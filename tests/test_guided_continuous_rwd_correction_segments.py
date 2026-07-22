from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import inspect
from pathlib import Path

import numpy as np
import pytest

from photometry_pipeline.core.baseline import DeterministicReservoir
from photometry_pipeline.core.types import PerRoiCorrectionSpec
from photometry_pipeline import guided_continuous_rwd_block_plan as block_subject
from photometry_pipeline import guided_continuous_rwd_correction_segments as subject
from photometry_pipeline import guided_continuous_rwd_target_grid as grid_subject
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    CONTINUITY_PASSED,
    evaluate_continuous_rwd_timestamp_continuity,
)
from photometry_pipeline.guided_continuous_rwd_projection import (
    GuidedContinuousRwdProjectedBlock,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    build_guided_continuous_rwd_review_binding,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.guided_new_analysis_plan import GuidedPlanCorrectionChoice
from photometry_pipeline.guided_execution_payloads import (
    GUIDED_CONFIG_DEFAULT_OVERRIDES,
    build_guided_execution_startup_mapping_contract,
)
from photometry_pipeline.io.rwd_continuous_projection_reader import (
    ContinuousRwdProjectionReaderError,
    iter_project_guided_continuous_rwd_blocks,
)
from photometry_pipeline.io.rwd_continuous_source import (
    inspect_continuous_rwd_acquisition_folder,
)


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_c4a") / "recording"
    folder.mkdir()
    source = folder / "Fluorescence.csv"
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(6001):
        timestamp = index / 10.0
        lines.append(
            f"{timestamp:.1f},{100 + index * 0.01},{200 + index * 0.02},"
            f"{300 + index * 0.03},{400 + index * 0.04}\n"
        )
    source.write_text("".join(lines), encoding="utf-8", newline="")
    inspection = inspect_continuous_rwd_acquisition_folder(folder)
    assert inspection.status == "completed"
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("ROI1", "ROI2")
    )
    evaluation = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=source
    )
    assert evaluation.outcome == CONTINUITY_PASSED
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(folder),
        resolved_input_source_path=str(folder),
        input_format="rwd",
        acquisition_mode="continuous",
        continuous_window_sec=20.0,
        continuous_step_sec=20.0,
        discovered_roi_ids=["ROI1", "ROI2"],
        included_roi_ids=["ROI1", "ROI2"],
        excluded_roi_ids=[],
        output_base_path=str(folder / "output"),
        global_correction_strategy="global_linear_regression",
        per_roi_correction_strategy_choices=list(
            _choices({"ROI1": "global_linear_regression", "ROI2": "signal_only_f0"})
        ),
        feature_event_profile_id="default",
        feature_event_values={"peak_threshold_k": 8.0},
    )
    binding = build_guided_continuous_rwd_review_binding(
        draft,
        recording=recording,
        continuity_evaluation=evaluation,
        current_source_path=source,
    )
    grid = grid_subject.build_guided_continuous_rwd_target_grid(
        recording, evaluation
    )
    return binding, grid, draft, build_guided_execution_startup_mapping_contract()


def _choices(strategies, *, evidence_tag="accepted"):
    return tuple(
        GuidedPlanCorrectionChoice(
            roi_id=roi_id,
            selected_strategy=strategy,
            source_type="local_correction_preview",
            current_or_stale="current",
            explicit_user_mark=True,
            evidence_reference={
                "evidence_source_type": "local_correction_preview",
                "evidence_reference_id": f"{evidence_tag}-{roi_id}",
                "strategy": strategy,
            },
        )
        for roi_id, strategy in strategies.items()
    )


def _accepted_variant(
    accepted_case,
    *,
    duration=None,
    strategies=None,
    evidence_tag="accepted",
    startup_mapping_contract=None,
):
    binding, grid, draft, contract = accepted_case
    duration = draft.continuous_window_sec if duration is None else duration
    if strategies is None:
        strategies = {
            choice.roi_id: choice.selected_strategy
            for choice in draft.per_roi_correction_strategy_choices
        }
    variant_draft = replace(
        draft,
        continuous_window_sec=duration,
        continuous_step_sec=duration,
        per_roi_correction_strategy_choices=list(
            _choices(strategies, evidence_tag=evidence_tag)
        ),
    )
    variant_binding = build_guided_continuous_rwd_review_binding(
        variant_draft,
        recording=binding.recording,
        continuity_evaluation=binding.continuity_evaluation,
        current_source_path=binding.current_source_path,
    )
    return (
        variant_binding,
        grid,
        variant_draft,
        startup_mapping_contract or contract,
    )


def _specs(*, roi2="signal_only_f0"):
    result = {
        "ROI1": PerRoiCorrectionSpec(
            roi_id="ROI1",
            strategy_family="dynamic_fit",
            selected_strategy="global_linear_regression",
            dynamic_fit_mode="global_linear_regression",
            parameter_identity="parameter-roi1",
            evidence_identity="evidence-roi1",
        )
    }
    if roi2 == "signal_only_f0":
        result["ROI2"] = PerRoiCorrectionSpec(
            roi_id="ROI2",
            strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0",
            dynamic_fit_mode=None,
            parameter_identity="parameter-roi2",
            evidence_identity="evidence-roi2",
        )
    else:
        result["ROI2"] = PerRoiCorrectionSpec(
            roi_id="ROI2",
            strategy_family="dynamic_fit",
            selected_strategy=roi2,
            dynamic_fit_mode=roi2,
            parameter_identity="parameter-roi2",
            evidence_identity="evidence-roi2",
        )
    return result


def _signal_only_specs():
    return {
        roi: PerRoiCorrectionSpec(
            roi_id=roi,
            strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0",
            dynamic_fit_mode=None,
            parameter_identity=f"parameter-{roi}",
            evidence_identity=f"evidence-{roi}",
        )
        for roi in ("ROI1", "ROI2")
    }


def _plan(accepted_case, *, duration=None, specs=None):
    strategies = None
    if specs is not None:
        strategies = {roi: spec.selected_strategy for roi, spec in specs.items()}
    case = _accepted_variant(
        accepted_case,
        duration=duration,
        strategies=strategies,
    )
    binding, grid, draft, contract = case
    return subject.build_guided_continuous_rwd_correction_segment_plan(
        binding,
        grid,
        accepted_draft=draft,
        startup_mapping_contract=contract,
    )


def _projected_blocks(accepted_case, block_plan):
    binding, grid = accepted_case[:2]
    roi_order = binding.recording.roi.included_roi_ids
    for block in block_plan.iter_blocks():
        indices = np.arange(
            block.start_target_index, block.stop_target_index, dtype=np.float64
        )
        time = indices * float(grid.cadence_fraction)
        control = np.column_stack((100.0 + indices * 0.01, 300.0 + indices * 0.03))
        signal = np.column_stack((200.0 + indices * 0.02, 400.0 + indices * 0.04))
        for array in (time, control, signal):
            array.setflags(write=False)
        yield GuidedContinuousRwdProjectedBlock(
            schema_name="guided_continuous_rwd_projected_block",
            schema_version="v1",
            projection_policy_name="continuous-rwd-linear-projection",
            projection_policy_version="v1",
            recording_identity=binding.recording.recording_identity,
            source_content_identity=binding.recording.source.source_content_identity,
            target_grid_identity=grid.target_grid_identity,
            block_index=block.block_index,
            start_target_index=block.start_target_index,
            stop_target_index=block.stop_target_index,
            included_roi_ids=roi_order,
            source_row_start=block.start_target_index,
            source_row_stop=block.stop_target_index,
            target_elapsed_seconds=time,
            control_values=control,
            signal_values=signal,
        )


def _partition_result(accepted_case, monkeypatch, block_size):
    binding, grid, draft, contract = accepted_case
    monkeypatch.setattr(
        block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", block_size
    )
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    segment_plan = _plan(accepted_case)
    blocks = tuple(_projected_blocks(accepted_case, block_plan))
    assembled = tuple(
        subject.iter_assemble_guided_continuous_rwd_correction_segments(
            binding,
            grid,
            block_plan,
            segment_plan,
            iter(blocks),
            accepted_draft=draft,
            startup_mapping_contract=contract,
        )
    )
    authority = subject.prepare_guided_continuous_rwd_dynamic_f0_authority(
        binding,
        grid,
        block_plan,
        segment_plan,
        iter(blocks),
        accepted_draft=draft,
        startup_mapping_contract=contract,
    )
    return segment_plan, assembled, authority


def test_nominal_600_seconds_uses_exact_c1_cadence_and_existing_rounding(accepted_case):
    plan = _plan(accepted_case, duration=600.0)
    assert plan.nominal_segment_sample_count == 6000
    assert plan.descriptors == (
        subject.GuidedContinuousRwdCorrectionSegmentDescription(
            segment_index=0,
            start_target_index=0,
            stop_target_index=6000,
            is_final=True,
            absorbed_short_tail=False,
        ),
    )
    assert (plan.segment_duration_seconds_numerator, plan.segment_duration_seconds_denominator) == (600, 1)
    assert (plan.cadence_seconds_numerator, plan.cadence_seconds_denominator) == (
        accepted_case[1].cadence_seconds_numerator,
        accepted_case[1].cadence_seconds_denominator,
    )


def test_nonintegral_ratio_uses_exact_python_round_ties_to_even(accepted_case):
    assert subject._rounded_segment_sample_count(
        Fraction(21, 20), Fraction(1, 10)
    ) == 10
    assert subject._rounded_segment_sample_count(
        Fraction(23, 20), Fraction(1, 10)
    ) == 12
    plan = _plan(accepted_case, duration=1.05, specs=_signal_only_specs())
    assert plan.nominal_segment_sample_count == round(21 / 2) == 10
    assert plan.descriptors[0].stop_target_index == 10
    assert all(
        left.stop_target_index == right.start_target_index
        for left, right in zip(plan.descriptors, plan.descriptors[1:])
    )
    assert plan.descriptors[-1].stop_target_index == plan.target_sample_count


def test_exact_full_multiple_and_viable_short_tail_are_retained(accepted_case):
    exact = _plan(accepted_case, duration=600.0)
    assert exact.nominal_segment_sample_count == 6000
    assert exact.descriptors[0].sample_count == 6000
    assert not exact.descriptors[0].absorbed_short_tail

    short = _plan(accepted_case, duration=298.9)
    assert [item.sample_count for item in short.descriptors] == [2989, 2989, 22]
    assert not short.descriptors[-1].absorbed_short_tail


def test_nonviable_tail_merges_without_gap_or_float_drift(accepted_case):
    plan = _plan(accepted_case, duration=299.9)
    assert [item.sample_count for item in plan.descriptors] == [2999, 3001]
    assert plan.descriptors[-1].absorbed_short_tail
    assert [
        index
        for item in plan.descriptors
        for index in range(item.start_target_index, item.stop_target_index)
    ] == list(range(plan.target_sample_count))


def test_complete_recording_below_shared_minimum_is_refused(accepted_case):
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject._canonical_segment_ranges(9, 20, 10)
    assert caught.value.category == "invalid_correction_settings"


def test_mixed_strategy_uses_filter_safe_shared_minimum(accepted_case):
    plan = _plan(accepted_case)
    assert plan.shared_minimum_viable_sample_count == 13
    signal_only = _plan(accepted_case, specs=_signal_only_specs())
    assert signal_only.shared_minimum_viable_sample_count == 10


def test_plan_is_frozen_deterministic_and_binds_correction_settings(accepted_case):
    first = _plan(accepted_case)
    second = _plan(accepted_case)
    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.segment_count = 0
    changed_specs = _specs(roi2="robust_global_event_reject")
    assert _plan(accepted_case, specs=changed_specs).correction_contract_identity != first.correction_contract_identity


def test_public_apis_remove_free_scientific_inputs():
    plan_parameters = set(
        inspect.signature(
            subject.build_guided_continuous_rwd_correction_segment_plan
        ).parameters
    )
    f0_parameters = set(
        inspect.signature(
            subject.prepare_guided_continuous_rwd_dynamic_f0_authority
        ).parameters
    )
    assert {"accepted_draft", "startup_mapping_contract"} <= plan_parameters
    assert {"continuous_window_sec", "per_roi_correction", "config"}.isdisjoint(
        plan_parameters
    )
    assert {"accepted_draft", "startup_mapping_contract"} <= f0_parameters
    assert {"continuous_window_sec", "per_roi_correction", "config"}.isdisjoint(
        f0_parameters
    )


def test_accepted_600_second_duration_refuses_substituted_20_second_draft(accepted_case):
    accepted_600 = _accepted_variant(accepted_case, duration=600.0)
    binding, grid, draft, contract = accepted_600
    substituted = replace(
        draft,
        continuous_window_sec=20.0,
        continuous_step_sec=20.0,
    )
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject.build_guided_continuous_rwd_correction_segment_plan(
            binding,
            grid,
            accepted_draft=substituted,
            startup_mapping_contract=contract,
        )
    assert caught.value.category == "accepted_correction_binding_mismatch"
    assert caught.value.context["field"] == "accepted_guided_plan_identity"


@pytest.mark.parametrize(
    ("accepted_strategy", "substituted_strategy"),
    [
        ("robust_global_event_reject", "global_linear_regression"),
        ("global_linear_regression", "signal_only_f0"),
    ],
)
def test_accepted_strategy_or_family_substitution_is_refused(
    accepted_case,
    accepted_strategy,
    substituted_strategy,
):
    accepted = _accepted_variant(
        accepted_case,
        strategies={"ROI1": accepted_strategy, "ROI2": "signal_only_f0"},
    )
    binding, grid, draft, contract = accepted
    substituted = replace(
        draft,
        per_roi_correction_strategy_choices=list(
            _choices(
                {"ROI1": substituted_strategy, "ROI2": "signal_only_f0"},
                evidence_tag="accepted",
            )
        ),
    )
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject.build_guided_continuous_rwd_correction_segment_plan(
            binding,
            grid,
            accepted_draft=substituted,
            startup_mapping_contract=contract,
        )
    assert caught.value.category == "accepted_correction_binding_mismatch"


def test_parameter_contract_and_evidence_substitution_are_refused(accepted_case):
    binding, grid, draft, contract = accepted_case
    parameter_substitution = replace(
        draft,
        dynamic_fit_parameter_contract=replace(
            draft.dynamic_fit_parameter_contract,
            min_slope=draft.dynamic_fit_parameter_contract.min_slope + 0.25,
        ),
    )
    evidence_substitution = replace(
        draft,
        per_roi_correction_strategy_choices=list(
            _choices(
                {"ROI1": "global_linear_regression", "ROI2": "signal_only_f0"},
                evidence_tag="substituted",
            )
        ),
    )
    for substituted in (parameter_substitution, evidence_substitution):
        with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
            subject.build_guided_continuous_rwd_correction_segment_plan(
                binding,
                grid,
                accepted_draft=substituted,
                startup_mapping_contract=contract,
            )
        assert caught.value.category == "accepted_correction_binding_mismatch"


def test_fixed_correction_setting_substitution_is_refused(accepted_case):
    binding, grid, draft, _ = accepted_case
    changed = dict(GUIDED_CONFIG_DEFAULT_OVERRIDES)
    changed["lowpass_hz"] = 0.5
    substituted_contract = build_guided_execution_startup_mapping_contract(
        fixed_config_overrides=changed
    )
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject.build_guided_continuous_rwd_correction_segment_plan(
            binding,
            grid,
            accepted_draft=draft,
            startup_mapping_contract=substituted_contract,
        )
    assert caught.value.category == "accepted_correction_binding_mismatch"
    assert caught.value.context["field"] == "startup_mapping_contract"


def test_recomputed_payload_and_plan_digests_cannot_legitimize_strategy_substitution(
    accepted_case,
):
    binding, grid, draft, contract = accepted_case
    plan = _plan(accepted_case)
    context = subject._resolve_accepted_correction_context(binding, draft, contract)
    changed_binding = replace(
        context.bindings[0],
        selected_strategy="robust_global_event_reject",
        dynamic_fit_mode="robust_global_event_reject",
    )
    changed_bindings = (changed_binding,) + context.bindings[1:]
    changed_payload = subject._correction_payload_identity_from_bindings(
        context.roi_order,
        changed_bindings,
    )
    changed_contract = subject._correction_contract_identity(
        accepted_guided_plan_identity=context.accepted_guided_plan_identity,
        correction_payload_identity_value=changed_payload,
        fixed_correction_settings_identity=context.fixed_correction_settings_identity,
    )
    changed_plan = replace(
        plan,
        correction_payload_identity=changed_payload,
        correction_contract_identity=changed_contract,
        plan_identity="",
    )
    changed_plan = replace(
        changed_plan,
        plan_identity=(
            subject.compute_guided_continuous_rwd_correction_segment_plan_identity(
                changed_plan
            )
        ),
    )
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject._validate_segment_plan(changed_plan, grid, binding, context)
    assert caught.value.category == "accepted_correction_binding_mismatch"
    assert caught.value.context["field"] == "segment_plan_correction_provenance"


def test_tampered_plan_identity_and_descriptor_are_refused(accepted_case):
    binding, grid = accepted_case[:2]
    plan = _plan(accepted_case)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError):
        subject._validate_segment_plan(
            replace(plan, plan_identity="0" * 64), grid, binding
        )
    bad_descriptor = replace(plan.descriptors[0], start_target_index=1)
    bad_plan = replace(plan, descriptors=(bad_descriptor,) + plan.descriptors[1:])
    bad_plan = replace(
        bad_plan,
        plan_identity=subject.compute_guided_continuous_rwd_correction_segment_plan_identity(bad_plan),
    )
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject._validate_segment_plan(bad_plan, grid, binding)
    assert caught.value.category == "invalid_segment_descriptor"


def test_assembler_handles_many_segments_per_block_and_cross_block_segments(accepted_case, monkeypatch):
    binding, grid, draft, contract = accepted_case
    segment_plan = _plan(accepted_case, duration=20.0)
    monkeypatch.setattr(block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", 550)
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    segments = tuple(
        subject.iter_assemble_guided_continuous_rwd_correction_segments(
            binding,
            grid,
            block_plan,
            segment_plan,
            _projected_blocks(accepted_case, block_plan),
            accepted_draft=draft,
            startup_mapping_contract=contract,
        )
    )
    assert len(segments) == segment_plan.segment_count
    assert segments[0].stop_target_index == 200
    assert segments[2].start_target_index == 400
    assert segments[2].stop_target_index == 600  # crosses the C2 boundary at 550
    for segment, descriptor in zip(segments, segment_plan.descriptors):
        assert (segment.start_target_index, segment.stop_target_index) == (
            descriptor.start_target_index,
            descriptor.stop_target_index,
        )
        assert segment.included_roi_ids == ("ROI1", "ROI2")
        assert segment.target_elapsed_seconds.dtype == np.float64
        assert segment.control_values.dtype == np.float64
        assert segment.signal_values.dtype == np.float64
        assert not segment.target_elapsed_seconds.flags.writeable
        assert not segment.control_values.flags.writeable
        assert not segment.signal_values.flags.writeable


def test_final_merged_segment_can_span_a_c2_boundary(accepted_case, monkeypatch):
    merged_case = _accepted_variant(accepted_case, duration=299.9)
    binding, grid, draft, contract = merged_case
    segment_plan = _plan(merged_case)
    assert segment_plan.descriptors[-1].absorbed_short_tail
    monkeypatch.setattr(block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", 3500)
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    segments = tuple(
        subject.iter_assemble_guided_continuous_rwd_correction_segments(
            binding,
            grid,
            block_plan,
            segment_plan,
            _projected_blocks(merged_case, block_plan),
            accepted_draft=draft,
            startup_mapping_contract=contract,
        )
    )
    final = segments[-1]
    assert (final.start_target_index, final.stop_target_index) == (2999, 6000)
    assert final.start_target_index < 3500 < final.stop_target_index
    np.testing.assert_array_equal(
        final.target_elapsed_seconds,
        np.arange(2999, 6000, dtype=np.float64) * float(grid.cadence_fraction),
    )


def test_partition_invariance_includes_segments_counts_f0_and_identity(accepted_case, monkeypatch):
    first = _partition_result(accepted_case, monkeypatch, 137)
    second = _partition_result(accepted_case, monkeypatch, 1000)
    assert first[0] == second[0]
    assert len(first[1]) == len(second[1])
    for left, right in zip(first[1], second[1]):
        assert (left.start_target_index, left.stop_target_index) == (
            right.start_target_index,
            right.stop_target_index,
        )
        np.testing.assert_array_equal(left.target_elapsed_seconds, right.target_elapsed_seconds)
        np.testing.assert_array_equal(left.control_values, right.control_values)
        np.testing.assert_array_equal(left.signal_values, right.signal_values)
    assert first[2].values == second[2].values
    assert first[2].authority_identity == second[2].authority_identity


def test_assembler_refuses_wrong_order_shape_roi_and_time(accepted_case, monkeypatch):
    binding, grid, draft, contract = accepted_case
    monkeypatch.setattr(block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", 1000)
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    plan = _plan(accepted_case)
    blocks = list(_projected_blocks(accepted_case, block_plan))
    cases = (
        (replace(blocks[0], block_index=1), "projected_block_order_mismatch"),
        (replace(blocks[0], included_roi_ids=("ROI2", "ROI1")), "projected_roi_order_mismatch"),
        (replace(blocks[0], control_values=blocks[0].control_values[:, :1]), "projected_shape_mismatch"),
        (replace(blocks[0], target_elapsed_seconds=blocks[0].target_elapsed_seconds + 1.0), "target_time_mismatch"),
    )
    for bad, category in cases:
        if isinstance(bad.target_elapsed_seconds, np.ndarray):
            bad.target_elapsed_seconds.setflags(write=False)
        with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
            list(
                subject.iter_assemble_guided_continuous_rwd_correction_segments(
                    binding,
                    grid,
                    block_plan,
                    plan,
                    iter([bad] + blocks[1:]),
                    accepted_draft=draft,
                    startup_mapping_contract=contract,
                )
            )
        assert caught.value.category == category


def test_assembler_cancellation_and_incomplete_exhaustion(accepted_case, monkeypatch):
    binding, grid, draft, contract = accepted_case
    monkeypatch.setattr(block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", 1000)
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    plan = _plan(accepted_case)
    blocks = tuple(_projected_blocks(accepted_case, block_plan))
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        list(
            subject.iter_assemble_guided_continuous_rwd_correction_segments(
                binding,
                grid,
                block_plan,
                plan,
                iter(blocks),
                accepted_draft=draft,
                startup_mapping_contract=contract,
                cancellation_requested=lambda: True,
            )
        )
    assert caught.value.category == "f0_preparation_interrupted"
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        list(
            subject.iter_assemble_guided_continuous_rwd_correction_segments(
                binding,
                grid,
                block_plan,
                plan,
                iter(blocks[:-1]),
                accepted_draft=draft,
                startup_mapping_contract=contract,
            )
        )
    assert caught.value.category == "segment_assembly_incomplete"


def test_pass1_real_c3b_uses_raw_control_dynamic_only_and_finalizes(accepted_case, monkeypatch):
    binding, grid, draft, contract = accepted_case
    monkeypatch.setattr(block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", 1000)
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    plan = _plan(accepted_case)
    authority = subject.prepare_guided_continuous_rwd_dynamic_f0_authority(
        binding,
        grid,
        block_plan,
        plan,
        iter_project_guided_continuous_rwd_blocks(binding, grid, block_plan),
        accepted_draft=draft,
        startup_mapping_contract=contract,
    )
    assert authority.dynamic_roi_ids == ("ROI1",)
    assert authority.values[0].finite_value_count == grid.target_sample_count
    assert authority.values[0].retained_value_count == grid.target_sample_count
    expected = np.percentile(
        np.asarray(100.0 + np.arange(6000) * 0.01, dtype=np.float32), 10.0
    )
    assert authority.values[0].scalar_f0 == expected
    assert authority.percentile == 10.0
    assert authority.capacity == 200_000
    assert authority.storage_dtype == "float32"
    assert authority.seed == 0
    assert authority.finalized
    assert authority.completion_state == "complete_source_verified"


def test_pass1_matches_direct_reservoir_canonical_segment_and_roi_order(accepted_case, monkeypatch):
    plan, segments, authority = _partition_result(accepted_case, monkeypatch, 733)
    direct = DeterministicReservoir(seed=0)
    for segment in segments:
        direct.add("ROI1", segment.control_values[:, 0])
    assert authority.values[0].scalar_f0 == direct.get_percentile("ROI1", 10.0)
    assert authority.values[0].finite_value_count == direct.count["ROI1"]
    assert authority.correction_segment_plan_identity == plan.plan_identity


def test_pass1_fixes_segment_then_b1_dynamic_roi_update_order(accepted_case, monkeypatch):
    dynamic_case = _accepted_variant(
        accepted_case,
        strategies={
            "ROI1": "global_linear_regression",
            "ROI2": "adaptive_event_gated_regression",
        },
    )
    binding, grid, draft, contract = dynamic_case
    plan = _plan(dynamic_case)
    monkeypatch.setattr(block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", 733)
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    blocks = tuple(_projected_blocks(dynamic_case, block_plan))
    calls = []
    original_add = subject.DeterministicReservoir.add

    def tracked_add(self, channel, data):
        calls.append((channel, len(data)))
        original_add(self, channel, data)

    monkeypatch.setattr(subject.DeterministicReservoir, "add", tracked_add)
    authority = subject.prepare_guided_continuous_rwd_dynamic_f0_authority(
        binding,
        grid,
        block_plan,
        plan,
        iter(blocks),
        accepted_draft=draft,
        startup_mapping_contract=contract,
    )
    expected = []
    for descriptor in plan.descriptors:
        expected.extend(
            [("ROI1", descriptor.sample_count), ("ROI2", descriptor.sample_count)]
        )
    assert calls == expected
    assert authority.dynamic_roi_ids == ("ROI1", "ROI2")


def test_all_signal_only_returns_empty_authority_without_consuming_iterator(accepted_case, monkeypatch):
    signal_case = _accepted_variant(
        accepted_case,
        strategies={"ROI1": "signal_only_f0", "ROI2": "signal_only_f0"},
    )
    binding, grid, draft, contract = signal_case
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    plan = _plan(signal_case)

    class Bomb:
        def __iter__(self):
            raise AssertionError("all-Signal-Only Pass 1 must not open C3b")

    authority = subject.prepare_guided_continuous_rwd_dynamic_f0_authority(
        binding,
        grid,
        block_plan,
        plan,
        Bomb(),
        accepted_draft=draft,
        startup_mapping_contract=contract,
    )
    assert authority.dynamic_roi_ids == ()
    assert authority.values == ()
    assert authority.completion_state == "not_required_all_signal_only"


def test_pass1_empty_support_and_too_small_f0_are_structured_refusals(accepted_case, monkeypatch):
    binding, grid, draft, contract = accepted_case
    monkeypatch.setattr(block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", 1000)
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    plan = _plan(accepted_case)
    blocks = tuple(_projected_blocks(accepted_case, block_plan))
    original_add = subject.DeterministicReservoir.add

    def skip_add(self, channel, data):
        if channel != "ROI1":
            original_add(self, channel, data)

    monkeypatch.setattr(subject.DeterministicReservoir, "add", skip_add)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject.prepare_guided_continuous_rwd_dynamic_f0_authority(
            binding, grid, block_plan, plan, iter(blocks),
            accepted_draft=draft, startup_mapping_contract=contract
        )
    assert caught.value.category == "no_finite_control_support"
    assert caught.value.context["roi"] == "ROI1"

    monkeypatch.setattr(subject.DeterministicReservoir, "add", original_add)
    zero_blocks = tuple(
        replace(block, control_values=np.zeros_like(block.control_values))
        for block in blocks
    )
    for block in zero_blocks:
        block.control_values.setflags(write=False)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject.prepare_guided_continuous_rwd_dynamic_f0_authority(
            binding, grid, block_plan, plan, iter(zero_blocks),
            accepted_draft=draft, startup_mapping_contract=contract
        )
    assert caught.value.category == "invalid_final_f0"
    assert caught.value.context["reason"] == "invalid_scalar_f0"


def test_late_source_failure_and_cancellation_never_return_authority(accepted_case, monkeypatch):
    binding, grid, draft, contract = accepted_case
    monkeypatch.setattr(block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", 1000)
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    plan = _plan(accepted_case)
    blocks = tuple(_projected_blocks(accepted_case, block_plan))

    def late_failure():
        yield from blocks
        raise ContinuousRwdProjectionReaderError(
            "source_content_mismatch", "late verification failed"
        )

    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject.prepare_guided_continuous_rwd_dynamic_f0_authority(
            binding, grid, block_plan, plan, late_failure(),
            accepted_draft=draft, startup_mapping_contract=contract
        )
    assert caught.value.category == "source_verification_failed"

    calls = 0
    def cancel_after_progress():
        nonlocal calls
        calls += 1
        return calls > 3

    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject.prepare_guided_continuous_rwd_dynamic_f0_authority(
            binding, grid, block_plan, plan, iter(blocks),
            accepted_draft=draft, startup_mapping_contract=contract,
            cancellation_requested=cancel_after_progress,
        )
    assert caught.value.category == "f0_preparation_interrupted"


def test_tampered_f0_authority_identity_is_refused(accepted_case, monkeypatch):
    binding, grid, draft, contract = accepted_case
    context = subject._resolve_accepted_correction_context(binding, draft, contract)
    plan, _, authority = _partition_result(accepted_case, monkeypatch, 777)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
        subject._validate_dynamic_f0_authority(
            replace(authority, authority_identity="0" * 64),
            review_binding=binding,
            target_grid=grid,
            segment_plan=plan,
            accepted_context=context,
        )
    assert caught.value.category == "invalid_final_f0"

    for field, value in (
        ("parameter_identity", "substituted-parameter-identity"),
        ("evidence_identity", "substituted-evidence-identity"),
    ):
        changed_binding = replace(authority.correction_bindings[0], **{field: value})
        changed_bindings = (changed_binding,) + authority.correction_bindings[1:]
        changed_payload = subject._correction_payload_identity_from_bindings(
            authority.canonical_roi_order,
            changed_bindings,
        )
        changed = replace(
            authority,
            correction_bindings=changed_bindings,
            correction_payload_identity=changed_payload,
            authority_identity="",
        )
        changed = replace(
            changed,
            authority_identity=(
                subject.compute_guided_continuous_rwd_dynamic_f0_authority_identity(
                    changed
                )
            ),
        )
        with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError) as caught:
            subject._validate_dynamic_f0_authority(
                changed,
                review_binding=binding,
                target_grid=grid,
                segment_plan=plan,
                accepted_context=context,
            )
        assert caught.value.category == "accepted_correction_binding_mismatch"
        assert caught.value.context["field"] == "correction_payload_identity"


def test_dynamic_fit_mode_substitution_is_refused_even_with_recomputed_authority_digest(
    accepted_case,
    monkeypatch,
):
    binding, grid, draft, contract = accepted_case
    plan, _, authority = _partition_result(accepted_case, monkeypatch, 777)
    context = subject._resolve_accepted_correction_context(binding, draft, contract)
    changed_binding = replace(
        authority.correction_bindings[0],
        dynamic_fit_mode="robust_global_event_reject",
    )
    changed = replace(
        authority,
        correction_bindings=(changed_binding,) + authority.correction_bindings[1:],
        authority_identity="",
    )
    changed = replace(
        changed,
        authority_identity=(
            subject.compute_guided_continuous_rwd_dynamic_f0_authority_identity(
                changed
            )
        ),
    )
    with pytest.raises(subject.GuidedContinuousRwdCorrectionSegmentError):
        subject._validate_dynamic_f0_authority(
            changed,
            review_binding=binding,
            target_grid=grid,
            segment_plan=plan,
            accepted_context=context,
        )
