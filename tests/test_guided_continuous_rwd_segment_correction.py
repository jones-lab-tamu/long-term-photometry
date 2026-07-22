from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from photometry_pipeline import guided_continuous_rwd_block_plan as block_subject
from photometry_pipeline import guided_continuous_rwd_correction_segments as c4a
from photometry_pipeline import guided_continuous_rwd_segment_correction as subject
from photometry_pipeline import guided_continuous_rwd_target_grid as grid_subject
from photometry_pipeline.core import preprocessing, regression
from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    evaluate_continuous_rwd_timestamp_continuity,
)
from photometry_pipeline.guided_continuous_rwd_projection import (
    GuidedContinuousRwdProjectedBlock,
    _target_coordinates,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    build_guided_continuous_rwd_review_binding,
)
from photometry_pipeline.guided_execution_payloads import (
    build_guided_execution_startup_mapping_contract,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedPlanCorrectionChoice,
)
from photometry_pipeline.io.rwd_continuous_source import (
    inspect_continuous_rwd_acquisition_folder,
)
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.signal_only_f0 import compute_signal_only_f0_production


def _choices(strategies):
    return [
        GuidedPlanCorrectionChoice(
            roi_id=roi_id,
            selected_strategy=strategy,
            source_type="local_correction_preview",
            current_or_stale="current",
            explicit_user_mark=True,
            evidence_reference={"strategy": strategy, "roi": roi_id},
        )
        for roi_id, strategy in strategies.items()
    ]


def _values(indices):
    time = indices / 10.0
    control1 = 2.0 + 0.15 * np.cos(0.17 * time)
    control2 = 3.0 + 0.12 * np.sin(0.11 * time + 0.4)
    signal1 = 5.0 + 1.6 * control1 + 0.08 * np.sin(0.7 * time)
    signal2 = 7.0 + 0.30 * np.cos(0.23 * time) + 0.04 * np.sin(1.3 * time)
    return (
        time.astype(np.float64),
        np.column_stack((control1, control2)).astype(np.float64),
        np.column_stack((signal1, signal2)).astype(np.float64),
    )


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_c4b") / "recording"
    folder.mkdir()
    source = folder / "Fluorescence.csv"
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(6001):
        time, control, signal = _values(np.array([index], dtype=float))
        lines.append(
            f"{time[0]:.1f},{control[0,0]:.12f},{signal[0,0]:.12f},"
            f"{control[0,1]:.12f},{signal[0,1]:.12f}\n"
        )
    source.write_text("".join(lines), encoding="utf-8", newline="")
    inspection = inspect_continuous_rwd_acquisition_folder(folder)
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("ROI1", "ROI2")
    )
    continuity = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=source
    )
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
        per_roi_correction_strategy_choices=_choices(
            {"ROI1": "global_linear_regression", "ROI2": "signal_only_f0"}
        ),
        feature_event_profile_id="default",
        feature_event_values={"peak_threshold_k": 8.0},
    )
    binding = build_guided_continuous_rwd_review_binding(
        draft,
        recording=recording,
        continuity_evaluation=continuity,
        current_source_path=source,
    )
    grid = grid_subject.build_guided_continuous_rwd_target_grid(recording, continuity)
    contract = build_guided_execution_startup_mapping_contract()
    return binding, grid, draft, contract


def _projected(binding, grid, block_plan, *, control_shift=0.0, signal_shift=0.0):
    for block in block_plan.iter_blocks():
        indices = np.arange(block.start_target_index, block.stop_target_index, dtype=float)
        _, control, signal = _values(indices)
        time = _target_coordinates(
            grid, block.start_target_index, block.stop_target_index
        )
        control = control.copy()
        signal = signal.copy()
        control[:, 1] += control_shift
        signal[:, 1] += signal_shift
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
            included_roi_ids=binding.recording.roi.included_roi_ids,
            source_row_start=block.start_target_index,
            source_row_stop=block.stop_target_index,
            target_elapsed_seconds=time,
            control_values=control,
            signal_values=signal,
        )


def _artifacts(accepted_case):
    binding, grid, draft, contract = accepted_case
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    segment_plan = c4a.build_guided_continuous_rwd_correction_segment_plan(
        binding, grid, accepted_draft=draft, startup_mapping_contract=contract
    )
    f0 = c4a.prepare_guided_continuous_rwd_dynamic_f0_authority(
        binding,
        grid,
        block_plan,
        segment_plan,
        _projected(binding, grid, block_plan),
        accepted_draft=draft,
        startup_mapping_contract=contract,
    )
    raw = next(
        c4a.iter_assemble_guided_continuous_rwd_correction_segments(
            binding,
            grid,
            block_plan,
            segment_plan,
            _projected(binding, grid, block_plan),
            accepted_draft=draft,
            startup_mapping_contract=contract,
        )
    )
    return binding, grid, draft, contract, segment_plan, f0, raw


def _variant(accepted_case, dynamic_strategy):
    binding, grid, draft, contract = accepted_case
    variant_draft = replace(
        draft,
        global_correction_strategy=dynamic_strategy,
        per_roi_correction_strategy_choices=_choices(
            {"ROI1": dynamic_strategy, "ROI2": "signal_only_f0"}
        ),
    )
    variant_binding = build_guided_continuous_rwd_review_binding(
        variant_draft,
        recording=binding.recording,
        continuity_evaluation=binding.continuity_evaluation,
        current_source_path=binding.current_source_path,
    )
    return variant_binding, grid, variant_draft, contract


def _contract_with_override(contract, name, value):
    overrides = tuple(
        replace(item, value=value) if item.name == name else item
        for item in contract.fixed_config_overrides
    )
    return replace(contract, fixed_config_overrides=overrides)


def _correct(artifacts, **kwargs):
    binding, grid, draft, contract, plan, f0, raw = artifacts
    return subject.correct_guided_continuous_rwd_segment(
        kwargs.get("binding", binding),
        kwargs.get("grid", grid),
        kwargs.get("plan", plan),
        kwargs.get("f0", f0),
        kwargs.get("raw", raw),
        accepted_draft=kwargs.get("draft", draft),
        startup_mapping_contract=kwargs.get("contract", contract),
        cancellation_requested=kwargs.get("cancellation"),
    )


def test_mixed_segment_matches_native_global_and_signal_only_references(accepted_case):
    artifacts = _artifacts(accepted_case)
    result = _correct(artifacts)
    binding, grid, draft, contract, plan, f0, raw = artifacts
    assert result.included_roi_ids == ("ROI1", "ROI2")
    assert [item.reference_kind for item in result.per_roi_results] == [
        subject.REFERENCE_FITTED_CONTROL,
        subject.REFERENCE_SIGNAL_DERIVED_F0,
    ]
    assert result.per_roi_results[0].scalar_f0 == f0.values[0].scalar_f0
    assert result.per_roi_results[1].scalar_f0 is None
    np.testing.assert_array_equal(result.raw_control_values, raw.control_values)
    np.testing.assert_array_equal(result.raw_signal_values, raw.signal_values)

    accepted = c4a._resolve_accepted_correction_context(binding, draft, contract)
    local_time = raw.target_elapsed_seconds - raw.target_elapsed_seconds[0]
    chunk = Chunk(
        chunk_id=raw.segment_index,
        source_file="reference",
        format="rwd",
        time_sec=local_time,
        uv_raw=raw.control_values[:, :1].copy(),
        sig_raw=raw.signal_values[:, :1].copy(),
        fs_hz=10.0,
        channel_names=["ROI1"],
        metadata={},
    )
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(
        chunk.uv_raw, chunk.fs_hz, accepted.config
    )
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(
        chunk.sig_raw, chunk.fs_hz, accepted.config
    )
    expected_fit, expected_delta = regression.fit_chunk_dynamic(
        chunk,
        accepted.config,
        mode="phasic",
        per_roi_correction={"ROI1": accepted.correction_specs["ROI1"]},
    )
    np.testing.assert_allclose(result.correction_reference_values[:, 0], expected_fit[:, 0], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(result.delta_f_values[:, 0], expected_delta[:, 0], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        result.dff_values[:, 0],
        100.0 * expected_delta[:, 0] / f0.values[0].scalar_f0,
        rtol=1e-11,
        atol=1e-12,
    )
    expected_signal = compute_signal_only_f0_production(
        raw.signal_values[:, 1],
        local_time,
        signal_state_config=dict(vars(accepted.config)),
        signal_only_f0_config=dict(vars(accepted.config)),
        coverage_fraction=accepted.config.signal_only_f0_min_coverage_fraction,
        f0_min_value=accepted.config.f0_min_value,
    )
    np.testing.assert_array_equal(result.correction_reference_values[:, 1], expected_signal.baseline)
    np.testing.assert_array_equal(result.delta_f_values[:, 1], expected_signal.delta_f)
    np.testing.assert_array_equal(result.dff_values[:, 1], expected_signal.dff)
    pipeline = Pipeline(accepted.config)
    pipeline_values = pipeline._compute_signal_only_f0_production(
        Chunk(
            chunk_id=raw.segment_index,
            source_file="reference",
            format="rwd",
            time_sec=local_time,
            uv_raw=raw.control_values[:, 1:2].copy(),
            sig_raw=raw.signal_values[:, 1:2].copy(),
            fs_hz=10.0,
            channel_names=["ROI2"],
            metadata={},
        ),
        roi_index=0,
        roi_id="ROI2",
        chunk_id=raw.segment_index,
    )
    np.testing.assert_array_equal(result.delta_f_values[:, 1], pipeline_values[0])
    np.testing.assert_array_equal(result.dff_values[:, 1], pipeline_values[1])
    np.testing.assert_array_equal(result.correction_reference_values[:, 1], pipeline_values[2])
    for array in (
        result.target_elapsed_seconds,
        result.raw_control_values,
        result.raw_signal_values,
        result.correction_reference_values,
        result.delta_f_values,
        result.dff_values,
    ):
        assert array.dtype == np.float64
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: {"binding": replace(a[0], draft_plan_identity="0" * 64)},
        lambda a: {"grid": replace(a[1], target_grid_identity="0" * 64)},
        lambda a: {"plan": replace(a[4], plan_identity="0" * 64)},
        lambda a: {"raw": replace(a[6], recording_identity="0" * 64)},
        lambda a: {"raw": replace(a[6], segment_index=1)},
        lambda a: {"raw": replace(a[6], stop_target_index=a[6].stop_target_index - 1)},
        lambda a: {"draft": replace(a[2], continuous_window_sec=21.0)},
        lambda a: {"contract": replace(a[3], contract_version="substituted")},
        lambda a: {"f0": replace(a[5], authority_identity="0" * 64)},
        lambda a: {"f0": replace(a[5], recording_identity="0" * 64)},
        lambda a: {"f0": replace(a[5], values=())},
        lambda a: {"f0": replace(a[5], dynamic_roi_ids=("ROI1", "ROI2"))},
    ],
)
def test_substituted_authorities_are_refused(accepted_case, mutation):
    artifacts = _artifacts(accepted_case)
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError):
        _correct(artifacts, **mutation(artifacts))


def test_result_validator_refuses_recomputed_substituted_metadata(accepted_case):
    artifacts = _artifacts(accepted_case)
    result = _correct(artifacts)
    first = replace(result.per_roi_results[0], selected_strategy="robust_global_event_reject")
    tampered = replace(result, per_roi_results=(first, result.per_roi_results[1]), result_identity="")
    tampered = replace(tampered, result_identity=subject._compute_result_identity(tampered))
    accepted = c4a._resolve_accepted_correction_context(
        artifacts[0], artifacts[2], artifacts[3]
    )
    _, settings_identity = subject._resolve_segment_correction_settings(artifacts[3])
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError, match="substituted"):
        subject._validate_result(
            tampered,
            raw_segment=artifacts[6],
            dynamic_f0_authority=artifacts[5],
            accepted=accepted,
            segment_correction_settings_identity=settings_identity,
        )


def test_result_validator_refuses_recomputed_f0_identity_substitution(accepted_case):
    artifacts = _artifacts(accepted_case)
    result = _correct(artifacts)
    tampered = replace(result, dynamic_f0_authority_identity="0" * 64, result_identity="")
    tampered = replace(tampered, result_identity=subject._compute_result_identity(tampered))
    accepted = c4a._resolve_accepted_correction_context(
        artifacts[0], artifacts[2], artifacts[3]
    )
    _, settings_identity = subject._resolve_segment_correction_settings(artifacts[3])
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError):
        subject._validate_result(
            tampered,
            raw_segment=artifacts[6],
            dynamic_f0_authority=artifacts[5],
            accepted=accepted,
            segment_correction_settings_identity=settings_identity,
        )


def test_signal_only_scientifically_ignores_control(accepted_case):
    artifacts = _artifacts(accepted_case)
    baseline = _correct(artifacts)
    raw = artifacts[6]
    changed_control = raw.control_values.copy()
    changed_control[:, 1] += 1000.0
    changed_control.setflags(write=False)
    changed = _correct(artifacts, raw=replace(raw, control_values=changed_control))
    np.testing.assert_array_equal(
        changed.correction_reference_values[:, 1], baseline.correction_reference_values[:, 1]
    )
    np.testing.assert_array_equal(changed.delta_f_values[:, 1], baseline.delta_f_values[:, 1])
    np.testing.assert_array_equal(changed.dff_values[:, 1], baseline.dff_values[:, 1])
    changed_dynamic_control = raw.control_values.copy()
    changed_dynamic_control[:, 0] += 0.1 * np.sin(np.arange(changed_dynamic_control.shape[0]))
    changed_dynamic_control.setflags(write=False)
    dynamic_changed = _correct(
        artifacts, raw=replace(raw, control_values=changed_dynamic_control)
    )
    assert not np.array_equal(
        dynamic_changed.correction_reference_values[:, 0],
        baseline.correction_reference_values[:, 0],
    )
    np.testing.assert_array_equal(
        dynamic_changed.correction_reference_values[:, 1],
        baseline.correction_reference_values[:, 1],
    )


@pytest.mark.parametrize(
    "strategy",
    ["robust_global_event_reject", "adaptive_event_gated_regression"],
)
def test_robust_and_adaptive_match_native_per_chunk_reference(accepted_case, strategy):
    artifacts = _artifacts(_variant(accepted_case, strategy))
    result = _correct(artifacts)
    binding, grid, draft, contract, plan, f0, raw = artifacts
    accepted = c4a._resolve_accepted_correction_context(binding, draft, contract)
    local_time = raw.target_elapsed_seconds - raw.target_elapsed_seconds[0]
    chunk = Chunk(
        chunk_id=raw.segment_index,
        source_file="reference",
        format="rwd",
        time_sec=local_time,
        uv_raw=raw.control_values[:, :1].copy(),
        sig_raw=raw.signal_values[:, :1].copy(),
        fs_hz=10.0,
        channel_names=["ROI1"],
        metadata={},
    )
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(
        chunk.uv_raw, chunk.fs_hz, accepted.config
    )
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(
        chunk.sig_raw, chunk.fs_hz, accepted.config
    )
    expected_fit, expected_delta = regression.fit_chunk_dynamic(
        chunk,
        accepted.config,
        mode="phasic",
        per_roi_correction={"ROI1": accepted.correction_specs["ROI1"]},
    )
    np.testing.assert_allclose(
        result.correction_reference_values[:, 0], expected_fit[:, 0], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        result.delta_f_values[:, 0], expected_delta[:, 0], rtol=1e-12, atol=1e-12
    )
    assert result.per_roi_results[0].fallback_path[0] == strategy
    detail = json.loads(result.per_roi_results[0].qc_json)["detail"]
    if strategy == "robust_global_event_reject":
        assert "keep_mask" in detail
    else:
        assert "trusted_mask" in detail
        assert "coef_slope" in detail


def test_event_contaminated_robust_segment_uses_native_accepted_path(accepted_case):
    artifacts = _artifacts(_variant(accepted_case, "robust_global_event_reject"))
    raw = artifacts[6]
    signal = raw.signal_values.copy()
    signal[75:95, 0] += 4.0
    signal.setflags(write=False)
    result = _correct(artifacts, raw=replace(raw, signal_values=signal))
    assert np.all(np.isfinite(result.correction_reference_values[:, 0]))
    assert result.per_roi_results[0].fallback_path[0] == "robust_global_event_reject"


def test_robust_and_adaptive_preserve_existing_fallback_paths(accepted_case, monkeypatch):
    robust_artifacts = _artifacts(_variant(accepted_case, "robust_global_event_reject"))

    def refuse_robust(*args, **kwargs):
        raise ValueError("forced robust refusal")

    monkeypatch.setattr(regression, "fit_robust_global_event_reject", refuse_robust)
    robust_result = _correct(robust_artifacts)
    assert robust_result.per_roi_results[0].applied_strategy == "global_linear_regression"
    assert robust_result.per_roi_results[0].fallback_path == (
        "robust_global_event_reject",
        "global_linear_regression",
    )

    adaptive_artifacts = _artifacts(_variant(accepted_case, "adaptive_event_gated_regression"))

    def refuse_adaptive(*args, **kwargs):
        raise ValueError("forced adaptive refusal")

    monkeypatch.setattr(regression, "fit_adaptive_event_gated_regression", refuse_adaptive)
    adaptive_result = _correct(adaptive_artifacts)
    assert adaptive_result.per_roi_results[0].applied_strategy == "global_linear_regression"
    assert adaptive_result.per_roi_results[0].fallback_path == (
        "adaptive_event_gated_regression",
        "robust_global_event_reject",
        "global_linear_regression",
    )


def test_exhausted_dynamic_fallback_fails_complete_segment_with_context(accepted_case):
    artifacts = _artifacts(_variant(accepted_case, "robust_global_event_reject"))
    raw = artifacts[6]
    control = raw.control_values.copy()
    control[:, 0] = 2.0
    control.setflags(write=False)
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError) as exc_info:
        _correct(artifacts, raw=replace(raw, control_values=control))
    assert exc_info.value.category == "dynamic_fit_failure"
    assert exc_info.value.context["roi"] == "ROI1"
    assert exc_info.value.context["segment_index"] == raw.segment_index


@pytest.mark.parametrize(
    "reason",
    [
        "production F0 finite coverage is insufficient",
        "production F0 baseline contains non-positive values",
    ],
)
def test_signal_only_failure_is_segment_scoped_without_fallback(
    accepted_case, monkeypatch, reason
):
    artifacts = _artifacts(accepted_case)

    def refuse(*args, **kwargs):
        raise subject.SignalOnlyF0ProductionError(reason)

    monkeypatch.setattr(subject, "compute_signal_only_f0_production", refuse)
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError) as exc_info:
        _correct(artifacts)
    assert exc_info.value.category == "signal_only_failure"
    assert exc_info.value.context["roi"] == "ROI2"
    assert exc_info.value.context["attempted_fallback_chain"] == ("signal_only_f0",)
    assert exc_info.value.context["reason"] == reason


def test_adjacent_segments_are_independent_and_neighbor_changes_do_not_leak(accepted_case):
    binding, grid, draft, contract = _variant(
        accepted_case, "adaptive_event_gated_regression"
    )
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    plan = c4a.build_guided_continuous_rwd_correction_segment_plan(
        binding, grid, accepted_draft=draft, startup_mapping_contract=contract
    )
    f0 = c4a.prepare_guided_continuous_rwd_dynamic_f0_authority(
        binding, grid, block_plan, plan, _projected(binding, grid, block_plan),
        accepted_draft=draft, startup_mapping_contract=contract,
    )
    raw_segments = list(c4a.iter_assemble_guided_continuous_rwd_correction_segments(
        binding, grid, block_plan, plan, _projected(binding, grid, block_plan),
        accepted_draft=draft, startup_mapping_contract=contract,
    ))
    first = subject.correct_guided_continuous_rwd_segment(
        binding, grid, plan, f0, raw_segments[0], accepted_draft=draft,
        startup_mapping_contract=contract,
    )
    second = subject.correct_guided_continuous_rwd_segment(
        binding, grid, plan, f0, raw_segments[1], accepted_draft=draft,
        startup_mapping_contract=contract,
    )
    changed_second_signal = raw_segments[1].signal_values.copy()
    changed_second_signal[:, 1] += 0.1 * np.sin(np.arange(changed_second_signal.shape[0]))
    changed_second_signal.setflags(write=False)
    changed_second = subject.correct_guided_continuous_rwd_segment(
        binding, grid, plan, f0,
        replace(raw_segments[1], signal_values=changed_second_signal),
        accepted_draft=draft, startup_mapping_contract=contract,
    )
    assert first.segment_index == 0 and second.segment_index == 1
    assert first.result_identity == subject._compute_result_identity(first)
    assert changed_second.result_identity != second.result_identity
    np.testing.assert_array_equal(first.raw_signal_values, raw_segments[0].signal_values)


def test_cancellation_before_and_between_rois_returns_no_result(accepted_case):
    artifacts = _artifacts(accepted_case)
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError) as exc_info:
        _correct(artifacts, cancellation=lambda: True)
    assert exc_info.value.category == "segment_correction_interrupted"
    calls = 0

    def cancel_between():
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError) as exc_info:
        _correct(artifacts, cancellation=cancel_between)
    assert exc_info.value.category == "segment_correction_interrupted"


def test_malformed_and_nonfinite_raw_arrays_are_refused(accepted_case):
    artifacts = _artifacts(accepted_case)
    raw = artifacts[6]
    malformed = raw.signal_values[:-1].copy()
    malformed.setflags(write=False)
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError):
        _correct(artifacts, raw=replace(raw, signal_values=malformed))
    nonfinite = raw.signal_values.copy()
    nonfinite[0, 0] = np.nan
    nonfinite.setflags(write=False)
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError):
        _correct(artifacts, raw=replace(raw, signal_values=nonfinite))


def test_result_is_deterministic_frozen_and_array_tampering_is_detected(accepted_case):
    artifacts = _artifacts(accepted_case)
    first = _correct(artifacts)
    second = _correct(artifacts)
    assert first.result_identity == second.result_identity
    with pytest.raises(FrozenInstanceError):
        first.segment_index = 99
    with pytest.raises(FrozenInstanceError):
        first.per_roi_results[0].roi_id = "other"
    changed = first.dff_values.copy()
    changed[0, 0] += 1.0
    changed.setflags(write=False)
    tampered = replace(first, dff_values=changed)
    accepted = c4a._resolve_accepted_correction_context(
        artifacts[0], artifacts[2], artifacts[3]
    )
    _, settings_identity = subject._resolve_segment_correction_settings(artifacts[3])
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError):
        subject._validate_result(
            tampered,
            raw_segment=artifacts[6],
            dynamic_f0_authority=artifacts[5],
            accepted=accepted,
            segment_correction_settings_identity=settings_identity,
        )


def test_segment_correction_settings_identity_present_and_deterministic(accepted_case):
    artifacts = _artifacts(accepted_case)
    result = _correct(artifacts)
    identity = result.segment_correction_settings_identity
    assert isinstance(identity, str)
    assert len(identity) == 64
    assert identity == identity.lower()
    assert all(character in "0123456789abcdef" for character in identity)

    second = _correct(artifacts)
    assert second.segment_correction_settings_identity == identity
    assert second.result_identity == result.result_identity


def test_result_binds_both_settings_identities(accepted_case):
    artifacts = _artifacts(accepted_case)
    result = _correct(artifacts)
    assert result.fixed_correction_settings_identity
    assert result.segment_correction_settings_identity
    assert result.fixed_correction_settings_identity != result.segment_correction_settings_identity

    payload = subject._identity_payload(result)
    assert payload["fixed_correction_settings_identity"] == result.fixed_correction_settings_identity
    assert (
        payload["segment_correction_settings_identity"]
        == result.segment_correction_settings_identity
    )

    tampered_fixed = replace(result, fixed_correction_settings_identity="0" * 64)
    assert subject._compute_result_identity(tampered_fixed) != result.result_identity
    tampered_segment = replace(result, segment_correction_settings_identity="0" * 64)
    assert subject._compute_result_identity(tampered_segment) != result.result_identity


@pytest.mark.parametrize(
    "field_name,new_value",
    [
        ("robust_event_reject_residual_z_thresh", 4.5),
        ("robust_event_reject_max_iters", 7),
    ],
)
def test_robust_setting_tampering_is_refused(accepted_case, field_name, new_value):
    _, _, _, contract = accepted_case
    tampered_contract = _contract_with_override(contract, field_name, new_value)
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError) as exc_info:
        subject._resolve_segment_correction_settings(tampered_contract)
    assert exc_info.value.category == "accepted_correction_binding_mismatch"


@pytest.mark.parametrize(
    "field_name,new_value",
    [
        ("adaptive_event_gate_min_trust_fraction", 0.9),
        ("adaptive_event_gate_smooth_window_sec", 120.0),
    ],
)
def test_adaptive_setting_tampering_is_refused(accepted_case, field_name, new_value):
    _, _, _, contract = accepted_case
    tampered_contract = _contract_with_override(contract, field_name, new_value)
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError) as exc_info:
        subject._resolve_segment_correction_settings(tampered_contract)
    assert exc_info.value.category == "accepted_correction_binding_mismatch"


@pytest.mark.parametrize(
    "field_name,new_value",
    [
        ("signal_only_f0_window_fraction", 0.30),
        ("signal_only_f0_low_quantile", 0.20),
        ("signal_only_f0_state_aware_enabled", False),
    ],
)
def test_signal_only_setting_tampering_is_refused(accepted_case, field_name, new_value):
    _, _, _, contract = accepted_case
    tampered_contract = _contract_with_override(contract, field_name, new_value)
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError) as exc_info:
        subject._resolve_segment_correction_settings(tampered_contract)
    assert exc_info.value.category == "accepted_correction_binding_mismatch"


@pytest.mark.parametrize(
    "field_name,new_value",
    [
        ("lowpass_hz", 2.0),
        ("filter_order", 4),
    ],
)
def test_filtering_setting_tampering_is_refused(accepted_case, field_name, new_value):
    _, _, _, contract = accepted_case
    tampered_contract = _contract_with_override(contract, field_name, new_value)
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError) as exc_info:
        subject._resolve_segment_correction_settings(tampered_contract)
    assert exc_info.value.category == "accepted_correction_binding_mismatch"
    # C4a's own narrower settings identity resolution independently refuses too,
    # but that alone must not be assumed to prove the C4b identity binds it.
    with pytest.raises(c4a.GuidedContinuousRwdCorrectionSegmentError):
        c4a._resolve_fixed_correction_config(tampered_contract)


def test_tampered_segment_correction_settings_identity_is_rejected(accepted_case):
    artifacts = _artifacts(accepted_case)
    result = _correct(artifacts)
    other_valid_form_digest = "a" * 64
    assert other_valid_form_digest != result.segment_correction_settings_identity
    tampered = replace(
        result,
        segment_correction_settings_identity=other_valid_form_digest,
        result_identity="",
    )
    tampered = replace(tampered, result_identity=subject._compute_result_identity(tampered))
    accepted = c4a._resolve_accepted_correction_context(artifacts[0], artifacts[2], artifacts[3])
    _, real_identity = subject._resolve_segment_correction_settings(artifacts[3])
    with pytest.raises(subject.GuidedContinuousRwdSegmentCorrectionError):
        subject._validate_result(
            tampered,
            raw_segment=artifacts[6],
            dynamic_f0_authority=artifacts[5],
            accepted=accepted,
            segment_correction_settings_identity=real_identity,
        )


def test_future_default_drift_on_previously_unbound_setting_changes_c4b_identity(
    accepted_case, monkeypatch
):
    binding, grid, draft, contract = accepted_case
    baseline_config, baseline_identity = subject._resolve_segment_correction_settings(contract)
    accepted_baseline = c4a._resolve_accepted_correction_context(binding, draft, contract)

    field_name = "adaptive_event_gate_min_trust_fraction"
    assert field_name not in c4a._C4A_FIXED_SETTING_NAMES
    altered_overrides = dict(subject.GUIDED_CONFIG_DEFAULT_OVERRIDES)
    altered_overrides[field_name] = altered_overrides[field_name] + 0.1
    monkeypatch.setattr(
        "photometry_pipeline.guided_execution_payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES",
        altered_overrides,
    )
    monkeypatch.setattr(subject, "GUIDED_CONFIG_DEFAULT_OVERRIDES", altered_overrides)

    drifted_contract = subject.build_guided_execution_startup_mapping_contract()
    assert drifted_contract != contract

    drifted_config, drifted_identity = subject._resolve_segment_correction_settings(
        drifted_contract
    )
    assert drifted_identity != baseline_identity
    assert drifted_config.adaptive_event_gate_min_trust_fraction == altered_overrides[field_name]

    accepted_drifted = c4a._resolve_accepted_correction_context(binding, draft, drifted_contract)
    assert (
        accepted_drifted.fixed_correction_settings_identity
        == accepted_baseline.fixed_correction_settings_identity
    )


def test_pipeline_delegates_to_extracted_signal_only_function(monkeypatch):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        raise subject.SignalOnlyF0ProductionError("delegated")

    monkeypatch.setattr("photometry_pipeline.pipeline.compute_signal_only_f0_production", fake)
    pipeline = Pipeline(Config())
    chunk = Chunk(
        chunk_id=1, source_file="source", format="rwd", time_sec=np.arange(20) / 10,
        uv_raw=np.ones((20, 1)), sig_raw=np.ones((20, 1)), fs_hz=10.0,
        channel_names=["ROI1"], metadata={},
    )
    with pytest.raises(Exception, match="delegated"):
        pipeline._compute_signal_only_f0_production(chunk, roi_index=0, roi_id="ROI1", chunk_id=1)
    assert len(calls) == 1
