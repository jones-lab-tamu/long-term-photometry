from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from photometry_pipeline import guided_continuous_rwd_block_plan as block_subject
from photometry_pipeline import guided_continuous_rwd_correction_pass as subject
from photometry_pipeline import guided_continuous_rwd_correction_segments as c4a
from photometry_pipeline import guided_continuous_rwd_segment_correction as c4b
from photometry_pipeline import guided_continuous_rwd_target_grid as grid_subject
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    evaluate_continuous_rwd_timestamp_continuity,
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
from photometry_pipeline.io.rwd_continuous_projection_reader import (
    iter_project_guided_continuous_rwd_blocks,
)
from photometry_pipeline.io.rwd_continuous_source import (
    inspect_continuous_rwd_acquisition_folder,
)


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


def _build_case(folder, *, continuous_window_sec=20.0, strategies=None):
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
    strategies = strategies or {
        "ROI1": "global_linear_regression",
        "ROI2": "signal_only_f0",
    }
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(folder),
        resolved_input_source_path=str(folder),
        input_format="rwd",
        acquisition_mode="continuous",
        continuous_window_sec=continuous_window_sec,
        continuous_step_sec=continuous_window_sec,
        discovered_roi_ids=["ROI1", "ROI2"],
        included_roi_ids=["ROI1", "ROI2"],
        excluded_roi_ids=[],
        output_base_path=str(folder / "output"),
        global_correction_strategy=next(iter(strategies.values())),
        per_roi_correction_strategy_choices=_choices(strategies),
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


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_c4c") / "recording"
    return _build_case(folder)


def _variant(tmp_path_factory, *, continuous_window_sec, strategies=None):
    folder = tmp_path_factory.mktemp("cr1_c4c_variant") / "recording"
    return _build_case(
        folder, continuous_window_sec=continuous_window_sec, strategies=strategies
    )


def _pass_inputs(case):
    binding, grid, draft, contract = case
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    segment_plan = c4a.build_guided_continuous_rwd_correction_segment_plan(
        binding, grid, accepted_draft=draft, startup_mapping_contract=contract
    )
    f0 = c4a.prepare_guided_continuous_rwd_dynamic_f0_authority(
        binding,
        grid,
        block_plan,
        segment_plan,
        iter_project_guided_continuous_rwd_blocks(binding, grid, block_plan),
        accepted_draft=draft,
        startup_mapping_contract=contract,
    )
    return binding, grid, draft, contract, block_plan, segment_plan, f0


def _run(inputs, **kwargs):
    binding, grid, draft, contract, block_plan, segment_plan, f0 = inputs
    return subject.iterate_guided_continuous_rwd_corrected_segments(
        kwargs.get("binding", binding),
        kwargs.get("grid", grid),
        kwargs.get("block_plan", block_plan),
        kwargs.get("segment_plan", segment_plan),
        kwargs.get("f0", f0),
        accepted_draft=kwargs.get("draft", draft),
        startup_mapping_contract=kwargs.get("contract", contract),
        cancellation_requested=kwargs.get("cancellation"),
    )


# ---------------------------------------------------------------------------
# Successful traversal
# ---------------------------------------------------------------------------


def test_one_segment_recording_completes(tmp_path_factory):
    case = _variant(tmp_path_factory, continuous_window_sec=600.0)
    inputs = _pass_inputs(case)
    traversal = _run(inputs)
    results = list(traversal)
    assert len(results) == 1
    assert results[0].segment_index == 0
    assert results[0].stop_target_index == inputs[1].target_sample_count
    assert traversal.state == "completed"
    completion = traversal.completion
    assert completion.corrected_segment_count == 1
    assert completion.target_sample_count == inputs[1].target_sample_count


def test_multiple_segments_canonical_order_and_full_coverage(accepted_case):
    inputs = _pass_inputs(accepted_case)
    traversal = _run(inputs)
    results = list(traversal)
    segment_plan = inputs[5]
    grid = inputs[1]
    assert [item.segment_index for item in results] == list(range(segment_plan.segment_count))
    assert results[0].start_target_index == 0
    assert results[-1].stop_target_index == grid.target_sample_count
    for left, right in zip(results, results[1:]):
        assert left.stop_target_index == right.start_target_index
    completion = traversal.completion
    assert completion.corrected_segment_count == segment_plan.segment_count
    assert completion.target_sample_count == grid.target_sample_count


def test_viable_final_short_tail_uses_accepted_c4a_plan(tmp_path_factory):
    case = _variant(tmp_path_factory, continuous_window_sec=298.9)
    inputs = _pass_inputs(case)
    segment_plan = inputs[5]
    assert [d.sample_count for d in segment_plan.descriptors] == [2989, 2989, 22]
    assert not segment_plan.descriptors[-1].absorbed_short_tail
    traversal = _run(inputs)
    results = list(traversal)
    assert [item.stop_target_index - item.start_target_index for item in results] == [
        2989,
        2989,
        22,
    ]
    assert traversal.completion.corrected_segment_count == 3


def test_merged_nonviable_final_tail_uses_accepted_c4a_plan(tmp_path_factory):
    case = _variant(tmp_path_factory, continuous_window_sec=299.9)
    inputs = _pass_inputs(case)
    segment_plan = inputs[5]
    assert [d.sample_count for d in segment_plan.descriptors] == [2999, 3001]
    assert segment_plan.descriptors[-1].absorbed_short_tail
    traversal = _run(inputs)
    results = list(traversal)
    assert [item.stop_target_index - item.start_target_index for item in results] == [
        2999,
        3001,
    ]
    assert traversal.completion.corrected_segment_count == 2


def test_mixed_per_roi_strategies_are_delegated_to_c4b(accepted_case):
    inputs = _pass_inputs(accepted_case)
    results = list(_run(inputs))
    for corrected in results:
        kinds = [item.reference_kind for item in corrected.per_roi_results]
        assert kinds == [c4b.REFERENCE_FITTED_CONTROL, c4b.REFERENCE_SIGNAL_DERIVED_F0]


def test_provisional_segments_and_completion_are_immutable(accepted_case):
    inputs = _pass_inputs(accepted_case)
    results = list(_run(inputs))
    with pytest.raises(FrozenInstanceError):
        results[0].segment_index = 99
    completion = _run(inputs).__iter__()
    remaining = list(completion)
    final = completion.completion
    with pytest.raises(FrozenInstanceError):
        final.corrected_segment_count = 0
    assert len(remaining) == len(results)


def test_completion_identity_is_deterministic(accepted_case):
    inputs = _pass_inputs(accepted_case)
    first_results = list(_run(inputs))
    first = _run(inputs)
    list(first)
    second = _run(inputs)
    list(second)
    assert first.completion.completion_identity == second.completion.completion_identity
    assert [item.result_identity for item in first_results] == [
        item.result_identity for item in list(_run(inputs))
    ]


def test_completion_unavailable_until_normal_exhaustion(accepted_case):
    inputs = _pass_inputs(accepted_case)
    traversal = _run(inputs)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as exc_info:
        traversal.completion
    assert exc_info.value.category == "completion_not_available"
    iterator = iter(traversal)
    next(iterator)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        traversal.completion
    for _ in iterator:
        pass
    assert traversal.state == "completed"
    assert traversal.completion.completion_state == "complete_all_segments_verified"


# ---------------------------------------------------------------------------
# Authority mismatches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation",
    [
        lambda i: {"binding": replace(i[0], draft_plan_identity="0" * 64)},
        lambda i: {"grid": replace(i[1], target_grid_identity="0" * 64)},
        lambda i: {"segment_plan": replace(i[5], plan_identity="0" * 64)},
        lambda i: {"draft": replace(i[2], continuous_window_sec=i[2].continuous_window_sec + 1.0)},
        lambda i: {"contract": replace(i[3], contract_version="substituted")},
        lambda i: {"f0": replace(i[6], authority_identity="0" * 64)},
        lambda i: {"f0": replace(i[6], recording_identity="0" * 64)},
    ],
)
def test_authority_mismatches_are_refused_before_traversal(accepted_case, mutation):
    inputs = _pass_inputs(accepted_case)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as exc_info:
        _run(inputs, **mutation(inputs))
    assert exc_info.value.category == "accepted_correction_binding_mismatch"


def test_c2_block_plan_type_mismatch_is_refused(accepted_case):
    inputs = _pass_inputs(accepted_case)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as exc_info:
        _run(inputs, block_plan=object())
    assert exc_info.value.category == "accepted_correction_binding_mismatch"


# ---------------------------------------------------------------------------
# Traversal integrity
# ---------------------------------------------------------------------------


def _consume_real_raw_segments(inputs):
    binding, grid, draft, contract, block_plan, segment_plan, f0 = inputs
    projected = iter_project_guided_continuous_rwd_blocks(binding, grid, block_plan)
    return list(
        c4a.iter_assemble_guided_continuous_rwd_correction_segments(
            binding,
            grid,
            block_plan,
            segment_plan,
            projected,
            accepted_draft=draft,
            startup_mapping_contract=contract,
        )
    )


def _patched_assembler(monkeypatch, sequence_fn):
    def fake_assemble(*args, **kwargs):
        for item in sequence_fn():
            yield item

    monkeypatch.setattr(
        subject, "iter_assemble_guided_continuous_rwd_correction_segments", fake_assemble
    )


def test_duplicate_segment_is_rejected(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    raws = _consume_real_raw_segments(inputs)
    _patched_assembler(monkeypatch, lambda: [raws[0], raws[0], *raws[1:]])
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        list(_run(inputs))


def test_skipped_segment_is_rejected(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    raws = _consume_real_raw_segments(inputs)
    _patched_assembler(monkeypatch, lambda: [raws[0], *raws[2:]])
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        list(_run(inputs))


def test_out_of_order_segment_is_rejected(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    raws = _consume_real_raw_segments(inputs)
    _patched_assembler(monkeypatch, lambda: [raws[1], raws[0], *raws[2:]])
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        list(_run(inputs))


def test_overlapping_segment_is_rejected(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    raws = _consume_real_raw_segments(inputs)
    overlapped = replace(
        raws[1], start_target_index=raws[1].start_target_index - 5
    )
    _patched_assembler(monkeypatch, lambda: [raws[0], overlapped, *raws[2:]])
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        list(_run(inputs))


def test_target_index_gap_is_rejected(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    raws = _consume_real_raw_segments(inputs)
    gapped = replace(raws[1], start_target_index=raws[1].start_target_index + 5)
    _patched_assembler(monkeypatch, lambda: [raws[0], gapped, *raws[2:]])
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        list(_run(inputs))


def test_incomplete_final_coverage_is_rejected(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    raws = _consume_real_raw_segments(inputs)
    _patched_assembler(monkeypatch, lambda: raws[:-1])
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        list(_run(inputs))


def test_unexpected_extra_segment_is_rejected(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    raws = _consume_real_raw_segments(inputs)
    extra = replace(
        raws[-1],
        segment_index=raws[-1].segment_index + 1,
        start_target_index=raws[-1].stop_target_index,
        stop_target_index=raws[-1].stop_target_index + raws[-1].stop_target_index
        - raws[-1].start_target_index,
    )
    _patched_assembler(monkeypatch, lambda: [*raws, extra])
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        list(_run(inputs))


def test_segment_range_inconsistent_with_descriptor_is_rejected(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    raws = _consume_real_raw_segments(inputs)
    truncated = replace(
        raws[0],
        stop_target_index=raws[0].stop_target_index - 1,
        target_elapsed_seconds=raws[0].target_elapsed_seconds[:-1],
        control_values=raws[0].control_values[:-1],
        signal_values=raws[0].signal_values[:-1],
    )
    _patched_assembler(monkeypatch, lambda: [truncated, *raws[1:]])
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        list(_run(inputs))


def test_noncanonical_roi_order_on_raw_segment_is_rejected(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    raws = _consume_real_raw_segments(inputs)
    reordered = replace(
        raws[0],
        included_roi_ids=tuple(reversed(raws[0].included_roi_ids)),
        control_values=raws[0].control_values[:, ::-1].copy(),
        signal_values=raws[0].signal_values[:, ::-1].copy(),
    )
    _patched_assembler(monkeypatch, lambda: [reordered, *raws[1:]])
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        list(_run(inputs))


def test_corrected_segment_with_wrong_upstream_identity_is_rejected(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)

    real_correct = subject.correct_guided_continuous_rwd_segment
    calls = {"count": 0}

    def tampering_correct(*args, **kwargs):
        result = real_correct(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] == 1:
            tampered = replace(
                result, target_grid_identity="0" * 64, result_identity=""
            )
            tampered = replace(
                tampered,
                result_identity=c4b._compute_result_identity(tampered),
            )
            return tampered
        return result

    monkeypatch.setattr(subject, "correct_guided_continuous_rwd_segment", tampering_correct)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as exc_info:
        list(_run(inputs))
    assert exc_info.value.category == "segment_identity_mismatch"


# ---------------------------------------------------------------------------
# Late failure
# ---------------------------------------------------------------------------


def test_late_c3b_failure_after_yielded_segments_leaves_no_completion(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)

    real_iter_project = subject.iter_project_guided_continuous_rwd_blocks

    def flaky_iter_project(review_binding, target_grid, block_plan, **kwargs):
        inner = real_iter_project(review_binding, target_grid, block_plan, **kwargs)

        def generator():
            count = 0
            for block in inner:
                count += 1
                yield block
                if count == 1:
                    raise RuntimeError("simulated late C3b failure")

        return generator()

    monkeypatch.setattr(subject, "iter_project_guided_continuous_rwd_blocks", flaky_iter_project)
    traversal = _run(inputs)
    produced = []
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        for corrected in traversal:
            produced.append(corrected)
    # The exact required sequence: one or more provisional segments were
    # yielded before the simulated late C3b failure, not zero.
    assert produced
    assert traversal.state == "failed"
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as exc_info:
        traversal.completion
    assert exc_info.value.category == "completion_not_available"
    for item in produced:
        assert isinstance(item, c4b.GuidedContinuousRwdCorrectedSegment)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancellation_before_traversal_returns_no_completion(accepted_case):
    inputs = _pass_inputs(accepted_case)
    traversal = _run(inputs, cancellation=lambda: True)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as exc_info:
        list(traversal)
    assert exc_info.value.category == "segment_correction_pass_interrupted"
    assert traversal.state == "cancelled"
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError):
        traversal.completion


def test_cancellation_between_segments_returns_no_completion(accepted_case):
    inputs = _pass_inputs(accepted_case)
    produced = []

    def cancel_after_first_segment():
        return len(produced) >= 1

    traversal = _run(inputs, cancellation=cancel_after_first_segment)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as exc_info:
        for corrected in traversal:
            produced.append(corrected)
    assert len(produced) >= 1
    # Cancellation may be detected by C4c's own checkpoint or by a nested
    # C3b/C4a checkpoint reached first; either way it must translate to the
    # single precise "cancelled" state, never "failed".
    assert exc_info.value.category == "segment_correction_pass_interrupted"
    assert traversal.state == "cancelled"
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as completion_exc:
        traversal.completion
    assert completion_exc.value.category == "completion_not_available"


def test_cancellation_at_finalization_checkpoint_is_precisely_isolated(accepted_case):
    """Cancellation requested only at the pre-publication finalization check.

    _finalize()'s own ``_check_cancellation`` call is deterministically the
    very last cancellation checkpoint reached during one fully successful
    traversal (every earlier checkpoint -- C3b block reads, C4a segment
    assembly, C4c's own per-segment checks -- runs strictly before it). A
    callback that returns False for every call except the last one during an
    identical run therefore isolates cancellation to that specific
    checkpoint, never to earlier segment processing.
    """
    inputs = _pass_inputs(accepted_case)

    total_calls = 0

    def counting_only():
        nonlocal total_calls
        total_calls += 1
        return False

    list(_run(inputs, cancellation=counting_only))
    assert total_calls > 0

    calls = 0

    def cancel_on_last_call():
        nonlocal calls
        calls += 1
        return calls >= total_calls

    traversal = _run(inputs, cancellation=cancel_on_last_call)
    produced = []
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as exc_info:
        for corrected in traversal:
            produced.append(corrected)
    segment_count = inputs[5].segment_count
    assert len(produced) == segment_count
    assert exc_info.value.category == "segment_correction_pass_interrupted"
    assert traversal.state == "cancelled"
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as completion_exc:
        traversal.completion
    assert completion_exc.value.category == "completion_not_available"

    # Retrying next() after terminal cancellation must not retry finalization
    # or ever publish completion.
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as retry_exc:
        next(traversal)
    assert retry_exc.value.category == "pass_already_terminal"
    assert traversal.state == "cancelled"
    with pytest.raises(subject.GuidedContinuousRwdCorrectionPassError) as completion_exc_2:
        traversal.completion
    assert completion_exc_2.value.category == "completion_not_available"


def test_next_on_pending_traversal_begins_it_like_iter(accepted_case):
    inputs = _pass_inputs(accepted_case)
    traversal = _run(inputs)
    assert traversal.state == "pending"
    first = next(traversal)
    assert traversal.state == "running"
    assert first.segment_index == 0
    remaining = list(traversal)
    assert len(remaining) == inputs[5].segment_count - 1
    assert traversal.state == "completed"
    assert traversal.completion.corrected_segment_count == inputs[5].segment_count


# ---------------------------------------------------------------------------
# C4b delegation
# ---------------------------------------------------------------------------


def test_c4c_delegates_to_c4b_and_adds_no_correction_mathematics(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    calls = []
    real_correct = subject.correct_guided_continuous_rwd_segment

    def spy(*args, **kwargs):
        calls.append(1)
        return real_correct(*args, **kwargs)

    monkeypatch.setattr(subject, "correct_guided_continuous_rwd_segment", spy)
    results = list(_run(inputs))
    assert len(calls) == len(results) == inputs[5].segment_count


# ---------------------------------------------------------------------------
# C2 partition independence
# ---------------------------------------------------------------------------


def test_c2_partition_independence(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)

    # block_count is a live property of the current MAXIMUM_OWNED_SAMPLES_PER_BLOCK
    # global, not a value frozen into the plan instance, so the "large" partition
    # must be captured (built and run) before the constant is ever patched.
    original_block_count = inputs[4].block_count
    large_results = list(_run(inputs))
    large_completion = _run(inputs)
    list(large_completion)

    monkeypatch.setattr(block_subject, "MAXIMUM_OWNED_SAMPLES_PER_BLOCK", 733)
    small_block_plan = block_subject.build_guided_continuous_rwd_block_plan(inputs[1])
    assert small_block_plan.block_count != original_block_count
    small_inputs = (
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        small_block_plan,
        inputs[5],
        inputs[6],
    )
    small_results = list(_run(small_inputs))
    small_completion = _run(small_inputs)
    list(small_completion)

    assert [item.result_identity for item in small_results] == [
        item.result_identity for item in large_results
    ]
    assert small_completion.completion.completion_identity == large_completion.completion.completion_identity


# ---------------------------------------------------------------------------
# One fresh second pass
# ---------------------------------------------------------------------------


def test_opens_exactly_one_fresh_c3b_traversal(accepted_case, monkeypatch):
    inputs = _pass_inputs(accepted_case)
    real_iter_project = subject.iter_project_guided_continuous_rwd_blocks
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real_iter_project(*args, **kwargs)

    monkeypatch.setattr(subject, "iter_project_guided_continuous_rwd_blocks", spy)
    list(_run(inputs))
    assert len(calls) == 1
