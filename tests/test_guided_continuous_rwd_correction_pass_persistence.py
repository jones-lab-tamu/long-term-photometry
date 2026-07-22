from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from photometry_pipeline import guided_continuous_rwd_block_plan as block_subject
from photometry_pipeline import guided_continuous_rwd_correction_pass as c4c
from photometry_pipeline import guided_continuous_rwd_correction_pass_persistence as subject
from photometry_pipeline import guided_continuous_rwd_correction_segments as c4a
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    evaluate_continuous_rwd_timestamp_continuity,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    build_guided_continuous_rwd_review_binding,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    build_guided_continuous_rwd_target_grid,
)
from photometry_pipeline.guided_execution_payloads import (
    build_guided_execution_startup_mapping_contract,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedPlanCorrectionChoice,
)
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_attrs,
    load_cache_chunk_fields,
    open_phasic_cache,
)
from photometry_pipeline.io.rwd_continuous_projection_reader import (
    iter_project_guided_continuous_rwd_blocks,
)
from photometry_pipeline.io.rwd_continuous_source import (
    inspect_continuous_rwd_acquisition_folder,
)


def _values(indices, *, phase=0.0):
    time = indices / 10.0
    control1 = 2.0 + 0.15 * np.cos(0.17 * time + phase)
    control2 = 3.0 + 0.12 * np.sin(0.11 * time + 0.4 + phase)
    signal1 = 5.0 + 1.6 * control1 + 0.08 * np.sin(0.7 * time)
    signal2 = 7.0 + 0.30 * np.cos(0.23 * time) + 0.04 * np.sin(1.3 * time)
    return (
        time.astype(np.float64),
        np.column_stack((control1, control2)).astype(np.float64),
        np.column_stack((signal1, signal2)).astype(np.float64),
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


def _build_case(folder, *, continuous_window_sec=20.0, phase=0.0):
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "Fluorescence.csv"
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(6001):
        time, control, signal = _values(np.array([index], dtype=float), phase=phase)
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
    strategies = {"ROI1": "global_linear_regression", "ROI2": "signal_only_f0"}
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
        feature_event_values={},
    )
    binding = build_guided_continuous_rwd_review_binding(
        draft,
        recording=recording,
        continuity_evaluation=continuity,
        current_source_path=source,
    )
    grid = build_guided_continuous_rwd_target_grid(recording, continuity)
    contract = build_guided_execution_startup_mapping_contract()
    return binding, grid, draft, contract, str(source)


def _pass_inputs(case):
    binding, grid, draft, contract, source = case
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
    return binding, grid, draft, contract, block_plan, segment_plan, f0, source


def _traversal(inputs):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    return c4c.iterate_guided_continuous_rwd_corrected_segments(
        binding,
        grid,
        block_plan,
        segment_plan,
        f0,
        accepted_draft=draft,
        startup_mapping_contract=contract,
    )


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_d1") / "recording"
    return _build_case(folder)


@pytest.fixture(scope="module")
def real_config(accepted_case):
    from photometry_pipeline.guided_continuous_rwd_segment_correction import (
        _resolve_segment_correction_settings,
    )

    _binding, _grid, _draft, contract, _source = accepted_case
    config, _identity = _resolve_segment_correction_settings(contract)
    return config


def _expected_bindings_from_real_run(inputs):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    return c4c._resolve_expected_bindings(
        binding, grid, block_plan, segment_plan, f0, draft, contract
    )


def _fake_traversal(segments, expected):
    def generator():
        for item in segments:
            yield item

    return c4c.GuidedContinuousRwdCorrectionPassTraversal(
        generator(), expected=expected, cancellation_requested=None
    )


def _real_segments(inputs):
    return list(_traversal(inputs))


def _assert_no_artifact(output_path):
    import os

    assert not os.path.exists(output_path)
    assert not os.path.exists(output_path + ".tmp")


# ---------------------------------------------------------------------------
# Successful write and reader round-trip
# ---------------------------------------------------------------------------


def test_multiple_segments_write_and_round_trip(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    traversal = _traversal(inputs)
    output_path = str(tmp_path / "continuous_phasic_trace_cache.h5")

    completion = subject.persist_guided_continuous_rwd_correction_pass(
        traversal,
        review_binding=inputs[0],
        target_grid=inputs[1],
        output_path=output_path,
        config=real_config,
    )

    segment_plan = inputs[5]
    grid = inputs[1]
    assert completion.corrected_segment_count == segment_plan.segment_count
    assert completion.target_sample_count == grid.target_sample_count

    import os

    assert os.path.exists(output_path)
    assert not os.path.exists(output_path + ".tmp")

    cache = open_phasic_cache(output_path)
    try:
        rois = list_cache_rois(cache)
        assert rois == ["ROI1", "ROI2"]
        chunk_ids = list_cache_chunk_ids(cache)
        assert chunk_ids == list(range(segment_plan.segment_count))

        first_time, first_dff = load_cache_chunk_fields(
            cache, "ROI1", 0, ["time_sec", "dff"]
        )
        assert first_time[0] == 0.0
        assert first_dff.shape == first_time.shape
        assert np.all(np.isfinite(first_dff))

        last_id = chunk_ids[-1]
        last_attrs = load_cache_chunk_attrs(cache, "ROI2", last_id)
        assert last_attrs["acquisition_mode"] == "continuous"
        assert last_attrs["window_index"] == last_id

        assert cache["meta"].attrs["continuous_acquisition_mode"] == "continuous"
        assert (
            cache["meta"].attrs["continuous_completion_identity"]
            == completion.completion_identity
        )
        assert (
            int(cache["meta"].attrs["continuous_target_sample_count"])
            == grid.target_sample_count
        )
        assert (
            int(cache["meta"].attrs["continuous_corrected_segment_count"])
            == segment_plan.segment_count
        )

        # Correction provenance for both strategy families made it through.
        roi1_attrs = load_cache_chunk_attrs(cache, "ROI1", 0)
        assert roi1_attrs["correction_strategy_family"] == "dynamic_fit"
        assert roi1_attrs["correction_selected_strategy"] == "global_linear_regression"
        roi2_attrs = load_cache_chunk_attrs(cache, "ROI2", 0)
        assert roi2_attrs["correction_strategy_family"] == "signal_only_f0"
        assert "correction_qc_json" in roi2_attrs
    finally:
        cache.close()


def test_one_segment_recording_writes_a_single_chunk(real_config, tmp_path, tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_d1_single") / "recording"
    case = _build_case(folder, continuous_window_sec=600.0)
    inputs = _pass_inputs(case)
    traversal = _traversal(inputs)
    output_path = str(tmp_path / "single.h5")

    completion = subject.persist_guided_continuous_rwd_correction_pass(
        traversal,
        review_binding=inputs[0],
        target_grid=inputs[1],
        output_path=output_path,
        config=real_config,
    )
    assert completion.corrected_segment_count == 1

    cache = open_phasic_cache(output_path)
    try:
        assert list_cache_chunk_ids(cache) == [0]
    finally:
        cache.close()


def test_completion_authority_returned_matches_traversal(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    traversal = _traversal(inputs)
    output_path = str(tmp_path / "identity_check.h5")
    completion = subject.persist_guided_continuous_rwd_correction_pass(
        traversal,
        review_binding=inputs[0],
        target_grid=inputs[1],
        output_path=output_path,
        config=real_config,
    )
    assert traversal.completion is completion


def test_round_trip_stores_authority_derived_source_and_rate(
    accepted_case, real_config, tmp_path
):
    """fs_hz and source_file in the artifact must come from the accepted
    review_binding/target_grid authorities, not any caller-invented value
    (there is no longer a way for a caller to supply one directly)."""
    inputs = _pass_inputs(accepted_case)
    traversal = _traversal(inputs)
    review_binding, target_grid = inputs[0], inputs[1]
    output_path = str(tmp_path / "authority_round_trip.h5")
    subject.persist_guided_continuous_rwd_correction_pass(
        traversal,
        review_binding=review_binding,
        target_grid=target_grid,
        output_path=output_path,
        config=real_config,
    )
    expected_fs_hz = 1.0 / float(target_grid.cadence_fraction)
    expected_source_file = review_binding.recording.source.fluorescence_path_canonical

    cache = open_phasic_cache(output_path)
    try:
        attrs = load_cache_chunk_attrs(cache, "ROI1", 0)
        assert attrs["fs_hz"] == pytest.approx(expected_fs_hz)
        assert attrs["source_file"] == expected_source_file
        last_attrs = load_cache_chunk_attrs(
            cache, "ROI1", list_cache_chunk_ids(cache)[-1]
        )
        assert last_attrs["fs_hz"] == pytest.approx(expected_fs_hz)
        assert last_attrs["source_file"] == expected_source_file
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# Bounded consumption: one segment in memory at a time
# ---------------------------------------------------------------------------


def test_segments_are_consumed_one_at_a_time_not_materialized(
    accepted_case, real_config, tmp_path
):
    inputs = _pass_inputs(accepted_case)
    traversal = _traversal(inputs)
    concurrent_outstanding = []
    max_concurrent = 0
    real_iter = type(traversal).__next__

    def spying_next(self):
        item = real_iter(self)
        concurrent_outstanding.append(item)
        nonlocal max_concurrent
        max_concurrent = max(max_concurrent, len(concurrent_outstanding))
        return item

    real_add_chunk = subject.Hdf5TraceCacheWriter.add_chunk

    def spying_add_chunk(self, chunk, chunk_id, source_file):
        assert len(concurrent_outstanding) <= 1, (
            "persist_guided_continuous_rwd_correction_pass must not accumulate "
            "more than one segment's data before writing it"
        )
        result = real_add_chunk(self, chunk, chunk_id, source_file)
        concurrent_outstanding.clear()
        return result

    traversal.__class__.__next__ = spying_next
    try:
        output_path = str(tmp_path / "bounded.h5")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(subject.Hdf5TraceCacheWriter, "add_chunk", spying_add_chunk)
            subject.persist_guided_continuous_rwd_correction_pass(
                traversal,
                review_binding=inputs[0],
                target_grid=inputs[1],
                output_path=output_path,
                config=real_config,
            )
    finally:
        traversal.__class__.__next__ = real_iter
    assert max_concurrent == 1


# ---------------------------------------------------------------------------
# Pre-traversal authority validation (fails before the .tmp file is created)
# ---------------------------------------------------------------------------


def test_review_binding_from_another_recording_is_rejected(
    accepted_case, real_config, tmp_path, tmp_path_factory
):
    inputs = _pass_inputs(accepted_case)
    traversal = _traversal(inputs)
    other_folder = tmp_path_factory.mktemp("cr1_d1_other_a") / "recording"
    other_case = _build_case(other_folder, phase=1.7)
    other_binding = other_case[0]
    assert (
        other_binding.recording.recording_identity
        != inputs[0].recording.recording_identity
    )
    output_path = str(tmp_path / "wrong_binding.h5")
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=other_binding,
            target_grid=inputs[1],
            output_path=output_path,
            config=real_config,
        )
    _assert_no_artifact(output_path)


def test_target_grid_from_another_recording_is_rejected(
    accepted_case, real_config, tmp_path, tmp_path_factory
):
    inputs = _pass_inputs(accepted_case)
    traversal = _traversal(inputs)
    other_folder = tmp_path_factory.mktemp("cr1_d1_other_b") / "recording"
    other_case = _build_case(other_folder, phase=2.3)
    other_grid = other_case[1]
    assert other_grid.recording_identity != inputs[1].recording_identity
    output_path = str(tmp_path / "wrong_grid_authority.h5")
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=other_grid,
            output_path=output_path,
            config=real_config,
        )
    _assert_no_artifact(output_path)


def test_review_binding_with_wrong_source_content_identity_is_rejected(
    accepted_case, real_config, tmp_path
):
    """recording_identity is derived from source_content_identity, so a
    tampered source_content_identity alone (recording_identity left intact)
    passes the pre-traversal same-recording check and must be caught when
    the first segment's true source_content_identity is compared against it.
    """
    inputs = _pass_inputs(accepted_case)
    traversal = _traversal(inputs)
    binding = inputs[0]
    tampered_source = replace(
        binding.recording.source, source_content_identity="0" * 64
    )
    tampered_recording = replace(binding.recording, source=tampered_source)
    tampered_binding = replace(binding, recording=tampered_recording)
    output_path = str(tmp_path / "wrong_source_content.h5")
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=tampered_binding,
            target_grid=inputs[1],
            output_path=output_path,
            config=real_config,
        )
    _assert_no_artifact(output_path)


def test_substituted_target_grid_identity_is_rejected(accepted_case, real_config, tmp_path):
    """target_grid_identity is not part of the pre-traversal same-recording
    check (only recording_identity is), so this must be caught per-segment.
    """
    inputs = _pass_inputs(accepted_case)
    traversal = _traversal(inputs)
    tampered_grid = replace(inputs[1], target_grid_identity="0" * 64)
    output_path = str(tmp_path / "substituted_grid_identity.h5")
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=tampered_grid,
            output_path=output_path,
            config=real_config,
        )
    _assert_no_artifact(output_path)


# ---------------------------------------------------------------------------
# First-segment authority validation (the first segment must not be trusted
# merely because it agrees with itself -- it must match the supplied
# review_binding/target_grid authorities).
# ---------------------------------------------------------------------------


def test_first_segment_wrong_recording_identity_is_rejected(
    accepted_case, real_config, tmp_path
):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    tampered = [replace(segments[0], recording_identity="0" * 64), *segments[1:]]
    traversal = _fake_traversal(tampered, expected)
    output_path = str(tmp_path / "first_seg_wrong_recording.h5")
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=output_path,
            config=real_config,
        )
    _assert_no_artifact(output_path)


def test_first_segment_wrong_source_content_identity_is_rejected(
    accepted_case, real_config, tmp_path
):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    tampered = [replace(segments[0], source_content_identity="0" * 64), *segments[1:]]
    traversal = _fake_traversal(tampered, expected)
    output_path = str(tmp_path / "first_seg_wrong_source.h5")
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=output_path,
            config=real_config,
        )
    _assert_no_artifact(output_path)


def test_first_segment_wrong_target_grid_identity_is_rejected(
    accepted_case, real_config, tmp_path
):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    tampered = [replace(segments[0], target_grid_identity="0" * 64), *segments[1:]]
    traversal = _fake_traversal(tampered, expected)
    output_path = str(tmp_path / "first_seg_wrong_grid.h5")
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=output_path,
            config=real_config,
        )
    _assert_no_artifact(output_path)


# ---------------------------------------------------------------------------
# Traversal integrity (defense-in-depth via a bare fake traversal, so C4c's
# own internal validation is bypassed and the writer's own checks are what
# is actually exercised)
# ---------------------------------------------------------------------------


def test_wrong_target_grid_identity_segment_is_rejected(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    tampered = [
        segments[0],
        replace(segments[1], target_grid_identity="0" * 64),
        *segments[2:],
    ]
    traversal = _fake_traversal(tampered, expected)
    output_path = str(tmp_path / "wrong_grid.h5")
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=output_path,
            config=real_config,
        )
    _assert_no_artifact(output_path)


def test_noncanonical_roi_order_segment_is_rejected(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    reordered = replace(
        segments[1], included_roi_ids=tuple(reversed(segments[1].included_roi_ids))
    )
    traversal = _fake_traversal([segments[0], reordered, *segments[2:]], expected)
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=str(tmp_path / "wrong_roi.h5"),
            config=real_config,
        )


def test_out_of_order_segment_is_rejected(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    traversal = _fake_traversal([segments[1], segments[0], *segments[2:]], expected)
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=str(tmp_path / "out_of_order.h5"),
            config=real_config,
        )


def test_duplicate_segment_is_rejected(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    traversal = _fake_traversal([segments[0], segments[0], *segments[1:]], expected)
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=str(tmp_path / "duplicate.h5"),
            config=real_config,
        )


def test_skipped_segment_is_rejected(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    traversal = _fake_traversal([segments[0], *segments[2:]], expected)
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=str(tmp_path / "skipped.h5"),
            config=real_config,
        )


def test_overlapping_segment_is_rejected(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    overlapped = replace(
        segments[1], start_target_index=segments[1].start_target_index - 5
    )
    traversal = _fake_traversal([segments[0], overlapped, *segments[2:]], expected)
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=str(tmp_path / "overlap.h5"),
            config=real_config,
        )


def test_incomplete_traversal_is_rejected(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    traversal = _fake_traversal(segments[:-1], expected)
    output_path = str(tmp_path / "incomplete.h5")
    with pytest.raises(Exception):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=output_path,
            config=real_config,
        )
    _assert_no_artifact(output_path)


def test_extra_segment_is_rejected(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)
    extra = replace(
        segments[-1],
        segment_index=segments[-1].segment_index + 1,
        start_target_index=segments[-1].stop_target_index,
        stop_target_index=segments[-1].stop_target_index
        + (segments[-1].stop_target_index - segments[-1].start_target_index),
    )
    traversal = _fake_traversal([*segments, extra], expected)
    with pytest.raises(Exception):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=str(tmp_path / "extra.h5"),
            config=real_config,
        )


# ---------------------------------------------------------------------------
# Late failure and cancellation
# ---------------------------------------------------------------------------


def test_late_traversal_failure_after_segments_written_leaves_no_artifact(
    accepted_case, real_config, tmp_path, monkeypatch
):
    inputs = _pass_inputs(accepted_case)

    real_correct = c4c.correct_guided_continuous_rwd_segment
    calls = {"count": 0}

    def flaky_correct(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("simulated late correction failure")
        return real_correct(*args, **kwargs)

    monkeypatch.setattr(c4c, "correct_guided_continuous_rwd_segment", flaky_correct)
    traversal = _traversal(inputs)
    output_path = str(tmp_path / "late_failure.h5")
    with pytest.raises(Exception):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=output_path,
            config=real_config,
        )
    assert calls["count"] >= 3
    _assert_no_artifact(output_path)


def test_cancellation_after_segments_written_leaves_no_artifact(
    accepted_case, real_config, tmp_path, monkeypatch
):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = _pass_inputs(
        accepted_case
    )
    written = {"count": 0}
    real_add_chunk = subject.Hdf5TraceCacheWriter.add_chunk

    def counting_add_chunk(self, chunk, chunk_id, source_file):
        result = real_add_chunk(self, chunk, chunk_id, source_file)
        written["count"] += 1
        return result

    monkeypatch.setattr(subject.Hdf5TraceCacheWriter, "add_chunk", counting_add_chunk)

    def cancel_after_first_segment():
        return written["count"] >= 1

    traversal = c4c.iterate_guided_continuous_rwd_corrected_segments(
        binding,
        grid,
        block_plan,
        segment_plan,
        f0,
        accepted_draft=draft,
        startup_mapping_contract=contract,
        cancellation_requested=cancel_after_first_segment,
    )
    output_path = str(tmp_path / "cancelled.h5")
    with pytest.raises(Exception):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=binding,
            target_grid=grid,
            output_path=output_path,
            config=real_config,
        )
    assert written["count"] >= 1
    assert traversal.state in {"cancelled", "failed"}
    _assert_no_artifact(output_path)


def test_completion_unavailable_after_finalization_failure_leaves_no_artifact(
    accepted_case, real_config, tmp_path
):
    """All apparent segments written, but the C4c completion authority never
    becomes available (traversal never reaches "completed") -> no artifact.
    """
    inputs = _pass_inputs(accepted_case)
    segments = _real_segments(inputs)
    expected = _expected_bindings_from_real_run(inputs)

    def generator():
        for item in segments:
            yield item
        raise RuntimeError("simulated finalization failure after all segments yielded")

    traversal = c4c.GuidedContinuousRwdCorrectionPassTraversal(
        generator(), expected=expected, cancellation_requested=None
    )
    output_path = str(tmp_path / "finalize_failure.h5")
    with pytest.raises(Exception):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=output_path,
            config=real_config,
        )
    assert traversal.state != "completed"
    with pytest.raises(Exception):
        traversal.completion
    _assert_no_artifact(output_path)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_non_traversal_input(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            [1, 2, 3],
            review_binding=inputs[0],
            target_grid=inputs[1],
            output_path=str(tmp_path / "bad.h5"),
            config=real_config,
        )


def test_rejects_wrong_type_review_binding(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    traversal = _traversal(inputs)
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=object(),
            target_grid=inputs[1],
            output_path=str(tmp_path / "bad_binding_type.h5"),
            config=real_config,
        )


def test_rejects_wrong_type_target_grid(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    traversal = _traversal(inputs)
    with pytest.raises(subject.GuidedContinuousRwdPersistenceError):
        subject.persist_guided_continuous_rwd_correction_pass(
            traversal,
            review_binding=inputs[0],
            target_grid=object(),
            output_path=str(tmp_path / "bad_grid_type.h5"),
            config=real_config,
        )
