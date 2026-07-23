from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pandas as pd
import pytest

from photometry_pipeline import guided_continuous_rwd_combined_run as subject
from photometry_pipeline import guided_continuous_rwd_phasic_detection as detection_module
from photometry_pipeline.guided_continuous_rwd_phasic_run import (
    execute_guided_continuous_rwd_phasic_run,
)
from photometry_pipeline.guided_continuous_rwd_tonic_run import (
    execute_guided_continuous_rwd_tonic_run,
)
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_fields,
    open_phasic_cache,
    open_tonic_cache,
)
from photometry_pipeline.run_completion_contract import (
    TERMINAL_SUCCESS_CURRENT,
    classify_run_terminal_state,
)

# Reuse the D1 module's synthetic-recording builders (same accepted
# construction path already used by D1/D2/D3a/D3b-B's own test suites).
from tests.test_guided_continuous_rwd_correction_pass_persistence import (
    _build_case,
    _pass_inputs,
)


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_d4") / "recording"
    return _build_case(folder, continuous_window_sec=90.0)


@pytest.fixture(scope="module")
def real_config(accepted_case):
    from photometry_pipeline.guided_continuous_rwd_segment_correction import (
        _resolve_segment_correction_settings,
    )

    _binding, _grid, _draft, contract, _source = accepted_case
    config, _identity = _resolve_segment_correction_settings(contract)
    # Lower the detection threshold so the synthetic recording's small
    # periodic correction residual actually crosses it, giving genuine,
    # spread-out phasic events to exercise multi-window conservation and
    # tonic/phasic equivalence checks (mirrors tests/test_guided_continuous_
    # rwd_phasic_run.py's own real_config fixture).
    return dataclasses.replace(
        config,
        peak_threshold_method="percentile",
        peak_threshold_percentile=50.0,
        peak_min_distance_sec=1.0,
    )


def _run_combined(inputs, real_config, output_base, **kwargs):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    return subject.execute_guided_continuous_rwd_combined_run(
        binding,
        grid,
        block_plan,
        segment_plan,
        f0,
        accepted_draft=draft,
        startup_mapping_contract=contract,
        output_base=str(output_base),
        config=real_config,
        **kwargs,
    )


def _run_tonic_only(inputs, real_config, output_base, **kwargs):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    return execute_guided_continuous_rwd_tonic_run(
        binding,
        grid,
        block_plan,
        segment_plan,
        f0,
        accepted_draft=draft,
        startup_mapping_contract=contract,
        output_base=str(output_base),
        config=real_config,
        **kwargs,
    )


def _run_phasic_only(inputs, real_config, output_base, **kwargs):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    return execute_guided_continuous_rwd_phasic_run(
        binding,
        grid,
        block_plan,
        segment_plan,
        f0,
        accepted_draft=draft,
        startup_mapping_contract=contract,
        output_base=str(output_base),
        config=real_config,
        **kwargs,
    )


def _read_tonic_summary(run_dir, roi):
    path = os.path.join(run_dir, roi, "tables", "continuous_tonic_window_summary.csv")
    return pd.read_csv(path)


def _read_phasic_summary(run_dir, roi):
    path = os.path.join(run_dir, roi, "tables", "continuous_phasic_window_summary.csv")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Successful combined run
# ---------------------------------------------------------------------------


def test_successful_multi_chunk_combined_run_publishes_current_success(
    accepted_case, real_config, tmp_path
):
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    result = _run_combined(inputs, real_config, tmp_path)

    assert result.terminal_state == TERMINAL_SUCCESS_CURRENT
    included = list(binding.recording.roi.included_roi_ids)
    assert set(result.tonic_summary_paths) == set(included)
    assert set(result.phasic_summary_paths) == set(included)
    assert all(count >= 1 for count in result.tonic_summary_row_counts.values())
    assert all(count >= 1 for count in result.phasic_summary_row_counts.values())

    tonic_analysis_dir = os.path.join(result.run_dir, "_analysis", "tonic_out")
    phasic_analysis_dir = os.path.join(result.run_dir, "_analysis", "phasic_out")
    assert os.path.isfile(os.path.join(tonic_analysis_dir, "run_report.json"))
    assert os.path.isfile(os.path.join(tonic_analysis_dir, "config_used.yaml"))
    assert os.path.isfile(os.path.join(phasic_analysis_dir, "run_report.json"))
    assert os.path.isfile(os.path.join(phasic_analysis_dir, "config_used.yaml"))
    assert os.path.isfile(result.tonic_cache_path)
    assert os.path.isfile(result.phasic_cache_path)
    assert os.path.isfile(result.features_path)
    assert os.path.isfile(result.events_path)

    classification = classify_run_terminal_state(result.run_dir)
    assert classification.is_success
    assert classification.run_mode.get("tonic_analysis") is True
    assert classification.run_mode.get("phasic_analysis") is True
    assert classification.run_mode.get("feature_extraction_ran") is True
    assert classification.run_mode.get("acquisition_mode") == "continuous"
    assert classification.run_mode.get("chunked_input_processing") is False

    with open(os.path.join(result.run_dir, "run_report.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    narrative = report["summary"]["narrative"].lower()
    assert "tonic" in narrative
    assert "phasic" in narrative
    assert "completed" in narrative
    for internal_term in ("d3a", "d3b-a", "d3b-b", "kernel", "storage chunk"):
        assert internal_term not in narrative


# ---------------------------------------------------------------------------
# Single correction pass
# ---------------------------------------------------------------------------


def test_correction_executes_exactly_once(accepted_case, real_config, tmp_path, monkeypatch):
    real_persist = subject.persist_guided_continuous_rwd_correction_pass
    calls = {"count": 0}

    def counting_persist(*args, **kwargs):
        calls["count"] += 1
        return real_persist(*args, **kwargs)

    monkeypatch.setattr(subject, "persist_guided_continuous_rwd_correction_pass", counting_persist)

    inputs = _pass_inputs(accepted_case)
    result = _run_combined(inputs, real_config, tmp_path)

    assert calls["count"] == 1
    assert result.terminal_state == TERMINAL_SUCCESS_CURRENT


def test_shared_correction_cache_consumed_by_both_families(
    accepted_case, real_config, tmp_path, monkeypatch
):
    from photometry_pipeline import guided_continuous_rwd_phasic_run as phasic_module
    from photometry_pipeline import guided_continuous_rwd_tonic_run as tonic_module

    seen_paths: dict[str, set[str]] = {"tonic": set(), "phasic": set()}
    real_tonic_open = tonic_module.open_phasic_cache
    real_phasic_open = phasic_module.open_phasic_cache

    def tracking_tonic_open(path):
        seen_paths["tonic"].add(path)
        return real_tonic_open(path)

    def tracking_phasic_open(path):
        seen_paths["phasic"].add(path)
        return real_phasic_open(path)

    monkeypatch.setattr(tonic_module, "open_phasic_cache", tracking_tonic_open)
    monkeypatch.setattr(phasic_module, "open_phasic_cache", tracking_phasic_open)

    inputs = _pass_inputs(accepted_case)
    result = _run_combined(inputs, real_config, tmp_path)

    # Both families opened the one shared D1 corrected cache (phasic also
    # reopens its own just-written phasic_cache_path for validation -- the
    # membership check, not equality, is the correct proof here).
    assert result.corrected_cache_path in seen_paths["tonic"]
    assert result.corrected_cache_path in seen_paths["phasic"]


# ---------------------------------------------------------------------------
# Equivalence to the accepted tonic-only and phasic-only paths
# ---------------------------------------------------------------------------


def test_tonic_publication_matches_tonic_only_path(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    combined_base = tmp_path / "combined"
    tonic_base = tmp_path / "tonic_only"
    combined_base.mkdir()
    tonic_base.mkdir()

    combined_result = _run_combined(inputs, real_config, combined_base)
    tonic_result = _run_tonic_only(inputs, real_config, tonic_base)

    included = list(binding.recording.roi.included_roi_ids)
    combined_cache = open_tonic_cache(combined_result.tonic_cache_path)
    tonic_cache = open_tonic_cache(tonic_result.tonic_cache_path)
    try:
        chunk_ids = list_cache_chunk_ids(combined_cache)
        assert chunk_ids == list_cache_chunk_ids(tonic_cache)
        for roi in included:
            for chunk_id in chunk_ids:
                (combined_delta,) = load_cache_chunk_fields(
                    combined_cache, roi, chunk_id, ["deltaF"]
                )
                (tonic_delta,) = load_cache_chunk_fields(tonic_cache, roi, chunk_id, ["deltaF"])
                np.testing.assert_array_equal(combined_delta, tonic_delta)
    finally:
        combined_cache.close()
        tonic_cache.close()

    for roi in included:
        combined_df = _read_tonic_summary(combined_result.run_dir, roi).sort_values(
            "window_index"
        ).reset_index(drop=True)
        tonic_df = _read_tonic_summary(tonic_result.run_dir, roi).sort_values(
            "window_index"
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(combined_df, tonic_df)


def test_phasic_publication_matches_phasic_only_path(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    combined_base = tmp_path / "combined"
    phasic_base = tmp_path / "phasic_only"
    combined_base.mkdir()
    phasic_base.mkdir()

    combined_result = _run_combined(inputs, real_config, combined_base)
    phasic_result = _run_phasic_only(inputs, real_config, phasic_base)

    included = list(binding.recording.roi.included_roi_ids)
    for roi in included:
        c = combined_result.detection.per_roi[roi]
        p = phasic_result.detection.per_roi[roi]
        assert c.event_count == p.event_count
        np.testing.assert_array_equal(c.peak_global_times_sec, p.peak_global_times_sec)
        np.testing.assert_array_equal(c.peak_polarities, p.peak_polarities)

    combined_features = pd.read_csv(combined_result.features_path)
    phasic_features = pd.read_csv(phasic_result.features_path)
    pd.testing.assert_frame_equal(
        combined_features.reset_index(drop=True), phasic_features.reset_index(drop=True)
    )

    combined_events = pd.read_csv(combined_result.events_path)
    phasic_events = pd.read_csv(phasic_result.events_path)
    pd.testing.assert_frame_equal(
        combined_events.reset_index(drop=True), phasic_events.reset_index(drop=True)
    )

    for roi in included:
        combined_summary = _read_phasic_summary(combined_result.run_dir, roi)
        phasic_summary = _read_phasic_summary(phasic_result.run_dir, roi)
        assert int(combined_summary["event_count"].sum()) == int(
            phasic_summary["event_count"].sum()
        )


# ---------------------------------------------------------------------------
# Boundary events, count conservation, cross-family coherence, final tail
# ---------------------------------------------------------------------------


def test_phasic_event_conservation_no_duplication_across_windows(
    accepted_case, real_config, tmp_path
):
    """Every detected event -- including any that land exactly on a D1
    storage-chunk boundary -- must appear exactly once in the combined
    run's published phasic artifacts, and per-window counts must sum to
    D3b-A's own recording-global count."""
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    result = _run_combined(inputs, real_config, tmp_path)
    included = list(binding.recording.roi.included_roi_ids)

    features = pd.read_csv(result.features_path)
    events = pd.read_csv(result.events_path)
    total_detected = sum(result.detection.per_roi[roi].event_count for roi in included)
    assert total_detected > 0  # the tuned threshold must actually find events
    assert len(events) == total_detected

    for roi in included:
        roi_features = features[features["roi"] == roi]
        assert int(roi_features["peak_count"].sum()) == result.detection.per_roi[roi].event_count
        roi_events = events[events["roi"] == roi]
        assert roi_events["global_time_sec"].is_unique
        # Every event time is attributed to exactly one chunk_id.
        assert roi_events.groupby("global_time_sec")["chunk_id"].nunique().max() <= 1


def test_final_short_tail_represented_in_both_summary_families(
    accepted_case, real_config, tmp_path
):
    inputs = _pass_inputs(accepted_case)
    segment_plan = inputs[5]
    assert segment_plan.descriptors[-1].sample_count < segment_plan.nominal_segment_sample_count

    binding = inputs[0]
    result = _run_combined(inputs, real_config, tmp_path)
    for roi in binding.recording.roi.included_roi_ids:
        tonic_df = _read_tonic_summary(result.run_dir, roi).sort_values("window_index")
        phasic_df = _read_phasic_summary(result.run_dir, roi).sort_values("window_index")
        assert tonic_df.iloc[-1]["window_duration_sec"] < tonic_df.iloc[0]["window_duration_sec"]
        assert phasic_df.iloc[-1]["window_duration_sec"] < phasic_df.iloc[0]["window_duration_sec"]
        assert tonic_df.iloc[-1]["window_index"] == segment_plan.segment_count - 1
        assert phasic_df.iloc[-1]["window_index"] == segment_plan.segment_count - 1


def test_cross_family_roi_order_and_chunk_coverage_identical(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    result = _run_combined(inputs, real_config, tmp_path)

    tonic_cache = open_tonic_cache(result.tonic_cache_path)
    phasic_cache = open_phasic_cache(result.phasic_cache_path)
    try:
        assert list_cache_rois(tonic_cache) == list_cache_rois(phasic_cache)
        assert list_cache_chunk_ids(tonic_cache) == list_cache_chunk_ids(phasic_cache)
        assert list_cache_chunk_ids(tonic_cache) == list(
            range(result.completion.corrected_segment_count)
        )
    finally:
        tonic_cache.close()
        phasic_cache.close()


# ---------------------------------------------------------------------------
# Failure and cancellation
# ---------------------------------------------------------------------------


def test_failure_after_tonic_succeeds_before_phasic_completes_leaves_no_successful_run(
    accepted_case, real_config, tmp_path, monkeypatch
):
    inputs = _pass_inputs(accepted_case)

    def flaky_detect(*args, **kwargs):
        raise RuntimeError("simulated D3b-A failure after tonic succeeded")

    monkeypatch.setattr(subject, "detect_guided_continuous_rwd_phasic_features", flaky_detect)

    with pytest.raises(Exception):
        _run_combined(inputs, real_config, tmp_path)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    tonic_out_dir = os.path.join(str(run_dir), "_analysis", "tonic_out")
    assert os.path.isfile(os.path.join(tonic_out_dir, "tonic_trace_cache.h5"))
    binding = inputs[0]
    first_roi = binding.recording.roi.included_roi_ids[0]
    assert os.path.isfile(
        os.path.join(str(run_dir), first_roi, "tables", "continuous_tonic_window_summary.csv")
    )
    phasic_out_dir = os.path.join(str(run_dir), "_analysis", "phasic_out")
    assert not os.path.isfile(os.path.join(phasic_out_dir, "phasic_trace_cache.h5"))
    assert not os.path.exists(os.path.join(str(run_dir), "MANIFEST.json"))

    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "error"
    assert status["phase"] == "final"


def test_failure_after_both_families_exist_before_manifest_leaves_no_successful_run(
    accepted_case, real_config, tmp_path, monkeypatch
):
    inputs = _pass_inputs(accepted_case)

    def flaky_manifest_block(*args, **kwargs):
        raise RuntimeError("simulated manifest-build failure after both families published")

    monkeypatch.setattr(subject, "build_manifest_completion_block", flaky_manifest_block)

    with pytest.raises(Exception):
        _run_combined(inputs, real_config, tmp_path)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    tonic_out_dir = os.path.join(str(run_dir), "_analysis", "tonic_out")
    phasic_out_dir = os.path.join(str(run_dir), "_analysis", "phasic_out")
    assert os.path.isfile(os.path.join(tonic_out_dir, "tonic_trace_cache.h5"))
    assert os.path.isfile(os.path.join(phasic_out_dir, "phasic_trace_cache.h5"))
    assert os.path.isfile(os.path.join(phasic_out_dir, "features", "features.csv"))
    assert not os.path.exists(os.path.join(str(run_dir), "MANIFEST.json"))

    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "error"


def test_cancellation_during_correction_leaves_no_successful_run(
    accepted_case, real_config, tmp_path, monkeypatch
):
    written = {"count": 0}
    real_add_chunk = Hdf5TraceCacheWriter.add_chunk

    def counting_add_chunk(self, chunk, chunk_id, source_file):
        result = real_add_chunk(self, chunk, chunk_id, source_file)
        written["count"] += 1
        return result

    monkeypatch.setattr(Hdf5TraceCacheWriter, "add_chunk", counting_add_chunk)

    def cancel_after_first_segment():
        return written["count"] >= 1

    inputs = _pass_inputs(accepted_case)
    with pytest.raises(Exception):
        _run_combined(inputs, real_config, tmp_path, cancellation_requested=cancel_after_first_segment)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "final"
    assert status["status"] == "cancelled"
    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success


def test_cancellation_during_d3b_a_detection_leaves_no_successful_run(
    accepted_case, real_config, tmp_path, monkeypatch
):
    real_detect_roi = detection_module._detect_roi
    state = {"rois_detected": 0}

    def counting_detect_roi(*args, **kwargs):
        result = real_detect_roi(*args, **kwargs)
        state["rois_detected"] += 1
        return result

    monkeypatch.setattr(detection_module, "_detect_roi", counting_detect_roi)

    def cancel_after_first_roi_detected():
        return state["rois_detected"] >= 1

    inputs = _pass_inputs(accepted_case)
    with pytest.raises(Exception):
        _run_combined(
            inputs, real_config, tmp_path, cancellation_requested=cancel_after_first_roi_detected
        )

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    # Tonic already completed before D3b-A cancellation fired.
    tonic_out_dir = os.path.join(str(run_dir), "_analysis", "tonic_out")
    assert os.path.isfile(os.path.join(tonic_out_dir, "tonic_trace_cache.h5"))

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "final"
    assert status["status"] == "cancelled"
    assert status["errors"] == []

    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success


def test_genuine_d3b_a_failure_remains_terminal_error_not_cancelled(
    accepted_case, real_config, tmp_path, monkeypatch
):
    def flaky_detect_roi(*args, **kwargs):
        raise detection_module.GuidedContinuousRwdPhasicDetectionError(
            "simulated genuine D3b-A detection failure"
        )

    monkeypatch.setattr(detection_module, "_detect_roi", flaky_detect_roi)

    inputs = _pass_inputs(accepted_case)
    with pytest.raises(Exception):
        _run_combined(inputs, real_config, tmp_path)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "final"
    assert status["status"] == "error"

    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success
