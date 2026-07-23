from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pandas as pd
import pytest

from photometry_pipeline import guided_continuous_rwd_phasic_run as subject
from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import compute_auc_above_threshold
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.guided_continuous_rwd_phasic_detection import (
    GuidedContinuousRwdPhasicDetectionResult,
    GuidedContinuousRwdPhasicRoiDetection,
)
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_attrs,
    load_cache_chunk_fields,
    open_phasic_cache,
)
from photometry_pipeline.run_completion_contract import (
    TERMINAL_SUCCESS_CURRENT,
    classify_run_terminal_state,
)

# Reuse the D1 module's synthetic-recording builders (same accepted
# construction path already used by D1/D2/D3a's own test suites).
from tests.test_guided_continuous_rwd_correction_pass_persistence import (
    _build_case,
    _pass_inputs,
)

FS_HZ = 10.0


# ---------------------------------------------------------------------------
# Part A: direct, hand-crafted tests of the publication logic
# (_publish_phasic_cache_and_features) -- proves window ownership, count
# conservation, global-not-local threshold reuse, and AUC re-integration,
# independent of what a real correction+detection pass happens to produce.
# ---------------------------------------------------------------------------


def _write_min_phasic_cache(
    path: str,
    *,
    included_roi_ids: list[str],
    chunk_sample_counts: list[int],
    roi_dff: dict[str, np.ndarray],
    config: Config,
) -> None:
    """Write a minimal, valid phasic-mode D1-shaped cache with explicit
    per-chunk boundaries and deterministic per-ROI ``dff`` content. No
    continuous-identity meta attrs are stamped: ``_publish_phasic_cache_and_
    features`` never validates cache authorities (D3b-A already owns that),
    it only reads chunk fields/attrs.
    """
    writer = Hdf5TraceCacheWriter(path, "phasic", config)
    start = 0
    for chunk_id, n in enumerate(chunk_sample_counts):
        stop = start + n
        local_time = np.arange(n, dtype=np.float64) / FS_HZ
        window_start = start / FS_HZ
        window_end = (stop - 1) / FS_HZ
        dff_cols = [np.asarray(roi_dff[roi][start:stop], dtype=np.float64) for roi in included_roi_ids]
        chunk = Chunk(
            chunk_id=chunk_id,
            source_file="synthetic-source",
            format="rwd",
            time_sec=local_time,
            uv_raw=np.zeros((n, len(included_roi_ids)), dtype=np.float64),
            sig_raw=np.zeros((n, len(included_roi_ids)), dtype=np.float64),
            uv_fit=np.zeros((n, len(included_roi_ids)), dtype=np.float64),
            delta_f=np.column_stack(dff_cols),
            dff=np.column_stack(dff_cols),
            fs_hz=FS_HZ,
            channel_names=included_roi_ids,
            metadata={
                "acquisition_mode": "continuous",
                "window_index": chunk_id,
                "window_start_sec": window_start,
                "window_end_sec": window_end,
                "window_duration_sec": window_end - window_start,
            },
        )
        writer.add_chunk(chunk, chunk_id=chunk_id, source_file="synthetic-source")
        start = stop
    writer.finalize()


def _roi_detection(
    *, mean, median, std, mad, event_count, peak_times, peak_polarities
) -> GuidedContinuousRwdPhasicRoiDetection:
    return GuidedContinuousRwdPhasicRoiDetection(
        roi_id="unused",
        mean=mean,
        median=median,
        std=std,
        mad=mad,
        event_count=event_count,
        auc=0.0,
        peak_global_times_sec=np.asarray(peak_times, dtype=np.float64),
        peak_polarities=np.asarray(peak_polarities, dtype=np.int64),
    )


@pytest.fixture
def boundary_scenario(tmp_path):
    """3 chunks of [50, 50, 30] samples at 10 Hz (total 130 samples). ROI1
    carries one event on the last sample of chunk 0, one on the first sample
    of chunk 1, one on the last sample of chunk 1, and one on the first
    sample of chunk 2 -- covering every adjacent-boundary ownership case.
    ROI2 carries no events. Each chunk's ROI1 dff is a distinct constant
    amplitude (1.0, 2.0, 3.0) so each window's AUC is independently
    predictable and provably NOT identical across chunks."""
    chunk_sample_counts = [50, 50, 30]
    total = sum(chunk_sample_counts)
    roi1_dff = np.concatenate(
        [np.full(50, 1.0), np.full(50, 2.0), np.full(30, 3.0)]
    )
    roi2_dff = np.zeros(total)

    cache_path = str(tmp_path / "corrected.h5")
    config = Config()
    _write_min_phasic_cache(
        cache_path,
        included_roi_ids=["ROI1", "ROI2"],
        chunk_sample_counts=chunk_sample_counts,
        roi_dff={"ROI1": roi1_dff, "ROI2": roi2_dff},
        config=config,
    )

    detection = GuidedContinuousRwdPhasicDetectionResult(
        recording_identity="rec-1",
        target_grid_identity="grid-1",
        completion_identity="comp-1",
        sampling_rate_hz=FS_HZ,
        included_roi_ids=("ROI1", "ROI2"),
        target_sample_count=total,
        global_time_start_sec=0.0,
        global_time_end_sec=(total - 1) / FS_HZ,
        execution_strategy="roi_at_a_time",
        detector_parameter_identity="digest-1",
        per_roi={
            "ROI1": _roi_detection(
                mean=11.0,
                median=12.0,
                std=13.0,
                mad=14.0,
                event_count=4,
                peak_times=[49 / FS_HZ, 50 / FS_HZ, 99 / FS_HZ, 100 / FS_HZ],
                peak_polarities=[1, -1, 1, -1],
            ),
            "ROI2": _roi_detection(
                mean=1.0, median=2.0, std=3.0, mad=4.0,
                event_count=0, peak_times=[], peak_polarities=[],
            ),
        },
    )

    return {
        "cache_path": cache_path,
        "chunk_sample_counts": chunk_sample_counts,
        "detection": detection,
        "config": config,
        "tmp_path": tmp_path,
    }


def test_boundary_events_are_partitioned_without_loss_or_duplication(boundary_scenario):
    out_path = str(boundary_scenario["tmp_path"] / "phasic_trace_cache.h5")
    feature_rows, event_rows = subject._publish_phasic_cache_and_features(
        corrected_cache_path=boundary_scenario["cache_path"],
        phasic_cache_path=out_path,
        included_roi_ids=("ROI1", "ROI2"),
        detection=boundary_scenario["detection"],
        config=boundary_scenario["config"],
    )

    roi1_rows = {r["chunk_id"]: r for r in feature_rows if r["roi"] == "ROI1"}
    assert roi1_rows[0]["peak_count"] == 1  # sample 49 (chunk 0's own last sample)
    assert roi1_rows[1]["peak_count"] == 2  # samples 50 and 99
    assert roi1_rows[2]["peak_count"] == 1  # sample 100 (chunk 2's own first sample)
    assert sum(r["peak_count"] for r in roi1_rows.values()) == 4

    roi1_events = [e for e in event_rows if e["roi"] == "ROI1"]
    assert len(roi1_events) == 4
    assert sorted(e["chunk_id"] for e in roi1_events) == [0, 1, 1, 2]
    assert sorted(e["global_time_sec"] for e in roi1_events) == pytest.approx(
        sorted([49 / FS_HZ, 50 / FS_HZ, 99 / FS_HZ, 100 / FS_HZ])
    )

    roi2_rows = [r for r in feature_rows if r["roi"] == "ROI2"]
    assert all(r["peak_count"] == 0 for r in roi2_rows)
    assert not [e for e in event_rows if e["roi"] == "ROI2"]


def test_window_rows_reuse_global_threshold_statistics_not_local(boundary_scenario):
    feature_rows, _ = subject._publish_phasic_cache_and_features(
        corrected_cache_path=boundary_scenario["cache_path"],
        phasic_cache_path=str(boundary_scenario["tmp_path"] / "phasic_trace_cache.h5"),
        included_roi_ids=("ROI1", "ROI2"),
        detection=boundary_scenario["detection"],
        config=boundary_scenario["config"],
    )
    roi1_rows = [r for r in feature_rows if r["roi"] == "ROI1"]
    assert len(roi1_rows) == 3
    for row in roi1_rows:
        assert row["mean"] == 11.0
        assert row["median"] == 12.0
        assert row["std"] == 13.0
        assert row["mad"] == 14.0


def test_window_auc_is_independently_reintegrated_per_chunk(boundary_scenario):
    feature_rows, _ = subject._publish_phasic_cache_and_features(
        corrected_cache_path=boundary_scenario["cache_path"],
        phasic_cache_path=str(boundary_scenario["tmp_path"] / "phasic_trace_cache.h5"),
        included_roi_ids=("ROI1", "ROI2"),
        detection=boundary_scenario["detection"],
        config=boundary_scenario["config"],
    )
    roi1_rows = {r["chunk_id"]: r for r in feature_rows if r["roi"] == "ROI1"}
    # zero baseline (Config default event_auc_baseline='zero'), positive polarity:
    # AUC of a constant-amplitude trace over its own local duration == amplitude * duration.
    expected_auc_0 = 1.0 * (49 / FS_HZ)
    expected_auc_1 = 2.0 * (49 / FS_HZ)
    expected_auc_2 = 3.0 * (29 / FS_HZ)
    assert roi1_rows[0]["auc"] == pytest.approx(expected_auc_0)
    assert roi1_rows[1]["auc"] == pytest.approx(expected_auc_1)
    assert roi1_rows[2]["auc"] == pytest.approx(expected_auc_2)
    # Distinct per window -- proves this is a real re-integration, not a
    # copy of one global value.
    assert len({roi1_rows[0]["auc"], roi1_rows[1]["auc"], roi1_rows[2]["auc"]}) == 3


def test_publish_raises_when_event_partition_does_not_conserve_count(boundary_scenario):
    detection = boundary_scenario["detection"]
    bad_roi1 = dataclasses.replace(
        detection.per_roi["ROI1"],
        event_count=5,  # one more than the 4 events actually supplied
    )
    bad_detection = dataclasses.replace(
        detection, per_roi={**detection.per_roi, "ROI1": bad_roi1}
    )
    with pytest.raises(subject.GuidedContinuousRwdPhasicRunError):
        subject._publish_phasic_cache_and_features(
            corrected_cache_path=boundary_scenario["cache_path"],
            phasic_cache_path=str(boundary_scenario["tmp_path"] / "phasic_trace_cache.h5"),
            included_roi_ids=("ROI1", "ROI2"),
            detection=bad_detection,
            config=boundary_scenario["config"],
        )


def test_republished_cache_round_trips_dff(boundary_scenario):
    out_path = str(boundary_scenario["tmp_path"] / "phasic_trace_cache.h5")
    subject._publish_phasic_cache_and_features(
        corrected_cache_path=boundary_scenario["cache_path"],
        phasic_cache_path=out_path,
        included_roi_ids=("ROI1", "ROI2"),
        detection=boundary_scenario["detection"],
        config=boundary_scenario["config"],
    )
    cache = open_phasic_cache(out_path)
    try:
        assert list_cache_rois(cache) == ["ROI1", "ROI2"]
        assert list_cache_chunk_ids(cache) == [0, 1, 2]
        (dff,) = load_cache_chunk_fields(cache, "ROI1", 2, ["dff"])
        np.testing.assert_array_equal(dff, np.full(30, 3.0))
        attrs = load_cache_chunk_attrs(cache, "ROI1", 0)
        assert attrs["window_start_sec"] == 0.0
    finally:
        cache.close()


def test_events_csv_round_trip(tmp_path, boundary_scenario):
    _, event_rows = subject._publish_phasic_cache_and_features(
        corrected_cache_path=boundary_scenario["cache_path"],
        phasic_cache_path=str(boundary_scenario["tmp_path"] / "phasic_trace_cache.h5"),
        included_roi_ids=("ROI1", "ROI2"),
        detection=boundary_scenario["detection"],
        config=boundary_scenario["config"],
    )
    features_dir = str(tmp_path / "features")
    events_path = subject._write_events_csv(features_dir, event_rows)
    subject._validate_events_csv(events_path, detection=boundary_scenario["detection"])
    df = pd.read_csv(events_path)
    assert len(df) == 4
    assert set(df.columns) == {"roi", "chunk_id", "window_index", "global_time_sec", "polarity"}


# ---------------------------------------------------------------------------
# Part B: end-to-end runs through the real correction + D3b-A detection path.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_d3b_b") / "recording"
    return _build_case(folder, continuous_window_sec=90.0)


@pytest.fixture(scope="module")
def real_config(accepted_case):
    from photometry_pipeline.guided_continuous_rwd_segment_correction import (
        _resolve_segment_correction_settings,
    )

    _binding, _grid, _draft, contract, _source = accepted_case
    config, _identity = _resolve_segment_correction_settings(contract)
    # Lower the detection threshold so the synthetic recording's small
    # periodic correction residual (see _build_case's underlying signal
    # construction) actually crosses it, giving genuine, spread-out events
    # to exercise multi-window conservation -- the default mean_std/k=2.5
    # threshold sits above this residual's amplitude by construction.
    return dataclasses.replace(
        config,
        peak_threshold_method="percentile",
        peak_threshold_percentile=50.0,
        peak_min_distance_sec=1.0,
    )


def _run(inputs, real_config, output_base, **kwargs):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    return subject.execute_guided_continuous_rwd_phasic_run(
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


def _read_roi_summary(run_dir, roi):
    path = os.path.join(run_dir, roi, "tables", "continuous_phasic_window_summary.csv")
    return pd.read_csv(path)


def test_successful_multi_chunk_run_publishes_current_run(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    result = _run(inputs, real_config, tmp_path)

    assert result.terminal_state == TERMINAL_SUCCESS_CURRENT
    included = list(binding.recording.roi.included_roi_ids)
    assert set(result.phasic_summary_paths) == set(included)
    assert all(count >= 1 for count in result.phasic_summary_row_counts.values())

    phasic_analysis_dir = os.path.join(result.run_dir, "_analysis", "phasic_out")
    assert os.path.isfile(os.path.join(phasic_analysis_dir, "run_report.json"))
    assert os.path.isfile(os.path.join(phasic_analysis_dir, "config_used.yaml"))
    assert result.phasic_cache_path == os.path.join(phasic_analysis_dir, "phasic_trace_cache.h5")
    assert os.path.isfile(result.phasic_cache_path)
    assert os.path.isfile(result.features_path)
    assert os.path.isfile(result.events_path)
    assert os.path.isfile(
        os.path.join(phasic_analysis_dir, "features", "feature_event_provenance.json")
    )

    classification = classify_run_terminal_state(result.run_dir)
    assert classification.is_success
    assert classification.run_mode.get("phasic_analysis") is True
    assert classification.run_mode.get("tonic_analysis") is False
    assert classification.run_mode.get("feature_extraction_ran") is True

    with open(os.path.join(result.run_dir, "run_report.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    assert "tonic" in report["summary"]["narrative"].lower()
    assert "not been run" in report["summary"]["narrative"].lower()


def test_features_csv_has_one_row_per_chunk_per_roi(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    result = _run(inputs, real_config, tmp_path)
    included = list(binding.recording.roi.included_roi_ids)

    features = pd.read_csv(result.features_path)
    for roi in included:
        roi_rows = features[features["roi"] == roi]
        assert len(roi_rows) == result.completion.corrected_segment_count
        assert sorted(roi_rows["chunk_id"].tolist()) == list(
            range(result.completion.corrected_segment_count)
        )
        # No fabricated per-window statistics: the threshold context is the
        # one global value D3b-A actually used, identical on every row.
        assert roi_rows["mean"].nunique() == 1
        assert roi_rows["median"].nunique() == 1


def test_events_csv_conserves_detection_exactly(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    result = _run(inputs, real_config, tmp_path)
    included = list(binding.recording.roi.included_roi_ids)

    events = pd.read_csv(result.events_path)
    total_detected = sum(result.detection.per_roi[roi].event_count for roi in included)
    assert len(events) == total_detected
    assert total_detected > 0  # the tuned threshold must actually find events

    for roi in included:
        roi_detection = result.detection.per_roi[roi]
        roi_events = events[events["roi"] == roi].sort_values("global_time_sec")
        assert len(roi_events) == roi_detection.event_count
        np.testing.assert_allclose(
            roi_events["global_time_sec"].to_numpy(),
            roi_detection.peak_global_times_sec,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            roi_events["polarity"].to_numpy(), roi_detection.peak_polarities
        )


def test_window_summary_conserves_events_and_covers_recording(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    result = _run(inputs, real_config, tmp_path)
    included = list(binding.recording.roi.included_roi_ids)

    for roi in included:
        df = _read_roi_summary(result.run_dir, roi).sort_values("window_index")
        assert len(df) == result.completion.corrected_segment_count
        assert df["window_index"].tolist() == list(range(len(df)))
        assert int(df["event_count"].sum()) == result.detection.per_roi[roi].event_count
        # Contiguous window coverage: one sampling interval between consecutive windows.
        starts = df["window_start_sec"].to_numpy()
        ends = df["window_end_sec"].to_numpy()
        gaps = starts[1:] - ends[:-1]
        assert np.allclose(gaps, 0.1, atol=1e-9)


def test_final_short_tail_is_included_not_dropped(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    segment_plan = inputs[5]
    assert segment_plan.descriptors[-1].sample_count < segment_plan.nominal_segment_sample_count

    result = _run(inputs, real_config, tmp_path)
    binding = inputs[0]
    for roi in binding.recording.roi.included_roi_ids:
        df = _read_roi_summary(result.run_dir, roi).sort_values("window_index")
        last_row = df.iloc[-1]
        first_row = df.iloc[0]
        assert last_row["window_duration_sec"] < first_row["window_duration_sec"]
        assert last_row["window_index"] == segment_plan.segment_count - 1

    features = pd.read_csv(result.features_path)
    assert features["chunk_id"].max() == segment_plan.segment_count - 1


def test_one_chunk_run_is_one_continuous_recording(real_config, tmp_path, tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_d3b_b_single") / "recording"
    case = _build_case(folder, continuous_window_sec=600.0)
    inputs = _pass_inputs(case)
    result = _run(inputs, real_config, tmp_path)

    assert result.terminal_state == TERMINAL_SUCCESS_CURRENT
    assert result.completion.corrected_segment_count == 1
    for roi, count in result.phasic_summary_row_counts.items():
        assert count == 1

    cache = open_phasic_cache(result.phasic_cache_path)
    try:
        assert list_cache_chunk_ids(cache) == [0]
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# Part C: failure and cancellation
# ---------------------------------------------------------------------------


def test_failure_during_phasic_publication_leaves_no_successful_run(
    accepted_case, real_config, tmp_path, monkeypatch
):
    inputs = _pass_inputs(accepted_case)

    def flaky_publish(*args, **kwargs):
        raise RuntimeError("simulated phasic cache production failure")

    monkeypatch.setattr(subject, "_publish_phasic_cache_and_features", flaky_publish)

    with pytest.raises(Exception):
        _run(inputs, real_config, tmp_path)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert os.path.isfile(os.path.join(str(run_dir), subject.CORRECTED_CACHE_RELATIVE_PATH))
    assert not os.path.exists(os.path.join(str(run_dir), "MANIFEST.json"))

    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "error"
    assert status["phase"] == "final"


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
        _run(inputs, real_config, tmp_path, cancellation_requested=cancel_after_first_segment)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "cancelled"
    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success


def test_failure_after_phasic_artifacts_written_leaves_no_successful_run(
    accepted_case, real_config, tmp_path, monkeypatch
):
    inputs = _pass_inputs(accepted_case)

    def flaky_manifest_block(*args, **kwargs):
        raise RuntimeError("simulated manifest-build failure after phasic artifacts written")

    monkeypatch.setattr(subject, "build_manifest_completion_block", flaky_manifest_block)

    with pytest.raises(Exception):
        _run(inputs, real_config, tmp_path)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    binding = inputs[0]
    first_roi = binding.recording.roi.included_roi_ids[0]
    phasic_analysis_dir = os.path.join(str(run_dir), "_analysis", "phasic_out")
    assert os.path.isfile(os.path.join(phasic_analysis_dir, "phasic_trace_cache.h5"))
    assert os.path.isfile(os.path.join(phasic_analysis_dir, "features", "features.csv"))
    assert os.path.isfile(
        os.path.join(str(run_dir), first_roi, "tables", "continuous_phasic_window_summary.csv")
    )
    assert not os.path.exists(os.path.join(str(run_dir), "MANIFEST.json"))

    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "error"


def test_cancellation_during_d3b_a_detection_is_recorded_as_cancelled_not_error(
    accepted_case, real_config, tmp_path, monkeypatch
):
    """C4c correction and D1 persistence complete normally; D3b-A detects
    the first ROI; the cancellation callback then returns true before the
    next ROI's checkpoint. The run must be recorded as ``cancelled``, never
    ``error`` -- proving CR1-D3b-B's exception handler now recognizes a
    D3b-A ``GuidedContinuousRwdPhasicDetectionError`` carrying the
    ``phasic_detection_interrupted`` category as cancellation, not merely a
    correction-stage interruption."""
    from photometry_pipeline import guided_continuous_rwd_phasic_detection as detection_module

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
        _run(inputs, real_config, tmp_path, cancellation_requested=cancel_after_first_roi_detected)

    # Correction actually completed before cancellation fired.
    assert state["rois_detected"] >= 1

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert os.path.isfile(os.path.join(str(run_dir), subject.CORRECTED_CACHE_RELATIVE_PATH))

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "final"
    assert status["status"] == "cancelled"
    assert status["errors"] == []

    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success


def test_genuine_d3b_a_failure_remains_terminal_error_not_cancelled(
    accepted_case, real_config, tmp_path, monkeypatch
):
    """A real D3b-A refusal (no cancellation category) must never be
    misclassified as cancelled merely because it is the same exception type
    that carries the cancellation category in the true-cancellation case."""
    from photometry_pipeline import guided_continuous_rwd_phasic_detection as detection_module

    def flaky_detect_roi(*args, **kwargs):
        raise detection_module.GuidedContinuousRwdPhasicDetectionError(
            "simulated genuine D3b-A detection failure"
        )

    monkeypatch.setattr(detection_module, "_detect_roi", flaky_detect_roi)

    inputs = _pass_inputs(accepted_case)
    with pytest.raises(Exception):
        _run(inputs, real_config, tmp_path)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "final"
    assert status["status"] == "error"
    assert status["errors"] == ["simulated genuine D3b-A detection failure"]

    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success


def test_is_lower_layer_cancellation_matches_exact_type_and_category():
    """Unit-level proof of the (type, category) pairing semantics: a
    same-named category on the wrong exception type must never match, and an
    uncategorized exception of a recognized cancellation type must never
    match either."""
    from photometry_pipeline.guided_continuous_rwd_correction_pass import (
        GuidedContinuousRwdCorrectionPassError,
    )
    from photometry_pipeline.guided_continuous_rwd_phasic_detection import (
        GuidedContinuousRwdPhasicDetectionError,
    )

    cancelled_detection_exc = GuidedContinuousRwdPhasicDetectionError(
        "cancelled", category="phasic_detection_interrupted"
    )
    assert subject._is_lower_layer_cancellation(cancelled_detection_exc)

    non_cancelled_detection_exc = GuidedContinuousRwdPhasicDetectionError("some other failure")
    assert not subject._is_lower_layer_cancellation(non_cancelled_detection_exc)

    wrong_category_exc = GuidedContinuousRwdPhasicDetectionError(
        "wrong category", category="not_a_real_category"
    )
    assert not subject._is_lower_layer_cancellation(wrong_category_exc)

    cancelled_correction_exc = GuidedContinuousRwdCorrectionPassError(
        "segment_correction_pass_interrupted", "cancelled"
    )
    assert subject._is_lower_layer_cancellation(cancelled_correction_exc)

    class _Impostor(Exception):
        category = "phasic_detection_interrupted"

    assert not subject._is_lower_layer_cancellation(_Impostor("not the real type"))
