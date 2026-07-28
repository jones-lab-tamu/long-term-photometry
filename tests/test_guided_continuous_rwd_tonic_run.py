from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from photometry_pipeline import guided_continuous_rwd_correction_pass as c4c
from photometry_pipeline import guided_continuous_rwd_tonic_run as subject
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_attrs,
    load_cache_chunk_fields,
    open_phasic_cache,
    open_tonic_cache,
)
from photometry_pipeline.run_completion_contract import (
    TERMINAL_SUCCESS_CURRENT,
    classify_run_terminal_state,
)

# Reuse the D1 module's synthetic-recording builders (same accepted
# construction path already used by D1/D2's own test suites).
from tests.test_guided_continuous_rwd_correction_pass_persistence import (
    _build_case,
    _pass_inputs,
)


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_d3a") / "recording"
    return _build_case(folder, continuous_window_sec=90.0)


@pytest.fixture(scope="module")
def real_config(accepted_case):
    from photometry_pipeline.guided_continuous_rwd_segment_correction import (
        _resolve_segment_correction_settings,
    )

    _binding, _grid, _draft, contract, _source = accepted_case
    config, _identity = _resolve_segment_correction_settings(contract)
    return config


def _run(inputs, real_config, output_base, **kwargs):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    return subject.execute_guided_continuous_rwd_tonic_run(
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
    path = os.path.join(run_dir, roi, "tables", "continuous_tonic_window_summary.csv")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Successful multi-chunk run
# ---------------------------------------------------------------------------


def test_successful_multi_chunk_run_publishes_current_run(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    binding, grid = inputs[0], inputs[1]
    result = _run(inputs, real_config, tmp_path)

    assert result.terminal_state == TERMINAL_SUCCESS_CURRENT
    included = list(binding.recording.roi.included_roi_ids)
    assert set(result.tonic_summary_paths) == set(included)
    assert all(count >= 1 for count in result.tonic_summary_row_counts.values())

    # The established tonic artifact: a genuine tonic-mode cache at the
    # classic analysis-directory location.
    tonic_analysis_dir = os.path.join(result.run_dir, "_analysis", "tonic_out")
    assert os.path.isfile(os.path.join(tonic_analysis_dir, "run_report.json"))
    assert os.path.isfile(os.path.join(tonic_analysis_dir, "config_used.yaml"))
    assert result.tonic_cache_path == os.path.join(tonic_analysis_dir, "tonic_trace_cache.h5")
    assert os.path.isfile(result.tonic_cache_path)

    for roi in included:
        df = _read_roi_summary(result.run_dir, roi)
        assert len(df) == result.tonic_summary_row_counts[roi]
        df = df.sort_values("window_index")
        starts = df["window_start_sec"].to_numpy()
        ends = df["window_end_sec"].to_numpy()
        gaps = starts[1:] - ends[:-1]
        assert np.allclose(gaps, 0.1, atol=1e-9)
        assert df["window_index"].tolist() == list(range(len(df)))
        assert df["continuous_window_sec"].tolist() == pytest.approx([90.0] * len(df))
        assert df["continuous_step_sec"].tolist() == pytest.approx([90.0] * len(df))
        overview = os.path.join(result.run_dir, roi, "summary", "tonic_overview.png")
        assert os.path.isfile(overview)
        assert os.path.getsize(overview) > 0
        assert not os.path.exists(
            os.path.join(result.run_dir, roi, "summary", "sampled_signal_reference.png")
        )

    classification = classify_run_terminal_state(result.run_dir)
    assert classification.is_success
    assert classification.run_mode.get("tonic_analysis") is True
    assert classification.run_mode.get("phasic_analysis") is False
    assert "tonic_shares_correction_cache" not in classification.run_mode

    with open(os.path.join(result.run_dir, "run_report.json"), encoding="utf-8") as fh:
        report = json.load(fh)
    assert "phasic" in report["summary"]["narrative"].lower()
    assert "not been run" in report["summary"]["narrative"].lower()
    assert report["saved_artifacts"]["window_timing"] == {
        "window_length_sec": pytest.approx(90.0),
        "window_step_sec": pytest.approx(90.0),
        "window_length_source": "accepted_draft.continuous_window_sec",
        "window_step_source": "accepted_draft.continuous_step_sec",
    }
    for roi in included:
        sampling = report["saved_artifacts"]["tonic_overview_sampling_by_roi"][roi]
        assert sampling["n_points_plotted"] <= sampling["max_plot_points"]
        assert sampling["trace_labels"] == [
            "Raw signal",
            "Raw reference",
            "Tonic signal (deltaF)",
        ]

    with open(os.path.join(result.run_dir, "MANIFEST.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    tonic_record = next(
        record
        for record in manifest["completion"]["deliverables"]["continuous_window_index"][
            "saved_artifacts"
        ]
        if record["relative_path"] == "ROI1/summary/tonic_overview.png"
    )
    assert tonic_record["family"] == "tonic_overview"
    assert tonic_record["analysis_family"] == "tonic"


def test_natural_tonic_publication_uses_two_shared_readable_panels(
    accepted_case, real_config, tmp_path, monkeypatch
):
    import matplotlib.pyplot as pyplot

    captured = []
    real_subplots = pyplot.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        flat_axes = np.asarray(axes, dtype=object).reshape(-1)
        if flat_axes.size == 2 and kwargs.get("sharex") is True:
            captured.append(flat_axes)
        return figure, axes

    monkeypatch.setattr(pyplot, "subplots", capture_subplots)
    result = _run(_pass_inputs(accepted_case), real_config, tmp_path)

    assert len(captured) == 2
    for raw_axis, tonic_axis in captured:
        assert raw_axis.get_shared_x_axes().joined(raw_axis, tonic_axis)
        assert [line.get_label() for line in raw_axis.get_lines()] == [
            "Raw signal",
            "Raw reference",
        ]
        assert [line.get_label() for line in tonic_axis.get_lines()] == [
            "Tonic signal (deltaF)",
        ]
        assert not any(
            line.get_label() == "Tonic signal (deltaF)"
            for line in raw_axis.get_lines()
        )
        assert not any(
            line.get_label() in {"Raw signal", "Raw reference"}
            for line in tonic_axis.get_lines()
        )

        raw_y = np.concatenate(
            [line.get_ydata() for line in raw_axis.get_lines()]
        )
        tonic_y = tonic_axis.get_lines()[0].get_ydata()
        raw_y = raw_y[np.isfinite(raw_y)]
        tonic_y = tonic_y[np.isfinite(tonic_y)]
        assert raw_y.size > 0
        assert tonic_y.size > 0
        assert raw_axis.get_ylim() != tonic_axis.get_ylim()
        assert (tonic_axis.get_ylim()[1] - tonic_axis.get_ylim()[0]) < (
            raw_axis.get_ylim()[1] - raw_axis.get_ylim()[0]
        )

        tonic_x = tonic_axis.get_lines()[0].get_xdata()
        assert all(
            np.array_equal(line.get_xdata(), tonic_x)
            for line in raw_axis.get_lines()
        )

    assert os.path.isfile(
        os.path.join(result.run_dir, "ROI1", "summary", "tonic_overview.png")
    )


def test_one_chunk_run_is_one_continuous_recording(real_config, tmp_path, tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_d3a_single") / "recording"
    case = _build_case(folder, continuous_window_sec=600.0)
    inputs = _pass_inputs(case)
    result = _run(inputs, real_config, tmp_path)

    assert result.terminal_state == TERMINAL_SUCCESS_CURRENT
    assert result.completion.corrected_segment_count == 1
    for roi, count in result.tonic_summary_row_counts.items():
        assert count == 1

    cache = open_tonic_cache(result.tonic_cache_path)
    try:
        assert list_cache_chunk_ids(cache) == [0]
    finally:
        cache.close()

    classification = classify_run_terminal_state(result.run_dir)
    assert classification.is_success
    assert classification.run_mode.get("chunked_input_processing") is False


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

    cache = open_tonic_cache(result.tonic_cache_path)
    try:
        chunk_ids = list_cache_chunk_ids(cache)
        assert chunk_ids == list(range(segment_plan.segment_count))
        last_attrs = load_cache_chunk_attrs(
            cache, binding.recording.roi.included_roi_ids[0], chunk_ids[-1]
        )
        assert last_attrs["window_duration_sec"] < first_row["window_duration_sec"]
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# Scientific reference tests
# ---------------------------------------------------------------------------


def test_tonic_deltaF_matches_established_correction_result(accepted_case, real_config, tmp_path):
    """The tonic cache's ``deltaF`` must equal C4b's already-established,
    already-tested per-segment correction result (``delta_f``) verbatim --
    see tests/test_guided_continuous_rwd_segment_correction.py::
    test_mixed_segment_matches_native_global_and_signal_only_references,
    which independently proves that same ``delta_f`` already equals
    ``regression.fit_chunk_dynamic(..., mode="phasic", ...)``, the exact
    call pipeline.py's native (per-ROI-correction) tonic route dispatches
    to. This test proves this module's own adapter republishes that
    established result without corruption, relabeling, or recomputation.
    """
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    result = _run(inputs, real_config, tmp_path)

    tonic_cache = open_tonic_cache(result.tonic_cache_path)
    correction_cache = open_phasic_cache(result.corrected_cache_path)
    try:
        chunk_ids = list_cache_chunk_ids(tonic_cache)
        assert chunk_ids == list_cache_chunk_ids(correction_cache)
        for roi in binding.recording.roi.included_roi_ids:
            for chunk_id in chunk_ids:
                (tonic_delta,) = load_cache_chunk_fields(
                    tonic_cache, roi, chunk_id, ["deltaF"]
                )
                (correction_delta,) = load_cache_chunk_fields(
                    correction_cache, roi, chunk_id, ["delta_f"]
                )
                np.testing.assert_array_equal(tonic_delta, correction_delta)
    finally:
        tonic_cache.close()
        correction_cache.close()


def test_tonic_cache_round_trip(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    result = _run(inputs, real_config, tmp_path)
    included = list(binding.recording.roi.included_roi_ids)

    cache = open_tonic_cache(result.tonic_cache_path)
    try:
        assert cache["meta"].attrs["mode"] == "tonic"
        assert list_cache_rois(cache) == included
        chunk_ids = list_cache_chunk_ids(cache)
        assert chunk_ids == list(range(result.completion.corrected_segment_count))
        first_attrs = load_cache_chunk_attrs(cache, included[0], chunk_ids[0])
        last_attrs = load_cache_chunk_attrs(cache, included[0], chunk_ids[-1])
        assert first_attrs["window_start_sec"] == 0.0
        assert last_attrs["window_end_sec"] > first_attrs["window_end_sec"]
    finally:
        cache.close()


def test_summary_is_derived_from_tonic_cache(accepted_case, real_config, tmp_path):
    """Prove ``continuous_tonic_window_summary.csv`` traces to the genuine
    tonic-mode artifact, not merely to the correction-stage cache: recompute
    the descriptive statistic independently from the tonic cache and compare
    against the summary row for the same window."""
    inputs = _pass_inputs(accepted_case)
    binding = inputs[0]
    result = _run(inputs, real_config, tmp_path)
    roi = binding.recording.roi.included_roi_ids[0]

    df = _read_roi_summary(result.run_dir, roi).sort_values("window_index")
    cache = open_tonic_cache(result.tonic_cache_path)
    try:
        for _, row in df.iterrows():
            chunk_id = int(row["chunk_id"])
            (delta_f,) = load_cache_chunk_fields(cache, roi, chunk_id, ["deltaF"])
            finite = delta_f[np.isfinite(delta_f)]
            assert row["tonic_mean"] == pytest.approx(float(np.mean(finite)))
            assert row["tonic_median"] == pytest.approx(float(np.median(finite)))
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# Failure and cancellation
# ---------------------------------------------------------------------------


def test_failure_during_tonic_cache_production_leaves_no_successful_run(
    accepted_case, real_config, tmp_path, monkeypatch
):
    inputs = _pass_inputs(accepted_case)

    def flaky_write(*args, **kwargs):
        raise RuntimeError("simulated tonic cache production failure")

    monkeypatch.setattr(subject, "_write_tonic_trace_cache", flaky_write)

    with pytest.raises(Exception):
        _run(inputs, real_config, tmp_path)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    # The corrected cache persisted before the simulated tonic failure.
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


def test_failure_after_tonic_artifacts_written_leaves_no_successful_run(
    accepted_case, real_config, tmp_path, monkeypatch
):
    inputs = _pass_inputs(accepted_case)

    def flaky_manifest_block(*args, **kwargs):
        raise RuntimeError("simulated manifest-build failure after tonic artifacts written")

    monkeypatch.setattr(subject, "build_manifest_completion_block", flaky_manifest_block)

    with pytest.raises(Exception):
        _run(inputs, real_config, tmp_path)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    binding = inputs[0]
    first_roi = binding.recording.roi.included_roi_ids[0]
    tonic_analysis_dir = os.path.join(str(run_dir), "_analysis", "tonic_out")
    # Tonic artifacts already written before the simulated failure remain.
    assert os.path.isfile(os.path.join(tonic_analysis_dir, "tonic_trace_cache.h5"))
    assert os.path.isfile(
        os.path.join(str(run_dir), first_roi, "tables", "continuous_tonic_window_summary.csv")
    )
    assert os.path.isfile(os.path.join(str(run_dir), subject.CORRECTED_CACHE_RELATIVE_PATH))
    assert not os.path.exists(os.path.join(str(run_dir), "MANIFEST.json"))

    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "error"
