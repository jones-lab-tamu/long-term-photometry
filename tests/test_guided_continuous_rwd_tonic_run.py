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


def _native_test_window(index, signal, reference, *, window_length_sec=120.0):
    signal = np.asarray(signal, dtype=float)
    reference = np.asarray(reference, dtype=float)
    assert signal.shape == reference.shape
    return {
        "chunk_id": int(index),
        "time_sec": np.arange(signal.size, dtype=float) / 10.0,
        "sig": signal,
        "uv": reference,
        "fs_hz": 10.0,
        "meta": {
            "roi": "ROI1",
            "source_file": "synthetic.csv",
            "chunk_id": int(index),
            "window_index": int(index),
            "window_start_sec": float(index) * window_length_sec,
            "window_end_sec": float(index + 1) * window_length_sec,
            "window_duration_sec": float(window_length_sec),
            "elapsed_hour_start": float(index) * window_length_sec / 3600.0,
            "elapsed_hour_mid": (float(index) + 0.5) * window_length_sec / 3600.0,
            "is_partial_final_window": False,
            "original_file_duration_sec": float(2 * window_length_sec),
            "continuous_window_sec": float(window_length_sec),
            "continuous_step_sec": float(window_length_sec),
            "acquisition_mode": "continuous",
        },
    }


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
        assert sampling["trace_labels"][:2] == ["Raw signal", "Raw reference"]
        assert sampling["trace_labels"][2].endswith("(P2 per window)")
        assert sampling["tonic_method"] in {
            subject.TONIC_METHOD_GLOBAL_ISOSBESTIC,
            subject.TONIC_METHOD_SIGNAL_ONLY,
        }
        assert sampling["tonic_units"] in {
            subject.TONIC_UNITS_FRACTIONAL,
            subject.TONIC_UNITS_RAW_AU,
        }

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
    included = list(accepted_case[0].recording.roi.included_roi_ids)
    for panel_index, (raw_axis, tonic_axis) in enumerate(captured):
        assert raw_axis.get_shared_x_axes().joined(raw_axis, tonic_axis)
        assert [line.get_label() for line in raw_axis.get_lines()] == [
            "Raw signal",
            "Raw reference",
        ]
        tonic_label = tonic_axis.get_lines()[0].get_label()
        assert tonic_label.endswith("(P2 per window)")
        assert [line.get_color() for line in raw_axis.get_lines()] == [
            "green",
            "purple",
        ]
        assert [line.get_color() for line in tonic_axis.get_lines()] == ["black"]
        assert not any(line.get_label().endswith("(P2 per window)") for line in raw_axis.get_lines())
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
        assert tonic_x.size < raw_axis.get_lines()[0].get_xdata().size
        summary = _read_roi_summary(result.run_dir, included[panel_index])
        np.testing.assert_allclose(
            tonic_axis.get_lines()[0].get_ydata(),
            summary.sort_values("window_index")["tonic_value"].to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )

    assert os.path.isfile(
        os.path.join(result.run_dir, "ROI1", "summary", "tonic_overview.png")
    )


def test_one_chunk_run_fails_if_recording_level_fallback_cannot_fit(
    real_config, tmp_path, tmp_path_factory
):
    folder = tmp_path_factory.mktemp("cr1_d3a_single") / "recording"
    case = _build_case(folder, continuous_window_sec=600.0)
    inputs = _pass_inputs(case)
    with pytest.raises(
        subject.GuidedContinuousRwdTonicRunError,
        match="signal-only fallback failed for ROI 'ROI2'.*valid output windows",
    ):
        _run(inputs, real_config, tmp_path)


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


def test_native_tonic_cache_contains_recording_wide_tonic_trace(
    accepted_case, real_config, tmp_path
):
    """The native tonic cache contains the raw-channel tonic result, not the
    phasic correction cache's per-segment ``delta_f`` field."""
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
                (tonic_trace,) = load_cache_chunk_fields(
                    tonic_cache, roi, chunk_id, ["deltaF"]
                )
                summary = _read_roi_summary(result.run_dir, roi)
                row = summary.loc[summary["chunk_id"] == chunk_id].iloc[0]
                finite = tonic_trace[np.isfinite(tonic_trace)]
                assert finite.size > 0
                assert row["tonic_value"] == pytest.approx(
                    float(np.percentile(finite, subject.TONIC_PERCENTILE))
                )
                assert row["tonic_method"] in {
                    subject.TONIC_METHOD_GLOBAL_ISOSBESTIC,
                    subject.TONIC_METHOD_SIGNAL_ONLY,
                }
                assert row["units"] in {
                    subject.TONIC_UNITS_FRACTIONAL,
                    subject.TONIC_UNITS_RAW_AU,
                }
                (phasic_delta,) = load_cache_chunk_fields(
                    correction_cache, roi, chunk_id, ["delta_f"]
                )
                assert not np.array_equal(tonic_trace, phasic_delta)
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


def test_summary_is_derived_from_authoritative_tonic_cache(
    accepted_case, real_config, tmp_path
):
    """The published window scalar is P2 of the authoritative tonic trace."""
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
            assert row["tonic_percentile"] == pytest.approx(subject.TONIC_PERCENTILE)
            assert row["tonic_value"] == pytest.approx(
                float(np.percentile(finite, subject.TONIC_PERCENTILE))
            )
    finally:
        cache.close()


def test_native_primary_is_one_recording_wide_signed_fractional_fit(monkeypatch):
    rng = np.random.default_rng(17)
    windows = []
    for index in range(2):
        local_time = np.arange(1200, dtype=float) / 10.0
        reference = (
            100.0
            + 7.0 * np.sin(2.0 * np.pi * local_time / 17.0)
            + rng.normal(0.0, 0.15, local_time.size)
        )
        fitted_reference = 1.8 * reference + 15.0
        tonic_component = (
            -12.0 + 2.0 * np.sin(2.0 * np.pi * local_time / 97.0)
            if index == 0
            else 8.0 + 2.0 * np.sin(2.0 * np.pi * local_time / 97.0)
        )
        signal = fitted_reference + tonic_component
        signal = signal + rng.normal(0.0, 0.15, local_time.size)
        signal[::101] += 45.0
        windows.append(_native_test_window(index, signal, reference))

    calls = []
    real_fit = subject.compute_global_iso_fit_robust

    def counted_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(subject, "compute_global_iso_fit_robust", counted_fit)
    result = subject._compute_native_roi_tonic_result("ROI1", windows)

    assert len(calls) == 1
    assert result["tonic_method"] == subject.TONIC_METHOD_GLOBAL_ISOSBESTIC
    assert result["units"] == subject.TONIC_UNITS_FRACTIONAL
    assert result["tonic_fallback"] is False
    rows = result["rows"]
    assert len({row["global_slope"] for row in rows}) == 1
    assert len({row["global_intercept"] for row in rows}) == 1
    assert rows[0]["global_slope"] == pytest.approx(1.8, abs=0.05)
    assert rows[0]["global_intercept"] == pytest.approx(15.0, abs=3.0)
    assert any(row["tonic_value"] < 0.0 for row in rows)
    assert all(row["tonic_status"] == subject.TONIC_STATUS_VALID for row in rows)

    slope = rows[0]["global_slope"]
    intercept = rows[0]["global_intercept"]
    assert slope > 0.0
    primary_method = {
        "slope": slope,
        "intercept": intercept,
        "n_used": int(rows[0]["global_fit_n_used"]),
        "tonic_method": subject.TONIC_METHOD_GLOBAL_ISOSBESTIC,
        "units": subject.TONIC_UNITS_FRACTIONAL,
        "tonic_fallback": False,
        "fallback_reason": "",
        "fallback_value_by_chunk": {},
    }
    for window, row in zip(windows, rows):
        fitted = subject.apply_global_fit(window["uv"], slope, intercept)
        usable = (
            np.isfinite(window["sig"])
            & np.isfinite(window["uv"])
            & np.isfinite(fitted)
            & (fitted > subject.MIN_FITTED_ISOSBESTIC)
        )
        expected_dff = (window["sig"][usable] - fitted[usable]) / fitted[usable]
        assert row["tonic_value"] == pytest.approx(
            float(np.percentile(expected_dff, subject.TONIC_PERCENTILE))
        )
        _, trace = subject._build_native_tonic_window_result(
            "ROI1", window, primary_method
        )
        np.testing.assert_allclose(
            trace[usable],
            expected_dff,
            rtol=0.0,
            atol=1e-12,
        )


def test_native_fallback_is_one_recording_level_raw_au_method(monkeypatch):
    windows = []
    for index in range(8):
        n = 200
        local_time = np.arange(n, dtype=float) / 2.0
        start = index * 100.0
        bleach = 400.0 + 250.0 * np.exp(-start / 500.0)
        slow_tonic = 8.0 * np.sin(2.0 * np.pi * start / 800.0)
        signal = bleach + slow_tonic + 0.1 * np.sin(local_time / 10.0)
        reference = np.full(n, 50.0)
        windows.append(
            _native_test_window(
                index,
                signal,
                reference,
                window_length_sec=100.0,
            )
        )

    global_fit_calls = []
    bleach_fit_calls = []
    real_global_fit = subject.compute_global_iso_fit_robust
    real_bleach_fit = subject._fit_exponential_bleach_trend

    def counted_global_fit(*args, **kwargs):
        global_fit_calls.append((args, kwargs))
        return real_global_fit(*args, **kwargs)

    def counted_bleach_fit(*args, **kwargs):
        bleach_fit_calls.append((args, kwargs))
        return real_bleach_fit(*args, **kwargs)

    monkeypatch.setattr(subject, "compute_global_iso_fit_robust", counted_global_fit)
    monkeypatch.setattr(subject, "_fit_exponential_bleach_trend", counted_bleach_fit)
    result = subject._compute_native_roi_tonic_result("ROI1", windows)

    assert len(global_fit_calls) == 1
    assert len(bleach_fit_calls) == 1
    assert result["tonic_method"] == subject.TONIC_METHOD_SIGNAL_ONLY
    assert result["units"] == subject.TONIC_UNITS_RAW_AU
    assert result["tonic_fallback"] is True
    rows = result["rows"]
    assert {row["tonic_method"] for row in rows} == {
        subject.TONIC_METHOD_SIGNAL_ONLY
    }
    assert {row["units"] for row in rows} == {subject.TONIC_UNITS_RAW_AU}
    assert {row["fallback_reason"] for row in rows} == {"global_fit_failed"}

    baselines = np.asarray(
        [
            np.percentile(window["sig"], subject.TONIC_PERCENTILE)
            for window in windows
        ],
        dtype=float,
    )
    times = np.asarray(
        [
            (window["meta"]["window_start_sec"] + window["meta"]["window_end_sec"])
            / 2.0
            for window in windows
        ],
        dtype=float,
    )
    bleach, fit_meta = real_bleach_fit(times, baselines)
    assert fit_meta["fallback"] is False
    expected = baselines - bleach + fit_meta["anchor"]
    fallback_method = {
        "slope": np.nan,
        "intercept": np.nan,
        "n_used": np.nan,
        "tonic_method": subject.TONIC_METHOD_SIGNAL_ONLY,
        "units": subject.TONIC_UNITS_RAW_AU,
        "tonic_fallback": True,
        "fallback_reason": "global_fit_failed",
        "fallback_value_by_chunk": {
            index: float(value) for index, value in enumerate(expected)
        },
    }
    for index, row in enumerate(rows):
        assert row["tonic_value"] == pytest.approx(float(expected[index]))
        _, trace = subject._build_native_tonic_window_result(
            "ROI1", windows[index], fallback_method
        )
        assert np.all(np.isfinite(trace))
        assert np.percentile(trace, subject.TONIC_PERCENTILE) == pytest.approx(
            row["tonic_value"]
        )


def test_native_global_fit_sampling_is_bounded_and_incremental(monkeypatch):
    n_windows = 6
    samples_per_window = 40_000
    state = {"active": 0, "peak_active": 0, "windows_seen": 0}
    fit_input_sizes = []

    def window_factory():
        for index in range(n_windows):
            state["active"] += 1
            state["peak_active"] = max(
                state["peak_active"], state["active"]
            )
            try:
                local = np.arange(samples_per_window, dtype=float)
                reference = 100.0 + 0.5 * np.sin(local / 100.0)
                signal = 1.7 * reference + 12.0
                state["windows_seen"] += 1
                yield {"sig": signal, "uv": reference}
            finally:
                state["active"] -= 1

    real_fit = subject.compute_global_iso_fit_robust

    def counted_fit(uv, sig, *args, **kwargs):
        fit_input_sizes.append((int(uv.size), int(sig.size)))
        return real_fit(uv, sig, *args, **kwargs)

    monkeypatch.setattr(subject, "compute_global_iso_fit_robust", counted_fit)
    fit = subject._fit_global_from_window_factory(window_factory)

    assert fit["fit_ok"] is True
    assert fit["n_pairs"] == n_windows * samples_per_window
    assert fit_input_sizes == [
        (subject.GLOBAL_FIT_MAX_POINTS, subject.GLOBAL_FIT_MAX_POINTS)
    ]
    assert all(
        size <= subject.GLOBAL_FIT_MAX_POINTS
        for pair in fit_input_sizes
        for size in pair
    )
    assert state["windows_seen"] == n_windows * 2
    assert state["peak_active"] == 1


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
