"""Acceptance checks for scientist-visible missing-session plotting (4J16k41c)."""

import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_source_files,
    open_phasic_cache,
    open_tonic_cache,
)
from photometry_pipeline.viz.phasic_data_prep import (
    build_authoritative_plot_sessions,
    compute_day_layout,
)
from tests.test_missing_session_backend import (
    _build_input,
    _config,
    _run,
    _source,
)


def _phasic_gap_run(tmp_path: Path):
    inp = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    cfg = _config(tmp_path, authorized_missing_sessions=[_source(inp, 1)])
    return inp, _run(tmp_path, cfg, inp), cfg


def test_authoritative_plot_layout_keeps_missing_middle_slot(tmp_path: Path):
    _inp, out, _cfg = _phasic_gap_run(tmp_path)
    cache = open_phasic_cache(str(out / "phasic_trace_cache.h5"))
    try:
        cids = list_cache_chunk_ids(cache)
        sources = list_cache_source_files(cache)
    finally:
        cache.close()

    sessions = build_authoritative_plot_sessions(str(out), cids, sources)
    assert [item["session_index"] for item in sessions] == [0, 1, 2]
    assert [item["status"] for item in sessions] == [
        "valid",
        "missing_corrupted",
        "valid",
    ]
    assert [item["cache_chunk_id"] for item in sessions] == [0, None, 1]

    layout = compute_day_layout(
        [(0, sources[0]), (1, sources[1])],
        None,
        "Region0",
        sessions_per_hour=1,
        session_index_entries=sessions,
    )
    assert [chunk.session_index for chunk in layout.chunks] == [0, 1, 2]
    assert [chunk.hour_idx for chunk in layout.chunks] == [0, 1, 2]
    assert layout.chunks[1].status == "missing_corrupted"
    assert layout.chunks[1].cache_chunk_id is None


def test_phasic_summary_exports_nan_missing_row_and_real_later_time(tmp_path: Path):
    _inp, out, _cfg = _phasic_gap_run(tmp_path)
    export_dir = tmp_path / "summary"
    cmd = [
        sys.executable,
        "tools/plot_phasic_time_series_summary.py",
        "--analysis-out",
        str(out),
        "--roi",
        "Region0",
        "--sessions-per-hour",
        "1",
        "--session-duration-s",
        "60",
        "--out-dir",
        str(export_dir),
        "--export-csv",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr

    peak = pd.read_csv(export_dir / "phasic_peak_rate_timeseries.csv")
    missing = peak.loc[peak["session_index"] == 1].iloc[0]
    later = peak.loc[peak["session_index"] == 2].iloc[0]
    assert missing["status"] == "missing_corrupted"
    assert math.isnan(float(missing["peak_count"]))
    assert math.isnan(float(missing["peak_rate_per_min"]))
    assert missing["expected_start_time"] == "2024-01-01T01:00:00"
    assert float(later["time_hours"]) == 2.0

    auc = pd.read_csv(export_dir / "phasic_auc_timeseries.csv")
    missing_auc = auc.loc[auc["session_index"] == 1].iloc[0]
    assert missing_auc["status"] == "missing_corrupted"
    assert math.isnan(float(missing_auc["auc_above_threshold_dff_s"]))


def test_dayplot_bundle_renders_explicit_gap_tiles(tmp_path: Path):
    _inp, out, _cfg = _phasic_gap_run(tmp_path)
    plot_dir = tmp_path / "dayplots"
    cmd = [
        sys.executable,
        "tools/plot_phasic_dayplot_bundle.py",
        "--analysis-out",
        str(out),
        "--roi",
        "Region0",
        "--output-dir",
        str(plot_dir),
        "--sessions-per-hour",
        "1",
        "--hide-peak-markers",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {
        "phasic_dFF_day_000.png",
        "phasic_sig_iso_day_000.png",
        "phasic_dynamic_fit_day_000.png",
        "phasic_stacked_day_000.png",
    }.issubset({path.name for path in plot_dir.glob("*.png")})


def test_tonic_marker_uses_missing_session_timestamp(tmp_path: Path):
    _inp = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    cfg = _config(tmp_path, authorized_missing_sessions=[_source(_inp, 1)])
    out = _run(tmp_path, cfg, _inp, mode="tonic")

    from tools.plot_tonic_48h import _build_missing_intervals, assemble_arrays

    cache = open_tonic_cache(str(out / "tonic_trace_cache.h5"))
    try:
        args = SimpleNamespace(
            analysis_out=str(out),
            input=None,
            format="auto",
            sessions_per_hour=1,
            tonic_output_mode="preserve_raw_session_shape",
            tonic_timeline_mode="real_elapsed_time",
            include_visual_separators=False,
        )
        _time, _sig, _uv, _df, missing = assemble_arrays(
            cache, "Region0", args, return_missing_metadata=True
        )
    finally:
        cache.close()
    intervals = _build_missing_intervals(missing, sessions_per_hour=1)
    assert len(intervals) == 1
    assert intervals[0]["session_index"] == 1
    assert intervals[0]["status"] == "missing_corrupted"
    # Session 1 starts at hour 1; the marker is centered within its 60 s slot.
    assert 1.0 < intervals[0]["x_hours"] < 1.02


def test_tonic_plot_lines_and_display_csv_break_decimated_missing_gap(tmp_path: Path):
    """Every tonic display series breaks across a real missing middle session."""
    _inp = _build_input(tmp_path, corrupted=(1,), n_sessions=3)
    cfg = _config(tmp_path, authorized_missing_sessions=[_source(_inp, 1)])
    out = _run(tmp_path, cfg, _inp, mode="tonic")

    from tools.plot_tonic_48h import (
        _annotate_missing_intervals,
        _build_missing_intervals,
        _plot_tonic_series,
        _write_tonic_display_series_csv,
        assemble_arrays,
        compute_tonic_overview_display_decimation,
    )

    cache = open_tonic_cache(str(out / "tonic_trace_cache.h5"))
    try:
        args = SimpleNamespace(
            analysis_out=str(out),
            input=None,
            format="auto",
            sessions_per_hour=1,
            tonic_output_mode="preserve_raw_session_shape",
            tonic_timeline_mode="real_elapsed_time",
            include_visual_separators=False,
        )
        time_sec, sig, uv, deltaf, missing = assemble_arrays(
            cache, "Region0", args, return_missing_metadata=True
        )
    finally:
        cache.close()

    intervals = _build_missing_intervals(missing, sessions_per_hour=1)
    t_hours = time_sec / 3600.0
    decimate = compute_tonic_overview_display_decimation(
        len(t_hours), target_points=30
    )
    t_plot = t_hours[::decimate]
    arrays = [sig[::decimate], uv[::decimate], deltaf[::decimate]]
    left = intervals[0]["x_hours"] - intervals[0]["width_hours"] / 2.0
    right = intervals[0]["x_hours"] + intervals[0]["width_hours"] / 2.0

    fig, (ax_raw, ax_tonic) = plt.subplots(2, 1)
    _plot_tonic_series(
        ax_raw,
        t_plot,
        arrays[0],
        label="Sig",
        color="green",
        linewidth=0.5,
        missing_intervals=intervals,
    )
    _plot_tonic_series(
        ax_raw,
        t_plot,
        arrays[1],
        label="Iso",
        color="purple",
        linewidth=0.5,
        missing_intervals=intervals,
    )
    _plot_tonic_series(
        ax_tonic,
        t_plot,
        arrays[2],
        label="Tonic (deltaF)",
        color="black",
        linewidth=0.8,
        missing_intervals=intervals,
    )
    _annotate_missing_intervals(ax_raw, intervals)

    # There must be no Line2D whose finite samples lie on both sides of the gap.
    assert len(ax_raw.lines) == 4  # two segments each for raw and iso
    assert len(ax_tonic.lines) == 2  # two segments for tonic deltaF
    for axis in (ax_raw, ax_tonic):
        for line in axis.lines:
            x_data = line.get_xdata()
            finite_x = x_data[np.isfinite(x_data)]
            assert not (np.any(finite_x < left) and np.any(finite_x > right))
    assert len(ax_raw.patches) == 1
    assert "Missing/corrupted session" in ax_raw.texts[0].get_text()

    # Real elapsed timestamps after the gap are unchanged by display decimation.
    plotted_after = np.sort(
        np.concatenate(
            [line.get_xdata()[line.get_xdata() > right] for line in ax_raw.lines[:2]]
        )
    )
    expected_after = np.sort(t_plot[t_plot > right])
    assert np.array_equal(plotted_after, expected_after)

    # The CSV carries segment IDs for every display series, so rows cannot be
    # interpreted as one continuous trace after NaN separators were filtered.
    output_path = tmp_path / "tonic.png"
    _write_tonic_display_series_csv(
        out_path=str(output_path),
        roi="Region0",
        source_run_profile="full",
        t_plot=t_plot,
        sig_plot=arrays[0],
        uv_plot=arrays[1],
        deltaf_plot=arrays[2],
        decimate=decimate,
        missing_intervals=intervals,
    )
    display = pd.read_csv(tmp_path / "tonic_display_series.csv")
    assert "segment_id" in display.columns
    assert (display["trace_kind"] == "missing_session_marker").any()
    for trace_kind in ("sig_raw_display", "iso_raw_display", "tonic_deltaf_display"):
        sample = display[display["trace_kind"] == trace_kind]
        assert sample["segment_id"].nunique() == 2
        for _segment, segment_rows in sample.groupby("segment_id"):
            assert not (
                (segment_rows["x"] < left).any()
                and (segment_rows["x"] > right).any()
            )
    plt.close(fig)


def test_clean_tonic_display_remains_one_continuous_segment():
    from tools.plot_tonic_48h import _plot_tonic_series

    t = np.linspace(0.0, 2.0, 20)
    y = np.sin(t)
    fig, ax = plt.subplots()
    lines = _plot_tonic_series(
        ax,
        t,
        y,
        label="Sig",
        color="green",
        linewidth=0.5,
        missing_intervals=[],
    )
    assert len(lines) == 1
    assert np.array_equal(lines[0].get_xdata(), t)
    assert np.array_equal(lines[0].get_ydata(), y)
    y_with_visual_separator = y.copy()
    y_with_visual_separator[10] = np.nan
    ax.clear()
    lines_with_separator = _plot_tonic_series(
        ax,
        t,
        y_with_visual_separator,
        label="Sig",
        color="green",
        linewidth=0.5,
        missing_intervals=[],
    )
    assert len(lines_with_separator) == 1
    plt.close(fig)
