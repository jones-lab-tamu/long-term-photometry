"""Focused tests for the two-panel session-level tonic figure."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.core.types import Chunk
from photometry_pipeline.input_processing_completeness import (
    DISPOSITION_AUTHORIZED_MISSING,
    DISPOSITION_PROCESS,
    INPUT_COMPLETENESS_CONTRACT_VERSION,
    INPUT_COMPLETENESS_FILENAME,
    expected_entries_digest,
)
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.tonic_session_plot import (
    ELAPSED_AXIS_LABEL,
    METHOD_GLOBAL_ISOSBESTIC,
    METHOD_SIGNAL_ONLY,
    RAW_ISOSBESTIC_LABEL,
    RAW_SIGNAL_LABEL,
    SESSION_INDEX_AXIS_LABEL,
    TONIC_SESSION_PLOT_FILENAME,
    TonicSessionPlotError,
    _valid_runs,
    generate_tonic_session_plots,
    session_elapsed_seconds,
    tonic_method_by_roi,
)
from photometry_pipeline.viz.semantic_colors import (
    DFF_COLOR,
    RAW_REFERENCE_COLOR,
    RAW_SIGNAL_COLOR,
    SUMMARY_TRACE_COLOR,
)
from photometry_pipeline.tonic_session_summary import (
    SUMMARY_COLUMNS,
    TONIC_SESSION_SUMMARY_FILENAME,
)

FS = 20.0
NS = 120
ROI = "Region0"


# --------------------------------------------------------------- fixtures ---


def _completeness_record(starts, missing_indices=()):
    """Authoritative session index with explicit per-slot start times."""
    expected, processed = [], []
    cache_slot = 0
    for index, start in enumerate(starts):
        entry = {
            "source": f"/frozen/chunk_{index:04d}",
            "size_bytes": 100 + index,
            "sha256": f"digest{index}",
            "index": index,
            "disposition": DISPOSITION_PROCESS,
            "expected_start_time": start,
            "expected_duration_sec": 60.0,
        }
        if index in missing_indices:
            entry["disposition"] = DISPOSITION_AUTHORIZED_MISSING
            entry["failure_category"] = "corrupted_session"
            entry["reason"] = "approved corrupted session"
            entry["authorization_source"] = "run_config"
        else:
            processed.append(
                {"index": index, "source": entry["source"], "cache_chunk_id": cache_slot}
            )
            cache_slot += 1
        expected.append(entry)
    return {
        "contract_version": INPUT_COMPLETENESS_CONTRACT_VERSION,
        "acquisition_mode": "intermittent",
        "input_format": "rwd",
        "frozen_manifest_digest": expected_entries_digest(expected),
        "expected": expected,
        "processed": processed,
        "missing": [e for e in expected if e["disposition"] == DISPOSITION_AUTHORIZED_MISSING],
    }


def _write_cache(tonic_out, n_contributing, rois=(ROI,), seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(tonic_out, exist_ok=True)
    cache_path = os.path.join(tonic_out, "tonic_trace_cache.h5")
    with Hdf5TraceCacheWriter(cache_path, "tonic", config=None) as writer:
        for chunk_id in range(n_contributing):
            uv = np.column_stack(
                [300.0 + rng.normal(0, 0.5, NS) for _ in rois]
            )
            sig = np.column_stack(
                [1.5 * uv[:, i] + 20.0 + rng.normal(0, 0.5, NS) for i, _ in enumerate(rois)]
            )
            chunk = Chunk(
                chunk_id=chunk_id,
                source_file=f"/frozen/chunk_{chunk_id:04d}",
                format="rwd",
                time_sec=np.arange(NS, dtype=float) / FS,
                uv_raw=uv,
                sig_raw=sig,
                delta_f=np.zeros_like(sig),
                fs_hz=FS,
                channel_names=list(rois),
                metadata={},
            )
            writer.add_chunk(chunk, chunk_id=chunk_id, source_file=chunk.source_file)
    return cache_path


def _row(roi, index, value, status, method, units, start="", reason=""):
    return {
        "roi": roi,
        "session_index": index,
        "source_file": f"/frozen/chunk_{index:04d}",
        "session_start_time": start,
        "tonic_value": value,
        "status": status,
        "tonic_method": method,
        "units": units,
        "n_finite_samples": NS,
        "percentile": 2.0,
        "global_slope": 1.5 if method == METHOD_GLOBAL_ISOSBESTIC else float("nan"),
        "global_intercept": 20.0 if method == METHOD_GLOBAL_ISOSBESTIC else float("nan"),
        "global_fit_n_used": 5000 if method == METHOD_GLOBAL_ISOSBESTIC else float("nan"),
        "fallback_reason": reason,
    }


def _build_run(
    tmp_path,
    starts,
    *,
    missing_indices=(),
    method=METHOD_GLOBAL_ISOSBESTIC,
    units="fraction_dff",
    reason="",
    rois=(ROI,),
    write_completeness=True,
    invalid_indices=(),
):
    """A run directory with an authoritative index, a tonic cache, and a summary."""
    run_dir = tmp_path / "run"
    tonic_out = run_dir / "_analysis" / "tonic_out"
    tonic_out.mkdir(parents=True, exist_ok=True)

    if write_completeness:
        record = _completeness_record(starts, missing_indices)
        (tonic_out / INPUT_COMPLETENESS_FILENAME).write_text(
            json.dumps(record), encoding="utf-8"
        )

    n_contributing = len(starts) - len(missing_indices)
    _write_cache(str(tonic_out), n_contributing, rois=rois)

    rows = []
    for roi in rois:
        for index, start in enumerate(starts):
            if index in missing_indices:
                rows.append(
                    _row(roi, index, float("nan"), "missing_corrupted", method, units,
                         start=start, reason=reason)
                )
            elif index in invalid_indices:
                rows.append(
                    _row(roi, index, float("nan"), "invalid_denominator", method, units,
                         start=start, reason=reason)
                )
            else:
                rows.append(
                    _row(roi, index, -0.03 + 0.01 * index, "valid", method, units,
                         start=start, reason=reason)
                )
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(
        os.path.join(str(run_dir), TONIC_SESSION_SUMMARY_FILENAME), index=False
    )
    return str(run_dir)


def _half_hourly(n, gap_after=None, gap_hours=0.0):
    """Session starts every 30 minutes, optionally with one deliberate gap."""
    from datetime import datetime, timedelta

    origin = datetime(2025, 1, 1, 0, 0, 0)
    starts, offset = [], 0.0
    for index in range(n):
        starts.append((origin + timedelta(hours=offset)).isoformat())
        offset += 0.5
        if gap_after is not None and index == gap_after:
            offset += gap_hours
    return starts


# -------------------------------------------------------------- time axis ---


def test_normal_run_uses_elapsed_hours_not_session_index(tmp_path):
    run_dir = _build_run(tmp_path, _half_hourly(8))
    result = generate_tonic_session_plots(run_dir)[0]

    assert result["x_label"] == ELAPSED_AXIS_LABEL
    assert result["elapsed_hours"] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])


def test_a_deliberate_multi_hour_gap_keeps_its_true_width(tmp_path):
    # Six sessions every 30 min, with an extra 5 h hole after session 2.
    run_dir = _build_run(tmp_path, _half_hourly(6, gap_after=2, gap_hours=5.0))
    hours = generate_tonic_session_plots(run_dir)[0]["elapsed_hours"]

    assert hours == pytest.approx([0.0, 0.5, 1.0, 6.5, 7.0, 7.5])
    assert hours[3] - hours[2] == pytest.approx(5.5)  # 0.5 h cadence + 5 h gap


def test_missing_middle_and_final_sessions_keep_their_elapsed_positions(tmp_path):
    run_dir = _build_run(tmp_path, _half_hourly(8), missing_indices=(3, 7))
    result = generate_tonic_session_plots(run_dir)[0]

    assert result["elapsed_hours"] == pytest.approx(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    )
    # Missing slots hold their place but contribute no point and no raw trace.
    assert result["n_sessions"] == 8
    assert result["n_plotted"] == 6
    assert result["n_raw_sessions"] == 6


def test_incomplete_row_timestamps_do_not_force_session_index_fallback(tmp_path):
    """The authoritative index supplies timing even when CSV cells are blank."""
    starts = _half_hourly(6)
    run_dir = _build_run(tmp_path, starts)
    path = os.path.join(run_dir, TONIC_SESSION_SUMMARY_FILENAME)
    frame = pd.read_csv(path)
    frame.loc[frame["session_index"].isin([1, 4]), "session_start_time"] = ""
    frame.to_csv(path, index=False)

    result = generate_tonic_session_plots(run_dir)[0]
    assert result["x_label"] == ELAPSED_AXIS_LABEL
    assert result["elapsed_hours"] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])


def test_untimed_interior_slot_is_linearly_interpolated():
    from datetime import datetime, timedelta

    origin = datetime(2025, 1, 1)
    records = [
        {"session_index": 0, "expected_start_time": origin},
        {"session_index": 1, "expected_start_time": None},
        {"session_index": 2, "expected_start_time": origin + timedelta(hours=1.0)},
        {"session_index": 3, "expected_start_time": origin + timedelta(hours=1.5)},
    ]
    elapsed = session_elapsed_seconds(records)
    assert elapsed[0] == pytest.approx(0.0)
    assert elapsed[1] == pytest.approx(1800.0)
    assert elapsed[2] == pytest.approx(3600.0)
    assert elapsed[3] == pytest.approx(5400.0)


def test_untimed_interior_slot_does_not_bridge_a_later_recording_gap():
    from datetime import datetime, timedelta

    origin = datetime(2025, 1, 1)
    records = [
        {"session_index": 0, "expected_start_time": origin},
        {"session_index": 1, "expected_start_time": None},
        {"session_index": 2, "expected_start_time": origin + timedelta(hours=1)},
        {"session_index": 3, "expected_start_time": origin + timedelta(hours=6)},
    ]

    elapsed = session_elapsed_seconds(records)

    assert elapsed[1] == pytest.approx(1800.0)
    assert elapsed[0] < elapsed[1] < elapsed[2]
    assert elapsed[3] - elapsed[2] == pytest.approx(5 * 3600.0)


def test_leading_slots_use_the_nearest_right_side_cadence():
    from datetime import datetime, timedelta

    origin = datetime(2025, 1, 1)
    records = [
        {"session_index": 0, "expected_start_time": None},
        {"session_index": 1, "expected_start_time": None},
        {"session_index": 2, "expected_start_time": origin + timedelta(hours=5)},
        {"session_index": 3, "expected_start_time": origin + timedelta(hours=5.5)},
        {"session_index": 4, "expected_start_time": origin + timedelta(hours=7.5)},
    ]

    elapsed = session_elapsed_seconds(records)

    assert [elapsed[index] for index in range(5)] == pytest.approx(
        [0.0, 0.5 * 3600.0, 1.0 * 3600.0, 1.5 * 3600.0, 3.5 * 3600.0]
    )


def test_trailing_slots_use_the_nearest_left_side_cadence():
    from datetime import datetime, timedelta

    origin = datetime(2025, 1, 1)
    records = [
        {"session_index": 0, "expected_start_time": origin},
        {"session_index": 1, "expected_start_time": origin + timedelta(hours=0.5)},
        {"session_index": 2, "expected_start_time": origin + timedelta(hours=2.5)},
        {"session_index": 3, "expected_start_time": None},
        {"session_index": 4, "expected_start_time": None},
    ]

    elapsed = session_elapsed_seconds(records)

    assert [elapsed[index] for index in range(5)] == pytest.approx(
        [0.0, 0.5 * 3600.0, 2.5 * 3600.0, 4.5 * 3600.0, 6.5 * 3600.0]
    )


def test_unsafe_partial_timing_leaves_the_slot_unplaced(tmp_path):
    from datetime import datetime

    origin = datetime(2025, 1, 1).isoformat()
    run_dir = _build_run(tmp_path, [origin, "", ""])

    result = generate_tonic_session_plots(run_dir)[0]

    assert result["x_label"] == ELAPSED_AXIS_LABEL
    assert result["elapsed_hours"][0] == pytest.approx(0.0)
    assert np.isnan(result["elapsed_hours"][1])
    assert np.isnan(result["elapsed_hours"][2])
    assert result["n_plotted"] == 1
    assert result["n_raw_sessions"] == 1


def test_session_index_is_used_only_when_no_timing_exists(tmp_path):
    run_dir = _build_run(tmp_path, _half_hourly(5), write_completeness=False)
    result = generate_tonic_session_plots(run_dir)[0]

    assert result["x_label"] == SESSION_INDEX_AXIS_LABEL
    assert result["elapsed_hours"] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_session_index_fallback_aligns_raw_sessions_with_tonic_points(
    tmp_path, monkeypatch
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    captured = {}
    real_subplots = plt.subplots
    real_close = plt.close

    def spy(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        captured["figure"] = figure
        return figure, axes

    monkeypatch.setattr(plt, "subplots", spy)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    run_dir = _build_run(tmp_path, _half_hourly(5), write_completeness=False)
    result = generate_tonic_session_plots(run_dir)[0]

    raw_axis, tonic_axis = captured["figure"].axes
    assert result["x_label"] == SESSION_INDEX_AXIS_LABEL
    assert tonic_axis.get_xlabel() == SESSION_INDEX_AXIS_LABEL
    assert tonic_axis.get_shared_x_axes().joined(raw_axis, tonic_axis)

    tonic_points = [
        line for line in tonic_axis.get_lines() if line.get_marker() == "o"
    ]
    assert len(tonic_points) == 1
    assert list(tonic_points[0].get_xdata()) == pytest.approx(
        [0.0, 1.0, 2.0, 3.0, 4.0]
    )

    raw_lines = raw_axis.get_lines()
    assert len(raw_lines) == 10  # signal and isosbestic for five sessions
    session_ranges = []
    for index in range(5):
        pair = raw_lines[index * 2 : index * 2 + 2]
        pair_ranges = []
        for line in pair:
            x = np.asarray(line.get_xdata(), dtype=float)
            assert x[0] == pytest.approx(float(index))
            assert np.nanmin(x) >= float(index)
            assert np.nanmax(x) <= float(index) + 0.8
            pair_ranges.append((np.nanmin(x), np.nanmax(x)))
        session_ranges.append(pair_ranges)

    assert max(bound for bounds in session_ranges[0] for bound in bounds) < min(
        bound for bounds in session_ranges[1] for bound in bounds
    )
    real_close(captured["figure"])


def test_session_elapsed_seconds_is_empty_without_any_start_time():
    assert session_elapsed_seconds(
        [
            {"session_index": 0, "expected_start_time": None},
            {"session_index": 1, "expected_start_time": None},
        ]
    ) == {}


# ------------------------------------------------------------- two panels ---


def _axes_of(path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def test_figure_has_exactly_two_panels_with_raw_above_tonic(tmp_path, monkeypatch):
    """Capture the real figure the producer builds and inspect its axes."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    captured = {}
    real_subplots = plt.subplots

    def spy(*args, **kwargs):
        figure, axes = real_subplots(*args, **kwargs)
        captured["figure"] = figure
        captured["axes"] = axes
        return figure, axes

    real_close = plt.close
    monkeypatch.setattr(plt, "subplots", spy)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    run_dir = _build_run(tmp_path, _half_hourly(6))
    result = generate_tonic_session_plots(run_dir)[0]

    figure = captured["figure"]
    axes = list(figure.axes)
    assert len(axes) == 2

    raw_axis, tonic_axis = axes
    raw_labels = [line.get_label() for line in raw_axis.get_lines()]
    assert RAW_SIGNAL_LABEL in raw_labels
    assert RAW_ISOSBESTIC_LABEL in raw_labels
    assert raw_axis.get_ylabel() == "Raw fluorescence (AU)"

    assert tonic_axis.get_ylabel() == "Slow dF/F₀ — P2 per session"
    assert tonic_axis.get_xlabel() == ELAPSED_AXIS_LABEL
    # Both panels share one elapsed axis.
    assert raw_axis.get_shared_x_axes().joined(raw_axis, tonic_axis)

    saved = [line for line in tonic_axis.get_lines() if line.get_marker() == "o"]
    assert len(saved) == 1
    assert saved[0].get_color() == DFF_COLOR
    assert raw_axis.get_lines()[0].get_color() == RAW_SIGNAL_COLOR
    assert raw_axis.get_lines()[1].get_color() == RAW_REFERENCE_COLOR
    assert list(saved[0].get_xdata()) == pytest.approx(result["elapsed_hours"])
    real_close(figure)


def test_lower_panel_line_breaks_across_invalid_and_missing_sessions(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    captured = {}
    real_subplots = plt.subplots
    monkeypatch.setattr(
        plt, "subplots",
        lambda *a, **k: captured.setdefault("v", real_subplots(*a, **k)),
    )
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    run_dir = _build_run(
        tmp_path, _half_hourly(8), missing_indices=(2,), invalid_indices=(5,)
    )
    generate_tonic_session_plots(run_dir)

    _figure, (_raw_axis, tonic_axis) = captured["v"]
    # Solid connectors only: the marker series is linestyle "none" and the zero
    # reference line is dotted.
    connectors = [
        line for line in tonic_axis.get_lines()
        if line.get_linestyle() == "-" and len(line.get_xdata()) > 1
    ]
    # Sessions 0-1, 3-4, 6-7 -> three separate connected runs.
    assert len(connectors) == 3
    assert [len(line.get_xdata()) for line in connectors] == [2, 2, 2]


def test_no_old_residual_trace_is_plotted(tmp_path, monkeypatch):
    """The black deltaF overview trace must never reappear on this figure."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    captured = {}
    real_subplots = plt.subplots
    monkeypatch.setattr(
        plt, "subplots",
        lambda *a, **k: captured.setdefault("v", real_subplots(*a, **k)),
    )
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    run_dir = _build_run(tmp_path, _half_hourly(6))
    generate_tonic_session_plots(run_dir)

    _figure, (raw_axis, _tonic_axis) = captured["v"]
    labels = {str(line.get_label()) for line in raw_axis.get_lines()}
    assert labels <= {RAW_SIGNAL_LABEL, RAW_ISOSBESTIC_LABEL, "_nolegend_"}
    assert not any("deltaF" in label or "Tonic (deltaF)" in label for label in labels)
    # Raw sample counts, not a per-session residual summary.
    assert all(len(line.get_xdata()) == NS for line in raw_axis.get_lines())


# ----------------------------------------------------------- labels/units ---


def test_primary_method_labels_and_units(tmp_path):
    run_dir = _build_run(tmp_path, _half_hourly(6))
    result = generate_tonic_session_plots(run_dir)[0]

    assert result["title"] == f"{ROI} — Slow dF/F₀"
    assert result["y_label"] == "Slow dF/F₀ — P2 per session"
    assert result["units"] == "fraction_dff"
    assert result["raw_title"] == f"{ROI} — Signal and Reference Overview"
    assert os.path.isfile(result["output_path"])


def test_fallback_method_labels_units_and_neutral_color(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    captured = {}
    real_subplots = plt.subplots

    def spy(*args, **kwargs):
        captured["axes"] = real_subplots(*args, **kwargs)
        return captured["axes"]

    monkeypatch.setattr(plt, "subplots", spy)
    run_dir = _build_run(
        tmp_path, _half_hourly(6), method=METHOD_SIGNAL_ONLY,
        units="raw_fluorescence_AU", reason="nonpositive_global_slope",
    )
    result = generate_tonic_session_plots(run_dir)[0]

    assert result["title"] == f"{ROI} — Slow fluorescence (signal-only bleach-corrected)"
    assert result["y_label"] == "Slow fluorescence — P2 per session (AU)"
    assert "AU" in result["y_label"]
    _figure, (_raw_axis, slow_axis) = captured["axes"]
    assert all(
        line.get_color() == SUMMARY_TRACE_COLOR
        for line in slow_axis.get_lines()
    )
    assert "ΔF/F" not in result["y_label"]
    assert "ΔF/F" not in result["title"]


def test_negative_values_are_plotted(tmp_path):
    run_dir = _build_run(tmp_path, _half_hourly(6))
    result = generate_tonic_session_plots(run_dir)[0]
    frame = pd.read_csv(os.path.join(run_dir, TONIC_SESSION_SUMMARY_FILENAME))
    assert (frame["tonic_value"] < 0).any()
    assert result["n_plotted"] == 6


def test_one_figure_per_roi(tmp_path):
    run_dir = _build_run(tmp_path, _half_hourly(6), rois=("Region0", "Region1"))
    results = {item["roi"]: item for item in generate_tonic_session_plots(run_dir)}

    assert set(results) == {"Region0", "Region1"}
    for item in results.values():
        assert os.path.isfile(item["output_path"])
        assert item["relative_path"].endswith(TONIC_SESSION_PLOT_FILENAME)


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["valid"] * 4, [[0, 1, 2, 3]]),
        (["valid", "missing_corrupted", "valid", "valid"], [[0], [2, 3]]),
        (["valid", "invalid_denominator", "valid"], [[0], [2]]),
        (["insufficient_samples", "no_finite_samples"], []),
        (["valid", "tonic_unavailable", "valid", "missing_corrupted", "valid"],
         [[0], [2], [4]]),
    ],
)
def test_line_is_broken_across_every_unusable_status(statuses, expected):
    assert _valid_runs(statuses) == expected


# ----------------------------------------------------------- failure paths ---


def test_missing_summary_fails_closed(tmp_path):
    with pytest.raises(TonicSessionPlotError):
        generate_tonic_session_plots(str(tmp_path / "absent"))


def test_missing_cache_fails_closed(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [_row(ROI, 0, 0.1, "valid", METHOD_GLOBAL_ISOSBESTIC, "fraction_dff")],
        columns=SUMMARY_COLUMNS,
    ).to_csv(os.path.join(str(run_dir), TONIC_SESSION_SUMMARY_FILENAME), index=False)

    with pytest.raises(TonicSessionPlotError, match="tonic trace cache"):
        generate_tonic_session_plots(str(run_dir))


# ---------------------------------------------------------- method records ---


def test_tonic_method_by_roi_reports_method_and_reason(tmp_path):
    run_dir = _build_run(
        tmp_path, _half_hourly(4), method=METHOD_SIGNAL_ONLY,
        units="raw_fluorescence_AU", reason="global_fit_failed",
    )
    records = tonic_method_by_roi(run_dir)
    assert records[ROI]["tonic_method"] == METHOD_SIGNAL_ONLY
    assert records[ROI]["fallback_reason"] == "global_fit_failed"


def test_tonic_method_by_roi_is_empty_for_a_run_without_the_summary(tmp_path):
    assert tonic_method_by_roi(str(tmp_path)) == {}
