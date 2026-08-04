"""Focused tests for the session-level tonic plot and its Results discovery."""

import os

import pandas as pd
import pytest

from photometry_pipeline.tonic_session_plot import (
    METHOD_GLOBAL_ISOSBESTIC,
    METHOD_SIGNAL_ONLY,
    TONIC_SESSION_PLOT_FILENAME,
    TonicSessionPlotError,
    _valid_runs,
    generate_tonic_session_plots,
    tonic_method_by_roi,
)
from photometry_pipeline.tonic_session_summary import (
    SUMMARY_COLUMNS,
    TONIC_SESSION_SUMMARY_FILENAME,
)


def _row(roi, index, value, status, method, units, start=None, reason=""):
    return {
        "roi": roi,
        "session_index": index,
        "source_file": f"sess{index:03d}/fluorescence.csv",
        "session_start_time": (
            start if start is not None else f"2025-01-01T{index // 2:02d}:{(index % 2) * 30:02d}:00"
        ),
        "tonic_value": value,
        "status": status,
        "tonic_method": method,
        "units": units,
        "n_finite_samples": 1000,
        "percentile": 2.0,
        "global_slope": 1.5 if method == METHOD_GLOBAL_ISOSBESTIC else float("nan"),
        "global_intercept": 20.0 if method == METHOD_GLOBAL_ISOSBESTIC else float("nan"),
        "global_fit_n_used": 5000 if method == METHOD_GLOBAL_ISOSBESTIC else float("nan"),
        "fallback_reason": reason,
    }


def _write_summary(tmp_path, rows):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = os.path.join(str(run_dir), TONIC_SESSION_SUMMARY_FILENAME)
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(path, index=False)
    return str(run_dir), path


def _primary_rows(roi="Region0", n=8):
    # Deliberately includes negative values.
    values = [-0.05, -0.02, 0.01, 0.04, 0.02, -0.01, -0.04, -0.06]
    return [
        _row(roi, i, values[i % len(values)], "valid", METHOD_GLOBAL_ISOSBESTIC, "fraction_dff")
        for i in range(n)
    ]


def _fallback_rows(roi="Region0", n=8, reason="nonpositive_global_slope"):
    return [
        _row(roi, i, 700.0 + i, "valid", METHOD_SIGNAL_ONLY, "raw_fluorescence_AU",
             reason=reason)
        for i in range(n)
    ]


# ------------------------------------------------------------ plot producer ---


def test_primary_method_plot_is_labeled_delta_f_over_f0(tmp_path):
    run_dir, _ = _write_summary(tmp_path, _primary_rows())
    results = generate_tonic_session_plots(run_dir)

    assert len(results) == 1
    result = results[0]
    assert result["tonic_method"] == METHOD_GLOBAL_ISOSBESTIC
    assert result["title"] == "Region0 — Tonic ΔF/F₀"
    assert result["y_label"] == "Tonic ΔF/F₀ (fraction)"
    assert result["units"] == "fraction_dff"
    assert os.path.isfile(result["output_path"])
    assert result["relative_path"] == f"Region0/summary/{TONIC_SESSION_PLOT_FILENAME}"


def test_fallback_method_plot_is_labeled_signal_only_and_uses_raw_au(tmp_path):
    run_dir, _ = _write_summary(tmp_path, _fallback_rows())
    result = generate_tonic_session_plots(run_dir)[0]

    assert result["tonic_method"] == METHOD_SIGNAL_ONLY
    assert result["title"] == "Region0 — Tonic F, signal-only bleach corrected"
    assert "raw fluorescence" in result["y_label"]
    assert "AU" in result["y_label"]
    assert "ΔF/F" not in result["y_label"]
    assert "ΔF/F" not in result["title"]
    assert os.path.isfile(result["output_path"])


def test_negative_values_are_plotted(tmp_path):
    rows = _primary_rows()
    assert any(row["tonic_value"] < 0 for row in rows)
    run_dir, _ = _write_summary(tmp_path, rows)
    result = generate_tonic_session_plots(run_dir)[0]
    assert result["n_plotted"] == len(rows)


def test_missing_and_invalid_sessions_remain_gaps(tmp_path):
    rows = _primary_rows(n=8)
    rows[2]["tonic_value"] = float("nan")
    rows[2]["status"] = "missing_corrupted"
    rows[5]["tonic_value"] = float("nan")
    rows[5]["status"] = "invalid_denominator"
    run_dir, _ = _write_summary(tmp_path, rows)
    result = generate_tonic_session_plots(run_dir)[0]

    assert result["n_sessions"] == 8
    assert result["n_plotted"] == 6  # the two unusable sessions are not drawn


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
    """No interpolation across missing or invalid sessions of any kind."""
    assert _valid_runs(statuses) == expected


def test_true_session_times_determine_x_positions(tmp_path):
    rows = _primary_rows(n=4)
    # Two sessions per hour, then a deliberate 6-hour hole before the last one.
    rows[0]["session_start_time"] = "2025-01-01T00:00:00"
    rows[1]["session_start_time"] = "2025-01-01T00:30:00"
    rows[2]["session_start_time"] = "2025-01-01T01:00:00"
    rows[3]["session_start_time"] = "2025-01-01T07:00:00"
    run_dir, _ = _write_summary(tmp_path, rows)
    result = generate_tonic_session_plots(run_dir)[0]
    assert result["x_label"] == "Time (hours from first session)"

    from photometry_pipeline.tonic_session_plot import _time_axis

    hours, _label = _time_axis([row["session_start_time"] for row in rows])
    assert hours == pytest.approx([0.0, 0.5, 1.0, 7.0])


def test_one_plot_per_roi(tmp_path):
    rows = _primary_rows("Region0") + _fallback_rows("Region1")
    run_dir, _ = _write_summary(tmp_path, rows)
    results = {item["roi"]: item for item in generate_tonic_session_plots(run_dir)}

    assert set(results) == {"Region0", "Region1"}
    assert results["Region0"]["tonic_method"] == METHOD_GLOBAL_ISOSBESTIC
    assert results["Region1"]["tonic_method"] == METHOD_SIGNAL_ONLY
    for item in results.values():
        assert os.path.isfile(item["output_path"])


def test_missing_summary_fails_closed(tmp_path):
    with pytest.raises(TonicSessionPlotError):
        generate_tonic_session_plots(str(tmp_path / "absent"))


# ---------------------------------------------------------- method records ---


def test_tonic_method_by_roi_reports_method_and_reason(tmp_path):
    rows = _primary_rows("Region0") + _fallback_rows("Region1", reason="global_fit_failed")
    run_dir, _ = _write_summary(tmp_path, rows)
    records = tonic_method_by_roi(run_dir)

    assert records["Region0"]["tonic_method"] == METHOD_GLOBAL_ISOSBESTIC
    assert records["Region0"]["fallback_reason"] == ""
    assert records["Region1"]["tonic_method"] == METHOD_SIGNAL_ONLY
    assert records["Region1"]["fallback_reason"] == "global_fit_failed"


def test_tonic_method_by_roi_is_empty_for_a_run_without_the_summary(tmp_path):
    """An older run simply has no record; no migration handling is needed."""
    assert tonic_method_by_roi(str(tmp_path)) == {}
