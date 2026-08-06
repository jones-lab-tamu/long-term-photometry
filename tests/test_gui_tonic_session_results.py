"""Focused Results-discovery tests for the session-level tonic view."""

import os

import pytest
from PySide6.QtWidgets import QApplication

from gui.run_report_viewer import (
    TAB_PHASIC_SUMMARY,
    TAB_TONIC,
    TAB_VERIFICATION,
    _NATIVE_CONTINUOUS_IMAGE_TABS,
    RunReportViewer,
)
from photometry_pipeline.tonic_session_plot import (
    METHOD_GLOBAL_ISOSBESTIC,
    METHOD_SIGNAL_ONLY,
    TONIC_FALLBACK_NOTE,
    TONIC_SESSION_PLOT_FILENAME,
    tonic_method_label,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _region_with(tmp_path, *filenames, region="Region0"):
    summary_dir = tmp_path / region / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (summary_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(tmp_path / region), str(summary_dir)


def test_tonic_tab_discovers_the_session_plot(qapp, tmp_path):
    _region_path, summary_dir = _region_with(
        tmp_path, TONIC_SESSION_PLOT_FILENAME, "tonic_overview.png"
    )
    viewer = RunReportViewer()
    found = viewer._discover_tonic_images(summary_dir)

    assert [os.path.basename(path) for path in found] == [TONIC_SESSION_PLOT_FILENAME]


def test_tonic_overview_is_not_shown_in_any_scientist_facing_tab(qapp, tmp_path):
    """The full residual trace stays on disk but is discovered by no tab."""
    region_path, _summary_dir = _region_with(
        tmp_path,
        TONIC_SESSION_PLOT_FILENAME,
        "tonic_overview.png",
        "phasic_correction_impact.png",
        "phasic_auc_timeseries.png",
        "phasic_peak_rate_timeseries.png",
    )
    viewer = RunReportViewer()
    by_tab = viewer._discover_region_tab_images(region_path)

    every_discovered = [
        os.path.basename(path) for paths in by_tab.values() for path in paths
    ]
    assert "tonic_overview.png" not in every_discovered
    assert os.path.isfile(os.path.join(region_path, "summary", "tonic_overview.png"))
    assert [os.path.basename(p) for p in by_tab[TAB_TONIC]] == [
        TONIC_SESSION_PLOT_FILENAME
    ]


def test_unrelated_tabs_are_unchanged(qapp, tmp_path):
    region_path, _summary_dir = _region_with(
        tmp_path,
        TONIC_SESSION_PLOT_FILENAME,
        "phasic_correction_impact.png",
        "phasic_auc_timeseries.png",
        "phasic_peak_rate_timeseries.png",
    )
    viewer = RunReportViewer()
    by_tab = viewer._discover_region_tab_images(region_path)

    assert [os.path.basename(p) for p in by_tab[TAB_VERIFICATION]] == [
        "phasic_correction_impact.png"
    ]
    assert sorted(os.path.basename(p) for p in by_tab[TAB_PHASIC_SUMMARY]) == [
        "phasic_auc_timeseries.png",
        "phasic_peak_rate_timeseries.png",
    ]


def test_continuous_tonic_discovery_is_unchanged():
    """Native continuous runs use their own artifact index, untouched here."""
    assert _NATIVE_CONTINUOUS_IMAGE_TABS["tonic_overview.png"] == TAB_TONIC


def test_run_without_the_session_plot_discovers_no_tonic_image(qapp, tmp_path):
    """An older run has nothing to show, rather than the residual trace."""
    _region_path, summary_dir = _region_with(tmp_path, "tonic_overview.png")
    viewer = RunReportViewer()
    assert viewer._discover_tonic_images(summary_dir) == []


# ------------------------------------------------------- fallback note ---


def _viewer_with_roi(roi, method, reason=""):
    viewer = RunReportViewer()
    viewer._tonic_method_by_roi = {
        roi: {"tonic_method": method, "units": "", "fallback_reason": reason}
    }
    viewer._region_combo.blockSignals(True)
    viewer._region_combo.clear()
    viewer._region_combo.addItem(roi)
    viewer._region_combo.setCurrentText(roi)
    viewer._region_combo.blockSignals(False)
    return viewer


def test_fallback_explanation_appears_once_for_a_fallback_roi(qapp):
    viewer = _viewer_with_roi("Region0", METHOD_SIGNAL_ONLY, "nonpositive_global_slope")
    viewer._refresh_tonic_method_note()

    text = viewer._tonic_method_note_label.text()
    assert text.startswith(
        f"Slow-signal method: {tonic_method_label(METHOD_SIGNAL_ONLY)}."
    )
    assert TONIC_FALLBACK_NOTE in text
    assert "nonpositive_global_slope" in text
    # Exactly one explanation for the ROI -- never repeated per session.
    assert text.count(TONIC_FALLBACK_NOTE) == 1
    assert not viewer._tonic_method_note_label.isHidden()


def test_primary_method_states_its_method_without_a_fallback_warning(qapp):
    viewer = _viewer_with_roi("Region0", METHOD_GLOBAL_ISOSBESTIC)
    viewer._refresh_tonic_method_note()

    text = viewer._tonic_method_note_label.text()
    assert text == (
        f"Slow-signal method: {tonic_method_label(METHOD_GLOBAL_ISOSBESTIC)}."
    )
    assert TONIC_FALLBACK_NOTE not in text


def test_note_switches_method_between_rois(qapp):
    viewer = RunReportViewer()
    viewer._tonic_method_by_roi = {
        "Region0": {"tonic_method": METHOD_SIGNAL_ONLY, "fallback_reason": "global_fit_failed"},
        "Region1": {"tonic_method": METHOD_GLOBAL_ISOSBESTIC, "fallback_reason": ""},
    }
    viewer._region_combo.blockSignals(True)
    viewer._region_combo.clear()
    viewer._region_combo.addItems(["Region0", "Region1"])
    viewer._region_combo.blockSignals(False)

    viewer._region_combo.setCurrentText("Region0")
    viewer._refresh_tonic_method_note()
    assert TONIC_FALLBACK_NOTE in viewer._tonic_method_note_label.text()

    viewer._region_combo.setCurrentText("Region1")
    viewer._refresh_tonic_method_note()
    assert viewer._tonic_method_note_label.text() == (
        f"Slow-signal method: {tonic_method_label(METHOD_GLOBAL_ISOSBESTIC)}."
    )


def test_stale_tonic_settings_lines_are_suppressed_for_new_runs(qapp):
    """Session shape / tonic timeline no longer govern the result, so they go."""
    viewer = RunReportViewer()
    viewer._completed_review_overview = {
        "tonic_settings": {
            "tonic_output_mode": "flatten_session_bleach_preserve_session_baseline",
            "tonic_timeline_mode": "real_elapsed_time",
            "timeline_anchor_mode": "fixed_daily_anchor",
        }
    }
    viewer._tonic_method_by_roi = {
        "Region0": {"tonic_method": METHOD_GLOBAL_ISOSBESTIC, "fallback_reason": ""}
    }
    viewer._refresh_tonic_settings_summary()

    text = viewer._tonic_settings_summary_label.text()
    assert text == ""
    assert "Tonic timeline" not in text
    assert "Session shape" not in text


def test_older_run_without_the_summary_keeps_its_original_line(qapp):
    """No migration code: a run predating the summary is untouched."""
    viewer = RunReportViewer()
    viewer._completed_review_overview = {
        "tonic_settings": {
            "tonic_output_mode": "flatten_session_bleach_preserve_session_baseline",
            "tonic_timeline_mode": "real_elapsed_time",
        }
    }
    viewer._tonic_method_by_roi = {}
    viewer._refresh_tonic_settings_summary()

    assert viewer._tonic_settings_summary_label.text() != ""
