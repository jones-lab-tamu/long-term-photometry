"""GUI-focused focused tests for the minimal Guided tonic-settings release.

Kept separate from tests/test_guided_tonic_settings.py so the fast unit
tests there don't pay Qt/MainWindow construction overhead.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import (
    GUIDED_TONIC_OUTPUT_MODE_CHOICES,
    GUIDED_TONIC_TIMELINE_MODE_CHOICES,
    MainWindow,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisTonicSettingsContract,
    evaluate_guided_new_analysis_execution_subset_readiness,
    evaluate_new_analysis_plan_readiness,
)
from tests.test_gui_guided_new_analysis_plan import _complete_new_analysis_plan_for_gui


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def test_both_tonic_controls_appear_in_guided_settings_step(window):
    assert window._guided_tonic_output_mode_combo.count() == len(
        GUIDED_TONIC_OUTPUT_MODE_CHOICES
    )
    assert window._guided_tonic_timeline_mode_combo.count() == len(
        GUIDED_TONIC_TIMELINE_MODE_CHOICES
    )
    assert window._guided_tonic_output_mode_combo.currentData() == (
        "preserve_raw_session_shape"
    )
    assert window._guided_tonic_timeline_mode_combo.currentData() == (
        "real_elapsed_time"
    )


def test_changing_output_mode_invalidates_prior_validation(window):
    before = int(getattr(window, "_guided_backend_validation_revision", 0))
    window._guided_tonic_output_mode_combo.setCurrentIndex(1)
    after = int(getattr(window, "_guided_backend_validation_revision", 0))
    assert after != before


def test_changing_timeline_mode_invalidates_prior_validation(window):
    before = int(getattr(window, "_guided_backend_validation_revision", 0))
    window._guided_tonic_timeline_mode_combo.setCurrentIndex(1)
    after = int(getattr(window, "_guided_backend_validation_revision", 0))
    assert after != before


def test_gap_free_selection_shows_persistent_note(window):
    window._guided_tonic_timeline_mode_combo.setCurrentIndex(1)
    assert window._guided_tonic_timeline_mode_combo.currentData() == (
        "gap_free_elapsed_time"
    )
    assert window._guided_tonic_gap_free_note_label.isHidden() is False
    window._guided_tonic_output_mode_combo.setCurrentIndex(0)
    # Real elapsed time (index 0) hides the note again.
    window._guided_tonic_timeline_mode_combo.setCurrentIndex(0)
    assert window._guided_tonic_gap_free_note_label.isHidden() is True


def test_gap_free_blocked_note_shown_when_missing_sessions_exist(window):
    window._guided_approved_missing_sessions = [object()]
    window._guided_tonic_timeline_mode_combo.setCurrentIndex(1)
    assert window._guided_tonic_gap_free_blocked_label.isHidden() is False
    assert "missing or excluded" in (
        window._guided_tonic_gap_free_blocked_label.text()
    )


def _render_review_checkpoint(window, plan):
    readiness = evaluate_new_analysis_plan_readiness(plan)
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    window._refresh_guided_review_plan_checkpoint(plan, readiness, subset)


def test_review_plan_displays_default_tonic_settings(window):
    plan = _complete_new_analysis_plan_for_gui()
    _render_review_checkpoint(window, plan)
    text = window._guided_review_analysis_summary_label.text()
    assert "Tonic session shape: Preserve session shape" in text
    assert "Tonic timeline: Real elapsed time" in text


def test_review_plan_displays_non_default_tonic_settings(window):
    plan = _complete_new_analysis_plan_for_gui(
        tonic_settings_contract=GuidedNewAnalysisTonicSettingsContract(
            tonic_output_mode="flatten_session_bleach_preserve_session_baseline",
            tonic_timeline_mode="gap_free_elapsed_time",
        )
    )
    _render_review_checkpoint(window, plan)
    text = window._guided_review_analysis_summary_label.text()
    assert "Tonic session shape: Remove within-session bleaching trend" in text
    assert "Tonic timeline: Gap-free elapsed time" in text
