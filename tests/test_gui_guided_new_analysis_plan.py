"""GUI tests for the new_analysis Guided draft plan summary."""

import json
from pathlib import Path
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from tests.test_gui_guided_workflow import (
    _configure_guided_raw_cache_setup,
    _write_minimal_guided_cache_outputs,
    _FakeDiagnosticCacheRunner,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def test_new_analysis_draft_plan_displays_summary_fields(window, tmp_path, monkeypatch):
    # Enter new_analysis mode
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    assert window._guided_workflow_mode == "new_analysis"

    # Configure setup
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)

    # Go to Draft Plan step
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    summary_text = window._guided_draft_run_plan_preview_label.text()
    assert "Status: new_analysis draft plan" in summary_text
    assert "Format: rwd" in summary_text
    assert "Acquisition mode: intermittent" in summary_text
    assert "ROI counts:" in summary_text
    assert "Diagnostic cache: missing" in summary_text
    assert "Correction strategy coverage:" in summary_text
    assert "Feature/event profile status: unavailable" in summary_text
    assert "Output policy status: unavailable" in summary_text
    assert "This draft plan is not executable yet. Final Run is not implemented in this stage." in summary_text


def test_new_analysis_draft_plan_reports_choices_as_current_after_build_and_mark(window, tmp_path, monkeypatch):
    # Enter new_analysis mode and configure
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)

    # Build cache
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)

    # Confirm strategy step: select ROI and mark
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Confirm strategy"))
    
    # Check that we can select and mark CH1
    window._guided_confirm_roi_combo.setCurrentIndex(window._guided_confirm_roi_combo.findData("CH1"))
    window._guided_confirm_chunk_combo.setCurrentIndex(0)
    window._guided_confirm_strategy_combo.setCurrentIndex(window._guided_confirm_strategy_combo.findText("Global Linear Regression"))
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    # Verify choices are recorded as current in Draft Plan summary
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    summary_text = window._guided_draft_run_plan_preview_label.text()
    
    assert "Diagnostic cache: current" in summary_text
    assert "1/3 ROIs covered" in summary_text


def test_new_analysis_draft_plan_marks_stale_when_roi_selection_changes(window, tmp_path, monkeypatch):
    # Enter mode, configure, build cache, mark strategy
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)

    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    _write_minimal_guided_cache_outputs(Path(fake_runner.run_dir))
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)

    # Confirm CH1
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Confirm strategy"))
    window._guided_confirm_roi_combo.setCurrentIndex(window._guided_confirm_roi_combo.findData("CH1"))
    window._guided_confirm_chunk_combo.setCurrentIndex(0)
    window._guided_confirm_strategy_combo.setCurrentIndex(window._guided_confirm_strategy_combo.findText("Global Linear Regression"))
    window._guided_confirm_ack_cb.setChecked(True)
    window._guided_confirm_mark_btn.click()

    # Switch back to Select data or ROI discovery and change ROI inclusion selection
    # Uncheck CH2
    for i in range(window._guided_roi_list.count()):
        item = window._guided_roi_list.item(i)
        if item.text() == "CH2":
            item.setCheckState(Qt.Unchecked)

    # Go to Draft Plan step and check that summary/cache is marked stale
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    summary_text = window._guided_draft_run_plan_preview_label.text()

    assert "Diagnostic cache: stale" in summary_text
    assert "stale_diagnostic_cache" in summary_text


def test_new_analysis_draft_plan_no_runspec_or_outputs_written(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)

    # Records initial files in temp directories
    initial_files = list(tmp_path.glob("**/*"))

    # Refresh plan summary
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._refresh_guided_draft_run_plan_preview()

    # Assert no new files/directories were created under output base
    current_files = list(tmp_path.glob("**/*"))
    assert len(current_files) == len(initial_files)

    # Assert no RunSpec was generated
    assert not hasattr(window, "_generated_run_spec") or window._generated_run_spec is None
    assert window._current_run_dir == ""
