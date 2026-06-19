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
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)    # Go to Draft Plan step
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))

    summary_text = window._guided_draft_run_plan_preview_label.text()
    assert "Status: new_analysis draft plan" in summary_text
    assert "Format: rwd" in summary_text
    assert "Acquisition mode: intermittent" in summary_text
    assert "ROI counts:" in summary_text
    assert "Diagnostic cache: missing" in summary_text
    assert "Correction strategy coverage:" in summary_text
    assert "Feature/event profile status: default_initialized" in summary_text
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


def test_new_analysis_feature_event_profile_gui_flows(window, tmp_path, monkeypatch):
    # Case F & K: Initialize from Full Control active baseline, not bare Config()
    custom_cfg_path = tmp_path / "custom_config.yaml"
    custom_cfg_content = """
event_signal: delta_f
signal_excursion_polarity: negative
peak_threshold_method: absolute
peak_threshold_k: 4.5
peak_threshold_percentile: 92.5
peak_threshold_abs: 0.123
peak_min_distance_sec: 12.0
peak_min_prominence_k: 2.5
peak_min_width_sec: 1.5
peak_pre_filter: lowpass
event_auc_baseline: median
"""
    custom_cfg_path.write_text(custom_cfg_content, encoding="utf-8")
    
    # Configure custom baseline YAML in Full Control
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(custom_cfg_path))
    
    # Enter new_analysis mode and configure setup
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    
    # Go to Draft Plan step to initialize feature/event editor
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    
    # Verify the defaults are initialized from active baseline custom config (Case F and Case K drift test)
    assert window._guided_feature_event_polarity_combo.currentText() == "negative"
    assert window._guided_feature_event_peak_method_combo.currentText() == "absolute"
    assert window._guided_feature_event_peak_k_edit.text() == "4.5"
    assert window._guided_feature_event_peak_pct_edit.text() == "92.5"
    assert window._guided_feature_event_peak_abs_edit.text() == "0.123"
    assert window._guided_feature_event_peak_distance_edit.text() == "12.0"
    assert window._guided_feature_event_peak_prominence_edit.text() == "2.5"
    assert window._guided_feature_event_peak_width_edit.text() == "1.5"
    assert window._guided_feature_event_pre_filter_combo.currentText() == "lowpass"
    assert window._guided_feature_event_auc_baseline_combo.currentText() == "median"
    
    # Verify profile status is default_initialized, which does not count as configured/passed (blocking issue)
    assert window._guided_new_analysis_feature_event_profile_status == "default_initialized"
    summary_text = window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profile status: default_initialized" in summary_text
    checklist_text = window._guided_draft_run_plan_checklist_label.text()
    assert "Feature/event settings: fail" in checklist_text
    
    # Case G: GUI Apply valid feature/event settings in new_analysis
    window._guided_feature_event_apply_btn.click()
    assert window._guided_new_analysis_feature_event_profile_status == "applied"
    
    # Checklist should now pass for Feature/event settings
    summary_text_applied = window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profile status: applied" in summary_text_applied
    checklist_text_applied = window._guided_draft_run_plan_checklist_label.text()
    assert "Feature/event settings: pass" in checklist_text_applied
    
    # Case H: GUI Invalid Apply is local and non-executing
    # Set invalid value
    window._guided_feature_event_peak_abs_edit.setText("invalid_non_numeric")
    # Record current files to verify no outputs are written
    initial_files = list(tmp_path.glob("**/*"))
    window._guided_feature_event_apply_btn.click()
    
    assert window._guided_new_analysis_feature_event_profile_status == "invalid"
    # Verify status label contains error message
    assert "invalid" in window._guided_feature_event_status_label.text().lower() or "must be a number" in window._guided_feature_event_status_label.text().lower()
    # Verify no files were written
    current_files = list(tmp_path.glob("**/*"))
    assert len(current_files) == len(initial_files)
    # The applied profile itself is NOT updated/overwritten with invalid values (still has previous applied values or defaults)
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.123
    
    # Case I: GUI Baseline change marks applied profile stale
    # Restore valid value in widget first
    window._guided_feature_event_peak_abs_edit.setText("0.123")
    window._guided_feature_event_apply_btn.click()
    assert window._guided_new_analysis_feature_event_profile_status == "applied"
    
    # Create another custom config representing baseline drift
    another_cfg_path = tmp_path / "another_config.yaml"
    another_cfg_path.write_text("event_signal: delta_f\nsignal_excursion_polarity: positive\npeak_pre_filter: none", encoding="utf-8")
    
    # Change active baseline config source path in Full Control
    window._config_path.setText(str(another_cfg_path))
    # Trigger sync manually to simulate UI panel refresh/step control sync
    window._sync_guided_feature_event_editor_to_current_run()
    
    assert window._guided_new_analysis_feature_event_profile_status == "stale"
    assert "active baseline config source path changed" in window._guided_new_analysis_feature_event_profile_stale_reasons
    # Values are preserved, not overwritten with defaults
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.123
    
    # Case J: GUI Completed-run feature/event behavior still works (separate state)
    # Change workflow mode to open_results
    window._set_guided_workflow_mode("open_results")
    assert window._guided_new_analysis_feature_event_profile_status == "stale"  # new_analysis state remains stale
    
    # Sync in open_results mode should reset or load for current run (which is empty, so should reset to defaults)
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    assert window._guided_feature_event_status_label.text() == "No draft feature/event profile applied."


def test_new_analysis_feature_event_forced_refresh_and_clear_rules(window, tmp_path, monkeypatch):
    custom_cfg_path = tmp_path / "custom_config.yaml"
    custom_cfg_content = """
event_signal: delta_f
signal_excursion_polarity: negative
peak_threshold_method: absolute
peak_threshold_abs: 0.123
"""
    custom_cfg_path.write_text(custom_cfg_content, encoding="utf-8")
    
    # Configure custom config in Full Control
    window._use_custom_config_cb.setChecked(True)
    window._config_path.setText(str(custom_cfg_path))
    
    # Enter new_analysis mode and configure setup
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    
    # Go to Draft Plan step to initialize feature/event editor defaults (default_initialized)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    assert window._guided_new_analysis_feature_event_profile_status == "default_initialized"
    
    # 1. Test: forced refresh (force=True) on default_initialized when widgets are NOT edited should reload defaults
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    assert window._guided_new_analysis_feature_event_profile_status == "default_initialized"
    
    # 2. Test: forced refresh (force=True) on default_initialized when widgets ARE edited must NOT overwrite unapplied edits
    # Edit the widget value
    window._guided_feature_event_peak_abs_edit.setText("0.999")
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    # The widget text must still be the user's unapplied edit "0.999", not reset to defaults
    assert window._guided_feature_event_peak_abs_edit.text() == "0.999"
    assert window._guided_new_analysis_feature_event_profile_status == "default_initialized"
    
    # Apply to move status to "applied"
    window._guided_feature_event_apply_btn.click()
    assert window._guided_new_analysis_feature_event_profile_status == "applied"
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.999
    
    # 3. Test: forced refresh (force=True) on "applied" status must NOT discard stored values
    # Let's type something else in widgets
    window._guided_feature_event_peak_abs_edit.setText("0.111")
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    # The stored values must be preserved, and the editor display must refresh from stored applied state ("0.999", not "0.111")
    assert window._guided_new_analysis_feature_event_profile_status == "applied"
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.999
    assert window._guided_feature_event_peak_abs_edit.text() == "0.999"
    
    # 4. Test: forced refresh (force=True) on "stale" status must NOT discard stored values
    # Force baseline config change to mark stale
    another_cfg_path = tmp_path / "another_config.yaml"
    another_cfg_path.write_text("event_signal: dff\nsignal_excursion_polarity: positive", encoding="utf-8")
    window._config_path.setText(str(another_cfg_path))
    
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    assert window._guided_new_analysis_feature_event_profile_status == "stale"
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.999
    assert window._guided_feature_event_peak_abs_edit.text() == "0.999"
    
    # 5. Test: forced refresh (force=True) on "invalid" status must NOT discard stored values
    # Restore config path so we can apply again
    window._config_path.setText(str(custom_cfg_path))
    window._guided_feature_event_apply_btn.click() # Re-applies successfully
    assert window._guided_new_analysis_feature_event_profile_status == "applied"
    
    # Apply invalid value to trigger "invalid" status
    window._guided_feature_event_peak_abs_edit.setText("invalid_non_numeric")
    window._guided_feature_event_apply_btn.click()
    assert window._guided_new_analysis_feature_event_profile_status == "invalid"
    
    # Run force=True sync on invalid status
    window._sync_guided_feature_event_editor_to_current_run(force=True)
    assert window._guided_new_analysis_feature_event_profile_status == "invalid"
    assert window._guided_new_analysis_feature_event_profile["peak_threshold_abs"] == 0.999 # preserves last valid applied value
    
    # 6. Test: Clear/Reset reloads defaults from the active Full Control baseline config and sets status to default_initialized
    window._guided_feature_event_clear_btn.click()
    assert window._guided_new_analysis_feature_event_profile_status == "default_initialized"
    assert window._guided_feature_event_peak_abs_edit.text() == "0.123" # loaded defaults from active baseline custom_cfg_path
    
    # 7. Test: Summary display shows baseline source, status, and details when present
    window._guided_feature_event_apply_btn.click()
    summary_text = window._guided_draft_run_plan_preview_label.text()
    assert "Feature/event profile status: applied" in summary_text
    assert "Feature/event profile baseline source:" in summary_text
    assert "custom_config.yaml" in summary_text
    assert "Feature/event profile baseline status: custom_config" in summary_text


