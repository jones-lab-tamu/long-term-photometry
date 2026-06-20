"""GUI tests for the new_analysis Guided draft plan summary."""

import json
from pathlib import Path
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedPlanCorrectionChoice,
    evaluate_new_analysis_plan_readiness,
)
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


def _complete_new_analysis_plan_for_gui(**overrides):
    plan = GuidedNewAnalysisDraftPlan(
        input_source_path="C:/raw/input",
        resolved_input_source_path="C:/raw/input",
        input_format="rwd",
        acquisition_mode="intermittent",
        sessions_per_hour=6,
        session_duration_sec=120.0,
        acquisition_structure_status="ready",
        discovered_roi_ids=["CH1"],
        included_roi_ids=["CH1"],
        cache_id="cache-1",
        cache_root_path="C:/cache",
        artifact_record_path="C:/cache/guided_diagnostic_cache_artifact.json",
        provenance_path="C:/cache/guided_diagnostic_cache_provenance.json",
        source_setup_signature="setup-1",
        diagnostic_scope_signature="scope-1",
        build_request_signature="build-1",
        stale_or_current="current",
        per_roi_correction_strategy_choices=[
            GuidedPlanCorrectionChoice(
                roi_id="CH1",
                selected_strategy="global_linear_regression",
                source_type="diagnostic_cache",
                diagnostic_cache_id="cache-1",
                diagnostic_cache_root="C:/cache",
                source_setup_signature="setup-1",
                diagnostic_scope_signature="scope-1",
                build_request_signature="build-1",
                current_or_stale="current",
                explicit_user_mark=True,
            )
        ],
        feature_event_profile_status="applied",
        feature_event_profile_id="feature-profile-1",
        feature_event_values={"event_signal": "dff"},
        feature_event_explicitly_applied=True,
        output_policy_status="applied",
        output_policy_path="C:/planned/output",
        output_policy_explicitly_applied=True,
    )
    for key, value in overrides.items():
        setattr(plan, key, value)
    return plan


def _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch, *, signal_only_f0=False):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    continuous_idx = window._guided_acquisition_mode_combo.findData("continuous")
    if continuous_idx >= 0:
        window._guided_acquisition_mode_combo.setCurrentIndex(continuous_idx)

    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Confirm strategy"))
    for index, roi in enumerate(("CH1", "CH2", "CH3")):
        window._guided_confirm_roi_combo.setCurrentIndex(window._guided_confirm_roi_combo.findData(roi))
        window._guided_confirm_chunk_combo.setCurrentIndex(0)
        strategy_text = "Signal-Only F0" if signal_only_f0 and index == 0 else "Global Linear Regression"
        window._guided_confirm_strategy_combo.setCurrentIndex(
            window._guided_confirm_strategy_combo.findText(strategy_text)
        )
        window._guided_confirm_ack_cb.setChecked(True)
        window._guided_confirm_mark_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_feature_event_apply_btn.click()

    output_parent = tmp_path / "planned_outputs"
    output_parent.mkdir()
    output_target = output_parent / "future_run_outputs"
    window._guided_output_path_edit.setText(str(output_target))
    window._guided_output_apply_btn.click()
    return output_parent, output_target


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
    assert "Output policy status: missing" in summary_text
    assert "Draft plan completeness: incomplete for future RunSpec handoff" in summary_text
    assert "Execution: unavailable, Final Guided Run/RunSpec is not implemented in this stage." in summary_text
    assert "This draft plan is not executable yet. Final Run is not implemented in this stage." in summary_text


def test_new_analysis_readiness_rendering_separates_planning_complete_from_execution_unavailable(window):
    plan = _complete_new_analysis_plan_for_gui()
    readiness = evaluate_new_analysis_plan_readiness(plan)

    summary = window._guided_new_analysis_draft_plan_summary_text(plan, readiness)
    readiness_summary = window._guided_new_analysis_readiness_summary_text(plan, readiness)
    window._refresh_guided_new_analysis_draft_plan_checklist(plan, readiness)
    checklist = window._guided_draft_run_plan_checklist_label.text()

    assert "Draft plan completeness: complete for future RunSpec handoff" in summary
    assert "Execution: unavailable, Final Guided Run/RunSpec is not implemented in this stage." in summary
    assert "Draft plan completeness: complete for future RunSpec handoff" in readiness_summary
    assert "Execution: unavailable, Final Guided Run/RunSpec is not implemented in this stage." in readiness_summary
    assert "Execution availability: unavailable" in checklist
    assert "Draft plan complete for handoff: true" in checklist
    assert "Execution available: false" in checklist


def test_new_analysis_readiness_rendering_shows_stale_feature_and_output_reasons(window):
    plan = _complete_new_analysis_plan_for_gui(
        feature_event_profile_status="stale",
        feature_event_stale_reasons=["baseline changed"],
        output_policy_status="stale",
        output_policy_stale_reasons=["target appeared"],
    )
    readiness = evaluate_new_analysis_plan_readiness(plan)

    summary = window._guided_new_analysis_draft_plan_summary_text(plan, readiness)
    readiness_summary = window._guided_new_analysis_readiness_summary_text(plan, readiness)
    window._refresh_guided_new_analysis_draft_plan_checklist(plan, readiness)
    checklist = window._guided_draft_run_plan_checklist_label.text()

    assert readiness.plan_complete_for_handoff is False
    assert "Feature/event profile stale reasons: baseline changed" in summary
    assert "Output policy stale reasons: target appeared" in summary
    assert "Feature/event settings (stale)" in readiness_summary
    assert "Output destination (stale)" in readiness_summary
    assert "Feature/event settings: fail - Feature/event profile is stale: baseline changed" in checklist
    assert "Output destination: fail - Output policy is stale: target appeared" in checklist
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


def test_new_analysis_run_preview_panel_renders_complete_plan(window, tmp_path, monkeypatch):
    _parent, output_target = _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Non-executing preview" in preview_text
    assert "Preview schema version: guided_new_analysis_run_preview.v1" in preview_text
    assert "Plan schema version: guided_new_analysis_plan.v1" in preview_text
    assert "Source/input:" in preview_text
    assert "Included ROIs: 3 (CH1, CH2, CH3)" in preview_text
    assert "Output destination:" in preview_text
    assert output_target.name in preview_text
    assert "Execution unavailable" in preview_text
    assert "Final Guided Run/RunSpec is not implemented in this stage." in preview_text
    assert "No files or directories were created." in preview_text
    assert not output_target.exists()


def test_new_analysis_run_preview_panel_shows_incomplete_plan_unresolved_items(window):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Non-executing preview" in preview_text
    assert "Draft plan completeness: incomplete for future RunSpec handoff" in preview_text
    assert "Run preview unresolved items:" in preview_text
    assert "missing_diagnostic_cache" in preview_text
    assert "missing_output_policy" in preview_text
    assert "Execution unavailable" in preview_text


def test_new_analysis_run_preview_complete_plan_keeps_execution_unavailable(window, tmp_path, monkeypatch):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "Draft plan completeness: complete for future RunSpec handoff" in preview_text
    assert "per_roi_correction_execution_contract_unresolved" in preview_text
    assert "preview preserves choices without collapsing them to a global strategy" in preview_text
    assert "global collapse false" in preview_text
    assert "Execution unavailable" in preview_text
    assert "ready to run" not in preview_text.lower()
    assert "ready for execution" not in preview_text.lower()


def test_new_analysis_run_preview_signal_only_f0_unresolved_routing(window, tmp_path, monkeypatch):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch, signal_only_f0=True)

    preview_text = window._guided_new_analysis_run_preview_label.text()

    assert "CH1: signal_only_f0" in preview_text
    assert "signal_only_f0_production_routing_unresolved" in preview_text
    assert "Signal-Only F0 production routing is not implemented" in preview_text
    assert "Execution unavailable" in preview_text


def test_new_analysis_run_preview_rendering_does_not_create_output_files(window, tmp_path, monkeypatch):
    output_parent, output_target = _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)
    before_files = sorted(str(path.relative_to(output_parent)) for path in output_parent.rglob("*"))

    window._refresh_guided_draft_run_plan_preview()

    after_files = sorted(str(path.relative_to(output_parent)) for path in output_parent.rglob("*"))
    assert after_files == before_files
    assert not output_target.exists()
    assert "No files or directories were created." in window._guided_new_analysis_run_preview_label.text()


def test_new_analysis_run_preview_rendering_does_not_mutate_completed_run_state(
    window, tmp_path, monkeypatch
):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)
    window._current_run_dir = str(tmp_path / "completed_run_sentinel")
    window._guided_draft_output_policy_by_run = {"completed": "policy"}
    window._guided_draft_feature_event_profiles_by_run = {"completed": [{"profile_id": "existing"}]}

    before_current_run = window._current_run_dir
    before_output_policies = dict(window._guided_draft_output_policy_by_run)
    before_feature_profiles = {
        key: list(value) for key, value in window._guided_draft_feature_event_profiles_by_run.items()
    }

    window._refresh_guided_draft_run_plan_preview()

    assert window._current_run_dir == before_current_run
    assert window._guided_draft_output_policy_by_run == before_output_policies
    assert window._guided_draft_feature_event_profiles_by_run == before_feature_profiles


def test_new_analysis_run_preview_rendering_does_not_call_execution_helpers(
    window, tmp_path, monkeypatch
):
    _configure_complete_guided_new_analysis_draft(window, tmp_path, monkeypatch)
    calls = []

    def forbidden(name):
        def _inner(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not be called by non-executing preview rendering")
        return _inner

    monkeypatch.setattr(window, "_build_run_spec", forbidden("_build_run_spec"))
    monkeypatch.setattr(window, "_build_argv", forbidden("_build_argv"))
    monkeypatch.setattr(window, "_on_validate", forbidden("_on_validate"))
    monkeypatch.setattr(window, "_on_run", forbidden("_on_run"))

    window._refresh_guided_draft_run_plan_preview()

    assert calls == []
    assert "Non-executing preview" in window._guided_new_analysis_run_preview_label.text()


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


def test_new_analysis_output_policy_typed_path_is_not_applied(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._refresh_guided_draft_run_plan_preview()

    assert window._guided_new_analysis_output_policy_status == "missing"
    summary_text = window._guided_draft_run_plan_preview_label.text()
    checklist_text = window._guided_draft_run_plan_checklist_label.text()
    assert "Output policy status: missing" in summary_text
    assert "Output destination: fail" in checklist_text
    assert not target.exists()


def test_new_analysis_output_policy_apply_valid_path_stores_state_without_creating_directory(
    window, tmp_path, monkeypatch
):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    before_files = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "applied"
    assert window._guided_new_analysis_output_policy_path == str(target.resolve())
    assert window._guided_new_analysis_output_policy_explicitly_applied is True
    assert "No directories or files were created" in window._guided_output_status_label.text()
    assert "Output policy status: applied" in window._guided_draft_run_plan_preview_label.text()
    assert target.name in window._guided_draft_run_plan_preview_label.text()
    assert "Output destination: pass" in window._guided_draft_run_plan_checklist_label.text()
    assert not target.exists()
    after_files = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert after_files == before_files


def test_new_analysis_output_policy_invalid_paths_are_rejected_without_creating_outputs(
    window, tmp_path, monkeypatch
):
    input_dir, _output_dir = _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    window._set_guided_workflow_mode("new_analysis")
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    valid_target = parent / "future_run_outputs"
    existing_target = parent / "existing_target"
    existing_target.mkdir()

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(existing_target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "already exists" in window._guided_output_status_label.text()

    window._guided_output_path_edit.setText(str(input_dir))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "must not be the same as the source path" in window._guided_output_status_label.text()

    window._guided_output_path_edit.setText(str(Path(input_dir) / "inside_input_outputs"))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "source/input folder" in window._guided_output_status_label.text()
    assert not (Path(input_dir) / "inside_input_outputs").exists()

    containing_target = tmp_path / "container_for_input"
    nested_source = containing_target / "raw_input"
    nested_source.mkdir(parents=True)
    nested_output = containing_target / "future_outputs"
    window._guided_output_path_edit.setText(str(nested_output))
    window._guided_input_dir_edit.setText(str(nested_source))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "Source/input folder must not be inside the output" in window._guided_output_status_label.text()
    assert not nested_output.exists()
    window._guided_input_dir_edit.setText(str(input_dir))

    window._guided_output_path_edit.setText(str(valid_target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "applied"
    assert not valid_target.exists()

    previous_path = window._guided_new_analysis_output_policy_path
    window._guided_output_path_edit.setText(str(existing_target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert window._guided_new_analysis_output_policy_path == previous_path


def test_new_analysis_output_policy_rejects_diagnostic_cache_overlap(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    fake_runner = _FakeDiagnosticCacheRunner()
    window._guided_diagnostic_cache_runner = fake_runner
    window._guided_diagnostic_cache_build_btn.click()
    cache_path = Path(fake_runner.run_dir)
    _write_minimal_guided_cache_outputs(cache_path)
    fake_runner.succeed()
    window._on_guided_diagnostic_cache_finished(0)

    target = cache_path / "future_run_outputs"
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "protected output/source root" in window._guided_output_status_label.text()
    assert not target.exists()

    preview_target = cache_path / "_guided_workflow" / "previews" / "future_outputs"
    window._guided_output_path_edit.setText(str(preview_target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "protected output/source root" in window._guided_output_status_label.text()
    assert not preview_target.exists()

    signal_target = cache_path / "_guided_workflow" / "signal_only_f0_diagnostics" / "future_outputs"
    window._guided_output_path_edit.setText(str(signal_target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "protected output/source root" in window._guided_output_status_label.text()
    assert not signal_target.exists()


def test_new_analysis_output_policy_marks_stale_when_context_changes(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    input_dir, _output_dir = _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "applied"

    new_input = tmp_path / "new_raw_input"
    new_input.mkdir()
    window._guided_input_dir_edit.setText(str(new_input))
    window._refresh_guided_draft_run_plan_preview()

    assert window._guided_new_analysis_output_policy_status == "stale"
    assert "input source path changed" in window._guided_new_analysis_output_policy_stale_reasons
    assert "Output policy status: stale" in window._guided_draft_run_plan_preview_label.text()
    assert "Output destination: fail" in window._guided_draft_run_plan_checklist_label.text()
    assert not target.exists()

    window._guided_input_dir_edit.setText(str(input_dir))
    window._refresh_guided_draft_run_plan_preview()

    assert window._guided_new_analysis_output_policy_status == "stale"
    assert "Output policy status: stale" in window._guided_draft_run_plan_preview_label.text()

    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "applied"
    assert window._guided_new_analysis_output_policy_stale_reasons == []
    assert not target.exists()


def test_new_analysis_output_policy_apply_requires_valid_source_context(window, tmp_path):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_input_dir_edit.setText("")
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "Raw input/source path is required" in window._guided_output_status_label.text()
    assert window._guided_new_analysis_output_policy_explicitly_applied is False
    assert not target.exists()

    missing_source = tmp_path / "missing_raw_source"
    window._guided_input_dir_edit.setText(str(missing_source))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()

    assert window._guided_new_analysis_output_policy_status == "invalid"
    assert "does not exist or is not a directory" in window._guided_output_status_label.text()
    assert window._guided_new_analysis_output_policy_explicitly_applied is False
    assert not target.exists()


def test_new_analysis_output_policy_marks_stale_when_target_appears(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "applied"

    target.mkdir()
    window._refresh_guided_draft_run_plan_preview()

    assert window._guided_new_analysis_output_policy_status == "stale"
    assert any("already exists" in reason for reason in window._guided_new_analysis_output_policy_stale_reasons)
    assert "Output destination: fail" in window._guided_draft_run_plan_checklist_label.text()


def test_new_analysis_output_policy_clear_removes_state(window, tmp_path, monkeypatch):
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()
    _configure_guided_raw_cache_setup(window, tmp_path, monkeypatch)
    parent = tmp_path / "planned_outputs"
    parent.mkdir()
    target = parent / "future_run_outputs"

    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Draft plan"))
    window._guided_output_path_edit.setText(str(target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "applied"

    window._guided_output_clear_btn.click()

    assert window._guided_new_analysis_output_policy_status == "missing"
    assert window._guided_new_analysis_output_policy_path is None
    assert window._guided_output_path_edit.text() == ""
    assert "Output policy status: missing" in window._guided_draft_run_plan_preview_label.text()
    assert not target.exists()
