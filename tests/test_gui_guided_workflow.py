import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QPushButton

from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


def _tab_labels(window: MainWindow) -> list[str]:
    tabs = window._workflow_mode_tabs
    return [tabs.tabText(i) for i in range(tabs.count())]


def _label_texts(widget) -> list[str]:
    return [label.text() for label in widget.findChildren(QLabel)]


def _make_window(qapp) -> MainWindow:
    return MainWindow()


def _close_window(window: MainWindow) -> None:
    window.close()
    window.deleteLater()


def _populate_fake_discovery(window: MainWindow) -> None:
    discovery = {
        "resolved_format": "rwd",
        "n_total_discovered": 2,
        "n_preview": 2,
        "sessions": [{"session_id": "s1"}, {"session_id": "s2"}],
        "rois": [{"roi_id": "CH1"}, {"roi_id": "CH2"}, {"roi_id": "CH3"}],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)


def _state_for_equivalence(window: MainWindow) -> dict[str, object]:
    state = dict(window._guided_setup_summary_state())
    return {
        key: state[key]
        for key in (
            "input_dir",
            "output_dir",
            "format",
            "resolved_format",
            "acquisition_mode",
            "sessions_per_hour",
            "session_duration_s",
            "continuous_window_sec",
            "continuous_step_sec",
            "allow_partial_final_window",
            "exclude_incomplete_final_rwd_chunk",
            "selected_roi_count",
            "total_roi_count",
            "selected_rois",
            "reference_correction_method",
            "reference_correction_label",
            "guided_correction_intent",
        )
    }


GUIDED_CARD_TO_DYNAMIC_MODE = {
    "Robust Global Event-Reject Fit": "robust_global_event_reject",
    "Adaptive Event-Gated Fit": "adaptive_event_gated_regression",
    "Global Linear Regression": "global_linear_regression",
}


def test_guided_workflow_and_full_control_tabs_are_accessible(window):
    assert _tab_labels(window) == ["Guided Workflow", "Full Control"]
    assert window._guided_workflow_tab.objectName() == "guidedWorkflowShell"
    assert window._full_control_tab.objectName() == "fullControlShell"


def test_guided_workflow_stepper_has_expected_steps(window):
    stepper = window._guided_workflow_stepper
    assert stepper.count() == len(GUIDED_WORKFLOW_STEPS)
    assert [stepper.item(i).data(0x0100) for i in range(stepper.count())] == list(GUIDED_WORKFLOW_STEPS)
    assert [stepper.item(i).text() for i in range(stepper.count())] == [
        f"{idx}. {step}" for idx, step in enumerate(GUIDED_WORKFLOW_STEPS, start=1)
    ]


def test_guided_workflow_stepper_switches_placeholder_panels(window):
    expected_panels = [
        "guidedStepSelectData",
        "guidedStepRecordingStructure",
        "guidedStepCorrectionApproach",
        "guidedStepDiagnostics",
        "guidedStepConfirmStrategy",
        "guidedStepRun",
        "guidedStepReview",
    ]
    for idx, expected_name in enumerate(expected_panels):
        window._guided_workflow_stepper.setCurrentRow(idx)
        assert window._guided_workflow_stack.currentWidget().objectName() == expected_name


def test_guided_correction_step_shows_expected_non_executing_cards(window):
    cards = window._guided_correction_cards
    assert list(cards) == [
        "Robust Global Event-Reject Fit",
        "Adaptive Event-Gated Fit",
        "Global Linear Regression",
        "Signal-Only F0",
        "Decision-Support Audit",
    ]
    assert "not recommended" in " ".join(_label_texts(cards["Global Linear Regression"])).lower()
    assert cards["Decision-Support Audit"].property("guidedCorrectionCardNonExecuting") is True
    assert "read-only evidence" in " ".join(_label_texts(cards["Decision-Support Audit"])).lower()
    assert "No Correction" not in cards
    assert "Decision-Support Audit" not in window._guided_correction_select_buttons


def test_no_correction_is_not_a_normal_guided_correction_card(window):
    card_titles = [
        card.property("guidedCorrectionCardTitle")
        for card in window._guided_workflow_tab.findChildren(QGroupBox)
        if card.property("guidedCorrectionCardTitle")
    ]
    assert "No Correction" not in card_titles


def test_full_control_preserves_existing_applied_dff_controls(window):
    tabs = window._workflow_mode_tabs
    tabs.setCurrentIndex(_tab_labels(window).index("Full Control"))
    assert window._applied_dff_group.isVisible() or window._applied_dff_group is not None
    assert window._applied_dff_save_manifest_btn.text() == "Save Manifest"
    assert window._applied_dff_dry_run_btn.text() == "Dry Run"
    assert window._applied_dff_run_batch_btn.text() == "Run Batch"


def test_guided_workflow_has_no_run_validate_or_manifest_action_buttons(window):
    forbidden = {"Run Pipeline", "Validate Only", "Save Manifest", "Dry Run", "Run Batch"}
    guided_button_texts = {
        button.text()
        for button in window._guided_workflow_tab.findChildren(QPushButton)
        if button.text()
    }
    assert guided_button_texts.isdisjoint(forbidden)


def test_guided_select_data_syncs_to_full_control_state(window, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    target_format = "custom_tabular"
    idx = window._guided_format_combo.findText(target_format)
    assert idx >= 0
    window._guided_format_combo.setCurrentIndex(idx)

    assert window._input_dir.text() == str(input_dir)
    assert window._output_dir.text() == str(output_dir)
    assert window._format_combo.currentText() == target_format


def test_full_control_select_data_syncs_to_guided_display(window, tmp_path):
    input_dir = tmp_path / "full_input"
    output_dir = tmp_path / "full_output"
    input_dir.mkdir()
    output_dir.mkdir()

    window._input_dir.setText(str(input_dir))
    window._output_dir.setText(str(output_dir))
    idx = window._format_combo.findText("auto")
    assert idx >= 0
    window._format_combo.setCurrentIndex(idx)

    assert window._guided_input_dir_edit.text() == str(input_dir)
    assert window._guided_output_dir_edit.text() == str(output_dir)
    assert window._guided_format_combo.currentText() == "auto"


def test_guided_recording_structure_syncs_to_full_control_state(window):
    idx = window._guided_acquisition_mode_combo.findData("continuous")
    assert idx >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)
    window._guided_continuous_window_sec_spin.setValue(900.0)
    window._guided_allow_partial_final_window_cb.setChecked(True)
    window._guided_exclude_incomplete_final_rwd_chunk_cb.setChecked(True)

    assert window._selected_acquisition_mode() == "continuous"
    assert float(window._continuous_window_sec_spin.value()) == 900.0
    assert float(window._continuous_step_sec_spin.value()) == 900.0
    assert window._allow_partial_final_window_cb.isChecked() is True
    assert window._exclude_incomplete_final_rwd_chunk_cb.isChecked() is True

    idx = window._guided_acquisition_mode_combo.findData("intermittent")
    window._guided_acquisition_mode_combo.setCurrentIndex(idx)
    window._guided_sessions_per_hour_edit.setText("6")
    window._guided_session_duration_edit.setText("300")

    assert window._selected_acquisition_mode() == "intermittent"
    assert window._sph_edit.text() == "6"
    assert window._duration_edit.text() == "300"


def test_full_control_recording_structure_syncs_to_guided_display(window):
    idx = window._acquisition_mode_combo.findData("continuous")
    assert idx >= 0
    window._acquisition_mode_combo.setCurrentIndex(idx)
    window._continuous_window_sec_spin.setValue(1200.0)
    window._allow_partial_final_window_cb.setChecked(True)

    assert window._guided_acquisition_mode_combo.currentData() == "continuous"
    assert float(window._guided_continuous_window_sec_spin.value()) == 1200.0
    assert window._guided_allow_partial_final_window_cb.isChecked() is True

    idx = window._acquisition_mode_combo.findData("intermittent")
    window._acquisition_mode_combo.setCurrentIndex(idx)
    window._sph_edit.setText("12")
    window._duration_edit.setText("100")

    assert window._guided_acquisition_mode_combo.currentData() == "intermittent"
    assert window._guided_sessions_per_hour_edit.text() == "12"
    assert window._guided_session_duration_edit.text() == "100"


def test_guided_roi_discovery_mirrors_existing_discovery_state(window):
    discovery = {
        "resolved_format": "rwd",
        "n_total_discovered": 2,
        "n_preview": 2,
        "sessions": [{"session_id": "s1"}, {"session_id": "s2"}],
        "rois": [{"roi_id": "CH1"}, {"roi_id": "CH2"}],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)

    assert window._guided_resolved_format_label.text() == "rwd"
    assert [window._guided_roi_list.item(i).text() for i in range(2)] == ["CH1", "CH2"]

    window._guided_roi_list.item(1).setCheckState(Qt.Unchecked)
    assert window._roi_list.item(1).checkState() == Qt.Unchecked

    window._roi_list.item(0).setCheckState(Qt.Unchecked)
    assert window._guided_roi_list.item(0).checkState() == Qt.Unchecked


def test_guided_roi_discovery_button_reuses_existing_discovery_handler(window, monkeypatch):
    called = {"discover": False}

    def _fake_discover():
        called["discover"] = True
        window._populate_discovery_ui(
            {
                "resolved_format": "custom_tabular",
                "n_total_discovered": 1,
                "n_preview": 1,
                "sessions": [{"session_id": "s1"}],
                "rois": [{"roi_id": "ROI_A"}],
            }
        )

    monkeypatch.setattr(window, "_on_discover", _fake_discover)
    window._on_guided_discover_rois()

    assert called["discover"] is True
    assert window._guided_roi_list.item(0).text() == "ROI_A"


def test_guided_setup_values_are_run_spec_relevant_state_equivalent(window, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._guided_format_combo.setCurrentText("custom_tabular")
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )
    window._guided_sessions_per_hour_edit.setText("4")
    window._guided_session_duration_edit.setText("600")

    assert window._input_dir.text() == str(input_dir)
    assert window._output_dir.text() == str(output_dir)
    assert window._format_combo.currentText() == "custom_tabular"
    assert window._selected_acquisition_mode() == "intermittent"
    assert window._sph_edit.text() == "4"
    assert window._duration_edit.text() == "600"

    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("continuous")
    )
    window._guided_continuous_window_sec_spin.setValue(750.0)

    assert window._selected_acquisition_mode() == "continuous"
    assert float(window._continuous_window_sec_spin.value()) == 750.0
    assert float(window._continuous_step_sec_spin.value()) == 750.0


def test_guided_setup_summary_is_read_only_and_tracks_current_state(window, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._guided_format_combo.setCurrentText("custom_tabular")
    window._guided_sessions_per_hour_edit.setText("8")

    text = window._guided_setup_summary_label.text()
    assert "Status: not validated" in text
    assert str(input_dir) in text
    assert str(output_dir) in text
    assert "custom_tabular" in text
    assert "sessions/hour=8" in text


def test_guided_and_full_control_intermittent_setup_summary_equivalence(qapp, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    guided = _make_window(qapp)
    full = _make_window(qapp)
    try:
        guided._guided_input_dir_edit.setText(str(input_dir))
        guided._guided_output_dir_edit.setText(str(output_dir))
        guided._guided_format_combo.setCurrentText("custom_tabular")
        guided._guided_acquisition_mode_combo.setCurrentIndex(
            guided._guided_acquisition_mode_combo.findData("intermittent")
        )
        guided._guided_sessions_per_hour_edit.setText("4")
        guided._guided_session_duration_edit.setText("600")

        full._input_dir.setText(str(input_dir))
        full._output_dir.setText(str(output_dir))
        full._format_combo.setCurrentText("custom_tabular")
        full._acquisition_mode_combo.setCurrentIndex(
            full._acquisition_mode_combo.findData("intermittent")
        )
        full._sph_edit.setText("4")
        full._duration_edit.setText("600")

        assert _state_for_equivalence(guided) == _state_for_equivalence(full)
    finally:
        _close_window(guided)
        _close_window(full)


def test_guided_and_full_control_continuous_setup_summary_equivalence(qapp, tmp_path):
    input_dir = tmp_path / "continuous_input"
    output_dir = tmp_path / "continuous_output"
    input_dir.mkdir()
    output_dir.mkdir()

    guided = _make_window(qapp)
    full = _make_window(qapp)
    try:
        guided._guided_input_dir_edit.setText(str(input_dir))
        guided._guided_output_dir_edit.setText(str(output_dir))
        guided._guided_format_combo.setCurrentText("auto")
        guided._guided_acquisition_mode_combo.setCurrentIndex(
            guided._guided_acquisition_mode_combo.findData("continuous")
        )
        guided._guided_continuous_window_sec_spin.setValue(900.0)
        guided._guided_allow_partial_final_window_cb.setChecked(True)

        full._input_dir.setText(str(input_dir))
        full._output_dir.setText(str(output_dir))
        full._format_combo.setCurrentText("auto")
        full._acquisition_mode_combo.setCurrentIndex(
            full._acquisition_mode_combo.findData("continuous")
        )
        full._continuous_window_sec_spin.setValue(900.0)
        full._allow_partial_final_window_cb.setChecked(True)

        assert _state_for_equivalence(guided) == _state_for_equivalence(full)
    finally:
        _close_window(guided)
        _close_window(full)


def test_guided_and_full_control_roi_selection_summary_equivalence(qapp):
    guided = _make_window(qapp)
    full = _make_window(qapp)
    try:
        _populate_fake_discovery(guided)
        _populate_fake_discovery(full)

        guided._guided_roi_list.item(1).setCheckState(Qt.Unchecked)
        full._roi_list.item(1).setCheckState(Qt.Unchecked)

        assert _state_for_equivalence(guided) == _state_for_equivalence(full)
    finally:
        _close_window(guided)
        _close_window(full)


def test_guided_and_full_control_rwd_final_chunk_option_equivalence(qapp):
    guided = _make_window(qapp)
    full = _make_window(qapp)
    try:
        guided._guided_exclude_incomplete_final_rwd_chunk_cb.setChecked(True)
        full._exclude_incomplete_final_rwd_chunk_cb.setChecked(True)

        assert _state_for_equivalence(guided) == _state_for_equivalence(full)
        assert guided._exclude_incomplete_final_rwd_chunk_cb.isChecked() is True
    finally:
        _close_window(guided)
        _close_window(full)


@pytest.mark.parametrize("card_title,mode", GUIDED_CARD_TO_DYNAMIC_MODE.items())
def test_guided_reference_correction_cards_sync_to_dynamic_fit_mode(window, card_title, mode):
    button = window._guided_correction_select_buttons[card_title]
    button.click()

    assert window._selected_dynamic_fit_mode() == mode
    assert window._guided_correction_cards[card_title].property("guidedCorrectionSelected") is True
    assert window._guided_correction_intent == card_title


@pytest.mark.parametrize("card_title,mode", GUIDED_CARD_TO_DYNAMIC_MODE.items())
def test_full_control_dynamic_fit_mode_syncs_to_guided_reference_card(window, card_title, mode):
    idx = window._dynamic_fit_mode_combo.findData(mode)
    assert idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)

    assert window._guided_correction_cards[card_title].property("guidedCorrectionSelected") is True
    assert window._guided_correction_intent == card_title


def test_guided_signal_only_f0_intent_does_not_change_dynamic_fit_mode_or_write_manifest(window, tmp_path):
    idx = window._dynamic_fit_mode_combo.findData("robust_global_event_reject")
    assert idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)
    before_mode = window._selected_dynamic_fit_mode()
    output_root = tmp_path / "out"
    window._applied_dff_output_root_edit.setText(str(output_root))

    window._guided_correction_select_buttons["Signal-Only F0"].click()

    assert window._selected_dynamic_fit_mode() == before_mode
    assert window._guided_correction_intent == "Signal-Only F0"
    assert window._guided_correction_cards["Signal-Only F0"].property("guidedCorrectionSelected") is True
    assert not (output_root / "gui_manifest").exists()
    assert not (output_root / "applied_dff_gui_provenance.json").exists()


def test_decision_support_audit_does_not_alter_dynamic_fit_mode(window):
    idx = window._dynamic_fit_mode_combo.findData("global_linear_regression")
    assert idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)
    before_mode = window._selected_dynamic_fit_mode()

    assert "Decision-Support Audit" not in window._guided_correction_select_buttons
    assert window._guided_correction_cards["Decision-Support Audit"].property("guidedCorrectionCardNonExecuting") is True
    assert window._selected_dynamic_fit_mode() == before_mode


def test_guided_setup_summary_reports_correction_state_without_validation_claim(window):
    window._guided_correction_select_buttons["Adaptive Event-Gated Fit"].click()
    text = window._guided_setup_summary_label.text()

    assert "Status: not validated" in text
    assert "diagnostics and strategy confirmation are not wired yet" in text
    assert "Reference correction method:" in text
    assert "adaptive_event_gated_regression" in text
    assert "Guided correction intent: Adaptive Event-Gated Fit" in text


def test_guided_diagnostics_step_has_status_context_and_slots(window):
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Diagnostics"))

    assert window._guided_workflow_stack.currentWidget().objectName() == "guidedStepDiagnostics"
    assert window._guided_diagnostics_status_label.text() == "Diagnostics: not generated; no completed run loaded"
    assert "Reference correction method:" in window._guided_diagnostics_context_label.text()
    assert "Decision-Support Audit: coming later / read-only evidence" in window._guided_diagnostics_context_label.text()
    assert "No Correction: not available in Guided Workflow" in window._guided_diagnostics_context_label.text()
    assert "Global linear baseline comparison" in window._guided_diagnostics_slot_labels
    assert "Decision-Support Audit evidence" in window._guided_diagnostics_slot_labels
    assert "not generated" in window._guided_diagnostics_slot_labels["Fit stability"].text()
    assert "No completed run is loaded" in window._guided_diagnostics_completed_run_label.text()


def test_guided_diagnostics_reports_existing_completed_run_artifacts_read_only(window, tmp_path):
    run_dir = tmp_path / "completed"
    run_dir.mkdir()
    (run_dir / "run_report.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (run_dir / "config_effective.yaml").write_text("event_signal: dff\n", encoding="utf-8")
    (run_dir / "gui_run_spec.json").write_text(json.dumps({"run": "spec"}), encoding="utf-8")
    (run_dir / "command_invoked.txt").write_text("python main.py\n", encoding="utf-8")
    summary_dir = run_dir / "CH1" / "summary"
    summary_dir.mkdir(parents=True)
    before = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))

    window._current_run_dir = str(run_dir)
    assert window._report_viewer.load_report(str(run_dir)) is True
    window._refresh_guided_diagnostics_panel()

    assert window._guided_diagnostics_status == "available"
    assert window._guided_diagnostics_status_label.text() == "Diagnostics: available from loaded completed-run artifacts"
    text = window._guided_diagnostics_completed_run_label.text()
    assert "Loaded completed run artifacts; separate from the active editable setup" in text
    assert "Run summary/report: run_report.json" in text
    assert "Status: status.json" in text
    assert "Manifest/provenance: MANIFEST.json" in text
    assert "Effective config: config_effective.yaml" in text
    assert "GUI run spec: gui_run_spec.json" in text
    assert "Command log: command_invoked.txt" in text
    assert "Region deliverable: Summary: CH1" in text
    after = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*"))
    assert after == before


def test_guided_diagnostics_loaded_run_without_recognized_artifacts_is_unavailable(window, tmp_path, monkeypatch):
    run_dir = tmp_path / "empty_loaded_run"
    run_dir.mkdir()

    window._current_run_dir = str(run_dir)
    monkeypatch.setattr(window._report_viewer, "has_loaded_results", lambda: True)
    window._refresh_guided_diagnostics_panel()

    assert window._guided_diagnostics_status == "unavailable"
    assert "Diagnostics: unavailable" in window._guided_diagnostics_status_label.text()
    text = window._guided_diagnostics_completed_run_label.text()
    assert str(run_dir) in text
    assert "No recognized completed-run diagnostic artifacts were found" in text


def test_guided_diagnostics_scope_loaded_artifacts_as_separate_from_active_setup(window, tmp_path):
    run_dir = tmp_path / "completed"
    run_dir.mkdir()
    (run_dir / "run_report.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    input_dir = tmp_path / "active_input"
    input_dir.mkdir()

    window._current_run_dir = str(run_dir)
    (run_dir / "CH1" / "summary").mkdir(parents=True)
    assert window._report_viewer.load_report(str(run_dir)) is True
    window._guided_input_dir_edit.setText(str(input_dir))
    window._refresh_guided_diagnostics_panel()

    assert window._guided_diagnostics_status == "available"
    text = window._guided_diagnostics_completed_run_label.text()
    assert "Loaded completed run artifacts; separate from the active editable setup" in text
    assert str(run_dir) in text
    assert str(input_dir) not in text


@pytest.mark.parametrize("card_title,mode", GUIDED_CARD_TO_DYNAMIC_MODE.items())
def test_guided_diagnostics_context_tracks_reference_correction_cards(window, card_title, mode):
    window._guided_correction_select_buttons[card_title].click()
    context = window._guided_diagnostics_context_label.text()

    assert mode in context
    assert f"Guided correction intent: {card_title}" in context
    assert window._guided_diagnostics_status_label.text() == "Diagnostics: not generated; no completed run loaded"


def test_guided_diagnostics_context_tracks_signal_only_intent(window):
    window._guided_correction_select_buttons["Signal-Only F0"].click()
    context = window._guided_diagnostics_context_label.text()

    assert "Guided correction intent: Signal-Only F0" in context
    assert "Signal-Only F0 intent: selected for later explicit confirmation" in context
    assert window._guided_diagnostics_status_label.text() == "Diagnostics: not generated; no completed run loaded"


def test_guided_diagnostics_step_has_no_generation_or_execution_buttons(window):
    forbidden = {
        "Run diagnostics",
        "Generate previews",
        "Compare fits",
        "Run correction tuning",
        "Apply diagnostics",
        "Auto choose strategy",
        "Validate Only",
        "Run Pipeline",
        "Save Manifest",
        "Run Batch",
    }
    button_texts = {
        button.text()
        for button in window._guided_workflow_tab.findChildren(QPushButton)
        if button.text()
    }
    assert button_texts.isdisjoint(forbidden)
