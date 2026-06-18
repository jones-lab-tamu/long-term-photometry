import json

import h5py
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QPushButton

import gui.main_window as main_window_module
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


def _make_preview_completed_run(tmp_path):
    run_dir = tmp_path / "completed_preview"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    (run_dir / "run_report.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (run_dir / "CH1" / "summary").mkdir(parents=True)
    (phasic_out / "config_used.yaml").write_text(
        "target_fs_hz: 20.0\nlowpass_hz: 1.0\nfilter_order: 3\n"
        "dynamic_fit_mode: robust_global_event_reject\n",
        encoding="utf-8",
    )
    t = np.arange(400, dtype=float) / 20.0
    uv = 1.0 + 0.02 * np.sin(t * 0.7)
    sig = 1.2 * uv + 0.05 * np.exp(-0.5 * ((t - 8.0) / 0.5) ** 2)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([b"CH1", b"CH2"]))
        meta.create_dataset("chunk_ids", data=np.asarray([0, 1], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"mock0.csv", b"mock1.csv"]))
        for roi in ("CH1", "CH2"):
            roi_group = h5.create_group(f"roi/{roi}")
            for chunk_id in (0, 1):
                grp = roi_group.create_group(f"chunk_{chunk_id}")
                grp.create_dataset("time_sec", data=t)
                grp.create_dataset("sig_raw", data=sig + chunk_id)
                grp.create_dataset("uv_raw", data=uv + 0.1 * chunk_id)
    return run_dir


def _load_preview_completed_run(window, run_dir, monkeypatch):
    window._current_run_dir = str(run_dir)
    monkeypatch.setattr(window._report_viewer, "has_loaded_results", lambda: True)
    window._refresh_guided_diagnostics_panel()


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


def test_guided_summary_and_planned_sections_are_collapsible(window, tmp_path, monkeypatch):
    summary_group = window._guided_workflow_tab.findChild(QGroupBox, "guidedSetupSummaryPanel")
    planned_group = window._guided_workflow_tab.findChild(QGroupBox, "guidedWorkflowPlannedStages")
    assert summary_group is not None
    assert planned_group is not None
    assert summary_group.isCheckable() is True
    assert planned_group.isCheckable() is True
    assert summary_group.isChecked() is False
    assert planned_group.isChecked() is False
    assert window._guided_setup_summary_content.isHidden() is True
    assert window._guided_planned_stages_content.isHidden() is True

    state_before = _state_for_equivalence(window)
    calls = {"preview": 0}

    def _fake_preview_backend(*_args, **_kwargs):
        calls["preview"] += 1
        return {}

    monkeypatch.setattr(main_window_module, "run_guided_correction_preview_comparison", _fake_preview_backend)
    summary_group.setChecked(True)
    planned_group.setChecked(True)
    assert window._guided_setup_summary_content.isHidden() is False
    assert window._guided_planned_stages_content.isHidden() is False
    summary_group.setChecked(False)
    planned_group.setChecked(False)
    assert window._guided_setup_summary_content.isHidden() is True
    assert window._guided_planned_stages_content.isHidden() is True
    assert _state_for_equivalence(window) == state_before
    assert calls["preview"] == 0

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_output_dir_edit.setText(str(output_dir))
    window._guided_format_combo.setCurrentText("custom_tabular")
    assert window._guided_setup_summary_content.isHidden() is True
    text = window._guided_setup_summary_label.text()
    assert str(input_dir) in text
    assert str(output_dir) in text
    assert "custom_tabular" in text


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
    assert "Load a completed run to generate preview-only correction comparisons" in window._guided_preview_source_status_label.text()
    assert window._guided_preview_generate_btn.text() == "Generate preview comparison"
    assert window._guided_preview_generate_btn.isEnabled() is False
    assert window._guided_preview_result_label.text() == ""
    signal_panel = window._guided_workflow_tab.findChild(QGroupBox, "guidedSignalOnlyF0DiagnosticPanel")
    preview_panel = window._guided_workflow_tab.findChild(QGroupBox, "guidedCorrectionPreviewPanel")
    assert signal_panel is not None
    assert preview_panel is not None
    assert signal_panel is not preview_panel
    assert "Load a completed run to generate Signal-Only F0 diagnostic review artifacts" in (
        window._guided_signal_f0_source_status_label.text()
    )
    assert window._guided_signal_f0_generate_btn.text() == "Generate Signal-Only F0 diagnostic review"
    assert window._guided_signal_f0_generate_btn.isEnabled() is False


def test_guided_correction_preview_panel_populates_from_loaded_completed_run(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)

    _load_preview_completed_run(window, run_dir, monkeypatch)

    assert "Preview is generated from the loaded completed run" in window._guided_preview_source_status_label.text()
    assert str(run_dir) in window._guided_preview_source_status_label.text()
    assert [window._guided_preview_roi_combo.itemText(i) for i in range(window._guided_preview_roi_combo.count())] == [
        "CH1",
        "CH2",
    ]
    assert [window._guided_preview_chunk_combo.itemData(i) for i in range(window._guided_preview_chunk_combo.count())] == [
        0,
        1,
    ]
    assert set(window._guided_preview_method_checkboxes) == {
        "robust_global_event_reject",
        "adaptive_event_gated_regression",
        "global_linear_regression",
    }
    assert all(cb.isChecked() for cb in window._guided_preview_method_checkboxes.values())
    assert window._guided_preview_generate_btn.isEnabled() is True
    method_text = " ".join(cb.text() for cb in window._guided_preview_method_checkboxes.values())
    assert "Signal-Only F0" not in method_text
    assert "Decision-Support Audit" not in method_text
    assert "No Correction" not in method_text


def test_guided_signal_only_f0_panel_populates_from_loaded_completed_run(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)

    _load_preview_completed_run(window, run_dir, monkeypatch)

    assert "Diagnostic review is generated from the loaded completed run" in (
        window._guided_signal_f0_source_status_label.text()
    )
    assert str(run_dir) in window._guided_signal_f0_source_status_label.text()
    assert [window._guided_signal_f0_roi_combo.itemText(i) for i in range(window._guided_signal_f0_roi_combo.count())] == [
        "CH1",
        "CH2",
    ]
    assert [
        window._guided_signal_f0_chunk_combo.itemData(i)
        for i in range(window._guided_signal_f0_chunk_combo.count())
    ] == [0, 1]
    assert window._guided_signal_f0_chunk_combo.currentData() == 0
    assert window._guided_signal_f0_generate_btn.isEnabled() is True
    method_text = " ".join(cb.text() for cb in window._guided_preview_method_checkboxes.values())
    assert "Signal-Only F0" not in method_text


def test_guided_correction_preview_button_generates_backend_preview_read_only(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)

    window._guided_preview_generate_btn.click()

    assert "Preview comparison generated: success." in window._guided_preview_status_label.text()
    artifacts_text = window._guided_preview_artifacts_label.text()
    assert "Preview directory:" in artifacts_text
    assert "Summary:" in artifacts_text
    assert "Provenance:" in artifacts_text
    table = window._guided_preview_method_table
    assert table.rowCount() == 3
    table_text = " ".join(
        table.item(row, col).text()
        for row in range(table.rowCount())
        for col in range(table.columnCount())
        if table.item(row, col) is not None
    )
    assert "Robust Global Event-Reject Fit" in table_text
    assert "Adaptive Event-Gated Fit" in table_text
    assert "Global Linear Regression" in table_text
    assert "method_global_linear_regression_diagnostics.json" in table_text
    assert "method_global_linear_regression_trace.csv" in table_text
    assert "Signal-Only F0" not in table_text
    assert "Decision-Support Audit" not in table_text
    assert "No Correction" not in table_text
    assert "auto" not in table_text
    assert "needs_review" not in table_text
    assert "Errors/warnings: none reported" in window._guided_preview_messages_label.text()
    text = window._guided_preview_result_label.text()
    assert "Strategy recommendation: none" in text
    preview_dir = run_dir / "_guided_workflow" / "previews"
    assert preview_dir.exists()
    assert list(preview_dir.glob("*/preview_summary.json"))
    assert list(preview_dir.glob("*/preview_provenance.json"))
    assert not (preview_dir / "MANIFEST.json").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "applied_dff").exists()
    assert not (run_dir / "_analysis" / "phasic_out" / "features").exists()


def test_guided_signal_only_f0_button_generates_backend_diagnostic_read_only(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    before = {
        str(path.relative_to(phasic)): path.read_bytes()
        for path in sorted(phasic.rglob("*"))
        if path.is_file()
    }
    _load_preview_completed_run(window, run_dir, monkeypatch)

    window._guided_signal_f0_generate_btn.click()

    assert "Signal-Only F0 diagnostic review generated: success." in window._guided_signal_f0_status_label.text()
    artifacts_text = window._guided_signal_f0_artifacts_label.text()
    assert "Diagnostic directory:" in artifacts_text
    assert "Provenance JSON:" in artifacts_text
    assert "Summary JSON:" in artifacts_text
    assert "Chunk CSV:" in artifacts_text
    assert "Strategy recommendation: none; not selected." in artifacts_text
    table = window._guided_signal_f0_chunk_table
    assert table.rowCount() == 1
    table_text = " ".join(
        table.item(row, col).text()
        for row in range(table.rowCount())
        for col in range(table.columnCount())
        if table.item(row, col) is not None
    )
    assert "0" in table_text
    assert "success" in table_text
    assert "best" not in table_text.lower()
    diagnostic_dir = run_dir / "_guided_workflow" / "signal_only_f0_diagnostics"
    assert diagnostic_dir.exists()
    assert list(diagnostic_dir.glob("*/signal_only_f0_diagnostic_provenance.json"))
    assert list(diagnostic_dir.glob("*/signal_only_f0_diagnostic_summary.json"))
    assert list(diagnostic_dir.glob("*/signal_only_f0_diagnostic_chunks.csv"))
    assert not list(diagnostic_dir.glob("*.png"))
    after = {
        str(path.relative_to(phasic)): path.read_bytes()
        for path in sorted(phasic.rglob("*"))
        if path.is_file()
    }
    assert after == before
    assert not (phasic / "qc").exists()
    assert not (phasic / "features").exists()
    assert not (phasic / "applied_dff").exists()
    assert not (run_dir / "manifest.csv").exists()


def test_guided_correction_preview_does_not_auto_generate(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    calls = {"count": 0}

    def _fake_backend(*_args, **_kwargs):
        calls["count"] += 1
        return {
            "ok": True,
            "status": "success",
            "preview_output_dir": "preview",
            "preview_summary_path": "summary",
            "preview_provenance_path": "provenance",
            "method_statuses": {},
            "warnings": [],
            "errors": [],
        }

    monkeypatch.setattr(main_window_module, "run_guided_correction_preview_comparison", _fake_backend)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Diagnostics"))
    assert calls["count"] == 0
    _load_preview_completed_run(window, run_dir, monkeypatch)
    assert calls["count"] == 0
    window._guided_preview_roi_combo.setCurrentIndex(1)
    window._guided_preview_chunk_combo.setCurrentIndex(1)
    window._guided_preview_method_checkboxes["global_linear_regression"].setChecked(False)
    window._guided_correction_select_buttons["Adaptive Event-Gated Fit"].click()
    assert calls["count"] == 0

    window._guided_preview_generate_btn.click()
    assert calls["count"] == 1


def test_guided_signal_only_f0_diagnostic_is_explicit_button_only(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    calls = {"count": 0, "kwargs": None, "args": None}

    def _fake_backend(*args, **kwargs):
        calls["count"] += 1
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {
            "ok": True,
            "status": "success",
            "diagnostic_id": "signal_only_f0_test",
            "output_dir": "diagnostic_dir",
            "provenance_path": "provenance.json",
            "summary_path": "summary.json",
            "chunk_csv_path": "chunks.csv",
            "trace_csv_paths": [],
            "warnings": [],
            "errors": [],
            "chunk_statuses": {"1": {"status": "success", "error": ""}},
        }

    monkeypatch.setattr(main_window_module, "run_signal_only_f0_diagnostic_review", _fake_backend)
    window._guided_workflow_stepper.setCurrentRow(list(GUIDED_WORKFLOW_STEPS).index("Diagnostics"))
    assert calls["count"] == 0
    _load_preview_completed_run(window, run_dir, monkeypatch)
    assert calls["count"] == 0
    window._guided_signal_f0_roi_combo.setCurrentIndex(1)
    window._guided_signal_f0_chunk_combo.setCurrentIndex(1)
    assert calls["count"] == 0
    window._guided_correction_select_buttons["Adaptive Event-Gated Fit"].click()
    assert calls["count"] == 0
    window._guided_preview_generate_btn.click()
    assert calls["count"] == 0

    window._guided_signal_f0_generate_btn.click()

    assert calls["count"] == 1
    assert calls["args"] == (str(run_dir),)
    assert calls["kwargs"]["roi"] == "CH2"
    assert calls["kwargs"]["chunk_ids"] == [1]
    assert calls["kwargs"]["allow_existing"] is False
    assert "output_dir" not in calls["kwargs"]
    assert "diagnostic_id" not in calls["kwargs"]


def test_guided_correction_preview_result_marks_stale_on_selection_change(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)

    window._guided_preview_generate_btn.click()
    assert "Preview comparison generated: success." in window._guided_preview_status_label.text()

    window._guided_preview_chunk_combo.setCurrentIndex(1)

    assert "Displayed preview is stale because the preview selection changed" in window._guided_preview_status_label.text()


def test_guided_signal_only_f0_result_displays_partial_failed_and_does_not_select_strategy(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    before_intent = window._guided_correction_intent

    def _fake_backend(*_args, **_kwargs):
        return {
            "ok": False,
            "status": "partial",
            "diagnostic_id": "signal_only_f0_test",
            "output_dir": "diagnostic_dir",
            "provenance_path": "provenance.json",
            "summary_path": "summary.json",
            "chunk_csv_path": "",
            "trace_csv_paths": [],
            "warnings": ["caution"],
            "errors": ["chunk 1: failed"],
            "chunk_statuses": {
                "0": {"status": "success", "error": ""},
                "1": {"status": "failed", "error": "failed"},
            },
        }

    monkeypatch.setattr(main_window_module, "run_signal_only_f0_diagnostic_review", _fake_backend)

    window._guided_signal_f0_generate_btn.click()

    assert "partial" in window._guided_signal_f0_status_label.text()
    assert "chunk 1: failed" in window._guided_signal_f0_messages_label.text()
    table_text = " ".join(
        window._guided_signal_f0_chunk_table.item(row, col).text()
        for row in range(window._guided_signal_f0_chunk_table.rowCount())
        for col in range(window._guided_signal_f0_chunk_table.columnCount())
        if window._guided_signal_f0_chunk_table.item(row, col) is not None
    )
    assert "failed" in table_text
    assert window._guided_correction_intent == before_intent

    def _fake_failed(*_args, **_kwargs):
        return {
            "ok": False,
            "status": "failed",
            "diagnostic_id": "signal_only_f0_test",
            "output_dir": "",
            "provenance_path": "",
            "summary_path": "",
            "chunk_csv_path": "",
            "trace_csv_paths": [],
            "warnings": [],
            "errors": ["source failed"],
            "chunk_statuses": {},
        }

    monkeypatch.setattr(main_window_module, "run_signal_only_f0_diagnostic_review", _fake_failed)
    window._guided_signal_f0_generate_btn.click()
    assert "failed" in window._guided_signal_f0_status_label.text().lower()
    assert "source failed" in window._guided_signal_f0_messages_label.text()
    assert window._guided_correction_intent == before_intent


def test_guided_signal_only_f0_result_marks_stale_on_selection_change(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_signal_f0_generate_btn.click()
    assert "Signal-Only F0 diagnostic review generated: success." in window._guided_signal_f0_status_label.text()

    window._guided_signal_f0_chunk_combo.setCurrentIndex(1)

    assert "Displayed Signal-Only F0 diagnostic review is stale because the selection changed" in (
        window._guided_signal_f0_status_label.text()
    )


def test_guided_correction_preview_refresh_preserves_non_default_selection_with_result(window, tmp_path, monkeypatch):
    run_dir = _make_preview_completed_run(tmp_path)
    _load_preview_completed_run(window, run_dir, monkeypatch)
    window._guided_preview_roi_combo.setCurrentIndex(window._guided_preview_roi_combo.findData("CH2"))
    window._guided_preview_chunk_combo.setCurrentIndex(window._guided_preview_chunk_combo.findData(1))
    window._guided_preview_has_result = True
    window._guided_preview_result_stale = False
    window._guided_preview_status_label.setText("Preview comparison generated: success.")
    window._guided_preview_result_label.setText("Preview status: success")

    window._refresh_guided_diagnostics_panel()

    assert window._guided_preview_roi_combo.currentData() == "CH2"
    assert window._guided_preview_chunk_combo.currentData() == 1
    assert "Preview comparison generated: success." in window._guided_preview_status_label.text()
    assert window._guided_preview_result_stale is False


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
    assert "Generate preview comparison" in button_texts
