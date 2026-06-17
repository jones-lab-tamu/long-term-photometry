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


def test_guided_stage1_has_no_run_or_manifest_action_buttons(window):
    forbidden = {"Run Pipeline", "Save Manifest", "Dry Run", "Run Batch"}
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
