import pytest
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
    assert [stepper.item(i).text() for i in range(stepper.count())] == list(GUIDED_WORKFLOW_STEPS)


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
    assert not cards["Decision-Support Audit"].isEnabled()
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

