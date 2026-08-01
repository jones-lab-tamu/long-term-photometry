"""What the Run page and the Start screen say about state the scientist owns.

Every assertion reads the visible widget after driving a real control, so a
label that is set but never shown, or a stepper item that is navigated to but
never enabled, cannot make a test pass.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, Qt, QTimer
from PySide6.QtWidgets import QApplication, QLabel

from gui.main_window import (
    GUIDED_LOAD_COMPLETED_RUN_HINT,
    GUIDED_RUN_COMPLETED_MESSAGE,
    GUIDED_SETUP_CHECK_PASSED_FOR_COMPLETED_RUN,
    MainWindow,
)


pytestmark = pytest.mark.usefixtures("no_real_modals")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _spin(times: int = 150) -> None:
    for _ in range(times):
        QCoreApplication.processEvents()


def _pump(predicate, *, limit: int = 60000) -> bool:
    for _ in range(limit):
        if predicate():
            return True
        QCoreApplication.processEvents()
    return predicate()


def _write_csv(path: Path, *, rows: int = 12000, fs_hz: float = 20.0) -> None:
    t = np.arange(rows, dtype=float) / fs_hz
    rng = np.random.default_rng(19)
    pd.DataFrame(
        {
            "ElapsedSeconds": t,
            "R1_Signal": 1.7 + 0.05 * np.sin(t / 7.0) + rng.normal(0, 0.004, rows),
            "R1_Reference": 1.0 + 0.02 * np.sin(t / 7.0) + rng.normal(0, 0.002, rows),
        }
    ).to_csv(path, index=False)


@pytest.fixture(scope="module")
def continuous_csv_folder(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("lifecycle_continuous")
    _write_csv(folder / "continuous_recording.csv", rows=24000)
    return folder


# --------------------------------------------------------------------------
# A. First launch must not invent a completed run
# --------------------------------------------------------------------------


def test_fresh_launch_shows_no_completed_run(window):
    assert str(window._current_run_dir or "") == ""

    assert window._guided_start_open_status_label.text() == "Completed run: none"
    assert window._guided_start_open_status_label.toolTip() == ""

    status = window._guided_start_status_label.text()
    assert "Completed run loaded" not in status
    assert "No completed run loaded." in status


def test_fresh_launch_never_shows_the_working_directory_as_a_run(window):
    cwd = os.getcwd()
    for text in (
        window._guided_start_open_status_label.text(),
        window._guided_start_status_label.text(),
        window._guided_mode_banner_label.text(),
    ):
        assert cwd not in text


def test_fresh_launch_leaves_results_unloaded(window):
    assert window._report_viewer.has_loaded_results() is False
    assert window._guided_report_viewer.has_loaded_results() is False


def test_a_real_completed_run_still_displays(window, tmp_path):
    """The guard must not suppress a genuinely retained completed run."""
    run_dir = tmp_path / "continuous_run"
    run_dir.mkdir()
    window._current_run_dir = str(run_dir)
    window._guided_continuous_rwd_completed_run_dir = str(run_dir)
    window._refresh_guided_start_panel()
    _spin(40)

    assert "none" not in window._guided_start_open_status_label.text()
    assert str(run_dir) in window._guided_start_open_status_label.toolTip()
    assert "Completed run loaded:" in window._guided_start_status_label.text()


def test_unset_run_paths_do_not_compare_equal(window):
    """Two unset paths must not both normalize to the working directory."""
    assert window._guided_real_run_path("") == ""
    assert window._guided_real_run_path(None) == ""
    assert window._guided_real_run_path("   ") == ""
    assert window._guided_real_run_path(os.getcwd()) == os.path.realpath(os.getcwd())


# --------------------------------------------------------------------------
# B. The setup check must be visibly in progress
# --------------------------------------------------------------------------


def test_setup_check_paints_its_in_progress_state_before_validating(
    window, monkeypatch
):
    """The existing synchronous check must not freeze on the previous text.

    A queued zero-timer stands in for the pending repaint: it can only have
    fired if the handler processed pending GUI events before entering
    validation.
    """
    observed = {}

    def fake_workflow(context):
        observed["status"] = window._guided_backend_validation_status_label.text()
        observed["details"] = window._guided_backend_validation_details_label.text()
        observed["check_enabled"] = window._guided_backend_validate_btn.isEnabled()
        observed["run_enabled"] = window._guided_run_btn.isEnabled()
        observed["pending_events_processed"] = observed.get("timer_fired", False)
        observed["calls"] = observed.get("calls", 0) + 1
        raise RuntimeError("stop after observing the in-progress state")

    monkeypatch.setattr(
        window, "_run_guided_backend_validation_workflow", fake_workflow
    )
    monkeypatch.setattr(
        window, "_capture_guided_backend_validation_context", lambda: None
    )

    QTimer.singleShot(0, lambda: observed.__setitem__("timer_fired", True))
    window._guided_backend_validate_btn.click()

    assert observed["calls"] == 1
    assert observed["pending_events_processed"] is True
    assert observed["status"] == (
        "Checking your Guided setup… No run is being started."
    )
    assert "take a moment" in observed["details"]
    assert observed["check_enabled"] is False
    assert observed["run_enabled"] is False


def test_setup_check_restores_state_after_a_failed_validation(window, monkeypatch):
    def failing_workflow(context):
        raise RuntimeError("simulated validation failure")

    monkeypatch.setattr(
        window, "_run_guided_backend_validation_workflow", failing_workflow
    )
    monkeypatch.setattr(
        window, "_capture_guided_backend_validation_context", lambda: None
    )
    window._guided_backend_validate_btn.click()
    _spin(60)

    assert window._guided_backend_validation_active is False
    assert window._guided_backend_validate_btn.isEnabled() is True
    assert window._guided_run_btn.isEnabled() is False
    status = window._guided_backend_validation_status_label.text()
    assert "Checking your Guided setup" not in status


# --------------------------------------------------------------------------
# C/D/E. Continuous completion state
# --------------------------------------------------------------------------


def _complete_continuous_run(window, folder, output):
    """Drive the ordinary Guided buttons to a finished continuous run."""
    window._guided_start_setup_btn.click()
    _spin(60)
    fmt = window._guided_format_combo
    fmt.setCurrentIndex(fmt.findData("custom_tabular"))
    structure = window._guided_acquisition_mode_combo
    structure.setCurrentIndex(structure.findData("continuous"))
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(output))
    _spin(150)

    combo = window._guided_csv_time_column_combo
    combo.setCurrentIndex(combo.findData("ElapsedSeconds"))
    row = window._guided_csv_mapping_rows[0]
    row["name"].setText("R1")
    row["signal"].setCurrentIndex(row["signal"].findData("R1_Signal"))
    row["reference"].setCurrentIndex(row["reference"].findData("R1_Reference"))
    _spin(60)

    window._on_guided_discover_rois()
    _pump(lambda: not window._guided_roi_discovery_running)
    _pump(lambda: window._guided_roi_discovery_thread is None)
    _spin(100)
    for index in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(index).setCheckState(Qt.Checked)
    _spin(60)

    window._on_guided_continue_to_recording_structure()
    _spin(150)
    window._guided_fixed_daily_anchor_clock_edit.setText("07:00")
    window._guided_recording_start_clock_edit.setText("12:00:00")
    _spin(120)
    window._on_guided_continue_to_correction_approach()
    _spin(200)
    _pump(
        lambda: getattr(window, "_guided_continuous_rwd_check_thread", None) is None,
        limit=400000,
    )
    _spin(200)

    for checkbox in window._guided_preview_method_checkboxes.values():
        checkbox.setChecked(True)
    _spin(30)
    window._on_generate_guided_correction_preview()
    _pump(lambda: not getattr(window, "_guided_correction_preview_running", False))
    _pump(
        lambda: getattr(window, "_guided_correction_preview_thread", None) is None,
        limit=400000,
    )
    _spin(120)
    for confirm_row in dict(window._guided_local_preview_confirmation_rows).values():
        strategy = confirm_row["strategy_combo"]
        for index in range(strategy.count()):
            strategy.setCurrentIndex(index)
            if strategy.currentData():
                break
        _spin(20)
        confirm_row["action_button"].click()
        _spin(60)

    window._on_guided_continue_to_feature_detection()
    _spin(200)
    window._on_guided_continue_to_review_plan()
    _spin(300)
    confirm_btn = window._guided_review_dataset_contract_action_btn
    if not confirm_btn.isHidden():
        confirm_btn.click()
        _pump(lambda: not window._guided_dataset_contract_confirmation_active)
        _spin(200)
    window._guided_review_go_to_run_btn.click()
    _spin(200)
    window._guided_backend_validate_btn.click()
    _spin(100)
    _pump(
        lambda: not window._guided_continuous_rwd_preparation_active(), limit=400000
    )
    _spin(300)
    assert window._guided_run_btn.isEnabled() is True, (
        window._guided_continuous_rwd_status_message
    )
    window._guided_run_btn.click()
    _pump(
        lambda: not getattr(window, "_guided_continuous_rwd_execution_active", False),
        limit=600000,
    )
    _spin(600)


@pytest.fixture(scope="module")
def completed_continuous_window(qapp, continuous_csv_folder, tmp_path_factory):
    """One finished continuous run, shared by the completion-state tests."""
    output = tmp_path_factory.mktemp("lifecycle_output")
    instance = MainWindow()
    _complete_continuous_run(instance, continuous_csv_folder, output)
    yield instance
    instance.close()
    instance.deleteLater()


def _shown_on_page(widget, page) -> bool:
    """True only if no ancestor up to the page hides this widget.

    ``isHidden()`` reports a widget's own flag, so a label inside a hidden
    group still answers False; ``isVisible()`` needs a shown top-level window.
    """
    node = widget
    while node is not None:
        if node.isHidden():
            return False
        if node is page:
            return True
        node = node.parentWidget()
    return True


def _run_page_texts(window):
    page = window._guided_workflow_stack.currentWidget()
    return [
        label.text()
        for label in page.findChildren(QLabel)
        if _shown_on_page(label, page) and label.text().strip()
    ]


def test_continuous_completion_states_success(completed_continuous_window):
    window = completed_continuous_window
    assert window._guided_run_readiness_label.text() == GUIDED_RUN_COMPLETED_MESSAGE


def test_continuous_completion_does_not_claim_the_run_is_still_going(
    completed_continuous_window,
):
    window = completed_continuous_window
    # The live "Analysis progress" panel is the one that says the run is
    # still going; it must be put away, not merely left behind a stale text.
    assert window._guided_run_live_status_group.isHidden() is True
    for text in _run_page_texts(window):
        assert "Do not close this window" not in text
        assert "Running continuous analysis" not in text


def test_continuous_completion_does_not_claim_the_setup_was_never_checked(
    completed_continuous_window,
):
    window = completed_continuous_window
    assert window._guided_backend_validation_status_label.text() == (
        GUIDED_SETUP_CHECK_PASSED_FOR_COMPLETED_RUN
    )
    for text in _run_page_texts(window):
        assert "has not been checked yet" not in text


def test_continuous_completion_disables_run_with_a_truthful_reason(
    completed_continuous_window,
):
    window = completed_continuous_window
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_run_btn.toolTip() == GUIDED_RUN_COMPLETED_MESSAGE


def test_continuous_completion_retains_the_completed_run_directory(
    completed_continuous_window,
):
    window = completed_continuous_window
    run_dir = str(window._guided_continuous_rwd_completed_run_dir or "")
    assert run_dir
    assert os.path.isdir(run_dir)
    assert os.path.realpath(window._current_run_dir) == os.path.realpath(run_dir)


def test_continuous_completion_shows_the_results_folder(
    completed_continuous_window,
):
    window = completed_continuous_window
    run_dir = str(window._guided_continuous_rwd_completed_run_dir)
    details = window._guided_run_execution_details_label.text()

    assert f"Results folder: {run_dir}" in details
    assert window._guided_npm_open_output_btn.isHidden() is False
    assert window._guided_npm_open_output_btn.isEnabled() is True


def test_open_results_folder_targets_the_completed_run(
    completed_continuous_window, monkeypatch
):
    """The action must open the completed run, not the output base."""
    import gui.main_window as main_window_module

    opened = []
    monkeypatch.setattr(main_window_module, "_open_folder", opened.append)
    window = completed_continuous_window
    window._guided_npm_open_output_btn.click()
    _spin(40)

    assert opened == [str(window._guided_continuous_rwd_completed_run_dir)]


def test_continuous_completion_says_how_to_open_review(
    completed_continuous_window,
):
    details = completed_continuous_window._guided_run_execution_details_label.text()
    assert GUIDED_LOAD_COMPLETED_RUN_HINT in details
    assert "Review" in GUIDED_LOAD_COMPLETED_RUN_HINT


def _review_step_enabled(window) -> bool:
    stepper = window._guided_workflow_stepper
    index = window._guided_step_index("Review")
    return bool(stepper.item(index).flags() & Qt.ItemIsEnabled)


def test_review_step_unlocks_only_after_loading_the_completed_run(
    completed_continuous_window,
):
    window = completed_continuous_window
    assert _review_step_enabled(window) is False

    button = window._guided_load_completed_run_for_review_btn
    assert button.isHidden() is False
    button.click()
    _pump(
        lambda: getattr(window, "_guided_completed_review_load_thread", None) is None,
        limit=600000,
    )
    _spin(600)

    assert _review_step_enabled(window) is True
    assert window._guided_report_viewer.has_loaded_results() is True
    assert window._guided_workflow_stepper.currentRow() == (
        window._guided_step_index("Review")
    )
    assert window._guided_run_readiness_label.text() == (
        "Completed run loaded for review."
    )
