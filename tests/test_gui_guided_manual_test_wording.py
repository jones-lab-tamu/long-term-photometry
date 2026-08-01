"""Wording and accuracy defects found during the manual Guided test.

Each assertion reads a visible widget after driving real controls, so a string
that is defined but never shown cannot make a test pass.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QLabel

import gui.main_window as main_window_module
from gui.main_window import MainWindow


pytestmark = pytest.mark.usefixtures("no_real_modals")

EM_DASH = "—"


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
    rng = np.random.default_rng(23)
    pd.DataFrame(
        {
            "ElapsedSeconds": t,
            "R1_Signal": 1.7 + 0.05 * np.sin(t / 7.0) + rng.normal(0, 0.004, rows),
            "R1_Reference": 1.0 + 0.02 * np.sin(t / 7.0) + rng.normal(0, 0.002, rows),
            "R2_Signal": 1.8 + 0.05 * np.cos(t / 9.0) + rng.normal(0, 0.004, rows),
            "R2_Reference": 1.0 + 0.02 * np.cos(t / 9.0) + rng.normal(0, 0.002, rows),
        }
    ).to_csv(path, index=False)


@pytest.fixture(scope="module")
def intermittent_csv_folder(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("manual_sessions")
    for index in range(3):
        _write_csv(folder / f"session_{index + 1:04d}.csv")
    return folder


@pytest.fixture(scope="module")
def continuous_csv_folder(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("manual_continuous")
    _write_csv(folder / "continuous_recording.csv", rows=24000)
    return folder


@pytest.fixture(scope="module")
def rwd_folder(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("manual_rwd")
    fs_hz, rows = 5.0, int(600.0 * 5.0)
    rng = np.random.default_rng(4)
    for hour in range(4):
        session = folder / f"2025_01_01-{hour:02d}_00_00"
        session.mkdir(parents=True, exist_ok=True)
        t = np.arange(rows, dtype=float) / fs_hz
        pd.DataFrame(
            {
                "TimeStamp": t,
                "Region0-410": 1.0 + 0.02 * np.sin(t / 40) + rng.normal(0, 0.003, rows),
                "Region0-470": 1.7 + 0.05 * np.sin(t / 40) + rng.normal(0, 0.006, rows),
            }
        ).to_csv(session / "Fluorescence.csv", index=False)
    return folder


def _shown(widget, page) -> bool:
    node = widget
    while node is not None:
        if node.isHidden():
            return False
        if node is page:
            return True
        node = node.parentWidget()
    return True


def _page_texts(window):
    page = window._guided_workflow_stack.currentWidget()
    return [
        label.text()
        for label in page.findChildren(QLabel)
        if _shown(label, page) and label.text().strip()
    ]


def _select_data(window, folder, output, *, fmt="custom_tabular", structure=None):
    window._guided_start_setup_btn.click()
    _spin(60)
    combo = window._guided_format_combo
    combo.setCurrentIndex(combo.findData(fmt))
    if structure is not None:
        structure_combo = window._guided_acquisition_mode_combo
        structure_combo.setCurrentIndex(structure_combo.findData(structure))
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(output))
    _spin(150)


def _map_csv(window, rois=("R1", "R2")):
    combo = window._guided_csv_time_column_combo
    combo.setCurrentIndex(combo.findData("ElapsedSeconds"))
    window._guided_csv_order_confirm_cb.setChecked(True)
    while len(window._guided_csv_mapping_rows) < len(rois):
        window._add_guided_csv_mapping_row()
    for row, roi in zip(window._guided_csv_mapping_rows, rois):
        row["name"].setText(roi)
        row["signal"].setCurrentIndex(row["signal"].findData(f"{roi}_Signal"))
        row["reference"].setCurrentIndex(row["reference"].findData(f"{roi}_Reference"))
    _spin(60)


def _discover(window):
    window._on_guided_discover_rois()
    _pump(lambda: not window._guided_roi_discovery_running)
    _pump(lambda: window._guided_roi_discovery_thread is None)
    _spin(120)
    for index in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(index).setCheckState(Qt.Checked)
    _spin(80)


def _reach_recording_structure(window, folder, output, *, csv=True, **kwargs):
    _select_data(window, folder, output, **kwargs)
    if csv:
        _map_csv(window)
    _discover(window)
    window._on_guided_continue_to_recording_structure()
    _spin(200)


# --------------------------------------------------------------------------
# 3. Stepper capitalization
# --------------------------------------------------------------------------


def test_stepper_uses_sentence_case(window):
    labels = [
        window._guided_workflow_stepper.item(i).text()
        for i in range(window._guided_workflow_stepper.count())
    ]

    assert "5. Feature detection" in labels
    assert "6. Review plan" in labels
    assert "5. Feature Detection" not in labels
    assert "6. Review Plan" not in labels


# --------------------------------------------------------------------------
# 4. Duplicated recording-start instruction
# --------------------------------------------------------------------------


def test_recording_start_instruction_appears_once_when_blank(
    window, intermittent_csv_folder, tmp_path
):
    """Two labels carried the identical sentence; only the field help shows."""
    _reach_recording_structure(window, intermittent_csv_folder, tmp_path)
    message = "Enter the clock time when this recording began."
    help_label = window._guided_recording_start_clock_help_label
    validation_label = window._guided_timeline_validation_label

    assert help_label.text() == message
    assert help_label.isHidden() is False
    # Same sentence, suppressed rather than repeated a few rows below.
    assert validation_label.text() == message
    assert validation_label.isHidden() is True

    shown = [
        label
        for label in window._guided_timeline_group.findChildren(QLabel)
        if not label.isHidden() and label.text() == message
    ]
    assert len(shown) == 1


def test_recording_start_instruction_is_replaced_once_entered(
    window, intermittent_csv_folder, tmp_path
):
    _reach_recording_structure(window, intermittent_csv_folder, tmp_path)
    window._guided_recording_start_clock_edit.setFocus()
    QTest.keyClicks(window._guided_recording_start_clock_edit, "12:00:00")
    _spin(200)

    texts = _page_texts(window)
    assert "Enter the clock time when this recording began." not in texts
    assert "Using the recording-start time entered here." in texts


# --------------------------------------------------------------------------
# 5. Stale session-timing instruction
# --------------------------------------------------------------------------


def test_populated_session_timing_stops_asking_for_it(
    window, intermittent_csv_folder, tmp_path
):
    _reach_recording_structure(window, intermittent_csv_folder, tmp_path)
    assert any(
        t.startswith("Enter the sessions per hour") for t in _page_texts(window)
    )

    window._guided_sessions_per_hour_edit.setFocus()
    QTest.keyClicks(window._guided_sessions_per_hour_edit, "2")
    window._guided_session_duration_edit.setFocus()
    QTest.keyClicks(window._guided_session_duration_edit, "600")
    _spin(200)

    assert window._guided_recording_structure_help_label.text() == ""
    assert not any(
        t.startswith("Enter the sessions per hour") for t in _page_texts(window)
    )


# --------------------------------------------------------------------------
# 6/7. Continuous guidance and CSV mapping language
# --------------------------------------------------------------------------


def test_continuous_guidance_names_no_vendor(window, continuous_csv_folder, tmp_path):
    _select_data(window, continuous_csv_folder, tmp_path, structure="continuous")
    text = window._guided_continuous_explanation_label.text()

    assert window._guided_continuous_explanation_label.isHidden() is False
    assert "RWD" not in text
    assert "CSV" not in text
    assert "one continuous recording rather than as repeated sessions" in text


def test_one_continuous_csv_gets_no_session_order_language(
    window, continuous_csv_folder, tmp_path
):
    _select_data(window, continuous_csv_folder, tmp_path, structure="continuous")

    status = window._guided_csv_status_label.text()
    count = window._guided_csv_session_count_label.text()

    assert status == (
        "Choose the elapsed-time column and map the signal and reference "
        "columns for each ROI."
    )
    assert "sessions" not in status
    assert "rename the files" not in status
    assert count == "Selected recording file"
    assert "session order" not in count.lower()


def test_multiple_csv_files_keep_session_order_language(
    window, intermittent_csv_folder, tmp_path
):
    _select_data(window, intermittent_csv_folder, tmp_path)

    assert "for all sessions" in window._guided_csv_status_label.text()
    assert "rename the files" in window._guided_csv_status_label.text()
    assert window._guided_csv_session_count_label.text() == (
        "CSV session order: 3 sessions"
    )
    assert window._guided_csv_order_confirm_cb.isHidden() is False


ORDER_CONFIRM_TEXT = "I confirm this is the intended recording order."


@pytest.mark.parametrize("structure", [None, "continuous"], ids=["auto", "explicit"])
def test_one_continuous_csv_hides_the_order_confirmation(
    window, continuous_csv_folder, tmp_path, structure
):
    """One recording has no order to confirm, so the control is not offered."""
    _select_data(window, continuous_csv_folder, tmp_path, structure=structure)
    checkbox = window._guided_csv_order_confirm_cb

    assert checkbox.text() == ORDER_CONFIRM_TEXT
    assert checkbox.isHidden() is True


@pytest.mark.parametrize(
    "structure,folder_name",
    [(None, "intermittent_csv_folder"), ("intermittent", "continuous_csv_folder")],
    ids=["auto_many_files", "explicit_intermittent_one_file"],
)
def test_repeated_sessions_still_offer_the_order_confirmation(
    window, tmp_path, request, structure, folder_name
):
    folder = request.getfixturevalue(folder_name)
    _select_data(window, folder, tmp_path, structure=structure)
    checkbox = window._guided_csv_order_confirm_cb

    assert checkbox.isHidden() is False
    assert checkbox.text() == ORDER_CONFIRM_TEXT


def test_intermittent_order_confirmation_still_gates_discovery(
    window, intermittent_csv_folder, tmp_path, monkeypatch
):
    """The confirmation requirement itself is unchanged for repeated sessions."""
    seen = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "critical",
        staticmethod(lambda *args, **_kw: seen.append(str(args[2]))),
    )
    _select_data(window, intermittent_csv_folder, tmp_path)
    combo = window._guided_csv_time_column_combo
    combo.setCurrentIndex(combo.findData("ElapsedSeconds"))
    row = window._guided_csv_mapping_rows[0]
    row["name"].setText("R1")
    row["signal"].setCurrentIndex(row["signal"].findData("R1_Signal"))
    row["reference"].setCurrentIndex(row["reference"].findData("R1_Reference"))
    _spin(60)
    assert window._guided_csv_order_confirm_cb.isChecked() is False

    window._on_guided_discover_rois()
    _pump(lambda: not window._guided_roi_discovery_running)
    _pump(lambda: window._guided_roi_discovery_thread is None)
    _spin(120)

    assert seen and "session-file order" in seen[-1]


def test_order_confirmation_visibility_follows_a_structure_change(
    window, continuous_csv_folder, tmp_path
):
    """It must not stay hidden after the scientist states repeated sessions."""
    _select_data(window, continuous_csv_folder, tmp_path)
    checkbox = window._guided_csv_order_confirm_cb
    assert checkbox.isHidden() is True

    combo = window._guided_acquisition_mode_combo
    combo.setCurrentIndex(combo.findData("intermittent"))
    _spin(150)
    assert checkbox.isHidden() is False
    assert window._guided_csv_session_count_label.text() == (
        "CSV session order: 1 sessions"
    )

    combo.setCurrentIndex(combo.findData("continuous"))
    _spin(150)
    assert checkbox.isHidden() is True
    assert window._guided_csv_session_count_label.text() == "Selected recording file"


def test_continuous_csv_mapping_area_has_no_order_or_session_language(
    window, continuous_csv_folder, tmp_path
):
    """Nothing visible inside the CSV mapping group asks about order/sessions."""
    _select_data(window, continuous_csv_folder, tmp_path, structure="continuous")
    group = window._guided_csv_interpretation_group

    visible = [
        widget.text()
        for widget in (*group.findChildren(QLabel), *group.findChildren(QCheckBox))
        if not widget.isHidden() and widget.text().strip()
    ]
    assert visible
    for text in visible:
        lowered = text.lower()
        assert "order" not in lowered
        assert "session" not in lowered


# --------------------------------------------------------------------------
# 8/9. Feature Detection guidance and punctuation
# --------------------------------------------------------------------------


def _reach_feature_detection(window, folder, output):
    _reach_recording_structure(window, folder, output)
    window._guided_sessions_per_hour_edit.setText("2")
    window._guided_session_duration_edit.setText("600")
    window._guided_fixed_daily_anchor_clock_edit.setText("07:00")
    window._guided_recording_start_clock_edit.setText("12:00:00")
    _spin(150)
    window._on_guided_continue_to_correction_approach()
    _spin(250)
    for checkbox in window._guided_preview_method_checkboxes.values():
        checkbox.setChecked(True)
    _spin(30)
    combo = window._guided_preview_roi_combo
    for index in range(combo.count()):
        combo.setCurrentIndex(index)
        _spin(30)
        window._on_generate_guided_correction_preview()
        _pump(lambda: not getattr(window, "_guided_correction_preview_running", False))
        _pump(
            lambda: getattr(window, "_guided_correction_preview_thread", None) is None,
            limit=400000,
        )
        _spin(120)
    for row in dict(window._guided_local_preview_confirmation_rows).values():
        strategy = row["strategy_combo"]
        for index in range(strategy.count()):
            strategy.setCurrentIndex(index)
            if strategy.currentData():
                break
        _spin(20)
        row["action_button"].click()
        _spin(60)
    window._on_guided_continue_to_feature_detection()
    _spin(250)


def test_feature_detection_guidance_makes_no_sufficiency_claim(
    window, intermittent_csv_folder, tmp_path
):
    _reach_feature_detection(window, intermittent_csv_folder, tmp_path)
    joined = "\n".join(_page_texts(window))

    assert "Most ROIs can use" not in joined
    assert "Most users can leave" not in joined
    # Nor the opposite claim that every ROI needs customizing.
    assert "every ROI needs" not in joined
    assert "Review the preview for each ROI" in joined


def test_feature_detection_notes_use_no_em_dash(
    window, intermittent_csv_folder, tmp_path
):
    _reach_feature_detection(window, intermittent_csv_folder, tmp_path)

    for text in _page_texts(window):
        assert EM_DASH not in text


# --------------------------------------------------------------------------
# 10/11. Review Plan per-ROI feature settings
# --------------------------------------------------------------------------


class _StubRoiDialog:
    """Stands in for the per-ROI dialog, returning one changed numeric field."""

    def __init__(self, roi_id, seed_values, parent=None):
        self._seed = dict(seed_values)

    def exec(self):
        return QDialog.Accepted

    def result_values(self):
        values = dict(self._seed)
        values["peak_threshold_k"] = 3.5
        return values


def test_review_plan_distinguishes_default_and_custom_rois(
    window, intermittent_csv_folder, tmp_path, monkeypatch
):
    _reach_feature_detection(window, intermittent_csv_folder, tmp_path)
    monkeypatch.setattr(
        main_window_module, "_GuidedRoiFeatureEventDialog", _StubRoiDialog
    )
    window._on_guided_customize_roi_feature_event("R2")
    _spin(200)
    assert window._guided_feature_detection_readiness()[0] is True

    window._on_guided_continue_to_review_plan()
    _spin(300)
    summary = window._guided_review_feature_detection_summary_label.text()

    assert "Settings used by each ROI:" in summary
    assert "R1: Default -" in summary
    assert "R2: Custom -" in summary


def test_review_plan_shows_the_actual_custom_settings(
    window, intermittent_csv_folder, tmp_path, monkeypatch
):
    _reach_feature_detection(window, intermittent_csv_folder, tmp_path)
    monkeypatch.setattr(
        main_window_module, "_GuidedRoiFeatureEventDialog", _StubRoiDialog
    )
    window._on_guided_customize_roi_feature_event("R2")
    _spin(200)
    window._on_guided_continue_to_review_plan()
    _spin(300)
    summary = window._guided_review_feature_detection_summary_label.text()

    # The customized threshold itself, not merely the word "Custom".
    assert "threshold (3.5)" in summary
    assert "threshold (2.5)" in summary  # the untouched Default ROI
    assert "mean_std" not in summary
    assert "dff signal" not in summary


# --------------------------------------------------------------------------
# 11. Completed-run Review language
# --------------------------------------------------------------------------


def test_review_step_has_no_future_stages_sentence(window):
    window._guided_workflow_stepper.setCurrentRow(
        window._guided_step_index("Review")
    )
    _spin(80)

    for text in _page_texts(window):
        assert "future guided stages" not in text
        assert "applied-dF/F routing" not in text


def test_completed_review_titles_use_the_csv_display_label():
    """The workspace title comes from the shared format display label."""
    from photometry_pipeline.guided_display_labels import format_display_label

    assert format_display_label("custom_tabular") == "CSV files"
    assert format_display_label("rwd") == "RWD"
    assert "CUSTOM_TABULAR" != format_display_label("custom_tabular").upper()


def test_completed_review_feature_summary_uses_display_names():
    """The completed-run per-ROI formatter shares the Step 5 display names."""
    from photometry_pipeline.guided_completed_feature_event_reload import (
        _effective_settings_summary_text,
    )

    summary = _effective_settings_summary_text(
        {
            "peak_threshold_method": "mean_std",
            "peak_threshold_k": 2.5,
            "event_signal": "dff",
        }
    )

    assert "mean + standard-deviation threshold (2.5)" in summary
    assert "dF/F signal" in summary
    assert "mean_std" not in summary


# --------------------------------------------------------------------------
# 12. The loaded-run banner
# --------------------------------------------------------------------------


def test_loaded_completed_run_is_named_in_the_start_banner(
    window, continuous_csv_folder, tmp_path
):
    """A normally loaded completed run must not still read "none"."""
    _select_data(window, continuous_csv_folder, tmp_path, structure="continuous")
    combo = window._guided_csv_time_column_combo
    combo.setCurrentIndex(combo.findData("ElapsedSeconds"))
    row = window._guided_csv_mapping_rows[0]
    row["name"].setText("R1")
    row["signal"].setCurrentIndex(row["signal"].findData("R1_Signal"))
    row["reference"].setCurrentIndex(row["reference"].findData("R1_Reference"))
    _spin(60)
    _discover(window)
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
    window._guided_run_btn.click()
    _pump(
        lambda: not getattr(window, "_guided_continuous_rwd_execution_active", False),
        limit=600000,
    )
    _spin(600)

    run_dir = str(window._guided_continuous_rwd_completed_run_dir or "")
    assert run_dir, window._guided_continuous_rwd_status_message

    window._guided_load_completed_run_for_review_btn.click()
    _pump(
        lambda: getattr(window, "_guided_completed_review_load_thread", None) is None,
        limit=600000,
    )
    _spin(600)

    assert window._guided_report_viewer.has_loaded_results() is True
    banner = window._guided_start_open_status_label.text()
    assert banner != "Completed run: none"
    assert os.path.basename(run_dir) in window._guided_start_open_status_label.toolTip()
    assert "Completed run loaded:" in window._guided_start_status_label.text()


# --------------------------------------------------------------------------
# 16. RWD shared paths
# --------------------------------------------------------------------------


def test_rwd_recording_structure_is_unaffected(window, rwd_folder, tmp_path):
    _reach_recording_structure(window, rwd_folder, tmp_path, fmt="auto", csv=False)

    assert window._guided_sessions_per_hour_edit.text() == "1"
    assert window._guided_session_duration_edit.text() == "600"
    inference = window._guided_recording_timing_inference_label.text()
    assert "Detected" in inference and "confirm" in inference.lower()

    joined = "\n".join(_page_texts(window))
    assert "elapsed-time column" not in joined
    assert "Selected recording file" not in joined
