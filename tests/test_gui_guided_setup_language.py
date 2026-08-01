"""The words the Guided setup and Review Plan screens actually show.

Everything here reads the visible widget after driving the real controls, so a
label that is merely defined but never shown cannot make a test pass.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel

from gui.main_window import MainWindow


pytestmark = pytest.mark.usefixtures("no_real_modals")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_csv(path: Path, *, rows: int = 12000, fs_hz: float = 20.0) -> None:
    t = np.arange(rows, dtype=float) / fs_hz
    rng = np.random.default_rng(11)
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
    folder = tmp_path_factory.mktemp("lang_sessions")
    for index in range(3):
        _write_csv(folder / f"session_{index + 1:04d}.csv")
    return folder


@pytest.fixture(scope="module")
def continuous_csv_folder(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("lang_continuous")
    _write_csv(folder / "continuous_recording.csv", rows=24000)
    return folder


@pytest.fixture(scope="module")
def rwd_folder(tmp_path_factory) -> Path:
    """Four one-hour RWD sessions, so timing really is detected."""
    folder = tmp_path_factory.mktemp("lang_rwd")
    fs_hz = 5.0
    rows = int(600.0 * fs_hz)
    rng = np.random.default_rng(3)
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


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _spin(times: int = 150) -> None:
    for _ in range(times):
        QCoreApplication.processEvents()


def _pump(predicate, *, limit: int = 40000) -> bool:
    for _ in range(limit):
        if predicate():
            return True
        QCoreApplication.processEvents()
    return predicate()


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


def _reach_recording_structure(window, folder, output, **kwargs):
    csv = kwargs.pop("csv", True)
    _select_data(window, folder, output, **kwargs)
    if csv:
        _map_csv(window)
    _discover(window)
    window._on_guided_continue_to_recording_structure()
    _spin(200)


def _visible_texts(window):
    page = window._guided_workflow_stack.currentWidget()
    return [
        label.text()
        for label in page.findChildren(QLabel)
        if not label.isHidden() and label.text().strip()
    ]


def _group_titles_for(label) -> list[str]:
    titles, parent = [], label.parentWidget()
    while parent is not None:
        if isinstance(parent, QGroupBox):
            titles.append(parent.title())
        parent = parent.parentWidget()
    return titles


def _primary_texts(window):
    """Visible text outside the clearly labelled technical-details area."""
    page = window._guided_workflow_stack.currentWidget()
    texts = []
    for label in page.findChildren(QLabel):
        if label.isHidden() or not label.text().strip():
            continue
        titles = _group_titles_for(label)
        if any("Technical" in title for title in titles):
            continue
        texts.append(label.text())
    return texts


# --------------------------------------------------------------------------
# Session timing fields
# --------------------------------------------------------------------------


def test_sessions_per_hour_placeholder_does_not_claim_optional(
    window, intermittent_csv_folder, tmp_path
):
    _reach_recording_structure(window, intermittent_csv_folder, tmp_path)
    placeholder = window._guided_sessions_per_hour_edit.placeholderText()

    assert placeholder == "Sessions per hour"
    assert "optional" not in placeholder.lower()


def test_sessions_per_hour_tooltip_has_no_duty_cycled_jargon(
    window, intermittent_csv_folder, tmp_path
):
    _reach_recording_structure(window, intermittent_csv_folder, tmp_path)
    tooltip = window._guided_sessions_per_hour_edit.toolTip()

    assert "duty-cycled" not in tooltip
    assert "sessions occur in each hour" in tooltip


def test_session_duration_tooltip_never_says_to_leave_it_blank(
    window, intermittent_csv_folder, tmp_path
):
    _reach_recording_structure(window, intermittent_csv_folder, tmp_path)
    tooltip = window._guided_session_duration_edit.toolTip()

    assert "Leave blank" not in tooltip
    assert window._guided_session_duration_edit.placeholderText() == (
        "Session duration in seconds"
    )


def test_undetected_timing_asks_the_scientist_to_enter_the_values(
    window, intermittent_csv_folder, tmp_path
):
    _reach_recording_structure(window, intermittent_csv_folder, tmp_path)

    assert window._guided_sessions_per_hour_edit.text() == ""
    assert window._guided_session_duration_edit.text() == ""
    inference = window._guided_recording_timing_inference_label.text()
    helper = window._guided_recording_structure_help_label.text()

    assert inference.startswith("Enter the sessions per hour")
    assert helper.startswith("Enter the sessions per hour")
    assert "Confirm sessions per hour" not in inference
    assert "Confirm sessions per hour" not in helper


def test_detected_timing_keeps_confirmation_wording_and_names_the_values(
    window, rwd_folder, tmp_path
):
    _reach_recording_structure(window, rwd_folder, tmp_path, fmt="auto", csv=False)

    assert window._guided_sessions_per_hour_edit.text() == "1"
    assert window._guided_session_duration_edit.text() == "600"
    inference = window._guided_recording_timing_inference_label.text()

    assert "Detected" in inference and "confirm" in inference.lower()
    assert "1 sessions/hour" in inference
    assert "600 s/session" in inference
    assert "read this from the recording" in (
        window._guided_session_duration_edit.toolTip()
    )


# --------------------------------------------------------------------------
# Detect automatically
# --------------------------------------------------------------------------


def test_auto_with_one_csv_explains_the_one_file_rule(
    window, continuous_csv_folder, tmp_path
):
    _select_data(window, continuous_csv_folder, tmp_path)
    label = window._guided_auto_structure_explanation_label

    assert label.isHidden() is False
    assert "one CSV file is read as one continuous recording" in label.text()
    assert "several CSV files are read as repeated sessions" in label.text()
    assert "custom_tabular" not in label.text()
    assert window._guided_intermittent_explanation_label.isHidden() is True


def test_auto_with_several_csv_files_explains_the_same_rule(
    window, intermittent_csv_folder, tmp_path
):
    _select_data(window, intermittent_csv_folder, tmp_path)
    label = window._guided_auto_structure_explanation_label

    assert label.isHidden() is False
    assert "several CSV files are read as repeated sessions" in label.text()
    # The intermittent explanation describes a stated choice, not a pending one.
    assert window._guided_intermittent_explanation_label.isHidden() is True


def test_auto_for_npm_states_only_the_supported_structure(
    window, intermittent_csv_folder, tmp_path
):
    _select_data(window, intermittent_csv_folder, tmp_path, fmt="npm")
    label = window._guided_auto_structure_explanation_label

    assert label.isHidden() is False
    assert label.text() == (
        "Detect automatically: NPM recordings are read as repeated sessions."
    )


def test_choosing_a_structure_replaces_the_automatic_explanation(
    window, intermittent_csv_folder, tmp_path
):
    _select_data(window, intermittent_csv_folder, tmp_path, structure="intermittent")

    assert window._guided_auto_structure_explanation_label.isHidden() is True
    assert window._guided_intermittent_explanation_label.isHidden() is False


# --------------------------------------------------------------------------
# Recording structure screen
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "structure,folder_name",
    [("continuous", "continuous_csv_folder"), (None, "intermittent_csv_folder")],
    ids=["continuous", "intermittent"],
)
def test_recording_structure_subtitle_is_source_neutral(
    window, tmp_path, request, structure, folder_name
):
    folder = request.getfixturevalue(folder_name)
    _reach_recording_structure(window, folder, tmp_path, structure=structure)
    subtitles = [t for t in _visible_texts(window) if t.startswith("Tell the app")]

    assert subtitles, "recording-structure subtitle was not visible"
    assert subtitles[0] == (
        "Tell the app how your recording is organized so it can interpret "
        "the timeline correctly."
    )
    assert "align sessions" not in subtitles[0]


def test_continuous_recording_structure_shows_no_session_alignment_wording(
    window, continuous_csv_folder, tmp_path
):
    _reach_recording_structure(
        window, continuous_csv_folder, tmp_path, structure="continuous"
    )

    for text in _visible_texts(window):
        assert "align sessions" not in text
        assert "sessions per hour" not in text.lower()


# --------------------------------------------------------------------------
# Correction approach and Feature Detection
# --------------------------------------------------------------------------


def _confirm_corrections(window):
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


def _reach_review_plan(window, folder, output, **kwargs):
    _reach_recording_structure(window, folder, output, **kwargs)
    if kwargs.get("structure") != "continuous":
        window._guided_sessions_per_hour_edit.setText("2")
        window._guided_session_duration_edit.setText("600")
    window._guided_fixed_daily_anchor_clock_edit.setText("07:00")
    window._guided_recording_start_clock_edit.setText("12:00:00")
    _spin(150)
    window._on_guided_continue_to_correction_approach()
    _spin(250)
    if kwargs.get("structure") == "continuous":
        _pump(
            lambda: getattr(window, "_guided_continuous_rwd_check_thread", None) is None,
            limit=400000,
        )
        _spin(300)
    _confirm_corrections(window)
    window._on_guided_continue_to_feature_detection()
    _spin(250)


def test_correction_completion_names_feature_detection_as_the_next_step(
    window, intermittent_csv_folder, tmp_path
):
    _reach_recording_structure(window, intermittent_csv_folder, tmp_path)
    window._guided_sessions_per_hour_edit.setText("2")
    window._guided_session_duration_edit.setText("600")
    window._guided_fixed_daily_anchor_clock_edit.setText("07:00")
    window._guided_recording_start_clock_edit.setText("12:00:00")
    _spin(150)
    window._on_guided_continue_to_correction_approach()
    _spin(250)
    _confirm_corrections(window)

    next_action = window._guided_correction_next_action_label.text()

    assert "Continue to Feature Detection." in next_action
    assert "Draft plan" not in next_action


def test_feature_detection_summary_uses_scientist_facing_names(
    window, intermittent_csv_folder, tmp_path
):
    _reach_review_plan(window, intermittent_csv_folder, tmp_path)
    summary = window._guided_feature_detection_summary_label.text()

    assert "Event signal: dF/F" in summary
    assert "Threshold: mean + standard-deviation" in summary
    assert "AUC baseline: zero line" in summary
    assert "dff" not in summary
    assert "mean_std" not in summary
    assert "AUC baseline: zero\n" not in summary


def test_feature_detection_per_roi_table_uses_the_same_names(
    window, intermittent_csv_folder, tmp_path
):
    _reach_review_plan(window, intermittent_csv_folder, tmp_path)
    table = window._guided_feature_event_per_roi_table
    cells = [
        table.item(row, 2).text()
        for row in range(table.rowCount())
        if table.item(row, 2) is not None
    ]

    assert cells
    for cell in cells:
        assert "mean + standard-deviation threshold" in cell
        assert "dF/F signal" in cell
        assert "mean_std" not in cell
        assert "· dff" not in cell


# --------------------------------------------------------------------------
# Review Plan
# --------------------------------------------------------------------------


def test_review_plan_uses_the_csv_display_label(
    window, intermittent_csv_folder, tmp_path
):
    _reach_review_plan(window, intermittent_csv_folder, tmp_path)
    window._on_guided_continue_to_review_plan()
    _spin(400)
    summary = window._guided_review_analysis_summary_label.text()

    assert "Input format: CSV files" in summary
    for text in _primary_texts(window):
        assert "CUSTOM_TABULAR" not in text
        assert "custom_tabular" not in text


def test_review_plan_primary_text_has_no_backend_vocabulary(
    window, intermittent_csv_folder, tmp_path
):
    _reach_review_plan(window, intermittent_csv_folder, tmp_path)
    window._on_guided_continue_to_review_plan()
    _spin(400)

    primary = _primary_texts(window)
    joined = "\n".join(primary)

    assert "backend validation" not in joined
    assert "Execution availability" not in joined
    assert "Run readiness:" in window._guided_review_plan_status_label.text()
    assert "Confirm that the detected files, timing, and included data" in joined


def test_review_plan_names_the_recording_structure_in_plain_words(
    window, intermittent_csv_folder, tmp_path
):
    _reach_review_plan(window, intermittent_csv_folder, tmp_path)
    window._on_guided_continue_to_review_plan()
    _spin(400)
    summary = window._guided_review_analysis_summary_label.text()

    assert "Recording structure: Repeated sessions" in summary
    assert "Acquisition mode" not in summary
    assert "intermittent" not in summary


def test_review_plan_shows_the_discovered_session_count_for_repeated_sessions(
    window, intermittent_csv_folder, tmp_path
):
    _reach_review_plan(window, intermittent_csv_folder, tmp_path)
    window._on_guided_continue_to_review_plan()
    _spin(400)
    summary = window._guided_review_analysis_summary_label.text()

    assert "Sessions discovered: 3" in summary


def test_review_plan_speaks_only_for_final_outputs(
    window, intermittent_csv_folder, tmp_path
):
    """Correction previews have already written files under this destination."""
    _reach_review_plan(window, intermittent_csv_folder, tmp_path)
    window._on_guided_continue_to_review_plan()
    _spin(400)
    output_status = window._guided_review_output_status_label.text()

    assert "Final analysis outputs: not created yet." in output_status
    assert "Files written so far: none" not in output_status


def test_continuous_review_plan_states_sessions_do_not_apply(
    window, continuous_csv_folder, tmp_path
):
    _reach_review_plan(
        window, continuous_csv_folder, tmp_path, structure="continuous"
    )
    window._on_guided_continue_to_review_plan()
    _spin(400)
    summary = window._guided_review_analysis_summary_label.text()

    assert "Recording structure: Continuous recording" in summary
    assert "Sessions discovered" not in summary
    assert "not available" not in summary
    assert "Recording sessions: not applicable" in summary
    # The internal value must never be the displayed one. ("continuous" still
    # appears in the folder path and in the sentence above, which is fine.)
    structure_line = next(
        line for line in summary.splitlines()
        if line.startswith("Recording structure:")
    )
    assert structure_line.startswith("Recording structure: Continuous recording")


def test_technical_details_may_keep_internal_tokens(
    window, intermittent_csv_folder, tmp_path
):
    """Only the clearly labelled technical area is allowed to show them."""
    _reach_review_plan(window, intermittent_csv_folder, tmp_path)
    window._on_guided_continue_to_review_plan()
    _spin(400)

    page = window._guided_workflow_stack.currentWidget()
    technical = [
        label.text()
        for label in page.findChildren(QLabel)
        if not label.isHidden()
        and any("Technical" in title for title in _group_titles_for(label))
    ]

    assert technical, "technical-details area was not visible"
    assert any("custom_tabular" in text for text in technical)
