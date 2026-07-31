"""What Select data tells a scientist when it refuses.

Two refusals are covered, both driven through the real Select data controls:

* ROI discovery failed. The message must name the CSV control to change for a
  CSV source, and must keep the RWD-specific guidance for an RWD source.
* The output destination cannot be used. That must be answered here, not
  several steps later by a correction preview blaming the source recording.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from gui.main_window import (
    GUIDED_CSV_DISCOVERY_GENERIC_FAILURE_MESSAGE,
    GUIDED_OUTPUT_DESTINATION_MISSING_MESSAGE,
    MainWindow,
)


pytestmark = pytest.mark.usefixtures("no_real_modals")

CSV_HEADERS = ("ElapsedSeconds", "R1_Signal", "R1_Reference", "R2_Signal", "R2_Reference")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_csv(path: Path, *, rows: int = 12000, fs_hz: float = 20.0) -> None:
    """One 600 s session at 20 Hz, matching the session length the pipeline
    reads by default -- a shorter file is rejected for end coverage, which is
    a different refusal than the ones under test here."""
    t = np.arange(rows, dtype=float) / fs_hz
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "ElapsedSeconds": t,
            "R1_Signal": 1.7 + 0.05 * np.sin(t / 7.0) + rng.normal(0, 0.004, rows),
            "R1_Reference": 1.0 + 0.02 * np.sin(t / 7.0) + rng.normal(0, 0.002, rows),
            "R2_Signal": 1.8 + 0.05 * np.cos(t / 9.0) + rng.normal(0, 0.004, rows),
            "R2_Reference": 1.0 + 0.02 * np.cos(t / 9.0) + rng.normal(0, 0.002, rows),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture(scope="module")
def intermittent_csv_folder(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("csv_sessions")
    for index in range(3):
        _write_csv(folder / f"session_{index + 1:04d}.csv")
    return folder


@pytest.fixture(scope="module")
def continuous_csv_folder(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("csv_continuous")
    _write_csv(folder / "continuous_recording.csv", rows=24000)
    return folder


@pytest.fixture(scope="module")
def unreadable_rwd_folder(tmp_path_factory) -> Path:
    """An RWD acquisition folder whose fluorescence header is unrecognizable."""
    folder = tmp_path_factory.mktemp("rwd_bad")
    session = folder / "2025_01_01-00_00_00"
    session.mkdir(parents=True)
    (session / "Fluorescence.csv").write_text(
        "colA,colB,colC\n1,2,3\n4,5,6\n", encoding="utf-8"
    )
    return folder


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


@pytest.fixture
def captured_modals(monkeypatch):
    """Capture the blocking dialog this workflow legitimately raises."""
    seen: list[tuple[str, str]] = []

    def _capture(*args, **_kwargs):
        seen.append(
            (
                str(args[1]) if len(args) > 1 else "",
                str(args[2]) if len(args) > 2 else "",
            )
        )
        return QMessageBox.Ok

    monkeypatch.setattr(QMessageBox, "critical", staticmethod(_capture))
    return seen


def _pump(predicate, *, limit: int = 40000) -> bool:
    for _ in range(limit):
        if predicate():
            return True
        QCoreApplication.processEvents()
    return predicate()


def _spin(times: int = 120) -> None:
    for _ in range(times):
        QCoreApplication.processEvents()


def _select_data(window, folder, output, *, fmt="custom_tabular", structure=None):
    """Enter Select data exactly as the Start screen does."""
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


def _map_csv(window, rois=("R1", "R2"), *, time_column="ElapsedSeconds", confirm_order=True):
    if time_column is not None:
        combo = window._guided_csv_time_column_combo
        combo.setCurrentIndex(combo.findData(time_column))
    if confirm_order:
        window._guided_csv_order_confirm_cb.setChecked(True)
    while len(window._guided_csv_mapping_rows) < len(rois):
        window._add_guided_csv_mapping_row()
    for row, roi in zip(window._guided_csv_mapping_rows, rois):
        row["name"].setText(roi)
        row["signal"].setCurrentIndex(row["signal"].findData(f"{roi}_Signal"))
        row["reference"].setCurrentIndex(
            row["reference"].findData(f"{roi}_Reference")
        )
    _spin(60)


def _discover(window):
    """Press the real Select ROIs action and wait for the worker to finish."""
    window._on_guided_discover_rois()
    _pump(lambda: not window._guided_roi_discovery_running)
    _pump(lambda: window._guided_roi_discovery_thread is None)
    _spin(120)


def _refuse(window, captured_modals) -> tuple[str, str]:
    """Return (modal text, persistent inline text) for the failed discovery."""
    _discover(window)
    assert captured_modals, "discovery was expected to fail with a dialog"
    title, text = captured_modals[-1]
    assert title == "ROI Selection Failed"
    return text, window._guided_discovery_summary_label.text()


def _assert_scientist_facing(*texts: str) -> None:
    for text in texts:
        assert "Fluorescence.csv" not in text
        assert "custom_tabular" not in text
        assert "Traceback" not in text
        for exception_name in ("ValueError", "RuntimeError", "OSError", "Exception"):
            assert exception_name not in text


# --------------------------------------------------------------------------
# CSV discovery refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("structure", ["intermittent", None])
def test_missing_time_column_names_the_time_column(
    window, intermittent_csv_folder, tmp_path, captured_modals, structure
):
    _select_data(window, intermittent_csv_folder, tmp_path, structure=structure)
    _map_csv(window, time_column=None)
    modal, inline = _refuse(window, captured_modals)

    assert modal == "Choose the CSV time column, then select ROIs again."
    assert inline == modal
    _assert_scientist_facing(modal, inline)


def test_incomplete_roi_mapping_names_the_roi_columns(
    window, intermittent_csv_folder, tmp_path, captured_modals
):
    _select_data(window, intermittent_csv_folder, tmp_path)
    combo = window._guided_csv_time_column_combo
    combo.setCurrentIndex(combo.findData("ElapsedSeconds"))
    window._guided_csv_order_confirm_cb.setChecked(True)
    _spin(60)
    modal, inline = _refuse(window, captured_modals)

    assert modal == (
        "Map both the signal and reference columns for each ROI, then "
        "select ROIs again."
    )
    assert inline == modal
    _assert_scientist_facing(modal, inline)


DUPLICATE_COLUMN_MESSAGE = (
    "Each CSV column can be assigned only once. Choose a different signal "
    "or reference column for the repeated assignment, then select ROIs again."
)


def _map_signal_reference_same_roi(window):
    _map_csv(window, rois=("R1",))
    row = window._guided_csv_mapping_rows[0]
    row["reference"].setCurrentIndex(row["reference"].findData("R1_Signal"))


def _map_signal_reused_by_two_rois(window):
    _map_csv(window, rois=("R1", "R2"))
    # The second ROI takes the first ROI's signal column.
    second = window._guided_csv_mapping_rows[1]
    second["signal"].setCurrentIndex(second["signal"].findData("R1_Signal"))


def _map_reference_reused_by_two_rois(window):
    _map_csv(window, rois=("R1", "R2"))
    # The second ROI takes the first ROI's reference column.
    second = window._guided_csv_mapping_rows[1]
    second["reference"].setCurrentIndex(second["reference"].findData("R1_Reference"))


@pytest.mark.parametrize(
    "mapper",
    [
        _map_signal_reference_same_roi,
        _map_signal_reused_by_two_rois,
        _map_reference_reused_by_two_rois,
    ],
    ids=["same_roi", "signal_reused_across_rois", "reference_reused_across_rois"],
)
def test_any_repeated_column_assignment_gets_the_same_accurate_message(
    window, intermittent_csv_folder, tmp_path, captured_modals, mapper
):
    """The rule is uniqueness across the whole mapping, not within one ROI."""
    _select_data(window, intermittent_csv_folder, tmp_path)
    mapper(window)
    _spin(60)
    modal, inline = _refuse(window, captured_modals)

    assert modal == DUPLICATE_COLUMN_MESSAGE
    assert inline == modal
    _assert_scientist_facing(modal, inline)


def test_unconfirmed_session_order_names_the_order_confirmation(
    window, intermittent_csv_folder, tmp_path, captured_modals
):
    _select_data(window, intermittent_csv_folder, tmp_path)
    _map_csv(window, confirm_order=False)
    modal, inline = _refuse(window, captured_modals)

    assert modal == "Confirm the displayed session-file order, then select ROIs again."
    assert inline == modal
    _assert_scientist_facing(modal, inline)


def test_explicit_intermittent_with_one_file_still_confirms_session_order(
    window, continuous_csv_folder, tmp_path, captured_modals
):
    """The scientist chose the session-based interpretation outright."""
    _select_data(window, continuous_csv_folder, tmp_path, structure="intermittent")
    _map_csv(window, confirm_order=False)
    modal, inline = _refuse(window, captured_modals)

    assert modal == "Confirm the displayed session-file order, then select ROIs again."
    assert inline == modal


@pytest.mark.parametrize("structure", ["continuous", None], ids=["explicit", "auto"])
def test_one_continuous_recording_needs_no_order_confirmation(
    window, continuous_csv_folder, tmp_path, captured_modals, structure
):
    """One file has no order to confirm, so discovery must not demand one."""
    _select_data(window, continuous_csv_folder, tmp_path, structure=structure)
    _map_csv(window, confirm_order=False)
    assert window._guided_csv_order_confirm_cb.isChecked() is False
    assert window._guided_csv_recording_structure_in_effect() == "continuous"

    _discover(window)

    assert captured_modals == []
    assert window._guided_roi_list.count() > 0
    assert window._guided_resolved_acquisition_mode == "continuous"


@pytest.mark.parametrize("structure", ["continuous", None], ids=["explicit", "auto"])
def test_continuous_csv_reports_the_real_missing_setting_with_order_unchecked(
    window, continuous_csv_folder, tmp_path, captured_modals, structure
):
    _select_data(window, continuous_csv_folder, tmp_path, structure=structure)
    _map_csv(window, time_column=None, confirm_order=False)
    modal, inline = _refuse(window, captured_modals)

    assert modal == "Choose the CSV time column, then select ROIs again."
    assert inline == modal
    _assert_scientist_facing(modal, inline)


@pytest.mark.parametrize("structure", ["continuous", None], ids=["explicit", "auto"])
def test_continuous_csv_reports_incomplete_mapping_with_order_unchecked(
    window, continuous_csv_folder, tmp_path, captured_modals, structure
):
    _select_data(window, continuous_csv_folder, tmp_path, structure=structure)
    combo = window._guided_csv_time_column_combo
    combo.setCurrentIndex(combo.findData("ElapsedSeconds"))
    _spin(60)
    modal, inline = _refuse(window, captured_modals)

    assert modal == (
        "Map both the signal and reference columns for each ROI, then "
        "select ROIs again."
    )
    assert inline == modal


def test_continuous_csv_duplicate_columns_reported_with_order_unchecked(
    window, continuous_csv_folder, tmp_path, captured_modals
):
    _select_data(window, continuous_csv_folder, tmp_path, structure="continuous")
    _map_csv(window, rois=("R1",), confirm_order=False)
    row = window._guided_csv_mapping_rows[0]
    row["reference"].setCurrentIndex(row["reference"].findData("R1_Signal"))
    _spin(60)
    modal, inline = _refuse(window, captured_modals)

    assert modal == DUPLICATE_COLUMN_MESSAGE
    assert inline == modal


def test_no_refusal_ever_asks_to_confirm_a_single_recording_file(
    window, continuous_csv_folder, tmp_path, captured_modals
):
    """The old continuous stand-in for session order is gone, not reworded."""
    import gui.main_window as main_window_module

    assert not hasattr(
        main_window_module, "GUIDED_CSV_SETUP_ORDER_CONTINUOUS_MESSAGE"
    )
    _select_data(window, continuous_csv_folder, tmp_path, structure="continuous")
    _map_csv(window, time_column=None, confirm_order=False)
    modal, _inline = _refuse(window, captured_modals)

    assert "recording file" not in modal


def test_unknown_csv_read_failure_uses_the_csv_fallback(
    window, intermittent_csv_folder, tmp_path, captured_modals, monkeypatch
):
    """An unclassified failure on a CSV source must not borrow RWD guidance."""
    _select_data(window, intermittent_csv_folder, tmp_path)
    _map_csv(window)

    def _explode(_snapshot):
        raise RuntimeError("simulated unreadable CSV")

    monkeypatch.setattr(window, "_build_discovery_spec_from_snapshot", _explode)
    modal, inline = _refuse(window, captured_modals)

    assert modal == GUIDED_CSV_DISCOVERY_GENERIC_FAILURE_MESSAGE
    assert inline == modal
    assert "simulated unreadable CSV" not in modal
    _assert_scientist_facing(modal, inline)


def test_rwd_failure_keeps_the_rwd_specific_guidance(
    window, unreadable_rwd_folder, tmp_path, captured_modals
):
    _select_data(
        window, unreadable_rwd_folder, tmp_path, fmt="rwd", structure="intermittent"
    )
    modal, inline = _refuse(window, captured_modals)

    assert modal == (
        "The selected recording does not contain a recognizable "
        "fluorescence header. Check that you selected the folder "
        "containing the recording-session folders and their "
        "Fluorescence.csv files."
    )
    assert inline == modal
    assert modal != GUIDED_CSV_DISCOVERY_GENERIC_FAILURE_MESSAGE


# --------------------------------------------------------------------------
# Output-destination readiness
# --------------------------------------------------------------------------


def _ready_csv_selection(window, folder, output):
    _select_data(window, folder, output)
    _map_csv(window)
    _discover(window)
    for index in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(index).setCheckState(Qt.Checked)
    _spin(80)


def _select_data_state(window) -> tuple[bool, str]:
    return (
        window._guided_select_data_continue_btn.isEnabled(),
        window._guided_select_data_continue_status.text(),
    )


def test_valid_output_folder_is_ready(window, intermittent_csv_folder, tmp_path):
    _ready_csv_selection(window, intermittent_csv_folder, tmp_path / "results")
    enabled, reason = _select_data_state(window)

    assert enabled is True
    assert reason == "Data selection is ready."


def test_blank_output_folder_blocks_and_names_the_output_folder(
    window, intermittent_csv_folder, tmp_path
):
    _ready_csv_selection(window, intermittent_csv_folder, tmp_path / "results")
    assert _select_data_state(window)[0] is True

    window._guided_output_dir_edit.setText("")
    _spin(120)
    enabled, reason = _select_data_state(window)

    assert enabled is False
    assert reason == GUIDED_OUTPUT_DESTINATION_MISSING_MESSAGE
    assert "ROI" not in reason


def test_unreachable_output_path_blocks_and_names_the_output_folder(
    window, intermittent_csv_folder, tmp_path
):
    unreachable = tmp_path / "missing_parent" / "results"
    _ready_csv_selection(window, intermittent_csv_folder, unreachable)
    enabled, reason = _select_data_state(window)

    assert enabled is False
    assert reason.startswith("The selected output folder cannot be used")
    assert "could not be reached" in reason
    assert "ROI" not in reason
    assert "recording" not in reason


def test_output_path_that_is_a_file_blocks_and_names_the_output_folder(
    window, intermittent_csv_folder, tmp_path
):
    occupied = tmp_path / "results.txt"
    occupied.write_text("not a folder", encoding="utf-8")
    _ready_csv_selection(window, intermittent_csv_folder, occupied)
    enabled, reason = _select_data_state(window)

    assert enabled is False
    assert reason.startswith("The selected output folder cannot be used")
    assert "is a file, not a folder" in reason
    assert "ROI" not in reason


@pytest.fixture
def deny_write_access(monkeypatch):
    """Report every path as non-writable.

    Read-only directory behaviour is not portable, so only the OS answer is
    stubbed; each test still supplies the real filesystem state that tells
    the two destinations apart.
    """
    import photometry_pipeline.guided_execution_request_builder as builder

    monkeypatch.setattr(builder.os, "access", lambda *_args, **_kwargs: False)


def test_existing_but_non_writable_output_folder_is_described_as_non_writable(
    window, intermittent_csv_folder, tmp_path, deny_write_access
):
    existing = tmp_path / "results"
    existing.mkdir()
    _ready_csv_selection(window, intermittent_csv_folder, existing)
    enabled, reason = _select_data_state(window)

    assert enabled is False
    assert reason == (
        "The selected output folder cannot be used because the app cannot "
        "write to it. Choose a folder you can write to."
    )
    assert "cannot be created" not in reason
    assert "ROI" not in reason


def test_non_creatable_child_destination_is_described_as_non_creatable(
    window, intermittent_csv_folder, tmp_path, deny_write_access
):
    parent = tmp_path / "parent"
    parent.mkdir()
    destination = parent / "results"
    assert destination.exists() is False
    _ready_csv_selection(window, intermittent_csv_folder, destination)
    enabled, reason = _select_data_state(window)

    assert enabled is False
    assert reason == (
        "The selected output folder cannot be used because it does not "
        "exist and cannot be created in that parent folder. Choose "
        "another folder you can write to."
    )
    assert "cannot write to it" not in reason
    assert "ROI" not in reason


def test_readiness_check_creates_nothing(window, intermittent_csv_folder, tmp_path):
    destination = tmp_path / "results"
    before = sorted(p.name for p in tmp_path.iterdir())
    _ready_csv_selection(window, intermittent_csv_folder, destination)

    assert _select_data_state(window)[0] is True
    assert destination.exists() is False
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_repairing_the_output_folder_restores_readiness(
    window, intermittent_csv_folder, tmp_path
):
    _ready_csv_selection(
        window, intermittent_csv_folder, tmp_path / "missing_parent" / "results"
    )
    assert _select_data_state(window)[0] is False

    window._guided_output_dir_edit.setText(str(tmp_path / "results"))
    _spin(150)
    enabled, reason = _select_data_state(window)

    assert enabled is True
    assert reason == "Data selection is ready."


def test_breaking_the_output_folder_withdraws_readiness(
    window, intermittent_csv_folder, tmp_path
):
    _ready_csv_selection(window, intermittent_csv_folder, tmp_path / "results")
    assert _select_data_state(window)[0] is True

    window._guided_output_dir_edit.setText(str(tmp_path / "missing_parent" / "results"))
    _spin(150)
    enabled, reason = _select_data_state(window)

    assert enabled is False
    assert reason.startswith("The selected output folder cannot be used")
