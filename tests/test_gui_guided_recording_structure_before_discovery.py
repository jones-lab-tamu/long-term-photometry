"""CR1-F1-A: the recording structure is chosen before ROI discovery reads the source.

The first real scientist-facing validation of the continuous-RWD Guided
workflow failed at Select data: ROI discovery interpreted a continuous
acquisition folder as repeated sessions, refused it, and left the scientist
unable to reach the acquisition-structure control that would have corrected
it -- because that control lived in the *next* step.

These tests drive the real Select-data boundary: the structure control's
placement, the routing of discovery on the explicit choice, the visible
ordering, and what happens when the choice or the source changes.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox

import gui.main_window as main_window_module
from gui.main_window import (
    GuidedContinuousRwdRoiDiscoveryError,
    MainWindow,
    _discover_continuous_rwd_rois,
)

from tests.test_guided_continuous_rwd_correction_pass_persistence import _values


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


def _continuous_folder(folder):
    """One small continuous RWD acquisition folder, including the
    non-fluorescence CSVs a real RWD folder carries."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(600):
        time_s, control, signal = _values(np.array([index], dtype=float))
        lines.append(
            f"{time_s[0]:.1f},{control[0,0]:.12f},{signal[0,0]:.12f},"
            f"{control[0,1]:.12f},{signal[0,1]:.12f}\n"
        )
    (folder / "Fluorescence.csv").write_text(
        "".join(lines), encoding="utf-8", newline=""
    )
    # These exist in real RWD acquisition folders and are not recordings. The
    # intermittent path used to try to read one of them as a session chunk.
    (folder / "Events.csv").write_text(
        "Time(s),Event\n0.0,start\n", encoding="utf-8", newline=""
    )
    (folder / "Outputs.csv").write_text(
        "Time(s),Output\n0.0,0\n", encoding="utf-8", newline=""
    )
    return folder


def _select_data(window, folder, *, fmt="rwd", structure="continuous", output=None):
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText(fmt)
    index = window._guided_acquisition_mode_combo.findData(structure)
    assert index >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(index)
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(output or (folder / "output")))


def _pump_discovery(window, qapp, timeout_ms=180_000):
    from PySide6.QtCore import QDeadlineTimer

    deadline = QDeadlineTimer(timeout_ms)
    while window._guided_roi_discovery_running and not deadline.hasExpired():
        qapp.processEvents()
    deadline = QDeadlineTimer(20_000)
    while (
        getattr(window, "_guided_roi_discovery_thread", None) is not None
        and not deadline.hasExpired()
    ):
        qapp.processEvents()


# ---------------------------------------------------------------------------
# Control placement and availability
# ---------------------------------------------------------------------------


def test_recording_structure_control_lives_in_select_data(window):
    """It must be reachable before discovery, not in the step the scientist
    could not get to."""
    select_data = window._guided_raw_setup_controls["Select data"]
    recording_structure = window._guided_raw_setup_controls["Recording structure"]

    assert window._guided_acquisition_mode_combo in select_data.findChildren(
        QComboBox
    )
    assert window._guided_acquisition_mode_combo not in (
        recording_structure.findChildren(QComboBox)
    )
    # Exactly one such control exists -- it moved, it was not duplicated.
    structure_combos = [
        combo
        for combo in window._guided_workflow_tab.findChildren(QComboBox)
        if combo.objectName() == "guidedAcquisitionModeCombo"
    ]
    assert structure_combos == [window._guided_acquisition_mode_combo]


def test_structure_specific_settings_stay_in_recording_structure(window):
    recording_structure = window._guided_raw_setup_controls["Recording structure"]
    for widget in (
        window._guided_sessions_per_hour_edit,
        window._guided_session_duration_edit,
        window._guided_continuous_window_sec_spin,
    ):
        assert widget in recording_structure.findChildren(type(widget))


def test_rwd_offers_both_structures(window):
    window._guided_format_combo.setCurrentText("rwd")
    model = window._guided_acquisition_mode_combo.model()
    for mode in ("intermittent", "continuous"):
        index = window._guided_acquisition_mode_combo.findData(mode)
        assert index >= 0
        assert model.item(index).isEnabled() is True


@pytest.mark.parametrize("unsupported_format", ["npm", "custom_tabular"])
def test_unsupported_formats_disable_continuous(window, unsupported_format):
    window._guided_format_combo.setCurrentText(unsupported_format)
    index = window._guided_acquisition_mode_combo.findData("continuous")
    item = window._guided_acquisition_mode_combo.model().item(index)
    assert item.isEnabled() is False
    assert "RWD" in item.toolTip()


def test_leaving_rwd_returns_structure_to_intermittent(window, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)
    assert window._guided_selected_acquisition_mode() == "continuous"

    window._guided_format_combo.setCurrentText("npm")

    assert window._guided_selected_acquisition_mode() == "intermittent"
    draft = window._build_guided_new_analysis_draft_plan()
    assert draft.acquisition_mode == "intermittent"


def test_live_draft_reflects_the_explicit_selection(window, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder)
    assert window._build_guided_new_analysis_draft_plan().acquisition_mode == (
        "continuous"
    )
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )
    assert window._build_guided_new_analysis_draft_plan().acquisition_mode == (
        "intermittent"
    )


# ---------------------------------------------------------------------------
# Discovery routing follows the explicit choice
# ---------------------------------------------------------------------------


def test_continuous_rwd_routes_to_continuous_discovery(window, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")

    snapshot = window._snapshot_guided_discovery_inputs()
    assert snapshot["acquisition_mode"] == "continuous"
    runner = window._guided_discovery_runner_for_snapshot(snapshot)

    assert runner is _discover_continuous_rwd_rois


def test_intermittent_rwd_routes_to_existing_discovery(window, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="intermittent")

    snapshot = window._snapshot_guided_discovery_inputs()
    assert snapshot["acquisition_mode"] == "intermittent"
    runner = window._guided_discovery_runner_for_snapshot(snapshot)

    assert runner is not _discover_continuous_rwd_rois
    assert runner.__name__ == "run_intermittent"


@pytest.mark.parametrize("unsupported_format", ["npm", "custom_tabular"])
def test_unsupported_formats_never_reach_continuous_discovery(
    window, tmp_path, unsupported_format
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")
    window._guided_format_combo.setCurrentText(unsupported_format)

    snapshot = window._snapshot_guided_discovery_inputs()
    runner = window._guided_discovery_runner_for_snapshot(snapshot)

    assert runner is not _discover_continuous_rwd_rois


def test_continuous_discovery_calls_only_the_accepted_reader(
    window, qapp, tmp_path, monkeypatch
):
    """One continuous parsing implementation: the accepted source inspector.
    The intermittent chunk-contract inference must not run."""
    import photometry_pipeline.io.rwd_continuous_source as source_module

    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")

    inspect_calls: list[str] = []
    real_inspect = source_module.inspect_continuous_rwd_acquisition_folder

    def counted(path, **kwargs):
        inspect_calls.append(str(path))
        return real_inspect(path, **kwargs)

    monkeypatch.setattr(
        source_module, "inspect_continuous_rwd_acquisition_folder", counted
    )
    monkeypatch.setattr(
        window,
        "_infer_rwd_dataset_contract_overrides",
        lambda *a, **k: pytest.fail(
            "intermittent RWD chunk-contract inference ran for a continuous plan"
        ),
    )

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    assert len(inspect_calls) == 1
    assert [
        window._guided_roi_list.item(i).text()
        for i in range(window._guided_roi_list.count())
    ] == ["ROI1", "ROI2"]


def test_continuous_discovery_does_no_execution_preparation(tmp_path, monkeypatch):
    """Select data identifies ROIs only -- no grid, block plan, segment plan,
    dynamic-F0 authority, or Config."""
    import importlib

    folder = _continuous_folder(tmp_path / "rec")
    for module_path, attr in (
        (
            "photometry_pipeline.guided_continuous_rwd_target_grid",
            "build_guided_continuous_rwd_target_grid",
        ),
        (
            "photometry_pipeline.guided_continuous_rwd_block_plan",
            "build_guided_continuous_rwd_block_plan",
        ),
        (
            "photometry_pipeline.guided_continuous_rwd_correction_segments",
            "build_guided_continuous_rwd_correction_segment_plan",
        ),
        (
            "photometry_pipeline.guided_continuous_rwd_correction_segments",
            "prepare_guided_continuous_rwd_dynamic_f0_authority",
        ),
        (
            "photometry_pipeline.guided_continuous_rwd_run_config",
            "build_guided_continuous_rwd_run_config",
        ),
    ):
        monkeypatch.setattr(
            importlib.import_module(module_path),
            attr,
            lambda *a, _attr=attr, **k: pytest.fail(
                f"{_attr} ran during ROI discovery"
            ),
        )

    result = _discover_continuous_rwd_rois({"input_dir": str(folder)})

    assert result["resolved_format"] == "rwd"
    assert result["acquisition_mode"] == "continuous"
    assert [roi["roi_id"] for roi in result["rois"]] == ["ROI1", "ROI2"]
    # One recording, not a set of sessions.
    assert result["sessions"] == []


def test_unreadable_continuous_folder_gives_plain_guidance(tmp_path):
    empty = tmp_path / "not_a_recording"
    empty.mkdir()
    with pytest.raises(GuidedContinuousRwdRoiDiscoveryError) as excinfo:
        _discover_continuous_rwd_rois({"input_dir": str(empty)})
    message = str(excinfo.value)
    for forbidden in (
        "Traceback",
        "ValueError",
        "projection",
        "parser_interpretation",
        "contract_version",
    ):
        assert forbidden not in message


def test_continuous_discovery_failure_is_shown_as_written(window):
    """The scientist-facing reason replaces the session-oriented guidance."""
    shown = window._guided_roi_discovery_failure_message(
        "reason: This folder could not be read as a continuous RWD recording."
    )
    assert shown == (
        "This folder could not be read as a continuous RWD recording."
    )
    assert "Fluorescence.csv files" not in shown

    # An unexpected internal error still maps to the existing guidance.
    generic = window._guided_roi_discovery_failure_message(
        "ValueError: no usable roi found"
    )
    assert "ValueError" not in generic


# ---------------------------------------------------------------------------
# The corrected visible ordering
# ---------------------------------------------------------------------------


def test_visible_order_reaches_rois_before_recording_structure(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")

    # The scientist has not visited Recording structure at all yet.
    assert window._guided_step_index("Recording structure") not in (
        window._guided_reached_step_indices
    )

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    rois = [
        window._guided_roi_list.item(i).text()
        for i in range(window._guided_roi_list.count())
    ]
    assert rois == ["ROI1", "ROI2"]
    assert window._guided_select_data_ready_to_continue() is True

    window._on_guided_continue_to_recording_structure()
    assert window._guided_workflow_stack.currentWidget().objectName() == (
        "guidedStepRecordingStructure"
    )
    # Continuous settings only.
    assert window._guided_continuous_window_sec_spin.isHidden() is False
    assert window._guided_sessions_per_hour_edit.isHidden() is True
    assert window._guided_session_duration_edit.isHidden() is True


def test_continue_stays_disabled_until_a_roi_is_included(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")
    assert window._guided_select_data_ready_to_continue() is False

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)
    assert window._guided_select_data_ready_to_continue() is True

    for index in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(index).setCheckState(Qt.Unchecked)
    assert window._guided_select_data_ready_to_continue() is False


def test_continuous_select_data_never_describes_sessions(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")
    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    summary = window._guided_discovery_summary_label.text()
    assert "session" not in summary.lower()
    assert "continuous recording" in summary.lower()


# ---------------------------------------------------------------------------
# Invalidation on structure and source change
# ---------------------------------------------------------------------------


def test_changing_structure_discards_discovered_rois(window, qapp, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")
    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)
    assert window._guided_roi_list.count() == 2

    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )

    assert window._guided_roi_list.count() == 0
    assert window._discovery_cache is None
    assert window._guided_select_data_ready_to_continue() is False
    assert window._guided_continuous_rwd_review_binding is None
    assert window._guided_continuous_rwd_prepared_run is None


def test_changing_structure_the_other_way_also_discards(window, qapp, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")
    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)
    assert window._guided_roi_list.count() == 2

    # continuous -> intermittent -> continuous: no ROI survives a round trip
    # without a fresh discovery.
    for mode in ("intermittent", "continuous"):
        window._guided_acquisition_mode_combo.setCurrentIndex(
            window._guided_acquisition_mode_combo.findData(mode)
        )
    assert window._guided_roi_list.count() == 0
    assert window._guided_select_data_ready_to_continue() is False


def test_stale_discovery_completion_is_discarded(window, qapp, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")
    generation = window._guided_discovery_generation

    # The scientist changes the structure while a discovery is in flight.
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )

    window._on_guided_roi_discovery_succeeded(
        {
            "resolved_format": "rwd",
            "acquisition_mode": "continuous",
            "rois": [{"roi_id": "STALE1"}, {"roi_id": "STALE2"}],
            "sessions": [],
        },
        generation=generation,
    )

    assert window._discovery_cache is None
    assert window._guided_roi_list.count() == 0
    assert window._guided_select_data_ready_to_continue() is False


def test_stale_discovery_failure_is_not_shown(window, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")
    generation = window._guided_discovery_generation
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )

    dialogs: list[str] = []
    from PySide6.QtWidgets import QMessageBox

    original = QMessageBox.critical
    try:
        QMessageBox.critical = staticmethod(
            lambda *a, **k: dialogs.append(str(a[1:3])) or QMessageBox.Ok
        )
        window._on_guided_roi_discovery_failed(
            "reason: stale continuous failure", generation=generation
        )
    finally:
        QMessageBox.critical = original

    assert dialogs == []


def test_discovery_results_are_applied_on_the_gui_thread(window, qapp, tmp_path):
    """The result slots must be queued to the GUI thread.

    The generation travels inside the signal precisely so the connections can
    stay bound methods. Connecting a lambda instead gives Qt no QObject
    receiver, which makes the connection direct and runs the whole
    ROI-population/`_on_config_changed` cascade on the discovery worker
    thread -- observed to deadlock.
    """
    import threading

    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")

    gui_thread = threading.get_ident()
    seen: dict[str, int] = {}
    real_populate = window._populate_discovery_ui

    def recording_populate(disco):
        seen["thread"] = threading.get_ident()
        return real_populate(disco)

    window._populate_discovery_ui = recording_populate

    window._on_guided_discover_rois()
    _pump_discovery(window, qapp)

    assert seen.get("thread") == gui_thread
    assert window._guided_roi_list.count() == 2


def test_worker_captures_the_mode_and_never_reads_the_combo(window, tmp_path):
    """The worker consumes only its immutable snapshot."""
    import inspect

    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, folder, structure="continuous")
    snapshot = window._snapshot_guided_discovery_inputs()
    assert snapshot["acquisition_mode"] == "continuous"

    worker = main_window_module._GuidedRoiDiscoveryWorker(
        snapshot,
        window._guided_discovery_runner_for_snapshot(snapshot),
        start_monotonic=0.0,
        gui_thread_id=0,
    )
    assert worker._snapshot["acquisition_mode"] == "continuous"
    source = inspect.getsource(main_window_module._GuidedRoiDiscoveryWorker)
    assert "_guided_acquisition_mode_combo" not in source
    assert "_selected_acquisition_mode" not in source
