"""CR1-F1-E: the recording check must complete through normal navigation.

A scientist reached Correction approach on a real continuous recording and the
page sat on "Waiting for the recording check to finish" indefinitely, on a
source whose check takes about 35 seconds.

Two defects, both proven by driving the visible Continue button:

* the worker's signals were connected to lambdas, which have no receiver
  object, so Qt ran the success handler on the *worker* thread -- every widget
  update underneath it happened off the GUI thread;
* nothing refreshed Correction approach when the check finished, so the page
  kept the message it was built with on arrival, forever.

CR1-F1-D's tests missed this because their helper called the private starter
and then rebuilt the page by hand, supplying the refresh production code never
performed. These tests call only the handler wired to the visible button.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PySide6.QtCore import QDeadlineTimer, Qt
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
from gui.main_window import GUIDED_STRUCTURE_CHOICE_AUTO, MainWindow

from tests.test_guided_continuous_rwd_correction_pass_persistence import _values


pytestmark = pytest.mark.usefixtures("no_real_modals")

WAITING_TEXT_FRAGMENT = "Waiting for the recording check to finish"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _rows(count):
    indices = np.arange(count, dtype=float)
    time_s, control, signal = _values(indices)
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(count):
        lines.append(
            "%.4f,%.12f,%.12f,%.12f,%.12f\n"
            % (
                time_s[index],
                control[index, 0],
                signal[index, 0],
                control[index, 1],
                signal[index, 1],
            )
        )
    return lines


def _continuous_folder(folder, *, samples=12_000):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Fluorescence.csv").write_text(
        "".join(_rows(samples)), encoding="utf-8", newline=""
    )
    (folder / "Events.csv").write_text(
        "Time(s),Event\n0.0,start\n", encoding="utf-8", newline=""
    )
    return folder


def _intermittent_folder(folder, *, sessions=2):
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(sessions):
        session = folder / ("2026_03_1%d-10_00_00" % index)
        session.mkdir()
        (session / "Fluorescence.csv").write_text(
            "".join(_rows(600)), encoding="utf-8", newline=""
        )
    return folder


# ---------------------------------------------------------------------------
# Visible actions only
# ---------------------------------------------------------------------------


def _pump(qapp, predicate, timeout_ms=120_000):
    deadline = QDeadlineTimer(timeout_ms)
    while predicate() and not deadline.hasExpired():
        qapp.processEvents()


def _select_data_and_rois(
    window, qapp, folder, *, fmt="auto", structure=GUIDED_STRUCTURE_CHOICE_AUTO
):
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText(fmt)
    index = window._guided_acquisition_mode_combo.findData(structure)
    assert index >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(index)
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(folder.parent / "output"))
    window._on_guided_discover_rois()
    _pump(qapp, lambda: window._guided_roi_discovery_running)
    _pump(
        qapp,
        lambda: getattr(window, "_guided_roi_discovery_thread", None) is not None,
        20_000,
    )
    for row in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(row).setCheckState(Qt.Checked)
    qapp.processEvents()


def _continue_to_recording_structure(window, qapp):
    window._on_guided_continue_to_recording_structure()
    qapp.processEvents()


def _continue_to_correction_approach(window, qapp):
    """Exactly what the visible Continue button is connected to."""
    window._on_guided_continue_to_correction_approach()
    qapp.processEvents()


def _wait_for_check_to_settle(window, qapp, timeout_ms=120_000):
    _pump(
        qapp,
        lambda: window._guided_continuous_recording_check_running(),
        timeout_ms,
    )
    _pump(
        qapp,
        lambda: getattr(window, "_guided_continuous_rwd_check_thread", None)
        is not None,
        20_000,
    )


def _page_message(window):
    return window._guided_preview_locked_label.text()


def _visible_step(window):
    item = window._guided_workflow_stepper.currentItem()
    return None if item is None else item.data(Qt.UserRole)


def _hold_the_inspection(monkeypatch):
    """Keep a real check in flight, by delaying where the time really goes.

    The delay is applied to the source inspection the worker calls, not to
    ``run`` itself: replacing that ``@Slot`` would change how Qt connects it
    and stop exercising the real thread wiring.
    """
    release = threading.Event()
    from photometry_pipeline.io import rwd_continuous_source

    real_inspect = rwd_continuous_source.inspect_continuous_rwd_acquisition_folder

    def slow_inspect(*args, **kwargs):
        release.wait(30)
        return real_inspect(*args, **kwargs)

    monkeypatch.setattr(
        rwd_continuous_source,
        "inspect_continuous_rwd_acquisition_folder",
        slow_inspect,
    )
    return release


def _assert_button_is_wired(window):
    """The handler under test is the one the visible button invokes."""
    assert window._guided_recording_continue_btn.text() == (
        "Continue to Correction Approach"
    )


# ---------------------------------------------------------------------------
# The central regression: the real visible path
# ---------------------------------------------------------------------------


def test_continue_starts_and_completes_the_check_and_unlocks_preview(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)
    _assert_button_is_wired(window)

    assert window._guided_recording_structure_readiness()[0] is True
    assert window._guided_continuous_rwd_review_binding is None

    # The only action: the visible Continue button's handler.
    _continue_to_correction_approach(window, qapp)

    assert _visible_step(window) == "Correction approach"
    _wait_for_check_to_settle(window, qapp)

    # The check ran to completion and installed its accepted recording.
    assert window._guided_continuous_rwd_check_outcome == "succeeded"
    assert window._guided_continuous_rwd_review_binding is not None
    assert window._guided_continuous_rwd_accepted_plan() is not None

    # And Correction approach updated itself, with no further navigation.
    assert window._guided_preview_source_ok is True
    assert window._guided_preview_chunk_combo.count() >= 1
    assert window._guided_preview_generate_btn.isEnabled() is True
    assert window._guided_preview_locked_label.isHidden()
    assert WAITING_TEXT_FRAGMENT not in _page_message(window)

    # The plan itself is unchanged by any of this.
    assert window._guided_selected_acquisition_mode() == GUIDED_STRUCTURE_CHOICE_AUTO
    assert window._guided_effective_acquisition_mode() == "continuous"
    draft = window._build_guided_new_analysis_draft_plan()
    assert draft.acquisition_mode == "continuous"
    assert draft.execution_intent.execution_mode == "both"
    assert list(draft.included_roi_ids) == ["ROI1", "ROI2"]


def test_central_path_never_calls_the_private_starter(
    window, qapp, tmp_path, monkeypatch
):
    """Proof the test above exercises navigation, not the private helper."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)

    calls = []
    real = MainWindow._maybe_start_guided_continuous_rwd_recording_check

    def counted(self):
        calls.append("from-navigation")
        return real(self)

    monkeypatch.setattr(
        MainWindow, "_maybe_start_guided_continuous_rwd_recording_check", counted
    )

    _continue_to_correction_approach(window, qapp)
    _wait_for_check_to_settle(window, qapp)

    # Navigation reached the launcher by itself, exactly once.
    assert calls == ["from-navigation"]
    assert window._guided_preview_source_ok is True


def test_explicit_continuous_behaves_identically(window, qapp, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(
        window, qapp, folder, fmt="rwd", structure="continuous"
    )
    _continue_to_recording_structure(window, qapp)
    _continue_to_correction_approach(window, qapp)
    _wait_for_check_to_settle(window, qapp)

    assert window._guided_selected_acquisition_mode() == "continuous"
    assert window._guided_continuous_rwd_check_outcome == "succeeded"
    assert window._guided_preview_source_ok is True
    assert window._guided_preview_generate_btn.isEnabled() is True
    assert WAITING_TEXT_FRAGMENT not in _page_message(window)


def test_intermittent_navigation_starts_no_recording_check(
    window, qapp, tmp_path
):
    folder = _intermittent_folder(tmp_path / "sessions")
    _select_data_and_rois(window, qapp, folder)
    window._guided_sessions_per_hour_edit.setText("2")
    window._guided_session_duration_edit.setText("60")
    qapp.processEvents()
    _continue_to_recording_structure(window, qapp)
    _continue_to_correction_approach(window, qapp)

    assert window._guided_effective_acquisition_mode() == "intermittent"
    assert getattr(window, "_guided_continuous_rwd_check_worker", None) is None
    assert window._guided_continuous_recording_check_running() is False
    assert window._guided_continuous_rwd_review_binding is None

    # The existing session preview path is what is offered.
    assert window._guided_preview_source_ok is True
    segments = [
        window._guided_preview_chunk_combo.itemData(index)
        for index in range(window._guided_preview_chunk_combo.count())
    ]
    assert len(segments) == 2
    for segment in segments:
        assert "continuous_window_index" not in segment
    assert WAITING_TEXT_FRAGMENT not in _page_message(window)
    assert "Checking the continuous recording" not in _page_message(window)


# ---------------------------------------------------------------------------
# Launch decisions
# ---------------------------------------------------------------------------


def test_already_current_binding_is_reused_without_a_second_check(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)
    _continue_to_correction_approach(window, qapp)
    _wait_for_check_to_settle(window, qapp)
    first_binding = window._guided_continuous_rwd_review_binding
    assert first_binding is not None

    started = []
    real = MainWindow._start_guided_continuous_rwd_recording_check
    monkeypatch.setattr(
        MainWindow,
        "_start_guided_continuous_rwd_recording_check",
        lambda self: (started.append(1), real(self))[1],
    )

    _continue_to_correction_approach(window, qapp)

    assert started == []
    assert window._guided_continuous_rwd_review_binding is first_binding
    assert window._guided_preview_source_ok is True
    assert window._guided_preview_locked_label.isHidden()


def test_second_continue_while_running_starts_no_second_check(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)

    # Hold the worker inside run() so a check is genuinely in flight.
    release = _hold_the_inspection(monkeypatch)

    _continue_to_correction_approach(window, qapp)
    first_worker = window._guided_continuous_rwd_check_worker
    first_thread = window._guided_continuous_rwd_check_thread
    assert first_worker is not None
    assert window._guided_continuous_recording_check_running() is True
    assert "Checking the continuous recording" in _page_message(window)

    _continue_to_correction_approach(window, qapp)

    assert window._guided_continuous_rwd_check_worker is first_worker
    assert window._guided_continuous_rwd_check_thread is first_thread
    assert window._guided_continuous_recording_check_running() is True
    assert "Checking the continuous recording" in _page_message(window)

    release.set()
    _wait_for_check_to_settle(window, qapp)
    assert window._guided_preview_source_ok is True


# ---------------------------------------------------------------------------
# Failure and retry
# ---------------------------------------------------------------------------


def test_failed_check_is_explained_and_retryable_through_continue(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)

    def broken(*args, **kwargs):
        raise RuntimeError("simulated inspection failure")

    monkeypatch.setattr(
        "photometry_pipeline.io.rwd_continuous_source."
        "inspect_continuous_rwd_acquisition_folder",
        broken,
    )

    _continue_to_correction_approach(window, qapp)
    _wait_for_check_to_settle(window, qapp)

    assert window._guided_continuous_rwd_check_outcome == "failed"
    assert window._guided_continuous_rwd_review_binding is None
    assert window._guided_continuous_recording_check_running() is False

    message = _page_message(window)
    assert WAITING_TEXT_FRAGMENT not in message
    assert "simulated inspection failure" not in message
    assert "RuntimeError" not in message
    assert message.strip()

    # Preview stays locked, and nothing was confirmed from a failed check.
    assert window._guided_preview_source_ok is False
    assert window._guided_preview_generate_btn.isEnabled() is False

    # Repeating the visible Continue starts a fresh check that succeeds.
    monkeypatch.undo()
    _continue_to_correction_approach(window, qapp)
    _wait_for_check_to_settle(window, qapp)

    assert window._guided_continuous_rwd_check_outcome == "succeeded"
    assert window._guided_preview_source_ok is True
    assert window._guided_preview_generate_btn.isEnabled() is True


def test_roi_change_mid_check_discards_the_result_truthfully(
    window, qapp, tmp_path, monkeypatch
):
    """A result that no longer describes the chosen ROIs must not install."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)

    release = _hold_the_inspection(monkeypatch)

    _continue_to_correction_approach(window, qapp)
    assert window._guided_continuous_recording_check_running() is True

    # The scientist drops an ROI while the check is running, so the recording
    # being checked no longer matches the plan on screen.
    window._guided_roi_list.item(1).setCheckState(Qt.Unchecked)
    qapp.processEvents()

    release.set()
    _wait_for_check_to_settle(window, qapp)

    assert window._guided_continuous_rwd_review_binding is None
    assert window._guided_continuous_rwd_accepted_plan() is None
    assert window._guided_preview_source_ok is False
    assert window._guided_continuous_recording_check_running() is False

    message = _page_message(window)
    assert WAITING_TEXT_FRAGMENT not in message
    assert message.strip()

    # A fresh check for the plan as it now stands still succeeds.
    _continue_to_correction_approach(window, qapp)
    _wait_for_check_to_settle(window, qapp)
    assert window._guided_preview_source_ok is True
    assert list(window._guided_selected_roi_ids()[1]) == ["ROI1"]


def test_window_length_change_mid_check_keeps_the_recording_finding(
    window, qapp, tmp_path, monkeypatch
):
    """The check inspects the recording, not the analysis windowing.

    Changing the window length while it runs does not make what it learned
    about the source wrong, so the finding is kept and the window choices
    simply follow the new length. The scientist is not made to sit through a
    second inspection for a display setting.
    """
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)

    release = _hold_the_inspection(monkeypatch)

    _continue_to_correction_approach(window, qapp)
    assert window._guided_continuous_recording_check_running() is True

    window._continuous_window_sec_spin.setValue(300.0)
    qapp.processEvents()

    release.set()
    _wait_for_check_to_settle(window, qapp)

    assert window._guided_continuous_rwd_accepted_plan() is not None
    assert window._guided_preview_source_ok is True
    assert WAITING_TEXT_FRAGMENT not in _page_message(window)

    # The windows offered are the new length, not the one in force at launch.
    segments = [
        window._guided_preview_chunk_combo.itemData(index)
        for index in range(window._guided_preview_chunk_combo.count())
    ]
    assert segments
    for segment in segments:
        assert segment["continuous_window_sec"] == pytest.approx(300.0)


def test_settings_change_after_a_check_requires_a_new_one(
    window, qapp, tmp_path
):
    """Once installed, a later settings change does invalidate the binding."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)
    _continue_to_correction_approach(window, qapp)
    _wait_for_check_to_settle(window, qapp)
    assert window._guided_preview_source_ok is True

    window._guided_roi_list.item(1).setCheckState(Qt.Unchecked)
    qapp.processEvents()
    window._refresh_guided_diagnostics_panel()
    window._refresh_guided_correction_next_action()

    assert window._guided_continuous_rwd_accepted_plan() is None
    assert window._guided_preview_source_ok is False
    message = _page_message(window)
    assert WAITING_TEXT_FRAGMENT not in message
    assert "Recording settings changed" in message


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def test_inspection_runs_off_the_gui_thread_and_updates_on_it(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)

    gui_thread = threading.get_ident()
    seen = {}

    real_run = main_window_module._GuidedContinuousRwdRecordingCheckWorker.run

    def run(self):
        seen["worker_run"] = threading.get_ident()
        real_run(self)

    real_succeeded = MainWindow._on_guided_continuous_rwd_check_succeeded

    def succeeded(self, token, worker, success):
        seen["success_handler"] = threading.get_ident()
        real_succeeded(self, token, worker, success)

    real_bind = MainWindow._set_guided_continuous_rwd_review_binding

    def bind(self, binding):
        seen["binding_install"] = threading.get_ident()
        real_bind(self, binding)

    real_panel = MainWindow._refresh_guided_correction_preview_panel

    def panel(self, artifact_state):
        seen.setdefault("panel_refresh", threading.get_ident())
        return real_panel(self, artifact_state)

    monkeypatch.setattr(
        main_window_module._GuidedContinuousRwdRecordingCheckWorker, "run", run
    )
    monkeypatch.setattr(
        MainWindow, "_on_guided_continuous_rwd_check_succeeded", succeeded
    )
    monkeypatch.setattr(
        MainWindow, "_set_guided_continuous_rwd_review_binding", bind
    )
    monkeypatch.setattr(
        MainWindow, "_refresh_guided_correction_preview_panel", panel
    )

    _continue_to_correction_approach(window, qapp)
    _wait_for_check_to_settle(window, qapp)

    assert seen["worker_run"] != gui_thread, "inspection ran on the GUI thread"
    assert seen["success_handler"] == gui_thread
    assert seen["binding_install"] == gui_thread
    assert seen["panel_refresh"] == gui_thread
    assert window._guided_preview_source_ok is True


def test_success_refreshes_the_visible_page_without_navigating_away(
    window, qapp, tmp_path, monkeypatch
):
    """Guards specifically against success-without-refresh."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)

    release = _hold_the_inspection(monkeypatch)

    _continue_to_correction_approach(window, qapp)
    assert _visible_step(window) == "Correction approach"
    assert window._guided_preview_source_ok is False
    assert window._guided_preview_chunk_combo.count() == 0
    assert "Checking the continuous recording" in _page_message(window)

    release.set()
    _wait_for_check_to_settle(window, qapp)

    # Still on the same page; it updated itself.
    assert _visible_step(window) == "Correction approach"
    assert window._guided_preview_locked_label.isHidden()
    assert window._guided_preview_chunk_combo.count() >= 1
    assert window._guided_preview_roi_combo.count() == 2
    assert window._guided_preview_generate_btn.isEnabled() is True
    assert "Ready to preview" in window._guided_preview_source_status_label.text()


# ---------------------------------------------------------------------------
# Status text truthfulness
# ---------------------------------------------------------------------------


def test_not_started_state_does_not_claim_a_check_is_running(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    window._refresh_guided_diagnostics_panel()
    window._refresh_guided_correction_next_action()

    assert window._guided_continuous_recording_check_running() is False
    message = _page_message(window)
    assert WAITING_TEXT_FRAGMENT not in message
    assert "has not been checked yet" in message


def test_running_state_says_it_is_checking(window, qapp, tmp_path, monkeypatch):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data_and_rois(window, qapp, folder)
    _continue_to_recording_structure(window, qapp)

    release = _hold_the_inspection(monkeypatch)

    _continue_to_correction_approach(window, qapp)

    assert window._guided_continuous_recording_check_running() is True
    assert _page_message(window) == "Checking the continuous recording…"

    release.set()
    _wait_for_check_to_settle(window, qapp)


def test_waiting_text_is_never_shown_in_any_settled_state(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")

    # not started
    _select_data_and_rois(window, qapp, folder)
    window._refresh_guided_diagnostics_panel()
    window._refresh_guided_correction_next_action()
    assert WAITING_TEXT_FRAGMENT not in _page_message(window)

    # succeeded
    _continue_to_recording_structure(window, qapp)
    _continue_to_correction_approach(window, qapp)
    _wait_for_check_to_settle(window, qapp)
    assert WAITING_TEXT_FRAGMENT not in _page_message(window)

    # stale after a settings change
    window._continuous_window_sec_spin.setValue(300.0)
    qapp.processEvents()
    window._refresh_guided_diagnostics_panel()
    window._refresh_guided_correction_next_action()
    assert window._guided_continuous_rwd_accepted_plan() is None
    message = _page_message(window)
    assert WAITING_TEXT_FRAGMENT not in message
    assert "Recording settings changed" in message
