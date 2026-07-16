from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from time import monotonic

import pytest
from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication

from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from tests.terminal_run_fixtures import write_current_run
from tests.test_guided_gui_run_execution_wiring import _result


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


def _completed_candidate(tmp_path: Path) -> Path:
    # A candidate the loader accepts must present the whole verified terminal
    # set, not merely a success status.
    return write_current_run(tmp_path / "completed", region="Region0")


def _set_result(window, result) -> None:
    window._guided_backend_execution_result = result
    window._refresh_guided_review_handoff_display()


def _wait_for_review_loader(qapp, window, timeout=3.0):
    deadline = monotonic() + timeout
    while window._guided_completed_review_loading and monotonic() < deadline:
        qapp.processEvents()
    qapp.processEvents()
    assert window._guided_completed_review_loading is False


def test_handoff_control_is_hidden_before_run(window):
    button = window._guided_load_completed_run_for_review_btn
    assert button.text() == "Load completed run for review"
    assert button.isVisible() is False
    assert button.isEnabled() is False
    assert window._guided_workflow_tab.isAncestorOf(button)
    assert not window._full_control_tab.isAncestorOf(button)


@pytest.mark.parametrize(
    "status",
    (
        "refused_before_startup",
        "startup_allocation_failed",
        "wrapper_running",
        "wrapper_failed",
    ),
)
def test_noncompleted_results_do_not_enable_handoff(window, status):
    result = _result(status, "Guided Run could not continue.")
    _set_result(window, result)
    assert window._guided_load_completed_run_for_review_btn.isVisible() is False
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is False


def test_completed_candidate_enables_handoff_without_auto_loading(
    window, tmp_path, monkeypatch
):
    run_dir = _completed_candidate(tmp_path)
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    monkeypatch.setattr(
        window,
        "_open_completed_results_dir",
        lambda *_args, **_kwargs: pytest.fail("Review auto-loaded"),
    )
    _set_result(window, completed)
    button = window._guided_load_completed_run_for_review_btn
    assert button.isHidden() is False
    assert button.isEnabled() is True
    assert completed.completed_run_claim is False


def test_accepted_candidate_is_classified_then_opened_once(
    qapp, window, tmp_path, monkeypatch
):
    run_dir = _completed_candidate(tmp_path)
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    import gui.main_window as main_window_module

    worker_calls = []
    overview = {
        "run_dir": str(run_dir),
        "terminal_state": "success",
        "analysis_branches": ["phasic"],
        "included_rois": ["Region0"],
        "requested_by_roi": {},
    }
    monkeypatch.setattr(
        main_window_module,
        "load_completed_review_overview",
        lambda path: worker_calls.append(path) or overview,
    )
    opened = []
    monkeypatch.setattr(
        window._guided_report_viewer,
        "load_report",
        lambda path, *, review_overview=None: (
            opened.append((path, review_overview)) or True
        ),
    )
    _set_result(window, completed)
    window._guided_load_completed_run_for_review_btn.click()
    _wait_for_review_loader(qapp, window)
    assert worker_calls == [str(run_dir)]
    assert opened == [(str(run_dir), overview)]
    assert window._guided_workflow_stepper.currentRow() == (
        list(GUIDED_WORKFLOW_STEPS).index("Review")
    )
    assert window._guided_run_readiness_label.text() == (
        "Completed run loaded for review."
    )


def test_incomplete_candidate_is_rejected_without_review_transition(
    qapp, window, tmp_path, monkeypatch
):
    run_dir = tmp_path / "startup-stop"
    (run_dir / "Region0" / "summary").mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {"schema_version": 1, "phase": "running", "status": "running"}
        ),
        encoding="utf-8",
    )
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    before = window._guided_workflow_stepper.currentRow()
    monkeypatch.setattr(
        window,
        "_open_completed_results_dir",
        lambda *_args, **_kwargs: pytest.fail("Rejected run was opened"),
    )
    _set_result(window, completed)
    window._guided_load_completed_run_for_review_btn.click()
    _wait_for_review_loader(qapp, window)
    assert window._guided_workflow_stepper.currentRow() == before
    assert window._guided_run_readiness_label.text() == (
        "The completed analysis could not be opened for review. "
        "Check that the completed output folder is still available "
        "and try again."
    )


def test_completed_review_worker_keeps_gui_responsive_and_blocks_duplicates(
    qapp, window, tmp_path, monkeypatch
):
    run_dir = _completed_candidate(tmp_path)
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    import gui.main_window as main_window_module

    worker_threads = []
    calls = []

    def slow_overview(path):
        calls.append(path)
        worker_threads.append(QThread.currentThread())
        time.sleep(0.15)
        return {
            "run_dir": path,
            "terminal_state": "success",
            "analysis_branches": ["phasic"],
            "included_rois": ["Region0"],
            "requested_by_roi": {},
        }

    monkeypatch.setattr(
        main_window_module, "load_completed_review_overview", slow_overview
    )
    gui_slots = []
    # Signals are already connected to the original bound slot, so thread
    # identity is also asserted through the viewer mutation below.
    monkeypatch.setattr(
        window._guided_report_viewer,
        "load_report",
        lambda *_args, **_kwargs: gui_slots.append(QThread.currentThread()) or True,
    )
    ticks = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(monotonic()))
    timer.start()

    _set_result(window, completed)
    window._guided_load_completed_run_for_review_btn.click()
    assert window._guided_completed_review_loading is True
    assert window._guided_completed_review_load_progress.isHidden() is False
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is False
    window._on_guided_load_completed_run_for_review_clicked()
    _wait_for_review_loader(qapp, window)
    timer.stop()

    assert calls == [str(run_dir)]
    assert len(ticks) >= 2
    assert worker_threads and worker_threads[0] is not qapp.thread()
    assert gui_slots == [qapp.thread()]
    assert window._guided_completed_review_load_progress.isHidden() is True


def test_completed_review_failure_is_sanitized_and_restores_controls(
    qapp, window, tmp_path, monkeypatch
):
    run_dir = _completed_candidate(tmp_path)
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    import gui.main_window as main_window_module

    raw = r"ValueError: internal C:\secret\trace_cache.h5 exploded"
    failure_worker_threads = []

    def fail_in_worker(_path):
        failure_worker_threads.append(QThread.currentThread())
        raise ValueError(raw)

    monkeypatch.setattr(
        main_window_module,
        "load_completed_review_overview",
        fail_in_worker,
    )
    logged = []
    failure_gui_threads = []

    def capture_log(message):
        failure_gui_threads.append(QThread.currentThread())
        logged.append(message)

    monkeypatch.setattr(window, "_append_log", capture_log)
    _set_result(window, completed)
    window._guided_load_completed_run_for_review_btn.click()
    _wait_for_review_loader(qapp, window)

    visible = window._guided_run_readiness_label.text()
    assert "secret" not in visible
    assert "trace_cache" not in visible
    assert "could not be opened for review" in visible
    assert any("secret" in line and "trace_cache" in line for line in logged)
    assert len(failure_worker_threads) == 1
    assert failure_worker_threads[0] is not qapp.thread()
    assert failure_gui_threads == [qapp.thread()]
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is True
    assert window._guided_report_viewer._current_run_dir == ""


def test_backend_status_without_candidate_path_refuses(window):
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        completed_run_candidate_path=None,
    )
    _set_result(window, completed)
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is False
    window._on_guided_load_completed_run_for_review_clicked()
    assert window._guided_run_readiness_label.text() == (
        "Guided Run did not provide a completed-run candidate."
    )


def test_invalidation_clears_and_hides_handoff(window, tmp_path):
    run_dir = _completed_candidate(tmp_path)
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    _set_result(window, completed)
    assert window._guided_load_completed_run_for_review_btn.isEnabled()
    window._invalidate_guided_backend_validation("input changed")
    assert window._guided_backend_execution_result is None
    assert window._guided_load_completed_run_for_review_btn.isVisible() is False
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is False


def test_handoff_text_is_safe_and_full_control_unchanged(
    window, tmp_path, monkeypatch
):
    run_dir = _completed_candidate(tmp_path)
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    full_control_before = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    monkeypatch.setattr(
        window, "_open_completed_results_dir", lambda _path: True
    )
    _set_result(window, completed)
    window._guided_load_completed_run_for_review_btn.click()
    full_control_after = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    assert full_control_after == full_control_before
    visible = " ".join(
        (
            window._guided_load_completed_run_for_review_btn.text(),
            window._guided_load_completed_run_for_review_btn.toolTip(),
            window._guided_run_readiness_label.text(),
        )
    ).lower()
    assert str(run_dir).lower() not in visible
    prohibited = (
        "manifest",
        "preallocated",
        "command_invoked",
        "wrapper claim",
        "startup transaction",
        "hash",
        "--guided",
        "config_effective.yaml",
        "runner_request",
        "startup_transaction_unavailable",
        "guided_candidate_manifest",
        "guided_startup",
        "wrapper_claim",
        "backend adapter",
        "orchestration",
        "subprocess",
        "raw command",
    )
    assert not any(term in visible for term in prohibited)
