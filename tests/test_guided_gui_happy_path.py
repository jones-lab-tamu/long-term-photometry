from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from gui.run_report_parser import classify_completed_run_candidate
from tests.test_guided_gui_run_completed_boundary import _completion_runner
from tests.test_guided_gui_run_execution_wiring import (
    _pump_until,
    _run_production_validation_update,
)
from tests.test_guided_startup_allocation import allocation_case


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    # Defensive cleanup: a failing test must never leave the close guard
    # active and block teardown on a real (unmocked) QMessageBox dialog.
    instance._guided_backend_execution_active = False
    thread = getattr(instance, "_guided_run_execution_thread", None)
    if thread is not None and thread.isRunning():
        thread.quit()
        thread.wait(2000)
    instance.close()
    instance.deleteLater()


def test_guided_validate_run_and_loader_gated_review_handoff_happy_path(
    window, allocation_case, monkeypatch, qapp
):
    request, _plan = allocation_case
    full_control_before = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    auto_load_calls = []
    monkeypatch.setattr(
        window,
        "_open_completed_results_dir",
        lambda path: auto_load_calls.append(path) or True,
    )

    _run_production_validation_update(window, request, monkeypatch)
    assert window._guided_run_btn.isEnabled() is True
    assert (
        window._guided_load_completed_run_for_review_btn.isHidden() is True
    )
    assert (
        window._guided_load_completed_run_for_review_btn.isEnabled() is False
    )

    runner, runner_calls = _completion_runner(monkeypatch)
    window._guided_backend_execution_runner = runner
    window._guided_run_btn.click()

    # Guided Run executes on a worker thread: control returns immediately
    # with the running guard active, and the final result only reaches the
    # GUI thread once the event loop is pumped. Asserting the result right
    # after the click would race the worker (stale synchronous assumption).
    assert window._guided_backend_execution_active is True
    assert window._guided_run_btn.isEnabled() is False
    _pump_until(
        qapp,
        lambda: window._guided_run_execution_thread is None,
        timeout_s=60.0,
    )
    assert window._guided_backend_execution_active is False

    result = window._guided_backend_execution_result
    run_dir = Path(request.planned_allocated_run_dir)
    assert result is not None
    assert result.status == "wrapper_completed_needs_review_loading"
    assert result.completed_run_candidate_path == str(run_dir)
    assert result.requires_completed_run_loader_validation is True
    assert result.completed_run_claim is False
    assert window._guided_run_readiness_label.text() == (
        "Guided Run finished. Load the completed run for review."
    )
    assert (
        window._guided_load_completed_run_for_review_btn.isHidden() is False
    )
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is True
    assert auto_load_calls == []
    assert runner_calls["prepared_validation"] >= 2
    assert runner_calls["live_verification"] == 1
    assert runner_calls["analysis_stub"] == 1
    assert classify_completed_run_candidate(str(run_dir))[0] is True

    call_order = []
    original_classifier = window._is_openable_completed_results_dir

    def classify(path):
        accepted = original_classifier(path)
        call_order.append(("classify", path, accepted[0]))
        return accepted

    def open_results(path):
        call_order.append(("open", path, True))
        return True

    monkeypatch.setattr(
        window, "_is_openable_completed_results_dir", classify
    )
    monkeypatch.setattr(
        window, "_open_completed_results_dir", open_results
    )
    window._guided_load_completed_run_for_review_btn.click()

    assert call_order == [
        ("classify", str(run_dir), True),
        ("open", str(run_dir), True),
    ]
    assert window._guided_workflow_stepper.currentRow() == (
        list(GUIDED_WORKFLOW_STEPS).index("Review")
    )
    assert window._guided_workflow_mode == "open_results"
    assert window._guided_run_readiness_label.text() == (
        "Completed run loaded for review."
    )
    assert result.completed_run_claim is False

    visible_text = " ".join(
        (
            window._guided_run_btn.text(),
            window._guided_run_btn.toolTip(),
            window._guided_load_completed_run_for_review_btn.text(),
            window._guided_load_completed_run_for_review_btn.toolTip(),
            window._guided_run_readiness_label.text(),
        )
    ).lower()
    assert str(run_dir).lower() not in visible_text
    internal_terms = (
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
    assert not any(term in visible_text for term in internal_terms)
    full_control_after = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    assert full_control_after == full_control_before
