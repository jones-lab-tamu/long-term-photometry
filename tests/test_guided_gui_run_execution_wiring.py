from __future__ import annotations

import inspect
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

import photometry_pipeline.guided_backend_execution as backend
import photometry_pipeline.guided_startup_claim as claim
from photometry_pipeline.guided_startup_transaction import (
    GuidedStartupTransactionRequest,
)
from gui.main_window import MainWindow, _GuidedRunExecutionWorker
from gui.run_report_parser import classify_completed_run_candidate
from tests.test_gui_guided_backend_validation_context import (
    _accepted_outcome,
    _failure_outcome,
)
from tests.test_guided_startup_allocation import allocation_case
from tests.test_guided_startup_orchestration_real_wrapper_boundary import (
    _real_wrapper_runner,
)
from tests.test_guided_startup_transaction import startup_request


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


def _pump_until(qapp, condition, *, timeout_s: float = 5.0) -> None:
    """Process GUI-thread events until `condition()` is true.

    Guided Run now executes on a worker thread; its result reaches the GUI
    thread via a queued signal, which is only delivered when the GUI event
    loop is pumped. Tests must pump explicitly since pytest does not run
    `QApplication.exec()`.
    """
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met before timeout")
        qapp.processEvents()


def _set_ready(window, request):
    window._guided_backend_validation_revision = request.current_guided_revision
    window._guided_backend_validation_outcome = _accepted_outcome()
    window._guided_backend_validation_outcome_revision = (
        request.current_guided_revision
    )
    window._guided_run_authorization_result = request.authorization_result
    window._guided_execution_payload_result = request.payload_result
    window._guided_startup_transaction_request = request
    window._guided_backend_execution_result = None
    window._refresh_guided_run_readiness_display()


def _result(status: str, summary: str):
    completed = status == "wrapper_completed_needs_review_loading"
    running = status == "wrapper_running"
    return backend.GuidedBackendExecutionResult(
        status=status,
        ok=running or completed,
        user_visible_state=(
            "running"
            if running
            else (
                "run_finished_review_required"
                if completed
                else "failed_during_run"
            )
        ),
        user_summary=summary,
        run_directory=r"C:\output\guided-run",
        completed_run_candidate_path=(
            r"C:\output\guided-run" if completed else None
        ),
        requires_completed_run_loader_validation=completed,
        wrapper_started=True,
        wrapper_completed=completed or status == "wrapper_failed",
        blocking_issues=(),
        diagnostics=backend.GuidedBackendExecutionDiagnostics(
            orchestration_status="test",
            pure_plan_status=None,
            allocation_status=None,
            materialization_status=None,
            wrapper_started=True,
            wrapper_completed=completed,
            wrapper_returncode=0 if completed else None,
            failure_marker_path=None,
            startup_transaction_identity=None,
            wrapper_command=None,
        ),
    )


def _visible_text(window):
    return " ".join(
        (
            window._guided_run_btn.text(),
            window._guided_run_btn.toolTip(),
            window._guided_run_readiness_label.text(),
        )
    ).lower()


def _run_production_validation_update(
    window,
    request,
    monkeypatch,
    *,
    outcome=None,
):
    window._guided_backend_validation_revision = request.current_guided_revision
    context = SimpleNamespace(revision=request.current_guided_revision)
    accepted = outcome or _accepted_outcome()
    monkeypatch.setattr(
        window,
        "_capture_guided_backend_validation_context",
        lambda: context,
    )
    monkeypatch.setattr(
        window,
        "_run_guided_backend_validation_workflow",
        lambda _context: accepted,
    )
    monkeypatch.setattr(
        window,
        "_derive_guided_execution_state_from_validation",
        lambda _context, _outcome: (
            request.authorization_result,
            request.payload_result,
            request,
        ),
    )
    window._on_guided_backend_validate_clicked()
    return accepted


def test_production_validation_update_retains_real_bound_request_and_enables(
    window, startup_request, monkeypatch
):
    _run_production_validation_update(window, startup_request, monkeypatch)
    assert window._guided_run_authorization_result is (
        startup_request.authorization_result
    )
    assert window._guided_execution_payload_result is (
        startup_request.payload_result
    )
    assert window._guided_startup_transaction_request is startup_request
    assert isinstance(
        window._guided_startup_transaction_request,
        GuidedStartupTransactionRequest,
    )
    assert window._guided_backend_execution_result is None
    assert window._guided_run_btn.isEnabled() is True


def test_production_derivation_builds_real_exactly_bound_request(
    window, startup_request, monkeypatch
):
    import photometry_pipeline.guided_execution_request_builder as request_builder

    revision = startup_request.current_guided_revision
    window._guided_backend_validation_revision = revision
    context = SimpleNamespace(revision=revision)
    outcome = _accepted_outcome()
    built = request_builder.GuidedExecutionRequestBuildResult(
        status="built",
        ok=True,
        authorization_result=startup_request.authorization_result,
        payload_result=startup_request.payload_result,
        startup_transaction_request=startup_request,
        blocking_issues=(),
        current_gui_revision=revision,
        request_ready=True,
    )
    monkeypatch.setattr(
        request_builder,
        "build_guided_startup_request_from_validation",
        lambda **_kwargs: built,
    )
    state = window._derive_guided_execution_state_from_validation(
        context, outcome
    )
    assert state is not None
    authorized, derived, request = state
    assert isinstance(request, GuidedStartupTransactionRequest)
    assert request.current_guided_revision == revision
    assert request.authorization_result is authorized
    assert request.payload_result is derived
    assert authorized is startup_request.authorization_result
    assert derived is startup_request.payload_result


def test_production_validation_state_executes_retained_request_once(
    window, startup_request, monkeypatch, qapp
):
    _run_production_validation_update(window, startup_request, monkeypatch)
    calls = []
    running = _result("wrapper_running", "Guided Run is running.")
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: calls.append(request) or running,
    )
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: bool(calls))
    assert calls == [startup_request]
    assert isinstance(calls[0], GuidedStartupTransactionRequest)


def test_production_retained_state_is_cleared_on_invalidation(
    window, startup_request, monkeypatch
):
    _run_production_validation_update(window, startup_request, monkeypatch)
    window._invalidate_guided_backend_validation("input changed")
    assert window._guided_run_authorization_result is None
    assert window._guided_execution_payload_result is None
    assert window._guided_startup_transaction_request is None
    assert window._guided_backend_execution_result is None
    assert window._guided_run_btn.isEnabled() is False


def test_validation_start_and_refusal_clear_retained_state(
    window, startup_request, monkeypatch
):
    _run_production_validation_update(window, startup_request, monkeypatch)
    context = SimpleNamespace(revision=startup_request.current_guided_revision)
    refused = _failure_outcome("validator_refused")
    observed_cleared = []

    def run_refused(_context):
        observed_cleared.append(
            (
                window._guided_run_authorization_result,
                window._guided_execution_payload_result,
                window._guided_startup_transaction_request,
            )
        )
        return refused

    monkeypatch.setattr(
        window,
        "_capture_guided_backend_validation_context",
        lambda: context,
    )
    monkeypatch.setattr(
        window, "_run_guided_backend_validation_workflow", run_refused
    )
    window._on_guided_backend_validate_clicked()
    assert observed_cleared == [(None, None, None)]
    assert window._guided_run_authorization_result is None
    assert window._guided_execution_payload_result is None
    assert window._guided_startup_transaction_request is None
    assert window._guided_run_btn.isEnabled() is False


def test_ready_click_calls_backend_seam_once_with_real_request(
    window, startup_request, monkeypatch, qapp
):
    calls = []
    expected = _result("wrapper_running", "Guided Run is running.")

    def execute(*, request, runner=None):
        calls.append(request)
        return expected

    monkeypatch.setattr(backend, "execute_guided_backend_run", execute)
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    # Immediately after click: control has returned, worker is dispatched.
    assert window._guided_backend_execution_active is True
    assert window._guided_run_btn.isEnabled() is False
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)
    assert calls == [startup_request]
    assert isinstance(calls[0], GuidedStartupTransactionRequest)
    assert window._guided_backend_execution_result is expected
    assert window._guided_run_readiness_label.text() == "Guided Run has started."
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_backend_execution_active is False


def test_run_click_starts_worker_and_returns_control_before_completion(
    window, startup_request, monkeypatch, qapp
):
    """Proves clicking Run no longer blocks waiting for the backend call.

    A controllable fake backend blocks on a threading.Event until the test
    releases it. If execution were still synchronous on the GUI thread,
    `.click()` itself would hang here (and the test would time out via
    Event.wait). Instead, control returns immediately with the worker still
    in flight, and only after `release.set()` + pumping does the final
    result appear.
    """
    release = threading.Event()
    calls = []

    def slow_execute(*, request, runner=None):
        calls.append(request)
        assert release.wait(timeout=5), "release was never set"
        return _result("wrapper_running", "Guided Run is running.")

    monkeypatch.setattr(backend, "execute_guided_backend_run", slow_execute)
    _set_ready(window, startup_request)

    window._guided_run_btn.click()

    # Control has returned to the caller; the worker is still blocked on
    # `release`, so no result has been stored yet.
    assert window._guided_backend_execution_active is True
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_backend_execution_result is None
    assert "running" in window._guided_run_readiness_label.text().lower()

    release.set()
    _pump_until(qapp, lambda: window._guided_run_execution_thread is None)
    assert calls == [startup_request]
    assert window._guided_backend_execution_active is False
    assert window._guided_backend_execution_result is not None
    assert window._guided_run_execution_worker is None


def test_double_click_does_not_start_second_worker(
    window, startup_request, monkeypatch, qapp
):
    release = threading.Event()
    calls = []

    def slow_execute(*, request, runner=None):
        calls.append(request)
        assert release.wait(timeout=5), "release was never set"
        return _result("wrapper_running", "Guided Run is running.")

    monkeypatch.setattr(backend, "execute_guided_backend_run", slow_execute)
    _set_ready(window, startup_request)

    window._guided_run_btn.click()
    assert window._guided_run_btn.isEnabled() is False
    # A second click while disabled must be a no-op (Qt itself refuses to
    # click a disabled button); also exercise the handler directly in case
    # something invokes it while active regardless of button state.
    window._guided_run_btn.click()
    window._on_guided_run_clicked_backend_guarded()

    release.set()
    _pump_until(qapp, lambda: window._guided_run_execution_thread is None)
    assert len(calls) == 1


def test_worker_internal_exception_shows_internal_error_and_recovers(
    window, startup_request, monkeypatch, qapp
):
    def raising_execute(*, request, runner=None):
        raise RuntimeError("simulated unexpected failure")

    monkeypatch.setattr(backend, "execute_guided_backend_run", raising_execute)
    _set_ready(window, startup_request)
    validate_btn = window._guided_backend_validate_btn

    window._guided_run_btn.click()
    assert validate_btn.isEnabled() is False

    _pump_until(qapp, lambda: window._guided_run_execution_thread is None)
    assert "internal error" in window._guided_run_readiness_label.text().lower()
    assert window._guided_backend_execution_active is False
    assert window._guided_backend_execution_result is None
    assert validate_btn.isEnabled() is True
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is False
    assert window._guided_run_execution_worker is None


def test_click_refuses_if_readiness_becomes_stale_after_enablement(
    window, startup_request, monkeypatch
):
    _run_production_validation_update(window, startup_request, monkeypatch)
    assert window._guided_run_btn.isEnabled()
    window._guided_backend_validation_revision += 1
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda **_kwargs: pytest.fail("backend called for stale readiness"),
    )
    window._guided_run_btn.click()
    assert window._guided_run_readiness.status == "validation_stale"
    assert window._guided_run_btn.isEnabled() is False
    assert "Validate again" in window._guided_run_readiness_label.text()
    # Stale revision is caught before any worker is launched.
    assert window._guided_backend_execution_active is False
    assert window._guided_run_execution_thread is None


@pytest.mark.parametrize(
    ("result", "expected_text"),
    (
        (
            _result(
                "refused_before_startup",
                "Guided Run could not start because the validated setup is "
                "no longer current.",
            ),
            "Guided Run could not start because the validated setup is no "
            "longer current.",
        ),
        (
            _result(
                "startup_allocation_failed",
                "Guided Run could not create a safe output folder.",
            ),
            "Guided Run could not create a safe output folder.",
        ),
        (
            _result(
                "startup_status_write_failed",
                "Guided Run created an output folder but could not write its "
                "startup status.",
            ),
            "Guided Run created an output folder but could not write its "
            "startup status.",
        ),
        (
            _result(
                "startup_materialization_failed",
                "Guided Run could not prepare the internal run files.",
            ),
            "Guided Run could not prepare the internal run files.",
        ),
        (
            _result(
                "wrapper_start_failed",
                "Guided Run prepared the output folder but could not start "
                "the analysis.",
            ),
            "Guided Run prepared the output folder but could not start the "
            "analysis.",
        ),
        (
            _result(
                "wrapper_failed",
                "Guided Run started, but the analysis reported an error.",
            ),
            "Guided Run started, but the analysis reported an error.",
        ),
        (
            _result(
                "wrapper_completed_needs_review_loading",
                "Guided Run finished. Load the completed run for review.",
            ),
            "Guided Run finished. Load the completed run for review.",
        ),
    ),
)
def test_backend_result_maps_to_safe_text_and_is_stored(
    window, startup_request, monkeypatch, result, expected_text, qapp
):
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: result,
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)
    assert window._guided_backend_execution_result is result
    assert window._guided_run_readiness_label.text() == expected_text
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_backend_execution_active is False
    # A failed/refused status is a real, structured result, not an internal
    # error, so it must not show the completed-run handoff (no false success).
    if result.status != "wrapper_completed_needs_review_loading":
        handoff_btn = window._guided_load_completed_run_for_review_btn
        assert handoff_btn.isVisible() is False


def test_completed_result_does_not_auto_load_or_claim_success(
    window, startup_request, monkeypatch, qapp
):
    completed = _result(
        "wrapper_completed_needs_review_loading",
        "Guided Run finished. Load the completed run for review.",
    )
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: completed,
    )
    monkeypatch.setattr(
        window,
        "_open_completed_results_dir",
        lambda *_args, **_kwargs: pytest.fail("Review auto-loaded"),
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_run_execution_thread is None)
    assert completed.requires_completed_run_loader_validation is True
    assert completed.completed_run_claim is False
    assert window._guided_backend_execution_result is completed
    assert window._guided_backend_execution_active is False
    # Async completion must still expose the handoff, unchanged from the
    # synchronous contract. (isVisible() is not checked here: the top-level
    # window is never shown in this test, so Qt reports all children as not
    # visible regardless of setVisible(True); isEnabled() is unaffected by
    # that and reflects the same "ready" gate.)
    assert window._guided_load_completed_run_for_review_btn.isEnabled() is True
    # Thread/worker cleanup must leave no dangling reference.
    assert window._guided_run_execution_worker is None


def test_fake_backend_click_writes_no_files(
    window, startup_request, tmp_path, monkeypatch, qapp
):
    running = _result("wrapper_running", "Guided Run is running.")
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: running,
    )
    _set_ready(window, startup_request)
    before = tuple(tmp_path.iterdir())

    def fail(*_args, **_kwargs):
        raise AssertionError("filesystem write attempted")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)
    assert tuple(tmp_path.iterdir()) == before


def test_execution_text_excludes_internal_terms(
    window, startup_request, monkeypatch, qapp
):
    result = _result(
        "wrapper_failed",
        "Guided Run started, but the analysis reported an error.",
    )
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: result,
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)
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
    assert not any(term in _visible_text(window) for term in prohibited)


def test_full_control_is_unchanged_by_guided_execution(
    window, startup_request, monkeypatch, qapp
):
    before = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: _result(
            "wrapper_running", "Guided Run is running."
        ),
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)
    after = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    assert after == before


def test_gui_execution_methods_use_only_backend_adapter():
    source = (
        inspect.getsource(MainWindow._on_guided_run_clicked_backend_guarded)
        + inspect.getsource(MainWindow._start_guided_run_execution_worker)
        + inspect.getsource(_GuidedRunExecutionWorker.run)
    )
    assert "execute_guided_backend_run" in source
    prohibited = (
        "run_guided_startup_to_wrapper",
        "allocate_guided_startup_directory",
        "materialize_guided_startup_artifacts",
        "run_full_pipeline_deliverables",
        "subprocess",
        "Pipeline",
    )
    assert not any(name in source for name in prohibited)


def test_guided_run_execution_moves_off_gui_thread_via_worker():
    """Guards the async-threading contract at the source level.

    The GUI click handler must not call the backend seam directly anymore;
    it must dispatch through a QThread-backed worker so the GUI event loop
    is never blocked for the run's duration.
    """
    handler_source = inspect.getsource(
        MainWindow._on_guided_run_clicked_backend_guarded
    )
    assert "execute_guided_backend_run" not in handler_source
    assert "_start_guided_run_execution_worker" in handler_source
    worker_launch_source = inspect.getsource(
        MainWindow._start_guided_run_execution_worker
    )
    assert "QThread" in worker_launch_source
    assert "moveToThread" in worker_launch_source


def test_run_worker_does_not_reference_main_window():
    """The worker must hold no MainWindow reference and touch no GUI state.

    It receives only plain, already-captured values (`request`, `runner`)
    and calls only the module-level `execute_guided_backend_run` function.
    """
    init_params = list(
        inspect.signature(_GuidedRunExecutionWorker.__init__).parameters
    )
    assert init_params == ["self", "request", "runner"]

    run_source = inspect.getsource(_GuidedRunExecutionWorker.run)
    assert "execute_guided_backend_run(" in run_source
    prohibited = (
        "MainWindow",
        "self._window",
        "self.window",
        "_execute_guided_backend_run_for_gui",
        "_guided_backend_execution_runner",
        "_guided_run_btn",
        "_guided_run_readiness_label",
        "_guided_backend_execution_active",
    )
    assert not any(term in run_source for term in prohibited)


def test_runner_override_captured_on_gui_thread_and_used_inside_worker(
    window, startup_request, monkeypatch, qapp
):
    """Proves the runner override crosses the request/runner boundary intact.

    `window._guided_backend_execution_runner` (set here, on the GUI thread)
    must reach `execute_guided_backend_run`'s `runner=` kwarg exactly, and
    that call must happen on a thread other than the GUI thread that set it.
    """
    marker_runner = object()
    captured = {}
    gui_thread_id = threading.get_ident()

    def fake_execute(*, request, runner=None):
        captured["request"] = request
        captured["runner"] = runner
        captured["thread_id"] = threading.get_ident()
        return _result("wrapper_running", "Guided Run is running.")

    monkeypatch.setattr(backend, "execute_guided_backend_run", fake_execute)
    window._guided_backend_execution_runner = marker_runner
    _set_ready(window, startup_request)

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_run_execution_thread is None)

    assert captured["request"] is startup_request
    assert captured["runner"] is marker_runner
    assert captured["thread_id"] != gui_thread_id


def test_gui_validation_derivation_uses_only_backend_builder():
    source = inspect.getsource(
        MainWindow._derive_guided_execution_state_from_validation
    )
    assert "build_guided_startup_request_from_validation" in source
    prohibited = (
        "resolve_application_build_identity",
        "authorize_guided_run",
        "derive_guided_execution_payloads",
        "GuidedStartupTransactionRequest",
        "GuidedStartupFilesystemPolicy",
        "GuidedWrapperEntrypointIdentity",
        "detect_guided_diagnostic_cache_candidate",
        "hashlib",
        "token_urlsafe",
    )
    assert not any(name in source for name in prohibited)


def test_real_backend_reaches_initial_status_boundary_only(
    window, allocation_case, monkeypatch, qapp
):
    request, _plan = allocation_case
    runner, calls = _real_wrapper_runner(monkeypatch)
    _set_ready(window, request)
    window._guided_backend_execution_runner = runner
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)

    result = window._guided_backend_execution_result
    run_dir = Path(request.planned_allocated_run_dir)
    status = json.loads((run_dir / "status.json").read_bytes())
    assert result.status == "wrapper_running"
    assert result.completed_run_claim is False
    assert result.completed_run_candidate_path is None
    assert window._guided_run_readiness_label.text() == "Guided Run has started."
    assert calls == {"live_verify": 1, "analysis": 0, "root_makedirs": 0}
    assert (
        run_dir / claim.GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME
    ).is_file()
    assert status["phase"] == "initializing"
    assert status["status"] == "running"
    for prohibited in (
        "MANIFEST.json",
        "run_report.json",
        "qc",
        "cache",
        "events",
        "figures",
        "events.ndjson",
    ):
        assert not (run_dir / prohibited).exists()
    assert classify_completed_run_candidate(str(run_dir))[0] is False


def test_close_event_refused_while_guided_run_active(window, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *args, **kwargs: shown.append(args)),
    )
    window._guided_backend_execution_active = True
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is False
    assert shown, "expected a plain message to be shown"
    # Message must be plain and actionable, no cancellation/progress/log
    # language implied.
    message_text = str(shown[0][2])
    assert "still running" in message_text.lower()
    assert "wait" in message_text.lower()


def test_close_event_allowed_when_guided_run_not_active(window):
    window._guided_backend_execution_active = False
    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is True
