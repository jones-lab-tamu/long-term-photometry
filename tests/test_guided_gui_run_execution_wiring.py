from __future__ import annotations

import inspect
from dataclasses import replace
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
    from photometry_pipeline.guided_plan_identity import (
        compute_guided_new_analysis_draft_plan_identity,
    )

    window._guided_backend_validation_revision = request.current_guided_revision
    window._guided_backend_validation_outcome = replace(
        _accepted_outcome(),
        request_identity=request.startup_authority.rwd.stored_request_identity,
    )
    window._guided_backend_validation_outcome_revision = (
        request.current_guided_revision
    )
    window._guided_startup_authority = request.startup_authority
    window._guided_execution_payload_result = request.payload_result
    window._guided_startup_transaction_request = request
    window._guided_backend_execution_result = None
    # This helper synthesizes "already validated and authorized" state
    # directly, bypassing a real Validate click. The authoritative identity
    # check now requires the current, freshly-built draft plan's canonical
    # identity to match what was "validated" -- stamp it from the window's
    # current (unmodified-by-this-helper) draft state so the synthetic
    # state is internally consistent with the real Run-guard.
    window._guided_validated_plan_identity = (
        compute_guided_new_analysis_draft_plan_identity(
            window._build_guided_new_analysis_draft_plan()
        )
    )
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
    context = SimpleNamespace(
        revision=request.current_guided_revision,
        draft=window._build_guided_new_analysis_draft_plan(),
    )
    accepted = outcome or replace(
        _accepted_outcome(),
        request_identity=request.startup_authority.rwd.stored_request_identity,
    )
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
            request.startup_authority,
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
    assert window._guided_startup_authority is (
        startup_request.startup_authority
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
    context = SimpleNamespace(
        revision=revision, draft=SimpleNamespace(input_format="rwd")
    )
    outcome = _accepted_outcome()
    built = request_builder.GuidedExecutionRequestBuildResult(
        status="built",
        ok=True,
        startup_authority=startup_request.startup_authority,
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
    assert request.startup_authority is authorized
    assert request.payload_result is derived
    assert authorized is startup_request.startup_authority
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
    assert window._guided_startup_authority is None
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
                window._guided_startup_authority,
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
    assert window._guided_startup_authority is None
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


@pytest.mark.parametrize("status", ("wrapper_failed", "wrapper_start_failed"))
def test_wrapper_failure_surfaces_actionable_scientist_message(
    window, startup_request, monkeypatch, qapp, status
):
    """A technical worker failure must not be the primary GUI instruction."""
    issue = backend.GuidedBackendExecutionIssue(
        category="wrapper_returned_nonzero",
        section="wrapper",
        message=(
            "Guided manifest verification refused: "
            "guided_manifest_parser_contract_mismatch"
        ),
        user_safe_message="Guided Run started, but the analysis reported an error.",
    )
    result = backend.GuidedBackendExecutionResult(
        status=status,
        ok=False,
        user_visible_state="failed_during_run",
        user_summary="Guided Run started, but the analysis reported an error.",
        run_directory=r"C:\output\guided-run",
        completed_run_candidate_path=None,
        requires_completed_run_loader_validation=False,
        wrapper_started=True,
        wrapper_completed=True,
        blocking_issues=(issue,),
        diagnostics=backend.GuidedBackendExecutionDiagnostics(
            orchestration_status="test",
            pure_plan_status=None,
            allocation_status=None,
            materialization_status=None,
            wrapper_started=True,
            wrapper_completed=True,
            wrapper_returncode=1,
            failure_marker_path=None,
            startup_transaction_identity=None,
            wrapper_command=None,
        ),
    )
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: result,
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)
    detail = window._guided_run_execution_details_label.text()
    assert detail == (
        "The analysis stopped before results were completed. Check the "
        "selected recording and setup, then try again. If the problem "
        "repeats, keep the run folder and ask for support."
    )
    assert "manifest" not in detail.lower()
    assert "parser_contract" not in detail.lower()
    # The real detail (issue.message, which for a genuine wrapper failure
    # already carries the subprocess's own stderr) must not be silently
    # lost -- it is persisted to the existing app log even though the
    # scientist-facing panel above stays generic.
    logged = window._log_view.toPlainText()
    assert "guided_manifest_parser_contract_mismatch" in logged
    assert "wrapper_returncode=1" in logged
    assert status in logged


def test_execution_details_label_clears_on_next_run_start(
    window, startup_request, monkeypatch, qapp
):
    """A stale error message from a previous failed run must not linger
    once a new run starts."""
    issue = backend.GuidedBackendExecutionIssue(
        category="wrapper_returned_nonzero",
        section="wrapper",
        message="stale failure detail",
        user_safe_message="Guided Run started, but the analysis reported an error.",
    )
    failed_result = backend.GuidedBackendExecutionResult(
        status="wrapper_failed",
        ok=False,
        user_visible_state="failed_during_run",
        user_summary="Guided Run started, but the analysis reported an error.",
        run_directory=r"C:\output\guided-run",
        completed_run_candidate_path=None,
        requires_completed_run_loader_validation=False,
        wrapper_started=True,
        wrapper_completed=True,
        blocking_issues=(issue,),
        diagnostics=backend.GuidedBackendExecutionDiagnostics(
            orchestration_status="test",
            pure_plan_status=None,
            allocation_status=None,
            materialization_status=None,
            wrapper_started=True,
            wrapper_completed=True,
            wrapper_returncode=1,
            failure_marker_path=None,
            startup_transaction_identity=None,
            wrapper_command=None,
        ),
    )
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: failed_result,
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)
    assert "analysis stopped before results were completed" in (
        window._guided_run_execution_details_label.text().lower()
    )

    _set_ready(window, startup_request)
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: _result(
            "wrapper_running", "Guided Run is running."
        ),
    )
    window._guided_run_btn.click()
    assert window._guided_run_execution_details_label.text() == ""


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


def test_rwd_completion_shows_exact_output_and_clears_it_after_edit(
    window, startup_request, tmp_path, monkeypatch, qapp
):
    run_dir = tmp_path / "guided-run"
    run_dir.mkdir()
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: completed,
    )
    opened = []
    monkeypatch.setattr("gui.main_window._open_folder", opened.append)

    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_run_execution_thread is None)

    assert window._guided_run_execution_details_label.text() == (
        f"Results folder: {run_dir}"
    )
    assert window._guided_npm_completed_output_dir == str(run_dir)
    assert window._guided_completed_output_format == "rwd"
    assert window._guided_npm_open_output_btn.isEnabled() is True
    window._on_guided_npm_open_output_folder_clicked()
    assert opened == [str(run_dir)]

    window._invalidate_guided_backend_validation("feature settings changed")
    assert window._guided_npm_completed_output_dir is None
    assert window._guided_completed_output_format is None
    assert window._guided_npm_open_output_btn.isHidden() is True
    assert window._guided_run_execution_details_label.text() == ""


def test_rwd_completion_does_not_infer_exclusion_from_selected_policy(
    window, startup_request, tmp_path, monkeypatch, qapp
):
    run_dir = tmp_path / "guided-run-with-exclusion"
    run_dir.mkdir()
    completed = replace(
        _result(
            "wrapper_completed_needs_review_loading",
            "Guided Run finished. Load the completed run for review.",
        ),
        run_directory=str(run_dir),
        completed_run_candidate_path=str(run_dir),
    )
    monkeypatch.setattr(
        backend,
        "execute_guided_backend_run",
        lambda *, request, runner=None: completed,
    )
    window._discovery_cache = {
        "sessions": [
            {"session_id": "2025_01_01-00_00_00"},
            {"session_id": "2025_01_01-00_30_00"},
        ]
    }
    window._guided_exclude_incomplete_final_rwd_chunk_cb.setChecked(True)

    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_run_execution_thread is None)

    detail = window._guided_run_execution_details_label.text()
    assert f"Results folder: {run_dir}" in detail
    assert "Excluded final recording session" not in detail
    assert "2025_01_01-00_30_00" not in detail


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


def test_run_button_has_one_authoritative_guarded_entry_point():
    source = Path(inspect.getfile(MainWindow)).read_text(encoding="utf-8")
    assert source.count(
        "self._guided_run_btn.clicked.connect(\n"
        "            self._on_guided_run_clicked_backend_guarded\n"
        "        )"
    ) == 1


def test_run_worker_does_not_reference_main_window():
    """The worker must hold no MainWindow reference and touch no GUI state.

    It receives only plain, already-captured values (`request`, `runner`)
    and calls only the module-level `execute_guided_backend_run` function.
    """
    init_params = list(
        inspect.signature(_GuidedRunExecutionWorker.__init__).parameters
    )
    # CR1-E2 added one optional `continuous_execution` parameter (default
    # None) for the internal continuous-RWD execution bridge; no existing
    # call site passes it and the intermittent request/runner contract is
    # unchanged (see test_no_existing_call_site_passes_continuous_execution
    # in tests/test_guided_continuous_rwd_execution_worker.py).
    assert init_params == ["self", "request", "runner", "continuous_execution"]

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


@pytest.mark.parametrize(
    "strategy_label",
    (
        "Robust Global Event-Reject Fit",
        "Adaptive Event-Gated Fit",
        "Global Linear Regression",
    ),
)
def test_real_gui_path_press_run_after_authorization(
    window, tmp_path, monkeypatch, qapp, strategy_label
):
    """4J16k11: drives the real cache-free local-preview new_analysis path
    (the same one 4J16k10 gets to validator_accepted / authorized / Run
    button enabled) through an actual click of the real Guided Run button,
    with nothing mocked below the click except the actual expensive/
    environment-dependent analysis subprocess: real worker thread, real
    execute_guided_backend_run -> run_guided_startup_to_wrapper -> real
    allocation -> real materialization -> real wrapper invocation.

    Every other test in this file hand-builds a GuidedStartupTransactionRequest
    via the `startup_request`/`allocation_case` fixtures, which hard-code a
    correct GuidedStartupFilesystemPolicy. That coverage proves the
    worker/close-guard/signal wiring is correct, but it never exercised the
    real computation in guided_execution_request_builder.py that derives
    filesystem_policy from actual GUI/output-destination state -- which is
    exactly what this test adds.

    4J16k11 audit finding (fixed here): for a brand-new output destination
    (the standard case for "new analysis" -- the output folder is never
    created ahead of Run, by design: see
    GuidedProductionExecutionProfile.allocate_output_at_future_run_start_only
    and the "no_directories_created" contract flags asserted throughout
    authorization/payload derivation), `build_guided_startup_request_from_
    validation` used to compute `output_base_exists_or_creatable =
    output_base.exists()` -- literal *current* existence only, never
    whether the path was actually creatable -- so every real Guided Run
    press failed immediately with the misleading message "the validated
    setup is no longer current." `guided_execution_request_builder.
    _output_base_creatability` now walks up to the nearest existing
    ancestor; `guided_startup_allocation.allocate_guided_startup_directory`
    now actually creates output_base (once every safety check has passed)
    instead of requiring it to already exist.

    This test uses the same controlled real-wrapper-boundary technique as
    test_real_backend_reaches_initial_status_boundary_only below: the real
    wrapper script runs in-process up to writing its initial "running"
    status, with the actual analysis subprocess forbidden. This proves the
    fix reaches a real, stable boundary without paying for -- or being
    fragile to unrelated failures in -- the full pipeline run (a live
    manual run of this path further confirms real allocation,
    materialization, and wrapper invocation now succeed; the wrapper's
    own analysis subprocess then fails for reasons in
    analyze_photometry.py/tools/run_full_pipeline_deliverables.py, a
    separate backend/runner concern out of scope for this patch).
    """
    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping
    import photometry_pipeline.guided_startup_transaction as startup_transaction
    from gui.main_window import GUIDED_WORKFLOW_STEPS
    from tests.test_gui_guided_new_analysis_plan import (
        _configure_complete_guided_new_analysis_draft_without_diagnostic_cache,
        _confirm_detected_dataset_settings_via_review_plan_button,
    )

    strategy_by_roi = {roi: strategy_label for roi in ("CH1", "CH2", "CH3")}
    _configure_complete_guided_new_analysis_draft_without_diagnostic_cache(
        window, tmp_path, monkeypatch, strategy_by_roi=strategy_by_roi
    )
    _confirm_detected_dataset_settings_via_review_plan_button(window, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    window._guided_review_go_to_run_btn.click()

    build_identity = production_mapping.build_application_build_identity(
        distribution_name="photometry-pipeline",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="abc123",
        source_tree_state="clean",
    )
    monkeypatch.setattr(
        request_builder,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(build_identity=build_identity),
    )
    window._guided_backend_validate_btn.click()
    outcome = window._guided_backend_validation_outcome
    assert outcome.status == "validator_accepted"
    auth = window._guided_startup_authority
    assert getattr(auth.rwd, "status", None) == "authorized"
    assert window._guided_run_btn.isEnabled() is True

    # The gate now accepts this real, GUI-constructed, never-before-used
    # output destination.
    retained_request = window._current_guided_startup_transaction_request()
    assert retained_request is not None
    gate_issue = startup_transaction._gate_issue(retained_request)
    assert gate_issue is None

    # Output base and the planned run directory must not exist yet --
    # nothing is allocated during Validate/authorization, only at Run press.
    output_base = Path(auth.rwd.production_intent.output_policy.output_base_canonical)
    planned_run_dir = Path(retained_request.planned_allocated_run_dir)
    assert not output_base.exists()
    assert not planned_run_dir.exists()

    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *args, **kwargs: None),
    )

    runner, calls = _real_wrapper_runner(monkeypatch)
    window._guided_backend_execution_runner = runner

    # Real click: real worker thread, real execute_guided_backend_run, real
    # run_guided_startup_to_wrapper, real allocation, real materialization,
    # real (controlled-boundary) wrapper invocation. Nothing below the
    # click is mocked except the actual analysis subprocess.
    window._guided_run_btn.click()
    # Worker started off the GUI thread: control returns immediately with
    # the running-state guard already active (no GUI-thread blocking).
    assert window._guided_backend_execution_active is True
    assert window._guided_backend_validate_btn.isEnabled() is False

    # Close guard must refuse to close while active.
    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *args, **kwargs: shown.append(args)),
    )
    close_event = QCloseEvent()
    window.closeEvent(close_event)
    assert close_event.isAccepted() is False
    assert shown, "close guard did not activate while execution was active"

    _pump_until(qapp, lambda: window._guided_run_execution_thread is None)

    result = window._guided_backend_execution_result
    assert result.status == "wrapper_running"
    assert result.ok is True
    assert result.wrapper_started is True
    assert result.wrapper_completed is False
    assert result.blocking_issues == ()
    assert calls == {"live_verify": 1, "analysis": 0, "root_makedirs": 0}

    # Output base and the run directory are created only now, at Run press.
    assert output_base.is_dir()
    assert planned_run_dir.is_dir()
    run_status = json.loads((planned_run_dir / "status.json").read_bytes())
    assert run_status["phase"] == "initializing"
    assert run_status["status"] == "running"

    # Coherent post-click state: no hang, guards cleared, an accurate
    # "running" message (not the misleading pre-fix staleness message).
    assert window._guided_backend_execution_active is False
    assert window._guided_backend_validate_btn.isEnabled() is True
    assert window._guided_run_readiness_label.text() == "Guided Run has started."
    assert window._guided_run_execution_worker is None
    assert window._guided_run_execution_thread is None

    # The accurate, non-misleading message for a genuinely not-creatable
    # output destination (a scenario the real GUI cannot actually reach --
    # gui/main_window.py's Apply-output-destination step already refuses to
    # apply a policy whose parent folder does not exist, and re-verifies
    # this at Validate time) is covered directly against
    # execute_guided_backend_run in
    # tests/test_guided_backend_execution.py::
    # test_output_not_creatable_maps_to_accurate_message.


def _drive_real_guided_rwd_setup(window, tmp_path, monkeypatch, *, apply_feature_defaults):
    """Drive the real GUI new-analysis RWD path up to (not including) the
    Run press, returning after Check My Setup. Shared by the loaded-Defaults
    visible regression tests below."""
    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping
    from gui.main_window import GUIDED_WORKFLOW_STEPS
    from tests.test_gui_guided_new_analysis_plan import (
        _configure_complete_guided_new_analysis_draft_without_diagnostic_cache,
        _confirm_detected_dataset_settings_via_review_plan_button,
    )

    strategy_by_roi = {roi: "Global Linear Regression" for roi in ("CH1", "CH2", "CH3")}
    _configure_complete_guided_new_analysis_draft_without_diagnostic_cache(
        window,
        tmp_path,
        monkeypatch,
        strategy_by_roi=strategy_by_roi,
        apply_feature_defaults=apply_feature_defaults,
    )
    _confirm_detected_dataset_settings_via_review_plan_button(window, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    window._guided_review_go_to_run_btn.click()
    build_identity = production_mapping.build_application_build_identity(
        distribution_name="photometry-pipeline",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="abc123",
        source_tree_state="clean",
    )
    monkeypatch.setattr(
        request_builder,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(build_identity=build_identity),
    )
    window._guided_backend_validate_btn.click()


def test_natural_path_fresh_output_reaches_beyond_previous_refusal(
    window, tmp_path, monkeypatch, qapp
):
    """Phase 3C natural-path regression: drives the real production
    validation -> authorization -> payload derivation -> startup request
    -> Run click -> worker seam -> execute_guided_backend_run ->
    startup orchestration path for an ordinary "new analysis" output
    destination (never used before, not a completed run). Proves the
    pure-plan handoff itself was never broken for this, the standard
    case -- the request is accepted and allocation is reached, so the
    previous refusal point (`refused_before_startup` /
    `pure_plan_not_accepted`) is never returned.
    """
    _drive_real_guided_rwd_setup(
        window, tmp_path, monkeypatch, apply_feature_defaults=True
    )
    outcome = window._guided_backend_validation_outcome
    assert outcome.status == "validator_accepted"
    assert window._guided_run_btn.isEnabled() is True

    retained_request = window._current_guided_startup_transaction_request()
    assert retained_request is not None
    assert (
        retained_request.filesystem_policy.output_base_is_completed_run_root
        is False
    )

    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    runner, calls = _real_wrapper_runner(monkeypatch)
    window._guided_backend_execution_runner = runner

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)

    result = window._guided_backend_execution_result
    assert result.status != "refused_before_startup"
    assert result.status == "wrapper_running"
    assert result.blocking_issues == ()
    assert "no longer current" not in result.user_summary
    assert calls == {"live_verify": 1, "analysis": 0, "root_makedirs": 0}


def test_natural_path_completed_run_root_gets_truthful_message(
    window, tmp_path, monkeypatch, qapp
):
    """Phase 3C natural-path regression: reproduces the real defect Jeff
    hit end-to-end through the real production path. The output
    destination is itself a genuinely completed prior run's folder (the
    exact condition confirmed on Jeff's real machine: the real artifact's
    planned run directory was nested directly under a previous run whose
    status.json read schema_version=1 / phase="final" / status="success").

    Before the Phase 3C fix, this reached `execute_guided_backend_run` and
    returned the false "the validated setup is no longer current" message
    even though the request was genuinely current -- the real problem was
    the (correct, safety-preserving) refusal to allocate under a
    completed run. After the fix, the request is still correctly refused
    (output safety is not weakened), but with a truthful, actionable
    summary and no output is created.
    """
    _drive_real_guided_rwd_setup(
        window, tmp_path, monkeypatch, apply_feature_defaults=True
    )
    outcome = window._guided_backend_validation_outcome
    assert outcome.status == "validator_accepted"

    output_dir = tmp_path / "output"
    (output_dir / "status.json").write_text(
        json.dumps(
            {"schema_version": 1, "phase": "final", "status": "success"}
        ),
        encoding="utf-8",
    )
    # Re-validate so the request is rebuilt against the output folder's
    # current (now completed-run-marked) on-disk state -- filesystem_policy
    # is computed fresh at build time, not cached from the first click.
    window._guided_backend_validate_btn.click()
    assert window._guided_backend_validation_outcome.status == "validator_accepted"
    assert window._guided_run_btn.isEnabled() is True

    retained_request = window._current_guided_startup_transaction_request()
    assert retained_request is not None
    assert (
        retained_request.filesystem_policy.output_base_is_completed_run_root
        is True
    )

    calls = []
    window._guided_backend_execution_runner = (
        lambda command: calls.append(command) or pytest.fail("runner called")
    )

    window._guided_run_btn.click()
    _pump_until(qapp, lambda: window._guided_backend_execution_result is not None)

    result = window._guided_backend_execution_result
    assert result.status == "refused_before_startup"
    assert result.blocking_issues[0].category == "pure_plan_output_unsafe"
    assert calls == []
    assert "no longer current" not in result.user_summary
    assert "output destination" in result.user_summary.lower()
    # Output safety preserved: nothing was allocated under the completed
    # run, and the completed run's own contents are untouched.
    assert not (output_dir / retained_request.planned_run_id).exists()
    assert (output_dir / "status.json").read_text(encoding="utf-8") != ""


def test_real_gui_loaded_defaults_without_apply_enables_run(
    window, tmp_path, monkeypatch, qapp
):
    """B2 Phase 2 visible regression: a valid loaded Feature Detection Default
    profile that was NEVER explicitly applied must reach ready_hidden and
    enable Run through the real validate -> authorize -> payload derivation
    path. This is the exact real interactive failure the payload predicate
    repair fixes. Nothing about authorization/payload/readiness is assigned
    directly."""
    _drive_real_guided_rwd_setup(
        window, tmp_path, monkeypatch, apply_feature_defaults=False
    )
    outcome = window._guided_backend_validation_outcome
    assert outcome.status == "validator_accepted"
    # The saved Default profile was loaded but never explicitly applied.
    draft = window._build_guided_new_analysis_draft_plan()
    assert draft.feature_event_profile_status == "default_initialized"
    assert draft.feature_event_explicitly_applied is False
    # With the repaired payload predicate, authorization + payload succeed
    # naturally and Run is enabled.
    auth = window._guided_startup_authority
    assert getattr(auth.rwd, "status", None) == "authorized"
    assert window._guided_execution_derivation_reason is None
    assert window._guided_run_readiness.status == "ready_hidden"
    assert window._guided_run_btn.isEnabled() is True
    # The old-code failure for this exact loaded-Defaults case is pinned at
    # the source in tests/test_guided_execution_payloads.py::
    # test_old_predicate_would_have_refused_loaded_default. It cannot be
    # pinned by monkeypatching is_saved_feature_event_profile_current here,
    # because that shared predicate is also consumed by backend validation
    # (reverting it would refuse validation before payload derivation is
    # ever reached), whereas the real defect was payload derivation using
    # its own obsolete explicitly-applied gate instead of the shared rule.


def test_applied_defaults_also_enable_run_after_repair(
    window, tmp_path, monkeypatch, qapp
):
    """Regression guard: explicitly applied Defaults must still reach
    ready_hidden and enable Run (the repair must not only fix the
    default_initialized case)."""
    _drive_real_guided_rwd_setup(
        window, tmp_path, monkeypatch, apply_feature_defaults=True
    )
    assert window._guided_backend_validation_outcome.status == "validator_accepted"
    draft = window._build_guided_new_analysis_draft_plan()
    assert draft.feature_event_profile_status == "applied"
    assert draft.feature_event_explicitly_applied is True
    assert getattr(window._guided_startup_authority.rwd, "status", None) == (
        "authorized"
    )
    assert window._guided_run_btn.isEnabled() is True
    # Baseline applied-Default value is in the payload.
    assert _retained_config_values(window)["peak_threshold_k"] == 2.5

    # Now genuinely edit AND apply a distinguishable value; the newly applied
    # value must be the one that reaches the payload, and the former saved
    # value must be gone for that active field.
    window._guided_feature_event_peak_k_edit.setText("3.25")
    window._guided_feature_event_apply_btn.click()
    window._guided_backend_validate_btn.click()
    assert window._guided_backend_validation_outcome.status == "validator_accepted"
    assert window._guided_run_btn.isEnabled() is True
    applied_values = _retained_config_values(window)
    assert applied_values["peak_threshold_k"] == 3.25
    assert applied_values["peak_threshold_k"] != 2.5


def _retained_config_values(window):
    """The actual retained execution config payload as a {name: value} dict."""
    request = window._guided_startup_transaction_request
    assert request is not None, "no startup request was retained"
    return {v.name: v.value for v in request.payload_result.config_payload.values}


def test_real_gui_dirty_default_editor_uses_saved_value_not_draft(
    window, tmp_path, monkeypatch, qapp
):
    """Visible dirty-editor payload regression: a valid saved Default profile
    is loaded (peak_threshold_k = 2.5). The shared Default editor is then
    edited to a clearly distinguishable 4.75 WITHOUT pressing "Use these as
    Default settings". After re-checking the setup, payload derivation must
    use the SAVED 2.5, never the unapplied draft 4.75, while Run still
    enables. The actual retained config payload is inspected directly."""
    _drive_real_guided_rwd_setup(
        window, tmp_path, monkeypatch, apply_feature_defaults=False
    )
    # Baseline: the saved Default value is in the payload.
    assert window._guided_run_btn.isEnabled() is True
    assert _retained_config_values(window)["peak_threshold_k"] == 2.5

    # Edit the visible shared Default form to a distinguishable value and do
    # NOT apply it.
    window._guided_feature_event_peak_k_edit.setText("4.75")
    window._guided_feature_event_peak_k_edit.editingFinished.emit()

    # Re-check the setup (the edit invalidated the prior check).
    window._guided_backend_validate_btn.click()

    # The draft still reflects the saved (never-applied) Default profile.
    draft = window._build_guided_new_analysis_draft_plan()
    assert draft.feature_event_profile_status == "default_initialized"
    assert draft.feature_event_explicitly_applied is False

    # Payload derivation succeeded and Run is enabled again...
    assert window._guided_backend_validation_outcome.status == "validator_accepted"
    assert window._guided_run_btn.isEnabled() is True
    # ...and the ACTUAL retained config uses the saved 2.5, not the draft 4.75.
    values = _retained_config_values(window)
    assert values["peak_threshold_k"] == 2.5
    assert values["peak_threshold_k"] != 4.75


def _customize_all_included_rois(window, monkeypatch, config_fields):
    """Give every included ROI a valid Custom Feature Detection override via
    the fake per-ROI dialog (no real UI)."""
    import gui.main_window as main_window_module

    class _FakeRoiDialog:
        def __init__(self, *_a, **_k):
            pass

        def exec(self):
            return main_window_module.QDialog.Accepted

        def result_values(self):
            return dict(config_fields)

    monkeypatch.setattr(
        main_window_module, "_GuidedRoiFeatureEventDialog", _FakeRoiDialog
    )
    for roi in ("CH1", "CH2", "CH3"):
        window._on_guided_customize_roi_feature_event(roi)


def test_real_gui_all_custom_with_dirty_default_editor_enables_run(
    window, tmp_path, monkeypatch, qapp
):
    """Visible all-Custom regression: every included ROI uses a valid Custom
    Feature Detection override (peak_threshold_k = 3.5) while the shared
    Default editor holds an unapplied dirty 4.75. The plan -> validation ->
    authorization -> payload path must reach Run-ready, the Custom values
    must be in the effective configuration, and the dirty Default draft must
    not enter the payload. No readiness state is injected."""
    _drive_real_guided_rwd_setup(
        window, tmp_path, monkeypatch, apply_feature_defaults=True
    )
    assert window._guided_run_btn.isEnabled() is True

    _customize_all_included_rois(window, monkeypatch, {"peak_threshold_k": 3.5})
    assert set(window._guided_per_roi_feature_event_overrides) == {"CH1", "CH2", "CH3"}

    # Unapplied dirty edit in the shared Default editor.
    window._guided_feature_event_peak_k_edit.setText("4.75")
    window._guided_feature_event_peak_k_edit.editingFinished.emit()

    window._guided_backend_validate_btn.click()

    assert window._guided_backend_validation_outcome.status == "validator_accepted"
    assert window._guided_run_btn.isEnabled() is True

    request = window._guided_startup_transaction_request
    assert request is not None
    per_roi = request.startup_authority.rwd.production_intent.feature_event.per_roi_feature_event_map
    by_roi = {entry.roi_id: entry for entry in per_roi}
    for roi in ("CH1", "CH2", "CH3"):
        entry = by_roi[roi]
        assert entry.source == "override"  # genuinely Custom
        override = {v.field_name: v.value for v in entry.override_config_fields}
        effective = {v.field_name: v.value for v in entry.effective_config_fields}
        # The Custom value is present in the effective execution config...
        assert override.get("peak_threshold_k") == 3.5
        assert effective.get("peak_threshold_k") == 3.5
        # ...and the dirty unapplied Default draft (4.75) is nowhere.
        assert override.get("peak_threshold_k") != 4.75
        assert effective.get("peak_threshold_k") != 4.75

    # The base config payload also never picked up the dirty 4.75.
    base_values = _retained_config_values(window)
    assert base_values["peak_threshold_k"] != 4.75
