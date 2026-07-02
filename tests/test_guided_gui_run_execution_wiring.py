from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

import photometry_pipeline.guided_backend_execution as backend
import photometry_pipeline.guided_startup_claim as claim
from photometry_pipeline.guided_startup_transaction import (
    GuidedStartupTransactionRequest,
)
from gui.main_window import MainWindow
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
    instance.close()
    instance.deleteLater()


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
    import photometry_pipeline.application_build_identity as build_identity
    import photometry_pipeline.guided_execution_payloads as payloads
    import photometry_pipeline.guided_run_authorization as authorization

    revision = startup_request.current_guided_revision
    window._guided_backend_validation_revision = revision
    context = SimpleNamespace(revision=revision)
    outcome = _accepted_outcome()
    monkeypatch.setattr(
        build_identity,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(
            build_identity=startup_request.application_build_identity
        ),
    )
    monkeypatch.setattr(
        authorization,
        "build_guided_run_authorization_request",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        authorization,
        "authorize_guided_run",
        lambda _request: startup_request.authorization_result,
    )
    monkeypatch.setattr(
        payloads,
        "build_guided_execution_startup_mapping_contract",
        lambda: startup_request.startup_mapping_contract,
    )
    monkeypatch.setattr(
        payloads,
        "derive_guided_execution_payloads",
        lambda _authorization, **_kwargs: startup_request.payload_result,
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
    window, startup_request, monkeypatch
):
    _run_production_validation_update(window, startup_request, monkeypatch)
    calls = []
    running = _result("wrapper_running", "Guided Run is running.")
    monkeypatch.setattr(
        window,
        "_execute_guided_backend_run_for_gui",
        lambda request: calls.append(request) or running,
    )
    window._guided_run_btn.click()
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
    window, startup_request, monkeypatch
):
    calls = []
    expected = _result("wrapper_running", "Guided Run is running.")

    def execute(request):
        calls.append(request)
        return expected

    monkeypatch.setattr(window, "_execute_guided_backend_run_for_gui", execute)
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    assert calls == [startup_request]
    assert isinstance(calls[0], GuidedStartupTransactionRequest)
    assert window._guided_backend_execution_result is expected
    assert window._guided_run_readiness_label.text() == "Guided Run has started."
    assert window._guided_run_btn.isEnabled() is False


def test_click_refuses_if_readiness_becomes_stale_after_enablement(
    window, startup_request, monkeypatch
):
    _run_production_validation_update(window, startup_request, monkeypatch)
    assert window._guided_run_btn.isEnabled()
    window._guided_backend_validation_revision += 1
    monkeypatch.setattr(
        window,
        "_execute_guided_backend_run_for_gui",
        lambda _request: pytest.fail("backend called for stale readiness"),
    )
    window._guided_run_btn.click()
    assert window._guided_run_readiness.status == "validation_stale"
    assert window._guided_run_btn.isEnabled() is False
    assert "Validate again" in window._guided_run_readiness_label.text()


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
    window, startup_request, monkeypatch, result, expected_text
):
    monkeypatch.setattr(
        window,
        "_execute_guided_backend_run_for_gui",
        lambda _request: result,
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    assert window._guided_backend_execution_result is result
    assert window._guided_run_readiness_label.text() == expected_text
    assert window._guided_run_btn.isEnabled() is False


def test_completed_result_does_not_auto_load_or_claim_success(
    window, startup_request, monkeypatch
):
    completed = _result(
        "wrapper_completed_needs_review_loading",
        "Guided Run finished. Load the completed run for review.",
    )
    monkeypatch.setattr(
        window,
        "_execute_guided_backend_run_for_gui",
        lambda _request: completed,
    )
    monkeypatch.setattr(
        window,
        "_open_completed_results_dir",
        lambda *_args, **_kwargs: pytest.fail("Review auto-loaded"),
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    assert completed.requires_completed_run_loader_validation is True
    assert completed.completed_run_claim is False
    assert window._guided_backend_execution_result is completed


def test_fake_backend_click_writes_no_files(
    window, startup_request, tmp_path, monkeypatch
):
    running = _result("wrapper_running", "Guided Run is running.")
    monkeypatch.setattr(
        window,
        "_execute_guided_backend_run_for_gui",
        lambda _request: running,
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
    assert tuple(tmp_path.iterdir()) == before


def test_execution_text_excludes_internal_terms(
    window, startup_request, monkeypatch
):
    result = _result(
        "wrapper_failed",
        "Guided Run started, but the analysis reported an error.",
    )
    monkeypatch.setattr(
        window,
        "_execute_guided_backend_run_for_gui",
        lambda _request: result,
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
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
    window, startup_request, monkeypatch
):
    before = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    monkeypatch.setattr(
        window,
        "_execute_guided_backend_run_for_gui",
        lambda _request: _result("wrapper_running", "Guided Run is running."),
    )
    _set_ready(window, startup_request)
    window._guided_run_btn.click()
    after = (
        window._run_btn.text(),
        window._run_btn.isEnabled(),
        window._run_btn.toolTip(),
    )
    assert after == before


def test_gui_execution_methods_use_only_backend_adapter():
    source = inspect.getsource(
        MainWindow._execute_guided_backend_run_for_gui
    ) + inspect.getsource(MainWindow._on_guided_run_clicked_backend_guarded)
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


def test_real_backend_reaches_initial_status_boundary_only(
    window, allocation_case, monkeypatch
):
    request, _plan = allocation_case
    runner, calls = _real_wrapper_runner(monkeypatch)
    _set_ready(window, request)
    window._guided_backend_execution_runner = runner
    window._guided_run_btn.click()

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
