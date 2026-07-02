from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import photometry_pipeline.guided_backend_execution as backend
import photometry_pipeline.guided_startup_orchestration as orchestration
from gui.run_report_parser import classify_completed_run_candidate
from tests.test_guided_startup_allocation import allocation_case


def _process(command, *, returncode=0, started=True, completed=True, stderr=""):
    return orchestration.GuidedWrapperProcessResult(
        returncode=returncode,
        stdout="stdout",
        stderr=stderr,
        command=command,
        started=started,
        completed=completed,
    )


def test_refused_plan_maps_to_not_started(allocation_case):
    request, _plan = allocation_case
    request = replace(request, explicit_user_run_transition=False)
    result = backend.execute_guided_backend_run(
        request=request,
        runner=lambda command: pytest.fail("runner called"),
    )
    assert result.status == "refused_before_startup"
    assert result.user_visible_state == "not_started"
    assert result.ok is False
    assert result.run_directory is None


def test_allocation_failure_maps_to_safe_output_failure(
    allocation_case, monkeypatch
):
    request, _plan = allocation_case
    original_mkdir = Path.mkdir

    def fail_run_dir(self, *args, **kwargs):
        if str(self) == request.planned_allocated_run_dir:
            raise OSError("simulated allocation failure")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_run_dir)
    result = backend.execute_guided_backend_run(
        request=request,
        runner=lambda command: pytest.fail("runner called"),
    )
    assert result.status == "startup_allocation_failed"
    assert result.user_visible_state == "failed_to_prepare"
    assert result.diagnostics.allocation_status == "allocation_failed"


def test_startup_status_failure_preserves_run_directory(
    allocation_case, monkeypatch
):
    request, _plan = allocation_case
    original_open = Path.open

    def fail_status(self, mode="r", *args, **kwargs):
        if self.name == "guided_startup_status.json" and mode == "xb":
            raise OSError("simulated status failure")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_status)
    result = backend.execute_guided_backend_run(
        request=request,
        runner=lambda command: pytest.fail("runner called"),
    )
    assert result.status == "startup_status_write_failed"
    assert result.user_visible_state == "failed_to_prepare"
    assert result.run_directory == request.planned_allocated_run_dir
    assert Path(result.run_directory).is_dir()
    assert result.diagnostics.allocation_status == "allocated_status_write_failed"


def test_materialization_failure_maps_to_failed_prepare(
    allocation_case, monkeypatch
):
    request, _plan = allocation_case
    original_open = Path.open

    def fail_config(self, mode="r", *args, **kwargs):
        if self.name == "config_effective.yaml" and mode == "xb":
            raise OSError("simulated config failure")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_config)
    result = backend.execute_guided_backend_run(
        request=request,
        runner=lambda command: pytest.fail("runner called"),
    )
    assert result.status == "startup_materialization_failed"
    assert result.user_visible_state == "failed_to_prepare"
    assert result.diagnostics.materialization_status == (
        "materialization_failed_partial"
    )


def test_wrapper_start_failure_maps_to_failed_to_start(allocation_case):
    request, _plan = allocation_case
    result = backend.execute_guided_backend_run(
        request=request,
        runner=lambda command: _process(
            command,
            returncode=None,
            started=False,
            completed=False,
            stderr="start failed",
        ),
    )
    assert result.status == "wrapper_start_failed"
    assert result.user_visible_state == "failed_to_start"
    assert result.wrapper_started is False
    assert result.diagnostics.failure_marker_path is not None


def test_nonzero_wrapper_maps_to_failed_during_run(allocation_case):
    request, _plan = allocation_case
    result = backend.execute_guided_backend_run(
        request=request,
        runner=lambda command: _process(
            command, returncode=9, stderr="analysis failed"
        ),
    )
    assert result.status == "wrapper_failed"
    assert result.user_visible_state == "failed_during_run"
    assert result.wrapper_started and result.wrapper_completed
    assert result.diagnostics.wrapper_returncode == 9


def test_started_not_completed_maps_to_running(allocation_case):
    request, _plan = allocation_case
    result = backend.execute_guided_backend_run(
        request=request,
        runner=lambda command: _process(
            command,
            returncode=None,
            started=True,
            completed=False,
        ),
    )
    assert result.status == "wrapper_running"
    assert result.user_visible_state == "running"
    assert result.ok is True
    assert result.completed_run_candidate_path is None


def test_zero_return_requires_completed_run_loader_validation(allocation_case):
    request, _plan = allocation_case
    result = backend.execute_guided_backend_run(
        request=request,
        runner=lambda command: _process(command, returncode=0),
    )
    assert result.status == "wrapper_completed_needs_review_loading"
    assert result.user_visible_state == "run_finished_review_required"
    assert result.ok is True
    assert result.completed_run_candidate_path == result.run_directory
    assert result.requires_completed_run_loader_validation is True
    assert result.completed_run_claim is False
    assert classify_completed_run_candidate(result.run_directory)[0] is False


def test_diagnostics_preserve_internal_state_without_user_exposure(
    allocation_case,
):
    request, _plan = allocation_case
    result = backend.execute_guided_backend_run(
        request=request,
        runner=lambda command: _process(command, returncode=4, stderr="failed"),
    )
    assert result.diagnostics.orchestration_status == "wrapper_failed"
    assert result.diagnostics.wrapper_returncode == 4
    assert result.diagnostics.startup_transaction_identity
    assert "--guided-preallocated-run-dir" in result.diagnostics.wrapper_command
    assert result.exposes_manifest_path_to_user is False
    assert result.exposes_internal_cli_to_user is False
    assert result.gui_run_enabled is False


def test_all_user_summaries_exclude_internal_terms():
    prohibited = (
        "manifest",
        "preallocated",
        "command_invoked",
        "wrapper claim",
        "startup transaction",
        "hash",
        "--guided",
        "config_effective.yaml",
    )
    for _status, _state, _ok, summary in backend._STATUS_MAP.values():
        lowered = summary.lower()
        assert not any(term in lowered for term in prohibited)


def test_backend_adapter_import_boundary():
    source = Path(backend.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    prohibited = (
        "gui",
        "subprocess",
        "photometry_pipeline.pipeline",
        "tools.run_full_pipeline_deliverables",
        "gui.run_report_parser",
    )
    assert not any(
        name == marker or name.startswith(f"{marker}.")
        for name in imported
        for marker in prohibited
    )
    assert "Pipeline(" not in source
