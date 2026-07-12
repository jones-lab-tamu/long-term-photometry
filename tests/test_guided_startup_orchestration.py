from __future__ import annotations

import ast
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_startup_orchestration as orchestration
from gui.run_report_parser import classify_completed_run_candidate
from tests.test_guided_startup_allocation import allocation_case


def _process(command, *, returncode=0, started=True, completed=True, stderr=""):
    return orchestration.GuidedWrapperProcessResult(
        returncode=returncode,
        stdout="wrapper stdout",
        stderr=stderr,
        command=command,
        started=started,
        completed=completed,
    )


def test_valid_orchestration_allocates_materializes_and_invokes_once(
    allocation_case,
):
    request, _plan = allocation_case
    calls = []

    def runner(command):
        calls.append(command)
        return _process(command)

    result = orchestration.run_guided_startup_to_wrapper(
        request=request, subprocess_runner=runner
    )
    assert result.status == "wrapper_completed"
    assert result.ok
    assert result.wrapper_started and result.wrapper_completed
    assert result.wrapper_returncode == 0
    assert calls == [result.wrapper_command]
    assert result.allocation_status == "allocated_startup_status_written"
    assert result.materialization_status == "startup_artifacts_materialized"
    assert result.completed_run_claim is False
    assert result.gui_run_enabled is False
    assert result.user_facing_manifest_workflow is False


def test_wrapper_command_is_exact_internal_first_subset(allocation_case):
    request, _plan = allocation_case
    observed = []

    def runner(command):
        observed.append(command)
        return _process(command)

    result = orchestration.run_guided_startup_to_wrapper(
        request=request, subprocess_runner=runner
    )
    command = observed[0]
    assert "--guided-preallocated-run-dir" in command
    assert command[command.index("--guided-candidate-manifest") + 1].endswith(
        "guided_candidate_manifest.json"
    )
    assert command[command.index("--out") + 1] == result.allocated_run_dir
    assert command[command.index("--config") + 1].endswith(
        "config_effective.yaml"
    )
    assert command[command.index("--mode") + 1] == "phasic"
    assert command[command.index("--run-type") + 1] == "full"
    prohibited = {
        "--out-base",
        "--overwrite",
        "tonic",
        "both",
        "--preview-first-n",
        "--discover",
        "--validate-only",
        "--traces-only",
        "--include-rois",
        "--exclude-rois",
    }
    assert not prohibited.intersection(command)
    assert tuple(
        (
            Path(result.allocated_run_dir) / "command_invoked.txt"
        ).read_text(encoding="utf-8").splitlines()
    ) == command


def test_refused_plan_writes_nothing_and_does_not_invoke(allocation_case):
    request, _plan = allocation_case
    refused_request = replace(request, explicit_user_run_transition=False)
    calls = []
    result = orchestration.run_guided_startup_to_wrapper(
        request=refused_request,
        subprocess_runner=lambda command: calls.append(command),
    )
    assert result.status == "refused_before_allocation"
    assert calls == []
    assert not Path(refused_request.planned_allocated_run_dir).exists()


def test_supplied_stale_plan_refuses_before_writes(allocation_case):
    request, plan = allocation_case
    stale = replace(plan, status="refused")
    result = orchestration.run_guided_startup_to_wrapper(
        request=request,
        pure_plan=stale,
        subprocess_runner=lambda command: pytest.fail("runner called"),
    )
    assert result.blocking_issues[0].category == "pure_plan_stale_or_tampered"
    assert not Path(request.planned_allocated_run_dir).exists()


def test_allocation_refusal_stops_before_materialization_and_runner(
    allocation_case,
):
    request, _plan = allocation_case
    Path(request.planned_allocated_run_dir).mkdir()
    result = orchestration.run_guided_startup_to_wrapper(
        request=request,
        subprocess_runner=lambda command: pytest.fail("runner called"),
    )
    assert result.status == "refused_before_allocation"
    assert result.materialization_status is None
    assert result.wrapper_started is False


def test_allocated_status_write_failure_reports_side_effect_truthfully(
    allocation_case, monkeypatch
):
    request, _plan = allocation_case
    original_open = Path.open
    runner_calls = []

    def fail_startup_status(self, mode="r", *args, **kwargs):
        if self.name == "guided_startup_status.json" and mode == "xb":
            raise OSError("simulated startup status failure")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_startup_status)
    result = orchestration.run_guided_startup_to_wrapper(
        request=request,
        subprocess_runner=lambda command: runner_calls.append(command),
    )
    assert result.status == "startup_status_write_failed"
    assert result.status != "refused_before_allocation"
    assert result.allocation_status == "allocated_status_write_failed"
    assert result.allocated_run_dir == request.planned_allocated_run_dir
    assert Path(result.allocated_run_dir).is_dir()
    assert result.materialization_status is None
    assert result.wrapper_started is False
    assert runner_calls == []
    accepted, _reason = classify_completed_run_candidate(
        result.allocated_run_dir
    )
    assert accepted is False


def test_materialization_partial_failure_never_invokes_wrapper(
    allocation_case, monkeypatch
):
    request, _plan = allocation_case
    original_open = Path.open

    def fail_config(self, mode="r", *args, **kwargs):
        if self.name == "config_effective.yaml" and mode == "xb":
            raise OSError("simulated config failure")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_config)
    result = orchestration.run_guided_startup_to_wrapper(
        request=request,
        subprocess_runner=lambda command: pytest.fail("runner called"),
    )
    assert result.status == "materialization_failed"
    assert result.materialization_status == "materialization_failed_partial"
    assert result.wrapper_started is False
    accepted, _reason = classify_completed_run_candidate(
        result.allocated_run_dir
    )
    assert accepted is False


def test_supplied_accepted_allocation_and_materialization_are_reused(
    allocation_case,
):
    import photometry_pipeline.guided_startup_allocation as allocation
    import photometry_pipeline.guided_startup_materialization as materialization

    request, plan = allocation_case
    allocated = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    prepared = materialization.materialize_guided_startup_artifacts(
        request=request,
        pure_plan=plan,
        allocation_result=allocated,
    )
    result = orchestration.run_guided_startup_to_wrapper(
        request=request,
        pure_plan=plan,
        allocation_result=allocated,
        materialization_result=prepared,
        subprocess_runner=lambda command: _process(command),
    )
    assert result.status == "wrapper_completed"
    assert result.allocated_run_dir == allocated.allocated_run_dir


def test_wrapper_start_failure_is_distinct_and_records_nonproduction_marker(
    allocation_case,
):
    request, _plan = allocation_case

    def runner(command):
        return _process(
            command,
            returncode=None,
            started=False,
            completed=False,
            stderr="could not start",
        )

    result = orchestration.run_guided_startup_to_wrapper(
        request=request, subprocess_runner=runner
    )
    assert result.status == "wrapper_start_failed"
    assert result.wrapper_started is False
    assert result.wrapper_completed is False
    assert result.wrapper_returncode is None
    assert result.failure_marker_written is True
    marker = json.loads(Path(result.failure_marker_path).read_bytes())
    assert marker["runner_started"] is False
    assert marker["completed_run_claim"] is False
    assert marker["startup_transaction_identity"] == (
        result.startup_transaction_identity
    )
    assert not (Path(result.allocated_run_dir) / "status.json").exists()
    accepted, _reason = classify_completed_run_candidate(
        result.allocated_run_dir
    )
    assert accepted is False


def test_runner_exception_is_a_start_failure(allocation_case):
    request, _plan = allocation_case

    def runner(_command):
        raise OSError("spawn failed")

    result = orchestration.run_guided_startup_to_wrapper(
        request=request, subprocess_runner=runner
    )
    assert result.status == "wrapper_start_failed"
    assert result.wrapper_started is False
    assert result.failure_marker_written is True


def test_nonzero_after_start_is_wrapper_failure_without_orchestration_marker(
    allocation_case,
):
    request, _plan = allocation_case
    result = orchestration.run_guided_startup_to_wrapper(
        request=request,
        subprocess_runner=lambda command: _process(
            command, returncode=7, stderr="wrapper failed"
        ),
    )
    assert result.status == "wrapper_failed"
    assert result.wrapper_started and result.wrapper_completed
    assert result.wrapper_returncode == 7
    assert result.failure_marker_written is False
    assert not (
        Path(result.allocated_run_dir)
        / orchestration.GUIDED_STARTUP_ORCHESTRATION_FAILURE_FILENAME
    ).exists()


def test_started_but_not_completed_is_reported_without_completion_claim(
    allocation_case,
):
    request, _plan = allocation_case
    result = orchestration.run_guided_startup_to_wrapper(
        request=request,
        subprocess_runner=lambda command: _process(
            command,
            returncode=None,
            started=True,
            completed=False,
        ),
    )
    assert result.status == "wrapper_started"
    assert result.wrapper_started is True
    assert result.wrapper_completed is False
    assert result.completed_run_claim is False


def test_zero_return_does_not_claim_completed_run_validity(allocation_case):
    request, _plan = allocation_case
    result = orchestration.run_guided_startup_to_wrapper(
        request=request,
        subprocess_runner=lambda command: _process(command, returncode=0),
    )
    assert result.status == "wrapper_completed"
    assert result.ok is True
    assert result.completed_run_claim is False
    assert {item.name for item in Path(result.allocated_run_dir).iterdir()} == {
        "guided_startup_status.json",
        "guided_candidate_manifest.json",
        "config_effective.yaml",
        "command_invoked.txt",
        "guided_startup_provenance.json",
    }
    accepted, _reason = classify_completed_run_candidate(
        result.allocated_run_dir
    )
    assert accepted is False


def test_invalid_runner_result_is_start_failure(allocation_case):
    request, _plan = allocation_case
    result = orchestration.run_guided_startup_to_wrapper(
        request=request,
        subprocess_runner=lambda _command: object(),
    )
    assert result.status == "wrapper_start_failed"
    assert result.wrapper_started is False


def test_orchestration_import_boundary_and_no_gui_pipeline_calls():
    source = Path(orchestration.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    prohibited = ("gui", "photometry_pipeline.pipeline")
    assert not any(
        name == marker or name.startswith(f"{marker}.")
        for name in imported
        for marker in prohibited
    )
    assert "Pipeline(" not in source
    assert "--guided-preallocated-run-dir" in source
    assert "production_intent.execution_profile.execution_mode" in source
