from __future__ import annotations

import json
from pathlib import Path

import pytest

import photometry_pipeline.guided_startup_claim as claim
import photometry_pipeline.guided_startup_orchestration as orchestration
import tools.run_full_pipeline_deliverables as wrapper
from gui.run_report_parser import classify_completed_run_candidate
from tests.test_guided_startup_allocation import allocation_case


class _StopAfterInitialStatus(RuntimeError):
    pass


def _real_wrapper_runner(monkeypatch, *, live_failure=False, prestatus_failure=False):
    calls = {"live_verify": 0, "analysis": 0, "root_makedirs": 0}

    def runner(command):
        monkeypatch.setattr(wrapper.sys, "argv", [command[1], *command[2:]])

        def live_verify(_args):
            calls["live_verify"] += 1
            if live_failure:
                raise RuntimeError("simulated live manifest refusal")
            return object()

        monkeypatch.setattr(
            wrapper, "verify_guided_manifest_before_output", live_verify
        )

        original_resolve = wrapper.resolve_run_dir

        def resolve(args):
            if prestatus_failure:
                raise RuntimeError("simulated pre-status refusal")
            return original_resolve(args)

        monkeypatch.setattr(wrapper, "resolve_run_dir", resolve)

        def root_makedirs_forbidden(*_args, **_kwargs):
            calls["root_makedirs"] += 1
            raise AssertionError("preallocated root must not be created")

        monkeypatch.setattr(wrapper.os, "makedirs", root_makedirs_forbidden)

        def analysis_forbidden(*_args, **_kwargs):
            calls["analysis"] += 1
            raise AssertionError("analysis subprocess must not run")

        monkeypatch.setattr(wrapper, "run_cmd", analysis_forbidden)
        monkeypatch.setattr(wrapper.subprocess, "run", analysis_forbidden)

        def stop_after_status(**_kwargs):
            raise _StopAfterInitialStatus()

        monkeypatch.setattr(
            wrapper, "_GUIDED_TEST_STOP_AFTER_INITIAL_STATUS", stop_after_status
        )
        try:
            wrapper.main()
        except _StopAfterInitialStatus:
            return orchestration.GuidedWrapperProcessResult(
                returncode=None,
                stdout="stopped after initial status",
                stderr="",
                command=command,
                started=True,
                completed=False,
            )
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            return orchestration.GuidedWrapperProcessResult(
                returncode=code,
                stdout="",
                stderr="wrapper refused before boundary",
                command=command,
                started=True,
                completed=True,
            )
        except RuntimeError as exc:
            return orchestration.GuidedWrapperProcessResult(
                returncode=1,
                stdout="",
                stderr=str(exc),
                command=command,
                started=True,
                completed=True,
            )
        raise AssertionError("real wrapper passed the test stop boundary")

    return runner, calls


def test_orchestration_enters_real_wrapper_and_stops_after_initial_status(
    allocation_case, monkeypatch
):
    request, _plan = allocation_case
    runner, calls = _real_wrapper_runner(monkeypatch)
    result = orchestration.run_guided_startup_to_wrapper(
        request=request, subprocess_runner=runner
    )
    run_dir = Path(result.allocated_run_dir)
    status = json.loads((run_dir / "status.json").read_bytes())
    assert result.status == "wrapper_started"
    assert result.wrapper_started is True
    assert result.wrapper_completed is False
    assert result.completed_run_claim is False
    assert calls == {"live_verify": 1, "analysis": 0, "root_makedirs": 0}
    assert (
        run_dir / claim.GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME
    ).is_file()
    assert status["phase"] == "initializing"
    assert status["status"] == "running"
    assert status["run_root"] == str(run_dir)

    expected = {
        "guided_startup_status.json",
        "guided_candidate_manifest.json",
        "config_effective.yaml",
        "command_invoked.txt",
        "guided_startup_provenance.json",
        claim.GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME,
        "status.json",
    }
    assert {item.name for item in run_dir.iterdir()} == expected
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
    accepted, _reason = classify_completed_run_candidate(str(run_dir))
    assert accepted is False


def test_live_manifest_refusal_creates_no_claim_or_production_status(
    allocation_case, monkeypatch
):
    request, _plan = allocation_case
    runner, calls = _real_wrapper_runner(monkeypatch, live_failure=True)
    result = orchestration.run_guided_startup_to_wrapper(
        request=request, subprocess_runner=runner
    )
    run_dir = Path(result.allocated_run_dir)
    assert result.status == "wrapper_failed"
    assert calls["live_verify"] == 1
    assert not (
        run_dir / claim.GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME
    ).exists()
    assert not (run_dir / "status.json").exists()
    assert calls["analysis"] == 0
    assert classify_completed_run_candidate(str(run_dir))[0] is False


def test_nonwriting_prestatus_failure_does_not_claim(
    allocation_case, monkeypatch
):
    request, _plan = allocation_case
    runner, calls = _real_wrapper_runner(monkeypatch, prestatus_failure=True)
    result = orchestration.run_guided_startup_to_wrapper(
        request=request, subprocess_runner=runner
    )
    run_dir = Path(result.allocated_run_dir)
    assert result.status == "wrapper_failed"
    assert calls["live_verify"] == 1
    assert not (
        run_dir / claim.GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME
    ).exists()
    assert not (run_dir / "status.json").exists()
    assert classify_completed_run_candidate(str(run_dir))[0] is False


def test_real_wrapper_replay_after_claim_is_refused(
    allocation_case, monkeypatch
):
    request, _plan = allocation_case
    runner, _calls = _real_wrapper_runner(monkeypatch)
    first = orchestration.run_guided_startup_to_wrapper(
        request=request, subprocess_runner=runner
    )
    assert first.status == "wrapper_started"
    replay = runner(first.wrapper_command)
    assert replay.started is True
    assert replay.completed is True
    assert replay.returncode != 0
    run_dir = Path(first.allocated_run_dir)
    assert (run_dir / claim.GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME).is_file()
    assert classify_completed_run_candidate(str(run_dir))[0] is False


def test_wrapper_stop_hook_is_internal_and_inactive_by_default():
    source = Path(wrapper.__file__).read_text(encoding="utf-8")
    assert wrapper._GUIDED_TEST_STOP_AFTER_INITIAL_STATUS is None
    assert "_GUIDED_TEST_STOP_AFTER_INITIAL_STATUS = None" in source
    assert "--guided-test-stop" not in source
    assert "gui." not in source
