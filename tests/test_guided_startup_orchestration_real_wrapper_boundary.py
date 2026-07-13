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
            return object(), object()

        monkeypatch.setattr(
            wrapper, "verify_guided_manifest_before_output", live_verify
        )
        monkeypatch.setattr(
            wrapper,
            "verify_guided_normalized_recording_description_before_output",
            lambda _args, _facts, _verified: None,
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
        "guided_normalized_recording_description.json",
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


class _StopAfterPhasicCommandCapture(RuntimeError):
    pass


def _real_wrapper_runner_capture_phasic_cmd(monkeypatch):
    """Run the real wrapper far enough to actually construct cmd_phasic (the
    real argv it would hand to analyze_photometry.py) and capture it, then
    stop before any subprocess is actually spawned. Unlike
    _real_wrapper_runner (which forbids run_cmd entirely to stop at the
    initial-status boundary), this lets validate/config/manifest
    materialization run for real and only intercepts at the analysis
    subprocess call itself."""
    captured = {}

    def runner(command):
        monkeypatch.setattr(wrapper.sys, "argv", [command[1], *command[2:]])
        monkeypatch.setattr(
            wrapper,
            "verify_guided_manifest_before_output",
            lambda _args: (object(), object()),
        )
        monkeypatch.setattr(
            wrapper,
            "verify_guided_normalized_recording_description_before_output",
            lambda _args, _facts, _verified: None,
        )

        def capture_and_stop(cmd, roi_label=None):
            captured["cmd_phasic"] = cmd
            raise _StopAfterPhasicCommandCapture()

        monkeypatch.setattr(wrapper, "run_cmd", capture_and_stop)
        try:
            wrapper.main()
        except _StopAfterPhasicCommandCapture:
            pass
        except SystemExit:
            # The wrapper's own broad exception handling catches the stop
            # signal (it cannot distinguish it from a real failure),
            # finalizes an error status, and exits -- cmd_phasic was
            # already captured above before that unwinding began.
            pass
        if "cmd_phasic" not in captured:
            raise AssertionError("real wrapper never reached run_cmd(cmd_phasic)")
        return orchestration.GuidedWrapperProcessResult(
            returncode=None,
            stdout="stopped after phasic command capture",
            stderr="",
            command=command,
            started=True,
            completed=False,
        )

    return runner, captured


def test_real_wrapper_phasic_command_excludes_overwrite_and_includes_sessions_per_hour(
    allocation_case, monkeypatch
):
    """4J16k12: proves, against the real (unmocked) wrapper argv-construction
    logic in tools/run_full_pipeline_deliverables.py, that the actual
    cmd_phasic handed to analyze_photometry.py for a Guided manifest run:

    1. Does not include --overwrite (analyze_photometry.py explicitly
       refuses --overwrite together with --guided-candidate-manifest with
       "Error: unsupported internal Guided manifest execution state.",
       which made every real Guided run fail before analysis started).
    2. Does include --sessions-per-hour with the value threaded from the
       Guided production intent via the wrapper's own top-level
       --sessions-per-hour flag (guided_startup_transaction.py), proving
       the wrapper's internal resolved_sessions_per_hour is no longer None.
    """
    request, _plan = allocation_case
    assert (
        request.authorization_result.production_intent.acquisition.sessions_per_hour
        == 6
    )
    runner, captured = _real_wrapper_runner_capture_phasic_cmd(monkeypatch)
    result = orchestration.run_guided_startup_to_wrapper(
        request=request, subprocess_runner=runner
    )
    assert result.status == "wrapper_started"
    cmd_phasic = captured["cmd_phasic"]
    assert "--overwrite" not in cmd_phasic
    assert "--guided-candidate-manifest" in cmd_phasic
    assert "--sessions-per-hour" in cmd_phasic
    assert (
        cmd_phasic[cmd_phasic.index("--sessions-per-hour") + 1] == "6"
    )


def test_non_guided_phasic_command_still_includes_overwrite(tmp_path, monkeypatch):
    """4J16k12: the --overwrite fix must be scoped exactly to the Guided-
    manifest case. A plain, non-Guided CLI invocation (no
    --guided-candidate-manifest at all, matching every scenario in
    tests/test_full_pipeline_deliverables.py and any manual/non-Guided use
    of this wrapper) must still receive --overwrite in cmd_phasic, exactly
    as before this patch. This is a fast, direct check of the real
    (unmocked) argv-construction logic -- no synthetic dataset or Guided
    orchestration layer involved -- to avoid needing the very slow, heavy
    real-dataset fixtures used by the legacy full-pipeline suite."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    config_path = Path(__file__).resolve().parent / "qc_universal_config.yaml"

    captured = {}

    def capture_and_stop(cmd, roi_label=None):
        captured["cmd_phasic"] = cmd
        raise _StopAfterPhasicCommandCapture()

    monkeypatch.setattr(wrapper, "run_cmd", capture_and_stop)
    monkeypatch.setattr(
        wrapper.sys,
        "argv",
        [
            "run_full_pipeline_deliverables.py",
            "--input", str(input_dir),
            "--out", str(out_dir),
            "--config", str(config_path),
            "--format", "rwd",
            "--overwrite",
            "--sessions-per-hour", "2",
        ],
    )
    try:
        wrapper.main()
    except (_StopAfterPhasicCommandCapture, SystemExit):
        pass
    assert "cmd_phasic" in captured, "real wrapper never reached run_cmd(cmd_phasic)"
    cmd_phasic = captured["cmd_phasic"]
    assert "--overwrite" in cmd_phasic
    assert "--guided-candidate-manifest" not in cmd_phasic


def test_wrapper_stop_hook_is_internal_and_inactive_by_default():
    source = Path(wrapper.__file__).read_text(encoding="utf-8")
    assert wrapper._GUIDED_TEST_STOP_AFTER_INITIAL_STATUS is None
    assert "_GUIDED_TEST_STOP_AFTER_INITIAL_STATUS = None" in source
    assert "--guided-test-stop" not in source
    assert "gui." not in source
