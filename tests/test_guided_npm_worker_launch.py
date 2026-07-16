from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import photometry_pipeline.guided_npm_worker_entry as entry_module
import photometry_pipeline.guided_npm_worker_launch as launch_module
from photometry_pipeline.guided_npm_worker_entry import (
    GUIDED_NPM_WORKER_SMOKE_ARGUMENT,
    build_guided_npm_pipeline_runtime,
    load_verified_guided_npm_worker_request,
    run_guided_npm_worker,
)
from photometry_pipeline.guided_npm_worker_launch import (
    GUIDED_NPM_LAUNCHER_KIND,
    GUIDED_NPM_LAUNCH_CONTEXT_ARGUMENT,
    GUIDED_NPM_WORKER_ENTRY_MODULE,
    GUIDED_NPM_WORKER_REQUEST_ARGUMENT,
    GuidedNpmLaunchedWorkerRuntime,
    GuidedNpmPostLaunchRuntime,
    GuidedNpmStartedProcess,
    GuidedNpmStartedProcessRuntime,
    GuidedNpmWorkerExecutionStartReceipt,
    GuidedNpmWorkerLaunchCancelled,
    GuidedNpmWorkerLaunchFailure,
    GuidedNpmWorkerLaunchInvocation,
    GuidedNpmWorkerPostLaunchFailure,
    build_guided_npm_worker_launch_invocation,
    compute_guided_npm_worker_execution_start_receipt_identity,
    compute_guided_npm_worker_launch_invocation_identity,
    launch_guided_npm_worker,
    launch_guided_npm_worker_runtime,
    verify_guided_npm_worker_execution_start_receipt,
    verify_guided_npm_worker_launch_invocation,
)
from photometry_pipeline.guided_npm_worker_prelaunch_claim import (
    GuidedNpmWorkerPrelaunchClaim,
    compute_guided_npm_worker_prelaunch_claim_identity,
)
from photometry_pipeline.guided_npm_worker_acknowledgement import (
    GuidedNpmLaunchContextCleanupError,
    expected_guided_npm_launch_context_path,
    read_guided_npm_worker_launch_context,
)
from tests.test_guided_npm_worker_prelaunch_claim import (
    _claim,
    _different_build,
)


def _valid_claim(tmp_path: Path) -> GuidedNpmWorkerPrelaunchClaim:
    _, _, claim = _claim(tmp_path)
    assert isinstance(claim, GuidedNpmWorkerPrelaunchClaim)
    return claim


def _invocation(claim):
    return build_guided_npm_worker_launch_invocation(
        claim,
        current_application_build_identity=claim.application_build_identity,
    )


def _reidentify_invocation(invocation, **changes):
    changed = replace(
        invocation,
        **changes,
        canonical_launch_invocation_identity="0" * 64,
    )
    return replace(
        changed,
        canonical_launch_invocation_identity=(
            compute_guided_npm_worker_launch_invocation_identity(changed)
        ),
    )


def _receipt(claim, invocation, pid=4242):
    started = GuidedNpmStartedProcess(pid, GUIDED_NPM_LAUNCHER_KIND)
    return launch_module._build_execution_start_receipt(claim, invocation, started)


def _reidentify_receipt(receipt, **changes):
    changed = replace(
        receipt,
        **changes,
        canonical_execution_start_receipt_identity="0" * 64,
    )
    return replace(
        changed,
        canonical_execution_start_receipt_identity=(
            compute_guided_npm_worker_execution_start_receipt_identity(changed)
        ),
    )


def test_invocation_is_deterministic_frozen_and_exact(tmp_path):
    claim = _valid_claim(tmp_path)
    first = _invocation(claim)
    second = _invocation(claim)
    assert first == second
    assert first.executable_path == sys.executable
    assert first.argument_vector == (
        sys.executable,
        "-m",
        GUIDED_NPM_WORKER_ENTRY_MODULE,
        GUIDED_NPM_WORKER_REQUEST_ARGUMENT,
        claim.worker_request_artifact_path,
        GUIDED_NPM_LAUNCH_CONTEXT_ARGUMENT,
        expected_guided_npm_launch_context_path(claim.run_directory_path),
    )
    assert first.working_directory_path == os.path.dirname(
        os.path.dirname(launch_module.__file__)
    )
    assert first.environment_policy == "inherit_unchanged"
    assert first.shell is False
    assert (
        first.invocation_status,
        first.launch_status,
        first.execution_status,
        first.completion_status,
    ) == ("constructed_for_launch", "not_launched", "not_started", "not_available")
    verify_guided_npm_worker_launch_invocation(first, claim)
    with pytest.raises(FrozenInstanceError):
        first.shell = True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_prelaunch_claim_identity", "1" * 64),
        ("source_prelaunch_freshness_evidence_identity", "1" * 64),
        ("source_worker_request_identity", "1" * 64),
        ("source_execution_request_identity", "1" * 64),
        ("guided_plan_identity", "other-plan"),
        ("validation_revision", 999),
        ("execution_mode", "tonic"),
        ("executable_path", os.path.abspath("other-python")),
        ("working_directory_path", os.path.abspath("other-cwd")),
        ("worker_request_artifact_path", os.path.abspath("other-worker.json")),
        ("launch_context_artifact_path", os.path.abspath("other-context.json")),
        ("run_directory_path", os.path.abspath("other-run")),
        ("environment_policy", "replace"),
        ("shell", True),
        ("invocation_status", "ready"),
        ("launch_status", "process_created"),
        ("execution_status", "started"),
        ("completion_status", "complete"),
    ],
)
def test_outer_reidentified_invocation_field_tampering_refuses(tmp_path, field, value):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    if field == "execution_mode" and invocation.execution_mode == value:
        value = "phasic"
    if field in {"worker_request_artifact_path", "launch_context_artifact_path", "run_directory_path"}:
        style = claim.worker_request.execution_request.output_runtime_projection.output_base_path_style
        if style == "windows_drive":
            value = r"C:\Other\guided_npm_worker_request.json" if field.startswith("worker") else r"C:\Other\guided_npm_launch_context.json" if field.startswith("launch") else r"C:\Other"
        else:
            value = "/other/guided_npm_worker_request.json" if field.startswith("worker") else "/other/guided_npm_launch_context.json" if field.startswith("launch") else "/other"
    with pytest.raises(ValueError):
        verify_guided_npm_worker_launch_invocation(
            _reidentify_invocation(invocation, **{field: value}), claim
        )


@pytest.mark.parametrize("kind", ["insert", "remove", "reorder", "duplicate", "alternate"])
def test_argument_vector_tampering_refuses(tmp_path, kind):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    argv = list(invocation.argument_vector)
    if kind == "insert":
        argv.insert(3, "--debug")
    elif kind == "remove":
        argv.pop(1)
    elif kind == "reorder":
        argv[1], argv[2] = argv[2], argv[1]
    elif kind == "duplicate":
        argv.extend((GUIDED_NPM_WORKER_REQUEST_ARGUMENT, claim.worker_request_artifact_path))
    else:
        argv[-1] = os.path.abspath("alternate-worker.json")
    changed = _reidentify_invocation(invocation, argument_vector=tuple(argv))
    with pytest.raises(ValueError):
        verify_guided_npm_worker_launch_invocation(changed, claim)


def test_invocation_build_and_identity_tampering_refuse(tmp_path):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    changed_build = _reidentify_invocation(
        invocation,
        application_build_identity=_different_build(invocation.application_build_identity),
    )
    with pytest.raises(ValueError):
        verify_guided_npm_worker_launch_invocation(changed_build, claim)
    with pytest.raises(ValueError, match="launch_invocation_identity_mismatch"):
        verify_guided_npm_worker_launch_invocation(
            replace(invocation, canonical_launch_invocation_identity="1" * 64), claim
        )


def test_fake_launch_calls_once_and_returns_truthful_receipt(tmp_path):
    claim = _valid_claim(tmp_path)
    expected = _invocation(claim)
    calls = []

    def launcher(argv, *, cwd, shell):
        calls.append((argv, cwd, shell))
        return GuidedNpmStartedProcess(43210, GUIDED_NPM_LAUNCHER_KIND)

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=launcher,
    )
    assert isinstance(result, GuidedNpmWorkerExecutionStartReceipt)
    assert calls == [(expected.argument_vector, expected.working_directory_path, False)]
    assert (
        result.launch_status,
        result.execution_status,
        result.completion_status,
        result.consumed_authority_status,
    ) == ("process_created", "start_unconfirmed", "not_available", "not_available")
    assert result.process_id == 43210
    verify_guided_npm_worker_execution_start_receipt(result, claim, expected)


def test_real_popen_smoke_uses_repository_owned_child_mode():
    argv = (
        sys.executable,
        "-m",
        GUIDED_NPM_WORKER_ENTRY_MODULE,
        GUIDED_NPM_WORKER_SMOKE_ARGUMENT,
    )
    started = launch_module._subprocess_popen_launcher(
        argv, cwd=launch_module._application_root(), shell=False
    )
    assert isinstance(started.process_id, int) and started.process_id > 0
    assert started.launcher_kind == GUIDED_NPM_LAUNCHER_KIND
    assert GUIDED_NPM_WORKER_SMOKE_ARGUMENT not in _invocation_source_shape()


def _invocation_source_shape():
    return (
        "-m",
        GUIDED_NPM_WORKER_ENTRY_MODULE,
        GUIDED_NPM_WORKER_REQUEST_ARGUMENT,
    )


def test_worker_entry_maps_authority_into_real_pipeline_path(tmp_path):
    claim = _valid_claim(tmp_path)
    worker = load_verified_guided_npm_worker_request(
        claim.worker_request_artifact_path,
        current_application_build_identity=claim.application_build_identity,
    )
    runtime = build_guided_npm_pipeline_runtime(worker)
    assert runtime["input_dir"] == worker.execution_request.source_runtime_projection.source_root_canonical
    assert runtime["output_dir"] == worker.run_directory_path
    assert runtime["force_format"] == "npm"
    assert runtime["selected_canonical_roi_ids"] == tuple(
        worker.execution_request.roi_runtime_projection.selected_canonical_roi_ids
    )
    observed = {}

    class FakePipeline:
        def __init__(self, config, **kwargs):
            observed["init"] = (config, kwargs)

        def run_guided_npm_authorized(self, *args, **kwargs):
            observed["run"] = (args, kwargs)

    run_guided_npm_worker(worker, pipeline_factory=FakePipeline)
    assert observed["init"][1]["mode"] == worker.execution_request.execution_mode
    assert observed["run"][0] == (
        runtime["authorized_runtime"],
        runtime["output_dir"],
    )


def test_invalid_claim_and_wrong_build_refuse_before_launcher(tmp_path):
    claim = _valid_claim(tmp_path)
    calls = []
    invalid = replace(claim, claim_status="invalid")
    result = launch_guided_npm_worker(
        invalid,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert result.blocking_issues[0].category == "prelaunch_claim_state_invalid"
    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=_different_build(claim.application_build_identity),
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert calls == []


@pytest.mark.parametrize(
    "category",
    [
        "launch_executable_invalid",
        "launch_entry_point_missing",
        "launch_working_directory_invalid",
    ],
)
def test_launch_filesystem_gate_failure_prevents_process_creation(
    monkeypatch, tmp_path, category
):
    claim = _valid_claim(tmp_path)
    calls = []
    monkeypatch.setattr(
        launch_module,
        "_verify_launch_filesystem",
        lambda *_: (_ for _ in ()).throw(ValueError(category)),
    )
    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert result.blocking_issues[0].category == category
    assert calls == []


@pytest.mark.parametrize(
    ("target", "category"),
    [
        ("worker", "launch_worker_artifact_changed"),
        ("startup", "launch_startup_artifact_changed"),
        ("source", "launch_source_freshness_changed"),
    ],
)
def test_final_gate_detects_changed_authority_before_launcher(tmp_path, target, category):
    claim = _valid_claim(tmp_path)
    if target == "worker":
        path = Path(claim.worker_request_artifact_path)
    elif target == "startup":
        path = Path(claim.worker_request.startup_artifact_path)
    else:
        path = Path(
            claim.worker_request.execution_request.source_runtime_projection.ordered_source_paths[0]
        )
    path.write_bytes(path.read_bytes() + b"changed")
    calls = []
    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert result.blocking_issues[0].category == category
    assert calls == []


@pytest.mark.parametrize("exc", [FileNotFoundError(), PermissionError(), OSError("boom")])
def test_process_creation_exception_returns_failure_without_retry(tmp_path, exc):
    claim = _valid_claim(tmp_path)
    calls = []

    def launcher(*args, **kwargs):
        calls.append(1)
        raise exc

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=launcher,
    )
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert result.blocking_issues[0].category == "process_creation_failed"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "started",
    [
        GuidedNpmStartedProcess(0, GUIDED_NPM_LAUNCHER_KIND),
        GuidedNpmStartedProcess(-1, GUIDED_NPM_LAUNCHER_KIND),
        GuidedNpmStartedProcess(True, GUIDED_NPM_LAUNCHER_KIND),
        GuidedNpmStartedProcess(12, "other"),
        object(),
    ],
)
def test_invalid_process_identity_is_post_launch_indeterminate(tmp_path, started):
    claim = _valid_claim(tmp_path)
    calls = []

    def launcher(*args, **kwargs):
        calls.append(1)
        return started

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=launcher,
    )
    assert isinstance(result, GuidedNpmWorkerPostLaunchFailure)
    assert result.blocking_issues[0].category == "process_identity_invalid"
    assert len(calls) == 1


@pytest.mark.parametrize("cancel_at", [1, 2, 3, 4, 5])
def test_cancellation_before_process_call_launches_nothing(tmp_path, cancel_at):
    claim = _valid_claim(tmp_path)
    checks = 0
    calls = []

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= cancel_at

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        cancellation_check=cancelled,
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert isinstance(result, GuidedNpmWorkerLaunchCancelled)
    assert calls == []


def test_cancellation_after_process_creation_still_returns_receipt(tmp_path):
    claim = _valid_claim(tmp_path)
    state = {"cancelled": False}
    calls = []

    def launcher(*args, **kwargs):
        calls.append(1)
        state["cancelled"] = True
        return GuidedNpmStartedProcess(9876, GUIDED_NPM_LAUNCHER_KIND)

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        cancellation_check=lambda: state["cancelled"],
        process_launcher=launcher,
    )
    assert isinstance(result, GuidedNpmWorkerExecutionStartReceipt)
    assert result.process_id == 9876
    assert calls == [1]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_prelaunch_claim_identity", "1" * 64),
        ("source_launch_invocation_identity", "1" * 64),
        ("source_worker_request_identity", "1" * 64),
        ("source_execution_request_identity", "1" * 64),
        ("source_materialization_receipt_identity", "1" * 64),
        ("source_prelaunch_freshness_evidence_identity", "1" * 64),
        ("guided_plan_identity", "other"),
        ("validation_revision", 999),
        ("execution_mode", "tonic"),
        ("executable_path", os.path.abspath("other-python")),
        ("argument_vector", ("other",)),
        ("working_directory_path", os.path.abspath("other-cwd")),
        ("worker_request_artifact_path", os.path.abspath("other-worker.json")),
        ("run_directory_path", os.path.abspath("other-run")),
        ("process_id", 0),
        ("launcher_kind", "other"),
        ("launch_status", "running"),
        ("execution_status", "started"),
        ("completion_status", "complete"),
        ("consumed_authority_status", "consumed"),
    ],
)
def test_outer_reidentified_receipt_tampering_refuses(tmp_path, field, value):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    receipt = _receipt(claim, invocation)
    if field == "execution_mode" and receipt.execution_mode == value:
        value = "phasic"
    with pytest.raises(ValueError):
        verify_guided_npm_worker_execution_start_receipt(
            _reidentify_receipt(receipt, **{field: value}), claim, invocation
        )


def test_receipt_build_tampering_refuses(tmp_path):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    receipt = _receipt(claim, invocation)
    changed = _reidentify_receipt(
        receipt,
        application_build_identity=_different_build(receipt.application_build_identity),
    )
    with pytest.raises(ValueError):
        verify_guided_npm_worker_execution_start_receipt(changed, claim, invocation)
    with pytest.raises(ValueError, match="execution_start_receipt_identity_mismatch"):
        verify_guided_npm_worker_execution_start_receipt(
            replace(receipt, canonical_execution_start_receipt_identity="1" * 64),
            claim,
            invocation,
        )


def test_different_authorities_do_not_cross_verify(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    claim_a = _valid_claim(tmp_path / "a")
    claim_b = _valid_claim(tmp_path / "b")
    invocation_a, invocation_b = _invocation(claim_a), _invocation(claim_b)
    receipt_a = _receipt(claim_a, invocation_a)
    with pytest.raises(ValueError):
        verify_guided_npm_worker_launch_invocation(invocation_a, claim_b)
    with pytest.raises(ValueError):
        verify_guided_npm_worker_execution_start_receipt(receipt_a, claim_b, invocation_a)
    with pytest.raises(ValueError):
        verify_guided_npm_worker_execution_start_receipt(receipt_a, claim_a, invocation_b)


def test_pure_verifiers_access_no_filesystem_or_process(monkeypatch, tmp_path):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    receipt = _receipt(claim, invocation)
    for name in ("exists", "stat", "resolve", "open", "read_bytes"):
        monkeypatch.setattr(Path, name, lambda *a, _name=name, **k: pytest.fail(f"Path.{_name}"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Popen"))
    verify_guided_npm_worker_launch_invocation(invocation, claim)
    verify_guided_npm_worker_execution_start_receipt(receipt, claim, invocation)


def test_parent_launch_does_not_reinterpret_or_handle_completion(monkeypatch, tmp_path):
    claim = _valid_claim(tmp_path)
    source = inspect.getsource(launch_module)
    for forbidden in (
        "pandas",
        "read_csv",
        "Pipeline(",
        ".wait(",
        ".communicate(",
        ".poll(",
        "completed_run",
        "terminal_receipt",
    ):
        assert forbidden not in source
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("subprocess.run"))
    monkeypatch.setattr(
        entry_module,
        "build_guided_npm_pipeline_runtime",
        lambda *a, **k: pytest.fail("child scientific mapping reached from parent"),
    )
    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda *a, **k: GuidedNpmStartedProcess(
            7654, GUIDED_NPM_LAUNCHER_KIND
        ),
    )
    assert isinstance(result, GuidedNpmWorkerExecutionStartReceipt)


def test_launch_failure_and_cancellation_preserve_authority_artifacts(tmp_path):
    claim = _valid_claim(tmp_path)
    paths = [
        Path(claim.worker_request_artifact_path),
        Path(claim.worker_request.startup_artifact_path),
        *map(
            Path,
            claim.worker_request.execution_request.source_runtime_projection.ordered_source_paths,
        ),
    ]
    before = {path: path.read_bytes() for path in paths}
    failed = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda *a, **k: (_ for _ in ()).throw(OSError("no")),
    )
    cancelled = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        cancellation_check=lambda: True,
    )
    assert isinstance(failed, GuidedNpmWorkerLaunchFailure)
    assert isinstance(cancelled, GuidedNpmWorkerLaunchCancelled)
    assert {path: path.read_bytes() for path in paths} == before


def _persist_with_flag(monkeypatch, state):
    original_persist = launch_module.persist_guided_npm_worker_launch_context

    def wrapped_persist(context):
        path = original_persist(context)
        state["persisted"] = True
        return path

    monkeypatch.setattr(
        launch_module, "persist_guided_npm_worker_launch_context", wrapped_persist
    )


def test_persist_then_cancellation_check_then_launcher_ordering(monkeypatch, tmp_path):
    claim = _valid_claim(tmp_path)
    events = []
    state = {"persisted": False}
    original_persist = launch_module.persist_guided_npm_worker_launch_context

    def wrapped_persist(context):
        path = original_persist(context)
        state["persisted"] = True
        events.append("context_persisted")
        return path

    monkeypatch.setattr(
        launch_module, "persist_guided_npm_worker_launch_context", wrapped_persist
    )

    def cancelled():
        if state["persisted"]:
            events.append("cancellation_checked_after_persistence")
        return False

    def launcher(argv, *, cwd, shell):
        events.append("launcher_called")
        return GuidedNpmStartedProcess(1234, GUIDED_NPM_LAUNCHER_KIND)

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        cancellation_check=cancelled,
        process_launcher=launcher,
    )
    assert isinstance(result, GuidedNpmWorkerExecutionStartReceipt)
    assert events == [
        "context_persisted",
        "cancellation_checked_after_persistence",
        "launcher_called",
    ]


def test_cancellation_immediately_after_persistence_cleans_up_and_launches_nothing(
    monkeypatch, tmp_path
):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    state = {"persisted": False}
    _persist_with_flag(monkeypatch, state)
    calls = []

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        cancellation_check=lambda: state["persisted"],
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert isinstance(result, GuidedNpmWorkerLaunchCancelled)
    assert calls == []
    assert not Path(invocation.launch_context_artifact_path).exists()


def test_retry_after_post_persistence_cancellation_cleanup_succeeds(monkeypatch, tmp_path):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    state = {"persisted": False}
    _persist_with_flag(monkeypatch, state)
    calls = []

    first = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        cancellation_check=lambda: state["persisted"],
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert isinstance(first, GuidedNpmWorkerLaunchCancelled)
    assert calls == []
    assert not Path(invocation.launch_context_artifact_path).exists()

    monkeypatch.undo()

    second = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda argv, **kwargs: GuidedNpmStartedProcess(
            5555, GUIDED_NPM_LAUNCHER_KIND
        ),
    )
    assert isinstance(second, GuidedNpmWorkerExecutionStartReceipt)
    assert second.process_id == 5555
    assert Path(invocation.launch_context_artifact_path).is_file()


@pytest.mark.parametrize("exc", [FileNotFoundError(), PermissionError(), OSError("boom")])
def test_launcher_exception_cleans_up_persisted_context(tmp_path, exc):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    calls = []

    def launcher(*args, **kwargs):
        calls.append(1)
        raise exc

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=launcher,
    )
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert result.blocking_issues[0].category == "process_creation_failed"
    assert len(calls) == 1
    assert not Path(invocation.launch_context_artifact_path).exists()


@pytest.mark.parametrize(
    "started",
    [
        GuidedNpmStartedProcess(0, GUIDED_NPM_LAUNCHER_KIND),
        GuidedNpmStartedProcess(-1, GUIDED_NPM_LAUNCHER_KIND),
        GuidedNpmStartedProcess(True, GUIDED_NPM_LAUNCHER_KIND),
        GuidedNpmStartedProcess(12, "other"),
        object(),
    ],
)
def test_malformed_launcher_return_preserves_launch_context(tmp_path, started):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda *a, **k: started,
    )
    assert isinstance(result, GuidedNpmWorkerPostLaunchFailure)
    path = Path(invocation.launch_context_artifact_path)
    assert path.is_file()
    context = read_guided_npm_worker_launch_context(
        os.fspath(path), worker_request=claim.worker_request
    )
    assert (
        context.source_launch_invocation_identity
        == invocation.canonical_launch_invocation_identity
    )


def test_post_launch_receipt_failure_preserves_launch_context(monkeypatch, tmp_path):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    real_verify = launch_module.verify_guided_npm_worker_execution_start_receipt

    def flaky_verify(receipt, prelaunch_claim, invocation_):
        if receipt.process_id == 4321:
            raise ValueError("post_launch_receipt_forced_failure")
        return real_verify(receipt, prelaunch_claim, invocation_)

    monkeypatch.setattr(
        launch_module, "verify_guided_npm_worker_execution_start_receipt", flaky_verify
    )

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda *a, **k: GuidedNpmStartedProcess(
            4321, GUIDED_NPM_LAUNCHER_KIND
        ),
    )
    assert isinstance(result, GuidedNpmWorkerPostLaunchFailure)
    assert result.blocking_issues[0].category == "process_created_receipt_failed"
    path = Path(invocation.launch_context_artifact_path)
    assert path.is_file()
    context = read_guided_npm_worker_launch_context(
        os.fspath(path), worker_request=claim.worker_request
    )
    assert (
        context.source_launch_invocation_identity
        == invocation.canonical_launch_invocation_identity
    )


def test_cleanup_failure_after_cancellation_reports_truthfully(monkeypatch, tmp_path):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    state = {"persisted": False}
    original_persist = launch_module.persist_guided_npm_worker_launch_context

    def tampering_persist(context):
        path = original_persist(context)
        data = json.loads(Path(path).read_bytes())
        data["source_worker_request_identity"] = "1" * 64
        Path(path).write_bytes(
            (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        state["persisted"] = True
        return path

    monkeypatch.setattr(
        launch_module, "persist_guided_npm_worker_launch_context", tampering_persist
    )
    calls = []

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        cancellation_check=lambda: state["persisted"],
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert result.blocking_issues[0].category == "launch_context_cleanup_failed"
    assert result.process_creation_status == "not_created"
    assert calls == []
    assert Path(invocation.launch_context_artifact_path).is_file()


def test_cancellation_cleanup_initial_stat_permission_failure_reports_cleanup_failed(
    monkeypatch, tmp_path
):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    context_path = invocation.launch_context_artifact_path
    state = {"persisted": False}
    _persist_with_flag(monkeypatch, state)
    original_stat = Path.stat

    def patched_stat(self, *args, **kwargs):
        if state["persisted"] and os.fspath(self) == context_path:
            raise PermissionError("denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", patched_stat)
    calls = []

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        cancellation_check=lambda: state["persisted"],
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert calls == []
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert result.blocking_issues[0].category == "launch_context_cleanup_failed"
    assert result.process_creation_status == "not_created"
    assert not isinstance(result, GuidedNpmWorkerLaunchCancelled)


def test_launcher_exception_cleanup_initial_stat_generic_oserror_reports_cleanup_failed(
    monkeypatch, tmp_path
):
    claim = _valid_claim(tmp_path)
    invocation = _invocation(claim)
    context_path = invocation.launch_context_artifact_path
    state = {"persisted": False}
    _persist_with_flag(monkeypatch, state)
    original_stat = Path.stat

    def patched_stat(self, *args, **kwargs):
        if state["persisted"] and os.fspath(self) == context_path:
            raise OSError("boom")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", patched_stat)
    calls = []

    def launcher(*args, **kwargs):
        calls.append(1)
        raise FileNotFoundError()

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=launcher,
    )
    assert len(calls) == 1
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert result.blocking_issues[0].category == "launch_context_cleanup_failed"
    assert result.process_creation_status == "not_created"


def test_post_persistence_cancellation_preserves_authority_artifacts(monkeypatch, tmp_path):
    claim = _valid_claim(tmp_path)
    paths = [
        Path(claim.worker_request_artifact_path),
        Path(claim.worker_request.startup_artifact_path),
        *map(
            Path,
            claim.worker_request.execution_request.source_runtime_projection.ordered_source_paths,
        ),
    ]
    before = {path: path.read_bytes() for path in paths}
    state = {"persisted": False}
    _persist_with_flag(monkeypatch, state)

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        cancellation_check=lambda: state["persisted"],
        process_launcher=lambda *a, **k: pytest.fail("launcher should not be called"),
    )
    assert isinstance(result, GuidedNpmWorkerLaunchCancelled)
    assert {path: path.read_bytes() for path in paths} == before


# ---------------------------------------------------------------------------
# B2-D2B: launch_guided_npm_worker_runtime (retains the exact process handle)
# ---------------------------------------------------------------------------


class _FakeHandle:
    def __init__(self, pid, *, wait_result=0):
        self.pid = pid
        self._wait_result = wait_result

    def wait(self, timeout=None):
        return self._wait_result


def test_launch_runtime_delegates_unmodified_and_retains_handle(tmp_path):
    claim = _valid_claim(tmp_path)
    expected = _invocation(claim)
    calls = []

    def fake_launcher(argv, *, cwd, shell):
        calls.append((argv, cwd, shell))
        return GuidedNpmStartedProcessRuntime(54321, GUIDED_NPM_LAUNCHER_KIND, _FakeHandle(54321))

    result = launch_guided_npm_worker_runtime(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=fake_launcher,
    )
    assert isinstance(result, GuidedNpmLaunchedWorkerRuntime)
    assert calls == [(expected.argument_vector, expected.working_directory_path, False)]
    assert result.process_handle.pid == 54321
    assert result.execution_start_receipt.process_id == 54321
    assert result.prelaunch_claim is claim
    assert result.launch_invocation == expected
    verify_guided_npm_worker_execution_start_receipt(
        result.execution_start_receipt, claim, result.launch_invocation
    )
    # Same durability standard as launch_guided_npm_worker: the launch context
    # is genuinely persisted on disk, not merely constructed in memory.
    assert Path(result.launch_context.consumed_authority_receipt_path).parent.is_dir()


def test_launch_runtime_started_process_runtime_requires_matching_pid():
    with pytest.raises(ValueError, match="launch_runtime_started_process_invalid"):
        GuidedNpmStartedProcessRuntime(111, GUIDED_NPM_LAUNCHER_KIND, _FakeHandle(222))


def test_launch_runtime_passes_through_non_success_results_unchanged(tmp_path):
    claim = _valid_claim(tmp_path)
    invalid = replace(claim, claim_status="invalid")
    result = launch_guided_npm_worker_runtime(
        invalid,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=lambda *a, **k: pytest.fail("launcher should not be called"),
    )
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)


def test_launch_runtime_cancellation_passes_through_and_launches_nothing(tmp_path):
    claim = _valid_claim(tmp_path)
    calls = []
    result = launch_guided_npm_worker_runtime(
        claim,
        current_application_build_identity=claim.application_build_identity,
        cancellation_check=lambda: True,
        process_launcher=lambda *a, **k: calls.append(1),
    )
    assert isinstance(result, GuidedNpmWorkerLaunchCancelled)
    assert calls == []


def test_launch_runtime_malformed_runtime_launcher_return_refuses(tmp_path):
    claim = _valid_claim(tmp_path)

    def bad_launcher(argv, *, cwd, shell):
        return GuidedNpmStartedProcess(4242, GUIDED_NPM_LAUNCHER_KIND)  # wrong type

    result = launch_guided_npm_worker_runtime(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=bad_launcher,
    )
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert result.blocking_issues[0].category == "process_creation_failed"


def test_launch_runtime_existing_launch_guided_npm_worker_unaffected(tmp_path):
    """launch_guided_npm_worker itself is completely unmodified by this addition."""
    claim = _valid_claim(tmp_path)
    calls = []

    def launcher(argv, *, cwd, shell):
        calls.append((argv, cwd, shell))
        return GuidedNpmStartedProcess(43210, GUIDED_NPM_LAUNCHER_KIND)

    result = launch_guided_npm_worker(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=launcher,
    )
    assert isinstance(result, GuidedNpmWorkerExecutionStartReceipt)
    assert result.process_id == 43210
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# B2-D2B narrow follow-up: never discard a handle after possible process
# creation (post-launch runtime preservation)
# ---------------------------------------------------------------------------


def test_launch_runtime_preserves_handle_on_malformed_process_identity(tmp_path):
    claim = _valid_claim(tmp_path)

    def bad_kind_launcher(argv, *, cwd, shell):
        return GuidedNpmStartedProcessRuntime(4242, "wrong_kind", _FakeHandle(4242))

    result = launch_guided_npm_worker_runtime(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=bad_kind_launcher,
    )
    assert isinstance(result, GuidedNpmPostLaunchRuntime)
    assert result.process_handle.pid == 4242
    assert result.process_id == 4242
    assert result.launch_failure.blocking_issues[0].category == "process_identity_invalid"
    assert result.execution_start_receipt is None


def test_launch_runtime_preserves_handle_on_start_receipt_verification_failure(
    monkeypatch, tmp_path
):
    claim = _valid_claim(tmp_path)
    handle = _FakeHandle(54321)
    real_verify = launch_module.verify_guided_npm_worker_execution_start_receipt

    def flaky_verify(receipt, prelaunch_claim, invocation):
        # The provisional pre-validation receipt (built with a fake PID of 1,
        # before launch-context persistence) must keep succeeding; only the
        # real post-launcher receipt (bound to the real captured PID) fails.
        if receipt.process_id == 54321:
            raise ValueError("forced_verification_failure")
        return real_verify(receipt, prelaunch_claim, invocation)

    def good_launcher(argv, *, cwd, shell):
        return GuidedNpmStartedProcessRuntime(54321, GUIDED_NPM_LAUNCHER_KIND, handle)

    monkeypatch.setattr(
        launch_module, "verify_guided_npm_worker_execution_start_receipt", flaky_verify
    )
    result = launch_guided_npm_worker_runtime(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=good_launcher,
    )
    assert isinstance(result, GuidedNpmPostLaunchRuntime)
    assert result.process_handle is handle
    assert result.process_id == 54321
    assert result.launch_failure.blocking_issues[0].category == "process_created_receipt_failed"
    assert result.execution_start_receipt is None


def test_launch_runtime_no_process_failure_remains_ordinary(tmp_path):
    claim = _valid_claim(tmp_path)

    def raising_launcher(argv, *, cwd, shell):
        raise OSError("no process")

    result = launch_guided_npm_worker_runtime(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=raising_launcher,
    )
    assert isinstance(result, GuidedNpmWorkerLaunchFailure)
    assert result.blocking_issues[0].category == "process_creation_failed"


def test_launch_runtime_retained_handle_can_be_reconciled_via_post_launch_path(
    monkeypatch, tmp_path
):
    from photometry_pipeline.guided_npm_worker_reconciliation import (
        OUTCOME_POST_LAUNCH_EVIDENCE_FAILED,
        reconcile_guided_npm_post_launch_runtime,
    )

    claim = _valid_claim(tmp_path)
    handle = _FakeHandle(9999, wait_result=0)
    real_verify = launch_module.verify_guided_npm_worker_execution_start_receipt

    def flaky_verify(receipt, prelaunch_claim, invocation):
        if receipt.process_id == 9999:
            raise ValueError("forced_verification_failure")
        return real_verify(receipt, prelaunch_claim, invocation)

    def good_launcher(argv, *, cwd, shell):
        return GuidedNpmStartedProcessRuntime(9999, GUIDED_NPM_LAUNCHER_KIND, handle)

    monkeypatch.setattr(
        launch_module, "verify_guided_npm_worker_execution_start_receipt", flaky_verify
    )
    runtime = launch_guided_npm_worker_runtime(
        claim,
        current_application_build_identity=claim.application_build_identity,
        process_launcher=good_launcher,
    )
    assert isinstance(runtime, GuidedNpmPostLaunchRuntime)
    result = reconcile_guided_npm_post_launch_runtime(runtime)
    assert result.final_outcome == OUTCOME_POST_LAUNCH_EVIDENCE_FAILED
    assert result.observed_exit_code == 0
    assert result.observed_process_id == 9999
    assert result.launch_failure_category == "process_created_receipt_failed"
    assert result.terminal_evidence_present is False
    assert result.consumed_authority_evidence_present is False
