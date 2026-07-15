"""Exact B2-C6B2B process creation from verified Guided NPM authority.

This module records only the operating-system process-creation boundary.  It
does not wait for the child, infer that it consumed authority, inspect outputs,
or claim numerical progress or completion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Protocol

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_startup_persistence import (
    verify_application_build_identity,
)
from photometry_pipeline.guided_npm_worker_prelaunch_claim import (
    GuidedNpmPrelaunchAuthorityLiveCancelled,
    GuidedNpmWorkerPrelaunchClaim,
    stored_paths_equal,
    verify_guided_npm_worker_prelaunch_authority_live,
    verify_guided_npm_worker_prelaunch_claim,
)
from photometry_pipeline.guided_production_mapping import ApplicationBuildIdentity


GUIDED_NPM_WORKER_LAUNCH_INVOCATION_SCHEMA_NAME = (
    "guided_npm_worker_launch_invocation"
)
GUIDED_NPM_WORKER_LAUNCH_INVOCATION_SCHEMA_VERSION = "v1"
GUIDED_NPM_WORKER_LAUNCH_INVOCATION_CONTRACT_VERSION = (
    "guided_npm_worker_launch_invocation.v1"
)
GUIDED_NPM_WORKER_EXECUTION_START_RECEIPT_SCHEMA_NAME = (
    "guided_npm_worker_execution_start_receipt"
)
GUIDED_NPM_WORKER_EXECUTION_START_RECEIPT_SCHEMA_VERSION = "v1"
GUIDED_NPM_WORKER_EXECUTION_START_RECEIPT_CONTRACT_VERSION = (
    "guided_npm_worker_execution_start_receipt.v1"
)
GUIDED_NPM_WORKER_ENTRY_MODULE = "photometry_pipeline.guided_npm_worker_entry"
GUIDED_NPM_WORKER_REQUEST_ARGUMENT = "--guided-npm-worker-request"
GUIDED_NPM_LAUNCHER_KIND = "subprocess_popen"
GUIDED_NPM_LAUNCH_ENVIRONMENT_POLICY = "inherit_unchanged"

GUIDED_NPM_WORKER_LAUNCH_FAILURE_CATEGORIES = (
    "prelaunch_claim_invalid",
    "prelaunch_claim_state_invalid",
    "current_build_invalid",
    "current_build_mismatch",
    "launch_invocation_invalid",
    "launch_invocation_identity_mismatch",
    "launch_executable_invalid",
    "launch_entry_point_missing",
    "launch_working_directory_invalid",
    "launch_worker_artifact_changed",
    "launch_startup_artifact_changed",
    "launch_source_freshness_changed",
    "launch_cancelled",
    "process_creation_failed",
    "process_identity_invalid",
    "process_created_receipt_failed",
    "launch_internal_error",
)
_FAILURE_CATEGORIES = frozenset(GUIDED_NPM_WORKER_LAUNCH_FAILURE_CATEGORIES)


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("launch_nonfinite")
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("launch_mapping_key_invalid")
        return {key: _canonical(item) for key, item in value.items()}
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    raise ValueError(f"launch_value_unsupported:{type(value).__name__}")


def _identity(value: Any, field_name: str, domain: str) -> str:
    payload = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name != field_name
    }
    return hashlib.sha256(
        domain.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(_canonical(payload))
    ).hexdigest()


@dataclass(frozen=True)
class GuidedNpmWorkerLaunchInvocation:
    invocation_schema_name: str
    invocation_schema_version: str
    invocation_contract_version: str
    source_prelaunch_claim_identity: str
    source_worker_request_identity: str
    source_execution_request_identity: str
    application_build_identity: ApplicationBuildIdentity
    guided_plan_identity: str
    validation_revision: int
    execution_mode: str
    executable_path: str
    argument_vector: tuple[str, ...]
    working_directory_path: str
    worker_request_artifact_path: str
    run_directory_path: str
    environment_policy: str
    shell: bool
    invocation_status: str
    launch_status: str
    execution_status: str
    completion_status: str
    canonical_launch_invocation_identity: str


@dataclass(frozen=True)
class GuidedNpmStartedProcess:
    process_id: int
    launcher_kind: str


class GuidedNpmProcessLauncher(Protocol):
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        cwd: str,
        shell: bool,
    ) -> GuidedNpmStartedProcess: ...


@dataclass(frozen=True)
class GuidedNpmWorkerExecutionStartReceipt:
    receipt_schema_name: str
    receipt_schema_version: str
    receipt_contract_version: str
    source_prelaunch_claim_identity: str
    source_launch_invocation_identity: str
    source_worker_request_identity: str
    source_execution_request_identity: str
    source_materialization_receipt_identity: str
    source_prelaunch_freshness_evidence_identity: str
    application_build_identity: ApplicationBuildIdentity
    guided_plan_identity: str
    validation_revision: int
    execution_mode: str
    executable_path: str
    argument_vector: tuple[str, ...]
    working_directory_path: str
    worker_request_artifact_path: str
    run_directory_path: str
    process_id: int
    launcher_kind: str
    launch_status: str
    execution_status: str
    completion_status: str
    consumed_authority_status: str
    canonical_execution_start_receipt_identity: str


@dataclass(frozen=True)
class GuidedNpmWorkerLaunchIssue:
    category: str
    section: str
    message: str
    detail_code: str
    exception_type: str = ""

    def __post_init__(self) -> None:
        if self.category not in _FAILURE_CATEGORIES:
            raise ValueError("launch_issue_category_invalid")


@dataclass(frozen=True)
class GuidedNpmWorkerLaunchFailure:
    blocking_issues: tuple[GuidedNpmWorkerLaunchIssue, ...]
    process_creation_status: str = "not_created"
    status: str = "refused"

    def __post_init__(self) -> None:
        if (
            len(self.blocking_issues) != 1
            or self.process_creation_status != "not_created"
            or self.status != "refused"
        ):
            raise ValueError("launch_failure_invalid")


@dataclass(frozen=True)
class GuidedNpmWorkerLaunchCancelled:
    blocking_issues: tuple[GuidedNpmWorkerLaunchIssue, ...]
    process_creation_status: str = "not_created"
    status: str = "cancelled"

    def __post_init__(self) -> None:
        if (
            len(self.blocking_issues) != 1
            or self.blocking_issues[0].category != "launch_cancelled"
            or self.process_creation_status != "not_created"
            or self.status != "cancelled"
        ):
            raise ValueError("launch_cancelled_result_invalid")


@dataclass(frozen=True)
class GuidedNpmWorkerPostLaunchFailure:
    blocking_issues: tuple[GuidedNpmWorkerLaunchIssue, ...]
    source_launch_invocation_identity: str
    observed_process_id: int | None
    process_creation_status: str = "indeterminate_or_created"
    status: str = "post_launch_failure"

    def __post_init__(self) -> None:
        if (
            len(self.blocking_issues) != 1
            or self.process_creation_status != "indeterminate_or_created"
            or self.status != "post_launch_failure"
        ):
            raise ValueError("post_launch_failure_invalid")


GuidedNpmWorkerLaunchResult = (
    GuidedNpmWorkerExecutionStartReceipt
    | GuidedNpmWorkerLaunchFailure
    | GuidedNpmWorkerLaunchCancelled
    | GuidedNpmWorkerPostLaunchFailure
)


class _LaunchCancelled(RuntimeError):
    pass


def _check_cancelled(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise _LaunchCancelled("launch_cancelled")


def _application_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))


def _executable_path() -> str:
    return os.fspath(sys.executable)


def _expected_argv(worker_path: str) -> tuple[str, ...]:
    return (
        _executable_path(),
        "-m",
        GUIDED_NPM_WORKER_ENTRY_MODULE,
        GUIDED_NPM_WORKER_REQUEST_ARGUMENT,
        worker_path,
    )


def compute_guided_npm_worker_launch_invocation_identity(
    invocation: GuidedNpmWorkerLaunchInvocation,
) -> str:
    return _identity(
        invocation,
        "canonical_launch_invocation_identity",
        GUIDED_NPM_WORKER_LAUNCH_INVOCATION_CONTRACT_VERSION,
    )


def compute_guided_npm_worker_execution_start_receipt_identity(
    receipt: GuidedNpmWorkerExecutionStartReceipt,
) -> str:
    return _identity(
        receipt,
        "canonical_execution_start_receipt_identity",
        GUIDED_NPM_WORKER_EXECUTION_START_RECEIPT_CONTRACT_VERSION,
    )


def _verify_build_binding(
    current: ApplicationBuildIdentity,
    claim: GuidedNpmWorkerPrelaunchClaim,
) -> None:
    verify_application_build_identity(current)
    worker = claim.worker_request
    if not (
        current
        == claim.application_build_identity
        == worker.application_build_identity
        == worker.execution_request.application_build_identity
        == claim.materialization_receipt.application_build_identity
        == claim.prelaunch_freshness_evidence.current_application_build_identity
    ):
        raise ValueError("current_build_mismatch")


def build_guided_npm_worker_launch_invocation(
    prelaunch_claim: GuidedNpmWorkerPrelaunchClaim,
    *,
    current_application_build_identity: ApplicationBuildIdentity,
) -> GuidedNpmWorkerLaunchInvocation:
    """Purely derive the sole production invocation from frozen authority."""
    verify_guided_npm_worker_prelaunch_claim(prelaunch_claim)
    _verify_build_binding(current_application_build_identity, prelaunch_claim)
    worker = prelaunch_claim.worker_request
    execution = worker.execution_request
    invocation = GuidedNpmWorkerLaunchInvocation(
        GUIDED_NPM_WORKER_LAUNCH_INVOCATION_SCHEMA_NAME,
        GUIDED_NPM_WORKER_LAUNCH_INVOCATION_SCHEMA_VERSION,
        GUIDED_NPM_WORKER_LAUNCH_INVOCATION_CONTRACT_VERSION,
        prelaunch_claim.canonical_prelaunch_claim_identity,
        worker.canonical_worker_request_identity,
        execution.canonical_execution_request_identity,
        current_application_build_identity,
        worker.guided_plan_identity,
        worker.validation_revision,
        execution.execution_mode,
        _executable_path(),
        _expected_argv(prelaunch_claim.worker_request_artifact_path),
        _application_root(),
        prelaunch_claim.worker_request_artifact_path,
        prelaunch_claim.run_directory_path,
        GUIDED_NPM_LAUNCH_ENVIRONMENT_POLICY,
        False,
        "constructed_for_launch",
        "not_launched",
        "not_started",
        "not_available",
        "0" * 64,
    )
    invocation = replace(
        invocation,
        canonical_launch_invocation_identity=(
            compute_guided_npm_worker_launch_invocation_identity(invocation)
        ),
    )
    verify_guided_npm_worker_launch_invocation(invocation, prelaunch_claim)
    return invocation


def verify_guided_npm_worker_launch_invocation(
    invocation: GuidedNpmWorkerLaunchInvocation,
    prelaunch_claim: GuidedNpmWorkerPrelaunchClaim,
) -> None:
    """Purely verify one complete deterministic launch invocation."""
    if type(invocation) is not GuidedNpmWorkerLaunchInvocation:
        raise ValueError("launch_invocation_type_invalid")
    verify_guided_npm_worker_prelaunch_claim(prelaunch_claim)
    _verify_build_binding(invocation.application_build_identity, prelaunch_claim)
    worker = prelaunch_claim.worker_request
    execution = worker.execution_request
    style = execution.output_runtime_projection.output_base_path_style
    try:
        worker_path_matches = stored_paths_equal(
            invocation.worker_request_artifact_path,
            prelaunch_claim.worker_request_artifact_path,
            style,
        )
        run_path_matches = stored_paths_equal(
            invocation.run_directory_path,
            prelaunch_claim.run_directory_path,
            style,
        )
    except ValueError as exc:
        raise ValueError("launch_invocation_path_invalid") from exc
    expected_argv = _expected_argv(prelaunch_claim.worker_request_artifact_path)
    if (
        (invocation.invocation_schema_name, invocation.invocation_schema_version,
         invocation.invocation_contract_version)
        != (GUIDED_NPM_WORKER_LAUNCH_INVOCATION_SCHEMA_NAME,
            GUIDED_NPM_WORKER_LAUNCH_INVOCATION_SCHEMA_VERSION,
            GUIDED_NPM_WORKER_LAUNCH_INVOCATION_CONTRACT_VERSION)
        or invocation.source_prelaunch_claim_identity
        != prelaunch_claim.canonical_prelaunch_claim_identity
        or invocation.source_worker_request_identity
        != worker.canonical_worker_request_identity
        or invocation.source_execution_request_identity
        != execution.canonical_execution_request_identity
        or invocation.guided_plan_identity != worker.guided_plan_identity
        or invocation.validation_revision != worker.validation_revision
        or invocation.execution_mode != execution.execution_mode
        or invocation.executable_path != _executable_path()
        or not os.path.isabs(invocation.executable_path)
        or invocation.argument_vector != expected_argv
        or invocation.argument_vector.count(GUIDED_NPM_WORKER_REQUEST_ARGUMENT) != 1
        or invocation.working_directory_path != _application_root()
        or not os.path.isabs(invocation.working_directory_path)
        or not worker_path_matches
        or not run_path_matches
        or invocation.environment_policy != GUIDED_NPM_LAUNCH_ENVIRONMENT_POLICY
        or invocation.shell is not False
        or (invocation.invocation_status, invocation.launch_status,
            invocation.execution_status, invocation.completion_status)
        != ("constructed_for_launch", "not_launched", "not_started", "not_available")
    ):
        raise ValueError("launch_invocation_invalid")
    if (
        compute_guided_npm_worker_launch_invocation_identity(invocation)
        != invocation.canonical_launch_invocation_identity
    ):
        raise ValueError("launch_invocation_identity_mismatch")


def _build_execution_start_receipt(
    claim: GuidedNpmWorkerPrelaunchClaim,
    invocation: GuidedNpmWorkerLaunchInvocation,
    started: GuidedNpmStartedProcess,
) -> GuidedNpmWorkerExecutionStartReceipt:
    receipt = GuidedNpmWorkerExecutionStartReceipt(
        GUIDED_NPM_WORKER_EXECUTION_START_RECEIPT_SCHEMA_NAME,
        GUIDED_NPM_WORKER_EXECUTION_START_RECEIPT_SCHEMA_VERSION,
        GUIDED_NPM_WORKER_EXECUTION_START_RECEIPT_CONTRACT_VERSION,
        claim.canonical_prelaunch_claim_identity,
        invocation.canonical_launch_invocation_identity,
        claim.source_worker_request_identity,
        claim.source_execution_request_identity,
        claim.source_materialization_receipt_identity,
        claim.source_prelaunch_freshness_evidence_identity,
        claim.application_build_identity,
        claim.guided_plan_identity,
        claim.validation_revision,
        claim.execution_mode,
        invocation.executable_path,
        invocation.argument_vector,
        invocation.working_directory_path,
        invocation.worker_request_artifact_path,
        invocation.run_directory_path,
        started.process_id,
        started.launcher_kind,
        "process_created",
        "start_unconfirmed",
        "not_available",
        "not_available",
        "0" * 64,
    )
    return replace(
        receipt,
        canonical_execution_start_receipt_identity=(
            compute_guided_npm_worker_execution_start_receipt_identity(receipt)
        ),
    )


def verify_guided_npm_worker_execution_start_receipt(
    receipt: GuidedNpmWorkerExecutionStartReceipt,
    prelaunch_claim: GuidedNpmWorkerPrelaunchClaim,
    invocation: GuidedNpmWorkerLaunchInvocation,
) -> None:
    """Purely verify process-creation evidence; access no process or files."""
    if type(receipt) is not GuidedNpmWorkerExecutionStartReceipt:
        raise ValueError("execution_start_receipt_type_invalid")
    verify_guided_npm_worker_prelaunch_claim(prelaunch_claim)
    verify_guided_npm_worker_launch_invocation(invocation, prelaunch_claim)
    worker = prelaunch_claim.worker_request
    if (
        (receipt.receipt_schema_name, receipt.receipt_schema_version,
         receipt.receipt_contract_version)
        != (GUIDED_NPM_WORKER_EXECUTION_START_RECEIPT_SCHEMA_NAME,
            GUIDED_NPM_WORKER_EXECUTION_START_RECEIPT_SCHEMA_VERSION,
            GUIDED_NPM_WORKER_EXECUTION_START_RECEIPT_CONTRACT_VERSION)
        or receipt.source_prelaunch_claim_identity
        != prelaunch_claim.canonical_prelaunch_claim_identity
        or receipt.source_launch_invocation_identity
        != invocation.canonical_launch_invocation_identity
        or receipt.source_worker_request_identity != worker.canonical_worker_request_identity
        or receipt.source_execution_request_identity
        != worker.execution_request.canonical_execution_request_identity
        or receipt.source_materialization_receipt_identity
        != prelaunch_claim.materialization_receipt.canonical_materialization_receipt_identity
        or receipt.source_prelaunch_freshness_evidence_identity
        != prelaunch_claim.prelaunch_freshness_evidence.canonical_prelaunch_freshness_evidence_identity
        or receipt.application_build_identity != prelaunch_claim.application_build_identity
        or receipt.guided_plan_identity != prelaunch_claim.guided_plan_identity
        or receipt.validation_revision != prelaunch_claim.validation_revision
        or receipt.execution_mode != prelaunch_claim.execution_mode
        or receipt.executable_path != invocation.executable_path
        or receipt.argument_vector != invocation.argument_vector
        or receipt.working_directory_path != invocation.working_directory_path
        or receipt.worker_request_artifact_path != invocation.worker_request_artifact_path
        or receipt.run_directory_path != invocation.run_directory_path
        or isinstance(receipt.process_id, bool)
        or not isinstance(receipt.process_id, int)
        or receipt.process_id <= 0
        or receipt.launcher_kind != GUIDED_NPM_LAUNCHER_KIND
        or (receipt.launch_status, receipt.execution_status,
            receipt.completion_status, receipt.consumed_authority_status)
        != ("process_created", "start_unconfirmed", "not_available", "not_available")
    ):
        raise ValueError("execution_start_receipt_invalid")
    if (
        compute_guided_npm_worker_execution_start_receipt_identity(receipt)
        != receipt.canonical_execution_start_receipt_identity
    ):
        raise ValueError("execution_start_receipt_identity_mismatch")


def _issue(
    category: str,
    section: str,
    message: str,
    detail_code: str,
    exception_type: str = "",
) -> GuidedNpmWorkerLaunchIssue:
    return GuidedNpmWorkerLaunchIssue(
        category if category in _FAILURE_CATEGORIES else "launch_internal_error",
        section,
        message,
        detail_code,
        exception_type,
    )


def _failure(category: str, section: str, detail: str, exc: BaseException | None = None):
    issue = _issue(
        category,
        section,
        "The NPM worker process could not be created from verified authority.",
        detail,
        type(exc).__name__ if exc is not None else "",
    )
    return GuidedNpmWorkerLaunchFailure((issue,))


def _cancelled() -> GuidedNpmWorkerLaunchCancelled:
    return GuidedNpmWorkerLaunchCancelled(
        (_issue(
            "launch_cancelled",
            "cancellation",
            "NPM worker launch was cancelled before process creation.",
            "launch_cancelled",
        ),)
    )


def _post_launch_failure(
    category: str,
    invocation: GuidedNpmWorkerLaunchInvocation,
    started: Any,
    detail: str,
    exc: BaseException | None = None,
) -> GuidedNpmWorkerPostLaunchFailure:
    pid = getattr(started, "process_id", None)
    observed = pid if isinstance(pid, int) and not isinstance(pid, bool) else None
    return GuidedNpmWorkerPostLaunchFailure(
        (_issue(
            category,
            "post_launch",
            "Process creation was attempted and its resulting state requires later reconciliation.",
            detail,
            type(exc).__name__ if exc is not None else "",
        ),),
        invocation.canonical_launch_invocation_identity,
        observed,
    )


def _verify_launch_filesystem(invocation: GuidedNpmWorkerLaunchInvocation) -> None:
    executable = Path(invocation.executable_path)
    entry = Path(_application_root(), "photometry_pipeline", "guided_npm_worker_entry.py")
    cwd = Path(invocation.working_directory_path)
    try:
        executable_stat = executable.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError("launch_executable_invalid") from exc
    if not stat.S_ISREG(executable_stat.st_mode):
        raise ValueError("launch_executable_invalid")
    if os.name != "nt" and not os.access(executable, os.X_OK):
        raise ValueError("launch_executable_invalid")
    try:
        entry_stat = entry.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError("launch_entry_point_missing") from exc
    if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISREG(entry_stat.st_mode):
        raise ValueError("launch_entry_point_missing")
    try:
        cwd_stat = cwd.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError("launch_working_directory_invalid") from exc
    if stat.S_ISLNK(cwd_stat.st_mode) or not stat.S_ISDIR(cwd_stat.st_mode):
        raise ValueError("launch_working_directory_invalid")


def _subprocess_popen_launcher(
    argv: tuple[str, ...],
    *,
    cwd: str,
    shell: bool,
) -> GuidedNpmStartedProcess:
    process = subprocess.Popen(argv, cwd=cwd, shell=shell)
    return GuidedNpmStartedProcess(process.pid, GUIDED_NPM_LAUNCHER_KIND)


def _category_from_value_error(exc: ValueError, fallback: str) -> str:
    detail = str(exc)
    if detail in _FAILURE_CATEGORIES:
        return detail
    if "state" in detail and fallback == "prelaunch_claim_invalid":
        return "prelaunch_claim_state_invalid"
    if "identity" in detail and fallback == "launch_invocation_invalid":
        return "launch_invocation_identity_mismatch"
    return fallback


def launch_guided_npm_worker(
    prelaunch_claim: GuidedNpmWorkerPrelaunchClaim,
    *,
    current_application_build_identity: ApplicationBuildIdentity,
    cancellation_check: Callable[[], bool] | None = None,
    process_launcher: GuidedNpmProcessLauncher | None = None,
) -> GuidedNpmWorkerLaunchResult:
    """Attempt one exact process creation and return truthful start evidence."""
    try:
        _check_cancelled(cancellation_check)
        if type(prelaunch_claim) is not GuidedNpmWorkerPrelaunchClaim:
            return _failure(
                "prelaunch_claim_invalid", "claim", "prelaunch_claim_type_invalid"
            )
        if (
            prelaunch_claim.claim_status,
            prelaunch_claim.launch_status,
            prelaunch_claim.execution_status,
            prelaunch_claim.completion_status,
            prelaunch_claim.runnable,
        ) != (
            "verified_for_prelaunch",
            "not_launched",
            "not_started",
            "not_available",
            False,
        ):
            return _failure(
                "prelaunch_claim_state_invalid",
                "claim",
                "prelaunch_claim_state_invalid",
            )
        try:
            verify_guided_npm_worker_prelaunch_claim(prelaunch_claim)
        except (TypeError, ValueError) as exc:
            return _failure(
                _category_from_value_error(exc, "prelaunch_claim_invalid"),
                "claim",
                str(exc) or type(exc).__name__,
                exc,
            )
        try:
            verify_application_build_identity(current_application_build_identity)
            _verify_build_binding(current_application_build_identity, prelaunch_claim)
        except (TypeError, ValueError) as exc:
            category = "current_build_mismatch" if "mismatch" in str(exc) else "current_build_invalid"
            return _failure(category, "build", str(exc) or type(exc).__name__, exc)

        _check_cancelled(cancellation_check)
        try:
            invocation = build_guided_npm_worker_launch_invocation(
                prelaunch_claim,
                current_application_build_identity=current_application_build_identity,
            )
            verify_guided_npm_worker_launch_invocation(invocation, prelaunch_claim)
        except (TypeError, ValueError) as exc:
            return _failure(
                _category_from_value_error(exc, "launch_invocation_invalid"),
                "invocation",
                str(exc) or type(exc).__name__,
                exc,
            )

        try:
            _verify_launch_filesystem(invocation)
        except ValueError as exc:
            return _failure(
                _category_from_value_error(exc, "launch_internal_error"),
                "launch_filesystem",
                str(exc),
                exc,
            )

        _check_cancelled(cancellation_check)
        try:
            verify_guided_npm_worker_prelaunch_authority_live(
                prelaunch_claim,
                current_application_build_identity=current_application_build_identity,
                cancellation_check=cancellation_check,
            )
        except GuidedNpmPrelaunchAuthorityLiveCancelled:
            return _cancelled()
        except (TypeError, ValueError) as exc:
            category = _category_from_value_error(
                exc, "launch_worker_artifact_changed"
            )
            return _failure(category, "final_gate", str(exc) or type(exc).__name__, exc)

        # Prevalidate every receipt field except the actual process identity.
        provisional = _build_execution_start_receipt(
            prelaunch_claim,
            invocation,
            GuidedNpmStartedProcess(1, GUIDED_NPM_LAUNCHER_KIND),
        )
        verify_guided_npm_worker_execution_start_receipt(
            provisional, prelaunch_claim, invocation
        )
        _check_cancelled(cancellation_check)

        launcher = process_launcher or _subprocess_popen_launcher
        try:
            started = launcher(
                invocation.argument_vector,
                cwd=invocation.working_directory_path,
                shell=False,
            )
        except Exception as exc:
            return _failure(
                "process_creation_failed",
                "process_creation",
                type(exc).__name__,
                exc,
            )

        if type(started) is not GuidedNpmStartedProcess:
            return _post_launch_failure(
                "process_identity_invalid",
                invocation,
                started,
                "started_process_type_invalid",
            )
        if (
            isinstance(started.process_id, bool)
            or not isinstance(started.process_id, int)
            or started.process_id <= 0
            or started.launcher_kind != GUIDED_NPM_LAUNCHER_KIND
        ):
            return _post_launch_failure(
                "process_identity_invalid",
                invocation,
                started,
                "started_process_identity_invalid",
            )
        try:
            receipt = _build_execution_start_receipt(
                prelaunch_claim, invocation, started
            )
            verify_guided_npm_worker_execution_start_receipt(
                receipt, prelaunch_claim, invocation
            )
            return receipt
        except Exception as exc:
            return _post_launch_failure(
                "process_created_receipt_failed",
                invocation,
                started,
                str(exc) or type(exc).__name__,
                exc,
            )
    except _LaunchCancelled:
        return _cancelled()
    except Exception as exc:
        return _failure(
            "launch_internal_error", "launch", type(exc).__name__, exc
        )
