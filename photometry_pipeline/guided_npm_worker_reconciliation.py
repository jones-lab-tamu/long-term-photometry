"""B2-D2B: parent-side process termination and authoritative run reconciliation.

This module owns the single boundary the parent needed but did not yet have:
retaining the exact process handle it created, observing the exact exit code,
and reconciling that observation against the durable evidence the child
already produced (the execution-start receipt, the consumed-authority
receipt, and the terminal receipt).

It does not redesign any child contract, does not infer completion from any
single source alone, and does not add GUI wiring, retry UX, or a general
scheduler.  A later GUI patch invokes ``reconcile_guided_npm_worker_runtime``
from its own worker thread; this module only provides the backend operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import threading
from typing import Any

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_worker_acknowledgement import (
    expected_guided_npm_consumed_authority_receipt_path,
    read_and_verify_guided_npm_consumed_authority_receipt,
)
from photometry_pipeline.guided_npm_worker_launch import (
    GuidedNpmLaunchedWorkerRuntime,
    GuidedNpmPostLaunchRuntime,
    verify_guided_npm_worker_execution_start_receipt,
)
from photometry_pipeline.guided_npm_worker_terminal import (
    GuidedNpmTerminalError,
    TERMINAL_OUTCOME_COMPLETED,
    build_guided_npm_terminal_output_record,
    compute_guided_npm_completed_run_evidence_identity,
    expected_guided_npm_terminal_receipt_path,
    read_guided_npm_worker_terminal_receipt,
    verify_guided_npm_worker_terminal_receipt,
)
from photometry_pipeline.guided_production_mapping import ApplicationBuildIdentity


GUIDED_NPM_RECONCILIATION_RESULT_FILENAME = "guided_npm_reconciliation_result.json"
GUIDED_NPM_RECONCILIATION_RESULT_SCHEMA_NAME = "guided_npm_worker_reconciliation_result"
GUIDED_NPM_RECONCILIATION_RESULT_SCHEMA_VERSION = "v1"
GUIDED_NPM_RECONCILIATION_RESULT_CONTRACT_VERSION = (
    "guided_npm_worker_reconciliation_result.v1"
)

OUTCOME_VERIFIED_COMPLETED = "verified_completed"
OUTCOME_VERIFIED_FAILED_BEFORE_CONSUMED = "verified_failed_before_consumed_authority"
OUTCOME_VERIFIED_FAILED_AFTER_CONSUMED = "verified_failed_after_consumed_authority"
OUTCOME_VERIFIED_FAILED_DURING_OUTPUT_FINALIZATION = (
    "verified_failed_during_output_finalization"
)
OUTCOME_AUTHORITY_REFUSED = "authority_refused"
OUTCOME_TERMINAL_RECEIPT_PUBLICATION_FAILED = "terminal_receipt_publication_failed"
OUTCOME_PROCESS_FAILED_WITHOUT_TERMINAL_EVIDENCE = (
    "process_failed_without_terminal_evidence"
)
OUTCOME_PROCESS_EXITED_ZERO_WITHOUT_TERMINAL_EVIDENCE = (
    "process_exited_zero_without_terminal_evidence"
)
OUTCOME_TERMINAL_EVIDENCE_INVALID = "terminal_evidence_invalid"
OUTCOME_CONSUMED_AUTHORITY_EVIDENCE_INVALID = "consumed_authority_evidence_invalid"
OUTCOME_PROCESS_IDENTITY_MISMATCH = "process_identity_mismatch"
OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED = "completed_output_integrity_failed"
OUTCOME_INDETERMINATE = "indeterminate"

GUIDED_NPM_RECONCILIATION_FINAL_OUTCOMES = (
    OUTCOME_VERIFIED_COMPLETED,
    OUTCOME_VERIFIED_FAILED_BEFORE_CONSUMED,
    OUTCOME_VERIFIED_FAILED_AFTER_CONSUMED,
    OUTCOME_VERIFIED_FAILED_DURING_OUTPUT_FINALIZATION,
    OUTCOME_AUTHORITY_REFUSED,
    OUTCOME_TERMINAL_RECEIPT_PUBLICATION_FAILED,
    OUTCOME_PROCESS_FAILED_WITHOUT_TERMINAL_EVIDENCE,
    OUTCOME_PROCESS_EXITED_ZERO_WITHOUT_TERMINAL_EVIDENCE,
    OUTCOME_TERMINAL_EVIDENCE_INVALID,
    OUTCOME_CONSUMED_AUTHORITY_EVIDENCE_INVALID,
    OUTCOME_PROCESS_IDENTITY_MISMATCH,
    OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED,
    OUTCOME_INDETERMINATE,
)

# Deliberately NOT part of GUIDED_NPM_RECONCILIATION_FINAL_OUTCOMES: a process
# whose durable launch confirmation never completed has no execution-start
# receipt to bind a durable GuidedNpmWorkerReconciliationResult to, so this
# outcome is only ever produced in-memory by
# reconcile_guided_npm_post_launch_runtime and is never persisted.  Keeping it
# out of the durable closed set means the durable verifier's own membership
# check already refuses any attempt to smuggle it into a published artifact.
OUTCOME_POST_LAUNCH_EVIDENCE_FAILED = "post_launch_evidence_failed"

_TERMINAL_OUTCOME_TO_FINAL_OUTCOME = {
    "failed_before_consumed_authority": OUTCOME_VERIFIED_FAILED_BEFORE_CONSUMED,
    "failed_after_consumed_authority": OUTCOME_VERIFIED_FAILED_AFTER_CONSUMED,
    "failed_during_output_finalization": (
        OUTCOME_VERIFIED_FAILED_DURING_OUTPUT_FINALIZATION
    ),
}

_EVIDENCE_STATUSES = ("absent", "invalid", "verified")
_OUTPUT_RECONCILIATION_STATUSES = ("not_applicable", "verified", "failed")


class GuidedNpmReconciliationTimeout(RuntimeError):
    """The process had not terminated within the requested timeout.

    The process was not killed.  A later call may still reconcile the same
    runtime once it has actually exited.
    """


class GuidedNpmReconciliationObservationError(RuntimeError):
    """The parent could not trust what ``process_handle.wait()`` returned."""


class GuidedNpmReconciliationPublicationError(RuntimeError):
    """The computed result could not be durably published."""


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("reconciliation_nonfinite")
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("reconciliation_mapping_key_invalid")
        return {key: _canonical(item) for key, item in value.items()}
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    raise ValueError(f"reconciliation_value_unsupported:{type(value).__name__}")


def _identity(value: Any, field_name: str, domain: str) -> str:
    payload = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name != field_name
    }
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + encode_canonical_value(_canonical(payload))
    ).hexdigest()


@dataclass(frozen=True)
class GuidedNpmWorkerReconciliationResult:
    result_schema_name: str
    result_schema_version: str
    result_contract_version: str

    source_prelaunch_claim_identity: str
    source_launch_invocation_identity: str
    source_execution_start_receipt_identity: str
    source_worker_request_identity: str
    source_execution_request_identity: str
    source_launch_context_identity: str

    application_build_identity: ApplicationBuildIdentity
    guided_plan_identity: str
    validation_revision: int
    execution_mode: str
    run_directory_path: str

    observed_process_id: int
    observed_exit_code: int

    consumed_authority_receipt_identity: str | None
    terminal_receipt_identity: str | None
    completed_run_evidence_identity: str | None

    process_terminal_status: str
    consumed_authority_status: str
    terminal_evidence_status: str
    output_reconciliation_status: str
    final_outcome: str

    failure_category: str | None
    failure_detail_code: str | None

    canonical_reconciliation_result_identity: str


def compute_guided_npm_worker_reconciliation_result_identity(
    value: GuidedNpmWorkerReconciliationResult,
) -> str:
    return _identity(
        value,
        "canonical_reconciliation_result_identity",
        GUIDED_NPM_RECONCILIATION_RESULT_CONTRACT_VERSION,
    )


def expected_guided_npm_reconciliation_result_path(run_directory_path: str) -> str:
    return os.path.join(run_directory_path, GUIDED_NPM_RECONCILIATION_RESULT_FILENAME)


def _pids_agree(
    *,
    reference_pid: int,
    execution_start_receipt_pid: int,
    consumed_status: str,
    consumed_pid: int | None,
    terminal_status: str,
    terminal_pid: int | None,
) -> bool:
    """Whether every authoritative PID source presently available agrees.

    Shared by the orchestration path (where ``reference_pid`` is the live
    ``process_handle.pid``) and the pure verifier (where ``reference_pid`` is
    the durable result's own stored ``observed_process_id``), so both use the
    exact same agreement rule.
    """
    candidates = [reference_pid, execution_start_receipt_pid]
    if consumed_status == "verified":
        candidates.append(consumed_pid)
    if terminal_status == "verified":
        candidates.append(terminal_pid)
    return (
        all(isinstance(pid, int) and not isinstance(pid, bool) for pid in candidates)
        and len(set(candidates)) == 1
    )


def _classify(
    *,
    exit_code: int,
    pid_ok: bool,
    consumed_status: str,
    terminal_status: str,
    terminal_outcome: str | None,
    output_ok: bool | None,
) -> str:
    """The single shared final-outcome decision tree.

    Used both by orchestration (to compute a fresh result) and by the pure
    verifier (to independently re-derive what a durable result's own stored
    fields imply, and require it match the result's stored ``final_outcome``)
    so the two can never drift into inconsistent classifications.
    """
    if not pid_ok:
        return OUTCOME_PROCESS_IDENTITY_MISMATCH
    # Consumed-authority corruption is checked first: when the consumed
    # receipt is genuinely invalid (not merely absent), a terminal receipt
    # that depends on it (e.g. a "completed" or "failed_after_consumed"
    # claim) will also fail to verify as a downstream symptom. Reporting the
    # root cause is more actionable than the cascading terminal failure.
    if consumed_status == "invalid":
        return OUTCOME_CONSUMED_AUTHORITY_EVIDENCE_INVALID
    if terminal_status == "invalid":
        return OUTCOME_TERMINAL_EVIDENCE_INVALID

    if exit_code == 0:
        if terminal_status != "verified":
            return OUTCOME_PROCESS_EXITED_ZERO_WITHOUT_TERMINAL_EVIDENCE
        if terminal_outcome != TERMINAL_OUTCOME_COMPLETED:
            return OUTCOME_INDETERMINATE
        if not output_ok:
            return OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED
        return OUTCOME_VERIFIED_COMPLETED
    if exit_code == 2:
        if terminal_status == "absent":
            return OUTCOME_AUTHORITY_REFUSED
        return OUTCOME_INDETERMINATE
    if exit_code == 3:
        if terminal_status != "verified":
            return OUTCOME_PROCESS_FAILED_WITHOUT_TERMINAL_EVIDENCE
        if terminal_outcome == TERMINAL_OUTCOME_COMPLETED:
            return OUTCOME_INDETERMINATE
        return _TERMINAL_OUTCOME_TO_FINAL_OUTCOME.get(terminal_outcome, OUTCOME_INDETERMINATE)
    if exit_code == 4:
        if terminal_status != "verified":
            return OUTCOME_TERMINAL_RECEIPT_PUBLICATION_FAILED
        return OUTCOME_INDETERMINATE
    if terminal_status != "verified":
        return OUTCOME_PROCESS_FAILED_WITHOUT_TERMINAL_EVIDENCE
    return OUTCOME_INDETERMINATE


def verify_guided_npm_worker_reconciliation_result(
    result: GuidedNpmWorkerReconciliationResult,
    *,
    prelaunch_claim,
    launch_invocation,
    execution_start_receipt,
    launch_context,
    consumed_authority_receipt=None,
    terminal_receipt=None,
) -> None:
    """Purely reconcile a durable result against the exact live evidence objects.

    Enforces the entire closed final-outcome matrix by deriving the expected
    outcome from the same ``_classify`` decision tree the orchestration path
    uses, plus the outcome-specific field bindings ``_classify`` alone cannot
    express (exact identity matches, and which evidence statuses each
    specific outcome structurally requires beyond bare agreement with the
    exit code and terminal claim).
    """
    if type(result) is not GuidedNpmWorkerReconciliationResult:
        raise ValueError("reconciliation_result_type_invalid")
    worker = prelaunch_claim.worker_request
    execution = worker.execution_request
    if (
        (result.result_schema_name, result.result_schema_version, result.result_contract_version)
        != (
            GUIDED_NPM_RECONCILIATION_RESULT_SCHEMA_NAME,
            GUIDED_NPM_RECONCILIATION_RESULT_SCHEMA_VERSION,
            GUIDED_NPM_RECONCILIATION_RESULT_CONTRACT_VERSION,
        )
        or result.source_prelaunch_claim_identity != prelaunch_claim.canonical_prelaunch_claim_identity
        or result.source_launch_invocation_identity != launch_invocation.canonical_launch_invocation_identity
        or result.source_execution_start_receipt_identity
        != execution_start_receipt.canonical_execution_start_receipt_identity
        or result.source_worker_request_identity != worker.canonical_worker_request_identity
        or result.source_execution_request_identity != execution.canonical_execution_request_identity
        or result.source_launch_context_identity != launch_context.canonical_launch_context_identity
        or result.application_build_identity != worker.application_build_identity
        or result.guided_plan_identity != worker.guided_plan_identity
        or result.validation_revision != worker.validation_revision
        or result.execution_mode != execution.execution_mode
        or result.run_directory_path != worker.run_directory_path
        or isinstance(result.observed_process_id, bool)
        or not isinstance(result.observed_process_id, int)
        or result.observed_process_id <= 0
        or isinstance(result.observed_exit_code, bool)
        or not isinstance(result.observed_exit_code, int)
        or result.process_terminal_status != "exited"
        or result.consumed_authority_status not in _EVIDENCE_STATUSES
        or result.terminal_evidence_status not in _EVIDENCE_STATUSES
        or result.output_reconciliation_status not in _OUTPUT_RECONCILIATION_STATUSES
        or result.final_outcome not in GUIDED_NPM_RECONCILIATION_FINAL_OUTCOMES
    ):
        raise ValueError("reconciliation_result_invalid")

    if result.consumed_authority_status == "verified":
        if (
            consumed_authority_receipt is None
            or result.consumed_authority_receipt_identity
            != consumed_authority_receipt.canonical_consumed_authority_receipt_identity
        ):
            raise ValueError("reconciliation_result_consumed_identity_invalid")
    elif result.consumed_authority_receipt_identity is not None:
        raise ValueError("reconciliation_result_consumed_identity_invalid")

    if result.terminal_evidence_status == "verified":
        if (
            terminal_receipt is None
            or result.terminal_receipt_identity != terminal_receipt.canonical_terminal_receipt_identity
        ):
            raise ValueError("reconciliation_result_terminal_identity_invalid")
    elif result.terminal_receipt_identity is not None:
        raise ValueError("reconciliation_result_terminal_identity_invalid")

    pid_ok = _pids_agree(
        reference_pid=result.observed_process_id,
        execution_start_receipt_pid=execution_start_receipt.process_id,
        consumed_status=result.consumed_authority_status,
        consumed_pid=(
            consumed_authority_receipt.observed_process_id
            if result.consumed_authority_status == "verified"
            else None
        ),
        terminal_status=result.terminal_evidence_status,
        terminal_pid=(
            terminal_receipt.observed_process_id
            if result.terminal_evidence_status == "verified"
            else None
        ),
    )
    if result.final_outcome == OUTCOME_PROCESS_IDENTITY_MISMATCH and pid_ok:
        raise ValueError("reconciliation_result_pid_mismatch_outcome_unjustified")

    terminal_outcome = (
        terminal_receipt.terminal_outcome if result.terminal_evidence_status == "verified" else None
    )
    if result.output_reconciliation_status == "verified":
        output_ok: bool | None = True
    elif result.output_reconciliation_status == "failed":
        output_ok = False
    else:
        output_ok = None

    expected_final_outcome = _classify(
        exit_code=result.observed_exit_code,
        pid_ok=pid_ok,
        consumed_status=result.consumed_authority_status,
        terminal_status=result.terminal_evidence_status,
        terminal_outcome=terminal_outcome,
        output_ok=output_ok,
    )
    if result.final_outcome != expected_final_outcome:
        raise ValueError("reconciliation_result_final_outcome_invalid")

    # _classify alone does not express every structural requirement a
    # specific outcome carries (e.g. it maps exit=3 + a verified
    # "failed_before_consumed_authority" terminal claim to the matching
    # outcome regardless of what consumed_authority_status actually is) --
    # the following close those gaps outcome by outcome.
    if result.final_outcome == OUTCOME_VERIFIED_COMPLETED:
        expected_output_status = "verified"
    elif result.final_outcome == OUTCOME_COMPLETED_OUTPUT_INTEGRITY_FAILED:
        expected_output_status = "failed"
    else:
        expected_output_status = "not_applicable"
    if result.output_reconciliation_status != expected_output_status:
        raise ValueError("reconciliation_result_output_status_invalid")

    if result.final_outcome == OUTCOME_VERIFIED_COMPLETED:
        if (
            terminal_receipt is None
            or terminal_receipt.terminal_outcome != TERMINAL_OUTCOME_COMPLETED
            or result.terminal_evidence_status != "verified"
            or result.consumed_authority_status != "verified"
            or result.output_reconciliation_status != "verified"
            or result.observed_exit_code != 0
            or result.completed_run_evidence_identity != terminal_receipt.completed_run_evidence_identity
            or result.failure_category is not None
            or result.failure_detail_code is not None
        ):
            raise ValueError("reconciliation_result_completed_invalid")
    else:
        if result.completed_run_evidence_identity is not None:
            raise ValueError("reconciliation_result_completed_evidence_unexpected")
        # Schema v1: every non-success outcome carries its own name as both
        # controlled failure fields (codified explicitly rather than left
        # implicit -- see B2-D2B narrow follow-up section 14).
        if result.failure_category != result.final_outcome or result.failure_detail_code != result.final_outcome:
            raise ValueError("reconciliation_result_failure_fields_invalid")

    if result.final_outcome == OUTCOME_AUTHORITY_REFUSED:
        if result.consumed_authority_status != "absent" or result.terminal_evidence_status != "absent":
            raise ValueError("reconciliation_result_authority_refused_invalid")
    elif result.final_outcome == OUTCOME_VERIFIED_FAILED_BEFORE_CONSUMED:
        if result.consumed_authority_status != "absent":
            raise ValueError("reconciliation_result_failed_before_consumed_invalid")
    elif result.final_outcome in (
        OUTCOME_VERIFIED_FAILED_AFTER_CONSUMED,
        OUTCOME_VERIFIED_FAILED_DURING_OUTPUT_FINALIZATION,
    ):
        if result.consumed_authority_status != "verified":
            raise ValueError("reconciliation_result_failed_after_consumed_invalid")

    if (
        compute_guided_npm_worker_reconciliation_result_identity(result)
        != result.canonical_reconciliation_result_identity
    ):
        raise ValueError("reconciliation_result_identity_mismatch")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("reconciliation_duplicate_json_key")
        result[key] = value
    return result


def _strict_fields(value: Any, cls: type) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(cls)}:
        raise ValueError("reconciliation_field_set_invalid")
    return dict(value)


def _decode_build(value: Any) -> ApplicationBuildIdentity:
    return ApplicationBuildIdentity(**_strict_fields(value, ApplicationBuildIdentity))


def serialize_guided_npm_worker_reconciliation_result(
    value: GuidedNpmWorkerReconciliationResult,
) -> bytes:
    return _canonical_bytes(value)


def decode_guided_npm_worker_reconciliation_result_bytes(
    content: bytes,
) -> GuidedNpmWorkerReconciliationResult:
    try:
        data = _strict_fields(
            json.loads(content, object_pairs_hook=_strict_object),
            GuidedNpmWorkerReconciliationResult,
        )
        data["application_build_identity"] = _decode_build(data["application_build_identity"])
        result = GuidedNpmWorkerReconciliationResult(**data)
        if serialize_guided_npm_worker_reconciliation_result(result) != content:
            raise ValueError("reconciliation_result_noncanonical")
        return result
    except Exception as exc:
        raise ValueError("reconciliation_result_decode_invalid") from exc


def _stat_facts(item: os.stat_result) -> tuple[int, int, int, int, int]:
    return (item.st_size, item.st_mtime_ns, item.st_dev, item.st_ino, item.st_mode)


def _publish_new_exclusive(path: str, content: bytes) -> None:
    target = Path(path)
    parent = target.parent
    if target.name != GUIDED_NPM_RECONCILIATION_RESULT_FILENAME:
        raise GuidedNpmReconciliationPublicationError("reconciliation_destination_invalid")
    try:
        parent_facts = parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise GuidedNpmReconciliationPublicationError(
            "reconciliation_destination_invalid"
        ) from exc
    if (
        stat.S_ISLNK(parent_facts.st_mode)
        or not stat.S_ISDIR(parent_facts.st_mode)
        or target.exists()
        or target.is_symlink()
    ):
        raise GuidedNpmReconciliationPublicationError("reconciliation_destination_conflict")
    temporary = parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise GuidedNpmReconciliationPublicationError("reconciliation_temporary_conflict")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, target)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _stable_read(path: str) -> bytes:
    target = Path(path)
    before = target.stat(follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise GuidedNpmReconciliationPublicationError("reconciliation_artifact_not_regular")
    with target.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        content = handle.read()
        after_open = os.fstat(handle.fileno())
    after = target.stat(follow_symlinks=False)
    if not (_stat_facts(before) == _stat_facts(opened) == _stat_facts(after_open) == _stat_facts(after)):
        raise GuidedNpmReconciliationPublicationError("reconciliation_artifact_changed_during_read")
    return content


def publish_guided_npm_worker_reconciliation_result(
    result: GuidedNpmWorkerReconciliationResult,
) -> str:
    """Publish exactly once. Refuses (never overwrites) a pre-existing destination."""
    path = expected_guided_npm_reconciliation_result_path(result.run_directory_path)
    _publish_new_exclusive(path, serialize_guided_npm_worker_reconciliation_result(result))
    observed = decode_guided_npm_worker_reconciliation_result_bytes(_stable_read(path))
    if observed != result:
        raise GuidedNpmReconciliationPublicationError("reconciliation_result_reread_mismatch")
    return path


def read_guided_npm_worker_reconciliation_result(
    path: str,
) -> GuidedNpmWorkerReconciliationResult:
    return decode_guided_npm_worker_reconciliation_result_bytes(_stable_read(path))


# ---------------------------------------------------------------------------
# Post-exit output reconciliation
# ---------------------------------------------------------------------------


def _reconcile_completed_outputs(worker_request, terminal_receipt) -> tuple[bool, str | None]:
    """Re-derive and re-hash every required output; never glob, never trust mtime alone."""
    fresh_records = []
    for stored in terminal_receipt.output_evidence:
        try:
            fresh = build_guided_npm_terminal_output_record(
                worker_request.run_directory_path,
                stored.output_role,
                terminal_receipt.execution_mode,
            )
        except GuidedNpmTerminalError:
            return False, None
        if (
            fresh.output_role != stored.output_role
            or fresh.canonical_relative_path != stored.canonical_relative_path
            or fresh.output_path != stored.output_path
            or fresh.source_size_bytes != stored.source_size_bytes
            or fresh.source_sha256 != stored.source_sha256
            or fresh.source_mode != stored.source_mode
            or fresh.source_device != stored.source_device
            or fresh.source_inode != stored.source_inode
        ):
            return False, None
        fresh_records.append(fresh)
    recomputed_identity = compute_guided_npm_completed_run_evidence_identity(
        source_worker_request_identity=terminal_receipt.source_worker_request_identity,
        source_execution_request_identity=terminal_receipt.source_execution_request_identity,
        source_consumed_authority_receipt_identity=(
            terminal_receipt.source_consumed_authority_receipt_identity
        ),
        guided_plan_identity=terminal_receipt.guided_plan_identity,
        validation_revision=terminal_receipt.validation_revision,
        output_evidence=tuple(fresh_records),
    )
    if recomputed_identity != terminal_receipt.completed_run_evidence_identity:
        return False, None
    return True, recomputed_identity


# ---------------------------------------------------------------------------
# Reconciliation orchestration (I/O; not a pure verifier)
#
# Per-runtime lifecycle state lives on the runtime object's own
# ``lifecycle_slot`` (a generic, reconciliation-agnostic mutable box owned by
# guided_npm_worker_launch.py), never in a module-level registry keyed by
# runtime identity/equality/weak-reference -- that would force hashing or
# weak-referencing an object holding an arbitrary, possibly non-hashable
# process handle, and would serialize unrelated runtimes' reconciliations
# behind one global lock.  Each runtime's own state has its own condition
# variable, so ``process_handle.wait()`` for one worker never blocks another.
# ---------------------------------------------------------------------------


@dataclass
class _ReconciliationRuntimeState:
    """Mutable per-runtime lifecycle state. Never serialized."""

    condition: threading.Condition = field(default_factory=threading.Condition)
    computation_status: str = "not_started"  # not_started | waiting | computed
    publication_status: str = "not_requested"  # not_requested | publishing | published | failed_or_indeterminate
    result: "GuidedNpmWorkerReconciliationResult | None" = None
    publication_error: BaseException | None = None


@dataclass
class _PostLaunchRuntimeState:
    """Mutable per-runtime lifecycle state for post-launch reconciliation.

    No publication state: a post-launch reconciliation is never durable (see
    ``GuidedNpmPostLaunchReconciliation``), so there is nothing to publish.
    """

    condition: threading.Condition = field(default_factory=threading.Condition)
    computation_status: str = "not_started"  # not_started | waiting | computed
    result: "GuidedNpmPostLaunchReconciliation | None" = None


def _run_once_per_runtime(lifecycle_slot, state_factory, compute_fn):
    """Ensure ``compute_fn`` runs to completion exactly once for this runtime.

    Concurrent callers for the *same* runtime wait on its own condition and
    receive the same result once computed; a timeout or observation error
    reverts the state to ``not_started`` (consuming no one-use guard) so a
    later call -- from this thread or another waiter -- may become the new
    computing caller once the process has actually exited.  The lock is held
    only to inspect/transition ``computation_status``, never while
    ``compute_fn`` itself runs (which is where the blocking ``wait()`` lives).
    """
    state = lifecycle_slot.get_or_create(state_factory)
    am_i_computing = False
    with state.condition:
        while True:
            if state.computation_status == "not_started":
                state.computation_status = "waiting"
                am_i_computing = True
                break
            if state.computation_status == "computed":
                return state, state.result
            state.condition.wait()

    try:
        result = compute_fn()
    except BaseException:
        with state.condition:
            state.computation_status = "not_started"
            state.condition.notify_all()
        raise

    with state.condition:
        state.result = result
        state.computation_status = "computed"
        state.condition.notify_all()
    return state, result


def _wait_for_exit_code(process_handle: Any, timeout_sec: float | None) -> int:
    try:
        exit_code = process_handle.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        raise GuidedNpmReconciliationTimeout("reconciliation_wait_timed_out") from None
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise GuidedNpmReconciliationObservationError("reconciliation_exit_code_invalid")
    return exit_code


def _compute_reconciliation(
    runtime: GuidedNpmLaunchedWorkerRuntime, timeout_sec: float | None
) -> GuidedNpmWorkerReconciliationResult:
    exit_code = _wait_for_exit_code(runtime.process_handle, timeout_sec)

    verify_guided_npm_worker_execution_start_receipt(
        runtime.execution_start_receipt, runtime.prelaunch_claim, runtime.launch_invocation
    )
    worker = runtime.prelaunch_claim.worker_request
    launch_context = runtime.launch_context

    consumed_status = "absent"
    consumed_receipt = None
    try:
        consumed_receipt = read_and_verify_guided_npm_consumed_authority_receipt(
            expected_guided_npm_consumed_authority_receipt_path(worker.run_directory_path),
            prelaunch_claim=runtime.prelaunch_claim,
            launch_invocation=runtime.launch_invocation,
            execution_start_receipt=runtime.execution_start_receipt,
        )
        consumed_status = "verified"
    except FileNotFoundError:
        consumed_status = "absent"
    except (OSError, ValueError):
        consumed_status = "invalid"

    terminal_status = "absent"
    terminal_receipt = None
    try:
        decoded = read_guided_npm_worker_terminal_receipt(
            expected_guided_npm_terminal_receipt_path(worker.run_directory_path),
            worker_request=worker,
        )
        verify_guided_npm_worker_terminal_receipt(
            decoded,
            worker_request=worker,
            launch_context=launch_context,
            consumed_authority_receipt=(
                consumed_receipt if consumed_status == "verified" else None
            ),
        )
        terminal_receipt = decoded
        terminal_status = "verified"
    except FileNotFoundError:
        terminal_status = "absent"
    except (OSError, ValueError, GuidedNpmTerminalError):
        terminal_status = "invalid"

    handle_pid = runtime.process_handle.pid
    pid_ok = _pids_agree(
        reference_pid=handle_pid,
        execution_start_receipt_pid=runtime.execution_start_receipt.process_id,
        consumed_status=consumed_status,
        consumed_pid=(consumed_receipt.observed_process_id if consumed_status == "verified" else None),
        terminal_status=terminal_status,
        terminal_pid=(terminal_receipt.observed_process_id if terminal_status == "verified" else None),
    )

    terminal_outcome = terminal_receipt.terminal_outcome if terminal_status == "verified" else None
    output_ok = None
    completed_run_evidence_identity = None
    output_reconciliation_status = "not_applicable"
    if (
        pid_ok
        and terminal_status == "verified"
        and consumed_status == "verified"
        and exit_code == 0
        and terminal_outcome == TERMINAL_OUTCOME_COMPLETED
    ):
        output_ok, recomputed_identity = _reconcile_completed_outputs(worker, terminal_receipt)
        output_reconciliation_status = "verified" if output_ok else "failed"
        if output_ok:
            completed_run_evidence_identity = recomputed_identity

    final_outcome = _classify(
        exit_code=exit_code,
        pid_ok=pid_ok,
        consumed_status=consumed_status,
        terminal_status=terminal_status,
        terminal_outcome=terminal_outcome,
        output_ok=output_ok,
    )
    if final_outcome != OUTCOME_VERIFIED_COMPLETED:
        completed_run_evidence_identity = None
    failure_category = None if final_outcome == OUTCOME_VERIFIED_COMPLETED else final_outcome

    result = GuidedNpmWorkerReconciliationResult(
        GUIDED_NPM_RECONCILIATION_RESULT_SCHEMA_NAME,
        GUIDED_NPM_RECONCILIATION_RESULT_SCHEMA_VERSION,
        GUIDED_NPM_RECONCILIATION_RESULT_CONTRACT_VERSION,
        runtime.prelaunch_claim.canonical_prelaunch_claim_identity,
        runtime.launch_invocation.canonical_launch_invocation_identity,
        runtime.execution_start_receipt.canonical_execution_start_receipt_identity,
        worker.canonical_worker_request_identity,
        worker.execution_request.canonical_execution_request_identity,
        launch_context.canonical_launch_context_identity,
        worker.application_build_identity,
        worker.guided_plan_identity,
        worker.validation_revision,
        worker.execution_request.execution_mode,
        worker.run_directory_path,
        handle_pid,
        exit_code,
        (
            consumed_receipt.canonical_consumed_authority_receipt_identity
            if consumed_status == "verified"
            else None
        ),
        (
            terminal_receipt.canonical_terminal_receipt_identity
            if terminal_status == "verified"
            else None
        ),
        completed_run_evidence_identity,
        "exited",
        consumed_status,
        terminal_status,
        output_reconciliation_status,
        final_outcome,
        failure_category,
        failure_category,
        "0" * 64,
    )
    result = replace(
        result,
        canonical_reconciliation_result_identity=(
            compute_guided_npm_worker_reconciliation_result_identity(result)
        ),
    )
    verify_guided_npm_worker_reconciliation_result(
        result,
        prelaunch_claim=runtime.prelaunch_claim,
        launch_invocation=runtime.launch_invocation,
        execution_start_receipt=runtime.execution_start_receipt,
        launch_context=launch_context,
        consumed_authority_receipt=(consumed_receipt if consumed_status == "verified" else None),
        terminal_receipt=(terminal_receipt if terminal_status == "verified" else None),
    )
    return result


def _publish_once(state: _ReconciliationRuntimeState, result: GuidedNpmWorkerReconciliationResult) -> None:
    """Publish exactly once per runtime; never retry a failed/indeterminate attempt."""
    with state.condition:
        if state.publication_status == "published":
            return
        if state.publication_status == "failed_or_indeterminate":
            raise GuidedNpmReconciliationPublicationError(
                "reconciliation_publication_previously_failed"
            ) from state.publication_error
        while state.publication_status == "publishing":
            state.condition.wait()
        if state.publication_status == "published":
            return
        if state.publication_status == "failed_or_indeterminate":
            raise GuidedNpmReconciliationPublicationError(
                "reconciliation_publication_previously_failed"
            ) from state.publication_error
        state.publication_status = "publishing"

    try:
        publish_guided_npm_worker_reconciliation_result(result)
    except BaseException as exc:
        with state.condition:
            state.publication_status = "failed_or_indeterminate"
            state.publication_error = exc
            state.condition.notify_all()
        raise
    else:
        with state.condition:
            state.publication_status = "published"
            state.condition.notify_all()


def reconcile_guided_npm_worker_runtime(
    runtime: GuidedNpmLaunchedWorkerRuntime,
    *,
    timeout_sec: float | None = None,
    publish: bool = True,
) -> GuidedNpmWorkerReconciliationResult:
    """Wait for the exact retained process and produce one immutable result.

    Blocking.  Not wired to any GUI thread by this patch.  Computation
    (waiting + evidence reconciliation) happens exactly once per runtime,
    regardless of how many times this is called or with what ``publish``
    value; concurrent callers for the same runtime receive the same computed
    result without duplicating the wait.  A timeout does not consume the
    one-use computation guard, so a later call may still reconcile the same
    runtime once it exits.

    Publication is tracked separately from computation: a first call with
    ``publish=False`` still computes and caches the result but leaves it
    unpublished, and a later call with ``publish=True`` publishes the exact
    cached result without recomputing or waiting again.  A failed or
    indeterminate publication attempt is remembered and never retried
    automatically; the computed in-memory result remains available, but
    every subsequent ``publish=True`` call raises the same controlled
    publication failure instead of touching the filesystem again.
    """
    state, result = _run_once_per_runtime(
        runtime.lifecycle_slot,
        _ReconciliationRuntimeState,
        lambda: _compute_reconciliation(runtime, timeout_sec),
    )
    if publish:
        _publish_once(state, result)
    return result


# ---------------------------------------------------------------------------
# Post-launch-confirmation-failure reconciliation
#
# A process was created but the parent could not durably confirm the launch
# (no execution-start receipt).  This path never produces a durable,
# publishable GuidedNpmWorkerReconciliationResult -- there is no execution
# start receipt identity to bind one to -- and it can never classify as
# verified_completed regardless of what child evidence happens to exist.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuidedNpmPostLaunchReconciliation:
    """In-memory-only diagnostic result for a created process whose durable
    launch confirmation failed. Never serialized; never published."""

    observed_process_id: int
    observed_exit_code: int
    launch_failure_category: str
    terminal_evidence_present: bool
    consumed_authority_evidence_present: bool
    final_outcome: str = OUTCOME_POST_LAUNCH_EVIDENCE_FAILED


def _compute_post_launch_reconciliation(
    runtime: GuidedNpmPostLaunchRuntime, timeout_sec: float | None
) -> GuidedNpmPostLaunchReconciliation:
    exit_code = _wait_for_exit_code(runtime.process_handle, timeout_sec)
    worker = runtime.prelaunch_claim.worker_request

    def _exists(path: str) -> bool:
        try:
            Path(path).stat(follow_symlinks=False)
            return True
        except OSError:
            return False

    terminal_present = _exists(
        expected_guided_npm_terminal_receipt_path(worker.run_directory_path)
    )
    consumed_present = _exists(
        expected_guided_npm_consumed_authority_receipt_path(worker.run_directory_path)
    )
    return GuidedNpmPostLaunchReconciliation(
        observed_process_id=runtime.process_handle.pid,
        observed_exit_code=exit_code,
        launch_failure_category=runtime.launch_failure.blocking_issues[0].category,
        terminal_evidence_present=terminal_present,
        consumed_authority_evidence_present=consumed_present,
    )


def reconcile_guided_npm_post_launch_runtime(
    runtime: GuidedNpmPostLaunchRuntime,
    *,
    timeout_sec: float | None = None,
) -> GuidedNpmPostLaunchReconciliation:
    """Wait for a created-but-unconfirmed process; never classify as completed.

    Child terminal/consumed evidence, if present, is inspected only for the
    diagnostic ``*_present`` flags -- never as sufficient authority to call
    the run complete, since the parent never durably confirmed the launch
    that would be required to trust the child's PID binding in the first
    place.  ``final_outcome`` is always ``post_launch_evidence_failed``.
    """
    _, result = _run_once_per_runtime(
        runtime.lifecycle_slot,
        _PostLaunchRuntimeState,
        lambda: _compute_post_launch_reconciliation(runtime, timeout_sec),
    )
    return result
