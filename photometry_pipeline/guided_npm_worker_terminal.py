"""Child-owned B2-D2A terminal receipt for one Guided NPM worker execution.

This module records only what the exact child verified about its own
execution up to and including its own terminal-receipt publication.  It does
not wait for, poll, or reconcile the child process from the parent, does not
interpret exit codes, and makes no claim about parent-observed cancellation
or post-exit output mutation -- those belong to a later B2-D2B patch.

A ``completed`` receipt is durable proof that: the exact worker and launch
context were accepted, the exact consumed-authority receipt was published and
reread, the authorized numerical Pipeline call returned normally, the
required output artifacts for this execution mode exist as exact regular
files bound to this attempt's consumed-authority identity, and the terminal
receipt itself was published atomically and reread successfully.

Any other outcome makes no success claim and truthfully records the furthest
verified stage, retaining the consumed-authority identity when it was
already durably established.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_worker_prelaunch_claim import stored_paths_equal
from photometry_pipeline.guided_production_mapping import ApplicationBuildIdentity


GUIDED_NPM_TERMINAL_RECEIPT_FILENAME = "guided_npm_terminal_receipt.json"
GUIDED_NPM_TERMINAL_RECEIPT_SCHEMA_NAME = "guided_npm_worker_terminal_receipt"
GUIDED_NPM_TERMINAL_RECEIPT_SCHEMA_VERSION = "v1"
GUIDED_NPM_TERMINAL_RECEIPT_CONTRACT_VERSION = "guided_npm_worker_terminal_receipt.v1"
GUIDED_NPM_TERMINAL_OUTPUT_RECORD_CONTRACT_VERSION = "guided_npm_terminal_output_record.v1"
GUIDED_NPM_COMPLETED_RUN_EVIDENCE_IDENTITY_DOMAIN = "guided_npm_completed_run_evidence.v1"

# Closed set of outcomes a *persisted* receipt may declare.  A receipt can
# never truthfully claim its own publication failed (if it exists and
# verifies, publication plainly succeeded) -- that classification exists only
# as an in-memory / exit-code concept, tracked separately below.
TERMINAL_OUTCOME_COMPLETED = "completed"
TERMINAL_OUTCOME_FAILED_BEFORE_CONSUMED = "failed_before_consumed_authority"
TERMINAL_OUTCOME_FAILED_AFTER_CONSUMED = "failed_after_consumed_authority"
TERMINAL_OUTCOME_FAILED_DURING_OUTPUT_FINALIZATION = "failed_during_output_finalization"

GUIDED_NPM_TERMINAL_OUTCOMES = (
    TERMINAL_OUTCOME_COMPLETED,
    TERMINAL_OUTCOME_FAILED_BEFORE_CONSUMED,
    TERMINAL_OUTCOME_FAILED_AFTER_CONSUMED,
    TERMINAL_OUTCOME_FAILED_DURING_OUTPUT_FINALIZATION,
)
_FAILURE_OUTCOMES = frozenset(GUIDED_NPM_TERMINAL_OUTCOMES) - {TERMINAL_OUTCOME_COMPLETED}

# In-memory-only: never a legal value of a persisted receipt's terminal_outcome.
GUIDED_NPM_TERMINAL_RECEIPT_PUBLICATION_FAILED = "terminal_receipt_publication_failed"

# Closed set of furthest-verified-stage values.  Adapted from the suggested
# stage list to the boundaries the child can actually observe: Pipeline.run()
# exposes exactly one mid-run hook (the consumed-authority callback, which
# fires once Pass 1 has consumed every source) and one terminal boundary (the
# call returning without raising, which already encompasses Pass 2, feature
# processing, and output finalization together).  Those three suggested
# stages are therefore represented by the single honestly-observable
# "numerical_pipeline_returned" boundary rather than claimed separately.
STAGE_WORKER_REQUEST_LOADING = "worker_request_loading"
STAGE_LAUNCH_CONTEXT_VERIFICATION = "launch_context_verification"
STAGE_AUTHORIZED_RUNTIME_BUILD = "authorized_runtime_build"
STAGE_NUMERICAL_DISPATCH = "numerical_dispatch"
STAGE_PASS_1 = "pass_1"
STAGE_CONSUMED_AUTHORITY_PUBLICATION = "consumed_authority_publication"
STAGE_NUMERICAL_PIPELINE_RETURNED = "numerical_pipeline_returned"
STAGE_COMPLETED_RUN_VERIFICATION = "completed_run_verification"
STAGE_TERMINAL_RECEIPT_PUBLICATION = "terminal_receipt_publication"
STAGE_TERMINAL = "terminal"

GUIDED_NPM_TERMINAL_STAGES = (
    STAGE_WORKER_REQUEST_LOADING,
    STAGE_LAUNCH_CONTEXT_VERIFICATION,
    STAGE_AUTHORIZED_RUNTIME_BUILD,
    STAGE_NUMERICAL_DISPATCH,
    STAGE_PASS_1,
    STAGE_CONSUMED_AUTHORITY_PUBLICATION,
    STAGE_NUMERICAL_PIPELINE_RETURNED,
    STAGE_COMPLETED_RUN_VERIFICATION,
    STAGE_TERMINAL_RECEIPT_PUBLICATION,
    STAGE_TERMINAL,
)
_STAGE_INDEX = {name: index for index, name in enumerate(GUIDED_NPM_TERMINAL_STAGES)}

# Closed set of required-output roles.  Guided-NPM-authorized runs write these
# core analysis artifacts flat at the run directory root (verified directly
# against a real end-to-end execution) -- this is a materially different
# layout from tools/run_full_pipeline_deliverables.py's wrapper convention,
# which nests per-analysis outputs under "_analysis/<mode>_out/" and further
# requires per-ROI plot/table deliverables that depend on dataset richness
# (representative-session selection, calendar-day coverage) the authorized
# runtime does not guarantee for every valid input.  Reusing that wrapper
# contract's path list here would therefore fail closed on genuinely
# successful minimal runs, so the required set below is derived independently
# from the two facts that actually vary it: execution mode and whether
# feature extraction ran.  photometry_pipeline.run_completion_contract's pure
# sha256_file hashing convention is mirrored (not imported) so this module
# does not depend on wrapper-specific run_mode plumbing.
OUTPUT_ROLE_RUN_REPORT = "run_report"
OUTPUT_ROLE_RUN_METADATA = "run_metadata"
OUTPUT_ROLE_CONFIG_USED = "config_used"
OUTPUT_ROLE_TRACE_CACHE = "trace_cache"
OUTPUT_ROLE_FEATURES_CSV = "features_csv"
OUTPUT_ROLE_FEATURE_EVENT_PROVENANCE = "feature_event_provenance"

GUIDED_NPM_TERMINAL_OUTPUT_ROLES = (
    OUTPUT_ROLE_RUN_REPORT,
    OUTPUT_ROLE_RUN_METADATA,
    OUTPUT_ROLE_CONFIG_USED,
    OUTPUT_ROLE_TRACE_CACHE,
    OUTPUT_ROLE_FEATURES_CSV,
    OUTPUT_ROLE_FEATURE_EVENT_PROVENANCE,
)

_FIXED_ROLE_RELATIVE_PATHS = {
    OUTPUT_ROLE_RUN_REPORT: "run_report.json",
    OUTPUT_ROLE_RUN_METADATA: "run_metadata.json",
    OUTPUT_ROLE_CONFIG_USED: "config_used.yaml",
    OUTPUT_ROLE_FEATURES_CSV: os.path.join("features", "features.csv"),
    OUTPUT_ROLE_FEATURE_EVENT_PROVENANCE: os.path.join(
        "features", "feature_event_provenance.json"
    ),
}


class GuidedNpmTerminalError(RuntimeError):
    """A terminal-receipt construction, verification, or publication step refused."""


def required_guided_npm_terminal_output_roles(
    execution_mode: str, *, traces_only: bool
) -> tuple[str, ...]:
    """The exact required output roles for this execution mode -- never globbed."""
    if execution_mode not in ("phasic", "tonic"):
        raise ValueError("terminal_execution_mode_invalid")
    roles = [
        OUTPUT_ROLE_RUN_REPORT,
        OUTPUT_ROLE_RUN_METADATA,
        OUTPUT_ROLE_CONFIG_USED,
        OUTPUT_ROLE_TRACE_CACHE,
    ]
    if execution_mode == "phasic" and not traces_only:
        roles += [OUTPUT_ROLE_FEATURES_CSV, OUTPUT_ROLE_FEATURE_EVENT_PROVENANCE]
    return tuple(roles)


def _output_role_relative_path(role: str, execution_mode: str) -> str:
    if role == OUTPUT_ROLE_TRACE_CACHE:
        return f"{execution_mode}_trace_cache.h5"
    try:
        return _FIXED_ROLE_RELATIVE_PATHS[role]
    except KeyError as exc:
        raise ValueError("terminal_output_role_invalid") from exc


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("terminal_nonfinite")
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("terminal_mapping_key_invalid")
        return {key: _canonical(item) for key, item in value.items()}
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    raise ValueError(f"terminal_value_unsupported:{type(value).__name__}")


def _identity(value: Any, field_name: str, domain: str) -> str:
    payload = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name != field_name
    }
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + encode_canonical_value(_canonical(payload))
    ).hexdigest()


def _digest_payload(domain: str, payload: Any) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + encode_canonical_value(_canonical(payload))
    ).hexdigest()


@dataclass(frozen=True)
class GuidedNpmTerminalOutputRecord:
    output_role: str
    output_path: str
    canonical_relative_path: str
    source_size_bytes: int
    source_sha256: str
    source_mtime_ns: int
    source_device: int
    source_inode: int
    source_mode: int
    canonical_output_record_identity: str


@dataclass(frozen=True)
class GuidedNpmWorkerTerminalReceipt:
    receipt_schema_name: str
    receipt_schema_version: str
    receipt_contract_version: str

    source_worker_request_identity: str
    source_execution_request_identity: str
    source_launch_invocation_identity: str
    source_launch_context_identity: str
    source_execution_start_receipt_identity: str | None
    source_consumed_authority_receipt_identity: str | None

    application_build_identity: ApplicationBuildIdentity
    guided_plan_identity: str
    validation_revision: int
    execution_mode: str
    observed_process_id: int

    worker_request_artifact_path: str
    run_directory_path: str

    terminal_outcome: str
    terminal_stage: str
    worker_acceptance_status: str
    consumed_authority_status: str
    numerical_dispatch_status: str
    completion_status: str

    authorized_runtime_identity: str | None
    correction_authority_identity: str | None
    feature_authority_identity: str | None

    output_evidence: tuple[GuidedNpmTerminalOutputRecord, ...]
    completed_run_evidence_identity: str | None

    failure_category: str | None
    failure_exception_type: str | None
    failure_detail_code: str | None

    canonical_terminal_receipt_identity: str


def compute_guided_npm_terminal_output_record_identity(
    value: GuidedNpmTerminalOutputRecord,
) -> str:
    return _identity(
        value,
        "canonical_output_record_identity",
        GUIDED_NPM_TERMINAL_OUTPUT_RECORD_CONTRACT_VERSION,
    )


def compute_guided_npm_worker_terminal_receipt_identity(
    value: GuidedNpmWorkerTerminalReceipt,
) -> str:
    return _identity(
        value,
        "canonical_terminal_receipt_identity",
        GUIDED_NPM_TERMINAL_RECEIPT_CONTRACT_VERSION,
    )


def compute_guided_npm_completed_run_evidence_identity(
    *,
    source_worker_request_identity: str,
    source_execution_request_identity: str,
    source_consumed_authority_receipt_identity: str,
    guided_plan_identity: str,
    validation_revision: int,
    output_evidence: tuple[GuidedNpmTerminalOutputRecord, ...],
) -> str:
    """Bind output evidence to this exact attempt, not merely to the worker.

    Including the consumed-authority receipt identity (fresh every launch
    attempt, even when a failed attempt's run directory is reused for a
    retry) prevents a stale output left over from an earlier attempt in the
    same directory from silently satisfying a later attempt's success claim.
    """
    payload = {
        "source_worker_request_identity": source_worker_request_identity,
        "source_execution_request_identity": source_execution_request_identity,
        "source_consumed_authority_receipt_identity": (
            source_consumed_authority_receipt_identity
        ),
        "guided_plan_identity": guided_plan_identity,
        "validation_revision": validation_revision,
        "output_evidence": output_evidence,
    }
    return _digest_payload(GUIDED_NPM_COMPLETED_RUN_EVIDENCE_IDENTITY_DOMAIN, payload)


def expected_guided_npm_terminal_receipt_path(run_directory_path: str) -> str:
    return os.path.join(run_directory_path, GUIDED_NPM_TERMINAL_RECEIPT_FILENAME)


def _stat_facts(item: os.stat_result) -> tuple[int, int, int, int, int]:
    return (item.st_size, item.st_mtime_ns, item.st_dev, item.st_ino, item.st_mode)


def build_guided_npm_terminal_output_record(
    run_directory_path: str,
    role: str,
    execution_mode: str,
    *,
    not_before_mtime_ns: int | None = None,
) -> GuidedNpmTerminalOutputRecord:
    """Verify and describe one required output artifact -- never by globbing.

    Requires a regular, non-symlink, single-link file at the exact
    deterministic path for this role and execution mode.  When
    ``not_before_mtime_ns`` is given, the artifact must not predate it: a
    stale file left over from strictly before this attempt's own
    consumed-authority acknowledgement cannot satisfy this attempt's
    completion evidence merely by continuing to exist.
    """
    if role not in GUIDED_NPM_TERMINAL_OUTPUT_ROLES:
        raise GuidedNpmTerminalError("terminal_output_role_invalid")
    relative_path = _output_role_relative_path(role, execution_mode)
    run_dir = Path(run_directory_path)
    target = run_dir / relative_path
    resolved_run_dir = os.path.realpath(os.fspath(run_dir))
    resolved_target = os.path.realpath(os.fspath(target))
    if os.path.commonpath([resolved_run_dir, resolved_target]) != resolved_run_dir:
        raise GuidedNpmTerminalError("terminal_output_path_outside_run_directory")
    try:
        before = target.stat(follow_symlinks=False)
    except OSError as exc:
        raise GuidedNpmTerminalError("terminal_output_missing") from exc
    if stat.S_ISLNK(before.st_mode):
        raise GuidedNpmTerminalError("terminal_output_is_symlink")
    if not stat.S_ISREG(before.st_mode):
        raise GuidedNpmTerminalError("terminal_output_not_regular")
    if before.st_nlink != 1:
        raise GuidedNpmTerminalError("terminal_output_unexpected_link_count")
    if not_before_mtime_ns is not None and before.st_mtime_ns < not_before_mtime_ns:
        raise GuidedNpmTerminalError("terminal_output_stale")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = target.stat(follow_symlinks=False)
    if _stat_facts(before) != _stat_facts(after):
        raise GuidedNpmTerminalError("terminal_output_changed_during_read")
    record = GuidedNpmTerminalOutputRecord(
        role,
        os.fspath(target),
        relative_path.replace(os.sep, "/"),
        int(before.st_size),
        digest.hexdigest(),
        int(before.st_mtime_ns),
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_mode),
        "0" * 64,
    )
    return replace(
        record,
        canonical_output_record_identity=(
            compute_guided_npm_terminal_output_record_identity(record)
        ),
    )


def build_guided_npm_required_output_evidence(
    run_directory_path: str,
    execution_mode: str,
    *,
    traces_only: bool,
    not_before_mtime_ns: int | None = None,
) -> tuple[GuidedNpmTerminalOutputRecord, ...]:
    roles = required_guided_npm_terminal_output_roles(
        execution_mode, traces_only=traces_only
    )
    records = tuple(
        build_guided_npm_terminal_output_record(
            run_directory_path,
            role,
            execution_mode,
            not_before_mtime_ns=not_before_mtime_ns,
        )
        for role in roles
    )
    roles_seen = [record.output_role for record in records]
    paths_seen = [record.canonical_relative_path for record in records]
    if len(set(roles_seen)) != len(roles_seen) or len(set(paths_seen)) != len(paths_seen):
        raise GuidedNpmTerminalError("terminal_output_duplicate_role_or_path")
    return records


def _base_receipt_kwargs(
    *,
    worker_request,
    launch_context,
    observed_process_id: int,
    terminal_stage: str,
) -> dict[str, Any]:
    execution = worker_request.execution_request
    return dict(
        receipt_schema_name=GUIDED_NPM_TERMINAL_RECEIPT_SCHEMA_NAME,
        receipt_schema_version=GUIDED_NPM_TERMINAL_RECEIPT_SCHEMA_VERSION,
        receipt_contract_version=GUIDED_NPM_TERMINAL_RECEIPT_CONTRACT_VERSION,
        source_worker_request_identity=worker_request.canonical_worker_request_identity,
        source_execution_request_identity=(
            execution.canonical_execution_request_identity
        ),
        source_launch_invocation_identity=launch_context.source_launch_invocation_identity,
        source_launch_context_identity=launch_context.canonical_launch_context_identity,
        source_execution_start_receipt_identity=None,
        application_build_identity=worker_request.application_build_identity,
        guided_plan_identity=worker_request.guided_plan_identity,
        validation_revision=worker_request.validation_revision,
        execution_mode=execution.execution_mode,
        observed_process_id=observed_process_id,
        worker_request_artifact_path=worker_request.worker_request_artifact_path,
        run_directory_path=worker_request.run_directory_path,
        terminal_stage=terminal_stage,
        worker_acceptance_status="accepted_exact_worker_authority",
    )


def build_guided_npm_worker_terminal_success_receipt(
    *,
    worker_request,
    launch_context,
    consumed_authority_receipt,
    observed_process_id: int,
    output_evidence: tuple[GuidedNpmTerminalOutputRecord, ...],
) -> GuidedNpmWorkerTerminalReceipt:
    if isinstance(observed_process_id, bool) or not isinstance(observed_process_id, int) or observed_process_id <= 0:
        raise GuidedNpmTerminalError("terminal_process_id_invalid")
    if consumed_authority_receipt.observed_process_id != observed_process_id:
        raise GuidedNpmTerminalError("terminal_process_id_mismatch")
    completed_run_evidence_identity = compute_guided_npm_completed_run_evidence_identity(
        source_worker_request_identity=worker_request.canonical_worker_request_identity,
        source_execution_request_identity=(
            worker_request.execution_request.canonical_execution_request_identity
        ),
        source_consumed_authority_receipt_identity=(
            consumed_authority_receipt.canonical_consumed_authority_receipt_identity
        ),
        guided_plan_identity=worker_request.guided_plan_identity,
        validation_revision=worker_request.validation_revision,
        output_evidence=output_evidence,
    )
    kwargs = _base_receipt_kwargs(
        worker_request=worker_request,
        launch_context=launch_context,
        observed_process_id=observed_process_id,
        terminal_stage=STAGE_TERMINAL,
    )
    kwargs.update(
        source_consumed_authority_receipt_identity=(
            consumed_authority_receipt.canonical_consumed_authority_receipt_identity
        ),
        terminal_outcome=TERMINAL_OUTCOME_COMPLETED,
        consumed_authority_status="verified",
        numerical_dispatch_status="entered",
        completion_status="verified_completed",
        authorized_runtime_identity=consumed_authority_receipt.authorized_runtime_identity,
        correction_authority_identity=(
            consumed_authority_receipt.correction_authority_identity
        ),
        feature_authority_identity=consumed_authority_receipt.feature_authority_identity,
        output_evidence=output_evidence,
        completed_run_evidence_identity=completed_run_evidence_identity,
        failure_category=None,
        failure_exception_type=None,
        failure_detail_code=None,
        canonical_terminal_receipt_identity="0" * 64,
    )
    receipt = GuidedNpmWorkerTerminalReceipt(**kwargs)
    receipt = replace(
        receipt,
        canonical_terminal_receipt_identity=(
            compute_guided_npm_worker_terminal_receipt_identity(receipt)
        ),
    )
    verify_guided_npm_worker_terminal_receipt(
        receipt,
        worker_request=worker_request,
        launch_context=launch_context,
        consumed_authority_receipt=consumed_authority_receipt,
    )
    return receipt


def build_guided_npm_worker_terminal_failure_receipt(
    *,
    worker_request,
    launch_context,
    observed_process_id: int,
    terminal_outcome: str,
    terminal_stage: str,
    consumed_authority_receipt=None,
    failure_category: str,
    failure_exception_type: str = "",
    failure_detail_code: str = "",
) -> GuidedNpmWorkerTerminalReceipt:
    if terminal_outcome not in _FAILURE_OUTCOMES:
        raise GuidedNpmTerminalError("terminal_failure_outcome_invalid")
    if isinstance(observed_process_id, bool) or not isinstance(observed_process_id, int) or observed_process_id <= 0:
        raise GuidedNpmTerminalError("terminal_process_id_invalid")
    consumed_available = consumed_authority_receipt is not None
    if terminal_outcome == TERMINAL_OUTCOME_FAILED_BEFORE_CONSUMED and consumed_available:
        raise GuidedNpmTerminalError("terminal_failure_outcome_stage_mismatch")
    if terminal_outcome != TERMINAL_OUTCOME_FAILED_BEFORE_CONSUMED and not consumed_available:
        raise GuidedNpmTerminalError("terminal_failure_outcome_stage_mismatch")
    kwargs = _base_receipt_kwargs(
        worker_request=worker_request,
        launch_context=launch_context,
        observed_process_id=observed_process_id,
        terminal_stage=terminal_stage,
    )
    kwargs.update(
        source_consumed_authority_receipt_identity=(
            consumed_authority_receipt.canonical_consumed_authority_receipt_identity
            if consumed_available
            else None
        ),
        terminal_outcome=terminal_outcome,
        consumed_authority_status=("verified" if consumed_available else "not_available"),
        numerical_dispatch_status=(
            "entered"
            if terminal_stage != STAGE_AUTHORIZED_RUNTIME_BUILD
            else "not_available"
        ),
        completion_status="failed",
        authorized_runtime_identity=(
            consumed_authority_receipt.authorized_runtime_identity
            if consumed_available
            else None
        ),
        correction_authority_identity=(
            consumed_authority_receipt.correction_authority_identity
            if consumed_available
            else None
        ),
        feature_authority_identity=(
            consumed_authority_receipt.feature_authority_identity
            if consumed_available
            else None
        ),
        output_evidence=(),
        completed_run_evidence_identity=None,
        failure_category=str(failure_category),
        failure_exception_type=str(failure_exception_type or ""),
        failure_detail_code=str(failure_detail_code or ""),
        canonical_terminal_receipt_identity="0" * 64,
    )
    receipt = GuidedNpmWorkerTerminalReceipt(**kwargs)
    receipt = replace(
        receipt,
        canonical_terminal_receipt_identity=(
            compute_guided_npm_worker_terminal_receipt_identity(receipt)
        ),
    )
    verify_guided_npm_worker_terminal_receipt(
        receipt,
        worker_request=worker_request,
        launch_context=launch_context,
        consumed_authority_receipt=consumed_authority_receipt,
    )
    return receipt


def verify_guided_npm_worker_terminal_receipt(
    receipt: GuidedNpmWorkerTerminalReceipt,
    *,
    worker_request,
    launch_context,
    consumed_authority_receipt=None,
) -> None:
    """Purely reconcile a terminal receipt. Performs no process or file access."""
    if type(receipt) is not GuidedNpmWorkerTerminalReceipt:
        raise ValueError("terminal_receipt_type_invalid")
    execution = worker_request.execution_request
    if (
        (receipt.receipt_schema_name, receipt.receipt_schema_version, receipt.receipt_contract_version)
        != (GUIDED_NPM_TERMINAL_RECEIPT_SCHEMA_NAME, GUIDED_NPM_TERMINAL_RECEIPT_SCHEMA_VERSION, GUIDED_NPM_TERMINAL_RECEIPT_CONTRACT_VERSION)
        or receipt.source_worker_request_identity != worker_request.canonical_worker_request_identity
        or receipt.source_execution_request_identity != execution.canonical_execution_request_identity
        or receipt.source_launch_invocation_identity != launch_context.source_launch_invocation_identity
        or receipt.source_launch_context_identity != launch_context.canonical_launch_context_identity
        or receipt.application_build_identity != worker_request.application_build_identity
        or receipt.guided_plan_identity != worker_request.guided_plan_identity
        or receipt.validation_revision != worker_request.validation_revision
        or receipt.execution_mode != execution.execution_mode
        or isinstance(receipt.observed_process_id, bool)
        or not isinstance(receipt.observed_process_id, int)
        or receipt.observed_process_id <= 0
        or receipt.worker_request_artifact_path != worker_request.worker_request_artifact_path
        or receipt.run_directory_path != worker_request.run_directory_path
        or receipt.terminal_outcome not in GUIDED_NPM_TERMINAL_OUTCOMES
        or receipt.terminal_stage not in _STAGE_INDEX
        or receipt.worker_acceptance_status != "accepted_exact_worker_authority"
        # Schema v1 deliberately defers execution-start-receipt binding to a
        # later parent-reconciliation patch; no receipt under this contract
        # version may claim to have verified it.
        or receipt.source_execution_start_receipt_identity is not None
    ):
        raise ValueError("terminal_receipt_invalid")

    if receipt.terminal_outcome == TERMINAL_OUTCOME_COMPLETED:
        if consumed_authority_receipt is None:
            raise ValueError("terminal_receipt_completed_requires_consumed_receipt")
        if (
            receipt.terminal_stage != STAGE_TERMINAL
            or receipt.consumed_authority_status != "verified"
            or receipt.numerical_dispatch_status != "entered"
            or receipt.completion_status != "verified_completed"
            or receipt.source_consumed_authority_receipt_identity
            != consumed_authority_receipt.canonical_consumed_authority_receipt_identity
            or receipt.observed_process_id != consumed_authority_receipt.observed_process_id
            or receipt.authorized_runtime_identity != consumed_authority_receipt.authorized_runtime_identity
            or receipt.correction_authority_identity != consumed_authority_receipt.correction_authority_identity
            or receipt.feature_authority_identity != consumed_authority_receipt.feature_authority_identity
            or not receipt.output_evidence
            or receipt.completed_run_evidence_identity is None
            or receipt.failure_category is not None
            or receipt.failure_exception_type is not None
            or receipt.failure_detail_code is not None
        ):
            raise ValueError("terminal_receipt_completed_invalid")
        base_roles = required_guided_npm_terminal_output_roles(
            receipt.execution_mode, traces_only=True
        )
        full_roles = required_guided_npm_terminal_output_roles(
            receipt.execution_mode, traces_only=False
        )
        valid_role_sequences = (
            (base_roles,) if receipt.execution_mode == "tonic" else (base_roles, full_roles)
        )
        observed_roles = tuple(record.output_role for record in receipt.output_evidence)
        if observed_roles not in valid_role_sequences:
            raise ValueError("terminal_receipt_output_roles_invalid")
        style = execution.output_runtime_projection.output_base_path_style
        seen_paths = set()
        for record in receipt.output_evidence:
            if type(record) is not GuidedNpmTerminalOutputRecord:
                raise ValueError("terminal_output_record_type_invalid")
            if record.output_role not in GUIDED_NPM_TERMINAL_OUTPUT_ROLES:
                raise ValueError("terminal_output_record_role_invalid")
            if record.canonical_relative_path in seen_paths:
                raise ValueError("terminal_output_record_duplicate_path")
            seen_paths.add(record.canonical_relative_path)
            expected_relative = _output_role_relative_path(
                record.output_role, receipt.execution_mode
            )
            expected_absolute = os.path.join(receipt.run_directory_path, expected_relative)
            if (
                record.canonical_relative_path.replace("/", os.sep) != expected_relative
                or os.path.normpath(record.canonical_relative_path).startswith("..")
                or os.path.isabs(record.canonical_relative_path)
                # The stored absolute output_path must be the exact
                # deterministic path inside this receipt's own authorized run
                # directory -- not merely a path whose recomputed identities
                # happen to match, and not merely lexically similar to it.
                or not os.path.isabs(record.output_path)
                or not stored_paths_equal(record.output_path, expected_absolute, style)
            ):
                raise ValueError("terminal_output_record_path_invalid")
            if (
                isinstance(record.source_size_bytes, bool)
                or not isinstance(record.source_size_bytes, int)
                or record.source_size_bytes < 0
                or len(record.source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in record.source_sha256)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (
                        record.source_mtime_ns,
                        record.source_device,
                        record.source_inode,
                        record.source_mode,
                    )
                )
                or compute_guided_npm_terminal_output_record_identity(record)
                != record.canonical_output_record_identity
            ):
                raise ValueError("terminal_output_record_invalid")
        expected_evidence_identity = compute_guided_npm_completed_run_evidence_identity(
            source_worker_request_identity=receipt.source_worker_request_identity,
            source_execution_request_identity=receipt.source_execution_request_identity,
            source_consumed_authority_receipt_identity=(
                receipt.source_consumed_authority_receipt_identity
            ),
            guided_plan_identity=receipt.guided_plan_identity,
            validation_revision=receipt.validation_revision,
            output_evidence=receipt.output_evidence,
        )
        if expected_evidence_identity != receipt.completed_run_evidence_identity:
            raise ValueError("terminal_receipt_completed_run_evidence_mismatch")
    else:
        if (
            receipt.completion_status != "failed"
            or receipt.output_evidence != ()
            or receipt.completed_run_evidence_identity is not None
            or not receipt.failure_category
        ):
            raise ValueError("terminal_receipt_failure_invalid")
        consumed_available = consumed_authority_receipt is not None
        if receipt.terminal_outcome == TERMINAL_OUTCOME_FAILED_BEFORE_CONSUMED:
            if (
                consumed_available
                or receipt.consumed_authority_status != "not_available"
                or receipt.source_consumed_authority_receipt_identity is not None
                or receipt.authorized_runtime_identity is not None
                or receipt.correction_authority_identity is not None
                or receipt.feature_authority_identity is not None
            ):
                raise ValueError("terminal_receipt_failed_before_consumed_invalid")
        else:
            if not consumed_available or receipt.consumed_authority_status != "verified":
                raise ValueError("terminal_receipt_failed_after_consumed_invalid")
            if (
                receipt.source_consumed_authority_receipt_identity
                != consumed_authority_receipt.canonical_consumed_authority_receipt_identity
                or receipt.authorized_runtime_identity != consumed_authority_receipt.authorized_runtime_identity
                or receipt.correction_authority_identity != consumed_authority_receipt.correction_authority_identity
                or receipt.feature_authority_identity != consumed_authority_receipt.feature_authority_identity
            ):
                raise ValueError("terminal_receipt_failed_after_consumed_invalid")

    if (
        compute_guided_npm_worker_terminal_receipt_identity(receipt)
        != receipt.canonical_terminal_receipt_identity
    ):
        raise ValueError("terminal_receipt_identity_mismatch")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("terminal_duplicate_json_key")
        result[key] = value
    return result


def _strict_fields(value: Any, cls: type) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(cls)}:
        raise ValueError("terminal_field_set_invalid")
    return dict(value)


def _decode_build(value: Any) -> ApplicationBuildIdentity:
    return ApplicationBuildIdentity(**_strict_fields(value, ApplicationBuildIdentity))


def _decode_output_record(value: Any) -> GuidedNpmTerminalOutputRecord:
    data = _strict_fields(value, GuidedNpmTerminalOutputRecord)
    return GuidedNpmTerminalOutputRecord(**data)


def serialize_guided_npm_worker_terminal_receipt(value: GuidedNpmWorkerTerminalReceipt) -> bytes:
    return _canonical_bytes(value)


def decode_guided_npm_worker_terminal_receipt_bytes(content: bytes) -> GuidedNpmWorkerTerminalReceipt:
    try:
        data = _strict_fields(
            json.loads(content, object_pairs_hook=_strict_object),
            GuidedNpmWorkerTerminalReceipt,
        )
        data["application_build_identity"] = _decode_build(data["application_build_identity"])
        records = data["output_evidence"]
        if not isinstance(records, list):
            raise ValueError("terminal_output_evidence_invalid")
        data["output_evidence"] = tuple(_decode_output_record(item) for item in records)
        result = GuidedNpmWorkerTerminalReceipt(**data)
        if serialize_guided_npm_worker_terminal_receipt(result) != content:
            raise ValueError("terminal_receipt_noncanonical")
        return result
    except Exception as exc:
        raise ValueError("terminal_receipt_decode_invalid") from exc


def _publish_new_exclusive(path: str, content: bytes) -> None:
    target = Path(path)
    parent = target.parent
    if target.name != GUIDED_NPM_TERMINAL_RECEIPT_FILENAME:
        raise GuidedNpmTerminalError("terminal_destination_invalid")
    try:
        parent_facts = parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise GuidedNpmTerminalError("terminal_destination_invalid") from exc
    if (
        stat.S_ISLNK(parent_facts.st_mode)
        or not stat.S_ISDIR(parent_facts.st_mode)
        or target.exists()
        or target.is_symlink()
    ):
        raise GuidedNpmTerminalError("terminal_destination_conflict")
    temporary = parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise GuidedNpmTerminalError("terminal_temporary_conflict")
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
        raise GuidedNpmTerminalError("terminal_artifact_not_regular")
    with target.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        content = handle.read()
        after_open = os.fstat(handle.fileno())
    after = target.stat(follow_symlinks=False)
    if not (_stat_facts(before) == _stat_facts(opened) == _stat_facts(after_open) == _stat_facts(after)):
        raise GuidedNpmTerminalError("terminal_artifact_changed_during_read")
    return content


def publish_guided_npm_worker_terminal_receipt(
    receipt: GuidedNpmWorkerTerminalReceipt,
) -> str:
    """Publish exactly once. Refuses (never overwrites) a pre-existing destination."""
    path = expected_guided_npm_terminal_receipt_path(receipt.run_directory_path)
    _publish_new_exclusive(path, serialize_guided_npm_worker_terminal_receipt(receipt))
    observed = decode_guided_npm_worker_terminal_receipt_bytes(_stable_read(path))
    if observed != receipt:
        raise GuidedNpmTerminalError("terminal_receipt_reread_mismatch")
    return path


def read_guided_npm_worker_terminal_receipt(
    path: str, *, worker_request
) -> GuidedNpmWorkerTerminalReceipt:
    expected = expected_guided_npm_terminal_receipt_path(worker_request.run_directory_path)
    style = worker_request.execution_request.output_runtime_projection.output_base_path_style
    if not stored_paths_equal(path, expected, style):
        raise ValueError("terminal_receipt_path_invalid")
    return decode_guided_npm_worker_terminal_receipt_bytes(_stable_read(path))
