"""B2-C6B2A path-based worker claim and final prelaunch verification.

This module creates an in-memory, non-runnable claim.  It performs no launch,
constructs no command, and persists no artifact.  B2-C6B2B must accept only a
fully verified claim and remains responsible for any future launch transaction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import math
import ntpath
import os
from pathlib import Path
import posixpath
import stat
from typing import Any

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_startup_persistence import verify_application_build_identity
from photometry_pipeline.guided_npm_worker_request import (
    GUIDED_NPM_WORKER_REQUEST_FILENAME,
    GuidedNpmLiveFileFacts,
    GuidedNpmLiveSourceFreshnessEvidence,
    GuidedNpmWorkerRequest,
    decode_canonical_guided_npm_worker_request_bytes,
    verify_guided_npm_live_source_freshness_evidence,
    verify_guided_npm_worker_request,
)
from photometry_pipeline.guided_npm_worker_request_materialization import (
    GuidedNpmWorkerRequestMaterializationReceipt,
    _Cancelled as _LiveCancelled,
    _Refusal as _LiveRefusal,
    compute_guided_npm_worker_request_materialization_receipt_identity,
    verify_guided_npm_source_freshness_live,
    verify_guided_npm_startup_artifact_live,
    verify_guided_npm_worker_request_materialization_binding,
    verify_guided_npm_worker_request_materialization_receipt,
)
from photometry_pipeline.guided_production_mapping import ApplicationBuildIdentity


GUIDED_NPM_PRELAUNCH_FRESHNESS_SCHEMA_NAME = "guided_npm_prelaunch_freshness_evidence"
GUIDED_NPM_PRELAUNCH_FRESHNESS_SCHEMA_VERSION = "v1"
GUIDED_NPM_PRELAUNCH_FRESHNESS_CONTRACT_VERSION = "guided_npm_prelaunch_freshness_evidence.v1"
GUIDED_NPM_WORKER_PRELAUNCH_CLAIM_SCHEMA_NAME = "guided_npm_worker_prelaunch_claim"
GUIDED_NPM_WORKER_PRELAUNCH_CLAIM_SCHEMA_VERSION = "v1"
GUIDED_NPM_WORKER_PRELAUNCH_CLAIM_CONTRACT_VERSION = "guided_npm_worker_prelaunch_claim.v1"
_HASH_BLOCK_SIZE = 1024 * 1024
_HEX = frozenset("0123456789abcdef")

GUIDED_NPM_WORKER_PRELAUNCH_REFUSAL_CATEGORIES = (
    "worker_artifact_path_invalid", "worker_artifact_missing",
    "worker_artifact_alias_invalid", "worker_artifact_not_regular",
    "worker_artifact_mutated", "worker_artifact_size_mismatch",
    "worker_artifact_digest_mismatch", "worker_artifact_noncanonical",
    "worker_request_invalid", "worker_request_state_invalid",
    "materialization_receipt_invalid", "materialization_binding_mismatch",
    "current_build_invalid", "current_build_mismatch",
    "startup_artifact_missing", "startup_artifact_alias_invalid",
    "startup_artifact_mutated", "startup_artifact_size_mismatch",
    "startup_artifact_digest_mismatch", "startup_artifact_authority_mismatch",
    "source_root_missing", "source_root_alias_invalid", "source_root_replaced",
    "source_membership_missing", "source_membership_extra",
    "source_membership_changed", "source_file_missing",
    "source_file_alias_invalid", "source_file_not_regular",
    "source_file_size_mismatch", "source_file_digest_mismatch",
    "source_file_mutated", "source_runtime_identity_mismatch",
    "prelaunch_claim_cancelled", "prelaunch_claim_identity_mismatch",
    "prelaunch_claim_internal_error",
)
_CATEGORY_SET = frozenset(GUIDED_NPM_WORKER_PRELAUNCH_REFUSAL_CATEGORIES)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("prelaunch_nonfinite")
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("prelaunch_mapping_key_invalid")
        return {key: _canonical(item) for key, item in value.items()}
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    raise ValueError(f"prelaunch_value_unsupported:{type(value).__name__}")


def _identity(value: Any, field_name: str, domain: str) -> str:
    payload = {item.name: getattr(value, item.name) for item in fields(value) if item.name != field_name}
    return hashlib.sha256(domain.encode() + b"\x00" + encode_canonical_value(_canonical(payload))).hexdigest()


@dataclass(frozen=True)
class GuidedNpmPrelaunchFreshnessEvidence:
    freshness_schema_name: str
    freshness_schema_version: str
    freshness_contract_version: str
    source_worker_request_identity: str
    source_execution_request_identity: str
    source_materialization_receipt_identity: str
    worker_artifact_path: str
    worker_artifact_size_bytes: int
    worker_artifact_sha256: str
    worker_artifact_pre_facts: GuidedNpmLiveFileFacts
    worker_artifact_opened_facts: GuidedNpmLiveFileFacts
    worker_artifact_post_read_facts: GuidedNpmLiveFileFacts
    worker_artifact_final_facts: GuidedNpmLiveFileFacts
    startup_artifact_path: str
    startup_artifact_size_bytes: int
    startup_artifact_sha256: str
    startup_artifact_final_facts: GuidedNpmLiveFileFacts
    source_freshness_evidence: GuidedNpmLiveSourceFreshnessEvidence
    current_application_build_identity: ApplicationBuildIdentity
    freshness_status: str
    canonical_prelaunch_freshness_evidence_identity: str


@dataclass(frozen=True)
class GuidedNpmWorkerPrelaunchClaim:
    claim_schema_name: str
    claim_schema_version: str
    claim_contract_version: str
    source_worker_request_identity: str
    source_execution_request_identity: str
    source_materialization_receipt_identity: str
    source_prelaunch_freshness_evidence_identity: str
    worker_request_artifact_path: str
    worker_request_artifact_sha256: str
    worker_request_artifact_size_bytes: int
    application_build_identity: ApplicationBuildIdentity
    guided_plan_identity: str
    validation_revision: int
    execution_mode: str
    run_directory_path: str
    worker_request: GuidedNpmWorkerRequest
    materialization_receipt: GuidedNpmWorkerRequestMaterializationReceipt
    prelaunch_freshness_evidence: GuidedNpmPrelaunchFreshnessEvidence
    claim_status: str
    launch_status: str
    execution_status: str
    completion_status: str
    runnable: bool
    canonical_prelaunch_claim_identity: str


@dataclass(frozen=True)
class GuidedNpmWorkerPrelaunchClaimIssue:
    category: str
    section: str
    message: str
    detail_code: str


@dataclass(frozen=True)
class GuidedNpmWorkerPrelaunchClaimFailure:
    blocking_issues: tuple[GuidedNpmWorkerPrelaunchClaimIssue, ...]
    status: str = "refused"


@dataclass(frozen=True)
class GuidedNpmWorkerPrelaunchClaimCancelled:
    blocking_issues: tuple[GuidedNpmWorkerPrelaunchClaimIssue, ...]
    status: str = "cancelled"


GuidedNpmWorkerPrelaunchClaimResult = (
    GuidedNpmWorkerPrelaunchClaim
    | GuidedNpmWorkerPrelaunchClaimFailure
    | GuidedNpmWorkerPrelaunchClaimCancelled
)


class GuidedNpmPrelaunchAuthorityLiveCancelled(RuntimeError):
    """Cancellation observed while performing the immediate launch gate."""



class _Refusal(ValueError):
    def __init__(self, category: str, section: str, message: str, detail_code: str):
        self.category, self.section, self.message, self.detail_code = category, section, message, detail_code
        super().__init__(message)


class _Cancelled(RuntimeError):
    pass


def _refuse(category: str, section: str, message: str, detail_code: str) -> None:
    raise _Refusal(category if category in _CATEGORY_SET else "prelaunch_claim_internal_error", section, message, detail_code)


def _check_cancelled(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise _Cancelled("prelaunch_claim_cancelled")


def verify_stored_path_style(path: str, expected_style: str) -> None:
    """Purely require one canonical absolute path in the declared path style."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path_invalid")
    if expected_style == "windows_drive":
        drive, _ = ntpath.splitdrive(path)
        if (
            not ntpath.isabs(path)
            or len(drive) != 2
            or drive[1:] != ":"
            or path.startswith(("\\\\", "//"))
            or posixpath.isabs(path)
        ):
            raise ValueError("path_style_mismatch")
        module = ntpath
    elif expected_style == "posix_absolute":
        if not posixpath.isabs(path) or ntpath.splitdrive(path)[0]:
            raise ValueError("path_style_mismatch")
        module = posixpath
    else:
        raise ValueError("path_style_unsupported")
    if module.normpath(path) != path or ".." in path.replace("\\", "/").split("/"):
        raise ValueError("path_not_canonical")


def stored_paths_equal(left: str, right: str, expected_style: str) -> bool:
    """Compare two validated stored paths without host or filesystem semantics."""
    verify_stored_path_style(left, expected_style)
    verify_stored_path_style(right, expected_style)
    module = ntpath if expected_style == "windows_drive" else posixpath
    if expected_style == "windows_drive":
        return module.normcase(module.normpath(left)) == module.normcase(module.normpath(right))
    return module.normpath(left) == module.normpath(right)


def stored_parent_and_name(path: str, expected_style: str) -> tuple[str, str]:
    """Return a validated stored parent and filename in the declared style."""
    verify_stored_path_style(path, expected_style)
    module = ntpath if expected_style == "windows_drive" else posixpath
    normalized = module.normpath(path)
    return module.dirname(normalized), module.basename(normalized)


def _stored_name_equal(left: str, right: str, style: str) -> bool:
    return ntpath.normcase(left) == ntpath.normcase(right) if style == "windows_drive" else left == right


def _path_model(path: str) -> tuple[str, Any]:
    if isinstance(path, str) and ntpath.isabs(path) and len(ntpath.splitdrive(path)[0]) == 2:
        style, module = "windows_drive", ntpath
    elif isinstance(path, str) and posixpath.isabs(path):
        style, module = "posix_absolute", posixpath
    else:
        raise ValueError("path_not_absolute")
    verify_stored_path_style(path, style)
    return style, module


_same_stored_path = stored_paths_equal


def _file_facts(path: Path, metadata: os.stat_result) -> GuidedNpmLiveFileFacts:
    return GuidedNpmLiveFileFacts(
        os.fspath(path), metadata.st_size, metadata.st_mtime_ns, "regular_file",
        metadata.st_dev, metadata.st_ino, metadata.st_mode,
    )


def _stable_tuple(facts: GuidedNpmLiveFileFacts) -> tuple[int, ...]:
    return facts.size_bytes, facts.mtime_ns, facts.device, facts.inode, facts.mode


def _fact_metadata(facts: GuidedNpmLiveFileFacts) -> tuple[Any, ...]:
    return (
        facts.size_bytes, facts.mtime_ns, facts.file_type,
        facts.device, facts.inode, facts.mode,
    )


def _inspect_worker_artifact(path: Path) -> GuidedNpmLiveFileFacts:
    if not os.path.lexists(path):
        _refuse("worker_artifact_missing", "artifact", "The worker artifact is missing.", "worker_artifact_missing")
    try:
        metadata = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        _refuse("worker_artifact_alias_invalid", "artifact", "The worker artifact could not be resolved safely.", type(exc).__name__)
    if stat.S_ISLNK(metadata.st_mode) or not _same_stored_path(os.fspath(path), os.fspath(resolved), "windows_drive" if os.name == "nt" else "posix_absolute"):
        _refuse("worker_artifact_alias_invalid", "artifact", "Worker artifact aliases are not accepted.", "worker_artifact_alias_invalid")
    if not stat.S_ISREG(metadata.st_mode):
        _refuse("worker_artifact_not_regular", "artifact", "The worker artifact is not a regular file.", "worker_artifact_not_regular")
    return _file_facts(path, metadata)


def _stable_read_worker_artifact(path: Path, check):
    _check_cancelled(check)
    pre = _inspect_worker_artifact(path)
    digest = hashlib.sha256()
    content = bytearray()
    try:
        with path.open("rb") as handle:
            opened = _file_facts(path, os.fstat(handle.fileno()))
            if _stable_tuple(pre) != _stable_tuple(opened):
                _refuse("worker_artifact_mutated", "artifact", "The worker artifact changed while opening.", "pre_open_facts_changed")
            while True:
                _check_cancelled(check)
                block = handle.read(_HASH_BLOCK_SIZE)
                if not block:
                    break
                digest.update(block)
                content.extend(block)
            post = _file_facts(path, os.fstat(handle.fileno()))
    except (_Refusal, _Cancelled):
        raise
    except OSError as exc:
        _refuse("worker_artifact_mutated", "artifact", "The worker artifact could not be read stably.", type(exc).__name__)
    final = _inspect_worker_artifact(path)
    if not (_stable_tuple(pre) == _stable_tuple(opened) == _stable_tuple(post) == _stable_tuple(final)):
        _refuse("worker_artifact_mutated", "artifact", "The worker artifact changed during verification.", "stable_facts_changed")
    return pre, opened, post, final, digest.hexdigest(), bytes(content)


def compute_guided_npm_prelaunch_freshness_evidence_identity(value) -> str:
    return _identity(value, "canonical_prelaunch_freshness_evidence_identity", GUIDED_NPM_PRELAUNCH_FRESHNESS_CONTRACT_VERSION)


def compute_guided_npm_worker_prelaunch_claim_identity(value) -> str:
    return _identity(value, "canonical_prelaunch_claim_identity", GUIDED_NPM_WORKER_PRELAUNCH_CLAIM_CONTRACT_VERSION)


def verify_guided_npm_prelaunch_freshness_evidence(
    evidence: GuidedNpmPrelaunchFreshnessEvidence,
    worker_request: GuidedNpmWorkerRequest,
    receipt: GuidedNpmWorkerRequestMaterializationReceipt,
) -> None:
    if type(evidence) is not GuidedNpmPrelaunchFreshnessEvidence:
        raise ValueError("prelaunch_freshness_type_invalid")
    verify_guided_npm_worker_request_materialization_binding(receipt, worker_request)
    execution = worker_request.execution_request
    path_style = execution.output_runtime_projection.output_base_path_style
    try:
        worker_paths_match = all(
            stored_paths_equal(path, worker_request.worker_request_artifact_path, path_style)
            for path in (
                evidence.worker_artifact_path,
                receipt.worker_request_artifact_path,
            )
        )
        startup_paths_match = all(
            stored_paths_equal(path, execution.startup_artifact_path, path_style)
            for path in (
                evidence.startup_artifact_path,
                worker_request.startup_artifact_path,
            )
        )
    except ValueError as exc:
        raise ValueError("prelaunch_freshness_path_invalid") from exc
    if (
        (evidence.freshness_schema_name, evidence.freshness_schema_version, evidence.freshness_contract_version)
        != (GUIDED_NPM_PRELAUNCH_FRESHNESS_SCHEMA_NAME, GUIDED_NPM_PRELAUNCH_FRESHNESS_SCHEMA_VERSION, GUIDED_NPM_PRELAUNCH_FRESHNESS_CONTRACT_VERSION)
        or evidence.freshness_status != "verified_for_prelaunch"
        or evidence.source_worker_request_identity != worker_request.canonical_worker_request_identity
        or evidence.source_execution_request_identity != execution.canonical_execution_request_identity
        or evidence.source_materialization_receipt_identity != receipt.canonical_materialization_receipt_identity
        or not worker_paths_match
        or evidence.worker_artifact_size_bytes != receipt.worker_request_artifact_size_bytes
        or evidence.worker_artifact_sha256 != receipt.worker_request_artifact_sha256
        or not startup_paths_match
        or evidence.startup_artifact_size_bytes != execution.startup_artifact_size_bytes
        or evidence.startup_artifact_sha256 != execution.startup_artifact_sha256
        or evidence.current_application_build_identity != worker_request.application_build_identity
    ):
        raise ValueError("prelaunch_freshness_authority_mismatch")
    worker_facts = (
        evidence.worker_artifact_pre_facts, evidence.worker_artifact_opened_facts,
        evidence.worker_artifact_post_read_facts, evidence.worker_artifact_final_facts,
    )
    if any(type(item) is not GuidedNpmLiveFileFacts for item in worker_facts):
        raise ValueError("prelaunch_worker_artifact_mutated")
    try:
        worker_fact_paths_match = all(
            stored_paths_equal(item.canonical_path, evidence.worker_artifact_path, path_style)
            for item in worker_facts
        )
        startup_fact_path_matches = stored_paths_equal(
            evidence.startup_artifact_final_facts.canonical_path,
            evidence.startup_artifact_path,
            path_style,
        )
    except (AttributeError, ValueError) as exc:
        raise ValueError("prelaunch_artifact_facts_path_invalid") from exc
    if len({_fact_metadata(item) for item in worker_facts}) != 1:
        raise ValueError("prelaunch_worker_artifact_mutated")
    if (
        not worker_fact_paths_match
        or worker_facts[0].size_bytes != evidence.worker_artifact_size_bytes
        or type(evidence.startup_artifact_final_facts) is not GuidedNpmLiveFileFacts
        or not startup_fact_path_matches
        or evidence.startup_artifact_final_facts.size_bytes != evidence.startup_artifact_size_bytes
    ):
        raise ValueError("prelaunch_artifact_facts_mismatch")
    verify_guided_npm_live_source_freshness_evidence(evidence.source_freshness_evidence, execution)
    if compute_guided_npm_prelaunch_freshness_evidence_identity(evidence) != evidence.canonical_prelaunch_freshness_evidence_identity:
        raise ValueError("prelaunch_freshness_identity_mismatch")


def verify_guided_npm_worker_prelaunch_claim(claim: GuidedNpmWorkerPrelaunchClaim) -> None:
    """Purely verify one complete, non-launched B2-C6B2A claim."""
    if type(claim) is not GuidedNpmWorkerPrelaunchClaim:
        raise ValueError("prelaunch_claim_type_invalid")
    verify_guided_npm_worker_request(claim.worker_request)
    verify_guided_npm_worker_request_materialization_receipt(claim.materialization_receipt)
    verify_guided_npm_worker_request_materialization_binding(claim.materialization_receipt, claim.worker_request)
    verify_guided_npm_prelaunch_freshness_evidence(
        claim.prelaunch_freshness_evidence, claim.worker_request, claim.materialization_receipt
    )
    worker, receipt, evidence = claim.worker_request, claim.materialization_receipt, claim.prelaunch_freshness_evidence
    path_style = worker.execution_request.output_runtime_projection.output_base_path_style
    try:
        artifact_paths_match = all(
            stored_paths_equal(path, worker.worker_request_artifact_path, path_style)
            for path in (
                claim.worker_request_artifact_path,
                evidence.worker_artifact_path,
                receipt.worker_request_artifact_path,
            )
        )
        run_paths_match = all(
            stored_paths_equal(path, worker.run_directory_path, path_style)
            for path in (
                claim.run_directory_path,
                worker.execution_request.output_runtime_projection.run_directory_path,
                receipt.run_directory_path,
            )
        )
        artifact_parent, artifact_name = stored_parent_and_name(
            worker.worker_request_artifact_path, path_style
        )
        artifact_relationship_valid = (
            _stored_name_equal(artifact_name, GUIDED_NPM_WORKER_REQUEST_FILENAME, path_style)
            and stored_paths_equal(artifact_parent, worker.run_directory_path, path_style)
        )
    except ValueError as exc:
        raise ValueError("prelaunch_claim_path_invalid") from exc
    if (
        (claim.claim_schema_name, claim.claim_schema_version, claim.claim_contract_version)
        != (GUIDED_NPM_WORKER_PRELAUNCH_CLAIM_SCHEMA_NAME, GUIDED_NPM_WORKER_PRELAUNCH_CLAIM_SCHEMA_VERSION, GUIDED_NPM_WORKER_PRELAUNCH_CLAIM_CONTRACT_VERSION)
        or (claim.claim_status, claim.launch_status, claim.execution_status, claim.completion_status, claim.runnable)
        != ("verified_for_prelaunch", "not_launched", "not_started", "not_available", False)
        or claim.source_worker_request_identity != worker.canonical_worker_request_identity
        or claim.source_execution_request_identity != worker.source_execution_request_identity
        or claim.source_materialization_receipt_identity != receipt.canonical_materialization_receipt_identity
        or claim.source_prelaunch_freshness_evidence_identity != evidence.canonical_prelaunch_freshness_evidence_identity
        or not artifact_paths_match
        or claim.worker_request_artifact_sha256 != receipt.worker_request_artifact_sha256
        or claim.worker_request_artifact_size_bytes != receipt.worker_request_artifact_size_bytes
        or claim.application_build_identity != worker.application_build_identity
        or claim.guided_plan_identity != worker.guided_plan_identity
        or claim.validation_revision != worker.validation_revision
        or claim.execution_mode != worker.execution_request.execution_mode
        or not run_paths_match
        or not artifact_relationship_valid
        or (claim.launch_status, claim.execution_status, claim.completion_status, claim.runnable)
        != (worker.launch_status, worker.execution_status, worker.completion_status, worker.runnable)
        or compute_guided_npm_worker_request_materialization_receipt_identity(receipt)
        != claim.source_materialization_receipt_identity
        or compute_guided_npm_worker_prelaunch_claim_identity(claim) != claim.canonical_prelaunch_claim_identity
    ):
        raise ValueError("prelaunch_claim_identity_mismatch")


def verify_guided_npm_worker_prelaunch_authority_live(
    claim: GuidedNpmWorkerPrelaunchClaim,
    *,
    current_application_build_identity: ApplicationBuildIdentity,
    cancellation_check: Callable[[], bool] | None = None,
) -> None:
    """Immediately reverify frozen authority before process creation.

    This closes the parent-side claim-to-launch window as far as the current
    process can truthfully observe it.  It does not prove which bytes a future
    child opens; consumed-authority evidence remains a later lifecycle stage.
    """
    try:
        _check_cancelled(cancellation_check)
        verify_guided_npm_worker_prelaunch_claim(claim)
        verify_application_build_identity(current_application_build_identity)
        worker = claim.worker_request
        receipt = claim.materialization_receipt
        evidence = claim.prelaunch_freshness_evidence
        if not (
            current_application_build_identity
            == claim.application_build_identity
            == worker.application_build_identity
            == worker.execution_request.application_build_identity
            == receipt.application_build_identity
            == evidence.current_application_build_identity
        ):
            raise ValueError("current_build_mismatch")

        _check_cancelled(cancellation_check)
        path = Path(claim.worker_request_artifact_path)
        pre, opened, post, final, digest, content = _stable_read_worker_artifact(
            path, cancellation_check
        )
        if (
            len(content) != claim.worker_request_artifact_size_bytes
            or digest != claim.worker_request_artifact_sha256
            or digest != receipt.worker_request_artifact_sha256
        ):
            raise ValueError("launch_worker_artifact_changed")
        style = worker.execution_request.output_runtime_projection.output_base_path_style
        for facts in (pre, opened, post, final):
            if (
                not stored_paths_equal(
                    facts.canonical_path, claim.worker_request_artifact_path, style
                )
                or _fact_metadata(facts)
                != _fact_metadata(evidence.worker_artifact_final_facts)
            ):
                raise ValueError("launch_worker_artifact_changed")
        decoded = decode_canonical_guided_npm_worker_request_bytes(content)
        verify_guided_npm_worker_request(decoded)
        if decoded != worker or decoded.canonical_worker_request_identity != (
            claim.source_worker_request_identity
        ):
            raise ValueError("launch_worker_artifact_changed")
        verify_guided_npm_worker_request_materialization_binding(receipt, decoded)

        _check_cancelled(cancellation_check)
        startup_facts = verify_guided_npm_startup_artifact_live(
            worker.execution_request, cancellation_check
        )
        if (
            not stored_paths_equal(
                startup_facts.canonical_path, evidence.startup_artifact_path, style
            )
            or _fact_metadata(startup_facts)
            != _fact_metadata(evidence.startup_artifact_final_facts)
        ):
            raise ValueError("launch_startup_artifact_changed")

        _check_cancelled(cancellation_check)
        source_freshness = verify_guided_npm_source_freshness_live(
            worker.execution_request, cancellation_check
        )
        verify_guided_npm_live_source_freshness_evidence(
            source_freshness, worker.execution_request
        )
        if source_freshness != evidence.source_freshness_evidence:
            raise ValueError("launch_source_freshness_changed")
        _final_preclaim_gate(worker, evidence)
        _check_cancelled(cancellation_check)
    except (_Cancelled, _LiveCancelled) as exc:
        raise GuidedNpmPrelaunchAuthorityLiveCancelled(
            "launch_cancelled"
        ) from exc
    except _LiveRefusal as exc:
        if exc.category.startswith("startup_artifact_"):
            category = "launch_startup_artifact_changed"
        elif exc.category.startswith("source_"):
            category = "launch_source_freshness_changed"
        else:
            category = "launch_worker_artifact_changed"
        raise ValueError(category) from exc
    except _Refusal as exc:
        if exc.category.startswith("startup_artifact_"):
            category = "launch_startup_artifact_changed"
        elif exc.category.startswith("source_"):
            category = "launch_source_freshness_changed"
        else:
            category = "launch_worker_artifact_changed"
        raise ValueError(category) from exc


def _final_live_gate(path: Path, expected: GuidedNpmLiveFileFacts, category: str) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError:
        _refuse(category, "freshness", "A verified artifact disappeared before claim completion.", category)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _refuse(category, "freshness", "A verified artifact changed type before claim completion.", category)
    if _stable_tuple(_file_facts(path, metadata)) != _stable_tuple(expected):
        _refuse(category, "freshness", "A verified artifact changed before claim completion.", category)


def _final_preclaim_gate(worker, evidence) -> None:
    _final_live_gate(Path(evidence.worker_artifact_path), evidence.worker_artifact_final_facts, "worker_artifact_mutated")
    _final_live_gate(Path(evidence.startup_artifact_path), evidence.startup_artifact_final_facts, "startup_artifact_mutated")
    source = evidence.source_freshness_evidence
    root = Path(source.source_root_canonical)
    try:
        root_metadata = root.stat(follow_symlinks=False)
        entries = list(os.scandir(root))
    except OSError:
        _refuse("source_root_replaced", "freshness", "The source root changed before claim completion.", "source_root_replaced")
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        _refuse("source_root_replaced", "freshness", "The source root changed type.", "source_root_replaced")
    post = source.source_root_post_facts
    if (root_metadata.st_dev, root_metadata.st_ino, root_metadata.st_mode, root_metadata.st_mtime_ns) != (post.device, post.inode, post.mode, post.mtime_ns):
        _refuse("source_root_replaced", "freshness", "The source root metadata changed.", "source_root_replaced")
    members = {
        os.path.normcase(entry.name) if os.name == "nt" else entry.name
        for entry in entries
        if entry.name.lower().endswith(".csv") and entry.is_file(follow_symlinks=False)
    }
    if members != set(worker.execution_request.source_runtime_projection.ordered_source_relative_paths):
        _refuse("source_membership_changed", "freshness", "Source membership changed before claim completion.", "source_membership_changed")
    for item in source.ordered_verified_files:
        _final_live_gate(Path(item.source_path), item.final_path_file_facts, "source_file_mutated")


def _translate_live_refusal(exc: _LiveRefusal) -> None:
    category = exc.category
    if category not in _CATEGORY_SET:
        if category.startswith("startup_artifact_"):
            category = "startup_artifact_authority_mismatch"
        elif category.startswith("source_"):
            category = "source_runtime_identity_mismatch"
        else:
            category = "prelaunch_claim_internal_error"
    _refuse(category, "freshness", exc.message, exc.detail_code)


def claim_guided_npm_worker_for_prelaunch(
    worker_request_artifact_path: str,
    materialization_receipt: GuidedNpmWorkerRequestMaterializationReceipt,
    *,
    current_application_build_identity: ApplicationBuildIdentity,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedNpmWorkerPrelaunchClaimResult:
    """Verify the exact durable NPM worker authority without launching it."""
    try:
        _check_cancelled(cancellation_check)
        if type(materialization_receipt) is not GuidedNpmWorkerRequestMaterializationReceipt:
            _refuse("materialization_receipt_invalid", "receipt", "A valid B2-C6B1 receipt is required.", "receipt_type_invalid")
        try:
            verify_guided_npm_worker_request_materialization_receipt(materialization_receipt)
        except (TypeError, ValueError) as exc:
            _refuse("materialization_receipt_invalid", "receipt", "The materialization receipt is invalid.", str(exc) or type(exc).__name__)
        try:
            verify_application_build_identity(current_application_build_identity)
        except (TypeError, ValueError) as exc:
            _refuse("current_build_invalid", "build", "The current build identity is invalid.", str(exc) or type(exc).__name__)
        try:
            style, module = _path_model(worker_request_artifact_path)
        except ValueError as exc:
            _refuse("worker_artifact_path_invalid", "artifact", "The worker-artifact path is invalid.", str(exc))
        if not _stored_name_equal(module.basename(worker_request_artifact_path), GUIDED_NPM_WORKER_REQUEST_FILENAME, style):
            _refuse("worker_artifact_path_invalid", "artifact", "The worker-artifact filename is invalid.", "worker_artifact_filename_invalid")
        if not _same_stored_path(worker_request_artifact_path, materialization_receipt.worker_request_artifact_path, style):
            _refuse("worker_artifact_path_invalid", "artifact", "The worker-artifact path differs from the receipt.", "receipt_artifact_path_mismatch")
        _check_cancelled(cancellation_check)
        path = Path(worker_request_artifact_path)
        pre, opened, post, final, observed_digest, content = _stable_read_worker_artifact(path, cancellation_check)
        if len(content) != materialization_receipt.worker_request_artifact_size_bytes:
            _refuse("worker_artifact_size_mismatch", "artifact", "The worker-artifact size differs from the receipt.", "worker_artifact_size_mismatch")
        if observed_digest != materialization_receipt.worker_request_artifact_sha256:
            _refuse("worker_artifact_digest_mismatch", "artifact", "The worker-artifact digest differs from the receipt.", "worker_artifact_digest_mismatch")
        _check_cancelled(cancellation_check)
        _final_live_gate(path, final, "worker_artifact_mutated")
        try:
            worker = decode_canonical_guided_npm_worker_request_bytes(content)
            verify_guided_npm_worker_request(worker)
        except ValueError as exc:
            _refuse("worker_artifact_noncanonical", "artifact", "The worker artifact is not canonical authority.", str(exc))
        worker_style = worker.execution_request.output_runtime_projection.output_base_path_style
        if style != worker_style or not _same_stored_path(worker_request_artifact_path, worker.worker_request_artifact_path, worker_style):
            _refuse("worker_artifact_path_invalid", "artifact", "The artifact path differs from worker authority.", "worker_authorized_path_mismatch")
        expected_parent, _ = stored_parent_and_name(worker_request_artifact_path, worker_style)
        if not all(_same_stored_path(expected_parent, value, worker_style) for value in (
            materialization_receipt.run_directory_path, worker.run_directory_path,
            worker.execution_request.output_runtime_projection.run_directory_path,
        )):
            _refuse("worker_artifact_path_invalid", "artifact", "The artifact parent differs from run authority.", "worker_artifact_parent_mismatch")
        _check_cancelled(cancellation_check)
        try:
            verify_guided_npm_worker_request_materialization_binding(materialization_receipt, worker)
        except ValueError as exc:
            _refuse("materialization_binding_mismatch", "receipt", "The receipt does not bind to this worker.", str(exc))
        if not (
            current_application_build_identity == worker.application_build_identity
            == worker.execution_request.application_build_identity
            == materialization_receipt.application_build_identity
        ):
            _refuse("current_build_mismatch", "build", "The current build differs from worker authority.", "current_build_mismatch")
        _check_cancelled(cancellation_check)
        try:
            startup_facts = verify_guided_npm_startup_artifact_live(worker.execution_request, cancellation_check)
            _check_cancelled(cancellation_check)
            source_freshness = verify_guided_npm_source_freshness_live(worker.execution_request, cancellation_check)
        except _LiveCancelled:
            raise _Cancelled("prelaunch_claim_cancelled")
        except _LiveRefusal as exc:
            _translate_live_refusal(exc)
        _check_cancelled(cancellation_check)
        authorized_worker_artifact_path = worker.worker_request_artifact_path
        authorized_startup_artifact_path = worker.startup_artifact_path
        authorized_run_directory_path = worker.run_directory_path
        pre, opened, post, final = tuple(
            replace(item, canonical_path=authorized_worker_artifact_path)
            for item in (pre, opened, post, final)
        )
        evidence = GuidedNpmPrelaunchFreshnessEvidence(
            GUIDED_NPM_PRELAUNCH_FRESHNESS_SCHEMA_NAME,
            GUIDED_NPM_PRELAUNCH_FRESHNESS_SCHEMA_VERSION,
            GUIDED_NPM_PRELAUNCH_FRESHNESS_CONTRACT_VERSION,
            worker.canonical_worker_request_identity,
            worker.source_execution_request_identity,
            materialization_receipt.canonical_materialization_receipt_identity,
            authorized_worker_artifact_path, len(content), observed_digest,
            pre, opened, post, final,
            authorized_startup_artifact_path, worker.startup_artifact_size_bytes,
            worker.startup_artifact_sha256, startup_facts,
            source_freshness, current_application_build_identity,
            "verified_for_prelaunch", "0" * 64,
        )
        evidence = replace(
            evidence,
            canonical_prelaunch_freshness_evidence_identity=compute_guided_npm_prelaunch_freshness_evidence_identity(evidence),
        )
        verify_guided_npm_prelaunch_freshness_evidence(evidence, worker, materialization_receipt)
        _check_cancelled(cancellation_check)
        claim = GuidedNpmWorkerPrelaunchClaim(
            GUIDED_NPM_WORKER_PRELAUNCH_CLAIM_SCHEMA_NAME,
            GUIDED_NPM_WORKER_PRELAUNCH_CLAIM_SCHEMA_VERSION,
            GUIDED_NPM_WORKER_PRELAUNCH_CLAIM_CONTRACT_VERSION,
            worker.canonical_worker_request_identity, worker.source_execution_request_identity,
            materialization_receipt.canonical_materialization_receipt_identity,
            evidence.canonical_prelaunch_freshness_evidence_identity,
            authorized_worker_artifact_path, observed_digest, len(content),
            worker.application_build_identity, worker.guided_plan_identity,
            worker.validation_revision, worker.execution_request.execution_mode,
            authorized_run_directory_path, worker, materialization_receipt, evidence,
            "verified_for_prelaunch", "not_launched", "not_started", "not_available", False,
            "0" * 64,
        )
        claim = replace(claim, canonical_prelaunch_claim_identity=compute_guided_npm_worker_prelaunch_claim_identity(claim))
        verify_guided_npm_worker_prelaunch_claim(claim)
        _final_preclaim_gate(worker, evidence)
        _check_cancelled(cancellation_check)
        return claim
    except _Cancelled:
        issue = GuidedNpmWorkerPrelaunchClaimIssue(
            "prelaunch_claim_cancelled", "prelaunch", "NPM prelaunch claiming was cancelled.", "prelaunch_claim_cancelled"
        )
        return GuidedNpmWorkerPrelaunchClaimCancelled((issue,))
    except _Refusal as exc:
        issue = GuidedNpmWorkerPrelaunchClaimIssue(exc.category, exc.section, exc.message, exc.detail_code)
        return GuidedNpmWorkerPrelaunchClaimFailure((issue,))
    except Exception as exc:
        issue = GuidedNpmWorkerPrelaunchClaimIssue(
            "prelaunch_claim_internal_error", "prelaunch", "NPM prelaunch claiming failed internally.", type(exc).__name__
        )
        return GuidedNpmWorkerPrelaunchClaimFailure((issue,))
