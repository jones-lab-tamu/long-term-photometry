"""B2-C6B1 live verification and durable worker-request materialization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
import hashlib
import ntpath
import os
from pathlib import Path
import posixpath
import secrets
import stat
from typing import Any

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_production_execution_request import (
    GuidedNpmProductionExecutionRequest,
    compute_guided_npm_production_session_sequence_identity,
    compute_guided_npm_production_source_content_identity,
    compute_guided_npm_production_source_membership_identity,
    verify_guided_npm_production_execution_request,
)
from photometry_pipeline.guided_npm_startup_persistence import (
    GUIDED_NPM_STARTUP_ARTIFACT_FILENAME,
    verify_application_build_identity,
    verify_guided_npm_startup_artifact_path,
)
from photometry_pipeline.guided_production_mapping import ApplicationBuildIdentity
from photometry_pipeline.guided_npm_worker_request import (
    GUIDED_NPM_DISCOVERY_CONTRACT_VERSION,
    GUIDED_NPM_LIVE_FRESHNESS_CONTRACT_VERSION,
    GUIDED_NPM_LIVE_FRESHNESS_SCHEMA_NAME,
    GUIDED_NPM_LIVE_FRESHNESS_SCHEMA_VERSION,
    GUIDED_NPM_WORKER_REQUEST_FILENAME,
    GuidedNpmLiveDirectoryFacts,
    GuidedNpmLiveFileFacts,
    GuidedNpmLiveSourceFreshnessEvidence,
    GuidedNpmLiveVerifiedSourceFile,
    GuidedNpmWorkerRequest,
    build_guided_npm_worker_request,
    canonical_guided_npm_worker_request_bytes,
    compute_guided_npm_live_freshness_evidence_identity,
    compute_guided_npm_live_verified_file_sequence_identity,
    compute_guided_npm_live_verified_source_file_identity,
    decode_canonical_guided_npm_worker_request_bytes,
    verify_guided_npm_worker_request,
)


GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_SCHEMA_NAME = "guided_npm_worker_request_materialization_receipt"
GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_SCHEMA_VERSION = "v1"
GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_CONTRACT_VERSION = "guided_npm_worker_request_materialization_receipt.v1"
GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_IDENTITY_DOMAIN = "guided_npm_worker_request_materialization_receipt.v1"
_HASH_BLOCK_SIZE = 1024 * 1024

GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_REFUSAL_CATEGORIES = (
    "execution_request_missing_or_invalid", "execution_request_schema_unsupported",
    "execution_request_identity_mismatch", "execution_request_state_invalid",
    "execution_request_build_mismatch", "startup_artifact_missing",
    "startup_artifact_path_invalid", "startup_artifact_alias_invalid",
    "startup_artifact_mutated", "startup_artifact_size_mismatch",
    "startup_artifact_digest_mismatch", "startup_artifact_payload_identity_mismatch",
    "startup_artifact_build_mismatch", "startup_artifact_plan_mismatch",
    "startup_artifact_revision_mismatch", "source_root_missing",
    "source_root_not_directory", "source_root_alias_invalid", "source_root_replaced",
    "source_discovery_failed", "source_membership_missing", "source_membership_extra",
    "source_membership_changed", "source_path_mismatch", "source_file_missing",
    "source_file_alias_invalid", "source_file_not_regular", "source_file_size_mismatch",
    "source_file_digest_mismatch", "source_file_mutated", "source_runtime_identity_mismatch",
    "run_directory_missing", "run_directory_not_directory", "run_directory_alias_invalid",
    "run_directory_dirty", "worker_request_artifact_conflict", "worker_request_write_failed",
    "worker_request_flush_failed", "worker_request_publish_failed", "worker_request_readback_failed",
    "worker_request_size_mismatch", "worker_request_digest_mismatch", "worker_request_bytes_mismatch",
    "worker_request_noncanonical", "worker_request_identity_mismatch",
    "materialization_cancelled", "materialization_internal_error",
)
_CATEGORY_SET = frozenset(GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_REFUSAL_CATEGORIES)


@dataclass(frozen=True)
class GuidedNpmWorkerRequestMaterializationIssue:
    category: str
    section: str
    message: str
    detail_code: str


@dataclass(frozen=True)
class GuidedNpmWorkerRequestMaterializationFailure:
    blocking_issues: tuple[GuidedNpmWorkerRequestMaterializationIssue, ...]
    unverified_worker_request_artifact_path: str | None = None
    status: str = "refused"


@dataclass(frozen=True)
class GuidedNpmWorkerRequestMaterializationCancelled:
    blocking_issues: tuple[GuidedNpmWorkerRequestMaterializationIssue, ...]
    unverified_worker_request_artifact_path: str | None = None
    status: str = "cancelled"


@dataclass(frozen=True)
class GuidedNpmWorkerRequestMaterializationReceipt:
    receipt_schema_name: str
    receipt_schema_version: str
    receipt_contract_version: str
    source_execution_request_identity: str
    source_worker_request_identity: str
    source_live_freshness_evidence_identity: str
    run_directory_path: str
    worker_request_artifact_path: str
    worker_request_artifact_sha256: str
    worker_request_artifact_size_bytes: int
    readback_worker_request_identity: str
    application_build_identity: ApplicationBuildIdentity
    guided_plan_identity: str
    validation_revision: int
    materialization_status: str
    launch_status: str
    execution_status: str
    completion_status: str
    runnable: bool
    canonical_materialization_receipt_identity: str


GuidedNpmWorkerRequestMaterializationResult = (
    GuidedNpmWorkerRequestMaterializationReceipt
    | GuidedNpmWorkerRequestMaterializationFailure
    | GuidedNpmWorkerRequestMaterializationCancelled
)


class _Refusal(ValueError):
    def __init__(self, category, section, message, detail_code, artifact_path=None):
        self.category, self.section, self.message, self.detail_code = category, section, message, detail_code
        self.artifact_path = artifact_path
        super().__init__(message)


class _Cancelled(RuntimeError):
    def __init__(self, artifact_path=None):
        self.artifact_path = artifact_path
        super().__init__("materialization_cancelled")


def _refuse(category, section, message, detail_code, artifact_path=None):
    raise _Refusal(category if category in _CATEGORY_SET else "materialization_internal_error", section, message, detail_code, artifact_path)


def _check_cancelled(check, artifact_path=None):
    if check is not None and check():
        raise _Cancelled(artifact_path)


def _same_path(left: Path | str, right: Path | str) -> bool:
    return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(os.path.abspath(os.fspath(right)))


def _file_facts(path: Path, metadata: os.stat_result) -> GuidedNpmLiveFileFacts:
    return GuidedNpmLiveFileFacts(os.fspath(path), metadata.st_size, metadata.st_mtime_ns, "regular_file", metadata.st_dev, metadata.st_ino, metadata.st_mode)


def _directory_facts(path: Path, metadata: os.stat_result) -> GuidedNpmLiveDirectoryFacts:
    return GuidedNpmLiveDirectoryFacts(os.fspath(path), metadata.st_mtime_ns, "directory", metadata.st_dev, metadata.st_ino, metadata.st_mode)


def _stable_tuple(facts: GuidedNpmLiveFileFacts) -> tuple[int, ...]:
    return facts.size_bytes, facts.mtime_ns, facts.device, facts.inode, facts.mode


def _directory_identity(facts: GuidedNpmLiveDirectoryFacts) -> tuple[int, ...]:
    return facts.device, facts.inode, facts.mode


def _lstat_regular(path: Path, *, missing_category: str, alias_category: str, regular_category: str) -> GuidedNpmLiveFileFacts:
    if not os.path.lexists(path):
        _refuse(missing_category, "filesystem", "The required file is missing.", missing_category)
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        _refuse(missing_category, "filesystem", "The required file could not be inspected.", type(exc).__name__)
    if stat.S_ISLNK(metadata.st_mode):
        _refuse(alias_category, "filesystem", "A symbolic-link file is not accepted.", alias_category)
    if not stat.S_ISREG(metadata.st_mode):
        _refuse(regular_category, "filesystem", "The required path is not a regular file.", regular_category)
    return _file_facts(path, metadata)


def _stable_read(path: Path, check, *, startup: bool = False):
    prefix = "startup_artifact" if startup else "source_file"
    pre = _lstat_regular(path, missing_category=f"{prefix}_missing", alias_category=f"{prefix}_alias_invalid", regular_category=f"{prefix}_path_invalid" if startup else "source_file_not_regular")
    digest = hashlib.sha256()
    content = bytearray() if startup else None
    try:
        with path.open("rb") as handle:
            opened = _file_facts(path, os.fstat(handle.fileno()))
            if _stable_tuple(pre) != _stable_tuple(opened):
                _refuse(f"{prefix}_mutated", "filesystem", "The file changed while it was opened.", "pre_open_facts_changed")
            while True:
                _check_cancelled(check)
                block = handle.read(_HASH_BLOCK_SIZE)
                if not block:
                    break
                digest.update(block)
                if content is not None:
                    content.extend(block)
            post = _file_facts(path, os.fstat(handle.fileno()))
    except (_Refusal, _Cancelled):
        raise
    except OSError as exc:
        _refuse(f"{prefix}_mutated", "filesystem", "The file could not be read stably.", type(exc).__name__)
    final = _lstat_regular(path, missing_category=f"{prefix}_missing", alias_category=f"{prefix}_alias_invalid", regular_category=f"{prefix}_path_invalid" if startup else "source_file_not_regular")
    if not (_stable_tuple(pre) == _stable_tuple(opened) == _stable_tuple(post) == _stable_tuple(final)):
        _refuse(f"{prefix}_mutated", "filesystem", "The file changed during verification.", "stable_file_facts_changed")
    return pre, opened, post, final, digest.hexdigest(), bytes(content) if content is not None else None


def _verify_startup_artifact(execution, check):
    path = Path(execution.startup_artifact_path)
    _check_cancelled(check)
    if not path.is_absolute() or path.name != GUIDED_NPM_STARTUP_ARTIFACT_FILENAME:
        _refuse("startup_artifact_path_invalid", "startup", "The startup-artifact path is invalid.", "startup_artifact_path_invalid")
    pre, _, _, final, digest, content = _stable_read(path, check, startup=True)
    try:
        verified = verify_guided_npm_startup_artifact_path(os.fspath(path))
    except FileNotFoundError:
        _refuse("startup_artifact_missing", "startup", "The startup artifact is missing.", "startup_artifact_missing")
    except (OSError, ValueError) as exc:
        code = str(exc)
        category = code if code in _CATEGORY_SET else "startup_artifact_mutated"
        _refuse(category, "startup", "The startup artifact failed live verification.", code or type(exc).__name__)
    final_after_public = _lstat_regular(path, missing_category="startup_artifact_missing", alias_category="startup_artifact_alias_invalid", regular_category="startup_artifact_path_invalid")
    if _stable_tuple(pre) != _stable_tuple(final_after_public):
        _refuse("startup_artifact_mutated", "startup", "The startup artifact changed during canonical verification.", "public_verifier_race")
    if verified.startup_artifact_path != execution.startup_artifact_path:
        _refuse("startup_artifact_alias_invalid", "startup", "The startup artifact resolved to another path.", "startup_artifact_resolved_path_mismatch")
    if len(content) != execution.startup_artifact_size_bytes or verified.startup_artifact_size_bytes != execution.startup_artifact_size_bytes:
        _refuse("startup_artifact_size_mismatch", "startup", "The startup-artifact size changed.", "startup_artifact_size_mismatch")
    if digest != execution.startup_artifact_sha256 or verified.startup_artifact_sha256 != execution.startup_artifact_sha256:
        _refuse("startup_artifact_digest_mismatch", "startup", "The startup-artifact digest changed.", "startup_artifact_digest_mismatch")
    payload = verified.payload
    comparisons = (
        (payload.canonical_startup_payload_identity, execution.source_startup_payload_identity, "startup_artifact_payload_identity_mismatch"),
        (payload.application_build_identity, execution.application_build_identity, "startup_artifact_build_mismatch"),
        (payload.guided_plan_identity, execution.guided_plan_identity, "startup_artifact_plan_mismatch"),
        (payload.validation_revision, execution.validation_revision, "startup_artifact_revision_mismatch"),
        (payload.source_authorization_identity, execution.source_authorization_identity, "startup_artifact_payload_identity_mismatch"),
        (payload.source_authority_identity, execution.source_authority_identity, "startup_artifact_payload_identity_mismatch"),
        (payload.source_production_intent_identity, execution.source_production_intent_identity, "startup_artifact_payload_identity_mismatch"),
        (payload.source_request_identity, execution.source_request_identity, "startup_artifact_payload_identity_mismatch"),
    )
    for observed, expected, category in comparisons:
        if observed != expected:
            _refuse(category, "startup", "Startup-artifact provenance no longer matches the execution request.", category)
    return final


def _inspect_directory(path: Path, *, missing, not_directory, alias) -> GuidedNpmLiveDirectoryFacts:
    if not os.path.lexists(path):
        _refuse(missing, "filesystem", "The required directory is missing.", missing)
    try:
        metadata = path.stat(follow_symlinks=False)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        _refuse(alias, "filesystem", "The directory could not be resolved safely.", type(exc).__name__)
    if stat.S_ISLNK(metadata.st_mode) or not _same_path(path, resolved):
        _refuse(alias, "filesystem", "Directory aliases are not accepted.", alias)
    if not stat.S_ISDIR(metadata.st_mode):
        _refuse(not_directory, "filesystem", "The required path is not a directory.", not_directory)
    return _directory_facts(path, metadata)


def _discover_csv(root: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        _refuse("source_discovery_failed", "source", "The source root could not be discovered.", type(exc).__name__)
    for entry in entries:
        if not entry.name.lower().endswith(".csv"):
            continue
        path = root / entry.name
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            _refuse("source_discovery_failed", "source", "A source member could not be inspected.", type(exc).__name__)
        if stat.S_ISLNK(metadata.st_mode):
            _refuse("source_file_alias_invalid", "source", "Symlinked CSV files are not accepted.", "source_csv_symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            _refuse("source_file_not_regular", "source", "A CSV source member is not regular.", "source_csv_not_regular")
        canonical_name = os.path.normcase(entry.name) if os.name == "nt" else entry.name
        if canonical_name in discovered:
            _refuse("source_file_alias_invalid", "source", "Duplicate source aliases are not accepted.", "duplicate_source_alias")
        discovered[canonical_name] = path
    return discovered


def _compare_membership(discovered: dict[str, Path], expected: tuple[str, ...], *, changed=False):
    actual = set(discovered)
    wanted = set(expected)
    if actual == wanted:
        return
    if changed:
        _refuse("source_membership_changed", "source", "Source membership changed during verification.", "final_membership_changed")
    if wanted - actual:
        _refuse("source_membership_missing", "source", "An authorized CSV source is missing.", "source_membership_missing")
    _refuse("source_membership_extra", "source", "An unexpected CSV source is present.", "source_membership_extra")


def _verify_live_sources(execution, check):
    source = execution.source_runtime_projection
    expected_style = "windows_drive" if os.name == "nt" else "posix_absolute"
    if source.source_path_style != expected_style:
        _refuse("source_root_alias_invalid", "source", "The source path style does not match this host.", "source_path_style_mismatch")
    root = Path(source.source_root_canonical)
    if not root.is_absolute():
        _refuse("source_root_alias_invalid", "source", "The source root is not absolute.", "source_root_not_absolute")
    _check_cancelled(check)
    pre_root = _inspect_directory(root, missing="source_root_missing", not_directory="source_root_not_directory", alias="source_root_alias_invalid")
    _check_cancelled(check)
    before = _discover_csv(root)
    expected_relative = source.ordered_source_relative_paths
    _compare_membership(before, expected_relative)
    verified_files = []
    for session in source.ordered_sessions:
        _check_cancelled(check)
        expected_path = root / session.canonical_relative_path
        if session.canonical_relative_path not in before or not _same_path(expected_path, session.source_path) or not _same_path(before[session.canonical_relative_path], expected_path):
            _refuse("source_path_mismatch", "source", "A live source path differs from authority.", "source_path_mismatch")
        pre, opened, post, final, digest, _ = _stable_read(expected_path, check)
        if pre.size_bytes != session.source_size_bytes:
            _refuse("source_file_size_mismatch", "source", "A source file size changed.", "source_file_size_mismatch")
        if digest != session.source_sha256:
            _refuse("source_file_digest_mismatch", "source", "A source file digest changed.", "source_file_digest_mismatch")
        live = GuidedNpmLiveVerifiedSourceFile(
            session.chronological_position, session.source_path, session.canonical_relative_path,
            session.source_size_bytes, pre.size_bytes, session.source_sha256, digest,
            pre, opened, post, final, session.canonical_session_runtime_identity, "0" * 64,
        )
        live = replace(live, canonical_live_verified_source_file_identity=compute_guided_npm_live_verified_source_file_identity(live))
        verified_files.append(live)
        _check_cancelled(check)
    _check_cancelled(check)
    after = _discover_csv(root)
    _compare_membership(after, expected_relative, changed=True)
    post_root = _inspect_directory(root, missing="source_root_missing", not_directory="source_root_not_directory", alias="source_root_alias_invalid")
    if _directory_identity(pre_root) != _directory_identity(post_root):
        _refuse("source_root_replaced", "source", "The source root was replaced.", "source_root_identity_changed")
    sessions = source.ordered_sessions
    evidence = GuidedNpmLiveSourceFreshnessEvidence(
        GUIDED_NPM_LIVE_FRESHNESS_SCHEMA_NAME, GUIDED_NPM_LIVE_FRESHNESS_SCHEMA_VERSION,
        GUIDED_NPM_LIVE_FRESHNESS_CONTRACT_VERSION, source.source_root_canonical,
        source.source_path_style, GUIDED_NPM_DISCOVERY_CONTRACT_VERSION,
        source.canonical_source_runtime_projection_identity, source.runtime_source_membership_identity,
        source.runtime_source_content_identity, source.runtime_session_sequence_identity,
        tuple(verified_files), compute_guided_npm_production_source_membership_identity(sessions),
        compute_guided_npm_production_source_content_identity(sessions),
        compute_guided_npm_production_session_sequence_identity(sessions),
        compute_guided_npm_live_verified_file_sequence_identity(tuple(verified_files)),
        pre_root, post_root, "live_verified", "0" * 64,
    )
    return replace(evidence, canonical_live_freshness_evidence_identity=compute_guided_npm_live_freshness_evidence_identity(evidence))


def _verify_run_directory(execution):
    output = execution.output_runtime_projection
    run = Path(output.run_directory_path)
    startup = Path(execution.startup_artifact_path)
    base = Path(output.output_base_canonical)
    if not run.is_absolute() or startup.parent != run or run.parent != base:
        _refuse("run_directory_alias_invalid", "output", "The run-directory relationship is invalid.", "run_directory_relationship_invalid")
    facts = _inspect_directory(run, missing="run_directory_missing", not_directory="run_directory_not_directory", alias="run_directory_alias_invalid")
    try:
        names = {entry.name for entry in os.scandir(run)}
    except OSError as exc:
        _refuse("run_directory_dirty", "output", "The run directory could not be inspected.", type(exc).__name__)
    if GUIDED_NPM_WORKER_REQUEST_FILENAME in names:
        _refuse("worker_request_artifact_conflict", "output", "A worker-request artifact already exists.", "worker_request_artifact_conflict")
    if names != {GUIDED_NPM_STARTUP_ARTIFACT_FILENAME}:
        _refuse("run_directory_dirty", "output", "The run directory contains unexpected entries.", "run_directory_allowlist_mismatch")
    return run, facts


def _write_temp(path: Path, content: bytes):
    try:
        with path.open("xb") as handle:
            if handle.write(content) != len(content):
                _refuse("worker_request_write_failed", "publication", "The complete worker request was not written.", "partial_write")
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except OSError as exc:
                _refuse("worker_request_flush_failed", "publication", "The worker request could not be flushed.", type(exc).__name__)
    except _Refusal:
        raise
    except OSError as exc:
        _refuse("worker_request_write_failed", "publication", "The worker request could not be written.", type(exc).__name__)


def _publish_no_replace(temp: Path, final: Path):
    try:
        os.link(temp, final, follow_symlinks=False)
        temp.unlink()
    except FileExistsError:
        _refuse("worker_request_artifact_conflict", "publication", "A worker request raced publication.", "worker_request_artifact_conflict", os.fspath(final))
    except OSError as exc:
        _refuse("worker_request_publish_failed", "publication", "The worker request could not be published atomically.", type(exc).__name__)


def _fsync_directory(path: Path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def compute_guided_npm_worker_request_materialization_receipt_identity(receipt):
    """Compute the structural identity of one materialization receipt."""
    def canonical(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, tuple):
            return [canonical(item) for item in value]
        if hasattr(value, "__dataclass_fields__"):
            return {item.name: canonical(getattr(value, item.name)) for item in fields(value)}
        raise ValueError("materialization_receipt_value_invalid")

    payload = {
        item.name: canonical(getattr(receipt, item.name))
        for item in fields(receipt)
        if item.name != "canonical_materialization_receipt_identity"
    }
    return hashlib.sha256(GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_IDENTITY_DOMAIN.encode() + b"\x00" + encode_canonical_value(payload)).hexdigest()


def _stored_path_style(path: str) -> str:
    if ntpath.isabs(path) and ntpath.splitdrive(path)[0]:
        return "windows_drive"
    if posixpath.isabs(path):
        return "posix_absolute"
    raise ValueError("materialization_receipt_path_invalid")


def _stored_path_equal(left: str, right: str, style: str) -> bool:
    module = ntpath if style == "windows_drive" else posixpath
    normalize = lambda value: module.normcase(module.normpath(value)) if style == "windows_drive" else module.normpath(value)
    return normalize(left) == normalize(right)


def _stored_parent_and_name(path: str, style: str) -> tuple[str, str]:
    module = ntpath if style == "windows_drive" else posixpath
    return module.dirname(module.normpath(path)), module.basename(module.normpath(path))


def verify_guided_npm_worker_request_materialization_receipt(receipt):
    """Verify structural validity and internal consistency only.

    This pure verifier does not establish correspondence to a worker request
    and is therefore insufficient as future launch authority.
    """
    if type(receipt) is not GuidedNpmWorkerRequestMaterializationReceipt:
        raise ValueError("materialization_receipt_type_invalid")
    verify_application_build_identity(receipt.application_build_identity)
    identity_fields = (
        receipt.source_execution_request_identity,
        receipt.source_worker_request_identity,
        receipt.source_live_freshness_evidence_identity,
        receipt.worker_request_artifact_sha256,
        receipt.readback_worker_request_identity,
        receipt.guided_plan_identity,
        receipt.canonical_materialization_receipt_identity,
    )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in identity_fields
    ):
        raise ValueError("materialization_receipt_identity_field_invalid")
    if (
        isinstance(receipt.worker_request_artifact_size_bytes, bool)
        or not isinstance(receipt.worker_request_artifact_size_bytes, int)
        or receipt.worker_request_artifact_size_bytes <= 0
    ):
        raise ValueError("materialization_receipt_artifact_size_invalid")
    if (
        isinstance(receipt.validation_revision, bool)
        or not isinstance(receipt.validation_revision, int)
        or receipt.validation_revision < 0
    ):
        raise ValueError("materialization_receipt_revision_invalid")
    if (
        not isinstance(receipt.run_directory_path, str)
        or not isinstance(receipt.worker_request_artifact_path, str)
    ):
        raise ValueError("materialization_receipt_path_invalid")
    run_style = _stored_path_style(receipt.run_directory_path)
    artifact_style = _stored_path_style(receipt.worker_request_artifact_path)
    artifact_parent, artifact_name = _stored_parent_and_name(
        receipt.worker_request_artifact_path, artifact_style
    )
    if (
        run_style != artifact_style
        or artifact_name != GUIDED_NPM_WORKER_REQUEST_FILENAME
        or not _stored_path_equal(artifact_parent, receipt.run_directory_path, run_style)
    ):
        raise ValueError("materialization_receipt_path_invalid")
    if (receipt.receipt_schema_name, receipt.receipt_schema_version, receipt.receipt_contract_version) != (
        GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_SCHEMA_NAME,
        GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
        GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_CONTRACT_VERSION,
    ) or (receipt.materialization_status, receipt.launch_status, receipt.execution_status, receipt.completion_status, receipt.runnable) != (
        "persisted_and_verified", "not_launched", "not_started", "not_available", False,
    ) or receipt.source_worker_request_identity != receipt.readback_worker_request_identity or compute_guided_npm_worker_request_materialization_receipt_identity(receipt) != receipt.canonical_materialization_receipt_identity:
        raise ValueError("materialization_receipt_invalid")


def verify_guided_npm_worker_request_materialization_binding(
    receipt: GuidedNpmWorkerRequestMaterializationReceipt,
    worker_request: GuidedNpmWorkerRequest,
) -> None:
    """Purely bind a structurally valid receipt to one exact worker authority.

    A structurally valid receipt is insufficient for future B2-C6B2 launch.
    Launch must additionally re-read the exact artifact, verify its observed
    digest and size, decode the worker request, and pass this semantic binding.
    """
    verify_guided_npm_worker_request_materialization_receipt(receipt)
    verify_guided_npm_worker_request(worker_request)
    worker_identity = worker_request.canonical_worker_request_identity
    if (
        receipt.source_worker_request_identity != worker_identity
        or receipt.readback_worker_request_identity != worker_identity
    ):
        raise ValueError("materialization_binding_worker_identity_mismatch")
    execution_identity = worker_request.execution_request.canonical_execution_request_identity
    if (
        receipt.source_execution_request_identity
        != worker_request.source_execution_request_identity
        or receipt.source_execution_request_identity != execution_identity
    ):
        raise ValueError("materialization_binding_execution_identity_mismatch")
    if (
        receipt.source_live_freshness_evidence_identity
        != worker_request.live_freshness_evidence.canonical_live_freshness_evidence_identity
    ):
        raise ValueError("materialization_binding_freshness_identity_mismatch")
    style = worker_request.execution_request.output_runtime_projection.output_base_path_style
    if not _stored_path_equal(receipt.run_directory_path, worker_request.run_directory_path, style):
        raise ValueError("materialization_binding_run_directory_mismatch")
    if not _stored_path_equal(
        receipt.worker_request_artifact_path,
        worker_request.worker_request_artifact_path,
        style,
    ):
        raise ValueError("materialization_binding_artifact_path_mismatch")
    artifact_parent, artifact_name = _stored_parent_and_name(
        worker_request.worker_request_artifact_path, style
    )
    if (
        artifact_name != GUIDED_NPM_WORKER_REQUEST_FILENAME
        or not _stored_path_equal(artifact_parent, worker_request.run_directory_path, style)
    ):
        raise ValueError("materialization_binding_artifact_path_mismatch")
    if (
        receipt.application_build_identity != worker_request.application_build_identity
        or receipt.application_build_identity
        != worker_request.execution_request.application_build_identity
    ):
        raise ValueError("materialization_binding_build_mismatch")
    if (
        receipt.guided_plan_identity != worker_request.guided_plan_identity
        or receipt.guided_plan_identity != worker_request.execution_request.guided_plan_identity
    ):
        raise ValueError("materialization_binding_plan_mismatch")
    if (
        receipt.validation_revision != worker_request.validation_revision
        or receipt.validation_revision != worker_request.execution_request.validation_revision
    ):
        raise ValueError("materialization_binding_revision_mismatch")
    if (
        (receipt.launch_status, receipt.execution_status, receipt.completion_status, receipt.runnable)
        != (
            worker_request.launch_status,
            worker_request.execution_status,
            worker_request.completion_status,
            worker_request.runnable,
        )
        or worker_request.request_status != "constructed_for_worker"
    ):
        raise ValueError("materialization_binding_state_mismatch")


def materialize_guided_npm_worker_request(
    execution_request: GuidedNpmProductionExecutionRequest,
    *,
    current_application_build_identity: ApplicationBuildIdentity,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedNpmWorkerRequestMaterializationResult:
    temp = final = None
    final_owned = False
    try:
        _check_cancelled(cancellation_check)
        if type(execution_request) is not GuidedNpmProductionExecutionRequest:
            _refuse("execution_request_missing_or_invalid", "request", "A valid B2-C6A request is required.", "execution_request_type_invalid")
        try:
            verify_guided_npm_production_execution_request(execution_request)
        except (TypeError, ValueError) as exc:
            code = str(exc)
            category = "execution_request_identity_mismatch" if "identity" in code else "execution_request_state_invalid" if "state" in code else "execution_request_missing_or_invalid"
            _refuse(category, "request", "The B2-C6A request failed standalone verification.", code or type(exc).__name__)
        try:
            verify_application_build_identity(current_application_build_identity)
        except (TypeError, ValueError) as exc:
            _refuse("execution_request_build_mismatch", "request", "The current build identity is invalid.", str(exc) or type(exc).__name__)
        if current_application_build_identity != execution_request.application_build_identity:
            _refuse("execution_request_build_mismatch", "request", "The current build differs from the request build.", "current_build_identity_mismatch")
        _verify_startup_artifact(execution_request, cancellation_check)
        _check_cancelled(cancellation_check)
        evidence = _verify_live_sources(execution_request, cancellation_check)
        _check_cancelled(cancellation_check)
        run, _ = _verify_run_directory(execution_request)
        worker = build_guided_npm_worker_request(execution_request, evidence)
        content = canonical_guided_npm_worker_request_bytes(worker)
        final = Path(worker.worker_request_artifact_path)
        _check_cancelled(cancellation_check)
        # Final freshness gate closes the discovery/hash-to-publication window.
        final_discovery = _discover_csv(Path(evidence.source_root_canonical))
        _compare_membership(final_discovery, execution_request.source_runtime_projection.ordered_source_relative_paths, changed=True)
        for live in evidence.ordered_verified_files:
            current = _lstat_regular(Path(live.source_path), missing_category="source_file_missing", alias_category="source_file_alias_invalid", regular_category="source_file_not_regular")
            if _stable_tuple(current) != _stable_tuple(live.final_path_file_facts):
                _refuse("source_file_mutated", "source", "A source file changed before publication.", "final_publication_gate_changed")
        _check_cancelled(cancellation_check)
        temp = run / f".{GUIDED_NPM_WORKER_REQUEST_FILENAME}.{secrets.token_hex(16)}.tmp"
        _write_temp(temp, content)
        _check_cancelled(cancellation_check)
        _publish_no_replace(temp, final)
        temp = None
        final_owned = True
        try:
            _fsync_directory(run)
        except OSError as exc:
            _refuse(
                "worker_request_flush_failed",
                "publication",
                "The worker-request directory entry could not be durably flushed.",
                type(exc).__name__,
                os.fspath(final),
            )
        _check_cancelled(cancellation_check, os.fspath(final))
        try:
            readback = final.read_bytes()
        except OSError as exc:
            _refuse("worker_request_readback_failed", "readback", "The worker request could not be read back.", type(exc).__name__, os.fspath(final))
        if len(readback) != len(content):
            _refuse("worker_request_size_mismatch", "readback", "Worker-request readback size differs.", "worker_request_size_mismatch", os.fspath(final))
        if hashlib.sha256(readback).hexdigest() != hashlib.sha256(content).hexdigest():
            _refuse("worker_request_digest_mismatch", "readback", "Worker-request readback digest differs.", "worker_request_digest_mismatch", os.fspath(final))
        if readback != content:
            _refuse("worker_request_bytes_mismatch", "readback", "Worker-request readback bytes differ.", "worker_request_bytes_mismatch", os.fspath(final))
        try:
            restored = decode_canonical_guided_npm_worker_request_bytes(readback)
        except ValueError as exc:
            _refuse("worker_request_noncanonical", "readback", "Worker-request readback is not canonical.", str(exc), os.fspath(final))
        if restored != worker or restored.canonical_worker_request_identity != worker.canonical_worker_request_identity:
            _refuse("worker_request_identity_mismatch", "readback", "Worker-request readback identity differs.", "worker_request_identity_mismatch", os.fspath(final))
        _check_cancelled(cancellation_check, os.fspath(final))
        receipt = GuidedNpmWorkerRequestMaterializationReceipt(
            GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_SCHEMA_NAME,
            GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
            GUIDED_NPM_WORKER_REQUEST_MATERIALIZATION_RECEIPT_CONTRACT_VERSION,
            restored.source_execution_request_identity,
            restored.canonical_worker_request_identity,
            restored.live_freshness_evidence.canonical_live_freshness_evidence_identity,
            restored.run_directory_path, restored.worker_request_artifact_path,
            hashlib.sha256(readback).hexdigest(), len(readback),
            restored.canonical_worker_request_identity, restored.application_build_identity,
            restored.guided_plan_identity, restored.validation_revision,
            "persisted_and_verified", "not_launched", "not_started", "not_available", False, "0" * 64,
        )
        receipt = replace(
            receipt,
            canonical_materialization_receipt_identity=compute_guided_npm_worker_request_materialization_receipt_identity(receipt),
        )
        verify_guided_npm_worker_request_materialization_receipt(receipt)
        try:
            verify_guided_npm_worker_request_materialization_binding(receipt, restored)
        except ValueError as exc:
            _refuse(
                "worker_request_identity_mismatch",
                "readback",
                "The materialization receipt does not bind to the readback worker request.",
                str(exc) or "materialization_binding_failed",
                os.fspath(final),
            )
        final_owned = False
        return receipt
    except _Cancelled as exc:
        if temp is not None:
            try: temp.unlink()
            except OSError: pass
        if final_owned and final is not None:
            try: final.unlink()
            except OSError: pass
        issue = GuidedNpmWorkerRequestMaterializationIssue("materialization_cancelled", "materialization", "NPM worker-request materialization was cancelled.", "materialization_cancelled")
        remaining = os.fspath(final) if final is not None and os.path.lexists(final) else None
        return GuidedNpmWorkerRequestMaterializationCancelled((issue,), remaining)
    except _Refusal as exc:
        if temp is not None:
            try: temp.unlink()
            except OSError: pass
        if final_owned and final is not None:
            try: final.unlink()
            except OSError: pass
        issue = GuidedNpmWorkerRequestMaterializationIssue(exc.category, exc.section, exc.message, exc.detail_code)
        remaining = os.fspath(final) if final is not None and os.path.lexists(final) else exc.artifact_path
        return GuidedNpmWorkerRequestMaterializationFailure((issue,), remaining)
    except Exception as exc:
        if temp is not None:
            try: temp.unlink()
            except OSError: pass
        if final_owned and final is not None:
            try: final.unlink()
            except OSError: pass
        issue = GuidedNpmWorkerRequestMaterializationIssue("materialization_internal_error", "materialization", "NPM worker-request materialization failed internally.", type(exc).__name__)
        remaining = os.fspath(final) if final is not None and os.path.lexists(final) else None
        return GuidedNpmWorkerRequestMaterializationFailure((issue,), remaining)
