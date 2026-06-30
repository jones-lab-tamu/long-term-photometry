"""Read-only RWD source candidate snapshots and no-exclusion classification.

These models are non-authorizing source facts. This module does not inspect
headers, scan timestamps, infer completeness, write files, or invoke execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Callable, Mapping

from photometry_pipeline.guided_identity import (
    CANONICALIZATION_ALGORITHM_VERSION,
    GuidedIdentityError,
    canonicalize_absolute_path,
    encode_canonical_value,
)


RWD_SOURCE_SNAPSHOT_SCHEMA_NAME = "guided_rwd_source_candidate_snapshot"
RWD_SOURCE_SNAPSHOT_SCHEMA_VERSION = "v1"
RWD_SOURCE_DISCOVERY_RULE_VERSION = "immediate_child_exact_fluorescence_csv.v1"
RWD_RELATIVE_PATH_RULE_VERSION = "canonical_forward_slash_relative_path.v1"
RWD_IGNORED_FILES_POLICY = (
    "ignore_non_target_entries_bounded_nested_root_check.v1"
)
SOURCE_SNAPSHOT_BUILD_MODE = "read_only"
DIGEST_ALGORITHM = "sha256"
SOURCE_FORMAT = "rwd"
ACQUISITION_MODE = "intermittent"

INCOMPLETE_FINAL_CLASSIFICATION_SCHEMA_NAME = (
    "guided_rwd_incomplete_final_chunk_classification"
)
INCOMPLETE_FINAL_CLASSIFICATION_SCHEMA_VERSION = "v1"
INCOMPLETE_FINAL_CLASSIFIER_VERSION = "not_requested_only.v1"
NOT_REQUESTED_STATUS = "not_requested"

_SET_DIGEST_DOMAIN = "guided-rwd-source-candidate-set:v1"
_CONTENT_DIGEST_DOMAIN = "guided-rwd-source-candidate-content:v1"
_CLASSIFICATION_DIGEST_DOMAIN = "guided-rwd-incomplete-final-classification:v1"
_READ_CHUNK_SIZE = 4 * 1024 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")


def _canonical_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, (tuple, list)):
        return all(_canonical_safe(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _canonical_safe(item)
            for key, item in value.items()
        )
    return False


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


class RwdSourceSnapshotError(ValueError):
    """Categorized failure to build or consume an RWD source snapshot."""

    def __init__(
        self,
        category: str,
        message: str,
        context: Mapping[str, Any] | None = None,
        *,
        retryable: bool = False,
    ) -> None:
        copied = dict(context or {})
        if not _canonical_safe(copied):
            raise TypeError("RWD source snapshot context must be canonical-safe.")
        self.category = str(category)
        self.message = str(message)
        self.context = _freeze(copied)
        self.retryable = bool(retryable)
        super().__init__(self.message)


@dataclass(frozen=True)
class GuidedRwdSourceCandidateFile:
    canonical_relative_path: str
    size_bytes: int
    sha256_content_digest: str


@dataclass(frozen=True)
class GuidedRwdIgnoredFilesSummary:
    root_non_target_entry_count: int = 0
    session_non_target_entry_count: int = 0
    nested_entry_count_not_scanned: int = 0


@dataclass(frozen=True)
class GuidedRwdSourceCandidateSnapshot:
    snapshot_schema_name: str
    snapshot_schema_version: str
    discovery_rule_version: str
    path_canonicalization_version: str
    relative_path_rule_version: str
    digest_algorithm: str
    source_root_canonical: str
    source_root_path_style: str
    source_format: str
    acquisition_mode: str
    candidates: tuple[GuidedRwdSourceCandidateFile, ...]
    source_candidate_set_digest: str
    source_candidate_content_digest: str
    ignored_files_policy: str
    ignored_summary: GuidedRwdIgnoredFilesSummary
    build_mode: str
    unresolved_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuidedIncompleteFinalChunkPolicy:
    exclude_incomplete_final_rwd_chunk: bool
    non_final_short_chunks_block: bool = True
    malformed_final_chunk_blocks: bool = True
    exactly_one_final_chunk_may_be_excluded: bool = True


@dataclass(frozen=True)
class GuidedIncompleteFinalChunkClassification:
    schema_name: str
    schema_version: str
    classifier_version: str
    classification_status: str
    source_candidate_set_digest: str
    source_candidate_content_digest: str
    excluded_canonical_relative_path: str | None
    reason: str | None
    evidence: None
    policy: GuidedIncompleteFinalChunkPolicy
    parsing_contract_digest: str | None
    timing_contract_digest: str | None
    unresolved_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CandidatePath:
    runtime_path: str
    canonical_relative_path: str


@dataclass(frozen=True)
class _Discovery:
    candidates: tuple[_CandidatePath, ...]
    ignored_summary: GuidedRwdIgnoredFilesSummary


@dataclass(frozen=True)
class _StableFacts:
    size: int
    mtime_ns: int
    ctime_ns: int | None
    device: int | None
    inode: int | None


@dataclass(frozen=True)
class _HashedCandidate:
    candidate: GuidedRwdSourceCandidateFile
    stable_facts: _StableFacts


def _error(
    category: str,
    message: str,
    *,
    retryable: bool = False,
    **context: Any,
) -> RwdSourceSnapshotError:
    return RwdSourceSnapshotError(
        category,
        message,
        context,
        retryable=retryable,
    )


def _check_cancelled(
    cancellation_check: Callable[[], bool] | None,
) -> None:
    if cancellation_check is not None and cancellation_check():
        raise _error(
            "source_candidate_snapshot_cancelled",
            "RWD source candidate snapshot was cancelled.",
        )


def _scandir_entries(path: str) -> list[os.DirEntry[str]]:
    with os.scandir(path) as entries:
        return list(entries)


def _canonical_relative_path(session_name: str, path_style: str) -> str:
    if (
        not session_name
        or session_name in {".", ".."}
        or "/" in session_name
        or "\\" in session_name
    ):
        raise _error(
            "candidate_path_collision",
            "RWD session directory name cannot form a canonical relative path.",
            session_name=session_name,
        )
    normalized_name = (
        session_name.casefold()
        if path_style in {"windows_drive", "windows_unc"}
        else session_name
    )
    return f"{normalized_name}/fluorescence.csv"


def _entry_target_path(entry: os.DirEntry[str]) -> str:
    return os.path.join(entry.path, "fluorescence.csv")


def _has_bounded_nested_root(entry: os.DirEntry[str]) -> bool:
    try:
        children = _scandir_entries(entry.path)
    except OSError as exc:
        raise _error(
            "source_root_unreadable",
            "Unable to inspect an RWD source child directory.",
            path=entry.path,
            reason=str(exc),
        ) from exc
    for child in children:
        try:
            if not child.is_dir(follow_symlinks=False):
                continue
        except OSError as exc:
            raise _error(
                "source_root_unreadable",
                "Unable to inspect an RWD nested source entry.",
                path=child.path,
                reason=str(exc),
            ) from exc
        if os.path.lexists(os.path.join(child.path, "fluorescence.csv")):
            return True
    return False


def _scan_candidates(source_root: str, path_style: str) -> _Discovery:
    root_target = os.path.join(source_root, "fluorescence.csv")
    if os.path.lexists(root_target):
        raise _error(
            "unsupported_root_level_rwd_layout",
            "Root-level fluorescence.csv is not a supported forced-RWD layout.",
            path=root_target,
        )

    try:
        root_entries = _scandir_entries(source_root)
    except OSError as exc:
        raise _error(
            "source_root_unreadable",
            "Unable to inspect the RWD source root.",
            path=source_root,
            reason=str(exc),
        ) from exc

    candidates: list[_CandidatePath] = []
    root_ignored = 0
    session_ignored = 0
    nested_not_scanned = 0
    seen_relative_paths: set[str] = set()

    for entry in root_entries:
        try:
            is_directory = entry.is_dir(follow_symlinks=False)
        except OSError as exc:
            raise _error(
                "source_root_unreadable",
                "Unable to inspect an RWD source entry.",
                path=entry.path,
                reason=str(exc),
            ) from exc
        if not is_directory:
            root_ignored += 1
            continue

        target_path = _entry_target_path(entry)
        if os.path.lexists(target_path):
            try:
                target_stat = os.lstat(target_path)
            except OSError as exc:
                raise _error(
                    "candidate_unreadable",
                    "Unable to inspect an RWD fluorescence candidate.",
                    path=target_path,
                    reason=str(exc),
                ) from exc
            if not stat.S_ISREG(target_stat.st_mode):
                raise _error(
                    "candidate_non_regular",
                    "RWD fluorescence candidate must be a regular file.",
                    path=target_path,
                )
            relative_path = _canonical_relative_path(entry.name, path_style)
            if relative_path in seen_relative_paths:
                raise _error(
                    "candidate_path_collision",
                    "RWD candidates collide after path canonicalization.",
                    canonical_relative_path=relative_path,
                )
            seen_relative_paths.add(relative_path)
            candidates.append(_CandidatePath(target_path, relative_path))
            try:
                session_entries = _scandir_entries(entry.path)
            except OSError as exc:
                raise _error(
                    "source_root_unreadable",
                    "Unable to inspect an RWD session directory.",
                    path=entry.path,
                    reason=str(exc),
                ) from exc
            session_ignored += sum(
                1 for item in session_entries if item.name != "fluorescence.csv"
            )
            nested_not_scanned += sum(
                1
                for item in session_entries
                if item.name != "fluorescence.csv"
                and item.is_dir(follow_symlinks=False)
            )
            continue

        if _has_bounded_nested_root(entry):
            raise _error(
                "ambiguous_nested_rwd_root",
                "Nested RWD acquisition root is ambiguous.",
                path=entry.path,
            )
        root_ignored += 1

    if not candidates:
        raise _error(
            "no_rwd_fluorescence_files",
            "No immediate-child fluorescence.csv files were found.",
            source_root=source_root,
        )

    candidates.sort(key=lambda item: item.canonical_relative_path)
    return _Discovery(
        candidates=tuple(candidates),
        ignored_summary=GuidedRwdIgnoredFilesSummary(
            root_non_target_entry_count=root_ignored,
            session_non_target_entry_count=session_ignored,
            nested_entry_count_not_scanned=nested_not_scanned,
        ),
    )


def _stable_facts(result: os.stat_result, path: str) -> _StableFacts:
    mtime_ns = getattr(result, "st_mtime_ns", None)
    if not isinstance(mtime_ns, int) or mtime_ns < 0:
        raise _error(
            "unstable_filesystem_facts",
            "Usable nanosecond mtime is required for snapshot stability.",
            path=path,
        )
    ctime_ns = getattr(result, "st_ctime_ns", None)
    device = int(result.st_dev) if int(result.st_dev) != 0 else None
    inode = int(result.st_ino) if int(result.st_ino) != 0 else None
    return _StableFacts(
        size=int(result.st_size),
        mtime_ns=mtime_ns,
        ctime_ns=int(ctime_ns) if isinstance(ctime_ns, int) else None,
        device=device,
        inode=inode,
    )


def _facts_match(before: _StableFacts, after: _StableFacts) -> bool:
    if before.size != after.size or before.mtime_ns != after.mtime_ns:
        return False
    if (
        before.device is not None
        and before.inode is not None
        and after.device is not None
        and after.inode is not None
        and (before.device, before.inode) != (after.device, after.inode)
    ):
        return False
    return True


def _open_candidate(path: str):
    return open(path, "rb")


def _read_candidate_digest(
    candidate: _CandidatePath,
    cancellation_check: Callable[[], bool] | None,
) -> _HashedCandidate:
    try:
        handle = _open_candidate(candidate.runtime_path)
    except OSError as exc:
        raise _error(
            "candidate_unreadable",
            "Unable to open an RWD fluorescence candidate.",
            path=candidate.runtime_path,
            reason=str(exc),
        ) from exc

    try:
        with handle:
            try:
                before_stat = os.fstat(handle.fileno())
            except OSError as exc:
                raise _error(
                    "unstable_filesystem_facts",
                    "Unable to inspect an open RWD candidate.",
                    path=candidate.runtime_path,
                    reason=str(exc),
                ) from exc
            if not stat.S_ISREG(before_stat.st_mode):
                raise _error(
                    "candidate_non_regular",
                    "RWD fluorescence candidate must be a regular file.",
                    path=candidate.runtime_path,
                )
            before = _stable_facts(before_stat, candidate.runtime_path)
            digest = hashlib.sha256()
            byte_count = 0
            while True:
                _check_cancelled(cancellation_check)
                try:
                    block = handle.read(_READ_CHUNK_SIZE)
                except OSError as exc:
                    raise _error(
                        "candidate_unreadable",
                        "Unable to read an RWD fluorescence candidate.",
                        path=candidate.runtime_path,
                        reason=str(exc),
                    ) from exc
                if not block:
                    break
                digest.update(block)
                byte_count += len(block)
            try:
                after_stat = os.fstat(handle.fileno())
            except OSError as exc:
                raise _error(
                    "unstable_filesystem_facts",
                    "Unable to re-inspect an open RWD candidate.",
                    path=candidate.runtime_path,
                    reason=str(exc),
                ) from exc
            after = _stable_facts(after_stat, candidate.runtime_path)
    finally:
        if not handle.closed:
            handle.close()

    if byte_count != before.size or byte_count != after.size:
        raise _error(
            "source_changed_during_snapshot",
            "RWD candidate size changed while it was read.",
            retryable=True,
            path=candidate.runtime_path,
        )
    if not _facts_match(before, after):
        raise _error(
            "source_changed_during_snapshot",
            "RWD candidate facts changed while it was read.",
            retryable=True,
            path=candidate.runtime_path,
        )
    return _HashedCandidate(
        candidate=GuidedRwdSourceCandidateFile(
            canonical_relative_path=candidate.canonical_relative_path,
            size_bytes=byte_count,
            sha256_content_digest=digest.hexdigest(),
        ),
        stable_facts=after,
    )


def _candidate_keys(discovery: _Discovery) -> tuple[str, ...]:
    return tuple(item.canonical_relative_path for item in discovery.candidates)


def _validate_path_after_read(
    candidate_path: _CandidatePath,
    hashed: _HashedCandidate,
) -> None:
    try:
        current_stat = os.stat(candidate_path.runtime_path, follow_symlinks=False)
    except OSError as exc:
        raise _error(
            "source_changed_during_snapshot",
            "RWD candidate disappeared after it was read.",
            retryable=True,
            path=candidate_path.runtime_path,
            reason=str(exc),
        ) from exc
    if not stat.S_ISREG(current_stat.st_mode):
        raise _error(
            "source_changed_during_snapshot",
            "RWD candidate was replaced after it was read.",
            retryable=True,
            path=candidate_path.runtime_path,
        )
    current = _stable_facts(current_stat, candidate_path.runtime_path)
    if not _facts_match(hashed.stable_facts, current):
        raise _error(
            "source_changed_during_snapshot",
            "RWD candidate changed after it was read.",
            retryable=True,
            path=candidate_path.runtime_path,
        )


def _digest(domain: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + encode_canonical_value(dict(payload))
    ).hexdigest()


def _candidate_from_value(value: Any) -> GuidedRwdSourceCandidateFile:
    if isinstance(value, GuidedRwdSourceCandidateFile):
        return value
    if isinstance(value, Mapping):
        try:
            return GuidedRwdSourceCandidateFile(
                canonical_relative_path=value["canonical_relative_path"],
                size_bytes=value["size_bytes"],
                sha256_content_digest=value["sha256_content_digest"],
            )
        except (KeyError, TypeError) as exc:
            raise _error(
                "invalid_rwd_source_snapshot",
                "Malformed RWD source candidate entry.",
            ) from exc
    raise _error(
        "invalid_rwd_source_snapshot",
        "RWD source candidate entry has an unsupported type.",
        received_type=type(value).__name__,
    )


def _snapshot_values(value: Any) -> tuple[dict[str, Any], bool]:
    is_snapshot = isinstance(value, GuidedRwdSourceCandidateSnapshot)
    if is_snapshot:
        source: Mapping[str, Any] = {
            "snapshot_schema_name": value.snapshot_schema_name,
            "snapshot_schema_version": value.snapshot_schema_version,
            "discovery_rule_version": value.discovery_rule_version,
            "path_canonicalization_version": value.path_canonicalization_version,
            "relative_path_rule_version": value.relative_path_rule_version,
            "digest_algorithm": value.digest_algorithm,
            "source_root_canonical": value.source_root_canonical,
            "source_root_path_style": value.source_root_path_style,
            "source_format": value.source_format,
            "acquisition_mode": value.acquisition_mode,
            "candidates": value.candidates,
            "ignored_files_policy": value.ignored_files_policy,
            "build_mode": value.build_mode,
            "unresolved_inputs": value.unresolved_inputs,
        }
    elif isinstance(value, Mapping):
        source = value
    else:
        raise _error(
            "invalid_rwd_source_snapshot",
            "Expected an RWD source snapshot or semantic payload mapping.",
            received_type=type(value).__name__,
        )

    required = (
        "snapshot_schema_name",
        "snapshot_schema_version",
        "discovery_rule_version",
        "path_canonicalization_version",
        "relative_path_rule_version",
        "digest_algorithm",
        "source_root_canonical",
        "source_root_path_style",
        "source_format",
        "acquisition_mode",
        "candidates",
        "ignored_files_policy",
        "build_mode",
        "unresolved_inputs",
    )
    missing = tuple(name for name in required if name not in source)
    if missing:
        raise _error(
            "invalid_rwd_source_snapshot",
            "RWD source snapshot is missing required fields.",
            missing_fields=missing,
        )
    try:
        raw_candidates = source["candidates"]
        if isinstance(raw_candidates, (str, bytes, Mapping)):
            raise TypeError
        candidates = tuple(_candidate_from_value(item) for item in raw_candidates)
        unresolved = tuple(source["unresolved_inputs"])
    except TypeError as exc:
        raise _error(
            "invalid_rwd_source_snapshot",
            "RWD source snapshot candidates or unresolved inputs are malformed.",
        ) from exc

    values = {name: source[name] for name in required if name not in {"candidates", "unresolved_inputs"}}
    values["candidates"] = candidates
    values["unresolved_inputs"] = unresolved
    _validate_snapshot_semantics(values)
    return values, is_snapshot


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _HEX_DIGITS
    )


def _validate_snapshot_semantics(values: Mapping[str, Any]) -> None:
    expected = {
        "snapshot_schema_name": RWD_SOURCE_SNAPSHOT_SCHEMA_NAME,
        "snapshot_schema_version": RWD_SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "discovery_rule_version": RWD_SOURCE_DISCOVERY_RULE_VERSION,
        "path_canonicalization_version": CANONICALIZATION_ALGORITHM_VERSION,
        "relative_path_rule_version": RWD_RELATIVE_PATH_RULE_VERSION,
        "digest_algorithm": DIGEST_ALGORITHM,
        "source_format": SOURCE_FORMAT,
        "acquisition_mode": ACQUISITION_MODE,
        "ignored_files_policy": RWD_IGNORED_FILES_POLICY,
        "build_mode": SOURCE_SNAPSHOT_BUILD_MODE,
    }
    for name, expected_value in expected.items():
        if values[name] != expected_value:
            raise _error(
                "invalid_rwd_source_snapshot",
                f"Unsupported RWD source snapshot field: {name}.",
                field=name,
            )
    if values["unresolved_inputs"]:
        raise _error(
            "invalid_rwd_source_snapshot",
            "RWD source snapshot has unresolved inputs.",
            unresolved_inputs=values["unresolved_inputs"],
        )
    try:
        root = canonicalize_absolute_path(values["source_root_canonical"])
    except (GuidedIdentityError, TypeError) as exc:
        raise _error(
            "invalid_rwd_source_snapshot",
            "RWD source snapshot root is not canonical.",
        ) from exc
    if (
        root.canonical_path != values["source_root_canonical"]
        or root.path_style != values["source_root_path_style"]
    ):
        raise _error(
            "invalid_rwd_source_snapshot",
            "RWD source snapshot root/path style mismatch.",
        )

    candidates = values["candidates"]
    if not candidates:
        raise _error(
            "invalid_rwd_source_snapshot",
            "RWD source snapshot candidate list is empty.",
        )
    paths = tuple(candidate.canonical_relative_path for candidate in candidates)
    if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
        raise _error(
            "invalid_rwd_source_snapshot",
            "RWD source snapshot candidates must be sorted and unique.",
        )
    for candidate in candidates:
        path = candidate.canonical_relative_path
        parts = path.split("/")
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
            or parts[-1] != "fluorescence.csv"
        ):
            raise _error(
                "invalid_rwd_source_snapshot",
                "RWD candidate relative path is invalid.",
                canonical_relative_path=path,
            )
        if (
            isinstance(candidate.size_bytes, bool)
            or not isinstance(candidate.size_bytes, int)
            or candidate.size_bytes < 0
        ):
            raise _error(
                "invalid_rwd_source_snapshot",
                "RWD candidate size is invalid.",
                canonical_relative_path=path,
            )
        if not _valid_sha256(candidate.sha256_content_digest):
            raise _error(
                "invalid_rwd_source_snapshot",
                "RWD candidate content digest is invalid.",
                canonical_relative_path=path,
            )


def _set_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_schema_name": values["snapshot_schema_name"],
        "snapshot_schema_version": values["snapshot_schema_version"],
        "discovery_rule_version": values["discovery_rule_version"],
        "path_canonicalization_version": values["path_canonicalization_version"],
        "relative_path_rule_version": values["relative_path_rule_version"],
        "digest_algorithm": values["digest_algorithm"],
        "source_root_canonical": values["source_root_canonical"],
        "source_root_path_style": values["source_root_path_style"],
        "source_format": values["source_format"],
        "acquisition_mode": values["acquisition_mode"],
        "candidates": [
            {
                "canonical_relative_path": candidate.canonical_relative_path,
                "size_bytes": candidate.size_bytes,
            }
            for candidate in values["candidates"]
        ],
        "ignored_files_policy": values["ignored_files_policy"],
        "build_mode": values["build_mode"],
        "unresolved_inputs": list(values["unresolved_inputs"]),
    }


def _content_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    payload = _set_payload(values)
    payload["candidates"] = [
        {
            "canonical_relative_path": candidate.canonical_relative_path,
            "size_bytes": candidate.size_bytes,
            "sha256_content_digest": candidate.sha256_content_digest,
        }
        for candidate in values["candidates"]
    ]
    return payload


def compute_rwd_source_candidate_set_digest(snapshot_or_payload: Any) -> str:
    values, is_snapshot = _snapshot_values(snapshot_or_payload)
    digest = _digest(_SET_DIGEST_DOMAIN, _set_payload(values))
    if (
        is_snapshot
        and snapshot_or_payload.source_candidate_set_digest != digest
    ):
        raise _error(
            "source_candidate_digest_mismatch",
            "Stored RWD candidate set digest does not match the snapshot.",
        )
    return digest


def compute_rwd_source_candidate_content_digest(snapshot_or_payload: Any) -> str:
    values, is_snapshot = _snapshot_values(snapshot_or_payload)
    digest = _digest(_CONTENT_DIGEST_DOMAIN, _content_payload(values))
    if (
        is_snapshot
        and snapshot_or_payload.source_candidate_content_digest != digest
    ):
        raise _error(
            "source_candidate_digest_mismatch",
            "Stored RWD candidate content digest does not match the snapshot.",
        )
    return digest


def _semantic_mapping(
    canonical_root: str,
    path_style: str,
    candidates: tuple[GuidedRwdSourceCandidateFile, ...],
) -> dict[str, Any]:
    return {
        "snapshot_schema_name": RWD_SOURCE_SNAPSHOT_SCHEMA_NAME,
        "snapshot_schema_version": RWD_SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "discovery_rule_version": RWD_SOURCE_DISCOVERY_RULE_VERSION,
        "path_canonicalization_version": CANONICALIZATION_ALGORITHM_VERSION,
        "relative_path_rule_version": RWD_RELATIVE_PATH_RULE_VERSION,
        "digest_algorithm": DIGEST_ALGORITHM,
        "source_root_canonical": canonical_root,
        "source_root_path_style": path_style,
        "source_format": SOURCE_FORMAT,
        "acquisition_mode": ACQUISITION_MODE,
        "candidates": candidates,
        "ignored_files_policy": RWD_IGNORED_FILES_POLICY,
        "build_mode": SOURCE_SNAPSHOT_BUILD_MODE,
        "unresolved_inputs": (),
    }


def _build_snapshot_attempt(
    source_root: str,
    canonical_root: str,
    path_style: str,
    cancellation_check: Callable[[], bool] | None,
) -> GuidedRwdSourceCandidateSnapshot:
    _check_cancelled(cancellation_check)
    before = _scan_candidates(source_root, path_style)
    _check_cancelled(cancellation_check)

    hashed: list[_HashedCandidate] = []
    for candidate in before.candidates:
        _check_cancelled(cancellation_check)
        hashed_candidate = _read_candidate_digest(candidate, cancellation_check)
        _validate_path_after_read(candidate, hashed_candidate)
        hashed.append(hashed_candidate)

    _check_cancelled(cancellation_check)
    after = _scan_candidates(source_root, path_style)
    if _candidate_keys(before) != _candidate_keys(after):
        raise _error(
            "source_discovery_changed",
            "RWD source candidate set changed during snapshot construction.",
            retryable=True,
            before=_candidate_keys(before),
            after=_candidate_keys(after),
        )
    for candidate_path, hashed_candidate in zip(before.candidates, hashed):
        _validate_path_after_read(candidate_path, hashed_candidate)

    _check_cancelled(cancellation_check)
    candidates = tuple(item.candidate for item in hashed)
    semantic = _semantic_mapping(canonical_root, path_style, candidates)
    set_digest = compute_rwd_source_candidate_set_digest(semantic)
    content_digest = compute_rwd_source_candidate_content_digest(semantic)
    return GuidedRwdSourceCandidateSnapshot(
        snapshot_schema_name=RWD_SOURCE_SNAPSHOT_SCHEMA_NAME,
        snapshot_schema_version=RWD_SOURCE_SNAPSHOT_SCHEMA_VERSION,
        discovery_rule_version=RWD_SOURCE_DISCOVERY_RULE_VERSION,
        path_canonicalization_version=CANONICALIZATION_ALGORITHM_VERSION,
        relative_path_rule_version=RWD_RELATIVE_PATH_RULE_VERSION,
        digest_algorithm=DIGEST_ALGORITHM,
        source_root_canonical=canonical_root,
        source_root_path_style=path_style,
        source_format=SOURCE_FORMAT,
        acquisition_mode=ACQUISITION_MODE,
        candidates=candidates,
        source_candidate_set_digest=set_digest,
        source_candidate_content_digest=content_digest,
        ignored_files_policy=RWD_IGNORED_FILES_POLICY,
        ignored_summary=after.ignored_summary,
        build_mode=SOURCE_SNAPSHOT_BUILD_MODE,
        unresolved_inputs=(),
    )


def build_rwd_source_candidate_snapshot(
    source_root: str,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedRwdSourceCandidateSnapshot:
    """Build a stable, content-bound RWD candidate snapshot without writes."""
    if not isinstance(source_root, str) or not source_root.strip():
        raise _error(
            "source_root_missing",
            "RWD source root is required.",
        )
    try:
        canonical = canonicalize_absolute_path(source_root)
    except GuidedIdentityError as exc:
        raise _error(
            "source_root_relative",
            "RWD source root must be an unambiguous absolute path.",
            source_root=source_root,
            reason=str(exc),
        ) from exc

    supported_styles = (
        {"windows_drive", "windows_unc"} if os.name == "nt" else {"posix"}
    )
    if canonical.path_style not in supported_styles:
        raise _error(
            "source_root_relative",
            "RWD source root path style is unsupported on this host.",
            path_style=canonical.path_style,
        )

    source_path = str(Path(source_root))
    try:
        root_stat = os.stat(source_path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise _error(
            "source_root_not_directory",
            "RWD source root does not exist.",
            source_root=source_root,
        ) from exc
    except OSError as exc:
        raise _error(
            "source_root_unreadable",
            "Unable to inspect the RWD source root.",
            source_root=source_root,
            reason=str(exc),
        ) from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise _error(
            "source_root_not_directory",
            "RWD source root must be a directory.",
            source_root=source_root,
        )

    last_error: RwdSourceSnapshotError | None = None
    for attempt in range(2):
        try:
            return _build_snapshot_attempt(
                source_path,
                canonical.canonical_path,
                canonical.path_style,
                cancellation_check,
            )
        except RwdSourceSnapshotError as exc:
            if not exc.retryable:
                raise
            last_error = exc
            if attempt == 0:
                continue
    raise _error(
        "source_changed_during_snapshot",
        "RWD source remained unstable after one complete retry.",
        last_category=last_error.category if last_error else None,
    )


def _validate_classification(
    classification: GuidedIncompleteFinalChunkClassification,
) -> dict[str, Any]:
    if not isinstance(classification, GuidedIncompleteFinalChunkClassification):
        raise _error(
            "invalid_rwd_source_snapshot",
            "Expected a fixed incomplete-final classification.",
        )
    if (
        classification.schema_name
        != INCOMPLETE_FINAL_CLASSIFICATION_SCHEMA_NAME
        or classification.schema_version
        != INCOMPLETE_FINAL_CLASSIFICATION_SCHEMA_VERSION
        or classification.classifier_version
        != INCOMPLETE_FINAL_CLASSIFIER_VERSION
        or classification.classification_status != NOT_REQUESTED_STATUS
    ):
        raise _error(
            "invalid_rwd_source_snapshot",
            "Unsupported incomplete-final classification contract.",
        )
    if (
        not _valid_sha256(classification.source_candidate_set_digest)
        or not _valid_sha256(classification.source_candidate_content_digest)
    ):
        raise _error(
            "invalid_rwd_source_snapshot",
            "Incomplete-final classification candidate digest is invalid.",
        )
    if (
        classification.excluded_canonical_relative_path is not None
        or classification.reason is not None
        or classification.evidence is not None
        or classification.parsing_contract_digest is not None
        or classification.timing_contract_digest is not None
        or classification.unresolved_inputs
        or classification.policy.exclude_incomplete_final_rwd_chunk is not False
        or classification.policy.non_final_short_chunks_block is not True
        or classification.policy.malformed_final_chunk_blocks is not True
        or classification.policy.exactly_one_final_chunk_may_be_excluded is not True
    ):
        raise _error(
            "invalid_rwd_source_snapshot",
            "Invalid not_requested incomplete-final classification state.",
        )
    return {
        "schema_name": classification.schema_name,
        "schema_version": classification.schema_version,
        "classifier_version": classification.classifier_version,
        "classification_status": classification.classification_status,
        "source_candidate_set_digest": classification.source_candidate_set_digest,
        "source_candidate_content_digest": (
            classification.source_candidate_content_digest
        ),
        "excluded_canonical_relative_path": None,
        "reason": None,
        "evidence": None,
        "policy": {
            "exclude_incomplete_final_rwd_chunk": False,
            "non_final_short_chunks_block": True,
            "malformed_final_chunk_blocks": True,
            "exactly_one_final_chunk_may_be_excluded": True,
        },
        "parsing_contract_digest": None,
        "timing_contract_digest": None,
        "unresolved_inputs": [],
    }


def make_not_requested_incomplete_final_chunk_classification(
    snapshot: GuidedRwdSourceCandidateSnapshot,
) -> GuidedIncompleteFinalChunkClassification:
    """Bind a deterministic no-exclusion policy to a complete snapshot."""
    if not isinstance(snapshot, GuidedRwdSourceCandidateSnapshot):
        raise _error(
            "invalid_rwd_source_snapshot",
            "A complete RWD source candidate snapshot is required.",
        )
    set_digest = compute_rwd_source_candidate_set_digest(snapshot)
    content_digest = compute_rwd_source_candidate_content_digest(snapshot)
    return GuidedIncompleteFinalChunkClassification(
        schema_name=INCOMPLETE_FINAL_CLASSIFICATION_SCHEMA_NAME,
        schema_version=INCOMPLETE_FINAL_CLASSIFICATION_SCHEMA_VERSION,
        classifier_version=INCOMPLETE_FINAL_CLASSIFIER_VERSION,
        classification_status=NOT_REQUESTED_STATUS,
        source_candidate_set_digest=set_digest,
        source_candidate_content_digest=content_digest,
        excluded_canonical_relative_path=None,
        reason=None,
        evidence=None,
        policy=GuidedIncompleteFinalChunkPolicy(
            exclude_incomplete_final_rwd_chunk=False,
        ),
        parsing_contract_digest=None,
        timing_contract_digest=None,
        unresolved_inputs=(),
    )


def compute_incomplete_final_chunk_classification_digest(
    classification: GuidedIncompleteFinalChunkClassification,
) -> str:
    """Digest the fixed non-authorizing not_requested classification."""
    return _digest(
        _CLASSIFICATION_DIGEST_DOMAIN,
        _validate_classification(classification),
    )
