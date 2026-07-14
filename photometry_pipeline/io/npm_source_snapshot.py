"""Immutable, read-only NPM source snapshots and filename chronology."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping

from photometry_pipeline.guided_identity import (
    CANONICALIZATION_ALGORITHM_VERSION,
    GuidedIdentityError,
    canonicalize_absolute_path,
    encode_canonical_value,
)


NPM_SOURCE_SNAPSHOT_SCHEMA_NAME = "guided_npm_source_candidate_snapshot"
NPM_SOURCE_SNAPSHOT_SCHEMA_VERSION = "v1"
NPM_SOURCE_DISCOVERY_RULE_VERSION = "immediate_child_csv_exact_filename_timestamp.v1"
NPM_RELATIVE_PATH_RULE_VERSION = "canonical_forward_slash_relative_path.v1"
NPM_IGNORED_FILES_POLICY = "ignore_non_csv_immediate_children.v1"
NPM_SOURCE_BUILD_MODE = "read_only"
NPM_FILENAME_TIMESTAMP_RULE = "exactly_one_yyyy_mm_dd_thh_mm_ss_in_stem.v1"
NPM_FILENAME_TIMESTAMP_FORMAT = "%Y-%m-%dT%H_%M_%S"
NPM_SOURCE_CANDIDATE_SET_DOMAIN = "guided-npm-source-candidate-set:v1"
NPM_SOURCE_CANDIDATE_CONTENT_DOMAIN = "guided-npm-source-candidate-content:v1"
_TIMESTAMP_TOKEN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}")


class NpmSourceSnapshotError(ValueError):
    def __init__(self, category: str, message: str, **context: Any) -> None:
        self.category = category
        self.message = message
        self.context = dict(context)
        super().__init__(message)


@dataclass(frozen=True)
class GuidedNpmSourceCandidateFile:
    canonical_relative_path: str
    size_bytes: int
    sha256_content_digest: str
    authoritative_source_start_time: str


@dataclass(frozen=True)
class GuidedNpmSourceCandidateSnapshot:
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
    candidates: tuple[GuidedNpmSourceCandidateFile, ...]
    source_candidate_set_digest: str
    source_candidate_content_digest: str
    ignored_files_policy: str
    build_mode: str
    filename_timestamp_rule: str = NPM_FILENAME_TIMESTAMP_RULE
    unresolved_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuidedNpmDiscoveredSourceFile:
    """One source-set member found by the accepted NPM discovery rule."""

    canonical_relative_path: str
    absolute_path: str


@dataclass(frozen=True)
class GuidedNpmDiscoveredSourceSet:
    """Membership-only discovery result; no chronology or CSV inspection."""

    source_root_canonical: str
    source_root_path_style: str
    discovery_rule_version: str
    ignored_files_policy: str
    files: tuple[GuidedNpmDiscoveredSourceFile, ...]


def parse_npm_filename_timestamp(path: str) -> datetime:
    """Parse one and only one valid vendor timestamp from a filename stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    tokens = _TIMESTAMP_TOKEN.findall(stem)
    if len(tokens) != 1:
        raise NpmSourceSnapshotError(
            "malformed_filename_timestamp",
            "Each NPM CSV filename must contain exactly one valid "
            "YYYY-MM-DDTHH_MM_SS timestamp.",
            filename=os.path.basename(path),
        )
    try:
        return datetime.strptime(tokens[0], NPM_FILENAME_TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise NpmSourceSnapshotError(
            "malformed_filename_timestamp",
            "An NPM CSV filename contains an invalid timestamp.",
            filename=os.path.basename(path),
        ) from exc


def _canonical_relative_path(root: str, path: str, path_style: str) -> str:
    relative = os.path.relpath(path, root).replace("\\", "/")
    if path_style in {"windows_drive", "windows_unc"}:
        relative = relative.casefold()
    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise NpmSourceSnapshotError(
            "invalid_source_candidate_path",
            "NPM source candidate path is not a canonical relative path.",
        )
    return relative


def _candidate_paths(source_path: str) -> tuple[str, str, list[str]]:
    if os.path.isfile(source_path):
        if not source_path.lower().endswith(".csv"):
            raise NpmSourceSnapshotError(
                "unsupported_source", "The selected NPM source file is not CSV."
            )
        return os.path.dirname(source_path), os.path.basename(source_path), [source_path]
    if not os.path.isdir(source_path):
        raise NpmSourceSnapshotError(
            "source_root_missing", "The NPM source path does not exist."
        )
    paths = [
        os.path.join(source_path, name)
        for name in os.listdir(source_path)
        if name.lower().endswith(".csv") and os.path.isfile(os.path.join(source_path, name))
    ]
    if not paths:
        raise NpmSourceSnapshotError(
            "no_npm_csv_files", "No NPM CSV files were found in the selected source."
        )
    return source_path, "", paths


def discover_npm_source_files(source_path: str) -> GuidedNpmDiscoveredSourceSet:
    """Apply the exact B2-B NPM source-membership predicate only.

    Discovery is source-root-only, case-insensitive for the ``.csv`` suffix,
    excludes directories, and ignores non-CSV immediate children.  It does
    not parse filename timestamps, inspect CSV contents, hash files, or impose
    execution order.
    """
    if not isinstance(source_path, str) or not source_path.strip():
        raise NpmSourceSnapshotError(
            "source_root_missing", "An NPM source path is required."
        )
    try:
        canonical_input = canonicalize_absolute_path(source_path)
    except GuidedIdentityError as exc:
        raise NpmSourceSnapshotError(
            "source_root_relative", "NPM source path must be an absolute path."
        ) from exc
    root, _single_name, paths = _candidate_paths(source_path)
    canonical_root = canonicalize_absolute_path(root)
    if canonical_input.path_style != canonical_root.path_style:
        raise NpmSourceSnapshotError(
            "source_path_style_unavailable", "NPM source path style is inconsistent."
        )
    return GuidedNpmDiscoveredSourceSet(
        source_root_canonical=canonical_root.canonical_path,
        source_root_path_style=canonical_root.path_style,
        discovery_rule_version=NPM_SOURCE_DISCOVERY_RULE_VERSION,
        ignored_files_policy=NPM_IGNORED_FILES_POLICY,
        files=tuple(
            GuidedNpmDiscoveredSourceFile(
                canonical_relative_path=_canonical_relative_path(
                    root, path, canonical_root.path_style
                ),
                absolute_path=path,
            )
            for path in paths
        ),
    )


def _hash_stable(path: str) -> tuple[int, str]:
    try:
        before = os.stat(path, follow_symlinks=False)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        after = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise NpmSourceSnapshotError(
            "candidate_unreadable", "Unable to read an NPM source candidate.", path=path
        ) from exc
    if not stat.S_ISREG(before.st_mode) or not stat.S_ISREG(after.st_mode):
        raise NpmSourceSnapshotError(
            "candidate_non_regular", "An NPM source candidate is not a regular file."
        )
    if (
        before.st_size != after.st_size
        or getattr(before, "st_mtime_ns", None) != getattr(after, "st_mtime_ns", None)
    ):
        raise NpmSourceSnapshotError(
            "source_changed_during_snapshot",
            "An NPM source candidate changed while the snapshot was built.",
            path=path,
        )
    return int(after.st_size), digest


def _digest(domain: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + encode_canonical_value(dict(payload))
    ).hexdigest()


def _snapshot_payload(snapshot: GuidedNpmSourceCandidateSnapshot, *, content: bool) -> dict[str, Any]:
    return {
        "snapshot_schema_name": snapshot.snapshot_schema_name,
        "snapshot_schema_version": snapshot.snapshot_schema_version,
        "discovery_rule_version": snapshot.discovery_rule_version,
        "path_canonicalization_version": snapshot.path_canonicalization_version,
        "relative_path_rule_version": snapshot.relative_path_rule_version,
        "digest_algorithm": "sha256",
        "source_root_canonical": snapshot.source_root_canonical,
        "source_root_path_style": snapshot.source_root_path_style,
        "source_format": snapshot.source_format,
        "acquisition_mode": snapshot.acquisition_mode,
        "candidates": [
            {
                "canonical_relative_path": item.canonical_relative_path,
                "size_bytes": item.size_bytes,
                **(
                    {
                        "sha256_content_digest": item.sha256_content_digest,
                        "authoritative_source_start_time": item.authoritative_source_start_time,
                    }
                    if content
                    else {
                        "authoritative_source_start_time": item.authoritative_source_start_time,
                    }
                ),
            }
            for item in snapshot.candidates
        ],
        "ignored_files_policy": snapshot.ignored_files_policy,
        "build_mode": snapshot.build_mode,
        "filename_timestamp_rule": snapshot.filename_timestamp_rule,
        "unresolved_inputs": list(snapshot.unresolved_inputs),
    }


def compute_npm_source_candidate_set_digest(snapshot: GuidedNpmSourceCandidateSnapshot) -> str:
    return _digest(NPM_SOURCE_CANDIDATE_SET_DOMAIN, _snapshot_payload(snapshot, content=False))


def compute_npm_source_candidate_content_digest(snapshot: GuidedNpmSourceCandidateSnapshot) -> str:
    return _digest(NPM_SOURCE_CANDIDATE_CONTENT_DOMAIN, _snapshot_payload(snapshot, content=True))


def build_npm_source_candidate_snapshot(
    source_path: str,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedNpmSourceCandidateSnapshot:
    """Build a stable NPM source snapshot in authoritative filename order."""
    discovered = discover_npm_source_files(source_path)
    stamped: list[tuple[datetime, str]] = []
    for item in discovered.files:
        if cancellation_check and cancellation_check():
            raise NpmSourceSnapshotError("source_snapshot_cancelled", "NPM source snapshot was cancelled.")
        stamped.append((parse_npm_filename_timestamp(item.absolute_path), item.absolute_path))
    timestamps = [item[0] for item in stamped]
    if len(set(timestamps)) != len(timestamps):
        raise NpmSourceSnapshotError(
            "duplicate_filename_timestamp",
            "NPM source filenames contain duplicate acquisition timestamps.",
        )
    stamped.sort(key=lambda item: item[0])
    candidates: list[GuidedNpmSourceCandidateFile] = []
    relative_by_path = {
        item.absolute_path: item.canonical_relative_path for item in discovered.files
    }
    for timestamp, path in stamped:
        size, digest = _hash_stable(path)
        candidates.append(
            GuidedNpmSourceCandidateFile(
                canonical_relative_path=relative_by_path[path],
                size_bytes=size,
                sha256_content_digest=digest,
                authoritative_source_start_time=timestamp.isoformat(),
            )
        )
    snapshot = GuidedNpmSourceCandidateSnapshot(
        snapshot_schema_name=NPM_SOURCE_SNAPSHOT_SCHEMA_NAME,
        snapshot_schema_version=NPM_SOURCE_SNAPSHOT_SCHEMA_VERSION,
        discovery_rule_version=NPM_SOURCE_DISCOVERY_RULE_VERSION,
        path_canonicalization_version=CANONICALIZATION_ALGORITHM_VERSION,
        relative_path_rule_version=NPM_RELATIVE_PATH_RULE_VERSION,
        digest_algorithm="sha256",
        source_root_canonical=discovered.source_root_canonical,
        source_root_path_style=discovered.source_root_path_style,
        source_format="npm",
        acquisition_mode="intermittent",
        candidates=tuple(candidates),
        source_candidate_set_digest="",
        source_candidate_content_digest="",
        ignored_files_policy=NPM_IGNORED_FILES_POLICY,
        build_mode=NPM_SOURCE_BUILD_MODE,
    )
    return GuidedNpmSourceCandidateSnapshot(
        **{
            **snapshot.__dict__,
            "source_candidate_set_digest": compute_npm_source_candidate_set_digest(snapshot),
            "source_candidate_content_digest": compute_npm_source_candidate_content_digest(snapshot),
        }
    )
