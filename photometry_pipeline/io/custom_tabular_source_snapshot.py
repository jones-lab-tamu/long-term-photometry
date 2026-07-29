"""Read-only source snapshot for Guided one-CSV-per-session recordings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Callable

from photometry_pipeline.core.utils import natural_sort_key
from photometry_pipeline.guided_identity import (
    CANONICALIZATION_ALGORITHM_VERSION,
    canonicalize_absolute_path,
    encode_canonical_value,
)


CUSTOM_TABULAR_SOURCE_SNAPSHOT_SCHEMA_NAME = (
    "guided_custom_tabular_source_candidate_snapshot"
)
CUSTOM_TABULAR_SOURCE_SNAPSHOT_SCHEMA_VERSION = "v1"
CUSTOM_TABULAR_SOURCE_DISCOVERY_RULE_VERSION = "top_level_csv_natural_order.v1"
CUSTOM_TABULAR_RELATIVE_PATH_RULE_VERSION = (
    "canonical_forward_slash_relative_path.v1"
)
CUSTOM_TABULAR_IGNORED_FILES_POLICY = "ignore_non_csv_and_nested_entries.v1"
CUSTOM_TABULAR_SOURCE_SNAPSHOT_BUILD_MODE = "read_only"


class CustomTabularSourceSnapshotError(ValueError):
    def __init__(self, category: str, message: str, *, retryable: bool = False):
        self.category = category
        self.message = message
        self.context = {}
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True)
class GuidedCustomTabularSourceCandidateFile:
    canonical_relative_path: str
    size_bytes: int
    sha256_content_digest: str


@dataclass(frozen=True)
class GuidedCustomTabularIgnoredFilesSummary:
    root_non_target_entry_count: int = 0
    session_non_target_entry_count: int = 0
    nested_entry_count_not_scanned: int = 0


@dataclass(frozen=True)
class GuidedCustomTabularSourceCandidateSnapshot:
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
    candidates: tuple[GuidedCustomTabularSourceCandidateFile, ...]
    source_candidate_set_digest: str
    source_candidate_content_digest: str
    ignored_files_policy: str
    ignored_summary: GuidedCustomTabularIgnoredFilesSummary
    build_mode: str
    unresolved_inputs: tuple[str, ...] = ()


def _digest(domain: bytes, payload) -> str:
    return hashlib.sha256(domain + b"\x00" + encode_canonical_value(payload)).hexdigest()


def build_custom_tabular_source_candidate_snapshot(
    source_root: str,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedCustomTabularSourceCandidateSnapshot:
    try:
        canonical_root = canonicalize_absolute_path(os.path.abspath(source_root))
    except Exception as exc:
        raise CustomTabularSourceSnapshotError(
            "source_root_invalid", "The selected CSV source folder is invalid."
        ) from exc
    try:
        entries = list(os.scandir(source_root))
    except OSError as exc:
        raise CustomTabularSourceSnapshotError(
            "source_root_unreadable", "The selected CSV source folder could not be read."
        ) from exc
    csv_entries = [
        item
        for item in entries
        if item.is_file(follow_symlinks=False)
        and item.name.lower().endswith(".csv")
    ]
    csv_entries.sort(key=lambda item: natural_sort_key(item.name))
    if not csv_entries:
        raise CustomTabularSourceSnapshotError(
            "no_custom_tabular_files",
            "No top-level CSV files were found in the selected folder.",
        )
    collision_keys = [
        item.name.casefold()
        if canonical_root.path_style in {"windows_drive", "windows_unc"}
        else item.name
        for item in csv_entries
    ]
    if len(set(collision_keys)) != len(collision_keys):
        raise CustomTabularSourceSnapshotError(
            "candidate_path_collision", "The CSV source entries are not unique."
        )

    candidates: list[GuidedCustomTabularSourceCandidateFile] = []
    for entry in csv_entries:
        if cancellation_check is not None and cancellation_check():
            raise CustomTabularSourceSnapshotError(
                "source_candidate_snapshot_cancelled",
                "CSV source checking was cancelled.",
            )
        try:
            before = os.stat(entry.path, follow_symlinks=False)
            digest = hashlib.sha256()
            size = 0
            with open(entry.path, "rb") as handle:
                while True:
                    if cancellation_check is not None and cancellation_check():
                        raise CustomTabularSourceSnapshotError(
                            "source_candidate_snapshot_cancelled",
                            "CSV source checking was cancelled.",
                        )
                    block = handle.read(4 * 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    size += len(block)
            after = os.stat(entry.path, follow_symlinks=False)
        except OSError as exc:
            raise CustomTabularSourceSnapshotError(
                "candidate_unreadable",
                f"The CSV file {entry.name} could not be read.",
            ) from exc
        if (
            size != before.st_size
            or size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise CustomTabularSourceSnapshotError(
                "source_changed_during_snapshot",
                f"The CSV file {entry.name} changed while it was checked.",
                retryable=True,
            )
        candidates.append(
            GuidedCustomTabularSourceCandidateFile(
                canonical_relative_path=Path(entry.name).as_posix(),
                size_bytes=size,
                sha256_content_digest=digest.hexdigest(),
            )
        )

    common = {
        "source_root_canonical": canonical_root.canonical_path,
        "source_format": "custom_tabular",
        "acquisition_mode": "intermittent",
        "ordered_candidates": [
            {
                "canonical_relative_path": item.canonical_relative_path,
                "size_bytes": item.size_bytes,
                "sha256_content_digest": item.sha256_content_digest,
            }
            for item in candidates
        ],
    }
    return GuidedCustomTabularSourceCandidateSnapshot(
        snapshot_schema_name=CUSTOM_TABULAR_SOURCE_SNAPSHOT_SCHEMA_NAME,
        snapshot_schema_version=CUSTOM_TABULAR_SOURCE_SNAPSHOT_SCHEMA_VERSION,
        discovery_rule_version=CUSTOM_TABULAR_SOURCE_DISCOVERY_RULE_VERSION,
        path_canonicalization_version=CANONICALIZATION_ALGORITHM_VERSION,
        relative_path_rule_version=CUSTOM_TABULAR_RELATIVE_PATH_RULE_VERSION,
        digest_algorithm="sha256",
        source_root_canonical=canonical_root.canonical_path,
        source_root_path_style=canonical_root.path_style,
        source_format="custom_tabular",
        acquisition_mode="intermittent",
        candidates=tuple(candidates),
        source_candidate_set_digest=_digest(
            b"guided-custom-tabular-source-candidate-set:v1",
            {
                **common,
                "ordered_candidates": [
                    item["canonical_relative_path"]
                    for item in common["ordered_candidates"]
                ],
            },
        ),
        source_candidate_content_digest=_digest(
            b"guided-custom-tabular-source-candidate-content:v1", common
        ),
        ignored_files_policy=CUSTOM_TABULAR_IGNORED_FILES_POLICY,
        ignored_summary=GuidedCustomTabularIgnoredFilesSummary(
            root_non_target_entry_count=len(entries) - len(csv_entries)
        ),
        build_mode=CUSTOM_TABULAR_SOURCE_SNAPSHOT_BUILD_MODE,
    )
