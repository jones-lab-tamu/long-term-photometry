"""Read-only candidate-manifest and ROI preflights for Guided production."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import os
from pathlib import PurePosixPath
from typing import Any, Callable

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_production_mapping import (
    GuidedProductionExecutionIntent,
)
from photometry_pipeline.io.rwd_contract import (
    RwdHeaderInspectionError,
    RwdHeaderParsingContract,
    compute_rwd_header_parsing_contract_digest,
    inspect_rwd_header_contract,
)
from photometry_pipeline.io.rwd_source_snapshot import (
    RWD_IGNORED_FILES_POLICY,
    RWD_RELATIVE_PATH_RULE_VERSION,
    RWD_SOURCE_DISCOVERY_RULE_VERSION,
    RWD_SOURCE_SNAPSHOT_SCHEMA_NAME,
    RWD_SOURCE_SNAPSHOT_SCHEMA_VERSION,
    SOURCE_FORMAT,
    RwdSourceSnapshotError,
    build_rwd_source_candidate_snapshot,
)


GUIDED_CANDIDATE_PREFLIGHT_CONTRACT_VERSION = (
    "exact_candidate_manifest_preflight.v1"
)
GUIDED_ROI_PREFLIGHT_CONTRACT_VERSION = "exact_included_roi_tuple.v1"
GUIDED_CANDIDATE_PREFLIGHT_IDENTITY_DOMAIN = (
    "guided-candidate-execution-preflight:v1"
)
GUIDED_ROI_PREFLIGHT_IDENTITY_DOMAIN = "guided-roi-execution-preflight:v1"
GUIDED_STRICT_ROI_INVENTORY_IDENTITY_DOMAIN = (
    "guided-strict-roi-inventory:v1"
)

GUIDED_CANDIDATE_PREFLIGHT_REFUSAL_CATEGORIES = (
    "candidate_preflight_contract_unavailable",
    "source_root_missing",
    "source_root_not_readable",
    "source_path_canonicalization_mismatch",
    "source_format_unsupported",
    "candidate_discovery_failed",
    "candidate_relative_path_mismatch",
    "candidate_file_missing",
    "candidate_file_extra",
    "candidate_file_size_mismatch",
    "candidate_file_digest_mismatch",
    "candidate_set_digest_mismatch",
    "candidate_content_digest_mismatch",
    "candidate_duplicate_path",
    "candidate_manifest_internal_error",
)
GUIDED_CANDIDATE_PREFLIGHT_REFUSAL_CATEGORY_SET = frozenset(
    GUIDED_CANDIDATE_PREFLIGHT_REFUSAL_CATEGORIES
)
GUIDED_CANDIDATE_PREFLIGHT_RESERVED_REFUSAL_CATEGORIES = (
    "candidate_open_binding_failed",
    "candidate_changed_after_preflight",
    "candidate_runner_consumption_mismatch",
)

GUIDED_ROI_PREFLIGHT_REFUSAL_CATEGORIES = (
    "roi_preflight_contract_unavailable",
    "parser_contract_unsupported",
    "parser_digest_mismatch",
    "roi_discovery_failed",
    "roi_tuple_mismatch",
    "roi_inventory_digest_mismatch",
    "roi_include_tuple_mismatch",
    "roi_exclude_tuple_mismatch",
    "roi_missing_included",
    "roi_extra_analyzed",
    "roi_duplicate",
    "roi_ambiguous",
    "roi_parser_column_mismatch",
    "roi_source_digest_mismatch",
    "roi_preflight_internal_error",
)
GUIDED_ROI_PREFLIGHT_REFUSAL_CATEGORY_SET = frozenset(
    GUIDED_ROI_PREFLIGHT_REFUSAL_CATEGORIES
)
GUIDED_ROI_PREFLIGHT_RESERVED_REFUSAL_CATEGORIES = (
    "roi_candidate_changed_during_inspection",
    "roi_runner_consumption_mismatch",
    "roi_inventory_identity_contract_unavailable",
)

_HEX = frozenset("0123456789abcdef")
_PLACEHOLDERS = frozenset(
    {"", "unknown", "unavailable", "placeholder", "unset", "none"}
)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _version(value: Any) -> bool:
    return _text(value) and value.strip().lower() not in _PLACEHOLDERS


def _canonical_relative_path(value: Any) -> bool:
    if not _text(value) or "\\" in value or value.startswith("/"):
        return False
    parts = value.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _no_side_effects(values: tuple[bool, ...]) -> bool:
    return all(value is True for value in values)


@dataclass(frozen=True)
class GuidedCandidateManifestEntry:
    canonical_relative_path: str
    size_bytes: int
    sha256_content_digest: str

    def __post_init__(self) -> None:
        if not _canonical_relative_path(self.canonical_relative_path):
            raise ValueError("Candidate relative path is not canonical.")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("Candidate size must be a non-negative integer.")
        if not _sha256(self.sha256_content_digest):
            raise ValueError("Candidate digest must be a lowercase SHA-256.")


@dataclass(frozen=True)
class GuidedCandidateManifestExecutionPreflightRequest:
    contract_version: str
    runner_contract_version: str
    source_root_canonical: str
    source_root_path_style: str
    source_format: str
    snapshot_schema_name: str
    snapshot_schema_version: str
    discovery_rule_version: str
    path_canonicalization_version: str
    relative_path_rule_version: str
    ignored_files_policy: str
    source_identity_level: str
    expected_candidate_set_digest: str
    expected_candidate_content_digest: str
    expected_candidates: tuple[GuidedCandidateManifestEntry, ...]
    parser_contract_version: str
    parser_contract_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.expected_candidates, tuple):
            raise ValueError("expected_candidates must be a tuple.")


@dataclass(frozen=True)
class GuidedCandidateManifestExecutionPreflightIssue:
    category: str
    section: str
    message: str
    detail_code: str = ""

    def __post_init__(self) -> None:
        if self.category not in GUIDED_CANDIDATE_PREFLIGHT_REFUSAL_CATEGORY_SET:
            raise ValueError("Unsupported candidate preflight issue.")


@dataclass(frozen=True)
class GuidedCandidateManifestExecutionPreflightResult:
    status: str
    accepted: bool
    contract_version: str
    runner_contract_version: str
    expected_candidate_set_digest: str
    expected_candidate_content_digest: str
    actual_candidate_set_digest: str | None
    actual_candidate_content_digest: str | None
    actual_candidates: tuple[GuidedCandidateManifestEntry, ...]
    blocking_issues: tuple[GuidedCandidateManifestExecutionPreflightIssue, ...]
    canonical_preflight_identity: str | None
    no_files_written: bool = True
    no_directories_created: bool = True
    no_artifacts_created: bool = True
    no_run_id_allocated: bool = True
    no_config_or_argv_generated: bool = True
    no_runner_invoked: bool = True

    def __post_init__(self) -> None:
        if not _no_side_effects(
            (
                self.no_files_written,
                self.no_directories_created,
                self.no_artifacts_created,
                self.no_run_id_allocated,
                self.no_config_or_argv_generated,
                self.no_runner_invoked,
            )
        ):
            raise ValueError("Candidate preflight cannot report side effects.")
        if self.accepted:
            if (
                self.status != "accepted"
                or self.blocking_issues
                or not self.actual_candidates
                or not _sha256(self.actual_candidate_set_digest)
                or not _sha256(self.actual_candidate_content_digest)
                or not _sha256(self.canonical_preflight_identity)
            ):
                raise ValueError("Accepted candidate preflight is incomplete.")
        elif (
            self.status != "refused"
            or not self.blocking_issues
            or self.canonical_preflight_identity is not None
        ):
            raise ValueError("Refused candidate preflight is inconsistent.")


def _candidate_issue(
    category: str, section: str, message: str, detail_code: str
) -> GuidedCandidateManifestExecutionPreflightIssue:
    return GuidedCandidateManifestExecutionPreflightIssue(
        category, section, message, detail_code
    )


def _candidate_refused(
    request: Any,
    issue: GuidedCandidateManifestExecutionPreflightIssue,
    *,
    actual_candidates: tuple[GuidedCandidateManifestEntry, ...] = (),
    actual_set_digest: str | None = None,
    actual_content_digest: str | None = None,
) -> GuidedCandidateManifestExecutionPreflightResult:
    return GuidedCandidateManifestExecutionPreflightResult(
        status="refused",
        accepted=False,
        contract_version=str(getattr(request, "contract_version", "") or ""),
        runner_contract_version=str(
            getattr(request, "runner_contract_version", "") or ""
        ),
        expected_candidate_set_digest=str(
            getattr(request, "expected_candidate_set_digest", "") or ""
        ),
        expected_candidate_content_digest=str(
            getattr(request, "expected_candidate_content_digest", "") or ""
        ),
        actual_candidate_set_digest=actual_set_digest,
        actual_candidate_content_digest=actual_content_digest,
        actual_candidates=actual_candidates,
        blocking_issues=(issue,),
        canonical_preflight_identity=None,
    )


def _candidate_identity_payload(
    result: GuidedCandidateManifestExecutionPreflightResult,
) -> dict[str, Any]:
    return {
        "identity_domain": GUIDED_CANDIDATE_PREFLIGHT_IDENTITY_DOMAIN,
        "result": {
            "contract_version": result.contract_version,
            "runner_contract_version": result.runner_contract_version,
            "expected_candidate_set_digest": result.expected_candidate_set_digest,
            "expected_candidate_content_digest": (
                result.expected_candidate_content_digest
            ),
            "actual_candidate_set_digest": result.actual_candidate_set_digest,
            "actual_candidate_content_digest": result.actual_candidate_content_digest,
            "actual_candidates": [
                {
                    "canonical_relative_path": item.canonical_relative_path,
                    "size_bytes": item.size_bytes,
                    "sha256_content_digest": item.sha256_content_digest,
                }
                for item in result.actual_candidates
            ],
            "accepted": result.accepted,
        },
    }


def compute_guided_candidate_preflight_identity(
    result: GuidedCandidateManifestExecutionPreflightResult,
) -> str:
    if (
        not isinstance(result, GuidedCandidateManifestExecutionPreflightResult)
        or not result.accepted
    ):
        raise ValueError("Only an accepted candidate preflight has an identity.")
    return hashlib.sha256(
        encode_canonical_value(_candidate_identity_payload(result))
    ).hexdigest()


def derive_candidate_manifest_preflight_request_from_intent(
    intent: GuidedProductionExecutionIntent,
) -> GuidedCandidateManifestExecutionPreflightRequest:
    if not isinstance(intent, GuidedProductionExecutionIntent):
        raise TypeError("intent must be a GuidedProductionExecutionIntent.")
    source = intent.input_source
    return GuidedCandidateManifestExecutionPreflightRequest(
        contract_version=source.candidate_manifest_execution_contract_version,
        runner_contract_version=intent.runner_contract_version,
        source_root_canonical=source.source_root_canonical,
        source_root_path_style=source.source_root_path_style,
        source_format=source.source_format,
        snapshot_schema_name=source.snapshot_schema_name,
        snapshot_schema_version=source.snapshot_schema_version,
        discovery_rule_version=source.discovery_rule_version,
        path_canonicalization_version=source.path_canonicalization_version,
        relative_path_rule_version=source.relative_path_rule_version,
        ignored_files_policy=source.ignored_files_policy,
        source_identity_level=source.source_identity_level,
        expected_candidate_set_digest=source.source_candidate_set_digest,
        expected_candidate_content_digest=source.source_candidate_content_digest,
        expected_candidates=tuple(
            GuidedCandidateManifestEntry(
                item.canonical_relative_path,
                item.size_bytes,
                item.sha256_content_digest,
            )
            for item in source.candidate_files
        ),
        parser_contract_version=intent.parser.schema_version,
        parser_contract_digest=intent.parser.parser_contract_digest,
    )


def run_candidate_manifest_execution_preflight(
    request: GuidedCandidateManifestExecutionPreflightRequest,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedCandidateManifestExecutionPreflightResult:
    if not isinstance(request, GuidedCandidateManifestExecutionPreflightRequest):
        return _candidate_refused(
            request,
            _candidate_issue(
                "candidate_preflight_contract_unavailable",
                "candidate_preflight",
                "Candidate preflight request is invalid.",
                "request_invalid_type",
            ),
        )
    if (
        request.contract_version != GUIDED_CANDIDATE_PREFLIGHT_CONTRACT_VERSION
        or not _version(request.runner_contract_version)
    ):
        return _candidate_refused(
            request,
            _candidate_issue(
                "candidate_preflight_contract_unavailable",
                "candidate_preflight",
                "Candidate preflight contract is unsupported.",
                "contract_unsupported",
            ),
        )
    if request.source_format != SOURCE_FORMAT:
        return _candidate_refused(
            request,
            _candidate_issue(
                "source_format_unsupported",
                "source",
                "Candidate preflight requires RWD source.",
                "source_format_not_rwd",
            ),
        )
    if not _text(request.source_root_canonical):
        return _candidate_refused(
            request,
            _candidate_issue(
                "source_root_missing",
                "source",
                "Source root is required.",
                "source_root_missing",
            ),
        )
    if (
        request.source_identity_level != "content_bound_candidate_snapshot"
        or request.snapshot_schema_name != RWD_SOURCE_SNAPSHOT_SCHEMA_NAME
        or request.snapshot_schema_version != RWD_SOURCE_SNAPSHOT_SCHEMA_VERSION
        or request.discovery_rule_version != RWD_SOURCE_DISCOVERY_RULE_VERSION
        or request.relative_path_rule_version != RWD_RELATIVE_PATH_RULE_VERSION
        or request.ignored_files_policy != RWD_IGNORED_FILES_POLICY
    ):
        return _candidate_refused(
            request,
            _candidate_issue(
                "candidate_preflight_contract_unavailable",
                "source",
                "Source snapshot policy is unsupported.",
                "snapshot_policy_unsupported",
            ),
        )
    if (
        not request.expected_candidates
        or not _sha256(request.expected_candidate_set_digest)
        or not _sha256(request.expected_candidate_content_digest)
    ):
        return _candidate_refused(
            request,
            _candidate_issue(
                "candidate_manifest_internal_error",
                "candidate_manifest",
                "Expected candidate manifest is incomplete.",
                "expected_manifest_invalid",
            ),
        )
    paths = tuple(item.canonical_relative_path for item in request.expected_candidates)
    if len(paths) != len(set(paths)):
        return _candidate_refused(
            request,
            _candidate_issue(
                "candidate_duplicate_path",
                "candidate_manifest",
                "Expected candidate paths contain duplicates.",
                "expected_candidate_duplicate",
            ),
        )
    try:
        snapshot = build_rwd_source_candidate_snapshot(
            request.source_root_canonical,
            cancellation_check=cancellation_check,
        )
    except RwdSourceSnapshotError as exc:
        if exc.category in {"source_root_not_directory", "source_root_missing"}:
            category = "source_root_missing"
        elif exc.category in {"source_root_unreadable", "candidate_unreadable"}:
            category = "source_root_not_readable"
        elif exc.category == "no_rwd_fluorescence_files":
            category = "candidate_file_missing"
        else:
            category = "candidate_discovery_failed"
        return _candidate_refused(
            request,
            _candidate_issue(
                category,
                "source",
                "Candidate snapshot could not be built.",
                str(exc.category),
            ),
        )
    except Exception:
        return _candidate_refused(
            request,
            _candidate_issue(
                "candidate_manifest_internal_error",
                "candidate_manifest",
                "Candidate preflight could not complete.",
                "snapshot_exception",
            ),
        )
    actual = tuple(
        GuidedCandidateManifestEntry(
            item.canonical_relative_path,
            item.size_bytes,
            item.sha256_content_digest,
        )
        for item in snapshot.candidates
    )
    actual_paths = tuple(item.canonical_relative_path for item in actual)
    expected_set = set(paths)
    actual_set = set(actual_paths)
    if expected_set - actual_set:
        category, detail = "candidate_file_missing", "candidate_missing"
    elif actual_set - expected_set:
        category, detail = "candidate_file_extra", "candidate_extra"
    elif actual_paths != paths:
        category, detail = (
            "candidate_relative_path_mismatch",
            "candidate_order_mismatch",
        )
    else:
        category = detail = ""
    if category:
        return _candidate_refused(
            request,
            _candidate_issue(
                category,
                "candidate_manifest",
                "Actual candidate paths do not match the expected manifest.",
                detail,
            ),
            actual_candidates=actual,
            actual_set_digest=snapshot.source_candidate_set_digest,
            actual_content_digest=snapshot.source_candidate_content_digest,
        )
    for expected, observed in zip(request.expected_candidates, actual):
        if expected.size_bytes != observed.size_bytes:
            return _candidate_refused(
                request,
                _candidate_issue(
                    "candidate_file_size_mismatch",
                    "candidate_manifest",
                    "A candidate file size changed.",
                    "candidate_size_mismatch",
                ),
                actual_candidates=actual,
                actual_set_digest=snapshot.source_candidate_set_digest,
                actual_content_digest=snapshot.source_candidate_content_digest,
            )
        if expected.sha256_content_digest != observed.sha256_content_digest:
            return _candidate_refused(
                request,
                _candidate_issue(
                    "candidate_file_digest_mismatch",
                    "candidate_manifest",
                    "A candidate file content digest changed.",
                    "candidate_digest_mismatch",
                ),
                actual_candidates=actual,
                actual_set_digest=snapshot.source_candidate_set_digest,
                actual_content_digest=snapshot.source_candidate_content_digest,
            )
    if snapshot.source_candidate_set_digest != request.expected_candidate_set_digest:
        return _candidate_refused(
            request,
            _candidate_issue(
                "candidate_set_digest_mismatch",
                "candidate_manifest",
                "Candidate set digest changed.",
                "candidate_set_digest_mismatch",
            ),
            actual_candidates=actual,
            actual_set_digest=snapshot.source_candidate_set_digest,
            actual_content_digest=snapshot.source_candidate_content_digest,
        )
    if (
        snapshot.source_candidate_content_digest
        != request.expected_candidate_content_digest
    ):
        return _candidate_refused(
            request,
            _candidate_issue(
                "candidate_content_digest_mismatch",
                "candidate_manifest",
                "Candidate content digest changed.",
                "candidate_content_digest_mismatch",
            ),
            actual_candidates=actual,
            actual_set_digest=snapshot.source_candidate_set_digest,
            actual_content_digest=snapshot.source_candidate_content_digest,
        )
    if (
        snapshot.source_root_canonical != request.source_root_canonical
        or snapshot.source_root_path_style != request.source_root_path_style
        or snapshot.path_canonicalization_version
        != request.path_canonicalization_version
    ):
        return _candidate_refused(
            request,
            _candidate_issue(
                "source_path_canonicalization_mismatch",
                "source",
                "Canonical source root does not match.",
                "source_canonicalization_mismatch",
            ),
            actual_candidates=actual,
            actual_set_digest=snapshot.source_candidate_set_digest,
            actual_content_digest=snapshot.source_candidate_content_digest,
        )
    provisional = GuidedCandidateManifestExecutionPreflightResult(
        "accepted",
        True,
        request.contract_version,
        request.runner_contract_version,
        request.expected_candidate_set_digest,
        request.expected_candidate_content_digest,
        snapshot.source_candidate_set_digest,
        snapshot.source_candidate_content_digest,
        actual,
        (),
        "0" * 64,
    )
    return replace(
        provisional,
        canonical_preflight_identity=compute_guided_candidate_preflight_identity(
            provisional
        ),
    )


@dataclass(frozen=True)
class GuidedRoiParserPreflightContract:
    schema_name: str
    schema_version: str
    header_search_line_limit: int
    time_column_candidates: tuple[str, ...]
    uv_suffix_candidates: tuple[str, ...]
    signal_suffix_candidates: tuple[str, ...]
    column_normalization_rule: str
    roi_name_rule: str
    ambiguity_policy: str
    unresolved_inputs: tuple[str, ...]


@dataclass(frozen=True)
class GuidedRoiExecutionPreflightRequest:
    contract_version: str
    runner_contract_version: str
    accepted_candidate_preflight_identity: str
    source_root_canonical: str
    source_candidate_content_digest: str
    candidate_files: tuple[GuidedCandidateManifestEntry, ...]
    parser_contract: GuidedRoiParserPreflightContract
    parser_contract_digest: str
    expected_selected_time_column: str
    expected_uv_suffix: str
    expected_signal_suffix: str
    expected_discovered_roi_ids: tuple[str, ...]
    expected_included_roi_ids: tuple[str, ...]
    expected_excluded_roi_ids: tuple[str, ...]
    selection_mode: str
    expected_inventory_source_content_digest: str
    expected_strict_roi_inventory_digest: str
    roi_execution_contract_version: str


@dataclass(frozen=True)
class GuidedRoiExecutionPreflightIssue:
    category: str
    section: str
    message: str
    detail_code: str = ""

    def __post_init__(self) -> None:
        if self.category not in GUIDED_ROI_PREFLIGHT_REFUSAL_CATEGORY_SET:
            raise ValueError("Unsupported ROI preflight issue.")


@dataclass(frozen=True)
class GuidedRoiExecutionPreflightResult:
    status: str
    accepted: bool
    contract_version: str
    runner_contract_version: str
    accepted_candidate_preflight_identity: str
    source_candidate_content_digest: str
    parser_contract_digest: str
    expected_strict_roi_inventory_digest: str
    actual_strict_roi_inventory_digest: str | None
    actual_discovered_roi_ids: tuple[str, ...]
    actual_included_roi_ids: tuple[str, ...]
    actual_excluded_roi_ids: tuple[str, ...]
    blocking_issues: tuple[GuidedRoiExecutionPreflightIssue, ...]
    canonical_preflight_identity: str | None
    no_files_written: bool = True
    no_directories_created: bool = True
    no_artifacts_created: bool = True
    no_run_id_allocated: bool = True
    no_config_or_argv_generated: bool = True
    no_runner_invoked: bool = True

    def __post_init__(self) -> None:
        if not _no_side_effects(
            (
                self.no_files_written,
                self.no_directories_created,
                self.no_artifacts_created,
                self.no_run_id_allocated,
                self.no_config_or_argv_generated,
                self.no_runner_invoked,
            )
        ):
            raise ValueError("ROI preflight cannot report side effects.")
        if self.accepted:
            if (
                self.status != "accepted"
                or self.blocking_issues
                or not _sha256(self.actual_strict_roi_inventory_digest)
                or not _sha256(self.canonical_preflight_identity)
            ):
                raise ValueError("Accepted ROI preflight is incomplete.")
        elif (
            self.status != "refused"
            or not self.blocking_issues
            or self.canonical_preflight_identity is not None
        ):
            raise ValueError("Refused ROI preflight is inconsistent.")


def compute_guided_strict_roi_inventory_digest(
    *,
    source_candidate_content_digest: str,
    parser_contract_digest: str,
    discovered_roi_ids: tuple[str, ...],
    included_roi_ids: tuple[str, ...],
    excluded_roi_ids: tuple[str, ...],
    selection_mode: str,
) -> str:
    if not _sha256(source_candidate_content_digest) or not _sha256(
        parser_contract_digest
    ):
        raise ValueError("ROI inventory source identities must be SHA-256.")
    for name, values in (
        ("discovered_roi_ids", discovered_roi_ids),
        ("included_roi_ids", included_roi_ids),
        ("excluded_roi_ids", excluded_roi_ids),
    ):
        if (
            not isinstance(values, tuple)
            or any(not _text(value) for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"{name} must be a unique tuple of non-empty strings.")
    if not _text(selection_mode):
        raise ValueError("selection_mode is required.")
    payload = {
        "identity_domain": GUIDED_STRICT_ROI_INVENTORY_IDENTITY_DOMAIN,
        "inventory": {
            "source_candidate_content_digest": source_candidate_content_digest,
            "parser_contract_digest": parser_contract_digest,
            "discovered_roi_ids": list(discovered_roi_ids),
            "included_roi_ids": list(included_roi_ids),
            "excluded_roi_ids": list(excluded_roi_ids),
            "selection_mode": selection_mode,
        },
    }
    return hashlib.sha256(encode_canonical_value(payload)).hexdigest()


def derive_roi_execution_preflight_request_from_intent(
    intent: GuidedProductionExecutionIntent,
    *,
    accepted_candidate_preflight_identity: str,
) -> GuidedRoiExecutionPreflightRequest:
    if not isinstance(intent, GuidedProductionExecutionIntent):
        raise TypeError("intent must be a GuidedProductionExecutionIntent.")
    parser = intent.parser
    roi = intent.roi_scope
    source = intent.input_source
    strict_digest = compute_guided_strict_roi_inventory_digest(
        source_candidate_content_digest=source.source_candidate_content_digest,
        parser_contract_digest=parser.parser_contract_digest,
        discovered_roi_ids=roi.discovered_roi_ids,
        included_roi_ids=roi.included_roi_ids,
        excluded_roi_ids=roi.excluded_roi_ids,
        selection_mode=roi.selection_mode,
    )
    return GuidedRoiExecutionPreflightRequest(
        contract_version=roi.roi_execution_contract_version,
        runner_contract_version=intent.runner_contract_version,
        accepted_candidate_preflight_identity=accepted_candidate_preflight_identity,
        source_root_canonical=source.source_root_canonical,
        source_candidate_content_digest=source.source_candidate_content_digest,
        candidate_files=tuple(
            GuidedCandidateManifestEntry(
                item.canonical_relative_path,
                item.size_bytes,
                item.sha256_content_digest,
            )
            for item in source.candidate_files
        ),
        parser_contract=GuidedRoiParserPreflightContract(
            parser.schema_name,
            parser.schema_version,
            parser.header_search_line_limit,
            parser.time_column_candidates,
            parser.uv_suffix_candidates,
            parser.signal_suffix_candidates,
            parser.column_normalization_rule,
            parser.roi_name_rule,
            parser.ambiguity_policy,
            (),
        ),
        parser_contract_digest=parser.parser_contract_digest,
        expected_selected_time_column=intent.acquisition.rwd_time_col,
        expected_uv_suffix=intent.acquisition.uv_suffix,
        expected_signal_suffix=intent.acquisition.sig_suffix,
        expected_discovered_roi_ids=roi.discovered_roi_ids,
        expected_included_roi_ids=roi.included_roi_ids,
        expected_excluded_roi_ids=roi.excluded_roi_ids,
        selection_mode=roi.selection_mode,
        expected_inventory_source_content_digest=(
            roi.inventory_source_content_digest
        ),
        expected_strict_roi_inventory_digest=strict_digest,
        roi_execution_contract_version=roi.roi_execution_contract_version,
    )


def _roi_issue(
    category: str, section: str, message: str, detail_code: str
) -> GuidedRoiExecutionPreflightIssue:
    return GuidedRoiExecutionPreflightIssue(
        category, section, message, detail_code
    )


def _roi_refused(
    request: Any,
    issue: GuidedRoiExecutionPreflightIssue,
    *,
    discovered: tuple[str, ...] = (),
    included: tuple[str, ...] = (),
    excluded: tuple[str, ...] = (),
    actual_digest: str | None = None,
) -> GuidedRoiExecutionPreflightResult:
    return GuidedRoiExecutionPreflightResult(
        "refused",
        False,
        str(getattr(request, "contract_version", "") or ""),
        str(getattr(request, "runner_contract_version", "") or ""),
        str(getattr(request, "accepted_candidate_preflight_identity", "") or ""),
        str(getattr(request, "source_candidate_content_digest", "") or ""),
        str(getattr(request, "parser_contract_digest", "") or ""),
        str(getattr(request, "expected_strict_roi_inventory_digest", "") or ""),
        actual_digest,
        discovered,
        included,
        excluded,
        (issue,),
        None,
    )


def _roi_identity_payload(result: GuidedRoiExecutionPreflightResult) -> dict[str, Any]:
    return {
        "identity_domain": GUIDED_ROI_PREFLIGHT_IDENTITY_DOMAIN,
        "result": {
            "contract_version": result.contract_version,
            "candidate_preflight_identity": (
                result.accepted_candidate_preflight_identity
            ),
            "source_candidate_content_digest": (
                result.source_candidate_content_digest
            ),
            "parser_contract_digest": result.parser_contract_digest,
            "expected_strict_roi_inventory_digest": (
                result.expected_strict_roi_inventory_digest
            ),
            "actual_strict_roi_inventory_digest": (
                result.actual_strict_roi_inventory_digest
            ),
            "actual_discovered_roi_ids": list(result.actual_discovered_roi_ids),
            "actual_included_roi_ids": list(result.actual_included_roi_ids),
            "actual_excluded_roi_ids": list(result.actual_excluded_roi_ids),
            "accepted": result.accepted,
        },
    }


def compute_guided_roi_preflight_identity(
    result: GuidedRoiExecutionPreflightResult,
) -> str:
    if not isinstance(result, GuidedRoiExecutionPreflightResult) or not result.accepted:
        raise ValueError("Only an accepted ROI preflight has an identity.")
    return hashlib.sha256(
        encode_canonical_value(_roi_identity_payload(result))
    ).hexdigest()


def run_roi_execution_preflight(
    request: GuidedRoiExecutionPreflightRequest,
    *,
    cancellation_check: Callable[[], bool] | None = None,
) -> GuidedRoiExecutionPreflightResult:
    if not isinstance(request, GuidedRoiExecutionPreflightRequest):
        return _roi_refused(
            request,
            _roi_issue(
                "roi_preflight_contract_unavailable",
                "roi_preflight",
                "ROI preflight request is invalid.",
                "request_invalid_type",
            ),
        )
    if (
        request.contract_version != GUIDED_ROI_PREFLIGHT_CONTRACT_VERSION
        or request.roi_execution_contract_version
        != GUIDED_ROI_PREFLIGHT_CONTRACT_VERSION
        or not _version(request.runner_contract_version)
    ):
        return _roi_refused(
            request,
            _roi_issue(
                "roi_preflight_contract_unavailable",
                "roi_preflight",
                "ROI preflight contract is unsupported.",
                "contract_unsupported",
            ),
        )
    if not _sha256(request.accepted_candidate_preflight_identity):
        return _roi_refused(
            request,
            _roi_issue(
                "roi_source_digest_mismatch",
                "candidate_preflight",
                "Accepted candidate preflight identity is invalid.",
                "candidate_preflight_identity_invalid",
            ),
        )
    if (
        not _sha256(request.source_candidate_content_digest)
        or request.expected_inventory_source_content_digest
        != request.source_candidate_content_digest
    ):
        return _roi_refused(
            request,
            _roi_issue(
                "roi_source_digest_mismatch",
                "source",
                "ROI inventory is not bound to candidate content.",
                "source_content_digest_mismatch",
            ),
        )
    try:
        current_snapshot = build_rwd_source_candidate_snapshot(
            request.source_root_canonical,
            cancellation_check=cancellation_check,
        )
    except Exception:
        return _roi_refused(
            request,
            _roi_issue(
                "roi_source_digest_mismatch",
                "source",
                "Current candidate content could not be verified.",
                "candidate_snapshot_rebuild_failed",
            ),
        )
    current_candidates = tuple(
        GuidedCandidateManifestEntry(
            item.canonical_relative_path,
            item.size_bytes,
            item.sha256_content_digest,
        )
        for item in current_snapshot.candidates
    )
    if (
        current_candidates != request.candidate_files
        or current_snapshot.source_candidate_content_digest
        != request.source_candidate_content_digest
    ):
        return _roi_refused(
            request,
            _roi_issue(
                "roi_source_digest_mismatch",
                "source",
                "Current candidate files do not match the accepted manifest.",
                "candidate_content_binding_mismatch",
            ),
        )
    parser_value = request.parser_contract
    try:
        parser = RwdHeaderParsingContract(
            schema_name=parser_value.schema_name,
            schema_version=parser_value.schema_version,
            header_search_line_limit=parser_value.header_search_line_limit,
            time_column_candidates=parser_value.time_column_candidates,
            uv_suffix_candidates=parser_value.uv_suffix_candidates,
            signal_suffix_candidates=parser_value.signal_suffix_candidates,
            column_normalization_rule=parser_value.column_normalization_rule,
            roi_name_rule=parser_value.roi_name_rule,
            ambiguity_policy=parser_value.ambiguity_policy,
            unresolved_inputs=parser_value.unresolved_inputs,
        )
        actual_parser_digest = compute_rwd_header_parsing_contract_digest(parser)
    except Exception:
        return _roi_refused(
            request,
            _roi_issue(
                "parser_contract_unsupported",
                "parser",
                "ROI parser contract is unsupported.",
                "parser_contract_invalid",
            ),
        )
    if actual_parser_digest != request.parser_contract_digest:
        return _roi_refused(
            request,
            _roi_issue(
                "parser_digest_mismatch",
                "parser",
                "ROI parser digest does not match.",
                "parser_digest_mismatch",
            ),
        )
    if not request.candidate_files:
        return _roi_refused(
            request,
            _roi_issue(
                "roi_discovery_failed",
                "source",
                "ROI preflight requires candidate files.",
                "candidate_files_empty",
            ),
        )
    inspections = []
    for candidate in request.candidate_files:
        if cancellation_check is not None and cancellation_check():
            return _roi_refused(
                request,
                _roi_issue(
                    "roi_discovery_failed",
                    "roi_preflight",
                    "ROI preflight was cancelled.",
                    "roi_preflight_cancelled",
                ),
            )
        runtime_path = os.path.join(
            request.source_root_canonical,
            *PurePosixPath(candidate.canonical_relative_path).parts,
        )
        try:
            inspection = inspect_rwd_header_contract(
                runtime_path, parsing_contract=parser
            )
        except RwdHeaderInspectionError as exc:
            return _roi_refused(
                request,
                _roi_issue(
                    "roi_discovery_failed",
                    "parser",
                    "A candidate header could not be inspected.",
                    str(exc.category),
                ),
            )
        except Exception:
            return _roi_refused(
                request,
                _roi_issue(
                    "roi_preflight_internal_error",
                    "roi_preflight",
                    "ROI preflight could not complete.",
                    "header_inspection_exception",
                ),
            )
        if not inspection.acceptable_for_strict_identity:
            return _roi_refused(
                request,
                _roi_issue(
                    "roi_ambiguous",
                    "parser",
                    "A candidate header is ambiguous for strict ROI identity.",
                    "header_not_strict",
                ),
            )
        if inspection.selected_time_column != request.expected_selected_time_column:
            return _roi_refused(
                request,
                _roi_issue(
                    "roi_parser_column_mismatch",
                    "parser",
                    "Selected time column changed.",
                    "time_column_mismatch",
                ),
            )
        if any(
            pair.matched_uv_suffix != request.expected_uv_suffix
            or pair.matched_signal_suffix != request.expected_signal_suffix
            for pair in inspection.roi_channel_pairs
        ):
            return _roi_refused(
                request,
                _roi_issue(
                    "roi_parser_column_mismatch",
                    "parser",
                    "Selected ROI channel suffixes changed.",
                    "suffix_mismatch",
                ),
            )
        inspections.append(inspection)
    discovered = tuple(inspections[0].roi_ids)
    if any(tuple(item.roi_ids) != discovered for item in inspections[1:]):
        return _roi_refused(
            request,
            _roi_issue(
                "roi_tuple_mismatch",
                "roi_scope",
                "Candidate files do not share one ordered ROI tuple.",
                "candidate_roi_tuple_mismatch",
            ),
            discovered=discovered,
        )
    if len(discovered) != len(set(discovered)):
        return _roi_refused(
            request,
            _roi_issue(
                "roi_duplicate",
                "roi_scope",
                "Discovered ROI tuple contains duplicates.",
                "discovered_roi_duplicate",
            ),
            discovered=discovered,
        )
    missing_included = tuple(
        item
        for item in request.expected_included_roi_ids
        if item not in discovered
    )
    if missing_included:
        return _roi_refused(
            request,
            _roi_issue(
                "roi_missing_included",
                "roi_scope",
                "An included ROI is missing.",
                "included_roi_missing",
            ),
            discovered=discovered,
        )
    if set(discovered) - set(request.expected_discovered_roi_ids):
        return _roi_refused(
            request,
            _roi_issue(
                "roi_extra_analyzed",
                "roi_scope",
                "Runner-visible ROI inventory contains an unexpected ROI.",
                "extra_discovered_roi",
            ),
            discovered=discovered,
        )
    if discovered != request.expected_discovered_roi_ids:
        return _roi_refused(
            request,
            _roi_issue(
                "roi_tuple_mismatch",
                "roi_scope",
                "Discovered ROI tuple changed.",
                "discovered_roi_tuple_mismatch",
            ),
            discovered=discovered,
        )
    included_expected = request.expected_included_roi_ids
    excluded_expected = request.expected_excluded_roi_ids
    if (
        request.selection_mode != "include"
        or len(included_expected) != len(set(included_expected))
        or len(excluded_expected) != len(set(excluded_expected))
        or set(included_expected) & set(excluded_expected)
        or set(included_expected) | set(excluded_expected) != set(discovered)
    ):
        return _roi_refused(
            request,
            _roi_issue(
                "roi_include_tuple_mismatch",
                "roi_scope",
                "Expected ROI partition is invalid.",
                "expected_roi_partition_invalid",
            ),
            discovered=discovered,
        )
    actual_included = tuple(item for item in discovered if item in included_expected)
    actual_excluded = tuple(item for item in discovered if item not in actual_included)
    if actual_included != included_expected:
        return _roi_refused(
            request,
            _roi_issue(
                "roi_include_tuple_mismatch",
                "roi_scope",
                "Included ROI tuple order changed.",
                "included_roi_tuple_mismatch",
            ),
            discovered=discovered,
            included=actual_included,
            excluded=actual_excluded,
        )
    if actual_excluded != excluded_expected:
        return _roi_refused(
            request,
            _roi_issue(
                "roi_exclude_tuple_mismatch",
                "roi_scope",
                "Excluded ROI tuple order changed.",
                "excluded_roi_tuple_mismatch",
            ),
            discovered=discovered,
            included=actual_included,
            excluded=actual_excluded,
        )
    actual_digest = compute_guided_strict_roi_inventory_digest(
        source_candidate_content_digest=request.source_candidate_content_digest,
        parser_contract_digest=request.parser_contract_digest,
        discovered_roi_ids=discovered,
        included_roi_ids=actual_included,
        excluded_roi_ids=actual_excluded,
        selection_mode=request.selection_mode,
    )
    if actual_digest != request.expected_strict_roi_inventory_digest:
        return _roi_refused(
            request,
            _roi_issue(
                "roi_inventory_digest_mismatch",
                "roi_scope",
                "Strict ROI inventory digest changed.",
                "strict_roi_inventory_digest_mismatch",
            ),
            discovered=discovered,
            included=actual_included,
            excluded=actual_excluded,
            actual_digest=actual_digest,
        )
    provisional = GuidedRoiExecutionPreflightResult(
        "accepted",
        True,
        request.contract_version,
        request.runner_contract_version,
        request.accepted_candidate_preflight_identity,
        request.source_candidate_content_digest,
        request.parser_contract_digest,
        request.expected_strict_roi_inventory_digest,
        actual_digest,
        discovered,
        actual_included,
        actual_excluded,
        (),
        "0" * 64,
    )
    return replace(
        provisional,
        canonical_preflight_identity=compute_guided_roi_preflight_identity(
            provisional
        ),
    )


CANDIDATE_IDENTITY_FIELDS = (
    "contract_version",
    "runner_contract_version",
    "expected_candidate_set_digest",
    "expected_candidate_content_digest",
    "actual_candidate_set_digest",
    "actual_candidate_content_digest",
    "actual_candidates",
    "accepted",
)
ROI_IDENTITY_FIELDS = (
    "contract_version",
    "accepted_candidate_preflight_identity",
    "source_candidate_content_digest",
    "parser_contract_digest",
    "expected_strict_roi_inventory_digest",
    "actual_strict_roi_inventory_digest",
    "actual_discovered_roi_ids",
    "actual_included_roi_ids",
    "actual_excluded_roi_ids",
    "accepted",
)
CANDIDATE_INTENT_DERIVATION_FIELDS = frozenset(
    {"input_source", "parser", "runner_contract_version"}
)
ROI_INTENT_DERIVATION_FIELDS = frozenset(
    {"input_source", "parser", "roi_scope", "acquisition", "runner_contract_version"}
)
