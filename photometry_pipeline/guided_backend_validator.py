"""Pure, non-writing validation of compiled Guided backend requests."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from photometry_pipeline.guided_backend_validation_request import (
    GUIDED_BACKEND_DIAGNOSTIC_CACHE_SCHEMA_VERSION,
    GUIDED_BACKEND_FEATURE_EVENT_PROFILE_SCHEMA_VERSION,
    GUIDED_BACKEND_INCOMPLETE_FINAL_CLASSIFIER_VERSION,
    GUIDED_BACKEND_INCOMPLETE_FINAL_SCHEMA_NAME,
    GUIDED_BACKEND_INCOMPLETE_FINAL_SCHEMA_VERSION,
    GUIDED_BACKEND_LOCAL_CHECK_CONTRACT_VERSION,
    GUIDED_BACKEND_SOURCE_DISCOVERY_RULE_VERSION,
    GUIDED_BACKEND_SOURCE_IGNORED_FILES_POLICY,
    GUIDED_BACKEND_SOURCE_RELATIVE_PATH_RULE_VERSION,
    GUIDED_BACKEND_SOURCE_SNAPSHOT_SCHEMA_NAME,
    GUIDED_BACKEND_SOURCE_SNAPSHOT_SCHEMA_VERSION,
    GUIDED_BACKEND_VALIDATION_COMPILER_VERSION,
    GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION,
    GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME,
    GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION,
    GUIDED_BACKEND_VALIDATION_SCOPE,
    GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION,
    GuidedBackendValidationRequest,
    GuidedBackendValidatorContract,
    compute_guided_backend_validation_request_identity,
)
from photometry_pipeline.guided_identity import CANONICALIZATION_ALGORITHM_VERSION
from photometry_pipeline.guided_new_analysis_plan import (
    FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
)


GUIDED_BACKEND_VALIDATOR_REFUSAL_CATEGORIES = (
    "request_missing_or_invalid",
    "request_identity_missing",
    "request_identity_invalid",
    "request_identity_mismatch",
    "request_identity_computation_failed",
    "validator_contract_unavailable",
    "unsupported_request_schema",
    "unsupported_validation_scope",
    "unsupported_source_format",
    "unsupported_acquisition_mode",
    "unsupported_analysis_scope",
    "incomplete_final_policy_not_supported",
    "parser_contract_unavailable",
    "parser_unresolved_inputs",
    "missing_or_stale_dataset_contract",
    "missing_roi_inventory",
    "empty_included_roi_set",
    "duplicate_roi_id",
    "included_excluded_roi_conflict",
    "roi_selection_stale",
    "missing_confirmed_strategy_mark",
    "duplicate_confirmed_strategy_mark",
    "stale_strategy_mark",
    "mixed_dynamic_fit_modes",
    "signal_only_not_supported_for_validate",
    "forbidden_strategy_state",
    "diagnostic_cache_identity_mismatch",
    "diagnostic_cache_not_completed_run_ineligible",
    "missing_or_stale_diagnostic_cache",
    "evidence_reference_missing_or_stale",
    "local_preview_setup_signature_mismatch",
    "invalid_feature_event_profile",
    "feature_event_effective_value_unresolved",
    "missing_output_policy",
    "stale_output_policy",
    "unsafe_output_base",
    "overwrite_not_allowed",
    "protected_root_context_incomplete",
    "unsupported_request_field",
    "local_contract_not_clean",
    "validator_internal_error",
)
GUIDED_BACKEND_VALIDATOR_REFUSAL_CATEGORY_SET = frozenset(
    GUIDED_BACKEND_VALIDATOR_REFUSAL_CATEGORIES
)

GUIDED_BACKEND_VALIDATOR_CAPABILITY_VERSION = "guided_backend_validator.v1"
GUIDED_BACKEND_VALIDATOR_CAPABILITY_PLACEHOLDERS = frozenset(
    {"unknown", "placeholder", "unset", "none"}
)
GUIDED_BACKEND_VALIDATOR_ACCEPTED_OUTPUT_RELATIONSHIP_STATUSES = frozenset(
    {"safe"}
)
GUIDED_BACKEND_VALIDATOR_DEFERRED_CAPABILITIES = (
    "backend_validation",
    "run_authorization",
    "app_build_identity",
    "full_source_manifest_identity",
    "strict_roi_inventory_identity",
)
GUIDED_BACKEND_VALIDATOR_FILESYSTEM_FACT_SCOPE = (
    "read_only_path_relationships_no_writability_probe"
)
GUIDED_BACKEND_VALIDATOR_SOURCE_IDENTITY_LEVEL = (
    "content_bound_candidate_snapshot"
)
GUIDED_BACKEND_VALIDATOR_ROI_IDENTITY_STATUS = "deferred_not_authoritative"


def _is_scalar(value: Any) -> bool:
    return (
        value is None
        or isinstance(value, (str, bool, int))
        or (isinstance(value, float) and math.isfinite(value))
    )


def _is_sha256_lower(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_tuple(value: Any) -> bool:
    return isinstance(value, tuple)


@dataclass(frozen=True)
class GuidedBackendValidationDebugValue:
    key: str
    value: str | bool | int | float | None

    def __post_init__(self) -> None:
        if not _is_non_empty_string(self.key):
            raise ValueError("key must be a non-empty string.")
        if not _is_scalar(self.value):
            raise ValueError("debug value must be a canonical scalar.")


@dataclass(frozen=True)
class GuidedBackendValidationIssue:
    category: str
    section: str
    message: str
    detail_code: str = ""
    debug_context: tuple[GuidedBackendValidationDebugValue, ...] = ()

    def __post_init__(self) -> None:
        if self.category not in GUIDED_BACKEND_VALIDATOR_REFUSAL_CATEGORY_SET:
            raise ValueError(f"Unsupported validator category: {self.category}.")
        if not _is_non_empty_string(self.section):
            raise ValueError("section must be a non-empty string.")
        if not _is_non_empty_string(self.message):
            raise ValueError("message must be a non-empty string.")
        if not isinstance(self.detail_code, str):
            raise ValueError("detail_code must be a string.")
        if not _is_tuple(self.debug_context):
            raise ValueError("debug_context must be a tuple.")
        if not all(
            isinstance(item, GuidedBackendValidationDebugValue)
            for item in self.debug_context
        ):
            raise ValueError("debug_context contains an invalid value.")


@dataclass(frozen=True)
class GuidedBackendValidationResult:
    status: str
    request_identity: str | None
    accepted: bool
    blocking_issues: tuple[GuidedBackendValidationIssue, ...]
    warning_categories: tuple[str, ...]
    validator_contract_version: str
    validator_capability_version: str
    validated_request_scope: str
    no_files_written: bool = True
    no_directories_created: bool = True
    no_run_id_allocated: bool = True
    no_runner_invoked: bool = True
    no_artifacts_created: bool = True
    run_authorization: bool = False

    def __post_init__(self) -> None:
        if not _is_tuple(self.blocking_issues):
            raise ValueError("blocking_issues must be a tuple.")
        if not all(
            isinstance(item, GuidedBackendValidationIssue)
            for item in self.blocking_issues
        ):
            raise ValueError("blocking_issues contains an invalid issue.")
        if not _is_tuple(self.warning_categories) or not all(
            isinstance(item, str) for item in self.warning_categories
        ):
            raise ValueError("warning_categories must be a tuple of strings.")
        if any(
            value is not True
            for value in (
                self.no_files_written,
                self.no_directories_created,
                self.no_run_id_allocated,
                self.no_runner_invoked,
                self.no_artifacts_created,
            )
        ):
            raise ValueError("Validator results must assert no side effects.")
        if self.run_authorization is not False:
            raise ValueError("Validator results cannot authorize a run.")
        if self.accepted:
            if (
                self.status != "accepted"
                or not _is_sha256_lower(self.request_identity)
                or self.blocking_issues
                or self.warning_categories
            ):
                raise ValueError("Accepted validator result is inconsistent.")
        elif self.status != "refused" or not self.blocking_issues:
            raise ValueError("Refused validator result is inconsistent.")
        if self.status not in {"accepted", "refused"}:
            raise ValueError("status must be accepted or refused.")


def _issue(
    category: str,
    section: str,
    message: str,
    detail_code: str,
) -> GuidedBackendValidationIssue:
    return GuidedBackendValidationIssue(
        category=category,
        section=section,
        message=message,
        detail_code=detail_code,
    )


def _result_metadata(
    request: Any,
    validator_contract: Any,
) -> tuple[str, str, str]:
    contract_version = getattr(
        validator_contract, "validation_contract_version", ""
    )
    capability_version = getattr(
        validator_contract, "validator_capability_version", ""
    )
    request_scope = getattr(request, "validation_scope", "")
    return (
        contract_version if isinstance(contract_version, str) else "",
        capability_version if isinstance(capability_version, str) else "",
        request_scope if isinstance(request_scope, str) else "",
    )


def _refused(
    issue: GuidedBackendValidationIssue,
    *,
    request: Any,
    validator_contract: Any,
    request_identity: str | None,
) -> GuidedBackendValidationResult:
    contract_version, capability_version, request_scope = _result_metadata(
        request, validator_contract
    )
    return GuidedBackendValidationResult(
        status="refused",
        request_identity=request_identity,
        accepted=False,
        blocking_issues=(issue,),
        warning_categories=(),
        validator_contract_version=contract_version,
        validator_capability_version=capability_version,
        validated_request_scope=request_scope,
    )


def _accepted(
    request: GuidedBackendValidationRequest,
    validator_contract: GuidedBackendValidatorContract,
    request_identity: str,
) -> GuidedBackendValidationResult:
    return GuidedBackendValidationResult(
        status="accepted",
        request_identity=request_identity,
        accepted=True,
        blocking_issues=(),
        warning_categories=(),
        validator_contract_version=(
            validator_contract.validation_contract_version
        ),
        validator_capability_version=(
            validator_contract.validator_capability_version
        ),
        validated_request_scope=request.validation_scope,
    )


def _typed_values_valid(values: Any) -> bool:
    return (
        _is_tuple(values)
        and bool(values)
        and all(
            _is_non_empty_string(getattr(item, "field_name", None))
            and _is_non_empty_string(getattr(item, "value_type", None))
            and _is_non_empty_string(
                getattr(item, "source_classification", None)
            )
            and _is_scalar(getattr(item, "value", object()))
            for item in values
        )
    )


def _canonical_candidate_path(value: Any) -> bool:
    if not _is_non_empty_string(value):
        return False
    if value.startswith(("/", "\\")) or "\\" in value:
        return False
    return all(segment not in {"", ".", ".."} for segment in value.split("/"))


def _validate_semantics(
    request: GuidedBackendValidationRequest,
) -> GuidedBackendValidationIssue | None:
    if (
        request.request_schema_name
        != GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME
        or request.request_schema_version
        != GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION
    ):
        return _issue(
            "unsupported_request_schema",
            "request",
            "The request schema is unsupported.",
            "request_schema_mismatch",
        )
    if (
        request.validation_scope != GUIDED_BACKEND_VALIDATION_SCOPE
        or request.validation_contract_version
        != GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
    ):
        return _issue(
            "unsupported_validation_scope",
            "request",
            "The validation scope or contract is unsupported.",
            "validation_scope_or_contract_mismatch",
        )
    if request.compiler_version != GUIDED_BACKEND_VALIDATION_COMPILER_VERSION:
        return _issue(
            "unsupported_request_field",
            "request",
            "The request compiler version is unsupported.",
            "compiler_version_mismatch",
        )
    if (
        request.subset_rule_version
        != GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
    ):
        return _issue(
            "unsupported_analysis_scope",
            "request",
            "The requested validation subset is unsupported.",
            "subset_rule_version_mismatch",
        )
    if (
        request.canonicalization_algorithm_version
        != CANONICALIZATION_ALGORITHM_VERSION
    ):
        return _issue(
            "unsupported_request_field",
            "request",
            "The canonicalization algorithm is unsupported.",
            "canonicalization_algorithm_mismatch",
        )

    source = request.source
    if source.source_format != "rwd":
        return _issue(
            "unsupported_source_format",
            "source",
            "The first validator subset requires RWD source data.",
            "source_format_not_rwd",
        )
    if (
        not _is_non_empty_string(source.source_root_canonical)
        or source.source_root_path_style not in {
            "windows_drive",
            "windows_unc",
            "posix",
        }
        or source.snapshot_schema_name
        != GUIDED_BACKEND_SOURCE_SNAPSHOT_SCHEMA_NAME
        or source.snapshot_schema_version
        != GUIDED_BACKEND_SOURCE_SNAPSHOT_SCHEMA_VERSION
        or source.discovery_rule_version
        != GUIDED_BACKEND_SOURCE_DISCOVERY_RULE_VERSION
        or source.path_canonicalization_version
        != CANONICALIZATION_ALGORITHM_VERSION
        or source.relative_path_rule_version
        != GUIDED_BACKEND_SOURCE_RELATIVE_PATH_RULE_VERSION
        or source.ignored_files_policy
        != GUIDED_BACKEND_SOURCE_IGNORED_FILES_POLICY
        or source.build_mode != "read_only"
        or source.source_identity_level
        != GUIDED_BACKEND_VALIDATOR_SOURCE_IDENTITY_LEVEL
        or not _is_sha256_lower(source.source_candidate_set_digest)
        or not _is_sha256_lower(source.source_candidate_content_digest)
        or not _is_tuple(source.unresolved_source_identity_inputs)
        or source.unresolved_source_identity_inputs
    ):
        return _issue(
            "unsupported_request_field",
            "source",
            "Source snapshot contract fields are incomplete or unsupported.",
            "source_contract_invalid",
        )
    if not _is_tuple(source.candidate_files) or not source.candidate_files:
        return _issue(
            "missing_or_stale_dataset_contract",
            "source",
            "The source candidate inventory is missing.",
            "source_candidates_missing",
        )
    for candidate in source.candidate_files:
        if (
            not _canonical_candidate_path(
                getattr(candidate, "canonical_relative_path", None)
            )
            or isinstance(getattr(candidate, "size_bytes", None), bool)
            or not isinstance(getattr(candidate, "size_bytes", None), int)
            or candidate.size_bytes < 0
            or not _is_sha256_lower(
                getattr(candidate, "sha256_content_digest", None)
            )
        ):
            return _issue(
                "unsupported_request_field",
                "source",
                "A source candidate entry is invalid.",
                "source_candidate_invalid",
            )

    dataset = request.acquisition_dataset
    if dataset.acquisition_mode != "intermittent":
        return _issue(
            "unsupported_acquisition_mode",
            "acquisition_dataset",
            "The first validator subset requires intermittent acquisition.",
            "acquisition_mode_not_intermittent",
        )
    if (
        isinstance(dataset.sessions_per_hour, bool)
        or not isinstance(dataset.sessions_per_hour, int)
        or dataset.sessions_per_hour <= 0
        or isinstance(dataset.session_duration_sec, bool)
        or not isinstance(dataset.session_duration_sec, (int, float))
        or not math.isfinite(float(dataset.session_duration_sec))
        or dataset.session_duration_sec <= 0
        or dataset.timeline_anchor_mode != "civil"
        or dataset.fixed_daily_anchor_clock is not None
    ):
        return _issue(
            "missing_or_stale_dataset_contract",
            "acquisition_dataset",
            "Dataset timing or timeline fields are invalid.",
            "dataset_timing_invalid",
        )
    if (
        dataset.allow_partial_final_window is not False
        or dataset.exclude_incomplete_final_rwd_chunk is not False
        or dataset.classification_schema_name
        != GUIDED_BACKEND_INCOMPLETE_FINAL_SCHEMA_NAME
        or dataset.classification_schema_version
        != GUIDED_BACKEND_INCOMPLETE_FINAL_SCHEMA_VERSION
        or dataset.classifier_version
        != GUIDED_BACKEND_INCOMPLETE_FINAL_CLASSIFIER_VERSION
        or dataset.classification_status != "not_requested"
        or not _is_sha256_lower(
            dataset.not_requested_classification_digest
        )
    ):
        return _issue(
            "incomplete_final_policy_not_supported",
            "acquisition_dataset",
            "The incomplete-final policy is unsupported.",
            "incomplete_final_contract_invalid",
        )
    if (
        not _is_non_empty_string(dataset.dataset_snapshot_schema_version)
        or dataset.dataset_status != "applied"
        or dataset.dataset_current_applied is not True
        or not _is_non_empty_string(dataset.rwd_time_col)
        or not _is_non_empty_string(dataset.uv_suffix)
        or not _is_non_empty_string(dataset.sig_suffix)
        or not _typed_values_valid(dataset.semantic_values)
        or not _is_non_empty_string(
            dataset.dataset_source_setup_signature
        )
        or not _is_non_empty_string(
            dataset.diagnostic_cache_contract_identity
        )
        or not _is_tuple(dataset.validation_issue_categories)
        or dataset.validation_issue_categories
        or not _is_tuple(dataset.stale_reason_categories)
        or dataset.stale_reason_categories
    ):
        return _issue(
            "missing_or_stale_dataset_contract",
            "acquisition_dataset",
            "The applied dataset contract is incomplete or stale.",
            "dataset_contract_invalid",
        )

    parser = request.parser
    if (
        not _is_non_empty_string(parser.schema_name)
        or not _is_non_empty_string(parser.schema_version)
        or isinstance(parser.header_search_line_limit, bool)
        or not isinstance(parser.header_search_line_limit, int)
        or parser.header_search_line_limit <= 0
        or not _is_tuple(parser.time_column_candidates)
        or not parser.time_column_candidates
        or not _is_tuple(parser.uv_suffix_candidates)
        or not parser.uv_suffix_candidates
        or not _is_tuple(parser.signal_suffix_candidates)
        or not parser.signal_suffix_candidates
        or not _is_non_empty_string(parser.column_normalization_rule)
        or not _is_non_empty_string(parser.roi_name_rule)
        or not _is_non_empty_string(parser.ambiguity_policy)
        or not _is_sha256_lower(parser.parser_contract_digest)
    ):
        return _issue(
            "parser_contract_unavailable",
            "parser",
            "The parser contract is incomplete.",
            "parser_contract_invalid",
        )
    if not _is_tuple(parser.unresolved_inputs) or parser.unresolved_inputs:
        return _issue(
            "parser_unresolved_inputs",
            "parser",
            "The parser contract has unresolved inputs.",
            "parser_inputs_unresolved",
        )

    roi = request.roi_scope
    if (
        not _is_tuple(roi.discovered_roi_ids)
        or not roi.discovered_roi_ids
        or not _is_tuple(roi.included_roi_ids)
        or not _is_tuple(roi.excluded_roi_ids)
    ):
        return _issue(
            "missing_roi_inventory",
            "roi_scope",
            "The ROI inventory is missing.",
            "roi_inventory_missing",
        )
    if not roi.included_roi_ids:
        return _issue(
            "empty_included_roi_set",
            "roi_scope",
            "At least one ROI must be included.",
            "included_roi_ids_empty",
        )
    roi_groups = (
        roi.discovered_roi_ids,
        roi.included_roi_ids,
        roi.excluded_roi_ids,
    )
    if any(
        any(not _is_non_empty_string(value) for value in values)
        or len(values) != len(set(values))
        for values in roi_groups
    ):
        return _issue(
            "duplicate_roi_id",
            "roi_scope",
            "ROI identifiers must be non-empty and unique.",
            "roi_id_invalid_or_duplicate",
        )
    discovered = set(roi.discovered_roi_ids)
    included = set(roi.included_roi_ids)
    excluded = set(roi.excluded_roi_ids)
    if included & excluded or included | excluded != discovered:
        return _issue(
            "included_excluded_roi_conflict",
            "roi_scope",
            "Included and excluded ROIs must partition the inventory.",
            "roi_partition_invalid",
        )
    if (
        roi.selection_mode != "include"
        or not _is_non_empty_string(roi.inventory_status)
        or not _is_sha256_lower(roi.inventory_source_content_digest)
        or roi.roi_inventory_identity_status
        != GUIDED_BACKEND_VALIDATOR_ROI_IDENTITY_STATUS
    ):
        return _issue(
            "roi_selection_stale",
            "roi_scope",
            "ROI selection metadata is stale or unsupported.",
            "roi_selection_metadata_invalid",
        )

    correction = request.correction
    mode = correction.global_dynamic_fit_mode
    if mode == "signal_only_f0":
        return _issue(
            "signal_only_not_supported_for_validate",
            "correction",
            "Signal-Only is not supported by this validator subset.",
            "signal_only_mode",
        )
    if (
        correction.strategy_scope != "global"
        or correction.global_correction_strategy != "dynamic_fit"
        or mode not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES
    ):
        return _issue(
            "forbidden_strategy_state",
            "correction",
            "The correction strategy is unsupported.",
            "correction_strategy_forbidden",
        )
    if (
        not _typed_values_valid(correction.dynamic_fit_parameter_values)
        or not _is_tuple(correction.confirmed_marks)
        or not correction.confirmed_marks
    ):
        return _issue(
            "missing_confirmed_strategy_mark",
            "correction",
            "Dynamic-fit parameters and confirmed marks are required.",
            "correction_binding_missing",
        )
    mark_rois = [getattr(mark, "roi_id", "") for mark in correction.confirmed_marks]
    if len(mark_rois) != len(set(mark_rois)):
        return _issue(
            "duplicate_confirmed_strategy_mark",
            "correction",
            "Confirmed strategy marks contain duplicate ROI bindings.",
            "confirmed_mark_duplicate",
        )
    if set(mark_rois) != included:
        return _issue(
            "missing_confirmed_strategy_mark",
            "correction",
            "Every included ROI must have one confirmed strategy mark.",
            "confirmed_mark_coverage_mismatch",
        )
    if any(
        getattr(mark, "explicit_user_mark", None) is not True
        or getattr(mark, "current", None) is not True
        for mark in correction.confirmed_marks
    ):
        return _issue(
            "stale_strategy_mark",
            "correction",
            "Confirmed strategy marks must be explicit and current.",
            "confirmed_mark_not_current",
        )
    if {
        getattr(mark, "selected_dynamic_fit_mode", "")
        for mark in correction.confirmed_marks
    } != {mode}:
        return _issue(
            "mixed_dynamic_fit_modes",
            "correction",
            "Confirmed marks must unanimously match the global mode.",
            "confirmed_mark_mode_mismatch",
        )
    if (
        not _is_non_empty_string(correction.mark_rule_version)
        or not _is_non_empty_string(correction.currentness_rule_version)
        or not _is_non_empty_string(correction.unanimity_rule_version)
        or not _is_tuple(correction.blocked_strategy_states)
        or correction.blocked_strategy_states
    ):
        return _issue(
            "forbidden_strategy_state",
            "correction",
            "Correction rule state is unsupported.",
            "correction_rule_state_invalid",
        )

    diagnostic = request.diagnostic_evidence
    if diagnostic.available:
        if (
            not _is_non_empty_string(diagnostic.cache_id)
            or not _is_non_empty_string(diagnostic.cache_root_canonical)
            or not _is_non_empty_string(diagnostic.source_setup_signature)
            or not _is_non_empty_string(diagnostic.diagnostic_scope_signature)
            or not _is_non_empty_string(diagnostic.build_request_signature)
            or diagnostic.artifact_contract_version
            != GUIDED_BACKEND_DIAGNOSTIC_CACHE_SCHEMA_VERSION
            or diagnostic.provenance_schema_version
            != GUIDED_BACKEND_DIAGNOSTIC_CACHE_SCHEMA_VERSION
            or not _is_sha256_lower(diagnostic.artifact_semantic_digest)
            or not _is_sha256_lower(diagnostic.provenance_semantic_digest)
        ):
            return _issue(
                "diagnostic_cache_identity_mismatch",
                "diagnostic_evidence",
                "Diagnostic-cache identity is incomplete or unsupported.",
                "diagnostic_cache_identity_invalid",
            )
        if (
            diagnostic.completed_run_rejection_category
            != "guided_diagnostic_cache_ineligible"
        ):
            return _issue(
                "diagnostic_cache_not_completed_run_ineligible",
                "diagnostic_evidence",
                "The cache lacks the required completed-run rejection category.",
                "completed_run_rejection_mismatch",
            )
        if (
            diagnostic.resolver_status != "current"
            or diagnostic.preliminary_cache is not True
            or diagnostic.production_analysis is not False
            or not _is_tuple(diagnostic.stale_reasons)
            or diagnostic.stale_reasons
            or not _is_tuple(diagnostic.unresolved_inputs)
            or diagnostic.unresolved_inputs
        ):
            return _issue(
                "missing_or_stale_diagnostic_cache",
                "diagnostic_evidence",
                "The diagnostic cache is stale or not preliminary.",
                "diagnostic_cache_not_current",
            )
    else:
        if (
            _is_non_empty_string(diagnostic.cache_id)
            or _is_non_empty_string(diagnostic.cache_root_canonical)
            or diagnostic.preliminary_cache is not False
            or diagnostic.production_analysis is not False
        ):
            return _issue(
                "diagnostic_cache_identity_mismatch",
                "diagnostic_evidence",
                "Unavailable diagnostic evidence must not carry cache "
                "identity fields.",
                "unavailable_diagnostic_evidence_carries_cache_identity",
            )
    if (
        not _is_tuple(diagnostic.evidence_references)
        or not diagnostic.evidence_references
    ):
        return _issue(
            "evidence_reference_missing_or_stale",
            "diagnostic_evidence",
            "Evidence references are required.",
            "evidence_references_missing",
        )
    _evidence_identity_fields = (
        ("evidence_reference_id", "evidence_kind", "roi_id", "selected_dynamic_fit_mode")
        if not diagnostic.available
        else (
            "evidence_reference_id",
            "evidence_kind",
            "diagnostic_cache_id",
            "source_setup_signature",
            "diagnostic_scope_signature",
            "build_request_signature",
            "roi_id",
            "selected_dynamic_fit_mode",
        )
    )
    for reference in diagnostic.evidence_references:
        if getattr(reference, "current", None) is not True:
            return _issue(
                "evidence_reference_missing_or_stale",
                "diagnostic_evidence",
                "Every evidence reference must be current.",
                "evidence_reference_not_current",
            )
        if any(
            not _is_non_empty_string(getattr(reference, field_name, None))
            for field_name in _evidence_identity_fields
        ):
            return _issue(
                "evidence_reference_missing_or_stale",
                "diagnostic_evidence",
                "Evidence reference identity fields must be complete.",
                "evidence_reference_identity_incomplete",
            )
        if not diagnostic.available and getattr(
            reference, "evidence_kind", ""
        ) != "local_correction_preview":
            return _issue(
                "evidence_reference_missing_or_stale",
                "diagnostic_evidence",
                "Cache-free evidence must be local-preview evidence.",
                "evidence_kind_mismatch_for_unavailable_cache",
            )
        evidence_chunk = getattr(reference, "evidence_chunk", None)
        if (
            evidence_chunk is None
            or isinstance(evidence_chunk, bool)
            or not isinstance(evidence_chunk, int)
            or evidence_chunk < 0
        ):
            return _issue(
                "evidence_reference_missing_or_stale",
                "diagnostic_evidence",
                "Evidence reference chunk must be a non-negative integer.",
                "evidence_reference_chunk_invalid",
            )

    feature = request.feature_event
    if (
        feature.profile_schema_version
        != GUIDED_BACKEND_FEATURE_EVENT_PROFILE_SCHEMA_VERSION
        or not _is_non_empty_string(feature.profile_id)
        or not _typed_values_valid(feature.effective_values)
        or not _is_tuple(feature.active_fields)
        or not feature.active_fields
        or not _is_tuple(feature.inactive_fields)
        or feature.profile_status != "applied"
        or feature.explicitly_applied is not True
        or feature.current is not True
        or feature.visible_unapplied_changes is not False
        or not _is_tuple(feature.validation_issue_categories)
        or feature.validation_issue_categories
        or not _is_tuple(feature.stale_reason_categories)
        or feature.stale_reason_categories
    ):
        return _issue(
            "invalid_feature_event_profile",
            "feature_event",
            "The feature/event profile is incomplete, stale, or unsupported.",
            "feature_event_profile_invalid",
        )
    if any(
        value.source_classification == "unresolved"
        for value in feature.effective_values
    ):
        return _issue(
            "feature_event_effective_value_unresolved",
            "feature_event",
            "Feature/event values contain an unresolved value.",
            "feature_event_value_unresolved",
        )
    active_fields = tuple(feature.active_fields)
    inactive_fields = tuple(feature.inactive_fields)
    effective_fields = tuple(
        value.field_name for value in feature.effective_values
    )
    if (
        len(active_fields) != len(set(active_fields))
        or len(inactive_fields) != len(set(inactive_fields))
        or set(active_fields) & set(inactive_fields)
        or set(active_fields) | set(inactive_fields) != set(effective_fields)
        or len(effective_fields) != len(set(effective_fields))
    ):
        return _issue(
            "invalid_feature_event_profile",
            "feature_event",
            "Feature/event activity does not cover effective values exactly.",
            "feature_event_activity_invalid",
        )

    output = request.output
    if (
        not _is_non_empty_string(output.output_base_canonical)
        or not _is_non_empty_string(output.output_base_path_style)
    ):
        return _issue(
            "missing_output_policy",
            "output",
            "The output base is missing.",
            "output_base_missing",
        )
    if output.policy_status != "applied" or output.policy_current is not True:
        return _issue(
            "stale_output_policy",
            "output",
            "The output policy is stale or unapplied.",
            "output_policy_not_current",
        )
    if output.overwrite is not False:
        return _issue(
            "overwrite_not_allowed",
            "output",
            "Overwrite is not allowed.",
            "output_overwrite_true",
        )
    if output.protected_root_context_complete is not True:
        return _issue(
            "protected_root_context_incomplete",
            "output",
            "Protected-root context is incomplete.",
            "protected_root_context_incomplete",
        )
    if (
        output.path_role != "output_base"
        or output.future_output_owner != "runner"
        or output.run_directory_strategy
        != "derive_unique_run_id_under_output_base"
        or output.creation_timing != "future_execution_start_only"
        or output.precreate is not False
        or not _is_non_empty_string(output.safety_classifier_version)
        or output.filesystem_fact_scope
        != GUIDED_BACKEND_VALIDATOR_FILESYSTEM_FACT_SCOPE
    ):
        return _issue(
            "unsupported_request_field",
            "output",
            "Output ownership or filesystem-fact scope is unsupported.",
            "output_contract_unsupported",
        )
    if (
        not _is_tuple(output.blocker_categories)
        or output.blocker_categories
        or not _is_tuple(output.relationships)
        or any(
            getattr(relationship, "status", None)
            not in GUIDED_BACKEND_VALIDATOR_ACCEPTED_OUTPUT_RELATIONSHIP_STATUSES
            for relationship in output.relationships
        )
    ):
        return _issue(
            "unsafe_output_base",
            "output",
            "Output safety relationships contain a blocker or unsafe status.",
            "output_safety_invalid",
        )

    local = request.local_contract
    if (
        local.local_check_contract_version
        != GUIDED_BACKEND_LOCAL_CHECK_CONTRACT_VERSION
        or not _is_tuple(local.blocking_issue_categories)
        or local.blocking_issue_categories
        or not _is_tuple(local.warning_categories)
        or local.warning_categories
        or not _is_tuple(local.unsupported_state_flags)
        or local.unsupported_state_flags
        or not _is_tuple(local.unresolved_required_inputs)
        or local.unresolved_required_inputs
        or local.deferred_capabilities
        != GUIDED_BACKEND_VALIDATOR_DEFERRED_CAPABILITIES
    ):
        return _issue(
            "local_contract_not_clean",
            "local_contract",
            "The local contract is not clean for backend validation.",
            "local_contract_invalid",
        )

    if (
        roi.inventory_source_content_digest
        != source.source_candidate_content_digest
    ):
        return _issue(
            "roi_selection_stale",
            "cross_section",
            "ROI inventory identity does not match source content.",
            "roi_source_content_digest_mismatch",
        )
    if diagnostic.available and (
        dataset.dataset_source_setup_signature
        != diagnostic.source_setup_signature
        or dataset.diagnostic_cache_contract_identity
        != diagnostic.build_request_signature
    ):
        return _issue(
            "diagnostic_cache_identity_mismatch",
            "cross_section",
            "Dataset and diagnostic-cache identities disagree.",
            "dataset_cache_identity_mismatch",
        )

    evidence_by_roi: dict[str, Any] = {}
    for reference in diagnostic.evidence_references:
        roi_id = getattr(reference, "roi_id", "")
        if not _is_non_empty_string(roi_id) or roi_id in evidence_by_roi:
            return _issue(
                "evidence_reference_missing_or_stale",
                "cross_section",
                "Evidence ROI bindings must be non-empty and unique.",
                "evidence_roi_duplicate_or_missing",
            )
        evidence_by_roi[roi_id] = reference
    if set(evidence_by_roi) != included:
        return _issue(
            "evidence_reference_missing_or_stale",
            "cross_section",
            "Evidence references must cover every included ROI.",
            "evidence_roi_coverage_mismatch",
        )

    marks_by_roi = {mark.roi_id: mark for mark in correction.confirmed_marks}
    for roi_id in roi.included_roi_ids:
        mark = marks_by_roi[roi_id]
        reference = evidence_by_roi[roi_id]
        if (
            mark.selected_dynamic_fit_mode != mode
            or reference.selected_dynamic_fit_mode != mode
        ):
            return _issue(
                "mixed_dynamic_fit_modes",
                "cross_section",
                "Correction marks and evidence disagree on selected mode.",
                "mark_evidence_mode_mismatch",
            )
        if diagnostic.available:
            if (
                mark.diagnostic_cache_id != diagnostic.cache_id
                or reference.diagnostic_cache_id != diagnostic.cache_id
                or mark.source_setup_signature
                != diagnostic.source_setup_signature
                or reference.source_setup_signature
                != diagnostic.source_setup_signature
                or mark.diagnostic_scope_signature
                != diagnostic.diagnostic_scope_signature
                or reference.diagnostic_scope_signature
                != diagnostic.diagnostic_scope_signature
                or mark.build_request_signature
                != diagnostic.build_request_signature
                or reference.build_request_signature
                != diagnostic.build_request_signature
            ):
                return _issue(
                    "diagnostic_cache_identity_mismatch",
                    "cross_section",
                    "Correction marks or evidence do not bind to the cache.",
                    "mark_evidence_cache_binding_mismatch",
                )
        else:
            if (
                mark.diagnostic_cache_id
                or reference.diagnostic_cache_id
                or not mark.source_setup_signature
                or mark.source_setup_signature
                != reference.source_setup_signature
            ):
                return _issue(
                    "local_preview_setup_signature_mismatch",
                    "cross_section",
                    "Correction marks or evidence do not bind to a "
                    "consistent local-preview source/setup signature.",
                    "mark_evidence_local_preview_binding_mismatch",
                )
        if (
            mark.evidence_reference_id != reference.evidence_reference_id
            or mark.evidence_chunk != reference.evidence_chunk
        ):
            return _issue(
                "evidence_reference_missing_or_stale",
                "cross_section",
                "Correction mark and evidence reference identity disagree.",
                "mark_evidence_reference_mismatch",
            )

    # The request does not carry the source set/content digests used by the
    # incomplete-final classifier. That binding was checked by the compiler
    # before canonical request identity was computed; this validator does not
    # reread mutable facts to reconstruct it.
    return None


def validate_guided_backend_validation_request(
    request: GuidedBackendValidationRequest,
    *,
    canonical_request_identity: str,
    validator_contract: GuidedBackendValidatorContract,
) -> GuidedBackendValidationResult:
    """Validate one identified request without I/O or execution authority."""
    if not isinstance(request, GuidedBackendValidationRequest):
        return _refused(
            _issue(
                "request_missing_or_invalid",
                "request",
                "A compiled Guided backend validation request is required.",
                "request_missing_or_invalid_type",
            ),
            request=request,
            validator_contract=validator_contract,
            request_identity=None,
        )
    if not isinstance(canonical_request_identity, str) or not canonical_request_identity:
        return _refused(
            _issue(
                "request_identity_missing",
                "identity",
                "Canonical request identity is required.",
                "request_identity_missing",
            ),
            request=request,
            validator_contract=validator_contract,
            request_identity=None,
        )
    if not _is_sha256_lower(canonical_request_identity):
        return _refused(
            _issue(
                "request_identity_invalid",
                "identity",
                "Canonical request identity must be a lowercase SHA-256 digest.",
                "request_identity_invalid",
            ),
            request=request,
            validator_contract=validator_contract,
            request_identity=None,
        )
    try:
        recomputed_identity = (
            compute_guided_backend_validation_request_identity(request)
        )
    except Exception:
        return _refused(
            _issue(
                "request_identity_computation_failed",
                "identity",
                "Canonical request identity could not be recomputed.",
                "request_identity_computation_failed",
            ),
            request=request,
            validator_contract=validator_contract,
            request_identity=None,
        )
    if recomputed_identity != canonical_request_identity:
        return _refused(
            _issue(
                "request_identity_mismatch",
                "identity",
                "Supplied request identity does not match request content.",
                "request_identity_mismatch",
            ),
            request=request,
            validator_contract=validator_contract,
            request_identity=None,
        )

    if not isinstance(validator_contract, GuidedBackendValidatorContract):
        return _refused(
            _issue(
                "validator_contract_unavailable",
                "validator_contract",
                "A supported validator contract is required.",
                "validator_contract_missing_or_invalid_type",
            ),
            request=request,
            validator_contract=validator_contract,
            request_identity=recomputed_identity,
        )
    if (
        validator_contract.validation_scope != request.validation_scope
        or validator_contract.validation_scope
        != GUIDED_BACKEND_VALIDATION_SCOPE
    ):
        detail_code = "validation_scope_mismatch"
    elif (
        validator_contract.validation_contract_version
        != request.validation_contract_version
        or validator_contract.validation_contract_version
        != GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
    ):
        detail_code = "validation_contract_version_mismatch"
    elif (
        validator_contract.supported_subset_rule_version
        != request.subset_rule_version
        or validator_contract.supported_subset_rule_version
        != GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
    ):
        detail_code = "supported_subset_rule_version_mismatch"
    elif (
        not _is_non_empty_string(
            validator_contract.validator_capability_version
        )
        or validator_contract.validator_capability_version.strip().lower()
        in GUIDED_BACKEND_VALIDATOR_CAPABILITY_PLACEHOLDERS
    ):
        detail_code = "validator_capability_version_invalid"
    elif (
        validator_contract.validator_capability_version
        != request.validator_capability_version
    ):
        detail_code = "validator_capability_version_mismatch"
    else:
        detail_code = ""
    if detail_code:
        return _refused(
            _issue(
                "validator_contract_unavailable",
                "validator_contract",
                "Validator contract is unavailable or incompatible.",
                detail_code,
            ),
            request=request,
            validator_contract=validator_contract,
            request_identity=recomputed_identity,
        )

    try:
        semantic_issue = _validate_semantics(request)
    except Exception:
        semantic_issue = _issue(
            "validator_internal_error",
            "validator",
            "Validator could not complete semantic request checks.",
            "validator_semantic_check_failed",
        )
    if semantic_issue is not None:
        return _refused(
            semantic_issue,
            request=request,
            validator_contract=validator_contract,
            request_identity=recomputed_identity,
        )
    return _accepted(request, validator_contract, recomputed_identity)
