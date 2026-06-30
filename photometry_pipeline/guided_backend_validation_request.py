"""Pure contracts for a future exact Guided backend validation request.

This Stage 1 module defines immutable request, materialized-fact, validator,
success, and refusal models. It performs no filesystem access, materialization,
backend validation, execution, or artifact generation. The compiler currently
refuses every path before request construction because read-only materialized
facts are not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from photometry_pipeline.guided_identity import CANONICALIZATION_ALGORITHM_VERSION
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan


GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME = (
    "guided_backend_validation_request"
)
GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION = "v1"
GUIDED_BACKEND_VALIDATION_SCOPE = (
    "guided_rwd_intermittent_phasic_full_validate"
)
GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION = (
    "guided_backend_validation_contract.v1"
)
GUIDED_BACKEND_VALIDATION_COMPILER_VERSION = (
    "guided_backend_validation_request_compiler.v1"
)
GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION = "global_dynamic_fit_only.v1"
GUIDED_BACKEND_VALIDATION_IDENTITY_DOMAIN = (
    "guided-backend-validation-request:v1"
)


SOURCE_DATASET_REFUSAL_CATEGORIES = (
    "missing_source",
    "unsupported_source_format",
    "unsupported_acquisition_mode",
    "source_snapshot_unavailable",
    "source_snapshot_stale",
    "source_snapshot_digest_mismatch",
    "unsupported_incomplete_final_exclusion",
    "incomplete_final_classification_mismatch",
    "missing_or_stale_dataset_contract",
    "dataset_source_binding_mismatch",
    "invalid_sessions_per_hour",
    "invalid_session_duration",
)
ROI_REFUSAL_CATEGORIES = (
    "missing_roi_inventory",
    "empty_included_roi_set",
    "roi_selection_stale",
    "included_roi_not_discovered",
    "included_excluded_roi_conflict",
    "duplicate_roi_id",
)
CORRECTION_EVIDENCE_REFUSAL_CATEGORIES = (
    "missing_confirmed_strategy_mark",
    "duplicate_confirmed_strategy_mark",
    "stale_strategy_mark",
    "non_explicit_strategy_mark",
    "mixed_dynamic_fit_modes",
    "signal_only_not_supported_for_validate",
    "forbidden_strategy_state",
    "global_intent_confirmed_marks_mismatch",
    "dynamic_fit_parameter_contract_mismatch",
    "unresolved_dynamic_fit_parameter",
    "missing_or_stale_diagnostic_cache",
    "diagnostic_cache_not_completed_run_ineligible",
    "diagnostic_cache_identity_mismatch",
    "evidence_reference_missing_or_stale",
)
FEATURE_EVENT_REFUSAL_CATEGORIES = (
    "missing_feature_event_profile",
    "invalid_feature_event_profile",
    "stale_feature_event_profile",
    "unapplied_feature_event_changes",
    "unresolved_feature_event_effective_value",
)
OUTPUT_REFUSAL_CATEGORIES = (
    "missing_output_policy",
    "stale_output_policy",
    "unapplied_output_policy_changes",
    "unsafe_output_base",
    "overwrite_not_allowed",
    "output_overlaps_source",
    "output_overlaps_completed_run",
    "output_overlaps_diagnostic_cache",
    "protected_root_context_incomplete",
)
IDENTITY_SUPPORT_REFUSAL_CATEGORIES = (
    "parser_contract_missing",
    "parser_digest_unavailable",
    "parser_digest_mismatch",
    "unresolved_required_identity_input",
    "unsupported_first_subset_state",
    "compiler_contract_unavailable",
    "validator_contract_unavailable",
)
GUIDED_BACKEND_VALIDATION_REFUSAL_CATEGORIES = (
    SOURCE_DATASET_REFUSAL_CATEGORIES
    + ROI_REFUSAL_CATEGORIES
    + CORRECTION_EVIDENCE_REFUSAL_CATEGORIES
    + FEATURE_EVENT_REFUSAL_CATEGORIES
    + OUTPUT_REFUSAL_CATEGORIES
    + IDENTITY_SUPPORT_REFUSAL_CATEGORIES
)
GUIDED_BACKEND_VALIDATION_REFUSAL_CATEGORY_SET = frozenset(
    GUIDED_BACKEND_VALIDATION_REFUSAL_CATEGORIES
)

PROHIBITED_REQUEST_FIELD_NAMES = frozenset(
    {
        "run_id",
        "production_run_id",
        "run_dir",
        "output_run_dir",
        "config_path",
        "argv",
        "command_text",
        "status_path",
        "run_report_path",
        "manifest_path",
        "production_artifact_path",
        "completed_run_metadata",
        "gui_label",
        "display_text",
        "widget_state",
    }
)


class GuidedBackendValidationRequestContractError(ValueError):
    """Raised when an immutable Stage 1 contract violates its invariants."""


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GuidedBackendValidationRequestContractError(
            f"{field_name} must be a non-empty string."
        )


def _require_tuple(value: Any, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise GuidedBackendValidationRequestContractError(
            f"{field_name} must be a tuple."
        )


def _require_sha256(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise GuidedBackendValidationRequestContractError(
            f"{field_name} must be a lowercase SHA-256 digest."
        )


def _require_scalar(value: Any, field_name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise GuidedBackendValidationRequestContractError(
        f"{field_name} must contain a canonical scalar value."
    )


@dataclass(frozen=True)
class GuidedBackendTypedFieldValue:
    field_name: str
    value_type: str
    value: str | bool | int | float | None
    source_classification: str = "explicit"

    def __post_init__(self) -> None:
        _require_non_empty(self.field_name, "field_name")
        _require_non_empty(self.value_type, "value_type")
        _require_non_empty(self.source_classification, "source_classification")
        _require_scalar(self.value, self.field_name)


@dataclass(frozen=True)
class GuidedBackendSourceCandidateFile:
    canonical_relative_path: str
    size_bytes: int
    sha256_content_digest: str

    def __post_init__(self) -> None:
        _require_non_empty(self.canonical_relative_path, "canonical_relative_path")
        if (
            self.canonical_relative_path.startswith(("/", "\\"))
            or "\\" in self.canonical_relative_path
            or any(
                segment in {"", ".", ".."}
                for segment in self.canonical_relative_path.split("/")
            )
        ):
            raise GuidedBackendValidationRequestContractError(
                "canonical_relative_path must be a canonical relative path."
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise GuidedBackendValidationRequestContractError(
                "size_bytes must be a non-negative integer."
            )
        _require_sha256(self.sha256_content_digest, "sha256_content_digest")


@dataclass(frozen=True)
class GuidedBackendSourceRequest:
    source_root_canonical: str
    source_root_path_style: str
    source_format: str
    snapshot_schema_name: str
    snapshot_schema_version: str
    discovery_rule_version: str
    path_canonicalization_version: str
    relative_path_rule_version: str
    ignored_files_policy: str
    build_mode: str
    source_candidate_set_digest: str
    source_candidate_content_digest: str
    candidate_files: tuple[GuidedBackendSourceCandidateFile, ...]
    unresolved_source_identity_inputs: tuple[str, ...] = ()
    source_identity_level: str = "content_bound_candidate_snapshot"

    def __post_init__(self) -> None:
        for name in (
            "source_root_canonical",
            "source_root_path_style",
            "snapshot_schema_name",
            "snapshot_schema_version",
            "discovery_rule_version",
            "path_canonicalization_version",
            "relative_path_rule_version",
            "ignored_files_policy",
            "build_mode",
            "source_identity_level",
        ):
            _require_non_empty(getattr(self, name), name)
        if self.source_format != "rwd":
            raise GuidedBackendValidationRequestContractError(
                "source_format must be rwd."
            )
        _require_sha256(
            self.source_candidate_set_digest,
            "source_candidate_set_digest",
        )
        _require_sha256(
            self.source_candidate_content_digest,
            "source_candidate_content_digest",
        )
        _require_tuple(self.candidate_files, "candidate_files")
        if not self.candidate_files:
            raise GuidedBackendValidationRequestContractError(
                "candidate_files must not be empty."
            )
        _require_tuple(
            self.unresolved_source_identity_inputs,
            "unresolved_source_identity_inputs",
        )
        if self.unresolved_source_identity_inputs:
            raise GuidedBackendValidationRequestContractError(
                "Source request cannot contain unresolved identity inputs."
            )


@dataclass(frozen=True)
class GuidedBackendAcquisitionDatasetRequest:
    acquisition_mode: str
    sessions_per_hour: int
    session_duration_sec: float
    timeline_anchor_mode: str
    fixed_daily_anchor_clock: None
    allow_partial_final_window: bool
    exclude_incomplete_final_rwd_chunk: bool
    classification_schema_name: str
    classification_schema_version: str
    classifier_version: str
    classification_status: str
    not_requested_classification_digest: str
    dataset_snapshot_schema_version: str
    dataset_status: str
    dataset_current_applied: bool
    rwd_time_col: str
    uv_suffix: str
    sig_suffix: str
    semantic_values: tuple[GuidedBackendTypedFieldValue, ...] = ()
    dataset_source_setup_signature: str = ""
    diagnostic_cache_contract_identity: str = ""
    validation_issue_categories: tuple[str, ...] = ()
    stale_reason_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.acquisition_mode != "intermittent":
            raise GuidedBackendValidationRequestContractError(
                "acquisition_mode must be intermittent."
            )
        if (
            isinstance(self.sessions_per_hour, bool)
            or not isinstance(self.sessions_per_hour, int)
            or self.sessions_per_hour <= 0
        ):
            raise GuidedBackendValidationRequestContractError(
                "sessions_per_hour must be a positive integer."
            )
        if (
            isinstance(self.session_duration_sec, bool)
            or not isinstance(self.session_duration_sec, (int, float))
            or not math.isfinite(float(self.session_duration_sec))
            or self.session_duration_sec <= 0
        ):
            raise GuidedBackendValidationRequestContractError(
                "session_duration_sec must be positive and finite."
            )
        if self.timeline_anchor_mode != "civil":
            raise GuidedBackendValidationRequestContractError(
                "timeline_anchor_mode must be civil."
            )
        if (
            self.fixed_daily_anchor_clock is not None
            or self.allow_partial_final_window is not False
            or self.exclude_incomplete_final_rwd_chunk is not False
        ):
            raise GuidedBackendValidationRequestContractError(
                "Unsupported first-subset acquisition policy."
            )
        if self.classification_status != "not_requested":
            raise GuidedBackendValidationRequestContractError(
                "Incomplete-final classification must be not_requested."
            )
        if self.dataset_status != "applied" or self.dataset_current_applied is not True:
            raise GuidedBackendValidationRequestContractError(
                "Dataset contract must be currently applied."
            )
        for name in (
            "classification_schema_name",
            "classification_schema_version",
            "classifier_version",
            "dataset_snapshot_schema_version",
            "rwd_time_col",
            "uv_suffix",
            "sig_suffix",
            "dataset_source_setup_signature",
            "diagnostic_cache_contract_identity",
        ):
            _require_non_empty(getattr(self, name), name)
        _require_sha256(
            self.not_requested_classification_digest,
            "not_requested_classification_digest",
        )
        for name in (
            "semantic_values",
            "validation_issue_categories",
            "stale_reason_categories",
        ):
            _require_tuple(getattr(self, name), name)
        if self.validation_issue_categories or self.stale_reason_categories:
            raise GuidedBackendValidationRequestContractError(
                "Dataset request cannot contain validation or stale issues."
            )


@dataclass(frozen=True)
class GuidedBackendRwdParserRequest:
    schema_name: str
    schema_version: str
    header_search_line_limit: int
    time_column_candidates: tuple[str, ...]
    uv_suffix_candidates: tuple[str, ...]
    signal_suffix_candidates: tuple[str, ...]
    column_normalization_rule: str
    roi_name_rule: str
    ambiguity_policy: str
    parser_contract_digest: str
    unresolved_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "schema_name",
            "schema_version",
            "column_normalization_rule",
            "roi_name_rule",
            "ambiguity_policy",
        ):
            _require_non_empty(getattr(self, name), name)
        if (
            isinstance(self.header_search_line_limit, bool)
            or not isinstance(self.header_search_line_limit, int)
            or self.header_search_line_limit <= 0
        ):
            raise GuidedBackendValidationRequestContractError(
                "header_search_line_limit must be a positive integer."
            )
        for name in (
            "time_column_candidates",
            "uv_suffix_candidates",
            "signal_suffix_candidates",
            "unresolved_inputs",
        ):
            _require_tuple(getattr(self, name), name)
        if (
            not self.time_column_candidates
            or not self.uv_suffix_candidates
            or not self.signal_suffix_candidates
            or self.unresolved_inputs
        ):
            raise GuidedBackendValidationRequestContractError(
                "Parser candidates must be complete and resolved."
            )
        _require_sha256(self.parser_contract_digest, "parser_contract_digest")


@dataclass(frozen=True)
class GuidedBackendRoiScopeRequest:
    discovered_roi_ids: tuple[str, ...]
    included_roi_ids: tuple[str, ...]
    excluded_roi_ids: tuple[str, ...]
    selection_mode: str = "include"
    inventory_status: str = "plan_inventory_current_for_snapshot"
    inventory_source_content_digest: str = ""
    roi_inventory_identity_status: str = "deferred_not_authoritative"

    def __post_init__(self) -> None:
        for name in (
            "discovered_roi_ids",
            "included_roi_ids",
            "excluded_roi_ids",
        ):
            values = getattr(self, name)
            _require_tuple(values, name)
            if any(not isinstance(value, str) or not value for value in values):
                raise GuidedBackendValidationRequestContractError(
                    f"{name} must contain non-empty strings."
                )
            if len(values) != len(set(values)):
                raise GuidedBackendValidationRequestContractError(
                    f"{name} must not contain duplicates."
                )
        if not self.discovered_roi_ids or not self.included_roi_ids:
            raise GuidedBackendValidationRequestContractError(
                "Discovered and included ROI sets must not be empty."
            )
        included = set(self.included_roi_ids)
        excluded = set(self.excluded_roi_ids)
        discovered = set(self.discovered_roi_ids)
        if included & excluded:
            raise GuidedBackendValidationRequestContractError(
                "Included and excluded ROI sets must be disjoint."
            )
        if included | excluded != discovered:
            raise GuidedBackendValidationRequestContractError(
                "Included and excluded ROI sets must partition discovered ROIs."
            )
        if self.selection_mode != "include":
            raise GuidedBackendValidationRequestContractError(
                "selection_mode must be include."
            )
        _require_non_empty(self.inventory_status, "inventory_status")
        _require_non_empty(
            self.roi_inventory_identity_status,
            "roi_inventory_identity_status",
        )
        _require_sha256(
            self.inventory_source_content_digest,
            "inventory_source_content_digest",
        )


@dataclass(frozen=True)
class GuidedBackendConfirmedStrategyMark:
    roi_id: str
    selected_dynamic_fit_mode: str
    diagnostic_cache_id: str
    source_setup_signature: str
    diagnostic_scope_signature: str
    build_request_signature: str
    evidence_reference_id: str
    evidence_chunk: int | None = None
    explicit_user_mark: bool = True
    current: bool = True

    def __post_init__(self) -> None:
        for name in (
            "roi_id",
            "selected_dynamic_fit_mode",
            "diagnostic_cache_id",
            "source_setup_signature",
            "diagnostic_scope_signature",
            "build_request_signature",
            "evidence_reference_id",
        ):
            _require_non_empty(getattr(self, name), name)
        if self.explicit_user_mark is not True or self.current is not True:
            raise GuidedBackendValidationRequestContractError(
                "Confirmed strategy marks must be explicit and current."
            )
        if self.evidence_chunk is not None and (
            isinstance(self.evidence_chunk, bool)
            or not isinstance(self.evidence_chunk, int)
            or self.evidence_chunk < 0
        ):
            raise GuidedBackendValidationRequestContractError(
                "evidence_chunk must be a non-negative integer."
            )


@dataclass(frozen=True)
class GuidedBackendCorrectionRequest:
    strategy_scope: str
    global_correction_strategy: str
    global_dynamic_fit_mode: str
    dynamic_fit_parameter_values: tuple[GuidedBackendTypedFieldValue, ...]
    confirmed_marks: tuple[GuidedBackendConfirmedStrategyMark, ...]
    mark_rule_version: str
    currentness_rule_version: str
    unanimity_rule_version: str
    blocked_strategy_states: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.strategy_scope != "global":
            raise GuidedBackendValidationRequestContractError(
                "strategy_scope must be global."
            )
        if self.global_correction_strategy != "dynamic_fit":
            raise GuidedBackendValidationRequestContractError(
                "global_correction_strategy must be dynamic_fit."
            )
        _require_non_empty(
            self.global_dynamic_fit_mode,
            "global_dynamic_fit_mode",
        )
        for name in (
            "dynamic_fit_parameter_values",
            "confirmed_marks",
            "blocked_strategy_states",
        ):
            _require_tuple(getattr(self, name), name)
        if not self.dynamic_fit_parameter_values or not self.confirmed_marks:
            raise GuidedBackendValidationRequestContractError(
                "Correction parameters and confirmed marks must not be empty."
            )
        if self.blocked_strategy_states:
            raise GuidedBackendValidationRequestContractError(
                "blocked_strategy_states must be empty."
            )
        modes = {mark.selected_dynamic_fit_mode for mark in self.confirmed_marks}
        if modes != {self.global_dynamic_fit_mode}:
            raise GuidedBackendValidationRequestContractError(
                "Confirmed marks must unanimously match global_dynamic_fit_mode."
            )
        for name in (
            "mark_rule_version",
            "currentness_rule_version",
            "unanimity_rule_version",
        ):
            _require_non_empty(getattr(self, name), name)


@dataclass(frozen=True)
class GuidedBackendEvidenceReference:
    evidence_reference_id: str
    evidence_kind: str
    diagnostic_cache_id: str
    source_setup_signature: str
    current: bool = True
    diagnostic_scope_signature: str = ""
    build_request_signature: str = ""
    evidence_chunk: int | None = None
    roi_id: str = ""
    selected_dynamic_fit_mode: str = ""

    def __post_init__(self) -> None:
        for name in (
            "evidence_reference_id",
            "evidence_kind",
            "diagnostic_cache_id",
            "source_setup_signature",
        ):
            _require_non_empty(getattr(self, name), name)
        if self.evidence_chunk is not None and (
            isinstance(self.evidence_chunk, bool)
            or not isinstance(self.evidence_chunk, int)
            or self.evidence_chunk < 0
        ):
            raise GuidedBackendValidationRequestContractError(
                "evidence_chunk must be a non-negative integer."
            )
        if self.roi_id or self.selected_dynamic_fit_mode:
            _require_non_empty(self.roi_id, "roi_id")
            _require_non_empty(
                self.selected_dynamic_fit_mode,
                "selected_dynamic_fit_mode",
            )
        if self.current is not True:
            raise GuidedBackendValidationRequestContractError(
                "Evidence references must be current."
            )


@dataclass(frozen=True)
class GuidedBackendDiagnosticEvidenceRequest:
    cache_id: str
    cache_root_canonical: str
    source_setup_signature: str
    diagnostic_scope_signature: str
    build_request_signature: str
    artifact_contract_version: str
    provenance_schema_version: str
    artifact_semantic_digest: str
    provenance_semantic_digest: str
    evidence_references: tuple[GuidedBackendEvidenceReference, ...]
    completed_run_rejection_category: str
    resolver_status: str
    preliminary_cache: bool
    production_analysis: bool
    stale_reasons: tuple[str, ...] = ()
    unresolved_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "cache_id",
            "cache_root_canonical",
            "source_setup_signature",
            "diagnostic_scope_signature",
            "build_request_signature",
            "artifact_contract_version",
            "provenance_schema_version",
        ):
            _require_non_empty(getattr(self, name), name)
        _require_sha256(self.artifact_semantic_digest, "artifact_semantic_digest")
        _require_sha256(
            self.provenance_semantic_digest,
            "provenance_semantic_digest",
        )
        for name in ("evidence_references", "stale_reasons", "unresolved_inputs"):
            _require_tuple(getattr(self, name), name)
        if (
            not self.evidence_references
            or self.completed_run_rejection_category
            != "guided_diagnostic_cache_ineligible"
            or self.resolver_status != "current"
            or self.preliminary_cache is not True
            or self.production_analysis is not False
            or self.stale_reasons
            or self.unresolved_inputs
        ):
            raise GuidedBackendValidationRequestContractError(
                "Diagnostic evidence must be current, preliminary, and non-production."
            )


@dataclass(frozen=True)
class GuidedBackendFeatureEventRequest:
    profile_schema_version: str
    profile_id: str
    effective_values: tuple[GuidedBackendTypedFieldValue, ...]
    active_fields: tuple[str, ...]
    inactive_fields: tuple[str, ...]
    profile_status: str
    explicitly_applied: bool
    current: bool
    visible_unapplied_changes: bool
    validation_issue_categories: tuple[str, ...] = ()
    stale_reason_categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.profile_schema_version, "profile_schema_version")
        _require_non_empty(self.profile_id, "profile_id")
        for name in (
            "effective_values",
            "active_fields",
            "inactive_fields",
            "validation_issue_categories",
            "stale_reason_categories",
        ):
            _require_tuple(getattr(self, name), name)
        if (
            not self.effective_values
            or self.profile_status != "applied"
            or self.explicitly_applied is not True
            or self.current is not True
            or self.visible_unapplied_changes is not False
            or self.validation_issue_categories
            or self.stale_reason_categories
        ):
            raise GuidedBackendValidationRequestContractError(
                "Feature/event profile must be complete, applied, and current."
            )


@dataclass(frozen=True)
class GuidedBackendOutputRelationship:
    relationship: str
    root_kind: str
    status: str

    def __post_init__(self) -> None:
        for name in ("relationship", "root_kind", "status"):
            _require_non_empty(getattr(self, name), name)


@dataclass(frozen=True)
class GuidedBackendOutputRequest:
    output_base_canonical: str
    output_base_path_style: str
    path_role: str
    future_output_owner: str
    run_directory_strategy: str
    creation_timing: str
    overwrite: bool
    precreate: bool
    policy_status: str
    policy_current: bool
    safety_classifier_version: str
    relationships: tuple[GuidedBackendOutputRelationship, ...]
    protected_root_context_complete: bool
    blocker_categories: tuple[str, ...]
    filesystem_fact_scope: str

    def __post_init__(self) -> None:
        for name in (
            "output_base_canonical",
            "output_base_path_style",
            "run_directory_strategy",
            "creation_timing",
            "safety_classifier_version",
            "filesystem_fact_scope",
        ):
            _require_non_empty(getattr(self, name), name)
        if self.path_role != "output_base":
            raise GuidedBackendValidationRequestContractError(
                "path_role must be output_base."
            )
        if self.future_output_owner != "runner":
            raise GuidedBackendValidationRequestContractError(
                "future_output_owner must be runner."
            )
        if self.overwrite is not False or self.precreate is not False:
            raise GuidedBackendValidationRequestContractError(
                "Overwrite and output precreation are prohibited."
            )
        if self.policy_status != "applied" or self.policy_current is not True:
            raise GuidedBackendValidationRequestContractError(
                "Output policy must be applied and current."
            )
        if self.protected_root_context_complete is not True:
            raise GuidedBackendValidationRequestContractError(
                "Protected-root context must be complete."
            )
        _require_tuple(self.relationships, "relationships")
        _require_tuple(self.blocker_categories, "blocker_categories")
        if self.blocker_categories:
            raise GuidedBackendValidationRequestContractError(
                "Output request cannot contain blockers."
            )


@dataclass(frozen=True)
class GuidedBackendLocalContractState:
    local_check_contract_version: str
    blocking_issue_categories: tuple[str, ...]
    warning_categories: tuple[str, ...]
    unsupported_state_flags: tuple[str, ...]
    unresolved_required_inputs: tuple[str, ...]
    deferred_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty(
            self.local_check_contract_version,
            "local_check_contract_version",
        )
        for name in (
            "blocking_issue_categories",
            "warning_categories",
            "unsupported_state_flags",
            "unresolved_required_inputs",
            "deferred_capabilities",
        ):
            _require_tuple(getattr(self, name), name)
        if (
            self.blocking_issue_categories
            or self.unsupported_state_flags
            or self.unresolved_required_inputs
        ):
            raise GuidedBackendValidationRequestContractError(
                "Successful request local contract state must be resolved."
            )
        overlap = set(self.deferred_capabilities) & set(
            self.unresolved_required_inputs
        )
        if overlap:
            raise GuidedBackendValidationRequestContractError(
                "Deferred capabilities cannot be unresolved required inputs."
            )


@dataclass(frozen=True)
class GuidedBackendValidationRequest:
    request_schema_name: str
    request_schema_version: str
    validation_scope: str
    validation_contract_version: str
    validator_capability_version: str
    compiler_version: str
    subset_rule_version: str
    canonicalization_algorithm_version: str
    source: GuidedBackendSourceRequest
    acquisition_dataset: GuidedBackendAcquisitionDatasetRequest
    parser: GuidedBackendRwdParserRequest
    roi_scope: GuidedBackendRoiScopeRequest
    correction: GuidedBackendCorrectionRequest
    diagnostic_evidence: GuidedBackendDiagnosticEvidenceRequest
    feature_event: GuidedBackendFeatureEventRequest
    output: GuidedBackendOutputRequest
    local_contract: GuidedBackendLocalContractState

    def __post_init__(self) -> None:
        expected = {
            "request_schema_name": GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME,
            "request_schema_version": GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION,
            "validation_scope": GUIDED_BACKEND_VALIDATION_SCOPE,
            "validation_contract_version": GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION,
            "compiler_version": GUIDED_BACKEND_VALIDATION_COMPILER_VERSION,
            "subset_rule_version": GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION,
            "canonicalization_algorithm_version": CANONICALIZATION_ALGORITHM_VERSION,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise GuidedBackendValidationRequestContractError(
                    f"{field_name} must equal {expected_value}."
                )
        _require_non_empty(
            self.validator_capability_version,
            "validator_capability_version",
        )
        if self.validator_capability_version.strip().lower() == "unknown":
            raise GuidedBackendValidationRequestContractError(
                "validator_capability_version cannot be unknown."
            )


@dataclass(frozen=True)
class GuidedBackendSourceSnapshotFacts:
    available: bool = False
    source_root_canonical: str = ""
    source_candidate_set_digest: str = ""
    source_candidate_content_digest: str = ""
    candidate_files: tuple[GuidedBackendSourceCandidateFile, ...] = ()
    stale: bool = False

    def __post_init__(self) -> None:
        _require_tuple(self.candidate_files, "candidate_files")


@dataclass(frozen=True)
class GuidedBackendIncompleteFinalClassificationFacts:
    available: bool = False
    classification_status: str = ""
    classification_digest: str = ""
    source_candidate_set_digest: str = ""
    source_candidate_content_digest: str = ""


@dataclass(frozen=True)
class GuidedBackendParserFacts:
    available: bool = False
    parser_contract_digest: str = ""
    unresolved_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_tuple(self.unresolved_inputs, "unresolved_inputs")


@dataclass(frozen=True)
class GuidedBackendEvidenceReferenceFacts:
    references: tuple[GuidedBackendEvidenceReference, ...] = ()
    complete: bool = False

    def __post_init__(self) -> None:
        _require_tuple(self.references, "references")


@dataclass(frozen=True)
class GuidedBackendDiagnosticCacheFacts:
    available: bool = False
    cache_id: str = ""
    cache_root_canonical: str = ""
    artifact_semantic_digest: str = ""
    provenance_semantic_digest: str = ""
    completed_run_rejection_category: str = ""
    resolver_status: str = ""
    source_setup_signature: str = ""
    diagnostic_scope_signature: str = ""
    build_request_signature: str = ""
    preliminary_cache: bool = False
    production_analysis: bool = False


@dataclass(frozen=True)
class GuidedBackendOutputFacts:
    available: bool = False
    output_base_canonical: str = ""
    protected_root_context_complete: bool = False
    relationships: tuple[GuidedBackendOutputRelationship, ...] = ()
    blocker_categories: tuple[str, ...] = ()
    filesystem_fact_scope: str = ""

    def __post_init__(self) -> None:
        _require_tuple(self.relationships, "relationships")
        _require_tuple(self.blocker_categories, "blocker_categories")


@dataclass(frozen=True)
class GuidedBackendValidationMaterializedFacts:
    source_snapshot: GuidedBackendSourceSnapshotFacts = field(
        default_factory=GuidedBackendSourceSnapshotFacts
    )
    incomplete_final_classification: GuidedBackendIncompleteFinalClassificationFacts = field(
        default_factory=GuidedBackendIncompleteFinalClassificationFacts
    )
    parser: GuidedBackendParserFacts = field(
        default_factory=GuidedBackendParserFacts
    )
    diagnostic_cache: GuidedBackendDiagnosticCacheFacts = field(
        default_factory=GuidedBackendDiagnosticCacheFacts
    )
    output: GuidedBackendOutputFacts = field(
        default_factory=GuidedBackendOutputFacts
    )
    evidence_references: GuidedBackendEvidenceReferenceFacts = field(
        default_factory=GuidedBackendEvidenceReferenceFacts
    )
    effective_feature_event_values: tuple[GuidedBackendTypedFieldValue, ...] = ()
    complete_for_compilation: bool = False
    unresolved_required_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_tuple(
            self.effective_feature_event_values,
            "effective_feature_event_values",
        )
        _require_tuple(
            self.unresolved_required_inputs,
            "unresolved_required_inputs",
        )


@dataclass(frozen=True)
class GuidedBackendValidatorContract:
    validation_scope: str
    validation_contract_version: str
    validator_capability_version: str
    supported_subset_rule_version: str

    def __post_init__(self) -> None:
        expected = {
            "validation_scope": GUIDED_BACKEND_VALIDATION_SCOPE,
            "validation_contract_version": GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION,
            "supported_subset_rule_version": GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION,
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise GuidedBackendValidationRequestContractError(
                    f"{field_name} must equal {expected_value}."
                )
        _require_non_empty(
            self.validator_capability_version,
            "validator_capability_version",
        )
        if self.validator_capability_version.strip().lower() == "unknown":
            raise GuidedBackendValidationRequestContractError(
                "validator_capability_version cannot be unknown."
            )


@dataclass(frozen=True)
class GuidedBackendValidationCompileDebugValue:
    field_name: str
    value: str | bool | int | float | None

    def __post_init__(self) -> None:
        _require_non_empty(self.field_name, "field_name")
        _require_scalar(self.value, self.field_name)


@dataclass(frozen=True)
class GuidedBackendValidationCompileIssue:
    category: str
    section: str
    message: str
    detail_code: str | None = None
    debug_context: tuple[GuidedBackendValidationCompileDebugValue, ...] = ()

    def __post_init__(self) -> None:
        if self.category not in GUIDED_BACKEND_VALIDATION_REFUSAL_CATEGORY_SET:
            raise GuidedBackendValidationRequestContractError(
                f"Unsupported refusal category: {self.category}."
            )
        _require_non_empty(self.section, "section")
        _require_non_empty(self.message, "message")
        if self.detail_code is not None:
            _require_non_empty(self.detail_code, "detail_code")
        _require_tuple(self.debug_context, "debug_context")


@dataclass(frozen=True)
class GuidedBackendValidationCompileFailure:
    blocking_issues: tuple[GuidedBackendValidationCompileIssue, ...]
    warning_categories: tuple[str, ...] = ()
    status: str = field(default="refused", init=False)
    no_partial_request: bool = field(default=True, init=False)
    no_request_identity: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        _require_tuple(self.blocking_issues, "blocking_issues")
        _require_tuple(self.warning_categories, "warning_categories")
        if not self.blocking_issues:
            raise GuidedBackendValidationRequestContractError(
                "Compile failure requires at least one blocking issue."
            )


@dataclass(frozen=True)
class GuidedBackendValidationCompileSuccess:
    request: GuidedBackendValidationRequest
    canonical_request_identity: str
    status: str = field(default="compiled", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, GuidedBackendValidationRequest):
            raise GuidedBackendValidationRequestContractError(
                "Compile success requires a GuidedBackendValidationRequest."
            )
        _require_sha256(
            self.canonical_request_identity,
            "canonical_request_identity",
        )


GuidedBackendValidationCompileResult = (
    GuidedBackendValidationCompileSuccess
    | GuidedBackendValidationCompileFailure
)


def _failure(
    category: str,
    section: str,
    message: str,
    *,
    detail_code: str | None = None,
) -> GuidedBackendValidationCompileFailure:
    return GuidedBackendValidationCompileFailure(
        blocking_issues=(
            GuidedBackendValidationCompileIssue(
                category=category,
                section=section,
                message=message,
                detail_code=detail_code,
            ),
        )
    )


def _validator_contract_is_accepted(value: Any) -> bool:
    if not isinstance(value, GuidedBackendValidatorContract):
        return False
    return (
        getattr(value, "validation_scope", None)
        == GUIDED_BACKEND_VALIDATION_SCOPE
        and getattr(value, "validation_contract_version", None)
        == GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
        and getattr(value, "supported_subset_rule_version", None)
        == GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
        and isinstance(getattr(value, "validator_capability_version", None), str)
        and bool(value.validator_capability_version.strip())
        and value.validator_capability_version.strip().lower() != "unknown"
    )


def compile_guided_backend_validation_request(
    draft: GuidedNewAnalysisDraftPlan | None,
    *,
    facts: GuidedBackendValidationMaterializedFacts | None,
    validator_contract: GuidedBackendValidatorContract | None,
) -> GuidedBackendValidationCompileResult:
    """Refuse until the separately scoped read-only materializer is available."""
    if not isinstance(draft, GuidedNewAnalysisDraftPlan):
        return _failure(
            "compiler_contract_unavailable",
            "compiler",
            "A materialized Guided new-analysis draft is required.",
            detail_code="draft_missing_or_invalid",
        )
    if not isinstance(facts, GuidedBackendValidationMaterializedFacts):
        return _failure(
            "unresolved_required_identity_input",
            "materialized_facts",
            "Read-only materialized validation facts are required.",
            detail_code="facts_missing_or_invalid",
        )
    if not _validator_contract_is_accepted(validator_contract):
        return _failure(
            "validator_contract_unavailable",
            "validator_contract",
            "An accepted backend validator contract is required.",
            detail_code="validator_contract_missing_or_invalid",
        )
    return _failure(
        "compiler_contract_unavailable",
        "compiler",
        (
            "Guided backend validation request population is not implemented "
            "in the Stage 1 contract checkpoint."
        ),
        detail_code="stage_1_refusal_only",
    )


def compute_guided_backend_validation_request_identity(
    request: GuidedBackendValidationRequest,
) -> str:
    """Identity is deferred until a complete semantic request can be compiled."""
    if not isinstance(request, GuidedBackendValidationRequest):
        raise GuidedBackendValidationRequestContractError(
            "request must be a GuidedBackendValidationRequest."
        )
    raise GuidedBackendValidationRequestContractError(
        "Canonical backend validation request identity is deferred until "
        "complete request population is implemented."
    )
