"""Pure contracts for a future exact Guided backend validation request.

This Stage 1 module defines immutable request, materialized-fact, validator,
success, and refusal models. It performs no filesystem access, materialization,
backend validation, execution, or artifact generation. The compiler currently
refuses every path before request construction because read-only materialized
facts are not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping

from photometry_pipeline.guided_identity import (
    CANONICALIZATION_ALGORITHM_VERSION,
    GuidedIdentityError,
    encode_canonical_value,
)
from photometry_pipeline.guided_new_analysis_plan import (
    FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
    GuidedNewAnalysisDraftPlan,
)


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
GUIDED_BACKEND_FEATURE_EVENT_PROFILE_SCHEMA_VERSION = (
    "guided_feature_event_profile.v1"
)


def is_saved_feature_event_profile_current(
    profile_status: str, explicitly_applied: bool
) -> bool:
    """Whether persisted feature settings are usable for validation.

    Loaded valid Defaults are already consumed; the Apply action is required
    only to accept edits to that saved Default profile.
    """
    return bool(
        profile_status == "default_initialized"
        or (profile_status == "applied" and explicitly_applied is True)
    )


GUIDED_BACKEND_LOCAL_CHECK_CONTRACT_VERSION = "guided_backend_local_checks.v1"
GUIDED_BACKEND_DIAGNOSTIC_CACHE_SCHEMA_VERSION = "guided_diagnostic_cache.v1"
GUIDED_BACKEND_SOURCE_SNAPSHOT_SCHEMA_NAME = (
    "guided_rwd_source_candidate_snapshot"
)
GUIDED_BACKEND_SOURCE_SNAPSHOT_SCHEMA_VERSION = "v1"
GUIDED_BACKEND_SOURCE_DISCOVERY_RULE_VERSION = (
    "immediate_child_exact_fluorescence_csv.v1"
)
GUIDED_BACKEND_SOURCE_RELATIVE_PATH_RULE_VERSION = (
    "canonical_forward_slash_relative_path.v1"
)
GUIDED_BACKEND_SOURCE_IGNORED_FILES_POLICY = (
    "ignore_non_target_entries_bounded_nested_root_check.v1"
)
GUIDED_BACKEND_NPM_SOURCE_SNAPSHOT_SCHEMA_NAME = (
    "guided_npm_source_candidate_snapshot"
)
GUIDED_BACKEND_NPM_SOURCE_DISCOVERY_RULE_VERSION = (
    "immediate_child_csv_exact_filename_timestamp.v1"
)
GUIDED_BACKEND_NPM_SOURCE_IGNORED_FILES_POLICY = (
    "ignore_non_csv_immediate_children.v1"
)
GUIDED_BACKEND_INCOMPLETE_FINAL_SCHEMA_NAME = (
    "guided_rwd_incomplete_final_chunk_classification"
)
GUIDED_BACKEND_INCOMPLETE_FINAL_SCHEMA_VERSION = "v1"
GUIDED_BACKEND_INCOMPLETE_FINAL_CLASSIFIER_VERSION = "not_requested_only.v1"
GUIDED_BACKEND_DISPOSITION_POLICY_SCHEMA_NAME = "guided_backend_disposition_policy"
GUIDED_BACKEND_DISPOSITION_POLICY_SCHEMA_VERSION = "v1"


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
    "local_preview_setup_signature_mismatch",
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
    "incomplete_materialized_facts",
    "unresolved_materialized_inputs",
    "missing_source_snapshot",
    "unsupported_analysis_scope",
    "incomplete_final_policy_not_supported",
    "parser_contract_unavailable",
    "parser_unresolved_inputs",
    "feature_event_effective_value_unresolved",
    "unsupported_request_field",
    "compiler_internal_error",
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
    approved_missing_candidates: tuple[GuidedBackendSourceCandidateFile, ...] = ()
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
        if self.source_format not in {"rwd", "npm"}:
            raise GuidedBackendValidationRequestContractError(
                "source_format must be rwd or npm."
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
        _require_tuple(self.approved_missing_candidates, "approved_missing_candidates")
        _require_tuple(self.unresolved_source_identity_inputs, "unresolved_source_identity_inputs")
        if self.unresolved_source_identity_inputs:
            raise GuidedBackendValidationRequestContractError(
                "Source request cannot contain unresolved identity inputs."
            )
        all_by_path = {item.canonical_relative_path: item for item in self.candidate_files}
        approved_paths = [item.canonical_relative_path for item in self.approved_missing_candidates]
        if len(approved_paths) != len(set(approved_paths)):
            raise GuidedBackendValidationRequestContractError(
                "Approved missing source entries must be unique."
            )
        if any(all_by_path.get(item.canonical_relative_path) != item for item in self.approved_missing_candidates):
            raise GuidedBackendValidationRequestContractError(
                "Approved missing sources must be exact members of the source candidate set."
            )


@dataclass(frozen=True)
class GuidedBackendDispositionPolicyRequest:
    schema_name: str
    schema_version: str
    admitted_dispositions: tuple[str, ...]
    missing_session_policy: str
    excluded_session_policy: str
    partial_support_owner: str

    def __post_init__(self) -> None:
        if self.schema_name != GUIDED_BACKEND_DISPOSITION_POLICY_SCHEMA_NAME:
            raise GuidedBackendValidationRequestContractError(
                "Unsupported disposition policy schema name."
            )
        if self.schema_version != GUIDED_BACKEND_DISPOSITION_POLICY_SCHEMA_VERSION:
            raise GuidedBackendValidationRequestContractError(
                "Unsupported disposition policy schema version."
            )
        _require_tuple(self.admitted_dispositions, "admitted_dispositions")
        if any(not isinstance(value, str) or not value for value in self.admitted_dispositions):
            raise GuidedBackendValidationRequestContractError(
                "admitted_dispositions must contain non-empty strings."
            )
        for name in (
            "missing_session_policy",
            "excluded_session_policy",
            "partial_support_owner",
        ):
            _require_non_empty(getattr(self, name), name)


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
    execution_mode: str = "phasic"

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
class GuidedBackendNpmAcquisitionDatasetRequest:
    """NPM acquisition facts without RWD incomplete-final classifier fields."""

    acquisition_mode: str
    sessions_per_hour: int
    session_duration_sec: float
    timeline_anchor_mode: str
    fixed_daily_anchor_clock: None
    allow_partial_final_window: bool
    dataset_snapshot_schema_version: str
    dataset_status: str
    dataset_current_applied: bool
    semantic_values: tuple[GuidedBackendTypedFieldValue, ...] = ()
    dataset_source_setup_signature: str = ""
    diagnostic_cache_contract_identity: str = ""
    validation_issue_categories: tuple[str, ...] = ()
    stale_reason_categories: tuple[str, ...] = ()
    execution_mode: str = "phasic"
    source_format: str = "npm"
    npm_time_axis: str = ""
    npm_system_ts_col: str = ""
    npm_computer_ts_col: str = ""
    npm_led_col: str = ""
    npm_region_prefix: str = ""
    npm_region_suffix: str = ""
    npm_target_fs_hz: float | None = None
    npm_adapter_value_nan_policy: str = ""
    disposition_policy: GuidedBackendDispositionPolicyRequest | None = None

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
        if self.timeline_anchor_mode != "civil" or self.fixed_daily_anchor_clock is not None:
            raise GuidedBackendValidationRequestContractError(
                "NPM timeline anchor policy is unsupported."
            )
        if not isinstance(self.allow_partial_final_window, bool):
            raise GuidedBackendValidationRequestContractError(
                "allow_partial_final_window must be boolean."
            )
        if self.source_format != "npm":
            raise GuidedBackendValidationRequestContractError(
                "NPM acquisition facts must have source_format npm."
            )
        if self.dataset_status != "applied" or self.dataset_current_applied is not True:
            raise GuidedBackendValidationRequestContractError(
                "Dataset contract must be currently applied."
            )
        for name in (
            "dataset_snapshot_schema_version",
            "dataset_source_setup_signature",
            "diagnostic_cache_contract_identity",
            "npm_time_axis",
            "npm_led_col",
            "npm_region_prefix",
            "npm_region_suffix",
            "npm_adapter_value_nan_policy",
        ):
            _require_non_empty(getattr(self, name), name)
        if self.npm_adapter_value_nan_policy not in {"strict", "mask"}:
            raise GuidedBackendValidationRequestContractError(
                "Unsupported NPM ROI NaN policy."
            )
        if self.npm_target_fs_hz is None or float(self.npm_target_fs_hz) <= 0:
            raise GuidedBackendValidationRequestContractError(
                "NPM target_fs_hz must be positive."
            )
        if not isinstance(self.disposition_policy, GuidedBackendDispositionPolicyRequest):
            raise GuidedBackendValidationRequestContractError(
                "NPM disposition policy is required."
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
class GuidedBackendNpmParserRequest:
    schema_name: str
    schema_version: str
    timestamp_column_candidates: tuple[str, ...]
    parser_contract_digest: str
    parser_contract_content: Mapping[str, Any]
    unresolved_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("schema_name", "schema_version"):
            _require_non_empty(getattr(self, name), name)
        _require_tuple(self.timestamp_column_candidates, "timestamp_column_candidates")
        if not self.timestamp_column_candidates or any(
            not isinstance(value, str) or not value for value in self.timestamp_column_candidates
        ):
            raise GuidedBackendValidationRequestContractError(
                "NPM timestamp column candidates must be complete."
            )
        _require_sha256(self.parser_contract_digest, "parser_contract_digest")
        if not isinstance(self.parser_contract_content, Mapping):
            raise GuidedBackendValidationRequestContractError(
                "NPM parser contract content must be an object."
            )
        def freeze(value: Any) -> Any:
            if isinstance(value, Mapping):
                if any(not isinstance(key, str) for key in value):
                    raise GuidedBackendValidationRequestContractError(
                        "NPM parser contract content keys must be strings."
                    )
                return MappingProxyType({key: freeze(item) for key, item in value.items()})
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            if value is None or isinstance(value, (str, bool, int)):
                return value
            if isinstance(value, float) and math.isfinite(value):
                return value
            raise GuidedBackendValidationRequestContractError(
                "NPM parser contract content contains a non-canonical value."
            )
        frozen_content = freeze(self.parser_contract_content)
        object.__setattr__(self, "parser_contract_content", frozen_content)
        if self.unresolved_inputs:
            raise GuidedBackendValidationRequestContractError(
                "NPM parser contract has unresolved inputs."
            )
        try:
            from photometry_pipeline.guided_normalized_recording import (
                compute_npm_parser_contract_digest,
            )

            expected = compute_npm_parser_contract_digest(self.parser_contract_content)
        except Exception as exc:
            raise GuidedBackendValidationRequestContractError(
                "NPM parser contract content is invalid."
            ) from exc
        if expected != self.parser_contract_digest:
            raise GuidedBackendValidationRequestContractError(
                "NPM parser contract digest does not match its content."
            )


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
            "source_setup_signature",
            "evidence_reference_id",
        ):
            _require_non_empty(getattr(self, name), name)
        # diagnostic_cache_id/diagnostic_scope_signature/build_request_signature
        # are cache-identity fields: non-empty for diagnostic_cache-backed
        # marks, empty for local_correction_preview marks (which prove
        # currentness via source_setup_signature instead).
        if bool(self.diagnostic_cache_id) != bool(self.diagnostic_scope_signature) or bool(
            self.diagnostic_cache_id
        ) != bool(self.build_request_signature):
            raise GuidedBackendValidationRequestContractError(
                "diagnostic_cache_id, diagnostic_scope_signature, and "
                "build_request_signature must be either all present or all "
                "empty."
            )
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
class GuidedBackendPerRoiProductionStrategy:
    roi_id: str
    strategy_family: str
    dynamic_fit_mode: str | None
    selected_strategy: str
    evidence_source_type: str
    evidence_reference_json: str
    explicit_user_mark: bool
    current_or_stale: str


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
    production_strategy_map_version: str = ""
    per_roi_production_strategy_map: tuple[
        GuidedBackendPerRoiProductionStrategy, ...
    ] = ()
    # Deprecated: the obsolete Guided post-hoc applied-dF/F route has been
    # retired from current-Guided production. Retained only as inert
    # deprecated input threaded through for canonicalization/identity
    # purposes; no execution logic branches on it.
    applied_dff_orchestration_enabled: bool = False

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
            "per_roi_production_strategy_map",
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
        if not self.per_roi_production_strategy_map and modes != {self.global_dynamic_fit_mode}:
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
        required_fields = ["evidence_reference_id", "evidence_kind", "source_setup_signature"]
        if self.evidence_kind != "local_correction_preview":
            required_fields.append("diagnostic_cache_id")
        for name in required_fields:
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
    available: bool = True

    def __post_init__(self) -> None:
        for name in ("evidence_references", "stale_reasons", "unresolved_inputs"):
            _require_tuple(getattr(self, name), name)
        if not self.available:
            # Cache-free local-preview evidence path: no diagnostic-cache
            # identity is required or claimed. All cache-identity fields
            # must be empty so this state cannot be mistaken for a resolved
            # diagnostic cache.
            if (
                self.cache_id
                or self.cache_root_canonical
                or self.source_setup_signature
                or self.diagnostic_scope_signature
                or self.build_request_signature
                or self.artifact_semantic_digest
                or self.provenance_semantic_digest
                or self.completed_run_rejection_category
                or self.resolver_status
                or self.preliminary_cache is not False
                or self.production_analysis is not False
                or not self.evidence_references
                or self.stale_reasons
                or self.unresolved_inputs
            ):
                raise GuidedBackendValidationRequestContractError(
                    "Unavailable diagnostic evidence must not carry cache "
                    "identity fields."
                )
            return
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
class GuidedBackendPerRoiFeatureEvent:
    """One ROI's resolved feature/event settings, mirroring
    GuidedBackendPerRoiProductionStrategy's role for correction strategy.

    override_config_fields is whatever fields the resolved source (the
    plan's default profile, or one explicit per-ROI override) set, which may
    be a SPARSE subset of feature-detection fields. effective_config_fields
    is always the COMPLETE, resolved set of feature-detection fields
    actually in effect for this ROI (override_config_fields layered onto the
    plan's default profile). Backend execution and applied-dF/F feature
    config files must use effective_config_fields, never
    override_config_fields alone.
    """

    roi_id: str
    source: str  # "default" or "override"
    feature_event_profile_id: str
    override_config_fields: tuple[GuidedBackendTypedFieldValue, ...]
    effective_config_fields: tuple[GuidedBackendTypedFieldValue, ...]
    explicit_user_mark: bool
    current_or_stale: str


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
    per_roi_feature_event_map_version: str = ""
    per_roi_feature_event_map: tuple[GuidedBackendPerRoiFeatureEvent, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.profile_schema_version, "profile_schema_version")
        _require_non_empty(self.profile_id, "profile_id")
        for name in (
            "effective_values",
            "active_fields",
            "inactive_fields",
            "validation_issue_categories",
            "stale_reason_categories",
            "per_roi_feature_event_map",
        ):
            _require_tuple(getattr(self, name), name)
        if (
            not self.effective_values
            or not is_saved_feature_event_profile_current(
                self.profile_status, self.explicitly_applied
            )
            or self.current is not True
            or self.visible_unapplied_changes is not False
            or self.validation_issue_categories
            or self.stale_reason_categories
        ):
            raise GuidedBackendValidationRequestContractError(
                "Feature/event profile must be complete and current."
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
    acquisition_dataset: (
        GuidedBackendAcquisitionDatasetRequest
        | GuidedBackendNpmAcquisitionDatasetRequest
    )
    parser: GuidedBackendRwdParserRequest | GuidedBackendNpmParserRequest
    roi_scope: GuidedBackendRoiScopeRequest
    correction: GuidedBackendCorrectionRequest
    diagnostic_evidence: GuidedBackendDiagnosticEvidenceRequest
    feature_event: GuidedBackendFeatureEventRequest
    output: GuidedBackendOutputRequest
    local_contract: GuidedBackendLocalContractState
    # B1: threads the shared, format-neutral normalized recording
    # description's canonical identity (guided_normalized_recording.py)
    # into the request identity chain -- compute_guided_backend_validation_request_identity
    # already covers it once listed in
    # _GUIDED_BACKEND_VALIDATION_IDENTITY_FIELDS below, so the existing
    # authorization/startup identity-consistency checks
    # (guided_run_authorization.py, guided_production_mapping.py) already
    # refuse a stale or altered normalized description without further
    # changes there. Only the identity (not the full payload) lives on
    # this strict, exhaustively identity-covered request contract; the
    # full serialized description lives on
    # GuidedBackendValidationMaterializedFacts, which downstream stages
    # that need the full content read directly (same process) or rebuild
    # and verify against this identity (see
    # guided_normalized_recording.build_rwd_normalized_recording_description).
    normalized_recording_description_identity: str = ""
    normalized_recording_description: Mapping[str, Any] | None = None

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
        if not isinstance(
            self.acquisition_dataset,
            (
                GuidedBackendAcquisitionDatasetRequest,
                GuidedBackendNpmAcquisitionDatasetRequest,
            ),
        ):
            raise GuidedBackendValidationRequestContractError(
                "acquisition_dataset has an unsupported request variant."
            )
        if self.source.source_format == "npm":
            if not isinstance(
                self.acquisition_dataset,
                GuidedBackendNpmAcquisitionDatasetRequest,
            ):
                raise GuidedBackendValidationRequestContractError(
                    "The NPM recording setup could not be confirmed."
                )
            if not isinstance(self.normalized_recording_description, Mapping):
                raise GuidedBackendValidationRequestContractError(
                    "The NPM recording setup could not be confirmed."
                )
        elif not isinstance(
            self.acquisition_dataset,
            GuidedBackendAcquisitionDatasetRequest,
        ):
            raise GuidedBackendValidationRequestContractError(
                "RWD source requires the RWD acquisition request variant."
            )


@dataclass(frozen=True)
class GuidedBackendSourceSnapshotFacts:
    available: bool = False
    source_root_canonical: str = ""
    source_root_path_style: str = ""
    source_candidate_set_digest: str = ""
    source_candidate_content_digest: str = ""
    candidate_files: tuple[GuidedBackendSourceCandidateFile, ...] = ()
    approved_missing_candidates: tuple[GuidedBackendSourceCandidateFile, ...] = ()
    stale: bool = False

    def __post_init__(self) -> None:
        _require_tuple(self.candidate_files, "candidate_files")
        _require_tuple(self.approved_missing_candidates, "approved_missing_candidates")


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
    schema_name: str = ""
    schema_version: str = ""
    header_search_line_limit: int | None = None
    time_column_candidates: tuple[str, ...] = ()
    uv_suffix_candidates: tuple[str, ...] = ()
    signal_suffix_candidates: tuple[str, ...] = ()
    column_normalization_rule: str = ""
    roi_name_rule: str = ""
    ambiguity_policy: str = ""
    parser_contract_digest: str = ""
    unresolved_inputs: tuple[str, ...] = ()
    npm_timestamp_column_candidates: tuple[str, ...] = ()
    npm_parser_contract_content: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in (
            "time_column_candidates",
            "uv_suffix_candidates",
            "signal_suffix_candidates",
        ):
            _require_tuple(getattr(self, name), name)
        _require_tuple(self.unresolved_inputs, "unresolved_inputs")
        _require_tuple(
            self.npm_timestamp_column_candidates,
            "npm_timestamp_column_candidates",
        )


@dataclass(frozen=True)
class GuidedBackendAcquisitionDatasetFacts:
    available: bool = False
    acquisition_mode: str = ""
    sessions_per_hour: int | None = None
    session_duration_sec: float | None = None
    timeline_anchor_mode: str = ""
    fixed_daily_anchor_clock: str | None = None
    allow_partial_final_window: bool = False
    exclude_incomplete_final_rwd_chunk: bool = False
    dataset_snapshot_schema_version: str = ""
    dataset_status: str = ""
    dataset_current_applied: bool = False
    rwd_time_col: str = ""
    uv_suffix: str = ""
    sig_suffix: str = ""
    semantic_values: tuple[GuidedBackendTypedFieldValue, ...] = ()
    dataset_source_setup_signature: str = ""
    diagnostic_cache_contract_identity: str = ""
    validation_issue_categories: tuple[str, ...] = ()
    stale_reason_categories: tuple[str, ...] = ()
    execution_mode: str = "phasic"
    source_format: str = "rwd"
    npm_time_axis: str = ""
    npm_system_ts_col: str = ""
    npm_computer_ts_col: str = ""
    npm_led_col: str = ""
    npm_region_prefix: str = ""
    npm_region_suffix: str = ""
    npm_target_fs_hz: float | None = None
    npm_adapter_value_nan_policy: str = ""

    def __post_init__(self) -> None:
        for name in (
            "semantic_values",
            "validation_issue_categories",
            "stale_reason_categories",
        ):
            _require_tuple(getattr(self, name), name)


@dataclass(frozen=True)
class GuidedBackendRoiScopeFacts:
    available: bool = False
    discovered_roi_ids: tuple[str, ...] = ()
    included_roi_ids: tuple[str, ...] = ()
    excluded_roi_ids: tuple[str, ...] = ()
    selection_mode: str = "include"
    inventory_status: str = ""
    inventory_source_content_digest: str = ""
    roi_inventory_identity_status: str = "deferred_not_authoritative"

    def __post_init__(self) -> None:
        for name in (
            "discovered_roi_ids",
            "included_roi_ids",
            "excluded_roi_ids",
        ):
            _require_tuple(getattr(self, name), name)


@dataclass(frozen=True)
class GuidedBackendConfirmedStrategyMarkFacts:
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


@dataclass(frozen=True)
class GuidedBackendCorrectionFacts:
    available: bool = False
    strategy_scope: str = "global"
    global_correction_strategy: str = "dynamic_fit"
    global_dynamic_fit_mode: str = ""
    dynamic_fit_parameter_values: tuple[GuidedBackendTypedFieldValue, ...] = ()
    confirmed_marks: tuple[GuidedBackendConfirmedStrategyMarkFacts, ...] = ()
    mark_rule_version: str = "explicit_confirmed_mark.v1"
    currentness_rule_version: str = "cache_bound_currentness.v1"
    unanimity_rule_version: str = "included_roi_unanimous_dynamic_fit.v1"
    blocked_strategy_states: tuple[str, ...] = ()
    production_strategy_map_version: str = ""
    per_roi_production_strategy_map: tuple[
        GuidedBackendPerRoiProductionStrategy, ...
    ] = ()

    def __post_init__(self) -> None:
        for name in (
            "dynamic_fit_parameter_values",
            "confirmed_marks",
            "blocked_strategy_states",
            "per_roi_production_strategy_map",
        ):
            _require_tuple(getattr(self, name), name)


@dataclass(frozen=True)
class GuidedBackendFeatureEventFacts:
    available: bool = False
    profile_schema_version: str = ""
    profile_id: str = ""
    effective_values: tuple[GuidedBackendTypedFieldValue, ...] = ()
    active_fields: tuple[str, ...] = ()
    inactive_fields: tuple[str, ...] = ()
    profile_status: str = ""
    explicitly_applied: bool = False
    current: bool = False
    visible_unapplied_changes: bool = False
    validation_issue_categories: tuple[str, ...] = ()
    stale_reason_categories: tuple[str, ...] = ()
    per_roi_feature_event_map_version: str = ""
    per_roi_feature_event_map: tuple[GuidedBackendPerRoiFeatureEvent, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "effective_values",
            "active_fields",
            "inactive_fields",
            "validation_issue_categories",
            "stale_reason_categories",
            "per_roi_feature_event_map",
        ):
            _require_tuple(getattr(self, name), name)


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
    output_base_path_style: str = ""
    path_role: str = ""
    future_output_owner: str = ""
    run_directory_strategy: str = ""
    creation_timing: str = ""
    overwrite: bool = False
    precreate: bool = False
    policy_status: str = ""
    policy_current: bool = False
    safety_classifier_version: str = ""
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
    acquisition_dataset: GuidedBackendAcquisitionDatasetFacts = field(
        default_factory=GuidedBackendAcquisitionDatasetFacts
    )
    roi_scope: GuidedBackendRoiScopeFacts = field(
        default_factory=GuidedBackendRoiScopeFacts
    )
    correction: GuidedBackendCorrectionFacts = field(
        default_factory=GuidedBackendCorrectionFacts
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
    feature_event: GuidedBackendFeatureEventFacts = field(
        default_factory=GuidedBackendFeatureEventFacts
    )
    effective_feature_event_values: tuple[GuidedBackendTypedFieldValue, ...] = ()
    complete_for_compilation: bool = False
    unresolved_required_inputs: tuple[str, ...] = ()
    # B1: the shared, format-neutral normalized recording description
    # (see guided_normalized_recording.py), built from the RWD facts
    # materialized above and stored as its canonical serialized payload
    # (guided_normalized_recording.serialize_normalized_recording_description)
    # so later stages read the actual persisted facts -- session order,
    # dispositions, timing, ROI/channel pairing -- rather than rebuilding
    # them independently. ``normalized_recording_description_identity`` is
    # duplicated at top level for cheap comparison without deserializing.
    # Both are "" / empty only for facts objects built before this field
    # existed (never for a real success result).
    normalized_recording_description_identity: str = ""
    normalized_recording_description: Mapping[str, Any] = field(default_factory=dict)

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
    request_identity_deferred: bool = False
    status: str = field(default="compiled", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, GuidedBackendValidationRequest):
            raise GuidedBackendValidationRequestContractError(
                "Compile success requires a GuidedBackendValidationRequest."
            )
        if self.request_identity_deferred is not False:
            raise GuidedBackendValidationRequestContractError(
                "Compile success cannot defer request identity."
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


def compile_guided_backend_validation_request(
    draft: GuidedNewAnalysisDraftPlan | None,
    *,
    facts: GuidedBackendValidationMaterializedFacts | None,
    validator_contract: GuidedBackendValidatorContract | None,
) -> GuidedBackendValidationCompileResult:
    """Compile a request from complete immutable facts without performing I/O."""
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
    if not isinstance(validator_contract, GuidedBackendValidatorContract):
        return _failure(
            "validator_contract_unavailable",
            "validator_contract",
            "An accepted backend validator contract is required.",
            detail_code="validator_contract_missing_or_invalid_type",
        )
    if (
        getattr(validator_contract, "validation_scope", None)
        != GUIDED_BACKEND_VALIDATION_SCOPE
    ):
        return _failure(
            "validator_contract_unavailable",
            "validator_contract",
            "Validator contract validation scope is unsupported.",
            detail_code="validation_scope_mismatch",
        )
    if (
        getattr(validator_contract, "validation_contract_version", None)
        != GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
    ):
        return _failure(
            "validator_contract_unavailable",
            "validator_contract",
            "Validator contract version is unsupported.",
            detail_code="validation_contract_version_mismatch",
        )
    if (
        getattr(validator_contract, "supported_subset_rule_version", None)
        != GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION
    ):
        return _failure(
            "validator_contract_unavailable",
            "validator_contract",
            "Validator contract subset rule is unsupported.",
            detail_code="supported_subset_rule_version_mismatch",
        )
    validator_capability_version = getattr(
        validator_contract,
        "validator_capability_version",
        None,
    )
    if (
        not isinstance(validator_capability_version, str)
        or not validator_capability_version.strip()
        or validator_capability_version.strip().lower()
        in {"unknown", "placeholder", "unset", "none"}
    ):
        return _failure(
            "validator_contract_unavailable",
            "validator_contract",
            "Validator capability version is missing or a placeholder.",
            detail_code="validator_capability_version_invalid",
        )

    if facts.complete_for_compilation is not True:
        return _failure(
            "incomplete_materialized_facts",
            "materialized_facts",
            "Materialized facts are not complete for request compilation.",
            detail_code="complete_for_compilation_false",
        )
    if facts.unresolved_required_inputs:
        return _failure(
            "unresolved_materialized_inputs",
            "materialized_facts",
            "Materialized facts contain unresolved required inputs.",
            detail_code="unresolved_required_inputs",
        )

    if draft.input_format not in {"rwd", "npm"}:
        return _failure(
            "unsupported_source_format",
            "source",
            "The authorized compiler subset requires RWD or intermittent NPM input.",
            detail_code="source_format_unsupported",
        )
    is_npm = draft.input_format == "npm"
    if is_npm and (
        not facts.parser.available
        or not facts.parser.npm_timestamp_column_candidates
        or facts.parser.npm_parser_contract_content is None
    ):
        return _failure(
            "unsupported_source_format",
            "source",
            "The NPM import settings are incomplete. Return to the NPM settings "
            "step, apply the intended settings, and rerun Setup check.",
            detail_code="npm_parser_policy_missing",
        )
    if draft.acquisition_mode != "intermittent":
        return _failure(
            "unsupported_acquisition_mode",
            "acquisition",
            "The first compiler subset requires intermittent acquisition.",
            detail_code="acquisition_not_intermittent",
        )
    if (
        draft.execution_intent.execution_mode not in {"phasic", "tonic", "both"}
        or draft.execution_intent.run_profile != "full"
        or draft.execution_intent.timeline_anchor_mode != "civil"
        or draft.execution_intent.fixed_daily_anchor_clock is not None
        or getattr(draft, "traces_only", False) is not False
    ):
        return _failure(
            "unsupported_analysis_scope",
            "analysis_scope",
            "The first compiler subset requires civil phasic/full analysis.",
            detail_code="analysis_scope_unsupported",
        )
    if not is_npm and draft.allow_partial_final_window is not False:
        return _failure(
            "incomplete_final_policy_not_supported",
            "incomplete_final",
            "The first compiler subset requires the no-exclusion final-chunk policy.",
            detail_code="incomplete_final_policy_unsupported",
        )
    if is_npm and draft.exclude_incomplete_final_rwd_chunk:
        return _failure(
            "incomplete_final_policy_not_supported",
            "incomplete_final",
            "NPM validation does not support excluded final sessions.",
            detail_code="npm_exclusion_unsupported",
        )

    source_facts = facts.source_snapshot
    if (
        not source_facts.available
        or source_facts.stale
        or not source_facts.source_root_canonical
        or not source_facts.source_root_path_style
        or not source_facts.candidate_files
    ):
        return _failure(
            "missing_source_snapshot",
            "source",
            "A current request-ready source snapshot is required.",
            detail_code="source_snapshot_incomplete",
        )

    incomplete_facts = facts.incomplete_final_classification
    if not is_npm and (
        not incomplete_facts.available
        or incomplete_facts.classification_status != "not_requested"
        or incomplete_facts.source_candidate_set_digest
        != source_facts.source_candidate_set_digest
        or incomplete_facts.source_candidate_content_digest
        != source_facts.source_candidate_content_digest
    ):
        return _failure(
            "incomplete_final_policy_not_supported",
            "incomplete_final",
            "Incomplete-final facts are unavailable or do not match the source snapshot.",
            detail_code="incomplete_final_facts_invalid",
        )

    parser_facts = facts.parser
    parser_complete = (
        parser_facts.available
        and bool(parser_facts.schema_name)
        and bool(parser_facts.schema_version)
        and bool(parser_facts.parser_contract_digest)
        and (
            (
                bool(parser_facts.npm_timestamp_column_candidates)
                and parser_facts.npm_parser_contract_content is not None
            )
            if is_npm
            else (
                parser_facts.header_search_line_limit is not None
                and bool(parser_facts.time_column_candidates)
                and bool(parser_facts.uv_suffix_candidates)
                and bool(parser_facts.signal_suffix_candidates)
                and bool(parser_facts.column_normalization_rule)
                and bool(parser_facts.roi_name_rule)
                and bool(parser_facts.ambiguity_policy)
            )
        )
    )
    if not parser_complete:
        return _failure(
            "parser_contract_unavailable",
            "parser",
            "Request-ready parser contract facts are required.",
            detail_code="parser_facts_incomplete",
        )
    if parser_facts.unresolved_inputs:
        return _failure(
            "parser_unresolved_inputs",
            "parser",
            "Parser facts contain unresolved inputs.",
            detail_code="parser_inputs_unresolved",
        )

    dataset_facts = facts.acquisition_dataset
    if not dataset_facts.available:
        return _failure(
            "missing_or_stale_dataset_contract",
            "dataset",
            "Request-ready acquisition/dataset facts are required.",
            detail_code="dataset_facts_unavailable",
        )
    if (
        dataset_facts.acquisition_mode != "intermittent"
        or dataset_facts.source_format != draft.input_format
        or dataset_facts.sessions_per_hour is None
        or dataset_facts.sessions_per_hour <= 0
        or dataset_facts.session_duration_sec is None
        or dataset_facts.session_duration_sec <= 0
        or dataset_facts.timeline_anchor_mode != "civil"
        or dataset_facts.fixed_daily_anchor_clock is not None
        or (not is_npm and dataset_facts.allow_partial_final_window is not False)
        or dataset_facts.dataset_status != "applied"
        or dataset_facts.dataset_current_applied is not True
        or (
            not is_npm
            and (not dataset_facts.rwd_time_col or not dataset_facts.uv_suffix or not dataset_facts.sig_suffix)
        )
        or (
            is_npm
            and (
                not dataset_facts.npm_time_axis
                or not dataset_facts.npm_led_col
                or not dataset_facts.npm_region_prefix
                or not dataset_facts.npm_region_suffix
                or dataset_facts.npm_target_fs_hz is None
            )
        )
        or not dataset_facts.semantic_values
        or dataset_facts.validation_issue_categories
        or dataset_facts.stale_reason_categories
        ):
        return _failure(
            "missing_or_stale_dataset_contract",
            "dataset",
            "Acquisition/dataset facts are incomplete, stale, or unsupported.",
            detail_code="dataset_facts_invalid",
        )

    roi_facts = facts.roi_scope
    if not roi_facts.available:
        return _failure(
            "missing_roi_inventory",
            "roi_scope",
            "Request-ready ROI scope facts are required.",
            detail_code="roi_facts_unavailable",
        )
    discovered = set(roi_facts.discovered_roi_ids)
    included = set(roi_facts.included_roi_ids)
    excluded = set(roi_facts.excluded_roi_ids)
    if not discovered or not included:
        return _failure(
            "empty_included_roi_set",
            "roi_scope",
            "Discovered and included ROI sets must be non-empty.",
            detail_code="roi_scope_empty",
        )
    if (
        len(discovered) != len(roi_facts.discovered_roi_ids)
        or len(included) != len(roi_facts.included_roi_ids)
        or len(excluded) != len(roi_facts.excluded_roi_ids)
    ):
        return _failure(
            "duplicate_roi_id",
            "roi_scope",
            "ROI scope facts contain duplicate IDs.",
            detail_code="duplicate_roi_id",
        )
    if included & excluded or included | excluded != discovered:
        return _failure(
            "included_excluded_roi_conflict",
            "roi_scope",
            "Included and excluded ROI sets must partition discovered ROIs.",
            detail_code="roi_partition_invalid",
        )
    if (
        roi_facts.selection_mode != "include"
        or not roi_facts.inventory_status
        or not roi_facts.inventory_source_content_digest
        or not roi_facts.roi_inventory_identity_status
    ):
        return _failure(
            "roi_selection_stale",
            "roi_scope",
            "ROI scope metadata is unavailable or stale.",
            detail_code="roi_scope_metadata_invalid",
        )

    correction_facts = facts.correction
    if (
        not correction_facts.available
        or correction_facts.strategy_scope != "global"
        or correction_facts.global_correction_strategy != "dynamic_fit"
        or not correction_facts.global_dynamic_fit_mode
        or not correction_facts.dynamic_fit_parameter_values
        or not correction_facts.confirmed_marks
        or correction_facts.blocked_strategy_states
    ):
        return _failure(
            "missing_confirmed_strategy_mark",
            "correction",
            "Request-ready correction facts are required.",
            detail_code="correction_facts_incomplete",
        )
    global_mode = correction_facts.global_dynamic_fit_mode
    if global_mode not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
        return _failure(
            "forbidden_strategy_state",
            "correction",
            "The selected correction mode is unsupported.",
            detail_code="correction_mode_forbidden",
        )
    mark_rois = [mark.roi_id for mark in correction_facts.confirmed_marks]
    if len(mark_rois) != len(set(mark_rois)):
        return _failure(
            "duplicate_confirmed_strategy_mark",
            "correction",
            "Correction facts contain duplicate confirmed marks.",
            detail_code="duplicate_confirmed_mark",
        )
    if set(mark_rois) != included:
        return _failure(
            "missing_confirmed_strategy_mark",
            "correction",
            "Every included ROI must have exactly one confirmed mark.",
            detail_code="confirmed_mark_coverage_mismatch",
        )
    if (
        not correction_facts.per_roi_production_strategy_map
        and {
            mark.selected_dynamic_fit_mode
            for mark in correction_facts.confirmed_marks
        }
        != {global_mode}
    ):
        return _failure(
            "mixed_dynamic_fit_modes",
            "correction",
            "Confirmed marks must resolve to one global dynamic-fit mode.",
            detail_code="confirmed_mark_modes_mixed",
        )
    if any(
        mark.explicit_user_mark is not True or mark.current is not True
        for mark in correction_facts.confirmed_marks
    ):
        return _failure(
            "stale_strategy_mark",
            "correction",
            "Confirmed marks must be explicit and current.",
            detail_code="confirmed_mark_not_current",
        )

    cache_facts = facts.diagnostic_cache
    evidence_facts = facts.evidence_references
    if not evidence_facts.complete or not evidence_facts.references:
        return _failure(
            "evidence_reference_missing_or_stale",
            "diagnostic_evidence",
            "Complete current evidence references are required.",
            detail_code="evidence_facts_incomplete",
        )
    evidence_by_roi = {
        reference.roi_id: reference for reference in evidence_facts.references
    }
    if set(evidence_by_roi) != included:
        return _failure(
            "evidence_reference_missing_or_stale",
            "diagnostic_evidence",
            "Evidence references must cover every included ROI exactly once.",
            detail_code="evidence_roi_coverage_mismatch",
        )

    if cache_facts.available:
        if (
            cache_facts.resolver_status != "current"
            or cache_facts.preliminary_cache is not True
            or cache_facts.production_analysis is not False
        ):
            return _failure(
                "missing_or_stale_diagnostic_cache",
                "diagnostic_evidence",
                "A current preliminary diagnostic cache is required.",
                detail_code="diagnostic_cache_unavailable",
            )
        if (
            cache_facts.completed_run_rejection_category
            != "guided_diagnostic_cache_ineligible"
        ):
            return _failure(
                "diagnostic_cache_not_completed_run_ineligible",
                "diagnostic_evidence",
                "Diagnostic cache lacks the required completed-run rejection category.",
                detail_code="completed_run_rejection_mismatch",
            )
        if (
            not cache_facts.cache_id
            or not cache_facts.cache_root_canonical
            or not cache_facts.source_setup_signature
            or not cache_facts.diagnostic_scope_signature
            or not cache_facts.build_request_signature
            or not cache_facts.artifact_semantic_digest
            or not cache_facts.provenance_semantic_digest
            or dataset_facts.dataset_source_setup_signature
            != cache_facts.source_setup_signature
            or dataset_facts.diagnostic_cache_contract_identity
            != cache_facts.build_request_signature
        ):
            return _failure(
                "diagnostic_cache_identity_mismatch",
                "diagnostic_evidence",
                "Diagnostic cache identity is incomplete or disagrees with dataset facts.",
                detail_code="diagnostic_cache_identity_mismatch",
            )
        for mark in correction_facts.confirmed_marks:
            reference = evidence_by_roi.get(mark.roi_id)
            if (
                reference is None
                or reference.current is not True
                or reference.evidence_reference_id != mark.evidence_reference_id
                or reference.selected_dynamic_fit_mode
                != mark.selected_dynamic_fit_mode
                or reference.diagnostic_cache_id != cache_facts.cache_id
                or mark.diagnostic_cache_id != cache_facts.cache_id
                or reference.source_setup_signature
                != cache_facts.source_setup_signature
                or mark.source_setup_signature != cache_facts.source_setup_signature
                or reference.diagnostic_scope_signature
                != cache_facts.diagnostic_scope_signature
                or mark.diagnostic_scope_signature
                != cache_facts.diagnostic_scope_signature
                or reference.build_request_signature
                != cache_facts.build_request_signature
                or mark.build_request_signature != cache_facts.build_request_signature
            ):
                return _failure(
                    "diagnostic_cache_identity_mismatch",
                    "diagnostic_evidence",
                    "Correction marks and evidence references do not bind to cache identity.",
                    detail_code="mark_evidence_cache_binding_mismatch",
                )
    else:
        # Cache-free local-preview evidence path: currentness is proven by
        # a source/setup signature bound to the plan's resolved input
        # source, acquisition/session settings, and included-ROI scope
        # (see compute_guided_local_preview_source_setup_signature), not by
        # diagnostic-cache identity.
        for mark in correction_facts.confirmed_marks:
            reference = evidence_by_roi.get(mark.roi_id)
            if (
                reference is None
                or reference.current is not True
                or reference.evidence_kind != "local_correction_preview"
                or reference.evidence_reference_id != mark.evidence_reference_id
                or reference.selected_dynamic_fit_mode
                != mark.selected_dynamic_fit_mode
                or reference.diagnostic_cache_id
                or mark.diagnostic_cache_id
                or not reference.source_setup_signature
                or reference.source_setup_signature != mark.source_setup_signature
            ):
                return _failure(
                    "local_preview_setup_signature_mismatch",
                    "diagnostic_evidence",
                    "Correction marks and evidence references do not bind to "
                    "a consistent local-preview source/setup signature.",
                    detail_code="mark_evidence_local_preview_binding_mismatch",
                )

    feature_facts = facts.feature_event
    if (
        not feature_facts.available
        or not is_saved_feature_event_profile_current(
            feature_facts.profile_status, feature_facts.explicitly_applied
        )
        or feature_facts.current is not True
        or feature_facts.visible_unapplied_changes is not False
        or not feature_facts.profile_schema_version
        or not feature_facts.profile_id
        or not feature_facts.effective_values
        or not feature_facts.active_fields
        or feature_facts.validation_issue_categories
        or feature_facts.stale_reason_categories
    ):
        return _failure(
            "invalid_feature_event_profile",
            "feature_event",
            "Request-ready feature/event facts are required.",
            detail_code="feature_event_facts_invalid",
        )
    if any(
        value.source_classification == "unresolved"
        for value in feature_facts.effective_values
    ):
        return _failure(
            "feature_event_effective_value_unresolved",
            "feature_event",
            "Feature/event facts contain unresolved effective values.",
            detail_code="feature_event_value_unresolved",
        )

    output_facts = facts.output
    if not output_facts.available:
        return _failure(
            "missing_output_policy",
            "output",
            "Request-ready output facts are required.",
            detail_code="output_facts_unavailable",
        )
    if output_facts.policy_status != "applied" or output_facts.policy_current is not True:
        return _failure(
            "stale_output_policy",
            "output",
            "Output policy facts are not current and applied.",
            detail_code="output_policy_not_current",
        )
    if output_facts.overwrite is not False:
        return _failure(
            "overwrite_not_allowed",
            "output",
            "Overwrite is prohibited.",
            detail_code="output_overwrite_true",
        )
    if (
        output_facts.blocker_categories
        or output_facts.protected_root_context_complete is not True
    ):
        return _failure(
            "protected_root_context_incomplete",
            "output",
            "Output safety facts contain blockers or incomplete protected-root context.",
            detail_code="output_safety_incomplete",
        )
    if (
        not output_facts.output_base_canonical
        or not output_facts.output_base_path_style
        or output_facts.path_role != "output_base"
        or output_facts.future_output_owner != "runner"
        or output_facts.run_directory_strategy
        != "derive_unique_run_id_under_output_base"
        or output_facts.creation_timing != "future_execution_start_only"
        or output_facts.precreate is not False
        or not output_facts.safety_classifier_version
        or not output_facts.filesystem_fact_scope
    ):
        return _failure(
            "unsupported_request_field",
            "output",
            "Output facts are outside the supported runner-owned future mapping.",
            detail_code="output_ownership_unsupported",
        )

    if is_npm:
        try:
            from photometry_pipeline.guided_normalized_recording import (
                compute_normalized_recording_description_identity,
                deserialize_normalized_recording_description,
            )

            normalized_description = deserialize_normalized_recording_description(
                facts.normalized_recording_description
            )
            if (
                compute_normalized_recording_description_identity(normalized_description)
                != facts.normalized_recording_description_identity
                or normalized_description.adapter_format != "npm"
                or any(
                    session.disposition != "process"
                    for session in normalized_description.sessions
                )
                or source_facts.approved_missing_candidates
            ):
                raise ValueError("NPM normalized disposition facts are not process-only.")
        except Exception:
            return _failure(
                "incomplete_materialized_facts",
                "normalized_recording",
                "The NPM recording sessions could not all be confirmed for "
                "processing from the current settings. Return to the NPM settings "
                "step and rerun Setup check.",
                detail_code="npm_normalized_disposition_invalid",
            )

    try:
        source_request = GuidedBackendSourceRequest(
            source_root_canonical=source_facts.source_root_canonical,
            source_root_path_style=source_facts.source_root_path_style,
            source_format=draft.input_format,
            snapshot_schema_name=(
                GUIDED_BACKEND_NPM_SOURCE_SNAPSHOT_SCHEMA_NAME
                if is_npm
                else GUIDED_BACKEND_SOURCE_SNAPSHOT_SCHEMA_NAME
            ),
            snapshot_schema_version=GUIDED_BACKEND_SOURCE_SNAPSHOT_SCHEMA_VERSION,
            discovery_rule_version=(
                GUIDED_BACKEND_NPM_SOURCE_DISCOVERY_RULE_VERSION
                if is_npm
                else GUIDED_BACKEND_SOURCE_DISCOVERY_RULE_VERSION
            ),
            path_canonicalization_version=CANONICALIZATION_ALGORITHM_VERSION,
            relative_path_rule_version=GUIDED_BACKEND_SOURCE_RELATIVE_PATH_RULE_VERSION,
            ignored_files_policy=(
                GUIDED_BACKEND_NPM_SOURCE_IGNORED_FILES_POLICY
                if is_npm
                else GUIDED_BACKEND_SOURCE_IGNORED_FILES_POLICY
            ),
            build_mode="read_only",
            source_candidate_set_digest=source_facts.source_candidate_set_digest,
            source_candidate_content_digest=(
                source_facts.source_candidate_content_digest
            ),
            candidate_files=source_facts.candidate_files,
            approved_missing_candidates=source_facts.approved_missing_candidates,
            unresolved_source_identity_inputs=(),
            source_identity_level="content_bound_candidate_snapshot",
        )
        if is_npm:
            acquisition_request = GuidedBackendNpmAcquisitionDatasetRequest(
                acquisition_mode=dataset_facts.acquisition_mode,
                sessions_per_hour=dataset_facts.sessions_per_hour,
                session_duration_sec=dataset_facts.session_duration_sec,
                timeline_anchor_mode=dataset_facts.timeline_anchor_mode,
                fixed_daily_anchor_clock=dataset_facts.fixed_daily_anchor_clock,
                allow_partial_final_window=dataset_facts.allow_partial_final_window,
                dataset_snapshot_schema_version=dataset_facts.dataset_snapshot_schema_version,
                dataset_status=dataset_facts.dataset_status,
                dataset_current_applied=dataset_facts.dataset_current_applied,
                semantic_values=dataset_facts.semantic_values,
                dataset_source_setup_signature=dataset_facts.dataset_source_setup_signature,
                diagnostic_cache_contract_identity=dataset_facts.diagnostic_cache_contract_identity,
                execution_mode=dataset_facts.execution_mode,
                validation_issue_categories=dataset_facts.validation_issue_categories,
                stale_reason_categories=dataset_facts.stale_reason_categories,
                npm_time_axis=dataset_facts.npm_time_axis,
                npm_system_ts_col=dataset_facts.npm_system_ts_col,
                npm_computer_ts_col=dataset_facts.npm_computer_ts_col,
                npm_led_col=dataset_facts.npm_led_col,
                npm_region_prefix=dataset_facts.npm_region_prefix,
                npm_region_suffix=dataset_facts.npm_region_suffix,
                npm_target_fs_hz=dataset_facts.npm_target_fs_hz,
                npm_adapter_value_nan_policy=dataset_facts.npm_adapter_value_nan_policy,
                disposition_policy=GuidedBackendDispositionPolicyRequest(
                    schema_name=GUIDED_BACKEND_DISPOSITION_POLICY_SCHEMA_NAME,
                    schema_version=GUIDED_BACKEND_DISPOSITION_POLICY_SCHEMA_VERSION,
                    admitted_dispositions=("process",),
                    missing_session_policy="unsupported",
                    excluded_session_policy="unsupported",
                    partial_support_owner="parser_contract",
                ),
            )
        else:
            acquisition_request = GuidedBackendAcquisitionDatasetRequest(
                acquisition_mode=dataset_facts.acquisition_mode,
                sessions_per_hour=dataset_facts.sessions_per_hour,
                session_duration_sec=dataset_facts.session_duration_sec,
                timeline_anchor_mode=dataset_facts.timeline_anchor_mode,
                fixed_daily_anchor_clock=dataset_facts.fixed_daily_anchor_clock,
                allow_partial_final_window=dataset_facts.allow_partial_final_window,
                exclude_incomplete_final_rwd_chunk=dataset_facts.exclude_incomplete_final_rwd_chunk,
                classification_schema_name=GUIDED_BACKEND_INCOMPLETE_FINAL_SCHEMA_NAME,
                classification_schema_version=GUIDED_BACKEND_INCOMPLETE_FINAL_SCHEMA_VERSION,
                classifier_version=GUIDED_BACKEND_INCOMPLETE_FINAL_CLASSIFIER_VERSION,
                classification_status=incomplete_facts.classification_status,
                not_requested_classification_digest=incomplete_facts.classification_digest,
                dataset_snapshot_schema_version=dataset_facts.dataset_snapshot_schema_version,
                dataset_status=dataset_facts.dataset_status,
                dataset_current_applied=dataset_facts.dataset_current_applied,
                rwd_time_col=dataset_facts.rwd_time_col,
                uv_suffix=dataset_facts.uv_suffix,
                sig_suffix=dataset_facts.sig_suffix,
                semantic_values=dataset_facts.semantic_values,
                dataset_source_setup_signature=dataset_facts.dataset_source_setup_signature,
                diagnostic_cache_contract_identity=dataset_facts.diagnostic_cache_contract_identity,
                execution_mode=dataset_facts.execution_mode,
                validation_issue_categories=dataset_facts.validation_issue_categories,
                stale_reason_categories=dataset_facts.stale_reason_categories,
            )
        if is_npm:
            parser_request = GuidedBackendNpmParserRequest(
                schema_name=parser_facts.schema_name,
                schema_version=parser_facts.schema_version,
                timestamp_column_candidates=parser_facts.npm_timestamp_column_candidates,
                parser_contract_digest=parser_facts.parser_contract_digest,
                parser_contract_content=parser_facts.npm_parser_contract_content or {},
                unresolved_inputs=parser_facts.unresolved_inputs,
            )
        else:
            parser_request = GuidedBackendRwdParserRequest(
                schema_name=parser_facts.schema_name,
                schema_version=parser_facts.schema_version,
                header_search_line_limit=parser_facts.header_search_line_limit,
                time_column_candidates=parser_facts.time_column_candidates,
                uv_suffix_candidates=parser_facts.uv_suffix_candidates,
                signal_suffix_candidates=parser_facts.signal_suffix_candidates,
                column_normalization_rule=parser_facts.column_normalization_rule,
                roi_name_rule=parser_facts.roi_name_rule,
                ambiguity_policy=parser_facts.ambiguity_policy,
                parser_contract_digest=parser_facts.parser_contract_digest,
                unresolved_inputs=parser_facts.unresolved_inputs,
            )
        roi_request = GuidedBackendRoiScopeRequest(
            discovered_roi_ids=roi_facts.discovered_roi_ids,
            included_roi_ids=roi_facts.included_roi_ids,
            excluded_roi_ids=roi_facts.excluded_roi_ids,
            selection_mode=roi_facts.selection_mode,
            inventory_status=roi_facts.inventory_status,
            inventory_source_content_digest=(
                roi_facts.inventory_source_content_digest
            ),
            roi_inventory_identity_status=roi_facts.roi_inventory_identity_status,
        )
        confirmed_marks = tuple(
            GuidedBackendConfirmedStrategyMark(
                roi_id=mark.roi_id,
                selected_dynamic_fit_mode=mark.selected_dynamic_fit_mode,
                diagnostic_cache_id=mark.diagnostic_cache_id,
                source_setup_signature=mark.source_setup_signature,
                diagnostic_scope_signature=mark.diagnostic_scope_signature,
                build_request_signature=mark.build_request_signature,
                evidence_reference_id=mark.evidence_reference_id,
                evidence_chunk=mark.evidence_chunk,
                explicit_user_mark=mark.explicit_user_mark,
                current=mark.current,
            )
            for mark in correction_facts.confirmed_marks
        )
        correction_request = GuidedBackendCorrectionRequest(
            strategy_scope=correction_facts.strategy_scope,
            global_correction_strategy=(
                correction_facts.global_correction_strategy
            ),
            global_dynamic_fit_mode=correction_facts.global_dynamic_fit_mode,
            dynamic_fit_parameter_values=(
                correction_facts.dynamic_fit_parameter_values
            ),
            confirmed_marks=confirmed_marks,
            mark_rule_version=correction_facts.mark_rule_version,
            currentness_rule_version=correction_facts.currentness_rule_version,
            unanimity_rule_version=correction_facts.unanimity_rule_version,
            blocked_strategy_states=correction_facts.blocked_strategy_states,
            production_strategy_map_version=(
                correction_facts.production_strategy_map_version
            ),
            per_roi_production_strategy_map=(
                correction_facts.per_roi_production_strategy_map
            ),
            applied_dff_orchestration_enabled=draft.applied_dff_orchestration_enabled,
        )
        if cache_facts.available:
            diagnostic_request = GuidedBackendDiagnosticEvidenceRequest(
                cache_id=cache_facts.cache_id,
                cache_root_canonical=cache_facts.cache_root_canonical,
                source_setup_signature=cache_facts.source_setup_signature,
                diagnostic_scope_signature=cache_facts.diagnostic_scope_signature,
                build_request_signature=cache_facts.build_request_signature,
                artifact_contract_version=(
                    GUIDED_BACKEND_DIAGNOSTIC_CACHE_SCHEMA_VERSION
                ),
                provenance_schema_version=(
                    GUIDED_BACKEND_DIAGNOSTIC_CACHE_SCHEMA_VERSION
                ),
                artifact_semantic_digest=cache_facts.artifact_semantic_digest,
                provenance_semantic_digest=cache_facts.provenance_semantic_digest,
                evidence_references=evidence_facts.references,
                completed_run_rejection_category=(
                    cache_facts.completed_run_rejection_category
                ),
                resolver_status=cache_facts.resolver_status,
                preliminary_cache=cache_facts.preliminary_cache,
                production_analysis=cache_facts.production_analysis,
                stale_reasons=(),
                unresolved_inputs=(),
                available=True,
            )
        else:
            diagnostic_request = GuidedBackendDiagnosticEvidenceRequest(
                cache_id="",
                cache_root_canonical="",
                source_setup_signature="",
                diagnostic_scope_signature="",
                build_request_signature="",
                artifact_contract_version="",
                provenance_schema_version="",
                artifact_semantic_digest="",
                provenance_semantic_digest="",
                evidence_references=evidence_facts.references,
                completed_run_rejection_category="",
                resolver_status="",
                preliminary_cache=False,
                production_analysis=False,
                stale_reasons=(),
                unresolved_inputs=(),
                available=False,
            )
        feature_request = GuidedBackendFeatureEventRequest(
            profile_schema_version=feature_facts.profile_schema_version,
            profile_id=feature_facts.profile_id,
            effective_values=feature_facts.effective_values,
            active_fields=feature_facts.active_fields,
            inactive_fields=feature_facts.inactive_fields,
            profile_status=feature_facts.profile_status,
            explicitly_applied=feature_facts.explicitly_applied,
            current=feature_facts.current,
            visible_unapplied_changes=feature_facts.visible_unapplied_changes,
            validation_issue_categories=feature_facts.validation_issue_categories,
            stale_reason_categories=feature_facts.stale_reason_categories,
            per_roi_feature_event_map_version=(
                feature_facts.per_roi_feature_event_map_version
            ),
            per_roi_feature_event_map=feature_facts.per_roi_feature_event_map,
        )
        output_request = GuidedBackendOutputRequest(
            output_base_canonical=output_facts.output_base_canonical,
            output_base_path_style=output_facts.output_base_path_style,
            path_role=output_facts.path_role,
            future_output_owner=output_facts.future_output_owner,
            run_directory_strategy=output_facts.run_directory_strategy,
            creation_timing=output_facts.creation_timing,
            overwrite=output_facts.overwrite,
            precreate=output_facts.precreate,
            policy_status=output_facts.policy_status,
            policy_current=output_facts.policy_current,
            safety_classifier_version=output_facts.safety_classifier_version,
            relationships=output_facts.relationships,
            protected_root_context_complete=(
                output_facts.protected_root_context_complete
            ),
            blocker_categories=output_facts.blocker_categories,
            filesystem_fact_scope=output_facts.filesystem_fact_scope,
        )
        local_contract = GuidedBackendLocalContractState(
            local_check_contract_version=GUIDED_BACKEND_LOCAL_CHECK_CONTRACT_VERSION,
            blocking_issue_categories=(),
            warning_categories=(),
            unsupported_state_flags=(),
            unresolved_required_inputs=(),
            deferred_capabilities=(
                "backend_validation",
                "run_authorization",
                "app_build_identity",
                "full_source_manifest_identity",
                "strict_roi_inventory_identity",
            ),
        )
        request = GuidedBackendValidationRequest(
            request_schema_name=GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME,
            request_schema_version=GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION,
            validation_scope=GUIDED_BACKEND_VALIDATION_SCOPE,
            validation_contract_version=(
                GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION
            ),
            validator_capability_version=(
                validator_contract.validator_capability_version
            ),
            compiler_version=GUIDED_BACKEND_VALIDATION_COMPILER_VERSION,
            subset_rule_version=GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION,
            canonicalization_algorithm_version=(
                CANONICALIZATION_ALGORITHM_VERSION
            ),
            source=source_request,
            acquisition_dataset=acquisition_request,
            parser=parser_request,
            roi_scope=roi_request,
            correction=correction_request,
            diagnostic_evidence=diagnostic_request,
            feature_event=feature_request,
            output=output_request,
            local_contract=local_contract,
            normalized_recording_description_identity=(
                facts.normalized_recording_description_identity
            ),
            normalized_recording_description=(
                facts.normalized_recording_description if is_npm else None
            ),
        )
    except Exception:
        return _failure(
            "compiler_internal_error",
            "compiler",
            "Request construction failed after completeness gates passed.",
            detail_code="request_contract_construction_failed",
        )
    try:
        canonical_request_identity = (
            compute_guided_backend_validation_request_identity(request)
        )
    except Exception:
        return _failure(
            "compiler_internal_error",
            "compiler",
            "Canonical request identity computation failed.",
            detail_code="request_identity_computation_failed",
        )
    return GuidedBackendValidationCompileSuccess(
        request=request,
        canonical_request_identity=canonical_request_identity,
        request_identity_deferred=False,
    )


_GUIDED_BACKEND_VALIDATION_IDENTITY_FIELDS = {
    GuidedBackendTypedFieldValue: (
        "field_name",
        "value_type",
        "value",
        "source_classification",
    ),
    GuidedBackendSourceCandidateFile: (
        "canonical_relative_path",
        "size_bytes",
        "sha256_content_digest",
    ),
    GuidedBackendSourceRequest: (
        "source_root_canonical",
        "source_root_path_style",
        "source_format",
        "snapshot_schema_name",
        "snapshot_schema_version",
        "discovery_rule_version",
        "path_canonicalization_version",
        "relative_path_rule_version",
        "ignored_files_policy",
        "build_mode",
        "source_candidate_set_digest",
        "source_candidate_content_digest",
        "candidate_files",
        "approved_missing_candidates",
        "unresolved_source_identity_inputs",
        "source_identity_level",
    ),
    GuidedBackendAcquisitionDatasetRequest: (
        "acquisition_mode",
        "sessions_per_hour",
        "session_duration_sec",
        "timeline_anchor_mode",
        "fixed_daily_anchor_clock",
        "allow_partial_final_window",
        "exclude_incomplete_final_rwd_chunk",
        "classification_schema_name",
        "classification_schema_version",
        "classifier_version",
        "classification_status",
        "not_requested_classification_digest",
        "dataset_snapshot_schema_version",
        "dataset_status",
        "dataset_current_applied",
        "rwd_time_col",
        "uv_suffix",
        "sig_suffix",
        "semantic_values",
        "dataset_source_setup_signature",
        "diagnostic_cache_contract_identity",
        "validation_issue_categories",
        "stale_reason_categories",
        "execution_mode",
    ),
    GuidedBackendDispositionPolicyRequest: (
        "schema_name",
        "schema_version",
        "admitted_dispositions",
        "missing_session_policy",
        "excluded_session_policy",
        "partial_support_owner",
    ),
    GuidedBackendNpmAcquisitionDatasetRequest: (
        "acquisition_mode",
        "sessions_per_hour",
        "session_duration_sec",
        "timeline_anchor_mode",
        "fixed_daily_anchor_clock",
        "allow_partial_final_window",
        "dataset_snapshot_schema_version",
        "dataset_status",
        "dataset_current_applied",
        "semantic_values",
        "dataset_source_setup_signature",
        "diagnostic_cache_contract_identity",
        "validation_issue_categories",
        "stale_reason_categories",
        "execution_mode",
        "source_format",
        "npm_time_axis",
        "npm_system_ts_col",
        "npm_computer_ts_col",
        "npm_led_col",
        "npm_region_prefix",
        "npm_region_suffix",
        "npm_target_fs_hz",
        "npm_adapter_value_nan_policy",
        "disposition_policy",
    ),
    GuidedBackendRwdParserRequest: (
        "schema_name",
        "schema_version",
        "header_search_line_limit",
        "time_column_candidates",
        "uv_suffix_candidates",
        "signal_suffix_candidates",
        "column_normalization_rule",
        "roi_name_rule",
        "ambiguity_policy",
        "parser_contract_digest",
        "unresolved_inputs",
    ),
    GuidedBackendRoiScopeRequest: (
        "discovered_roi_ids",
        "included_roi_ids",
        "excluded_roi_ids",
        "selection_mode",
        "inventory_status",
        "inventory_source_content_digest",
        "roi_inventory_identity_status",
    ),
    GuidedBackendConfirmedStrategyMark: (
        "roi_id",
        "selected_dynamic_fit_mode",
        "diagnostic_cache_id",
        "source_setup_signature",
        "diagnostic_scope_signature",
        "build_request_signature",
        "evidence_reference_id",
        "evidence_chunk",
        "explicit_user_mark",
        "current",
    ),
    GuidedBackendPerRoiProductionStrategy: (
        "roi_id",
        "strategy_family",
        "dynamic_fit_mode",
        "selected_strategy",
        "evidence_source_type",
        "evidence_reference_json",
        "explicit_user_mark",
        "current_or_stale",
    ),
    GuidedBackendCorrectionRequest: (
        "strategy_scope",
        "global_correction_strategy",
        "global_dynamic_fit_mode",
        "dynamic_fit_parameter_values",
        "confirmed_marks",
        "mark_rule_version",
        "currentness_rule_version",
        "unanimity_rule_version",
        "blocked_strategy_states",
        "production_strategy_map_version",
        "per_roi_production_strategy_map",
        "applied_dff_orchestration_enabled",
    ),
    GuidedBackendEvidenceReference: (
        "evidence_reference_id",
        "evidence_kind",
        "diagnostic_cache_id",
        "source_setup_signature",
        "current",
        "diagnostic_scope_signature",
        "build_request_signature",
        "evidence_chunk",
        "roi_id",
        "selected_dynamic_fit_mode",
    ),
    GuidedBackendDiagnosticEvidenceRequest: (
        "cache_id",
        "cache_root_canonical",
        "source_setup_signature",
        "diagnostic_scope_signature",
        "build_request_signature",
        "artifact_contract_version",
        "provenance_schema_version",
        "artifact_semantic_digest",
        "provenance_semantic_digest",
        "evidence_references",
        "completed_run_rejection_category",
        "resolver_status",
        "preliminary_cache",
        "production_analysis",
        "stale_reasons",
        "unresolved_inputs",
        "available",
    ),
    GuidedBackendPerRoiFeatureEvent: (
        "roi_id",
        "source",
        "feature_event_profile_id",
        "override_config_fields",
        "effective_config_fields",
        "explicit_user_mark",
        "current_or_stale",
    ),
    GuidedBackendFeatureEventRequest: (
        "profile_schema_version",
        "profile_id",
        "effective_values",
        "active_fields",
        "inactive_fields",
        "profile_status",
        "explicitly_applied",
        "current",
        "visible_unapplied_changes",
        "validation_issue_categories",
        "stale_reason_categories",
        "per_roi_feature_event_map_version",
        "per_roi_feature_event_map",
    ),
    GuidedBackendOutputRelationship: (
        "relationship",
        "root_kind",
        "status",
    ),
    GuidedBackendOutputRequest: (
        "output_base_canonical",
        "output_base_path_style",
        "path_role",
        "future_output_owner",
        "run_directory_strategy",
        "creation_timing",
        "overwrite",
        "precreate",
        "policy_status",
        "policy_current",
        "safety_classifier_version",
        "relationships",
        "protected_root_context_complete",
        "blocker_categories",
        "filesystem_fact_scope",
    ),
    GuidedBackendLocalContractState: (
        "local_check_contract_version",
        "blocking_issue_categories",
        "warning_categories",
        "unsupported_state_flags",
        "unresolved_required_inputs",
        "deferred_capabilities",
    ),
    GuidedBackendValidationRequest: (
        "request_schema_name",
        "request_schema_version",
        "validation_scope",
        "validation_contract_version",
        "validator_capability_version",
        "compiler_version",
        "subset_rule_version",
        "canonicalization_algorithm_version",
        "source",
        "acquisition_dataset",
        "parser",
        "roi_scope",
        "correction",
        "diagnostic_evidence",
        "feature_event",
        "output",
        "local_contract",
        "normalized_recording_description_identity",
        "normalized_recording_description",
    ),
}


def _map_guided_backend_validation_identity_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GuidedBackendValidationRequestContractError(
                "Canonical request identity does not support non-finite floats."
            )
        return value
    if isinstance(value, (tuple, list)):
        return [
            _map_guided_backend_validation_identity_value(item)
            for item in value
        ]
    if isinstance(value, Mapping):
        return {
            str(key): _map_guided_backend_validation_identity_value(item)
            for key, item in value.items()
        }
    if isinstance(value, GuidedBackendNpmParserRequest):
        return {
            field_name: _map_guided_backend_validation_identity_value(
                getattr(value, field_name)
            )
            for field_name in (
                "schema_name",
                "schema_version",
                "timestamp_column_candidates",
                "parser_contract_digest",
                "parser_contract_content",
                "unresolved_inputs",
            )
        }
    if isinstance(value, GuidedBackendNpmAcquisitionDatasetRequest):
        field_names = _GUIDED_BACKEND_VALIDATION_IDENTITY_FIELDS[
            GuidedBackendNpmAcquisitionDatasetRequest
        ]
        return {
            field_name: _map_guided_backend_validation_identity_value(
                getattr(value, field_name)
            )
            for field_name in field_names
        }
    if isinstance(value, GuidedBackendAcquisitionDatasetRequest):
        field_names = _GUIDED_BACKEND_VALIDATION_IDENTITY_FIELDS[
            GuidedBackendAcquisitionDatasetRequest
        ]
        return {
            field_name: _map_guided_backend_validation_identity_value(
                getattr(value, field_name)
            )
            for field_name in field_names
        }
    if isinstance(value, GuidedBackendValidationRequest):
        field_names = _GUIDED_BACKEND_VALIDATION_IDENTITY_FIELDS[
            GuidedBackendValidationRequest
        ]
        if value.normalized_recording_description is None:
            field_names = tuple(
                name
                for name in field_names
                if name != "normalized_recording_description"
            )
        return {
            field_name: _map_guided_backend_validation_identity_value(
                getattr(value, field_name)
            )
            for field_name in field_names
        }
    field_names = _GUIDED_BACKEND_VALIDATION_IDENTITY_FIELDS.get(type(value))
    if field_names is not None:
        return {
            field_name: _map_guided_backend_validation_identity_value(
                getattr(value, field_name)
            )
            for field_name in field_names
        }
    raise GuidedBackendValidationRequestContractError(
        "Canonical request identity encountered an unsupported value type."
    )


def _guided_backend_validation_request_identity_payload(
    request: GuidedBackendValidationRequest,
) -> dict[str, Any]:
    if not isinstance(request, GuidedBackendValidationRequest):
        raise GuidedBackendValidationRequestContractError(
            "request must be a GuidedBackendValidationRequest."
        )
    return {
        "identity_domain": GUIDED_BACKEND_VALIDATION_IDENTITY_DOMAIN,
        "request": _map_guided_backend_validation_identity_value(request),
    }


def compute_guided_backend_validation_request_identity(
    request: GuidedBackendValidationRequest,
) -> str:
    """Return the deterministic content identity of a frozen request."""
    payload = _guided_backend_validation_request_identity_payload(request)
    try:
        encoded = encode_canonical_value(payload)
        digest = hashlib.sha256(encoded).hexdigest()
        _require_sha256(digest, "canonical_request_identity")
    except (GuidedIdentityError, TypeError, ValueError) as exc:
        raise GuidedBackendValidationRequestContractError(
            "Canonical backend validation request identity could not be computed."
        ) from exc
    return digest
