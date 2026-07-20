"""Pure, immutable B2-C2 authority for accepted intermittent NPM intents.

This module deliberately sits after the B2-C1 production-intent boundary.  It
does not inspect source files, discover paths, resolve timestamps, or allocate
outputs.  It only projects the facts already frozen into a
``GuidedNpmProductionExecutionIntent`` into a separately versioned,
non-authorizing execution authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
import ntpath
import posixpath
from typing import Any

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_new_analysis_plan import (
    FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
)
from photometry_pipeline.guided_normalized_recording import (
    NormalizedRecordingError,
    build_normalized_recording_description_payload,
    compute_npm_parser_contract_digest,
    compute_npm_support_policy_identity,
    compute_normalized_recording_description_identity,
    deserialize_normalized_recording_description,
)
from photometry_pipeline.guided_production_mapping import (
    GUIDED_NPM_PRODUCTION_CAPABILITY_STATUS,
    GUIDED_NPM_PRODUCTION_INTENT_SCHEMA_NAME,
    GUIDED_NPM_PRODUCTION_INTENT_SCHEMA_VERSION,
    GUIDED_NPM_PRODUCTION_RUNNER_CONTRACT_VERSION,
    GUIDED_PRODUCTION_MAPPING_CONTRACT_VERSION,
    ApplicationBuildIdentity,
    GuidedNpmProductionExecutionIntent,
    GuidedProductionFeatureEvent,
    GuidedProductionOutputPolicy,
    GuidedProductionOutputRelationship,
    GuidedProductionPerRoiFeatureEvent,
    GuidedProductionPerRoiStrategy,
    GuidedProductionSourceCandidate,
    GuidedProductionTypedValue,
    build_application_build_identity,
    compute_guided_npm_production_execution_intent_identity,
    feature_entry_provenance_valid,
)


GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_NAME = "guided_npm_execution_authority"
GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_VERSION = "v1"
GUIDED_NPM_EXECUTION_AUTHORITY_IDENTITY_DOMAIN = (
    "guided_npm_execution_authority.v1"
)
GUIDED_NPM_EXECUTION_AUTHORITY_CONTRACT_VERSION = (
    "guided_npm_execution_authority.v1"
)
GUIDED_NPM_AUTHORIZATION_STATUS_NOT_AUTHORIZED = "not_authorized"
GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED = "not_materialized"
GUIDED_NPM_CANONICAL_ROI_NAMING_RULE = (
    "normalized_physical_to_canonical_mapping_order"
)


GUIDED_NPM_EXECUTION_AUTHORITY_REFUSAL_CATEGORIES = (
    "intent_missing_or_invalid",
    "intent_schema_unsupported",
    "intent_identity_mismatch",
    "intent_not_npm",
    "intent_not_intermittent",
    "intent_validation_status_invalid",
    "intent_unexpectedly_runnable",
    "intent_startup_state_invalid",
    "recording_policy_missing",
    "recording_policy_malformed",
    "recording_policy_identity_mismatch",
    "session_evidence_missing",
    "session_count_mismatch",
    "session_position_invalid",
    "session_order_mismatch",
    "session_path_invalid",
    "session_path_duplicate",
    "session_candidate_missing",
    "session_candidate_duplicate",
    "session_candidate_extra",
    "session_size_mismatch",
    "session_digest_mismatch",
    "session_timestamp_missing",
    "session_timestamp_column_missing",
    "session_support_geometry_invalid",
    "session_output_time_basis_mismatch",
    "session_identity_mismatch",
    "roi_inventory_invalid",
    "roi_mapping_missing",
    "roi_mapping_duplicate",
    "roi_mapping_not_one_to_one",
    "roi_mapping_inconsistent_across_sessions",
    "selected_roi_unmapped",
    "selected_roi_scope_mismatch",
    "roi_authority_identity_mismatch",
    "correction_authority_incomplete",
    "correction_strategy_unsupported",
    "correction_selected_roi_coverage_mismatch",
    "correction_authority_identity_mismatch",
    "feature_authority_incomplete",
    "feature_selected_roi_coverage_mismatch",
    "feature_authority_identity_mismatch",
    "output_authority_invalid",
    "output_authority_identity_mismatch",
    "authority_identity_mismatch",
    "authority_serialization_invalid",
    "authority_internal_error",
)
_REFUSAL_CATEGORY_SET = frozenset(GUIDED_NPM_EXECUTION_AUTHORITY_REFUSAL_CATEGORIES)
_SHA_HEX = frozenset("0123456789abcdef")
# The set of dynamic-fit strategy IDs this authority accepts must be the
# same canonical set Guided Mode actually offers and Stage 1 mapping
# already validated against (guided_new_analysis_plan.
# FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES) -- not an independently
# hand-maintained duplicate. A second, separately typed allowlist here
# previously drifted from the real strategy IDs (invented a
# "dynamic_fit_" prefix and renamed two strategies), silently refusing
# every genuinely supported NPM correction choice except
# "global_linear_regression".
_SUPPORTED_DYNAMIC_STRATEGIES = frozenset(FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA_HEX


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _scalar(value: Any) -> bool:
    return value is None or (
        isinstance(value, (str, bool, int, float))
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _validate_typed_values(values: tuple[GuidedProductionTypedValue, ...]) -> bool:
    return all(
        isinstance(item, GuidedProductionTypedValue)
        and _text(item.field_name)
        and _text(item.value_type)
        and _text(item.source_classification)
        and _scalar(item.value)
        for item in values
    )


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite values cannot be canonicalized.")
        return value
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Canonical mapping keys must be strings.")
        return {key: _canonical(value[key]) for key in value}
    if hasattr(value, "__dataclass_fields__"):
        return {
            item.name: _canonical(getattr(value, item.name))
            for item in fields(value)
        }
    raise ValueError(f"Unsupported canonical value: {type(value).__name__}.")


def _without_identity(value: Any, identity_field: str) -> dict[str, Any]:
    return {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name != identity_field
    }


def _digest(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(_canonical(value))
    ).hexdigest()


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Required field {key!r} is missing.")
    return mapping[key]


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = _required(mapping, key)
    if not _text(value):
        raise ValueError(f"Required text field {key!r} is invalid.")
    return value


def _required_sha(mapping: Mapping[str, Any], key: str) -> str:
    value = _required(mapping, key)
    if not _sha256(value):
        raise ValueError(f"Required identity field {key!r} is invalid.")
    return value


def _required_tuple(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an ordered sequence.")
    return tuple(value)


class _AuthorityRefusal(ValueError):
    def __init__(
        self,
        category: str,
        section: str,
        message: str,
        detail_code: str,
    ) -> None:
        self.category = category
        self.section = section
        self.message = message
        self.detail_code = detail_code
        super().__init__(message)


def _refuse(
    category: str,
    section: str,
    message: str,
    detail_code: str,
) -> None:
    if category not in _REFUSAL_CATEGORY_SET:
        category = "authority_internal_error"
    raise _AuthorityRefusal(category, section, message, detail_code)


@dataclass(frozen=True)
class GuidedNpmExecutionAuthorityIssue:
    category: str
    section: str
    message: str
    detail_code: str

    def __post_init__(self) -> None:
        if self.category not in _REFUSAL_CATEGORY_SET:
            raise ValueError("Unsupported NPM authority refusal category.")
        if not _text(self.section) or not _text(self.message) or not _text(
            self.detail_code
        ):
            raise ValueError("Authority issues require complete text fields.")


@dataclass(frozen=True)
class GuidedNpmExecutionAuthorityFailure:
    blocking_issues: tuple[GuidedNpmExecutionAuthorityIssue, ...]
    status: str = "refused"

    def __post_init__(self) -> None:
        if not isinstance(self.blocking_issues, tuple) or not self.blocking_issues:
            raise ValueError("Authority failure requires blocking issues.")
        if any(
            not isinstance(item, GuidedNpmExecutionAuthorityIssue)
            for item in self.blocking_issues
        ):
            raise ValueError("Authority failure issues have an invalid type.")


@dataclass(frozen=True)
class GuidedNpmRoiMappingEntry:
    physical_source_column: str
    canonical_roi_id: str

    def __post_init__(self) -> None:
        if not _text(self.physical_source_column) or not _text(self.canonical_roi_id):
            raise ValueError("ROI mapping entries require both identifiers.")


@dataclass(frozen=True)
class GuidedNpmRecordingPolicy:
    parser_policy_identity: str
    parser_policy_content_json: str
    time_axis_mode: str
    ordered_timestamp_candidates: tuple[str, ...]
    timestamp_unit: str
    timestamp_finite_policy: str
    led_state_column: str
    reference_led_value: int | float | str
    signal_led_value: int | float | str
    roi_prefix: str
    roi_suffix: str
    roi_ordering_rule: str
    canonical_roi_naming_rule: str
    support_policy: str
    support_policy_identity: str
    roi_value_nan_policy: str
    target_fs_hz: float
    configured_session_duration_sec: float
    sessions_per_hour: int
    chronology_policy: str
    gap_policy: str
    overlap_policy: str
    output_time_basis: str
    canonical_policy_identity: str

    def __post_init__(self) -> None:
        for name in (
            "parser_policy_identity",
            "support_policy_identity",
            "canonical_policy_identity",
        ):
            if not _sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256.")
        if compute_npm_support_policy_identity(self.support_policy) != self.support_policy_identity:
            raise ValueError("support_policy_identity_mismatch")
        for name in (
            "parser_policy_content_json",
            "time_axis_mode",
            "timestamp_unit",
            "timestamp_finite_policy",
            "led_state_column",
            "roi_prefix",
            "roi_suffix",
            "roi_ordering_rule",
            "canonical_roi_naming_rule",
            "support_policy",
            "roi_value_nan_policy",
            "chronology_policy",
            "gap_policy",
            "overlap_policy",
            "output_time_basis",
        ):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} must be non-empty.")
        if not isinstance(self.ordered_timestamp_candidates, tuple) or not self.ordered_timestamp_candidates:
            raise ValueError("ordered_timestamp_candidates must be a non-empty tuple.")
        if any(not _text(item) for item in self.ordered_timestamp_candidates):
            raise ValueError("Timestamp candidates must be non-empty strings.")
        if not _finite(self.target_fs_hz) or float(self.target_fs_hz) <= 0:
            raise ValueError("target_fs_hz must be positive and finite.")
        if not _finite(self.configured_session_duration_sec) or float(
            self.configured_session_duration_sec
        ) <= 0:
            raise ValueError("configured_session_duration_sec must be positive.")
        if (
            isinstance(self.sessions_per_hour, bool)
            or not isinstance(self.sessions_per_hour, int)
            or self.sessions_per_hour <= 0
        ):
            raise ValueError("sessions_per_hour must be a positive integer.")


@dataclass(frozen=True)
class GuidedNpmAuthorizedSession:
    chronological_position: int
    canonical_relative_path: str
    authorized_absolute_source_reference: str
    size_bytes: int
    sha256_content_digest: str
    authoritative_source_start_time: str
    actual_elapsed_sec: float
    nominal_expected_elapsed_sec: float
    resolved_timestamp_column: str
    resolved_led_column: str | None
    timestamp_unit: str
    source_timing_evidence: str
    physical_roi_inventory: tuple[str, ...]
    physical_to_canonical_roi_mapping: tuple[GuidedNpmRoiMappingEntry, ...]
    overlap_origin_absolute: float
    resolved_support_start_offset_sec: float
    resolved_support_end_offset_sec: float
    resolved_support_start_absolute: float
    resolved_support_end_absolute: float
    observed_support_duration_sec: float
    support_policy: str
    support_policy_identity: str
    output_time_basis: str
    warning_categories: tuple[str, ...]
    canonical_session_identity: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.chronological_position, bool)
            or not isinstance(self.chronological_position, int)
            or self.chronological_position < 0
        ):
            raise ValueError("chronological_position must be non-negative.")
        if not _text(self.canonical_relative_path) or not _text(
            self.authorized_absolute_source_reference
        ):
            raise ValueError("Session paths are required.")
        if (
            self.canonical_relative_path.startswith(("/", "\\"))
            or "\\" in self.canonical_relative_path
            or any(
                part in {"", ".", ".."}
                for part in self.canonical_relative_path.split("/")
            )
        ):
            raise ValueError("canonical_relative_path is not canonical.")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
            or not _sha256(self.sha256_content_digest)
        ):
            raise ValueError("Session source facts are invalid.")
        for name in (
            "authoritative_source_start_time",
            "resolved_timestamp_column",
            "timestamp_unit",
            "source_timing_evidence",
            "support_policy",
            "output_time_basis",
        ):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} is required.")
        if self.resolved_led_column is not None and not _text(self.resolved_led_column):
            raise ValueError("resolved_led_column must be text when present.")
        for name in (
            "actual_elapsed_sec",
            "nominal_expected_elapsed_sec",
            "overlap_origin_absolute",
            "resolved_support_start_offset_sec",
            "resolved_support_end_offset_sec",
            "resolved_support_start_absolute",
            "resolved_support_end_absolute",
            "observed_support_duration_sec",
        ):
            if not _finite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if self.resolved_support_end_offset_sec <= self.resolved_support_start_offset_sec:
            raise ValueError("Support end offset must be greater than start offset.")
        if self.resolved_support_end_absolute <= self.resolved_support_start_absolute:
            raise ValueError("Absolute support end must be greater than start.")
        if self.observed_support_duration_sec < 0:
            raise ValueError("Observed support duration must be non-negative.")
        if not _sha256(self.support_policy_identity) or not _sha256(
            self.canonical_session_identity
        ):
            raise ValueError("Session identities are invalid.")
        if compute_npm_support_policy_identity(self.support_policy) != self.support_policy_identity:
            raise ValueError("session_support_policy_identity_mismatch")
        if not isinstance(self.physical_roi_inventory, tuple) or not self.physical_roi_inventory:
            raise ValueError("physical_roi_inventory must be non-empty.")
        if len(set(self.physical_roi_inventory)) != len(self.physical_roi_inventory):
            raise ValueError("Physical ROI inventory must be unique.")
        if not isinstance(self.physical_to_canonical_roi_mapping, tuple) or (
            len(self.physical_to_canonical_roi_mapping)
            != len(self.physical_roi_inventory)
        ):
            raise ValueError("Session ROI mapping must cover the inventory.")
        if any(
            not isinstance(item, GuidedNpmRoiMappingEntry)
            for item in self.physical_to_canonical_roi_mapping
        ):
            raise ValueError("Session ROI mapping entry type is invalid.")
        if tuple(
            item.physical_source_column
            for item in self.physical_to_canonical_roi_mapping
        ) != self.physical_roi_inventory:
            raise ValueError("Session ROI mapping order must match inventory.")
        if len(
            {
                item.canonical_roi_id
                for item in self.physical_to_canonical_roi_mapping
            }
        ) != len(self.physical_to_canonical_roi_mapping):
            raise ValueError("Session canonical ROI mapping must be unique.")
        if not isinstance(self.warning_categories, tuple) or any(
            not _text(item) for item in self.warning_categories
        ):
            raise ValueError("warning_categories must be a tuple of text.")


@dataclass(frozen=True)
class GuidedNpmRoiAuthority:
    complete_canonical_roi_ids: tuple[str, ...]
    selected_canonical_roi_ids: tuple[str, ...]
    excluded_canonical_roi_ids: tuple[str, ...]
    complete_physical_source_columns: tuple[str, ...]
    physical_to_canonical_roi_mapping: tuple[GuidedNpmRoiMappingEntry, ...]
    selected_physical_source_columns: tuple[str, ...]
    selected_physical_to_canonical_roi_mapping: tuple[GuidedNpmRoiMappingEntry, ...]
    roi_ordering_rule: str
    canonical_roi_naming_rule: str
    canonical_roi_authority_identity: str

    def __post_init__(self) -> None:
        for name in (
            "complete_canonical_roi_ids",
            "selected_canonical_roi_ids",
            "excluded_canonical_roi_ids",
            "complete_physical_source_columns",
            "selected_physical_source_columns",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not _text(item) for item in value):
                raise ValueError(f"{name} must be a tuple of text.")
        if not self.complete_canonical_roi_ids:
            raise ValueError("Complete ROI inventory is required.")
        if len(set(self.complete_canonical_roi_ids)) != len(self.complete_canonical_roi_ids):
            raise ValueError("Complete canonical ROI inventory must be unique.")
        if len(set(self.selected_canonical_roi_ids)) != len(self.selected_canonical_roi_ids):
            raise ValueError("Selected canonical ROIs must be unique.")
        if len(set(self.excluded_canonical_roi_ids)) != len(self.excluded_canonical_roi_ids):
            raise ValueError("Excluded canonical ROIs must be unique.")
        if set(self.selected_canonical_roi_ids) - set(self.complete_canonical_roi_ids):
            raise ValueError("Selected ROI is outside the complete inventory.")
        if set(self.excluded_canonical_roi_ids) - set(self.complete_canonical_roi_ids):
            raise ValueError("Excluded ROI is outside the complete inventory.")
        if set(self.selected_canonical_roi_ids) & set(self.excluded_canonical_roi_ids):
            raise ValueError("Selected and excluded ROI scopes overlap.")
        if set(self.selected_canonical_roi_ids) | set(self.excluded_canonical_roi_ids) != set(
            self.complete_canonical_roi_ids
        ):
            raise ValueError("Selected and excluded ROI scope is incomplete.")
        if not isinstance(self.physical_to_canonical_roi_mapping, tuple) or len(
            self.physical_to_canonical_roi_mapping
        ) != len(self.complete_canonical_roi_ids):
            raise ValueError("Complete ROI mapping is incomplete.")
        if tuple(item.canonical_roi_id for item in self.physical_to_canonical_roi_mapping) != self.complete_canonical_roi_ids:
            raise ValueError("Complete ROI mapping order is not authoritative.")
        if tuple(item.physical_source_column for item in self.physical_to_canonical_roi_mapping) != self.complete_physical_source_columns:
            raise ValueError("Physical inventory does not match its mapping.")
        if len(set(self.complete_physical_source_columns)) != len(self.complete_physical_source_columns):
            raise ValueError("Physical ROI columns must be unique.")
        selected_map = tuple(
            item
            for item in self.physical_to_canonical_roi_mapping
            if item.canonical_roi_id in set(self.selected_canonical_roi_ids)
        )
        if selected_map != self.selected_physical_to_canonical_roi_mapping:
            raise ValueError("Selected ROI mapping was not mechanically derived.")
        if tuple(item.physical_source_column for item in selected_map) != self.selected_physical_source_columns:
            raise ValueError("Selected physical ROI columns were not mechanically derived.")
        if not _text(self.roi_ordering_rule) or not _text(self.canonical_roi_naming_rule):
            raise ValueError("ROI ordering rules are required.")
        if not _sha256(self.canonical_roi_authority_identity):
            raise ValueError("ROI authority identity is invalid.")


@dataclass(frozen=True)
class GuidedNpmCorrectionAuthority:
    selected_canonical_roi_ids: tuple[str, ...]
    correction_parameter_values: tuple[GuidedProductionTypedValue, ...]
    per_roi_correction_strategy_map: tuple[GuidedProductionPerRoiStrategy, ...]
    correction_payload_identity: str
    canonical_correction_authority_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.selected_canonical_roi_ids, tuple) or not self.selected_canonical_roi_ids:
            raise ValueError("Correction selected ROI scope is required.")
        if not isinstance(self.correction_parameter_values, tuple) or not isinstance(
            self.per_roi_correction_strategy_map, tuple
        ):
            raise ValueError("Correction authority sequences must be tuples.")
        if any(
            not isinstance(item, GuidedProductionTypedValue)
            for item in self.correction_parameter_values
        ) or any(
            not isinstance(item, GuidedProductionPerRoiStrategy)
            for item in self.per_roi_correction_strategy_map
        ):
            raise ValueError("Correction authority entry type is invalid.")
        roi_ids = tuple(item.roi_id for item in self.per_roi_correction_strategy_map)
        if len(set(roi_ids)) != len(roi_ids) or set(roi_ids) != set(
            self.selected_canonical_roi_ids
        ):
            raise ValueError("Correction authority ROI coverage is incomplete.")
        if not _sha256(self.correction_payload_identity) or not _sha256(
            self.canonical_correction_authority_identity
        ):
            raise ValueError("Correction authority identities are invalid.")


@dataclass(frozen=True)
class GuidedNpmFeatureAuthority:
    execution_mode: str
    profile_schema_version: str
    profile_id: str
    effective_values: tuple[GuidedProductionTypedValue, ...]
    active_fields: tuple[str, ...]
    inactive_fields: tuple[str, ...]
    profile_status: str
    explicitly_applied: bool
    current: bool
    visible_unapplied_changes: bool
    per_roi_feature_event_map_version: str
    per_roi_feature_event_map: tuple[GuidedProductionPerRoiFeatureEvent, ...]
    selected_canonical_roi_ids: tuple[str, ...]
    feature_payload_identity: str
    inactive_for_execution: bool
    canonical_feature_authority_identity: str

    def __post_init__(self) -> None:
        if self.execution_mode not in {"phasic", "tonic", "both"}:
            raise ValueError("Unsupported feature execution mode.")
        for name in (
            "profile_schema_version",
            "profile_id",
            "profile_status",
            "per_roi_feature_event_map_version",
        ):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} is required.")
        if not isinstance(self.effective_values, tuple) or any(
            not isinstance(item, GuidedProductionTypedValue)
            for item in self.effective_values
        ):
            raise ValueError("Feature effective values are invalid.")
        if not isinstance(self.active_fields, tuple) or not isinstance(
            self.inactive_fields, tuple
        ) or any(not _text(item) for item in self.active_fields + self.inactive_fields):
            raise ValueError("Feature field scopes are invalid.")
        if not isinstance(self.per_roi_feature_event_map, tuple) or any(
            not isinstance(item, GuidedProductionPerRoiFeatureEvent)
            for item in self.per_roi_feature_event_map
        ):
            raise ValueError("Feature per-ROI map is invalid.")
        if not isinstance(self.selected_canonical_roi_ids, tuple) or not self.selected_canonical_roi_ids:
            raise ValueError("Feature selected ROI scope is required.")
        if len({item.roi_id for item in self.per_roi_feature_event_map}) != len(
            self.per_roi_feature_event_map
        ) or {item.roi_id for item in self.per_roi_feature_event_map} != set(
            self.selected_canonical_roi_ids
        ):
            raise ValueError("Feature selected ROI coverage is incomplete.")
        if self.inactive_for_execution is not (self.execution_mode == "tonic"):
            raise ValueError("Tonic feature inactivity marker is inconsistent.")
        if not _sha256(self.feature_payload_identity) or not _sha256(
            self.canonical_feature_authority_identity
        ):
            raise ValueError("Feature authority identities are invalid.")


@dataclass(frozen=True)
class GuidedNpmOutputAuthority:
    output_base_canonical: str
    output_base_path_style: str
    path_role: str
    future_output_owner: str
    run_directory_strategy: str
    creation_timing: str
    overwrite: bool
    precreate: bool
    safety_classifier_version: str
    relationships: tuple[GuidedProductionOutputRelationship, ...]
    protected_root_context_complete: bool
    filesystem_fact_scope: str
    canonical_output_authority_identity: str

    def __post_init__(self) -> None:
        for name in (
            "output_base_canonical",
            "output_base_path_style",
            "path_role",
            "future_output_owner",
            "run_directory_strategy",
            "creation_timing",
            "safety_classifier_version",
            "filesystem_fact_scope",
        ):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} is required.")
        if not isinstance(self.relationships, tuple) or any(
            not isinstance(item, GuidedProductionOutputRelationship)
            for item in self.relationships
        ):
            raise ValueError("Output relationships are invalid.")
        if self.overwrite is not False or self.precreate is not False:
            raise ValueError("B2-C2 output authority cannot mutate output state.")
        if not _sha256(self.canonical_output_authority_identity):
            raise ValueError("Output authority identity is invalid.")


@dataclass(frozen=True)
class GuidedNpmExecutionAuthority:
    authority_schema_name: str
    authority_schema_version: str
    authority_contract_version: str
    source_production_intent_identity: str
    source_request_identity: str
    validation_revision: int
    guided_plan_identity: str
    application_build_identity: ApplicationBuildIdentity
    source_format: str
    acquisition_mode: str
    execution_mode: str
    run_type: str
    recording_policy: GuidedNpmRecordingPolicy
    sessions: tuple[GuidedNpmAuthorizedSession, ...]
    session_sequence_identity: str
    roi_authority: GuidedNpmRoiAuthority
    correction_authority: GuidedNpmCorrectionAuthority
    feature_authority: GuidedNpmFeatureAuthority
    output_authority: GuidedNpmOutputAuthority
    authorization_status: str
    startup_status: str
    runnable: bool
    canonical_authority_identity: str

    def __post_init__(self) -> None:
        if self.authority_schema_name != GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_NAME:
            raise ValueError("Unsupported NPM authority schema name.")
        if self.authority_schema_version != GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_VERSION:
            raise ValueError("Unsupported NPM authority schema version.")
        if self.authority_contract_version != GUIDED_NPM_EXECUTION_AUTHORITY_CONTRACT_VERSION:
            raise ValueError("Unsupported NPM authority contract version.")
        for name in (
            "source_production_intent_identity",
            "source_request_identity",
            "guided_plan_identity",
            "session_sequence_identity",
            "canonical_authority_identity",
        ):
            if not _sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256.")
        if (
            isinstance(self.validation_revision, bool)
            or not isinstance(self.validation_revision, int)
            or self.validation_revision < 0
        ):
            raise ValueError("validation_revision must be non-negative.")
        if self.source_format != "npm" or self.acquisition_mode != "intermittent":
            raise ValueError("Authority must represent intermittent NPM.")
        if self.execution_mode not in {"phasic", "tonic", "both"} or self.run_type != "full":
            raise ValueError("Authority execution subset is unsupported.")
        if not isinstance(self.application_build_identity, ApplicationBuildIdentity):
            raise ValueError("Build identity is required.")
        if not isinstance(self.sessions, tuple) or not self.sessions:
            raise ValueError("Authority requires sessions.")
        if any(not isinstance(item, GuidedNpmAuthorizedSession) for item in self.sessions):
            raise ValueError("Authority session type is invalid.")
        positions = tuple(item.chronological_position for item in self.sessions)
        paths = tuple(item.canonical_relative_path for item in self.sessions)
        if positions != tuple(range(len(self.sessions))) or len(set(paths)) != len(paths):
            raise ValueError("Authority session sequence is not exact.")
        for session in self.sessions:
            if session.resolved_timestamp_column not in self.recording_policy.ordered_timestamp_candidates:
                raise ValueError("resolved_timestamp_column_not_authorized")
            if (
                session.support_policy != self.recording_policy.support_policy
                or session.support_policy_identity
                != self.recording_policy.support_policy_identity
                or compute_npm_support_policy_identity(session.support_policy)
                != session.support_policy_identity
            ):
                raise ValueError("session_support_policy_identity_mismatch")
        if self.authorization_status != GUIDED_NPM_AUTHORIZATION_STATUS_NOT_AUTHORIZED:
            raise ValueError("B2-C2 authority cannot authorize execution.")
        if self.startup_status != GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED:
            raise ValueError("B2-C2 authority cannot claim startup materialization.")
        if self.runnable is not False:
            raise ValueError("B2-C2 authority cannot be runnable.")
        for value in (
            self.recording_policy,
            self.roi_authority,
            self.correction_authority,
            self.feature_authority,
            self.output_authority,
        ):
            if not isinstance(
                value,
                (
                    GuidedNpmRecordingPolicy,
                    GuidedNpmRoiAuthority,
                    GuidedNpmCorrectionAuthority,
                    GuidedNpmFeatureAuthority,
                    GuidedNpmOutputAuthority,
                ),
            ):
                raise ValueError("Authority nested contract type is invalid.")


GuidedNpmExecutionAuthorityResult = (
    GuidedNpmExecutionAuthority | GuidedNpmExecutionAuthorityFailure
)


def compute_guided_npm_recording_policy_identity(
    policy: GuidedNpmRecordingPolicy,
) -> str:
    if not isinstance(policy, GuidedNpmRecordingPolicy):
        raise ValueError("policy must be a GuidedNpmRecordingPolicy.")
    return _digest(
        "guided_npm_recording_policy.v1",
        _without_identity(policy, "canonical_policy_identity"),
    )


def compute_guided_npm_authorized_session_identity(
    session: GuidedNpmAuthorizedSession,
) -> str:
    if not isinstance(session, GuidedNpmAuthorizedSession):
        raise ValueError("session must be a GuidedNpmAuthorizedSession.")
    return _digest(
        "guided_npm_authorized_session.v1",
        _without_identity(session, "canonical_session_identity"),
    )


def compute_guided_npm_session_sequence_identity(
    sessions: tuple[GuidedNpmAuthorizedSession, ...],
) -> str:
    if not isinstance(sessions, tuple):
        raise ValueError("sessions must be a tuple.")
    return _digest(
        "guided_npm_session_sequence.v1",
        tuple(
            {
                "chronological_position": item.chronological_position,
                "canonical_session_identity": item.canonical_session_identity,
            }
            for item in sessions
        ),
    )


def compute_guided_npm_roi_authority_identity(
    authority: GuidedNpmRoiAuthority,
) -> str:
    if not isinstance(authority, GuidedNpmRoiAuthority):
        raise ValueError("authority must be a GuidedNpmRoiAuthority.")
    return _digest(
        "guided_npm_roi_authority.v1",
        _without_identity(authority, "canonical_roi_authority_identity"),
    )


def compute_guided_npm_correction_authority_identity(
    authority: GuidedNpmCorrectionAuthority,
) -> str:
    if not isinstance(authority, GuidedNpmCorrectionAuthority):
        raise ValueError("authority must be a GuidedNpmCorrectionAuthority.")
    return _digest(
        "guided_npm_correction_authority.v1",
        _without_identity(authority, "canonical_correction_authority_identity"),
    )


def compute_guided_npm_feature_authority_identity(
    authority: GuidedNpmFeatureAuthority,
) -> str:
    if not isinstance(authority, GuidedNpmFeatureAuthority):
        raise ValueError("authority must be a GuidedNpmFeatureAuthority.")
    return _digest(
        "guided_npm_feature_authority.v1",
        _without_identity(authority, "canonical_feature_authority_identity"),
    )


def compute_guided_npm_output_authority_identity(
    authority: GuidedNpmOutputAuthority,
) -> str:
    if not isinstance(authority, GuidedNpmOutputAuthority):
        raise ValueError("authority must be a GuidedNpmOutputAuthority.")
    return _digest(
        "guided_npm_output_authority.v1",
        _without_identity(authority, "canonical_output_authority_identity"),
    )


def compute_guided_npm_execution_authority_identity(
    authority: GuidedNpmExecutionAuthority,
) -> str:
    if not isinstance(authority, GuidedNpmExecutionAuthority):
        raise ValueError("authority must be a GuidedNpmExecutionAuthority.")
    return _digest(
        GUIDED_NPM_EXECUTION_AUTHORITY_IDENTITY_DOMAIN,
        _without_identity(authority, "canonical_authority_identity"),
    )


def _verify_authority_identity_chain(authority: GuidedNpmExecutionAuthority) -> None:
    for session in authority.sessions:
        if session.resolved_timestamp_column not in authority.recording_policy.ordered_timestamp_candidates:
            raise ValueError("resolved_timestamp_column_not_authorized")
        if (
            session.support_policy != authority.recording_policy.support_policy
            or session.support_policy_identity
            != authority.recording_policy.support_policy_identity
            or compute_npm_support_policy_identity(session.support_policy)
            != session.support_policy_identity
        ):
            raise ValueError("session_support_policy_identity_mismatch")
    expected_build = build_application_build_identity(
        distribution_name=authority.application_build_identity.distribution_name,
        distribution_version=authority.application_build_identity.distribution_version,
        source_revision_kind=authority.application_build_identity.source_revision_kind,
        source_revision=authority.application_build_identity.source_revision,
        source_tree_state=authority.application_build_identity.source_tree_state,
        source_tree_digest=authority.application_build_identity.source_tree_digest,
        build_artifact_digest=authority.application_build_identity.build_artifact_digest,
        identity_provider_version=authority.application_build_identity.identity_provider_version,
    )
    if expected_build.canonical_identity != authority.application_build_identity.canonical_identity:
        raise ValueError("application_build_identity_mismatch")
    if compute_guided_npm_recording_policy_identity(
        authority.recording_policy
    ) != authority.recording_policy.canonical_policy_identity:
        raise ValueError("recording_policy_identity_mismatch")
    if any(
        compute_guided_npm_authorized_session_identity(item)
        != item.canonical_session_identity
        for item in authority.sessions
    ):
        raise ValueError("session_identity_mismatch")
    if compute_guided_npm_session_sequence_identity(
        authority.sessions
    ) != authority.session_sequence_identity:
        raise ValueError("session_sequence_identity_mismatch")
    if compute_guided_npm_roi_authority_identity(
        authority.roi_authority
    ) != authority.roi_authority.canonical_roi_authority_identity:
        raise ValueError("roi_authority_identity_mismatch")
    if compute_guided_npm_correction_authority_identity(
        authority.correction_authority
    ) != authority.correction_authority.canonical_correction_authority_identity:
        raise ValueError("correction_authority_identity_mismatch")
    if compute_guided_npm_feature_authority_identity(
        authority.feature_authority
    ) != authority.feature_authority.canonical_feature_authority_identity:
        raise ValueError("feature_authority_identity_mismatch")
    if compute_guided_npm_output_authority_identity(
        authority.output_authority
    ) != authority.output_authority.canonical_output_authority_identity:
        raise ValueError("output_authority_identity_mismatch")
    if compute_guided_npm_execution_authority_identity(
        authority
    ) != authority.canonical_authority_identity:
        raise ValueError("authority_identity_mismatch")


def _failure(exc: _AuthorityRefusal) -> GuidedNpmExecutionAuthorityFailure:
    return GuidedNpmExecutionAuthorityFailure(
        (
            GuidedNpmExecutionAuthorityIssue(
                exc.category, exc.section, exc.message, exc.detail_code
            ),
        )
    )


def _lexical_join_source_root(root: str, relative: str) -> str:
    if not _text(root):
        _refuse(
            "session_path_invalid",
            "source",
            "The accepted source root is missing.",
            "source_root_missing",
        )
    if not _text(relative) or relative.startswith(("/", "\\")):
        _refuse(
            "session_path_invalid",
            "source",
            "The canonical relative path is invalid.",
            "canonical_relative_path_invalid",
        )
    if ntpath.splitdrive(relative)[0] or any(
        part in {"", ".", ".."} for part in relative.split("/")
    ) or "\\" in relative:
        _refuse(
            "session_path_invalid",
            "source",
            "The canonical relative path cannot escape the accepted source root.",
            "canonical_relative_path_invalid",
        )
    if not (ntpath.isabs(root) or posixpath.isabs(root)):
        _refuse(
            "session_path_invalid",
            "source",
            "The accepted source root is not absolute.",
            "source_root_not_absolute",
        )
    separator = "\\" if "\\" in root or ntpath.splitdrive(root)[0] else "/"
    if root in {"/", "\\"}:
        base = root
    else:
        base = root.rstrip("/\\")
    return base + ("" if base.endswith(("/", "\\")) else separator) + relative.replace(
        "/", separator
    )


def _parser_policy_from_intent(
    intent: GuidedNpmProductionExecutionIntent,
) -> tuple[GuidedNpmRecordingPolicy, dict[str, Any]]:
    if not _sha256(intent.parser_policy_identity) or not _text(
        intent.parser_policy_content_json
    ):
        _refuse(
            "recording_policy_missing",
            "recording_policy",
            "The accepted parser policy is missing.",
            "parser_policy_missing",
        )
    try:
        payload = json.loads(intent.parser_policy_content_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        _refuse(
            "recording_policy_malformed",
            "recording_policy",
            "The accepted parser policy JSON is malformed.",
            "parser_policy_json_invalid",
        )
    if not isinstance(payload, Mapping):
        _refuse(
            "recording_policy_malformed",
            "recording_policy",
            "The accepted parser policy is not an object.",
            "parser_policy_object_invalid",
        )
    sampling = _required(payload, "sampling")
    if not isinstance(sampling, Mapping):
        _refuse(
            "recording_policy_malformed",
            "recording_policy",
            "The accepted parser policy sampling section is malformed.",
            "parser_policy_sampling_invalid",
        )
    filename = _required(sampling, "filename_chronology")
    if not isinstance(filename, Mapping):
        _refuse(
            "recording_policy_malformed",
            "recording_policy",
            "The accepted chronology policy is malformed.",
            "parser_policy_chronology_invalid",
        )
    led_values = _required(sampling, "led_values")
    if not isinstance(led_values, Mapping):
        _refuse(
            "recording_policy_malformed",
            "recording_policy",
            "The accepted LED selectors are malformed.",
            "parser_policy_led_values_invalid",
        )
    try:
        support_policy = _required_text(sampling, "support_policy")
        support_policy_identity = compute_npm_support_policy_identity(support_policy)
    except (KeyError, TypeError, ValueError, NormalizedRecordingError):
        _refuse(
            "recording_policy_malformed",
            "recording_policy",
            "The accepted support policy is malformed.",
            "support_policy_invalid",
        )
    if intent.support_policy_identity != support_policy_identity:
        _refuse(
            "recording_policy_identity_mismatch",
            "recording_policy",
            "The production intent support-policy identity is not the policy-value identity.",
            "support_policy_identity_mismatch",
        )
    try:
        candidates = tuple(_required(sampling, "timestamp_column_candidates"))
        policy = GuidedNpmRecordingPolicy(
            parser_policy_identity=intent.parser_policy_identity,
            parser_policy_content_json=intent.parser_policy_content_json,
            time_axis_mode=_required_text(sampling, "time_axis"),
            ordered_timestamp_candidates=candidates,
            timestamp_unit=_required_text(sampling, "timestamp_unit"),
            timestamp_finite_policy=_required_text(sampling, "timestamp_finite_policy"),
            led_state_column=_required_text(sampling, "led_column"),
            reference_led_value=_required(led_values, "uv"),
            signal_led_value=_required(led_values, "signal"),
            roi_prefix=_required_text(sampling, "region_prefix"),
            roi_suffix=_required_text(sampling, "region_suffix"),
            roi_ordering_rule=_required_text(sampling, "roi_order_policy"),
            canonical_roi_naming_rule=GUIDED_NPM_CANONICAL_ROI_NAMING_RULE,
            support_policy=support_policy,
            support_policy_identity=support_policy_identity,
            roi_value_nan_policy=_required_text(sampling, "roi_nan_policy"),
            target_fs_hz=float(_required(sampling, "target_fs_hz")),
            configured_session_duration_sec=float(
                _required(sampling, "session_duration_sec")
            ),
            sessions_per_hour=int(intent.sessions_per_hour),
            chronology_policy=_required_text(filename, "sort"),
            gap_policy=_required_text(filename, "gaps"),
            overlap_policy=_required_text(filename, "overlap"),
            output_time_basis=_required_text(sampling, "output_time_basis"),
            canonical_policy_identity="0" * 64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _refuse(
            "recording_policy_malformed",
            "recording_policy",
            "The accepted parser policy contains malformed fields.",
            "parser_policy_fields_invalid",
        )
    if policy.target_fs_hz != float(intent.target_fs_hz) or policy.configured_session_duration_sec != float(
        intent.session_duration_sec
    ):
        _refuse(
            "recording_policy_identity_mismatch",
            "recording_policy",
            "Parser sampling values do not match the production intent.",
            "parser_sampling_intent_mismatch",
        )
    try:
        if compute_npm_parser_contract_digest(payload) != intent.parser_policy_identity:
            _refuse(
                "recording_policy_identity_mismatch",
                "recording_policy",
                "Parser policy content does not match its identity.",
                "parser_policy_identity_mismatch",
            )
    except Exception:
        _refuse(
            "recording_policy_malformed",
            "recording_policy",
            "Parser policy identity could not be recomputed.",
            "parser_policy_identity_unavailable",
        )
    policy = replace(
        policy,
        canonical_policy_identity=compute_guided_npm_recording_policy_identity(policy),
    )
    return policy, payload


def _build_sessions(
    intent: GuidedNpmProductionExecutionIntent,
    normalized: Any,
    policy: GuidedNpmRecordingPolicy,
) -> tuple[GuidedNpmAuthorizedSession, ...]:
    raw_sessions = normalized.adapter_evidence.get("npm_sessions")
    if not isinstance(raw_sessions, (list, tuple)):
        _refuse(
            "session_evidence_missing",
            "sessions",
            "The normalized NPM session evidence is missing.",
            "npm_sessions_missing",
        )
    if len(raw_sessions) != len(normalized.sessions):
        _refuse(
            "session_count_mismatch",
            "sessions",
            "Normalized sessions and resolved evidence have different counts.",
            "session_evidence_count_mismatch",
        )
    candidates = intent.source_candidate_files
    if not isinstance(candidates, tuple) or not candidates:
        _refuse(
            "session_candidate_missing",
            "source_candidates",
            "The production intent has no source candidates.",
            "source_candidates_missing",
        )
    candidate_paths = tuple(item.canonical_relative_path for item in candidates)
    normalized_paths = tuple(item.stable_source_identity for item in normalized.sessions)
    if len(set(candidate_paths)) != len(candidate_paths):
        _refuse(
            "session_candidate_duplicate",
            "source_candidates",
            "The production intent contains duplicate source candidates.",
            "source_candidate_path_duplicate",
        )
    if len(set(normalized_paths)) != len(normalized_paths):
        _refuse(
            "session_path_duplicate",
            "sessions",
            "The normalized recording contains duplicate session paths.",
            "normalized_session_path_duplicate",
        )
    if len(candidates) != len(normalized.sessions):
        _refuse(
            "session_count_mismatch",
            "sessions",
            "Source candidate and normalized session counts differ.",
            "candidate_session_count_mismatch",
        )
    if candidate_paths != normalized_paths:
        missing = set(normalized_paths) - set(candidate_paths)
        extra = set(candidate_paths) - set(normalized_paths)
        if missing:
            _refuse(
                "session_candidate_missing",
                "source_candidates",
                "A normalized session has no matching source candidate.",
                "source_candidate_missing",
            )
        if extra:
            _refuse(
                "session_candidate_extra",
                "source_candidates",
                "A source candidate has no matching normalized session.",
                "source_candidate_extra",
            )
        _refuse(
            "session_order_mismatch",
            "sessions",
            "Source candidate order does not match normalized session order.",
            "source_candidate_order_mismatch",
        )
    mapping_raw = normalized.adapter_evidence.get("physical_to_canonical_roi_mapping")
    if not isinstance(mapping_raw, (list, tuple)) or not mapping_raw:
        _refuse(
            "roi_mapping_missing",
            "roi",
            "The normalized recording has no complete ROI mapping.",
            "roi_mapping_missing",
        )
    global_mapping: tuple[GuidedNpmRoiMappingEntry, ...]
    try:
        global_mapping = tuple(
            GuidedNpmRoiMappingEntry(
                physical_source_column=_required_text(item, "physical_source_column"),
                canonical_roi_id=_required_text(item, "canonical_roi_id"),
            )
            for item in mapping_raw
        )
    except (TypeError, ValueError):
        _refuse(
            "roi_mapping_missing",
            "roi",
            "The normalized ROI mapping is malformed.",
            "roi_mapping_malformed",
        )
    if len({item.physical_source_column for item in global_mapping}) != len(global_mapping) or len(
        {item.canonical_roi_id for item in global_mapping}
    ) != len(global_mapping):
        _refuse(
            "roi_mapping_not_one_to_one",
            "roi",
            "The normalized ROI mapping is not one-to-one.",
            "roi_mapping_not_one_to_one",
        )
    sessions: list[GuidedNpmAuthorizedSession] = []
    for position, (normalized_session, raw, candidate) in enumerate(
        zip(normalized.sessions, raw_sessions, candidates)
    ):
        if normalized_session.chronological_position != position:
            _refuse(
                "session_position_invalid",
                "sessions",
                "Normalized session positions are not contiguous and ordered.",
                "session_position_invalid",
            )
        if not isinstance(raw, Mapping):
            _refuse(
                "session_evidence_missing",
                "sessions",
                "A normalized NPM session evidence record is malformed.",
                "session_evidence_record_invalid",
            )
        for name in (
            "canonical_relative_path",
            "resolved_timestamp_column",
            "timestamp_unit",
            "physical_roi_inventory",
            "physical_to_canonical_roi_mapping",
            "overlap_origin_absolute",
            "resolved_support_start_offset_sec",
            "resolved_support_end_offset_sec",
            "resolved_support_start_absolute",
            "resolved_support_end_absolute",
            "observed_duration_sec",
            "output_time_basis",
            "support_policy",
            "support_policy_identity",
            "warning_categories",
            "actual_elapsed_sec",
            "nominal_expected_elapsed_sec",
        ):
            if name not in raw:
                _refuse(
                    "session_evidence_missing",
                    "sessions",
                    "A required per-session authority fact is missing.",
                    f"session_field_missing_{name}",
                )
        if raw["canonical_relative_path"] != normalized_session.stable_source_identity:
            _refuse(
                "session_order_mismatch",
                "sessions",
                "Per-session evidence path does not match normalized order.",
                "session_evidence_path_mismatch",
            )
        resolved_timestamp_column = raw["resolved_timestamp_column"]
        if (
            not _text(resolved_timestamp_column)
            or resolved_timestamp_column not in policy.ordered_timestamp_candidates
        ):
            _refuse(
                "session_timestamp_column_missing",
                "sessions",
                "The resolved timestamp column is not authorized by the recording policy.",
                "resolved_timestamp_column_not_authorized",
            )
        if candidate.canonical_relative_path != normalized_session.stable_source_identity:
            _refuse(
                "session_order_mismatch",
                "source_candidates",
                "Source candidate path does not match normalized order.",
                "source_candidate_order_mismatch",
            )
        if candidate.size_bytes != normalized_session.size_bytes:
            _refuse(
                "session_size_mismatch",
                "source_candidates",
                "Source candidate size does not match normalized evidence.",
                "source_candidate_size_mismatch",
            )
        if candidate.sha256_content_digest != normalized_session.content_digest:
            _refuse(
                "session_digest_mismatch",
                "source_candidates",
                "Source candidate digest does not match normalized evidence.",
                "source_candidate_digest_mismatch",
            )
        try:
            absolute_reference = _lexical_join_source_root(
                intent.source_root_canonical, normalized_session.stable_source_identity
            )
        except _AuthorityRefusal:
            raise
        if normalized_session.canonical_source_reference != absolute_reference:
            _refuse(
                "session_path_invalid",
                "source",
                "The accepted normalized absolute source reference is not the lexical root join.",
                "absolute_source_reference_mismatch",
            )
        try:
            raw_mapping = tuple(
                GuidedNpmRoiMappingEntry(
                    physical_source_column=_required_text(item, "physical_source_column"),
                    canonical_roi_id=_required_text(item, "canonical_roi_id"),
                )
                for item in raw["physical_to_canonical_roi_mapping"]
            )
        except (TypeError, ValueError):
            _refuse(
                "roi_mapping_missing",
                "roi",
                "A session ROI mapping is malformed.",
                "session_roi_mapping_malformed",
            )
        inventory = tuple(raw["physical_roi_inventory"])
        if raw_mapping != global_mapping or tuple(
            item.physical_source_column for item in raw_mapping
        ) != inventory:
            _refuse(
                "roi_mapping_inconsistent_across_sessions",
                "roi",
                "A session ROI mapping differs from the recording authority.",
                "session_roi_mapping_mismatch",
            )
        if raw["timestamp_unit"] != policy.timestamp_unit:
            _refuse(
                "session_timestamp_column_missing",
                "sessions",
                "A session timestamp unit differs from recording policy.",
                "session_timestamp_unit_mismatch",
            )
        if raw["support_policy"] != policy.support_policy:
            _refuse(
                "session_output_time_basis_mismatch",
                "sessions",
                "Session support policy differs from recording policy.",
                "session_support_policy_mismatch",
            )
        try:
            expected_support_policy_identity = compute_npm_support_policy_identity(
                raw["support_policy"]
            )
        except (TypeError, ValueError, NormalizedRecordingError):
            _refuse(
                "session_support_geometry_invalid",
                "sessions",
                "Session support policy is invalid.",
                "session_support_policy_identity_mismatch",
            )
        if (
            raw["support_policy_identity"] != expected_support_policy_identity
            or raw["support_policy_identity"] != policy.support_policy_identity
        ):
            _refuse(
                "session_support_geometry_invalid",
                "sessions",
                "Session support policy identity does not match its policy value.",
                "session_support_policy_identity_mismatch",
            )
        if raw["output_time_basis"] != policy.output_time_basis:
            _refuse(
                "session_output_time_basis_mismatch",
                "sessions",
                "Session output-time policy differs from recording policy.",
                "session_recording_policy_mismatch",
            )
        try:
            session = GuidedNpmAuthorizedSession(
                chronological_position=position,
                canonical_relative_path=normalized_session.stable_source_identity,
                authorized_absolute_source_reference=absolute_reference,
                size_bytes=candidate.size_bytes,
                sha256_content_digest=candidate.sha256_content_digest,
                authoritative_source_start_time=normalized_session.authoritative_source_start_time,
                actual_elapsed_sec=float(raw["actual_elapsed_sec"]),
                nominal_expected_elapsed_sec=float(raw["nominal_expected_elapsed_sec"]),
                resolved_timestamp_column=_required_text(raw, "resolved_timestamp_column"),
                resolved_led_column=policy.led_state_column,
                timestamp_unit=_required_text(raw, "timestamp_unit"),
                source_timing_evidence=normalized_session.source_timing_evidence,
                physical_roi_inventory=inventory,
                physical_to_canonical_roi_mapping=raw_mapping,
                overlap_origin_absolute=float(raw["overlap_origin_absolute"]),
                resolved_support_start_offset_sec=float(
                    raw["resolved_support_start_offset_sec"]
                ),
                resolved_support_end_offset_sec=float(
                    raw["resolved_support_end_offset_sec"]
                ),
                resolved_support_start_absolute=float(
                    raw["resolved_support_start_absolute"]
                ),
                resolved_support_end_absolute=float(
                    raw["resolved_support_end_absolute"]
                ),
                observed_support_duration_sec=float(raw["observed_duration_sec"]),
                support_policy=_required_text(raw, "support_policy"),
                support_policy_identity=_required_sha(raw, "support_policy_identity"),
                output_time_basis=_required_text(raw, "output_time_basis"),
                warning_categories=tuple(raw["warning_categories"]),
                canonical_session_identity="0" * 64,
            )
        except (TypeError, ValueError, KeyError):
            _refuse(
                "session_support_geometry_invalid",
                "sessions",
                "A normalized NPM session contains malformed authority facts.",
                "session_authority_fields_invalid",
            )
        session = replace(
            session,
            canonical_session_identity=compute_guided_npm_authorized_session_identity(
                session
            ),
        )
        sessions.append(session)
    return tuple(sessions)


def _build_roi_authority(
    intent: GuidedNpmProductionExecutionIntent,
    normalized: Any,
    policy: GuidedNpmRecordingPolicy,
    sessions: tuple[GuidedNpmAuthorizedSession, ...],
) -> GuidedNpmRoiAuthority:
    complete = tuple(intent.discovered_roi_ids)
    selected = tuple(intent.selected_roi_ids)
    excluded = tuple(intent.excluded_roi_ids)
    raw_mapping = normalized.adapter_evidence["physical_to_canonical_roi_mapping"]
    mapping = tuple(
        GuidedNpmRoiMappingEntry(
            physical_source_column=item["physical_source_column"],
            canonical_roi_id=item["canonical_roi_id"],
        )
        for item in raw_mapping
    )
    try:
        authority = GuidedNpmRoiAuthority(
            complete_canonical_roi_ids=complete,
            selected_canonical_roi_ids=selected,
            excluded_canonical_roi_ids=excluded,
            complete_physical_source_columns=tuple(
                item.physical_source_column for item in mapping
            ),
            physical_to_canonical_roi_mapping=mapping,
            selected_physical_source_columns=tuple(
                item.physical_source_column for item in mapping if item.canonical_roi_id in set(selected)
            ),
            selected_physical_to_canonical_roi_mapping=tuple(
                item for item in mapping if item.canonical_roi_id in set(selected)
            ),
            roi_ordering_rule=policy.roi_ordering_rule,
            canonical_roi_naming_rule=policy.canonical_roi_naming_rule,
            canonical_roi_authority_identity="0" * 64,
        )
    except (TypeError, ValueError, KeyError):
        _refuse(
            "roi_inventory_invalid",
            "roi",
            "The accepted ROI inventory or mapping is incomplete.",
            "roi_authority_fields_invalid",
        )
    expected_session_mapping = sessions[0].physical_to_canonical_roi_mapping
    if any(
        item.physical_to_canonical_roi_mapping != expected_session_mapping
        for item in sessions
    ):
        _refuse(
            "roi_mapping_inconsistent_across_sessions",
            "roi",
            "Per-session ROI mapping differs from the recording authority.",
            "session_roi_mapping_mismatch",
        )
    authority = replace(
        authority,
        canonical_roi_authority_identity=compute_guided_npm_roi_authority_identity(
            authority
        ),
    )
    return authority


def _build_correction_authority(
    intent: GuidedNpmProductionExecutionIntent,
) -> GuidedNpmCorrectionAuthority:
    selected = tuple(intent.selected_roi_ids)
    strategy_map = tuple(
        sorted(intent.per_roi_correction_strategy_map, key=lambda item: item.roi_id)
    )
    if len({item.roi_id for item in strategy_map}) != len(strategy_map) or {
        item.roi_id for item in strategy_map
    } != set(selected):
        _refuse(
            "correction_selected_roi_coverage_mismatch",
            "correction",
            "Correction authority does not cover selected ROIs exactly.",
            "correction_selected_roi_coverage_mismatch",
        )
    for entry in strategy_map:
        if (
            entry.strategy_family not in {"signal_only_f0", "dynamic_fit"}
            or not entry.explicit_user_mark
            or entry.current_or_stale != "current"
            or not _text(entry.selected_strategy)
            or not _text(entry.evidence_source_type)
            or not _text(entry.evidence_reference_json)
            or (
                entry.strategy_family == "signal_only_f0"
                and (entry.selected_strategy != "signal_only_f0" or entry.dynamic_fit_mode is not None)
            )
            or (
                entry.strategy_family == "dynamic_fit"
                and entry.selected_strategy not in _SUPPORTED_DYNAMIC_STRATEGIES
            )
            or (
                entry.strategy_family == "dynamic_fit"
                and entry.dynamic_fit_mode != entry.selected_strategy
            )
        ):
            _refuse(
                "correction_strategy_unsupported",
                "correction",
                "A correction strategy is unsupported or incomplete.",
                "correction_strategy_unsupported",
            )
    correction_parameters = tuple(intent.correction_parameter_values)
    if not _validate_typed_values(correction_parameters):
        _refuse(
            "correction_authority_incomplete",
            "correction",
            "Correction parameter values are malformed.",
            "correction_parameter_values_invalid",
        )
    payload_identity = _digest(
        "npm-correction-payload:v1",
        {
            "parameters": correction_parameters,
            "per_roi_strategy_map": tuple(strategy_map),
        },
    )
    if payload_identity != intent.correction_payload_identity:
        _refuse(
            "correction_authority_identity_mismatch",
            "correction",
            "Correction authority does not match the accepted payload identity.",
            "correction_payload_identity_mismatch",
        )
    authority = GuidedNpmCorrectionAuthority(
        selected_canonical_roi_ids=selected,
        correction_parameter_values=correction_parameters,
        per_roi_correction_strategy_map=strategy_map,
        correction_payload_identity=intent.correction_payload_identity,
        canonical_correction_authority_identity="0" * 64,
    )
    return replace(
        authority,
        canonical_correction_authority_identity=compute_guided_npm_correction_authority_identity(
            authority
        ),
    )


def _build_feature_authority(
    intent: GuidedNpmProductionExecutionIntent,
) -> GuidedNpmFeatureAuthority:
    feature = intent.feature_event
    if not isinstance(feature, GuidedProductionFeatureEvent):
        _refuse(
            "feature_authority_incomplete",
            "feature",
            "The accepted feature authority is missing.",
            "feature_event_missing",
        )
    entries = tuple(sorted(feature.per_roi_feature_event_map, key=lambda item: item.roi_id))
    selected = tuple(intent.selected_roi_ids)
    if len({item.roi_id for item in entries}) != len(entries) or {
        item.roi_id for item in entries
    } != set(selected):
        _refuse(
            "feature_selected_roi_coverage_mismatch",
            "feature",
            "Feature authority does not cover selected ROIs exactly.",
            "feature_selected_roi_coverage_mismatch",
        )
    if (
        not feature.effective_values
        or feature.profile_status not in ("applied", "default_initialized")
        or (
            feature.profile_status == "applied"
            and feature.explicitly_applied is not True
        )
        or feature.current is not True
        or feature.visible_unapplied_changes is not False
    ):
        _refuse(
            "feature_authority_incomplete",
            "feature",
            "Feature authority is not complete and current.",
            "feature_event_incomplete",
        )
    effective_values = tuple(feature.effective_values)
    if not _validate_typed_values(effective_values):
        _refuse(
            "feature_authority_incomplete",
            "feature",
            "Feature effective values are malformed.",
            "feature_effective_values_invalid",
        )
    for entry in entries:
        if (
            entry.source not in {"default", "override"}
            or not _text(entry.feature_event_profile_id)
            or not feature_entry_provenance_valid(
                entry_source=entry.source,
                entry_feature_event_profile_id=entry.feature_event_profile_id,
                entry_explicit_user_mark=entry.explicit_user_mark,
                enclosing_profile_status=feature.profile_status,
                enclosing_profile_id=feature.profile_id,
                enclosing_current=feature.current,
                enclosing_visible_unapplied_changes=feature.visible_unapplied_changes,
            )
            or entry.current_or_stale != "current"
            or not entry.effective_config_fields
            or not _validate_typed_values(tuple(entry.effective_config_fields))
            or not _validate_typed_values(tuple(entry.override_config_fields))
        ):
            _refuse(
                "feature_authority_incomplete",
                "feature",
                "A per-ROI feature authority entry is incomplete.",
                "feature_per_roi_entry_invalid",
            )
    payload_identity = _digest("npm-feature-payload:v1", feature)
    if payload_identity != intent.feature_payload_identity:
        _refuse(
            "feature_authority_identity_mismatch",
            "feature",
            "Feature authority does not match the accepted payload identity.",
            "feature_payload_identity_mismatch",
        )
    authority = GuidedNpmFeatureAuthority(
        execution_mode=intent.execution_mode,
        profile_schema_version=feature.profile_schema_version,
        profile_id=feature.profile_id,
        effective_values=effective_values,
        active_fields=tuple(feature.active_fields),
        inactive_fields=tuple(feature.inactive_fields),
        profile_status=feature.profile_status,
        explicitly_applied=feature.explicitly_applied,
        current=feature.current,
        visible_unapplied_changes=feature.visible_unapplied_changes,
        per_roi_feature_event_map_version=feature.per_roi_feature_event_map_version,
        per_roi_feature_event_map=entries,
        selected_canonical_roi_ids=selected,
        feature_payload_identity=intent.feature_payload_identity,
        inactive_for_execution=intent.execution_mode == "tonic",
        canonical_feature_authority_identity="0" * 64,
    )
    return replace(
        authority,
        canonical_feature_authority_identity=compute_guided_npm_feature_authority_identity(
            authority
        ),
    )


def _build_output_authority(
    intent: GuidedNpmProductionExecutionIntent,
) -> GuidedNpmOutputAuthority:
    output = intent.output_policy
    if not isinstance(output, GuidedProductionOutputPolicy):
        _refuse(
            "output_authority_invalid",
            "output",
            "The accepted output policy is missing.",
            "output_policy_missing",
        )
    if (
        output.overwrite
        or output.precreate
        or output.creation_timing != "future_execution_start_only"
        or output.path_role != "output_base"
        or output.future_output_owner != "runner"
    ):
        _refuse(
            "output_authority_invalid",
            "output",
            "The output policy is not future-only and non-mutating.",
            "output_policy_not_future_only",
        )
    authority = GuidedNpmOutputAuthority(
        output_base_canonical=output.output_base_canonical,
        output_base_path_style=output.output_base_path_style,
        path_role=output.path_role,
        future_output_owner=output.future_output_owner,
        run_directory_strategy=output.child_directory_strategy,
        creation_timing=output.creation_timing,
        overwrite=output.overwrite,
        precreate=output.precreate,
        safety_classifier_version=output.safety_classifier_version,
        relationships=tuple(output.relationships),
        protected_root_context_complete=output.protected_root_context_complete,
        filesystem_fact_scope=output.filesystem_fact_scope,
        canonical_output_authority_identity="0" * 64,
    )
    return replace(
        authority,
        canonical_output_authority_identity=compute_guided_npm_output_authority_identity(
            authority
        ),
    )


def _build_authority(
    intent: GuidedNpmProductionExecutionIntent,
) -> GuidedNpmExecutionAuthority:
    if not isinstance(intent, GuidedNpmProductionExecutionIntent):
        _refuse(
            "intent_missing_or_invalid",
            "intent",
            "A B2-C1 NPM production intent is required.",
            "intent_type_invalid",
        )
    if (
        intent.intent_schema_name != GUIDED_NPM_PRODUCTION_INTENT_SCHEMA_NAME
        or intent.intent_schema_version != GUIDED_NPM_PRODUCTION_INTENT_SCHEMA_VERSION
    ):
        _refuse(
            "intent_schema_unsupported",
            "intent",
            "The NPM production-intent schema is unsupported.",
            "intent_schema_unsupported",
        )
    try:
        expected_intent_identity = compute_guided_npm_production_execution_intent_identity(
            intent
        )
    except Exception:
        _refuse(
            "intent_identity_mismatch",
            "intent",
            "The NPM production-intent identity cannot be recomputed.",
            "intent_identity_unavailable",
        )
    if expected_intent_identity != intent.canonical_intent_identity:
        _refuse(
            "intent_identity_mismatch",
            "intent",
            "The NPM production-intent identity is stale or tampered.",
            "intent_identity_mismatch",
        )
    try:
        expected_build_identity = build_application_build_identity(
            distribution_name=intent.application_build_identity.distribution_name,
            distribution_version=intent.application_build_identity.distribution_version,
            source_revision_kind=intent.application_build_identity.source_revision_kind,
            source_revision=intent.application_build_identity.source_revision,
            source_tree_state=intent.application_build_identity.source_tree_state,
            source_tree_digest=intent.application_build_identity.source_tree_digest,
            build_artifact_digest=intent.application_build_identity.build_artifact_digest,
            identity_provider_version=intent.application_build_identity.identity_provider_version,
        )
    except Exception:
        _refuse(
            "intent_missing_or_invalid",
            "intent",
            "The application build identity is malformed.",
            "application_build_identity_invalid",
        )
    if expected_build_identity.canonical_identity != intent.application_build_identity.canonical_identity:
        _refuse(
            "intent_identity_mismatch",
            "intent",
            "The application build identity is stale or tampered.",
            "application_build_identity_mismatch",
        )
    if intent.source_format != "npm":
        _refuse(
            "intent_not_npm",
            "intent",
            "The production intent is not NPM.",
            "source_format_not_npm",
        )
    if intent.acquisition_mode != "intermittent":
        _refuse(
            "intent_not_intermittent",
            "intent",
            "The production intent is not intermittent NPM.",
            "acquisition_mode_not_intermittent",
        )
    if intent.validation_status != "validator_accepted":
        _refuse(
            "intent_validation_status_invalid",
            "intent",
            "The production intent is not an accepted validation mapping.",
            "validation_status_invalid",
        )
    if intent.capability_status != GUIDED_NPM_PRODUCTION_CAPABILITY_STATUS:
        _refuse(
            "intent_unexpectedly_runnable",
            "intent",
            "The production intent has an unexpected capability state.",
            "intent_capability_state_invalid",
        )
    if (
        intent.mapping_contract_version != GUIDED_PRODUCTION_MAPPING_CONTRACT_VERSION
        or intent.runner_contract_version != GUIDED_NPM_PRODUCTION_RUNNER_CONTRACT_VERSION
    ):
        _refuse(
            "intent_startup_state_invalid",
            "intent",
            "The production intent startup contract is not the B2-C2 non-runnable contract.",
            "intent_startup_contract_invalid",
        )
    if "npm_startup_orchestration" not in intent.deferred_capabilities:
        _refuse(
            "intent_startup_state_invalid",
            "intent",
            "The production intent does not preserve unavailable NPM startup.",
            "intent_startup_state_invalid",
        )
    policy, parser_payload = _parser_policy_from_intent(intent)
    try:
        normalized_payload = json.loads(intent.normalized_recording_payload_json)
        normalized = deserialize_normalized_recording_description(normalized_payload)
        normalized_identity = compute_normalized_recording_description_identity(normalized)
    except (TypeError, ValueError, json.JSONDecodeError, NormalizedRecordingError) as exc:
        _refuse(
            "session_evidence_missing",
            "normalized_recording",
            "The embedded normalized NPM recording description is invalid.",
            getattr(exc, "category", "normalized_recording_invalid"),
        )
    if normalized_identity != intent.normalized_recording_description_identity:
        _refuse(
            "session_identity_mismatch",
            "normalized_recording",
            "The normalized recording description identity does not match the intent.",
            "normalized_recording_identity_mismatch",
        )
    if normalized.adapter_format != "npm" or normalized.acquisition_mode != "intermittent":
        _refuse(
            "intent_not_npm",
            "normalized_recording",
            "The embedded normalized recording is not intermittent NPM.",
            "normalized_adapter_not_npm",
        )
    if (
        normalized.sampling.parser_contract_identity != intent.parser_policy_identity
        or _canonical(normalized.sampling.parser_contract_content)
        != _canonical(parser_payload)
    ):
        _refuse(
            "recording_policy_identity_mismatch",
            "recording_policy",
            "Embedded normalized parser policy does not match the intent.",
            "normalized_parser_policy_mismatch",
        )
    if (
        not _sha256(intent.source_snapshot_set_identity)
        or not _sha256(intent.source_snapshot_content_identity)
        or not _sha256(intent.source_snapshot_identity)
        or normalized.adapter_evidence.get("npm_source_candidate_set_digest")
        != intent.source_snapshot_set_identity
        or normalized.adapter_evidence.get("npm_source_candidate_content_digest")
        != intent.source_snapshot_content_identity
    ):
        _refuse(
            "session_candidate_missing",
            "source_candidates",
            "The accepted source-candidate identity is incomplete or mismatched.",
            "source_candidate_snapshot_identity_mismatch",
        )
    sessions = _build_sessions(intent, normalized, policy)
    ordered_session_identity = _digest(
        "npm-ordered-session-identity:v1",
        [
            {
                "chronological_position": session.chronological_position,
                "stable_source_identity": session.canonical_relative_path,
                "authoritative_source_start_time": session.authoritative_source_start_time,
                "content_digest": session.sha256_content_digest,
            }
            for session in sessions
        ],
    )
    if ordered_session_identity != intent.ordered_session_identity:
        _refuse(
            "session_order_mismatch",
            "sessions",
            "The accepted ordered-session identity does not match the authority.",
            "ordered_session_identity_mismatch",
        )
    try:
        normalized_projection = build_normalized_recording_description_payload(
            normalized
        )
        per_session_identity = _digest(
            "npm-per-session-resolved-evidence:v1",
            normalized_projection["npm_per_session_resolved_evidence"],
        )
        mapping_identity = _digest(
            "npm-physical-to-canonical-roi-mapping:v1",
            normalized_projection["npm_physical_to_canonical_roi_mapping"],
        )
        source_snapshot_identity = _digest(
            "npm-source-snapshot:v1",
            {
                "source_root_canonical": intent.source_root_canonical,
                "source_candidate_set_digest": intent.source_snapshot_set_identity,
                "source_candidate_content_digest": intent.source_snapshot_content_identity,
                "candidate_files": [
                    {
                        "canonical_relative_path": item.canonical_relative_path,
                        "size_bytes": item.size_bytes,
                        "sha256_content_digest": item.sha256_content_digest,
                    }
                    for item in intent.source_candidate_files
                ],
            },
        )
    except Exception:
        _refuse(
            "session_identity_mismatch",
            "sessions",
            "The accepted normalized session identities could not be recomputed.",
            "session_identity_recompute_failed",
        )
    if per_session_identity != intent.per_session_resolved_evidence_identity:
        _refuse(
            "session_identity_mismatch",
            "sessions",
            "The accepted per-session evidence identity does not match.",
            "per_session_evidence_identity_mismatch",
        )
    if mapping_identity != intent.physical_to_canonical_roi_mapping_identity:
        _refuse(
            "roi_authority_identity_mismatch",
            "roi",
            "The accepted ROI mapping identity does not match.",
            "physical_to_canonical_mapping_identity_mismatch",
        )
    if source_snapshot_identity != intent.source_snapshot_identity:
        _refuse(
            "session_candidate_missing",
            "source_candidates",
            "The accepted source snapshot identity does not match.",
            "source_snapshot_identity_mismatch",
        )
    if _digest(
        "npm-output-time-basis:v1", [item.output_time_basis for item in sessions]
    ) != intent.output_time_basis_identity:
        _refuse(
            "session_output_time_basis_mismatch",
            "sessions",
            "Output time basis identity does not match the accepted intent.",
            "output_time_basis_identity_mismatch",
        )
    roi_authority = _build_roi_authority(intent, normalized, policy, sessions)
    correction_authority = _build_correction_authority(intent)
    feature_authority = _build_feature_authority(intent)
    output_authority = _build_output_authority(intent)
    session_sequence_identity = compute_guided_npm_session_sequence_identity(sessions)
    authority = GuidedNpmExecutionAuthority(
        authority_schema_name=GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_NAME,
        authority_schema_version=GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_VERSION,
        authority_contract_version=GUIDED_NPM_EXECUTION_AUTHORITY_CONTRACT_VERSION,
        source_production_intent_identity=intent.canonical_intent_identity,
        source_request_identity=intent.source_request_identity,
        validation_revision=intent.validation_revision,
        guided_plan_identity=intent.current_plan_identity,
        application_build_identity=intent.application_build_identity,
        source_format=intent.source_format,
        acquisition_mode=intent.acquisition_mode,
        execution_mode=intent.execution_mode,
        run_type=intent.run_type,
        recording_policy=policy,
        sessions=sessions,
        session_sequence_identity=session_sequence_identity,
        roi_authority=roi_authority,
        correction_authority=correction_authority,
        feature_authority=feature_authority,
        output_authority=output_authority,
        authorization_status=GUIDED_NPM_AUTHORIZATION_STATUS_NOT_AUTHORIZED,
        startup_status=GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED,
        runnable=False,
        canonical_authority_identity="0" * 64,
    )
    return replace(
        authority,
        canonical_authority_identity=compute_guided_npm_execution_authority_identity(
            authority
        ),
    )


def build_guided_npm_execution_authority(
    intent: GuidedNpmProductionExecutionIntent,
) -> GuidedNpmExecutionAuthorityResult:
    """Build the pure, identity-bound, non-runnable B2-C2 authority."""
    try:
        return _build_authority(intent)
    except _AuthorityRefusal as exc:
        return _failure(exc)
    except Exception as exc:
        return _failure(
            _AuthorityRefusal(
                "authority_internal_error",
                "authority",
                "NPM execution-authority construction failed.",
                type(exc).__name__,
            )
        )


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite authority value.")
        return value
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Authority mapping keys must be strings.")
        return {key: _serialize(value[key]) for key in value}
    if hasattr(value, "__dataclass_fields__"):
        return {item.name: _serialize(getattr(value, item.name)) for item in fields(value)}
    raise ValueError(f"Unsupported authority serialization type: {type(value).__name__}.")


def serialize_guided_npm_execution_authority(
    authority: GuidedNpmExecutionAuthority,
) -> dict[str, Any]:
    if not isinstance(authority, GuidedNpmExecutionAuthority):
        raise ValueError("authority must be a GuidedNpmExecutionAuthority.")
    _verify_authority_identity_chain(authority)
    payload = _serialize(authority)
    return {
        "identity_domain": GUIDED_NPM_EXECUTION_AUTHORITY_IDENTITY_DOMAIN,
        **payload,
    }


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required(payload, key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object.")
    return value


def _build_identity(payload: Mapping[str, Any]) -> ApplicationBuildIdentity:
    return ApplicationBuildIdentity(
        schema_name=_required_text(payload, "schema_name"),
        schema_version=_required_text(payload, "schema_version"),
        identity_provider_version=_required_text(payload, "identity_provider_version"),
        distribution_name=_required_text(payload, "distribution_name"),
        distribution_version=_required_text(payload, "distribution_version"),
        source_revision_kind=_required_text(payload, "source_revision_kind"),
        source_revision=_required_text(payload, "source_revision"),
        source_tree_state=_required_text(payload, "source_tree_state"),
        source_tree_digest=payload.get("source_tree_digest"),
        build_artifact_digest=payload.get("build_artifact_digest"),
        canonical_identity=_required_sha(payload, "canonical_identity"),
    )


def _build_typed_value(payload: Mapping[str, Any]) -> GuidedProductionTypedValue:
    return GuidedProductionTypedValue(
        field_name=_required_text(payload, "field_name"),
        value_type=_required_text(payload, "value_type"),
        value=_required(payload, "value"),
        source_classification=_required_text(payload, "source_classification"),
    )


def _build_strategy(payload: Mapping[str, Any]) -> GuidedProductionPerRoiStrategy:
    return GuidedProductionPerRoiStrategy(
        roi_id=_required_text(payload, "roi_id"),
        strategy_family=_required_text(payload, "strategy_family"),
        dynamic_fit_mode=payload.get("dynamic_fit_mode"),
        selected_strategy=_required_text(payload, "selected_strategy"),
        evidence_source_type=_required_text(payload, "evidence_source_type"),
        evidence_reference_json=_required_text(payload, "evidence_reference_json"),
        explicit_user_mark=_required(payload, "explicit_user_mark"),
        current_or_stale=_required_text(payload, "current_or_stale"),
    )


def _build_feature_entry(payload: Mapping[str, Any]) -> GuidedProductionPerRoiFeatureEvent:
    return GuidedProductionPerRoiFeatureEvent(
        roi_id=_required_text(payload, "roi_id"),
        source=_required_text(payload, "source"),
        feature_event_profile_id=_required_text(payload, "feature_event_profile_id"),
        override_config_fields=tuple(
            _build_typed_value(item)
            for item in _required_tuple(_required(payload, "override_config_fields"), "override_config_fields")
        ),
        effective_config_fields=tuple(
            _build_typed_value(item)
            for item in _required_tuple(_required(payload, "effective_config_fields"), "effective_config_fields")
        ),
        explicit_user_mark=_required(payload, "explicit_user_mark"),
        current_or_stale=_required_text(payload, "current_or_stale"),
    )


def _deserialize_authority_payload(payload: Mapping[str, Any]) -> GuidedNpmExecutionAuthority:
    if _required(payload, "identity_domain") != GUIDED_NPM_EXECUTION_AUTHORITY_IDENTITY_DOMAIN:
        raise ValueError("authority_serialization_invalid")
    policy_payload = _mapping(payload, "recording_policy")
    policy = GuidedNpmRecordingPolicy(
        parser_policy_identity=_required_sha(policy_payload, "parser_policy_identity"),
        parser_policy_content_json=_required_text(policy_payload, "parser_policy_content_json"),
        time_axis_mode=_required_text(policy_payload, "time_axis_mode"),
        ordered_timestamp_candidates=tuple(_required(policy_payload, "ordered_timestamp_candidates")),
        timestamp_unit=_required_text(policy_payload, "timestamp_unit"),
        timestamp_finite_policy=_required_text(policy_payload, "timestamp_finite_policy"),
        led_state_column=_required_text(policy_payload, "led_state_column"),
        reference_led_value=_required(policy_payload, "reference_led_value"),
        signal_led_value=_required(policy_payload, "signal_led_value"),
        roi_prefix=_required_text(policy_payload, "roi_prefix"),
        roi_suffix=_required_text(policy_payload, "roi_suffix"),
        roi_ordering_rule=_required_text(policy_payload, "roi_ordering_rule"),
        canonical_roi_naming_rule=_required_text(policy_payload, "canonical_roi_naming_rule"),
        support_policy=_required_text(policy_payload, "support_policy"),
        support_policy_identity=_required_sha(policy_payload, "support_policy_identity"),
        roi_value_nan_policy=_required_text(policy_payload, "roi_value_nan_policy"),
        target_fs_hz=float(_required(policy_payload, "target_fs_hz")),
        configured_session_duration_sec=float(_required(policy_payload, "configured_session_duration_sec")),
        sessions_per_hour=_required(policy_payload, "sessions_per_hour"),
        chronology_policy=_required_text(policy_payload, "chronology_policy"),
        gap_policy=_required_text(policy_payload, "gap_policy"),
        overlap_policy=_required_text(policy_payload, "overlap_policy"),
        output_time_basis=_required_text(policy_payload, "output_time_basis"),
        canonical_policy_identity=_required_sha(policy_payload, "canonical_policy_identity"),
    )
    sessions_payload = _required(payload, "sessions")
    if not isinstance(sessions_payload, (list, tuple)):
        raise ValueError("authority_serialization_invalid")
    sessions: list[GuidedNpmAuthorizedSession] = []
    for item in sessions_payload:
        if not isinstance(item, Mapping):
            raise ValueError("authority_serialization_invalid")
        mapping_payload = _required(item, "physical_to_canonical_roi_mapping")
        if not isinstance(mapping_payload, (list, tuple)):
            raise ValueError("authority_serialization_invalid")
        sessions.append(
            GuidedNpmAuthorizedSession(
                chronological_position=_required(item, "chronological_position"),
                canonical_relative_path=_required_text(item, "canonical_relative_path"),
                authorized_absolute_source_reference=_required_text(item, "authorized_absolute_source_reference"),
                size_bytes=_required(item, "size_bytes"),
                sha256_content_digest=_required_sha(item, "sha256_content_digest"),
                authoritative_source_start_time=_required_text(item, "authoritative_source_start_time"),
                actual_elapsed_sec=float(_required(item, "actual_elapsed_sec")),
                nominal_expected_elapsed_sec=float(_required(item, "nominal_expected_elapsed_sec")),
                resolved_timestamp_column=_required_text(item, "resolved_timestamp_column"),
                resolved_led_column=item.get("resolved_led_column"),
                timestamp_unit=_required_text(item, "timestamp_unit"),
                source_timing_evidence=_required_text(item, "source_timing_evidence"),
                physical_roi_inventory=tuple(_required(item, "physical_roi_inventory")),
                physical_to_canonical_roi_mapping=tuple(
                    GuidedNpmRoiMappingEntry(
                        physical_source_column=_required_text(entry, "physical_source_column"),
                        canonical_roi_id=_required_text(entry, "canonical_roi_id"),
                    )
                    for entry in mapping_payload
                ),
                overlap_origin_absolute=float(_required(item, "overlap_origin_absolute")),
                resolved_support_start_offset_sec=float(_required(item, "resolved_support_start_offset_sec")),
                resolved_support_end_offset_sec=float(_required(item, "resolved_support_end_offset_sec")),
                resolved_support_start_absolute=float(_required(item, "resolved_support_start_absolute")),
                resolved_support_end_absolute=float(_required(item, "resolved_support_end_absolute")),
                observed_support_duration_sec=float(_required(item, "observed_support_duration_sec")),
                support_policy=_required_text(item, "support_policy"),
                support_policy_identity=_required_sha(item, "support_policy_identity"),
                output_time_basis=_required_text(item, "output_time_basis"),
                warning_categories=tuple(_required(item, "warning_categories")),
                canonical_session_identity=_required_sha(item, "canonical_session_identity"),
            )
        )
    roi_payload = _mapping(payload, "roi_authority")
    roi_mapping_payload = _required(roi_payload, "physical_to_canonical_roi_mapping")
    selected_mapping_payload = _required(
        roi_payload, "selected_physical_to_canonical_roi_mapping"
    )
    roi_authority = GuidedNpmRoiAuthority(
        complete_canonical_roi_ids=tuple(_required(roi_payload, "complete_canonical_roi_ids")),
        selected_canonical_roi_ids=tuple(_required(roi_payload, "selected_canonical_roi_ids")),
        excluded_canonical_roi_ids=tuple(_required(roi_payload, "excluded_canonical_roi_ids")),
        complete_physical_source_columns=tuple(_required(roi_payload, "complete_physical_source_columns")),
        physical_to_canonical_roi_mapping=tuple(
            GuidedNpmRoiMappingEntry(
                physical_source_column=_required_text(item, "physical_source_column"),
                canonical_roi_id=_required_text(item, "canonical_roi_id"),
            )
            for item in roi_mapping_payload
        ),
        selected_physical_source_columns=tuple(_required(roi_payload, "selected_physical_source_columns")),
        selected_physical_to_canonical_roi_mapping=tuple(
            GuidedNpmRoiMappingEntry(
                physical_source_column=_required_text(item, "physical_source_column"),
                canonical_roi_id=_required_text(item, "canonical_roi_id"),
            )
            for item in selected_mapping_payload
        ),
        roi_ordering_rule=_required_text(roi_payload, "roi_ordering_rule"),
        canonical_roi_naming_rule=_required_text(roi_payload, "canonical_roi_naming_rule"),
        canonical_roi_authority_identity=_required_sha(roi_payload, "canonical_roi_authority_identity"),
    )
    correction_payload = _mapping(payload, "correction_authority")
    correction_authority = GuidedNpmCorrectionAuthority(
        selected_canonical_roi_ids=tuple(_required(correction_payload, "selected_canonical_roi_ids")),
        correction_parameter_values=tuple(
            _build_typed_value(item)
            for item in _required_tuple(_required(correction_payload, "correction_parameter_values"), "correction_parameter_values")
        ),
        per_roi_correction_strategy_map=tuple(
            _build_strategy(item)
            for item in _required_tuple(_required(correction_payload, "per_roi_correction_strategy_map"), "per_roi_correction_strategy_map")
        ),
        correction_payload_identity=_required_sha(correction_payload, "correction_payload_identity"),
        canonical_correction_authority_identity=_required_sha(correction_payload, "canonical_correction_authority_identity"),
    )
    feature_payload = _mapping(payload, "feature_authority")
    feature_authority = GuidedNpmFeatureAuthority(
        execution_mode=_required_text(feature_payload, "execution_mode"),
        profile_schema_version=_required_text(feature_payload, "profile_schema_version"),
        profile_id=_required_text(feature_payload, "profile_id"),
        effective_values=tuple(
            _build_typed_value(item)
            for item in _required_tuple(_required(feature_payload, "effective_values"), "effective_values")
        ),
        active_fields=tuple(_required(feature_payload, "active_fields")),
        inactive_fields=tuple(_required(feature_payload, "inactive_fields")),
        profile_status=_required_text(feature_payload, "profile_status"),
        explicitly_applied=_required(feature_payload, "explicitly_applied"),
        current=_required(feature_payload, "current"),
        visible_unapplied_changes=_required(feature_payload, "visible_unapplied_changes"),
        per_roi_feature_event_map_version=_required_text(feature_payload, "per_roi_feature_event_map_version"),
        per_roi_feature_event_map=tuple(
            _build_feature_entry(item)
            for item in _required_tuple(_required(feature_payload, "per_roi_feature_event_map"), "per_roi_feature_event_map")
        ),
        selected_canonical_roi_ids=tuple(_required(feature_payload, "selected_canonical_roi_ids")),
        feature_payload_identity=_required_sha(feature_payload, "feature_payload_identity"),
        inactive_for_execution=_required(feature_payload, "inactive_for_execution"),
        canonical_feature_authority_identity=_required_sha(feature_payload, "canonical_feature_authority_identity"),
    )
    output_payload = _mapping(payload, "output_authority")
    output_authority = GuidedNpmOutputAuthority(
        output_base_canonical=_required_text(output_payload, "output_base_canonical"),
        output_base_path_style=_required_text(output_payload, "output_base_path_style"),
        path_role=_required_text(output_payload, "path_role"),
        future_output_owner=_required_text(output_payload, "future_output_owner"),
        run_directory_strategy=_required_text(output_payload, "run_directory_strategy"),
        creation_timing=_required_text(output_payload, "creation_timing"),
        overwrite=_required(output_payload, "overwrite"),
        precreate=_required(output_payload, "precreate"),
        safety_classifier_version=_required_text(output_payload, "safety_classifier_version"),
        relationships=tuple(
            GuidedProductionOutputRelationship(
                relationship=_required_text(item, "relationship"),
                root_kind=_required_text(item, "root_kind"),
                status=_required_text(item, "status"),
            )
            for item in _required_tuple(_required(output_payload, "relationships"), "relationships")
        ),
        protected_root_context_complete=_required(output_payload, "protected_root_context_complete"),
        filesystem_fact_scope=_required_text(output_payload, "filesystem_fact_scope"),
        canonical_output_authority_identity=_required_sha(output_payload, "canonical_output_authority_identity"),
    )
    authority = GuidedNpmExecutionAuthority(
        authority_schema_name=_required_text(payload, "authority_schema_name"),
        authority_schema_version=_required_text(payload, "authority_schema_version"),
        authority_contract_version=_required_text(payload, "authority_contract_version"),
        source_production_intent_identity=_required_sha(payload, "source_production_intent_identity"),
        source_request_identity=_required_sha(payload, "source_request_identity"),
        validation_revision=_required(payload, "validation_revision"),
        guided_plan_identity=_required_sha(payload, "guided_plan_identity"),
        application_build_identity=_build_identity(_mapping(payload, "application_build_identity")),
        source_format=_required_text(payload, "source_format"),
        acquisition_mode=_required_text(payload, "acquisition_mode"),
        execution_mode=_required_text(payload, "execution_mode"),
        run_type=_required_text(payload, "run_type"),
        recording_policy=policy,
        sessions=tuple(sessions),
        session_sequence_identity=_required_sha(payload, "session_sequence_identity"),
        roi_authority=roi_authority,
        correction_authority=correction_authority,
        feature_authority=feature_authority,
        output_authority=output_authority,
        authorization_status=_required_text(payload, "authorization_status"),
        startup_status=_required_text(payload, "startup_status"),
        runnable=_required(payload, "runnable"),
        canonical_authority_identity=_required_sha(payload, "canonical_authority_identity"),
    )
    if compute_guided_npm_recording_policy_identity(policy) != policy.canonical_policy_identity:
        raise ValueError("recording_policy_identity_mismatch")
    if any(
        compute_guided_npm_authorized_session_identity(item) != item.canonical_session_identity
        for item in sessions
    ):
        raise ValueError("session_identity_mismatch")
    if compute_guided_npm_session_sequence_identity(authority.sessions) != authority.session_sequence_identity:
        raise ValueError("session_sequence_identity_mismatch")
    if compute_guided_npm_roi_authority_identity(roi_authority) != roi_authority.canonical_roi_authority_identity:
        raise ValueError("roi_authority_identity_mismatch")
    if compute_guided_npm_correction_authority_identity(correction_authority) != correction_authority.canonical_correction_authority_identity:
        raise ValueError("correction_authority_identity_mismatch")
    if compute_guided_npm_feature_authority_identity(feature_authority) != feature_authority.canonical_feature_authority_identity:
        raise ValueError("feature_authority_identity_mismatch")
    if compute_guided_npm_output_authority_identity(output_authority) != output_authority.canonical_output_authority_identity:
        raise ValueError("output_authority_identity_mismatch")
    if compute_guided_npm_execution_authority_identity(authority) != authority.canonical_authority_identity:
        raise ValueError("authority_identity_mismatch")
    _verify_authority_identity_chain(authority)
    return authority


def deserialize_guided_npm_execution_authority(
    payload: Mapping[str, Any],
) -> GuidedNpmExecutionAuthority:
    """Reconstruct and verify an immutable B2-C2 authority payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("authority_serialization_invalid")
    try:
        return _deserialize_authority_payload(payload)
    except (TypeError, KeyError, ValueError, OverflowError) as exc:
        if str(exc) == "authority_serialization_invalid":
            raise
        raise ValueError("authority_serialization_invalid") from exc


def verify_guided_npm_execution_authority(
    authority: GuidedNpmExecutionAuthority,
) -> None:
    """Reverify the complete immutable B2-C2 authority contract in memory.

    The round trip deliberately reconstructs every frozen nested dataclass so
    callers do not need to duplicate B2-C2's identity or invariant checks.
    This helper performs no filesystem I/O.
    """
    if type(authority) is not GuidedNpmExecutionAuthority:
        raise ValueError("authority_type_invalid")
    payload = serialize_guided_npm_execution_authority(authority)
    restored = deserialize_guided_npm_execution_authority(payload)
    if restored != authority:
        raise ValueError("authority_round_trip_mismatch")
