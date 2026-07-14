"""Pure B2-C4 construction of the sole canonical NPM startup payload.

The payload is deterministic and in memory only.  This module performs no
filesystem access, creates no manifest or startup artifact, and grants no
claim, launch, or execution permission.

Top-level B2-C2 and B2-C3 identities are historical provenance references
established during construction.  Persisted execution authority comes from
the independently recomputable source evidence and B2-C4 projection chain.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import hashlib
import math
from typing import Any

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_authorization import (
    GUIDED_NPM_AUTHORIZATION_STARTUP_STATUS_NOT_MATERIALIZED,
    GUIDED_NPM_AUTHORIZATION_STATUS_AUTHORIZED_FOR_STARTUP_PREPARATION,
    GUIDED_NPM_EXECUTION_AUTHORIZATION_CONTRACT_VERSION,
    GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_NAME,
    GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
    GuidedNpmExecutionAuthorization,
    GuidedNpmVerifiedSourceFile,
    GuidedNpmVerifiedSourceSnapshot,
    compute_guided_npm_verified_source_content_identity,
    compute_guided_npm_verified_source_file_identity,
    compute_guided_npm_verified_source_sequence_identity,
    compute_guided_npm_verified_source_set_identity,
    compute_guided_npm_verified_source_snapshot_identity,
    verify_guided_npm_execution_authorization,
)
from photometry_pipeline.guided_npm_execution_authority import (
    GUIDED_NPM_AUTHORIZATION_STATUS_NOT_AUTHORIZED,
    GUIDED_NPM_EXECUTION_AUTHORITY_CONTRACT_VERSION,
    GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_NAME,
    GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_VERSION,
    GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED,
    GuidedNpmExecutionAuthority,
    verify_guided_npm_execution_authority,
)
from photometry_pipeline.guided_production_mapping import (
    ApplicationBuildIdentity,
    GuidedProductionOutputRelationship,
    GuidedProductionPerRoiFeatureEvent,
    GuidedProductionPerRoiStrategy,
    GuidedProductionTypedValue,
    build_application_build_identity,
)


GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_NAME = "guided_npm_startup_payload"
GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_VERSION = "v1"
GUIDED_NPM_STARTUP_PAYLOAD_CONTRACT_VERSION = "guided_npm_startup_payload.v1"
GUIDED_NPM_STARTUP_PAYLOAD_IDENTITY_DOMAIN = "guided_npm_startup_payload.v1"
GUIDED_NPM_STARTUP_PAYLOAD_STATUS_CONSTRUCTED_IN_MEMORY = "constructed_in_memory"
GUIDED_NPM_STARTUP_PAYLOAD_PERSISTENCE_STATUS_NOT_PERSISTED = "not_persisted"
GUIDED_NPM_STARTUP_PAYLOAD_CLAIM_STATUS_NOT_CLAIMED = "not_claimed"
GUIDED_NPM_STARTUP_EXECUTION_PROJECTION_SCHEMA_NAME = (
    "guided_npm_startup_execution_projection"
)
GUIDED_NPM_STARTUP_EXECUTION_PROJECTION_SCHEMA_VERSION = "v1"
GUIDED_NPM_STARTUP_DEFERRED_EXECUTION_CAPABILITIES = (
    "npm_startup_persistence",
    "npm_startup_claim",
    "npm_wrapper_launch",
    "npm_execution_adapter",
    "npm_terminal_reconciliation",
)

GUIDED_NPM_STARTUP_PAYLOAD_REFUSAL_CATEGORIES = (
    "authorization_missing_or_invalid",
    "authorization_schema_unsupported",
    "authorization_identity_mismatch",
    "authorization_state_invalid",
    "authority_missing_or_invalid",
    "authority_schema_unsupported",
    "authority_identity_mismatch",
    "authority_state_invalid",
    "authorization_authority_mismatch",
    "request_identity_mismatch",
    "production_intent_identity_mismatch",
    "validation_revision_mismatch",
    "guided_plan_identity_mismatch",
    "application_build_identity_mismatch",
    "execution_mode_mismatch",
    "selected_roi_scope_mismatch",
    "correction_authority_identity_mismatch",
    "feature_authority_identity_mismatch",
    "output_authority_identity_mismatch",
    "verified_source_snapshot_mismatch",
    "verified_source_file_missing",
    "verified_source_file_extra",
    "verified_source_file_duplicate",
    "verified_source_file_order_mismatch",
    "verified_source_file_path_mismatch",
    "verified_source_file_size_mismatch",
    "verified_source_file_digest_mismatch",
    "recording_policy_projection_invalid",
    "roi_projection_invalid",
    "correction_projection_invalid",
    "feature_projection_invalid",
    "output_projection_invalid",
    "execution_projection_invalid",
    "startup_payload_identity_mismatch",
    "startup_payload_serialization_invalid",
    "startup_payload_internal_error",
)
_CATEGORY_SET = frozenset(GUIDED_NPM_STARTUP_PAYLOAD_REFUSAL_CATEGORIES)
_SHA_HEX = frozenset("0123456789abcdef")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA_HEX


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _text_tuple(value: Any, *, allow_empty: bool = True) -> bool:
    return (
        isinstance(value, tuple)
        and (allow_empty or bool(value))
        and all(_text(item) for item in value)
    )


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite startup values are prohibited.")
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Startup mapping keys must be strings.")
        return {key: _canonical(value[key]) for key in value}
    if hasattr(value, "__dataclass_fields__"):
        return {
            item.name: _canonical(getattr(value, item.name)) for item in fields(value)
        }
    raise ValueError(f"Unsupported startup value: {type(value).__name__}.")


def _digest(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + encode_canonical_value(_canonical(value))
    ).hexdigest()


@dataclass(frozen=True)
class GuidedNpmStartupPayloadIssue:
    category: str
    section: str
    message: str
    detail_code: str

    def __post_init__(self) -> None:
        if self.category not in _CATEGORY_SET or not all(
            _text(value) for value in (self.section, self.message, self.detail_code)
        ):
            raise ValueError("Invalid NPM startup-payload issue.")


@dataclass(frozen=True)
class GuidedNpmStartupPayloadFailure:
    blocking_issues: tuple[GuidedNpmStartupPayloadIssue, ...]
    status: str = "refused"

    def __post_init__(self) -> None:
        if not self.blocking_issues or self.status != "refused":
            raise ValueError("Startup-payload failure requires blocking issues.")


@dataclass(frozen=True)
class GuidedNpmStartupRoiMappingEntry:
    physical_source_column: str
    canonical_roi_id: str

    def __post_init__(self) -> None:
        if not _text(self.physical_source_column) or not _text(self.canonical_roi_id):
            raise ValueError("Startup ROI mapping entry is incomplete.")


@dataclass(frozen=True)
class GuidedNpmStartupVerifiedSourceFileProjection:
    """Frozen B2-C3 source-verification evidence embedded for later rechecking."""

    chronological_position: int
    canonical_relative_path: str
    authorized_absolute_source_reference: str
    inspected_absolute_path: str
    expected_size_bytes: int
    observed_size_bytes: int
    expected_sha256_content_digest: str
    observed_sha256_content_digest: str
    pre_hash_stat_identity: str
    post_hash_stat_identity: str
    canonical_verified_file_identity: str

    def __post_init__(self) -> None:
        # Reuse B2-C3's immutable contract validation rather than maintaining a
        # second interpretation of verified-file evidence here.
        _reconstruct_verified_source_file(self)


@dataclass(frozen=True)
class GuidedNpmStartupSession:
    chronological_position: int
    canonical_relative_path: str
    authorized_absolute_source_reference: str
    size_bytes: int
    sha256_content_digest: str
    verified_source_file: GuidedNpmStartupVerifiedSourceFileProjection
    authoritative_source_start_time: str
    actual_elapsed_sec: float
    nominal_expected_elapsed_sec: float
    resolved_timestamp_column: str
    resolved_led_column: str | None
    timestamp_unit: str
    source_timing_evidence: str
    physical_roi_inventory: tuple[str, ...]
    physical_to_canonical_roi_mapping: tuple[GuidedNpmStartupRoiMappingEntry, ...]
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
    canonical_startup_session_identity: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.chronological_position, bool)
            or not isinstance(self.chronological_position, int)
            or self.chronological_position < 0
        ):
            raise ValueError("Startup session position is invalid.")
        for name in (
            "canonical_relative_path",
            "authorized_absolute_source_reference",
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
            raise ValueError("resolved_led_column is invalid.")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("Startup session size is invalid.")
        for name in (
            "sha256_content_digest",
            "support_policy_identity",
            "canonical_startup_session_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError(f"{name} is invalid.")
        if not isinstance(
            self.verified_source_file,
            GuidedNpmStartupVerifiedSourceFileProjection,
        ):
            raise ValueError("Startup verified-file evidence is required.")
        verified = self.verified_source_file
        if (
            self.chronological_position != verified.chronological_position
            or self.canonical_relative_path != verified.canonical_relative_path
            or self.authorized_absolute_source_reference
            != verified.authorized_absolute_source_reference
        ):
            raise ValueError("Startup session and verified-file paths differ.")
        if (
            self.size_bytes != verified.expected_size_bytes
            or self.size_bytes != verified.observed_size_bytes
        ):
            raise ValueError("Startup session and verified-file sizes differ.")
        if (
            self.sha256_content_digest
            != verified.expected_sha256_content_digest
            or self.sha256_content_digest
            != verified.observed_sha256_content_digest
        ):
            raise ValueError("Startup session and verified-file digests differ.")
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
        if not _text_tuple(self.physical_roi_inventory, allow_empty=False):
            raise ValueError("Physical ROI inventory is invalid.")
        if (
            not isinstance(self.physical_to_canonical_roi_mapping, tuple)
            or len(self.physical_to_canonical_roi_mapping)
            != len(self.physical_roi_inventory)
            or tuple(
                item.physical_source_column
                for item in self.physical_to_canonical_roi_mapping
            )
            != self.physical_roi_inventory
        ):
            raise ValueError("Startup session ROI mapping is invalid.")
        if any(
            not isinstance(item, GuidedNpmStartupRoiMappingEntry)
            for item in self.physical_to_canonical_roi_mapping
        ):
            raise ValueError("Startup session ROI mapping entries are invalid.")
        if not _text_tuple(self.warning_categories):
            raise ValueError("Warning categories are invalid.")


@dataclass(frozen=True)
class GuidedNpmStartupSourceProjection:
    source_root_canonical: str
    source_root_inspected: str
    discovery_contract_version: str
    ordered_sessions: tuple[GuidedNpmStartupSession, ...]
    ordered_session_sequence_identity: str
    verified_ordered_file_sequence_identity: str
    verified_source_set_identity: str
    verified_source_content_identity: str
    verified_source_snapshot_identity: str
    canonical_source_projection_identity: str

    def __post_init__(self) -> None:
        if (
            not _text(self.source_root_canonical)
            or not _text(self.source_root_inspected)
            or not _text(self.discovery_contract_version)
            or not isinstance(self.ordered_sessions, tuple)
            or not self.ordered_sessions
            or any(
                not isinstance(item, GuidedNpmStartupSession)
                for item in self.ordered_sessions
            )
        ):
            raise ValueError("Startup source projection is incomplete.")
        if tuple(item.chronological_position for item in self.ordered_sessions) != tuple(
            range(len(self.ordered_sessions))
        ):
            raise ValueError("Startup source positions are not exact.")
        if len({item.canonical_relative_path for item in self.ordered_sessions}) != len(
            self.ordered_sessions
        ):
            raise ValueError("Startup source paths are not unique.")
        for name in (
            "ordered_session_sequence_identity",
            "verified_ordered_file_sequence_identity",
            "verified_source_set_identity",
            "verified_source_content_identity",
            "verified_source_snapshot_identity",
            "canonical_source_projection_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError(f"{name} is invalid.")


@dataclass(frozen=True)
class GuidedNpmStartupRecordingPolicy:
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
    source_recording_policy_identity: str
    canonical_recording_policy_identity: str

    def __post_init__(self) -> None:
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
                raise ValueError(f"{name} is required.")
        if not _text_tuple(self.ordered_timestamp_candidates, allow_empty=False):
            raise ValueError("Timestamp candidates are required.")
        for name in (
            "parser_policy_identity",
            "support_policy_identity",
            "source_recording_policy_identity",
            "canonical_recording_policy_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError(f"{name} is invalid.")
        if not _finite(self.target_fs_hz) or self.target_fs_hz <= 0:
            raise ValueError("Target sampling rate is invalid.")
        if not _finite(self.configured_session_duration_sec) or self.configured_session_duration_sec <= 0:
            raise ValueError("Configured session duration is invalid.")
        if isinstance(self.sessions_per_hour, bool) or not isinstance(self.sessions_per_hour, int) or self.sessions_per_hour <= 0:
            raise ValueError("Sessions per hour is invalid.")


@dataclass(frozen=True)
class GuidedNpmStartupRoiProjection:
    complete_canonical_roi_ids: tuple[str, ...]
    selected_canonical_roi_ids: tuple[str, ...]
    excluded_canonical_roi_ids: tuple[str, ...]
    complete_physical_source_columns: tuple[str, ...]
    physical_to_canonical_roi_mapping: tuple[GuidedNpmStartupRoiMappingEntry, ...]
    selected_physical_source_columns: tuple[str, ...]
    selected_physical_to_canonical_roi_mapping: tuple[GuidedNpmStartupRoiMappingEntry, ...]
    roi_ordering_rule: str
    canonical_roi_naming_rule: str
    source_roi_authority_identity: str
    canonical_roi_projection_identity: str

    def __post_init__(self) -> None:
        if not _text_tuple(self.complete_canonical_roi_ids, allow_empty=False) or not _text_tuple(
            self.selected_canonical_roi_ids, allow_empty=False
        ) or not _text_tuple(self.excluded_canonical_roi_ids):
            raise ValueError("ROI scopes are invalid.")
        if (
            len(set(self.complete_canonical_roi_ids))
            != len(self.complete_canonical_roi_ids)
            or len(set(self.selected_canonical_roi_ids))
            != len(self.selected_canonical_roi_ids)
            or len(set(self.excluded_canonical_roi_ids))
            != len(self.excluded_canonical_roi_ids)
            or not isinstance(self.physical_to_canonical_roi_mapping, tuple)
            or not isinstance(self.selected_physical_to_canonical_roi_mapping, tuple)
            or any(
                not isinstance(item, GuidedNpmStartupRoiMappingEntry)
                for item in self.physical_to_canonical_roi_mapping
                + self.selected_physical_to_canonical_roi_mapping
            )
        ):
            raise ValueError("ROI scopes or mappings contain invalid entries.")
        if set(self.selected_canonical_roi_ids) | set(self.excluded_canonical_roi_ids) != set(
            self.complete_canonical_roi_ids
        ) or set(self.selected_canonical_roi_ids) & set(self.excluded_canonical_roi_ids):
            raise ValueError("ROI selection coverage is invalid.")
        if tuple(item.canonical_roi_id for item in self.physical_to_canonical_roi_mapping) != self.complete_canonical_roi_ids:
            raise ValueError("Complete ROI mapping is invalid.")
        if tuple(item.physical_source_column for item in self.physical_to_canonical_roi_mapping) != self.complete_physical_source_columns:
            raise ValueError("Physical ROI inventory is invalid.")
        selected = tuple(
            item
            for item in self.physical_to_canonical_roi_mapping
            if item.canonical_roi_id in set(self.selected_canonical_roi_ids)
        )
        if selected != self.selected_physical_to_canonical_roi_mapping or tuple(
            item.physical_source_column for item in selected
        ) != self.selected_physical_source_columns:
            raise ValueError("Selected ROI mapping was not mechanically derived.")
        if not _text(self.roi_ordering_rule) or not _text(self.canonical_roi_naming_rule):
            raise ValueError("ROI naming rules are required.")
        if not _sha(self.source_roi_authority_identity) or not _sha(
            self.canonical_roi_projection_identity
        ):
            raise ValueError("ROI projection identities are invalid.")


@dataclass(frozen=True)
class GuidedNpmStartupCorrectionProjection:
    selected_canonical_roi_ids: tuple[str, ...]
    correction_parameter_values: tuple[GuidedProductionTypedValue, ...]
    per_roi_correction_strategy_map: tuple[GuidedProductionPerRoiStrategy, ...]
    source_correction_payload_identity: str
    source_correction_authority_identity: str
    canonical_correction_projection_identity: str

    def __post_init__(self) -> None:
        if not _text_tuple(self.selected_canonical_roi_ids, allow_empty=False):
            raise ValueError("Correction selected ROI scope is invalid.")
        if not isinstance(self.correction_parameter_values, tuple) or any(
            not isinstance(item, GuidedProductionTypedValue)
            for item in self.correction_parameter_values
        ) or not isinstance(self.per_roi_correction_strategy_map, tuple) or any(
            not isinstance(item, GuidedProductionPerRoiStrategy)
            for item in self.per_roi_correction_strategy_map
        ):
            raise ValueError("Correction entries are invalid.")
        roi_ids = tuple(item.roi_id for item in self.per_roi_correction_strategy_map)
        if len(set(roi_ids)) != len(roi_ids) or set(roi_ids) != set(
            self.selected_canonical_roi_ids
        ):
            raise ValueError("Correction ROI coverage is invalid.")
        for name in (
            "source_correction_payload_identity",
            "source_correction_authority_identity",
            "canonical_correction_projection_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError(f"{name} is invalid.")


@dataclass(frozen=True)
class GuidedNpmStartupFeatureProjection:
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
    inactive_for_execution: bool
    source_feature_payload_identity: str
    source_feature_authority_identity: str
    canonical_feature_projection_identity: str

    def __post_init__(self) -> None:
        if self.execution_mode not in {"phasic", "tonic", "both"}:
            raise ValueError("Feature execution mode is invalid.")
        for name in (
            "profile_schema_version",
            "profile_id",
            "profile_status",
            "per_roi_feature_event_map_version",
        ):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} is required.")
        if not isinstance(self.effective_values, tuple) or any(
            not isinstance(item, GuidedProductionTypedValue) for item in self.effective_values
        ) or not _text_tuple(self.active_fields) or not _text_tuple(self.inactive_fields):
            raise ValueError("Feature configuration is invalid.")
        if (
            not _text_tuple(self.selected_canonical_roi_ids, allow_empty=False)
            or not isinstance(self.per_roi_feature_event_map, tuple)
            or any(
            not isinstance(item, GuidedProductionPerRoiFeatureEvent)
            for item in self.per_roi_feature_event_map
            )
        ):
            raise ValueError("Feature ROI entries are invalid.")
        roi_ids = tuple(item.roi_id for item in self.per_roi_feature_event_map)
        if len(set(roi_ids)) != len(roi_ids) or set(roi_ids) != set(
            self.selected_canonical_roi_ids
        ):
            raise ValueError("Feature ROI coverage is invalid.")
        if self.inactive_for_execution is not (self.execution_mode == "tonic"):
            raise ValueError("Feature inactivity marker is invalid.")
        for name in (
            "source_feature_payload_identity",
            "source_feature_authority_identity",
            "canonical_feature_projection_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError(f"{name} is invalid.")


@dataclass(frozen=True)
class GuidedNpmStartupOutputProjection:
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
    source_output_authority_identity: str
    canonical_output_projection_identity: str

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
        if self.creation_timing != "future_execution_start_only" or self.overwrite is not False or self.precreate is not False:
            raise ValueError("Output projection is not future-only and non-mutating.")
        if not isinstance(self.relationships, tuple) or any(
            not isinstance(item, GuidedProductionOutputRelationship)
            for item in self.relationships
        ):
            raise ValueError("Output relationships are invalid.")
        if not _sha(self.source_output_authority_identity) or not _sha(
            self.canonical_output_projection_identity
        ):
            raise ValueError("Output projection identities are invalid.")


@dataclass(frozen=True)
class GuidedNpmStartupExecutionProjection:
    execution_projection_schema_name: str
    execution_projection_schema_version: str
    source_format: str
    acquisition_mode: str
    execution_mode: str
    run_type: str
    ordered_source_paths: tuple[str, ...]
    ordered_source_relative_paths: tuple[str, ...]
    ordered_source_digests: tuple[str, ...]
    source_projection_identity: str
    parser_policy_identity: str
    recording_policy_identity: str
    session_sequence_identity: str
    roi_projection_identity: str
    correction_projection_identity: str
    feature_projection_identity: str
    output_projection_identity: str
    deferred_execution_capabilities: tuple[str, ...]
    canonical_execution_projection_identity: str

    def __post_init__(self) -> None:
        if self.execution_projection_schema_name != GUIDED_NPM_STARTUP_EXECUTION_PROJECTION_SCHEMA_NAME or self.execution_projection_schema_version != GUIDED_NPM_STARTUP_EXECUTION_PROJECTION_SCHEMA_VERSION:
            raise ValueError("Execution projection schema is unsupported.")
        if self.source_format != "npm" or self.acquisition_mode != "intermittent":
            raise ValueError("Execution projection source is invalid.")
        if self.execution_mode not in {"phasic", "tonic", "both"} or self.run_type != "full":
            raise ValueError("Execution projection mode is invalid.")
        if not _text_tuple(self.ordered_source_paths, allow_empty=False) or not _text_tuple(
            self.ordered_source_relative_paths, allow_empty=False
        ) or not _text_tuple(self.ordered_source_digests, allow_empty=False):
            raise ValueError("Ordered execution sources are invalid.")
        if len(self.ordered_source_paths) != len(self.ordered_source_relative_paths) or len(
            self.ordered_source_paths
        ) != len(self.ordered_source_digests) or any(
            not _sha(item) for item in self.ordered_source_digests
        ):
            raise ValueError("Ordered execution sources differ in length.")
        for name in (
            "source_projection_identity",
            "parser_policy_identity",
            "recording_policy_identity",
            "session_sequence_identity",
            "roi_projection_identity",
            "correction_projection_identity",
            "feature_projection_identity",
            "output_projection_identity",
            "canonical_execution_projection_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError(f"{name} is invalid.")
        if self.deferred_execution_capabilities != GUIDED_NPM_STARTUP_DEFERRED_EXECUTION_CAPABILITIES:
            raise ValueError("Deferred NPM execution capabilities are invalid.")


@dataclass(frozen=True)
class GuidedNpmStartupPayload:
    startup_schema_name: str
    startup_schema_version: str
    startup_contract_version: str
    source_authorization_identity: str
    source_authority_identity: str
    source_production_intent_identity: str
    source_request_identity: str
    validation_revision: int
    guided_plan_identity: str
    application_build_identity: ApplicationBuildIdentity
    source_format: str
    acquisition_mode: str
    execution_mode: str
    run_type: str
    source_projection: GuidedNpmStartupSourceProjection
    recording_policy: GuidedNpmStartupRecordingPolicy
    roi_projection: GuidedNpmStartupRoiProjection
    correction_projection: GuidedNpmStartupCorrectionProjection
    feature_projection: GuidedNpmStartupFeatureProjection
    output_projection: GuidedNpmStartupOutputProjection
    execution_projection: GuidedNpmStartupExecutionProjection
    authorization_status: str
    payload_status: str
    persistence_status: str
    claim_status: str
    startup_status: str
    runnable: bool
    canonical_startup_payload_identity: str

    def __post_init__(self) -> None:
        if self.startup_schema_name != GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_NAME or self.startup_schema_version != GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_VERSION or self.startup_contract_version != GUIDED_NPM_STARTUP_PAYLOAD_CONTRACT_VERSION:
            raise ValueError("Startup payload schema is unsupported.")
        for name in (
            "source_authorization_identity",
            "source_authority_identity",
            "source_production_intent_identity",
            "source_request_identity",
            "guided_plan_identity",
            "canonical_startup_payload_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError(f"{name} is invalid.")
        if isinstance(self.validation_revision, bool) or not isinstance(self.validation_revision, int) or self.validation_revision < 0:
            raise ValueError("Validation revision is invalid.")
        if not isinstance(self.application_build_identity, ApplicationBuildIdentity):
            raise ValueError("Application build identity is required.")
        nested_types = (
            (self.source_projection, GuidedNpmStartupSourceProjection),
            (self.recording_policy, GuidedNpmStartupRecordingPolicy),
            (self.roi_projection, GuidedNpmStartupRoiProjection),
            (self.correction_projection, GuidedNpmStartupCorrectionProjection),
            (self.feature_projection, GuidedNpmStartupFeatureProjection),
            (self.output_projection, GuidedNpmStartupOutputProjection),
            (self.execution_projection, GuidedNpmStartupExecutionProjection),
        )
        if any(not isinstance(value, expected) for value, expected in nested_types):
            raise ValueError("Startup payload projection type is invalid.")
        if self.source_format != "npm" or self.acquisition_mode != "intermittent" or self.execution_mode not in {"phasic", "tonic", "both"} or self.run_type != "full":
            raise ValueError("Startup execution subset is invalid.")
        if self.authorization_status != GUIDED_NPM_AUTHORIZATION_STATUS_AUTHORIZED_FOR_STARTUP_PREPARATION or self.payload_status != GUIDED_NPM_STARTUP_PAYLOAD_STATUS_CONSTRUCTED_IN_MEMORY or self.persistence_status != GUIDED_NPM_STARTUP_PAYLOAD_PERSISTENCE_STATUS_NOT_PERSISTED or self.claim_status != GUIDED_NPM_STARTUP_PAYLOAD_CLAIM_STATUS_NOT_CLAIMED or self.startup_status != GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED or self.runnable is not False:
            raise ValueError("Startup payload state is invalid.")


GuidedNpmStartupPayloadResult = GuidedNpmStartupPayload | GuidedNpmStartupPayloadFailure


class _StartupRefusal(ValueError):
    def __init__(self, category: str, section: str, message: str, detail_code: str) -> None:
        self.category = category
        self.section = section
        self.message = message
        self.detail_code = detail_code
        super().__init__(message)


def _refuse(category: str, section: str, message: str, detail_code: str) -> None:
    if category not in _CATEGORY_SET:
        category = "startup_payload_internal_error"
    raise _StartupRefusal(category, section, message, detail_code)


def _failure(exc: _StartupRefusal) -> GuidedNpmStartupPayloadFailure:
    return GuidedNpmStartupPayloadFailure(
        (GuidedNpmStartupPayloadIssue(exc.category, exc.section, exc.message, exc.detail_code),)
    )


def _identity_without(value: Any, identity_name: str) -> dict[str, Any]:
    return {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name != identity_name
    }


def compute_guided_npm_startup_session_identity(value: GuidedNpmStartupSession) -> str:
    return _digest("guided_npm_startup_session.v1", _identity_without(value, "canonical_startup_session_identity"))


def compute_guided_npm_startup_session_sequence_identity(values: tuple[GuidedNpmStartupSession, ...]) -> str:
    return _digest("guided_npm_startup_session_sequence.v1", tuple({"chronological_position": item.chronological_position, "canonical_startup_session_identity": item.canonical_startup_session_identity} for item in values))


def compute_guided_npm_startup_source_projection_identity(value: GuidedNpmStartupSourceProjection) -> str:
    return _digest("guided_npm_startup_source_projection.v1", _identity_without(value, "canonical_source_projection_identity"))


def compute_guided_npm_startup_recording_policy_identity(value: GuidedNpmStartupRecordingPolicy) -> str:
    return _digest("guided_npm_startup_recording_policy.v1", _identity_without(value, "canonical_recording_policy_identity"))


def compute_guided_npm_startup_roi_projection_identity(value: GuidedNpmStartupRoiProjection) -> str:
    return _digest("guided_npm_startup_roi_projection.v1", _identity_without(value, "canonical_roi_projection_identity"))


def compute_guided_npm_startup_correction_projection_identity(value: GuidedNpmStartupCorrectionProjection) -> str:
    return _digest("guided_npm_startup_correction_projection.v1", _identity_without(value, "canonical_correction_projection_identity"))


def compute_guided_npm_startup_feature_projection_identity(value: GuidedNpmStartupFeatureProjection) -> str:
    return _digest("guided_npm_startup_feature_projection.v1", _identity_without(value, "canonical_feature_projection_identity"))


def compute_guided_npm_startup_output_projection_identity(value: GuidedNpmStartupOutputProjection) -> str:
    return _digest("guided_npm_startup_output_projection.v1", _identity_without(value, "canonical_output_projection_identity"))


def compute_guided_npm_startup_execution_projection_identity(value: GuidedNpmStartupExecutionProjection) -> str:
    return _digest("guided_npm_startup_execution_projection.v1", _identity_without(value, "canonical_execution_projection_identity"))


def compute_guided_npm_startup_payload_identity(value: GuidedNpmStartupPayload) -> str:
    return _digest(GUIDED_NPM_STARTUP_PAYLOAD_IDENTITY_DOMAIN, _identity_without(value, "canonical_startup_payload_identity"))


def _project_verified_source_file(
    verified: GuidedNpmVerifiedSourceFile,
) -> GuidedNpmStartupVerifiedSourceFileProjection:
    return GuidedNpmStartupVerifiedSourceFileProjection(
        chronological_position=verified.chronological_position,
        canonical_relative_path=verified.canonical_relative_path,
        authorized_absolute_source_reference=(
            verified.authorized_absolute_source_reference
        ),
        inspected_absolute_path=verified.inspected_absolute_path,
        expected_size_bytes=verified.expected_size_bytes,
        observed_size_bytes=verified.observed_size_bytes,
        expected_sha256_content_digest=verified.expected_sha256_content_digest,
        observed_sha256_content_digest=verified.observed_sha256_content_digest,
        pre_hash_stat_identity=verified.pre_hash_stat_identity,
        post_hash_stat_identity=verified.post_hash_stat_identity,
        canonical_verified_file_identity=(
            verified.canonical_verified_file_identity
        ),
    )


def _reconstruct_verified_source_file(
    projection: GuidedNpmStartupVerifiedSourceFileProjection,
) -> GuidedNpmVerifiedSourceFile:
    return GuidedNpmVerifiedSourceFile(
        chronological_position=projection.chronological_position,
        canonical_relative_path=projection.canonical_relative_path,
        authorized_absolute_source_reference=(
            projection.authorized_absolute_source_reference
        ),
        inspected_absolute_path=projection.inspected_absolute_path,
        expected_size_bytes=projection.expected_size_bytes,
        observed_size_bytes=projection.observed_size_bytes,
        expected_sha256_content_digest=(
            projection.expected_sha256_content_digest
        ),
        observed_sha256_content_digest=(
            projection.observed_sha256_content_digest
        ),
        pre_hash_stat_identity=projection.pre_hash_stat_identity,
        post_hash_stat_identity=projection.post_hash_stat_identity,
        canonical_verified_file_identity=(
            projection.canonical_verified_file_identity
        ),
    )


def _reconstruct_verified_source_snapshot(
    source: GuidedNpmStartupSourceProjection,
) -> GuidedNpmVerifiedSourceSnapshot:
    return GuidedNpmVerifiedSourceSnapshot(
        source_root_canonical=source.source_root_canonical,
        source_root_inspected=source.source_root_inspected,
        discovery_contract_version=source.discovery_contract_version,
        ordered_files=tuple(
            _reconstruct_verified_source_file(item.verified_source_file)
            for item in source.ordered_sessions
        ),
        ordered_file_sequence_identity=(
            source.verified_ordered_file_sequence_identity
        ),
        source_set_identity=source.verified_source_set_identity,
        source_content_identity=source.verified_source_content_identity,
        canonical_verified_snapshot_identity=(
            source.verified_source_snapshot_identity
        ),
    )


def _map_entry(value: Any) -> GuidedNpmStartupRoiMappingEntry:
    return GuidedNpmStartupRoiMappingEntry(
        physical_source_column=value.physical_source_column,
        canonical_roi_id=value.canonical_roi_id,
    )


def _verify_inputs(authorization: Any, authority: Any) -> None:
    if type(authorization) is not GuidedNpmExecutionAuthorization:
        _refuse("authorization_missing_or_invalid", "authorization", "A valid B2-C3 authorization is required.", "authorization_type_invalid")
    if authorization.authorization_schema_name != GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_NAME or authorization.authorization_schema_version != GUIDED_NPM_EXECUTION_AUTHORIZATION_SCHEMA_VERSION or authorization.authorization_contract_version != GUIDED_NPM_EXECUTION_AUTHORIZATION_CONTRACT_VERSION:
        _refuse("authorization_schema_unsupported", "authorization", "The B2-C3 authorization schema is unsupported.", "authorization_schema_unsupported")
    if authorization.authorization_status != GUIDED_NPM_AUTHORIZATION_STATUS_AUTHORIZED_FOR_STARTUP_PREPARATION or authorization.startup_status != GUIDED_NPM_AUTHORIZATION_STARTUP_STATUS_NOT_MATERIALIZED or authorization.runnable is not False:
        _refuse("authorization_state_invalid", "authorization", "The B2-C3 authorization state is invalid.", "authorization_state_invalid")
    try:
        verify_guided_npm_execution_authorization(authorization)
    except (TypeError, ValueError) as exc:
        _refuse("authorization_identity_mismatch", "authorization", "The B2-C3 authorization identity chain is invalid.", str(exc) or "authorization_identity_mismatch")
    if type(authority) is not GuidedNpmExecutionAuthority:
        _refuse("authority_missing_or_invalid", "authority", "A valid B2-C2 authority is required.", "authority_type_invalid")
    if authority.authority_schema_name != GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_NAME or authority.authority_schema_version != GUIDED_NPM_EXECUTION_AUTHORITY_SCHEMA_VERSION or authority.authority_contract_version != GUIDED_NPM_EXECUTION_AUTHORITY_CONTRACT_VERSION:
        _refuse("authority_schema_unsupported", "authority", "The B2-C2 authority schema is unsupported.", "authority_schema_unsupported")
    if authority.authorization_status != GUIDED_NPM_AUTHORIZATION_STATUS_NOT_AUTHORIZED or authority.startup_status != GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED or authority.runnable is not False:
        _refuse("authority_state_invalid", "authority", "The B2-C2 authority state is invalid.", "authority_state_invalid")
    try:
        verify_guided_npm_execution_authority(authority)
    except (TypeError, ValueError) as exc:
        _refuse("authority_identity_mismatch", "authority", "The B2-C2 authority identity chain is invalid.", str(exc) or "authority_identity_mismatch")


def _verify_cross_object(authorization: GuidedNpmExecutionAuthorization, authority: GuidedNpmExecutionAuthority) -> None:
    checks = (
        (authorization.source_authority_identity, authority.canonical_authority_identity, "authorization_authority_mismatch"),
        (authorization.source_production_intent_identity, authority.source_production_intent_identity, "production_intent_identity_mismatch"),
        (authorization.source_request_identity, authority.source_request_identity, "request_identity_mismatch"),
        (authorization.validation_revision, authority.validation_revision, "validation_revision_mismatch"),
        (authorization.guided_plan_identity, authority.guided_plan_identity, "guided_plan_identity_mismatch"),
        (authorization.application_build_identity, authority.application_build_identity, "application_build_identity_mismatch"),
        (authorization.execution_mode, authority.execution_mode, "execution_mode_mismatch"),
        (authorization.selected_canonical_roi_ids, authority.roi_authority.selected_canonical_roi_ids, "selected_roi_scope_mismatch"),
        (authorization.correction_authority_identity, authority.correction_authority.canonical_correction_authority_identity, "correction_authority_identity_mismatch"),
        (authorization.feature_authority_identity, authority.feature_authority.canonical_feature_authority_identity, "feature_authority_identity_mismatch"),
        (authorization.output_authority_identity, authority.output_authority.canonical_output_authority_identity, "output_authority_identity_mismatch"),
    )
    for left, right, category in checks:
        if left != right:
            _refuse(category, "binding", "The authorization does not match its B2-C2 authority.", category)


def _startup_sessions(authorization: GuidedNpmExecutionAuthorization, authority: GuidedNpmExecutionAuthority) -> tuple[GuidedNpmStartupSession, ...]:
    verified = authorization.verified_source_snapshot.ordered_files
    if len(verified) < len(authority.sessions):
        _refuse("verified_source_file_missing", "source", "A verified source file is missing.", "verified_source_file_count_missing")
    if len(verified) > len(authority.sessions):
        _refuse("verified_source_file_extra", "source", "An extra verified source file is present.", "verified_source_file_count_extra")
    if len({item.canonical_relative_path for item in verified}) != len(verified):
        _refuse("verified_source_file_duplicate", "source", "Verified source paths are duplicated.", "verified_source_file_duplicate")
    sessions: list[GuidedNpmStartupSession] = []
    for position, (source, current) in enumerate(zip(authority.sessions, verified)):
        if source.chronological_position != position or current.chronological_position != position:
            _refuse("verified_source_file_order_mismatch", "source", "Verified source positions differ from authority order.", "verified_source_file_position_mismatch")
        if source.canonical_relative_path != current.canonical_relative_path or source.authorized_absolute_source_reference != current.authorized_absolute_source_reference:
            _refuse("verified_source_file_path_mismatch", "source", "Verified source paths differ from authority paths.", "verified_source_file_path_mismatch")
        if source.size_bytes != current.expected_size_bytes or source.size_bytes != current.observed_size_bytes:
            _refuse("verified_source_file_size_mismatch", "source", "Verified source size differs from authority.", "verified_source_file_size_mismatch")
        if source.sha256_content_digest != current.expected_sha256_content_digest or source.sha256_content_digest != current.observed_sha256_content_digest:
            _refuse("verified_source_file_digest_mismatch", "source", "Verified source digest differs from authority.", "verified_source_file_digest_mismatch")
        session = GuidedNpmStartupSession(
            chronological_position=position,
            canonical_relative_path=source.canonical_relative_path,
            authorized_absolute_source_reference=source.authorized_absolute_source_reference,
            size_bytes=source.size_bytes,
            sha256_content_digest=source.sha256_content_digest,
            verified_source_file=_project_verified_source_file(current),
            authoritative_source_start_time=source.authoritative_source_start_time,
            actual_elapsed_sec=source.actual_elapsed_sec,
            nominal_expected_elapsed_sec=source.nominal_expected_elapsed_sec,
            resolved_timestamp_column=source.resolved_timestamp_column,
            resolved_led_column=source.resolved_led_column,
            timestamp_unit=source.timestamp_unit,
            source_timing_evidence=source.source_timing_evidence,
            physical_roi_inventory=source.physical_roi_inventory,
            physical_to_canonical_roi_mapping=tuple(_map_entry(item) for item in source.physical_to_canonical_roi_mapping),
            overlap_origin_absolute=source.overlap_origin_absolute,
            resolved_support_start_offset_sec=source.resolved_support_start_offset_sec,
            resolved_support_end_offset_sec=source.resolved_support_end_offset_sec,
            resolved_support_start_absolute=source.resolved_support_start_absolute,
            resolved_support_end_absolute=source.resolved_support_end_absolute,
            observed_support_duration_sec=source.observed_support_duration_sec,
            support_policy=source.support_policy,
            support_policy_identity=source.support_policy_identity,
            output_time_basis=source.output_time_basis,
            warning_categories=source.warning_categories,
            canonical_startup_session_identity="0" * 64,
        )
        sessions.append(replace(session, canonical_startup_session_identity=compute_guided_npm_startup_session_identity(session)))
    return tuple(sessions)


def _build_payload(authorization: GuidedNpmExecutionAuthorization, authority: GuidedNpmExecutionAuthority) -> GuidedNpmStartupPayload:
    _verify_inputs(authorization, authority)
    _verify_cross_object(authorization, authority)
    snapshot = authorization.verified_source_snapshot
    if authorization.source_root_canonical != snapshot.source_root_canonical:
        _refuse("verified_source_snapshot_mismatch", "source", "Verified source root differs from authorization.", "verified_source_root_mismatch")
    sessions = _startup_sessions(authorization, authority)
    source = GuidedNpmStartupSourceProjection(
        source_root_canonical=authorization.source_root_canonical,
        source_root_inspected=snapshot.source_root_inspected,
        discovery_contract_version=snapshot.discovery_contract_version,
        ordered_sessions=sessions,
        ordered_session_sequence_identity=compute_guided_npm_startup_session_sequence_identity(sessions),
        verified_ordered_file_sequence_identity=(
            snapshot.ordered_file_sequence_identity
        ),
        verified_source_set_identity=snapshot.source_set_identity,
        verified_source_content_identity=snapshot.source_content_identity,
        verified_source_snapshot_identity=snapshot.canonical_verified_snapshot_identity,
        canonical_source_projection_identity="0" * 64,
    )
    source = replace(source, canonical_source_projection_identity=compute_guided_npm_startup_source_projection_identity(source))
    policy = authority.recording_policy
    recording = GuidedNpmStartupRecordingPolicy(
        parser_policy_identity=policy.parser_policy_identity,
        parser_policy_content_json=policy.parser_policy_content_json,
        time_axis_mode=policy.time_axis_mode,
        ordered_timestamp_candidates=policy.ordered_timestamp_candidates,
        timestamp_unit=policy.timestamp_unit,
        timestamp_finite_policy=policy.timestamp_finite_policy,
        led_state_column=policy.led_state_column,
        reference_led_value=policy.reference_led_value,
        signal_led_value=policy.signal_led_value,
        roi_prefix=policy.roi_prefix,
        roi_suffix=policy.roi_suffix,
        roi_ordering_rule=policy.roi_ordering_rule,
        canonical_roi_naming_rule=policy.canonical_roi_naming_rule,
        support_policy=policy.support_policy,
        support_policy_identity=policy.support_policy_identity,
        roi_value_nan_policy=policy.roi_value_nan_policy,
        target_fs_hz=policy.target_fs_hz,
        configured_session_duration_sec=policy.configured_session_duration_sec,
        sessions_per_hour=policy.sessions_per_hour,
        chronology_policy=policy.chronology_policy,
        gap_policy=policy.gap_policy,
        overlap_policy=policy.overlap_policy,
        output_time_basis=policy.output_time_basis,
        source_recording_policy_identity=policy.canonical_policy_identity,
        canonical_recording_policy_identity="0" * 64,
    )
    recording = replace(recording, canonical_recording_policy_identity=compute_guided_npm_startup_recording_policy_identity(recording))
    roi_authority = authority.roi_authority
    roi = GuidedNpmStartupRoiProjection(
        complete_canonical_roi_ids=roi_authority.complete_canonical_roi_ids,
        selected_canonical_roi_ids=roi_authority.selected_canonical_roi_ids,
        excluded_canonical_roi_ids=roi_authority.excluded_canonical_roi_ids,
        complete_physical_source_columns=roi_authority.complete_physical_source_columns,
        physical_to_canonical_roi_mapping=tuple(_map_entry(item) for item in roi_authority.physical_to_canonical_roi_mapping),
        selected_physical_source_columns=roi_authority.selected_physical_source_columns,
        selected_physical_to_canonical_roi_mapping=tuple(_map_entry(item) for item in roi_authority.selected_physical_to_canonical_roi_mapping),
        roi_ordering_rule=roi_authority.roi_ordering_rule,
        canonical_roi_naming_rule=roi_authority.canonical_roi_naming_rule,
        source_roi_authority_identity=roi_authority.canonical_roi_authority_identity,
        canonical_roi_projection_identity="0" * 64,
    )
    roi = replace(roi, canonical_roi_projection_identity=compute_guided_npm_startup_roi_projection_identity(roi))
    correction_authority = authority.correction_authority
    correction = GuidedNpmStartupCorrectionProjection(
        selected_canonical_roi_ids=correction_authority.selected_canonical_roi_ids,
        correction_parameter_values=correction_authority.correction_parameter_values,
        per_roi_correction_strategy_map=correction_authority.per_roi_correction_strategy_map,
        source_correction_payload_identity=correction_authority.correction_payload_identity,
        source_correction_authority_identity=correction_authority.canonical_correction_authority_identity,
        canonical_correction_projection_identity="0" * 64,
    )
    correction = replace(correction, canonical_correction_projection_identity=compute_guided_npm_startup_correction_projection_identity(correction))
    feature_authority = authority.feature_authority
    feature = GuidedNpmStartupFeatureProjection(
        execution_mode=feature_authority.execution_mode,
        profile_schema_version=feature_authority.profile_schema_version,
        profile_id=feature_authority.profile_id,
        effective_values=feature_authority.effective_values,
        active_fields=feature_authority.active_fields,
        inactive_fields=feature_authority.inactive_fields,
        profile_status=feature_authority.profile_status,
        explicitly_applied=feature_authority.explicitly_applied,
        current=feature_authority.current,
        visible_unapplied_changes=feature_authority.visible_unapplied_changes,
        per_roi_feature_event_map_version=feature_authority.per_roi_feature_event_map_version,
        per_roi_feature_event_map=feature_authority.per_roi_feature_event_map,
        selected_canonical_roi_ids=feature_authority.selected_canonical_roi_ids,
        inactive_for_execution=feature_authority.inactive_for_execution,
        source_feature_payload_identity=feature_authority.feature_payload_identity,
        source_feature_authority_identity=feature_authority.canonical_feature_authority_identity,
        canonical_feature_projection_identity="0" * 64,
    )
    feature = replace(feature, canonical_feature_projection_identity=compute_guided_npm_startup_feature_projection_identity(feature))
    output_authority = authority.output_authority
    output = GuidedNpmStartupOutputProjection(
        output_base_canonical=output_authority.output_base_canonical,
        output_base_path_style=output_authority.output_base_path_style,
        path_role=output_authority.path_role,
        future_output_owner=output_authority.future_output_owner,
        run_directory_strategy=output_authority.run_directory_strategy,
        creation_timing=output_authority.creation_timing,
        overwrite=output_authority.overwrite,
        precreate=output_authority.precreate,
        safety_classifier_version=output_authority.safety_classifier_version,
        relationships=output_authority.relationships,
        protected_root_context_complete=output_authority.protected_root_context_complete,
        filesystem_fact_scope=output_authority.filesystem_fact_scope,
        source_output_authority_identity=output_authority.canonical_output_authority_identity,
        canonical_output_projection_identity="0" * 64,
    )
    output = replace(output, canonical_output_projection_identity=compute_guided_npm_startup_output_projection_identity(output))
    execution = GuidedNpmStartupExecutionProjection(
        execution_projection_schema_name=GUIDED_NPM_STARTUP_EXECUTION_PROJECTION_SCHEMA_NAME,
        execution_projection_schema_version=GUIDED_NPM_STARTUP_EXECUTION_PROJECTION_SCHEMA_VERSION,
        source_format=authority.source_format,
        acquisition_mode=authority.acquisition_mode,
        execution_mode=authority.execution_mode,
        run_type=authority.run_type,
        ordered_source_paths=tuple(item.authorized_absolute_source_reference for item in sessions),
        ordered_source_relative_paths=tuple(item.canonical_relative_path for item in sessions),
        ordered_source_digests=tuple(
            item.verified_source_file.observed_sha256_content_digest
            for item in sessions
        ),
        source_projection_identity=source.canonical_source_projection_identity,
        parser_policy_identity=recording.parser_policy_identity,
        recording_policy_identity=recording.canonical_recording_policy_identity,
        session_sequence_identity=source.ordered_session_sequence_identity,
        roi_projection_identity=roi.canonical_roi_projection_identity,
        correction_projection_identity=correction.canonical_correction_projection_identity,
        feature_projection_identity=feature.canonical_feature_projection_identity,
        output_projection_identity=output.canonical_output_projection_identity,
        deferred_execution_capabilities=GUIDED_NPM_STARTUP_DEFERRED_EXECUTION_CAPABILITIES,
        canonical_execution_projection_identity="0" * 64,
    )
    execution = replace(execution, canonical_execution_projection_identity=compute_guided_npm_startup_execution_projection_identity(execution))
    payload = GuidedNpmStartupPayload(
        startup_schema_name=GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_NAME,
        startup_schema_version=GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_VERSION,
        startup_contract_version=GUIDED_NPM_STARTUP_PAYLOAD_CONTRACT_VERSION,
        source_authorization_identity=authorization.canonical_authorization_identity,
        source_authority_identity=authority.canonical_authority_identity,
        source_production_intent_identity=authority.source_production_intent_identity,
        source_request_identity=authority.source_request_identity,
        validation_revision=authority.validation_revision,
        guided_plan_identity=authority.guided_plan_identity,
        application_build_identity=authority.application_build_identity,
        source_format=authority.source_format,
        acquisition_mode=authority.acquisition_mode,
        execution_mode=authority.execution_mode,
        run_type=authority.run_type,
        source_projection=source,
        recording_policy=recording,
        roi_projection=roi,
        correction_projection=correction,
        feature_projection=feature,
        output_projection=output,
        execution_projection=execution,
        authorization_status=GUIDED_NPM_AUTHORIZATION_STATUS_AUTHORIZED_FOR_STARTUP_PREPARATION,
        payload_status=GUIDED_NPM_STARTUP_PAYLOAD_STATUS_CONSTRUCTED_IN_MEMORY,
        persistence_status=GUIDED_NPM_STARTUP_PAYLOAD_PERSISTENCE_STATUS_NOT_PERSISTED,
        claim_status=GUIDED_NPM_STARTUP_PAYLOAD_CLAIM_STATUS_NOT_CLAIMED,
        startup_status=GUIDED_NPM_STARTUP_STATUS_NOT_MATERIALIZED,
        runnable=False,
        canonical_startup_payload_identity="0" * 64,
    )
    payload = replace(
        payload,
        canonical_startup_payload_identity=(
            compute_guided_npm_startup_payload_identity(payload)
        ),
    )
    verify_guided_npm_startup_payload(payload)
    return payload


def build_guided_npm_startup_payload(authorization: GuidedNpmExecutionAuthorization, authority: GuidedNpmExecutionAuthority) -> GuidedNpmStartupPayloadResult:
    try:
        return _build_payload(authorization, authority)
    except _StartupRefusal as exc:
        return _failure(exc)
    except Exception as exc:
        return _failure(_StartupRefusal("startup_payload_internal_error", "startup_payload", "NPM startup-payload construction failed.", type(exc).__name__))


def _verify_build_identity(identity: ApplicationBuildIdentity) -> None:
    rebuilt = build_application_build_identity(
        distribution_name=identity.distribution_name,
        distribution_version=identity.distribution_version,
        source_revision_kind=identity.source_revision_kind,
        source_revision=identity.source_revision,
        source_tree_state=identity.source_tree_state,
        source_tree_digest=identity.source_tree_digest,
        build_artifact_digest=identity.build_artifact_digest,
        identity_provider_version=identity.identity_provider_version,
    )
    if rebuilt != identity:
        raise ValueError("application_build_identity_mismatch")


def _verify_startup_payload_identity_chain(payload: GuidedNpmStartupPayload) -> None:
    _verify_build_identity(payload.application_build_identity)
    source = payload.source_projection
    sessions = source.ordered_sessions
    verified_snapshot = _reconstruct_verified_source_snapshot(source)
    verified_files = verified_snapshot.ordered_files
    for session, verified in zip(sessions, verified_files):
        replace(session.verified_source_file)
        if (
            session.chronological_position != verified.chronological_position
            or session.canonical_relative_path != verified.canonical_relative_path
            or session.authorized_absolute_source_reference
            != verified.authorized_absolute_source_reference
        ):
            raise ValueError("startup_verified_file_path_mismatch")
        if (
            session.size_bytes != verified.expected_size_bytes
            or session.size_bytes != verified.observed_size_bytes
        ):
            raise ValueError("startup_verified_file_size_mismatch")
        if (
            session.sha256_content_digest
            != verified.expected_sha256_content_digest
            or session.sha256_content_digest
            != verified.observed_sha256_content_digest
        ):
            raise ValueError("startup_verified_file_digest_mismatch")
        if (
            compute_guided_npm_verified_source_file_identity(verified)
            != verified.canonical_verified_file_identity
        ):
            raise ValueError("startup_verified_file_identity_mismatch")
    if (
        compute_guided_npm_verified_source_sequence_identity(verified_files)
        != source.verified_ordered_file_sequence_identity
    ):
        raise ValueError("startup_verified_source_sequence_identity_mismatch")
    if (
        compute_guided_npm_verified_source_set_identity(verified_files)
        != source.verified_source_set_identity
    ):
        raise ValueError("startup_verified_source_set_identity_mismatch")
    if (
        compute_guided_npm_verified_source_content_identity(verified_files)
        != source.verified_source_content_identity
    ):
        raise ValueError("startup_verified_source_content_identity_mismatch")
    if (
        compute_guided_npm_verified_source_snapshot_identity(verified_snapshot)
        != source.verified_source_snapshot_identity
    ):
        raise ValueError("startup_verified_source_snapshot_identity_mismatch")
    for session in sessions:
        replace(session)
        if (
            compute_guided_npm_startup_session_identity(session)
            != session.canonical_startup_session_identity
        ):
            raise ValueError("startup_session_identity_mismatch")
    if (
        compute_guided_npm_startup_session_sequence_identity(sessions)
        != source.ordered_session_sequence_identity
    ):
        raise ValueError("startup_session_sequence_identity_mismatch")
    projections = (
        (
            source,
            compute_guided_npm_startup_source_projection_identity,
            source.canonical_source_projection_identity,
            "source_projection_identity_mismatch",
        ),
        (
            payload.recording_policy,
            compute_guided_npm_startup_recording_policy_identity,
            payload.recording_policy.canonical_recording_policy_identity,
            "recording_policy_projection_identity_mismatch",
        ),
        (
            payload.roi_projection,
            compute_guided_npm_startup_roi_projection_identity,
            payload.roi_projection.canonical_roi_projection_identity,
            "roi_projection_identity_mismatch",
        ),
        (
            payload.correction_projection,
            compute_guided_npm_startup_correction_projection_identity,
            payload.correction_projection.canonical_correction_projection_identity,
            "correction_projection_identity_mismatch",
        ),
        (
            payload.feature_projection,
            compute_guided_npm_startup_feature_projection_identity,
            payload.feature_projection.canonical_feature_projection_identity,
            "feature_projection_identity_mismatch",
        ),
        (
            payload.output_projection,
            compute_guided_npm_startup_output_projection_identity,
            payload.output_projection.canonical_output_projection_identity,
            "output_projection_identity_mismatch",
        ),
        (
            payload.execution_projection,
            compute_guided_npm_startup_execution_projection_identity,
            payload.execution_projection.canonical_execution_projection_identity,
            "execution_projection_identity_mismatch",
        ),
    )
    for projection, compute, identity, message in projections:
        replace(projection)
        if compute(projection) != identity:
            raise ValueError(message)

    recording = payload.recording_policy
    roi = payload.roi_projection
    correction = payload.correction_projection
    feature = payload.feature_projection
    execution = payload.execution_projection
    if execution.ordered_source_paths != tuple(
        item.authorized_absolute_source_reference for item in sessions
    ):
        raise ValueError("execution_source_path_mismatch")
    if execution.ordered_source_relative_paths != tuple(
        item.canonical_relative_path for item in sessions
    ):
        raise ValueError("execution_source_relative_path_mismatch")
    if execution.ordered_source_digests != tuple(
        item.verified_source_file.observed_sha256_content_digest
        for item in sessions
    ):
        raise ValueError("execution_source_digest_mismatch")
    reference_identities = (
        (
            execution.source_projection_identity,
            source.canonical_source_projection_identity,
        ),
        (execution.parser_policy_identity, recording.parser_policy_identity),
        (
            execution.recording_policy_identity,
            recording.canonical_recording_policy_identity,
        ),
        (execution.session_sequence_identity, source.ordered_session_sequence_identity),
        (execution.roi_projection_identity, roi.canonical_roi_projection_identity),
        (
            execution.correction_projection_identity,
            correction.canonical_correction_projection_identity,
        ),
        (
            execution.feature_projection_identity,
            feature.canonical_feature_projection_identity,
        ),
        (
            execution.output_projection_identity,
            payload.output_projection.canonical_output_projection_identity,
        ),
    )
    if any(actual != expected for actual, expected in reference_identities):
        raise ValueError("execution_projection_reference_mismatch")
    if (
        correction.selected_canonical_roi_ids != roi.selected_canonical_roi_ids
        or feature.selected_canonical_roi_ids != roi.selected_canonical_roi_ids
    ):
        raise ValueError("selected_roi_scope_mismatch")
    if (
        payload.source_format != execution.source_format
        or payload.acquisition_mode != execution.acquisition_mode
        or payload.execution_mode != execution.execution_mode
        or payload.execution_mode != feature.execution_mode
        or payload.run_type != execution.run_type
    ):
        raise ValueError("execution_projection_mode_mismatch")
    replace(payload)
    if (
        compute_guided_npm_startup_payload_identity(payload)
        != payload.canonical_startup_payload_identity
    ):
        raise ValueError("startup_payload_identity_mismatch")


def verify_guided_npm_startup_payload(payload: GuidedNpmStartupPayload) -> None:
    """Reverify the complete immutable B2-C4 payload without filesystem I/O."""
    if type(payload) is not GuidedNpmStartupPayload:
        raise ValueError("startup_payload_type_invalid")
    _verify_startup_payload_identity_chain(payload)


def _serialize_startup_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("startup_payload_serialization_invalid")
        return value
    if isinstance(value, tuple):
        return [_serialize_startup_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("startup_payload_serialization_invalid")
        return {key: _serialize_startup_value(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            item.name: _serialize_startup_value(getattr(value, item.name))
            for item in fields(value)
        }
    raise ValueError("startup_payload_serialization_invalid")


def serialize_guided_npm_startup_payload(
    payload: GuidedNpmStartupPayload,
) -> dict[str, Any]:
    """Return a deterministic JSON-compatible representation of a B2-C4 payload."""
    verify_guided_npm_startup_payload(payload)
    return {
        "identity_domain": GUIDED_NPM_STARTUP_PAYLOAD_IDENTITY_DOMAIN,
        **_serialize_startup_value(payload),
    }


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError("startup_payload_serialization_invalid")
    return payload[key]


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required(payload, key)
    if not isinstance(value, Mapping):
        raise ValueError("startup_payload_serialization_invalid")
    return value


def _sequence(payload: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    value = _required(payload, key)
    if not isinstance(value, (list, tuple)):
        raise ValueError("startup_payload_serialization_invalid")
    return tuple(value)


def _typed_value(payload: Mapping[str, Any]) -> GuidedProductionTypedValue:
    return GuidedProductionTypedValue(
        field_name=_required(payload, "field_name"),
        value_type=_required(payload, "value_type"),
        value=_required(payload, "value"),
        source_classification=_required(payload, "source_classification"),
    )


def _strategy(payload: Mapping[str, Any]) -> GuidedProductionPerRoiStrategy:
    return GuidedProductionPerRoiStrategy(
        roi_id=_required(payload, "roi_id"),
        strategy_family=_required(payload, "strategy_family"),
        dynamic_fit_mode=payload.get("dynamic_fit_mode"),
        selected_strategy=_required(payload, "selected_strategy"),
        evidence_source_type=_required(payload, "evidence_source_type"),
        evidence_reference_json=_required(payload, "evidence_reference_json"),
        explicit_user_mark=_required(payload, "explicit_user_mark"),
        current_or_stale=_required(payload, "current_or_stale"),
    )


def _feature_event(payload: Mapping[str, Any]) -> GuidedProductionPerRoiFeatureEvent:
    return GuidedProductionPerRoiFeatureEvent(
        roi_id=_required(payload, "roi_id"),
        source=_required(payload, "source"),
        feature_event_profile_id=_required(payload, "feature_event_profile_id"),
        override_config_fields=tuple(
            _typed_value(item)
            for item in _sequence(payload, "override_config_fields")
        ),
        effective_config_fields=tuple(
            _typed_value(item)
            for item in _sequence(payload, "effective_config_fields")
        ),
        explicit_user_mark=_required(payload, "explicit_user_mark"),
        current_or_stale=_required(payload, "current_or_stale"),
    )


def _relationship(payload: Mapping[str, Any]) -> GuidedProductionOutputRelationship:
    return GuidedProductionOutputRelationship(
        relationship=_required(payload, "relationship"),
        root_kind=_required(payload, "root_kind"),
        status=_required(payload, "status"),
    )


def _roi_mapping(payload: Mapping[str, Any]) -> GuidedNpmStartupRoiMappingEntry:
    return GuidedNpmStartupRoiMappingEntry(
        physical_source_column=_required(payload, "physical_source_column"),
        canonical_roi_id=_required(payload, "canonical_roi_id"),
    )


def _verified_source_file_projection(
    payload: Mapping[str, Any],
) -> GuidedNpmStartupVerifiedSourceFileProjection:
    return GuidedNpmStartupVerifiedSourceFileProjection(
        chronological_position=_required(payload, "chronological_position"),
        canonical_relative_path=_required(payload, "canonical_relative_path"),
        authorized_absolute_source_reference=_required(
            payload, "authorized_absolute_source_reference"
        ),
        inspected_absolute_path=_required(payload, "inspected_absolute_path"),
        expected_size_bytes=_required(payload, "expected_size_bytes"),
        observed_size_bytes=_required(payload, "observed_size_bytes"),
        expected_sha256_content_digest=_required(
            payload, "expected_sha256_content_digest"
        ),
        observed_sha256_content_digest=_required(
            payload, "observed_sha256_content_digest"
        ),
        pre_hash_stat_identity=_required(payload, "pre_hash_stat_identity"),
        post_hash_stat_identity=_required(payload, "post_hash_stat_identity"),
        canonical_verified_file_identity=_required(
            payload, "canonical_verified_file_identity"
        ),
    )


def _deserialize_startup_payload(payload: Mapping[str, Any]) -> GuidedNpmStartupPayload:
    if _required(payload, "identity_domain") != GUIDED_NPM_STARTUP_PAYLOAD_IDENTITY_DOMAIN:
        raise ValueError("startup_payload_serialization_invalid")
    source_data = _mapping(payload, "source_projection")
    sessions = tuple(
        GuidedNpmStartupSession(
            chronological_position=_required(item, "chronological_position"),
            canonical_relative_path=_required(item, "canonical_relative_path"),
            authorized_absolute_source_reference=_required(item, "authorized_absolute_source_reference"),
            size_bytes=_required(item, "size_bytes"),
            sha256_content_digest=_required(item, "sha256_content_digest"),
            verified_source_file=_verified_source_file_projection(
                _mapping(item, "verified_source_file")
            ),
            authoritative_source_start_time=_required(item, "authoritative_source_start_time"),
            actual_elapsed_sec=_required(item, "actual_elapsed_sec"),
            nominal_expected_elapsed_sec=_required(item, "nominal_expected_elapsed_sec"),
            resolved_timestamp_column=_required(item, "resolved_timestamp_column"),
            resolved_led_column=item.get("resolved_led_column"),
            timestamp_unit=_required(item, "timestamp_unit"),
            source_timing_evidence=_required(item, "source_timing_evidence"),
            physical_roi_inventory=tuple(_sequence(item, "physical_roi_inventory")),
            physical_to_canonical_roi_mapping=tuple(
                _roi_mapping(entry)
                for entry in _sequence(item, "physical_to_canonical_roi_mapping")
            ),
            overlap_origin_absolute=_required(item, "overlap_origin_absolute"),
            resolved_support_start_offset_sec=_required(item, "resolved_support_start_offset_sec"),
            resolved_support_end_offset_sec=_required(item, "resolved_support_end_offset_sec"),
            resolved_support_start_absolute=_required(item, "resolved_support_start_absolute"),
            resolved_support_end_absolute=_required(item, "resolved_support_end_absolute"),
            observed_support_duration_sec=_required(item, "observed_support_duration_sec"),
            support_policy=_required(item, "support_policy"),
            support_policy_identity=_required(item, "support_policy_identity"),
            output_time_basis=_required(item, "output_time_basis"),
            warning_categories=tuple(_sequence(item, "warning_categories")),
            canonical_startup_session_identity=_required(item, "canonical_startup_session_identity"),
        )
        for item in _sequence(source_data, "ordered_sessions")
    )
    source = GuidedNpmStartupSourceProjection(
        source_root_canonical=_required(source_data, "source_root_canonical"),
        source_root_inspected=_required(source_data, "source_root_inspected"),
        discovery_contract_version=_required(
            source_data, "discovery_contract_version"
        ),
        ordered_sessions=sessions,
        ordered_session_sequence_identity=_required(source_data, "ordered_session_sequence_identity"),
        verified_ordered_file_sequence_identity=_required(
            source_data, "verified_ordered_file_sequence_identity"
        ),
        verified_source_set_identity=_required(source_data, "verified_source_set_identity"),
        verified_source_content_identity=_required(source_data, "verified_source_content_identity"),
        verified_source_snapshot_identity=_required(source_data, "verified_source_snapshot_identity"),
        canonical_source_projection_identity=_required(source_data, "canonical_source_projection_identity"),
    )
    recording_data = _mapping(payload, "recording_policy")
    recording = GuidedNpmStartupRecordingPolicy(
        parser_policy_identity=_required(recording_data, "parser_policy_identity"),
        parser_policy_content_json=_required(recording_data, "parser_policy_content_json"),
        time_axis_mode=_required(recording_data, "time_axis_mode"),
        ordered_timestamp_candidates=tuple(_sequence(recording_data, "ordered_timestamp_candidates")),
        timestamp_unit=_required(recording_data, "timestamp_unit"),
        timestamp_finite_policy=_required(recording_data, "timestamp_finite_policy"),
        led_state_column=_required(recording_data, "led_state_column"),
        reference_led_value=_required(recording_data, "reference_led_value"),
        signal_led_value=_required(recording_data, "signal_led_value"),
        roi_prefix=_required(recording_data, "roi_prefix"),
        roi_suffix=_required(recording_data, "roi_suffix"),
        roi_ordering_rule=_required(recording_data, "roi_ordering_rule"),
        canonical_roi_naming_rule=_required(recording_data, "canonical_roi_naming_rule"),
        support_policy=_required(recording_data, "support_policy"),
        support_policy_identity=_required(recording_data, "support_policy_identity"),
        roi_value_nan_policy=_required(recording_data, "roi_value_nan_policy"),
        target_fs_hz=_required(recording_data, "target_fs_hz"),
        configured_session_duration_sec=_required(recording_data, "configured_session_duration_sec"),
        sessions_per_hour=_required(recording_data, "sessions_per_hour"),
        chronology_policy=_required(recording_data, "chronology_policy"),
        gap_policy=_required(recording_data, "gap_policy"),
        overlap_policy=_required(recording_data, "overlap_policy"),
        output_time_basis=_required(recording_data, "output_time_basis"),
        source_recording_policy_identity=_required(recording_data, "source_recording_policy_identity"),
        canonical_recording_policy_identity=_required(recording_data, "canonical_recording_policy_identity"),
    )
    roi_data = _mapping(payload, "roi_projection")
    roi = GuidedNpmStartupRoiProjection(
        complete_canonical_roi_ids=tuple(_sequence(roi_data, "complete_canonical_roi_ids")),
        selected_canonical_roi_ids=tuple(_sequence(roi_data, "selected_canonical_roi_ids")),
        excluded_canonical_roi_ids=tuple(_sequence(roi_data, "excluded_canonical_roi_ids")),
        complete_physical_source_columns=tuple(_sequence(roi_data, "complete_physical_source_columns")),
        physical_to_canonical_roi_mapping=tuple(
            _roi_mapping(item)
            for item in _sequence(roi_data, "physical_to_canonical_roi_mapping")
        ),
        selected_physical_source_columns=tuple(_sequence(roi_data, "selected_physical_source_columns")),
        selected_physical_to_canonical_roi_mapping=tuple(
            _roi_mapping(item)
            for item in _sequence(roi_data, "selected_physical_to_canonical_roi_mapping")
        ),
        roi_ordering_rule=_required(roi_data, "roi_ordering_rule"),
        canonical_roi_naming_rule=_required(roi_data, "canonical_roi_naming_rule"),
        source_roi_authority_identity=_required(roi_data, "source_roi_authority_identity"),
        canonical_roi_projection_identity=_required(roi_data, "canonical_roi_projection_identity"),
    )
    correction_data = _mapping(payload, "correction_projection")
    correction = GuidedNpmStartupCorrectionProjection(
        selected_canonical_roi_ids=tuple(_sequence(correction_data, "selected_canonical_roi_ids")),
        correction_parameter_values=tuple(
            _typed_value(item)
            for item in _sequence(correction_data, "correction_parameter_values")
        ),
        per_roi_correction_strategy_map=tuple(
            _strategy(item)
            for item in _sequence(correction_data, "per_roi_correction_strategy_map")
        ),
        source_correction_payload_identity=_required(correction_data, "source_correction_payload_identity"),
        source_correction_authority_identity=_required(correction_data, "source_correction_authority_identity"),
        canonical_correction_projection_identity=_required(correction_data, "canonical_correction_projection_identity"),
    )
    feature_data = _mapping(payload, "feature_projection")
    feature = GuidedNpmStartupFeatureProjection(
        execution_mode=_required(feature_data, "execution_mode"),
        profile_schema_version=_required(feature_data, "profile_schema_version"),
        profile_id=_required(feature_data, "profile_id"),
        effective_values=tuple(
            _typed_value(item) for item in _sequence(feature_data, "effective_values")
        ),
        active_fields=tuple(_sequence(feature_data, "active_fields")),
        inactive_fields=tuple(_sequence(feature_data, "inactive_fields")),
        profile_status=_required(feature_data, "profile_status"),
        explicitly_applied=_required(feature_data, "explicitly_applied"),
        current=_required(feature_data, "current"),
        visible_unapplied_changes=_required(feature_data, "visible_unapplied_changes"),
        per_roi_feature_event_map_version=_required(feature_data, "per_roi_feature_event_map_version"),
        per_roi_feature_event_map=tuple(
            _feature_event(item)
            for item in _sequence(feature_data, "per_roi_feature_event_map")
        ),
        selected_canonical_roi_ids=tuple(_sequence(feature_data, "selected_canonical_roi_ids")),
        inactive_for_execution=_required(feature_data, "inactive_for_execution"),
        source_feature_payload_identity=_required(feature_data, "source_feature_payload_identity"),
        source_feature_authority_identity=_required(feature_data, "source_feature_authority_identity"),
        canonical_feature_projection_identity=_required(feature_data, "canonical_feature_projection_identity"),
    )
    output_data = _mapping(payload, "output_projection")
    output = GuidedNpmStartupOutputProjection(
        output_base_canonical=_required(output_data, "output_base_canonical"),
        output_base_path_style=_required(output_data, "output_base_path_style"),
        path_role=_required(output_data, "path_role"),
        future_output_owner=_required(output_data, "future_output_owner"),
        run_directory_strategy=_required(output_data, "run_directory_strategy"),
        creation_timing=_required(output_data, "creation_timing"),
        overwrite=_required(output_data, "overwrite"),
        precreate=_required(output_data, "precreate"),
        safety_classifier_version=_required(output_data, "safety_classifier_version"),
        relationships=tuple(
            _relationship(item) for item in _sequence(output_data, "relationships")
        ),
        protected_root_context_complete=_required(output_data, "protected_root_context_complete"),
        filesystem_fact_scope=_required(output_data, "filesystem_fact_scope"),
        source_output_authority_identity=_required(output_data, "source_output_authority_identity"),
        canonical_output_projection_identity=_required(output_data, "canonical_output_projection_identity"),
    )
    execution_data = _mapping(payload, "execution_projection")
    execution = GuidedNpmStartupExecutionProjection(
        execution_projection_schema_name=_required(execution_data, "execution_projection_schema_name"),
        execution_projection_schema_version=_required(execution_data, "execution_projection_schema_version"),
        source_format=_required(execution_data, "source_format"),
        acquisition_mode=_required(execution_data, "acquisition_mode"),
        execution_mode=_required(execution_data, "execution_mode"),
        run_type=_required(execution_data, "run_type"),
        ordered_source_paths=tuple(_sequence(execution_data, "ordered_source_paths")),
        ordered_source_relative_paths=tuple(_sequence(execution_data, "ordered_source_relative_paths")),
        ordered_source_digests=tuple(_sequence(execution_data, "ordered_source_digests")),
        source_projection_identity=_required(
            execution_data, "source_projection_identity"
        ),
        parser_policy_identity=_required(execution_data, "parser_policy_identity"),
        recording_policy_identity=_required(execution_data, "recording_policy_identity"),
        session_sequence_identity=_required(execution_data, "session_sequence_identity"),
        roi_projection_identity=_required(execution_data, "roi_projection_identity"),
        correction_projection_identity=_required(execution_data, "correction_projection_identity"),
        feature_projection_identity=_required(execution_data, "feature_projection_identity"),
        output_projection_identity=_required(execution_data, "output_projection_identity"),
        deferred_execution_capabilities=tuple(_sequence(execution_data, "deferred_execution_capabilities")),
        canonical_execution_projection_identity=_required(execution_data, "canonical_execution_projection_identity"),
    )
    build_data = _mapping(payload, "application_build_identity")
    build_identity = ApplicationBuildIdentity(
        schema_name=_required(build_data, "schema_name"),
        schema_version=_required(build_data, "schema_version"),
        identity_provider_version=_required(build_data, "identity_provider_version"),
        distribution_name=_required(build_data, "distribution_name"),
        distribution_version=_required(build_data, "distribution_version"),
        source_revision_kind=_required(build_data, "source_revision_kind"),
        source_revision=_required(build_data, "source_revision"),
        source_tree_state=_required(build_data, "source_tree_state"),
        source_tree_digest=build_data.get("source_tree_digest"),
        build_artifact_digest=build_data.get("build_artifact_digest"),
        canonical_identity=_required(build_data, "canonical_identity"),
    )
    return GuidedNpmStartupPayload(
        startup_schema_name=_required(payload, "startup_schema_name"),
        startup_schema_version=_required(payload, "startup_schema_version"),
        startup_contract_version=_required(payload, "startup_contract_version"),
        source_authorization_identity=_required(payload, "source_authorization_identity"),
        source_authority_identity=_required(payload, "source_authority_identity"),
        source_production_intent_identity=_required(payload, "source_production_intent_identity"),
        source_request_identity=_required(payload, "source_request_identity"),
        validation_revision=_required(payload, "validation_revision"),
        guided_plan_identity=_required(payload, "guided_plan_identity"),
        application_build_identity=build_identity,
        source_format=_required(payload, "source_format"),
        acquisition_mode=_required(payload, "acquisition_mode"),
        execution_mode=_required(payload, "execution_mode"),
        run_type=_required(payload, "run_type"),
        source_projection=source,
        recording_policy=recording,
        roi_projection=roi,
        correction_projection=correction,
        feature_projection=feature,
        output_projection=output,
        execution_projection=execution,
        authorization_status=_required(payload, "authorization_status"),
        payload_status=_required(payload, "payload_status"),
        persistence_status=_required(payload, "persistence_status"),
        claim_status=_required(payload, "claim_status"),
        startup_status=_required(payload, "startup_status"),
        runnable=_required(payload, "runnable"),
        canonical_startup_payload_identity=_required(payload, "canonical_startup_payload_identity"),
    )


def deserialize_guided_npm_startup_payload(
    payload: Mapping[str, Any],
) -> GuidedNpmStartupPayload:
    """Reconstruct and reverify a serialized immutable B2-C4 payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("startup_payload_serialization_invalid")
    try:
        restored = _deserialize_startup_payload(payload)
        verify_guided_npm_startup_payload(restored)
        return restored
    except (TypeError, KeyError, ValueError, OverflowError) as exc:
        if str(exc) == "startup_payload_serialization_invalid":
            raise
        raise ValueError("startup_payload_serialization_invalid") from exc
