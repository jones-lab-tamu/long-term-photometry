"""Pure B2-C6A projection from a claimed NPM payload to a worker request.

This module performs no filesystem access and launches no work.  Component
identifiers document the existing numerical path that B2-C6B must adapt:
``io.adapters._load_npm`` loads, resolves timestamps, demultiplexes LED states,
extracts ROIs, and trims support; ``Pipeline`` preserves intermittent order,
applies per-ROI correction, dispatches phasic/tonic processing, and writes
results; ``core.feature_extraction.extract_features`` performs phasic feature
extraction.  The current Pipeline has no atomic ``both`` execution mode, so
combined requests refuse rather than being silently split.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
import ntpath
import posixpath
import types
from typing import Any, get_args, get_origin, get_type_hints

from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_npm_startup_claim import (
    GUIDED_NPM_STARTUP_CLAIM_SOURCE_DIRECT_ARTIFACT,
    GUIDED_NPM_STARTUP_CLAIM_SOURCE_PERSISTENCE_RECEIPT,
    GUIDED_NPM_STARTUP_WRAPPER_ARGUMENT,
    GuidedNpmStartupClaimReceipt,
    verify_guided_npm_startup_claim_receipt,
)
from photometry_pipeline.guided_npm_startup_payload import (
    GUIDED_NPM_STARTUP_PAYLOAD_CONTRACT_VERSION,
    GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_NAME,
    GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_VERSION,
    GuidedNpmStartupPayload,
    GuidedNpmStartupRoiMappingEntry,
    verify_guided_npm_startup_payload,
)
from photometry_pipeline.guided_npm_startup_persistence import (
    GUIDED_NPM_STARTUP_ARTIFACT_FILENAME,
    verify_application_build_identity,
)
from photometry_pipeline.guided_normalized_recording import (
    compute_npm_parser_contract_digest,
    compute_npm_support_policy_identity,
)
from photometry_pipeline.guided_production_mapping import (
    ApplicationBuildIdentity,
    GuidedProductionOutputRelationship,
    GuidedProductionPerRoiFeatureEvent,
    GuidedProductionPerRoiStrategy,
    GuidedProductionTypedValue,
)


GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_SCHEMA_NAME = "guided_npm_production_execution_request"
GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_SCHEMA_VERSION = "v1"
GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_CONTRACT_VERSION = "guided_npm_production_execution_request.v1"
GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_IDENTITY_DOMAIN = "guided_npm_production_execution_request.v1"
GUIDED_NPM_PRODUCTION_ADAPTER_SCHEMA_NAME = "guided_npm_production_adapter_runtime_projection"
GUIDED_NPM_PRODUCTION_ADAPTER_SCHEMA_VERSION = "v1"
GUIDED_NPM_PRODUCTION_ADAPTER_CONTRACT_VERSION = "guided_npm_production_adapter_runtime_projection.v1"

GUIDED_NPM_PRODUCTION_DEFERRED_ACTIONS = (
    "live_artifact_reverification",
    "live_source_freshness_reverification",
    "worker_request_materialization",
    "worker_launch",
    "execution_start_receipt",
    "consumed_evidence_capture",
    "terminal_reconciliation",
    "guided_completion",
)

GUIDED_NPM_PRODUCTION_COMPONENT_IDENTIFIERS = (
    "photometry_pipeline.io.adapters._load_npm",
    "photometry_pipeline.io.adapters._resolve_npm_time_col",
    "photometry_pipeline.io.adapters._load_npm:led_state_demultiplexing",
    "photometry_pipeline.io.adapters._load_npm:roi_extraction",
    "photometry_pipeline.io.npm_contract.resolve_npm_support_geometry",
    "photometry_pipeline.pipeline.Pipeline._iter_entry_chunks_for_pass",
    "photometry_pipeline.pipeline.Pipeline._resolve_correction_map_for_chunk",
    "photometry_pipeline.core.regression.validate_per_roi_correction_map",
    "photometry_pipeline.core.feature_extraction.extract_features",
    "photometry_pipeline.pipeline.Pipeline._process_chunk_tonic",
    "photometry_pipeline.pipeline.Pipeline.run_pass_2",
)

GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_REFUSAL_CATEGORIES = (
    "claim_receipt_missing_or_invalid", "claim_receipt_schema_unsupported",
    "claim_receipt_identity_mismatch", "claim_receipt_state_invalid",
    "startup_payload_missing_or_invalid", "startup_payload_schema_unsupported",
    "startup_payload_identity_mismatch", "startup_payload_state_invalid",
    "claim_payload_identity_mismatch", "claim_build_identity_mismatch",
    "claim_plan_identity_mismatch", "claim_validation_revision_mismatch",
    "claim_artifact_path_mismatch", "source_runtime_projection_invalid",
    "source_order_mismatch", "source_session_count_mismatch",
    "source_session_position_mismatch", "source_session_path_mismatch",
    "source_session_digest_mismatch", "parser_runtime_projection_invalid",
    "timing_runtime_projection_invalid", "roi_runtime_projection_invalid",
    "correction_runtime_projection_invalid", "feature_runtime_projection_invalid",
    "output_runtime_projection_invalid", "adapter_runtime_projection_invalid",
    "selected_roi_scope_mismatch", "correction_roi_coverage_mismatch",
    "feature_roi_coverage_mismatch", "execution_mode_mismatch",
    "production_component_mapping_unavailable", "execution_request_identity_mismatch",
    "execution_request_serialization_invalid", "execution_request_internal_error",
)
_CATEGORY_SET = frozenset(GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_REFUSAL_CATEGORIES)
_HEX = frozenset("0123456789abcdef")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("execution_request_nonfinite")
        return value
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("execution_request_mapping_key_invalid")
        return {key: _canonical(item) for key, item in value.items()}
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    raise ValueError(f"execution_request_value_unsupported:{type(value).__name__}")


def _digest(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode() + b"\x00" + encode_canonical_value(_canonical(value))).hexdigest()


def _without(value: Any, identity_field: str) -> dict[str, Any]:
    return {item.name: getattr(value, item.name) for item in fields(value) if item.name != identity_field}


def _identity(value: Any, identity_field: str, domain: str) -> str:
    return _digest(domain, _without(value, identity_field))


@dataclass(frozen=True)
class GuidedNpmProductionExecutionRequestIssue:
    category: str
    section: str
    message: str
    detail_code: str

    def __post_init__(self):
        if self.category not in _CATEGORY_SET or not all(_text(x) for x in (self.section, self.message, self.detail_code)):
            raise ValueError("execution_request_issue_invalid")


@dataclass(frozen=True)
class GuidedNpmProductionExecutionRequestFailure:
    blocking_issues: tuple[GuidedNpmProductionExecutionRequestIssue, ...]
    status: str = "refused"

    def __post_init__(self):
        if len(self.blocking_issues) != 1 or self.status != "refused":
            raise ValueError("execution_request_failure_invalid")


@dataclass(frozen=True)
class GuidedNpmProductionSessionRuntimeProjection:
    chronological_position: int
    source_path: str
    canonical_relative_path: str
    source_size_bytes: int
    source_sha256: str
    authoritative_source_start_time: str
    actual_elapsed_sec: float
    nominal_expected_elapsed_sec: float
    resolved_timestamp_column: str
    resolved_led_column: str | None
    timestamp_unit: str
    physical_roi_inventory: tuple[str, ...]
    physical_to_canonical_roi_mapping: tuple[GuidedNpmStartupRoiMappingEntry, ...]
    overlap_origin_absolute: float
    support_start_offset_sec: float
    support_end_offset_sec: float
    support_start_absolute: float
    support_end_absolute: float
    observed_support_duration_sec: float
    support_policy: str
    support_policy_identity: str
    output_time_basis: str
    warning_categories: tuple[str, ...]
    source_startup_session_identity_reference: str
    canonical_session_runtime_identity: str

    def __post_init__(self):
        if isinstance(self.chronological_position, bool) or not isinstance(self.chronological_position, int) or self.chronological_position < 0:
            raise ValueError("session_position_invalid")
        if isinstance(self.source_size_bytes, bool) or not isinstance(self.source_size_bytes, int) or self.source_size_bytes <= 0:
            raise ValueError("session_size_invalid")
        if not _sha(self.source_sha256) or not _sha(self.source_startup_session_identity_reference) or not _sha(self.canonical_session_runtime_identity):
            raise ValueError("session_identity_or_digest_invalid")


@dataclass(frozen=True)
class GuidedNpmProductionSourceRuntimeProjection:
    source_root_canonical: str
    source_root_inspected: str
    source_path_style: str
    ordered_source_paths: tuple[str, ...]
    ordered_source_relative_paths: tuple[str, ...]
    ordered_source_digests: tuple[str, ...]
    ordered_sessions: tuple[GuidedNpmProductionSessionRuntimeProjection, ...]
    source_startup_projection_identity: str
    source_startup_snapshot_identity: str
    source_startup_content_identity: str
    source_startup_membership_identity: str
    source_startup_session_sequence_identity: str
    runtime_source_membership_identity: str
    runtime_source_content_identity: str
    runtime_session_sequence_identity: str
    canonical_source_runtime_projection_identity: str

    def __post_init__(self):
        if self.source_path_style not in {"windows_drive", "posix_absolute"} or not self.ordered_sessions:
            raise ValueError("source_projection_primitive_invalid")
        for name in (
            "source_startup_projection_identity", "source_startup_snapshot_identity",
            "source_startup_content_identity", "source_startup_membership_identity",
            "source_startup_session_sequence_identity", "runtime_source_membership_identity",
            "runtime_source_content_identity", "runtime_session_sequence_identity",
            "canonical_source_runtime_projection_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError("source_projection_identity_invalid")


@dataclass(frozen=True)
class GuidedNpmProductionParserRuntimeProjection:
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
    roi_value_nan_policy: str
    support_policy: str
    support_policy_identity: str
    source_recording_policy_identity: str
    source_recording_projection_identity: str
    canonical_parser_runtime_projection_identity: str


@dataclass(frozen=True)
class GuidedNpmProductionTimingRuntimeProjection:
    target_fs_hz: float
    configured_session_duration_sec: float
    sessions_per_hour: int
    chronology_policy: str
    gap_policy: str
    overlap_policy: str
    output_time_basis: str
    ordered_authoritative_source_start_times: tuple[str, ...]
    ordered_actual_elapsed_sec: tuple[float, ...]
    ordered_nominal_expected_elapsed_sec: tuple[float, ...]
    ordered_support_start_offsets_sec: tuple[float, ...]
    ordered_support_end_offsets_sec: tuple[float, ...]
    ordered_support_start_absolute: tuple[float, ...]
    ordered_support_end_absolute: tuple[float, ...]
    ordered_overlap_origins_absolute: tuple[float, ...]
    source_session_sequence_identity: str
    canonical_timing_runtime_projection_identity: str


@dataclass(frozen=True)
class GuidedNpmProductionRoiRuntimeProjection:
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
    source_roi_projection_identity: str
    canonical_roi_runtime_projection_identity: str


@dataclass(frozen=True)
class GuidedNpmProductionCorrectionRuntimeProjection:
    selected_canonical_roi_ids: tuple[str, ...]
    correction_parameter_values: tuple[GuidedProductionTypedValue, ...]
    per_roi_correction_strategy_map: tuple[GuidedProductionPerRoiStrategy, ...]
    source_correction_payload_identity: str
    source_correction_authority_identity: str
    source_correction_projection_identity: str
    canonical_correction_runtime_projection_identity: str


@dataclass(frozen=True)
class GuidedNpmProductionFeatureRuntimeProjection:
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
    source_feature_projection_identity: str
    canonical_feature_runtime_projection_identity: str


@dataclass(frozen=True)
class GuidedNpmProductionOutputRuntimeProjection:
    output_base_canonical: str
    run_directory_path: str
    startup_artifact_path: str
    output_base_path_style: str
    path_role: str
    future_output_owner: str
    run_directory_strategy: str
    overwrite: bool
    safety_classifier_version: str
    relationships: tuple[GuidedProductionOutputRelationship, ...]
    protected_root_context_complete: bool
    filesystem_fact_scope: str
    source_output_authority_identity: str
    source_output_projection_identity: str
    canonical_output_runtime_projection_identity: str


@dataclass(frozen=True)
class GuidedNpmProductionAdapterRuntimeProjection:
    adapter_schema_name: str
    adapter_schema_version: str
    adapter_contract_version: str
    source_format: str
    acquisition_mode: str
    execution_mode: str
    run_type: str
    ordered_source_paths: tuple[str, ...]
    ordered_source_relative_paths: tuple[str, ...]
    source_runtime_projection_identity: str
    parser_runtime_projection_identity: str
    timing_runtime_projection_identity: str
    roi_runtime_projection_identity: str
    correction_runtime_projection_identity: str
    feature_runtime_projection_identity: str
    output_runtime_projection_identity: str
    production_loader_kind: str
    production_correction_kind: str
    production_feature_kind: str
    production_output_kind: str
    production_component_identifiers: tuple[str, ...]
    deferred_runtime_actions: tuple[str, ...]
    canonical_adapter_runtime_projection_identity: str

    def __post_init__(self):
        if (self.adapter_schema_name, self.adapter_schema_version, self.adapter_contract_version) != (
            GUIDED_NPM_PRODUCTION_ADAPTER_SCHEMA_NAME,
            GUIDED_NPM_PRODUCTION_ADAPTER_SCHEMA_VERSION,
            GUIDED_NPM_PRODUCTION_ADAPTER_CONTRACT_VERSION,
        ) or self.source_format != "npm" or self.acquisition_mode != "intermittent" or self.execution_mode not in {"phasic", "tonic"}:
            raise ValueError("adapter_projection_primitive_invalid")
        for name in (
            "source_runtime_projection_identity", "parser_runtime_projection_identity",
            "timing_runtime_projection_identity", "roi_runtime_projection_identity",
            "correction_runtime_projection_identity", "feature_runtime_projection_identity",
            "output_runtime_projection_identity", "canonical_adapter_runtime_projection_identity",
        ):
            if not _sha(getattr(self, name)):
                raise ValueError("adapter_projection_identity_invalid")


@dataclass(frozen=True)
class GuidedNpmProductionExecutionRequest:
    request_schema_name: str
    request_schema_version: str
    request_contract_version: str
    source_claim_receipt_identity: str
    source_startup_payload_identity: str
    source_authorization_identity: str
    source_authority_identity: str
    source_production_intent_identity: str
    source_request_identity: str
    validation_revision: int
    guided_plan_identity: str
    application_build_identity: ApplicationBuildIdentity
    startup_artifact_path: str
    startup_artifact_sha256: str
    startup_artifact_size_bytes: int
    source_format: str
    acquisition_mode: str
    execution_mode: str
    run_type: str
    source_runtime_projection: GuidedNpmProductionSourceRuntimeProjection
    parser_runtime_projection: GuidedNpmProductionParserRuntimeProjection
    roi_runtime_projection: GuidedNpmProductionRoiRuntimeProjection
    correction_runtime_projection: GuidedNpmProductionCorrectionRuntimeProjection
    feature_runtime_projection: GuidedNpmProductionFeatureRuntimeProjection
    timing_runtime_projection: GuidedNpmProductionTimingRuntimeProjection
    output_runtime_projection: GuidedNpmProductionOutputRuntimeProjection
    adapter_runtime_projection: GuidedNpmProductionAdapterRuntimeProjection
    claim_status: str
    request_status: str
    launch_status: str
    execution_status: str
    completion_status: str
    runnable: bool
    canonical_execution_request_identity: str

    def __post_init__(self):
        if (self.request_schema_name, self.request_schema_version, self.request_contract_version) != (
            GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_SCHEMA_NAME,
            GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_SCHEMA_VERSION,
            GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_CONTRACT_VERSION,
        ):
            raise ValueError("execution_request_schema_unsupported")
        if self.source_format != "npm" or self.acquisition_mode != "intermittent" or self.execution_mode not in {"phasic", "tonic"} or self.run_type != "full":
            raise ValueError("execution_request_mode_invalid")
        if (self.claim_status, self.request_status, self.launch_status, self.execution_status, self.completion_status, self.runnable) != (
            "claimed_for_npm_startup", "constructed_for_production_adapter", "not_launched", "not_started", "not_available", False,
        ):
            raise ValueError("execution_request_state_invalid")


GuidedNpmProductionExecutionRequestResult = GuidedNpmProductionExecutionRequest | GuidedNpmProductionExecutionRequestFailure


class _Refusal(ValueError):
    def __init__(self, category: str, section: str, message: str, detail_code: str):
        self.category, self.section, self.message, self.detail_code = category, section, message, detail_code
        super().__init__(message)


def _refuse(category: str, section: str, message: str, detail_code: str) -> None:
    raise _Refusal(category if category in _CATEGORY_SET else "execution_request_internal_error", section, message, detail_code)


def _failure(exc: _Refusal) -> GuidedNpmProductionExecutionRequestFailure:
    return GuidedNpmProductionExecutionRequestFailure((GuidedNpmProductionExecutionRequestIssue(exc.category, exc.section, exc.message, exc.detail_code),))


def compute_guided_npm_production_session_runtime_identity(value):
    # The B2-C4 startup-session identity is retained only for provenance and is
    # deliberately excluded from B2-C6A's independently recomputable authority.
    facts = {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if item.name not in {
            "source_startup_session_identity_reference",
            "canonical_session_runtime_identity",
        }
    }
    return _digest("guided_npm_production_session_runtime.v1", facts)


def compute_guided_npm_production_source_membership_identity(sessions):
    return _digest(
        "guided_npm_production_source_membership.v1",
        tuple(
            {
                "chronological_position": item.chronological_position,
                "canonical_relative_path": item.canonical_relative_path,
                "source_path": item.source_path,
            }
            for item in sessions
        ),
    )


def compute_guided_npm_production_source_content_identity(sessions):
    return _digest(
        "guided_npm_production_source_content.v1",
        tuple(
            {
                "canonical_relative_path": item.canonical_relative_path,
                "source_size_bytes": item.source_size_bytes,
                "source_sha256": item.source_sha256,
            }
            for item in sessions
        ),
    )


def compute_guided_npm_production_session_sequence_identity(sessions):
    return _digest(
        "guided_npm_production_session_sequence.v1",
        tuple(
            {
                "chronological_position": item.chronological_position,
                "canonical_session_runtime_identity": item.canonical_session_runtime_identity,
            }
            for item in sessions
        ),
    )


def compute_guided_npm_production_source_runtime_projection_identity(value):
    return _identity(value, "canonical_source_runtime_projection_identity", "guided_npm_production_source_runtime_projection.v1")


def compute_guided_npm_production_parser_runtime_projection_identity(value):
    return _identity(value, "canonical_parser_runtime_projection_identity", "guided_npm_production_parser_runtime_projection.v1")


def compute_guided_npm_production_timing_runtime_projection_identity(value):
    return _identity(value, "canonical_timing_runtime_projection_identity", "guided_npm_production_timing_runtime_projection.v1")


def compute_guided_npm_production_roi_runtime_projection_identity(value):
    return _identity(value, "canonical_roi_runtime_projection_identity", "guided_npm_production_roi_runtime_projection.v1")


def compute_guided_npm_production_correction_runtime_projection_identity(value):
    return _identity(value, "canonical_correction_runtime_projection_identity", "guided_npm_production_correction_runtime_projection.v1")


def compute_guided_npm_production_feature_runtime_projection_identity(value):
    return _identity(value, "canonical_feature_runtime_projection_identity", "guided_npm_production_feature_runtime_projection.v1")


def compute_guided_npm_production_output_runtime_projection_identity(value):
    return _identity(value, "canonical_output_runtime_projection_identity", "guided_npm_production_output_runtime_projection.v1")


def compute_guided_npm_production_adapter_runtime_projection_identity(value):
    return _identity(value, "canonical_adapter_runtime_projection_identity", "guided_npm_production_adapter_runtime_projection.v1")


def compute_guided_npm_production_execution_request_identity(value):
    return _identity(value, "canonical_execution_request_identity", GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_IDENTITY_DOMAIN)


def _identified(value, field_name: str, compute):
    return replace(value, **{field_name: compute(value)})


def _path_model(path: str, style: str) -> tuple[str, str, str]:
    module = ntpath if style == "windows_drive" else posixpath
    normalized = module.normpath(path)
    if not module.isabs(path) or normalized != path or ".." in path.replace("\\", "/").split("/"):
        raise ValueError("path_not_canonical")
    return normalized, module.dirname(normalized), module.basename(normalized)


def _source_path_style(path: str) -> str:
    if ntpath.isabs(path) and ntpath.splitdrive(path)[0]:
        return "windows_drive"
    if posixpath.isabs(path):
        return "posix_absolute"
    raise ValueError("source_root_not_absolute")


def _verify_source_path(root: str, path: str, relative: str, style: str) -> None:
    module = ntpath if style == "windows_drive" else posixpath
    root_value, _, _ = _path_model(root, style)
    path_value, _, _ = _path_model(path, style)
    if module.isabs(relative) or module.normpath(relative) != relative or relative in {"", "."} or ".." in relative.replace("\\", "/").split("/"):
        raise ValueError("source_relative_path_invalid")
    if _source_path_style(root_value) != style or _source_path_style(path_value) != style:
        raise ValueError("source_path_style_mismatch")
    try:
        derived = module.relpath(path_value, root_value)
    except ValueError as exc:
        raise ValueError("source_path_root_mismatch") from exc
    if derived != relative or derived == ".." or derived.startswith(".." + module.sep):
        raise ValueError("source_path_root_mismatch")


def _verify_typed_values(values: Any) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(item, GuidedProductionTypedValue)
        or not _text(item.field_name)
        or item.value_type not in {"str", "bool", "int", "float", "NoneType"}
        or not _text(item.source_classification)
        or (isinstance(item.value, float) and not math.isfinite(item.value))
        for item in values
    ):
        raise ValueError("typed_runtime_values_invalid")


def _verify_output_paths(output_base: str, artifact: str, style: str) -> str:
    output, _, _ = _path_model(output_base, style)
    artifact_path, run_dir, filename = _path_model(artifact, style)
    module = ntpath if style == "windows_drive" else posixpath
    if filename != GUIDED_NPM_STARTUP_ARTIFACT_FILENAME or run_dir == output or module.dirname(run_dir) != output:
        raise ValueError("output_run_directory_relationship_invalid")
    if module.dirname(artifact_path) != run_dir:
        raise ValueError("artifact_parent_invalid")
    return run_dir


def _verify_builder_inputs(claim: Any, payload: Any) -> None:
    if type(claim) is not GuidedNpmStartupClaimReceipt:
        _refuse("claim_receipt_missing_or_invalid", "claim", "A valid NPM claim receipt is required.", "claim_type_invalid")
    try:
        verify_guided_npm_startup_claim_receipt(claim)
    except ValueError as exc:
        category = "claim_receipt_identity_mismatch" if "identity" in str(exc) else "claim_receipt_missing_or_invalid"
        _refuse(category, "claim", "The NPM claim receipt is invalid.", str(exc))
    if claim.claim_source_kind not in {GUIDED_NPM_STARTUP_CLAIM_SOURCE_PERSISTENCE_RECEIPT, GUIDED_NPM_STARTUP_CLAIM_SOURCE_DIRECT_ARTIFACT} or claim.claim_status != "claimed_for_npm_startup" or claim.startup_status != "claimed_not_executed" or claim.runnable is not False:
        _refuse("claim_receipt_state_invalid", "claim", "The NPM claim state is invalid.", "claim_state_invalid")
    if type(payload) is not GuidedNpmStartupPayload:
        _refuse("startup_payload_missing_or_invalid", "payload", "A valid NPM startup payload is required.", "payload_type_invalid")
    try:
        verify_guided_npm_startup_payload(payload)
    except ValueError as exc:
        _refuse("startup_payload_identity_mismatch", "payload", "The NPM startup payload is invalid.", str(exc))
    if payload.startup_schema_name != GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_NAME or payload.startup_schema_version != GUIDED_NPM_STARTUP_PAYLOAD_SCHEMA_VERSION or payload.startup_contract_version != GUIDED_NPM_STARTUP_PAYLOAD_CONTRACT_VERSION:
        _refuse("startup_payload_schema_unsupported", "payload", "The NPM startup payload schema is unsupported.", "payload_schema_unsupported")
    for left, right, category in (
        (claim.source_startup_payload_identity, payload.canonical_startup_payload_identity, "claim_payload_identity_mismatch"),
        (claim.claimed_payload_identity, payload.canonical_startup_payload_identity, "claim_payload_identity_mismatch"),
        (claim.application_build_identity, payload.application_build_identity, "claim_build_identity_mismatch"),
        (claim.guided_plan_identity, payload.guided_plan_identity, "claim_plan_identity_mismatch"),
        (claim.validation_revision, payload.validation_revision, "claim_validation_revision_mismatch"),
        (claim.startup_artifact_path, claim.wrapper_argument_path, "claim_artifact_path_mismatch"),
    ):
        if left != right:
            _refuse(category, "binding", "The claim and startup payload do not correspond.", category)
    if payload.source_format != "npm" or payload.acquisition_mode != "intermittent" or payload.execution_mode not in {"phasic", "tonic", "both"}:
        _refuse("execution_mode_mismatch", "payload", "The startup execution subset is unsupported.", "execution_subset_invalid")
    if payload.execution_mode == "both":
        _refuse("production_component_mapping_unavailable", "adapter", "The existing production Pipeline has no atomic combined mode.", "pipeline_combined_mode_unavailable")


def build_guided_npm_production_execution_request(claim_receipt, startup_payload):
    try:
        _verify_builder_inputs(claim_receipt, startup_payload)
        p, c = startup_payload, claim_receipt
        sessions = []
        for source in p.source_projection.ordered_sessions:
            session = GuidedNpmProductionSessionRuntimeProjection(
                source.chronological_position, source.authorized_absolute_source_reference,
                source.canonical_relative_path, source.size_bytes, source.sha256_content_digest,
                source.authoritative_source_start_time, source.actual_elapsed_sec,
                source.nominal_expected_elapsed_sec, source.resolved_timestamp_column,
                source.resolved_led_column, source.timestamp_unit, source.physical_roi_inventory,
                source.physical_to_canonical_roi_mapping, source.overlap_origin_absolute,
                source.resolved_support_start_offset_sec, source.resolved_support_end_offset_sec,
                source.resolved_support_start_absolute, source.resolved_support_end_absolute,
                source.observed_support_duration_sec, source.support_policy,
                source.support_policy_identity, source.output_time_basis, source.warning_categories,
                source.canonical_startup_session_identity, "0" * 64,
            )
            sessions.append(_identified(session, "canonical_session_runtime_identity", compute_guided_npm_production_session_runtime_identity))
        sessions = tuple(sessions)
        source_style = _source_path_style(p.source_projection.source_root_canonical)
        runtime_membership = compute_guided_npm_production_source_membership_identity(sessions)
        runtime_content = compute_guided_npm_production_source_content_identity(sessions)
        runtime_sequence = compute_guided_npm_production_session_sequence_identity(sessions)
        source = GuidedNpmProductionSourceRuntimeProjection(
            p.source_projection.source_root_canonical, p.source_projection.source_root_inspected,
            source_style,
            tuple(x.source_path for x in sessions), tuple(x.canonical_relative_path for x in sessions),
            tuple(x.source_sha256 for x in sessions), sessions,
            p.source_projection.canonical_source_projection_identity,
            p.source_projection.verified_source_snapshot_identity,
            p.source_projection.verified_source_content_identity,
            p.source_projection.verified_source_set_identity,
            p.source_projection.ordered_session_sequence_identity,
            runtime_membership, runtime_content, runtime_sequence, "0" * 64,
        )
        source = _identified(source, "canonical_source_runtime_projection_identity", compute_guided_npm_production_source_runtime_projection_identity)
        r = p.recording_policy
        parser = GuidedNpmProductionParserRuntimeProjection(
            r.parser_policy_identity, r.parser_policy_content_json, r.time_axis_mode,
            r.ordered_timestamp_candidates, r.timestamp_unit, r.timestamp_finite_policy,
            r.led_state_column, r.reference_led_value, r.signal_led_value, r.roi_prefix,
            r.roi_suffix, r.roi_ordering_rule, r.canonical_roi_naming_rule,
            r.roi_value_nan_policy, r.support_policy, r.support_policy_identity,
            r.source_recording_policy_identity, r.canonical_recording_policy_identity, "0" * 64,
        )
        parser = _identified(parser, "canonical_parser_runtime_projection_identity", compute_guided_npm_production_parser_runtime_projection_identity)
        timing = GuidedNpmProductionTimingRuntimeProjection(
            r.target_fs_hz, r.configured_session_duration_sec, r.sessions_per_hour,
            r.chronology_policy, r.gap_policy, r.overlap_policy, r.output_time_basis,
            tuple(x.authoritative_source_start_time for x in sessions),
            tuple(x.actual_elapsed_sec for x in sessions), tuple(x.nominal_expected_elapsed_sec for x in sessions),
            tuple(x.support_start_offset_sec for x in sessions), tuple(x.support_end_offset_sec for x in sessions),
            tuple(x.support_start_absolute for x in sessions), tuple(x.support_end_absolute for x in sessions),
            tuple(x.overlap_origin_absolute for x in sessions), source.runtime_session_sequence_identity, "0" * 64,
        )
        timing = _identified(timing, "canonical_timing_runtime_projection_identity", compute_guided_npm_production_timing_runtime_projection_identity)
        q = p.roi_projection
        roi = GuidedNpmProductionRoiRuntimeProjection(
            q.complete_canonical_roi_ids, q.selected_canonical_roi_ids, q.excluded_canonical_roi_ids,
            q.complete_physical_source_columns, q.physical_to_canonical_roi_mapping,
            q.selected_physical_source_columns, q.selected_physical_to_canonical_roi_mapping,
            q.roi_ordering_rule, q.canonical_roi_naming_rule, q.source_roi_authority_identity,
            q.canonical_roi_projection_identity, "0" * 64,
        )
        roi = _identified(roi, "canonical_roi_runtime_projection_identity", compute_guided_npm_production_roi_runtime_projection_identity)
        q = p.correction_projection
        correction = GuidedNpmProductionCorrectionRuntimeProjection(
            q.selected_canonical_roi_ids, q.correction_parameter_values,
            q.per_roi_correction_strategy_map, q.source_correction_payload_identity,
            q.source_correction_authority_identity, q.canonical_correction_projection_identity, "0" * 64,
        )
        correction = _identified(correction, "canonical_correction_runtime_projection_identity", compute_guided_npm_production_correction_runtime_projection_identity)
        q = p.feature_projection
        feature = GuidedNpmProductionFeatureRuntimeProjection(
            q.execution_mode, q.profile_schema_version, q.profile_id, q.effective_values,
            q.active_fields, q.inactive_fields, q.profile_status, q.explicitly_applied,
            q.current, q.visible_unapplied_changes, q.per_roi_feature_event_map_version,
            q.per_roi_feature_event_map, q.selected_canonical_roi_ids, q.inactive_for_execution,
            q.source_feature_payload_identity, q.source_feature_authority_identity,
            q.canonical_feature_projection_identity, "0" * 64,
        )
        feature = _identified(feature, "canonical_feature_runtime_projection_identity", compute_guided_npm_production_feature_runtime_projection_identity)
        q = p.output_projection
        try:
            run_directory = _verify_output_paths(q.output_base_canonical, c.startup_artifact_path, q.output_base_path_style)
        except ValueError as exc:
            _refuse("output_runtime_projection_invalid", "output", "The claimed run directory is outside its authorized output base.", str(exc))
        output = GuidedNpmProductionOutputRuntimeProjection(
            q.output_base_canonical, run_directory, c.startup_artifact_path,
            q.output_base_path_style, q.path_role, q.future_output_owner,
            q.run_directory_strategy, q.overwrite, q.safety_classifier_version,
            q.relationships, q.protected_root_context_complete, q.filesystem_fact_scope,
            q.source_output_authority_identity, q.canonical_output_projection_identity, "0" * 64,
        )
        output = _identified(output, "canonical_output_runtime_projection_identity", compute_guided_npm_production_output_runtime_projection_identity)
        adapter = GuidedNpmProductionAdapterRuntimeProjection(
            GUIDED_NPM_PRODUCTION_ADAPTER_SCHEMA_NAME, GUIDED_NPM_PRODUCTION_ADAPTER_SCHEMA_VERSION,
            GUIDED_NPM_PRODUCTION_ADAPTER_CONTRACT_VERSION, p.source_format, p.acquisition_mode,
            p.execution_mode, p.run_type, source.ordered_source_paths, source.ordered_source_relative_paths,
            source.canonical_source_runtime_projection_identity,
            parser.canonical_parser_runtime_projection_identity, timing.canonical_timing_runtime_projection_identity,
            roi.canonical_roi_runtime_projection_identity, correction.canonical_correction_runtime_projection_identity,
            feature.canonical_feature_runtime_projection_identity, output.canonical_output_runtime_projection_identity,
            "photometry_pipeline.io.adapters._load_npm",
            "photometry_pipeline.pipeline.Pipeline._resolve_correction_map_for_chunk",
            "photometry_pipeline.core.feature_extraction.extract_features",
            "photometry_pipeline.pipeline.Pipeline.run_pass_2",
            GUIDED_NPM_PRODUCTION_COMPONENT_IDENTIFIERS, GUIDED_NPM_PRODUCTION_DEFERRED_ACTIONS, "0" * 64,
        )
        adapter = _identified(adapter, "canonical_adapter_runtime_projection_identity", compute_guided_npm_production_adapter_runtime_projection_identity)
        request = GuidedNpmProductionExecutionRequest(
            GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_SCHEMA_NAME,
            GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_SCHEMA_VERSION,
            GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_CONTRACT_VERSION,
            c.canonical_claim_receipt_identity, p.canonical_startup_payload_identity,
            p.source_authorization_identity, p.source_authority_identity,
            p.source_production_intent_identity, p.source_request_identity,
            p.validation_revision, p.guided_plan_identity, p.application_build_identity,
            c.startup_artifact_path, c.startup_artifact_sha256, c.startup_artifact_size_bytes,
            p.source_format, p.acquisition_mode, p.execution_mode, p.run_type,
            source, parser, roi, correction, feature, timing, output, adapter,
            "claimed_for_npm_startup", "constructed_for_production_adapter",
            "not_launched", "not_started", "not_available", False, "0" * 64,
        )
        request = _identified(request, "canonical_execution_request_identity", compute_guided_npm_production_execution_request_identity)
        verify_guided_npm_production_execution_request(request)
        return request
    except _Refusal as exc:
        return _failure(exc)
    except Exception as exc:
        return _failure(_Refusal("execution_request_internal_error", "request", "NPM production execution-request construction failed.", str(exc) or type(exc).__name__))


def _require_identity(value, field_name, compute, category):
    if compute(value) != getattr(value, field_name):
        raise ValueError(category)


def verify_guided_npm_production_execution_request(request: GuidedNpmProductionExecutionRequest) -> None:
    if type(request) is not GuidedNpmProductionExecutionRequest:
        raise ValueError("execution_request_type_invalid")
    if (request.request_schema_name, request.request_schema_version, request.request_contract_version) != (
        GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_SCHEMA_NAME, GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_SCHEMA_VERSION,
        GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_CONTRACT_VERSION):
        raise ValueError("execution_request_schema_unsupported")
    verify_application_build_identity(request.application_build_identity)
    if request.claim_status != "claimed_for_npm_startup" or request.request_status != "constructed_for_production_adapter" or request.launch_status != "not_launched" or request.execution_status != "not_started" or request.completion_status != "not_available" or request.runnable is not False:
        raise ValueError("execution_request_state_invalid")
    if request.source_format != "npm" or request.acquisition_mode != "intermittent" or request.execution_mode not in {"phasic", "tonic"} or request.run_type != "full":
        raise ValueError("execution_request_mode_invalid")
    if not all(_sha(value) for value in (
        request.source_claim_receipt_identity, request.source_startup_payload_identity,
        request.source_authorization_identity, request.source_authority_identity,
        request.source_production_intent_identity, request.source_request_identity,
        request.startup_artifact_sha256, request.guided_plan_identity,
    )) or isinstance(request.startup_artifact_size_bytes, bool) or request.startup_artifact_size_bytes <= 0 or isinstance(request.validation_revision, bool) or request.validation_revision < 0:
        raise ValueError("execution_request_provenance_invalid")
    s = request.source_runtime_projection
    if _source_path_style(s.source_root_canonical) != s.source_path_style or _source_path_style(s.source_root_inspected) != s.source_path_style:
        raise ValueError("source_root_path_model_invalid")
    _path_model(s.source_root_canonical, s.source_path_style)
    _path_model(s.source_root_inspected, s.source_path_style)
    if tuple(x.chronological_position for x in s.ordered_sessions) != tuple(range(len(s.ordered_sessions))):
        raise ValueError("source_session_position_mismatch")
    if s.ordered_source_paths != tuple(x.source_path for x in s.ordered_sessions) or s.ordered_source_relative_paths != tuple(x.canonical_relative_path for x in s.ordered_sessions) or s.ordered_source_digests != tuple(x.source_sha256 for x in s.ordered_sessions):
        raise ValueError("source_order_mismatch")
    for item in s.ordered_sessions:
        if (
            isinstance(item.source_size_bytes, bool)
            or not isinstance(item.source_size_bytes, int)
            or item.source_size_bytes <= 0
            or not _sha(item.source_sha256)
            or not _sha(item.source_startup_session_identity_reference)
            or not all(_finite(getattr(item, name)) for name in (
                "actual_elapsed_sec", "nominal_expected_elapsed_sec", "overlap_origin_absolute",
                "support_start_offset_sec", "support_end_offset_sec", "support_start_absolute",
                "support_end_absolute", "observed_support_duration_sec",
            ))
            or item.actual_elapsed_sec < 0
            or item.nominal_expected_elapsed_sec < 0
            or item.observed_support_duration_sec < 0
            or item.support_end_offset_sec < item.support_start_offset_sec
            or item.support_end_absolute < item.support_start_absolute
            or not isinstance(item.warning_categories, tuple)
            or any(not _text(value) for value in item.warning_categories)
        ):
            raise ValueError("source_session_primitive_invalid")
        _verify_source_path(s.source_root_canonical, item.source_path, item.canonical_relative_path, s.source_path_style)
        _require_identity(item, "canonical_session_runtime_identity", compute_guided_npm_production_session_runtime_identity, "session_runtime_identity_mismatch")
    if compute_guided_npm_production_source_membership_identity(s.ordered_sessions) != s.runtime_source_membership_identity:
        raise ValueError("runtime_source_membership_identity_mismatch")
    if compute_guided_npm_production_source_content_identity(s.ordered_sessions) != s.runtime_source_content_identity:
        raise ValueError("runtime_source_content_identity_mismatch")
    if compute_guided_npm_production_session_sequence_identity(s.ordered_sessions) != s.runtime_session_sequence_identity:
        raise ValueError("runtime_session_sequence_identity_mismatch")
    _require_identity(s, "canonical_source_runtime_projection_identity", compute_guided_npm_production_source_runtime_projection_identity, "source_runtime_projection_identity_mismatch")
    p, t, r, c, f, o, a = request.parser_runtime_projection, request.timing_runtime_projection, request.roi_runtime_projection, request.correction_runtime_projection, request.feature_runtime_projection, request.output_runtime_projection, request.adapter_runtime_projection
    try:
        parser_content = json.loads(p.parser_policy_content_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("parser_runtime_projection_invalid") from exc
    sampling = parser_content.get("sampling") if isinstance(parser_content, Mapping) else None
    led_values = sampling.get("led_values") if isinstance(sampling, Mapping) else None
    parser_runtime_values = (
        p.time_axis_mode, p.ordered_timestamp_candidates, p.timestamp_unit,
        p.timestamp_finite_policy, p.led_state_column, p.reference_led_value,
        p.signal_led_value, p.roi_prefix, p.roi_suffix, p.roi_ordering_rule,
        p.roi_value_nan_policy, p.support_policy,
    )
    canonical_parser_values = (
        sampling.get("time_axis") if isinstance(sampling, Mapping) else None,
        tuple(sampling.get("timestamp_column_candidates", ())) if isinstance(sampling, Mapping) else (),
        sampling.get("timestamp_unit") if isinstance(sampling, Mapping) else None,
        sampling.get("timestamp_finite_policy") if isinstance(sampling, Mapping) else None,
        sampling.get("led_column") if isinstance(sampling, Mapping) else None,
        led_values.get("uv") if isinstance(led_values, Mapping) else None,
        led_values.get("signal") if isinstance(led_values, Mapping) else None,
        sampling.get("region_prefix") if isinstance(sampling, Mapping) else None,
        sampling.get("region_suffix") if isinstance(sampling, Mapping) else None,
        sampling.get("roi_order_policy") if isinstance(sampling, Mapping) else None,
        sampling.get("roi_nan_policy") if isinstance(sampling, Mapping) else None,
        sampling.get("support_policy") if isinstance(sampling, Mapping) else None,
    )
    if (
        parser_content.get("schema_name") != "npm_normalized_parser_contract"
        or parser_content.get("schema_version") != "v1"
        or json.dumps(parser_content, sort_keys=True, separators=(",", ":"), ensure_ascii=False) != p.parser_policy_content_json
        or compute_npm_parser_contract_digest(parser_content) != p.parser_policy_identity
        or compute_npm_support_policy_identity(p.support_policy) != p.support_policy_identity
        or parser_runtime_values != canonical_parser_values
        or p.canonical_roi_naming_rule != "normalized_physical_to_canonical_mapping_order"
        or any(
            item.resolved_timestamp_column not in p.ordered_timestamp_candidates
            or item.resolved_led_column != p.led_state_column
            or item.timestamp_unit != p.timestamp_unit
            or item.support_policy != p.support_policy
            or item.support_policy_identity != p.support_policy_identity
            or item.output_time_basis != t.output_time_basis
            for item in s.ordered_sessions
        )
    ):
        raise ValueError("parser_runtime_projection_invalid")
    complete, selected, excluded = r.complete_canonical_roi_ids, r.selected_canonical_roi_ids, r.excluded_canonical_roi_ids
    if (
        any(type(value) is not tuple for value in (
            complete, selected, excluded, r.complete_physical_source_columns,
            r.physical_to_canonical_roi_mapping, r.selected_physical_source_columns,
            r.selected_physical_to_canonical_roi_mapping,
        ))
        or any(len(values) != len(set(values)) for values in (complete, selected, excluded, r.complete_physical_source_columns))
        or tuple(x for x in complete if x in set(selected)) != selected
        or tuple(x for x in complete if x in set(excluded)) != excluded
        or set(selected) | set(excluded) != set(complete)
        or set(selected) & set(excluded)
        or len(r.complete_physical_source_columns) != len(complete)
        or len(r.physical_to_canonical_roi_mapping) != len(complete)
        or len(r.selected_physical_source_columns) != len(selected)
        or len(r.selected_physical_to_canonical_roi_mapping) != len(selected)
        or tuple(x.physical_source_column for x in r.physical_to_canonical_roi_mapping) != r.complete_physical_source_columns
        or tuple(x.canonical_roi_id for x in r.physical_to_canonical_roi_mapping) != complete
        or len({x.physical_source_column for x in r.physical_to_canonical_roi_mapping}) != len(complete)
        or len({x.canonical_roi_id for x in r.physical_to_canonical_roi_mapping}) != len(complete)
    ):
        raise ValueError("roi_runtime_projection_invalid")
    expected_mapping = tuple(x for x in r.physical_to_canonical_roi_mapping if x.canonical_roi_id in set(selected))
    if r.selected_physical_to_canonical_roi_mapping != expected_mapping or r.selected_physical_source_columns != tuple(x.physical_source_column for x in expected_mapping) or tuple(x.canonical_roi_id for x in expected_mapping) != selected:
        raise ValueError("selected_roi_scope_mismatch")
    if any(item.physical_roi_inventory != r.complete_physical_source_columns or item.physical_to_canonical_roi_mapping != r.physical_to_canonical_roi_mapping for item in s.ordered_sessions):
        raise ValueError("session_global_roi_mapping_mismatch")
    correction_ids = tuple(x.roi_id for x in c.per_roi_correction_strategy_map)
    if c.selected_canonical_roi_ids != selected or correction_ids != selected or len(set(correction_ids)) != len(correction_ids) or any(not isinstance(x, GuidedProductionPerRoiStrategy) for x in c.per_roi_correction_strategy_map):
        raise ValueError("correction_roi_coverage_mismatch")
    _verify_typed_values(c.correction_parameter_values)
    if any(
        item.strategy_family not in {"dynamic_fit", "signal_only"}
        or not _text(item.selected_strategy)
        or not _text(item.evidence_source_type)
        or not _text(item.evidence_reference_json)
        or not isinstance(item.explicit_user_mark, bool)
        or not _text(item.current_or_stale)
        or (item.strategy_family == "signal_only" and (item.dynamic_fit_mode is not None or item.selected_strategy != "signal_only"))
        or (item.strategy_family == "dynamic_fit" and not _text(item.dynamic_fit_mode))
        for item in c.per_roi_correction_strategy_map
    ):
        raise ValueError("correction_strategy_invalid")
    feature_ids = tuple(x.roi_id for x in f.per_roi_feature_event_map)
    if f.selected_canonical_roi_ids != selected or feature_ids != selected or len(set(feature_ids)) != len(feature_ids) or any(not isinstance(x, GuidedProductionPerRoiFeatureEvent) for x in f.per_roi_feature_event_map):
        raise ValueError("feature_roi_coverage_mismatch")
    if f.execution_mode != request.execution_mode or f.inactive_for_execution is not (request.execution_mode == "tonic"):
        raise ValueError("feature_execution_mode_mismatch")
    _verify_typed_values(f.effective_values)
    if any(
        not _text(item.roi_id) or not _text(item.source) or not _text(item.feature_event_profile_id)
        or not isinstance(item.explicit_user_mark, bool) or not _text(item.current_or_stale)
        for item in f.per_roi_feature_event_map
    ):
        raise ValueError("feature_runtime_projection_invalid")
    for item in f.per_roi_feature_event_map:
        _verify_typed_values(item.override_config_fields)
        _verify_typed_values(item.effective_config_fields)
    timing_vectors = (
        t.ordered_authoritative_source_start_times,
        t.ordered_actual_elapsed_sec,
        t.ordered_nominal_expected_elapsed_sec,
        t.ordered_support_start_offsets_sec,
        t.ordered_support_end_offsets_sec,
        t.ordered_support_start_absolute,
        t.ordered_support_end_absolute,
        t.ordered_overlap_origins_absolute,
    )
    expected_timing_vectors = (
        tuple(x.authoritative_source_start_time for x in s.ordered_sessions),
        tuple(x.actual_elapsed_sec for x in s.ordered_sessions),
        tuple(x.nominal_expected_elapsed_sec for x in s.ordered_sessions),
        tuple(x.support_start_offset_sec for x in s.ordered_sessions),
        tuple(x.support_end_offset_sec for x in s.ordered_sessions),
        tuple(x.support_start_absolute for x in s.ordered_sessions),
        tuple(x.support_end_absolute for x in s.ordered_sessions),
        tuple(x.overlap_origin_absolute for x in s.ordered_sessions),
    )
    if any(len(vector) != len(s.ordered_sessions) for vector in timing_vectors) or timing_vectors != expected_timing_vectors or t.source_session_sequence_identity != s.runtime_session_sequence_identity:
        raise ValueError("timing_source_order_mismatch")
    try:
        expected_run = _verify_output_paths(o.output_base_canonical, request.startup_artifact_path, o.output_base_path_style)
    except ValueError as exc:
        raise ValueError("output_runtime_projection_invalid") from exc
    if o.run_directory_path != expected_run or o.startup_artifact_path != request.startup_artifact_path or o.overwrite is not False:
        raise ValueError("output_runtime_projection_invalid")
    if (
        a.ordered_source_paths != s.ordered_source_paths
        or a.ordered_source_relative_paths != s.ordered_source_relative_paths
        or a.source_runtime_projection_identity != s.canonical_source_runtime_projection_identity
        or (a.source_format, a.acquisition_mode, a.execution_mode, a.run_type)
        != (request.source_format, request.acquisition_mode, request.execution_mode, request.run_type)
        or a.production_component_identifiers != GUIDED_NPM_PRODUCTION_COMPONENT_IDENTIFIERS
        or a.deferred_runtime_actions != GUIDED_NPM_PRODUCTION_DEFERRED_ACTIONS
        or a.production_loader_kind != "photometry_pipeline.io.adapters._load_npm"
        or a.production_correction_kind != "photometry_pipeline.pipeline.Pipeline._resolve_correction_map_for_chunk"
        or a.production_feature_kind != "photometry_pipeline.core.feature_extraction.extract_features"
        or a.production_output_kind != "photometry_pipeline.pipeline.Pipeline.run_pass_2"
    ):
        raise ValueError("adapter_runtime_projection_invalid")
    refs = (a.parser_runtime_projection_identity, a.timing_runtime_projection_identity, a.roi_runtime_projection_identity, a.correction_runtime_projection_identity, a.feature_runtime_projection_identity, a.output_runtime_projection_identity)
    expected_refs = (p.canonical_parser_runtime_projection_identity, t.canonical_timing_runtime_projection_identity, r.canonical_roi_runtime_projection_identity, c.canonical_correction_runtime_projection_identity, f.canonical_feature_runtime_projection_identity, o.canonical_output_runtime_projection_identity)
    if refs != expected_refs:
        raise ValueError("adapter_runtime_projection_reference_mismatch")
    _require_identity(p, "canonical_parser_runtime_projection_identity", compute_guided_npm_production_parser_runtime_projection_identity, "parser_runtime_projection_identity_mismatch")
    _require_identity(t, "canonical_timing_runtime_projection_identity", compute_guided_npm_production_timing_runtime_projection_identity, "timing_runtime_projection_identity_mismatch")
    _require_identity(r, "canonical_roi_runtime_projection_identity", compute_guided_npm_production_roi_runtime_projection_identity, "roi_runtime_projection_identity_mismatch")
    _require_identity(c, "canonical_correction_runtime_projection_identity", compute_guided_npm_production_correction_runtime_projection_identity, "correction_runtime_projection_identity_mismatch")
    _require_identity(f, "canonical_feature_runtime_projection_identity", compute_guided_npm_production_feature_runtime_projection_identity, "feature_runtime_projection_identity_mismatch")
    _require_identity(o, "canonical_output_runtime_projection_identity", compute_guided_npm_production_output_runtime_projection_identity, "output_runtime_projection_identity_mismatch")
    _require_identity(a, "canonical_adapter_runtime_projection_identity", compute_guided_npm_production_adapter_runtime_projection_identity, "adapter_runtime_projection_identity_mismatch")
    _require_identity(request, "canonical_execution_request_identity", compute_guided_npm_production_execution_request_identity, "execution_request_identity_mismatch")


def serialize_guided_npm_production_execution_request(request):
    verify_guided_npm_production_execution_request(request)
    return {"identity_domain": GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_IDENTITY_DOMAIN, **_canonical(request)}


def _restore(annotation, value):
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError("tuple_expected")
        return tuple(_restore(args[0], item) for item in value)
    if origin in (types.UnionType,):
        if value is None and type(None) in args:
            return None
        for option in args:
            if option is type(None):
                continue
            try:
                return _restore(option, value)
            except (TypeError, ValueError):
                pass
        raise ValueError("union_value_invalid")
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ValueError("mapping_expected")
        names = {item.name for item in fields(annotation)}
        if set(value) != names:
            raise ValueError("field_set_invalid")
        hints = get_type_hints(annotation)
        return annotation(**{item.name: _restore(hints[item.name], value[item.name]) for item in fields(annotation)})
    if annotation in (str, int, float, bool):
        if annotation is float and type(value) in (int, float) and not isinstance(value, bool):
            return float(value)
        if type(value) is not annotation:
            raise ValueError("primitive_type_invalid")
    return value


def deserialize_guided_npm_production_execution_request(payload: Mapping[str, Any]):
    try:
        if not isinstance(payload, Mapping) or payload.get("identity_domain") != GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_IDENTITY_DOMAIN:
            raise ValueError("identity_domain_invalid")
        value = dict(payload)
        del value["identity_domain"]
        request = _restore(GuidedNpmProductionExecutionRequest, value)
        verify_guided_npm_production_execution_request(request)
        return request
    except Exception as exc:
        raise ValueError("execution_request_serialization_invalid") from exc
