"""Pure mapping from an identified Guided validation request to production intent."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from photometry_pipeline.guided_backend_validation_request import (
    GUIDED_BACKEND_VALIDATION_COMPILER_VERSION,
    GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION,
    GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME,
    GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION,
    GUIDED_BACKEND_VALIDATION_SCOPE,
    GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION,
    GuidedBackendAcquisitionDatasetRequest,
    GuidedBackendConfirmedStrategyMark,
    GuidedBackendCorrectionRequest,
    GuidedBackendDiagnosticEvidenceRequest,
    GuidedBackendEvidenceReference,
    GuidedBackendFeatureEventRequest,
    GuidedBackendPerRoiFeatureEvent,
    GuidedBackendLocalContractState,
    GuidedBackendNpmAcquisitionDatasetRequest,
    GuidedBackendNpmParserRequest,
    GuidedBackendOutputRelationship,
    GuidedBackendOutputRequest,
    GuidedBackendRoiScopeRequest,
    GuidedBackendRwdParserRequest,
    GuidedBackendSourceCandidateFile,
    GuidedBackendSourceRequest,
    GuidedBackendTypedFieldValue,
    GuidedBackendValidationRequest,
    GuidedBackendValidationCompileSuccess,
    compute_guided_backend_validation_request_identity,
)
from photometry_pipeline.guided_normalized_recording import (
    NormalizedRecordingError,
    build_normalized_recording_description_payload,
    compute_npm_parser_contract_digest,
    compute_npm_support_policy_identity,
    deserialize_normalized_recording_description,
    serialize_normalized_recording_description,
)
from photometry_pipeline.guided_identity import (
    CANONICALIZATION_ALGORITHM_VERSION,
    GuidedIdentityError,
    encode_canonical_value,
)
from photometry_pipeline.guided_new_analysis_plan import (
    FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
)
from photometry_pipeline.core.types import (
    CORRECTION_STRATEGY_FAMILIES,
    PerRoiCorrectionSpec,
)


APPLICATION_BUILD_IDENTITY_SCHEMA_NAME = "photometry_application_build_identity"
APPLICATION_BUILD_IDENTITY_SCHEMA_VERSION = "v1"
APPLICATION_BUILD_IDENTITY_DOMAIN = "photometry-application-build-identity:v1"
APPLICATION_BUILD_IDENTITY_PROVIDER_VERSION = (
    "application_build_identity_provider.v1"
)
GUIDED_PRODUCTION_INTENT_SCHEMA_NAME = "guided_production_execution_intent"
GUIDED_PRODUCTION_INTENT_SCHEMA_VERSION = "v1"
GUIDED_PRODUCTION_INTENT_IDENTITY_DOMAIN = (
    "guided-production-execution-intent:v1"
)
GUIDED_PRODUCTION_MAPPING_SCHEMA_NAME = "guided_production_mapping_contract"
GUIDED_PRODUCTION_MAPPING_SCHEMA_VERSION = "v1"
GUIDED_PRODUCTION_MAPPING_CONTRACT_VERSION = "guided_production_mapping.v1"
GUIDED_NPM_PRODUCTION_INTENT_SCHEMA_NAME = (
    "guided_production_npm_execution_intent"
)
GUIDED_NPM_PRODUCTION_INTENT_SCHEMA_VERSION = "v1"
GUIDED_NPM_PRODUCTION_INTENT_IDENTITY_DOMAIN = (
    "guided_production_npm_execution_intent.v1"
)
GUIDED_NPM_PRODUCTION_RUNNER_CONTRACT_VERSION = (
    "guided_npm_runner_not_yet_startable.v1"
)
GUIDED_NPM_PRODUCTION_CAPABILITY_STATUS = (
    "supported_by_production_mapping_but_startup_not_implemented"
)

GUIDED_PRODUCTION_MAPPING_REFUSAL_CATEGORIES = (
    "request_missing_or_invalid",
    "request_identity_invalid",
    "request_identity_mismatch",
    "unsupported_request_schema",
    "unsupported_validation_scope",
    "unsupported_subset_rule",
    "unsupported_source_format",
    "unsupported_acquisition_mode",
    "unsupported_analysis_scope",
    "unsupported_correction_strategy",
    "signal_only_not_supported",
    "mixed_strategy_not_supported",
    "incomplete_final_policy_not_supported",
    "output_policy_not_supported",
    "unresolved_request_field",
    "deferred_capability_blocks_mapping",
    "app_build_identity_missing",
    "app_build_identity_unusable",
    "mapping_contract_unavailable",
    "runner_contract_unavailable",
    "production_config_field_unmapped",
    "candidate_snapshot_execution_contract_unavailable",
    "roi_execution_contract_unavailable",
    "unsupported_analysis_configuration",
    "stale_or_mismatched_validation",
    "incomplete_correction_settings",
    "incomplete_feature_settings",
    "per_session_evidence_not_identity_bound",
    "mapping_internal_error",
)
GUIDED_PRODUCTION_MAPPING_REFUSAL_CATEGORY_SET = frozenset(
    GUIDED_PRODUCTION_MAPPING_REFUSAL_CATEGORIES
)
GUIDED_PRODUCTION_MAPPING_RESERVED_REFUSAL_CATEGORIES = (
    "source_manifest_required",
    "roi_identity_required",
    "runner_capability_mismatch",
    "config_mapping_contract_unavailable",
    "execution_intent_schema_unsupported",
)

_PLACEHOLDERS = frozenset({"", "unknown", "unavailable", "placeholder", "unset", "none"})
_HEX = frozenset("0123456789abcdef")
_SOURCE_REVISION_KINDS = frozenset({"git", "packaged_artifact", "unavailable"})
_SOURCE_TREE_STATES = frozenset({"clean", "dirty_content_bound", "unavailable"})


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _usable_version(value: Any) -> bool:
    return _text(value) and value.strip().lower() not in _PLACEHOLDERS


def _require_tuple(value: Any, name: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple.")


@dataclass(frozen=True)
class ApplicationBuildIdentity:
    schema_name: str
    schema_version: str
    identity_provider_version: str
    distribution_name: str
    distribution_version: str
    source_revision_kind: str
    source_revision: str
    source_tree_state: str
    source_tree_digest: str | None
    build_artifact_digest: str | None
    canonical_identity: str

    def __post_init__(self) -> None:
        for name in (
            "schema_name",
            "schema_version",
            "identity_provider_version",
            "distribution_name",
            "distribution_version",
            "source_revision_kind",
            "source_revision",
            "source_tree_state",
        ):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} must be a non-empty string.")
        if self.source_revision_kind not in _SOURCE_REVISION_KINDS:
            raise ValueError("Unsupported source_revision_kind.")
        if self.source_tree_state not in _SOURCE_TREE_STATES:
            raise ValueError("Unsupported source_tree_state.")
        for name in ("source_tree_digest", "build_artifact_digest"):
            value = getattr(self, name)
            if value is not None and not _sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256.")
        if self.source_tree_state == "dirty_content_bound" and not _sha256(
            self.source_tree_digest
        ):
            raise ValueError("dirty_content_bound requires source_tree_digest.")
        if self.source_tree_state == "unavailable" and not _sha256(
            self.build_artifact_digest
        ):
            raise ValueError("unavailable source requires build_artifact_digest.")
        if self.source_tree_state == "clean" and (
            self.source_revision.strip().lower() in _PLACEHOLDERS
            and not _sha256(self.build_artifact_digest)
        ):
            raise ValueError("clean source requires a revision or artifact digest.")
        if not _sha256(self.canonical_identity):
            raise ValueError("canonical_identity must be a lowercase SHA-256.")


def _build_identity_payload(identity: ApplicationBuildIdentity) -> dict[str, Any]:
    return {
        "identity_domain": APPLICATION_BUILD_IDENTITY_DOMAIN,
        "build": {
            name: getattr(identity, name)
            for name in (
                "schema_name",
                "schema_version",
                "identity_provider_version",
                "distribution_name",
                "distribution_version",
                "source_revision_kind",
                "source_revision",
                "source_tree_state",
                "source_tree_digest",
                "build_artifact_digest",
            )
        },
    }


def build_application_build_identity(
    *,
    distribution_name: str,
    distribution_version: str,
    source_revision_kind: str,
    source_revision: str,
    source_tree_state: str,
    source_tree_digest: str | None = None,
    build_artifact_digest: str | None = None,
    identity_provider_version: str = APPLICATION_BUILD_IDENTITY_PROVIDER_VERSION,
) -> ApplicationBuildIdentity:
    provisional = ApplicationBuildIdentity(
        schema_name=APPLICATION_BUILD_IDENTITY_SCHEMA_NAME,
        schema_version=APPLICATION_BUILD_IDENTITY_SCHEMA_VERSION,
        identity_provider_version=identity_provider_version,
        distribution_name=distribution_name,
        distribution_version=distribution_version,
        source_revision_kind=source_revision_kind,
        source_revision=source_revision,
        source_tree_state=source_tree_state,
        source_tree_digest=source_tree_digest,
        build_artifact_digest=build_artifact_digest,
        canonical_identity="0" * 64,
    )
    digest = hashlib.sha256(encode_canonical_value(_build_identity_payload(provisional))).hexdigest()
    return replace(provisional, canonical_identity=digest)


@dataclass(frozen=True)
class GuidedProductionMappingContract:
    mapping_schema_name: str
    mapping_schema_version: str
    mapping_contract_version: str
    supported_request_schema_name: str
    supported_request_schema_version: str
    supported_validation_scope: str
    supported_validation_contract_version: str
    supported_subset_rule_version: str
    supported_compiler_version: str
    supported_canonicalization_algorithm_version: str
    runner_contract_version: str
    candidate_manifest_execution_contract_version: str
    roi_execution_contract_version: str
    config_mapping_contract_version: str
    build_identity_policy: str
    allowed_nonblocking_deferred_capabilities: tuple[str, ...]
    stage_deferred_capabilities: tuple[str, ...]
    blocking_deferred_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name.endswith("capabilities"):
                _require_tuple(value, item.name)
            elif not _text(value):
                raise ValueError(f"{item.name} must be a non-empty string.")
        if "run_authorization" not in self.stage_deferred_capabilities:
            raise ValueError(
                "run_authorization must be explicitly deferred to authorization."
            )
        policy_groups = (
            set(self.allowed_nonblocking_deferred_capabilities),
            set(self.stage_deferred_capabilities),
            set(self.blocking_deferred_capabilities),
        )
        if any(
            left & right
            for index, left in enumerate(policy_groups)
            for right in policy_groups[index + 1 :]
        ):
            raise ValueError("Deferred capability policies overlap.")


def build_guided_production_mapping_contract(
    *,
    runner_contract_version: str = "guided_runner_candidate_roi_bound.v1",
    candidate_manifest_execution_contract_version: str = (
        "exact_candidate_manifest_preflight.v1"
    ),
    roi_execution_contract_version: str = "exact_included_roi_tuple.v1",
    config_mapping_contract_version: str = "guided_config_mapping.v1",
    mapping_contract_version: str = GUIDED_PRODUCTION_MAPPING_CONTRACT_VERSION,
    build_identity_policy: str = "content_bound_build_required.v1",
) -> GuidedProductionMappingContract:
    return GuidedProductionMappingContract(
        mapping_schema_name=GUIDED_PRODUCTION_MAPPING_SCHEMA_NAME,
        mapping_schema_version=GUIDED_PRODUCTION_MAPPING_SCHEMA_VERSION,
        mapping_contract_version=mapping_contract_version,
        supported_request_schema_name=GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME,
        supported_request_schema_version=GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION,
        supported_validation_scope=GUIDED_BACKEND_VALIDATION_SCOPE,
        supported_validation_contract_version=GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION,
        supported_subset_rule_version=GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION,
        supported_compiler_version=GUIDED_BACKEND_VALIDATION_COMPILER_VERSION,
        supported_canonicalization_algorithm_version=CANONICALIZATION_ALGORITHM_VERSION,
        runner_contract_version=runner_contract_version,
        candidate_manifest_execution_contract_version=(
            candidate_manifest_execution_contract_version
        ),
        roi_execution_contract_version=roi_execution_contract_version,
        config_mapping_contract_version=config_mapping_contract_version,
        build_identity_policy=build_identity_policy,
        allowed_nonblocking_deferred_capabilities=(
            "backend_validation",
            "full_source_manifest_identity",
            "strict_roi_inventory_identity",
        ),
        stage_deferred_capabilities=("run_authorization",),
        blocking_deferred_capabilities=(),
    )


@dataclass(frozen=True)
class GuidedProductionTypedValue:
    field_name: str
    value_type: str
    value: str | bool | int | float | None
    source_classification: str


@dataclass(frozen=True)
class GuidedProductionSourceCandidate:
    canonical_relative_path: str
    size_bytes: int
    sha256_content_digest: str


@dataclass(frozen=True)
class GuidedProductionInputSource:
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
    candidate_files: tuple[GuidedProductionSourceCandidate, ...]
    source_identity_level: str
    candidate_manifest_execution_contract_version: str
    approved_missing_candidates: tuple[GuidedProductionSourceCandidate, ...] = ()


@dataclass(frozen=True)
class GuidedProductionAcquisition:
    acquisition_mode: str
    sessions_per_hour: int
    session_duration_sec: float
    timeline_anchor_mode: str
    fixed_daily_anchor_clock: str | None
    allow_partial_final_window: bool
    exclude_incomplete_final_rwd_chunk: bool
    classification_schema_name: str
    classification_schema_version: str
    classifier_version: str
    classification_status: str
    not_requested_classification_digest: str
    dataset_snapshot_schema_version: str
    rwd_time_col: str
    uv_suffix: str
    sig_suffix: str
    semantic_values: tuple[GuidedProductionTypedValue, ...]
    dataset_source_setup_signature: str
    diagnostic_cache_contract_identity: str
    execution_mode: str = "phasic"


@dataclass(frozen=True)
class GuidedProductionParser:
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


@dataclass(frozen=True)
class GuidedProductionRoiScope:
    discovered_roi_ids: tuple[str, ...]
    included_roi_ids: tuple[str, ...]
    excluded_roi_ids: tuple[str, ...]
    selection_mode: str
    inventory_status: str
    inventory_source_content_digest: str
    roi_inventory_identity_status: str
    roi_execution_contract_version: str


@dataclass(frozen=True)
class GuidedProductionConfirmedMark:
    roi_id: str
    selected_dynamic_fit_mode: str
    diagnostic_cache_id: str
    source_setup_signature: str
    diagnostic_scope_signature: str
    build_request_signature: str
    evidence_reference_id: str
    evidence_chunk: int
    explicit_user_mark: bool
    current: bool


@dataclass(frozen=True)
class GuidedProductionPerRoiStrategy:
    roi_id: str
    strategy_family: str
    dynamic_fit_mode: str | None
    selected_strategy: str
    evidence_source_type: str
    evidence_reference_json: str
    explicit_user_mark: bool
    current_or_stale: str


@dataclass(frozen=True)
class GuidedProductionCorrection:
    strategy_scope: str
    global_correction_strategy: str
    global_dynamic_fit_mode: str
    dynamic_fit_parameter_values: tuple[GuidedProductionTypedValue, ...]
    confirmed_marks: tuple[GuidedProductionConfirmedMark, ...]
    mark_rule_version: str
    currentness_rule_version: str
    unanimity_rule_version: str
    production_strategy_map_version: str = ""
    per_roi_production_strategy_map: tuple[
        GuidedProductionPerRoiStrategy, ...
    ] = ()
    # Deprecated: the obsolete Guided post-hoc applied-dF/F route has been
    # retired from current-Guided production. Retained only as inert
    # deprecated input; guided_startup_materialization.py no longer acts
    # on it, and it is never serialized into new startup artifacts.
    applied_dff_orchestration_enabled: bool = False


@dataclass(frozen=True)
class GuidedProductionEvidenceReference:
    evidence_reference_id: str
    evidence_kind: str
    diagnostic_cache_id: str
    source_setup_signature: str
    current: bool
    diagnostic_scope_signature: str
    build_request_signature: str
    evidence_chunk: int
    roi_id: str
    selected_dynamic_fit_mode: str


@dataclass(frozen=True)
class GuidedProductionDiagnosticEvidence:
    cache_id: str
    cache_root_canonical: str
    source_setup_signature: str
    diagnostic_scope_signature: str
    build_request_signature: str
    artifact_contract_version: str
    provenance_schema_version: str
    artifact_semantic_digest: str
    provenance_semantic_digest: str
    evidence_references: tuple[GuidedProductionEvidenceReference, ...]
    completed_run_rejection_category: str
    resolver_status: str
    preliminary_cache: bool
    production_analysis: bool


@dataclass(frozen=True)
class GuidedProductionPerRoiFeatureEvent:
    roi_id: str
    source: str  # "default" or "override"
    feature_event_profile_id: str
    override_config_fields: tuple[GuidedProductionTypedValue, ...]
    effective_config_fields: tuple[GuidedProductionTypedValue, ...]
    explicit_user_mark: bool
    current_or_stale: str


@dataclass(frozen=True)
class GuidedProductionFeatureEvent:
    profile_schema_version: str
    profile_id: str
    effective_values: tuple[GuidedProductionTypedValue, ...]
    active_fields: tuple[str, ...]
    inactive_fields: tuple[str, ...]
    profile_status: str
    explicitly_applied: bool
    current: bool
    visible_unapplied_changes: bool
    per_roi_feature_event_map_version: str = ""
    per_roi_feature_event_map: tuple[GuidedProductionPerRoiFeatureEvent, ...] = ()


@dataclass(frozen=True)
class GuidedProductionOutputRelationship:
    relationship: str
    root_kind: str
    status: str


@dataclass(frozen=True)
class GuidedProductionOutputPolicy:
    output_base_canonical: str
    output_base_path_style: str
    path_role: str
    future_output_owner: str
    child_directory_strategy: str
    creation_timing: str
    overwrite: bool
    precreate: bool
    safety_classifier_version: str
    relationships: tuple[GuidedProductionOutputRelationship, ...]
    protected_root_context_complete: bool
    filesystem_fact_scope: str


@dataclass(frozen=True)
class GuidedProductionExecutionProfile:
    execution_mode: str = "phasic"
    run_type: str = "full"
    traces_only: bool = False
    allocate_output_at_future_run_start_only: bool = True


@dataclass(frozen=True)
class GuidedProductionProvenanceRequirements:
    validation_request_identity: str
    mapping_contract_version: str
    config_mapping_contract_version: str
    runner_contract_version: str
    validator_contract_version: str
    validator_capability_version: str
    source_candidate_set_digest: str
    source_candidate_content_digest: str
    parser_contract_digest: str
    diagnostic_cache_id: str
    feature_event_profile_id: str


@dataclass(frozen=True)
class GuidedProductionExecutionIntent:
    intent_schema_name: str
    intent_schema_version: str
    mapping_contract_version: str
    runner_contract_version: str
    source_request_identity: str
    source_validation_scope: str
    source_validation_contract_version: str
    source_validator_capability_version: str
    source_subset_rule_version: str
    application_build_identity: ApplicationBuildIdentity
    input_source: GuidedProductionInputSource
    acquisition: GuidedProductionAcquisition
    parser: GuidedProductionParser
    roi_scope: GuidedProductionRoiScope
    correction: GuidedProductionCorrection
    diagnostic_evidence: GuidedProductionDiagnosticEvidence
    feature_event: GuidedProductionFeatureEvent
    output_policy: GuidedProductionOutputPolicy
    execution_profile: GuidedProductionExecutionProfile
    provenance_requirements: GuidedProductionProvenanceRequirements
    deferred_capabilities: tuple[str, ...]
    # B1: the shared, format-neutral normalized recording description's
    # canonical identity (guided_normalized_recording.py), carried
    # through from the validated request. Automatically covered by
    # compute_guided_production_execution_intent_identity via
    # _INTENT_IDENTITY_MODEL_FIELDS' fields()-introspection, so
    # canonical_intent_identity already changes if this does.
    normalized_recording_description_identity: str
    canonical_intent_identity: str

    def __post_init__(self) -> None:
        if not _sha256(self.source_request_identity):
            raise ValueError("source_request_identity must be a lowercase SHA-256.")
        if not _sha256(self.normalized_recording_description_identity):
            raise ValueError(
                "normalized_recording_description_identity must be a lowercase SHA-256."
            )
        if not _sha256(self.canonical_intent_identity):
            raise ValueError("canonical_intent_identity must be a lowercase SHA-256.")
        _require_tuple(self.deferred_capabilities, "deferred_capabilities")
        if (
            self.execution_profile.execution_mode not in {"phasic", "tonic", "both"}
            or self.execution_profile.run_type != "full"
            or self.execution_profile.traces_only is not False
            or self.execution_profile.allocate_output_at_future_run_start_only
            is not True
            or self.output_policy.overwrite is not False
        ):
            raise ValueError("Intent is outside the first production subset.")


@dataclass(frozen=True)
class GuidedNpmProductionExecutionIntent:
    """Immutable, non-runnable production mapping for an accepted NPM setup.

    This sibling intentionally does not reuse the RWD execution-intent
    dataclass.  NPM has no RWD candidate-manifest or global dynamic-fit
    semantics, and keeping the variants separate prevents NPM fields from
    changing RWD identity, serialization, or authorization behavior.
    """

    intent_schema_name: str
    intent_schema_version: str
    mapping_contract_version: str
    runner_contract_version: str
    source_request_identity: str
    validation_status: str
    validation_revision: int
    current_plan_identity: str
    application_build_identity: ApplicationBuildIdentity
    source_format: str
    source_root_canonical: str
    acquisition_mode: str
    source_candidate_files: tuple[GuidedProductionSourceCandidate, ...]
    source_snapshot_set_identity: str
    source_snapshot_content_identity: str
    source_snapshot_identity: str
    normalized_recording_description_identity: str
    normalized_recording_payload_json: str
    parser_policy_identity: str
    parser_policy_content_json: str
    ordered_session_identity: str
    per_session_resolved_evidence_identity: str
    physical_to_canonical_roi_mapping_identity: str
    support_policy_identity: str
    output_time_basis_identity: str
    target_fs_hz: float
    session_duration_sec: float
    sessions_per_hour: int
    execution_mode: str
    run_type: str
    discovered_roi_ids: tuple[str, ...]
    selected_roi_ids: tuple[str, ...]
    excluded_roi_ids: tuple[str, ...]
    correction_parameter_values: tuple[GuidedProductionTypedValue, ...]
    per_roi_correction_strategy_map: tuple[GuidedProductionPerRoiStrategy, ...]
    correction_payload_identity: str
    feature_event: GuidedProductionFeatureEvent
    feature_payload_identity: str
    output_policy: GuidedProductionOutputPolicy
    deferred_capabilities: tuple[str, ...]
    capability_status: str
    canonical_intent_identity: str

    def __post_init__(self) -> None:
        if self.intent_schema_name != GUIDED_NPM_PRODUCTION_INTENT_SCHEMA_NAME:
            raise ValueError("Unsupported NPM production intent schema name.")
        if self.intent_schema_version != GUIDED_NPM_PRODUCTION_INTENT_SCHEMA_VERSION:
            raise ValueError("Unsupported NPM production intent schema version.")
        for name in (
            "source_request_identity",
            "current_plan_identity",
            "source_snapshot_set_identity",
            "source_snapshot_content_identity",
            "source_snapshot_identity",
            "normalized_recording_description_identity",
            "parser_policy_identity",
            "ordered_session_identity",
            "per_session_resolved_evidence_identity",
            "physical_to_canonical_roi_mapping_identity",
            "support_policy_identity",
            "output_time_basis_identity",
            "correction_payload_identity",
            "feature_payload_identity",
            "canonical_intent_identity",
        ):
            if not _sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256.")
        for name in (
            "mapping_contract_version",
            "runner_contract_version",
            "validation_status",
            "source_format",
            "source_root_canonical",
            "acquisition_mode",
            "normalized_recording_payload_json",
            "parser_policy_content_json",
            "run_type",
            "capability_status",
        ):
            if not _text(getattr(self, name)):
                raise ValueError(f"{name} must be a non-empty string.")
        if self.source_format != "npm" or self.acquisition_mode != "intermittent":
            raise ValueError("NPM production intent has unsupported source facts.")
        if (
            isinstance(self.validation_revision, bool)
            or not isinstance(self.validation_revision, int)
            or self.validation_revision < 0
        ):
            raise ValueError("validation_revision must be a non-negative integer.")
        if (
            isinstance(self.target_fs_hz, bool)
            or not isinstance(self.target_fs_hz, (int, float))
            or not math.isfinite(float(self.target_fs_hz))
            or self.target_fs_hz <= 0
            or isinstance(self.session_duration_sec, bool)
            or not isinstance(self.session_duration_sec, (int, float))
            or not math.isfinite(float(self.session_duration_sec))
            or self.session_duration_sec <= 0
            or isinstance(self.sessions_per_hour, bool)
            or not isinstance(self.sessions_per_hour, int)
            or self.sessions_per_hour <= 0
        ):
            raise ValueError("NPM sampling facts must be positive and finite.")
        if self.execution_mode not in {"phasic", "tonic", "both"}:
            raise ValueError("Unsupported NPM execution mode.")
        if self.run_type != "full":
            raise ValueError("Only full NPM runs are in the mapped subset.")
        for name in (
            "source_candidate_files",
            "discovered_roi_ids",
            "selected_roi_ids",
            "excluded_roi_ids",
            "correction_parameter_values",
            "per_roi_correction_strategy_map",
            "deferred_capabilities",
        ):
            _require_tuple(getattr(self, name), name)
        if not self.source_candidate_files:
            raise ValueError("NPM source candidate identity is required.")
        if (
            not self.discovered_roi_ids
            or not self.selected_roi_ids
            or len(set(self.discovered_roi_ids)) != len(self.discovered_roi_ids)
            or len(set(self.selected_roi_ids)) != len(self.selected_roi_ids)
            or len(set(self.excluded_roi_ids)) != len(self.excluded_roi_ids)
            or set(self.selected_roi_ids) - set(self.discovered_roi_ids)
            or set(self.excluded_roi_ids) - set(self.discovered_roi_ids)
            or set(self.selected_roi_ids) & set(self.excluded_roi_ids)
        ):
            raise ValueError("NPM ROI scope is incomplete or inconsistent.")
        strategy_by_roi = {
            entry.roi_id: entry for entry in self.per_roi_correction_strategy_map
        }
        if set(strategy_by_roi) != set(self.selected_roi_ids):
            raise ValueError("NPM correction strategy map must cover selected ROIs exactly.")
        for entry in self.per_roi_correction_strategy_map:
            if (
                entry.strategy_family not in CORRECTION_STRATEGY_FAMILIES
                or not entry.explicit_user_mark
                or entry.current_or_stale != "current"
            ):
                raise ValueError("NPM correction strategy map contains an unusable entry.")
            if entry.strategy_family == "signal_only_f0":
                if entry.selected_strategy != "signal_only_f0" or entry.dynamic_fit_mode is not None:
                    raise ValueError("Signal-Only NPM strategy entry is inconsistent.")
            elif (
                entry.selected_strategy not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES
                or entry.dynamic_fit_mode != entry.selected_strategy
            ):
                raise ValueError("Dynamic-fit NPM strategy entry is inconsistent.")
        if self.capability_status != GUIDED_NPM_PRODUCTION_CAPABILITY_STATUS:
            raise ValueError("Unsupported NPM capability status.")
        if self.output_policy.overwrite or self.output_policy.precreate:
            raise ValueError("NPM production intent cannot authorize output mutation.")


@dataclass(frozen=True)
class GuidedProductionMappingIssue:
    category: str
    section: str
    message: str
    detail_code: str = ""

    def __post_init__(self) -> None:
        if self.category not in GUIDED_PRODUCTION_MAPPING_REFUSAL_CATEGORY_SET:
            raise ValueError("Unsupported mapping refusal category.")
        if not _text(self.section) or not _text(self.message):
            raise ValueError("Mapping issue requires section and message.")


@dataclass(frozen=True)
class GuidedProductionMappingSuccess:
    intent: GuidedProductionExecutionIntent
    canonical_intent_identity: str
    source_request_identity: str
    status: str = "mapped"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.intent, GuidedProductionExecutionIntent)
            or self.canonical_intent_identity
            != self.intent.canonical_intent_identity
            or self.source_request_identity != self.intent.source_request_identity
        ):
            raise ValueError("Mapping success identities are inconsistent.")


@dataclass(frozen=True)
class GuidedProductionMappingFailure:
    blocking_issues: tuple[GuidedProductionMappingIssue, ...]
    partial_intent: None = None
    canonical_intent_identity: None = None
    status: str = "refused"

    def __post_init__(self) -> None:
        if not isinstance(self.blocking_issues, tuple) or not self.blocking_issues:
            raise ValueError("Mapping failure requires blocking issues.")


GuidedProductionMappingResult = (
    GuidedProductionMappingSuccess | GuidedProductionMappingFailure
)


@dataclass(frozen=True)
class GuidedNpmProductionCapabilityResult:
    status: str
    production_mapping_supported: bool
    startup_available: bool
    runnable: bool
    blocking_issues: tuple[GuidedProductionMappingIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {
            GUIDED_NPM_PRODUCTION_CAPABILITY_STATUS,
            "unsupported_analysis_configuration",
            "stale_or_mismatched_validation",
            "incomplete_correction_settings",
            "incomplete_feature_settings",
            "per_session_evidence_not_identity_bound",
        }:
            raise ValueError("Unsupported NPM capability status.")
        if not isinstance(self.blocking_issues, tuple):
            raise ValueError("blocking_issues must be a tuple.")
        if self.runnable is not False or self.startup_available is not False:
            raise ValueError("B2-C1 NPM capability cannot be runnable.")


@dataclass(frozen=True)
class GuidedNpmProductionMappingSuccess:
    intent: GuidedNpmProductionExecutionIntent
    capability: GuidedNpmProductionCapabilityResult
    canonical_intent_identity: str
    source_request_identity: str
    status: str = "mapped"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.intent, GuidedNpmProductionExecutionIntent)
            or not isinstance(self.capability, GuidedNpmProductionCapabilityResult)
            or self.canonical_intent_identity != self.intent.canonical_intent_identity
            or self.source_request_identity != self.intent.source_request_identity
        ):
            raise ValueError("NPM mapping success identities are inconsistent.")


GuidedNpmProductionMappingResult = (
    GuidedNpmProductionMappingSuccess | GuidedProductionMappingFailure
)


def _failure(
    category: str, section: str, message: str, detail_code: str
) -> GuidedProductionMappingFailure:
    return GuidedProductionMappingFailure(
        (GuidedProductionMappingIssue(category, section, message, detail_code),)
    )


_INTENT_IDENTITY_MODEL_FIELDS = {
    ApplicationBuildIdentity: tuple(item.name for item in fields(ApplicationBuildIdentity)),
    GuidedProductionTypedValue: tuple(item.name for item in fields(GuidedProductionTypedValue)),
    GuidedProductionSourceCandidate: tuple(item.name for item in fields(GuidedProductionSourceCandidate)),
    GuidedProductionInputSource: tuple(item.name for item in fields(GuidedProductionInputSource)),
    GuidedProductionAcquisition: tuple(item.name for item in fields(GuidedProductionAcquisition)),
    GuidedProductionParser: tuple(item.name for item in fields(GuidedProductionParser)),
    GuidedProductionRoiScope: tuple(item.name for item in fields(GuidedProductionRoiScope)),
    GuidedProductionConfirmedMark: tuple(item.name for item in fields(GuidedProductionConfirmedMark)),
    GuidedProductionPerRoiStrategy: tuple(item.name for item in fields(GuidedProductionPerRoiStrategy)),
    GuidedProductionCorrection: tuple(item.name for item in fields(GuidedProductionCorrection)),
    GuidedProductionEvidenceReference: tuple(item.name for item in fields(GuidedProductionEvidenceReference)),
    GuidedProductionDiagnosticEvidence: tuple(item.name for item in fields(GuidedProductionDiagnosticEvidence)),
    GuidedProductionPerRoiFeatureEvent: tuple(item.name for item in fields(GuidedProductionPerRoiFeatureEvent)),
    GuidedProductionFeatureEvent: tuple(item.name for item in fields(GuidedProductionFeatureEvent)),
    GuidedProductionOutputRelationship: tuple(item.name for item in fields(GuidedProductionOutputRelationship)),
    GuidedProductionOutputPolicy: tuple(item.name for item in fields(GuidedProductionOutputPolicy)),
    GuidedProductionExecutionProfile: tuple(item.name for item in fields(GuidedProductionExecutionProfile)),
    GuidedProductionProvenanceRequirements: tuple(item.name for item in fields(GuidedProductionProvenanceRequirements)),
    GuidedProductionExecutionIntent: tuple(
        item.name
        for item in fields(GuidedProductionExecutionIntent)
        if item.name != "canonical_intent_identity"
    ),
}

_NPM_INTENT_IDENTITY_MODEL_FIELDS = {
    GuidedNpmProductionExecutionIntent: tuple(
        item.name
        for item in fields(GuidedNpmProductionExecutionIntent)
        if item.name != "canonical_intent_identity"
    )
}


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite values are not canonical.")
        return value
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    names = _INTENT_IDENTITY_MODEL_FIELDS.get(type(value))
    if names is None:
        raise ValueError("Unsupported production intent value type.")
    return {name: _canonical_value(getattr(value, name)) for name in names}


def _npm_canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite values are not canonical.")
        return value
    if isinstance(value, tuple):
        return [_npm_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [_npm_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("NPM identity mapping keys must be strings.")
        return {
            key: _npm_canonical_value(item) for key, item in value.items()
        }
    names = _NPM_INTENT_IDENTITY_MODEL_FIELDS.get(type(value))
    if names is not None:
        return {
            name: _npm_canonical_value(getattr(value, name)) for name in names
        }
    if type(value) in _INTENT_IDENTITY_MODEL_FIELDS:
        return _canonical_value(value)
    raise ValueError("Unsupported NPM production intent value type.")


def compute_guided_production_execution_intent_identity(
    intent: GuidedProductionExecutionIntent,
) -> str:
    if not isinstance(intent, GuidedProductionExecutionIntent):
        raise ValueError("intent must be a GuidedProductionExecutionIntent.")
    payload = {
        "identity_domain": GUIDED_PRODUCTION_INTENT_IDENTITY_DOMAIN,
        "intent": _canonical_value(intent),
    }
    try:
        return hashlib.sha256(encode_canonical_value(payload)).hexdigest()
    except (GuidedIdentityError, TypeError, ValueError) as exc:
        raise ValueError("Production intent identity could not be computed.") from exc


def compute_guided_npm_production_execution_intent_identity(
    intent: GuidedNpmProductionExecutionIntent,
) -> str:
    """Compute the deterministic identity of the non-runnable NPM intent."""
    if not isinstance(intent, GuidedNpmProductionExecutionIntent):
        raise ValueError("intent must be a GuidedNpmProductionExecutionIntent.")
    payload = {
        "identity_domain": GUIDED_NPM_PRODUCTION_INTENT_IDENTITY_DOMAIN,
        "intent": _npm_canonical_value(intent),
    }
    try:
        return hashlib.sha256(encode_canonical_value(payload)).hexdigest()
    except (GuidedIdentityError, TypeError, ValueError) as exc:
        raise ValueError("NPM production intent identity could not be computed.") from exc


REQUEST_FIELD_CLASSIFICATIONS = {
    GuidedBackendValidationRequest: {name: "mapped_to_intent" for name in (
        "request_schema_name", "request_schema_version", "validation_scope",
        "validation_contract_version", "validator_capability_version",
        "compiler_version", "subset_rule_version",
        "canonicalization_algorithm_version", "source", "acquisition_dataset",
        "parser", "roi_scope", "correction", "diagnostic_evidence",
        "feature_event", "output", "local_contract",
        "normalized_recording_description_identity",
        "normalized_recording_description")},
    GuidedBackendSourceRequest: {name: ("gate_only" if name == "unresolved_source_identity_inputs" else "mapped_to_intent") for name in (item.name for item in fields(GuidedBackendSourceRequest))},
    GuidedBackendSourceCandidateFile: {name: "mapped_to_intent" for name in (item.name for item in fields(GuidedBackendSourceCandidateFile))},
    GuidedBackendAcquisitionDatasetRequest: {name: ("gate_only" if name in {"dataset_status", "dataset_current_applied", "validation_issue_categories", "stale_reason_categories"} else "mapped_to_intent") for name in (item.name for item in fields(GuidedBackendAcquisitionDatasetRequest))},
    GuidedBackendRwdParserRequest: {name: ("gate_only" if name == "unresolved_inputs" else "mapped_to_intent") for name in (item.name for item in fields(GuidedBackendRwdParserRequest))},
    GuidedBackendRoiScopeRequest: {name: "mapped_to_intent" for name in (item.name for item in fields(GuidedBackendRoiScopeRequest))},
    GuidedBackendCorrectionRequest: {name: ("gate_only" if name == "blocked_strategy_states" else "mapped_to_intent") for name in (item.name for item in fields(GuidedBackendCorrectionRequest))},
    GuidedBackendConfirmedStrategyMark: {name: "mapped_to_intent" for name in (item.name for item in fields(GuidedBackendConfirmedStrategyMark))},
    GuidedBackendDiagnosticEvidenceRequest: {name: ("gate_only" if name in {"stale_reasons", "unresolved_inputs"} else "mapped_to_intent") for name in (item.name for item in fields(GuidedBackendDiagnosticEvidenceRequest))},
    GuidedBackendEvidenceReference: {name: "mapped_to_intent" for name in (item.name for item in fields(GuidedBackendEvidenceReference))},
    GuidedBackendFeatureEventRequest: {name: ("gate_only" if name in {"validation_issue_categories", "stale_reason_categories"} else "mapped_to_intent") for name in (item.name for item in fields(GuidedBackendFeatureEventRequest))},
    GuidedBackendOutputRequest: {name: ("gate_only" if name in {"policy_status", "policy_current", "blocker_categories"} else "mapped_to_intent") for name in (item.name for item in fields(GuidedBackendOutputRequest))},
    GuidedBackendOutputRelationship: {name: "mapped_to_intent" for name in (item.name for item in fields(GuidedBackendOutputRelationship))},
    GuidedBackendLocalContractState: {name: ("deferred_allowed" if name == "deferred_capabilities" else "gate_only") for name in (item.name for item in fields(GuidedBackendLocalContractState))},
    GuidedBackendTypedFieldValue: {name: "mapped_to_intent" for name in (item.name for item in fields(GuidedBackendTypedFieldValue))},
}

ACQUISITION_TYPED_FIELD_CONFIG_MAP = frozenset(
    {
        "rwd_time_col", "uv_suffix", "sig_suffix", "target_fs_hz", "sessions_per_hour", "session_duration_sec",
        "acquisition_mode", "allow_partial_final_window", "exclude_incomplete_final_rwd_chunk",
        "input_format", "resolved_input_format", "continuous_window_sec", "continuous_step_sec",
    }
)
NPM_ACQUISITION_TYPED_FIELD_CONFIG_MAP = frozenset(
    {
        "npm_time_axis",
        "npm_system_ts_col",
        "npm_computer_ts_col",
        "npm_led_col",
        "npm_region_prefix",
        "npm_region_suffix",
        "target_fs_hz",
        "chunk_duration_sec",
        "allow_partial_final_chunk",
        "adapter_value_nan_policy",
        # The materialized dataset-contract snapshot's semantic_values
        # always carries the same universal draft-identity fields
        # regardless of format (guided_backend_validation_materialization.
        # materialize_guided_backend_validation_facts iterates every key of
        # snapshot.contract_values, not just the format-specific ones --
        # see ACQUISITION_TYPED_FIELD_CONFIG_MAP above, which already
        # allows these same fields for RWD). Omitting them here made every
        # real, GUI-driven NPM production mapping refuse with
        # "typed_field_unmapped", even for a genuinely accepted validation.
        "sessions_per_hour",
        "session_duration_sec",
        "acquisition_mode",
        "allow_partial_final_window",
        "exclude_incomplete_final_rwd_chunk",
        "input_format",
        "resolved_input_format",
        "continuous_window_sec",
        "continuous_step_sec",
    }
)
CORRECTION_TYPED_FIELD_CONFIG_MAP = frozenset(
    {
        "dynamic_fit_mode", "slope_constraint", "min_slope",
        "robust_event_reject_max_iters", "robust_event_reject_residual_z_thresh",
        "robust_event_reject_local_var_window_sec",
        "robust_event_reject_local_var_ratio_thresh",
        "robust_event_reject_min_keep_fraction",
        "adaptive_event_gate_residual_z_thresh",
        "adaptive_event_gate_local_var_window_sec",
        "adaptive_event_gate_local_var_ratio_thresh",
        "adaptive_event_gate_smooth_window_sec",
        "adaptive_event_gate_min_trust_fraction",
        "adaptive_event_gate_freeze_interp_method", "window_sec", "step_sec",
        "r_low", "r_high", "g_min", "min_samples_per_window",
        "min_valid_windows", "baseline_subtract_before_fit",
        "bleach_correction_mode",
    }
)
FEATURE_EVENT_TYPED_FIELD_CONFIG_MAP = frozenset(
    {
        "event_signal", "peak_threshold_method", "peak_threshold_k",
        "peak_threshold_percentile", "peak_threshold_abs",
        "peak_min_distance_sec", "peak_min_prominence_k",
        "peak_min_width_sec", "peak_pre_filter", "event_auc_baseline",
        "signal_excursion_polarity",
    }
)


def _typed(values: tuple[GuidedBackendTypedFieldValue, ...]) -> tuple[GuidedProductionTypedValue, ...]:
    return tuple(
        GuidedProductionTypedValue(
            item.field_name, item.value_type, item.value, item.source_classification
        )
        for item in values
    )


def _unknown_typed(
    values: tuple[GuidedBackendTypedFieldValue, ...], allowed: frozenset[str]
) -> bool:
    names = [item.field_name for item in values]
    return len(names) != len(set(names)) or any(name not in allowed for name in names)


def feature_entry_provenance_valid(
    *,
    entry_source: str,
    entry_feature_event_profile_id: str,
    entry_explicit_user_mark: bool,
    enclosing_profile_status: str,
    enclosing_profile_id: str,
    enclosing_current: bool,
    enclosing_visible_unapplied_changes: bool,
) -> bool:
    """Whether a per-ROI feature/event entry's provenance is acceptable.

    An entry the scientist explicitly marked is always acceptable. A
    default-sourced entry may also be acceptable without an explicit mark
    when the enclosing feature/event profile is itself a current, unedited
    loaded Default (``profile_status == "default_initialized"``) -- the
    same already-supported "valid without Apply" state recognized by
    is_saved_feature_event_profile_current -- and the entry's profile
    identity matches that enclosing profile exactly. An applied profile,
    and any override/custom-sourced entry, must still carry an explicit
    mark. Shared by production mapping and execution-authority per-ROI
    feature checks so the two layers cannot diverge.
    """
    if entry_explicit_user_mark:
        return True
    return bool(
        entry_source == "default"
        and enclosing_profile_status == "default_initialized"
        and enclosing_current is True
        and enclosing_visible_unapplied_changes is False
        and entry_feature_event_profile_id == enclosing_profile_id
    )


def _contract_problem(contract: Any) -> tuple[str, str] | None:
    if not isinstance(contract, GuidedProductionMappingContract):
        return ("mapping_contract_unavailable", "mapping_contract_invalid_type")
    expected = {
        "mapping_schema_name": GUIDED_PRODUCTION_MAPPING_SCHEMA_NAME,
        "mapping_schema_version": GUIDED_PRODUCTION_MAPPING_SCHEMA_VERSION,
        "supported_request_schema_name": GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_NAME,
        "supported_request_schema_version": GUIDED_BACKEND_VALIDATION_REQUEST_SCHEMA_VERSION,
        "supported_validation_scope": GUIDED_BACKEND_VALIDATION_SCOPE,
        "supported_validation_contract_version": GUIDED_BACKEND_VALIDATION_CONTRACT_VERSION,
        "supported_subset_rule_version": GUIDED_BACKEND_VALIDATION_SUBSET_RULE_VERSION,
        "supported_compiler_version": GUIDED_BACKEND_VALIDATION_COMPILER_VERSION,
        "supported_canonicalization_algorithm_version": CANONICALIZATION_ALGORITHM_VERSION,
    }
    if any(getattr(contract, name) != value for name, value in expected.items()):
        return ("mapping_contract_unavailable", "mapping_contract_mismatch")
    for name, category in (
        ("runner_contract_version", "runner_contract_unavailable"),
        ("candidate_manifest_execution_contract_version", "candidate_snapshot_execution_contract_unavailable"),
        ("roi_execution_contract_version", "roi_execution_contract_unavailable"),
        ("config_mapping_contract_version", "mapping_contract_unavailable"),
    ):
        if not _usable_version(getattr(contract, name)):
            return (category, f"{name}_invalid")
    return None


def map_guided_validation_request_to_execution_intent(
    request: GuidedBackendValidationRequest,
    *,
    canonical_request_identity: str,
    application_build_identity: ApplicationBuildIdentity,
    mapping_contract: GuidedProductionMappingContract,
) -> GuidedProductionMappingResult:
    if not isinstance(request, GuidedBackendValidationRequest):
        return _failure("request_missing_or_invalid", "request", "A Guided validation request is required.", "request_invalid_type")
    if not _sha256(canonical_request_identity):
        return _failure("request_identity_invalid", "identity", "Request identity must be lowercase SHA-256.", "request_identity_invalid")
    try:
        recomputed = compute_guided_backend_validation_request_identity(request)
    except Exception:
        return _failure("mapping_internal_error", "identity", "Request identity could not be recomputed.", "request_identity_recomputation_failed")
    if recomputed != canonical_request_identity:
        return _failure("request_identity_mismatch", "identity", "Request identity does not match request content.", "request_identity_mismatch")
    problem = _contract_problem(mapping_contract)
    if problem:
        return _failure(problem[0], "mapping_contract", "Production mapping contract is unavailable.", problem[1])
    if not isinstance(application_build_identity, ApplicationBuildIdentity):
        return _failure("app_build_identity_missing", "build_identity", "Application build identity is required.", "build_identity_invalid_type")
    try:
        expected_build = hashlib.sha256(
            encode_canonical_value(_build_identity_payload(application_build_identity))
        ).hexdigest()
    except Exception:
        expected_build = ""
    if (
        expected_build != application_build_identity.canonical_identity
        or not _usable_version(application_build_identity.distribution_name)
        or not _usable_version(application_build_identity.distribution_version)
    ):
        return _failure("app_build_identity_unusable", "build_identity", "Application build identity is unusable.", "build_identity_unusable")

    if request.request_schema_name != mapping_contract.supported_request_schema_name or request.request_schema_version != mapping_contract.supported_request_schema_version:
        return _failure("unsupported_request_schema", "request", "Request schema is unsupported.", "request_schema_mismatch")
    if request.validation_scope != mapping_contract.supported_validation_scope or request.validation_contract_version != mapping_contract.supported_validation_contract_version:
        return _failure("unsupported_validation_scope", "request", "Validation scope is unsupported.", "validation_scope_mismatch")
    if request.subset_rule_version != mapping_contract.supported_subset_rule_version or request.compiler_version != mapping_contract.supported_compiler_version or request.canonicalization_algorithm_version != mapping_contract.supported_canonicalization_algorithm_version:
        return _failure("unsupported_subset_rule", "request", "Request subset or compiler contract is unsupported.", "request_contract_mismatch")
    if request.source.source_format != "rwd":
        return _failure("unsupported_source_format", "source", "Only RWD source is supported.", "source_not_rwd")
    if not request.source.candidate_files or request.source.unresolved_source_identity_inputs:
        return _failure("unresolved_request_field", "source", "Source candidate snapshot is incomplete.", "source_snapshot_unresolved")
    if not _sha256(request.normalized_recording_description_identity):
        return _failure(
            "unresolved_request_field",
            "normalized_recording",
            "The normalized recording description identity is missing or invalid.",
            "normalized_recording_description_identity_invalid",
        )
    if request.acquisition_dataset.acquisition_mode != "intermittent":
        return _failure("unsupported_acquisition_mode", "acquisition", "Only intermittent acquisition is supported.", "acquisition_not_intermittent")
    if request.acquisition_dataset.allow_partial_final_window:
        return _failure("incomplete_final_policy_not_supported", "acquisition", "Final-window policy is unsupported.", "final_policy_unsupported")
    if request.parser.unresolved_inputs:
        return _failure("unresolved_request_field", "parser", "Parser inputs are unresolved.", "parser_unresolved")
    if not request.roi_scope.included_roi_ids:
        return _failure("unresolved_request_field", "roi_scope", "Included ROI set is empty.", "included_rois_empty")
    correction = request.correction
    native_per_roi = bool(correction.per_roi_production_strategy_map)
    if correction.strategy_scope != "global" or correction.global_correction_strategy != "dynamic_fit" or correction.global_dynamic_fit_mode not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
        return _failure("unsupported_correction_strategy", "correction", "Correction strategy is unsupported.", "correction_unsupported")
    modes = {item.selected_dynamic_fit_mode for item in correction.confirmed_marks}
    if not native_per_roi and modes != {correction.global_dynamic_fit_mode}:
        return _failure("mixed_strategy_not_supported", "correction", "Confirmed marks are not unanimous.", "confirmed_modes_mixed")
    output = request.output
    if (
        output.path_role != "output_base"
        or output.future_output_owner != "runner"
        or output.run_directory_strategy != "derive_unique_run_id_under_output_base"
        or output.creation_timing != "future_execution_start_only"
        or output.overwrite
        or output.precreate
        or output.blocker_categories
        or not output.protected_root_context_complete
    ):
        return _failure("output_policy_not_supported", "output", "Output policy is unsupported.", "output_policy_invalid")
    local = request.local_contract
    if local.blocking_issue_categories or local.warning_categories or local.unsupported_state_flags or local.unresolved_required_inputs:
        return _failure("unresolved_request_field", "local_contract", "Local contract is not clean.", "local_contract_not_clean")
    requested_deferred = set(local.deferred_capabilities)
    blocking = requested_deferred & set(
        mapping_contract.blocking_deferred_capabilities
    )
    if blocking:
        return _failure(
            "deferred_capability_blocks_mapping",
            "local_contract",
            "A production-mapping blocker remains deferred.",
            "blocking_deferred_capability",
        )
    known = (
        set(mapping_contract.allowed_nonblocking_deferred_capabilities)
        | set(mapping_contract.stage_deferred_capabilities)
        | {"app_build_identity"}
    )
    unknown = requested_deferred - known
    if unknown:
        return _failure("deferred_capability_blocks_mapping", "local_contract", "An unknown deferred capability blocks mapping.", "unknown_deferred_capability")
    if _unknown_typed(request.acquisition_dataset.semantic_values, ACQUISITION_TYPED_FIELD_CONFIG_MAP) or _unknown_typed(correction.dynamic_fit_parameter_values, CORRECTION_TYPED_FIELD_CONFIG_MAP) or _unknown_typed(request.feature_event.effective_values, FEATURE_EVENT_TYPED_FIELD_CONFIG_MAP):
        return _failure("production_config_field_unmapped", "typed_values", "A typed production field is unmapped.", "typed_field_unmapped")
    if any(
        _unknown_typed(entry.override_config_fields, FEATURE_EVENT_TYPED_FIELD_CONFIG_MAP)
        or _unknown_typed(entry.effective_config_fields, FEATURE_EVENT_TYPED_FIELD_CONFIG_MAP)
        for entry in request.feature_event.per_roi_feature_event_map
    ):
        return _failure("production_config_field_unmapped", "typed_values", "A per-ROI typed feature/event field is unmapped.", "per_roi_typed_field_unmapped")

    try:
        intent = GuidedProductionExecutionIntent(
            intent_schema_name=GUIDED_PRODUCTION_INTENT_SCHEMA_NAME,
            intent_schema_version=GUIDED_PRODUCTION_INTENT_SCHEMA_VERSION,
            mapping_contract_version=mapping_contract.mapping_contract_version,
            runner_contract_version=mapping_contract.runner_contract_version,
            source_request_identity=recomputed,
            source_validation_scope=request.validation_scope,
            source_validation_contract_version=request.validation_contract_version,
            source_validator_capability_version=request.validator_capability_version,
            source_subset_rule_version=request.subset_rule_version,
            application_build_identity=application_build_identity,
            input_source=GuidedProductionInputSource(
                request.source.source_root_canonical, request.source.source_root_path_style,
                request.source.source_format, request.source.snapshot_schema_name,
                request.source.snapshot_schema_version, request.source.discovery_rule_version,
                request.source.path_canonicalization_version, request.source.relative_path_rule_version,
                request.source.ignored_files_policy, request.source.build_mode,
                request.source.source_candidate_set_digest, request.source.source_candidate_content_digest,
                tuple(GuidedProductionSourceCandidate(item.canonical_relative_path, item.size_bytes, item.sha256_content_digest) for item in request.source.candidate_files),
                request.source.source_identity_level,
                mapping_contract.candidate_manifest_execution_contract_version,
                tuple(
                    GuidedProductionSourceCandidate(
                        item.canonical_relative_path,
                        item.size_bytes,
                        item.sha256_content_digest,
                    )
                    for item in request.source.approved_missing_candidates
                ),
            ),
            acquisition=GuidedProductionAcquisition(
                request.acquisition_dataset.acquisition_mode, request.acquisition_dataset.sessions_per_hour,
                request.acquisition_dataset.session_duration_sec, request.acquisition_dataset.timeline_anchor_mode,
                request.acquisition_dataset.fixed_daily_anchor_clock, request.acquisition_dataset.allow_partial_final_window,
                request.acquisition_dataset.exclude_incomplete_final_rwd_chunk,
                request.acquisition_dataset.classification_schema_name,
                request.acquisition_dataset.classification_schema_version,
                request.acquisition_dataset.classifier_version,
                request.acquisition_dataset.classification_status,
                request.acquisition_dataset.not_requested_classification_digest,
                request.acquisition_dataset.dataset_snapshot_schema_version,
                request.acquisition_dataset.rwd_time_col, request.acquisition_dataset.uv_suffix,
                request.acquisition_dataset.sig_suffix, _typed(request.acquisition_dataset.semantic_values),
                request.acquisition_dataset.dataset_source_setup_signature,
                request.acquisition_dataset.diagnostic_cache_contract_identity,
                request.acquisition_dataset.execution_mode,
            ),
            parser=GuidedProductionParser(
                request.parser.schema_name, request.parser.schema_version,
                request.parser.header_search_line_limit, request.parser.time_column_candidates,
                request.parser.uv_suffix_candidates, request.parser.signal_suffix_candidates,
                request.parser.column_normalization_rule, request.parser.roi_name_rule,
                request.parser.ambiguity_policy, request.parser.parser_contract_digest,
            ),
            roi_scope=GuidedProductionRoiScope(
                request.roi_scope.discovered_roi_ids, request.roi_scope.included_roi_ids,
                request.roi_scope.excluded_roi_ids, request.roi_scope.selection_mode,
                request.roi_scope.inventory_status, request.roi_scope.inventory_source_content_digest,
                request.roi_scope.roi_inventory_identity_status,
                mapping_contract.roi_execution_contract_version,
            ),
            correction=GuidedProductionCorrection(
                correction.strategy_scope, correction.global_correction_strategy,
                correction.global_dynamic_fit_mode, _typed(correction.dynamic_fit_parameter_values),
                tuple(GuidedProductionConfirmedMark(**{item.name: getattr(mark, item.name) for item in fields(GuidedBackendConfirmedStrategyMark)}) for mark in correction.confirmed_marks),
                correction.mark_rule_version, correction.currentness_rule_version,
                correction.unanimity_rule_version,
                correction.production_strategy_map_version,
                tuple(
                    GuidedProductionPerRoiStrategy(
                        roi_id=entry.roi_id,
                        strategy_family=entry.strategy_family,
                        dynamic_fit_mode=entry.dynamic_fit_mode,
                        selected_strategy=entry.selected_strategy,
                        evidence_source_type=entry.evidence_source_type,
                        evidence_reference_json=entry.evidence_reference_json,
                        explicit_user_mark=entry.explicit_user_mark,
                        current_or_stale=entry.current_or_stale,
                    )
                    for entry in correction.per_roi_production_strategy_map
                ),
                correction.applied_dff_orchestration_enabled,
            ),
            diagnostic_evidence=GuidedProductionDiagnosticEvidence(
                request.diagnostic_evidence.cache_id, request.diagnostic_evidence.cache_root_canonical,
                request.diagnostic_evidence.source_setup_signature,
                request.diagnostic_evidence.diagnostic_scope_signature,
                request.diagnostic_evidence.build_request_signature,
                request.diagnostic_evidence.artifact_contract_version,
                request.diagnostic_evidence.provenance_schema_version,
                request.diagnostic_evidence.artifact_semantic_digest,
                request.diagnostic_evidence.provenance_semantic_digest,
                tuple(GuidedProductionEvidenceReference(**{item.name: getattr(ref, item.name) for item in fields(GuidedBackendEvidenceReference)}) for ref in request.diagnostic_evidence.evidence_references),
                request.diagnostic_evidence.completed_run_rejection_category,
                request.diagnostic_evidence.resolver_status,
                request.diagnostic_evidence.preliminary_cache,
                request.diagnostic_evidence.production_analysis,
            ),
            feature_event=GuidedProductionFeatureEvent(
                request.feature_event.profile_schema_version, request.feature_event.profile_id,
                _typed(request.feature_event.effective_values), request.feature_event.active_fields,
                request.feature_event.inactive_fields, request.feature_event.profile_status,
                request.feature_event.explicitly_applied, request.feature_event.current,
                request.feature_event.visible_unapplied_changes,
                request.feature_event.per_roi_feature_event_map_version,
                tuple(
                    GuidedProductionPerRoiFeatureEvent(
                        roi_id=entry.roi_id,
                        source=entry.source,
                        feature_event_profile_id=entry.feature_event_profile_id,
                        override_config_fields=_typed(entry.override_config_fields),
                        effective_config_fields=_typed(entry.effective_config_fields),
                        explicit_user_mark=entry.explicit_user_mark,
                        current_or_stale=entry.current_or_stale,
                    )
                    for entry in request.feature_event.per_roi_feature_event_map
                ),
            ),
            output_policy=GuidedProductionOutputPolicy(
                output.output_base_canonical, output.output_base_path_style, output.path_role,
                output.future_output_owner, output.run_directory_strategy, output.creation_timing,
                output.overwrite, output.precreate, output.safety_classifier_version,
                tuple(GuidedProductionOutputRelationship(item.relationship, item.root_kind, item.status) for item in output.relationships),
                output.protected_root_context_complete, output.filesystem_fact_scope,
            ),
            execution_profile=GuidedProductionExecutionProfile(
                execution_mode=request.acquisition_dataset.execution_mode
            ),
            provenance_requirements=GuidedProductionProvenanceRequirements(
                recomputed, mapping_contract.mapping_contract_version,
                mapping_contract.config_mapping_contract_version,
                mapping_contract.runner_contract_version,
                request.validation_contract_version, request.validator_capability_version,
                request.source.source_candidate_set_digest,
                request.source.source_candidate_content_digest,
                request.parser.parser_contract_digest, request.diagnostic_evidence.cache_id,
                request.feature_event.profile_id,
            ),
            deferred_capabilities=tuple(
                item for item in local.deferred_capabilities if item != "app_build_identity"
            ),
            normalized_recording_description_identity=(
                request.normalized_recording_description_identity
            ),
            canonical_intent_identity="0" * 64,
        )
        digest = compute_guided_production_execution_intent_identity(intent)
        intent = replace(intent, canonical_intent_identity=digest)
        return GuidedProductionMappingSuccess(intent, digest, recomputed)
    except Exception:
        return _failure("mapping_internal_error", "mapping", "Production intent mapping failed.", "intent_construction_failed")


def _npm_json_value(value: Any) -> Any:
    """Convert accepted immutable/mapping values into canonical JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NPM intent JSON cannot contain non-finite floats.")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("NPM intent JSON object keys must be strings.")
        return {key: _npm_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_npm_json_value(item) for item in value]
    raise ValueError("NPM intent JSON encountered an unsupported value type.")


def _npm_canonical_json(value: Any) -> str:
    return json.dumps(
        _npm_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _npm_digest(label: str, value: Any) -> str:
    return hashlib.sha256(
        label.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(_npm_canonical_value(value))
    ).hexdigest()


def _npm_build_identity_is_usable(
    application_build_identity: Any,
) -> bool:
    if not isinstance(application_build_identity, ApplicationBuildIdentity):
        return False
    try:
        expected_build = hashlib.sha256(
            encode_canonical_value(_build_identity_payload(application_build_identity))
        ).hexdigest()
    except Exception:
        return False
    return (
        expected_build == application_build_identity.canonical_identity
        and _usable_version(application_build_identity.distribution_name)
        and _usable_version(application_build_identity.distribution_version)
    )


def _npm_output_policy(
    output: Any,
) -> GuidedProductionOutputPolicy:
    return GuidedProductionOutputPolicy(
        output.output_base_canonical,
        output.output_base_path_style,
        output.path_role,
        output.future_output_owner,
        output.run_directory_strategy,
        output.creation_timing,
        output.overwrite,
        output.precreate,
        output.safety_classifier_version,
        tuple(
            GuidedProductionOutputRelationship(
                item.relationship, item.root_kind, item.status
            )
            for item in output.relationships
        ),
        output.protected_root_context_complete,
        output.filesystem_fact_scope,
    )


def map_guided_npm_validation_outcome_to_execution_intent(
    outcome: "GuidedBackendValidationWorkflowOutcome",
    *,
    expected_validation_revision: int,
    expected_plan_identity: str,
    application_build_identity: ApplicationBuildIdentity,
    mapping_contract: GuidedProductionMappingContract,
) -> GuidedNpmProductionMappingResult:
    """Map only the actual accepted NPM validation workflow outcome.

    This is the supported B2-C1 production boundary.  It verifies the
    immutable workflow linkage before delegating to the pure request mapper;
    callers cannot assert acceptance by supplying a status string, inventing
    a revision, or substituting the request identity for the Guided plan
    identity.
    """
    workflow_module = __import__(
        "photometry_pipeline.guided_backend_validation_workflow",
        fromlist=["GuidedBackendValidationWorkflowOutcome"],
    )
    workflow_outcome_type = getattr(
        workflow_module, "GuidedBackendValidationWorkflowOutcome", None
    )
    if workflow_outcome_type is None or not isinstance(outcome, workflow_outcome_type):
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "An accepted Guided validation outcome is required.",
            "validation_outcome_type_invalid",
        )
    if (
        outcome.status != "validator_accepted"
        or outcome.accepted_for_backend_validation is not True
        or outcome.validation_result is None
        or outcome.validation_result.accepted is not True
    ):
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The Guided validation outcome was not accepted.",
            "validation_outcome_not_accepted",
        )
    if outcome.stale is not False:
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The Guided validation outcome is stale.",
            "validation_outcome_stale",
        )
    if outcome.blocking_issues:
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The accepted Guided validation outcome contains blockers.",
            "validation_outcome_has_blockers",
        )
    if not isinstance(outcome.compile_result, GuidedBackendValidationCompileSuccess):
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The accepted Guided validation outcome has no compile result.",
            "validation_compile_result_missing",
        )
    request = outcome.compile_result.request
    if not isinstance(request, GuidedBackendValidationRequest):
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The accepted Guided validation outcome has no request.",
            "validation_request_missing",
        )
    materialization_module = __import__(
        "photometry_pipeline.guided_backend_validation_materialization",
        fromlist=["GuidedBackendValidationMaterializationSuccess"],
    )
    materialization_success_type = getattr(
        materialization_module,
        "GuidedBackendValidationMaterializationSuccess",
        None,
    )
    if materialization_success_type is None or not isinstance(
        outcome.materialization_result, materialization_success_type
    ):
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The accepted Guided validation outcome has no materialization result.",
            "validation_materialization_result_missing",
        )

    accepted_request_identity = outcome.accepted_request_identity
    if not _sha256(accepted_request_identity):
        return _failure(
            "stale_or_mismatched_validation",
            "identity",
            "The accepted request identity is missing.",
            "accepted_request_identity_missing",
        )
    if (
        outcome.request_identity != accepted_request_identity
        or outcome.compile_result.canonical_request_identity
        != accepted_request_identity
    ):
        return _failure(
            "stale_or_mismatched_validation",
            "identity",
            "The accepted request identity is inconsistent with the outcome.",
            "accepted_request_identity_mismatch",
        )
    try:
        recomputed_request_identity = (
            compute_guided_backend_validation_request_identity(request)
        )
    except Exception:
        return _failure(
            "stale_or_mismatched_validation",
            "identity",
            "The accepted request identity could not be recomputed.",
            "accepted_request_identity_mismatch",
        )
    if recomputed_request_identity != accepted_request_identity:
        return _failure(
            "stale_or_mismatched_validation",
            "identity",
            "The accepted request identity does not match request content.",
            "accepted_request_identity_mismatch",
        )

    if expected_validation_revision is None:
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The expected validation revision is missing.",
            "validation_revision_missing",
        )
    if (
        isinstance(expected_validation_revision, bool)
        or not isinstance(expected_validation_revision, int)
        or expected_validation_revision < 0
        or outcome.validation_revision is None
        or isinstance(outcome.validation_revision, bool)
        or not isinstance(outcome.validation_revision, int)
        or outcome.validation_revision < 0
    ):
        detail_code = (
            "validation_revision_missing"
            if outcome.validation_revision is None
            else "validation_revision_mismatch"
        )
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The validation revision is missing or invalid.",
            detail_code,
        )
    if outcome.validation_revision != expected_validation_revision:
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The validation revision does not match the current revision.",
            "validation_revision_mismatch",
        )

    if expected_plan_identity is None:
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The expected Guided plan identity is missing.",
            "guided_plan_identity_missing",
        )
    if not _sha256(expected_plan_identity):
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The expected Guided plan identity is invalid.",
            "guided_plan_identity_mismatch",
        )
    if expected_plan_identity == accepted_request_identity:
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The request identity cannot stand in for the Guided plan identity.",
            "request_identity_used_as_plan_identity",
        )
    if not _sha256(outcome.guided_plan_identity):
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The accepted Guided plan identity is missing.",
            "guided_plan_identity_missing",
        )
    if outcome.guided_plan_identity == accepted_request_identity:
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The accepted outcome used request identity as its plan identity.",
            "request_identity_used_as_plan_identity",
        )
    if outcome.guided_plan_identity != expected_plan_identity:
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The Guided plan identity does not match the current plan.",
            "guided_plan_identity_mismatch",
        )

    return _map_verified_guided_npm_request_to_execution_intent(
        request,
        accepted_request_identity=accepted_request_identity,
        validation_revision=outcome.validation_revision,
        current_plan_identity=outcome.guided_plan_identity,
        application_build_identity=application_build_identity,
        mapping_contract=mapping_contract,
    )


def _map_verified_guided_npm_request_to_execution_intent(
    request: GuidedBackendValidationRequest,
    *,
    accepted_request_identity: str,
    validation_revision: int,
    current_plan_identity: str,
    application_build_identity: ApplicationBuildIdentity,
    mapping_contract: GuidedProductionMappingContract,
) -> GuidedNpmProductionMappingResult:
    """Map an already verified accepted NPM request to a non-runnable intent.

    The function consumes only already-materialized request content.  It does
    not discover files, inspect source data, recalculate support geometry, or
    allocate outputs.  Startup and wrapper execution are deliberately absent
    from this path and are represented only by the capability status. This is
    an internal helper; the public production boundary is
    ``map_guided_npm_validation_outcome_to_execution_intent``.
    """
    if not isinstance(request, GuidedBackendValidationRequest):
        return _failure(
            "request_missing_or_invalid",
            "request",
            "A Guided validation request is required.",
            "request_invalid_type",
        )
    if not _sha256(accepted_request_identity):
        return _failure(
            "stale_or_mismatched_validation",
            "identity",
            "The accepted request identity is missing or invalid.",
            "accepted_request_identity_missing",
        )
    try:
        recomputed = compute_guided_backend_validation_request_identity(request)
    except Exception:
        return _failure(
            "mapping_internal_error",
            "identity",
            "Request identity could not be recomputed.",
            "request_identity_recomputation_failed",
        )
    if recomputed != accepted_request_identity:
        return _failure(
            "stale_or_mismatched_validation",
            "identity",
            "The accepted request identity does not match request content.",
            "accepted_request_identity_mismatch",
        )
    if not _sha256(current_plan_identity):
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The current Guided plan identity is missing or invalid.",
            "current_plan_identity_invalid",
        )
    if (
        isinstance(validation_revision, bool)
        or not isinstance(validation_revision, int)
        or validation_revision < 0
    ):
        return _failure(
            "stale_or_mismatched_validation",
            "validation",
            "The NPM validation revision is invalid.",
            "validation_revision_invalid",
        )
    problem = _contract_problem(mapping_contract)
    if problem:
        return _failure(
            problem[0],
            "mapping_contract",
            "Production mapping contract is unavailable.",
            problem[1],
        )
    if not _npm_build_identity_is_usable(application_build_identity):
        return _failure(
            "app_build_identity_unusable",
            "build_identity",
            "Application build identity is unusable.",
            "build_identity_unusable",
        )
    if (
        request.request_schema_name != mapping_contract.supported_request_schema_name
        or request.request_schema_version
        != mapping_contract.supported_request_schema_version
    ):
        return _failure(
            "unsupported_request_schema",
            "request",
            "Request schema is unsupported.",
            "request_schema_mismatch",
        )
    if (
        request.validation_scope != mapping_contract.supported_validation_scope
        or request.validation_contract_version
        != mapping_contract.supported_validation_contract_version
    ):
        return _failure(
            "unsupported_validation_scope",
            "request",
            "Validation scope is unsupported.",
            "validation_scope_mismatch",
        )
    if (
        request.subset_rule_version != mapping_contract.supported_subset_rule_version
        or request.compiler_version != mapping_contract.supported_compiler_version
        or request.canonicalization_algorithm_version
        != mapping_contract.supported_canonicalization_algorithm_version
    ):
        return _failure(
            "unsupported_subset_rule",
            "request",
            "Request subset or compiler contract is unsupported.",
            "request_contract_mismatch",
        )
    if request.source.source_format != "npm":
        return _failure(
            "unsupported_source_format",
            "source",
            "Only NPM source is supported by the NPM production mapper.",
            "source_not_npm",
        )
    if not request.source.candidate_files or request.source.unresolved_source_identity_inputs:
        return _failure(
            "stale_or_mismatched_validation",
            "source",
            "The NPM source snapshot identity is incomplete.",
            "source_snapshot_unresolved",
        )
    if not isinstance(
        request.acquisition_dataset, GuidedBackendNpmAcquisitionDatasetRequest
    ):
        return _failure(
            "unsupported_analysis_configuration",
            "acquisition",
            "The NPM acquisition contract is unavailable.",
            "npm_acquisition_variant_missing",
        )
    acquisition = request.acquisition_dataset
    if (
        acquisition.source_format != "npm"
        or acquisition.acquisition_mode != "intermittent"
        or acquisition.allow_partial_final_window
        or acquisition.execution_mode not in {"phasic", "tonic", "both"}
    ):
        return _failure(
            "unsupported_analysis_configuration",
            "acquisition",
            "The NPM acquisition configuration is outside the mapped subset.",
            "npm_acquisition_unsupported",
        )
    if not isinstance(request.parser, GuidedBackendNpmParserRequest):
        return _failure(
            "stale_or_mismatched_validation",
            "parser",
            "The NPM parser policy is unavailable.",
            "npm_parser_variant_missing",
        )
    parser = request.parser
    if parser.unresolved_inputs:
        return _failure(
            "stale_or_mismatched_validation",
            "parser",
            "The NPM parser policy is unresolved.",
            "npm_parser_unresolved",
        )
    if not _sha256(request.normalized_recording_description_identity):
        return _failure(
            "stale_or_mismatched_validation",
            "normalized_recording",
            "The normalized recording identity is missing or invalid.",
            "normalized_recording_identity_invalid",
        )
    if not isinstance(request.normalized_recording_description, Mapping):
        return _failure(
            "stale_or_mismatched_validation",
            "normalized_recording",
            "The normalized recording description is missing.",
            "normalized_recording_payload_missing",
        )
    try:
        normalized = deserialize_normalized_recording_description(
            request.normalized_recording_description
        )
    except NormalizedRecordingError as exc:
        if exc.category in {
            "npm_per_session_evidence_not_identity_bound",
            "serialized_npm_session_evidence_mismatch",
        }:
            return _failure(
                "per_session_evidence_not_identity_bound",
                "normalized_recording",
                "NPM per-session evidence is not identity-bound.",
                exc.category,
            )
        return _failure(
            "stale_or_mismatched_validation",
            "normalized_recording",
            "The normalized recording description is stale or mismatched.",
            exc.category,
        )
    if (
        normalized.adapter_format != "npm"
        or normalized.acquisition_mode != acquisition.acquisition_mode
        or normalized.recording_source_identity != request.source.source_root_canonical
        or normalized.source_evidence_identity
        != request.source.source_candidate_content_digest
        or normalized.sampling.parser_contract_identity != parser.parser_contract_digest
        or normalized.sampling.target_fs_hz is None
        or float(normalized.sampling.target_fs_hz) != float(acquisition.npm_target_fs_hz)
        or normalized.sampling.sessions_per_hour != acquisition.sessions_per_hour
        or float(normalized.sampling.session_duration_sec or 0.0)
        != float(acquisition.session_duration_sec)
        or normalized.adapter_evidence.get("npm_source_candidate_set_digest")
        != request.source.source_candidate_set_digest
        or normalized.adapter_evidence.get("npm_source_candidate_content_digest")
        != request.source.source_candidate_content_digest
    ):
        return _failure(
            "stale_or_mismatched_validation",
            "normalized_recording",
            "NPM normalized recording facts do not match the accepted request.",
            "normalized_recording_request_mismatch",
        )
    try:
        parser_digest = compute_npm_parser_contract_digest(parser.parser_contract_content)
    except Exception:
        return _failure(
            "stale_or_mismatched_validation",
            "parser",
            "The NPM parser policy identity could not be verified.",
            "parser_policy_identity_unavailable",
        )
    if parser_digest != parser.parser_contract_digest:
        return _failure(
            "stale_or_mismatched_validation",
            "parser",
            "The NPM parser policy identity does not match its content.",
            "parser_policy_identity_mismatch",
        )
    if any(
        session.disposition != "process" for session in normalized.sessions
    ):
        return _failure(
            "unsupported_analysis_configuration",
            "acquisition",
            "The NPM production subset requires processable sessions only.",
            "npm_session_disposition_unsupported",
        )
    if not request.roi_scope.included_roi_ids:
        return _failure(
            "incomplete_correction_settings",
            "roi_scope",
            "The selected NPM ROI scope is empty.",
            "selected_roi_scope_empty",
        )
    if (
        tuple(item.roi_id for item in normalized.roi_channels if item.included)
        != tuple(request.roi_scope.included_roi_ids)
    ):
        return _failure(
            "stale_or_mismatched_validation",
            "roi_scope",
            "The NPM normalized ROI scope is stale.",
            "normalized_roi_scope_mismatch",
        )
    local = request.local_contract
    if (
        local.blocking_issue_categories
        or local.warning_categories
        or local.unsupported_state_flags
        or local.unresolved_required_inputs
    ):
        return _failure(
            "stale_or_mismatched_validation",
            "local_contract",
            "The accepted NPM local contract is not clean.",
            "local_contract_not_clean",
        )
    requested_deferred = set(local.deferred_capabilities)
    if requested_deferred & set(mapping_contract.blocking_deferred_capabilities):
        return _failure(
            "stale_or_mismatched_validation",
            "local_contract",
            "A production-mapping blocker remains deferred.",
            "blocking_deferred_capability",
        )
    known_deferred = (
        set(mapping_contract.allowed_nonblocking_deferred_capabilities)
        | set(mapping_contract.stage_deferred_capabilities)
        | {"app_build_identity"}
    )
    if requested_deferred - known_deferred:
        return _failure(
            "stale_or_mismatched_validation",
            "local_contract",
            "An unknown deferred capability remains on the NPM request.",
            "unknown_deferred_capability",
        )
    if _unknown_typed(
        acquisition.semantic_values, NPM_ACQUISITION_TYPED_FIELD_CONFIG_MAP
    ) or _unknown_typed(
        request.correction.dynamic_fit_parameter_values,
        CORRECTION_TYPED_FIELD_CONFIG_MAP,
    ):
        return _failure(
            "unsupported_analysis_configuration",
            "typed_values",
            "An NPM production configuration field is not mapped.",
            "typed_field_unmapped",
        )
    correction = request.correction
    raw_entries = tuple(correction.per_roi_production_strategy_map)
    if not raw_entries:
        return _failure(
            "incomplete_correction_settings",
            "correction",
            "NPM correction settings do not contain a per-ROI strategy map.",
            "per_roi_strategy_map_missing",
        )
    selected = tuple(request.roi_scope.included_roi_ids)
    selected_set = set(selected)
    if (
        len({entry.roi_id for entry in raw_entries}) != len(raw_entries)
        or {entry.roi_id for entry in raw_entries} != selected_set
    ):
        return _failure(
            "incomplete_correction_settings",
            "correction",
            "NPM correction settings must contain exactly one strategy per selected ROI.",
            "per_roi_strategy_map_incomplete",
        )
    production_strategy_map: list[GuidedProductionPerRoiStrategy] = []
    for entry in sorted(raw_entries, key=lambda item: item.roi_id):
        if (
            not entry.explicit_user_mark
            or entry.current_or_stale != "current"
            or not entry.evidence_source_type
            or not entry.evidence_reference_json
        ):
            return _failure(
                "incomplete_correction_settings",
                "correction",
                "NPM correction evidence is incomplete or stale.",
                "per_roi_strategy_evidence_incomplete",
            )
        if entry.strategy_family not in CORRECTION_STRATEGY_FAMILIES:
            return _failure(
                "unsupported_analysis_configuration",
                "correction",
                "An NPM ROI correction strategy is unsupported.",
                "per_roi_strategy_family_unsupported",
            )
        if entry.strategy_family == "signal_only_f0":
            if entry.selected_strategy != "signal_only_f0" or entry.dynamic_fit_mode is not None:
                return _failure(
                    "unsupported_analysis_configuration",
                    "correction",
                    "The NPM Signal-Only strategy entry is inconsistent.",
                    "signal_only_strategy_inconsistent",
                )
        elif (
            entry.selected_strategy not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES
            or entry.dynamic_fit_mode != entry.selected_strategy
        ):
            return _failure(
                "unsupported_analysis_configuration",
                "correction",
                "The NPM dynamic-fit strategy entry is unsupported.",
                "dynamic_fit_strategy_inconsistent",
            )
        production_strategy_map.append(
            GuidedProductionPerRoiStrategy(
                roi_id=entry.roi_id,
                strategy_family=entry.strategy_family,
                dynamic_fit_mode=entry.dynamic_fit_mode,
                selected_strategy=entry.selected_strategy,
                evidence_source_type=entry.evidence_source_type,
                evidence_reference_json=entry.evidence_reference_json,
                explicit_user_mark=entry.explicit_user_mark,
                current_or_stale=entry.current_or_stale,
            )
        )
    if _unknown_typed(
        request.feature_event.effective_values, FEATURE_EVENT_TYPED_FIELD_CONFIG_MAP
    ):
        return _failure(
            "incomplete_feature_settings",
            "feature_event",
            "NPM feature/event settings contain an unmapped field.",
            "feature_typed_field_unmapped",
        )
    raw_feature_entries = tuple(request.feature_event.per_roi_feature_event_map)
    if (
        not request.feature_event.effective_values
        or len({entry.roi_id for entry in raw_feature_entries}) != len(raw_feature_entries)
        or {entry.roi_id for entry in raw_feature_entries} != selected_set
    ):
        return _failure(
            "incomplete_feature_settings",
            "feature_event",
            "NPM feature/event settings must resolve one complete entry per selected ROI.",
            "per_roi_feature_map_incomplete",
        )
    for entry in raw_feature_entries:
        if (
            not feature_entry_provenance_valid(
                entry_source=entry.source,
                entry_feature_event_profile_id=entry.feature_event_profile_id,
                entry_explicit_user_mark=entry.explicit_user_mark,
                enclosing_profile_status=request.feature_event.profile_status,
                enclosing_profile_id=request.feature_event.profile_id,
                enclosing_current=request.feature_event.current,
                enclosing_visible_unapplied_changes=request.feature_event.visible_unapplied_changes,
            )
            or entry.current_or_stale != "current"
            or not entry.effective_config_fields
            or _unknown_typed(entry.effective_config_fields, FEATURE_EVENT_TYPED_FIELD_CONFIG_MAP)
            or _unknown_typed(entry.override_config_fields, FEATURE_EVENT_TYPED_FIELD_CONFIG_MAP)
        ):
            return _failure(
                "incomplete_feature_settings",
                "feature_event",
                "NPM per-ROI feature/event settings are incomplete or stale.",
                "per_roi_feature_entry_incomplete",
            )
    feature_event = GuidedProductionFeatureEvent(
        request.feature_event.profile_schema_version,
        request.feature_event.profile_id,
        _typed(request.feature_event.effective_values),
        request.feature_event.active_fields,
        request.feature_event.inactive_fields,
        request.feature_event.profile_status,
        request.feature_event.explicitly_applied,
        request.feature_event.current,
        request.feature_event.visible_unapplied_changes,
        request.feature_event.per_roi_feature_event_map_version,
        tuple(
            GuidedProductionPerRoiFeatureEvent(
                roi_id=entry.roi_id,
                source=entry.source,
                feature_event_profile_id=entry.feature_event_profile_id,
                override_config_fields=_typed(entry.override_config_fields),
                effective_config_fields=_typed(entry.effective_config_fields),
                explicit_user_mark=entry.explicit_user_mark,
                current_or_stale=entry.current_or_stale,
            )
            for entry in sorted(raw_feature_entries, key=lambda item: item.roi_id)
        ),
    )
    output = request.output
    if (
        output.path_role != "output_base"
        or output.future_output_owner != "runner"
        or output.run_directory_strategy != "derive_unique_run_id_under_output_base"
        or output.creation_timing != "future_execution_start_only"
        or output.overwrite
        or output.precreate
        or output.blocker_categories
        or not output.protected_root_context_complete
    ):
        return _failure(
            "output_policy_not_supported",
            "output",
            "NPM output policy is not a future-run output policy.",
            "output_policy_invalid",
        )
    try:
        normalized_payload = serialize_normalized_recording_description(normalized)
        normalized_payload_json = _npm_canonical_json(normalized_payload)
        parser_policy_content_json = _npm_canonical_json(parser.parser_contract_content)
        session_projection = build_normalized_recording_description_payload(normalized)[
            "npm_per_session_resolved_evidence"
        ]
        mapping_projection = normalized.adapter_evidence[
            "physical_to_canonical_roi_mapping"
        ]
        ordered_session_identity = _npm_digest(
            "npm-ordered-session-identity:v1",
            [
                {
                    "chronological_position": session.chronological_position,
                    "stable_source_identity": session.stable_source_identity,
                    "authoritative_source_start_time": session.authoritative_source_start_time,
                    "content_digest": session.content_digest,
                }
                for session in normalized.sessions
            ],
        )
        per_session_evidence_identity = _npm_digest(
            "npm-per-session-resolved-evidence:v1", session_projection
        )
        mapping_identity = _npm_digest(
            "npm-physical-to-canonical-roi-mapping:v1", mapping_projection
        )
        parser_sampling = parser.parser_contract_content["sampling"]
        support_policy = parser_sampling["support_policy"]
        for item in session_projection:
            if item["support_policy"] != support_policy:
                return _failure(
                    "per_session_evidence_not_identity_bound",
                    "normalized_recording",
                    "NPM session support policy differs from the recording-wide policy.",
                    "session_support_policy_mismatch",
                )
            if item["support_policy_identity"] != compute_npm_support_policy_identity(
                item["support_policy"]
            ):
                return _failure(
                    "per_session_evidence_not_identity_bound",
                    "normalized_recording",
                    "NPM session support policy identity does not match its value.",
                    "session_support_policy_identity_mismatch",
                )
        support_identity = compute_npm_support_policy_identity(support_policy)
        output_time_basis_identity = _npm_digest(
            "npm-output-time-basis:v1",
            [item["output_time_basis"] for item in session_projection],
        )
        source_snapshot_identity = _npm_digest(
            "npm-source-snapshot:v1",
            {
                "source_root_canonical": request.source.source_root_canonical,
                "source_candidate_set_digest": request.source.source_candidate_set_digest,
                "source_candidate_content_digest": request.source.source_candidate_content_digest,
                "candidate_files": [
                    {
                        "canonical_relative_path": item.canonical_relative_path,
                        "size_bytes": item.size_bytes,
                        "sha256_content_digest": item.sha256_content_digest,
                    }
                    for item in request.source.candidate_files
                ],
            },
        )
        correction_payload_identity = _npm_digest(
            "npm-correction-payload:v1",
            {
                "parameters": _typed(correction.dynamic_fit_parameter_values),
                "per_roi_strategy_map": tuple(production_strategy_map),
            },
        )
        feature_payload_identity = _npm_digest(
            "npm-feature-payload:v1", feature_event
        )
        intent = GuidedNpmProductionExecutionIntent(
            intent_schema_name=GUIDED_NPM_PRODUCTION_INTENT_SCHEMA_NAME,
            intent_schema_version=GUIDED_NPM_PRODUCTION_INTENT_SCHEMA_VERSION,
            mapping_contract_version=mapping_contract.mapping_contract_version,
            runner_contract_version=GUIDED_NPM_PRODUCTION_RUNNER_CONTRACT_VERSION,
            source_request_identity=recomputed,
            validation_status="validator_accepted",
            validation_revision=validation_revision,
            current_plan_identity=current_plan_identity,
            application_build_identity=application_build_identity,
            source_format=request.source.source_format,
            source_root_canonical=request.source.source_root_canonical,
            acquisition_mode=acquisition.acquisition_mode,
            source_candidate_files=tuple(
                GuidedProductionSourceCandidate(
                    item.canonical_relative_path,
                    item.size_bytes,
                    item.sha256_content_digest,
                )
                for item in request.source.candidate_files
            ),
            source_snapshot_set_identity=request.source.source_candidate_set_digest,
            source_snapshot_content_identity=request.source.source_candidate_content_digest,
            source_snapshot_identity=source_snapshot_identity,
            normalized_recording_description_identity=(
                request.normalized_recording_description_identity
            ),
            normalized_recording_payload_json=normalized_payload_json,
            parser_policy_identity=parser.parser_contract_digest,
            parser_policy_content_json=parser_policy_content_json,
            ordered_session_identity=ordered_session_identity,
            per_session_resolved_evidence_identity=per_session_evidence_identity,
            physical_to_canonical_roi_mapping_identity=mapping_identity,
            support_policy_identity=support_identity,
            output_time_basis_identity=output_time_basis_identity,
            target_fs_hz=float(acquisition.npm_target_fs_hz),
            session_duration_sec=float(acquisition.session_duration_sec),
            sessions_per_hour=acquisition.sessions_per_hour,
            execution_mode=acquisition.execution_mode,
            run_type="full",
            discovered_roi_ids=tuple(request.roi_scope.discovered_roi_ids),
            selected_roi_ids=selected,
            excluded_roi_ids=tuple(request.roi_scope.excluded_roi_ids),
            correction_parameter_values=_typed(correction.dynamic_fit_parameter_values),
            per_roi_correction_strategy_map=tuple(production_strategy_map),
            correction_payload_identity=correction_payload_identity,
            feature_event=feature_event,
            feature_payload_identity=feature_payload_identity,
            output_policy=_npm_output_policy(output),
            deferred_capabilities=tuple(
                sorted(
                    {
                        item
                        for item in local.deferred_capabilities
                        if item not in {"app_build_identity", "backend_validation"}
                    }
                    | {"npm_startup_orchestration"}
                )
            ),
            capability_status=GUIDED_NPM_PRODUCTION_CAPABILITY_STATUS,
            canonical_intent_identity="0" * 64,
        )
        digest = compute_guided_npm_production_execution_intent_identity(intent)
        intent = replace(intent, canonical_intent_identity=digest)
        capability = GuidedNpmProductionCapabilityResult(
            status=GUIDED_NPM_PRODUCTION_CAPABILITY_STATUS,
            production_mapping_supported=True,
            startup_available=False,
            runnable=False,
        )
        return GuidedNpmProductionMappingSuccess(
            intent=intent,
            capability=capability,
            canonical_intent_identity=digest,
            source_request_identity=recomputed,
        )
    except (NormalizedRecordingError, GuidedIdentityError, TypeError, ValueError, KeyError):
        return _failure(
            "mapping_internal_error",
            "mapping",
            "NPM production intent mapping failed.",
            "npm_intent_construction_failed",
        )

def build_per_roi_feature_event_backend_shapes(
    intent: GuidedProductionExecutionIntent,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Reshape a mapped intent's per-ROI feature/event map into plain-dict
    shapes for 32b's backend (Pipeline, applied-dF/F orchestration).

    This function is pure and does not execute anything: it does not
    construct Pipeline, call feature extraction, or invoke applied-dF/F
    orchestration. No caller in this codebase constructs Pipeline (or any
    applied-dF/F orchestration) from a GuidedProductionExecutionIntent today
    -- the same is true for correction strategy's
    per_roi_production_strategy_map, which also stops at this mapping layer.
    Building that execution call here would invent an execution route this
    codebase does not otherwise have, so this function stops at producing
    the ready-to-use data a future runner would need.

    Returns a dict with three keys:
    - "per_roi_override_config_fields": {roi_id: sparse override fields}
      for source="override" ROIs only. A future runner builds Pipeline's
      per_roi_feature_config from this via
      {roi: dataclasses.replace(base_config, **fields) for roi, fields in
      ...items()}, mirroring
      guided_new_analysis_plan.build_per_roi_feature_backend_config.
    - "per_roi_effective_feature_config_fields_for_overrides": {roi_id:
      COMPLETE effective fields} for source="override" ROIs only. Already
      complete (every FEATURE_EVENT_CONFIG_FIELDS name present) -- ready to
      pass directly to write_per_roi_feature_config_files (the current
      native per-ROI correction route) with no further merging. Never the
      sparse override_config_fields. (guided_applied_dff_orchestration.py's
      run_guided_applied_dff_orchestration_if_enabled has the same
      per_roi_feature_event_overrides argument shape, but that function is
      retired: it has no remaining caller anywhere in current-Guided
      production, including tools/run_full_pipeline_deliverables.py, whose
      former call site to it has been removed.)
    - "per_roi_feature_provenance": {roi_id: {"source", "feature_event_profile_id",
      "override_config_fields", "effective_config_fields"}} for every
      resolved ROI (both "default" and "override"). Ready to pass directly
      as Pipeline(per_roi_feature_provenance=...).
    """
    # Relaxed only as far as required: both intent types carry the
    # identical shared feature_event.per_roi_feature_event_map type this
    # function actually reads (GuidedProductionFeatureEvent /
    # GuidedProductionPerRoiFeatureEvent) -- nothing below reads any
    # RWD-only or NPM-only field.
    if not isinstance(
        intent, (GuidedProductionExecutionIntent, GuidedNpmProductionExecutionIntent)
    ):
        raise TypeError(
            "intent must be a GuidedProductionExecutionIntent or "
            "GuidedNpmProductionExecutionIntent"
        )

    override_config_fields_by_roi: dict[str, dict[str, Any]] = {}
    effective_config_fields_for_overrides_by_roi: dict[str, dict[str, Any]] = {}
    provenance_by_roi: dict[str, dict[str, Any]] = {}

    for entry in intent.feature_event.per_roi_feature_event_map:
        override_fields = {item.field_name: item.value for item in entry.override_config_fields}
        effective_fields = {item.field_name: item.value for item in entry.effective_config_fields}
        provenance_by_roi[entry.roi_id] = {
            "source": entry.source,
            "feature_event_profile_id": entry.feature_event_profile_id,
            "override_config_fields": override_fields,
            "effective_config_fields": effective_fields,
        }
        if entry.source == "override":
            override_config_fields_by_roi[entry.roi_id] = override_fields
            effective_config_fields_for_overrides_by_roi[entry.roi_id] = effective_fields

    return {
        "per_roi_override_config_fields": override_config_fields_by_roi,
        "per_roi_effective_feature_config_fields_for_overrides": (
            effective_config_fields_for_overrides_by_roi
        ),
        "per_roi_feature_provenance": provenance_by_roi,
    }


def guided_production_per_roi_strategy_to_correction_spec(
    entry: GuidedProductionPerRoiStrategy,
) -> PerRoiCorrectionSpec:
    """Fail-closed conversion from the one authorized production per-ROI
    strategy record (GuidedProductionPerRoiStrategy) into Pipeline's
    dispatch-facing PerRoiCorrectionSpec (photometry_pipeline.core.types).

    This is the sole conversion point between the Guided production layer's
    strategy representation and Pipeline's; nothing else in this codebase
    may build a second, independently authoritative interpretation of a
    GuidedProductionPerRoiStrategy.

    Field-by-field mapping, confirmed against build_guided_per_roi_production
    _strategy_map (guided_new_analysis_plan.py) and its GuidedProductionPerRoiStrategy
    construction site above:

    - roi_id: direct passthrough.
    - strategy_family: direct passthrough, but validated against
      CORRECTION_STRATEGY_FAMILIES here (fail-closed) because the draft-plan
      layer's own strategy_family can also be "unsupported" (an included ROI
      whose selected_strategy matched neither FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES
      nor "signal_only_f0") -- that state must never reach Pipeline dispatch.
    - selected_strategy / dynamic_fit_mode: both already hold the same
      internal dispatch identifier (e.g. "global_linear_regression"), not a
      scientist-facing label -- confirmed at the construction site, where
      `selected = str(choice.selected_strategy or "")` is used verbatim for
      both fields when strategy_family == "dynamic_fit". No alias/label
      translation happens at this layer, so none is performed here either;
      PerRoiCorrectionSpec.__post_init__ independently re-validates that
      dynamic_fit_mode is one of RESOLVED_DYNAMIC_FIT_MODES and that
      selected_strategy == dynamic_fit_mode, which is this adapter's actual
      fit-mode-mapping fail-closed check.
    - parameter_identity: left as "" (not mapped). GuidedProductionPerRoiStrategy
      carries no per-ROI parameter data -- dynamic-fit tunable parameters are
      a separate, run-wide field (GuidedProductionCorrection.dynamic_fit_parameter_values),
      not yet per-ROI in the authoritative production representation. Mapping
      a per-ROI field to a run-wide value would misrepresent it as ROI-specific;
      left unmapped rather than guessed.
    - evidence_identity: evidence_source_type and evidence_reference_json
      combined ("{source_type}::{reference_json}"), since either alone could
      collide across genuinely different evidence (e.g. the same JSON payload
      shape produced by two different evidence_source_type values).
    - explicit_user_mark / current_or_stale: not carried into PerRoiCorrectionSpec
      as fields (it has none for them); instead enforced as fail-closed
      preconditions below -- a non-explicit or stale entry must never reach
      Pipeline dispatch, so conversion refuses rather than silently accepting.

    Raises ValueError (fail closed) for a non-explicit mark, a stale entry,
    an unsupported strategy_family, or (via PerRoiCorrectionSpec's own
    validation) an inconsistent/invalid strategy/mode pairing.
    """
    if not entry.explicit_user_mark:
        raise ValueError(
            f"ROI {entry.roi_id!r} strategy is not an explicit user mark; "
            "refusing to convert a non-explicit selection into a production dispatch spec"
        )
    if entry.current_or_stale != "current":
        raise ValueError(
            f"ROI {entry.roi_id!r} strategy is {entry.current_or_stale!r}, not 'current'; "
            "refusing to convert a stale selection into a production dispatch spec"
        )
    if entry.strategy_family not in CORRECTION_STRATEGY_FAMILIES:
        raise ValueError(
            f"ROI {entry.roi_id!r} has unsupported strategy_family: {entry.strategy_family!r}"
        )
    evidence_identity = f"{entry.evidence_source_type}::{entry.evidence_reference_json}"
    return PerRoiCorrectionSpec(
        roi_id=entry.roi_id,
        strategy_family=entry.strategy_family,
        selected_strategy=entry.selected_strategy,
        dynamic_fit_mode=entry.dynamic_fit_mode,
        parameter_identity="",
        evidence_identity=evidence_identity,
    )


def guided_production_strategy_map_to_correction_specs(
    entries: tuple[GuidedProductionPerRoiStrategy, ...],
) -> dict[str, PerRoiCorrectionSpec]:
    """Convert a full production per-ROI strategy map into the
    {roi_id: PerRoiCorrectionSpec} shape Pipeline dispatch (regression.
    fit_chunk_dynamic's per_roi_correction argument) expects.

    Fails closed (ValueError) on any entry that fails
    guided_production_per_roi_strategy_to_correction_spec, or on a duplicate
    roi_id across entries.

    Unused by the GUI/Guided Run in this staging patch (see the module-level
    note on build_per_roi_feature_event_backend_shapes: no caller in this
    codebase constructs Pipeline from a GuidedProductionExecutionIntent yet).
    This function only prepares the ready-to-use, already-validated mapping
    a future runner would pass to Pipeline.
    """
    result: dict[str, PerRoiCorrectionSpec] = {}
    for entry in entries:
        spec = guided_production_per_roi_strategy_to_correction_spec(entry)
        if spec.roi_id in result:
            raise ValueError(f"duplicate roi_id in production strategy map: {spec.roi_id!r}")
        result[spec.roi_id] = spec
    return result
