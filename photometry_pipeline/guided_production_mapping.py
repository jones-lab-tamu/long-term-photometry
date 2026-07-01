"""Pure mapping from an identified Guided validation request to production intent."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import math
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
    GuidedBackendLocalContractState,
    GuidedBackendOutputRelationship,
    GuidedBackendOutputRequest,
    GuidedBackendRoiScopeRequest,
    GuidedBackendRwdParserRequest,
    GuidedBackendSourceCandidateFile,
    GuidedBackendSourceRequest,
    GuidedBackendTypedFieldValue,
    GuidedBackendValidationRequest,
    compute_guided_backend_validation_request_identity,
)
from photometry_pipeline.guided_identity import (
    CANONICALIZATION_ALGORITHM_VERSION,
    GuidedIdentityError,
    encode_canonical_value,
)
from photometry_pipeline.guided_new_analysis_plan import (
    FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
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
class GuidedProductionCorrection:
    strategy_scope: str
    global_correction_strategy: str
    global_dynamic_fit_mode: str
    dynamic_fit_parameter_values: tuple[GuidedProductionTypedValue, ...]
    confirmed_marks: tuple[GuidedProductionConfirmedMark, ...]
    mark_rule_version: str
    currentness_rule_version: str
    unanimity_rule_version: str


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
    canonical_intent_identity: str

    def __post_init__(self) -> None:
        if not _sha256(self.source_request_identity):
            raise ValueError("source_request_identity must be a lowercase SHA-256.")
        if not _sha256(self.canonical_intent_identity):
            raise ValueError("canonical_intent_identity must be a lowercase SHA-256.")
        _require_tuple(self.deferred_capabilities, "deferred_capabilities")
        if (
            self.execution_profile.execution_mode != "phasic"
            or self.execution_profile.run_type != "full"
            or self.execution_profile.traces_only is not False
            or self.execution_profile.allocate_output_at_future_run_start_only
            is not True
            or self.output_policy.overwrite is not False
        ):
            raise ValueError("Intent is outside the first production subset.")


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
    GuidedProductionCorrection: tuple(item.name for item in fields(GuidedProductionCorrection)),
    GuidedProductionEvidenceReference: tuple(item.name for item in fields(GuidedProductionEvidenceReference)),
    GuidedProductionDiagnosticEvidence: tuple(item.name for item in fields(GuidedProductionDiagnosticEvidence)),
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


REQUEST_FIELD_CLASSIFICATIONS = {
    GuidedBackendValidationRequest: {name: "mapped_to_intent" for name in (
        "request_schema_name", "request_schema_version", "validation_scope",
        "validation_contract_version", "validator_capability_version",
        "compiler_version", "subset_rule_version",
        "canonicalization_algorithm_version", "source", "acquisition_dataset",
        "parser", "roi_scope", "correction", "diagnostic_evidence",
        "feature_event", "output", "local_contract")},
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
    {"rwd_time_col", "uv_suffix", "sig_suffix", "target_fs_hz", "sessions_per_hour", "session_duration_sec"}
)
CORRECTION_TYPED_FIELD_CONFIG_MAP = frozenset(
    {
        "dynamic_fit_mode", "dynamic_fit_slope_constraint", "dynamic_fit_min_slope",
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
    if request.acquisition_dataset.acquisition_mode != "intermittent":
        return _failure("unsupported_acquisition_mode", "acquisition", "Only intermittent acquisition is supported.", "acquisition_not_intermittent")
    if request.acquisition_dataset.allow_partial_final_window or request.acquisition_dataset.exclude_incomplete_final_rwd_chunk:
        return _failure("incomplete_final_policy_not_supported", "acquisition", "Final-window policy is unsupported.", "final_policy_unsupported")
    if request.parser.unresolved_inputs:
        return _failure("unresolved_request_field", "parser", "Parser inputs are unresolved.", "parser_unresolved")
    if not request.roi_scope.included_roi_ids:
        return _failure("unresolved_request_field", "roi_scope", "Included ROI set is empty.", "included_rois_empty")
    correction = request.correction
    if correction.global_dynamic_fit_mode == "signal_only_f0":
        return _failure("signal_only_not_supported", "correction", "Signal-Only is unsupported.", "signal_only_mode")
    if correction.strategy_scope != "global" or correction.global_correction_strategy != "dynamic_fit" or correction.global_dynamic_fit_mode not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
        return _failure("unsupported_correction_strategy", "correction", "Correction strategy is unsupported.", "correction_unsupported")
    modes = {item.selected_dynamic_fit_mode for item in correction.confirmed_marks}
    if modes != {correction.global_dynamic_fit_mode}:
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
            ),
            output_policy=GuidedProductionOutputPolicy(
                output.output_base_canonical, output.output_base_path_style, output.path_role,
                output.future_output_owner, output.run_directory_strategy, output.creation_timing,
                output.overwrite, output.precreate, output.safety_classifier_version,
                tuple(GuidedProductionOutputRelationship(item.relationship, item.root_kind, item.status) for item in output.relationships),
                output.protected_root_context_complete, output.filesystem_fact_scope,
            ),
            execution_profile=GuidedProductionExecutionProfile(),
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
            canonical_intent_identity="0" * 64,
        )
        digest = compute_guided_production_execution_intent_identity(intent)
        intent = replace(intent, canonical_intent_identity=digest)
        return GuidedProductionMappingSuccess(intent, digest, recomputed)
    except Exception:
        return _failure("mapping_internal_error", "mapping", "Production intent mapping failed.", "intent_construction_failed")
