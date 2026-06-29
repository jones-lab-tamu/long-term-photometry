"""Non-executable new_analysis Guided draft plan model and pure validation helpers.

This module intentionally contains no GUI, RunSpec, pipeline, validation, feature
extraction, or output-writing imports. It defines only the data contract and pure
validation helpers for the new_analysis Guided draft plan state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from photometry_pipeline.config import Config
from photometry_pipeline.feature_event_config import (
    FEATURE_EVENT_CONFIG_FIELDS,
    FEATURE_EVENT_THRESHOLD_METHODS,
    validate_feature_event_config_fields,
)
from photometry_pipeline.workflow_safety import feature_event_defaults_from_config

SCHEMA_VERSION = "guided_new_analysis_plan.v1"
RUN_PREVIEW_SCHEMA_VERSION = "guided_new_analysis_run_preview.v1"
EXECUTION_SPEC_PREVIEW_SCHEMA_VERSION = "guided_new_analysis_execution_spec_preview.v1"
EXECUTION_SUBSET_SCHEMA_VERSION = "guided_new_analysis_execution_subset.v1"
DATASET_CONTRACT_SNAPSHOT_SCHEMA_VERSION = "guided_new_analysis_dataset_contract_snapshot.v1"
EXECUTION_INTENT_SCHEMA_VERSION = "guided_new_analysis_execution_intent.v1"
OUTPUT_CREATION_POLICY_SCHEMA_VERSION = "guided_new_analysis_output_creation_policy.v1"
DYNAMIC_FIT_PARAMETER_CONTRACT_SCHEMA_VERSION = "guided_new_analysis_dynamic_fit_parameter_contract.v1"
FIRST_EXECUTION_SUBSET_NAME = "global_dynamic_fit_only.v1"
SUPPORTED_INPUT_FORMATS = {"auto", "rwd", "npm", "custom_tabular"}
SUPPORTED_ACQUISITION_MODES = {"intermittent", "continuous"}
DATASET_CONTRACT_SNAPSHOT_STATUSES = {
    "missing",
    "inferred",
    "applied",
    "invalid",
    "stale",
    "unsupported",
}
RUNNABLE_CORRECTION_STRATEGIES = {
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
    "global_linear_regression",
    "signal_only_f0",
}
FORBIDDEN_CORRECTION_STRATEGIES = {"auto", "needs_review", "no_correction"}
FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES = {
    "global_linear_regression",
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
}
TIMELINE_ANCHOR_MODES = {"civil", "elapsed", "fixed_daily_anchor"}
EXECUTION_MODES = {"both", "phasic", "tonic"}
RUN_PROFILES = {"full", "tuning_prep"}
OUTPUT_PATH_ROLES = {"output_base"}
OUTPUT_CREATION_TIMINGS = {"future_execution_start_only"}
RUN_DIRECTORY_STRATEGIES = {"derive_unique_run_id_under_output_base"}
CONFIG_WRITE_TIMINGS = {"future_execution_or_validation_only"}
DYNAMIC_FIT_SLOPE_CONSTRAINTS = {"unconstrained", "nonnegative"}
ADAPTIVE_EVENT_GATE_FREEZE_INTERP_METHODS = {"linear_hold"}
FEATURE_EVENT_BACKEND_DEFAULT_SOURCE = "photometry_pipeline.config.Config"
FEATURE_EVENT_BACKEND_DEFAULT_PROVENANCE = (
    "backend Config default mechanically derived from "
    f"{FEATURE_EVENT_BACKEND_DEFAULT_SOURCE}"
)
FEATURE_EVENT_EFFECTIVE_VALUE_FIELDS = tuple(sorted(FEATURE_EVENT_CONFIG_FIELDS))
FEATURE_EVENT_THRESHOLD_PARAMETER_FIELDS = {
    "mean_std": "peak_threshold_k",
    "median_mad": "peak_threshold_k",
    "percentile": "peak_threshold_percentile",
    "absolute": "peak_threshold_abs",
}


def canonical_feature_event_backend_defaults() -> dict[str, Any]:
    """Return feature/event defaults from the canonical backend Config dataclass."""
    defaults = feature_event_defaults_from_config(Config())
    return {
        field_name: defaults[field_name]
        for field_name in FEATURE_EVENT_EFFECTIVE_VALUE_FIELDS
    }


_CANONICAL_FEATURE_EVENT_BACKEND_DEFAULTS = canonical_feature_event_backend_defaults()

BACKEND_DYNAMIC_FIT_DEFAULT_SOURCE = "photometry_pipeline.config.Config"
BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE = (
    "backend Config default mechanically derived from "
    f"{BACKEND_DYNAMIC_FIT_DEFAULT_SOURCE}"
)
BACKEND_DYNAMIC_FIT_DEFAULT_FIELD_MAP: dict[str, str] = {
    "dynamic_fit_mode": "dynamic_fit_mode",
    "slope_constraint": "dynamic_fit_slope_constraint",
    "min_slope": "dynamic_fit_min_slope",
    "robust_event_reject_max_iters": "robust_event_reject_max_iters",
    "robust_event_reject_residual_z_thresh": "robust_event_reject_residual_z_thresh",
    "robust_event_reject_local_var_window_sec": "robust_event_reject_local_var_window_sec",
    "robust_event_reject_local_var_ratio_thresh": "robust_event_reject_local_var_ratio_thresh",
    "robust_event_reject_min_keep_fraction": "robust_event_reject_min_keep_fraction",
    "adaptive_event_gate_residual_z_thresh": "adaptive_event_gate_residual_z_thresh",
    "adaptive_event_gate_local_var_window_sec": "adaptive_event_gate_local_var_window_sec",
    "adaptive_event_gate_local_var_ratio_thresh": "adaptive_event_gate_local_var_ratio_thresh",
    "adaptive_event_gate_smooth_window_sec": "adaptive_event_gate_smooth_window_sec",
    "adaptive_event_gate_min_trust_fraction": "adaptive_event_gate_min_trust_fraction",
    "adaptive_event_gate_freeze_interp_method": "adaptive_event_gate_freeze_interp_method",
    "window_sec": "window_sec",
    "step_sec": "step_sec",
    "r_low": "r_low",
    "r_high": "r_high",
    "g_min": "g_min",
    "min_samples_per_window": "min_samples_per_window",
    "min_valid_windows": "min_valid_windows",
}


def canonical_dynamic_fit_backend_defaults() -> dict[str, Any]:
    """Return dynamic-fit defaults from the canonical backend Config dataclass."""
    cfg = Config()
    return {
        contract_field: getattr(cfg, config_field)
        for contract_field, config_field in BACKEND_DYNAMIC_FIT_DEFAULT_FIELD_MAP.items()
    }


_CANONICAL_DYNAMIC_FIT_BACKEND_DEFAULTS = canonical_dynamic_fit_backend_defaults()


def _backend_dynamic_fit_default(field_name: str) -> Any:
    return _CANONICAL_DYNAMIC_FIT_BACKEND_DEFAULTS[field_name]


@dataclass(frozen=True)
class GuidedPlanIssue:
    category: str
    message: str
    severity: str  # "blocking", "warning", "info"


@dataclass(frozen=True)
class GuidedNewAnalysisSectionReadiness:
    key: str
    label: str
    status: str  # ready / missing / invalid / stale / warning / info / blocked
    blocking_issues: tuple[GuidedPlanIssue, ...] = ()
    warning_issues: tuple[GuidedPlanIssue, ...] = ()
    info_issues: tuple[GuidedPlanIssue, ...] = ()


@dataclass(frozen=True)
class GuidedNewAnalysisReadiness:
    sections: tuple[GuidedNewAnalysisSectionReadiness, ...]
    plan_complete_for_handoff: bool
    execution_available: bool
    execution_blocked_reason: str
    blocking_issues: tuple[GuidedPlanIssue, ...] = ()
    warning_issues: tuple[GuidedPlanIssue, ...] = ()
    info_issues: tuple[GuidedPlanIssue, ...] = ()


@dataclass(frozen=True)
class GuidedNewAnalysisRunPreviewIssue:
    category: str
    message: str
    severity: str  # "blocking", "warning", "info"


@dataclass(frozen=True)
class GuidedNewAnalysisRunPreview:
    preview_schema_version: str
    plan_schema_version: str
    source: dict[str, Any]
    acquisition: dict[str, Any]
    execution_intent: dict[str, Any]
    dataset_contract: dict[str, Any]
    roi_selection: dict[str, Any]
    diagnostic_cache: dict[str, Any]
    correction_strategy: dict[str, Any]
    evidence_references: dict[str, Any]
    feature_event: dict[str, Any]
    feature_event_consumption: dict[str, Any]
    output_policy: dict[str, Any]
    output_creation_policy: dict[str, Any]
    provenance: dict[str, Any]
    readiness_snapshot: dict[str, Any]
    unresolved_items: tuple[GuidedNewAnalysisRunPreviewIssue, ...] = ()
    warnings: tuple[GuidedNewAnalysisRunPreviewIssue, ...] = ()
    execution_available: bool = False
    execution_blocked_reason: str = "Final Guided Run/RunSpec is not implemented in this stage."


@dataclass(frozen=True)
class GuidedNewAnalysisExecutionSpecPreview:
    spec_preview_schema_version: str
    plan_schema_version: str
    subset_name: str
    spec_preview_available: bool
    first_subset_executable: bool
    execution_available: bool
    execution_blocked_reason: str
    backend_mapping_status: str
    source_acquisition: dict[str, Any]
    dataset_contract: dict[str, Any]
    roi: dict[str, Any]
    correction: dict[str, Any]
    execution_intent: dict[str, Any]
    feature_event: dict[str, Any]
    output: dict[str, Any]
    diagnostic_cache_provenance: dict[str, Any]
    provenance: dict[str, Any]
    blocked_reasons: tuple[str, ...] = ()
    blocking_issue_categories: tuple[str, ...] = ()
    field_classifications: tuple[GuidedNewAnalysisExecutionFieldClassification, ...] = ()
    warning_issue_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuidedNewAnalysisExecutionSubsetIssue:
    category: str
    message: str
    severity: str  # "blocking", "warning", "info"
    section: str = "execution_subset"


@dataclass(frozen=True)
class GuidedNewAnalysisExecutionFieldClassification:
    field_name: str
    status: str  # present / fixed_default / required_missing / deferred_full_control / not_relevant / selected / invalid / stale / unsupported
    value: Any | None = None
    provenance: str = ""
    blocks_subset: bool = False
    issue_category: str | None = None


@dataclass(frozen=True)
class GuidedNewAnalysisExecutionSubsetReadiness:
    subset_name: str
    subset_schema_version: str
    first_subset_executable: bool
    planning_complete_for_handoff: bool
    execution_available: bool
    execution_blocked_reason: str
    allowed_dynamic_fit_strategy: str | None = None
    blocking_issues: tuple[GuidedNewAnalysisExecutionSubsetIssue, ...] = ()
    warning_issues: tuple[GuidedNewAnalysisExecutionSubsetIssue, ...] = ()
    info_issues: tuple[GuidedNewAnalysisExecutionSubsetIssue, ...] = ()
    field_classifications: tuple[GuidedNewAnalysisExecutionFieldClassification, ...] = ()


@dataclass(frozen=True)
class GuidedNewAnalysisDatasetContractSourceIdentity:
    input_source_path: str | None = None
    resolved_input_source_path: str | None = None
    input_format: str | None = None
    resolved_input_format: str | None = None
    acquisition_mode: str | None = None
    sessions_per_hour: int | None = None
    session_duration_sec: float | None = None
    continuous_window_sec: float | None = None
    continuous_step_sec: float | None = None
    allow_partial_final_window: bool | None = None
    exclude_incomplete_final_rwd_chunk: bool | None = None
    discovered_roi_ids: tuple[str, ...] = ()
    included_roi_ids: tuple[str, ...] = ()
    source_setup_signature: str | None = None
    config_fingerprint: str | None = None
    diagnostic_cache_contract_identity: str | None = None


@dataclass(frozen=True)
class GuidedNewAnalysisExecutionIntent:
    schema_version: str = EXECUTION_INTENT_SCHEMA_VERSION
    timeline_anchor_mode: str = "civil"
    fixed_daily_anchor_clock: str | None = None
    execution_mode: str = "phasic"
    run_profile: str = "full"
    provenance: dict[str, Any] = field(default_factory=lambda: {
        "timeline_anchor_mode": "first_subset_fixed_default_matches_backend_default",
        "fixed_daily_anchor_clock": "not_relevant_for_civil_timeline_anchor",
        "execution_mode": "first_subset_fixed_default_phasic_for_global_dynamic_fit_only",
        "run_profile": "first_subset_fixed_default_matches_backend_default",
        "no_runspec": True,
        "no_argv": True,
        "no_config_written": True,
        "no_files_written": True,
    })

    def __post_init__(self) -> None:
        return None


@dataclass(frozen=True)
class GuidedNewAnalysisOutputCreationPolicy:
    schema_version: str = OUTPUT_CREATION_POLICY_SCHEMA_VERSION
    path_role: str = "output_base"
    creation_timing: str = "future_execution_start_only"
    run_directory_strategy: str = "derive_unique_run_id_under_output_base"
    overwrite: bool = False
    precreate_during_preview: bool = False
    config_write_timing: str = "future_execution_or_validation_only"
    gui_preflight_writes_enabled: bool = False
    provenance: dict[str, Any] = field(default_factory=lambda: {
        "path_role": "output_policy_path_is_a_base_for_future_unique_run_directory",
        "creation_timing": "no_directory_creation_until_future_execution_start",
        "overwrite": "first_subset_fixed_default_false",
        "precreate_during_preview": "disabled_for_non_executing_preview",
        "config_write_timing": "no_config_write_until_future_execution_or_validation",
        "gui_preflight_writes_enabled": "disabled_in_model_and_preview",
        "no_runspec": True,
        "no_argv": True,
        "no_config_written": True,
        "no_files_written": True,
    })

    def __post_init__(self) -> None:
        return None


@dataclass(frozen=True)
class GuidedNewAnalysisDynamicFitParameterContract:
    schema_version: str = DYNAMIC_FIT_PARAMETER_CONTRACT_SCHEMA_VERSION
    dynamic_fit_mode: str = "global_linear_regression"
    slope_constraint: str = _backend_dynamic_fit_default("slope_constraint")
    min_slope: float = _backend_dynamic_fit_default("min_slope")
    robust_event_reject_max_iters: int = _backend_dynamic_fit_default("robust_event_reject_max_iters")
    robust_event_reject_residual_z_thresh: float = _backend_dynamic_fit_default(
        "robust_event_reject_residual_z_thresh"
    )
    robust_event_reject_local_var_window_sec: float | None = _backend_dynamic_fit_default(
        "robust_event_reject_local_var_window_sec"
    )
    robust_event_reject_local_var_ratio_thresh: float | None = _backend_dynamic_fit_default(
        "robust_event_reject_local_var_ratio_thresh"
    )
    robust_event_reject_min_keep_fraction: float = _backend_dynamic_fit_default(
        "robust_event_reject_min_keep_fraction"
    )
    adaptive_event_gate_residual_z_thresh: float = _backend_dynamic_fit_default(
        "adaptive_event_gate_residual_z_thresh"
    )
    adaptive_event_gate_local_var_window_sec: float | None = _backend_dynamic_fit_default(
        "adaptive_event_gate_local_var_window_sec"
    )
    adaptive_event_gate_local_var_ratio_thresh: float | None = _backend_dynamic_fit_default(
        "adaptive_event_gate_local_var_ratio_thresh"
    )
    adaptive_event_gate_smooth_window_sec: float = _backend_dynamic_fit_default(
        "adaptive_event_gate_smooth_window_sec"
    )
    adaptive_event_gate_min_trust_fraction: float = _backend_dynamic_fit_default(
        "adaptive_event_gate_min_trust_fraction"
    )
    adaptive_event_gate_freeze_interp_method: str = _backend_dynamic_fit_default(
        "adaptive_event_gate_freeze_interp_method"
    )
    window_sec: float = _backend_dynamic_fit_default("window_sec")
    step_sec: float = _backend_dynamic_fit_default("step_sec")
    r_low: float = _backend_dynamic_fit_default("r_low")
    r_high: float = _backend_dynamic_fit_default("r_high")
    g_min: float = _backend_dynamic_fit_default("g_min")
    min_samples_per_window: int = _backend_dynamic_fit_default("min_samples_per_window")
    min_valid_windows: int = _backend_dynamic_fit_default("min_valid_windows")
    unresolved_parameters: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=lambda: {
        "dynamic_fit_mode": (
            "first_subset_model_default; must match unanimous explicit per-ROI strategy; "
            "not mirrored from backend Config dynamic_fit_mode"
        ),
        "backend_config_dynamic_fit_mode": _backend_dynamic_fit_default("dynamic_fit_mode"),
        "slope_constraint": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "min_slope": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "robust_event_reject_max_iters": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "robust_event_reject_residual_z_thresh": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "robust_event_reject_local_var_window_sec": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "robust_event_reject_local_var_ratio_thresh": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "robust_event_reject_min_keep_fraction": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "adaptive_event_gate_residual_z_thresh": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "adaptive_event_gate_local_var_window_sec": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "adaptive_event_gate_local_var_ratio_thresh": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "adaptive_event_gate_smooth_window_sec": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "adaptive_event_gate_min_trust_fraction": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "adaptive_event_gate_freeze_interp_method": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "window_sec": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "step_sec": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "r_low": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "r_high": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "g_min": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "min_samples_per_window": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "min_valid_windows": BACKEND_DYNAMIC_FIT_DEFAULT_PROVENANCE,
        "legacy_global_settings": (
            "inactive parameter defaults mechanically derived from backend Config; "
            "represented for provenance, not read from widgets"
        ),
        "no_runspec": True,
        "no_argv": True,
        "no_config_written": True,
        "no_files_written": True,
    })

    def __post_init__(self) -> None:
        if self.dynamic_fit_mode not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
            raise ValueError(f"Unsupported dynamic_fit_mode for first subset: {self.dynamic_fit_mode}")
        if self.slope_constraint not in DYNAMIC_FIT_SLOPE_CONSTRAINTS:
            raise ValueError(f"Unsupported dynamic_fit slope_constraint: {self.slope_constraint}")
        if self.adaptive_event_gate_freeze_interp_method not in ADAPTIVE_EVENT_GATE_FREEZE_INTERP_METHODS:
            raise ValueError(
                "adaptive_event_gate_freeze_interp_method must be one of "
                f"{sorted(ADAPTIVE_EVENT_GATE_FREEZE_INTERP_METHODS)}"
            )
        if self.robust_event_reject_max_iters < 1:
            raise ValueError("robust_event_reject_max_iters must be >= 1")
        if self.min_samples_per_window < 0:
            raise ValueError("min_samples_per_window must be >= 0")
        if self.min_valid_windows < 1:
            raise ValueError("min_valid_windows must be >= 1")
        positive_fields = (
            ("robust_event_reject_residual_z_thresh", self.robust_event_reject_residual_z_thresh),
            ("robust_event_reject_min_keep_fraction", self.robust_event_reject_min_keep_fraction),
            ("adaptive_event_gate_residual_z_thresh", self.adaptive_event_gate_residual_z_thresh),
            ("adaptive_event_gate_smooth_window_sec", self.adaptive_event_gate_smooth_window_sec),
            ("adaptive_event_gate_min_trust_fraction", self.adaptive_event_gate_min_trust_fraction),
            ("window_sec", self.window_sec),
            ("step_sec", self.step_sec),
        )
        for name, value in positive_fields:
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        optional_positive_fields = (
            ("robust_event_reject_local_var_window_sec", self.robust_event_reject_local_var_window_sec),
            ("robust_event_reject_local_var_ratio_thresh", self.robust_event_reject_local_var_ratio_thresh),
            ("adaptive_event_gate_local_var_window_sec", self.adaptive_event_gate_local_var_window_sec),
            ("adaptive_event_gate_local_var_ratio_thresh", self.adaptive_event_gate_local_var_ratio_thresh),
        )
        for name, value in optional_positive_fields:
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"{name} must be > 0 when provided")
        fraction_fields = (
            ("robust_event_reject_min_keep_fraction", self.robust_event_reject_min_keep_fraction),
            ("adaptive_event_gate_min_trust_fraction", self.adaptive_event_gate_min_trust_fraction),
        )
        for name, value in fraction_fields:
            if not (0.0 < float(value) <= 1.0):
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class GuidedNewAnalysisDatasetContractSnapshot:
    schema_version: str = DATASET_CONTRACT_SNAPSHOT_SCHEMA_VERSION
    status: str = "missing"  # missing / inferred / applied / invalid / stale / unsupported
    input_format: str | None = None
    resolved_input_format: str | None = None
    acquisition_mode: str | None = None
    contract_values: dict[str, Any] = field(default_factory=dict)
    format_specific: dict[str, Any] = field(default_factory=dict)
    source_identity: GuidedNewAnalysisDatasetContractSourceIdentity = field(
        default_factory=GuidedNewAnalysisDatasetContractSourceIdentity
    )
    validation_issues: tuple[str, ...] = ()
    stale_reasons: tuple[str, ...] = ()
    created_at_utc: str | None = None
    updated_at_utc: str | None = None
    explicitly_applied: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in DATASET_CONTRACT_SNAPSHOT_STATUSES:
            raise ValueError(f"Unsupported dataset contract snapshot status: {self.status}")

    @property
    def current_applied(self) -> bool:
        return (
            self.status == "applied"
            and self.explicitly_applied
            and not self.validation_issues
            and not self.stale_reasons
        )


NEW_ANALYSIS_READINESS_SECTIONS: tuple[tuple[str, str], ...] = (
    ("source_setup", "Source/setup"),
    ("roi_inclusion", "ROI inclusion"),
    ("diagnostic_cache", "Diagnostic cache"),
    ("correction_strategies", "Correction strategies"),
    ("evidence_references", "Evidence references"),
    ("feature_event", "Feature/event settings"),
    ("output_policy", "Output destination"),
    ("execution", "Execution availability"),
)


NEW_ANALYSIS_ISSUE_CATEGORY_TO_SECTION: dict[str, str] = {
    "missing_input_source": "source_setup",
    "invalid_or_missing_input_format": "source_setup",
    "missing_or_invalid_acquisition_structure": "source_setup",
    "no_roi_inventory": "roi_inclusion",
    "no_included_rois": "roi_inclusion",
    "missing_diagnostic_cache": "diagnostic_cache",
    "missing_diagnostic_cache_artifact_path": "diagnostic_cache",
    "missing_diagnostic_cache_provenance_path": "diagnostic_cache",
    "stale_diagnostic_cache": "diagnostic_cache",
    "missing_strategy_choice_for_included_roi": "correction_strategies",
    "stale_strategy_choice": "correction_strategies",
    "forbidden_strategy": "correction_strategies",
    "correction_preview_evidence_path_missing": "evidence_references",
    "correction_preview_source_identity_missing": "evidence_references",
    "signal_only_f0_evidence_path_missing": "evidence_references",
    "signal_only_f0_source_identity_missing": "evidence_references",
    "missing_feature_event_profile": "feature_event",
    "feature_event_profile_not_applied": "feature_event",
    "invalid_feature_event_profile": "feature_event",
    "stale_feature_event_profile": "feature_event",
    "missing_output_policy": "output_policy",
    "output_policy_not_applied": "output_policy",
    "invalid_output_policy": "output_policy",
    "stale_output_policy": "output_policy",
    "execution_not_implemented": "execution",
}


@dataclass(frozen=True)
class GuidedPlanCorrectionChoice:
    roi_id: str
    selected_strategy: str
    source_type: str
    diagnostic_cache_id: str | None = None
    diagnostic_cache_root: str | None = None
    diagnostic_cache_signature: str | None = None
    source_setup_signature: str | None = None
    diagnostic_scope_signature: str | None = None
    build_request_signature: str | None = None
    evidence_chunk: int | None = None
    evidence_summary: str | None = None
    current_or_stale: str | None = "stale"  # "current" or "stale"
    explicit_user_mark: bool = False
    selected_at_utc: str | None = None


@dataclass
class GuidedNewAnalysisDraftPlan:
    schema_version: str = SCHEMA_VERSION
    mode: str = "new_analysis"
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    updated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    input_source_path: str | None = None
    resolved_input_source_path: str | None = None
    input_format: str = "auto"
    acquisition_mode: str = "continuous"

    # timing & structure
    sessions_per_hour: int | None = None
    session_duration_sec: float | None = None
    continuous_window_sec: float | None = 600.0
    continuous_step_sec: float | None = 600.0
    allow_partial_final_window: bool = False
    exclude_incomplete_final_rwd_chunk: bool = False
    acquisition_structure_status: str = "unknown"  # "ready", "incomplete", "invalid", "unknown"

    # Dataset contract snapshot planning state. This is reviewed/applied plan
    # state only; it does not infer fields or generate executable config.
    dataset_contract_snapshot: GuidedNewAnalysisDatasetContractSnapshot = field(
        default_factory=GuidedNewAnalysisDatasetContractSnapshot
    )

    # First-subset execution intent and output creation policy are model-only
    # planning contracts. They do not instantiate RunSpec, generate argv/config,
    # create directories, or run validation/execution.
    execution_intent: GuidedNewAnalysisExecutionIntent = field(
        default_factory=GuidedNewAnalysisExecutionIntent
    )
    output_creation_policy: GuidedNewAnalysisOutputCreationPolicy = field(
        default_factory=GuidedNewAnalysisOutputCreationPolicy
    )
    dynamic_fit_parameter_contract: GuidedNewAnalysisDynamicFitParameterContract = field(
        default_factory=GuidedNewAnalysisDynamicFitParameterContract
    )

    # ROI inventory
    discovered_roi_ids: list[str] = field(default_factory=list)
    included_roi_ids: list[str] = field(default_factory=list)
    excluded_roi_ids: list[str] = field(default_factory=list)

    # Diagnostic cache reference
    cache_id: str | None = None
    cache_root_path: str | None = None
    artifact_record_path: str | None = None
    request_json_path: str | None = None
    provenance_path: str | None = None
    phasic_trace_cache_path: str | None = None
    config_used_path: str | None = None
    source_setup_signature: str | None = None
    diagnostic_scope_signature: str | None = None
    build_request_signature: str | None = None
    stale_or_current: str | None = None  # "current" or "stale"
    stale_reasons: list[str] = field(default_factory=list)
    production_analysis: bool = False
    preliminary_cache: bool = True

    # Correction evidence references
    correction_preview_result_id: str | None = None
    correction_preview_path: str | None = None
    correction_preview_status: str | None = None
    correction_preview_source_cache_id: str | None = None
    signal_only_f0_result_id: str | None = None
    signal_only_f0_path: str | None = None
    signal_only_f0_status: str | None = None
    signal_only_f0_source_cache_id: str | None = None
    selected_evidence_context: dict[str, Any] = field(default_factory=dict)

    # per-ROI correction strategy choices
    per_roi_correction_strategy_choices: list[GuidedPlanCorrectionChoice] = field(default_factory=list)

    # feature/event profile status
    feature_event_profile_status: str = "unavailable"  # missing / default_initialized / applied / stale / invalid / unavailable
    feature_event_profile_id: str | None = None
    feature_event_baseline_config_source: str | None = None
    feature_event_baseline_status: str | None = None
    feature_event_values: dict[str, Any] = field(default_factory=dict)
    feature_event_validation_issues: list[str] = field(default_factory=list)
    feature_event_stale_reasons: list[str] = field(default_factory=list)
    feature_event_updated_at_utc: str | None = None
    feature_event_explicitly_applied: bool = False

    # output policy status
    output_policy_status: str = "missing"  # missing / applied / invalid / stale / unavailable
    output_policy_path: str | None = None
    output_policy_validation_issues: list[str] = field(default_factory=list)
    output_policy_stale_reasons: list[str] = field(default_factory=list)
    output_policy_updated_at_utc: str | None = None
    output_policy_explicitly_applied: bool = False
    output_policy_safety_summary: str = ""

    # user-visible list properties (mirroring fields if needed, or initialized directly)
    warnings: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    informational_issues: list[str] = field(default_factory=list)
    completeness_readiness_summary: str = ""

    # non-executable flags
    execution_ready: bool = False
    executable: bool = False
    production_run_enabled: bool = False

    def __post_init__(self) -> None:
        if self.mode != "new_analysis":
            raise ValueError(f"mode must be 'new_analysis', got {self.mode}")


def _paths_match(p1: str | None, p2: str | None) -> bool:
    if p1 is None or p2 is None:
        return p1 == p2
    import os
    try:
        return os.path.normpath(os.path.normcase(p1)) == os.path.normpath(os.path.normcase(p2))
    except Exception:
        return p1 == p2


def evaluate_new_analysis_plan_issues(plan: GuidedNewAnalysisDraftPlan) -> list[GuidedPlanIssue]:
    """Pure validation function to evaluate plan decisions and return structured issues."""
    issues: list[GuidedPlanIssue] = []

    # 1. missing_input_source
    if not plan.input_source_path or not plan.input_source_path.strip():
        issues.append(GuidedPlanIssue(
            category="missing_input_source",
            message="Raw input/source path is not configured.",
            severity="blocking"
        ))

    # 2. invalid_or_missing_input_format
    if not plan.input_format or plan.input_format not in SUPPORTED_INPUT_FORMATS:
        issues.append(GuidedPlanIssue(
            category="invalid_or_missing_input_format",
            message=f"Input format '{plan.input_format}' is invalid or missing.",
            severity="blocking"
        ))

    # 3. missing_or_invalid_acquisition_structure
    acq_mode = str(plan.acquisition_mode).strip().lower()
    if acq_mode not in SUPPORTED_ACQUISITION_MODES:
        issues.append(GuidedPlanIssue(
            category="missing_or_invalid_acquisition_structure",
            message=f"Acquisition mode '{plan.acquisition_mode}' is unsupported.",
            severity="blocking"
        ))
    elif acq_mode == "intermittent":
        if plan.sessions_per_hour is None or plan.sessions_per_hour <= 0:
            issues.append(GuidedPlanIssue(
                category="missing_or_invalid_acquisition_structure",
                message="Intermittent mode requires positive sessions per hour.",
                severity="blocking"
            ))
        if plan.session_duration_sec is None or plan.session_duration_sec <= 0:
            issues.append(GuidedPlanIssue(
                category="missing_or_invalid_acquisition_structure",
                message="Intermittent mode requires a positive session duration.",
                severity="blocking"
            ))
    elif acq_mode == "continuous":
        if plan.continuous_window_sec is None or plan.continuous_window_sec <= 0:
            issues.append(GuidedPlanIssue(
                category="missing_or_invalid_acquisition_structure",
                message="Continuous mode requires a positive window duration.",
                severity="blocking"
            ))
        if plan.continuous_step_sec is None or plan.continuous_step_sec <= 0:
            issues.append(GuidedPlanIssue(
                category="missing_or_invalid_acquisition_structure",
                message="Continuous mode requires a positive step duration.",
                severity="blocking"
            ))

    # 4. no_roi_inventory
    if not plan.discovered_roi_ids:
        issues.append(GuidedPlanIssue(
            category="no_roi_inventory",
            message="No ROIs discovered in the input source.",
            severity="blocking"
        ))

    # 5. no_included_rois
    if plan.discovered_roi_ids and not plan.included_roi_ids:
        issues.append(GuidedPlanIssue(
            category="no_included_rois",
            message="No ROIs included in the current selection.",
            severity="blocking"
        ))

    # 6. missing_diagnostic_cache
    if not plan.cache_id:
        issues.append(GuidedPlanIssue(
            category="missing_diagnostic_cache",
            message="Diagnostic cache has not been generated for the current setup.",
            severity="blocking"
        ))
    else:
        # Check for missing path fields honestly
        if not plan.artifact_record_path:
            issues.append(GuidedPlanIssue(
                category="missing_diagnostic_cache_artifact_path",
                message="Diagnostic cache artifact record path is missing.",
                severity="warning"
            ))
        if not plan.provenance_path:
            issues.append(GuidedPlanIssue(
                category="missing_diagnostic_cache_provenance_path",
                message="Diagnostic cache provenance path is missing.",
                severity="warning"
            ))

    # 7. stale_diagnostic_cache
    if plan.cache_id and (plan.stale_or_current == "stale" or plan.stale_reasons):
        reasons = f" ({', '.join(plan.stale_reasons)})" if plan.stale_reasons else ""
        issues.append(GuidedPlanIssue(
            category="stale_diagnostic_cache",
            message=f"Diagnostic cache is stale{reasons}. Please rebuild cache.",
            severity="blocking"
        ))

    # ROI strategies checks
    choices_by_roi = {choice.roi_id: choice for choice in plan.per_roi_correction_strategy_choices}
    for roi in plan.included_roi_ids:
        # 8. missing_strategy_choice_for_included_roi
        if roi not in choices_by_roi:
            issues.append(GuidedPlanIssue(
                category="missing_strategy_choice_for_included_roi",
                message=f"Missing correction strategy choice for included ROI '{roi}'.",
                severity="blocking"
            ))
        else:
            choice = choices_by_roi[roi]
            is_choice_stale = False
            choice_stale_reasons = []

            # source_type check
            if choice.source_type != "diagnostic_cache":
                is_choice_stale = True
                choice_stale_reasons.append("invalid_choice_source_type")

            # cache_id check
            if not plan.cache_id or choice.diagnostic_cache_id != plan.cache_id:
                is_choice_stale = True
                choice_stale_reasons.append("source cache id mismatch")

            # cache_root check
            if not _paths_match(choice.diagnostic_cache_root, plan.cache_root_path):
                is_choice_stale = True
                choice_stale_reasons.append("source cache root mismatch")

            # build_request_signature check
            if choice.build_request_signature != plan.build_request_signature:
                is_choice_stale = True
                choice_stale_reasons.append("build request signature mismatch")

            # source_setup_signature check
            if choice.source_setup_signature != plan.source_setup_signature:
                is_choice_stale = True
                choice_stale_reasons.append("source setup signature mismatch")

            # diagnostic_scope_signature check
            if choice.diagnostic_scope_signature != plan.diagnostic_scope_signature:
                is_choice_stale = True
                choice_stale_reasons.append("diagnostic scope signature mismatch")

            # stale check
            if choice.current_or_stale == "stale":
                is_choice_stale = True
                choice_stale_reasons.append("choice explicitly marked stale")

            if plan.stale_or_current == "stale":
                is_choice_stale = True
                choice_stale_reasons.append("active cache is stale")

            if is_choice_stale:
                reasons_str = f" ({', '.join(choice_stale_reasons)})" if choice_stale_reasons else ""
                issues.append(GuidedPlanIssue(
                    category="stale_strategy_choice",
                    message=f"Correction strategy choice for ROI '{roi}' is stale{reasons_str}.",
                    severity="blocking"
                ))

            # 10. forbidden_strategy
            if choice.selected_strategy in FORBIDDEN_CORRECTION_STRATEGIES:
                issues.append(GuidedPlanIssue(
                    category="forbidden_strategy",
                    message=f"Strategy '{choice.selected_strategy}' for ROI '{roi}' is forbidden.",
                    severity="blocking"
                ))
            elif choice.selected_strategy not in RUNNABLE_CORRECTION_STRATEGIES:
                issues.append(GuidedPlanIssue(
                    category="forbidden_strategy",
                    message=f"Strategy '{choice.selected_strategy}' for ROI '{roi}' is unknown/invalid.",
                    severity="blocking"
                ))

    # Evidence result missing checks
    if plan.correction_preview_result_id:
        if not plan.correction_preview_path:
            issues.append(GuidedPlanIssue(
                category="correction_preview_evidence_path_missing",
                message="Correction preview evidence output path is missing.",
                severity="warning"
            ))
        if not plan.correction_preview_source_cache_id:
            issues.append(GuidedPlanIssue(
                category="correction_preview_source_identity_missing",
                message="Correction preview source diagnostic cache identity is missing.",
                severity="warning"
            ))

    if plan.signal_only_f0_result_id:
        if not plan.signal_only_f0_path:
            issues.append(GuidedPlanIssue(
                category="signal_only_f0_evidence_path_missing",
                message="Signal-Only F0 evidence output path is missing.",
                severity="warning"
            ))
        if not plan.signal_only_f0_source_cache_id:
            issues.append(GuidedPlanIssue(
                category="signal_only_f0_source_identity_missing",
                message="Signal-Only F0 source diagnostic cache identity is missing.",
                severity="warning"
            ))

    # 11. missing_feature_event_profile
    if plan.feature_event_profile_status in ("missing", "unavailable"):
        issues.append(GuidedPlanIssue(
            category="missing_feature_event_profile",
            message="Feature/event profile settings are missing or unavailable in this stage.",
            severity="blocking"
        ))

    # 12. feature_event_profile_not_applied
    elif plan.feature_event_profile_status == "default_initialized":
        issues.append(GuidedPlanIssue(
            category="feature_event_profile_not_applied",
            message="Defaults are loaded, but feature/event settings have not been explicitly applied to the draft plan.",
            severity="blocking"
        ))

    # 12b. invalid_feature_event_profile
    elif plan.feature_event_profile_status == "invalid":
        msg = "Feature/event profile settings are invalid."
        if plan.feature_event_validation_issues:
            msg = f"Feature/event profile settings are invalid: {'; '.join(plan.feature_event_validation_issues)}"
        issues.append(GuidedPlanIssue(
            category="invalid_feature_event_profile",
            message=msg,
            severity="blocking"
        ))

    # 12c. stale_feature_event_profile
    elif plan.feature_event_profile_status == "stale":
        reasons = "; ".join(plan.feature_event_stale_reasons) or "active baseline config changed"
        issues.append(GuidedPlanIssue(
            category="stale_feature_event_profile",
            message=f"Feature/event profile is stale: {reasons}",
            severity="blocking"
        ))

    elif plan.feature_event_profile_status == "applied":
        if not plan.feature_event_explicitly_applied:
            issues.append(GuidedPlanIssue(
                category="feature_event_profile_not_applied",
                message="Feature/event profile status is applied but explicit apply provenance is missing.",
                severity="blocking"
            ))
        if plan.feature_event_validation_issues:
            issues.append(GuidedPlanIssue(
                category="invalid_feature_event_profile",
                message=f"Feature/event profile has validation issues: {'; '.join(plan.feature_event_validation_issues)}",
                severity="blocking"
            ))

    # 13. output policy
    if plan.output_policy_status in ("missing", "unavailable"):
        issues.append(GuidedPlanIssue(
            category="missing_output_policy",
            message="Output policy is missing or unavailable.",
            severity="blocking"
        ))

    elif plan.output_policy_status == "selected":
        issues.append(GuidedPlanIssue(
            category="output_policy_not_applied",
            message="Output destination has been selected in the editor but not explicitly applied to the draft plan.",
            severity="blocking"
        ))

    elif plan.output_policy_status == "invalid":
        msg = "Output policy is invalid."
        if plan.output_policy_validation_issues:
            msg = f"Output policy is invalid: {'; '.join(plan.output_policy_validation_issues)}"
        issues.append(GuidedPlanIssue(
            category="invalid_output_policy",
            message=msg,
            severity="blocking"
        ))

    elif plan.output_policy_status == "stale":
        reasons = "; ".join(plan.output_policy_stale_reasons) or "source or diagnostic-cache context changed"
        issues.append(GuidedPlanIssue(
            category="stale_output_policy",
            message=f"Output policy is stale: {reasons}",
            severity="blocking"
        ))

    elif plan.output_policy_status == "applied":
        if not plan.output_policy_explicitly_applied:
            issues.append(GuidedPlanIssue(
                category="output_policy_not_applied",
                message="Output policy status is applied but explicit apply provenance is missing.",
                severity="blocking"
            ))
        if not plan.output_policy_path:
            issues.append(GuidedPlanIssue(
                category="invalid_output_policy",
                message="Output policy is applied but output destination path is missing.",
                severity="blocking"
            ))
        if plan.output_policy_validation_issues:
            issues.append(GuidedPlanIssue(
                category="invalid_output_policy",
                message=f"Output policy has validation issues: {'; '.join(plan.output_policy_validation_issues)}",
                severity="blocking"
            ))

    # 15. execution_not_implemented
    issues.append(GuidedPlanIssue(
        category="execution_not_implemented",
        message="This draft plan is not executable yet. Final Run is not implemented in this stage.",
        severity="info"
    ))

    return issues


def _readiness_status_from_issues(issues: list[GuidedPlanIssue]) -> str:
    if not issues:
        return "ready"
    blocking = [issue for issue in issues if issue.severity == "blocking"]
    if blocking:
        categories = {issue.category for issue in blocking}
        if any("stale" in category for category in categories):
            return "stale"
        if any("invalid" in category or category == "forbidden_strategy" for category in categories):
            return "invalid"
        if any("missing" in category or category.startswith("no_") or "not_applied" in category for category in categories):
            return "missing"
        return "blocked"
    if any(issue.severity == "warning" for issue in issues):
        return "warning"
    return "info"


def evaluate_new_analysis_plan_readiness(plan: GuidedNewAnalysisDraftPlan) -> GuidedNewAnalysisReadiness:
    """Evaluate new_analysis draft completeness without touching Qt, widgets, or files."""
    issues = evaluate_new_analysis_plan_issues(plan)
    issues_by_section: dict[str, list[GuidedPlanIssue]] = {
        key: [] for key, _label in NEW_ANALYSIS_READINESS_SECTIONS
    }
    for issue in issues:
        section_key = NEW_ANALYSIS_ISSUE_CATEGORY_TO_SECTION.get(issue.category, "source_setup")
        issues_by_section.setdefault(section_key, []).append(issue)

    sections: list[GuidedNewAnalysisSectionReadiness] = []
    for key, label in NEW_ANALYSIS_READINESS_SECTIONS:
        section_issues = issues_by_section.get(key, [])
        sections.append(GuidedNewAnalysisSectionReadiness(
            key=key,
            label=label,
            status=_readiness_status_from_issues(section_issues),
            blocking_issues=tuple(issue for issue in section_issues if issue.severity == "blocking"),
            warning_issues=tuple(issue for issue in section_issues if issue.severity == "warning"),
            info_issues=tuple(issue for issue in section_issues if issue.severity == "info"),
        ))

    blocking_for_handoff = tuple(
        issue for issue in issues
        if issue.severity == "blocking" and issue.category != "execution_not_implemented"
    )
    warning_issues = tuple(issue for issue in issues if issue.severity == "warning")
    info_issues = tuple(issue for issue in issues if issue.severity == "info")
    return GuidedNewAnalysisReadiness(
        sections=tuple(sections),
        plan_complete_for_handoff=not blocking_for_handoff,
        execution_available=False,
        execution_blocked_reason="Final Guided Run/RunSpec is not implemented in this stage.",
        blocking_issues=blocking_for_handoff,
        warning_issues=warning_issues,
        info_issues=info_issues,
    )


def _execution_subset_issue(category: str, message: str, severity: str = "blocking") -> GuidedNewAnalysisExecutionSubsetIssue:
    return GuidedNewAnalysisExecutionSubsetIssue(
        category=category,
        message=message,
        severity=severity,
    )


def _execution_field(
    field_name: str,
    status: str,
    *,
    value: Any | None = None,
    provenance: str,
    blocks_subset: bool = False,
    issue_category: str | None = None,
) -> GuidedNewAnalysisExecutionFieldClassification:
    return GuidedNewAnalysisExecutionFieldClassification(
        field_name=field_name,
        status=status,
        value=value,
        provenance=provenance,
        blocks_subset=blocks_subset,
        issue_category=issue_category,
    )


def _same_optional_value(expected: Any, actual: Any) -> bool:
    return expected == actual


def _same_optional_path(expected: str | None, actual: str | None) -> bool:
    return _paths_match(expected, actual)


def _normalized_format(value: str | None) -> str:
    return str(value or "").strip().lower()


def _dataset_contract_snapshot_plan_consistency_reasons(
    plan: GuidedNewAnalysisDraftPlan,
    snapshot: GuidedNewAnalysisDatasetContractSnapshot,
) -> tuple[str, ...]:
    identity = snapshot.source_identity
    reasons: list[str] = []

    if not _same_optional_path(plan.input_source_path, identity.input_source_path):
        reasons.append("input_source_path mismatch")
    if not _same_optional_path(plan.resolved_input_source_path, identity.resolved_input_source_path):
        reasons.append("resolved_input_source_path mismatch")
    if not _same_optional_value(_normalized_format(plan.input_format), _normalized_format(identity.input_format)):
        reasons.append("input_format mismatch")
    if not _same_optional_value(_normalized_format(plan.acquisition_mode), _normalized_format(identity.acquisition_mode)):
        reasons.append("acquisition_mode mismatch")
    if not _same_optional_value(plan.sessions_per_hour, identity.sessions_per_hour):
        reasons.append("sessions_per_hour mismatch")
    if not _same_optional_value(plan.session_duration_sec, identity.session_duration_sec):
        reasons.append("session_duration_sec mismatch")
    if not _same_optional_value(plan.continuous_window_sec, identity.continuous_window_sec):
        reasons.append("continuous_window_sec mismatch")
    if not _same_optional_value(plan.continuous_step_sec, identity.continuous_step_sec):
        reasons.append("continuous_step_sec mismatch")
    if not _same_optional_value(plan.allow_partial_final_window, identity.allow_partial_final_window):
        reasons.append("allow_partial_final_window mismatch")
    if not _same_optional_value(plan.exclude_incomplete_final_rwd_chunk, identity.exclude_incomplete_final_rwd_chunk):
        reasons.append("exclude_incomplete_final_rwd_chunk mismatch")
    if identity.included_roi_ids and tuple(plan.included_roi_ids) != identity.included_roi_ids:
        reasons.append("included_roi_ids mismatch")

    plan_fmt = _normalized_format(plan.input_format)
    snapshot_input_fmt = _normalized_format(snapshot.input_format)
    snapshot_resolved_fmt = _normalized_format(snapshot.resolved_input_format)
    if snapshot_input_fmt and snapshot_input_fmt != plan_fmt:
        reasons.append("snapshot input_format mismatch")
    if snapshot_resolved_fmt and snapshot_resolved_fmt != plan_fmt:
        reasons.append("snapshot resolved_input_format mismatch")
    if _normalized_format(snapshot.acquisition_mode) and _normalized_format(snapshot.acquisition_mode) != _normalized_format(plan.acquisition_mode):
        reasons.append("snapshot acquisition_mode mismatch")

    return tuple(dict.fromkeys(reasons))


def _dataset_contract_snapshot_execution_field(
    snapshot: GuidedNewAnalysisDatasetContractSnapshot,
    plan: GuidedNewAnalysisDraftPlan | None = None,
) -> GuidedNewAnalysisExecutionFieldClassification:
    consistency_reasons = (
        _dataset_contract_snapshot_plan_consistency_reasons(plan, snapshot)
        if plan is not None and snapshot.current_applied
        else ()
    )
    value = {
        "schema_version": snapshot.schema_version,
        "status": snapshot.status,
        "input_format": snapshot.input_format,
        "resolved_input_format": snapshot.resolved_input_format,
        "acquisition_mode": snapshot.acquisition_mode,
        "explicitly_applied": snapshot.explicitly_applied,
        "current_applied": snapshot.current_applied,
        "validation_issues": list(snapshot.validation_issues),
        "stale_reasons": list(snapshot.stale_reasons),
        "consistency_reasons": list(consistency_reasons),
    }
    if snapshot.status == "missing":
        status = "required_missing"
        provenance = "dataset contract snapshot is not represented as applied Guided planning state"
        blocks_subset = False
        issue_category = None
    elif snapshot.status == "unsupported":
        status = "unsupported"
        provenance = "dataset contract snapshot records an unsupported format/acquisition combination"
        blocks_subset = True
        issue_category = "unsupported_dataset_contract_snapshot"
    elif snapshot.validation_issues or snapshot.status == "invalid":
        status = "invalid"
        provenance = "dataset contract snapshot failed structural or reviewed validation"
        blocks_subset = True
        issue_category = "invalid_dataset_contract_snapshot"
    elif snapshot.stale_reasons or snapshot.status == "stale":
        status = "stale"
        provenance = "dataset contract snapshot is applied or inferred but no longer current"
        blocks_subset = True
        issue_category = "stale_dataset_contract_snapshot"
    elif consistency_reasons:
        status = "stale"
        provenance = "dataset contract snapshot source identity is inconsistent with current Guided plan state"
        blocks_subset = True
        issue_category = "inconsistent_dataset_contract_snapshot"
    elif snapshot.current_applied:
        status = "present"
        provenance = "applied GuidedNewAnalysisDraftPlan dataset contract snapshot"
        blocks_subset = False
        issue_category = None
    elif snapshot.status == "inferred":
        status = "selected"
        provenance = "dataset contract snapshot is visible/reviewable but not explicitly applied"
        blocks_subset = False
        issue_category = None
    elif snapshot.status == "applied" and not snapshot.explicitly_applied:
        status = "selected"
        provenance = "dataset contract snapshot status is applied but explicit apply provenance is missing"
        blocks_subset = False
        issue_category = None
    else:
        status = "selected"
        provenance = "dataset contract snapshot is not current applied planning state"
        blocks_subset = False
        issue_category = None
    return _execution_field(
        "dataset_contract_snapshot",
        status,
        value=value,
        provenance=provenance,
        blocks_subset=blocks_subset,
        issue_category=issue_category,
    )


def _snapshot_field_value(snapshot: GuidedNewAnalysisDatasetContractSnapshot) -> dict[str, Any]:
    return {
        "schema_version": snapshot.schema_version,
        "status": snapshot.status,
        "input_format": snapshot.input_format,
        "resolved_input_format": snapshot.resolved_input_format,
        "acquisition_mode": snapshot.acquisition_mode,
        "contract_keys": sorted(snapshot.contract_values.keys()),
        "format_specific_keys": sorted(snapshot.format_specific.keys()),
    }


def _snapshot_has_mapping_fields(
    snapshot: GuidedNewAnalysisDatasetContractSnapshot,
    required_keys: tuple[str, ...],
) -> bool:
    values = {**dict(snapshot.contract_values), **dict(snapshot.format_specific)}
    return all(bool(values.get(key)) for key in required_keys)


def _valid_fixed_anchor_clock(value: str | None) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parts = value.strip().split(":")
    if len(parts) not in {2, 3}:
        return False
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return False
    hour, minute = numbers[0], numbers[1]
    second = numbers[2] if len(numbers) == 3 else 0
    return 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59


def _execution_intent_value(intent: GuidedNewAnalysisExecutionIntent) -> dict[str, Any]:
    return {
        "schema_version": intent.schema_version,
        "timeline_anchor_mode": intent.timeline_anchor_mode,
        "fixed_daily_anchor_clock": intent.fixed_daily_anchor_clock,
        "execution_mode": intent.execution_mode,
        "run_profile": intent.run_profile,
        "provenance": dict(intent.provenance),
    }


def _execution_intent_fields(
    intent: GuidedNewAnalysisExecutionIntent,
) -> tuple[GuidedNewAnalysisExecutionFieldClassification, ...]:
    timeline_ok = intent.timeline_anchor_mode == "civil" and intent.fixed_daily_anchor_clock is None
    if intent.timeline_anchor_mode == "fixed_daily_anchor" and not _valid_fixed_anchor_clock(intent.fixed_daily_anchor_clock):
        timeline_provenance = "fixed_daily_anchor timeline mode requires a valid fixed_daily_anchor_clock"
    elif intent.timeline_anchor_mode != "civil":
        timeline_provenance = "first subset supports only civil timeline anchor mode"
    elif intent.fixed_daily_anchor_clock is not None:
        timeline_provenance = "civil timeline anchor mode must not carry fixed_daily_anchor_clock"
    else:
        timeline_provenance = "first subset fixed default; matches backend/Full Control civil timeline anchor default"

    execution_mode_ok = intent.execution_mode == "phasic"
    run_profile_ok = intent.run_profile == "full"

    return (
        _execution_field(
            "timeline_anchor_mode",
            "fixed_default" if timeline_ok else "invalid",
            value=intent.timeline_anchor_mode,
            provenance=timeline_provenance,
            blocks_subset=not timeline_ok,
            issue_category=None if timeline_ok else "invalid_timeline_anchor_mode",
        ),
        _execution_field(
            "fixed_daily_anchor_clock",
            "fixed_default" if intent.fixed_daily_anchor_clock is None else "invalid",
            value=intent.fixed_daily_anchor_clock,
            provenance=(
                "first subset fixed default; no fixed daily anchor"
                if intent.fixed_daily_anchor_clock is None
                else "fixed_daily_anchor_clock is not used by first subset civil timeline default"
            ),
            blocks_subset=intent.fixed_daily_anchor_clock is not None and intent.timeline_anchor_mode != "fixed_daily_anchor",
            issue_category="invalid_timeline_anchor_mode"
            if intent.fixed_daily_anchor_clock is not None and intent.timeline_anchor_mode != "fixed_daily_anchor"
            else None,
        ),
        _execution_field(
            "mode",
            "fixed_default" if execution_mode_ok else "unsupported",
            value=intent.execution_mode,
            provenance=(
                "first subset fixed default phasic for global dynamic-fit/phasic-output subset"
                if execution_mode_ok
                else "first subset supports only phasic execution mode"
            ),
            blocks_subset=not execution_mode_ok,
            issue_category=None if execution_mode_ok else "invalid_execution_mode",
        ),
        _execution_field(
            "run_profile",
            "fixed_default" if run_profile_ok else "unsupported",
            value=intent.run_profile,
            provenance=(
                "first subset fixed default full; matches backend/Full Control run profile default"
                if run_profile_ok
                else "first subset supports only full run profile"
            ),
            blocks_subset=not run_profile_ok,
            issue_category=None if run_profile_ok else "unsupported_run_profile_for_first_subset",
        ),
    )


def _output_creation_policy_value(policy: GuidedNewAnalysisOutputCreationPolicy) -> dict[str, Any]:
    return {
        "schema_version": policy.schema_version,
        "path_role": policy.path_role,
        "creation_timing": policy.creation_timing,
        "run_directory_strategy": policy.run_directory_strategy,
        "overwrite": policy.overwrite,
        "precreate_during_preview": policy.precreate_during_preview,
        "config_write_timing": policy.config_write_timing,
        "gui_preflight_writes_enabled": policy.gui_preflight_writes_enabled,
        "provenance": dict(policy.provenance),
    }


def _output_creation_policy_field(
    policy: GuidedNewAnalysisOutputCreationPolicy,
) -> GuidedNewAnalysisExecutionFieldClassification:
    safe = (
        policy.path_role == "output_base"
        and policy.creation_timing == "future_execution_start_only"
        and policy.run_directory_strategy == "derive_unique_run_id_under_output_base"
        and policy.overwrite is False
        and policy.precreate_during_preview is False
        and policy.config_write_timing == "future_execution_or_validation_only"
        and policy.gui_preflight_writes_enabled is False
    )
    return _execution_field(
        "output_creation_policy",
        "present" if safe else "invalid",
        value=_output_creation_policy_value(policy),
        provenance=(
            "safe first-subset output creation policy; classification only, no directories or files are created"
            if safe
            else "unsafe or unsupported output creation policy for first subset"
        ),
        blocks_subset=not safe,
        issue_category=None if safe else "unsafe_output_creation_policy",
    )


def _dynamic_fit_parameter_contract_value(
    contract: GuidedNewAnalysisDynamicFitParameterContract,
) -> dict[str, Any]:
    return {
        "schema_version": contract.schema_version,
        "dynamic_fit_mode": contract.dynamic_fit_mode,
        "slope_constraint": contract.slope_constraint,
        "min_slope": contract.min_slope,
        "robust_event_reject_max_iters": contract.robust_event_reject_max_iters,
        "robust_event_reject_residual_z_thresh": contract.robust_event_reject_residual_z_thresh,
        "robust_event_reject_local_var_window_sec": contract.robust_event_reject_local_var_window_sec,
        "robust_event_reject_local_var_ratio_thresh": contract.robust_event_reject_local_var_ratio_thresh,
        "robust_event_reject_min_keep_fraction": contract.robust_event_reject_min_keep_fraction,
        "adaptive_event_gate_residual_z_thresh": contract.adaptive_event_gate_residual_z_thresh,
        "adaptive_event_gate_local_var_window_sec": contract.adaptive_event_gate_local_var_window_sec,
        "adaptive_event_gate_local_var_ratio_thresh": contract.adaptive_event_gate_local_var_ratio_thresh,
        "adaptive_event_gate_smooth_window_sec": contract.adaptive_event_gate_smooth_window_sec,
        "adaptive_event_gate_min_trust_fraction": contract.adaptive_event_gate_min_trust_fraction,
        "adaptive_event_gate_freeze_interp_method": contract.adaptive_event_gate_freeze_interp_method,
        "window_sec": contract.window_sec,
        "step_sec": contract.step_sec,
        "r_low": contract.r_low,
        "r_high": contract.r_high,
        "g_min": contract.g_min,
        "min_samples_per_window": contract.min_samples_per_window,
        "min_valid_windows": contract.min_valid_windows,
        "backend_default_source": BACKEND_DYNAMIC_FIT_DEFAULT_SOURCE,
        "backend_default_values": dict(_CANONICAL_DYNAMIC_FIT_BACKEND_DEFAULTS),
        "unresolved_parameters": list(contract.unresolved_parameters),
        "provenance": dict(contract.provenance),
    }


def _dynamic_fit_parameter_contract_field(
    contract: GuidedNewAnalysisDynamicFitParameterContract,
) -> GuidedNewAnalysisExecutionFieldClassification:
    unresolved = tuple(contract.unresolved_parameters)
    return _execution_field(
        "dynamic_fit_parameter_contract",
        "present" if not unresolved else "required_missing",
        value=_dynamic_fit_parameter_contract_value(contract),
        provenance=(
            "stored GuidedNewAnalysisDraftPlan dynamic-fit parameter contract; "
            "backend defaults mechanically derived from Config where marked"
            if not unresolved
            else "dynamic-fit parameter contract has unresolved parameters for future executable mapping"
        ),
        blocks_subset=bool(unresolved),
        issue_category=None if not unresolved else "unresolved_dynamic_fit_parameter_contract",
    )


def _execution_field_classifications(plan: GuidedNewAnalysisDraftPlan) -> tuple[GuidedNewAnalysisExecutionFieldClassification, ...]:
    dataset_snapshot_field = _dataset_contract_snapshot_execution_field(plan.dataset_contract_snapshot, plan)
    dataset_snapshot_usable = (
        dataset_snapshot_field.status == "present"
        and not dataset_snapshot_field.blocks_subset
        and plan.dataset_contract_snapshot.current_applied
    )
    execution_intent_fields = _execution_intent_fields(plan.execution_intent)
    fields: list[GuidedNewAnalysisExecutionFieldClassification] = [
        dataset_snapshot_field,
        *execution_intent_fields[:2],
        _execution_field(
            "render_modes",
            "fixed_default",
            value={
                "signal_iso_render_mode": "default_backend_behavior",
                "dff_render_mode": "default_backend_behavior",
                "stacked_render_mode": "default_backend_behavior",
            },
            provenance="first subset fixed default; detailed render controls remain Full Control-only",
        ),
        _execution_field(
            "traces_only",
            "fixed_default",
            value=False,
            provenance="first subset fixed default for production feature/event handoff",
        ),
        _execution_field(
            "preview_first_n",
            "fixed_default",
            value=None,
            provenance="first subset fixed default; preview-first-N remains Full Control-only",
        ),
        _execution_field(
            "representative_session",
            "deferred_full_control",
            value=None,
            provenance="representative-session preview behavior is not part of first Guided execution subset",
        ),
        _execution_field(
            "validate_only_behavior",
            "fixed_default",
            value="explicit_future_validate_action_only",
            provenance="classification only; no validation process is implemented here",
        ),
        _execution_field(
            "provenance_run_metadata_fields",
            "present",
            value={
                "plan_schema_version": plan.schema_version,
                "created_at_utc": plan.created_at_utc,
                "updated_at_utc": plan.updated_at_utc,
                "cache_id": plan.cache_id,
            },
            provenance="stored GuidedNewAnalysisDraftPlan fields only",
        ),
    ]

    fields.extend(execution_intent_fields[2:])
    fields.append(_output_creation_policy_field(plan.output_creation_policy))
    fields.append(_dynamic_fit_parameter_contract_field(plan.dynamic_fit_parameter_contract))

    roi_identity_status = "present" if plan.included_roi_ids else "required_missing"
    fields.append(_execution_field(
        "roi_identity",
        roi_identity_status,
        value=list(plan.included_roi_ids),
        provenance="GuidedNewAnalysisDraftPlan included_roi_ids",
        blocks_subset=roi_identity_status != "present",
        issue_category=None if roi_identity_status == "present" else "missing_roi_identity",
    ))

    fmt = str(plan.input_format or "").strip().lower()
    acq = str(plan.acquisition_mode or "").strip().lower()
    if fmt == "rwd":
        if dataset_snapshot_usable:
            fields.append(_execution_field(
                "dataset_contract_overrides",
                "present",
                value=_snapshot_field_value(plan.dataset_contract_snapshot),
                provenance="applied GuidedNewAnalysisDraftPlan dataset contract snapshot consumed for first-subset readiness classification",
            ))
        else:
            blocks_category = dataset_snapshot_field.issue_category or "missing_rwd_dataset_contract"
            fields.append(_execution_field(
                "dataset_contract_overrides",
                dataset_snapshot_field.status if dataset_snapshot_field.status in {"invalid", "stale", "unsupported"} else "required_missing",
                value=_snapshot_field_value(plan.dataset_contract_snapshot) if plan.dataset_contract_snapshot.status != "missing" else None,
                provenance=(
                    dataset_snapshot_field.provenance
                    if dataset_snapshot_field.status in {"invalid", "stale", "unsupported"}
                    else "RWD dataset contract snapshot is not represented as current applied GuidedNewAnalysisDraftPlan state"
                ),
                blocks_subset=True,
                issue_category=blocks_category,
            ))
        fields.append(_execution_field(
            "acquisition_repair_fields",
            "present",
            value={"exclude_incomplete_final_rwd_chunk": plan.exclude_incomplete_final_rwd_chunk},
            provenance="GuidedNewAnalysisDraftPlan acquisition repair field",
        ))
    elif fmt == "npm":
        if acq == "continuous":
            fields.append(_execution_field(
                "format_acquisition_support",
                "required_missing",
                value={"input_format": fmt, "acquisition_mode": acq},
                provenance="first subset blocks NPM continuous before execution-field mapping",
                blocks_subset=True,
                issue_category="unsupported_npm_continuous",
            ))
        else:
            npm_has_mapping = dataset_snapshot_usable and _snapshot_has_mapping_fields(
                plan.dataset_contract_snapshot,
                ("signal_channel", "control_channel", "time_column"),
            )
            fields.append(_execution_field(
                "npm_channel_mapping",
                "present" if npm_has_mapping else "required_missing",
                value=_snapshot_field_value(plan.dataset_contract_snapshot) if npm_has_mapping else None,
                provenance=(
                    "applied GuidedNewAnalysisDraftPlan dataset contract snapshot provides signal_channel/control_channel/time_column"
                    if npm_has_mapping
                    else "NPM signal/control/time channel mapping is not represented in current applied GuidedNewAnalysisDraftPlan dataset contract snapshot"
                ),
                blocks_subset=not npm_has_mapping,
                issue_category=None if npm_has_mapping else "missing_npm_channel_mapping",
            ))
            fields.append(_execution_field(
                "dataset_contract_overrides",
                "present" if npm_has_mapping else "required_missing",
                value=_snapshot_field_value(plan.dataset_contract_snapshot) if npm_has_mapping else None,
                provenance=(
                    "applied GuidedNewAnalysisDraftPlan dataset contract snapshot consumed for NPM first-subset readiness classification"
                    if npm_has_mapping
                    else "NPM dataset contract snapshot does not contain required channel mapping fields"
                ),
                blocks_subset=not npm_has_mapping,
                issue_category=None if npm_has_mapping else "missing_npm_dataset_contract",
            ))
    elif fmt == "custom_tabular":
        custom_has_mapping = dataset_snapshot_usable and _snapshot_has_mapping_fields(
            plan.dataset_contract_snapshot,
            ("signal_column", "control_column", "time_column", "roi_column"),
        )
        fields.append(_execution_field(
            "custom_tabular_column_mapping",
            "present" if custom_has_mapping else "required_missing",
            value=_snapshot_field_value(plan.dataset_contract_snapshot) if custom_has_mapping else None,
            provenance=(
                "applied GuidedNewAnalysisDraftPlan dataset contract snapshot provides signal/control/time/ROI column mapping"
                if custom_has_mapping
                else "custom_tabular signal/control/time/ROI column mapping is not represented in current applied GuidedNewAnalysisDraftPlan dataset contract snapshot"
            ),
            blocks_subset=not custom_has_mapping,
            issue_category=None if custom_has_mapping else "missing_custom_tabular_column_mapping",
        ))
        fields.append(_execution_field(
            "dataset_contract_overrides",
            "present" if custom_has_mapping else "required_missing",
            value=_snapshot_field_value(plan.dataset_contract_snapshot) if custom_has_mapping else None,
            provenance=(
                "applied GuidedNewAnalysisDraftPlan dataset contract snapshot consumed for custom_tabular first-subset readiness classification"
                if custom_has_mapping
                else "custom_tabular dataset contract snapshot does not contain required column mapping fields"
            ),
            blocks_subset=not custom_has_mapping,
            issue_category=None if custom_has_mapping else "missing_custom_tabular_dataset_contract",
        ))
    elif fmt == "auto":
        fields.append(_execution_field(
            "format_acquisition_support",
            "required_missing",
            value={"input_format": fmt, "acquisition_mode": acq},
            provenance="first subset requires an explicit resolved input format; auto is planning-only here",
            blocks_subset=True,
            issue_category="unsupported_auto_format_for_execution_subset",
        ))
    else:
        fields.append(_execution_field(
            "format_acquisition_support",
            "required_missing",
            value={"input_format": fmt, "acquisition_mode": acq},
            provenance="input format is missing or unsupported by GuidedNewAnalysisDraftPlan",
            blocks_subset=True,
            issue_category="unsupported_input_format_for_execution_subset",
        ))

    if acq not in SUPPORTED_ACQUISITION_MODES:
        fields.append(_execution_field(
            "format_acquisition_support",
            "required_missing",
            value={"input_format": fmt, "acquisition_mode": acq},
            provenance="acquisition mode is missing or unsupported by GuidedNewAnalysisDraftPlan",
            blocks_subset=True,
            issue_category="unsupported_acquisition_mode_for_execution_subset",
        ))

    return tuple(fields)


def evaluate_guided_new_analysis_execution_subset_readiness(
    plan: GuidedNewAnalysisDraftPlan,
) -> GuidedNewAnalysisExecutionSubsetReadiness:
    """Evaluate first-subset execution readiness without side effects.

    This helper is model-only. It does not inspect files, instantiate RunSpec,
    build argv, read GUI widgets, write config, create directories, or mutate plan.
    """
    if not isinstance(plan, GuidedNewAnalysisDraftPlan):
        raise TypeError("plan must be a GuidedNewAnalysisDraftPlan")

    planning_readiness = evaluate_new_analysis_plan_readiness(plan)
    issues: list[GuidedNewAnalysisExecutionSubsetIssue] = []
    warnings: list[GuidedNewAnalysisExecutionSubsetIssue] = []
    info: list[GuidedNewAnalysisExecutionSubsetIssue] = [
        _execution_subset_issue(
            "execution_not_implemented",
            "Final Guided Run/RunSpec is not implemented in this stage.",
            severity="info",
        )
    ]

    if not planning_readiness.plan_complete_for_handoff:
        issues.append(_execution_subset_issue(
            "incomplete_planning_readiness",
            "Planning readiness is incomplete; first execution subset cannot be entered.",
        ))
        for planning_issue in planning_readiness.blocking_issues:
            issues.append(_execution_subset_issue(
                f"planning_{planning_issue.category}",
                planning_issue.message,
            ))

    choices_by_roi = {choice.roi_id: choice for choice in plan.per_roi_correction_strategy_choices}
    choice_counts_by_roi: dict[str, int] = {}
    for choice in plan.per_roi_correction_strategy_choices:
        choice_counts_by_roi[choice.roi_id] = choice_counts_by_roi.get(choice.roi_id, 0) + 1
    included_choices = [
        choices_by_roi[roi]
        for roi in plan.included_roi_ids
        if roi in choices_by_roi
    ]
    selected_strategies = [choice.selected_strategy for choice in included_choices]
    unique_strategies = tuple(dict.fromkeys(selected_strategies))
    allowed_dynamic_fit_strategy: str | None = None

    if len(included_choices) != len(plan.included_roi_ids):
        issues.append(_execution_subset_issue(
            "missing_strategy_choice_for_execution_subset",
            "Every included ROI must have one explicit strategy choice for the first execution subset.",
        ))

    for roi in plan.included_roi_ids:
        if choice_counts_by_roi.get(roi, 0) > 1:
            issues.append(_execution_subset_issue(
                "duplicate_strategy_choice_for_execution_subset",
                f"Included ROI '{roi}' has duplicate strategy choices; first subset requires exactly one choice per ROI.",
            ))

    for choice in included_choices:
        if not choice.explicit_user_mark:
            issues.append(_execution_subset_issue(
                "non_explicit_strategy_choice",
                f"Strategy choice for ROI '{choice.roi_id}' is not explicitly user-marked.",
            ))
        if choice.selected_strategy in FORBIDDEN_CORRECTION_STRATEGIES:
            issues.append(_execution_subset_issue(
                "forbidden_strategy_state",
                f"Strategy '{choice.selected_strategy}' for ROI '{choice.roi_id}' is forbidden for execution.",
            ))
        elif choice.selected_strategy == "signal_only_f0":
            issues.append(_execution_subset_issue(
                "signal_only_f0_execution_not_supported",
                "Signal-Only F0 remains planning/diagnostic only until applied-dF/F routing is designed.",
            ))
        elif choice.selected_strategy not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
            issues.append(_execution_subset_issue(
                "unsupported_dynamic_fit_strategy_for_first_subset",
                f"Strategy '{choice.selected_strategy}' is not supported by the first execution subset.",
            ))

    if len(unique_strategies) > 1:
        issues.append(_execution_subset_issue(
            "mixed_per_roi_strategies",
            "Included ROIs use mixed correction strategies; first subset requires one shared dynamic-fit strategy.",
        ))
    elif len(unique_strategies) == 1 and unique_strategies[0] in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
        allowed_dynamic_fit_strategy = unique_strategies[0]

    if (
        allowed_dynamic_fit_strategy
        and plan.dynamic_fit_parameter_contract.dynamic_fit_mode != allowed_dynamic_fit_strategy
    ):
        issues.append(_execution_subset_issue(
            "dynamic_fit_parameter_contract_mismatch",
            (
                "Dynamic-fit parameter contract mode "
                f"'{plan.dynamic_fit_parameter_contract.dynamic_fit_mode}' does not match "
                f"the unanimous selected per-ROI strategy '{allowed_dynamic_fit_strategy}'."
            ),
        ))

    field_classifications = _execution_field_classifications(plan)
    for field in field_classifications:
        if field.blocks_subset:
            issues.append(_execution_subset_issue(
                field.issue_category or f"{field.field_name}_blocks_execution_subset",
                f"Execution field '{field.field_name}' is {field.status}: {field.provenance}.",
            ))

    blocking = tuple(dict.fromkeys(issues))
    first_subset_executable = not blocking
    reason = (
        "First subset readiness is complete for future execution-spec preview; actual execution remains unavailable in this stage."
        if first_subset_executable
        else "; ".join(issue.category for issue in blocking)
    )
    return GuidedNewAnalysisExecutionSubsetReadiness(
        subset_name=FIRST_EXECUTION_SUBSET_NAME,
        subset_schema_version=EXECUTION_SUBSET_SCHEMA_VERSION,
        first_subset_executable=first_subset_executable,
        planning_complete_for_handoff=planning_readiness.plan_complete_for_handoff,
        execution_available=False,
        execution_blocked_reason=reason,
        allowed_dynamic_fit_strategy=allowed_dynamic_fit_strategy,
        blocking_issues=blocking,
        warning_issues=tuple(warnings),
        info_issues=tuple(info),
        field_classifications=field_classifications,
    )


def _preview_issue_from_plan_issue(issue: GuidedPlanIssue) -> GuidedNewAnalysisRunPreviewIssue:
    return GuidedNewAnalysisRunPreviewIssue(
        category=issue.category,
        message=issue.message,
        severity=issue.severity,
    )


def _dataset_contract_snapshot_preview_dict(
    snapshot: GuidedNewAnalysisDatasetContractSnapshot,
    *,
    execution_consumption_enabled: bool = False,
) -> dict[str, Any]:
    identity = snapshot.source_identity
    return {
        "schema_version": snapshot.schema_version,
        "status": snapshot.status,
        "current_applied": snapshot.current_applied,
        "explicitly_applied": snapshot.explicitly_applied,
        "input_format": snapshot.input_format,
        "resolved_input_format": snapshot.resolved_input_format,
        "acquisition_mode": snapshot.acquisition_mode,
        "validation_issues": list(snapshot.validation_issues),
        "stale_reasons": list(snapshot.stale_reasons),
        "source_identity": {
            "input_source_path": identity.input_source_path,
            "resolved_input_source_path": identity.resolved_input_source_path,
            "input_format": identity.input_format,
            "resolved_input_format": identity.resolved_input_format,
            "acquisition_mode": identity.acquisition_mode,
            "sessions_per_hour": identity.sessions_per_hour,
            "session_duration_sec": identity.session_duration_sec,
            "continuous_window_sec": identity.continuous_window_sec,
            "continuous_step_sec": identity.continuous_step_sec,
            "allow_partial_final_window": identity.allow_partial_final_window,
            "exclude_incomplete_final_rwd_chunk": identity.exclude_incomplete_final_rwd_chunk,
            "discovered_roi_ids": list(identity.discovered_roi_ids),
            "included_roi_ids": list(identity.included_roi_ids),
            "source_setup_signature": identity.source_setup_signature,
            "config_fingerprint": identity.config_fingerprint,
            "diagnostic_cache_contract_identity": identity.diagnostic_cache_contract_identity,
        },
        "contract_values": dict(snapshot.contract_values),
        "format_specific": dict(snapshot.format_specific),
        "provenance": {
            **dict(snapshot.provenance),
            "no_runspec": True,
            "no_argv": True,
            "no_config_written": True,
            "no_files_written": True,
        },
        "execution_consumption_enabled": execution_consumption_enabled,
    }


RWD_NORMALIZATION_REQUIRED_SNAPSHOT_FIELDS = (
    "rwd_time_col",
    "sig_suffix",
    "uv_suffix",
)
RWD_NORMALIZATION_BACKEND_CONFIG_FIELDS = {
    *RWD_NORMALIZATION_REQUIRED_SNAPSHOT_FIELDS,
    "exclude_incomplete_final_rwd_chunk",
}
RWD_NORMALIZATION_STRUCTURAL_FIELDS = {
    "input_format",
    "resolved_input_format",
    "acquisition_mode",
    "sessions_per_hour",
    "session_duration_sec",
    "allow_partial_final_window",
}
RWD_NORMALIZATION_PROVENANCE_FIELDS = {
    "structural_only",
    "no_file_inspection",
    "real_backend_contract_values_not_inferred",
    "rwd_contract_validation",
    "rwd_excluded_source_files",
}


def _rwd_normalization_consistency_record(
    field_name: str,
    plan_value: Any,
    snapshot_value: Any,
    snapshot_location: str,
) -> dict[str, Any]:
    return {
        "field_name": field_name,
        "plan_value": plan_value,
        "snapshot_value": snapshot_value,
        "snapshot_location": snapshot_location,
    }


def build_guided_rwd_dataset_contract_normalization_preview(
    plan: GuidedNewAnalysisDraftPlan,
) -> dict[str, Any]:
    """Normalize stored RWD/intermittent plan state without I/O or inference."""
    if not isinstance(plan, GuidedNewAnalysisDraftPlan):
        raise TypeError("plan must be a GuidedNewAnalysisDraftPlan")

    snapshot = plan.dataset_contract_snapshot
    identity = snapshot.source_identity
    contract_values = dict(snapshot.contract_values)
    format_specific = dict(snapshot.format_specific)
    combined_values = {**contract_values, **format_specific}

    backend_config_values: dict[str, Any] = {}
    structural_values = {
        "input_format": plan.input_format,
        "resolved_input_format": snapshot.resolved_input_format,
        "acquisition_mode": plan.acquisition_mode,
        "sessions_per_hour": plan.sessions_per_hour,
        "session_duration_sec": plan.session_duration_sec,
        "allow_partial_final_window": plan.allow_partial_final_window,
    }
    provenance_values: dict[str, Any] = {
        "source_identity": {
            "input_source_path": identity.input_source_path,
            "resolved_input_source_path": identity.resolved_input_source_path,
            "input_format": identity.input_format,
            "resolved_input_format": identity.resolved_input_format,
            "acquisition_mode": identity.acquisition_mode,
            "sessions_per_hour": identity.sessions_per_hour,
            "session_duration_sec": identity.session_duration_sec,
            "allow_partial_final_window": identity.allow_partial_final_window,
            "exclude_incomplete_final_rwd_chunk": identity.exclude_incomplete_final_rwd_chunk,
            "source_setup_signature": identity.source_setup_signature,
            "config_fingerprint": identity.config_fingerprint,
            "diagnostic_cache_contract_identity": identity.diagnostic_cache_contract_identity,
        }
    }
    for field_name in RWD_NORMALIZATION_PROVENANCE_FIELDS:
        if field_name in combined_values:
            provenance_values[field_name] = combined_values[field_name]

    missing_required_fields: list[str] = []
    unresolved_fields: list[str] = []
    for field_name in RWD_NORMALIZATION_REQUIRED_SNAPSHOT_FIELDS:
        value = combined_values.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing_required_fields.append(field_name)
            unresolved_fields.append(field_name)
        elif not isinstance(value, str) or not value.strip():
            unresolved_fields.append(field_name)
        else:
            backend_config_values[field_name] = value
    if isinstance(plan.exclude_incomplete_final_rwd_chunk, bool):
        backend_config_values["exclude_incomplete_final_rwd_chunk"] = (
            plan.exclude_incomplete_final_rwd_chunk
        )
    else:
        unresolved_fields.append("exclude_incomplete_final_rwd_chunk")

    inconsistent_fields: list[dict[str, Any]] = []
    plan_format = _normalized_format(plan.input_format)
    plan_acquisition = _normalized_format(plan.acquisition_mode)
    consistency_candidates = (
        ("input_format", plan_format, _normalized_format(snapshot.input_format), "snapshot.input_format"),
        (
            "resolved_input_format",
            plan_format,
            _normalized_format(snapshot.resolved_input_format),
            "snapshot.resolved_input_format",
        ),
        (
            "acquisition_mode",
            plan_acquisition,
            _normalized_format(snapshot.acquisition_mode),
            "snapshot.acquisition_mode",
        ),
    )
    for field_name, plan_value, snapshot_value, location in consistency_candidates:
        if snapshot_value and plan_value != snapshot_value:
            inconsistent_fields.append(
                _rwd_normalization_consistency_record(
                    field_name,
                    plan_value,
                    snapshot_value,
                    location,
                )
            )

    identity_candidates = (
        ("input_format", plan_format, _normalized_format(identity.input_format), "source_identity.input_format"),
        (
            "resolved_input_format",
            plan_format,
            _normalized_format(identity.resolved_input_format),
            "source_identity.resolved_input_format",
        ),
        (
            "acquisition_mode",
            plan_acquisition,
            _normalized_format(identity.acquisition_mode),
            "source_identity.acquisition_mode",
        ),
        (
            "exclude_incomplete_final_rwd_chunk",
            bool(plan.exclude_incomplete_final_rwd_chunk),
            identity.exclude_incomplete_final_rwd_chunk,
            "source_identity.exclude_incomplete_final_rwd_chunk",
        ),
    )
    for field_name, plan_value, snapshot_value, location in identity_candidates:
        if snapshot_value is not None and plan_value != snapshot_value:
            inconsistent_fields.append(
                _rwd_normalization_consistency_record(
                    field_name,
                    plan_value,
                    snapshot_value,
                    location,
                )
            )

    for source_name, values in (
        ("contract_values", contract_values),
        ("format_specific", format_specific),
    ):
        for field_name in (
            "input_format",
            "resolved_input_format",
            "acquisition_mode",
            "exclude_incomplete_final_rwd_chunk",
        ):
            if field_name not in values:
                continue
            expected = (
                plan.exclude_incomplete_final_rwd_chunk
                if field_name == "exclude_incomplete_final_rwd_chunk"
                else plan_acquisition
                if field_name == "acquisition_mode"
                else plan_format
            )
            actual = (
                values[field_name]
                if field_name == "exclude_incomplete_final_rwd_chunk"
                else _normalized_format(values[field_name])
            )
            if expected != actual:
                inconsistent_fields.append(
                    _rwd_normalization_consistency_record(
                        field_name,
                        expected,
                        actual,
                        f"snapshot.{source_name}.{field_name}",
                    )
                )

    duplicate_keys = set(contract_values).intersection(format_specific)
    for field_name in sorted(duplicate_keys):
        if contract_values[field_name] != format_specific[field_name]:
            inconsistent_fields.append(
                _rwd_normalization_consistency_record(
                    field_name,
                    contract_values[field_name],
                    format_specific[field_name],
                    "snapshot.contract_values_vs_format_specific",
                )
            )

    recognized_fields = (
        RWD_NORMALIZATION_BACKEND_CONFIG_FIELDS
        | RWD_NORMALIZATION_STRUCTURAL_FIELDS
        | RWD_NORMALIZATION_PROVENANCE_FIELDS
    )
    rejected_fields = {
        field_name: value
        for field_name, value in combined_values.items()
        if field_name not in recognized_fields
    }

    blocker_categories: list[str] = []
    if plan_format != "rwd":
        blocker_categories.append("unsupported_format_for_rwd_first_subset")
    if plan_acquisition != "intermittent":
        blocker_categories.append("unsupported_acquisition_mode_for_rwd_first_subset")
    if not snapshot.current_applied:
        blocker_categories.append("dataset_contract_snapshot_not_current")
    if _dataset_contract_snapshot_plan_consistency_reasons(plan, snapshot):
        blocker_categories.append("inconsistent_rwd_contract_field")
    if inconsistent_fields:
        blocker_categories.append("inconsistent_rwd_contract_field")
    if missing_required_fields:
        blocker_categories.append("missing_required_rwd_contract_field")
    if unresolved_fields:
        blocker_categories.append("unresolved_rwd_dataset_contract_normalization")
    blocker_categories = list(dict.fromkeys(blocker_categories))

    if "unsupported_format_for_rwd_first_subset" in blocker_categories:
        normalization_status = "unsupported_format_for_rwd_first_subset"
        mapping_status = normalization_status
    elif "unsupported_acquisition_mode_for_rwd_first_subset" in blocker_categories:
        normalization_status = "unsupported_acquisition_mode_for_rwd_first_subset"
        mapping_status = normalization_status
    elif "dataset_contract_snapshot_not_current" in blocker_categories:
        normalization_status = "dataset_contract_snapshot_not_current"
        mapping_status = normalization_status
    elif "inconsistent_rwd_contract_field" in blocker_categories:
        normalization_status = "inconsistent_rwd_contract_field"
        mapping_status = "rwd_dataset_contract_inconsistent"
    elif blocker_categories:
        normalization_status = "unresolved_rwd_dataset_contract_normalization"
        mapping_status = "rwd_dataset_contract_unresolved"
    else:
        normalization_status = "ready_for_future_mapping"
        mapping_status = "rwd_dataset_contract_ready_for_future_mapping"

    return {
        "normalization_status": normalization_status,
        "backend_config_values": backend_config_values,
        "structural_values": structural_values,
        "provenance_values": provenance_values,
        "rejected_fields": rejected_fields,
        "missing_required_fields": missing_required_fields,
        "inconsistent_fields": inconsistent_fields,
        "unresolved_fields": unresolved_fields,
        "blocker_categories": blocker_categories,
        "execution_consumption_enabled": normalization_status == "ready_for_future_mapping",
        "backend_config_mapping_status": mapping_status,
        "supported_subset": {
            "input_format": "rwd",
            "acquisition_mode": "intermittent",
        },
        "no_file_inspection": True,
        "no_runspec": True,
        "no_argv": True,
        "no_config_written": True,
        "no_files_written": True,
    }


def _execution_intent_preview_dict(
    intent: GuidedNewAnalysisExecutionIntent,
    *,
    execution_consumption_enabled: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": intent.schema_version,
        "timeline_anchor_mode": intent.timeline_anchor_mode,
        "fixed_daily_anchor_clock": intent.fixed_daily_anchor_clock,
        "execution_mode": intent.execution_mode,
        "run_profile": intent.run_profile,
        "provenance": dict(intent.provenance),
        "execution_consumption_enabled": execution_consumption_enabled,
        "no_runspec": True,
        "no_argv": True,
        "no_config_written": True,
        "no_files_written": True,
    }


def _output_creation_policy_preview_dict(
    policy: GuidedNewAnalysisOutputCreationPolicy,
    *,
    execution_consumption_enabled: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": policy.schema_version,
        "path_role": policy.path_role,
        "creation_timing": policy.creation_timing,
        "run_directory_strategy": policy.run_directory_strategy,
        "overwrite": policy.overwrite,
        "precreate_during_preview": policy.precreate_during_preview,
        "config_write_timing": policy.config_write_timing,
        "gui_preflight_writes_enabled": policy.gui_preflight_writes_enabled,
        "provenance": dict(policy.provenance),
        "execution_consumption_enabled": execution_consumption_enabled,
        "directory_created": False,
        "files_written": False,
        "no_runspec": True,
        "no_argv": True,
        "no_config_written": True,
    }


def _feature_event_profile_current_for_first_subset(plan: GuidedNewAnalysisDraftPlan) -> bool:
    return (
        plan.feature_event_profile_status == "applied"
        and bool(plan.feature_event_explicitly_applied)
        and not plan.feature_event_validation_issues
        and not plan.feature_event_stale_reasons
    )


def _feature_event_field_activity(field_name: str, method: str) -> str:
    active_threshold_field = FEATURE_EVENT_THRESHOLD_PARAMETER_FIELDS.get(method)
    if field_name in {"peak_threshold_k", "peak_threshold_percentile", "peak_threshold_abs"}:
        if active_threshold_field is None:
            return "unresolved"
        if field_name == active_threshold_field:
            return "active"
        return "inactive_for_threshold_method"
    return "active"


def build_guided_feature_event_effective_values_preview(plan: GuidedNewAnalysisDraftPlan) -> dict[str, Any]:
    """Build a pure feature/event effective-value preview from stored plan state.

    The merge consumes only GuidedNewAnalysisDraftPlan feature/event fields plus
    canonical backend Config defaults. It does not inspect GUI widgets, generate
    config, instantiate RunSpec, write files, validate, run, or mutate the plan.
    """
    if not isinstance(plan, GuidedNewAnalysisDraftPlan):
        raise TypeError("plan must be a GuidedNewAnalysisDraftPlan")

    backend_defaults = dict(_CANONICAL_FEATURE_EVENT_BACKEND_DEFAULTS)
    applied_values = {
        key: value for key, value in dict(plan.feature_event_values).items()
        if key in FEATURE_EVENT_CONFIG_FIELDS
    }
    ignored_applied_fields = sorted(set(plan.feature_event_values) - FEATURE_EVENT_CONFIG_FIELDS)
    effective: dict[str, Any] = {}
    effective_values: list[dict[str, Any]] = []
    default_values: dict[str, Any] = {}
    unresolved_fields: list[str] = []
    inactive_fields: list[str] = []
    field_activity: dict[str, str] = {}

    profile_current = _feature_event_profile_current_for_first_subset(plan)
    method_value = applied_values.get(
        "peak_threshold_method",
        backend_defaults.get("peak_threshold_method"),
    )
    method = str(method_value or "")
    unsupported_threshold = method not in FEATURE_EVENT_THRESHOLD_METHODS

    for field_name in FEATURE_EVENT_EFFECTIVE_VALUE_FIELDS:
        if field_name in applied_values:
            value = applied_values[field_name]
            source = "applied_guided_profile"
            provenance = "stored GuidedNewAnalysisDraftPlan.feature_event_values"
        elif field_name in backend_defaults:
            value = backend_defaults[field_name]
            source = "backend_config_default"
            default_values[field_name] = value
            provenance = FEATURE_EVENT_BACKEND_DEFAULT_PROVENANCE
        else:
            value = None
            source = "unresolved"
            provenance = "no applied Guided value and no canonical backend Config default"
            unresolved_fields.append(field_name)

        activity = _feature_event_field_activity(field_name, method)
        if activity == "unresolved":
            unresolved_fields.append(field_name)
        elif activity != "active":
            inactive_fields.append(field_name)
        field_activity[field_name] = activity

        effective[field_name] = value
        effective_values.append({
            "field_name": field_name,
            "effective_value": value,
            "source": source,
            "consumed_by_first_subset": bool(profile_current and activity == "active" and source != "unresolved"),
            "active_or_inactive": activity,
            "provenance": provenance,
        })

    validation_payload = {
        field_name: value for field_name, value in effective.items()
        if field_activity.get(field_name) == "active"
    }
    validation_errors = validate_feature_event_config_fields(validation_payload)
    unresolved_unique = tuple(dict.fromkeys(unresolved_fields))
    inactive_unique = tuple(dict.fromkeys(inactive_fields))
    blocker_categories: list[str] = []
    if not profile_current:
        blocker_categories.append("feature_event_profile_not_current")
    if unsupported_threshold:
        blocker_categories.append("unsupported_threshold_method")
    if validation_errors:
        blocker_categories.append("invalid_feature_event_effective_values")
    if unresolved_unique:
        blocker_categories.append("unresolved_feature_event_effective_values")

    if "unsupported_threshold_method" in blocker_categories:
        mapping_status = "unsupported_threshold_method"
    elif "feature_event_profile_not_current" in blocker_categories:
        mapping_status = "feature_event_profile_not_current"
    elif unresolved_unique or validation_errors:
        mapping_status = "effective_values_unresolved"
    else:
        mapping_status = "effective_values_ready_for_future_mapping"

    return {
        "effective_values": effective_values,
        "applied_values": applied_values,
        "default_values": default_values,
        "unresolved_fields": list(unresolved_unique),
        "inactive_fields": list(inactive_unique),
        "ignored_applied_fields": ignored_applied_fields,
        "validation_errors": validation_errors,
        "backend_default_source": FEATURE_EVENT_BACKEND_DEFAULT_SOURCE,
        "backend_default_values": backend_defaults,
        "threshold_method": method,
        "threshold_activity": {
            "method": method,
            "active_threshold_field": FEATURE_EVENT_THRESHOLD_PARAMETER_FIELDS.get(method),
            "supported": not unsupported_threshold,
        },
        "execution_consumption_enabled": (
            mapping_status == "effective_values_ready_for_future_mapping"
            and profile_current
        ),
        "backend_config_mapping_status": mapping_status,
        "blocker_categories": list(dict.fromkeys(blocker_categories)),
        "no_runspec": True,
        "no_argv": True,
        "no_config_written": True,
        "no_files_written": True,
    }


def _feature_event_consumption_preview_dict(plan: GuidedNewAnalysisDraftPlan) -> dict[str, Any]:
    traces_only = False
    first_subset_contract = (
        plan.execution_intent.execution_mode == "phasic"
        and plan.execution_intent.run_profile == "full"
        and traces_only is False
    )
    current_profile = _feature_event_profile_current_for_first_subset(plan)
    effective_preview = build_guided_feature_event_effective_values_preview(plan)
    consumption_enabled = bool(
        first_subset_contract
        and current_profile
        and effective_preview["backend_config_mapping_status"] == "effective_values_ready_for_future_mapping"
    )
    return {
        "execution_mode": plan.execution_intent.execution_mode,
        "run_profile": plan.execution_intent.run_profile,
        "traces_only": traces_only,
        "feature_event_profile_required": True,
        "feature_event_profile_current_applied": current_profile,
        "feature_event_values_consumed": consumption_enabled,
        "feature_extraction_in_scope": bool(first_subset_contract),
        "feature_dependent_phasic_summaries_in_scope": bool(first_subset_contract),
        "tonic_outputs_in_scope": False,
        "full_both_mode_outputs_in_scope": False,
        "execution_consumption_enabled": consumption_enabled,
        "effective_values_preview": effective_preview,
        "provenance": (
            "first subset phasic full execution preview includes phasic feature extraction "
            "and feature-dependent summaries"
        ),
        "no_runspec": True,
        "no_argv": True,
        "no_config_written": True,
        "no_files_written": True,
    }


def _section_snapshot(section: GuidedNewAnalysisSectionReadiness) -> dict[str, Any]:
    return {
        "key": section.key,
        "label": section.label,
        "status": section.status,
        "blocking_categories": [issue.category for issue in section.blocking_issues],
        "warning_categories": [issue.category for issue in section.warning_issues],
        "info_categories": [issue.category for issue in section.info_issues],
    }


def _correction_strategy_run_preview_unresolved_items(
    plan: GuidedNewAnalysisDraftPlan,
) -> tuple[GuidedNewAnalysisRunPreviewIssue, ...]:
    choices = tuple(plan.per_roi_correction_strategy_choices)
    if not choices:
        return ()

    unresolved: list[GuidedNewAnalysisRunPreviewIssue] = []
    choices_by_roi: dict[str, GuidedPlanCorrectionChoice] = {}
    counts_by_roi: dict[str, int] = {}
    for choice in choices:
        counts_by_roi[choice.roi_id] = counts_by_roi.get(choice.roi_id, 0) + 1
        choices_by_roi[choice.roi_id] = choice

    included_choices = [
        choices_by_roi[roi]
        for roi in plan.included_roi_ids
        if roi in choices_by_roi
    ]

    if len(included_choices) != len(plan.included_roi_ids):
        unresolved.append(GuidedNewAnalysisRunPreviewIssue(
            category="missing_strategy_choice_for_execution_preview",
            message="Every included ROI needs one explicit strategy choice before execution preview mapping can be resolved.",
            severity="blocking",
        ))

    for roi in plan.included_roi_ids:
        if counts_by_roi.get(roi, 0) > 1:
            unresolved.append(GuidedNewAnalysisRunPreviewIssue(
                category="duplicate_strategy_choice_for_execution_preview",
                message=f"Included ROI '{roi}' has duplicate strategy choices.",
                severity="blocking",
            ))

    selected = [choice.selected_strategy for choice in included_choices]
    unique_strategies = tuple(dict.fromkeys(selected))
    if len(unique_strategies) > 1:
        unresolved.append(GuidedNewAnalysisRunPreviewIssue(
            category="mixed_per_roi_strategies",
            message="Included ROIs use mixed correction strategies; first subset requires one shared dynamic-fit strategy.",
            severity="blocking",
        ))

    for choice in included_choices:
        if not choice.explicit_user_mark:
            unresolved.append(GuidedNewAnalysisRunPreviewIssue(
                category="non_explicit_strategy_choice",
                message=f"Strategy choice for ROI '{choice.roi_id}' is not explicitly user-marked.",
                severity="blocking",
            ))
        if choice.current_or_stale != "current":
            unresolved.append(GuidedNewAnalysisRunPreviewIssue(
                category="stale_strategy_choice",
                message=f"Strategy choice for ROI '{choice.roi_id}' is not current.",
                severity="blocking",
            ))
        if choice.selected_strategy in FORBIDDEN_CORRECTION_STRATEGIES:
            unresolved.append(GuidedNewAnalysisRunPreviewIssue(
                category="forbidden_strategy_state",
                message=f"Strategy '{choice.selected_strategy}' for ROI '{choice.roi_id}' is forbidden for execution.",
                severity="blocking",
            ))
        elif choice.selected_strategy == "signal_only_f0":
            unresolved.append(GuidedNewAnalysisRunPreviewIssue(
                category="signal_only_f0_production_routing_unresolved",
                message="Signal-Only F0 production routing is not implemented for Guided new_analysis.",
                severity="blocking",
            ))
        elif choice.selected_strategy not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
            unresolved.append(GuidedNewAnalysisRunPreviewIssue(
                category="unsupported_dynamic_fit_strategy_for_first_subset",
                message=f"Strategy '{choice.selected_strategy}' is not supported by the first execution subset.",
                severity="blocking",
            ))

    return tuple(dict.fromkeys(unresolved))


def build_guided_new_analysis_run_preview(plan: GuidedNewAnalysisDraftPlan) -> GuidedNewAnalysisRunPreview:
    """Build a pure, non-executing preview contract from Guided new_analysis plan state.

    This function does not create directories, write files, generate argv, instantiate
    GUI RunSpec, inspect widgets, or read the filesystem. Ordinary incomplete plans
    still return a preview with unresolved_items populated.
    """
    if not isinstance(plan, GuidedNewAnalysisDraftPlan):
        raise TypeError("plan must be a GuidedNewAnalysisDraftPlan")

    readiness = evaluate_new_analysis_plan_readiness(plan)
    unresolved: list[GuidedNewAnalysisRunPreviewIssue] = [
        _preview_issue_from_plan_issue(issue) for issue in readiness.blocking_issues
    ]
    warnings: list[GuidedNewAnalysisRunPreviewIssue] = [
        _preview_issue_from_plan_issue(issue) for issue in readiness.warning_issues
    ]

    choices = tuple(plan.per_roi_correction_strategy_choices)
    unresolved.extend(_correction_strategy_run_preview_unresolved_items(plan))

    fixed_contract_defaults = {
        "execution_available": False,
        "execution_backend": "unavailable_in_this_stage",
    }

    execution_fields = _execution_field_classifications(plan)
    dataset_contract_consumed = any(
        field.field_name in {"dataset_contract_overrides", "npm_channel_mapping", "custom_tabular_column_mapping"}
        and field.status == "present"
        and not field.blocks_subset
        for field in execution_fields
    )
    execution_intent_consumed = all(
        any(field.field_name == name and field.status == "fixed_default" and not field.blocks_subset for field in execution_fields)
        for name in ("timeline_anchor_mode", "mode", "run_profile")
    )
    output_creation_policy_consumed = any(
        field.field_name == "output_creation_policy"
        and field.status == "present"
        and not field.blocks_subset
        for field in execution_fields
    )

    return GuidedNewAnalysisRunPreview(
        preview_schema_version=RUN_PREVIEW_SCHEMA_VERSION,
        plan_schema_version=plan.schema_version,
        source={
            "input_source_path": plan.input_source_path,
            "resolved_input_source_path": plan.resolved_input_source_path,
            "authoritative_input_source_path": plan.resolved_input_source_path or plan.input_source_path,
            "input_format": plan.input_format,
        },
        acquisition={
            "acquisition_mode": plan.acquisition_mode,
            "sessions_per_hour": plan.sessions_per_hour,
            "session_duration_sec": plan.session_duration_sec,
            "continuous_window_sec": plan.continuous_window_sec,
            "continuous_step_sec": plan.continuous_step_sec,
            "allow_partial_final_window": plan.allow_partial_final_window,
            "exclude_incomplete_final_rwd_chunk": plan.exclude_incomplete_final_rwd_chunk,
            "acquisition_structure_status": plan.acquisition_structure_status,
            "timeline_anchor_mode": {
                "status": "represented",
                "value": plan.execution_intent.timeline_anchor_mode,
                "fixed_daily_anchor_clock": plan.execution_intent.fixed_daily_anchor_clock,
                "source": "GuidedNewAnalysisDraftPlan.execution_intent",
            },
        },
        execution_intent=_execution_intent_preview_dict(
            plan.execution_intent,
            execution_consumption_enabled=execution_intent_consumed,
        ),
        dataset_contract=_dataset_contract_snapshot_preview_dict(
            plan.dataset_contract_snapshot,
            execution_consumption_enabled=dataset_contract_consumed,
        ),
        roi_selection={
            "discovered_roi_ids": list(plan.discovered_roi_ids),
            "included_roi_ids": list(plan.included_roi_ids),
            "excluded_roi_ids": list(plan.excluded_roi_ids),
            "execution_roi_filter": {
                "mode": "include",
                "roi_ids": list(plan.included_roi_ids),
            },
        },
        diagnostic_cache={
            "cache_id": plan.cache_id,
            "cache_root_path": plan.cache_root_path,
            "artifact_record_path": plan.artifact_record_path,
            "request_json_path": plan.request_json_path,
            "provenance_path": plan.provenance_path,
            "phasic_trace_cache_path": plan.phasic_trace_cache_path,
            "config_used_path": plan.config_used_path,
            "source_setup_signature": plan.source_setup_signature,
            "diagnostic_scope_signature": plan.diagnostic_scope_signature,
            "build_request_signature": plan.build_request_signature,
            "stale_or_current": plan.stale_or_current,
            "stale_reasons": list(plan.stale_reasons),
            "execution_consumes_cache_artifacts": False,
        },
        correction_strategy={
            "per_roi_choices": [
                {
                    "roi_id": choice.roi_id,
                    "selected_strategy": choice.selected_strategy,
                    "source_type": choice.source_type,
                    "diagnostic_cache_id": choice.diagnostic_cache_id,
                    "diagnostic_cache_root": choice.diagnostic_cache_root,
                    "diagnostic_cache_signature": choice.diagnostic_cache_signature,
                    "source_setup_signature": choice.source_setup_signature,
                    "diagnostic_scope_signature": choice.diagnostic_scope_signature,
                    "build_request_signature": choice.build_request_signature,
                    "evidence_chunk": choice.evidence_chunk,
                    "evidence_summary": choice.evidence_summary,
                    "current_or_stale": choice.current_or_stale,
                    "explicit_user_mark": choice.explicit_user_mark,
                    "selected_at_utc": choice.selected_at_utc,
                }
                for choice in choices
            ],
            "execution_mapping_status": "unresolved" if choices else "no_choices",
            "global_strategy_collapsed": False,
        },
        evidence_references={
            "correction_preview": {
                "result_id": plan.correction_preview_result_id,
                "path": plan.correction_preview_path,
                "status": plan.correction_preview_status,
                "source_cache_id": plan.correction_preview_source_cache_id,
            },
            "signal_only_f0": {
                "result_id": plan.signal_only_f0_result_id,
                "path": plan.signal_only_f0_path,
                "status": plan.signal_only_f0_status,
                "source_cache_id": plan.signal_only_f0_source_cache_id,
            },
            "selected_evidence_context": dict(plan.selected_evidence_context),
            "execution_input": False,
        },
        feature_event={
            "status": plan.feature_event_profile_status,
            "profile_id": plan.feature_event_profile_id,
            "values": dict(plan.feature_event_values),
            "validation_issues": list(plan.feature_event_validation_issues),
            "stale_reasons": list(plan.feature_event_stale_reasons),
            "explicitly_applied": plan.feature_event_explicitly_applied,
            "baseline_config_source": plan.feature_event_baseline_config_source,
            "baseline_status": plan.feature_event_baseline_status,
            "updated_at_utc": plan.feature_event_updated_at_utc,
        },
        feature_event_consumption=_feature_event_consumption_preview_dict(plan),
        output_policy={
            "status": plan.output_policy_status,
            "path": plan.output_policy_path,
            "validation_issues": list(plan.output_policy_validation_issues),
            "stale_reasons": list(plan.output_policy_stale_reasons),
            "updated_at_utc": plan.output_policy_updated_at_utc,
            "explicitly_applied": plan.output_policy_explicitly_applied,
            "safety_summary": plan.output_policy_safety_summary,
            "directory_created": False,
            "files_written": False,
        },
        output_creation_policy=_output_creation_policy_preview_dict(
            plan.output_creation_policy,
            execution_consumption_enabled=output_creation_policy_consumed,
        ),
        provenance={
            "plan_created_at_utc": plan.created_at_utc,
            "plan_updated_at_utc": plan.updated_at_utc,
            "production_analysis": plan.production_analysis,
            "preliminary_cache": plan.preliminary_cache,
            "fixed_contract_defaults": {
                **fixed_contract_defaults,
                "execution_intent": _execution_intent_preview_dict(
                    plan.execution_intent,
                    execution_consumption_enabled=execution_intent_consumed,
                ),
                "output_creation_policy": _output_creation_policy_preview_dict(
                    plan.output_creation_policy,
                    execution_consumption_enabled=output_creation_policy_consumed,
                ),
            },
            "no_gui_runspec": True,
            "no_argv_generated": True,
            "no_config_written": True,
            "no_output_directory_created": True,
        },
        readiness_snapshot={
            "plan_complete_for_handoff": readiness.plan_complete_for_handoff,
            "execution_available": readiness.execution_available,
            "execution_blocked_reason": readiness.execution_blocked_reason,
            "sections": [_section_snapshot(section) for section in readiness.sections],
            "blocking_categories": [issue.category for issue in readiness.blocking_issues],
            "warning_categories": [issue.category for issue in readiness.warning_issues],
            "info_categories": [issue.category for issue in readiness.info_issues],
        },
        unresolved_items=tuple(unresolved),
        warnings=tuple(warnings),
        execution_available=False,
        execution_blocked_reason=readiness.execution_blocked_reason,
    )


_CORRECTION_EXECUTION_SPEC_BLOCKERS = {
    "missing_strategy_choice_for_execution_subset",
    "duplicate_strategy_choice_for_execution_subset",
    "non_explicit_strategy_choice",
    "forbidden_strategy_state",
    "signal_only_f0_execution_not_supported",
    "unsupported_dynamic_fit_strategy_for_first_subset",
    "mixed_per_roi_strategies",
    "dynamic_fit_parameter_contract_mismatch",
    "unresolved_dynamic_fit_parameter_contract",
    "planning_missing_strategy_choice_for_included_roi",
    "planning_stale_strategy_choice",
    "planning_forbidden_strategy",
}


def _per_roi_choice_preview_dict(choice: GuidedPlanCorrectionChoice) -> dict[str, Any]:
    return {
        "roi_id": choice.roi_id,
        "selected_strategy": choice.selected_strategy,
        "source_type": choice.source_type,
        "diagnostic_cache_id": choice.diagnostic_cache_id,
        "diagnostic_cache_root": choice.diagnostic_cache_root,
        "diagnostic_cache_signature": choice.diagnostic_cache_signature,
        "source_setup_signature": choice.source_setup_signature,
        "diagnostic_scope_signature": choice.diagnostic_scope_signature,
        "build_request_signature": choice.build_request_signature,
        "evidence_chunk": choice.evidence_chunk,
        "evidence_summary": choice.evidence_summary,
        "current_or_stale": choice.current_or_stale,
        "explicit_user_mark": choice.explicit_user_mark,
        "selected_at_utc": choice.selected_at_utc,
    }


def _execution_spec_correction_preview_dict(
    plan: GuidedNewAnalysisDraftPlan,
    subset_readiness: GuidedNewAnalysisExecutionSubsetReadiness,
) -> dict[str, Any]:
    issue_categories = tuple(issue.category for issue in subset_readiness.blocking_issues)
    correction_blockers = tuple(
        category for category in issue_categories
        if category in _CORRECTION_EXECUTION_SPEC_BLOCKERS
    )
    selected_global_strategy = (
        subset_readiness.allowed_dynamic_fit_strategy
        if subset_readiness.allowed_dynamic_fit_strategy and not correction_blockers
        else None
    )
    contract_preview = _dynamic_fit_parameter_contract_preview_dict(
        plan.dynamic_fit_parameter_contract,
        selected_strategy=subset_readiness.allowed_dynamic_fit_strategy,
        issue_categories=issue_categories,
        execution_consumption_enabled=bool(selected_global_strategy),
    )
    return {
        "selected_global_dynamic_fit_strategy": selected_global_strategy,
        "global_strategy_derivation": (
            "unanimous_explicit_per_roi_choices"
            if selected_global_strategy
            else "unresolved"
        ),
        "global_strategy_collapsed": False,
        "per_roi_choices": [
            _per_roi_choice_preview_dict(choice)
            for choice in plan.per_roi_correction_strategy_choices
        ],
        "per_roi_choice_provenance_preserved": True,
        "signal_only_f0_production_routing_supported": False,
        "mixed_strategy_supported": False,
        "blocker_categories": list(correction_blockers),
        "dynamic_fit_parameter_contract": contract_preview,
    }


def _dynamic_fit_parameter_contract_preview_dict(
    contract: GuidedNewAnalysisDynamicFitParameterContract,
    *,
    selected_strategy: str | None,
    issue_categories: tuple[str, ...] = (),
    execution_consumption_enabled: bool = False,
) -> dict[str, Any]:
    robust_parameters = {
        "robust_event_reject_max_iters": contract.robust_event_reject_max_iters,
        "robust_event_reject_residual_z_thresh": contract.robust_event_reject_residual_z_thresh,
        "robust_event_reject_local_var_window_sec": contract.robust_event_reject_local_var_window_sec,
        "robust_event_reject_local_var_ratio_thresh": contract.robust_event_reject_local_var_ratio_thresh,
        "robust_event_reject_min_keep_fraction": contract.robust_event_reject_min_keep_fraction,
    }
    adaptive_parameters = {
        "adaptive_event_gate_residual_z_thresh": contract.adaptive_event_gate_residual_z_thresh,
        "adaptive_event_gate_local_var_window_sec": contract.adaptive_event_gate_local_var_window_sec,
        "adaptive_event_gate_local_var_ratio_thresh": contract.adaptive_event_gate_local_var_ratio_thresh,
        "adaptive_event_gate_smooth_window_sec": contract.adaptive_event_gate_smooth_window_sec,
        "adaptive_event_gate_min_trust_fraction": contract.adaptive_event_gate_min_trust_fraction,
        "adaptive_event_gate_freeze_interp_method": contract.adaptive_event_gate_freeze_interp_method,
    }
    legacy_global_parameters = {
        "window_sec": contract.window_sec,
        "step_sec": contract.step_sec,
        "r_low": contract.r_low,
        "r_high": contract.r_high,
        "g_min": contract.g_min,
        "min_samples_per_window": contract.min_samples_per_window,
        "min_valid_windows": contract.min_valid_windows,
    }
    common_parameters = {
        "slope_constraint": contract.slope_constraint,
        "min_slope": contract.min_slope,
    }
    if "dynamic_fit_parameter_contract_mismatch" in issue_categories:
        mapping_status = "contract_mismatch"
    elif contract.dynamic_fit_mode not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
        mapping_status = "unsupported_strategy"
    elif contract.unresolved_parameters:
        mapping_status = "label_ready_parameters_unresolved"
    else:
        mapping_status = "label_and_parameters_ready_for_future_mapping"

    active_parameter_set = "global_linear_regression"
    if contract.dynamic_fit_mode == "robust_global_event_reject":
        active_parameter_set = "robust_event_rejection"
    elif contract.dynamic_fit_mode == "adaptive_event_gated_regression":
        active_parameter_set = "adaptive_event_gate"

    inactive_parameter_sets: dict[str, dict[str, Any]] = {}
    if active_parameter_set != "robust_event_rejection":
        inactive_parameter_sets["robust_event_rejection"] = {
            "status": "inactive_parameter_defaults",
            "parameters": robust_parameters,
        }
    if active_parameter_set != "adaptive_event_gate":
        inactive_parameter_sets["adaptive_event_gate"] = {
            "status": "inactive_parameter_defaults",
            "parameters": adaptive_parameters,
        }
    if active_parameter_set != "global_linear_regression":
        inactive_parameter_sets["global_linear_regression_legacy_defaults"] = {
            "status": "inactive_parameter_defaults",
            "parameters": legacy_global_parameters,
        }

    active_parameters = dict(common_parameters)
    if active_parameter_set == "robust_event_rejection":
        active_parameters.update(robust_parameters)
        active_parameters["legacy_global_defaults"] = legacy_global_parameters
    elif active_parameter_set == "adaptive_event_gate":
        active_parameters.update(adaptive_parameters)
        active_parameters["legacy_global_defaults"] = legacy_global_parameters
    else:
        active_parameters.update(legacy_global_parameters)

    return {
        "schema_version": contract.schema_version,
        "dynamic_fit_mode": contract.dynamic_fit_mode,
        "selected_strategy": selected_strategy,
        "active_parameter_set": active_parameter_set,
        "active_parameters": active_parameters,
        "inactive_parameter_sets": inactive_parameter_sets,
        "unresolved_parameters": list(contract.unresolved_parameters),
        "backend_default_source": BACKEND_DYNAMIC_FIT_DEFAULT_SOURCE,
        "backend_default_values": dict(_CANONICAL_DYNAMIC_FIT_BACKEND_DEFAULTS),
        "provenance": dict(contract.provenance),
        "execution_consumption_enabled": execution_consumption_enabled,
        "backend_config_mapping_status": mapping_status,
        "no_runspec": True,
        "no_argv": True,
        "no_config_written": True,
        "no_files_written": True,
    }


def _output_safety_normalized_path(value: str | None) -> str:
    return _output_safety_parse_path(value)["normalized_path"]


def _output_safety_parse_path(value: str | None) -> dict[str, Any]:
    raw = str(value or "").strip()
    if not raw:
        return {
            "raw_path": raw,
            "path_style": "missing",
            "is_absolute": False,
            "anchor": "",
            "parts": (),
            "normalized_path": "",
        }
    text = raw.replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    path_style = "relative"
    anchor = ""
    remainder = text
    is_absolute = False
    if len(text) >= 3 and text[1] == ":" and text[0].isalpha() and text[2] == "/":
        path_style = "windows_drive"
        anchor = f"{text[0].lower()}:"
        remainder = text[3:]
        is_absolute = True
    elif text.startswith("/"):
        path_style = "posix"
        anchor = "/"
        remainder = text[1:]
        is_absolute = True

    parts: list[str] = []
    for raw_part in remainder.split("/"):
        if raw_part in ("", "."):
            continue
        if raw_part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(raw_part.lower() if path_style == "windows_drive" else raw_part)

    if not is_absolute:
        normalized = ""
    elif path_style == "windows_drive":
        normalized = f"{anchor}/" + "/".join(parts) if parts else f"{anchor}/"
    else:
        normalized = "/" + "/".join(parts) if parts else "/"

    return {
        "raw_path": raw,
        "path_style": path_style,
        "is_absolute": is_absolute,
        "anchor": anchor,
        "parts": tuple(parts),
        "normalized_path": normalized,
    }


def _output_safety_same_or_child(path_info: dict[str, Any], root_info: dict[str, Any]) -> bool | None:
    if not path_info.get("is_absolute") or not root_info.get("is_absolute"):
        return None
    if path_info.get("path_style") != root_info.get("path_style"):
        return None
    if path_info.get("anchor") != root_info.get("anchor"):
        return False
    path_parts = tuple(path_info.get("parts") or ())
    root_parts = tuple(root_info.get("parts") or ())
    return path_parts[:len(root_parts)] == root_parts


def classify_output_base_safety_ownership(
    *,
    output_base: str | None,
    source_path: str | None,
    output_policy_status: str,
    output_policy_explicitly_applied: bool,
    output_policy_validation_issues: tuple[str, ...] = (),
    output_policy_stale_reasons: tuple[str, ...] = (),
    path_role: str = "output_base",
    run_directory_strategy: str = "derive_unique_run_id_under_output_base",
    overwrite_requested: bool | None = False,
    precreate_during_preview: bool = False,
    protected_roots: tuple[tuple[str, str], ...] = (),
    protected_root_context_complete: bool = False,
    filesystem_facts: dict[str, Any] | None = None,
    write_context: str = "planning_preview",
) -> dict[str, Any]:
    """Classify output-base safety without filesystem I/O or side effects."""
    raw_output_base = str(output_base or "").strip()
    raw_source_path = str(source_path or "").strip()
    output_path_info = _output_safety_parse_path(raw_output_base)
    source_path_info = _output_safety_parse_path(raw_source_path)
    output_is_absolute = bool(output_path_info["is_absolute"])
    source_is_absolute = bool(source_path_info["is_absolute"])
    normalized_output = output_path_info["normalized_path"]
    normalized_source = source_path_info["normalized_path"]

    blockers: list[str] = []
    relationships: list[dict[str, Any]] = []

    def add_relationship(
        name: str,
        status: str,
        *,
        evidence: dict[str, Any],
        blocker_category: str | None = None,
    ) -> None:
        relationships.append({
            "relationship": name,
            "status": status,
            "evidence": evidence,
            "blocker_category": blocker_category,
        })
        if blocker_category:
            blockers.append(blocker_category)

    policy_current = (
        output_policy_status == "applied"
        and output_policy_explicitly_applied
        and not output_policy_validation_issues
        and not output_policy_stale_reasons
    )
    if not policy_current:
        blockers.append("output_policy_not_current")
    if not raw_output_base:
        blockers.append("output_base_missing")
    elif not output_is_absolute:
        blockers.append("output_base_relative")

    if normalized_output and normalized_source:
        styles_match = (
            output_path_info["path_style"] == source_path_info["path_style"]
            and output_path_info["anchor"] == source_path_info["anchor"]
        )
        evidence = {
            "output_base": normalized_output,
            "source_path": normalized_source,
            "output_path_style": output_path_info["path_style"],
            "source_path_style": source_path_info["path_style"],
        }
        if not styles_match:
            for name in (
                "output_base_equals_source",
                "output_base_inside_source",
                "source_inside_output_base",
            ):
                add_relationship(
                    name,
                    "unknown_mixed_path_style",
                    evidence=evidence,
                    blocker_category="output_path_style_mismatch",
                )
        elif normalized_output == normalized_source:
            add_relationship(
                "output_base_equals_source",
                "unsafe",
                evidence=evidence,
                blocker_category="unsafe_source_output_relationship",
            )
        else:
            add_relationship(
                "output_base_equals_source",
                "safe",
                evidence=evidence,
            )

        if styles_match:
            output_inside_source = _output_safety_same_or_child(output_path_info, source_path_info)
            add_relationship(
                "output_base_inside_source",
                "unsafe" if output_inside_source else "safe",
                evidence=evidence,
                blocker_category=(
                    "unsafe_source_output_relationship"
                    if output_inside_source
                    else None
                ),
            )
            source_inside_output = _output_safety_same_or_child(source_path_info, output_path_info)
            add_relationship(
                "source_inside_output_base",
                "unsafe" if source_inside_output else "safe",
                evidence=evidence,
                blocker_category=(
                    "unsafe_source_output_relationship"
                    if source_inside_output
                    else None
                ),
            )
    else:
        relationship_status = "not_applicable" if not raw_source_path else "unknown"
        evidence = {
            "output_base": raw_output_base,
            "source_path": raw_source_path,
            "output_path_style": output_path_info["path_style"],
            "source_path_style": source_path_info["path_style"],
        }
        for name in (
            "output_base_equals_source",
            "output_base_inside_source",
            "source_inside_output_base",
        ):
            add_relationship(
                name,
                relationship_status,
                evidence=evidence,
            )

    protected_relationships: list[dict[str, Any]] = []
    for root_kind, root_path in protected_roots:
        root_path_info = _output_safety_parse_path(root_path)
        normalized_root = root_path_info["normalized_path"]
        blocker_category = None
        if not normalized_output or not normalized_root:
            status = "unknown"
        elif (
            output_path_info["path_style"] != root_path_info["path_style"]
            or output_path_info["anchor"] != root_path_info["anchor"]
        ):
            status = "unknown_mixed_path_style"
            blocker_category = "unsafe_protected_output_location"
        else:
            output_inside_root = _output_safety_same_or_child(output_path_info, root_path_info)
            root_inside_output = _output_safety_same_or_child(root_path_info, output_path_info)
            if output_inside_root or root_inside_output:
                status = "unsafe"
                blocker_category = "unsafe_protected_output_location"
            else:
                status = "safe"
        if blocker_category:
            blockers.append(blocker_category)
        protected_relationships.append({
            "root_kind": root_kind,
            "root_path": normalized_root or str(root_path or ""),
            "root_path_style": root_path_info["path_style"],
            "status": status,
        })
    protected_root_status = (
        "verified_safe"
        if protected_root_context_complete
        and protected_relationships
        and all(item["status"] == "safe" for item in protected_relationships)
        else "unsafe"
        if any(item["status"] in ("unsafe", "unknown_mixed_path_style") for item in protected_relationships)
        else "unknown"
    )

    if overwrite_requested is not False:
        blockers.append("unsafe_overwrite_for_guided_first_subset")
    if path_role != "output_base" or run_directory_strategy != "derive_unique_run_id_under_output_base":
        blockers.append("output_ownership_policy_mismatch")
    if precreate_during_preview:
        blockers.append("output_ownership_policy_mismatch")

    supplied_facts = dict(filesystem_facts or {})
    exists = supplied_facts.get("exists")
    is_file = supplied_facts.get("is_file")
    is_directory = supplied_facts.get("is_directory")
    directory_empty = supplied_facts.get("directory_empty")
    writable = supplied_facts.get("writable")
    facts = {
        "facts_source": "supplied_read_only_facts" if filesystem_facts is not None else "not_inspected",
        "exists": exists if isinstance(exists, bool) else None,
        "is_file": is_file if isinstance(is_file, bool) else None,
        "is_directory": is_directory if isinstance(is_directory, bool) else None,
        "directory_empty": directory_empty if isinstance(directory_empty, bool) else None,
        "directory_non_empty": (
            not directory_empty if isinstance(directory_empty, bool) else None
        ),
        "requires_creation": not exists if isinstance(exists, bool) else None,
        "writable": writable if isinstance(writable, bool) else None,
        "writability_status": (
            "writable" if writable is True else "not_writable" if writable is False else "unknown"
        ),
        "filesystem_facts_required_for_planning_mapping": False,
    }
    if is_file is True:
        blockers.append("output_base_is_file")
    if writable is False:
        blockers.append("output_base_not_writable")

    blockers = list(dict.fromkeys(blockers))
    if "output_policy_not_current" in blockers:
        safety_status = "output_policy_not_current"
    elif "output_base_missing" in blockers:
        safety_status = "output_base_missing"
    elif "output_base_relative" in blockers:
        safety_status = "output_base_relative"
    elif "unsafe_overwrite_for_guided_first_subset" in blockers:
        safety_status = "unsafe_overwrite_for_guided_first_subset"
    elif "unsafe_source_output_relationship" in blockers:
        safety_status = "unsafe_source_output_relationship"
    elif "unsafe_protected_output_location" in blockers:
        safety_status = "unsafe_protected_output_location"
    elif "output_path_style_mismatch" in blockers:
        safety_status = "output_path_style_mismatch"
    elif blockers:
        safety_status = "output_base_safety_unresolved"
    else:
        safety_status = "output_base_ready_for_runner_owned_future_mapping"

    return {
        "output_safety_status": safety_status,
        "future_output_owner": "runner",
        "path_role": path_role,
        "run_directory_strategy": run_directory_strategy,
        "future_run_dir": "unresolved_until_execution_start",
        "future_run_directory_pattern": "<output_base>/<runner-generated-run-id>",
        "preview_path_kind": "pattern_only",
        "concrete_run_dir_known": False,
        "output_base": raw_output_base or None,
        "normalized_output_base": normalized_output or None,
        "write_context": write_context,
        "path_relationships": relationships,
        "filesystem_facts": facts,
        "overwrite_requested": overwrite_requested,
        "protected_root_status": protected_root_status,
        "protected_root_context_complete": protected_root_context_complete,
        "protected_root_relationships": protected_relationships,
        "blocker_categories": blockers,
        "execution_consumption_enabled": not blockers,
        "backend_config_mapping_status": safety_status,
        "no_directory_creation": True,
        "no_directory_reservation": True,
        "no_config_written": True,
        "no_command_written": True,
        "no_files_written": True,
        "no_validation": True,
        "no_run": True,
    }


def build_guided_output_base_safety_ownership_preview(
    plan: GuidedNewAnalysisDraftPlan,
    *,
    protected_roots: tuple[tuple[str, str], ...] | None = None,
    protected_root_context_complete: bool = False,
    filesystem_facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Guided output safety from stored plan state only."""
    if not isinstance(plan, GuidedNewAnalysisDraftPlan):
        raise TypeError("plan must be a GuidedNewAnalysisDraftPlan")
    stored_protected_roots: list[tuple[str, str]] = []
    if plan.cache_root_path:
        stored_protected_roots.append(("diagnostic_cache", plan.cache_root_path))
    stored_protected_roots.extend(protected_roots or ())
    source_path = plan.resolved_input_source_path or plan.input_source_path
    return classify_output_base_safety_ownership(
        output_base=plan.output_policy_path,
        source_path=source_path,
        output_policy_status=plan.output_policy_status,
        output_policy_explicitly_applied=plan.output_policy_explicitly_applied,
        output_policy_validation_issues=tuple(plan.output_policy_validation_issues),
        output_policy_stale_reasons=tuple(plan.output_policy_stale_reasons),
        path_role=plan.output_creation_policy.path_role,
        run_directory_strategy=plan.output_creation_policy.run_directory_strategy,
        overwrite_requested=plan.output_creation_policy.overwrite,
        precreate_during_preview=plan.output_creation_policy.precreate_during_preview,
        protected_roots=tuple(stored_protected_roots),
        protected_root_context_complete=protected_root_context_complete,
        filesystem_facts=filesystem_facts,
        write_context="planning_preview",
    )


def _execution_spec_output_preview_dict(
    plan: GuidedNewAnalysisDraftPlan,
    output_safety_ownership: dict[str, Any],
) -> dict[str, Any]:
    policy = _output_creation_policy_preview_dict(
        plan.output_creation_policy,
        execution_consumption_enabled=True,
    )
    return {
        "output_policy_status": plan.output_policy_status,
        "output_base": plan.output_policy_path,
        "output_policy_validation_issues": list(plan.output_policy_validation_issues),
        "output_policy_stale_reasons": list(plan.output_policy_stale_reasons),
        "output_policy_explicitly_applied": plan.output_policy_explicitly_applied,
        "path_role": policy["path_role"],
        "future_run_directory_strategy": policy["run_directory_strategy"],
        "future_run_dir": "unresolved_until_execution_start",
        "overwrite": policy["overwrite"],
        "precreate_during_preview": policy["precreate_during_preview"],
        "config_write_timing": policy["config_write_timing"],
        "gui_preflight_writes_enabled": policy["gui_preflight_writes_enabled"],
        "directory_created": False,
        "files_written": False,
        "config_written": False,
        "command_written": False,
        "validation_run": False,
        "execution_started": False,
        "RunSpec_instantiated": False,
        "argv_generated": False,
        "output_safety_ownership": output_safety_ownership,
    }


def build_guided_new_analysis_execution_spec_preview(
    plan: GuidedNewAnalysisDraftPlan,
) -> GuidedNewAnalysisExecutionSpecPreview:
    """Build a pure, non-writing execution-spec preview contract.

    This preview consumes only stored GuidedNewAnalysisDraftPlan state and
    documented first-subset defaults. It does not instantiate RunSpec, build
    argv, write config, create directories, validate, run, or mutate the plan.
    """
    if not isinstance(plan, GuidedNewAnalysisDraftPlan):
        raise TypeError("plan must be a GuidedNewAnalysisDraftPlan")

    subset_readiness = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    feature_event_consumption = _feature_event_consumption_preview_dict(plan)
    feature_event_effective_values = feature_event_consumption["effective_values_preview"]
    feature_event_blockers = tuple(feature_event_effective_values["blocker_categories"])
    rwd_normalization = build_guided_rwd_dataset_contract_normalization_preview(plan)
    rwd_normalization_blockers = tuple(rwd_normalization["blocker_categories"])
    output_safety_ownership = build_guided_output_base_safety_ownership_preview(plan)
    output_safety_blockers = tuple(output_safety_ownership["blocker_categories"])
    issue_categories = tuple(
        dict.fromkeys(
            tuple(issue.category for issue in subset_readiness.blocking_issues)
            + feature_event_blockers
            + rwd_normalization_blockers
            + output_safety_blockers
        )
    )
    blocked_reasons = tuple(issue.message for issue in subset_readiness.blocking_issues) + tuple(
        f"Feature/event effective-value preview blocker: {category}"
        for category in feature_event_blockers
    ) + tuple(
        f"RWD dataset normalization blocker: {category}"
        for category in rwd_normalization_blockers
    ) + tuple(
        f"Output safety/ownership blocker: {category}"
        for category in output_safety_blockers
    )
    spec_preview_available = bool(
        subset_readiness.first_subset_executable
        and not feature_event_blockers
        and not rwd_normalization_blockers
        and not output_safety_blockers
    )
    output_safety_ownership = {
        **output_safety_ownership,
        "execution_consumption_enabled": bool(
            spec_preview_available
            and output_safety_ownership["output_safety_status"]
            == "output_base_ready_for_runner_owned_future_mapping"
        ),
    }
    dataset_contract = _dataset_contract_snapshot_preview_dict(
        plan.dataset_contract_snapshot,
        execution_consumption_enabled=spec_preview_available,
    )
    dataset_contract["rwd_normalization"] = {
        **rwd_normalization,
        "execution_consumption_enabled": bool(
            spec_preview_available
            and rwd_normalization["normalization_status"] == "ready_for_future_mapping"
        ),
    }
    execution_intent = _execution_intent_preview_dict(
        plan.execution_intent,
        execution_consumption_enabled=spec_preview_available,
    )

    return GuidedNewAnalysisExecutionSpecPreview(
        spec_preview_schema_version=EXECUTION_SPEC_PREVIEW_SCHEMA_VERSION,
        plan_schema_version=plan.schema_version,
        subset_name=subset_readiness.subset_name,
        spec_preview_available=spec_preview_available,
        first_subset_executable=subset_readiness.first_subset_executable,
        execution_available=False,
        execution_blocked_reason="Final Guided Run/RunSpec is not implemented in this stage.",
        backend_mapping_status="preview_only_not_mapped_to_RunSpec",
        source_acquisition={
            "input_source_path": plan.input_source_path,
            "resolved_input_source_path": plan.resolved_input_source_path,
            "authoritative_input_source_path": plan.resolved_input_source_path or plan.input_source_path,
            "input_format": plan.input_format,
            "acquisition_mode": plan.acquisition_mode,
            "sessions_per_hour": plan.sessions_per_hour,
            "session_duration_sec": plan.session_duration_sec,
            "continuous_window_sec": plan.continuous_window_sec,
            "continuous_step_sec": plan.continuous_step_sec,
            "allow_partial_final_window": plan.allow_partial_final_window,
            "exclude_incomplete_final_rwd_chunk": plan.exclude_incomplete_final_rwd_chunk,
        },
        dataset_contract=dataset_contract,
        roi={
            "included_roi_ids": list(plan.included_roi_ids),
            "excluded_roi_ids": list(plan.excluded_roi_ids),
            "discovered_roi_ids": list(plan.discovered_roi_ids),
            "include_list_is_authoritative": True,
            "excluded_rois_are_provenance_only": True,
        },
        correction=_execution_spec_correction_preview_dict(plan, subset_readiness),
        execution_intent={
            **execution_intent,
            "traces_only": False,
            "tonic_outputs_in_scope": False,
            "full_both_mode_outputs_in_scope": False,
        },
        feature_event={
            "status": plan.feature_event_profile_status,
            "profile_id": plan.feature_event_profile_id,
            "values": dict(plan.feature_event_values),
            "validation_issues": list(plan.feature_event_validation_issues),
            "stale_reasons": list(plan.feature_event_stale_reasons),
            "explicitly_applied": plan.feature_event_explicitly_applied,
            "baseline_config_source": plan.feature_event_baseline_config_source,
            "baseline_status": plan.feature_event_baseline_status,
            "updated_at_utc": plan.feature_event_updated_at_utc,
            "consumption": feature_event_consumption,
            "feature_event_effective_values": feature_event_effective_values,
        },
        output=_execution_spec_output_preview_dict(plan, output_safety_ownership),
        diagnostic_cache_provenance={
            "cache_id": plan.cache_id,
            "cache_root_path": plan.cache_root_path,
            "artifact_record_path": plan.artifact_record_path,
            "request_json_path": plan.request_json_path,
            "provenance_path": plan.provenance_path,
            "phasic_trace_cache_path": plan.phasic_trace_cache_path,
            "config_used_path": plan.config_used_path,
            "source_setup_signature": plan.source_setup_signature,
            "diagnostic_scope_signature": plan.diagnostic_scope_signature,
            "build_request_signature": plan.build_request_signature,
            "stale_or_current": plan.stale_or_current,
            "stale_reasons": list(plan.stale_reasons),
            "execution_consumes_cache_artifacts": False,
        },
        provenance={
            "plan_created_at_utc": plan.created_at_utc,
            "plan_updated_at_utc": plan.updated_at_utc,
            "production_analysis": plan.production_analysis,
            "preliminary_cache": plan.preliminary_cache,
            "stored_plan_state_only": True,
            "no_gui_runspec": True,
            "no_argv_generated": True,
            "no_config_written": True,
            "no_output_directory_created": True,
            "no_validation_run": True,
            "no_pipeline_run": True,
        },
        blocked_reasons=blocked_reasons,
        blocking_issue_categories=issue_categories,
        field_classifications=subset_readiness.field_classifications,
        warning_issue_categories=tuple(issue.category for issue in subset_readiness.warning_issues),
    )
