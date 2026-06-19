"""Non-executable new_analysis Guided draft plan model and pure validation helpers.

This module intentionally contains no GUI, RunSpec, pipeline, validation, feature
extraction, or output-writing imports. It defines only the data contract and pure
validation helpers for the new_analysis Guided draft plan state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "guided_new_analysis_plan.v1"
SUPPORTED_INPUT_FORMATS = {"auto", "rwd", "npm", "custom_tabular"}
SUPPORTED_ACQUISITION_MODES = {"intermittent", "continuous"}
RUNNABLE_CORRECTION_STRATEGIES = {
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
    "global_linear_regression",
    "signal_only_f0",
}
FORBIDDEN_CORRECTION_STRATEGIES = {"auto", "needs_review", "no_correction"}


@dataclass(frozen=True)
class GuidedPlanIssue:
    category: str
    message: str
    severity: str  # "blocking", "warning", "info"


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
    output_policy_status: str = "unavailable"  # missing / selected / unsafe / ready / unavailable
    output_policy_root: str | None = None
    output_policy_not_represented_note: str = "output policy is not yet represented as new-analysis plan state"

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
            severity="warning" if plan.feature_event_profile_status == "unavailable" else "blocking"
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
            severity="warning"
        ))

    # 13. missing_output_policy
    if plan.output_policy_status in ("missing", "unavailable"):
        issues.append(GuidedPlanIssue(
            category="missing_output_policy",
            message="Output policy is missing or unavailable in this stage.",
            severity="warning" if plan.output_policy_status == "unavailable" else "blocking"
        ))

    # 14. unsafe_output_policy
    elif plan.output_policy_status == "unsafe":
        issues.append(GuidedPlanIssue(
            category="unsafe_output_policy",
            message="Output policy root path is unsafe (e.g. conflicts with input source).",
            severity="blocking"
        ))

    # 15. execution_not_implemented
    issues.append(GuidedPlanIssue(
        category="execution_not_implemented",
        message="This draft plan is not executable yet. Final Run is not implemented in this stage.",
        severity="blocking"
    ))

    return issues
