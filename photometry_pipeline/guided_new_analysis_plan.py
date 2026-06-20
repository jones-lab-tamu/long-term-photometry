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
RUN_PREVIEW_SCHEMA_VERSION = "guided_new_analysis_run_preview.v1"
EXECUTION_SUBSET_SCHEMA_VERSION = "guided_new_analysis_execution_subset.v1"
FIRST_EXECUTION_SUBSET_NAME = "global_dynamic_fit_only.v1"
SUPPORTED_INPUT_FORMATS = {"auto", "rwd", "npm", "custom_tabular"}
SUPPORTED_ACQUISITION_MODES = {"intermittent", "continuous"}
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
    roi_selection: dict[str, Any]
    diagnostic_cache: dict[str, Any]
    correction_strategy: dict[str, Any]
    evidence_references: dict[str, Any]
    feature_event: dict[str, Any]
    output_policy: dict[str, Any]
    provenance: dict[str, Any]
    readiness_snapshot: dict[str, Any]
    unresolved_items: tuple[GuidedNewAnalysisRunPreviewIssue, ...] = ()
    warnings: tuple[GuidedNewAnalysisRunPreviewIssue, ...] = ()
    execution_available: bool = False
    execution_blocked_reason: str = "Final Guided Run/RunSpec is not implemented in this stage."


@dataclass(frozen=True)
class GuidedNewAnalysisExecutionSubsetIssue:
    category: str
    message: str
    severity: str  # "blocking", "warning", "info"
    section: str = "execution_subset"


@dataclass(frozen=True)
class GuidedNewAnalysisExecutionFieldClassification:
    field_name: str
    status: str  # present / fixed_default / required_missing / deferred_full_control / not_relevant
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


def _execution_field_classifications(plan: GuidedNewAnalysisDraftPlan) -> tuple[GuidedNewAnalysisExecutionFieldClassification, ...]:
    fields: list[GuidedNewAnalysisExecutionFieldClassification] = [
        _execution_field(
            "timeline_anchor_mode",
            "required_missing",
            value=None,
            provenance="timeline anchor mode is not represented in GuidedNewAnalysisDraftPlan and no executable default is verified",
            blocks_subset=True,
            issue_category="missing_timeline_anchor_mode",
        ),
        _execution_field(
            "fixed_daily_anchor_clock",
            "fixed_default",
            value=None,
            provenance="first subset fixed default; no fixed daily anchor",
        ),
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

    fields.append(_execution_field(
        "mode",
        "required_missing",
        value=None,
        provenance="execution mode both/tonic/phasic is not represented in GuidedNewAnalysisDraftPlan",
        blocks_subset=True,
        issue_category="missing_execution_mode",
    ))
    fields.append(_execution_field(
        "run_profile",
        "required_missing",
        value=None,
        provenance="execution run profile is not represented in GuidedNewAnalysisDraftPlan",
        blocks_subset=True,
        issue_category="missing_run_profile",
    ))
    fields.append(_execution_field(
        "output_creation_policy",
        "required_missing",
        value=None,
        provenance="output destination is stored, but creation/overwrite execution policy is not represented",
        blocks_subset=True,
        issue_category="missing_output_creation_policy",
    ))

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
        fields.append(_execution_field(
            "dataset_contract_overrides",
            "required_missing",
            value=None,
            provenance="RWD dataset contract snapshot is not represented in GuidedNewAnalysisDraftPlan",
            blocks_subset=True,
            issue_category="missing_rwd_dataset_contract",
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
            fields.append(_execution_field(
                "npm_channel_mapping",
                "required_missing",
                value=None,
                provenance="NPM signal/control channel mapping is not represented in GuidedNewAnalysisDraftPlan",
                blocks_subset=True,
                issue_category="missing_npm_channel_mapping",
            ))
            fields.append(_execution_field(
                "dataset_contract_overrides",
                "required_missing",
                value=None,
                provenance="NPM dataset contract snapshot is not represented in GuidedNewAnalysisDraftPlan",
                blocks_subset=True,
                issue_category="missing_npm_dataset_contract",
            ))
    elif fmt == "custom_tabular":
        fields.append(_execution_field(
            "custom_tabular_column_mapping",
            "required_missing",
            value=None,
            provenance="custom_tabular column mapping is not represented in GuidedNewAnalysisDraftPlan",
            blocks_subset=True,
            issue_category="missing_custom_tabular_column_mapping",
        ))
        fields.append(_execution_field(
            "dataset_contract_overrides",
            "required_missing",
            value=None,
            provenance="custom_tabular dataset contract snapshot is not represented in GuidedNewAnalysisDraftPlan",
            blocks_subset=True,
            issue_category="missing_custom_tabular_dataset_contract",
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
        "First execution subset is available for future executable preview; execution remains unavailable in this stage."
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


def _section_snapshot(section: GuidedNewAnalysisSectionReadiness) -> dict[str, Any]:
    return {
        "key": section.key,
        "label": section.label,
        "status": section.status,
        "blocking_categories": [issue.category for issue in section.blocking_issues],
        "warning_categories": [issue.category for issue in section.warning_issues],
        "info_categories": [issue.category for issue in section.info_issues],
    }


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
    if choices:
        unresolved.append(GuidedNewAnalysisRunPreviewIssue(
            category="per_roi_correction_execution_contract_unresolved",
            message=(
                "Per-ROI correction strategy execution mapping is not implemented; "
                "preview preserves choices without collapsing them to a global strategy."
            ),
            severity="blocking",
        ))
    if any(choice.selected_strategy == "signal_only_f0" for choice in choices):
        unresolved.append(GuidedNewAnalysisRunPreviewIssue(
            category="signal_only_f0_production_routing_unresolved",
            message="Signal-Only F0 production routing is not implemented for Guided new_analysis.",
            severity="blocking",
        ))

    fixed_contract_defaults = {
        "execution_available": False,
        "execution_backend": "unavailable_in_this_stage",
    }

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
                "status": "unresolved",
                "reason": "GuidedNewAnalysisDraftPlan does not represent timeline anchor mode.",
            },
        },
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
        provenance={
            "plan_created_at_utc": plan.created_at_utc,
            "plan_updated_at_utc": plan.updated_at_utc,
            "production_analysis": plan.production_analysis,
            "preliminary_cache": plan.preliminary_cache,
            "fixed_contract_defaults": fixed_contract_defaults,
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
