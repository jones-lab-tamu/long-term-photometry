from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedPlanIssue,
    FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
)

@dataclass(frozen=True)
class GuidedValidationRequest:
    # Source/acquisition
    source_path: str | None
    source_format: str
    acquisition_mode: str
    sessions_per_hour: int | None
    session_duration_sec: float | None
    exclude_incomplete_final_rwd_chunk: bool
    timeline_anchor_mode: str

    # ROI Selection
    included_roi_ids: list[str] = field(default_factory=list)
    excluded_roi_ids: list[str] = field(default_factory=list)
    discovered_roi_ids: list[str] = field(default_factory=list)
    roi_state_status: str = "unknown"

    # Execution
    execution_mode: str = "phasic"
    run_profile: str = "full"
    traces_only: bool = False
    subset_contract_version: str = "global_dynamic_fit_only.v1"

    # Correction Strategy
    strategy_scope: str = "global"
    global_correction_strategy: str | None = None
    dynamic_fit_mode: str | None = None
    dynamic_fit_parameter_contract: dict[str, Any] = field(default_factory=dict)

    # Output Destination
    output_base_path: str | None = None
    output_overwrite: bool = False
    output_path_role: str = "output_base"
    output_creation_timing: str = "future_execution_start_only"
    run_directory_strategy: str = "derive_unique_run_id_under_output_base"


def build_guided_validation_request_from_plan(plan: GuidedNewAnalysisDraftPlan) -> GuidedValidationRequest:
    """
    Compile a GuidedNewAnalysisDraftPlan into a GuidedValidationRequest in memory.
    """
    # Sourced from GuidedNewAnalysisDraftPlan
    source_path = getattr(plan, "input_source_path", None)
    source_format = getattr(plan, "input_format", "auto")
    acq_mode = getattr(plan, "acquisition_mode", "continuous")
    sessions_per_hour = getattr(plan, "sessions_per_hour", None)
    session_duration_sec = getattr(plan, "session_duration_sec", None)
    exclude_incomplete_final_rwd_chunk = getattr(plan, "exclude_incomplete_final_rwd_chunk", False)
    
    # ROI selection
    included_roi_ids = list(getattr(plan, "included_roi_ids", []) or [])
    excluded_roi_ids = list(getattr(plan, "excluded_roi_ids", []) or [])
    discovered_roi_ids = list(getattr(plan, "discovered_roi_ids", []) or [])
    roi_state_status = getattr(plan, "acquisition_structure_status", "unknown")
    
    # Execution intent
    timeline_anchor_mode = "civil"
    execution_mode = "phasic"
    run_profile = "full"
    exec_intent = getattr(plan, "execution_intent", None)
    if exec_intent:
        timeline_anchor_mode = getattr(exec_intent, "timeline_anchor_mode", "civil")
        execution_mode = getattr(exec_intent, "execution_mode", "phasic")
        run_profile = getattr(exec_intent, "run_profile", "full")
        
    # Output policy fields
    output_base_path = getattr(plan, "output_base_path", None)
    output_overwrite = getattr(plan, "output_overwrite", False)
    
    output_path_role = "output_base"
    output_creation_timing = "future_execution_start_only"
    run_directory_strategy = "derive_unique_run_id_under_output_base"
    out_policy = getattr(plan, "output_creation_policy", None)
    if out_policy:
        output_path_role = getattr(out_policy, "path_role", "output_base")
        output_creation_timing = getattr(out_policy, "creation_timing", "future_execution_start_only")
        run_directory_strategy = getattr(out_policy, "run_directory_strategy", "derive_unique_run_id_under_output_base")
        
    # Correction strategy fields
    global_correction_strategy = getattr(plan, "global_correction_strategy", None)
    dynamic_fit_mode = getattr(plan, "dynamic_fit_mode", None)
    
    dynamic_fit_parameter_contract = {}
    df_contract = getattr(plan, "dynamic_fit_parameter_contract", None)
    if df_contract:
        from dataclasses import asdict
        dynamic_fit_parameter_contract = asdict(df_contract)

    return GuidedValidationRequest(
        source_path=source_path,
        source_format=source_format,
        acquisition_mode=acq_mode,
        sessions_per_hour=sessions_per_hour,
        session_duration_sec=session_duration_sec,
        exclude_incomplete_final_rwd_chunk=exclude_incomplete_final_rwd_chunk,
        timeline_anchor_mode=timeline_anchor_mode,
        included_roi_ids=included_roi_ids,
        excluded_roi_ids=excluded_roi_ids,
        discovered_roi_ids=discovered_roi_ids,
        roi_state_status=roi_state_status,
        execution_mode=execution_mode,
        run_profile=run_profile,
        traces_only=False,
        strategy_scope="global",
        global_correction_strategy=global_correction_strategy,
        dynamic_fit_mode=dynamic_fit_mode,
        dynamic_fit_parameter_contract=dynamic_fit_parameter_contract,
        output_base_path=output_base_path,
        output_overwrite=output_overwrite,
        output_path_role=output_path_role,
        output_creation_timing=output_creation_timing,
        run_directory_strategy=run_directory_strategy,
    )


def compute_request_identity(request: GuidedValidationRequest) -> str:
    """
    Compute a unique hash signature for the request.
    Uses basic attributes to define config/plan identity.
    
    NOTE: This is a minimal helper and is non-authorizing for Run execution.
    It is not wired to validation success or Run eligibility.
    """
    import hashlib
    # Concat important fields to create stable string representation
    elements = [
        str(request.source_path),
        str(request.source_format),
        str(request.acquisition_mode),
        str(request.sessions_per_hour),
        str(request.session_duration_sec),
        str(request.exclude_incomplete_final_rwd_chunk),
        str(request.timeline_anchor_mode),
        ",".join(sorted(request.included_roi_ids)),
        str(request.global_correction_strategy),
        str(request.dynamic_fit_mode),
        str(request.output_base_path),
        str(request.output_overwrite),
    ]
    raw_str = "|".join(elements)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


def validate_guided_validation_request(request: GuidedValidationRequest) -> list[GuidedPlanIssue]:
    """
    Perform local, fail-closed validation on the GuidedValidationRequest.
    """
    issues = []

    # 1. Source checks
    if not request.source_path:
        issues.append(GuidedPlanIssue(category="missing_raw_source", message="Raw input source path is missing.", severity="blocking"))
    else:
        if not os.path.isdir(request.source_path):
            issues.append(GuidedPlanIssue(category="invalid_raw_source", message=f"Raw input source path does not exist: {request.source_path}", severity="blocking"))

    if request.source_format not in {"rwd", "RWD"}:
        issues.append(GuidedPlanIssue(category="unsupported_source_format", message=f"Unsupported source format: {request.source_format}", severity="blocking"))

    # 2. Acquisition checks
    if request.acquisition_mode != "intermittent":
        issues.append(GuidedPlanIssue(category="unsupported_acquisition_mode", message="Only intermittent acquisition is supported in the first subset.", severity="blocking"))

    if request.sessions_per_hour is None or request.sessions_per_hour <= 0:
        issues.append(GuidedPlanIssue(category="invalid_sessions_per_hour", message="Sessions per hour must be positive.", severity="blocking"))

    if request.session_duration_sec is None or request.session_duration_sec <= 0:
        issues.append(GuidedPlanIssue(category="invalid_session_duration", message="Session duration must be positive.", severity="blocking"))

    # 3. ROI checks
    if not request.included_roi_ids:
        issues.append(GuidedPlanIssue(category="missing_roi_selection", message="At least one ROI must be selected.", severity="blocking"))

    # 4. Execution profile checks
    if request.execution_mode != "phasic":
        issues.append(GuidedPlanIssue(category="unsupported_execution_mode", message="Only phasic execution mode is supported.", severity="blocking"))

    if request.run_profile != "full":
        issues.append(GuidedPlanIssue(category="unsupported_run_profile", message="Only full run profile is supported.", severity="blocking"))

    if request.traces_only:
        issues.append(GuidedPlanIssue(category="unsupported_traces_only", message="traces_only=True is not supported.", severity="blocking"))

    # 5. Output destination checks
    if not request.output_base_path:
        issues.append(GuidedPlanIssue(category="missing_output_base", message="Output base directory is missing.", severity="blocking"))
    else:
        if request.output_overwrite:
            issues.append(GuidedPlanIssue(category="unsupported_overwrite", message="Overwrite is not permitted in Guided validation subset.", severity="blocking"))
            
        # Path safety / overlaps
        src = os.path.realpath(request.source_path) if request.source_path else ""
        out = os.path.realpath(request.output_base_path)
        
        if src:
            if src == out:
                issues.append(GuidedPlanIssue(category="output_source_overlap", message="Output base must not be the same as raw input directory.", severity="blocking"))
            elif out.startswith(src + os.sep):
                issues.append(GuidedPlanIssue(category="output_source_overlap", message="Output base must not be nested inside raw input directory.", severity="blocking"))
            elif src.startswith(out + os.sep):
                issues.append(GuidedPlanIssue(category="output_source_overlap", message="Raw input directory must not be nested inside output base.", severity="blocking"))

        # TODO: Defer completed-run directory detection from the minimal validator.
        # Completed-run directory rejection must be handled by a read-only non-GUI helper later.
        # Re-incorporate completed-run detection helper check here once relocated to a non-GUI module.

    # 6. Correction strategy checks
    if request.strategy_scope != "global":
        issues.append(GuidedPlanIssue(category="unsupported_strategy_scope", message="Only global strategy scope is supported.", severity="blocking"))

    if request.global_correction_strategy != "dynamic_fit":
        if request.global_correction_strategy == "signal_only_f0":
            issues.append(GuidedPlanIssue(category="unsupported_correction_strategy", message="Signal-Only F0 routing is not supported in this validation subset.", severity="blocking"))
        else:
            issues.append(GuidedPlanIssue(category="unsupported_correction_strategy", message=f"Unsupported strategy: {request.global_correction_strategy}", severity="blocking"))
    else:
        # Must have dynamic_fit_mode
        if not request.dynamic_fit_mode:
            issues.append(GuidedPlanIssue(category="missing_dynamic_fit_mode", message="Dynamic fit mode is missing.", severity="blocking"))
        elif request.dynamic_fit_mode not in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
            issues.append(GuidedPlanIssue(category="unsupported_dynamic_fit_mode", message=f"Unsupported dynamic fit mode: {request.dynamic_fit_mode}", severity="blocking"))

    return issues


@dataclass(frozen=True)
class GuidedValidationResultState:
    backend_validation_status: str = "unvalidated"
    backend_validated_request_identity: str = ""
    validation_result_identity: str = ""
    validator_version: str = ""
    validation_scope: str = "guided_first_subset"
    validation_timestamp: str | None = None
    backend_errors: list[str] = field(default_factory=list)
    backend_warnings: list[str] = field(default_factory=list)
    backend_info: list[str] = field(default_factory=list)
    validation_subset_contract_version: str = "global_dynamic_fit_only.v1"


def make_unvalidated_guided_validation_state() -> GuidedValidationResultState:
    return GuidedValidationResultState()


def make_passed_guided_validation_state(
    req_identity: str,
    result_identity: str,
    *,
    warnings: list[str] | None = None,
    info: list[str] | None = None,
    validator_version: str = "",
    validation_scope: str = "guided_first_subset",
    validation_timestamp: str | None = None,
    validation_subset_contract_version: str = "global_dynamic_fit_only.v1",
) -> GuidedValidationResultState:
    return GuidedValidationResultState(
        backend_validation_status="passed",
        backend_validated_request_identity=req_identity,
        validation_result_identity=result_identity,
        validator_version=validator_version,
        validation_scope=validation_scope,
        validation_timestamp=validation_timestamp,
        backend_errors=[],
        backend_warnings=list(warnings or []),
        backend_info=list(info or []),
        validation_subset_contract_version=validation_subset_contract_version,
    )


def make_failed_guided_validation_state(
    req_identity: str,
    errors: list[str],
    *,
    warnings: list[str] | None = None,
    info: list[str] | None = None,
    validator_version: str = "",
    validation_scope: str = "guided_first_subset",
    validation_timestamp: str | None = None,
    validation_subset_contract_version: str = "global_dynamic_fit_only.v1",
) -> GuidedValidationResultState:
    return GuidedValidationResultState(
        backend_validation_status="failed",
        backend_validated_request_identity=req_identity,
        validation_result_identity="",
        validator_version=validator_version,
        validation_scope=validation_scope,
        validation_timestamp=validation_timestamp,
        backend_errors=list(errors or []),
        backend_warnings=list(warnings or []),
        backend_info=list(info or []),
        validation_subset_contract_version=validation_subset_contract_version,
    )


def make_error_guided_validation_state(
    req_identity: str,
    errors: list[str],
    *,
    warnings: list[str] | None = None,
    info: list[str] | None = None,
    validator_version: str = "",
    validation_scope: str = "guided_first_subset",
    validation_timestamp: str | None = None,
    validation_subset_contract_version: str = "global_dynamic_fit_only.v1",
) -> GuidedValidationResultState:
    return GuidedValidationResultState(
        backend_validation_status="error",
        backend_validated_request_identity=req_identity,
        validation_result_identity="",
        validator_version=validator_version,
        validation_scope=validation_scope,
        validation_timestamp=validation_timestamp,
        backend_errors=list(errors or []),
        backend_warnings=list(warnings or []),
        backend_info=list(info or []),
        validation_subset_contract_version=validation_subset_contract_version,
    )


def is_guided_validation_state_stale(
    state: GuidedValidationResultState,
    current_request_identity: str,
) -> bool:
    if state.backend_validation_status == "unvalidated":
        return True
    if not state.backend_validated_request_identity or not current_request_identity:
        return True
    return state.backend_validated_request_identity != current_request_identity


def can_guided_run_unlock(
    state: GuidedValidationResultState,
    current_request_identity: str,
    local_issues: list[Any],
    production_run_allocated: bool = False,
) -> bool:
    if state.backend_validation_status != "passed":
        return False
    if is_guided_validation_state_stale(state, current_request_identity):
        return False
    if production_run_allocated:
        return False
    for iss in local_issues:
        if getattr(iss, "severity", None) == "blocking":
            return False
    return True

