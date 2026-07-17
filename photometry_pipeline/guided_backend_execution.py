"""GUI-safe backend adapter for internal Guided startup orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Callable

import photometry_pipeline.guided_startup_orchestration as orchestration
from photometry_pipeline.guided_startup_allocation import (
    GuidedStartupAllocationResult,
)
from photometry_pipeline.guided_startup_materialization import (
    GuidedStartupMaterializationResult,
)
from photometry_pipeline.guided_startup_transaction import (
    GuidedStartupPlanResult,
    GuidedStartupTransactionRequest,
)


@dataclass(frozen=True)
class GuidedBackendExecutionIssue:
    category: str
    section: str
    message: str
    user_safe_message: str


@dataclass(frozen=True)
class GuidedBackendExecutionDiagnostics:
    orchestration_status: str
    pure_plan_status: str | None
    allocation_status: str | None
    materialization_status: str | None
    wrapper_started: bool
    wrapper_completed: bool
    wrapper_returncode: int | None
    failure_marker_path: str | None
    startup_transaction_identity: str | None
    wrapper_command: tuple[str, ...] | None


@dataclass(frozen=True)
class GuidedExecutionFailureDetail:
    failure_type: str
    category: str
    phase: str
    source: str
    session_index: int
    reason: str
    eligible_for_missing_session_authorization: bool


@dataclass(frozen=True)
class GuidedBackendExecutionResult:
    status: str
    ok: bool
    user_visible_state: str
    user_summary: str
    run_directory: str | None
    completed_run_candidate_path: str | None
    requires_completed_run_loader_validation: bool
    wrapper_started: bool
    wrapper_completed: bool
    blocking_issues: tuple[GuidedBackendExecutionIssue, ...]
    diagnostics: GuidedBackendExecutionDiagnostics
    gui_run_enabled: bool = False
    exposes_manifest_path_to_user: bool = False
    exposes_internal_cli_to_user: bool = False
    completed_run_claim: bool = False
    failure_details: tuple[GuidedExecutionFailureDetail, ...] = ()


_OUTPUT_NOT_CREATABLE_SUMMARY = (
    "Guided Run could not find or create the selected output folder. "
    "Choose a writable output destination and try again."
)
# `pure_plan_output_unsafe` (see guided_startup_orchestration._validate_plan)
# represents a genuinely current setup that a later, more exhaustive
# startup-planning check has correctly refused for a specific,
# scientist-actionable reason -- not a stale/expired validation. The
# shared "no longer current" fallback below is actively false for this;
# a scientist would re-check their setup for no reason instead of picking
# a different output destination.
_OUTPUT_UNSAFE_SUMMARY = (
    "Guided Run could not start because the selected output destination "
    "is not safe to use for a new run (for example, it is a previous "
    "completed run's folder). Choose a different output destination and "
    "try again."
)

_STATUS_MAP = {
    "refused_before_allocation": (
        "refused_before_startup",
        "not_started",
        False,
        "Guided Run could not start because the validated setup is no longer current.",
    ),
    "allocation_failed": (
        "startup_allocation_failed",
        "failed_to_prepare",
        False,
        "Guided Run could not create a safe output folder.",
    ),
    "startup_status_write_failed": (
        "startup_status_write_failed",
        "failed_to_prepare",
        False,
        "Guided Run created an output folder but could not write its startup status.",
    ),
    "materialization_failed": (
        "startup_materialization_failed",
        "failed_to_prepare",
        False,
        "Guided Run could not prepare the internal run files.",
    ),
    "refused_before_wrapper": (
        "startup_materialization_failed",
        "failed_to_prepare",
        False,
        "Guided Run could not finish preparing the analysis.",
    ),
    "wrapper_start_failed": (
        "wrapper_start_failed",
        "failed_to_start",
        False,
        "Guided Run prepared the output folder but could not start the analysis.",
    ),
    "wrapper_started": (
        "wrapper_running",
        "running",
        True,
        "Guided Run is running.",
    ),
    "wrapper_failed": (
        "wrapper_failed",
        "failed_during_run",
        False,
        "Guided Run started, but the analysis reported an error.",
    ),
    "wrapper_completed": (
        "wrapper_completed_needs_review_loading",
        "run_finished_review_required",
        True,
        "Guided Run finished. Load the completed run for review.",
    ),
}


def execute_guided_backend_run(
    *,
    request: GuidedStartupTransactionRequest,
    pure_plan: GuidedStartupPlanResult | None = None,
    allocation_result: GuidedStartupAllocationResult | None = None,
    materialization_result: GuidedStartupMaterializationResult | None = None,
    runner: (
        Callable[
            [tuple[str, ...]],
            orchestration.GuidedWrapperProcessResult,
        ]
        | None
    ) = None,
) -> GuidedBackendExecutionResult:
    """Execute internal orchestration and return a user-safe lifecycle result."""
    internal = orchestration.run_guided_startup_to_wrapper(
        request=request,
        pure_plan=pure_plan,
        allocation_result=allocation_result,
        materialization_result=materialization_result,
        subprocess_runner=runner,
    )
    mapped = _STATUS_MAP.get(internal.status)
    if mapped is None:
        mapped = (
            "unexpected_internal_failure",
            "failed_to_prepare",
            False,
            "Guided Run could not continue because of an internal error.",
        )
    status, user_state, ok, summary = mapped
    if internal.status == "refused_before_allocation" and internal.blocking_issues:
        first_category = internal.blocking_issues[0].category
        if first_category == "pure_plan_output_not_creatable":
            summary = _OUTPUT_NOT_CREATABLE_SUMMARY
        elif first_category == "pure_plan_output_unsafe":
            summary = _OUTPUT_UNSAFE_SUMMARY
    completed_candidate = (
        internal.allocated_run_dir
        if status == "wrapper_completed_needs_review_loading"
        else None
    )
    diagnostics = GuidedBackendExecutionDiagnostics(
        orchestration_status=internal.status,
        pure_plan_status=internal.pure_plan_status,
        allocation_status=internal.allocation_status,
        materialization_status=internal.materialization_status,
        wrapper_started=internal.wrapper_started,
        wrapper_completed=internal.wrapper_completed,
        wrapper_returncode=internal.wrapper_returncode,
        failure_marker_path=internal.failure_marker_path,
        startup_transaction_identity=internal.startup_transaction_identity,
        wrapper_command=internal.wrapper_command,
    )
    issues = tuple(
        GuidedBackendExecutionIssue(
            category=issue.category,
            section=issue.section,
            message=issue.message,
            user_safe_message=summary,
        )
        for issue in internal.blocking_issues
    )

    failure_details = []
    if status == "wrapper_failed" and internal.allocated_run_dir:
        status_path = os.path.join(internal.allocated_run_dir, "status.json")
        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as handle:
                    status_data = json.load(handle)
                details_list = status_data.get("failure_details", [])
                if isinstance(details_list, list):
                    for item in details_list:
                        if isinstance(item, dict):
                            ft = item.get("failure_type")
                            cat = item.get("category")
                            phase = item.get("phase")
                            source = item.get("source")
                            s_idx = item.get("session_index")
                            reason = item.get("reason")
                            elig = item.get("eligible_for_missing_session_authorization")

                            # Validate types strictly and fail closed
                            if (
                                isinstance(ft, str)
                                and isinstance(cat, str)
                                and isinstance(phase, str)
                                and isinstance(source, str)
                                and not isinstance(s_idx, bool)
                                and isinstance(s_idx, int)
                                and s_idx >= 0
                                and isinstance(reason, str)
                                and isinstance(elig, bool)
                            ):
                                failure_details.append(
                                    GuidedExecutionFailureDetail(
                                        failure_type=ft,
                                        category=cat,
                                        phase=phase,
                                        source=source,
                                        session_index=s_idx,
                                        reason=reason,
                                        eligible_for_missing_session_authorization=elig,
                                    )
                                )
            except Exception:
                pass

    return GuidedBackendExecutionResult(
        status=status,
        ok=ok,
        user_visible_state=user_state,
        user_summary=summary,
        run_directory=internal.allocated_run_dir,
        completed_run_candidate_path=completed_candidate,
        requires_completed_run_loader_validation=completed_candidate is not None,
        wrapper_started=internal.wrapper_started,
        wrapper_completed=internal.wrapper_completed,
        blocking_issues=issues,
        diagnostics=diagnostics,
        failure_details=tuple(failure_details),
    )
