"""Pure hidden readiness evaluation for a future Guided Run control."""

from __future__ import annotations

from dataclasses import dataclass

from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationWorkflowOutcome,
)
from photometry_pipeline.guided_execution_payloads import (
    GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE,
    GuidedExecutionPayloadDerivationResult,
)
from photometry_pipeline.guided_run_authorization import (
    GuidedRunAuthorizationResult,
)


@dataclass(frozen=True)
class GuidedRunReadinessIssue:
    category: str
    section: str
    message: str
    user_safe_message: str


@dataclass(frozen=True)
class GuidedRunReadinessResult:
    status: str
    ready: bool
    user_visible_state: str
    user_summary: str
    blocking_issues: tuple[GuidedRunReadinessIssue, ...]
    validation_revision: int | None
    current_gui_revision: int
    authorization_identity: str | None
    payload_status: str | None
    backend_execution_available: bool
    startup_orchestration_available: bool
    execution_would_be_backend_only: bool = True
    visible_run_control_enabled: bool = False
    visible_run_control_present: bool = False
    execution_invoked: bool = False
    files_written: bool = False
    exposes_manifest_path_to_user: bool = False
    exposes_internal_cli_to_user: bool = False


_SUMMARIES = {
    "no_validation": "Validate the Guided setup before running.",
    "validation_not_accepted": (
        "The Guided setup was not accepted. Resolve its validation issues first."
    ),
    "validation_stale": (
        "The Guided setup changed after validation. Validate again before running."
    ),
    "authorization_missing": (
        "Guided validation succeeded, but Guided Run execution is unavailable "
        "in this build."
    ),
    "authorization_not_accepted": (
        "Guided Run could not authorize the validated setup."
    ),
    "authorization_stale": (
        "The Guided setup changed after authorization. Validate again before running."
    ),
    "payload_missing": "Guided Run has not prepared the validated setup yet.",
    "payload_not_ready": (
        "Guided Run could not prepare the validated setup for execution."
    ),
    "backend_unavailable": "Guided Run is not available in this build.",
    "ready_hidden": "Guided Run is ready to start.",
}


def _result(
    *,
    status: str,
    validation_revision: int | None,
    current_gui_revision: int,
    authorization: GuidedRunAuthorizationResult | None,
    payload: GuidedExecutionPayloadDerivationResult | None,
    backend_execution_available: bool,
    startup_orchestration_available: bool,
) -> GuidedRunReadinessResult:
    ready = status == "ready_hidden"
    summary = _SUMMARIES[status]
    issue = (
        ()
        if ready
        else (
            GuidedRunReadinessIssue(
                category=status,
                section="guided_run_readiness",
                message=summary,
                user_safe_message=summary,
            ),
        )
    )
    state = (
        "ready_for_future_run_hidden"
        if ready
        else (
            "needs_validation"
            if status
            in {
                "no_validation",
                "validation_not_accepted",
            }
            else (
                "needs_revalidation"
                if status in {"validation_stale", "authorization_stale"}
                else "cannot_run"
            )
        )
    )
    return GuidedRunReadinessResult(
        status=status,
        ready=ready,
        user_visible_state=state,
        user_summary=summary,
        blocking_issues=issue,
        validation_revision=validation_revision,
        current_gui_revision=current_gui_revision,
        authorization_identity=(
            authorization.canonical_authorization_identity
            if isinstance(authorization, GuidedRunAuthorizationResult)
            else None
        ),
        payload_status=(
            payload.status
            if isinstance(payload, GuidedExecutionPayloadDerivationResult)
            else None
        ),
        backend_execution_available=backend_execution_available,
        startup_orchestration_available=startup_orchestration_available,
    )


def _first_subset_current(authorization: GuidedRunAuthorizationResult) -> bool:
    intent = authorization.production_intent
    return bool(
        intent is not None
        and intent.input_source.source_format == "rwd"
        and intent.acquisition.acquisition_mode == "intermittent"
        and intent.execution_profile.execution_mode in {"phasic", "tonic", "both"}
        and intent.execution_profile.run_type == "full"
        and intent.execution_profile.traces_only is False
        and intent.roi_scope.selection_mode == "include"
        and bool(intent.roi_scope.included_roi_ids)
        and intent.correction.strategy_scope == "global"
        and intent.output_policy.overwrite is False
        and intent.output_policy.precreate is False
    )


def evaluate_guided_run_readiness(
    *,
    validation_outcome: GuidedBackendValidationWorkflowOutcome | None,
    validation_revision: int | None,
    current_gui_revision: int,
    authorization_result: GuidedRunAuthorizationResult | None = None,
    payload_result: GuidedExecutionPayloadDerivationResult | None = None,
    backend_execution_available: bool = True,
    startup_orchestration_available: bool = True,
) -> GuidedRunReadinessResult:
    """Inspect supplied state without validation, authorization, writes, or execution."""
    status = "ready_hidden"
    if not isinstance(validation_outcome, GuidedBackendValidationWorkflowOutcome):
        status = "no_validation"
    elif (
        validation_outcome.status != "validator_accepted"
        or validation_outcome.accepted_for_backend_validation is not True
    ):
        status = "validation_not_accepted"
    elif (
        validation_outcome.stale is not False
        or validation_revision != current_gui_revision
    ):
        status = "validation_stale"
    elif not isinstance(authorization_result, GuidedRunAuthorizationResult):
        status = "authorization_missing"
    elif (
        authorization_result.status != "authorized"
        or authorization_result.authorized is not True
        or authorization_result.run_authorization is not True
    ):
        status = "authorization_not_accepted"
    elif authorization_result.authorized_gui_revision != current_gui_revision:
        status = "authorization_stale"
    elif not isinstance(payload_result, GuidedExecutionPayloadDerivationResult):
        status = "payload_missing"
    elif (
        payload_result.status != GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE
        or payload_result.ok is not True
        or payload_result.runnable is not False
        or payload_result.runner_request is not None
        or payload_result.runner_request_identity is not None
        or len(payload_result.limiting_issues) != 1
        or payload_result.limiting_issues[0].category
        != "startup_transaction_unavailable"
        or payload_result.blocking_issues
        or not _first_subset_current(authorization_result)
    ):
        status = "payload_not_ready"
    elif (
        backend_execution_available is not True
        or startup_orchestration_available is not True
    ):
        status = "backend_unavailable"
    return _result(
        status=status,
        validation_revision=validation_revision,
        current_gui_revision=current_gui_revision,
        authorization=authorization_result,
        payload=payload_result,
        backend_execution_available=backend_execution_available,
        startup_orchestration_available=startup_orchestration_available,
    )
