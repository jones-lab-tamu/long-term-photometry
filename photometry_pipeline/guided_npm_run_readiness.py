"""Pure Guided NPM Run enablement predicate.

Mirrors ``guided_run_readiness.evaluate_guided_run_readiness`` (the
existing RWD predicate) but is scoped only to the NPM launch path added in
B2-E1: it never inspects ``GuidedRunAuthorizationResult`` or
``GuidedExecutionPayloadDerivationResult`` (both RWD-only), and it does not
require a prelaunch claim to already exist -- the claim is built (with its
accompanying disk writes) only once Run is actually pressed.
"""

from __future__ import annotations

from dataclasses import dataclass

from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationWorkflowOutcome,
)


@dataclass(frozen=True)
class GuidedNpmRunReadinessResult:
    status: str
    ready: bool
    user_summary: str
    validation_revision: int | None
    current_gui_revision: int


_SUMMARIES = {
    "no_validation": "Check your Guided setup before running analysis.",
    "validation_not_accepted": (
        "Your Guided setup was not accepted. Resolve the issues shown, "
        "then check it again."
    ),
    "validation_stale": (
        "The Guided setup changed after it was checked. Check it again "
        "before running."
    ),
    "run_active": "Your NPM analysis is already running.",
    "result_pending": (
        "Review the result of the last run before starting a new one."
    ),
    "ready": (
        "This NPM recording setup was checked successfully and is ready "
        "to run."
    ),
}


def evaluate_guided_npm_run_readiness(
    *,
    validation_outcome: GuidedBackendValidationWorkflowOutcome | None,
    validation_revision: int | None,
    current_gui_revision: int,
    execution_active: bool,
    execution_result_pending: bool,
) -> GuidedNpmRunReadinessResult:
    """Inspect supplied state without validation, writes, or execution."""
    status = "ready"
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
    elif execution_active:
        status = "run_active"
    elif execution_result_pending:
        status = "result_pending"
    ready = status == "ready"
    return GuidedNpmRunReadinessResult(
        status=status,
        ready=ready,
        user_summary=_SUMMARIES[status],
        validation_revision=validation_revision,
        current_gui_revision=current_gui_revision,
    )
