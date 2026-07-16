"""Map a Guided NPM parent result into a small scientist-facing vocabulary.

This module holds no backend evidence logic of its own. It only classifies
an already-computed reconciliation `final_outcome` (or a controlled
background-error marker) into one of three GUI-facing categories. Only
`final_outcome == "verified_completed"` may produce success; every other
value -- including any this module does not explicitly recognize -- is
treated as unconfirmed, never as success.
"""

from __future__ import annotations

from dataclasses import dataclass


GUIDED_NPM_RUN_OUTCOME_SUCCESS = "completed_successfully"
GUIDED_NPM_RUN_OUTCOME_FAILED = "did_not_complete"
GUIDED_NPM_RUN_OUTCOME_UNCONFIRMED = "could_not_confirm"

_VERIFIED_SUCCESS_OUTCOME = "verified_completed"

_DEFINITE_FAILURE_OUTCOMES = frozenset(
    {
        "verified_failed_before_consumed_authority",
        "verified_failed_after_consumed_authority",
        "verified_failed_during_output_finalization",
        "process_failed_without_terminal_evidence",
        "authority_refused",
    }
)

GUIDED_NPM_LAUNCH_FAILURE_PRIMARY_TEXT = (
    "The analysis could not be started. Review the highlighted setup "
    "issue, then check the setup again."
)

_LAUNCH_FAILURE_CANCELLED_CATEGORIES = frozenset({"launch_cancelled"})

# Stable category strings observed in GuidedNpmWorkerLaunchFailure.
# blocking_issues[0].category (guided_npm_worker_launch.py) and in
# GuidedNpmRunLaunchBuildResult.status (guided_npm_run_launch_builder.py)
# that mean the validated setup is no longer current and must be checked
# again, as opposed to an internal/environment failure.
_LAUNCH_FAILURE_STALE_SETUP_CATEGORIES = frozenset(
    {
        "current_build_invalid",
        "current_build_mismatch",
        "launch_worker_artifact_changed",
        "launch_startup_artifact_changed",
        "launch_source_freshness_changed",
        "launch_invocation_invalid",
        "launch_invocation_identity_mismatch",
        "prelaunch_claim_invalid",
        "prelaunch_claim_state_invalid",
        "invalid_context",
        "validation_not_current",
        "validation_not_accepted",
        "build_identity_unavailable",
        "plan_identity_unavailable",
    }
)


@dataclass(frozen=True)
class GuidedNpmRunUnexpectedError:
    """A controlled marker for a background reconciliation exception."""

    message: str


@dataclass(frozen=True)
class GuidedNpmRunResultPresentation:
    category: str
    title: str
    detail: str
    output_directory: str | None


def present_guided_npm_run_result(
    result: object,
) -> GuidedNpmRunResultPresentation:
    """Classify a reconciliation result or a controlled error marker.

    ``result`` is expected to be a ``GuidedNpmWorkerReconciliationResult``,
    a ``GuidedNpmPostLaunchReconciliation``, or a
    ``GuidedNpmRunUnexpectedError``. Any other shape (including an
    unrecognized future outcome value) is treated as unconfirmed.
    """
    if isinstance(result, GuidedNpmRunUnexpectedError):
        return GuidedNpmRunResultPresentation(
            category=GUIDED_NPM_RUN_OUTCOME_UNCONFIRMED,
            title=(
                "The application could not confirm that the analysis "
                "finished correctly."
            ),
            detail="Do not treat the output folder as a completed analysis.",
            output_directory=None,
        )

    final_outcome = getattr(result, "final_outcome", None)

    if final_outcome == _VERIFIED_SUCCESS_OUTCOME:
        return GuidedNpmRunResultPresentation(
            category=GUIDED_NPM_RUN_OUTCOME_SUCCESS,
            title="Your NPM analysis finished successfully.",
            detail="",
            output_directory=getattr(result, "run_directory_path", None),
        )

    if final_outcome in _DEFINITE_FAILURE_OUTCOMES:
        return GuidedNpmRunResultPresentation(
            category=GUIDED_NPM_RUN_OUTCOME_FAILED,
            title="The analysis did not finish.",
            detail="Review your setup, then check it again before running.",
            output_directory=None,
        )

    return GuidedNpmRunResultPresentation(
        category=GUIDED_NPM_RUN_OUTCOME_UNCONFIRMED,
        title=(
            "The application could not confirm that the analysis finished "
            "correctly."
        ),
        detail="Do not treat the output folder as a completed analysis.",
        output_directory=None,
    )


def present_guided_npm_launch_failure_detail(category: str | None) -> str:
    """Map a stable launch-failure/build-failure category to one safe,
    scientist-facing detail sentence.

    ``category`` is expected to be either a
    ``GuidedNpmWorkerLaunchFailure.blocking_issues[0].category`` value or a
    ``GuidedNpmRunLaunchBuildResult.status`` value -- both are internal
    stable identifiers, never displayed verbatim. This function never
    echoes a raw backend issue message, exception text, detail code, or
    path; any category it does not recognize (including one it has never
    seen before) falls through to the generic "could not start" sentence.
    """
    if category in _LAUNCH_FAILURE_CANCELLED_CATEGORIES:
        return "The analysis was not started."
    if category in _LAUNCH_FAILURE_STALE_SETUP_CATEGORIES:
        return "The setup must be checked again before running."
    return "The application could not start the analysis."
