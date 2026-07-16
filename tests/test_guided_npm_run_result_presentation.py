from __future__ import annotations

from types import SimpleNamespace

import pytest

from photometry_pipeline.guided_npm_run_result_presentation import (
    GUIDED_NPM_LAUNCH_FAILURE_PRIMARY_TEXT,
    GUIDED_NPM_RUN_OUTCOME_FAILED,
    GUIDED_NPM_RUN_OUTCOME_SUCCESS,
    GUIDED_NPM_RUN_OUTCOME_UNCONFIRMED,
    GuidedNpmRunUnexpectedError,
    present_guided_npm_launch_failure_detail,
    present_guided_npm_run_result,
)


def _result(final_outcome, **kwargs):
    return SimpleNamespace(final_outcome=final_outcome, **kwargs)


def test_only_verified_completed_is_success():
    presentation = present_guided_npm_run_result(
        _result("verified_completed", run_directory_path=r"C:\out\run")
    )
    assert presentation.category == GUIDED_NPM_RUN_OUTCOME_SUCCESS
    assert presentation.title == "Your NPM analysis finished successfully."
    assert presentation.output_directory == r"C:\out\run"


@pytest.mark.parametrize(
    "final_outcome",
    [
        "verified_failed_before_consumed_authority",
        "verified_failed_after_consumed_authority",
        "verified_failed_during_output_finalization",
        "process_failed_without_terminal_evidence",
        "authority_refused",
    ],
)
def test_definite_failure_outcomes_map_to_failed(final_outcome):
    presentation = present_guided_npm_run_result(_result(final_outcome))
    assert presentation.category == GUIDED_NPM_RUN_OUTCOME_FAILED
    assert presentation.title == "The analysis did not finish."
    assert presentation.output_directory is None


@pytest.mark.parametrize(
    "final_outcome",
    [
        "terminal_receipt_publication_failed",
        "process_exited_zero_without_terminal_evidence",
        "terminal_evidence_invalid",
        "consumed_authority_evidence_invalid",
        "process_identity_mismatch",
        "completed_output_integrity_failed",
        "indeterminate",
        "post_launch_evidence_failed",
        "some_future_outcome_this_module_has_never_seen",
        None,
        "",
    ],
)
def test_unconfirmed_and_unrecognized_outcomes_never_claim_success(final_outcome):
    presentation = present_guided_npm_run_result(_result(final_outcome))
    assert presentation.category == GUIDED_NPM_RUN_OUTCOME_UNCONFIRMED
    assert presentation.output_directory is None
    assert "successfully" not in presentation.title.lower()


def test_unexpected_error_marker_is_unconfirmed_never_success():
    presentation = present_guided_npm_run_result(
        GuidedNpmRunUnexpectedError("boom")
    )
    assert presentation.category == GUIDED_NPM_RUN_OUTCOME_UNCONFIRMED
    assert presentation.output_directory is None


def test_shapeless_object_is_unconfirmed_never_success():
    presentation = present_guided_npm_run_result(object())
    assert presentation.category == GUIDED_NPM_RUN_OUTCOME_UNCONFIRMED


def test_success_requires_run_directory_field_not_a_side_channel():
    # A "verified_completed" result missing run_directory_path must still
    # report success (the field is optional metadata for display, not part
    # of the success gate) but with no directory to show.
    presentation = present_guided_npm_run_result(
        SimpleNamespace(final_outcome="verified_completed")
    )
    assert presentation.category == GUIDED_NPM_RUN_OUTCOME_SUCCESS
    assert presentation.output_directory is None


# ---------------------------------------------------------------------------
# present_guided_npm_launch_failure_detail (B2-E1 narrow follow-up)
# ---------------------------------------------------------------------------


def test_launch_cancelled_category_maps_to_not_started():
    assert (
        present_guided_npm_launch_failure_detail("launch_cancelled")
        == "The analysis was not started."
    )


@pytest.mark.parametrize(
    "category",
    [
        # from GuidedNpmWorkerLaunchFailure.blocking_issues[0].category
        "current_build_invalid",
        "current_build_mismatch",
        "launch_worker_artifact_changed",
        "launch_startup_artifact_changed",
        "launch_source_freshness_changed",
        "launch_invocation_invalid",
        "launch_invocation_identity_mismatch",
        "prelaunch_claim_invalid",
        "prelaunch_claim_state_invalid",
        # from GuidedNpmRunLaunchBuildResult.status
        "invalid_context",
        "validation_not_current",
        "validation_not_accepted",
        "build_identity_unavailable",
        "plan_identity_unavailable",
    ],
)
def test_stale_setup_categories_map_to_check_again(category):
    assert (
        present_guided_npm_launch_failure_detail(category)
        == "The setup must be checked again before running."
    )


@pytest.mark.parametrize(
    "category",
    [
        "process_creation_failed",
        "launch_internal_error",
        "process_created_receipt_failed",
        "process_identity_invalid",
        "launch_executable_invalid",
        "launch_entry_point_missing",
        "launch_working_directory_invalid",
        "launch_context_persistence_failed",
        "launch_context_cleanup_failed",
        "unsupported_format",
        "production_mapping_failed",
        "worker_request_materialization_failed",
        "some_category_this_module_has_never_seen",
        None,
        "",
    ],
)
def test_internal_and_unrecognized_categories_map_to_generic_could_not_start(category):
    assert (
        present_guided_npm_launch_failure_detail(category)
        == "The application could not start the analysis."
    )


def test_launch_failure_primary_text_is_scientist_facing():
    lowered = GUIDED_NPM_LAUNCH_FAILURE_PRIMARY_TEXT.lower()
    for term in ("worker", "receipt", "artifact", "backend", "json", "exception"):
        assert term not in lowered
