from __future__ import annotations

from types import SimpleNamespace

import pytest

from photometry_pipeline.guided_backend_validation_materialization import (
    _npm_inspection_user_message,
    _npm_normalized_recording_user_message,
    _npm_settings_incomplete_message,
    _npm_settings_mismatch_message,
    _npm_source_snapshot_user_message,
)
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationWorkflowOutcome,
)
from photometry_pipeline.guided_run_readiness import evaluate_guided_run_readiness


_FORBIDDEN_PRIMARY_TERMS = (
    "adapter",
    "backend",
    "contract",
    "materialization",
    "normalized",
    "parser policy",
    "provenance",
    "authorization",
    "identity",
    "digest",
    "schema",
    "canonical",
)


def _assert_scientist_facing(text: str) -> None:
    lowered = text.lower()
    for term in _FORBIDDEN_PRIMARY_TERMS:
        assert term not in lowered, f"{term!r} leaked into visible text: {text!r}"


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (
            "npm_time_column_missing",
            "The timestamp column specified in the NPM settings was not found.",
        ),
        (
            "npm_led_column_missing",
            "The LED-state column specified in the NPM settings was not found.",
        ),
        (
            "npm_roi_columns_missing",
            "No ROI columns matched the current NPM naming pattern.",
        ),
        (
            "npm_non_monotonic_timestamp",
            "Timestamps are not in increasing order in one or more NPM sessions.",
        ),
        (
            "npm_insufficient_overlap_support",
            "The signal and reference samples do not overlap enough to analyze this recording.",
        ),
        (
            "npm_roi_inventory_mismatch",
            "The ROI columns are not consistent across all selected NPM sessions.",
        ),
    ],
)
def test_npm_inspection_failures_use_plain_recording_problems(category, expected):
    message = _npm_inspection_user_message(category)
    _assert_scientist_facing(message)
    assert message.startswith(
        "The app could not determine how to read this NPM recording from the current settings."
    )
    assert expected in message


def test_npm_settings_failures_are_actionable():
    incomplete = _npm_settings_incomplete_message()
    mismatch = _npm_settings_mismatch_message()
    for message in (incomplete, mismatch):
        _assert_scientist_facing(message)
        assert "NPM settings step" in message
        assert "rerun Setup check" in message
    assert incomplete.startswith("The NPM import settings are incomplete.")
    assert mismatch.startswith(
        "The current NPM import settings do not match the settings applied to this recording."
    )


def test_npm_source_and_roi_failures_identify_the_recording_problem():
    source_message = _npm_source_snapshot_user_message("no_npm_csv_files")
    roi_message = _npm_normalized_recording_user_message(
        "npm_physical_roi_mapping_mismatch"
    )
    _assert_scientist_facing(source_message)
    _assert_scientist_facing(roi_message)
    assert "No NPM CSV recordings were found" in source_message
    assert "ROI columns are not consistent" in roi_message


def _accepted_npm_outcome():
    return GuidedBackendValidationWorkflowOutcome(
        status="validator_accepted",
        accepted_for_backend_validation=True,
        run_authorization=False,
        request_identity="a" * 64,
        validation_result=SimpleNamespace(accepted=True),
        compile_result=SimpleNamespace(
            request=SimpleNamespace(
                source=SimpleNamespace(source_format="npm"),
            )
        ),
        materialization_result=SimpleNamespace(),
        blocking_issues=(),
        user_summary=(
            "This NPM recording setup was checked successfully. Running NPM "
            "analyses is not available yet."
        ),
    )


def test_validated_npm_readiness_keeps_run_unavailable_message():
    result = evaluate_guided_run_readiness(
        validation_outcome=_accepted_npm_outcome(),
        validation_revision=7,
        current_gui_revision=7,
    )
    assert result.status == "validated_npm_not_available"
    assert result.ready is False
    assert result.visible_run_control_enabled is False
    assert result.user_summary == (
        "This NPM recording setup was checked successfully. Running NPM analyses "
        "is not available yet."
    )
    _assert_scientist_facing(result.user_summary)
