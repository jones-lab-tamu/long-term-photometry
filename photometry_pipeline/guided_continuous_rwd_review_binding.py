"""Pure in-memory pre-Review binding for one continuous RWD recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    CONTINUITY_PASSED,
    ContinuousRwdDiscontinuityEvaluation,
)
from photometry_pipeline.guided_continuous_rwd_discontinuity_policy import (
    POLICY_NAME,
    POLICY_VERSION,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    ContinuousRwdRecordingAuthorityError,
    GuidedContinuousRwdRecordingDescription,
    _validate_description,
)
from photometry_pipeline.guided_new_analysis_plan import (
    SCHEMA_VERSION as GUIDED_DRAFT_SCHEMA_VERSION,
    GuidedNewAnalysisDraftPlan,
)
from photometry_pipeline.guided_plan_identity import (
    compute_guided_new_analysis_draft_plan_identity,
)


class GuidedContinuousRwdReviewBindingError(ValueError):
    """The supplied draft and scientific authorities cannot be bound."""


@dataclass(frozen=True)
class GuidedContinuousRwdReviewBinding:
    """One current draft bound to its exact accepted recording and continuity pass."""

    draft_plan_identity: str
    recording: GuidedContinuousRwdRecordingDescription
    continuity_evaluation: ContinuousRwdDiscontinuityEvaluation
    current_source_path: str


def _fail(message: str) -> None:
    raise GuidedContinuousRwdReviewBindingError(message)


def build_guided_continuous_rwd_review_binding(
    draft: GuidedNewAnalysisDraftPlan,
    *,
    recording: GuidedContinuousRwdRecordingDescription,
    continuity_evaluation: ContinuousRwdDiscontinuityEvaluation,
    current_source_path: str | Path,
) -> GuidedContinuousRwdReviewBinding:
    """Build a non-executing binding without inspecting or changing the source."""
    try:
        draft_plan_identity = compute_guided_new_analysis_draft_plan_identity(draft)
    except (TypeError, ValueError, AttributeError) as exc:
        raise GuidedContinuousRwdReviewBindingError(
            "Guided draft authority is invalid."
        ) from exc
    if draft.schema_version != GUIDED_DRAFT_SCHEMA_VERSION or draft.mode != "new_analysis":
        _fail("Guided draft authority is invalid.")
    if draft.input_format != "rwd" or draft.acquisition_mode != "continuous":
        _fail("Guided draft must describe one continuous RWD recording.")

    try:
        _validate_description(recording)
    except (ContinuousRwdRecordingAuthorityError, TypeError, ValueError) as exc:
        raise GuidedContinuousRwdReviewBindingError(
            "Continuous RWD recording authority is invalid."
        ) from exc

    available_roi_ids = tuple(
        channel.roi_id for channel in recording.roi.available_roi_channels
    )
    if tuple(draft.discovered_roi_ids) != available_roi_ids:
        _fail("Guided draft discovered ROI inventory does not match the recording.")
    if tuple(sorted(str(roi_id) for roi_id in draft.included_roi_ids)) != (
        recording.roi.included_roi_ids
    ):
        _fail("Guided draft included ROI IDs do not match the recording.")
    if tuple(sorted(str(roi_id) for roi_id in draft.excluded_roi_ids)) != (
        recording.roi.excluded_roi_ids
    ):
        _fail("Guided draft excluded ROI IDs do not match the recording.")

    if not isinstance(continuity_evaluation, ContinuousRwdDiscontinuityEvaluation):
        _fail("Continuity evaluation authority is invalid.")
    evaluation = continuity_evaluation
    if evaluation.outcome != CONTINUITY_PASSED:
        _fail("Continuity evaluation did not pass.")
    if evaluation.failure_reason is not None:
        _fail("Passing continuity evaluation must not contain a failure reason.")

    identity_pairs = (
        (evaluation.recording_identity, recording.recording_identity, "recording"),
        (
            evaluation.source_content_identity,
            recording.source.source_content_identity,
            "source-content",
        ),
        (
            evaluation.parser_interpretation_identity,
            recording.source.parser_interpretation_identity,
            "parser",
        ),
        (
            evaluation.cadence_evidence_identity,
            recording.cadence.cadence_evidence_identity,
            "cadence",
        ),
    )
    for actual, expected, name in identity_pairs:
        if actual != expected:
            _fail(f"Continuity {name} identity does not match the recording.")

    if (
        evaluation.observed_source_sha256 != recording.source.sha256
        or evaluation.observed_source_size_bytes != recording.source.file_size_bytes
    ):
        _fail("Continuity observed source facts do not match the recording.")
    if evaluation.policy_name != POLICY_NAME or evaluation.policy_version != POLICY_VERSION:
        _fail("Continuity policy does not match the committed policy.")

    valid_rows = evaluation.valid_row_count_evaluated
    positive_intervals = evaluation.positive_interval_count_evaluated
    if valid_rows != recording.source.valid_timestamp_count:
        _fail("Continuity valid-row count does not match the recording.")
    if positive_intervals != recording.cadence.positive_interval_count:
        _fail("Continuity positive-interval count does not match the recording.")
    if positive_intervals != valid_rows - 1:
        _fail("Continuity interval and row counts are inconsistent.")
    if evaluation.normal_interval_count != positive_intervals:
        _fail("Passing continuity normal-interval count is inconsistent.")
    if (
        evaluation.short_interval_anomaly_count != 0
        or evaluation.material_long_interval_count != 0
        or evaluation.short_examples != ()
        or evaluation.long_examples != ()
        or evaluation.maximum_short_residual_seconds is not None
        or evaluation.maximum_long_residual_seconds is not None
    ):
        _fail("Passing continuity result contains discontinuity evidence.")

    if not isinstance(current_source_path, (str, Path)):
        _fail("Current source path must be a string or Path.")
    source_path = str(current_source_path)
    if not source_path.strip():
        _fail("Current source path must be nonempty.")
    draft_source_folder = draft.resolved_input_source_path or draft.input_source_path
    if not isinstance(draft_source_folder, str) or not draft_source_folder.strip():
        _fail("Guided draft source folder is invalid.")
    current_source = Path(source_path)
    if current_source.parent != Path(draft_source_folder):
        _fail("Current source path does not belong to the Guided draft source folder.")
    accepted_source = Path(recording.source.fluorescence_path_canonical)
    if (
        not accepted_source.name
        or accepted_source.parent
        != Path(recording.source.selected_folder_canonical)
    ):
        _fail("Accepted B1 source filename provenance is invalid.")
    if current_source.name != accepted_source.name:
        _fail("Current source filename does not match the accepted B1 source.")

    return GuidedContinuousRwdReviewBinding(
        draft_plan_identity=draft_plan_identity,
        recording=recording,
        continuity_evaluation=evaluation,
        current_source_path=source_path,
    )
