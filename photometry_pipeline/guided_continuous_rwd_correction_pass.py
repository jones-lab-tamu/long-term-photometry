"""CR1-C4c: complete second-pass corrected-segment traversal and completion authority.

This module is pure orchestration. It opens exactly one fresh C3b ordered
projection traversal, reuses the accepted C4a public path to assemble raw
correction segments from it, and delegates every one-segment correction to
the accepted C4b public entry point. It contains no filtering, fitting,
fallback, Signal-Only, delta-F, dF/F, or QC mathematics of its own.

Provisional-vs-final contract
------------------------------
``iterate_guided_continuous_rwd_corrected_segments`` returns a
``GuidedContinuousRwdCorrectionPassTraversal``: a one-shot iterator. Iterating
it yields the exact immutable ``GuidedContinuousRwdCorrectedSegment`` objects
produced by C4b, in canonical segment-plan order. Every yielded segment
remains scoped to its own segment (it carries no field claiming the recording
finished correcting) and is therefore provisional by construction.

The only way to obtain a ``GuidedContinuousRwdCorrectionPassCompletion`` is
through the traversal's ``completion`` property, which raises
``GuidedContinuousRwdCorrectionPassError`` until the traversal has fully and
successfully exhausted. Cancellation, a mid-traversal failure, or an
ordering/coverage violation leaves the traversal in a ``"cancelled"`` or
``"failed"`` terminal state, and ``completion`` never becomes available from
that state -- there is no separate constructor path a caller could reach for
it instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterator

from photometry_pipeline.guided_continuous_rwd_block_plan import (
    GuidedContinuousRwdBlockPlan,
)
from photometry_pipeline.guided_continuous_rwd_correction_segments import (
    GuidedContinuousRwdCorrectionSegmentError,
    GuidedContinuousRwdCorrectionSegmentPlan,
    GuidedContinuousRwdDynamicF0Authority,
    _resolve_accepted_correction_context,
    _validate_dynamic_f0_authority,
    _validate_segment_plan,
    iter_assemble_guided_continuous_rwd_correction_segments,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBinding,
)
from photometry_pipeline.guided_continuous_rwd_segment_correction import (
    GuidedContinuousRwdCorrectedSegment,
    GuidedContinuousRwdSegmentCorrectionError,
    _resolve_segment_correction_settings,
    correct_guided_continuous_rwd_segment,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    GuidedContinuousRwdTargetGridDescription,
)
from photometry_pipeline.guided_execution_payloads import (
    GuidedExecutionStartupMappingContract,
)
from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.io.rwd_continuous_projection_reader import (
    ContinuousRwdProjectionReaderError,
    _validate_authorities,
    iter_project_guided_continuous_rwd_blocks,
)


SCHEMA_NAME = "guided_continuous_rwd_correction_pass_completion"
SCHEMA_VERSION = "v1"
POLICY_NAME = "second-pass-full-recording-correction-traversal"
POLICY_VERSION = "v1"
COMPLETION_IDENTITY_DOMAIN = "guided-continuous-rwd-correction-pass-completion:v1"
COMPLETION_STATE = "complete_all_segments_verified"

# Each reused lower-layer module raises its own cancellation exception type
# with its own narrow category name for "cancellation_requested() was true".
# Mapping the exact (type, category) pair -- rather than matching on category
# string alone -- avoids ever misclassifying an unrelated same-named failure
# from a different exception type as cancellation.
_LOWER_LAYER_CANCELLATION_CATEGORIES: dict[type, str] = {
    ContinuousRwdProjectionReaderError: "projection_interrupted",
    GuidedContinuousRwdCorrectionSegmentError: "f0_preparation_interrupted",
    GuidedContinuousRwdSegmentCorrectionError: "segment_correction_interrupted",
}


def _is_lower_layer_cancellation(exc: Exception) -> bool:
    expected_category = _LOWER_LAYER_CANCELLATION_CATEGORIES.get(type(exc))
    return expected_category is not None and getattr(exc, "category", None) == expected_category


ERROR_CATEGORIES = frozenset(
    {
        "accepted_correction_binding_mismatch",
        "segment_order_mismatch",
        "segment_identity_mismatch",
        "incomplete_traversal",
        "segment_correction_pass_interrupted",
        "segment_correction_pass_failed",
        "pass_already_terminal",
        "completion_not_available",
        "completion_identity_mismatch",
    }
)


class GuidedContinuousRwdCorrectionPassError(ValueError):
    """A narrow C4c refusal while orchestrating the full second-pass traversal."""

    def __init__(self, category: str, message: str, **context: Any) -> None:
        if category not in ERROR_CATEGORIES:
            raise ValueError(f"Unsupported CR1-C4c error category: {category!r}")
        super().__init__(message)
        self.category = category
        self.context = dict(context)


def _raise(category: str, message: str, **context: Any) -> None:
    raise GuidedContinuousRwdCorrectionPassError(category, message, **context)


def _check_cancellation(callback: Callable[[], bool] | None) -> None:
    if callback is None:
        return
    if not callable(callback):
        _raise(
            "segment_correction_pass_interrupted",
            "cancellation_requested must be callable or None.",
        )
    try:
        interrupted = callback()
    except Exception as exc:
        raise GuidedContinuousRwdCorrectionPassError(
            "segment_correction_pass_interrupted",
            "Cancellation callback failed.",
        ) from exc
    if type(interrupted) is not bool:
        _raise(
            "segment_correction_pass_interrupted",
            "Cancellation callback must return bool.",
        )
    if interrupted:
        _raise(
            "segment_correction_pass_interrupted",
            "The corrected-segment traversal was cancelled.",
        )


@dataclass(frozen=True)
class _ExpectedCompletionBindings:
    """Upstream identities/counts resolved once, before any traversal opens."""

    recording_identity: str
    source_content_identity: str
    target_grid_identity: str
    block_ownership_schema_name: str
    block_ownership_schema_version: str
    block_ownership_policy_name: str
    block_ownership_policy_version: str
    correction_segment_plan_identity: str
    dynamic_f0_authority_identity: str
    accepted_guided_plan_identity: str
    startup_mapping_contract_version: str
    correction_payload_identity: str
    fixed_correction_settings_identity: str
    segment_correction_settings_identity: str
    segment_count: int
    target_sample_count: int


@dataclass(frozen=True)
class GuidedContinuousRwdCorrectionPassCompletion:
    schema_name: str
    schema_version: str
    policy_name: str
    policy_version: str
    recording_identity: str
    source_content_identity: str
    target_grid_identity: str
    block_ownership_schema_name: str
    block_ownership_schema_version: str
    block_ownership_policy_name: str
    block_ownership_policy_version: str
    correction_segment_plan_identity: str
    dynamic_f0_authority_identity: str
    accepted_guided_plan_identity: str
    startup_mapping_contract_version: str
    correction_payload_identity: str
    fixed_correction_settings_identity: str
    segment_correction_settings_identity: str
    corrected_segment_count: int
    target_sample_count: int
    ordered_segment_identity_digest: str
    completion_state: str
    completion_identity: str


def _completion_identity_payload(
    completion: GuidedContinuousRwdCorrectionPassCompletion,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in completion.__dict__.items()
        if key != "completion_identity"
    }


def _compute_completion_identity(
    completion: GuidedContinuousRwdCorrectionPassCompletion,
) -> str:
    return hashlib.sha256(
        COMPLETION_IDENTITY_DOMAIN.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(_completion_identity_payload(completion))
    ).hexdigest()


def _identity(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _raise(
            "completion_identity_mismatch",
            f"{name} must be a lowercase 64-character identity.",
        )


def _validate_completion(
    completion: object,
    *,
    expected: _ExpectedCompletionBindings,
) -> None:
    if not isinstance(completion, GuidedContinuousRwdCorrectionPassCompletion):
        _raise("completion_identity_mismatch", "Completion authority has the wrong type.")
    if (
        completion.schema_name != SCHEMA_NAME
        or completion.schema_version != SCHEMA_VERSION
        or completion.policy_name != POLICY_NAME
        or completion.policy_version != POLICY_VERSION
        or completion.recording_identity != expected.recording_identity
        or completion.source_content_identity != expected.source_content_identity
        or completion.target_grid_identity != expected.target_grid_identity
        or completion.block_ownership_schema_name != expected.block_ownership_schema_name
        or completion.block_ownership_schema_version != expected.block_ownership_schema_version
        or completion.block_ownership_policy_name != expected.block_ownership_policy_name
        or completion.block_ownership_policy_version != expected.block_ownership_policy_version
        or completion.correction_segment_plan_identity != expected.correction_segment_plan_identity
        or completion.dynamic_f0_authority_identity != expected.dynamic_f0_authority_identity
        or completion.accepted_guided_plan_identity != expected.accepted_guided_plan_identity
        or completion.startup_mapping_contract_version != expected.startup_mapping_contract_version
        or completion.correction_payload_identity != expected.correction_payload_identity
        or completion.fixed_correction_settings_identity != expected.fixed_correction_settings_identity
        or completion.segment_correction_settings_identity != expected.segment_correction_settings_identity
        or completion.corrected_segment_count != expected.segment_count
        or completion.target_sample_count != expected.target_sample_count
        or completion.completion_state != COMPLETION_STATE
    ):
        _raise("completion_identity_mismatch", "Completion authority provenance is invalid.")
    for value, name in (
        (completion.correction_segment_plan_identity, "Correction-segment-plan identity"),
        (completion.dynamic_f0_authority_identity, "Dynamic-F0 authority identity"),
        (completion.accepted_guided_plan_identity, "Accepted Guided plan identity"),
        (completion.correction_payload_identity, "Correction-payload identity"),
        (completion.fixed_correction_settings_identity, "C4a fixed-settings identity"),
        (completion.segment_correction_settings_identity, "C4b segment-settings identity"),
        (completion.ordered_segment_identity_digest, "Ordered segment-identity digest"),
        (completion.completion_identity, "Completion identity"),
    ):
        _identity(value, name)
    if completion.completion_identity != _compute_completion_identity(completion):
        _raise("completion_identity_mismatch", "Completion identity mismatch.")


def _resolve_expected_bindings(
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    segment_plan: GuidedContinuousRwdCorrectionSegmentPlan,
    dynamic_f0_authority: GuidedContinuousRwdDynamicF0Authority,
    accepted_draft: GuidedNewAnalysisDraftPlan,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
) -> _ExpectedCompletionBindings:
    """Validate every accepted upstream authority once, before any traversal opens."""
    try:
        _validate_authorities(review_binding, target_grid, block_plan)
        accepted = _resolve_accepted_correction_context(
            review_binding, accepted_draft, startup_mapping_contract
        )
        _validate_segment_plan(segment_plan, target_grid, review_binding, accepted)
        _validate_dynamic_f0_authority(
            dynamic_f0_authority,
            review_binding=review_binding,
            target_grid=target_grid,
            segment_plan=segment_plan,
            accepted_context=accepted,
        )
        _, segment_correction_settings_identity = _resolve_segment_correction_settings(
            startup_mapping_contract
        )
    except Exception as exc:
        raise GuidedContinuousRwdCorrectionPassError(
            "accepted_correction_binding_mismatch",
            "Accepted upstream authorities are incompatible.",
            reason=str(exc),
        ) from exc
    return _ExpectedCompletionBindings(
        recording_identity=review_binding.recording.recording_identity,
        source_content_identity=review_binding.recording.source.source_content_identity,
        target_grid_identity=target_grid.target_grid_identity,
        block_ownership_schema_name=block_plan.schema_name,
        block_ownership_schema_version=block_plan.schema_version,
        block_ownership_policy_name=block_plan.block_policy_name,
        block_ownership_policy_version=block_plan.block_policy_version,
        correction_segment_plan_identity=segment_plan.plan_identity,
        dynamic_f0_authority_identity=dynamic_f0_authority.authority_identity,
        accepted_guided_plan_identity=accepted.accepted_guided_plan_identity,
        startup_mapping_contract_version=startup_mapping_contract.contract_version,
        correction_payload_identity=accepted.correction_payload_identity,
        fixed_correction_settings_identity=accepted.fixed_correction_settings_identity,
        segment_correction_settings_identity=segment_correction_settings_identity,
        segment_count=segment_plan.segment_count,
        target_sample_count=target_grid.target_sample_count,
    )


def _validate_corrected_segment_binding(
    corrected: GuidedContinuousRwdCorrectedSegment,
    *,
    raw_segment_index: int,
    raw_start_target_index: int,
    raw_stop_target_index: int,
    raw_included_roi_ids: tuple[str, ...],
    expected: _ExpectedCompletionBindings,
) -> None:
    if (
        corrected.segment_index != raw_segment_index
        or corrected.start_target_index != raw_start_target_index
        or corrected.stop_target_index != raw_stop_target_index
        or corrected.included_roi_ids != raw_included_roi_ids
        or corrected.recording_identity != expected.recording_identity
        or corrected.source_content_identity != expected.source_content_identity
        or corrected.target_grid_identity != expected.target_grid_identity
        or corrected.correction_segment_plan_identity != expected.correction_segment_plan_identity
        or corrected.dynamic_f0_authority_identity != expected.dynamic_f0_authority_identity
        or corrected.accepted_guided_plan_identity != expected.accepted_guided_plan_identity
        or corrected.correction_payload_identity != expected.correction_payload_identity
        or corrected.fixed_correction_settings_identity != expected.fixed_correction_settings_identity
        or corrected.segment_correction_settings_identity != expected.segment_correction_settings_identity
    ):
        _raise(
            "segment_identity_mismatch",
            "Corrected segment does not match the expected traversal authorities.",
            segment_index=raw_segment_index,
        )


def _run_correction_pass(
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    segment_plan: GuidedContinuousRwdCorrectionSegmentPlan,
    dynamic_f0_authority: GuidedContinuousRwdDynamicF0Authority,
    *,
    accepted_draft: GuidedNewAnalysisDraftPlan,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
    expected: _ExpectedCompletionBindings,
    cancellation_requested: Callable[[], bool] | None,
) -> Iterator[GuidedContinuousRwdCorrectedSegment]:
    """Yield provisional corrected segments; normal exhaustion proves completion."""
    _check_cancellation(cancellation_requested)
    projected_blocks = iter_project_guided_continuous_rwd_blocks(
        review_binding,
        target_grid,
        block_plan,
        cancellation_requested=cancellation_requested,
    )
    raw_segments = iter_assemble_guided_continuous_rwd_correction_segments(
        review_binding,
        target_grid,
        block_plan,
        segment_plan,
        projected_blocks,
        accepted_draft=accepted_draft,
        startup_mapping_contract=startup_mapping_contract,
        cancellation_requested=cancellation_requested,
    )
    expected_index = 0
    previous_stop = 0
    for raw in raw_segments:
        _check_cancellation(cancellation_requested)
        if raw.segment_index != expected_index or raw.start_target_index != previous_stop:
            _raise(
                "segment_order_mismatch",
                "Raw correction segment was produced out of canonical order.",
                expected_index=expected_index,
                actual_index=raw.segment_index,
            )
        _check_cancellation(cancellation_requested)
        corrected = correct_guided_continuous_rwd_segment(
            review_binding,
            target_grid,
            segment_plan,
            dynamic_f0_authority,
            raw,
            accepted_draft=accepted_draft,
            startup_mapping_contract=startup_mapping_contract,
            cancellation_requested=cancellation_requested,
        )
        _check_cancellation(cancellation_requested)
        _validate_corrected_segment_binding(
            corrected,
            raw_segment_index=raw.segment_index,
            raw_start_target_index=raw.start_target_index,
            raw_stop_target_index=raw.stop_target_index,
            raw_included_roi_ids=raw.included_roi_ids,
            expected=expected,
        )
        yield corrected
        previous_stop = corrected.stop_target_index
        expected_index += 1
    if expected_index != segment_plan.segment_count or previous_stop != target_grid.target_sample_count:
        _raise(
            "incomplete_traversal",
            "The corrected-segment traversal did not cover the full canonical target grid.",
            segments_produced=expected_index,
            last_stop_target_index=previous_stop,
        )


class GuidedContinuousRwdCorrectionPassTraversal:
    """One-shot iterator over provisional corrected segments for one recording.

    Iterating yields immutable ``GuidedContinuousRwdCorrectedSegment`` results
    in canonical segment order. ``completion`` raises
    ``GuidedContinuousRwdCorrectionPassError`` (category
    ``"completion_not_available"``) until the traversal reaches normal
    exhaustion; it never becomes available after cancellation or a
    mid-traversal failure. ``state`` reports ``"pending"``, ``"running"``,
    ``"completed"``, ``"failed"``, or ``"cancelled"``.
    """

    def __init__(
        self,
        segments: Iterator[GuidedContinuousRwdCorrectedSegment],
        *,
        expected: _ExpectedCompletionBindings,
        cancellation_requested: Callable[[], bool] | None,
    ) -> None:
        self._segments = segments
        self._expected = expected
        self._cancellation_requested = cancellation_requested
        self._state = "pending"
        self._completion: GuidedContinuousRwdCorrectionPassCompletion | None = None
        self._segment_count = 0
        self._last_stop_target_index = 0
        self._digest = hashlib.sha256()

    def __iter__(self) -> "GuidedContinuousRwdCorrectionPassTraversal":
        # Idempotent by design: repeated iter()/list() calls on the same
        # (possibly partially consumed) traversal must behave like the
        # standard "an iterator's __iter__ returns self" protocol rather than
        # refusing re-entry. __next__ is the sole terminal-state gatekeeper.
        if self._state == "pending":
            self._state = "running"
        return self

    def __next__(self) -> GuidedContinuousRwdCorrectedSegment:
        if self._state == "completed":
            raise StopIteration
        if self._state == "pending":
            # next(traversal) may begin the traversal exactly like
            # iter(traversal) does; only a genuine terminal state (failed,
            # cancelled) refuses re-entry below.
            self._state = "running"
        elif self._state != "running":
            _raise(
                "pass_already_terminal",
                "This traversal already reached a terminal failed/cancelled state.",
            )
        try:
            item = next(self._segments)
        except StopIteration:
            # Finalization exceptions are routed through the same _fail
            # classifier as segment-processing exceptions: neither cancellation
            # nor an unexpected error at this last, pre-publication checkpoint
            # may leave the traversal in "running", and none of it may
            # publish or later expose a completion authority.
            try:
                self._completion = self._finalize()
            except Exception as exc:
                self._fail(exc)
            self._state = "completed"
            raise
        except Exception as exc:
            self._fail(exc)
        else:
            self._segment_count += 1
            self._digest.update(
                str(item.segment_index).encode("ascii")
                + b":"
                + item.result_identity.encode("ascii")
                + b"\n"
            )
            self._last_stop_target_index = item.stop_target_index
            return item

    def _fail(self, exc: Exception) -> None:
        """Classify exc, permanently set a terminal state, and (always) raise.

        Shared by both the segment-processing exception path and the
        finalization exception path so a failure or cancellation detected at
        either point is classified identically and can never leave the
        traversal readable as "running".
        """
        if isinstance(exc, GuidedContinuousRwdCorrectionPassError):
            self._state = (
                "cancelled"
                if exc.category == "segment_correction_pass_interrupted"
                else "failed"
            )
            raise exc
        if _is_lower_layer_cancellation(exc):
            self._state = "cancelled"
            raise GuidedContinuousRwdCorrectionPassError(
                "segment_correction_pass_interrupted",
                "The corrected-segment traversal was cancelled.",
            ) from exc
        self._state = "failed"
        raise GuidedContinuousRwdCorrectionPassError(
            "segment_correction_pass_failed",
            "The corrected-segment traversal failed.",
            reason=str(exc),
        ) from exc

    def _finalize(self) -> GuidedContinuousRwdCorrectionPassCompletion:
        _check_cancellation(self._cancellation_requested)
        expected = self._expected
        if (
            self._segment_count != expected.segment_count
            or self._last_stop_target_index != expected.target_sample_count
        ):
            _raise(
                "incomplete_traversal",
                "The corrected-segment traversal did not cover the full canonical target grid.",
            )
        draft = GuidedContinuousRwdCorrectionPassCompletion(
            schema_name=SCHEMA_NAME,
            schema_version=SCHEMA_VERSION,
            policy_name=POLICY_NAME,
            policy_version=POLICY_VERSION,
            recording_identity=expected.recording_identity,
            source_content_identity=expected.source_content_identity,
            target_grid_identity=expected.target_grid_identity,
            block_ownership_schema_name=expected.block_ownership_schema_name,
            block_ownership_schema_version=expected.block_ownership_schema_version,
            block_ownership_policy_name=expected.block_ownership_policy_name,
            block_ownership_policy_version=expected.block_ownership_policy_version,
            correction_segment_plan_identity=expected.correction_segment_plan_identity,
            dynamic_f0_authority_identity=expected.dynamic_f0_authority_identity,
            accepted_guided_plan_identity=expected.accepted_guided_plan_identity,
            startup_mapping_contract_version=expected.startup_mapping_contract_version,
            correction_payload_identity=expected.correction_payload_identity,
            fixed_correction_settings_identity=expected.fixed_correction_settings_identity,
            segment_correction_settings_identity=expected.segment_correction_settings_identity,
            corrected_segment_count=self._segment_count,
            target_sample_count=expected.target_sample_count,
            ordered_segment_identity_digest=self._digest.hexdigest(),
            completion_state=COMPLETION_STATE,
            completion_identity="",
        )
        completion = GuidedContinuousRwdCorrectionPassCompletion(
            **{**draft.__dict__, "completion_identity": _compute_completion_identity(draft)}
        )
        _validate_completion(completion, expected=expected)
        return completion

    @property
    def state(self) -> str:
        return self._state

    @property
    def completion(self) -> GuidedContinuousRwdCorrectionPassCompletion:
        if self._state != "completed":
            _raise(
                "completion_not_available",
                "Finalized completion authority is only available after this "
                "traversal has fully and successfully exhausted.",
            )
        return self._completion


def iterate_guided_continuous_rwd_corrected_segments(
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    segment_plan: GuidedContinuousRwdCorrectionSegmentPlan,
    dynamic_f0_authority: GuidedContinuousRwdDynamicF0Authority,
    *,
    accepted_draft: GuidedNewAnalysisDraftPlan,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
    cancellation_requested: Callable[[], bool] | None = None,
) -> GuidedContinuousRwdCorrectionPassTraversal:
    """Validate accepted authorities, then return a fresh second-pass traversal.

    All supplied authorities are validated immediately, before any C3b
    traversal is opened. The returned traversal opens exactly one fresh C3b
    projection pass (via the accepted C3b public reader), assembles raw
    correction segments through the accepted C4a public path, and corrects
    each one through the accepted C4b public entry point -- never a second
    implementation of any of that mathematics.

    Cancellation is checked once the returned traversal is actually iterated
    (its first checkpoint is "before opening the source traversal"), not at
    this eager validation step, since nothing is opened yet here.
    """
    if not isinstance(review_binding, GuidedContinuousRwdReviewBinding):
        _raise("accepted_correction_binding_mismatch", "B3 Review binding has the wrong type.")
    expected = _resolve_expected_bindings(
        review_binding,
        target_grid,
        block_plan,
        segment_plan,
        dynamic_f0_authority,
        accepted_draft,
        startup_mapping_contract,
    )
    segments = _run_correction_pass(
        review_binding,
        target_grid,
        block_plan,
        segment_plan,
        dynamic_f0_authority,
        accepted_draft=accepted_draft,
        startup_mapping_contract=startup_mapping_contract,
        expected=expected,
        cancellation_requested=cancellation_requested,
    )
    return GuidedContinuousRwdCorrectionPassTraversal(
        segments,
        expected=expected,
        cancellation_requested=cancellation_requested,
    )
