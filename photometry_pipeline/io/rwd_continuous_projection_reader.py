"""Ordered bounded projection of one accepted continuous RWD source.

Projected blocks yielded by this module are provisional.  Complete-source
verification is established only when the returned iterator exhausts normally.
Abandonment, interruption, or any later exception leaves all earlier blocks as
an incomplete projection run.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from photometry_pipeline.guided_continuous_rwd_block_plan import (
    ContinuousRwdBlockPlanError,
    GuidedContinuousRwdBlockDescription,
    GuidedContinuousRwdBlockPlan,
    _validate_block_plan,
)
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    CONTINUITY_PASSED,
    ContinuousRwdDiscontinuityEvaluation,
    _close,
    _normalized_header,
    _stat_source,
)
from photometry_pipeline.guided_continuous_rwd_discontinuity_policy import (
    POLICY_NAME as DISCONTINUITY_POLICY_NAME,
    POLICY_VERSION as DISCONTINUITY_POLICY_VERSION,
)
from photometry_pipeline.guided_continuous_rwd_projection import (
    ContinuousRwdProjectionError,
    GuidedContinuousRwdProjectedBlock,
    project_guided_continuous_rwd_block,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    ContinuousRwdRecordingAuthorityError,
    GuidedContinuousRwdRecordingDescription,
    _validate_description,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBinding,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    ContinuousRwdTargetGridError,
    GuidedContinuousRwdTargetGridDescription,
    _decimal_fraction,
    _validate_target_grid_description,
    compute_continuous_rwd_discontinuity_evaluation_identity,
)


_CANCELLATION_ROW_INTERVAL = 10_000


class ContinuousRwdProjectionReaderError(ValueError):
    """The accepted source cannot complete one ordered projection pass."""

    def __init__(self, category: str, message: str):
        super().__init__(f"{category}: {message}")
        self.category = category


@dataclass(frozen=True)
class _ParsedSourceRow:
    row_index: int
    elapsed_seconds: float
    control_values: tuple[float, ...]
    signal_values: tuple[float, ...]


@dataclass(frozen=True)
class _ValidatedAuthorities:
    binding: GuidedContinuousRwdReviewBinding
    recording: GuidedContinuousRwdRecordingDescription
    evaluation: ContinuousRwdDiscontinuityEvaluation
    target_grid: GuidedContinuousRwdTargetGridDescription
    block_plan: GuidedContinuousRwdBlockPlan
    source_path: Path
    included_roi_ids: tuple[str, ...]
    time_index: int
    control_indices: tuple[int, ...]
    signal_indices: tuple[int, ...]


def _fail(category: str, message: str) -> None:
    raise ContinuousRwdProjectionReaderError(category, message)


def _validate_identity_text(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(
            "invalid_authority_binding",
            f"{name} must be a lowercase 64-character hexadecimal identity.",
        )


def _validate_b2_binding(
    recording: GuidedContinuousRwdRecordingDescription,
    evaluation: object,
) -> ContinuousRwdDiscontinuityEvaluation:
    if not isinstance(evaluation, ContinuousRwdDiscontinuityEvaluation):
        _fail("invalid_authority_binding", "Retained B2 authority has the wrong type.")
    if evaluation.outcome != CONTINUITY_PASSED or evaluation.failure_reason is not None:
        _fail("invalid_authority_binding", "Retained B2 authority did not pass cleanly.")
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
            _fail(
                "invalid_authority_binding",
                f"Retained B2 {name} identity does not match B1.",
            )
    if (
        evaluation.observed_source_sha256 != recording.source.sha256
        or evaluation.observed_source_size_bytes != recording.source.file_size_bytes
    ):
        _fail("invalid_authority_binding", "Retained B2 source facts do not match B1.")
    if (
        evaluation.policy_name != DISCONTINUITY_POLICY_NAME
        or evaluation.policy_version != DISCONTINUITY_POLICY_VERSION
    ):
        _fail("invalid_authority_binding", "Retained B2 policy is unsupported.")
    if evaluation.nominal_cadence_seconds != recording.cadence.nominal_cadence_seconds:
        _fail("invalid_authority_binding", "Retained B2 cadence does not match B1.")
    rows = evaluation.valid_row_count_evaluated
    intervals = evaluation.positive_interval_count_evaluated
    if rows != recording.source.valid_timestamp_count:
        _fail("invalid_authority_binding", "Retained B2 row count does not match B1.")
    if intervals != recording.cadence.positive_interval_count or intervals != rows - 1:
        _fail("invalid_authority_binding", "Retained B2 interval count does not match B1.")
    if evaluation.normal_interval_count != intervals:
        _fail("invalid_authority_binding", "Retained B2 normal count is inconsistent.")
    if (
        evaluation.short_interval_anomaly_count != 0
        or evaluation.material_long_interval_count != 0
        or evaluation.short_examples
        or evaluation.long_examples
        or evaluation.maximum_short_residual_seconds is not None
        or evaluation.maximum_long_residual_seconds is not None
    ):
        _fail("invalid_authority_binding", "Retained passing B2 contains anomaly evidence.")
    tolerance = evaluation.tolerance_seconds
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0.0
        or evaluation.nominal_cadence_seconds - float(tolerance) <= 0.0
    ):
        _fail("invalid_authority_binding", "Retained B2 tolerance cannot bound buffering.")
    return evaluation


def _validate_authorities(
    review_binding: object,
    target_grid: object,
    block_plan: object,
) -> _ValidatedAuthorities:
    if not isinstance(review_binding, GuidedContinuousRwdReviewBinding):
        _fail("invalid_authority_binding", "review_binding has the wrong type.")
    _validate_identity_text(review_binding.draft_plan_identity, "Draft-plan identity")
    recording = review_binding.recording
    try:
        _validate_description(recording)
    except (ContinuousRwdRecordingAuthorityError, TypeError, ValueError) as exc:
        raise ContinuousRwdProjectionReaderError(
            "invalid_authority_binding", "Retained B1 authority is invalid."
        ) from exc
    evaluation = _validate_b2_binding(recording, review_binding.continuity_evaluation)

    source_text = review_binding.current_source_path
    if not isinstance(source_text, str) or not source_text.strip():
        _fail("invalid_authority_binding", "B3 current source path is invalid.")
    source_path = Path(source_text)
    accepted_path = Path(recording.source.fluorescence_path_canonical)
    if (
        not accepted_path.name
        or accepted_path.parent != Path(recording.source.selected_folder_canonical)
    ):
        _fail("invalid_authority_binding", "B1 filename provenance is invalid.")
    if source_path.name != accepted_path.name:
        _fail("invalid_authority_binding", "B3 current source filename does not match B1.")

    if not isinstance(target_grid, GuidedContinuousRwdTargetGridDescription):
        _fail("invalid_authority_binding", "target_grid has the wrong type.")
    try:
        _validate_target_grid_description(target_grid)
    except (ContinuousRwdTargetGridError, TypeError, ValueError) as exc:
        raise ContinuousRwdProjectionReaderError(
            "invalid_authority_binding", "Retained C1 authority is invalid."
        ) from exc
    if target_grid.recording_identity != recording.recording_identity:
        _fail("invalid_authority_binding", "C1 recording identity does not match B1.")
    expected_continuity_identity = (
        compute_continuous_rwd_discontinuity_evaluation_identity(evaluation)
    )
    if target_grid.continuity_evaluation_identity != expected_continuity_identity:
        _fail("invalid_authority_binding", "C1 continuity identity does not match B2.")
    try:
        expected_cadence = _decimal_fraction(
            recording.cadence.nominal_cadence_seconds,
            "nominal cadence",
        )
        expected_support = _decimal_fraction(
            recording.time.measured_support_end_seconds,
            "measured support end",
        )
    except ContinuousRwdTargetGridError as exc:
        raise ContinuousRwdProjectionReaderError(
            "invalid_authority_binding", "B1 cannot establish C1 scalar authority."
        ) from exc
    if (
        target_grid.cadence_fraction != expected_cadence
        or target_grid.source_support_end_fraction != expected_support
    ):
        _fail("invalid_authority_binding", "C1 cadence or support does not match B1.")

    if not isinstance(block_plan, GuidedContinuousRwdBlockPlan):
        _fail("invalid_authority_binding", "block_plan has the wrong type.")
    try:
        _validate_block_plan(block_plan)
    except (ContinuousRwdBlockPlanError, TypeError, ValueError) as exc:
        raise ContinuousRwdProjectionReaderError(
            "invalid_authority_binding", "Retained C2 authority is invalid."
        ) from exc
    if block_plan.target_grid_identity != target_grid.target_grid_identity:
        _fail("invalid_authority_binding", "C2 target-grid identity does not match C1.")
    if block_plan.target_sample_count != target_grid.target_sample_count:
        _fail("invalid_authority_binding", "C2 target count does not match C1.")

    columns = recording.source.raw_columns
    try:
        time_index = columns.index(recording.source.selected_time_column)
    except ValueError as exc:
        raise ContinuousRwdProjectionReaderError(
            "invalid_authority_binding", "B1 time column is absent from raw columns."
        ) from exc
    channel_by_roi = {
        channel.roi_id: channel for channel in recording.roi.available_roi_channels
    }
    included_roi_ids = recording.roi.included_roi_ids
    try:
        selected_channels = tuple(channel_by_roi[roi_id] for roi_id in included_roi_ids)
        control_indices = tuple(columns.index(item.reference_column) for item in selected_channels)
        signal_indices = tuple(columns.index(item.signal_column) for item in selected_channels)
    except (KeyError, ValueError) as exc:
        raise ContinuousRwdProjectionReaderError(
            "invalid_authority_binding", "B1 included ROI mapping is incomplete."
        ) from exc
    return _ValidatedAuthorities(
        binding=review_binding,
        recording=recording,
        evaluation=evaluation,
        target_grid=target_grid,
        block_plan=block_plan,
        source_path=source_path,
        included_roi_ids=included_roi_ids,
        time_index=time_index,
        control_indices=control_indices,
        signal_indices=signal_indices,
    )


def _check_cancellation(callback: Callable[[], bool] | None) -> None:
    if callback is None:
        return
    try:
        cancelled = callback()
    except Exception as exc:
        raise ContinuousRwdProjectionReaderError(
            "projection_interrupted", "Cancellation callback failed."
        ) from exc
    if cancelled:
        _fail("projection_interrupted", "Continuous RWD projection was cancelled.")


def _block_target_bounds(
    grid: GuidedContinuousRwdTargetGridDescription,
    block: GuidedContinuousRwdBlockDescription,
) -> tuple[float, float]:
    cadence = np.float64(grid.cadence_fraction)
    first = float(np.float64(block.start_target_index) * cadence)
    last = float(np.float64(block.stop_target_index - 1) * cadence)
    if not math.isfinite(first) or not math.isfinite(last) or first > last:
        _fail("invalid_authority_binding", "C1 target bounds are not finite and ordered.")
    return first, last


def _buffer_row_limit(
    authorities: _ValidatedAuthorities,
    block: GuidedContinuousRwdBlockDescription,
) -> int:
    nominal = float(authorities.evaluation.nominal_cadence_seconds)
    tolerance = float(authorities.evaluation.tolerance_seconds)
    lower = nominal - tolerance
    upper = nominal + tolerance
    target_span = max(0, block.owned_sample_count - 1) * float(
        authorities.target_grid.cadence_fraction
    )
    return max(3, math.ceil((target_span + 2.0 * upper) / lower) + 1)


def _trim_to_left_bracket(buffer: list[_ParsedSourceRow], first_target: float) -> None:
    discard = 0
    while discard + 1 < len(buffer) and buffer[discard + 1].elapsed_seconds <= first_target:
        discard += 1
    if discard:
        del buffer[:discard]


def _source_arrays(
    rows: list[_ParsedSourceRow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elapsed = np.fromiter(
        (row.elapsed_seconds for row in rows),
        dtype=np.float64,
        count=len(rows),
    )
    control = np.asarray([row.control_values for row in rows], dtype=np.float64)
    signal = np.asarray([row.signal_values for row in rows], dtype=np.float64)
    return elapsed, control, signal


def _iterate_projection(
    authorities: _ValidatedAuthorities,
    cancellation_requested: Callable[[], bool] | None,
) -> Iterator[GuidedContinuousRwdProjectedBlock]:
    _check_cancellation(cancellation_requested)
    path = authorities.source_path
    recording = authorities.recording
    expected_columns = recording.source.raw_columns
    try:
        before = _stat_source(path)
    except (OSError, ValueError, TypeError) as exc:
        raise ContinuousRwdProjectionReaderError(
            "source_unavailable", "Current source cannot be inspected."
        ) from exc
    if not before.regular_file:
        _fail("source_unavailable", "Current source is not a regular file.")
    if before.size != recording.source.file_size_bytes:
        _fail("source_content_mismatch", "Starting source size does not match B1/B2.")

    def row_failure(category: str, message: str, cause: Exception | None = None):
        try:
            current = _stat_source(path)
            changed = (
                not current.regular_file
                or current.size != before.size
                or current.mtime_ns != before.mtime_ns
            )
        except (OSError, ValueError, TypeError):
            changed = True
        error = ContinuousRwdProjectionReaderError(
            "source_instability" if changed else category,
            "Source changed during projection." if changed else message,
        )
        if cause is not None:
            raise error from cause
        raise error

    digest = hashlib.sha256()
    bytes_read = 0

    def decoded_lines(raw):
        nonlocal bytes_read
        for raw_line in raw:
            digest.update(raw_line)
            bytes_read += len(raw_line)
            try:
                yield raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ContinuousRwdProjectionReaderError(
                    "malformed_data_row", "Source is not strict UTF-8."
                ) from exc

    try:
        raw_handle = path.open("rb")
    except (OSError, ValueError, TypeError) as exc:
        raise ContinuousRwdProjectionReaderError(
            "source_unavailable", "Current source cannot be opened."
        ) from exc

    valid_row_count = 0
    first_raw = last_raw = previous_elapsed = last_elapsed = None
    buffer: list[_ParsedSourceRow] = []
    with raw_handle:
        reader = csv.reader(decoded_lines(raw_handle), strict=True)
        header = None
        try:
            for _ in range(recording.source.header_row_index + 1):
                header = next(reader)
        except StopIteration as exc:
            raise ContinuousRwdProjectionReaderError(
                "parser_header_mismatch", "Accepted header row is absent."
            ) from exc
        except (csv.Error, UnicodeError, OSError) as exc:
            raise ContinuousRwdProjectionReaderError(
                "parser_header_mismatch", "Accepted header cannot be parsed."
            ) from exc
        except ContinuousRwdProjectionReaderError as exc:
            raise ContinuousRwdProjectionReaderError(
                "parser_header_mismatch", "Accepted header is not strict UTF-8."
            ) from exc
        if header is None or _normalized_header(header) != expected_columns:
            _fail("parser_header_mismatch", "Parsed header does not match B1 raw columns.")
        _check_cancellation(cancellation_requested)

        def read_next() -> _ParsedSourceRow | None:
            nonlocal valid_row_count, first_raw, last_raw, previous_elapsed, last_elapsed
            try:
                row = next(reader)
            except StopIteration:
                return None
            except ContinuousRwdProjectionReaderError:
                raise
            except (csv.Error, UnicodeError, OSError) as exc:
                row_failure("malformed_data_row", "Data row cannot be parsed.", exc)
            row_index = valid_row_count
            if len(row) != len(expected_columns):
                row_failure("malformed_data_row", "Data-row field count does not match B1.")
            try:
                raw_timestamp = float(row[authorities.time_index].strip())
            except ValueError as exc:
                row_failure("timestamp_violation", "Timestamp is not numeric.", exc)
            if not math.isfinite(raw_timestamp):
                row_failure("timestamp_violation", "Timestamp is not finite.")
            try:
                control = tuple(float(row[index].strip()) for index in authorities.control_indices)
                signal = tuple(float(row[index].strip()) for index in authorities.signal_indices)
            except ValueError as exc:
                row_failure(
                    "selected_channel_value_violation",
                    "Selected control or signal value is not numeric.",
                    exc,
                )
            if not all(math.isfinite(value) for value in control + signal):
                row_failure(
                    "selected_channel_value_violation",
                    "Selected control and signal values must be finite.",
                )
            elapsed = (
                raw_timestamp - recording.time.raw_first_timestamp
            ) * recording.time.raw_timestamp_scale_to_seconds
            if not math.isfinite(elapsed):
                _fail("timestamp_violation", "Normalized timestamp is not finite.")
            if first_raw is None:
                first_raw = raw_timestamp
                if not _close(raw_timestamp, recording.time.raw_first_timestamp) or elapsed != 0.0:
                    _fail("timestamp_violation", "First source timestamp does not establish zero.")
            elif elapsed <= previous_elapsed:
                _fail("timestamp_violation", "Source timestamps are not strictly increasing.")
            support_end = recording.time.measured_support_end_seconds
            if elapsed > support_end and not _close(elapsed, support_end):
                _fail("timestamp_violation", "Source extends beyond accepted B1 support.")
            valid_row_count += 1
            if (
                valid_row_count % _CANCELLATION_ROW_INTERVAL == 0
                and cancellation_requested is not None
            ):
                _check_cancellation(cancellation_requested)
            previous_elapsed = last_elapsed = elapsed
            last_raw = raw_timestamp
            return _ParsedSourceRow(row_index, elapsed, control, signal)

        for block in authorities.block_plan.iter_blocks():
            first_target, last_target = _block_target_bounds(authorities.target_grid, block)
            _trim_to_left_bracket(buffer, first_target)
            if buffer and buffer[0].elapsed_seconds > first_target:
                _fail("insufficient_left_support", "No source row brackets the first target.")
            while not buffer or buffer[-1].elapsed_seconds < last_target:
                parsed = read_next()
                if parsed is None:
                    _fail("insufficient_right_support", "Source ended before the final target bracket.")
                buffer.append(parsed)
                if len(buffer) > _buffer_row_limit(authorities, block):
                    _fail(
                        "bounded_buffer_limit_exceeded",
                        "Retained source support exceeds the B2-derived block ceiling.",
                    )
            _trim_to_left_bracket(buffer, first_target)
            if not buffer or buffer[0].elapsed_seconds > first_target:
                _fail("insufficient_left_support", "No source row brackets the first target.")
            if buffer[-1].elapsed_seconds < last_target:
                _fail("insufficient_right_support", "No source row brackets the final target.")
            _check_cancellation(cancellation_requested)
            elapsed, control, signal = _source_arrays(buffer)
            try:
                projected = project_guided_continuous_rwd_block(
                    authorities.target_grid,
                    authorities.block_plan,
                    block,
                    recording_identity=recording.recording_identity,
                    source_content_identity=recording.source.source_content_identity,
                    included_roi_ids=authorities.included_roi_ids,
                    source_row_start=buffer[0].row_index,
                    source_row_stop=buffer[-1].row_index + 1,
                    source_elapsed_seconds=elapsed,
                    source_control_values=control,
                    source_signal_values=signal,
                )
            except ContinuousRwdProjectionError as exc:
                raise ContinuousRwdProjectionReaderError(
                    "projection_failure", "C3a refused bounded source support."
                ) from exc
            yield projected
            _check_cancellation(cancellation_requested)

        while read_next() is not None:
            pass
        _check_cancellation(cancellation_requested)

    observed_sha256 = digest.hexdigest()
    try:
        after = _stat_source(path)
    except (OSError, ValueError, TypeError) as exc:
        raise ContinuousRwdProjectionReaderError(
            "source_instability", "Current source cannot be inspected after projection."
        ) from exc
    if (
        not after.regular_file
        or after.size != before.size
        or after.mtime_ns != before.mtime_ns
    ):
        _fail("source_instability", "Source metadata changed during projection.")
    if bytes_read != before.size or after.size != recording.source.file_size_bytes:
        _fail("source_content_mismatch", "Final source byte size does not match B1/B2.")
    if (
        observed_sha256 != recording.source.sha256
        or observed_sha256 != authorities.evaluation.observed_source_sha256
    ):
        _fail("source_content_mismatch", "Final source SHA-256 does not match B1/B2.")
    if valid_row_count != recording.source.valid_timestamp_count:
        _fail("source_content_mismatch", "Final valid-row count does not match B1.")
    if first_raw is None or not _close(first_raw, recording.time.raw_first_timestamp):
        _fail("timestamp_violation", "Final first endpoint does not match B1.")
    if last_raw is None or not _close(last_raw, recording.time.raw_last_timestamp):
        _fail("timestamp_violation", "Final raw endpoint does not match B1.")
    if last_elapsed is None or not _close(
        last_elapsed,
        recording.time.measured_support_end_seconds,
    ):
        _fail("timestamp_violation", "Final normalized endpoint does not match B1.")


def iter_project_guided_continuous_rwd_blocks(
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    block_plan: GuidedContinuousRwdBlockPlan,
    *,
    cancellation_requested: Callable[[], bool] | None = None,
) -> Iterator[GuidedContinuousRwdProjectedBlock]:
    """Return provisional blocks; only normal exhaustion verifies the full source."""
    if cancellation_requested is not None and not callable(cancellation_requested):
        _fail("invalid_authority_binding", "cancellation_requested must be callable or None.")
    authorities = _validate_authorities(review_binding, target_grid, block_plan)
    return _iterate_projection(authorities, cancellation_requested)
