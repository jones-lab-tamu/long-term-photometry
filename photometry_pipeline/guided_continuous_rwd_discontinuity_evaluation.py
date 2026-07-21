"""Bounded one-pass continuity evaluation for an accepted continuous RWD source.

Example row numbers are one-based data-row numbers after the frozen header;
they are not physical CSV line numbers.  This module grants no execution
authority and does not persist its result.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import stat
from typing import Callable

from photometry_pipeline.guided_continuous_rwd_discontinuity_policy import (
    MATERIAL_LONG_INTERVAL,
    NORMAL_INTERVAL,
    SHORT_INTERVAL_ANOMALY,
    classify_continuous_rwd_interval,
    resolve_continuous_rwd_discontinuity_tolerance,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    ContinuousRwdRecordingAuthorityError,
    GuidedContinuousRwdRecordingDescription,
    _validate_description,
)


CONTINUITY_PASSED = "continuity_passed"
SHORT_INTERVAL_ANOMALY_DETECTED = "short_interval_anomaly_detected"
MATERIAL_LONG_INTERVAL_DETECTED = "material_long_interval_detected"
SHORT_AND_LONG_DISCONTINUITIES_DETECTED = (
    "short_and_long_discontinuities_detected"
)
EVALUATION_INTERRUPTED = "evaluation_interrupted"
SOURCE_CHANGED_OR_MISMATCHED = "source_changed_or_mismatched"

_EXAMPLE_LIMIT = 10
_CANCELLATION_ROW_INTERVAL = 10_000
_REL_TOL = 1e-12
_ABS_TOL = 1e-12


class ContinuousRwdDiscontinuityEvaluationError(ValueError):
    """The supplied accepted-recording authority is malformed."""


@dataclass(frozen=True)
class ContinuousRwdDiscontinuityExample:
    """One classified interval located by one-based post-header data rows."""

    category: str
    data_row_number: int
    previous_data_row_number: int
    previous_elapsed_seconds: float
    current_elapsed_seconds: float
    dt_seconds: float
    nominal_cadence_seconds: float
    tolerance_seconds: float
    residual_seconds: float


@dataclass(frozen=True)
class ContinuousRwdDiscontinuityEvaluation:
    outcome: str
    recording_identity: str
    source_content_identity: str
    parser_interpretation_identity: str
    cadence_evidence_identity: str
    policy_name: str
    policy_version: str
    nominal_cadence_seconds: float
    tolerance_seconds: float
    valid_row_count_evaluated: int
    positive_interval_count_evaluated: int
    normal_interval_count: int
    short_interval_anomaly_count: int
    material_long_interval_count: int
    maximum_short_residual_seconds: float | None
    maximum_long_residual_seconds: float | None
    short_examples: tuple[ContinuousRwdDiscontinuityExample, ...]
    long_examples: tuple[ContinuousRwdDiscontinuityExample, ...]
    observed_source_sha256: str | None
    observed_source_size_bytes: int | None
    failure_reason: str | None


@dataclass(frozen=True)
class _SourceStat:
    size: int
    mtime_ns: int
    regular_file: bool


def _stat_source(path: Path) -> _SourceStat:
    facts = path.stat()
    return _SourceStat(facts.st_size, facts.st_mtime_ns, stat.S_ISREG(facts.st_mode))


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _normalized_header(row: list[str]) -> tuple[str, ...]:
    # This is CR1-A's committed column-normalization rule.
    return tuple(str(value).lstrip("\ufeff").strip() for value in row)


def _base_result(recording, tolerance, *, outcome, reason, observed_sha=None,
                 observed_size=None, counts=(0, 0, 0, 0, 0), maxima=(None, None),
                 examples=((), ())):
    valid, intervals, normal, short, long = counts
    return ContinuousRwdDiscontinuityEvaluation(
        outcome=outcome,
        recording_identity=recording.recording_identity,
        source_content_identity=recording.source.source_content_identity,
        parser_interpretation_identity=recording.source.parser_interpretation_identity,
        cadence_evidence_identity=recording.cadence.cadence_evidence_identity,
        policy_name=tolerance.policy_name,
        policy_version=tolerance.policy_version,
        nominal_cadence_seconds=tolerance.nominal_cadence_seconds,
        tolerance_seconds=tolerance.final_tolerance_seconds,
        valid_row_count_evaluated=valid,
        positive_interval_count_evaluated=intervals,
        normal_interval_count=normal,
        short_interval_anomaly_count=short,
        material_long_interval_count=long,
        maximum_short_residual_seconds=maxima[0],
        maximum_long_residual_seconds=maxima[1],
        short_examples=examples[0],
        long_examples=examples[1],
        observed_source_sha256=observed_sha,
        observed_source_size_bytes=observed_size,
        failure_reason=reason,
    )


def _interrupted(recording, tolerance, observed_size=None):
    return _base_result(
        recording, tolerance, outcome=EVALUATION_INTERRUPTED,
        reason="evaluation_cancelled", observed_size=observed_size,
    )


def _mismatch(recording, tolerance, reason, *, observed_sha=None,
              observed_size=None):
    return _base_result(
        recording, tolerance, outcome=SOURCE_CHANGED_OR_MISMATCHED,
        reason=reason, observed_sha=observed_sha, observed_size=observed_size,
    )


def _retain_example(examples, example, *, longest):
    examples.append(example)
    examples.sort(
        key=(lambda item: (-item.dt_seconds, item.data_row_number))
        if longest else
        (lambda item: (item.dt_seconds, item.data_row_number))
    )
    del examples[_EXAMPLE_LIMIT:]


def _completed_outcome(short_count: int, long_count: int) -> str:
    if short_count and long_count:
        return SHORT_AND_LONG_DISCONTINUITIES_DETECTED
    if short_count:
        return SHORT_INTERVAL_ANOMALY_DETECTED
    if long_count:
        return MATERIAL_LONG_INTERVAL_DETECTED
    return CONTINUITY_PASSED


def evaluate_continuous_rwd_timestamp_continuity(
    recording: GuidedContinuousRwdRecordingDescription,
    *,
    source_path: str | Path,
    cancellation_requested: Callable[[], bool] | None = None,
) -> ContinuousRwdDiscontinuityEvaluation:
    """Classify every positive interval while verifying the accepted source."""
    if not isinstance(recording, GuidedContinuousRwdRecordingDescription):
        raise ContinuousRwdDiscontinuityEvaluationError(
            "recording must be a GuidedContinuousRwdRecordingDescription."
        )
    try:
        _validate_description(recording)
        tolerance = resolve_continuous_rwd_discontinuity_tolerance(
            recording.cadence
        )
    except (ContinuousRwdRecordingAuthorityError, ValueError) as exc:
        raise ContinuousRwdDiscontinuityEvaluationError(
            "recording authority is invalid for continuity evaluation."
        ) from exc
    if cancellation_requested is not None and not callable(cancellation_requested):
        raise ContinuousRwdDiscontinuityEvaluationError(
            "cancellation_requested must be callable or None."
        )
    if cancellation_requested and cancellation_requested():
        return _interrupted(recording, tolerance)

    path = Path(source_path)
    try:
        before = _stat_source(path)
    except (OSError, ValueError, TypeError):
        return _mismatch(recording, tolerance, "path_not_regular_file")
    if not before.regular_file:
        return _mismatch(
            recording, tolerance, "path_not_regular_file",
            observed_size=before.size,
        )
    if before.size != recording.source.file_size_bytes:
        return _mismatch(
            recording, tolerance, "starting_size_mismatch",
            observed_size=before.size,
        )

    expected_columns = recording.source.raw_columns
    try:
        time_index = expected_columns.index(recording.source.selected_time_column)
    except ValueError as exc:
        raise ContinuousRwdDiscontinuityEvaluationError(
            "recording authority does not contain its selected time column."
        ) from exc

    digest = hashlib.sha256()
    valid_count = interval_count = normal_count = short_count = long_count = 0
    first = last = previous = previous_elapsed = None
    short_examples: list[ContinuousRwdDiscontinuityExample] = []
    long_examples: list[ContinuousRwdDiscontinuityExample] = []
    maximum_short = maximum_long = None

    def decoded_lines(raw):
        for raw_line in raw:
            digest.update(raw_line)
            yield raw_line.decode("utf-8")

    try:
        with path.open("rb") as raw:
            reader = csv.reader(decoded_lines(raw), strict=True)
            header = None
            for _ in range(recording.source.header_row_index + 1):
                header = next(reader)
            if header is None or _normalized_header(header) != expected_columns:
                return _mismatch(
                    recording, tolerance, "header_mismatch",
                    observed_size=before.size,
                )

            for data_row_number, row in enumerate(reader, start=1):
                if (
                    data_row_number % _CANCELLATION_ROW_INTERVAL == 0
                    and cancellation_requested
                    and cancellation_requested()
                ):
                    return _interrupted(recording, tolerance, before.size)
                if len(row) != len(expected_columns):
                    return _mismatch(
                        recording, tolerance, "malformed_data_row",
                        observed_size=before.size,
                    )
                try:
                    current = float(row[time_index].strip())
                except ValueError:
                    return _mismatch(
                        recording, tolerance, "nonnumeric_timestamp",
                        observed_size=before.size,
                    )
                if not math.isfinite(current):
                    return _mismatch(
                        recording, tolerance, "nonfinite_timestamp",
                        observed_size=before.size,
                    )

                valid_count += 1
                elapsed = (
                    current - recording.time.raw_first_timestamp
                ) * recording.time.raw_timestamp_scale_to_seconds
                if first is None:
                    first = current
                    if not _close(current, recording.time.raw_first_timestamp):
                        return _mismatch(
                            recording, tolerance, "first_endpoint_mismatch",
                            observed_size=before.size,
                        )
                    if not _close(elapsed, 0.0):
                        return _mismatch(
                            recording, tolerance, "first_endpoint_mismatch",
                            observed_size=before.size,
                        )
                else:
                    raw_dt = current - previous
                    if raw_dt == 0.0:
                        return _mismatch(
                            recording, tolerance, "duplicate_timestamp",
                            observed_size=before.size,
                        )
                    if raw_dt < 0.0:
                        return _mismatch(
                            recording, tolerance, "backward_timestamp",
                            observed_size=before.size,
                        )
                    classification = classify_continuous_rwd_interval(
                        raw_dt * recording.time.raw_timestamp_scale_to_seconds,
                        tolerance=tolerance,
                    )
                    interval_count += 1
                    if classification.category == NORMAL_INTERVAL:
                        normal_count += 1
                    else:
                        example = ContinuousRwdDiscontinuityExample(
                            classification.category,
                            data_row_number,
                            data_row_number - 1,
                            previous_elapsed,
                            elapsed,
                            classification.dt_seconds,
                            classification.nominal_cadence_seconds,
                            classification.tolerance_seconds,
                            classification.residual_seconds,
                        )
                        if classification.category == SHORT_INTERVAL_ANOMALY:
                            short_count += 1
                            maximum_short = max(
                                maximum_short or 0.0,
                                classification.residual_seconds,
                            )
                            _retain_example(short_examples, example, longest=False)
                        elif classification.category == MATERIAL_LONG_INTERVAL:
                            long_count += 1
                            maximum_long = max(
                                maximum_long or 0.0,
                                classification.residual_seconds,
                            )
                            _retain_example(long_examples, example, longest=True)
                        else:
                            raise ContinuousRwdDiscontinuityEvaluationError(
                                "policy returned an unsupported interval category."
                            )
                previous = last = current
                previous_elapsed = elapsed
    except StopIteration:
        return _mismatch(
            recording, tolerance, "header_mismatch", observed_size=before.size,
        )
    except UnicodeError:
        return _mismatch(
            recording, tolerance, "source_unreadable", observed_size=before.size,
        )
    except (csv.Error, OSError):
        return _mismatch(
            recording, tolerance, "source_unreadable", observed_size=before.size,
        )

    observed_sha = digest.hexdigest()
    try:
        after = _stat_source(path)
    except OSError:
        return _mismatch(
            recording, tolerance, "source_changed_during_evaluation",
            observed_sha=observed_sha,
        )
    if (
        not after.regular_file
        or after.size != before.size
        or after.mtime_ns != before.mtime_ns
    ):
        return _mismatch(
            recording, tolerance, "source_changed_during_evaluation",
            observed_sha=observed_sha, observed_size=after.size,
        )
    if cancellation_requested and cancellation_requested():
        return _interrupted(recording, tolerance, after.size)
    if after.size != recording.source.file_size_bytes:
        return _mismatch(
            recording, tolerance, "source_size_mismatch",
            observed_sha=observed_sha, observed_size=after.size,
        )
    if observed_sha != recording.source.sha256:
        return _mismatch(
            recording, tolerance, "source_sha256_mismatch",
            observed_sha=observed_sha, observed_size=after.size,
        )
    if valid_count != recording.source.valid_timestamp_count:
        return _mismatch(
            recording, tolerance, "row_count_mismatch",
            observed_sha=observed_sha, observed_size=after.size,
        )
    if interval_count != recording.cadence.positive_interval_count:
        return _mismatch(
            recording, tolerance, "interval_count_mismatch",
            observed_sha=observed_sha, observed_size=after.size,
        )
    if first is None or not _close(first, recording.time.raw_first_timestamp):
        return _mismatch(
            recording, tolerance, "first_endpoint_mismatch",
            observed_sha=observed_sha, observed_size=after.size,
        )
    if last is None or not _close(last, recording.time.raw_last_timestamp):
        return _mismatch(
            recording, tolerance, "last_endpoint_mismatch",
            observed_sha=observed_sha, observed_size=after.size,
        )
    duration = (last - first) * recording.time.raw_timestamp_scale_to_seconds
    if not _close(duration, recording.time.measured_duration_seconds):
        return _mismatch(
            recording, tolerance, "duration_mismatch",
            observed_sha=observed_sha, observed_size=after.size,
        )
    if interval_count != valid_count - 1:
        return _mismatch(
            recording, tolerance, "interval_count_mismatch",
            observed_sha=observed_sha, observed_size=after.size,
        )

    counts = (valid_count, interval_count, normal_count, short_count, long_count)
    return _base_result(
        recording,
        tolerance,
        outcome=_completed_outcome(short_count, long_count),
        reason=None,
        observed_sha=observed_sha,
        observed_size=after.size,
        counts=counts,
        maxima=(maximum_short, maximum_long),
        examples=(tuple(short_examples), tuple(long_examples)),
    )
