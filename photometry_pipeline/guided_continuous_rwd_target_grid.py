"""Pure scalar authority for the CR1 continuous RWD target timebase."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import math
from typing import Any, Mapping

from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    CONTINUITY_PASSED,
    ContinuousRwdDiscontinuityEvaluation,
)
from photometry_pipeline.guided_continuous_rwd_discontinuity_policy import (
    POLICY_NAME as DISCONTINUITY_POLICY_NAME,
    POLICY_VERSION as DISCONTINUITY_POLICY_VERSION,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    ContinuousRwdRecordingAuthorityError,
    GuidedContinuousRwdRecordingDescription,
    _validate_description as _validate_recording_description,
)
from photometry_pipeline.guided_identity import encode_canonical_value


SCHEMA_NAME = "guided_continuous_rwd_target_grid"
SCHEMA_VERSION = "v1"
GRID_POLICY_NAME = "continuous-rwd-observed-nominal-cadence-grid"
GRID_POLICY_VERSION = "v1"
CONTINUITY_EVALUATION_IDENTITY_DOMAIN = "guided-continuous-rwd-continuity-evaluation:v1"
TARGET_GRID_IDENTITY_DOMAIN = "guided-continuous-rwd-target-grid:v1"
MAX_TARGET_SAMPLE_COUNT = 2**63 - 1


class ContinuousRwdTargetGridError(ValueError):
    """The accepted CR1 authorities cannot establish one target grid."""


@dataclass(frozen=True)
class GuidedContinuousRwdTargetGridDescription:
    schema_name: str
    schema_version: str
    grid_policy_name: str
    grid_policy_version: str
    recording_identity: str
    continuity_evaluation_identity: str
    cadence_seconds_numerator: int
    cadence_seconds_denominator: int
    source_support_end_seconds_numerator: int
    source_support_end_seconds_denominator: int
    target_sample_count: int
    target_grid_identity: str

    @property
    def origin_elapsed_seconds(self) -> float:
        return 0.0

    @property
    def cadence_fraction(self) -> Fraction:
        return Fraction(
            self.cadence_seconds_numerator,
            self.cadence_seconds_denominator,
        )

    @property
    def cadence_seconds(self) -> float:
        return float(self.cadence_fraction)

    @property
    def source_support_end_fraction(self) -> Fraction:
        return Fraction(
            self.source_support_end_seconds_numerator,
            self.source_support_end_seconds_denominator,
        )

    @property
    def source_support_end_seconds(self) -> float:
        return float(self.source_support_end_fraction)

    @property
    def last_target_index(self) -> int:
        return self.target_sample_count - 1

    @property
    def last_target_elapsed_fraction(self) -> Fraction:
        return self.last_target_index * self.cadence_fraction

    @property
    def last_target_elapsed_seconds(self) -> float:
        return float(self.last_target_elapsed_fraction)

    @property
    def next_target_elapsed_fraction(self) -> Fraction:
        return self.target_sample_count * self.cadence_fraction

    @property
    def next_target_elapsed_seconds(self) -> float:
        return float(self.next_target_elapsed_fraction)


def _fail(message: str) -> None:
    raise ContinuousRwdTargetGridError(message)


def _validate_identity_text(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be a lowercase 64-character hexadecimal identity.")


def _digest(domain: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + encode_canonical_value(dict(payload))
    ).hexdigest()


def _decimal_fraction(value: Any, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be numeric.")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        _fail(f"{name} must be finite and greater than zero.")
    try:
        result = Fraction(Decimal(str(value)))
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise ContinuousRwdTargetGridError(
            f"{name} cannot be represented as an exact decimal fraction."
        ) from exc
    if result <= 0:
        _fail(f"{name} must be greater than zero.")
    return result


def _resolve_target_sample_count(support: Fraction, cadence: Fraction) -> int:
    """Return exact inclusive-support count for positive rational scalars."""
    if not isinstance(support, Fraction) or support <= 0:
        _fail("support must be a positive Fraction.")
    if not isinstance(cadence, Fraction) or cadence <= 0:
        _fail("cadence must be a positive Fraction.")
    return support // cadence + 1


def compute_continuous_rwd_discontinuity_evaluation_identity(
    evaluation: ContinuousRwdDiscontinuityEvaluation,
) -> str:
    """Identify every field of one frozen B2b evaluation deterministically."""
    if not isinstance(evaluation, ContinuousRwdDiscontinuityEvaluation):
        _fail("continuity_evaluation must be a ContinuousRwdDiscontinuityEvaluation.")
    return _digest(CONTINUITY_EVALUATION_IDENTITY_DOMAIN, asdict(evaluation))


def _target_grid_identity_payload(
    description: GuidedContinuousRwdTargetGridDescription,
) -> dict[str, Any]:
    return {
        "schema_name": description.schema_name,
        "schema_version": description.schema_version,
        "grid_policy_name": description.grid_policy_name,
        "grid_policy_version": description.grid_policy_version,
        "recording_identity": description.recording_identity,
        "continuity_evaluation_identity": (
            description.continuity_evaluation_identity
        ),
        "cadence_seconds": {
            "numerator": description.cadence_seconds_numerator,
            "denominator": description.cadence_seconds_denominator,
        },
        "origin_elapsed_seconds": {"numerator": 0, "denominator": 1},
        "source_support_end_seconds": {
            "numerator": description.source_support_end_seconds_numerator,
            "denominator": description.source_support_end_seconds_denominator,
        },
        "target_sample_count": description.target_sample_count,
    }


def compute_guided_continuous_rwd_target_grid_identity(
    description: GuidedContinuousRwdTargetGridDescription,
) -> str:
    if not isinstance(description, GuidedContinuousRwdTargetGridDescription):
        _fail("description must be a GuidedContinuousRwdTargetGridDescription.")
    return _digest(TARGET_GRID_IDENTITY_DOMAIN, _target_grid_identity_payload(description))


def _validate_b2_compatibility(
    recording: GuidedContinuousRwdRecordingDescription,
    evaluation: ContinuousRwdDiscontinuityEvaluation,
) -> None:
    if not isinstance(evaluation, ContinuousRwdDiscontinuityEvaluation):
        _fail("continuity_evaluation must be a ContinuousRwdDiscontinuityEvaluation.")
    if evaluation.outcome != CONTINUITY_PASSED:
        _fail("Continuity evaluation must have passed.")
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
        evaluation.policy_name != DISCONTINUITY_POLICY_NAME
        or evaluation.policy_version != DISCONTINUITY_POLICY_VERSION
    ):
        _fail("Continuity evaluation uses an unsupported policy.")
    if evaluation.nominal_cadence_seconds != recording.cadence.nominal_cadence_seconds:
        _fail("Continuity nominal cadence does not match the recording.")
    if evaluation.valid_row_count_evaluated != recording.source.valid_timestamp_count:
        _fail("Continuity valid-row count does not match the recording.")
    if (
        evaluation.positive_interval_count_evaluated
        != recording.cadence.positive_interval_count
    ):
        _fail("Continuity interval count does not match the recording.")
    if evaluation.normal_interval_count != evaluation.positive_interval_count_evaluated:
        _fail("Passing continuity normal-interval count is inconsistent.")
    if evaluation.short_interval_anomaly_count != 0 or evaluation.short_examples:
        _fail("Passing continuity result contains short-interval evidence.")
    if evaluation.material_long_interval_count != 0 or evaluation.long_examples:
        _fail("Passing continuity result contains long-interval evidence.")


def _validate_target_grid_description(
    description: GuidedContinuousRwdTargetGridDescription,
) -> None:
    if not isinstance(description, GuidedContinuousRwdTargetGridDescription):
        _fail("description must be a GuidedContinuousRwdTargetGridDescription.")
    if (
        description.schema_name != SCHEMA_NAME
        or description.schema_version != SCHEMA_VERSION
        or description.grid_policy_name != GRID_POLICY_NAME
        or description.grid_policy_version != GRID_POLICY_VERSION
    ):
        _fail("Unsupported continuous target-grid metadata.")
    _validate_identity_text(description.recording_identity, "Recording identity")
    _validate_identity_text(
        description.continuity_evaluation_identity,
        "Continuity-evaluation identity",
    )
    _validate_identity_text(description.target_grid_identity, "Target-grid identity")
    for value, name in (
        (description.cadence_seconds_numerator, "cadence numerator"),
        (description.cadence_seconds_denominator, "cadence denominator"),
        (description.source_support_end_seconds_numerator, "support numerator"),
        (description.source_support_end_seconds_denominator, "support denominator"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _fail(f"{name.capitalize()} must be a positive integer.")
    cadence = description.cadence_fraction
    support = description.source_support_end_fraction
    if (
        cadence.numerator != description.cadence_seconds_numerator
        or cadence.denominator != description.cadence_seconds_denominator
        or support.numerator != description.source_support_end_seconds_numerator
        or support.denominator != description.source_support_end_seconds_denominator
    ):
        _fail("Target-grid fractions must be reduced.")
    count = description.target_sample_count
    if isinstance(count, bool) or not isinstance(count, int) or count <= 1:
        _fail("Target sample count must be an integer greater than one.")
    if count > MAX_TARGET_SAMPLE_COUNT:
        _fail("Target sample count exceeds the signed 64-bit index limit.")
    if description.last_target_elapsed_fraction > support:
        _fail("Last target sample lies outside accepted source support.")
    if description.next_target_elapsed_fraction <= support:
        _fail("Target sample count omits a supported next sample.")
    expected_identity = compute_guided_continuous_rwd_target_grid_identity(description)
    if description.target_grid_identity != expected_identity:
        _fail("Target-grid identity mismatch.")


def build_guided_continuous_rwd_target_grid(
    recording: GuidedContinuousRwdRecordingDescription,
    continuity_evaluation: ContinuousRwdDiscontinuityEvaluation,
) -> GuidedContinuousRwdTargetGridDescription:
    """Build one immutable scalar target-grid authority without source access."""
    if not isinstance(recording, GuidedContinuousRwdRecordingDescription):
        _fail("recording must be a GuidedContinuousRwdRecordingDescription.")
    try:
        _validate_recording_description(recording)
    except (ContinuousRwdRecordingAuthorityError, TypeError, ValueError) as exc:
        raise ContinuousRwdTargetGridError(
            "Continuous RWD recording authority is invalid."
        ) from exc
    _validate_b2_compatibility(recording, continuity_evaluation)

    cadence = _decimal_fraction(
        recording.cadence.nominal_cadence_seconds,
        "nominal cadence",
    )
    support = _decimal_fraction(
        recording.time.measured_support_end_seconds,
        "measured support end",
    )
    target_sample_count = _resolve_target_sample_count(support, cadence)
    if target_sample_count <= 1:
        _fail("Target grid must contain at least two samples.")
    if target_sample_count > MAX_TARGET_SAMPLE_COUNT:
        _fail("Target sample count exceeds the signed 64-bit index limit.")

    continuity_identity = (
        compute_continuous_rwd_discontinuity_evaluation_identity(
            continuity_evaluation
        )
    )
    draft = GuidedContinuousRwdTargetGridDescription(
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        grid_policy_name=GRID_POLICY_NAME,
        grid_policy_version=GRID_POLICY_VERSION,
        recording_identity=recording.recording_identity,
        continuity_evaluation_identity=continuity_identity,
        cadence_seconds_numerator=cadence.numerator,
        cadence_seconds_denominator=cadence.denominator,
        source_support_end_seconds_numerator=support.numerator,
        source_support_end_seconds_denominator=support.denominator,
        target_sample_count=target_sample_count,
        target_grid_identity="",
    )
    description = GuidedContinuousRwdTargetGridDescription(
        **{
            **draft.__dict__,
            "target_grid_identity": (
                compute_guided_continuous_rwd_target_grid_identity(draft)
            ),
        }
    )
    _validate_target_grid_description(description)
    return description
