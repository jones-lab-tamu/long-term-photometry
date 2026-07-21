"""Pure CR1 policy for discontinuities relative to observed RWD cadence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from photometry_pipeline.guided_continuous_rwd_recording import (
    ContinuousRwdCadenceAuthority,
)


POLICY_NAME = "continuous-rwd-observed-cadence-discontinuity"
POLICY_VERSION = "v1"
ABSOLUTE_FLOOR_SECONDS = 0.001
RELATIVE_TOLERANCE_FRACTION = 0.02
EMPIRICAL_MULTIPLIER = 1.5
UPPER_CAP_FRACTION = 0.20

NORMAL_INTERVAL = "normal_interval"
SHORT_INTERVAL_ANOMALY = "short_interval_anomaly"
MATERIAL_LONG_INTERVAL = "material_long_interval"
INTERVAL_CATEGORIES = (
    NORMAL_INTERVAL,
    SHORT_INTERVAL_ANOMALY,
    MATERIAL_LONG_INTERVAL,
)


class ContinuousRwdDiscontinuityPolicyError(ValueError):
    """Observed cadence evidence cannot be used by the CR1 policy."""


@dataclass(frozen=True)
class ContinuousRwdDiscontinuityTolerance:
    policy_name: str
    policy_version: str
    nominal_cadence_seconds: float
    q01_seconds: float
    q99_seconds: float
    absolute_floor_seconds: float
    relative_component_seconds: float
    empirical_component_seconds: float
    upper_cap_seconds: float
    uncapped_tolerance_seconds: float
    final_tolerance_seconds: float


@dataclass(frozen=True)
class ContinuousRwdIntervalClassification:
    category: str
    dt_seconds: float
    nominal_cadence_seconds: float
    tolerance_seconds: float
    residual_seconds: float


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContinuousRwdDiscontinuityPolicyError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ContinuousRwdDiscontinuityPolicyError(f"{name} must be finite.")
    return result


def _required_quantiles(
    cadence: ContinuousRwdCadenceAuthority,
) -> tuple[float, float, float]:
    probabilities: list[float] = []
    values_by_probability: dict[float, list[float]] = {}
    for item in cadence.quantiles:
        probability = _finite_number(item.probability, "quantile probability")
        probabilities.append(probability)
        values_by_probability.setdefault(probability, []).append(item.dt_seconds)
    if any(left >= right for left, right in zip(probabilities, probabilities[1:])):
        raise ContinuousRwdDiscontinuityPolicyError(
            "Cadence quantile probabilities must be strictly ascending."
        )

    resolved: list[float] = []
    for probability in (0.01, 0.5, 0.99):
        matches = values_by_probability.get(probability, [])
        if len(matches) != 1:
            raise ContinuousRwdDiscontinuityPolicyError(
                f"Required cadence quantile {probability!r} must be present exactly once."
            )
        value = _finite_number(matches[0], f"quantile {probability!r} value")
        if value <= 0.0:
            raise ContinuousRwdDiscontinuityPolicyError(
                f"Quantile {probability!r} value must be greater than zero."
            )
        resolved.append(value)
    return resolved[0], resolved[1], resolved[2]


def _validate_tolerance(
    tolerance: ContinuousRwdDiscontinuityTolerance,
) -> None:
    if not isinstance(tolerance, ContinuousRwdDiscontinuityTolerance):
        raise ContinuousRwdDiscontinuityPolicyError(
            "tolerance must be a ContinuousRwdDiscontinuityTolerance."
        )
    if tolerance.policy_name != POLICY_NAME or tolerance.policy_version != POLICY_VERSION:
        raise ContinuousRwdDiscontinuityPolicyError(
            "Unsupported discontinuity policy metadata."
        )
    nominal = _finite_number(
        tolerance.nominal_cadence_seconds, "nominal cadence"
    )
    q01 = _finite_number(tolerance.q01_seconds, "q0.01")
    q99 = _finite_number(tolerance.q99_seconds, "q0.99")
    expected_relative = RELATIVE_TOLERANCE_FRACTION * nominal
    expected_empirical = EMPIRICAL_MULTIPLIER * max(
        nominal - q01, q99 - nominal, 0.0
    )
    expected_cap = UPPER_CAP_FRACTION * nominal
    expected_uncapped = max(
        ABSOLUTE_FLOOR_SECONDS, expected_relative, expected_empirical
    )
    expected_final = min(expected_cap, expected_uncapped)
    expected = (
        ABSOLUTE_FLOOR_SECONDS,
        expected_relative,
        expected_empirical,
        expected_cap,
        expected_uncapped,
        expected_final,
    )
    actual = (
        tolerance.absolute_floor_seconds,
        tolerance.relative_component_seconds,
        tolerance.empirical_component_seconds,
        tolerance.upper_cap_seconds,
        tolerance.uncapped_tolerance_seconds,
        tolerance.final_tolerance_seconds,
    )
    if actual != expected or expected_final <= 0.0:
        raise ContinuousRwdDiscontinuityPolicyError(
            "Discontinuity tolerance does not match policy v1."
        )


def resolve_continuous_rwd_discontinuity_tolerance(
    cadence: ContinuousRwdCadenceAuthority,
) -> ContinuousRwdDiscontinuityTolerance:
    """Resolve the bounded v1 tolerance from accepted observed cadence."""
    if not isinstance(cadence, ContinuousRwdCadenceAuthority):
        raise ContinuousRwdDiscontinuityPolicyError(
            "cadence must be a ContinuousRwdCadenceAuthority."
        )
    nominal = _finite_number(cadence.nominal_cadence_seconds, "nominal cadence")
    if nominal <= 0.0:
        raise ContinuousRwdDiscontinuityPolicyError(
            "Nominal cadence must be greater than zero."
        )
    q01, q50, q99 = _required_quantiles(cadence)
    if not math.isclose(q50, nominal, rel_tol=1e-12, abs_tol=1e-12):
        raise ContinuousRwdDiscontinuityPolicyError(
            "The q0.5 cadence quantile must agree with nominal cadence."
        )
    if q01 > nominal or q99 < nominal:
        raise ContinuousRwdDiscontinuityPolicyError(
            "Required cadence quantiles must bracket nominal cadence."
        )

    relative = RELATIVE_TOLERANCE_FRACTION * nominal
    empirical = EMPIRICAL_MULTIPLIER * max(nominal - q01, q99 - nominal, 0.0)
    cap = UPPER_CAP_FRACTION * nominal
    uncapped = max(ABSOLUTE_FLOOR_SECONDS, relative, empirical)
    final = min(cap, uncapped)
    for value, name in (
        (relative, "relative component"),
        (empirical, "empirical component"),
        (cap, "upper cap"),
        (uncapped, "uncapped tolerance"),
        (final, "final tolerance"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ContinuousRwdDiscontinuityPolicyError(
                f"{name.capitalize()} must be finite and nonnegative."
            )
    if cap <= 0.0 or final <= 0.0 or final > UPPER_CAP_FRACTION * nominal:
        raise ContinuousRwdDiscontinuityPolicyError(
            "Final discontinuity tolerance is outside policy v1 bounds."
        )
    tolerance = ContinuousRwdDiscontinuityTolerance(
        policy_name=POLICY_NAME,
        policy_version=POLICY_VERSION,
        nominal_cadence_seconds=nominal,
        q01_seconds=q01,
        q99_seconds=q99,
        absolute_floor_seconds=ABSOLUTE_FLOOR_SECONDS,
        relative_component_seconds=relative,
        empirical_component_seconds=empirical,
        upper_cap_seconds=cap,
        uncapped_tolerance_seconds=uncapped,
        final_tolerance_seconds=final,
    )
    _validate_tolerance(tolerance)
    return tolerance


def classify_continuous_rwd_interval(
    dt_seconds: float,
    *,
    tolerance: ContinuousRwdDiscontinuityTolerance,
) -> ContinuousRwdIntervalClassification:
    """Classify one positive interval relative to accepted observed cadence."""
    _validate_tolerance(tolerance)
    dt = _finite_number(dt_seconds, "dt_seconds")
    if dt <= 0.0:
        raise ContinuousRwdDiscontinuityPolicyError(
            "dt_seconds must be greater than zero."
        )
    nominal = tolerance.nominal_cadence_seconds
    final = tolerance.final_tolerance_seconds
    residual = abs(dt - nominal)
    lower_boundary = nominal - final
    upper_boundary = nominal + final
    if lower_boundary <= dt <= upper_boundary:
        category = NORMAL_INTERVAL
    elif dt < lower_boundary:
        category = SHORT_INTERVAL_ANOMALY
    else:
        category = MATERIAL_LONG_INTERVAL
    return ContinuousRwdIntervalClassification(
        category=category,
        dt_seconds=dt,
        nominal_cadence_seconds=nominal,
        tolerance_seconds=final,
        residual_seconds=residual,
    )
