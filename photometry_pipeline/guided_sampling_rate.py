"""Shared whole-number sampling-rate rule for Guided analysis."""

from __future__ import annotations

import math


GUIDED_SAMPLING_RATE_RELATIVE_TOLERANCE = 0.005
GUIDED_SAMPLING_RATE_FAILURE_MESSAGE = (
    "Sampling rate could not be determined reliably. "
    "Check the recording timestamps before continuing."
)


class GuidedSamplingRateError(ValueError):
    """The selected recording does not establish a reliable whole-Hz rate."""


def normalize_guided_sampling_rate_hz(inferred_rate_hz: float) -> float:
    """Return the nearest whole-Hz rate when it is within exactly 0.5%."""
    rate = float(inferred_rate_hz)
    if not math.isfinite(rate) or rate <= 0.0:
        raise GuidedSamplingRateError(GUIDED_SAMPLING_RATE_FAILURE_MESSAGE)
    whole_rate = int(math.floor(rate + 0.5))
    if whole_rate <= 0:
        raise GuidedSamplingRateError(GUIDED_SAMPLING_RATE_FAILURE_MESSAGE)
    relative_difference = abs(rate - whole_rate) / whole_rate
    if relative_difference > GUIDED_SAMPLING_RATE_RELATIVE_TOLERANCE:
        raise GuidedSamplingRateError(GUIDED_SAMPLING_RATE_FAILURE_MESSAGE)
    return float(whole_rate)
