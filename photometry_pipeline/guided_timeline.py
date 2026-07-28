"""Small, format-neutral helpers for the accepted Guided timeline contract.

The placement semantics remain owned by ``viz.phasic_data_prep`` for
intermittent layouts.  This module only validates/serializes the explicit
Guided choices and maps an elapsed coordinate when a continuous source has no
session datetime sequence.
"""

from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Any

GUIDED_TIMELINE_MODES = ("elapsed", "civil", "fixed_daily_anchor")
GUIDED_TIMELINE_MODE_SET = frozenset(GUIDED_TIMELINE_MODES)
GUIDED_TIMELINE_CLOCK_SOURCES = (
    "validated_metadata",
    "user_entered",
    "not_applicable",
)
GUIDED_TIMELINE_CLOCK_SOURCE_SET = frozenset(GUIDED_TIMELINE_CLOCK_SOURCES)
GUIDED_DEFAULT_FIXED_DAILY_ANCHOR_CLOCK = "07:00"
_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


class GuidedTimelineError(ValueError):
    """An accepted Guided timeline choice cannot be represented safely."""


def parse_guided_clock(value: str | None, *, field_name: str) -> tuple[int, str]:
    """Return seconds since midnight and a canonical HH:MM[:SS] string."""
    text = str(value or "").strip()
    match = _CLOCK_RE.fullmatch(text)
    if match is None:
        raise GuidedTimelineError(
            f"{field_name} must be a valid clock time in HH:MM format."
        )
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3) or 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise GuidedTimelineError(
            f"{field_name} must be a valid clock time in HH:MM format."
        )
    canonical = f"{hour:02d}:{minute:02d}"
    if second:
        canonical += f":{second:02d}"
    return hour * 3600 + minute * 60 + second, canonical


def valid_guided_clock(value: str | None) -> bool:
    try:
        parse_guided_clock(value, field_name="Clock time")
    except GuidedTimelineError:
        return False
    return True


def guided_clock_from_datetime(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise GuidedTimelineError("An authoritative datetime is required.")
    text = f"{value.hour:02d}:{value.minute:02d}"
    if value.second or value.microsecond:
        text += f":{value.second:02d}"
    return text


def normalize_guided_clock_source(value: str | None) -> str:
    """Validate and return the recording-start clock source name.

    Only the three current values are accepted.
    """
    source = str(value or "").strip().lower()
    if source not in GUIDED_TIMELINE_CLOCK_SOURCE_SET:
        raise GuidedTimelineError(
            f"Unsupported recording-start clock source: {value!r}."
        )
    return source


def guided_clock_display_offset_sec(
    effective_clock: str | None,
    validated_clock: str | None,
) -> float:
    """Seconds to shift display placement so the recording starts at the
    effective clock.

    This is the single offset rule for intermittent placement: the effective
    recording-start clock minus the validated first-session clock.  It is a
    display-placement quantity only; stored acquisition timestamps keep their
    validated values.

    The field carries a clock time, not a date, so the raw difference between
    two clocks is ambiguous across midnight: correcting a 23:59:30 start to
    00:00:30 is a 60-second correction, not a 23 h 59 m one.  The offset is
    therefore resolved to the *nearest clock occurrence* -- the smallest signed
    difference among the raw difference and that difference plus or minus one
    day.  This is a display-placement correction only; it introduces no date
    field and no timezone handling.
    """
    effective_sec, _ = parse_guided_clock(
        effective_clock, field_name="Clock time at recording start"
    )
    validated_sec, _ = parse_guided_clock(
        validated_clock, field_name="Validated recording start clock"
    )
    raw = float(effective_sec - validated_sec)
    day = 86400.0
    return min((raw, raw + day, raw - day), key=abs)


def timeline_provenance(
    *,
    timeline_anchor_mode: str,
    fixed_daily_anchor_clock: str | None,
    recording_start_clock: str | None,
    recording_start_clock_source: str,
) -> dict[str, Any]:
    """Serialize the four run-level timing facts with consistent nulls."""
    mode = str(timeline_anchor_mode or "").strip().lower()
    if mode not in GUIDED_TIMELINE_MODE_SET:
        raise GuidedTimelineError(f"Unsupported Guided timeline mode: {mode!r}.")
    source = normalize_guided_clock_source(recording_start_clock_source)
    fixed = None
    if mode == "fixed_daily_anchor":
        _, fixed = parse_guided_clock(
            fixed_daily_anchor_clock, field_name="Start of plotted day"
        )
    elif fixed_daily_anchor_clock is not None:
        raise GuidedTimelineError(
            "A fixed daily anchor clock is only valid for fixed daily anchor mode."
        )
    start = None
    if recording_start_clock is not None:
        _, start = parse_guided_clock(
            recording_start_clock, field_name="Clock time at recording start"
        )
    if source == "not_applicable" and start is not None:
        raise GuidedTimelineError(
            "A recording-start clock cannot be marked not applicable."
        )
    if mode != "elapsed" and start is None:
        raise GuidedTimelineError(
            "A recording-start clock is required for civil or fixed timeline placement."
        )
    if source != "not_applicable" and start is None:
        raise GuidedTimelineError(
            "A recording-start clock is required for the selected source."
        )
    if mode == "elapsed" and (start is not None or source != "not_applicable"):
        raise GuidedTimelineError(
            "Elapsed placement must not carry a recording-start clock."
        )
    return {
        "timeline_mode": mode,
        "fixed_daily_anchor_clock": fixed,
        "recording_start_clock": start,
        "recording_start_clock_source": source,
    }


def timeline_provenance_from_intent(intent: Any) -> dict[str, Any]:
    return timeline_provenance(
        timeline_anchor_mode=getattr(intent, "timeline_anchor_mode", ""),
        fixed_daily_anchor_clock=getattr(intent, "fixed_daily_anchor_clock", None),
        recording_start_clock=getattr(intent, "recording_start_clock", None),
        recording_start_clock_source=getattr(
            intent, "recording_start_clock_source", "not_applicable"
        ),
    )


def map_elapsed_coordinate(
    elapsed_sec: float,
    *,
    timeline_anchor_mode: str,
    fixed_daily_anchor_clock: str | None = None,
    recording_start_clock: str | None = None,
) -> tuple[int, float]:
    """Map one elapsed coordinate to ``(day_index, seconds_from_day_start)``.

    This is the continuous counterpart of the existing ``compute_day_layout``
    placement rules.  Python floor division keeps a sample before a fixed
    anchor in the preceding anchored day; day indices are then normalized
    relative to the anchored day containing the first sample.
    """
    try:
        elapsed = float(elapsed_sec)
    except (TypeError, ValueError) as exc:
        raise GuidedTimelineError("Elapsed time must be numeric.") from exc
    if not math.isfinite(elapsed):
        raise GuidedTimelineError("Elapsed time must be finite.")
    mode = str(timeline_anchor_mode or "").strip().lower()
    if mode not in GUIDED_TIMELINE_MODE_SET:
        raise GuidedTimelineError(f"Unsupported Guided timeline mode: {mode!r}.")
    if mode == "elapsed":
        offset = elapsed
    else:
        start_sec, _ = parse_guided_clock(
            recording_start_clock, field_name="Clock time at recording start"
        )
        offset = elapsed + start_sec
        if mode == "fixed_daily_anchor":
            anchor_sec, _ = parse_guided_clock(
                fixed_daily_anchor_clock, field_name="Start of plotted day"
            )
            offset -= anchor_sec
    base_day_index = 0
    if mode != "elapsed":
        # A continuous elapsed-zero sample may occur before the selected
        # fixed anchor.  Day indices are relative to the anchored day that
        # contains the first sample, so that first sample remains day 0 while
        # retaining its correct within-day coordinate (for example 03:00
        # with a 07:00 anchor maps to day 0, hour 20).
        base_day_index = math.floor(
            (start_sec - (anchor_sec if mode == "fixed_daily_anchor" else 0))
            / 86400.0
        )
    raw_day_index = math.floor(offset / 86400.0)
    day_index = raw_day_index - base_day_index
    within_day = offset - (raw_day_index * 86400.0)
    # Floating-point arithmetic can leave a value one ulp outside the day.
    if within_day < 0 and abs(within_day) < 1e-9:
        within_day = 0.0
    if within_day >= 86400.0 and within_day - 86400.0 < 1e-9:
        day_index += 1
        within_day = 0.0
    return int(day_index), float(within_day)


def accepted_continuous_window_timing(
    accepted_draft: Any,
) -> dict[str, Any]:
    """Return the accepted window length and authoritative segment step.

    The accepted Guided draft has independent ``continuous_window_sec`` and
    ``continuous_step_sec`` fields, and both are part of the accepted plan
    identity.  The correction segment plan's duration is the correction
    window length, not the publication step, so it must not be substituted
    for the accepted step.
    """
    length = float(getattr(accepted_draft, "continuous_window_sec", 0.0) or 0.0)
    if not math.isfinite(length) or length <= 0:
        raise GuidedTimelineError("The accepted continuous window length is invalid.")
    step_source = "accepted_draft.continuous_step_sec"
    step_value = getattr(accepted_draft, "continuous_step_sec", None)
    try:
        step = float(step_value)
    except (TypeError, ValueError) as exc:
        raise GuidedTimelineError("The accepted continuous window step is invalid.") from exc
    if not math.isfinite(step) or step <= 0:
        raise GuidedTimelineError("The accepted continuous window step is invalid.")
    return {
        "window_length_sec": length,
        "window_step_sec": step,
        "window_length_source": "accepted_draft.continuous_window_sec",
        "window_step_source": step_source,
    }


def timeline_mode_label(mode: str) -> str:
    return {
        "fixed_daily_anchor": "Fixed daily anchor",
        "civil": "Civil clock",
        "elapsed": "Elapsed from recording start",
    }.get(str(mode or "").strip().lower(), str(mode or ""))
