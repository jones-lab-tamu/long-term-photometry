"""Conservative recording-structure inference for Guided setup."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import median
from typing import Any, Iterable


@dataclass(frozen=True)
class GuidedRecordingStructureInference:
    supported: bool
    status: str
    sessions_per_hour: int | None = None
    session_duration_sec: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    message: str = ""


_RWD_SESSION_ID_FORMATS = (
    "%Y_%m_%d-%H_%M_%S",
    "%Y-%m-%d-%H-%M-%S",
    "%Y%m%d-%H%M%S",
)


def _display_session_duration(raw_duration: float) -> tuple[float, bool]:
    nearest_second = round(raw_duration)
    normalized = abs(raw_duration - nearest_second) <= 0.1
    return (
        float(nearest_second) if normalized else float(raw_duration),
        normalized,
    )


def _parse_rwd_session_start(value: object) -> datetime | None:
    text = str(value or "").strip()
    for format_text in _RWD_SESSION_ID_FORMATS:
        try:
            return datetime.strptime(text, format_text)
        except ValueError:
            continue
    return None


def infer_guided_recording_structure(
    discovery_cache: dict[str, Any] | None,
    source_path: str,
    input_format: str,
    *,
    rwd_chunk_contracts: Iterable[dict[str, Any]] = (),
) -> GuidedRecordingStructureInference:
    """Infer recording timing without guessing unsupported formats."""
    resolved = str(
        (discovery_cache or {}).get("resolved_format") or input_format or ""
    ).strip().lower()
    if resolved != "rwd":
        return GuidedRecordingStructureInference(
            supported=False,
            status="unsupported",
            evidence={"resolved_format": resolved, "source_path": source_path},
            message=(
                "Confirm sessions per hour and session duration for this "
                "recording."
            ),
        )

    contracts = [dict(contract) for contract in rwd_chunk_contracts]
    sessions = list((discovery_cache or {}).get("sessions") or ())
    sampling_evidence = {
        "duration_source": "sampled_rwd_chunk_contracts",
        "duration_sample_size": len(contracts),
        "duration_sample_session_ids": [
            str(contract.get("sample_session_id") or "")
            for contract in contracts
        ],
        "duration_sample_paths": [
            str(contract.get("csv_path") or "")
            for contract in contracts
        ],
        "n_sessions_available": len(sessions),
        "duration_sampling_strategy": "first_middle_last",
    }
    try:
        durations = [
            float(contract["chunk_duration_sec"])
            for contract in contracts
            if float(contract.get("chunk_duration_sec") or 0) > 0
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return GuidedRecordingStructureInference(
            supported=True,
            status="failed",
            evidence={
                "resolved_format": resolved,
                "source_path": source_path,
                **sampling_evidence,
                "error": f"{type(exc).__name__}: {exc}",
            },
            message=(
                "Could not confidently detect session timing. Enter sessions "
                "per hour and session duration manually."
            ),
        )
    duration = median(durations) if durations else None
    duration_tolerance = (
        max(0.5, 0.01 * duration) if duration is not None else None
    )
    if (
        duration is None
        or any(
            abs(value - duration) > duration_tolerance
            for value in durations
        )
    ):
        return GuidedRecordingStructureInference(
            supported=True,
            status="ambiguous",
            session_duration_sec=duration,
            evidence={
                "resolved_format": resolved,
                "source_path": source_path,
                **sampling_evidence,
                "chunk_duration_sec": durations,
            },
            message=(
                "Could not confidently detect session timing. Enter sessions "
                "per hour and session duration manually."
            ),
        )

    starts = [
        _parse_rwd_session_start(session.get("session_id"))
        for session in sessions
        if isinstance(session, dict)
    ]
    if len(starts) < 2 or any(value is None for value in starts):
        return GuidedRecordingStructureInference(
            supported=True,
            status="ambiguous",
            session_duration_sec=float(duration),
            evidence={
                "resolved_format": resolved,
                "source_path": source_path,
                **sampling_evidence,
                "chunk_duration_sec": durations,
                "session_ids": [
                    session.get("session_id")
                    for session in sessions
                    if isinstance(session, dict)
                ],
            },
            message=(
                "Could not confidently detect session timing. Enter sessions "
                "per hour and session duration manually."
            ),
        )

    intervals = [
        (current - previous).total_seconds()
        for previous, current in zip(starts, starts[1:])
    ]
    cadence = median(intervals)
    cadence_tolerance = max(1.0, 0.01 * cadence)
    rate = 3600.0 / cadence if cadence > 0 else 0.0
    rounded_rate = int(round(rate))
    if (
        cadence <= 0
        or any(abs(value - cadence) > cadence_tolerance for value in intervals)
        or rounded_rate <= 0
        or abs(rate - rounded_rate) > 0.05
        or rounded_rate * duration > 3600.0 + 1e-9
    ):
        return GuidedRecordingStructureInference(
            supported=True,
            status="ambiguous",
            session_duration_sec=float(duration),
            evidence={
                "resolved_format": resolved,
                "source_path": source_path,
                **sampling_evidence,
                "chunk_duration_sec": durations,
                "session_start_intervals_sec": intervals,
                "candidate_sessions_per_hour": rate,
            },
            message=(
                "Could not confidently detect session timing. Enter sessions "
                "per hour and session duration manually."
            ),
        )

    display_duration, duration_normalized = _display_session_duration(
        float(duration)
    )
    return GuidedRecordingStructureInference(
        supported=True,
        status="inferred",
        sessions_per_hour=rounded_rate,
        session_duration_sec=display_duration,
        evidence={
            "resolved_format": resolved,
            "source_path": source_path,
            **sampling_evidence,
            "cadence_source": "ordered_rwd_session_ids",
            "chunk_duration_sec": durations,
            "session_start_intervals_sec": intervals,
            "raw_session_duration_sec": float(duration),
            "display_session_duration_sec": display_duration,
            "duration_display_normalized": duration_normalized,
        },
        message=(
            f"Detected {rounded_rate} sessions/hour and approximately "
            f"{display_duration:g} s/session. Please confirm."
        ),
    )
