"""Curated synthetic demo dataset generation helpers for the GUI and examples."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


DEMO_FORMAT = "rwd"
DEMO_SESSIONS_PER_HOUR = 2
DEMO_MODE = "both"
FAST_DEMO_TYPE = "Fast quickstart demo"
LONG_DEMO_TYPE = "Long-duration intermittent demo"

REPO_ROOT = Path(__file__).resolve().parents[1]
FAST_SOURCE_DIR = REPO_ROOT / "examples" / "data" / "synthetic_photometry_basic"
SYNTH_SCRIPT = REPO_ROOT / "tools" / "synth_photometry_dataset.py"

GUIDED_DEMO_FOLDER_NAME = "long_term_photometry_guided_demo"
GUIDED_DEMO_TYPE = "Synthetic Guided CSV demo"
GUIDED_DEMO_FORMAT = "custom_tabular"
GUIDED_DEMO_SESSION_COUNT = 96
GUIDED_DEMO_SESSIONS_PER_HOUR = 2
GUIDED_DEMO_SESSIONS_PER_DAY = 24 * GUIDED_DEMO_SESSIONS_PER_HOUR
GUIDED_DEMO_SESSION_DURATION_SEC = 600.0
GUIDED_DEMO_FS_HZ = 20
GUIDED_DEMO_ROWS_PER_SESSION = 12000
GUIDED_DEMO_SEED = 2026
GUIDED_DEMO_TONIC_TRUTH_FILENAME = "tonic_truth.json"
GUIDED_DEMO_FIRST_SESSION_START = datetime(2025, 1, 1, 0, 0, 0)
GUIDED_DEMO_TONIC_OFFSET_AU = (0.0, 0.0)
GUIDED_DEMO_TONIC_AMPLITUDE_AU = (4.0, 4.5)
GUIDED_DEMO_TONIC_PERIOD_HOURS = (24.0, 24.0)
GUIDED_DEMO_TONIC_PHASE_HOURS = (7.0, 7.0)
GUIDED_DEMO_HEADERS = (
    "ElapsedSeconds",
    "ROI1_Signal",
    "ROI1_Reference",
    "ROI2_Signal",
    "ROI2_Reference",
)

# Fixed internal shape of the demonstration signals. These are not user options
# and are not exposed anywhere in the application.
_EVENT_RATE_PER_MIN = (3.2, 2.6)  # ROI1, ROI2 baseline transient rate
_EVENT_RATE_MODULATION = (0.55, 0.38)  # recording-scale swing of the rate
_EVENT_AMPLITUDE_MODULATION = (0.22, 0.14)  # recording-scale swing of amplitude
_EVENT_ROI_PHASE_OFFSET = 0.9  # ROI2 activity lags ROI1 across the recording
_EVENT_MAX_ABUNDANCE_HOUR = 7.0
_EVENT_MEDIAN_AMPLITUDE = (1.9, 2.15)
_EVENT_RISE_SEC = (0.08, 0.26)
_EVENT_DECAY_SEC = (0.45, 2.4)
_EVENT_TAIL_GUARD_SEC = 2.0  # keep transients from being cut off at session end
_REFERENCE_EVENT_BLEED = 0.05  # small event leak into the reference channel

# The continuous demo is one uninterrupted RWD acquisition folder, which is the
# only source structure the Guided continuous path accepts.
GUIDED_CONTINUOUS_DEMO_FOLDER_NAME = "long_term_photometry_continuous_demo"
GUIDED_CONTINUOUS_DEMO_TYPE = "Synthetic Guided continuous demo"
GUIDED_CONTINUOUS_DEMO_FORMAT = "custom_tabular"
# 8 Hz gives an exactly representable 0.125 s sample period, so every written
# timestamp is the same double the continuous target grid computes for it. A
# period such as 0.1 s is not binary-exact, and over 48 hours the final target
# lands one unit in the last place above the written timestamp, which the
# projection reader correctly rejects as missing right-hand support.
GUIDED_CONTINUOUS_DEMO_FS_HZ = 8
GUIDED_CONTINUOUS_DEMO_DURATION_SEC = 48 * 3600.0
GUIDED_CONTINUOUS_DEMO_BLOCK_SEC = 600.0
GUIDED_CONTINUOUS_DEMO_FILE_NAME = "continuous_recording.csv"
GUIDED_CONTINUOUS_DEMO_HEADERS = (
    "ElapsedSeconds",
    "ROI1_Signal",
    "ROI1_Reference",
    "ROI2_Signal",
    "ROI2_Reference",
)
_CONTINUOUS_DAY_SEC = 86400.0
_CONTINUOUS_EVENT_LOOKBACK_SEC = 20.0  # longest transient tail crossing a block


@dataclass(frozen=True)
class DemoGenerationResult:
    success: bool
    demo_type: str
    input_dir: Path
    config_path: Path
    format: str
    sessions_per_hour: int
    mode: str
    message: str
    stdout_path: Path | None = None
    stderr_path: Path | None = None


def guided_demo_readme_text() -> str:
    return """# Synthetic Guided CSV Demo

These CSV files are synthetic demonstration data, not real biological data.
Select this containing folder in Guided Mode.

- Format: CSV files
- Recording structure: Intermittent/session-based recording
- Sessions: 96 files across 48 scheduled hours
- Sessions per hour: 2
- Session duration: 600 seconds
- Time column: `ElapsedSeconds`
- Time unit: seconds
- Confirm the displayed natural filename order; session filenames include authoritative start timestamps
- ROI1 mapping: `ROI1_Signal` with `ROI1_Reference`
- ROI2 mapping: `ROI2_Signal` with `ROI2_Reference`

Recommended timeline display:
- Fixed daily anchor
- Start of plotted day: `07:00`
- Clock time at recording start: `00:00:00`

The tonic rhythms and base phasic modulation share a daily peak at `07:00`
and trough at `19:00`.

The two ROIs contain explicit signal-only tonic rhythms with known ground truth,
rhythmic phasic activity, shared optical nuisance, and realistic noise for
demonstrating the workflow.
Do not draw biological conclusions from this dataset.
"""


def guided_demo_session_start_time(session_index: int) -> datetime:
    """Return the deterministic authoritative start time for one demo session."""
    return GUIDED_DEMO_FIRST_SESSION_START + timedelta(
        hours=float(session_index) / float(GUIDED_DEMO_SESSIONS_PER_HOUR)
    )


def guided_demo_session_filename(session_index: int) -> str:
    """Return one naturally ordered session filename with its start timestamp."""
    start = guided_demo_session_start_time(session_index)
    return (
        f"session_{int(session_index) + 1:04d}_"
        f"{start:%Y_%m_%d-%H_%M_%S}.csv"
    )


def _guided_demo_tonic_value(session_index: int, roi_index: int) -> float:
    """Return the exact signal-only tonic component for one session and ROI."""
    elapsed_hours = float(session_index) / float(GUIDED_DEMO_SESSIONS_PER_HOUR)
    return float(
        GUIDED_DEMO_TONIC_OFFSET_AU[roi_index]
        + GUIDED_DEMO_TONIC_AMPLITUDE_AU[roi_index]
        * np.cos(
            2.0
            * np.pi
            * (elapsed_hours - GUIDED_DEMO_TONIC_PHASE_HOURS[roi_index])
            / GUIDED_DEMO_TONIC_PERIOD_HOURS[roi_index]
        )
    )


def _guided_demo_tonic_truth_payload(session_count: int) -> dict:
    records = []
    for session_index in range(int(session_count)):
        session_start = guided_demo_session_start_time(session_index)
        elapsed_hours = float(
            (session_start - GUIDED_DEMO_FIRST_SESSION_START).total_seconds()
            / 3600.0
        )
        session_id = Path(guided_demo_session_filename(session_index)).stem
        for roi_index, roi_id in enumerate(("ROI1", "ROI2")):
            records.append(
                {
                    "roi_id": roi_id,
                    "session_id": session_id,
                    "session_index": int(session_index),
                    "authoritative_session_start_time": session_start.isoformat(),
                    "elapsed_hours": elapsed_hours,
                    "tonic_value_au": _guided_demo_tonic_value(
                        session_index, roi_index
                    ),
                    "tonic_offset_au": GUIDED_DEMO_TONIC_OFFSET_AU[roi_index],
                    "tonic_period_hours": GUIDED_DEMO_TONIC_PERIOD_HOURS[roi_index],
                    "tonic_amplitude_au": GUIDED_DEMO_TONIC_AMPLITUDE_AU[roi_index],
                    "tonic_phase_hours": GUIDED_DEMO_TONIC_PHASE_HOURS[roi_index],
                }
            )
    return {
        "schema": "guided_csv_demo_tonic_truth.v1",
        "component": "signal_only",
        "value_units": "raw fluorescence AU",
        "time_units": "elapsed recording hours",
        "equation": (
            "tonic_offset_au + tonic_amplitude_au * cos(2*pi*"
            "(elapsed_hours - tonic_phase_hours) / tonic_period_hours)"
        ),
        "records": records,
    }


def _guided_demo_event_rate_modulation(session_index: int, roi_index: int) -> float:
    """Return the signed daily phasic rate modulation for one session and ROI."""
    recording_phase = 2.0 * np.pi * (
        float(session_index) / float(GUIDED_DEMO_SESSIONS_PER_DAY)
        + 0.25
        - _EVENT_MAX_ABUNDANCE_HOUR / 24.0
    )
    return float(np.sin(recording_phase))


def _slow_variation(
    rng: np.random.Generator,
    samples: int,
    fs_hz: float,
    tau_sec: float,
    std: float,
) -> np.ndarray:
    """Smoothly wandering zero-mean variation, from smoothed white noise.

    ``fs_hz`` may be a coarse grid rate, so the continuous demo can build
    whole-recording components cheaply and interpolate them per block.
    """
    white = rng.normal(0.0, 1.0, samples)
    if samples < 2 or std <= 0.0:
        return np.zeros(samples, dtype=np.float64)
    span = max(1, min(int(round(tau_sec * float(fs_hz))), max(1, samples // 4)))
    kernel_length = min(4 * span, samples)
    kernel = np.exp(-np.arange(kernel_length, dtype=np.float64) / float(span))
    kernel /= float(np.sqrt(np.sum(kernel * kernel)))
    return float(std) * np.convolve(white, kernel, mode="same")


def _transient_kernel(fs_hz: int, rise_sec: float, decay_sec: float) -> np.ndarray:
    """Causal calcium-like transient: fast rise, slower exponential decay."""
    length = max(2, int(round(6.0 * decay_sec * float(fs_hz))))
    offsets = np.arange(length, dtype=np.float64) / float(fs_hz)
    shape = np.exp(-offsets / decay_sec) - np.exp(-offsets / rise_sec)
    peak = float(np.max(shape))
    if peak <= 0.0:
        return np.zeros(length, dtype=np.float64)
    return shape / peak


def _event_start_times(
    rng: np.random.Generator,
    duration_sec: float,
    rate_per_sec: float,
) -> np.ndarray:
    """Irregular start times from exponential inter-event intervals."""
    usable_sec = float(duration_sec) - _EVENT_TAIL_GUARD_SEC
    expected = max(1.0, float(rate_per_sec) * max(usable_sec, 0.0))
    intervals = rng.exponential(1.0 / float(rate_per_sec), int(3.0 * expected) + 8)
    if usable_sec <= 0.0:
        return np.zeros(0, dtype=np.float64)
    start_times = np.cumsum(intervals)
    return start_times[start_times < usable_sec]


def _shared_artifact_trace(
    rng: np.random.Generator,
    samples: int,
    fs_hz: int,
) -> np.ndarray:
    """A few mild common-mode disturbances seen by both channels."""
    trace = np.zeros(samples, dtype=np.float64)
    for _ in range(int(rng.integers(0, 3))):
        amplitude = float(rng.uniform(0.35, 1.10))
        if rng.random() >= 0.25:
            amplitude = -amplitude
        length = max(2, int(round(float(rng.uniform(1.2, 4.5)) * float(fs_hz))))
        start = int(rng.integers(0, max(1, samples)))
        stop = min(samples, start + length)
        if stop <= start:
            continue
        progress = np.arange(stop - start, dtype=np.float64) / float(length)
        shape = np.sqrt(progress) * np.exp(-3.0 * progress)
        peak = float(np.max(shape))
        if peak <= 0.0:
            continue
        trace[start:stop] += amplitude * shape / peak
    return trace


def _guided_demo_session_arrays(
    session_index: int,
    *,
    rows_per_session: int,
    fs_hz: int,
    rng: np.random.Generator,
) -> np.ndarray:
    time_sec = np.arange(rows_per_session, dtype=np.float64) / float(fs_hz)
    duration_sec = float(rows_per_session) / float(fs_hz)
    # Each session is an independent recording on the same rig.
    session_scale = 1.0 + 0.03 * float(rng.normal())

    columns = [time_sec]
    for roi_index in range(2):
        # Non-calcium optical structure seen by both channels of this ROI.
        bleach = -float(rng.uniform(0.7, 1.6)) * (
            1.0 - np.exp(-time_sec / float(rng.uniform(180.0, 420.0)))
        )
        drift = _slow_variation(rng, rows_per_session, fs_hz, 30.0, 0.42)
        common_wander = _slow_variation(rng, rows_per_session, fs_hz, 1.8, 0.20)
        common_noise = rng.normal(0.0, 0.09, rows_per_session)
        common = session_scale * (bleach + drift + common_wander) + common_noise
        artifacts = _shared_artifact_trace(rng, rows_per_session, fs_hz)
        artifact_reference_scale = float(rng.uniform(0.78, 1.06))

        # Calcium-like transients, more active in one half of the recording.
        activity_modulation = _guided_demo_event_rate_modulation(
            session_index, roi_index
        )
        rate_per_min = (
            _EVENT_RATE_PER_MIN[roi_index]
            * (1.0 + _EVENT_RATE_MODULATION[roi_index] * activity_modulation)
            * float(np.exp(rng.normal(0.0, 0.16)))
        )
        amplitude_scale = 1.0 + _EVENT_AMPLITUDE_MODULATION[roi_index] * (
            activity_modulation
        )
        start_times = _event_start_times(
            rng, duration_sec, max(0.25, rate_per_min) / 60.0
        )
        amplitudes = np.clip(
            amplitude_scale
            * rng.lognormal(
                np.log(_EVENT_MEDIAN_AMPLITUDE[roi_index]), 0.45, start_times.size
            ),
            0.6,
            9.0,
        )
        rise_times = rng.uniform(*_EVENT_RISE_SEC, start_times.size)
        decay_times = rng.uniform(*_EVENT_DECAY_SEC, start_times.size)
        events = np.zeros(rows_per_session, dtype=np.float64)
        for start_time, amplitude, rise_sec, decay_sec in zip(
            start_times, amplitudes, rise_times, decay_times
        ):
            start = int(round(float(start_time) * float(fs_hz)))
            kernel = _transient_kernel(fs_hz, float(rise_sec), float(decay_sec))
            stop = min(rows_per_session, start + kernel.size)
            if stop > start:
                events[start:stop] += float(amplitude) * kernel[: stop - start]

        # Channel-specific level, coupling, and noise.
        reference_level = 100.0 + roi_index * 7.0 + float(rng.normal(0.0, 0.9))
        signal_level = 118.0 + roi_index * 8.5 + float(rng.normal(0.0, 1.1))
        reference_gain = float(rng.uniform(0.82, 0.95))
        signal_gain = float(rng.uniform(1.02, 1.18))
        reference_wander = _slow_variation(rng, rows_per_session, fs_hz, 2.5, 0.07)
        signal_wander = _slow_variation(rng, rows_per_session, fs_hz, 2.5, 0.08)
        reference = (
            reference_level
            + reference_gain * common
            + artifact_reference_scale * artifacts
            + _REFERENCE_EVENT_BLEED * events
            + reference_wander
            + rng.normal(0.0, 0.20, rows_per_session)
        )
        signal = (
            signal_level
            + signal_gain * common
            + artifacts
            + events
            + signal_wander
            + rng.normal(0.0, 0.22, rows_per_session)
        )
        # The known tonic is a session-level signal-only component.  It is
        # deliberately absent from the reference/control channel and from all
        # shared, phasic, and nuisance terms above.
        signal += _guided_demo_tonic_value(session_index, roi_index)
        columns.extend([signal, reference])
    return np.column_stack(columns)


def guided_continuous_demo_readme_text() -> str:
    return """# Synthetic Guided Continuous Demo

These data are synthetic demonstration data, not real biological data.
Select this containing folder in Guided Mode.

- Source type: CSV files, one continuous recording
- Recording structure: continuous, one uninterrupted recording
- Total duration: 48 hours
- Sampling rate: 8 Hz
- Time column: `ElapsedSeconds`
- Time unit: seconds
- ROI1 mapping: `ROI1_Signal` with `ROI1_Reference`
- ROI2 mapping: `ROI2_Signal` with `ROI2_Reference`

If Guided does not already establish the structure, choose the continuous
recording option.

Recommended timeline display:
- Fixed daily anchor
- Start of plotted day: `07:00`
- Clock time at recording start: `12:00:00`

The signals contain deterministic bleaching, shared optical variation, noise,
occasional common-mode artifacts, and irregular calcium-like transients for
demonstrating the workflow.
Do not draw biological conclusions from this dataset.
"""


def _continuous_event_schedule(
    rng: np.random.Generator,
    duration_sec: float,
    roi_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Irregular transients across the whole recording, busier once per day.

    The rate varies with time of day, so events are drawn at the daily maximum
    rate and thinned back to the wanted rate. That keeps the draw count bounded
    and the schedule continuous across block boundaries.
    """
    depth = _EVENT_RATE_MODULATION[roi_index]
    peak_rate_per_sec = _EVENT_RATE_PER_MIN[roi_index] * (1.0 + depth) / 60.0
    usable_sec = max(0.0, float(duration_sec) - _EVENT_TAIL_GUARD_SEC)
    expected = max(1.0, peak_rate_per_sec * usable_sec)
    intervals = rng.exponential(
        1.0 / peak_rate_per_sec, int(1.5 * expected) + 32
    )
    candidates = np.cumsum(intervals)
    candidates = candidates[candidates < usable_sec]
    phase = (
        2.0 * np.pi * candidates / _CONTINUOUS_DAY_SEC
        - roi_index * _EVENT_ROI_PHASE_OFFSET
    )
    keep = rng.random(candidates.size) < (1.0 + depth * np.sin(phase)) / (1.0 + depth)
    start_times = candidates[keep]
    amplitude_scale = 1.0 + _EVENT_AMPLITUDE_MODULATION[roi_index] * np.sin(
        2.0 * np.pi * start_times / _CONTINUOUS_DAY_SEC
        - roi_index * _EVENT_ROI_PHASE_OFFSET
    )
    amplitudes = np.clip(
        amplitude_scale
        * rng.lognormal(
            np.log(_EVENT_MEDIAN_AMPLITUDE[roi_index]), 0.45, start_times.size
        ),
        0.6,
        9.0,
    )
    rise_times = rng.uniform(*_EVENT_RISE_SEC, start_times.size)
    decay_times = rng.uniform(*_EVENT_DECAY_SEC, start_times.size)
    return start_times, amplitudes, rise_times, decay_times


def _continuous_artifact_schedule(
    rng: np.random.Generator,
    duration_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A few mild common-mode disturbances per hour of recording."""
    count = max(1, int(round(float(duration_sec) / 3600.0))) * 2
    start_times = np.sort(rng.uniform(0.0, max(1.0, float(duration_sec)), count))
    amplitudes = rng.uniform(0.35, 1.10, count)
    amplitudes = np.where(rng.random(count) < 0.25, amplitudes, -amplitudes)
    durations = rng.uniform(1.2, 4.5, count)
    return start_times, amplitudes, durations


def _render_continuous_events(
    block_time_sec: np.ndarray,
    fs_hz: int,
    schedule: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Add every transient overlapping this block, including tails from before."""
    start_times, amplitudes, rise_times, decay_times = schedule
    events = np.zeros(block_time_sec.size, dtype=np.float64)
    if block_time_sec.size == 0:
        return events
    block_start = float(block_time_sec[0])
    block_end = float(block_time_sec[-1])
    selected = np.flatnonzero(
        (start_times > block_start - _CONTINUOUS_EVENT_LOOKBACK_SEC)
        & (start_times <= block_end)
    )
    for index in selected:
        kernel = _transient_kernel(
            fs_hz, float(rise_times[index]), float(decay_times[index])
        )
        offset = int(round((float(start_times[index]) - block_start) * float(fs_hz)))
        start = max(0, offset)
        stop = min(events.size, offset + kernel.size)
        if stop > start:
            events[start:stop] += float(amplitudes[index]) * kernel[
                start - offset : stop - offset
            ]
    return events


def _render_continuous_artifacts(
    block_time_sec: np.ndarray,
    fs_hz: int,
    schedule: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    start_times, amplitudes, durations = schedule
    trace = np.zeros(block_time_sec.size, dtype=np.float64)
    if block_time_sec.size == 0:
        return trace
    block_start = float(block_time_sec[0])
    block_end = float(block_time_sec[-1])
    selected = np.flatnonzero(
        (start_times > block_start - float(np.max(durations, initial=0.0)))
        & (start_times <= block_end)
    )
    for index in selected:
        length = max(2, int(round(float(durations[index]) * float(fs_hz))))
        offset = int(round((float(start_times[index]) - block_start) * float(fs_hz)))
        start = max(0, offset)
        stop = min(trace.size, offset + length)
        if stop <= start:
            continue
        # Build the whole artifact and normalize it against its own complete
        # shape, then take this block's slice. Normalizing a partial slice
        # would rescale each side of a block boundary differently.
        progress = np.arange(length, dtype=np.float64) / float(length)
        shape = np.sqrt(progress) * np.exp(-3.0 * progress)
        peak = float(np.max(shape))
        if peak <= 0.0:
            continue
        waveform = float(amplitudes[index]) * shape / peak
        trace[start:stop] += waveform[start - offset : stop - offset]
    return trace


def _validate_guided_continuous_demo_folder(
    folder: Path,
    *,
    total_samples: int,
    fs_hz: int,
) -> None:
    """Stream the written file once and check the continuous source contract."""
    source = folder / GUIDED_CONTINUOUS_DEMO_FILE_NAME
    if not source.is_file():
        raise RuntimeError("Generated continuous recording file is missing.")
    expected_step = 1.0 / float(fs_hz)
    with source.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        if tuple(header) != GUIDED_CONTINUOUS_DEMO_HEADERS:
            raise RuntimeError("Generated continuous header is invalid.")
        row_count = 0
        previous = None
        for line in handle:
            fields = line.rstrip("\r\n").split(",")
            if len(fields) != len(GUIDED_CONTINUOUS_DEMO_HEADERS):
                raise RuntimeError("Generated continuous row is malformed.")
            values = np.array([float(item) for item in fields], dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError("Generated continuous values are not finite.")
            timestamp = float(fields[0])
            if previous is None:
                if timestamp != 0.0:
                    raise RuntimeError("Generated continuous recording must start at 0.")
            else:
                step = timestamp - previous
                if step <= 0.0 or abs(step - expected_step) > 1e-6:
                    raise RuntimeError("Generated continuous timestamps are invalid.")
            previous = timestamp
            row_count += 1
    if row_count != total_samples:
        raise RuntimeError("Generated continuous recording is incomplete.")
    if previous is None or abs((previous + expected_step) - total_samples * expected_step) > 1e-6:
        raise RuntimeError("Generated continuous duration is invalid.")
    if not (folder / "README.md").is_file():
        raise RuntimeError("Generated README is missing.")


def generate_guided_continuous_demo(
    destination_parent: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    _duration_sec: float = GUIDED_CONTINUOUS_DEMO_DURATION_SEC,
) -> DemoGenerationResult:
    """Generate the one fixed end-user Guided continuous demo transactionally."""
    parent = Path(destination_parent).expanduser()
    final_folder = parent / GUIDED_CONTINUOUS_DEMO_FOLDER_NAME
    temporary_folder: Path | None = None
    started = time.perf_counter()
    fs_hz = GUIDED_CONTINUOUS_DEMO_FS_HZ
    try:
        parent.mkdir(parents=True, exist_ok=True)
        if final_folder.exists():
            raise FileExistsError(
                f"The demo folder already exists: {final_folder}. "
                "Choose another destination or remove the existing folder."
            )
        duration_sec = float(_duration_sec)
        total_samples = int(round(duration_sec * float(fs_hz)))
        block_samples = int(round(GUIDED_CONTINUOUS_DEMO_BLOCK_SEC * float(fs_hz)))
        block_count = max(1, -(-total_samples // block_samples))
        temporary_folder = Path(
            tempfile.mkdtemp(
                prefix=f".{GUIDED_CONTINUOUS_DEMO_FOLDER_NAME}.tmp-",
                dir=str(parent),
            )
        )
        rng = np.random.default_rng(GUIDED_DEMO_SEED)

        # Whole-recording components on coarse grids, so one uninterrupted
        # recording stays continuous across every written block.
        drift_points = max(4, int(round(duration_sec / 60.0)) + 1)
        drift_grid = _slow_variation(rng, drift_points, 1.0 / 60.0, 10800.0, 0.9)
        gain_grid = 1.0 + _slow_variation(rng, drift_points, 1.0 / 60.0, 21600.0, 0.02)
        drift_times = np.arange(drift_points, dtype=np.float64) * 60.0
        wander_points = max(4, int(round(duration_sec / 2.0)) + 1)
        wander_times = np.arange(wander_points, dtype=np.float64) * 2.0
        common_wander_grid = _slow_variation(rng, wander_points, 0.5, 20.0, 0.28)
        settle_depth = float(rng.uniform(1.8, 2.6))
        settle_tau = float(rng.uniform(4.0, 7.0)) * 3600.0
        artifact_schedule = _continuous_artifact_schedule(rng, duration_sec)
        artifact_reference_scale = float(rng.uniform(0.78, 1.06))

        roi_state = []
        for roi_index in range(2):
            roi_state.append(
                {
                    "events": _continuous_event_schedule(rng, duration_sec, roi_index),
                    "reference_level": 100.0 + roi_index * 7.0 + float(rng.normal(0.0, 0.9)),
                    "signal_level": 118.0 + roi_index * 8.5 + float(rng.normal(0.0, 1.1)),
                    "reference_gain": float(rng.uniform(0.82, 0.95)),
                    "signal_gain": float(rng.uniform(1.02, 1.18)),
                    "reference_wander": _slow_variation(
                        rng, wander_points, 0.5, 25.0, 0.07
                    ),
                    "signal_wander": _slow_variation(rng, wander_points, 0.5, 25.0, 0.08),
                }
            )

        source_path = temporary_folder / GUIDED_CONTINUOUS_DEMO_FILE_NAME
        with source_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(",".join(GUIDED_CONTINUOUS_DEMO_HEADERS) + "\n")
            for block_index in range(block_count):
                first = block_index * block_samples
                last = min(total_samples, first + block_samples)
                block_time = np.arange(first, last, dtype=np.float64) / float(fs_hz)
                settle = -settle_depth * (1.0 - np.exp(-block_time / settle_tau))
                drift = np.interp(block_time, drift_times, drift_grid)
                gain = np.interp(block_time, drift_times, gain_grid)
                wander = np.interp(block_time, wander_times, common_wander_grid)
                common = gain * (settle + drift + wander) + rng.normal(
                    0.0, 0.09, block_time.size
                )
                artifacts = _render_continuous_artifacts(
                    block_time, fs_hz, artifact_schedule
                )
                columns = [block_time]
                for state in roi_state:
                    events = _render_continuous_events(
                        block_time, fs_hz, state["events"]
                    )
                    reference = (
                        state["reference_level"]
                        + state["reference_gain"] * common
                        + artifact_reference_scale * artifacts
                        + _REFERENCE_EVENT_BLEED * events
                        + np.interp(block_time, wander_times, state["reference_wander"])
                        + rng.normal(0.0, 0.20, block_time.size)
                    )
                    signal = (
                        state["signal_level"]
                        + state["signal_gain"] * common
                        + artifacts
                        + events
                        + np.interp(block_time, wander_times, state["signal_wander"])
                        + rng.normal(0.0, 0.22, block_time.size)
                    )
                    columns.extend([signal, reference])
                values = np.column_stack(columns)
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        f"Generated values are not finite near block {block_index + 1}."
                    )
                np.savetxt(
                    handle,
                    values,
                    fmt="%.3f,%.6f,%.6f,%.6f,%.6f",
                    delimiter="",
                    comments="",
                )
                if progress is not None:
                    progress(block_index + 1, block_count)
        (temporary_folder / "README.md").write_text(
            guided_continuous_demo_readme_text(), encoding="utf-8"
        )
        _validate_guided_continuous_demo_folder(
            temporary_folder,
            total_samples=total_samples,
            fs_hz=fs_hz,
        )
        os.replace(temporary_folder, final_folder)
        temporary_folder = None
        elapsed = time.perf_counter() - started
        return DemoGenerationResult(
            success=True,
            demo_type=GUIDED_CONTINUOUS_DEMO_TYPE,
            input_dir=final_folder,
            config_path=final_folder / "README.md",
            format=GUIDED_CONTINUOUS_DEMO_FORMAT,
            sessions_per_hour=0,
            mode=DEMO_MODE,
            message=(
                "Synthetic Guided continuous demo created successfully.\n"
                f"Recording folder: {final_folder}\n"
                'Use this folder in the Guided "Select data" step.\n'
                f"Generation time: {elapsed:.1f} seconds."
            ),
        )
    except BaseException as exc:
        if temporary_folder is not None and temporary_folder.exists():
            shutil.rmtree(temporary_folder, ignore_errors=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return DemoGenerationResult(
            success=False,
            demo_type=GUIDED_CONTINUOUS_DEMO_TYPE,
            input_dir=final_folder,
            config_path=final_folder / "README.md",
            format=GUIDED_CONTINUOUS_DEMO_FORMAT,
            sessions_per_hour=0,
            mode=DEMO_MODE,
            message=str(exc),
        )


def _validate_guided_demo_folder(
    folder: Path,
    *,
    session_count: int,
    rows_per_session: int,
    fs_hz: int,
) -> None:
    csv_files = sorted(folder.glob("session_*.csv"))
    expected_names = [
        guided_demo_session_filename(index)
        for index in range(int(session_count))
    ]
    if [path.name for path in csv_files] != expected_names:
        raise RuntimeError("Generated session file set is incomplete.")
    expected_header = ",".join(GUIDED_DEMO_HEADERS)
    expected_last = (rows_per_session - 1) / float(fs_hz)
    for path in csv_files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\r\n")
            first = handle.readline().rstrip("\r\n")
            row_count = 1
            last = first
            first_values = np.fromstring(first, sep=",")
            previous_timestamp = (
                float(first_values[0]) if first_values.size else float("nan")
            )
            values_are_finite = (
                first_values.size == len(GUIDED_DEMO_HEADERS)
                and np.isfinite(first_values).all()
            )
            timestamps_increase = True
            for line in handle:
                last = line.rstrip("\r\n")
                row_count += 1
                current_values = np.fromstring(last, sep=",")
                if (
                    current_values.size != len(GUIDED_DEMO_HEADERS)
                    or not np.isfinite(current_values).all()
                ):
                    values_are_finite = False
                    continue
                current_timestamp = float(current_values[0])
                if current_timestamp <= previous_timestamp:
                    timestamps_increase = False
                previous_timestamp = current_timestamp
        if header != expected_header or row_count != rows_per_session:
            raise RuntimeError(f"Generated file failed validation: {path.name}")
        last_values = np.fromstring(last, sep=",")
        if (
            not values_are_finite
            or not timestamps_increase
            or first_values.size != len(GUIDED_DEMO_HEADERS)
            or last_values.size != len(GUIDED_DEMO_HEADERS)
            or not np.isfinite(last_values).all()
            or first_values[0] != 0.0
            or not np.isclose(last_values[0], expected_last)
        ):
            raise RuntimeError(f"Generated timestamps are invalid: {path.name}")
    truth_path = folder / GUIDED_DEMO_TONIC_TRUTH_FILENAME
    if not truth_path.is_file():
        raise RuntimeError("Generated tonic truth is missing.")
    try:
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Generated tonic truth is not valid JSON.") from exc
    records = truth.get("records") if isinstance(truth, dict) else None
    if (
        not isinstance(truth, dict)
        or truth.get("schema") != "guided_csv_demo_tonic_truth.v1"
        or not isinstance(records, list)
        or len(records) != int(session_count) * 2
    ):
        raise RuntimeError("Generated tonic truth is incomplete.")
    if not (folder / "README.md").is_file():
        raise RuntimeError("Generated README is missing.")


def generate_guided_csv_demo(
    destination_parent: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    _session_count: int = GUIDED_DEMO_SESSION_COUNT,
    _rows_per_session: int = GUIDED_DEMO_ROWS_PER_SESSION,
) -> DemoGenerationResult:
    """Generate the one fixed end-user Guided CSV demo transactionally."""
    parent = Path(destination_parent).expanduser()
    final_folder = parent / GUIDED_DEMO_FOLDER_NAME
    temporary_folder: Path | None = None
    started = time.perf_counter()
    try:
        parent.mkdir(parents=True, exist_ok=True)
        if final_folder.exists():
            raise FileExistsError(
                f"The demo folder already exists: {final_folder}. "
                "Choose another destination or remove the existing folder."
            )
        temporary_folder = Path(
            tempfile.mkdtemp(
                prefix=f".{GUIDED_DEMO_FOLDER_NAME}.tmp-",
                dir=str(parent),
            )
        )
        rng = np.random.default_rng(GUIDED_DEMO_SEED)
        for session_index in range(int(_session_count)):
            values = _guided_demo_session_arrays(
                session_index,
                rows_per_session=int(_rows_per_session),
                fs_hz=GUIDED_DEMO_FS_HZ,
                rng=rng,
            )
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Generated values are not finite for session {session_index + 1}."
                )
            np.savetxt(
                temporary_folder / guided_demo_session_filename(session_index),
                values,
                delimiter=",",
                header=",".join(GUIDED_DEMO_HEADERS),
                comments="",
                fmt="%.6f",
            )
            if progress is not None:
                progress(session_index + 1, int(_session_count))
        (temporary_folder / GUIDED_DEMO_TONIC_TRUTH_FILENAME).write_text(
            json.dumps(
                _guided_demo_tonic_truth_payload(int(_session_count)),
                indent=2,
                sort_keys=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary_folder / "README.md").write_text(
            guided_demo_readme_text(), encoding="utf-8"
        )
        _validate_guided_demo_folder(
            temporary_folder,
            session_count=int(_session_count),
            rows_per_session=int(_rows_per_session),
            fs_hz=GUIDED_DEMO_FS_HZ,
        )
        os.replace(temporary_folder, final_folder)
        temporary_folder = None
        elapsed = time.perf_counter() - started
        return DemoGenerationResult(
            success=True,
            demo_type=GUIDED_DEMO_TYPE,
            input_dir=final_folder,
            config_path=final_folder / "README.md",
            format=GUIDED_DEMO_FORMAT,
            sessions_per_hour=GUIDED_DEMO_SESSIONS_PER_HOUR,
            mode=DEMO_MODE,
            message=(
                "Synthetic Guided CSV demo created successfully.\n"
                f"Recording folder: {final_folder}\n"
                'Use this folder in the Guided "Select data" step.\n'
                f"Generation time: {elapsed:.1f} seconds."
            ),
        )
    except BaseException as exc:
        if temporary_folder is not None and temporary_folder.exists():
            shutil.rmtree(temporary_folder, ignore_errors=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return DemoGenerationResult(
            success=False,
            demo_type=GUIDED_DEMO_TYPE,
            input_dir=final_folder,
            config_path=final_folder / "README.md",
            format=GUIDED_DEMO_FORMAT,
            sessions_per_hour=GUIDED_DEMO_SESSIONS_PER_HOUR,
            mode=DEMO_MODE,
            message=str(exc),
        )


def _destination_non_empty(destination: Path) -> bool:
    return destination.exists() and any(destination.iterdir())


def _prepare_destination(destination: Path, *, overwrite: bool) -> None:
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Destination exists and is not empty: {destination}. "
                "Pass overwrite=True to replace it."
            )
        shutil.rmtree(destination)
    elif destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)


def _result(
    *,
    success: bool,
    demo_type: str,
    destination: Path,
    message: str,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> DemoGenerationResult:
    destination = Path(destination)
    return DemoGenerationResult(
        success=bool(success),
        demo_type=str(demo_type),
        input_dir=destination,
        config_path=destination / "tutorial_config.yaml",
        format=DEMO_FORMAT,
        sessions_per_hour=DEMO_SESSIONS_PER_HOUR,
        mode=DEMO_MODE,
        message=str(message),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def copy_fast_quickstart_demo(
    destination: Path,
    *,
    overwrite: bool = False,
) -> DemoGenerationResult:
    """Copy the committed bundled quickstart dataset to destination."""
    destination = Path(destination)
    try:
        if not FAST_SOURCE_DIR.exists():
            raise FileNotFoundError(f"Bundled demo dataset not found: {FAST_SOURCE_DIR}")
        _prepare_destination(destination, overwrite=overwrite)
        shutil.copytree(FAST_SOURCE_DIR, destination)
        return _result(
            success=True,
            demo_type=FAST_DEMO_TYPE,
            destination=destination,
            message=f"Copied fast quickstart demo to {destination}",
        )
    except Exception as exc:
        return _result(
            success=False,
            demo_type=FAST_DEMO_TYPE,
            destination=destination,
            message=str(exc),
        )


def long_duration_tutorial_config_text(*, recording_duration_min: float = 10.0) -> str:
    chunk_duration_sec = float(recording_duration_min) * 60.0
    return "\n".join(
        [
            f"chunk_duration_sec: {chunk_duration_sec:.1f}",
            "target_fs_hz: 10.0",
            "allow_partial_final_chunk: false",
            "rwd_time_col: TimeStamp",
            "uv_suffix: \"-410\"",
            "sig_suffix: \"-470\"",
            "baseline_method: uv_raw_percentile_session",
            "baseline_percentile: 10",
            "peak_threshold_method: mean_std",
            "peak_threshold_k: 2.5",
            "peak_min_distance_sec: 1.0",
            "peak_min_prominence_k: 2.0",
            "peak_min_width_sec: 0.3",
            "dynamic_fit_mode: robust_global_event_reject",
            "window_sec: 20.0",
            "step_sec: 5.0",
            "r_low: -1.0",
            "r_high: 1.0",
            "g_min: 0.0",
            "min_valid_windows: 1",
            "min_samples_per_window: 20",
            "lowpass_hz: 2.0",
            "qc_max_chunk_fail_fraction: 1.0",
            "",
        ]
    )


def write_long_duration_demo_config(
    destination: Path,
    *,
    recording_duration_min: float = 10.0,
) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "tutorial_config.yaml"
    config_path.write_text(
        long_duration_tutorial_config_text(recording_duration_min=recording_duration_min),
        encoding="utf-8",
    )
    return config_path


def build_long_duration_demo_command(
    destination: Path,
    *,
    total_days: float = 2.0,
    recording_duration_min: float = 10.0,
) -> list[str]:
    """Build the fixed curated long-duration intermittent demo command."""
    destination = Path(destination)
    config_path = destination / "tutorial_config.yaml"
    return [
        sys.executable,
        str(SYNTH_SCRIPT),
        "--out",
        str(destination),
        "--format",
        "rwd",
        "--config",
        str(config_path),
        "--preset",
        "biological_shared_nuisance",
        "--total-days",
        str(float(total_days)).rstrip("0").rstrip("."),
        "--recording-duration-min",
        str(float(recording_duration_min)).rstrip("0").rstrip("."),
        "--recordings-per-hour",
        "2",
        "--acquisition-mode",
        "intermittent",
        "--fs-hz",
        "10",
        "--n-rois",
        "2",
        "--start-iso",
        "2025-01-03T11:22:00",
        "--seed",
        "2026",
        "--phasic-min-events-per-chunk",
        "3",
        "--artifact-enable-motion",
        "--artifact-motion-min-per-day",
        "1",
        "--artifact-motion-rate-per-day",
        "20",
    ]


def run_long_duration_demo(
    destination: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    overwrite: bool = False,
    total_days: float = 2.0,
    recording_duration_min: float = 10.0,
) -> DemoGenerationResult:
    """Generate the curated long-duration intermittent demo dataset."""
    destination = Path(destination)
    stdout_path = destination / "synthetic_generation_stdout.txt"
    stderr_path = destination / "synthetic_generation_stderr.txt"
    try:
        _prepare_destination(destination, overwrite=overwrite)
        destination.mkdir(parents=True, exist_ok=True)
        write_long_duration_demo_config(
            destination,
            recording_duration_min=recording_duration_min,
        )
        cmd = build_long_duration_demo_command(
            destination,
            total_days=total_days,
            recording_duration_min=recording_duration_min,
        )
        completed = runner(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path.write_text(getattr(completed, "stdout", "") or "", encoding="utf-8")
        stderr_path.write_text(getattr(completed, "stderr", "") or "", encoding="utf-8")
        code = int(getattr(completed, "returncode", 1))
        if code != 0:
            return _result(
                success=False,
                demo_type=LONG_DEMO_TYPE,
                destination=destination,
                message=f"Synthetic generator failed with exit code {code}.",
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        return _result(
            success=True,
            demo_type=LONG_DEMO_TYPE,
            destination=destination,
            message=f"Generated long-duration intermittent demo at {destination}",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    except Exception as exc:
        try:
            destination.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
        except Exception:
            pass
        return _result(
            success=False,
            demo_type=LONG_DEMO_TYPE,
            destination=destination,
            message=str(exc),
            stdout_path=stdout_path if stdout_path.exists() else None,
            stderr_path=stderr_path if stderr_path.exists() else None,
        )


def command_to_text(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)
