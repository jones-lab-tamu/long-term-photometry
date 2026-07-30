"""Curated synthetic demo dataset generation helpers for the GUI and examples."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
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
GUIDED_DEMO_SESSION_COUNT = 48
GUIDED_DEMO_SESSIONS_PER_HOUR = 2
GUIDED_DEMO_SESSION_DURATION_SEC = 600.0
GUIDED_DEMO_FS_HZ = 20
GUIDED_DEMO_ROWS_PER_SESSION = 12000
GUIDED_DEMO_SEED = 2026
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
_EVENT_MEDIAN_AMPLITUDE = (1.9, 2.15)
_EVENT_RISE_SEC = (0.08, 0.26)
_EVENT_DECAY_SEC = (0.45, 2.4)
_EVENT_TAIL_GUARD_SEC = 2.0  # keep transients from being cut off at session end
_REFERENCE_EVENT_BLEED = 0.05  # small event leak into the reference channel


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

- Source type: CSV files, one file per session
- Acquisition mode: intermittent
- Sessions per hour: 2
- Session duration: 600 seconds
- Time column: `ElapsedSeconds`
- Time unit: seconds
- Confirm the displayed natural filename order
- ROI1 mapping: `ROI1_Signal` with `ROI1_Reference`
- ROI2 mapping: `ROI2_Signal` with `ROI2_Reference`

Recommended timeline display:
- Fixed daily anchor
- Start of plotted day: `07:00`
- Clock time at recording start: `12:00:00`

The signals contain deterministic bleaching, shared optical variation, noise,
occasional common-mode artifacts, and irregular calcium-like transients for
demonstrating the workflow.
Do not draw biological conclusions from this dataset.
"""


def _slow_variation(
    rng: np.random.Generator,
    samples: int,
    fs_hz: int,
    tau_sec: float,
    std: float,
) -> np.ndarray:
    """Smoothly wandering zero-mean variation, from smoothed white noise."""
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
    session_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    time_sec = np.arange(rows_per_session, dtype=np.float64) / float(fs_hz)
    duration_sec = float(rows_per_session) / float(fs_hz)
    recording_phase = 2.0 * np.pi * float(session_index) / float(session_count)
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
        activity_phase = recording_phase - roi_index * _EVENT_ROI_PHASE_OFFSET
        rate_per_min = (
            _EVENT_RATE_PER_MIN[roi_index]
            * (1.0 + _EVENT_RATE_MODULATION[roi_index] * np.sin(activity_phase))
            * float(np.exp(rng.normal(0.0, 0.16)))
        )
        amplitude_scale = 1.0 + _EVENT_AMPLITUDE_MODULATION[roi_index] * np.sin(
            activity_phase
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
        columns.extend([signal, reference])
    return np.column_stack(columns)


def _validate_guided_demo_folder(
    folder: Path,
    *,
    session_count: int,
    rows_per_session: int,
    fs_hz: int,
) -> None:
    csv_files = sorted(folder.glob("session_*.csv"))
    expected_names = [
        f"session_{index:04d}.csv" for index in range(1, session_count + 1)
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
                session_count=int(_session_count),
                rng=rng,
            )
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Generated values are not finite for session {session_index + 1}."
                )
            np.savetxt(
                temporary_folder / f"session_{session_index + 1:04d}.csv",
                values,
                delimiter=",",
                header=",".join(GUIDED_DEMO_HEADERS),
                comments="",
                fmt="%.6f",
            )
            if progress is not None:
                progress(session_index + 1, int(_session_count))
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
