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

The signals contain deterministic baseline variation, mild bleaching, noise,
and synthetic transient events for demonstrating the workflow.
Do not draw biological conclusions from this dataset.
"""


def _guided_demo_session_arrays(
    session_index: int,
    *,
    rows_per_session: int,
    fs_hz: int,
    session_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    time_sec = np.arange(rows_per_session, dtype=np.float64) / float(fs_hz)
    recording_phase = 2.0 * np.pi * float(session_index) / float(session_count)
    shared_noise = rng.normal(0.0, 0.075, rows_per_session)
    shared_wave = 0.28 * np.sin(2.0 * np.pi * time_sec / 47.0 + recording_phase)
    bleach = -0.85 * (1.0 - np.exp(-time_sec / 260.0))
    slow_recording = 1.2 * np.sin(recording_phase)

    columns = [time_sec]
    for roi_index in range(2):
        roi_phase = recording_phase + roi_index * 0.55
        reference_noise = rng.normal(0.0, 0.055 + roi_index * 0.005, rows_per_session)
        reference = (
            100.0
            + roi_index * 7.0
            + slow_recording * (0.65 + roi_index * 0.1)
            + bleach * (0.75 + roi_index * 0.08)
            + shared_wave
            + shared_noise
            + reference_noise
        )
        signal = (
            118.0
            + roi_index * 8.5
            + 1.08 * (reference - (100.0 + roi_index * 7.0))
            + 0.45 * np.sin(2.0 * np.pi * time_sec / 91.0 + roi_phase)
            + rng.normal(0.0, 0.08, rows_per_session)
        )
        event_centers = np.array([62.0, 174.0, 286.0, 407.0, 526.0])
        event_centers += ((session_index * 7 + roi_index * 11) % 17) - 8
        for event_number, center in enumerate(event_centers):
            amplitude = (
                3.2
                + 0.35 * roi_index
                + 0.25 * np.sin(recording_phase + event_number)
            )
            width_sec = 0.85 + 0.08 * ((event_number + roi_index) % 3)
            signal += amplitude * np.exp(
                -0.5 * ((time_sec - center) / width_sec) ** 2
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
