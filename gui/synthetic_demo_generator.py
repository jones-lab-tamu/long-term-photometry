"""Curated synthetic demo dataset generation helpers for the GUI and examples."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


DEMO_FORMAT = "rwd"
DEMO_SESSIONS_PER_HOUR = 2
DEMO_MODE = "both"
FAST_DEMO_TYPE = "Fast quickstart demo"
LONG_DEMO_TYPE = "Long-duration intermittent demo"

REPO_ROOT = Path(__file__).resolve().parents[1]
FAST_SOURCE_DIR = REPO_ROOT / "examples" / "data" / "synthetic_photometry_basic"
SYNTH_SCRIPT = REPO_ROOT / "tools" / "synth_photometry_dataset.py"


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
