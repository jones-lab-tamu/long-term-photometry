import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from gui.run_spec import RunSpec
from photometry_pipeline.config import Config
from photometry_pipeline.discovery import discover_inputs
from photometry_pipeline.io.adapters import load_chunk


def _write_test_config(path: Path, *, chunk_duration_sec: int = 60, target_fs_hz: int = 5) -> None:
    path.write_text(
        "\n".join(
            [
                f"chunk_duration_sec: {chunk_duration_sec}",
                f"target_fs_hz: {target_fs_hz}",
                "baseline_method: uv_raw_percentile_session",
                "baseline_percentile: 10",
                "npm_time_axis: system_timestamp",
                "npm_frame_col: FrameCounter",
                "npm_system_ts_col: SystemTimestamp",
                "npm_computer_ts_col: ComputerTimestamp",
                "npm_led_col: LedState",
                "npm_region_prefix: Region",
                "npm_region_suffix: G",
                'rwd_time_col: "TimeStamp"',
                'uv_suffix: "-410"',
                'sig_suffix: "-470"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _expected_chunk_count(total_days: float, recordings_per_hour: int) -> int:
    return int(math.floor(total_days * 24.0 * recordings_per_hour))


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)


def _generate_vendor_npm_dataset(
    tmp_path: Path,
    *,
    total_days: float = 0.05,
    recording_duration_min: int = 1,
    recordings_per_hour: int = 2,
    fs_hz: int = 5,
    n_rois: int = 2,
    seed: int = 123,
) -> tuple[Path, Path]:
    out_dir = tmp_path / "npm_data"
    cfg_path = tmp_path / "cfg.yaml"
    _write_test_config(cfg_path, chunk_duration_sec=recording_duration_min * 60, target_fs_hz=fs_hz)

    cmd = [
        sys.executable,
        "tools/synth_photometry_dataset.py",
        "--out",
        str(out_dir),
        "--format",
        "npm",
        "--config",
        str(cfg_path),
        "--total-days",
        str(total_days),
        "--recording-duration-min",
        str(recording_duration_min),
        "--recordings-per-hour",
        str(recordings_per_hour),
        "--fs-hz",
        str(fs_hz),
        "--n-rois",
        str(n_rois),
        "--start-iso",
        "2025-03-05T15:37:44",
        "--seed",
        str(seed),
    ]
    res = _run(cmd, cwd=Path.cwd())
    assert res.returncode == 0, (
        "NPM generation failed\n"
        f"STDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}"
    )
    return out_dir, cfg_path


def test_legacy_generator_emits_flat_vendor_style_npm_sessions(tmp_path: Path):
    total_days = 0.05
    recordings_per_hour = 2
    out_dir, _cfg_path = _generate_vendor_npm_dataset(
        tmp_path,
        total_days=total_days,
        recordings_per_hour=recordings_per_hour,
        n_rois=2,
    )

    assert out_dir.is_dir()
    assert not any(p.is_dir() for p in out_dir.iterdir())

    files = sorted(out_dir.glob("*.csv"))
    assert len(files) == _expected_chunk_count(total_days, recordings_per_hour)

    pattern = re.compile(r"^photometryData\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.csv$")
    for p in files:
        assert pattern.match(p.name), p.name


def test_legacy_generator_emits_required_npm_columns_and_ledstate_layout(tmp_path: Path):
    fs_hz = 5
    recording_duration_min = 1
    n_rois = 3
    out_dir, _cfg_path = _generate_vendor_npm_dataset(
        tmp_path,
        fs_hz=fs_hz,
        recording_duration_min=recording_duration_min,
        n_rois=n_rois,
    )

    first_csv = sorted(out_dir.glob("*.csv"))[0]
    df = pd.read_csv(first_csv)

    expected_cols = [
        "FrameCounter",
        "SystemTimestamp",
        "LedState",
        "ComputerTimestamp",
        "Region0G",
        "Region1G",
        "Region2G",
    ]
    assert list(df.columns) == expected_cols

    expected_rows = 2 * fs_hz * (recording_duration_min * 60)
    assert len(df) == expected_rows

    leds = df["LedState"].to_numpy()
    assert set(leds.tolist()) == {1, 2}
    assert (leds[0::2] == 1).all()
    assert (leds[1::2] == 2).all()


def test_generated_vendor_npm_is_discoverable_and_loadable_via_shared_paths(tmp_path: Path):
    fs_hz = 5
    recording_duration_min = 1
    n_rois = 2
    out_dir, cfg_path = _generate_vendor_npm_dataset(
        tmp_path,
        fs_hz=fs_hz,
        recording_duration_min=recording_duration_min,
        n_rois=n_rois,
    )

    cfg = Config.from_yaml(str(cfg_path))
    auto = discover_inputs(str(out_dir), cfg, force_format="auto")
    forced = discover_inputs(str(out_dir), cfg, force_format="npm")

    assert auto["resolved_format"] == "NPM"
    assert forced["resolved_format"] == "NPM"
    assert auto["n_total_discovered"] == forced["n_total_discovered"] > 0

    chunk = load_chunk(forced["sessions"][0]["path"], "npm", cfg, chunk_id=0)
    expected_samples = fs_hz * (recording_duration_min * 60)
    assert chunk.uv_raw.shape == (expected_samples, n_rois)
    assert chunk.sig_raw.shape == (expected_samples, n_rois)
    assert chunk.channel_names == ["Region0", "Region1"]


def test_generated_vendor_npm_runs_cli_and_gui_adjacent_contract(tmp_path: Path):
    out_dir, cfg_path = _generate_vendor_npm_dataset(
        tmp_path,
        total_days=0.06,
        recordings_per_hour=2,
        fs_hz=5,
        recording_duration_min=1,
        n_rois=2,
        seed=77,
    )

    analyze_out = tmp_path / "analyze_out"
    analyze_cmd = [
        sys.executable,
        "analyze_photometry.py",
        "--input",
        str(out_dir),
        "--config",
        str(cfg_path),
        "--out",
        str(analyze_out),
        "--format",
        "npm",
        "--mode",
        "phasic",
        "--overwrite",
    ]
    res = _run(analyze_cmd, cwd=Path.cwd())
    assert res.returncode == 0, (
        "CLI analysis failed\n"
        f"STDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}"
    )
    assert (analyze_out / "run_report.json").is_file()
    assert (analyze_out / "features" / "features.csv").is_file()

    cfg = Config.from_yaml(str(cfg_path))
    discovered = discover_inputs(str(out_dir), cfg, force_format="npm")
    assert discovered["resolved_format"] == "NPM"
    assert discovered["n_total_discovered"] > 0

    gui_run_dir = tmp_path / "gui_run"
    spec = RunSpec(
        input_dir=str(out_dir),
        run_dir=str(gui_run_dir),
        format="npm",
        config_source_path=str(cfg_path),
    )
    effective_cfg = spec.generate_derived_config(str(gui_run_dir))
    RunSpec.validate_effective_config(effective_cfg)
    argv = spec.build_runner_argv()
    assert "--format" in argv
    assert argv[argv.index("--format") + 1] == "npm"
