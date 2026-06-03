import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gui.run_report_parser import is_successful_completed_run_dir, resolve_region_deliverables


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH = REPO_ROOT / "tools" / "synth_photometry_dataset.py"
WRAPPER = REPO_ROOT / "tools" / "run_full_pipeline_deliverables.py"


def _write_continuous_test_config(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "chunk_duration_sec: 600.0",
                "target_fs_hz: 10.0",
                "allow_partial_final_chunk: false",
                "acquisition_mode: continuous",
                "continuous_window_sec: 600.0",
                "continuous_step_sec: 600.0",
                "allow_partial_final_window: false",
                "custom_tabular_time_col: time_sec",
                "custom_tabular_uv_suffix: _iso",
                "custom_tabular_sig_suffix: _sig",
                "rwd_time_col: TimeStamp",
                "uv_suffix: '-410'",
                "sig_suffix: '-470'",
                "baseline_method: uv_raw_percentile_session",
                "baseline_percentile: 10",
                "peak_threshold_method: mean_std",
                "peak_threshold_k: 1.0",
                "peak_min_distance_sec: 5.0",
                "window_sec: 20.0",
                "step_sec: 5.0",
                "r_low: -1.0",
                "r_high: 1.0",
                "g_min: 0.0",
                "min_valid_windows: 1",
                "min_samples_per_window: 20",
                "lowpass_hz: 2.0",
                "qc_max_chunk_fail_fraction: 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)


def _synth_cmd(out_dir: Path, cfg: Path, fmt: str, *, hours: float = 0.67) -> list[str]:
    return [
        sys.executable,
        str(SYNTH),
        "--out",
        str(out_dir),
        "--format",
        fmt,
        "--config",
        str(cfg),
        "--acquisition-mode",
        "continuous",
        "--preset",
        "continuous_realistic",
        "--continuous-duration-hours",
        str(hours),
        "--fs-hz",
        "10",
        "--n-rois",
        "2",
        "--start-iso",
        "2025-01-01T13:37:11",
        "--seed",
        "2026",
    ]


def _load_manifest(out_dir: Path) -> dict:
    path = out_dir / "generation_manifest.yaml"
    assert path.exists()
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_custom_tabular_continuous_generator_creates_one_csv_and_manifest(tmp_path: Path):
    cfg = _write_continuous_test_config(tmp_path / "cfg.yaml")
    out_dir = tmp_path / "custom_continuous"

    res = _run(_synth_cmd(out_dir, cfg, "custom_tabular", hours=0.67))
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"

    csv_path = out_dir / "continuous_recording.csv"
    assert csv_path.exists()
    assert not [p for p in out_dir.iterdir() if p.is_dir()]
    df = pd.read_csv(csv_path)
    expected_cols = ["time_sec", "Region0_iso", "Region0_sig", "Region1_iso", "Region1_sig"]
    assert list(df.columns) == expected_cols
    assert np.all(np.diff(df["time_sec"].to_numpy(dtype=float)) > 0)
    values = df.drop(columns=["time_sec"]).to_numpy(dtype=float)
    assert np.all(np.isfinite(values))

    manifest = _load_manifest(out_dir)
    assert manifest["acquisition_mode"] == "continuous"
    assert manifest["format"] == "custom_tabular"
    assert manifest["preset"] == "continuous_realistic"
    assert manifest["n_rois"] == 2
    assert manifest["n_samples"] == len(df)
    assert manifest["output_files"] == ["continuous_recording.csv"]
    assert manifest["pipeline_channel_names"] == ["Region0", "Region1"]
    assert manifest["channel_columns"]["per_roi"]["Region0"]["pipeline_channel_name"] == "Region0"
    assert manifest["continuous_windows"]["expected_continuous_window_count"] >= 2


def test_rwd_continuous_generator_creates_one_fluorescence_file_and_manifest(tmp_path: Path):
    cfg = _write_continuous_test_config(tmp_path / "cfg.yaml")
    out_dir = tmp_path / "rwd_continuous"

    res = _run(_synth_cmd(out_dir, cfg, "rwd", hours=0.67))
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"

    subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    csv_path = subdirs[0] / "fluorescence.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path, header=1)
    expected_cols = ["TimeStamp", "Events", "CH1-410", "CH1-470", "CH2-410", "CH2-470"]
    assert list(df.columns) == expected_cols
    assert np.all(np.diff(df["TimeStamp"].to_numpy(dtype=float)) > 0)
    values = df.drop(columns=["Events"]).to_numpy(dtype=float)
    assert np.all(np.isfinite(values))

    manifest = _load_manifest(out_dir)
    assert manifest["acquisition_mode"] == "continuous"
    assert manifest["format"] == "rwd"
    assert manifest["n_rois"] == 2
    assert manifest["n_samples"] == len(df)
    assert manifest["output_files"] == [f"{subdirs[0].name}/fluorescence.csv"]
    assert manifest["roi_names"] == ["Region0", "Region1"]
    assert manifest["pipeline_channel_names"] == ["CH1", "CH2"]
    assert manifest["channel_columns"]["per_roi"]["Region0"] == {
        "pipeline_channel_name": "CH1",
        "uv": "CH1-410",
        "sig": "CH1-470",
    }
    assert manifest["channel_columns"]["per_roi"]["Region1"] == {
        "pipeline_channel_name": "CH2",
        "uv": "CH2-410",
        "sig": "CH2-470",
    }


def test_npm_continuous_generation_fails_clearly_without_outputs(tmp_path: Path):
    cfg = _write_continuous_test_config(tmp_path / "cfg.yaml")
    out_dir = tmp_path / "npm_continuous"

    res = _run(_synth_cmd(out_dir, cfg, "npm", hours=0.1))

    assert res.returncode != 0
    msg = f"{res.stdout}\n{res.stderr}"
    assert "continuous" in msg
    assert "NPM" in msg
    assert "unsupported" in msg or "not supported" in msg
    assert not out_dir.exists()


def test_custom_tabular_intermittent_generation_fails_clearly(tmp_path: Path):
    cfg = _write_continuous_test_config(tmp_path / "cfg.yaml")
    out_dir = tmp_path / "custom_tabular_intermittent"
    cmd = [
        sys.executable,
        str(SYNTH),
        "--out",
        str(out_dir),
        "--format",
        "custom_tabular",
        "--config",
        str(cfg),
        "--acquisition-mode",
        "intermittent",
        "--preset",
        "biological_shared_nuisance",
        "--total-days",
        "0.05",
        "--recording-duration-min",
        "10",
        "--recordings-per-hour",
        "1",
        "--fs-hz",
        "10",
        "--n-rois",
        "1",
        "--seed",
        "8",
    ]

    res = _run(cmd)

    assert res.returncode != 0
    msg = f"{res.stdout}\n{res.stderr}"
    assert "custom_tabular" in msg
    assert "intermittent" in msg
    assert "continuous" in msg
    assert not out_dir.exists()


def test_existing_intermittent_rwd_command_still_generates_session_layout(tmp_path: Path):
    cfg = _write_continuous_test_config(tmp_path / "cfg.yaml")
    out_dir = tmp_path / "rwd_intermittent"
    cmd = [
        sys.executable,
        str(SYNTH),
        "--out",
        str(out_dir),
        "--format",
        "rwd",
        "--config",
        str(cfg),
        "--preset",
        "biological_shared_nuisance",
        "--total-days",
        "0.05",
        "--recording-duration-min",
        "10",
        "--recordings-per-hour",
        "1",
        "--fs-hz",
        "10",
        "--n-rois",
        "1",
        "--seed",
        "7",
    ]

    res = _run(cmd)

    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert subdirs
    assert all((p / "fluorescence.csv").exists() for p in subdirs)
    manifest = _load_manifest(out_dir)
    assert manifest["format"] == "rwd"
    assert manifest["sessions_requested"] >= 1


def _validate_continuous_dataset(input_dir: Path, out_dir: Path, cfg: Path, fmt: str) -> dict:
    cmd = [
        sys.executable,
        str(WRAPPER),
        "--input",
        str(input_dir),
        "--out",
        str(out_dir),
        "--config",
        str(cfg),
        "--format",
        fmt,
        "--acquisition-mode",
        "continuous",
        "--continuous-window-sec",
        "600",
        "--continuous-step-sec",
        "600",
        "--validate-only",
        "--overwrite",
    ]
    res = _run(cmd)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert status["acquisition_mode"] == "continuous"
    assert "VALIDATE-ONLY: continuous plan" in res.stdout
    assert "planned_windows=" in res.stdout
    assert not (out_dir / "_analysis").exists()
    return status


def test_generated_rwd_continuous_full_phasic_run_succeeds(tmp_path: Path):
    cfg = _write_continuous_test_config(tmp_path / "cfg.yaml")
    input_dir = tmp_path / "rwd_full_input"
    res = _run(_synth_cmd(input_dir, cfg, "rwd", hours=0.34))
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    out_dir = tmp_path / "rwd_full_out"
    cmd = [
        sys.executable,
        str(WRAPPER),
        "--input",
        str(input_dir),
        "--out",
        str(out_dir),
        "--config",
        str(cfg),
        "--format",
        "rwd",
        "--mode",
        "phasic",
        "--acquisition-mode",
        "continuous",
        "--continuous-window-sec",
        "600",
        "--continuous-step-sec",
        "600",
        "--overwrite",
    ]

    res = _run(cmd)

    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert (out_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").exists()
    ok, reason = is_successful_completed_run_dir(str(out_dir))
    assert ok, reason

    regions = resolve_region_deliverables(str(out_dir))
    region_names = {r["name"] for r in regions}
    assert region_names == {"CH1", "CH2"}
    roi_dir = out_dir / sorted(region_names)[0]
    assert (roi_dir / "tables" / "continuous_phasic_window_summary.csv").exists()
    assert (roi_dir / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert (roi_dir / "summary" / "phasic_auc_timeseries.png").exists()
    assert (roi_dir / "summary" / "continuous_phasic_dff_trace_overview.png").exists()

    manifest = _load_manifest(input_dir)
    assert manifest["pipeline_channel_names"] == ["CH1", "CH2"]


def test_generated_custom_tabular_continuous_validate_only_succeeds(tmp_path: Path):
    cfg = _write_continuous_test_config(tmp_path / "cfg.yaml")
    input_dir = tmp_path / "custom_input"
    res = _run(_synth_cmd(input_dir, cfg, "custom_tabular", hours=0.67))
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"

    status = _validate_continuous_dataset(input_dir, tmp_path / "custom_validate", cfg, "custom_tabular")

    assert status["format"] == "custom_tabular"


def test_generated_rwd_continuous_validate_only_succeeds(tmp_path: Path):
    cfg = _write_continuous_test_config(tmp_path / "cfg.yaml")
    input_dir = tmp_path / "rwd_input"
    res = _run(_synth_cmd(input_dir, cfg, "rwd", hours=0.67))
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"

    status = _validate_continuous_dataset(input_dir, tmp_path / "rwd_validate", cfg, "rwd")

    assert status["format"] == "rwd"


def test_generated_custom_tabular_continuous_full_phasic_run_succeeds(tmp_path: Path):
    cfg = _write_continuous_test_config(tmp_path / "cfg.yaml")
    input_dir = tmp_path / "custom_full_input"
    res = _run(_synth_cmd(input_dir, cfg, "custom_tabular", hours=0.34))
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    out_dir = tmp_path / "custom_full_out"
    cmd = [
        sys.executable,
        str(WRAPPER),
        "--input",
        str(input_dir),
        "--out",
        str(out_dir),
        "--config",
        str(cfg),
        "--format",
        "custom_tabular",
        "--mode",
        "phasic",
        "--acquisition-mode",
        "continuous",
        "--continuous-window-sec",
        "600",
        "--continuous-step-sec",
        "600",
        "--overwrite",
    ]

    res = _run(cmd)

    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    assert (out_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").exists()
    assert (out_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv").exists()
    assert (out_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert (out_dir / "Region0" / "summary" / "phasic_auc_timeseries.png").exists()
    assert (out_dir / "Region0" / "summary" / "continuous_phasic_dff_trace_overview.png").exists()
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
