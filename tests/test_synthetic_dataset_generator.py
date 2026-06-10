import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH = REPO_ROOT / "tools" / "synth_photometry_dataset.py"


def _write_config(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "chunk_duration_sec: 60.0",
                "target_fs_hz: 5.0",
                "allow_partial_final_chunk: false",
                "rwd_time_col: TimeStamp",
                "uv_suffix: '-410'",
                "sig_suffix: '-470'",
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
                "min_samples_per_window: 10",
                "lowpass_hz: 1.0",
                "qc_max_chunk_fail_fraction: 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_synthetic_generator_creates_small_rwd_dataset(tmp_path):
    cfg = _write_config(tmp_path / "cfg.yaml")
    out_dir = tmp_path / "synthetic_rwd"
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
        "1",
        "--recordings-per-hour",
        "2",
        "--fs-hz",
        "5",
        "--n-rois",
        "1",
        "--start-iso",
        "2025-01-03T11:22:00",
        "--seed",
        "123",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(out_dir.glob("*/fluorescence.csv"))
    manifest_path = out_dir / "generation_manifest.yaml"
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "rwd"
    assert manifest["sessions_generated"] >= 1
    assert manifest["command"]["parsed_args"]["n_rois"] == 1

