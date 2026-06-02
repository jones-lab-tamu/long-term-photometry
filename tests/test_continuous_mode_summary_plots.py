import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gui.run_report_parser import is_successful_completed_run_dir, resolve_region_deliverables
from photometry_pipeline.continuous_outputs import generate_continuous_summary_plots


def _write_phasic_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "roi": ["Region0", "Region0"],
            "source_file": ["source.csv", "source.csv"],
            "chunk_id": [0, 1],
            "window_index": [0, 1],
            "window_start_sec": [0.0, 600.0],
            "window_end_sec": [600.0, 1200.0],
            "window_duration_sec": [600.0, 600.0],
            "elapsed_hour_start": [0.0, 600.0 / 3600.0],
            "elapsed_hour_mid": [300.0 / 3600.0, 900.0 / 3600.0],
            "event_count": [3, 5],
            "event_rate_per_min": [0.3, 0.5],
            "event_rate_per_hour": [18.0, 30.0],
            "event_signal_auc": [1.25, 2.5],
            "event_signal_auc_semantics": [
                "aggregate finite-run AUC from feature_extraction output; not per-event AUC",
                "aggregate finite-run AUC from feature_extraction output; not per-event AUC",
            ],
            "event_signal_mean": [0.1, 0.2],
            "event_signal_median": [0.05, 0.07],
            "event_signal_std": [0.01, 0.02],
            "event_signal_mad": [0.005, 0.006],
            "is_partial_final_window": [False, False],
            "original_file_duration_sec": [1200.0, 1200.0],
            "continuous_window_sec": [600.0, 600.0],
            "continuous_step_sec": [600.0, 600.0],
            "acquisition_mode": ["continuous", "continuous"],
        }
    ).to_csv(path, index=False)


def _write_tonic_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "roi": ["Region0", "Region0"],
            "source_file": ["source.csv", "source.csv"],
            "chunk_id": [0, 1],
            "window_index": [0, 1],
            "window_start_sec": [0.0, 600.0],
            "window_end_sec": [600.0, 1200.0],
            "window_duration_sec": [600.0, 600.0],
            "elapsed_hour_start": [0.0, 600.0 / 3600.0],
            "elapsed_hour_mid": [300.0 / 3600.0, 900.0 / 3600.0],
            "tonic_mean": [0.1, 0.2],
            "tonic_median": [0.09, 0.18],
            "tonic_min": [0.0, 0.1],
            "tonic_max": [0.2, 0.3],
            "tonic_p05": [0.01, 0.11],
            "tonic_p95": [0.19, 0.29],
            "tonic_n_finite": [6000, 6000],
            "tonic_nan_fraction": [0.0, 0.0],
            "is_partial_final_window": [False, False],
            "original_file_duration_sec": [1200.0, 1200.0],
            "continuous_window_sec": [600.0, 600.0],
            "continuous_step_sec": [600.0, 600.0],
            "acquisition_mode": ["continuous", "continuous"],
        }
    ).to_csv(path, index=False)


def _write_custom_tabular_csv(path: Path, duration_sec: float, fs_hz: float = 10.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    iso = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 2.0 + 0.9 * iso + 0.04 * np.sin(2.0 * np.pi * 0.08 * t + 0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_sec": t, "Region0_iso": iso, "Region0_sig": sig}).to_csv(path, index=False)


def _write_continuous_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "target_fs_hz: 10.0",
                "chunk_duration_sec: 600.0",
                "allow_partial_final_chunk: false",
                "acquisition_mode: continuous",
                "continuous_window_sec: 600.0",
                "continuous_step_sec: 600.0",
                "allow_partial_final_window: false",
                "custom_tabular_time_col: time_sec",
                "custom_tabular_uv_suffix: _iso",
                "custom_tabular_sig_suffix: _sig",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_wrapper(tmp_path: Path, mode: str) -> Path:
    input_dir = tmp_path / f"input_{mode}"
    out_dir = tmp_path / f"out_{mode}"
    cfg_path = tmp_path / f"cfg_{mode}.yaml"
    _write_custom_tabular_csv(input_dir / "session_000.csv", duration_sec=1200.0, fs_hz=10.0)
    _write_continuous_config(cfg_path)
    cmd = [
        sys.executable,
        "tools/run_full_pipeline_deliverables.py",
        "--input",
        str(input_dir),
        "--out",
        str(out_dir),
        "--config",
        str(cfg_path),
        "--format",
        "custom_tabular",
        "--mode",
        mode,
        "--overwrite",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    return out_dir


def test_continuous_phasic_plots_are_generated_from_summary_csv(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_phasic_summary(run_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv")

    result = generate_continuous_summary_plots(str(run_dir), mode="phasic")

    rate = run_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png"
    count = run_dir / "Region0" / "summary" / "phasic_peak_count_timeseries.png"
    auc = run_dir / "Region0" / "summary" / "phasic_auc_timeseries.png"
    assert rate.exists()
    assert count.exists()
    assert auc.exists()
    assert "Region0/summary/phasic_peak_rate_timeseries.png" in result["summary_plots"]
    assert "Region0/summary/phasic_auc_timeseries.png" in result["summary_plots"]
    assert any(skip["reason"] == "tonic mode not requested" for skip in result["plot_skips"])


def test_continuous_phasic_plot_skip_does_not_create_empty_summary_folder(tmp_path: Path):
    run_dir = tmp_path / "run"
    table_path = run_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv"
    _write_phasic_summary(table_path)
    df = pd.read_csv(table_path)
    df["event_rate_per_min"] = np.nan
    df["event_count"] = np.nan
    df["event_signal_auc"] = np.nan
    df.to_csv(table_path, index=False)

    result = generate_continuous_summary_plots(str(run_dir), mode="phasic")

    summary_dir = run_dir / "Region0" / "summary"
    assert not summary_dir.exists()
    assert not (summary_dir / "phasic_peak_rate_timeseries.png").exists()
    assert not (summary_dir / "phasic_peak_count_timeseries.png").exists()
    assert not (summary_dir / "phasic_auc_timeseries.png").exists()
    assert result["summary_plots"] == []
    assert len(result["phasic"]["skipped_outputs"]) == 3


def test_continuous_tonic_plot_is_generated_from_summary_csv(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_tonic_summary(run_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv")

    result = generate_continuous_summary_plots(str(run_dir), mode="tonic")

    tonic = run_dir / "Region0" / "summary" / "tonic_overview.png"
    assert tonic.exists()
    assert "Region0/summary/tonic_overview.png" in result["summary_plots"]
    assert any(skip["reason"] == "phasic mode not requested" for skip in result["plot_skips"])


def test_wrapper_continuous_phasic_generates_summary_plots_and_manifest_entries(tmp_path: Path):
    out_dir = _run_wrapper(tmp_path, "phasic")

    assert (out_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv").exists()
    assert (out_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert (out_dir / "Region0" / "summary" / "phasic_auc_timeseries.png").exists()
    assert not (out_dir / "Region0" / "summary" / "tonic_overview.png").exists()

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    continuous_outputs = manifest["continuous_outputs"]
    assert continuous_outputs["summary_plots_generated"] is True
    assert "Region0/summary/phasic_peak_rate_timeseries.png" in continuous_outputs["summary_plots"]
    assert "Region0/summary/phasic_auc_timeseries.png" in continuous_outputs["summary_plots"]
    assert any(skip["reason"] == "tonic mode not requested" for skip in continuous_outputs["plot_skips"])
    assert continuous_outputs["intermittent_only_outputs_skipped"]

    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["continuous_outputs"]["summary_plots_generated"] is True
    assert "Region0/summary/phasic_auc_timeseries.png" in status["continuous_outputs"]["summary_plots"]

    ok, reason = is_successful_completed_run_dir(str(out_dir))
    assert ok, reason
    regions = resolve_region_deliverables(str(out_dir))
    assert len(regions) == 1
    assert any(label == "Summary" and status == "ok" for label, _path, status in regions[0]["subfolders"])
    assert any(label == "Tables" and status == "ok" for label, _path, status in regions[0]["subfolders"])


def test_wrapper_continuous_both_generates_phasic_and_tonic_plots(tmp_path: Path):
    out_dir = _run_wrapper(tmp_path, "both")

    assert (out_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv").exists()
    assert (out_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv").exists()
    assert (out_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert (out_dir / "Region0" / "summary" / "phasic_auc_timeseries.png").exists()
    assert (out_dir / "Region0" / "summary" / "tonic_overview.png").exists()

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    continuous_outputs = manifest["continuous_outputs"]
    assert "Region0/tables/continuous_tonic_window_summary.csv" in continuous_outputs["summary_tables"]
    assert "Region0/summary/tonic_overview.png" in continuous_outputs["summary_plots"]


def test_wrapper_continuous_tonic_only_skips_phasic_plots(tmp_path: Path):
    out_dir = _run_wrapper(tmp_path, "tonic")

    assert (out_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv").exists()
    assert (out_dir / "Region0" / "summary" / "tonic_overview.png").exists()
    assert not (out_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert not (out_dir / "Region0" / "summary" / "phasic_auc_timeseries.png").exists()

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    continuous_outputs = manifest["continuous_outputs"]
    assert any(skip["reason"] == "phasic mode not requested" for skip in continuous_outputs["plot_skips"])
