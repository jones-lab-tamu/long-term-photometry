import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from gui.run_report_parser import is_successful_completed_run_dir, resolve_region_deliverables
from photometry_pipeline.tuning.cache_correction_retune import run_cache_correction_retune
from photometry_pipeline.tuning.cache_downstream_retune import run_cache_downstream_retune


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH = REPO_ROOT / "tools" / "synth_photometry_dataset.py"
WRAPPER = REPO_ROOT / "tools" / "run_full_pipeline_deliverables.py"
WINDOW_SEC = 600.0


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)


def _write_continuous_config(path: Path) -> Path:
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


def _generate_continuous_dataset(
    tmp_path: Path,
    fmt: str,
    *,
    hours: float = 0.34,
    cfg: Path | None = None,
) -> tuple[Path, dict]:
    if cfg is None:
        cfg = _write_continuous_config(tmp_path / f"{fmt}_config.yaml")
    input_dir = tmp_path / f"{fmt}_input"
    cmd = [
        sys.executable,
        str(SYNTH),
        "--out",
        str(input_dir),
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
    res = _run(cmd)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    manifest = yaml.safe_load((input_dir / "generation_manifest.yaml").read_text(encoding="utf-8"))
    return input_dir, manifest


def _run_full_workflow(input_dir: Path, out_dir: Path, cfg: Path, fmt: str, *, mode: str) -> dict:
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
        "--mode",
        mode,
        "--acquisition-mode",
        "continuous",
        "--continuous-window-sec",
        str(int(WINDOW_SEC)),
        "--continuous-step-sec",
        str(int(WINDOW_SEC)),
        "--overwrite",
    ]
    res = _run(cmd)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    status = _load_json(out_dir / "status.json")
    assert status["status"] == "success"
    assert status["acquisition_mode"] == "continuous"
    return status


def _load_json(path: Path) -> dict:
    assert path.exists()
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events(path: Path) -> list[dict]:
    assert path.exists()
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_full_window_count(generation_manifest: dict) -> int:
    return int(generation_manifest["continuous_windows"]["expected_continuous_window_count"])


def _assert_region_deliverables(run_dir: Path, expected_roi: str | None = None) -> list[dict]:
    ok, reason = is_successful_completed_run_dir(str(run_dir))
    assert ok, reason
    regions = resolve_region_deliverables(str(run_dir))
    assert regions
    if expected_roi is not None:
        assert expected_roi in {r["name"] for r in regions}
    for region in regions:
        labels = {label for label, _path, status in region["subfolders"] if status == "ok"}
        assert "Summary" in labels
        assert "Tables" in labels
    return regions


def _assert_phasic_summary_sane(path: Path, expected_rows: int) -> pd.DataFrame:
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == expected_rows
    assert set(df["acquisition_mode"].astype(str)) == {"continuous"}
    assert np.all(np.diff(df["elapsed_hour_mid"].to_numpy(dtype=float)) > 0)
    assert int(df["event_count"].sum()) > 0
    assert np.all(np.isfinite(df["event_signal_auc"].to_numpy(dtype=float)))
    return df


def _assert_tonic_summary_sane(path: Path, expected_rows: int) -> pd.DataFrame:
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == expected_rows
    assert set(df["acquisition_mode"].astype(str)) == {"continuous"}
    assert np.all(np.diff(df["elapsed_hour_mid"].to_numpy(dtype=float)) > 0)
    assert np.all(np.isfinite(df["tonic_mean"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(df["tonic_median"].to_numpy(dtype=float)))
    return df


def _assert_continuous_both_outputs(run_dir: Path, roi: str, expected_rows: int) -> None:
    assert (run_dir / "MANIFEST.json").exists()
    assert (run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").exists()
    assert (run_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5").exists()
    _assert_phasic_summary_sane(
        run_dir / roi / "tables" / "continuous_phasic_window_summary.csv",
        expected_rows,
    )
    _assert_tonic_summary_sane(
        run_dir / roi / "tables" / "continuous_tonic_window_summary.csv",
        expected_rows,
    )
    assert (run_dir / roi / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert (run_dir / roi / "summary" / "phasic_peak_count_timeseries.png").exists()
    assert (run_dir / roi / "summary" / "phasic_auc_timeseries.png").exists()
    assert (run_dir / roi / "summary" / "tonic_overview.png").exists()
    assert (run_dir / roi / "summary" / "continuous_phasic_dff_trace_overview.png").exists()
    assert (run_dir / roi / "summary" / "continuous_tonic_trace_overview.png").exists()


def _hash_production_outputs(run_dir: Path, roi: str) -> dict[Path, str]:
    paths = [
        run_dir / "status.json",
        run_dir / "MANIFEST.json",
        run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5",
        run_dir / "_analysis" / "phasic_out" / "features" / "features.csv",
        run_dir / roi / "tables" / "continuous_phasic_window_summary.csv",
        run_dir / roi / "summary" / "phasic_peak_rate_timeseries.png",
        run_dir / roi / "summary" / "phasic_peak_count_timeseries.png",
        run_dir / roi / "summary" / "phasic_auc_timeseries.png",
        run_dir / roi / "summary" / "continuous_phasic_dff_trace_overview.png",
    ]
    return {path: _sha256(path) for path in paths}


def _assert_retuned_continuous_outputs(result: dict, roi: str) -> None:
    retuned = result["retuned_continuous_outputs"]
    assert retuned["generated"] is True
    assert retuned["continuous_detected"] is True
    artifacts = result["artifacts"]
    retune_dir = Path(result["retune_dir"])
    summary_path = Path(artifacts["retuned_continuous_phasic_summary_csv"])
    rate_path = Path(artifacts["retuned_continuous_phasic_peak_rate_png"])
    count_path = Path(artifacts["retuned_continuous_phasic_peak_count_png"])
    auc_path = Path(artifacts["retuned_continuous_phasic_auc_png"])
    assert summary_path == retune_dir / f"retuned_continuous_phasic_window_summary_{roi}.csv"
    assert rate_path == retune_dir / f"retuned_phasic_peak_rate_timeseries_{roi}.png"
    assert count_path == retune_dir / f"retuned_phasic_peak_count_timeseries_{roi}.png"
    assert auc_path == retune_dir / f"retuned_phasic_auc_timeseries_{roi}.png"
    assert summary_path.exists()
    assert rate_path.exists()
    assert count_path.exists()
    assert auc_path.exists()
    assert not (retune_dir / roi / "summary").exists()
    assert not (retune_dir / roi / "tables").exists()
    saved = _load_json(retune_dir / "retune_result.json")
    assert saved["retuned_continuous_outputs"]["generated"] is True
    assert Path(artifacts["retuned_features_csv"]).exists()


def test_generated_custom_tabular_continuous_mode_both_e2e(tmp_path: Path):
    input_dir, generation_manifest = _generate_continuous_dataset(tmp_path, "custom_tabular")
    cfg = tmp_path / "custom_tabular_config.yaml"
    out_dir = tmp_path / "custom_tabular_out"

    _run_full_workflow(input_dir, out_dir, cfg, "custom_tabular", mode="both")

    expected_rows = _expected_full_window_count(generation_manifest)
    _assert_region_deliverables(out_dir, expected_roi="Region0")
    _assert_continuous_both_outputs(out_dir, "Region0", expected_rows)

    manifest = _load_json(out_dir / "MANIFEST.json")
    continuous_outputs = manifest["continuous_outputs"]
    assert "Region0/tables/continuous_phasic_window_summary.csv" in continuous_outputs["summary_tables"]
    assert "Region0/tables/continuous_tonic_window_summary.csv" in continuous_outputs["summary_tables"]
    assert "Region0/summary/phasic_auc_timeseries.png" in continuous_outputs["summary_plots"]
    assert "Region0/summary/tonic_overview.png" in continuous_outputs["summary_plots"]
    assert "Region0/summary/continuous_phasic_dff_trace_overview.png" in continuous_outputs["trace_overview_plots"]
    assert "Region0/summary/continuous_tonic_trace_overview.png" in continuous_outputs["trace_overview_plots"]

    phasic_cfg = yaml.safe_load(
        (out_dir / "_analysis" / "phasic_out" / "config_used.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert phasic_cfg["dynamic_fit_mode"] == "robust_global_event_reject"
    assert phasic_cfg["peak_min_prominence_k"] == 2.0
    assert phasic_cfg["peak_min_width_sec"] == 0.3

    status = _load_json(out_dir / "status.json")
    timing = status["timing"]
    expected_timing_phases = {
        "validate",
        "tonic_analysis",
        "phasic_analysis",
        "continuous_summary_tables",
        "continuous_summary_plots",
        "continuous_trace_overview_plots",
        "manifest_write",
        "finalize_artifacts",
    }
    history_by_phase = {record["phase"]: record for record in timing["phase_history"]}
    assert expected_timing_phases.issubset(history_by_phase)
    assert expected_timing_phases.issubset(timing["phase_elapsed_sec"])
    for phase in expected_timing_phases:
        record = history_by_phase[phase]
        assert record["started_utc"]
        assert record["finished_utc"]
        assert record["elapsed_sec"] >= 0
        assert timing["phase_elapsed_sec"][phase] >= 0

    events = _load_events(out_dir / "events.ndjson")
    timing_events = [event for event in events if event.get("stage") == "timing"]
    timing_start_phases = {
        event.get("payload", {}).get("phase")
        for event in timing_events
        if event.get("type") == "timing_start"
    }
    timing_done_phases = {
        event.get("payload", {}).get("phase")
        for event in timing_events
        if event.get("type") == "timing_done"
    }
    assert expected_timing_phases.issubset(timing_start_phases)
    assert expected_timing_phases.issubset(timing_done_phases)


def test_generated_rwd_continuous_mode_both_e2e(tmp_path: Path):
    input_dir, generation_manifest = _generate_continuous_dataset(tmp_path, "rwd")
    cfg = tmp_path / "rwd_config.yaml"
    out_dir = tmp_path / "rwd_out"

    _run_full_workflow(input_dir, out_dir, cfg, "rwd", mode="both")

    regions = _assert_region_deliverables(out_dir)
    resolved_names = {region["name"] for region in regions}
    expected_names = set(generation_manifest["pipeline_channel_names"])
    assert resolved_names == expected_names

    expected_rows = _expected_full_window_count(generation_manifest)
    roi = sorted(resolved_names)[0]
    _assert_continuous_both_outputs(out_dir, roi, expected_rows)


def test_generated_custom_tabular_continuous_retune_smoke(tmp_path: Path):
    input_dir, generation_manifest = _generate_continuous_dataset(tmp_path, "custom_tabular")
    cfg = tmp_path / "custom_tabular_config.yaml"
    run_dir = tmp_path / "custom_tabular_phasic_out"
    roi = "Region0"

    _run_full_workflow(input_dir, run_dir, cfg, "custom_tabular", mode="phasic")

    expected_rows = _expected_full_window_count(generation_manifest)
    _assert_phasic_summary_sane(
        run_dir / roi / "tables" / "continuous_phasic_window_summary.csv",
        expected_rows,
    )
    before = _hash_production_outputs(run_dir, roi)

    downstream = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi=roi,
        chunk_id=0,
        overrides={"peak_threshold_k": 1.25},
    )
    _assert_retuned_continuous_outputs(downstream, roi)
    assert _hash_production_outputs(run_dir, roi) == before

    correction = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi=roi,
        chunk_id=0,
        overrides={"dynamic_fit_mode": "global_linear_regression"},
    )
    assert Path(correction["artifacts"]["retuned_correction_cache_h5"]).exists()
    _assert_retuned_continuous_outputs(correction, roi)
    assert _hash_production_outputs(run_dir, roi) == before


def test_wrapper_continuous_cli_overrides_propagate_to_analysis_subprocess(tmp_path: Path):
    cfg = REPO_ROOT / "tests" / "test_config.yaml"
    input_dir, _generation_manifest = _generate_continuous_dataset(
        tmp_path,
        "custom_tabular",
        hours=0.67,
        cfg=cfg,
    )
    run_dir = tmp_path / "manual_failure_regression_out"

    _run_full_workflow(input_dir, run_dir, cfg, "custom_tabular", mode="both")

    phasic_cache = run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    tonic_cache = run_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5"
    assert phasic_cache.exists()
    assert tonic_cache.exists()
    assert (run_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv").exists()
    assert (run_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv").exists()
    assert (run_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png").exists()
    assert (run_dir / "Region0" / "summary" / "phasic_auc_timeseries.png").exists()
    assert (run_dir / "Region0" / "summary" / "tonic_overview.png").exists()
    assert (run_dir / "Region0" / "summary" / "continuous_phasic_dff_trace_overview.png").exists()
    assert (run_dir / "Region0" / "summary" / "continuous_tonic_trace_overview.png").exists()

    status = _load_json(run_dir / "status.json")
    assert status["status"] == "success"
    phasic_summary = pd.read_csv(run_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv")
    tonic_summary = pd.read_csv(run_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv")
    assert len(phasic_summary) > 1
    assert len(tonic_summary) > 1

    with h5py.File(phasic_cache, "r") as h5:
        attrs = h5["roi"]["Region0"]["chunk_0"].attrs
        assert attrs["acquisition_mode"] == "continuous"
        assert int(attrs["window_index"]) == 0
        assert float(attrs["window_start_sec"]) == 0.0
        assert float(attrs["window_end_sec"]) == WINDOW_SEC
        assert float(attrs["window_duration_sec"]) == WINDOW_SEC

    phasic_cfg = yaml.safe_load(
        (run_dir / "_analysis" / "phasic_out" / "config_used.yaml").read_text(encoding="utf-8")
    )
    tonic_cfg = yaml.safe_load(
        (run_dir / "_analysis" / "tonic_out" / "config_used.yaml").read_text(encoding="utf-8")
    )
    assert phasic_cfg["acquisition_mode"] == "continuous"
    assert tonic_cfg["acquisition_mode"] == "continuous"
    assert float(phasic_cfg["continuous_window_sec"]) == WINDOW_SEC
    assert float(tonic_cfg["continuous_window_sec"]) == WINDOW_SEC
    tonic_report = _load_json(run_dir / "_analysis" / "tonic_out" / "run_report.json")
    tonic_fit_provenance = tonic_report["derived_settings"]["tonic_global_fit_provenance"]["Region0"]
    assert tonic_fit_provenance["tonic_global_fit_sampling_mode"] == "bounded_paired_reservoir"
    assert tonic_fit_provenance["tonic_global_fit_sample_capacity"] == 200000
    assert tonic_fit_provenance["tonic_global_fit_seed"] == tonic_cfg["seed"]
    assert tonic_fit_provenance["tonic_global_fit_samples_seen"] >= tonic_fit_provenance["tonic_global_fit_samples_used"]
    assert tonic_fit_provenance["channel"] == "Region0"

    manifest = _load_json(run_dir / "MANIFEST.json")
    command_text = "\n".join(" ".join(entry["cmd"]) for entry in manifest["commands"])
    assert "--acquisition-mode continuous" in command_text
    assert "--continuous-window-sec 600.0" in command_text
    assert "--continuous-step-sec 600.0" in command_text
    assert "--no-allow-partial-final-window" in command_text
