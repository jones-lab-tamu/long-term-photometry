import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.adapters import plan_continuous_windows_for_source
from photometry_pipeline.pipeline import Pipeline


def _write_custom_tabular_csv(path: Path, duration_sec: float, fs_hz: float = 10.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    iso = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 2.0 + 0.9 * iso + 0.04 * np.sin(2.0 * np.pi * 0.08 * t + 0.2)
    df = pd.DataFrame(
        {
            "time_sec": t,
            "Region0_iso": iso,
            "Region0_sig": sig,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_rwd_csv(path: Path, duration_sec: float, fs_hz: float = 5.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    uv = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.01 * t)
    sig = 0.3 + 1.4 * uv + 0.03 * np.sin(2.0 * np.pi * 0.07 * t + 0.1)
    df = pd.DataFrame(
        {
            "TimeStamp": t,
            "Region0-410": uv,
            "Region0-470": sig,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _continuous_cfg() -> Config:
    cfg = Config()
    cfg.acquisition_mode = "continuous"
    cfg.continuous_window_sec = 600.0
    cfg.continuous_step_sec = 600.0
    cfg.allow_partial_final_window = False
    cfg.chunk_duration_sec = 600.0
    cfg.target_fs_hz = 10.0
    cfg.allow_partial_final_chunk = False
    return cfg


def test_continuous_window_planner_contract(tmp_path: Path):
    cfg = _continuous_cfg()
    csv_path = tmp_path / "source.csv"

    _write_custom_tabular_csv(csv_path, duration_sec=1200.0, fs_hz=10.0)
    w0 = plan_continuous_windows_for_source(str(csv_path), "custom_tabular", cfg)
    assert len(w0) == 2
    assert w0[0]["window_start_sec"] == pytest.approx(0.0)
    assert w0[0]["window_end_sec"] == pytest.approx(600.0)
    assert w0[1]["window_start_sec"] == pytest.approx(600.0)
    assert w0[1]["window_end_sec"] == pytest.approx(1200.0)

    _write_custom_tabular_csv(csv_path, duration_sec=1250.0, fs_hz=10.0)
    cfg.allow_partial_final_window = False
    w1 = plan_continuous_windows_for_source(str(csv_path), "custom_tabular", cfg)
    assert len(w1) == 2
    assert all(not bool(x["is_partial_final_window"]) for x in w1)

    cfg.allow_partial_final_window = True
    w2 = plan_continuous_windows_for_source(str(csv_path), "custom_tabular", cfg)
    assert len(w2) == 3
    assert bool(w2[-1]["is_partial_final_window"])
    assert w2[-1]["window_duration_sec"] == pytest.approx(50.0, abs=1e-6)

    _write_custom_tabular_csv(csv_path, duration_sec=300.0, fs_hz=10.0)
    cfg.allow_partial_final_window = False
    with pytest.raises(ValueError, match="shorter than configured continuous_window_sec"):
        plan_continuous_windows_for_source(str(csv_path), "custom_tabular", cfg)


def test_custom_tabular_continuous_pipeline_writes_multiple_chunks(tmp_path: Path):
    input_dir = tmp_path / "input_custom"
    output_dir = tmp_path / "out_custom"
    _write_custom_tabular_csv(input_dir / "session_000.csv", duration_sec=1200.0, fs_hz=10.0)

    cfg = _continuous_cfg()
    p = Pipeline(cfg, mode="phasic")
    p.run(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        force_format="custom_tabular",
        recursive=False,
        traces_only=True,
        sessions_per_hour=None,
    )

    cache_path = output_dir / "phasic_trace_cache.h5"
    assert cache_path.exists()
    with h5py.File(cache_path, "r") as h5:
        n_chunks = int(h5["meta"]["n_chunks"][0])
        assert n_chunks == 2
        g0 = h5["roi"]["Region0"]["chunk_0"]
        g1 = h5["roi"]["Region0"]["chunk_1"]
        assert g0.attrs["acquisition_mode"] == "continuous"
        assert g0.attrs["window_start_sec"] == pytest.approx(0.0)
        assert g1.attrs["window_start_sec"] == pytest.approx(600.0)
        assert g1.attrs["window_end_sec"] == pytest.approx(1200.0)
        total_cov = float(g0.attrs["window_duration_sec"]) + float(g1.attrs["window_duration_sec"])
        assert total_cov == pytest.approx(1200.0, abs=1e-3)


def test_rwd_continuous_pipeline_writes_multiple_chunks(tmp_path: Path):
    input_dir = tmp_path / "input_rwd"
    output_dir = tmp_path / "out_rwd"
    _write_rwd_csv(
        input_dir / "2025_01_01-00_00_00" / "fluorescence.csv",
        duration_sec=1200.0,
        fs_hz=5.0,
    )

    cfg = _continuous_cfg()
    cfg.target_fs_hz = 5.0
    p = Pipeline(cfg, mode="phasic")
    p.run(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        force_format="rwd",
        recursive=False,
        traces_only=True,
        sessions_per_hour=None,
    )

    cache_path = output_dir / "phasic_trace_cache.h5"
    assert cache_path.exists()
    with h5py.File(cache_path, "r") as h5:
        assert int(h5["meta"]["n_chunks"][0]) == 2
        g1 = h5["roi"]["Region0"]["chunk_1"]
        assert g1.attrs["source_file"].endswith("fluorescence.csv")
        assert g1.attrs["window_start_sec"] == pytest.approx(600.0)
        assert g1.attrs["window_end_sec"] == pytest.approx(1200.0)


def test_continuous_tonic_global_fit_uses_bounded_paired_sampler(monkeypatch):
    cfg = _continuous_cfg()
    cfg.seed = 123
    p = Pipeline(cfg, mode="tonic")
    p.file_list = [f"window_{i}" for i in range(3)]
    fit_calls = []

    def _fake_chunk(_entry: str, chunk_id: int, _force_format: str) -> Chunk:
        n = 100_000
        t = np.arange(n, dtype=float) / 10.0
        uv = 1.0 + 0.01 * chunk_id + 0.05 * np.sin(2.0 * np.pi * 0.03 * t)
        sig = 0.4 + 1.8 * uv + 0.02 * np.cos(2.0 * np.pi * 0.05 * t)
        return Chunk(
            chunk_id=chunk_id,
            source_file=f"window_{chunk_id}.csv",
            format="custom_tabular",
            time_sec=t,
            uv_raw=uv.reshape(-1, 1),
            sig_raw=sig.reshape(-1, 1),
            fs_hz=10.0,
            channel_names=["Region0"],
            metadata={
                "acquisition_mode": "continuous",
                "window_index": chunk_id,
                "window_start_sec": float(chunk_id * 600.0),
                "window_end_sec": float((chunk_id + 1) * 600.0),
                "window_duration_sec": 600.0,
                "original_file_duration_sec": 1800.0,
                "continuous_window_sec": 600.0,
                "continuous_step_sec": 600.0,
                "is_partial_final_window": False,
            },
        )

    def _fake_robust_fit(uv_sample, sig_sample):
        fit_calls.append((np.asarray(uv_sample), np.asarray(sig_sample)))
        assert len(uv_sample) <= 200_000
        assert len(sig_sample) <= 200_000
        assert len(uv_sample) == len(sig_sample)
        return 1.8, 0.4, True, len(uv_sample)

    monkeypatch.setattr(p, "_load_entry_chunk", _fake_chunk)
    monkeypatch.setattr(
        "photometry_pipeline.core.tonic_dff.compute_global_iso_fit_robust",
        _fake_robust_fit,
    )

    p.run_pass_1(force_format="custom_tabular")

    assert len(fit_calls) == 1
    assert len(fit_calls[0][0]) == 200_000
    assert "Region0" in p.stats.tonic_fit_params
    provenance = p.stats.tonic_global_fit_provenance["Region0"]
    assert provenance["tonic_global_fit_sampling_mode"] == "bounded_paired_reservoir"
    assert provenance["tonic_global_fit_sample_capacity"] == 200_000
    assert provenance["tonic_global_fit_seed"] == cfg.seed
    assert provenance["tonic_global_fit_samples_seen"] == 300_000
    assert provenance["tonic_global_fit_samples_used"] == 200_000
    assert provenance["tonic_global_fit_samples_seen"] > provenance["tonic_global_fit_samples_used"]
    assert np.isfinite(p.stats.tonic_fit_params["Region0"]["slope"])
    assert np.isfinite(p.stats.tonic_fit_params["Region0"]["intercept"])


def test_intermittent_tonic_global_fit_preserves_full_accumulation(monkeypatch):
    cfg = Config()
    cfg.seed = 123
    cfg.acquisition_mode = "intermittent"
    p = Pipeline(cfg, mode="tonic")
    p.file_list = [f"session_{i}" for i in range(3)]
    fit_calls = []

    def _fake_chunk(_entry: str, chunk_id: int, _force_format: str) -> Chunk:
        n = 500
        t = np.arange(n, dtype=float) / 10.0
        uv = 1.0 + 0.01 * chunk_id + 0.05 * np.sin(2.0 * np.pi * 0.03 * t)
        sig = 0.4 + 1.8 * uv + 0.02 * np.cos(2.0 * np.pi * 0.05 * t)
        return Chunk(
            chunk_id=chunk_id,
            source_file=f"session_{chunk_id}.csv",
            format="custom_tabular",
            time_sec=t,
            uv_raw=uv.reshape(-1, 1),
            sig_raw=sig.reshape(-1, 1),
            fs_hz=10.0,
            channel_names=["Region0"],
            metadata={},
        )

    def _fake_robust_fit(uv_full, sig_full):
        fit_calls.append((np.asarray(uv_full), np.asarray(sig_full)))
        assert len(uv_full) == 1500
        assert len(sig_full) == 1500
        return 1.8, 0.4, True, len(uv_full)

    monkeypatch.setattr(p, "_load_entry_chunk", _fake_chunk)
    monkeypatch.setattr(
        "photometry_pipeline.core.tonic_dff.compute_global_iso_fit_robust",
        _fake_robust_fit,
    )

    p.run_pass_1(force_format="custom_tabular")

    assert len(fit_calls) == 1
    provenance = p.stats.tonic_global_fit_provenance["Region0"]
    assert provenance["tonic_global_fit_sampling_mode"] == "full_accumulation"
    assert provenance["tonic_global_fit_samples_seen"] == 1500
    assert provenance["tonic_global_fit_samples_used"] == 1500
    assert "tonic_global_fit_sample_capacity" not in provenance
    assert "Region0" in p.stats.tonic_fit_params


def test_wrapper_continuous_skips_intermittent_outputs_and_succeeds(tmp_path: Path):
    input_dir = tmp_path / "input_wrap"
    out_dir = tmp_path / "out_wrap"
    cfg_path = tmp_path / "cfg.yaml"
    _write_custom_tabular_csv(input_dir / "session_000.csv", duration_sec=1200.0, fs_hz=10.0)
    cfg_path.write_text(
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
        "phasic",
        "--overwrite",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    assert (out_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").exists()
    assert not (out_dir / "day_plots").exists()

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest.get("acquisition_mode") == "continuous"
    assert "continuous_outputs" in manifest
    assert "intermittent_only_outputs_skipped" in manifest["continuous_outputs"]


def test_wrapper_continuous_npm_is_rejected(tmp_path: Path):
    input_dir = tmp_path / "input_npm"
    out_dir = tmp_path / "out_npm"
    cfg_path = tmp_path / "cfg.yaml"
    input_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "acquisition_mode: continuous\ncontinuous_window_sec: 600.0\ncontinuous_step_sec: 600.0\n",
        encoding="utf-8",
    )
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
        "npm",
        "--validate-only",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 1
    combined = f"{res.stdout}\n{res.stderr}"
    assert "Continuous acquisition mode is not yet implemented for NPM/interleaved inputs." in combined
