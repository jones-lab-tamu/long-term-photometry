import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from gui.run_report_parser import (
    is_successful_completed_run_dir,
    parse_run_report,
    resolve_internal_artifacts,
    resolve_primary_artifacts,
    resolve_region_deliverables,
)


def _write_custom_tabular_csv(path: Path, duration_sec: float, fs_hz: float = 10.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    iso = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 2.0 + 0.9 * iso + 0.04 * np.sin(2.0 * np.pi * 0.08 * t + 0.2)
    df = pd.DataFrame({"time_sec": t, "Region0_iso": iso, "Region0_sig": sig})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_rwd_csv(path: Path, duration_sec: float, fs_hz: float = 5.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    uv = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.01 * t)
    sig = 0.3 + 1.4 * uv + 0.03 * np.sin(2.0 * np.pi * 0.07 * t + 0.1)
    df = pd.DataFrame({"TimeStamp": t, "Region0-410": uv, "Region0-470": sig})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_continuous_config(path: Path, *, target_fs_hz: float, custom_tabular: bool) -> None:
    lines = [
        f"target_fs_hz: {target_fs_hz}",
        "chunk_duration_sec: 600.0",
        "allow_partial_final_chunk: false",
        "acquisition_mode: continuous",
        "continuous_window_sec: 600.0",
        "continuous_step_sec: 600.0",
        "allow_partial_final_window: false",
    ]
    if custom_tabular:
        lines.extend(
            [
                "custom_tabular_time_col: time_sec",
                "custom_tabular_uv_suffix: _iso",
                "custom_tabular_sig_suffix: _sig",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_wrapper(input_dir: Path, out_dir: Path, cfg_path: Path, fmt: str) -> subprocess.CompletedProcess:
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
        fmt,
        "--mode",
        "phasic",
        "--overwrite",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _assert_continuous_completed_run_contract(out_dir: Path) -> None:
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["schema_version"] == 1
    assert status["phase"] == "final"
    assert status["status"] == "success"
    assert status["acquisition_mode"] == "continuous"

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["acquisition_mode"] == "continuous"
    assert manifest["continuous_plan"]["planned_window_count"] >= 2
    skipped = manifest["continuous_outputs"]["intermittent_only_outputs_skipped"]
    assert isinstance(skipped, list)
    assert skipped

    ok, reason = is_successful_completed_run_dir(str(out_dir))
    assert ok, reason
    report, err = parse_run_report(str(out_dir / "run_report.json"))
    assert err is None
    regions = resolve_region_deliverables(str(out_dir))
    assert len(regions) == 1
    assert regions[0]["name"] == "Region0"
    assert any(label == "Summary" and status == "ok" for label, _path, status in regions[0]["subfolders"])
    assert any(label == "Tables" and status == "ok" for label, _path, status in regions[0]["subfolders"])

    internal = resolve_internal_artifacts(str(out_dir))
    assert any(label == "Phasic Analysis (Internal)" and status == "ok" for label, _path, status in internal)
    primary = resolve_primary_artifacts(str(out_dir), report)
    assert any(label == "Run Status" and status == "ok" for label, _path, status in primary)
    assert any(label == "Output Manifest" and status == "ok" for label, _path, status in primary)


def _assert_continuous_cache_window_contract(cache_path: Path) -> None:
    assert cache_path.exists()
    with h5py.File(cache_path, "r") as h5:
        assert int(h5["meta"]["n_chunks"][0]) >= 2
        roi = h5["roi"]["Region0"]
        g0 = roi["chunk_0"]
        g1 = roi["chunk_1"]
        assert g0.attrs["acquisition_mode"] == "continuous"
        assert g0.attrs["source_file"] == g1.attrs["source_file"]
        assert int(g0.attrs["window_index"]) == 0
        assert int(g1.attrs["window_index"]) == 1
        assert g0.attrs["window_start_sec"] == pytest.approx(0.0)
        assert g0.attrs["window_end_sec"] == pytest.approx(600.0)
        assert g1.attrs["window_start_sec"] == pytest.approx(600.0)
        assert g1.attrs["window_end_sec"] == pytest.approx(1200.0)
        assert g0.attrs["window_duration_sec"] == pytest.approx(600.0)
        assert g1.attrs["window_duration_sec"] == pytest.approx(600.0)
        assert g0.attrs["original_file_duration_sec"] == pytest.approx(1200.0, abs=0.25)
        assert not bool(g0.attrs["is_partial_final_window"])
        assert g0.attrs["continuous_window_sec"] == pytest.approx(600.0)
        assert g0.attrs["continuous_step_sec"] == pytest.approx(600.0)


def test_custom_tabular_continuous_wrapper_completed_run_loads_without_region_deliverables(
    tmp_path: Path,
):
    input_dir = tmp_path / "input_custom"
    out_dir = tmp_path / "out_custom"
    cfg_path = tmp_path / "custom_continuous.yaml"
    _write_custom_tabular_csv(input_dir / "session_000.csv", duration_sec=1200.0, fs_hz=10.0)
    _write_continuous_config(cfg_path, target_fs_hz=10.0, custom_tabular=True)

    res = _run_wrapper(input_dir, out_dir, cfg_path, "custom_tabular")
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"

    assert not (out_dir / "day_plots").exists()
    _assert_continuous_completed_run_contract(out_dir)
    _assert_continuous_cache_window_contract(
        out_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    )


def test_rwd_continuous_wrapper_full_run_writes_window_metadata(tmp_path: Path):
    input_dir = tmp_path / "input_rwd"
    out_dir = tmp_path / "out_rwd"
    cfg_path = tmp_path / "rwd_continuous.yaml"
    _write_rwd_csv(
        input_dir / "2025_01_01-00_00_00" / "fluorescence.csv",
        duration_sec=1200.0,
        fs_hz=5.0,
    )
    _write_continuous_config(cfg_path, target_fs_hz=5.0, custom_tabular=False)

    res = _run_wrapper(input_dir, out_dir, cfg_path, "rwd")
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"

    _assert_continuous_completed_run_contract(out_dir)
    _assert_continuous_cache_window_contract(
        out_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    )
