import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


def _write_custom_tabular_csv(path: Path, duration_sec: float, fs_hz: float = 10.0) -> None:
    n = int(round(duration_sec * fs_hz))
    t = np.arange(n, dtype=float) / float(fs_hz)
    iso = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 2.0 + 0.9 * iso + 0.04 * np.sin(2.0 * np.pi * 0.08 * t + 0.2)
    df = pd.DataFrame({"time_sec": t, "Region0_iso": iso, "Region0_sig": sig})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


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


def test_continuous_phasic_summary_table_uses_features_and_window_attrs(tmp_path: Path):
    out_dir = _run_wrapper(tmp_path, "phasic")
    summary_path = out_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv"
    assert summary_path.exists()

    summary = pd.read_csv(summary_path)
    features = pd.read_csv(out_dir / "_analysis" / "phasic_out" / "features" / "features.csv")
    features = features[features["roi"] == "Region0"].sort_values("chunk_id").reset_index(drop=True)
    summary = summary.sort_values("window_index").reset_index(drop=True)

    assert len(summary) == 2
    assert list(summary.columns) == [
        "roi",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "window_midpoint_sec",
        "window_duration_sec",
        "phasic_signal_auc",
        "event_count",
        "event_rate_per_min",
        "is_partial_final_window",
    ]
    assert list(summary["event_count"].astype(int)) == list(features["peak_count"].astype(int))
    np.testing.assert_allclose(summary["phasic_signal_auc"], features["auc"])
    np.testing.assert_allclose(
        summary["event_rate_per_min"],
        summary["event_count"] / (summary["window_duration_sec"] / 60.0),
    )
    np.testing.assert_allclose(
        summary["window_midpoint_sec"],
        (summary["window_start_sec"] + summary["window_end_sec"]) / 2.0,
    )
    for unavailable in [
        "event_signal_auc",
        "event_signal_mean",
        "event_signal_median",
        "event_signal_std",
        "event_signal_mad",
        "event_rate_per_hour",
        "elapsed_hour_start",
        "elapsed_hour_mid",
        "source_file",
        "chunk_id",
        "continuous_window_sec",
        "continuous_step_sec",
        "acquisition_mode",
        "event_signal_auc_semantics",
        "mean_event_amplitude",
        "peak_event_amplitude",
        "event_auc_mean",
        "mean_event_width_sec",
        "n_rejected_events",
    ]:
        assert unavailable not in summary.columns
    assert not summary["is_partial_final_window"].any()

    with h5py.File(
        out_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5", "r"
    ) as h5:
        for row in summary.itertuples(index=False):
            group = h5["roi"]["Region0"][f"chunk_{int(row.window_index)}"]
            local_time = group["time_sec"][()]
            start = float(group.attrs["window_start_sec"]) + float(local_time[0])
            end = float(group.attrs["window_start_sec"]) + float(local_time[-1])
            assert row.window_start_sec == pytest.approx(start)
            assert row.window_end_sec == pytest.approx(end)
            assert row.window_duration_sec == pytest.approx(end - start)

    phasic_report = json.loads(
        (
            out_dir
            / "_analysis"
            / "phasic_out"
            / "run_report.json"
        ).read_text(encoding="utf-8")
    )
    provenance = phasic_report["continuous_phasic_auc"]
    assert provenance["event_signal"] == "dff"
    assert provenance["auc_baseline_mode"] == "zero"
    assert provenance["polarity"] == "positive"
    assert provenance["prefilter"] == "none"
    assert provenance["signal_units"] == "dF/F"
    assert provenance["auc_units"] == "dF/F·s"
    assert provenance["window_length_sec"] == pytest.approx(600.0)
    assert provenance["window_step_sec"] == pytest.approx(600.0)
    assert provenance["partial_final_window_policy"]["determination"]

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    continuous_outputs = manifest["continuous_outputs"]
    assert continuous_outputs["summary_tables_generated"] is True
    assert "Region0/tables/continuous_phasic_window_summary.csv" in continuous_outputs["summary_tables"]
    assert any(skip["reason"] == "tonic mode not requested" for skip in continuous_outputs["summary_skips"])
    assert continuous_outputs["intermittent_only_outputs_skipped"]
    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["continuous_outputs"]["summary_tables_generated"] is True
    assert "Region0/tables/continuous_phasic_window_summary.csv" in status["continuous_outputs"]["summary_tables"]


def test_continuous_tonic_summary_table_uses_cached_deltaf_and_window_attrs(tmp_path: Path):
    out_dir = _run_wrapper(tmp_path, "both")
    summary_path = out_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv"
    assert summary_path.exists()

    summary = pd.read_csv(summary_path).sort_values("chunk_id").reset_index(drop=True)
    assert len(summary) == 2
    cache_path = out_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5"
    with h5py.File(cache_path, "r") as h5:
        for row in summary.itertuples(index=False):
            arr = h5["roi"]["Region0"][f"chunk_{int(row.chunk_id)}"]["deltaF"][()]
            finite = arr[np.isfinite(arr)]
            assert int(row.tonic_n_finite) == int(finite.size)
            assert row.tonic_nan_fraction == pytest.approx(
                1.0 - (finite.size / float(arr.size))
            )
            assert row.tonic_mean == pytest.approx(float(np.mean(finite)))
            assert row.tonic_median == pytest.approx(float(np.median(finite)))
            assert row.tonic_p05 == pytest.approx(float(np.percentile(finite, 5)))
            assert row.tonic_p95 == pytest.approx(float(np.percentile(finite, 95)))
            attrs = h5["roi"]["Region0"][f"chunk_{int(row.chunk_id)}"].attrs
            assert row.window_start_sec == pytest.approx(float(attrs["window_start_sec"]))
            assert row.window_end_sec == pytest.approx(float(attrs["window_end_sec"]))
            assert row.elapsed_hour_mid == pytest.approx(
                ((float(attrs["window_start_sec"]) + float(attrs["window_end_sec"])) / 2.0) / 3600.0
            )

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    continuous_outputs = manifest["continuous_outputs"]
    assert continuous_outputs["summary_tables_generated"] is True
    assert "Region0/tables/continuous_phasic_window_summary.csv" in continuous_outputs["summary_tables"]
    assert "Region0/tables/continuous_tonic_window_summary.csv" in continuous_outputs["summary_tables"]


def test_continuous_tonic_only_skips_phasic_summary_cleanly(tmp_path: Path):
    out_dir = _run_wrapper(tmp_path, "tonic")
    assert (out_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv").exists()
    assert not (out_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv").exists()

    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    continuous_outputs = manifest["continuous_outputs"]
    assert "Region0/tables/continuous_tonic_window_summary.csv" in continuous_outputs["summary_tables"]
    assert any(skip["reason"] == "phasic mode not requested" for skip in continuous_outputs["summary_skips"])
