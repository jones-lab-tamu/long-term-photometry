import dataclasses
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.continuous_outputs import generate_retuned_continuous_phasic_outputs
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.tuning.cache_correction_retune import run_cache_correction_retune
from photometry_pipeline.tuning.cache_downstream_retune import run_cache_downstream_retune


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_png_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00"
    )


def _write_completed_markers(run_dir: Path) -> None:
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "status": "success"}),
        encoding="utf-8",
    )


def _continuous_metadata(cid: int) -> dict:
    start = 600.0 * float(cid)
    end = start + 600.0
    return {
        "acquisition_mode": "continuous",
        "window_index": int(cid),
        "window_start_sec": start,
        "window_end_sec": end,
        "window_duration_sec": 600.0,
        "original_file_duration_sec": 1200.0,
        "is_partial_final_window": False,
        "continuous_window_sec": 600.0,
        "continuous_step_sec": 600.0,
    }


def _make_signal(t: np.ndarray, cid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uv = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.01 * t + 0.2)
    pulses = np.zeros_like(t)
    for center in (120.0, 300.0, 470.0):
        pulses += 0.35 * np.exp(-0.5 * ((t - center) / 4.0) ** 2)
    sig = 2.0 + 0.8 * uv + 0.04 * np.sin(2.0 * np.pi * 0.04 * t + cid) + pulses
    delta_f = sig - uv
    dff = 100.0 * delta_f / 50.0
    return uv, sig, delta_f, dff


def _make_completed_continuous_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "continuous_run"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    features_dir = phasic_out / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    _write_completed_markers(run_dir)

    cfg = Config(
        target_fs_hz=10.0,
        event_signal="dff",
        peak_threshold_method="mean_std",
        peak_threshold_k=1.0,
        peak_min_distance_sec=10.0,
        acquisition_mode="continuous",
        continuous_window_sec=600.0,
        continuous_step_sec=600.0,
        allow_partial_final_window=False,
    )
    with open(phasic_out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)

    pd.DataFrame(
        [
            {"chunk_id": 0, "roi": "Region0", "peak_count": 999, "auc": 999.0},
            {"chunk_id": 1, "roi": "Region0", "peak_count": 999, "auc": 999.0},
        ]
    ).to_csv(features_dir / "features.csv", index=False)

    cache_path = phasic_out / "phasic_trace_cache.h5"
    with Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg) as writer:
        t = np.arange(0.0, 600.0, 1.0 / cfg.target_fs_hz)
        for cid in (0, 1):
            uv, sig, delta_f, dff = _make_signal(t, cid)
            chunk = Chunk(
                chunk_id=cid,
                source_file="continuous_source.csv",
                format="cache",
                time_sec=t,
                uv_raw=uv.reshape(-1, 1),
                sig_raw=sig.reshape(-1, 1),
                fs_hz=cfg.target_fs_hz,
                channel_names=["Region0"],
                metadata=_continuous_metadata(cid),
            )
            chunk.delta_f = delta_f.reshape(-1, 1)
            chunk.dff = dff.reshape(-1, 1)
            writer.add_chunk(chunk, chunk_id=cid, source_file="continuous_source.csv")

    production_summary = run_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv"
    production_summary.parent.mkdir(parents=True, exist_ok=True)
    production_summary.write_text("sentinel,unchanged\n1,2\n", encoding="utf-8")
    for filename in (
        "phasic_peak_rate_timeseries.png",
        "phasic_peak_count_timeseries.png",
        "phasic_auc_timeseries.png",
    ):
        _write_png_placeholder(run_dir / "Region0" / "summary" / filename)
    return run_dir


def _make_completed_intermittent_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "intermittent_run"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    features_dir = phasic_out / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    _write_completed_markers(run_dir)

    cfg = Config(target_fs_hz=10.0, event_signal="dff", peak_threshold_method="mean_std")
    with open(phasic_out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)
    pd.DataFrame([{"chunk_id": 0, "roi": "Region0", "peak_count": 1, "auc": 0.1}]).to_csv(
        features_dir / "features.csv",
        index=False,
    )
    t = np.arange(0.0, 60.0, 1.0 / cfg.target_fs_hz)
    uv, sig, delta_f, dff = _make_signal(t, 0)
    cache_path = phasic_out / "phasic_trace_cache.h5"
    with Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg) as writer:
        chunk = Chunk(
            chunk_id=0,
            source_file="session_0.csv",
            format="cache",
            time_sec=t,
            uv_raw=uv.reshape(-1, 1),
            sig_raw=sig.reshape(-1, 1),
            fs_hz=cfg.target_fs_hz,
            channel_names=["Region0"],
            metadata={},
        )
        chunk.delta_f = delta_f.reshape(-1, 1)
        chunk.dff = dff.reshape(-1, 1)
        writer.add_chunk(chunk, chunk_id=0, source_file="session_0.csv")
    return run_dir


def _assert_retuned_continuous_outputs(result: dict, roi: str = "Region0") -> pd.DataFrame:
    retuned = result["retuned_continuous_outputs"]
    assert retuned["generated"] is True
    assert retuned["continuous_detected"] is True
    assert retuned["roi"] == roi
    artifacts = result["artifacts"]
    summary_path = Path(artifacts["retuned_continuous_phasic_summary_csv"])
    rate_path = Path(artifacts["retuned_continuous_phasic_peak_rate_png"])
    count_path = Path(artifacts["retuned_continuous_phasic_peak_count_png"])
    auc_path = Path(artifacts["retuned_continuous_phasic_auc_png"])
    assert summary_path.name == f"retuned_continuous_phasic_window_summary_{roi}.csv"
    assert rate_path.name == f"retuned_phasic_peak_rate_timeseries_{roi}.png"
    assert count_path.name == f"retuned_phasic_peak_count_timeseries_{roi}.png"
    assert auc_path.name == f"retuned_phasic_auc_timeseries_{roi}.png"
    assert summary_path.exists()
    assert rate_path.exists()
    assert count_path.exists()
    assert auc_path.exists()
    assert summary_path.parent == Path(result["retune_dir"])
    assert rate_path.parent == Path(result["retune_dir"])
    assert not (Path(result["retune_dir"]) / roi / "tables").exists()
    assert not (Path(result["retune_dir"]) / roi / "summary").exists()
    return pd.read_csv(summary_path).sort_values("chunk_id").reset_index(drop=True)


def test_downstream_continuous_retune_writes_flat_retuned_summary_and_plots(tmp_path: Path):
    run_dir = _make_completed_continuous_run(tmp_path)

    result = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={"peak_threshold_k": 1.25},
    )

    summary = _assert_retuned_continuous_outputs(result)
    retuned_features = pd.read_csv(result["artifacts"]["retuned_features_csv"]).sort_values("chunk_id")
    assert len(summary) == 2
    assert list(summary["event_count"].astype(int)) == list(retuned_features["peak_count"].astype(int))
    np.testing.assert_allclose(summary["event_signal_auc"], retuned_features["auc"])
    assert set(summary["acquisition_mode"]) == {"continuous"}

    saved = json.loads((Path(result["retune_dir"]) / "retune_result.json").read_text(encoding="utf-8"))
    assert saved["retuned_continuous_outputs"]["generated"] is True
    assert "retuned_continuous_phasic_summary_csv" in saved["artifacts"]
    saved_block = saved["retuned_continuous_outputs"]
    assert saved_block["continuous_detected"] is True
    assert saved_block["summary_csv"] == result["retuned_continuous_outputs"]["summary_csv"]
    assert saved_block["plots"] == result["retuned_continuous_outputs"]["plots"]
    assert saved_block["source_features_path"] == result["retuned_continuous_outputs"]["source_features_path"]
    assert saved_block["source_cache_path"] == result["retuned_continuous_outputs"]["source_cache_path"]
    assert saved_block["roi"] == "Region0"
    np.testing.assert_allclose(
        summary["elapsed_hour_mid"],
        ((summary["window_start_sec"] + summary["window_end_sec"]) / 2.0) / 3600.0,
    )


def test_correction_continuous_retune_preserves_attrs_and_writes_outputs(tmp_path: Path):
    run_dir = _make_completed_continuous_run(tmp_path)

    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={"dynamic_fit_mode": "global_linear_regression"},
    )

    summary = _assert_retuned_continuous_outputs(result)
    assert len(summary) == 2
    assert list(summary["window_index"].astype(int)) == [0, 1]
    assert list(summary["window_start_sec"].astype(float)) == [0.0, 600.0]

    cache_path = Path(result["artifacts"]["retuned_correction_cache_h5"])
    with h5py.File(cache_path, "r") as h5:
        attrs = h5["roi"]["Region0"]["chunk_1"].attrs
        assert attrs["acquisition_mode"] == "continuous"
        assert int(round(float(attrs["window_index"]))) == 1
        assert float(attrs["window_start_sec"]) == 600.0
        assert float(attrs["window_end_sec"]) == 1200.0
        assert float(attrs["window_duration_sec"]) == 600.0
        assert float(attrs["original_file_duration_sec"]) == 1200.0
        assert bool(attrs["is_partial_final_window"]) is False
        assert float(attrs["continuous_window_sec"]) == 600.0
        assert float(attrs["continuous_step_sec"]) == 600.0

    retuned_features = pd.read_csv(result["artifacts"]["retuned_features_csv"]).sort_values("chunk_id")
    np.testing.assert_allclose(summary["event_signal_auc"], retuned_features["auc"])
    saved = json.loads((Path(result["retune_dir"]) / "retune_result.json").read_text(encoding="utf-8"))
    assert saved["retuned_continuous_outputs"]["generated"] is True
    assert saved["retuned_continuous_outputs"]["continuous_detected"] is True
    assert saved["retuned_continuous_outputs"]["source_cache_path"] == str(cache_path)


def test_retunes_do_not_overwrite_production_outputs(tmp_path: Path):
    run_dir = _make_completed_continuous_run(tmp_path)
    production_paths = [
        run_dir / "status.json",
        run_dir / "MANIFEST.json",
        run_dir / "_analysis" / "phasic_out" / "features" / "features.csv",
        run_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv",
        run_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png",
        run_dir / "Region0" / "summary" / "phasic_peak_count_timeseries.png",
        run_dir / "Region0" / "summary" / "phasic_auc_timeseries.png",
    ]
    before = {path: _sha256(path) for path in production_paths}

    run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={"peak_threshold_k": 1.25},
    )
    run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={"dynamic_fit_mode": "global_linear_regression"},
    )

    after = {path: _sha256(path) for path in production_paths}
    assert after == before


def test_intermittent_retunes_skip_continuous_outputs_without_artifacts(tmp_path: Path):
    run_dir = _make_completed_intermittent_run(tmp_path)

    downstream = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={"peak_threshold_k": 1.25},
    )
    correction = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={"dynamic_fit_mode": "global_linear_regression"},
    )

    for result in (downstream, correction):
        retuned = result["retuned_continuous_outputs"]
        assert retuned["generated"] is False
        assert retuned["continuous_detected"] is False
        assert "source cache is not continuous" in retuned["reason"]
        assert "source cache is not continuous" in retuned["skips"][0]["reason"]
        assert "retuned_continuous_phasic_summary_csv" not in result["artifacts"]
        assert not any(Path(result["retune_dir"]).glob("retuned_continuous_phasic_window_summary_*.csv"))


def test_retuned_continuous_helper_skips_intermittent_cache_without_outputs(tmp_path: Path):
    run_dir = _make_completed_intermittent_run(tmp_path)
    output_dir = tmp_path / "retuned_outputs"

    result = generate_retuned_continuous_phasic_outputs(
        features_path=str(run_dir / "_analysis" / "phasic_out" / "features" / "features.csv"),
        cache_path=str(run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"),
        output_dir=str(output_dir),
        roi="Region0",
    )

    assert result["generated"] is False
    assert result["continuous_detected"] is False
    assert "source cache is not continuous" in result["reason"]
    assert "source cache is not continuous" in result["skips"][0]["reason"]
    assert not any(output_dir.glob("retuned_continuous_phasic_window_summary_*.csv"))
    assert not any(output_dir.glob("retuned_phasic_*_timeseries_*.png"))


def test_retuned_continuous_helper_raises_when_continuous_attrs_are_missing(tmp_path: Path):
    run_dir = _make_completed_continuous_run(tmp_path)
    cache_path = run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "a") as h5:
        del h5["roi"]["Region0"]["chunk_0"].attrs["window_start_sec"]

    output_dir = tmp_path / "retuned_outputs"
    try:
        generate_retuned_continuous_phasic_outputs(
            features_path=str(run_dir / "_analysis" / "phasic_out" / "features" / "features.csv"),
            cache_path=str(cache_path),
            output_dir=str(output_dir),
            roi="Region0",
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("continuous cache with missing window attrs must not silently skip")

    assert "Missing attrs" in message
    assert "window_start_sec" in message
    assert not any(output_dir.glob("retuned_continuous_phasic_window_summary_*.csv"))
    assert not any(output_dir.glob("retuned_phasic_*_timeseries_*.png"))
