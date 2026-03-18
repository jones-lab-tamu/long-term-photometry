import dataclasses
import json
import os

import pandas as pd
import pytest
import yaml
import numpy as np

from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.tuning.cache_downstream_retune import (
    run_cache_downstream_retune,
)


def _make_completed_run_fixture(tmp_path):
    run_dir = tmp_path / "run_complete"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    features_dir = phasic_out / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    status_payload = {
        "schema_version": 1,
        "phase": "final",
        "status": "success",
    }
    (run_dir / "status.json").write_text(json.dumps(status_payload), encoding="utf-8")

    manifest_payload = {"status": "success"}
    (run_dir / "MANIFEST.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    cfg = Config()
    with open(phasic_out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)

    original_features = pd.DataFrame(
        [
            {"chunk_id": 0, "roi": "Region0", "peak_count": 123, "auc": 9.9},
            {"chunk_id": 0, "roi": "Region1", "peak_count": 111, "auc": 8.8},
        ]
    )
    original_features.to_csv(features_dir / "features.csv", index=False)

    cache_path = phasic_out / "phasic_trace_cache.h5"
    with Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg) as writer:
        t = np.arange(0, 600.0, 1.0)
        for cid in (0, 1, 2):
            sig_r0 = np.sin(0.05 * t) + (cid * 0.05)
            uv_r0 = 0.4 * np.sin(0.05 * t + 0.2) + 0.2
            delta_r0 = sig_r0 - uv_r0
            dff_r0 = 100.0 * delta_r0 / 50.0

            sig_r1 = np.cos(0.04 * t) + 0.1
            uv_r1 = 0.3 * np.cos(0.04 * t + 0.3) + 0.2
            delta_r1 = sig_r1 - uv_r1
            dff_r1 = 100.0 * delta_r1 / 40.0

            chunk = Chunk(
                chunk_id=cid,
                source_file=f"session_{cid}.csv",
                format="cache",
                time_sec=t,
                uv_raw=np.column_stack([uv_r0, uv_r1]),
                sig_raw=np.column_stack([sig_r0, sig_r1]),
                fs_hz=1.0,
                channel_names=["Region0", "Region1"],
            )
            chunk.delta_f = np.column_stack([delta_r0, delta_r1])
            chunk.dff = np.column_stack([dff_r0, dff_r1])
            writer.add_chunk(chunk, chunk_id=cid, source_file=f"session_{cid}.csv")

    return run_dir


def test_cache_retune_success_and_output_isolation(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    orig_features_path = run_dir / "_analysis" / "phasic_out" / "features" / "features.csv"
    before = orig_features_path.read_text(encoding="utf-8")

    result = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={
            "event_signal": "dff",
            "peak_threshold_method": "absolute",
            "peak_threshold_abs": 0.2,
            "peak_min_distance_sec": 1.0,
        },
    )

    assert os.path.isdir(result["retune_dir"])
    assert os.path.commonpath([result["retune_dir"], str(run_dir / "tuning_retune")]) == str(run_dir / "tuning_retune")
    assert os.path.isfile(os.path.join(result["retune_dir"], "retuned_features_Region0.csv"))
    assert os.path.isfile(os.path.join(result["retune_dir"], "retune_request.json"))
    assert os.path.isfile(os.path.join(result["retune_dir"], "retuned_overlay_Region0_chunk_000.png"))
    assert os.path.isfile(os.path.join(result["retune_dir"], "retuned_events_Region0_chunk_000.csv"))
    assert result["inspection_chunk_id"] == 0
    assert before == orig_features_path.read_text(encoding="utf-8")


def test_cache_retune_rejects_correction_sensitive_override(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    with pytest.raises(ValueError, match="recompute"):
        run_cache_downstream_retune(
            run_dir=str(run_dir),
            roi="Region0",
            overrides={"window_sec": 120.0},
        )


def test_cache_retune_rejects_f0_min_value(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        run_cache_downstream_retune(
            run_dir=str(run_dir),
            roi="Region0",
            overrides={"f0_min_value": 0.1},
        )


def test_cache_retune_writes_provenance(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    overrides = {
        "event_signal": "delta_f",
        "peak_threshold_method": "mean_std",
        "peak_threshold_k": 1.5,
    }
    result = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides=overrides,
    )
    req_path = os.path.join(result["retune_dir"], "retune_request.json")
    with open(req_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["source_run_dir"] == os.path.abspath(str(run_dir))
    assert data["selected_roi"] == "Region0"
    assert data["inspection_chunk_id_requested"] is None
    assert data["inspection_chunk_id_used"] == 0
    assert data["event_signal_used"] == "delta_f"
    assert data["downstream_overrides_applied"] == overrides
    assert data["override_classification"]["correction_required"] == []


def test_cache_retune_selected_chunk_targeting(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={
            "event_signal": "dff",
            "peak_threshold_method": "mean_std",
            "peak_threshold_k": 1.0,
        },
        chunk_id=2,
    )
    assert result["inspection_chunk_id"] == 2
    assert os.path.isfile(os.path.join(result["retune_dir"], "retuned_overlay_Region0_chunk_002.png"))
    assert os.path.isfile(os.path.join(result["retune_dir"], "retuned_events_Region0_chunk_002.csv"))
    assert not os.path.exists(os.path.join(result["retune_dir"], "retuned_overlay_Region0_chunk_000.png"))

    req_path = os.path.join(result["retune_dir"], "retune_request.json")
    with open(req_path, "r", encoding="utf-8") as f:
        req = json.load(f)
    assert req["inspection_chunk_id_requested"] == 2
    assert req["inspection_chunk_id_used"] == 2


def test_cache_retune_scopes_outputs_to_selected_roi(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region1",
        overrides={
            "event_signal": "dff",
            "peak_threshold_method": "percentile",
            "peak_threshold_percentile": 90.0,
        },
    )
    features_path = os.path.join(result["retune_dir"], "retuned_features_Region1.csv")
    df = pd.read_csv(features_path)
    assert set(df["roi"].unique()) == {"Region1"}
    assert not os.path.exists(os.path.join(result["retune_dir"], "retuned_features_Region0.csv"))
