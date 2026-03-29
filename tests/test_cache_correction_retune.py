import dataclasses
import json
import os
import struct

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.tuning.cache_correction_retune import run_cache_correction_retune


def _read_png_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise AssertionError(f"Not a valid PNG header: {path}")
    width = struct.unpack(">I", header[16:20])[0]
    height = struct.unpack(">I", header[20:24])[0]
    return int(width), int(height)


def _make_completed_run_fixture(tmp_path):
    run_dir = tmp_path / "run_complete"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    features_dir = phasic_out / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )

    cfg = Config()
    with open(phasic_out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)

    # Deliberately non-authoritative source features to ensure retune recomputes instead of copying.
    source_features = pd.DataFrame(
        [
            {"chunk_id": 0, "roi": "Region0", "peak_count": 999, "auc": 111.0},
            {"chunk_id": 0, "roi": "Region1", "peak_count": 888, "auc": 222.0},
        ]
    )
    source_features.to_csv(features_dir / "features.csv", index=False)

    cache_path = phasic_out / "phasic_trace_cache.h5"
    with Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg) as writer:
        t = np.arange(0.0, 600.0, 1.0)
        for cid in (0, 1, 2):
            sig_r0 = 1.0 + 0.6 * np.sin(0.05 * t) + 0.2 * np.sin(0.5 * t + cid * 0.2)
            uv_r0 = 0.7 + 0.4 * np.sin(0.05 * t + 0.4)
            delta_r0 = sig_r0 - uv_r0
            dff_r0 = 100.0 * delta_r0 / 50.0

            sig_r1 = 1.2 + 0.4 * np.cos(0.03 * t + 0.1) + 0.1 * np.sin(0.4 * t + cid * 0.1)
            uv_r1 = 0.8 + 0.3 * np.cos(0.03 * t + 0.5)
            delta_r1 = sig_r1 - uv_r1
            dff_r1 = 100.0 * delta_r1 / 55.0

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


def _make_authoritative_completed_run_fixture(tmp_path, baseline_method: str):
    run_dir = tmp_path / f"run_authoritative_{baseline_method}"
    input_dir = run_dir / "input_RWD"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    input_dir.mkdir(parents=True, exist_ok=True)
    phasic_out.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        chunk_duration_sec=20.0,
        target_fs_hz=10.0,
        lowpass_hz=2.0,
        window_sec=6.0,
        step_sec=2.0,
        min_valid_windows=2,
        min_samples_per_window=20,
        r_low=0.05,
        r_high=0.95,
        g_min=0.1,
        baseline_method=baseline_method,
        baseline_percentile=10.0,
        peak_threshold_method="mean_std",
        peak_threshold_k=1.2,
        peak_min_distance_sec=0.4,
        peak_pre_filter="none",
        event_auc_baseline="zero",
        event_signal="dff",
    )

    n_samples = int(round(cfg.chunk_duration_sec * cfg.target_fs_hz))
    t = np.arange(n_samples) / cfg.target_fs_hz
    idx = np.arange(n_samples)
    for cid in (0, 1):
        session_dir = input_dir / f"chunk_{cid:04d}"
        session_dir.mkdir(parents=True, exist_ok=True)

        phase = 0.35 * cid
        uv = 0.8 + 0.15 * np.sin(0.35 * t + phase) + 0.03 * np.sin(0.08 * t)
        nuisance = 1.25 * uv + 0.02 * np.cos(0.9 * t + phase)
        phasic = 0.05 * np.sin(2.4 * t + 0.2 * cid) + 0.03 * np.sin(3.7 * t + 1.0)
        centers = [50 + (4 * cid), 120 + (3 * cid), 170 - (2 * cid)]
        pulses = sum(0.18 * np.exp(-0.5 * ((idx - c) / 3.0) ** 2) for c in centers)
        sig = nuisance + phasic + pulses

        pd.DataFrame(
            {
                "Time(s)": t,
                "Region0-410": uv,
                "Region0-470": sig,
            }
        ).to_csv(session_dir / "fluorescence.csv", index=False)

    pipe = Pipeline(cfg, mode="phasic")
    pipe.run(
        input_dir=str(input_dir),
        output_dir=str(phasic_out),
        force_format="rwd",
        recursive=False,
        glob_pattern="*.csv",
    )

    with open(phasic_out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)

    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    return run_dir, cfg


def test_correction_retune_success_and_output_isolation(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    source_features = run_dir / "_analysis" / "phasic_out" / "features" / "features.csv"
    before = source_features.read_text(encoding="utf-8")

    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={"window_sec": 45.0, "step_sec": 8.0},
    )

    assert os.path.isdir(result["retune_dir"])
    assert os.path.commonpath([result["retune_dir"], str(run_dir / "tuning_correction_retune")]) == str(
        run_dir / "tuning_correction_retune"
    )

    artifacts = result["artifacts"]
    assert os.path.isfile(artifacts["retuned_correction_cache_h5"])
    assert os.path.isfile(artifacts["retuned_features_csv"])
    assert os.path.isfile(artifacts["retuned_summary_csv"])
    assert os.path.isfile(artifacts["retuned_correction_session_csv"])
    inspection_pngs = artifacts["retuned_correction_inspection_pngs"]
    assert isinstance(inspection_pngs, list)
    assert len(inspection_pngs) == 4
    for p in inspection_pngs:
        assert os.path.isfile(p)
    assert artifacts["retuned_correction_inspection_panel_labels"] == [
        "Raw absolute sig/iso",
        "Centered common-gain sig/iso",
        "Dynamic fit",
        "Final corrected dF/F",
    ]

    # Production features remain untouched.
    assert before == source_features.read_text(encoding="utf-8")


def test_correction_retune_accepts_dynamic_fit_mode_override_and_records_it(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={"dynamic_fit_mode": "global_linear_regression"},
    )

    assert os.path.isdir(result["retune_dir"])
    assert os.path.isfile(result["artifacts"]["retuned_correction_cache_h5"])

    with open(os.path.join(result["retune_dir"], "retune_config_effective.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["dynamic_fit_mode"] == "global_linear_regression"

    with open(os.path.join(result["retune_dir"], "retune_request.json"), "r", encoding="utf-8") as f:
        req = json.load(f)
    assert req["correction_overrides_applied"]["dynamic_fit_mode"] == "global_linear_regression"


def test_correction_retune_accepts_baseline_subtract_override_and_records_it(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={
            "dynamic_fit_mode": "rolling_filtered_to_filtered",
            "baseline_subtract_before_fit": True,
        },
    )

    assert os.path.isdir(result["retune_dir"])
    assert os.path.isfile(result["artifacts"]["retuned_correction_cache_h5"])

    with open(os.path.join(result["retune_dir"], "retune_config_effective.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["dynamic_fit_mode"] == "rolling_filtered_to_filtered"
    assert cfg["baseline_subtract_before_fit"] is True

    with open(os.path.join(result["retune_dir"], "retune_request.json"), "r", encoding="utf-8") as f:
        req = json.load(f)
    assert req["correction_overrides_applied"]["baseline_subtract_before_fit"] is True
    assert "baseline_subtract_before_fit" in req["override_classification"]["correction_supported"]

    assert result["correction_overrides_applied"]["baseline_subtract_before_fit"] is True


def test_correction_retune_accepts_robust_event_reject_overrides_and_records_them(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={
            "dynamic_fit_mode": "robust_global_event_reject",
            "robust_event_reject_max_iters": 4,
            "robust_event_reject_residual_z_thresh": 3.1,
            "robust_event_reject_local_var_window_sec": 9.0,
            "robust_event_reject_local_var_ratio_thresh": 4.2,
            "robust_event_reject_min_keep_fraction": 0.6,
        },
    )

    with open(os.path.join(result["retune_dir"], "retune_config_effective.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["dynamic_fit_mode"] == "robust_global_event_reject"
    assert cfg["robust_event_reject_max_iters"] == 4
    assert cfg["robust_event_reject_residual_z_thresh"] == pytest.approx(3.1)
    assert cfg["robust_event_reject_local_var_window_sec"] == pytest.approx(9.0)
    assert cfg["robust_event_reject_local_var_ratio_thresh"] == pytest.approx(4.2)
    assert cfg["robust_event_reject_min_keep_fraction"] == pytest.approx(0.6)

    with open(os.path.join(result["retune_dir"], "retune_request.json"), "r", encoding="utf-8") as f:
        req = json.load(f)
    assert req["correction_overrides_applied"]["dynamic_fit_mode"] == "robust_global_event_reject"
    assert "robust_event_reject_max_iters" in req["override_classification"]["correction_supported"]
    assert "robust_event_reject_residual_z_thresh" in req["override_classification"]["correction_supported"]
    robust_diag = result["artifacts"].get("retuned_correction_inspection_robust_diagnostics")
    assert isinstance(robust_diag, dict)
    assert robust_diag["fit_mode_resolved"] == "robust_global_event_reject"
    assert robust_diag["iterations_completed"] >= 1
    assert 0.0 <= robust_diag["keep_fraction"] <= 1.0
    assert robust_diag["fallback_to_global_linear"] is False
    assert robust_diag["fallback_status"] == "no"
    assert 0.0 <= robust_diag["excluded_fraction"] <= 1.0


def test_correction_retune_robust_mode_emits_fallback_diagnostics_when_robust_fit_fails(
    tmp_path,
    monkeypatch,
):
    run_dir = _make_completed_run_fixture(tmp_path)

    def _raise_forced_failure(*args, **kwargs):
        raise RuntimeError("forced robust fit failure for retune diagnostics test")

    monkeypatch.setattr(
        "photometry_pipeline.core.regression.fit_robust_global_event_reject",
        _raise_forced_failure,
    )

    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={"dynamic_fit_mode": "robust_global_event_reject"},
    )
    robust_diag = result["artifacts"].get("retuned_correction_inspection_robust_diagnostics")
    assert isinstance(robust_diag, dict)
    assert robust_diag["fallback_to_global_linear"] is True
    assert robust_diag["fallback_failed"] is False
    assert robust_diag["fallback_status"] == "yes"
    assert robust_diag["keep_fraction"] == pytest.approx(1.0)
    assert robust_diag["excluded_count"] == 0


def test_correction_retune_non_robust_mode_does_not_emit_robust_diagnostics(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={"dynamic_fit_mode": "global_linear_regression"},
    )
    assert "retuned_correction_inspection_robust_diagnostics" not in result["artifacts"]


def test_correction_retune_rejects_downstream_only_override(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    with pytest.raises(ValueError, match="Downstream-only"):
        run_cache_correction_retune(
            run_dir=str(run_dir),
            roi="Region0",
            overrides={"event_signal": "delta_f"},
        )


def test_correction_retune_rejects_f0_min_value(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        run_cache_correction_retune(
            run_dir=str(run_dir),
            roi="Region0",
            overrides={"f0_min_value": 0.1},
        )


def test_correction_retune_rejects_unknown_override(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    with pytest.raises(ValueError, match="Unknown override"):
        run_cache_correction_retune(
            run_dir=str(run_dir),
            roi="Region0",
            overrides={"not_a_real_key": 123},
        )


def test_correction_retune_missing_phasic_cache_fails(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    os.remove(run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5")

    with pytest.raises(RuntimeError, match="Missing phasic cache"):
        run_cache_correction_retune(
            run_dir=str(run_dir),
            roi="Region0",
            overrides={"window_sec": 60.0},
        )


def test_correction_retune_missing_config_snapshot_fails(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    os.remove(run_dir / "_analysis" / "phasic_out" / "config_used.yaml")

    with pytest.raises(RuntimeError, match="Missing base config snapshot"):
        run_cache_correction_retune(
            run_dir=str(run_dir),
            roi="Region0",
            overrides={"window_sec": 60.0},
        )


def test_correction_retune_roi_scoping(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region1",
        overrides={"baseline_percentile": 15.0},
    )

    feats = pd.read_csv(result["artifacts"]["retuned_features_csv"])
    assert set(feats["roi"].unique()) == {"Region1"}

    with h5py.File(result["artifacts"]["retuned_correction_cache_h5"], "r") as f:
        rois = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["meta/rois"][()]]
        assert rois == ["Region1"]


def test_correction_retune_chunk_targeting_and_roi_wide_recompute(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={"r_low": 0.1, "r_high": 0.9},
        chunk_id=2,
    )

    assert result["inspection_chunk_id"] == 2
    assert result["inspection_source_file"] == "session_2.csv"
    assert result["n_chunks"] == 3

    summary = pd.read_csv(result["artifacts"]["retuned_summary_csv"])
    assert set(summary["chunk_id"].tolist()) == {0, 1, 2}

    req_path = os.path.join(result["retune_dir"], "retune_request.json")
    with open(req_path, "r", encoding="utf-8") as f:
        req = json.load(f)
    assert req["inspection_chunk_id_requested"] == 2
    assert req["inspection_chunk_id_used"] == 2

    assert result["artifacts"]["retuned_correction_inspection_png"].endswith("chunk_002_raw.png")
    assert len(result["artifacts"]["retuned_correction_inspection_pngs"]) == 4
    assert result["artifacts"]["retuned_correction_inspection_pngs"][0].endswith("chunk_002_raw.png")
    assert result["artifacts"]["retuned_correction_inspection_pngs"][1].endswith("chunk_002_centered.png")
    assert result["artifacts"]["retuned_correction_inspection_pngs"][2].endswith("chunk_002_fit.png")
    assert result["artifacts"]["retuned_correction_inspection_pngs"][3].endswith("chunk_002_dff.png")
    assert result["artifacts"]["retuned_correction_session_csv"].endswith("chunk_002.csv")


def test_correction_retune_features_are_recomputed_not_copied(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    source_features_path = run_dir / "_analysis" / "phasic_out" / "features" / "features.csv"
    source_df = pd.read_csv(source_features_path)

    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={"window_sec": 30.0, "step_sec": 5.0},
    )

    retuned_df = pd.read_csv(result["artifacts"]["retuned_features_csv"])
    assert len(retuned_df) >= 3
    assert set(retuned_df["chunk_id"].unique()) == {0, 1, 2}
    assert not (retuned_df["peak_count"] == 999).all()

    # Source production features are not overwritten.
    source_df_after = pd.read_csv(source_features_path)
    pd.testing.assert_frame_equal(source_df, source_df_after)


def test_correction_retune_required_correction_diagnostic_exists(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={"g_min": 0.1},
    )

    pngs = result["artifacts"]["retuned_correction_inspection_pngs"]
    csv = result["artifacts"]["retuned_correction_session_csv"]
    assert len(pngs) == 4
    for png in pngs:
        assert os.path.isfile(png)
    assert os.path.isfile(csv)


def test_correction_retune_inspection_png_has_useful_preview_resolution(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={"window_sec": 45.0, "step_sec": 8.0},
    )
    pngs = result["artifacts"]["retuned_correction_inspection_pngs"]
    assert len(pngs) == 4
    for png in pngs:
        assert os.path.isfile(png)
        width, height = _read_png_size(png)
        assert width >= 2800
        assert height >= 1100


def test_correction_retune_hdf5_internal_contract(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region1",
        overrides={"window_sec": 50.0, "step_sec": 9.0},
    )

    cache_path = result["artifacts"]["retuned_correction_cache_h5"]
    assert os.path.isfile(cache_path)
    required = {"time_sec", "sig_raw", "uv_raw", "fit_ref", "delta_f", "dff"}
    with h5py.File(cache_path, "r") as f:
        rois = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["meta/rois"][()]]
        chunk_ids = [int(x) for x in f["meta/chunk_ids"][()]]

        assert rois == ["Region1"]
        assert "roi/Region0" not in f
        assert chunk_ids == [0, 1, 2]

        for cid in chunk_ids:
            grp = f[f"roi/Region1/chunk_{cid}"]
            assert required.issubset(set(grp.keys()))


@pytest.mark.parametrize(
    "baseline_method",
    ["uv_raw_percentile_session", "uv_globalfit_percentile_session"],
)
def test_correction_retune_baseline_method_coverage(tmp_path, baseline_method):
    run_dir = _make_completed_run_fixture(tmp_path)
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides={
            "baseline_method": baseline_method,
            "baseline_percentile": 12.0,
            "window_sec": 45.0,
            "step_sec": 8.0,
        },
    )
    artifacts = result["artifacts"]
    assert os.path.isfile(artifacts["retuned_correction_cache_h5"])
    assert os.path.isfile(artifacts["retuned_features_csv"])


@pytest.mark.parametrize(
    "baseline_method",
    ["uv_raw_percentile_session", "uv_globalfit_percentile_session"],
)
def test_correction_retune_authoritative_parity_with_pipeline(tmp_path, baseline_method):
    run_dir, cfg = _make_authoritative_completed_run_fixture(tmp_path, baseline_method)
    overrides = {
        "window_sec": float(cfg.window_sec),
        "step_sec": float(cfg.step_sec),
        "min_valid_windows": int(cfg.min_valid_windows),
        "min_samples_per_window": int(cfg.min_samples_per_window),
        "r_low": float(cfg.r_low),
        "r_high": float(cfg.r_high),
        "g_min": float(cfg.g_min),
        "baseline_method": str(cfg.baseline_method),
        "baseline_percentile": float(cfg.baseline_percentile),
        "lowpass_hz": float(cfg.lowpass_hz),
    }
    result = run_cache_correction_retune(
        run_dir=str(run_dir),
        roi="Region0",
        overrides=overrides,
    )

    prod_cache_path = run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    retune_cache_path = result["artifacts"]["retuned_correction_cache_h5"]
    with h5py.File(prod_cache_path, "r") as prod, h5py.File(retune_cache_path, "r") as ret:
        for cid in (0, 1):
            prod_grp = prod[f"roi/Region0/chunk_{cid}"]
            ret_grp = ret[f"roi/Region0/chunk_{cid}"]
            np.testing.assert_allclose(
                prod_grp["delta_f"][()],
                ret_grp["delta_f"][()],
                rtol=1e-6,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                prod_grp["dff"][()],
                ret_grp["dff"][()],
                rtol=1e-6,
                atol=1e-8,
            )

    prod_features = pd.read_csv(run_dir / "_analysis" / "phasic_out" / "features" / "features.csv")
    prod_features = prod_features[prod_features["roi"] == "Region0"].copy()
    ret_features = pd.read_csv(result["artifacts"]["retuned_features_csv"])
    ret_features = ret_features[ret_features["roi"] == "Region0"].copy()

    prod_summary = (
        prod_features.groupby("chunk_id", as_index=False)[["peak_count", "auc"]]
        .sum()
        .sort_values("chunk_id")
        .reset_index(drop=True)
    )
    ret_summary = (
        ret_features.groupby("chunk_id", as_index=False)[["peak_count", "auc"]]
        .sum()
        .sort_values("chunk_id")
        .reset_index(drop=True)
    )

    assert list(prod_summary["chunk_id"]) == list(ret_summary["chunk_id"]) == [0, 1]
    assert prod_summary["peak_count"].astype(int).tolist() == ret_summary["peak_count"].astype(int).tolist()
    np.testing.assert_allclose(
        prod_summary["auc"].to_numpy(dtype=float),
        ret_summary["auc"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-8,
    )
