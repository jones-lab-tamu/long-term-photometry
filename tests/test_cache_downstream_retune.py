import dataclasses
import json
import os
import struct

import pandas as pd
import pytest
import yaml
import numpy as np

from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import apply_peak_prefilter
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.io.hdf5_cache_reader import open_phasic_cache
from photometry_pipeline.tuning.cache_downstream_retune import (
    _build_chunk_for_roi,
    _resolve_prefilter_config_for_chunk,
    run_cache_downstream_retune,
)


def _read_png_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        header = f.read(24)
    # PNG signature (8 bytes) + IHDR length/type (8 bytes) + width/height (8 bytes)
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


def _make_quantized_time_lowpass_fixture(tmp_path, *, lowpass_hz: float = 1.0):
    run_dir = tmp_path / "run_quantized_time"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    features_dir = phasic_out / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    (run_dir / "MANIFEST.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")

    cfg = Config(
        target_fs_hz=20.0,
        lowpass_hz=float(lowpass_hz),
        event_signal="dff",
        peak_threshold_method="mean_std",
        peak_threshold_k=2.0,
        peak_min_distance_sec=0.5,
    )
    with open(phasic_out / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=True)

    pd.DataFrame([{"chunk_id": 0, "roi": "Region0", "peak_count": 1, "auc": 0.1}]).to_csv(
        features_dir / "features.csv", index=False
    )

    fs_hz = float(cfg.target_fs_hz)
    t_true = np.arange(0.0, 60.0, 1.0 / fs_hz)
    t_quantized = np.floor(t_true)
    signal = 0.35 * np.sin(2.0 * np.pi * 0.3 * t_true) + 0.85 * np.sin(2.0 * np.pi * 4.5 * t_true)

    cache_path = phasic_out / "phasic_trace_cache.h5"
    with Hdf5TraceCacheWriter(str(cache_path), "phasic", cfg) as writer:
        chunk = Chunk(
            chunk_id=0,
            source_file="session_0.csv",
            format="cache",
            time_sec=t_quantized,
            uv_raw=np.zeros((len(t_quantized), 1), dtype=float),
            sig_raw=np.zeros((len(t_quantized), 1), dtype=float),
            fs_hz=fs_hz,
            channel_names=["Region0"],
        )
        chunk.delta_f = signal.reshape(-1, 1)
        chunk.dff = signal.reshape(-1, 1)
        writer.add_chunk(chunk, chunk_id=0, source_file="session_0.csv")
    return run_dir, signal, cfg


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


def test_cache_retune_overlay_png_has_high_enough_resolution_for_gui_preview(tmp_path):
    run_dir = _make_completed_run_fixture(tmp_path)
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
    overlay_path = os.path.join(result["retune_dir"], "retuned_overlay_Region0_chunk_000.png")
    assert os.path.isfile(overlay_path)
    width, height = _read_png_size(overlay_path)
    # Keep post-run tuning overlays sharp when shown in large result panes.
    assert width >= 2800
    assert height >= 1000


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


def test_build_chunk_prefers_cached_fs_metadata_when_time_axis_is_quantized(tmp_path):
    run_dir, _signal, cfg = _make_quantized_time_lowpass_fixture(tmp_path)
    cache_path = run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    with open_phasic_cache(str(cache_path)) as cache:
        grp = cache["roi/Region0/chunk_0"]
        chunk = _build_chunk_for_roi(
            "Region0",
            0,
            "session_0.csv",
            grp,
            cfg,
        )
    # Quantized time axis would imply ~1 Hz from dt; retune must trust cached fs metadata.
    assert chunk.fs_hz == pytest.approx(20.0)


def test_prefilter_config_resolution_keeps_smooth_mode():
    cfg = Config(peak_pre_filter="smooth", lowpass_hz=3.0)
    resolved = _resolve_prefilter_config_for_chunk(cfg, fs_hz=20.0)
    assert resolved.peak_pre_filter == "smooth"
    assert resolved.lowpass_hz == pytest.approx(3.0)


def test_prefilter_config_resolution_maps_legacy_lowpass_to_smooth():
    cfg = Config(peak_pre_filter="lowpass", lowpass_hz=10.0)
    resolved = _resolve_prefilter_config_for_chunk(cfg, fs_hz=20.0, n_samples=12000)
    assert resolved.peak_pre_filter == "smooth"
    assert resolved.lowpass_hz == pytest.approx(10.0)


def test_retune_overlay_uses_smoothed_y_data_when_smooth_selected(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from photometry_pipeline.tuning import cache_downstream_retune as retune_mod

    run_dir, signal, cfg = _make_quantized_time_lowpass_fixture(tmp_path)
    captured: list[dict[str, np.ndarray | str | bool]] = []
    rc_context_calls: list[dict] = []
    original_plot = Axes.plot
    original_rc_context = retune_mod.matplotlib.rc_context

    def _spy_plot(self, *args, **kwargs):
        if kwargs.get("color") == "steelblue" and len(args) >= 2:
            captured.append(
                {
                    "x": np.asarray(args[0], dtype=float).copy(),
                    "y": np.asarray(args[1], dtype=float).copy(),
                    "label": str(kwargs.get("label", "")),
                    "antialiased": bool(kwargs.get("antialiased", True)),
                }
            )
        return original_plot(self, *args, **kwargs)

    def _spy_rc_context(*args, **kwargs):
        if args and isinstance(args[0], dict):
            rc_context_calls.append(dict(args[0]))
        elif "rc" in kwargs and isinstance(kwargs["rc"], dict):
            rc_context_calls.append(dict(kwargs["rc"]))
        return original_rc_context(*args, **kwargs)

    monkeypatch.setattr(Axes, "plot", _spy_plot)
    monkeypatch.setattr(retune_mod.matplotlib, "rc_context", _spy_rc_context)

    common_overrides = {
        "event_signal": "dff",
        "peak_threshold_method": "mean_std",
        "peak_threshold_k": 2.0,
    }
    run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={**common_overrides, "peak_pre_filter": "none"},
    )
    run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={**common_overrides, "peak_pre_filter": "smooth"},
    )

    assert len(captured) >= 2
    none_call = captured[-2]
    smooth_call = captured[-1]
    y_none = np.asarray(none_call["y"], dtype=float)
    y_smooth = np.asarray(smooth_call["y"], dtype=float)

    assert str(none_call["label"]) == "dff trace"
    assert "smooth" in str(smooth_call["label"]).lower()
    assert bool(none_call["antialiased"]) is False
    assert bool(smooth_call["antialiased"]) is False
    assert any(
        call.get("path.simplify") is False and float(call.get("path.simplify_threshold", 1.0)) == 0.0
        for call in rc_context_calls
    )

    expected_smooth, _ = apply_peak_prefilter(
        signal,
        20.0,
        dataclasses.replace(cfg, peak_pre_filter="smooth"),
    )

    # This is the exact y-data passed into matplotlib for the visible trace line.
    assert np.allclose(y_none, signal, rtol=0.0, atol=1e-12)
    assert np.allclose(y_smooth, expected_smooth, rtol=0.0, atol=1e-9)
    assert np.max(np.abs(y_none - y_smooth)) > 0.25


def test_real_run_shape_nyquist_edge_smooth_changes_plotted_y_materially(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    run_dir, _signal, _cfg = _make_quantized_time_lowpass_fixture(tmp_path, lowpass_hz=10.0)
    captured: list[np.ndarray] = []
    original_plot = Axes.plot

    def _spy_plot(self, *args, **kwargs):
        if kwargs.get("color") == "steelblue" and len(args) >= 2:
            captured.append(np.asarray(args[1], dtype=float).copy())
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", _spy_plot)

    common_overrides = {
        "event_signal": "dff",
        "peak_threshold_method": "median_mad",
        "peak_threshold_k": 4.0,
        "peak_min_distance_sec": 0.5,
        "peak_min_prominence_k": 0.0,
        "peak_min_width_sec": 0.0,
        "event_auc_baseline": "zero",
    }
    run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={**common_overrides, "peak_pre_filter": "none"},
    )
    run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={**common_overrides, "peak_pre_filter": "smooth"},
    )

    assert len(captured) >= 2
    y_none = captured[-2]
    y_smooth = captured[-1]
    diff = np.abs(y_none - y_smooth)
    assert float(np.max(diff)) > 1e-3
    assert int(np.count_nonzero(diff > 1e-4)) > 0


def test_retune_debug_is_off_by_default(tmp_path):
    run_dir, _signal, _cfg = _make_quantized_time_lowpass_fixture(tmp_path, lowpass_hz=10.0)
    result = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={
            "event_signal": "dff",
            "peak_threshold_method": "median_mad",
            "peak_threshold_k": 4.0,
            "peak_pre_filter": "none",
        },
    )
    debug_path = os.path.join(result["retune_dir"], "retune_preview_debug_backend.json")
    assert not os.path.exists(debug_path)


def test_retune_debug_writes_backend_trace_and_image_hashes_when_enabled(tmp_path, monkeypatch):
    run_dir, _signal, _cfg = _make_quantized_time_lowpass_fixture(tmp_path, lowpass_hz=10.0)
    monkeypatch.setenv("PHOTOMETRY_RETUNE_DEBUG", "1")
    result = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={
            "event_signal": "dff",
            "peak_threshold_method": "median_mad",
            "peak_threshold_k": 4.0,
            "peak_pre_filter": "none",
        },
    )
    debug_path = os.path.join(result["retune_dir"], "retune_preview_debug_backend.json")
    assert os.path.isfile(debug_path)
    with open(debug_path, "r", encoding="utf-8") as f:
        debug = json.load(f)
    assert debug["peak_pre_filter"] == "none"
    assert debug["chunk_id"] == 0
    assert debug["trace_sha256_f64"].strip()
    assert debug["overlay_png_sha256"].strip()
    assert debug["overlay_png_path"] == result["artifacts"]["retuned_overlay_png"]


def test_retune_debug_trace_hash_differs_between_none_and_smooth(tmp_path, monkeypatch):
    run_dir, _signal, _cfg = _make_quantized_time_lowpass_fixture(tmp_path, lowpass_hz=10.0)
    monkeypatch.setenv("PHOTOMETRY_RETUNE_DEBUG", "1")
    common = {
        "event_signal": "dff",
        "peak_threshold_method": "median_mad",
        "peak_threshold_k": 4.0,
        "peak_min_distance_sec": 0.5,
        "peak_min_prominence_k": 0.0,
        "peak_min_width_sec": 0.0,
        "event_auc_baseline": "zero",
    }
    res_none = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={**common, "peak_pre_filter": "none"},
    )
    res_smooth = run_cache_downstream_retune(
        run_dir=str(run_dir),
        roi="Region0",
        chunk_id=0,
        overrides={**common, "peak_pre_filter": "smooth"},
    )
    with open(os.path.join(res_none["retune_dir"], "retune_preview_debug_backend.json"), "r", encoding="utf-8") as f:
        dbg_none = json.load(f)
    with open(os.path.join(res_smooth["retune_dir"], "retune_preview_debug_backend.json"), "r", encoding="utf-8") as f:
        dbg_smooth = json.load(f)
    assert dbg_none["peak_pre_filter"] == "none"
    assert dbg_smooth["peak_pre_filter"] == "smooth"
    assert dbg_none["trace_sha256_f64"] != dbg_smooth["trace_sha256_f64"]


def test_smooth_prefilter_short_trace_degrades_without_crash():
    trace = np.array([0.1, 0.2], dtype=float)
    cfg = Config(peak_pre_filter="smooth")
    smoothed, meta = apply_peak_prefilter(trace, 20.0, cfg)
    assert np.allclose(smoothed, trace, rtol=0.0, atol=1e-12)
    assert meta["mode"] == "smooth"
    assert bool(meta["prefilter_applied"]) is False
