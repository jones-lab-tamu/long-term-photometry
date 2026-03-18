"""
Cache-driven downstream retune backend.

This module implements a bounded retune path for completed runs:
- reads phasic cache artifacts from an existing successful run
- accepts only downstream-retunable overrides
- rejects correction-context overrides explicitly
- recomputes downstream features for a selected ROI
- writes outputs to an isolated tuning subtree
"""

from __future__ import annotations

import dataclasses
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import yaml

from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import extract_features
from photometry_pipeline.core.preprocessing import lowpass_filter
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    open_phasic_cache,
    resolve_cache_roi,
)


DOWNSTREAM_RETUNABLE_KEYS = {
    "event_signal",
    "peak_threshold_method",
    "peak_threshold_k",
    "peak_threshold_percentile",
    "peak_threshold_abs",
    "peak_min_distance_sec",
    "peak_pre_filter",
    "event_auc_baseline",
}

CORRECTION_RECOMPUTE_REQUIRED_KEYS = {
    "window_sec",
    "step_sec",
    "min_valid_windows",
    "min_samples_per_window",
    "r_low",
    "r_high",
    "g_min",
    "baseline_method",
    "baseline_percentile",
    "lowpass_hz",
}

EXPLICITLY_UNSUPPORTED_KEYS = {
    "f0_min_value",
}

_OVERRIDE_VALUE_CASTERS = {
    "event_signal": str,
    "peak_threshold_method": str,
    "peak_threshold_k": float,
    "peak_threshold_percentile": float,
    "peak_threshold_abs": float,
    "peak_min_distance_sec": float,
    "peak_pre_filter": str,
    "event_auc_baseline": str,
    "window_sec": float,
    "step_sec": float,
    "min_valid_windows": int,
    "min_samples_per_window": int,
    "r_low": float,
    "r_high": float,
    "g_min": float,
    "baseline_method": str,
    "baseline_percentile": float,
    "lowpass_hz": float,
    "f0_min_value": float,
}


def parse_key_value_overrides(items: Iterable[str]) -> Dict[str, Any]:
    """Parse KEY=VALUE strings into typed override values."""
    parsed: Dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Use KEY=VALUE.")
        key, value = raw.split("=", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid override '{raw}'. Empty key.")
        caster = _OVERRIDE_VALUE_CASTERS.get(key, str)
        try:
            parsed[key] = caster(value)
        except Exception as exc:
            raise ValueError(f"Invalid value for override '{key}': {value}") from exc
    return parsed


def classify_overrides(overrides: Dict[str, Any]) -> Dict[str, list[str]]:
    """Classify override keys by retune boundary class."""
    keys = set(overrides.keys())
    return {
        "downstream": sorted(keys & DOWNSTREAM_RETUNABLE_KEYS),
        "correction_required": sorted(keys & CORRECTION_RECOMPUTE_REQUIRED_KEYS),
        "unsupported": sorted(keys & EXPLICITLY_UNSUPPORTED_KEYS),
        "unknown": sorted(
            keys
            - DOWNSTREAM_RETUNABLE_KEYS
            - CORRECTION_RECOMPUTE_REQUIRED_KEYS
            - EXPLICITLY_UNSUPPORTED_KEYS
        ),
    }


def _assert_successful_completed_run(run_dir: str) -> str:
    """
    Verify run_dir is defensibly complete and successful.

    Evidence order:
      1) status.json (phase=final, status=success)
      2) MANIFEST.json (status=success)
    """
    status_path = os.path.join(run_dir, "status.json")
    if os.path.isfile(status_path):
        with open(status_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        phase = str(data.get("phase", "")).strip().lower()
        status = str(data.get("status", "")).strip().lower()
        if phase == "final" and status == "success":
            return "status.json"

    manifest_path = os.path.join(run_dir, "MANIFEST.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        status = str(data.get("status", "")).strip().lower()
        if status == "success":
            return "MANIFEST.json"

    raise RuntimeError(
        "Run directory is not defensibly a successful completed run. "
        "Require status.json final success or MANIFEST.json status=success."
    )


def _resolve_base_config(phasic_out_dir: str) -> Tuple[Config, str]:
    """
    Load base config used by the completed phasic run.
    """
    cfg_path = os.path.join(phasic_out_dir, "config_used.yaml")
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"Missing base config snapshot: {cfg_path}")
    return Config.from_yaml(cfg_path), cfg_path


def _apply_downstream_overrides(base_cfg: Config, overrides: Dict[str, Any]) -> Config:
    """
    Return a Config object with accepted downstream overrides applied.
    """
    cfg_dict = dataclasses.asdict(base_cfg)
    cfg_dict.update(overrides)
    # Re-validate through Config constructor + from_yaml semantics constraints.
    return Config(**cfg_dict)


def _compute_chunk_fs_hz(time_sec: np.ndarray, fallback: float) -> float:
    if len(time_sec) < 2:
        return float(fallback)
    dt = np.diff(time_sec)
    finite = dt[np.isfinite(dt) & (dt > 0)]
    if len(finite) == 0:
        return float(fallback)
    return float(1.0 / np.median(finite))


def _build_chunk_for_roi(
    roi: str,
    chunk_id: int,
    source_file: str,
    grp,
    cfg: Config,
) -> Chunk:
    required = {"time_sec", "sig_raw", "uv_raw"}
    signal_field = "dff" if cfg.event_signal == "dff" else "delta_f"
    required.add(signal_field)
    missing = sorted(k for k in required if k not in grp)
    if missing:
        raise RuntimeError(
            f"Cache chunk missing required dataset(s) for retune: "
            f"roi={roi} chunk={chunk_id} missing={missing}"
        )

    time_sec = grp["time_sec"][()]
    sig_raw = grp["sig_raw"][()]
    uv_raw = grp["uv_raw"][()]
    signal_arr = grp[signal_field][()]

    fs_hz = _compute_chunk_fs_hz(time_sec, cfg.target_fs_hz)
    chunk = Chunk(
        chunk_id=int(chunk_id),
        source_file=source_file,
        format="cache",
        time_sec=time_sec,
        uv_raw=uv_raw.reshape(-1, 1),
        sig_raw=sig_raw.reshape(-1, 1),
        fs_hz=fs_hz,
        channel_names=[roi],
        metadata={},
    )
    if signal_field == "dff":
        chunk.dff = signal_arr.reshape(-1, 1)
    else:
        chunk.delta_f = signal_arr.reshape(-1, 1)
    return chunk


def _make_retune_dir(run_dir: str, out_dir: str | None) -> str:
    base = os.path.abspath(out_dir) if out_dir else os.path.join(run_dir, "tuning_retune")
    os.makedirs(base, exist_ok=True)
    run_id = datetime.now().strftime("retune_%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    retune_dir = os.path.join(base, run_id)
    os.makedirs(retune_dir, exist_ok=False)
    return retune_dir


def _write_provenance(
    path: str,
    *,
    run_dir: str,
    phasic_out_dir: str,
    selected_roi: str,
    inspection_chunk_id_requested: int | None,
    inspection_chunk_id_used: int,
    event_signal_used: str,
    completed_evidence: str,
    base_config_path: str,
    base_config: Config,
    overrides: Dict[str, Any],
    classes: Dict[str, list[str]],
) -> None:
    payload = {
        "schema_version": 1,
        "tool": "cache_downstream_retune",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": os.path.abspath(run_dir),
        "source_phasic_out_dir": os.path.abspath(phasic_out_dir),
        "completed_run_evidence": completed_evidence,
        "selected_roi": selected_roi,
        "inspection_chunk_id_requested": inspection_chunk_id_requested,
        "inspection_chunk_id_used": inspection_chunk_id_used,
        "event_signal_used": event_signal_used,
        "base_config_source": os.path.abspath(base_config_path),
        "base_config_snapshot": dataclasses.asdict(base_config),
        "downstream_overrides_applied": overrides,
        "override_classification": classes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_retuned_diagnostics(
    retune_dir: str,
    roi: str,
    features_df: pd.DataFrame,
    median_session_duration_s: float,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    feats_csv = os.path.join(retune_dir, f"retuned_features_{roi}.csv")
    features_df.to_csv(feats_csv, index=False)
    out["retuned_features_csv"] = feats_csv

    per_chunk = (
        features_df.groupby("chunk_id", as_index=False)
        .agg({"peak_count": "sum", "auc": "sum"})
        .sort_values("chunk_id")
    )
    if median_session_duration_s > 0:
        per_chunk["peak_rate_per_min"] = per_chunk["peak_count"] / (median_session_duration_s / 60.0)
    else:
        per_chunk["peak_rate_per_min"] = np.nan

    summary_csv = os.path.join(retune_dir, f"retuned_summary_{roi}.csv")
    per_chunk.to_csv(summary_csv, index=False)
    out["retuned_summary_csv"] = summary_csv

    # Minimal tuning diagnostics (cache-driven, downstream feature based).
    peak_png = os.path.join(retune_dir, f"retuned_peak_count_{roi}.png")
    auc_png = os.path.join(retune_dir, f"retuned_auc_{roi}.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(per_chunk["chunk_id"], per_chunk["peak_count"], marker="o")
    ax.set_xlabel("Chunk ID")
    ax.set_ylabel("Peak Count")
    ax.set_title(f"Retuned Peak Count ({roi})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(peak_png, dpi=150)
    plt.close(fig)
    out["retuned_peak_count_png"] = peak_png

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(per_chunk["chunk_id"], per_chunk["auc"], marker="o")
    ax.set_xlabel("Chunk ID")
    ax.set_ylabel("AUC")
    ax.set_title(f"Retuned AUC ({roi})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(auc_png, dpi=150)
    plt.close(fig)
    out["retuned_auc_png"] = auc_png

    return out


def _compute_threshold(trace_use: np.ndarray, cfg: Config) -> tuple[float, dict[str, float]]:
    clean = trace_use[np.isfinite(trace_use)]
    if len(clean) < 2:
        raise RuntimeError("Cannot compute threshold: fewer than 2 finite samples in selected chunk trace.")

    mu = float(np.mean(clean))
    med = float(np.median(clean))
    sigma = float(np.std(clean))
    mad = float(np.median(np.abs(clean - med)))
    sigma_robust = 1.4826 * mad

    method = cfg.peak_threshold_method
    if method == "absolute":
        threshold = float(getattr(cfg, "peak_threshold_abs", 0.0))
    elif method == "mean_std":
        threshold = mu + float(cfg.peak_threshold_k) * sigma
    elif method == "percentile":
        threshold = float(np.nanpercentile(clean, cfg.peak_threshold_percentile))
    elif method == "median_mad":
        if sigma_robust == 0:
            threshold = med if float(cfg.peak_threshold_k) == 0.0 else float("inf")
        else:
            threshold = med + float(cfg.peak_threshold_k) * sigma_robust
    else:
        raise ValueError(
            f"Unknown peak_threshold_method: {method}. "
            "Supported: ['mean_std', 'percentile', 'median_mad', 'absolute']"
        )

    return threshold, {
        "mean": mu,
        "median": med,
        "std": sigma,
        "mad": mad,
        "sigma_robust": float(sigma_robust),
    }


def _detect_events_for_chunk(chunk: Chunk, cfg: Config) -> Dict[str, Any]:
    signal_field = "dff" if cfg.event_signal == "dff" else "delta_f"
    signal = chunk.dff[:, 0] if signal_field == "dff" else chunk.delta_f[:, 0]

    if getattr(cfg, "peak_pre_filter", "none") == "lowpass":
        trace_use = lowpass_filter(signal, chunk.fs_hz, cfg)
    else:
        trace_use = signal

    threshold, stats = _compute_threshold(trace_use, cfg)
    dist_samples = max(1, int(cfg.peak_min_distance_sec * chunk.fs_hz))

    is_valid_raw = np.isfinite(signal)
    padded = np.concatenate(([False], is_valid_raw, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    event_indices: list[int] = []
    for s, e in zip(starts, ends):
        seg_y = trace_use[s:e]
        seg_valid = np.isfinite(seg_y)
        if not np.any(seg_valid):
            continue
        run_pad = np.concatenate(([False], seg_valid, [False]))
        run_diff = np.diff(run_pad.astype(int))
        run_starts = np.where(run_diff == 1)[0]
        run_ends = np.where(run_diff == -1)[0]
        for rs, re in zip(run_starts, run_ends):
            run_y = seg_y[rs:re]
            if len(run_y) < 2:
                continue
            peaks, _ = find_peaks(run_y, height=threshold, distance=dist_samples)
            if len(peaks):
                event_indices.extend((s + rs + peaks).astype(int).tolist())

    event_idx = np.array(sorted(set(event_indices)), dtype=int)
    event_times = chunk.time_sec[event_idx] if len(event_idx) else np.array([], dtype=float)
    event_values = trace_use[event_idx] if len(event_idx) else np.array([], dtype=float)
    return {
        "chunk_id": int(chunk.chunk_id),
        "source_file": chunk.source_file,
        "signal_field": signal_field,
        "trace_time_sec": chunk.time_sec,
        "trace_used_for_calling": trace_use,
        "event_indices": event_idx,
        "event_times_sec": event_times,
        "event_values": event_values,
        "threshold": float(threshold),
        "threshold_stats": stats,
        "distance_samples": int(dist_samples),
    }


def _write_inspection_diagnostics(
    retune_dir: str,
    roi: str,
    detection: Dict[str, Any],
    cfg: Config,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    chunk_id = int(detection["chunk_id"])
    suffix = f"{roi}_chunk_{chunk_id:03d}"

    events_df = pd.DataFrame(
        {
            "chunk_id": chunk_id,
            "source_file": str(detection["source_file"]),
            "roi": roi,
            "event_signal": detection["signal_field"],
            "event_index": detection["event_indices"],
            "event_time_sec": detection["event_times_sec"],
            "event_value": detection["event_values"],
            "threshold": float(detection["threshold"]),
            "threshold_method": cfg.peak_threshold_method,
            "peak_min_distance_sec": float(cfg.peak_min_distance_sec),
        }
    )
    events_csv = os.path.join(retune_dir, f"retuned_events_{suffix}.csv")
    events_df.to_csv(events_csv, index=False)
    out["retuned_events_csv"] = events_csv

    overlay_png = os.path.join(retune_dir, f"retuned_overlay_{suffix}.png")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = detection["trace_time_sec"]
    y = detection["trace_used_for_calling"]
    ax.plot(x, y, linewidth=1.0, color="steelblue", label=f"{detection['signal_field']} trace")
    if len(detection["event_indices"]):
        ax.scatter(
            detection["event_times_sec"],
            detection["event_values"],
            color="crimson",
            marker="x",
            s=28,
            linewidths=1.0,
            label="Detected events",
            zorder=3,
        )
    ax.axhline(float(detection["threshold"]), color="darkorange", linestyle="--", linewidth=1.0, label="Threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(detection["signal_field"])
    ax.set_title(
        f"Retuned Event Overlay ({roi}) | chunk={chunk_id} | source={detection['source_file']}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(overlay_png, dpi=150)
    plt.close(fig)
    out["retuned_overlay_png"] = overlay_png
    return out


def run_cache_downstream_retune(
    *,
    run_dir: str,
    roi: str,
    overrides: Dict[str, Any],
    chunk_id: int | None = None,
    out_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Execute cache-driven downstream retune for a selected ROI.
    """
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"Run directory does not exist: {run_dir}")
    if not roi.strip():
        raise RuntimeError("ROI must be provided.")

    classes = classify_overrides(overrides)
    if classes["unknown"]:
        raise ValueError(f"Unknown override key(s): {classes['unknown']}")
    if classes["unsupported"]:
        raise ValueError(
            "Unsupported override key(s) for cache-driven downstream retune: "
            f"{classes['unsupported']}. "
            "These are not currently supported in this path."
        )
    if classes["correction_required"]:
        raise ValueError(
            "Override(s) require correction-context recompute and are not allowed in "
            "cache-driven downstream retune: "
            f"{classes['correction_required']}. "
            "Use an explicit recompute-correction workflow."
        )

    completed_evidence = _assert_successful_completed_run(run_dir)

    phasic_out_dir = os.path.join(run_dir, "_analysis", "phasic_out")
    if not os.path.isdir(phasic_out_dir):
        raise RuntimeError(f"Missing phasic analysis directory: {phasic_out_dir}")

    cache_path = os.path.join(phasic_out_dir, "phasic_trace_cache.h5")
    if not os.path.isfile(cache_path):
        raise RuntimeError(f"Missing phasic cache: {cache_path}")

    base_config, base_config_path = _resolve_base_config(phasic_out_dir)
    effective_config = _apply_downstream_overrides(base_config, overrides)

    with open_phasic_cache(cache_path) as cache:
        resolved_roi = resolve_cache_roi(cache, roi)
        chunk_ids = sorted(int(c) for c in list_cache_chunk_ids(cache))
        if not chunk_ids:
            raise RuntimeError(f"No chunks found in phasic cache: {cache_path}")
        if chunk_id is None:
            inspection_chunk_id = chunk_ids[0]
        else:
            inspection_chunk_id = int(chunk_id)
            if inspection_chunk_id not in chunk_ids:
                raise RuntimeError(
                    f"Requested chunk_id={inspection_chunk_id} not present in cache. "
                    f"Available chunk_ids={chunk_ids}"
                )

        retune_dir = _make_retune_dir(run_dir, out_dir)
        with open(os.path.join(retune_dir, "retune_config_effective.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dataclasses.asdict(effective_config), f, sort_keys=True)

        _write_provenance(
            os.path.join(retune_dir, "retune_request.json"),
            run_dir=run_dir,
            phasic_out_dir=phasic_out_dir,
            selected_roi=resolved_roi,
            inspection_chunk_id_requested=chunk_id,
            inspection_chunk_id_used=inspection_chunk_id,
            event_signal_used=effective_config.event_signal,
            completed_evidence=completed_evidence,
            base_config_path=base_config_path,
            base_config=base_config,
            overrides=overrides,
            classes=classes,
        )

        source_files = None
        meta = cache.get("meta")
        if meta is not None and "source_files" in meta:
            source_files = [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in meta["source_files"][()]]

        roi_group = cache.get(f"roi/{resolved_roi}")
        if roi_group is None:
            raise RuntimeError(f"ROI group missing in cache: {resolved_roi}")

        all_rows = []
        durations = []
        inspection_chunk = None
        for idx, cid in enumerate(chunk_ids):
            grp = roi_group.get(f"chunk_{int(cid)}")
            if grp is None:
                raise RuntimeError(f"Missing cache chunk group: roi={resolved_roi} chunk={cid}")
            src = source_files[idx] if source_files and idx < len(source_files) else f"chunk_{int(cid)}"
            chunk = _build_chunk_for_roi(resolved_roi, int(cid), src, grp, effective_config)
            if int(cid) == inspection_chunk_id:
                inspection_chunk = chunk
            if len(chunk.time_sec) >= 2:
                durations.append(float(chunk.time_sec[-1] - chunk.time_sec[0]))
            feats = extract_features(chunk, effective_config)
            all_rows.append(feats)
        if inspection_chunk is None:
            raise RuntimeError(f"Internal error: inspection chunk {inspection_chunk_id} was not loaded.")

    if not all_rows:
        raise RuntimeError("No features were produced from cache retune request.")
    features_df = pd.concat(all_rows, ignore_index=True)
    features_df = features_df[features_df["roi"] == resolved_roi].copy()
    features_df.sort_values(["chunk_id", "roi"], inplace=True)

    median_duration_s = float(np.median(durations)) if durations else 0.0
    artifact_paths = _write_retuned_diagnostics(
        retune_dir=retune_dir,
        roi=resolved_roi,
        features_df=features_df,
        median_session_duration_s=median_duration_s,
    )
    detection = _detect_events_for_chunk(inspection_chunk, effective_config)
    artifact_paths.update(
        _write_inspection_diagnostics(
            retune_dir=retune_dir,
            roi=resolved_roi,
            detection=detection,
            cfg=effective_config,
        )
    )

    result = {
        "retune_dir": retune_dir,
        "selected_roi": resolved_roi,
        "inspection_chunk_id": inspection_chunk_id,
        "inspection_source_file": inspection_chunk.source_file,
        "event_signal_used": effective_config.event_signal,
        "n_chunks": int(features_df["chunk_id"].nunique()) if not features_df.empty else 0,
        "n_rows": int(len(features_df)),
        "downstream_overrides_applied": dict(overrides),
        "artifacts": artifact_paths,
    }
    with open(os.path.join(retune_dir, "retune_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result
