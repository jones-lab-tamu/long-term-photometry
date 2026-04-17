"""
Cache-driven correction-sensitive retune backend.

This module implements an isolated correction recompute path for a completed run:
- validates successful completion provenance
- loads raw phasic cache context (time_sec, sig_raw, uv_raw)
- applies correction-sensitive overrides only
- recomputes baseline/correction/dff/features for one ROI across all chunks
- writes retuned outputs into an isolated subtree
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
import yaml

from photometry_pipeline.config import Config
from photometry_pipeline.core import baseline, regression, normalization, preprocessing
from photometry_pipeline.core.feature_extraction import extract_features
from photometry_pipeline.core.types import Chunk, SessionStats
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.io.hdf5_cache_reader import (
    open_phasic_cache,
    resolve_cache_roi,
)
from photometry_pipeline.viz.display_prep import prepare_centered_common_gain


CORRECTION_RETUNABLE_KEYS = {
    "dynamic_fit_mode",
    "signal_excursion_polarity",
    "baseline_subtract_before_fit",
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
    "robust_event_reject_max_iters",
    "robust_event_reject_residual_z_thresh",
    "robust_event_reject_local_var_window_sec",
    "robust_event_reject_local_var_ratio_thresh",
    "robust_event_reject_min_keep_fraction",
    "adaptive_event_gate_residual_z_thresh",
    "adaptive_event_gate_local_var_window_sec",
    "adaptive_event_gate_local_var_ratio_thresh",
    "adaptive_event_gate_smooth_window_sec",
    "adaptive_event_gate_min_trust_fraction",
    "adaptive_event_gate_freeze_interp_method",
}

DOWNSTREAM_ONLY_KEYS = {
    "event_signal",
    "peak_threshold_method",
    "peak_threshold_k",
    "peak_threshold_percentile",
    "peak_threshold_abs",
    "peak_min_distance_sec",
    "peak_min_prominence_k",
    "peak_min_width_sec",
    "peak_pre_filter",
    "event_auc_baseline",
}

EXPLICITLY_UNSUPPORTED_KEYS = {
    "f0_min_value",
}

_OVERRIDE_VALUE_CASTERS = {
    "dynamic_fit_mode": str,
    "signal_excursion_polarity": str,
    "baseline_subtract_before_fit": str,
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
    "robust_event_reject_max_iters": int,
    "robust_event_reject_residual_z_thresh": float,
    "robust_event_reject_local_var_window_sec": float,
    "robust_event_reject_local_var_ratio_thresh": float,
    "robust_event_reject_min_keep_fraction": float,
    "adaptive_event_gate_residual_z_thresh": float,
    "adaptive_event_gate_local_var_window_sec": float,
    "adaptive_event_gate_local_var_ratio_thresh": float,
    "adaptive_event_gate_smooth_window_sec": float,
    "adaptive_event_gate_min_trust_fraction": float,
    "adaptive_event_gate_freeze_interp_method": str,
    "event_signal": str,
    "peak_threshold_method": str,
    "peak_threshold_k": float,
    "peak_threshold_percentile": float,
    "peak_threshold_abs": float,
    "peak_min_distance_sec": float,
    "peak_min_prominence_k": float,
    "peak_min_width_sec": float,
    "peak_pre_filter": str,
    "event_auc_baseline": str,
    "f0_min_value": float,
}

_CORRECTION_INSPECTION_FIGSIZE = (14.5, 6.0)
_CORRECTION_INSPECTION_DPI = 200

_BOOL_TRUE_TOKENS = {"1", "true", "yes", "on"}
_BOOL_FALSE_TOKENS = {"0", "false", "no", "off"}


def _parse_bool_override(value: Any, *, key: str = "value") -> bool:
    """Parse explicit bool spellings and reject ambiguous values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False

    text = str(value).strip().lower()
    if text in _BOOL_TRUE_TOKENS:
        return True
    if text in _BOOL_FALSE_TOKENS:
        return False
    accepted = ", ".join(
        sorted(_BOOL_FALSE_TOKENS | _BOOL_TRUE_TOKENS, key=lambda x: (len(x), x))
    )
    raise ValueError(
        f"Invalid boolean override value for '{key}': {value!r}. "
        f"Accepted values: {accepted}."
    )


_OVERRIDE_VALUE_CASTERS["baseline_subtract_before_fit"] = (
    lambda value: _parse_bool_override(value, key="baseline_subtract_before_fit")
)


def _true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index spans where mask is True."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(m):
        if flag and start is None:
            start = idx
        elif (not flag) and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, m.size - 1))
    return spans


def parse_key_value_overrides(items: Iterable[str]) -> Dict[str, Any]:
    """Parse KEY=VALUE CLI items into typed overrides."""
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
            raise ValueError(
                f"Invalid value for override '{key}': {value}. {exc}"
            ) from exc
    return parsed


def classify_overrides(overrides: Dict[str, Any]) -> Dict[str, list[str]]:
    """Classify override keys by correction-retune boundary classes."""
    keys = set(overrides.keys())
    return {
        "correction_supported": sorted(keys & CORRECTION_RETUNABLE_KEYS),
        "downstream_only": sorted(keys & DOWNSTREAM_ONLY_KEYS),
        "unsupported": sorted(keys & EXPLICITLY_UNSUPPORTED_KEYS),
        "unknown": sorted(
            keys
            - CORRECTION_RETUNABLE_KEYS
            - DOWNSTREAM_ONLY_KEYS
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
    cfg_path = os.path.join(phasic_out_dir, "config_used.yaml")
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"Missing base config snapshot: {cfg_path}")
    return Config.from_yaml(cfg_path), cfg_path


def _coerce_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in overrides.items():
        if value is None:
            out[key] = None
            continue
        caster = _OVERRIDE_VALUE_CASTERS.get(key, str)
        out[key] = caster(value)
    return out


def _apply_correction_overrides(base_cfg: Config, overrides: Dict[str, Any]) -> Config:
    cfg_dict = dataclasses.asdict(base_cfg)
    cfg_dict.update(_coerce_overrides(overrides))
    return Config(**cfg_dict)


def _compute_chunk_fs_hz(time_sec: np.ndarray, fallback: float) -> float:
    if len(time_sec) < 2:
        return float(fallback)
    dt = np.diff(time_sec)
    finite = dt[np.isfinite(dt) & (dt > 0)]
    if len(finite) == 0:
        return float(fallback)
    return float(1.0 / np.median(finite))


def _make_retune_dir(run_dir: str, out_dir: str | None) -> str:
    base = os.path.abspath(out_dir) if out_dir else os.path.join(run_dir, "tuning_correction_retune")
    os.makedirs(base, exist_ok=True)
    run_id = datetime.now().strftime("retune_%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    retune_dir = os.path.join(base, run_id)
    os.makedirs(retune_dir, exist_ok=False)
    return retune_dir


def _load_roi_raw_entries(cache, roi: str, cfg: Config) -> list[dict[str, Any]]:
    meta = cache.get("meta")
    source_files: list[str] | None = None
    if meta is not None and "source_files" in meta:
        source_files = [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in meta["source_files"][()]]

    roi_group = cache.get(f"roi/{roi}")
    if roi_group is None:
        raise RuntimeError(f"ROI group missing in cache: {roi}")
    chunk_ids = sorted(
        int(name.split("_", 1)[1])
        for name in roi_group.keys()
        if str(name).startswith("chunk_")
    )
    if not chunk_ids:
        raise RuntimeError(f"No chunks found in phasic cache for ROI {roi}.")

    entries: list[dict[str, Any]] = []
    for idx, cid in enumerate(chunk_ids):
        grp = roi_group.get(f"chunk_{cid}")
        if grp is None:
            raise RuntimeError(f"Missing cache chunk group for ROI {roi}: chunk_{cid}")

        required = ["time_sec", "sig_raw", "uv_raw"]
        missing = [name for name in required if name not in grp]
        if missing:
            raise RuntimeError(
                f"Cache chunk missing required raw dataset(s): "
                f"roi={roi} chunk={cid} missing={missing}"
            )

        time_sec = grp["time_sec"][()]
        sig_raw = grp["sig_raw"][()]
        uv_raw = grp["uv_raw"][()]
        fs_hz = _compute_chunk_fs_hz(time_sec, cfg.target_fs_hz)
        source_file = source_files[idx] if source_files and idx < len(source_files) else f"chunk_{cid}"

        entries.append(
            {
                "chunk_id": int(cid),
                "source_file": source_file,
                "time_sec": np.asarray(time_sec),
                "sig_raw": np.asarray(sig_raw),
                "uv_raw": np.asarray(uv_raw),
                "fs_hz": float(fs_hz),
            }
        )

    return entries


def _compute_roi_baseline_stats(entries: list[dict[str, Any]], roi: str, cfg: Config) -> SessionStats:
    stats = SessionStats()
    method = cfg.baseline_method

    if method == "uv_raw_percentile_session":
        reservoir = baseline.DeterministicReservoir(seed=cfg.seed)
        for rec in entries:
            reservoir.add(roi, rec["uv_raw"])
        stats.method_used = method
        stats.f0_values[roi] = float(reservoir.get_percentile(roi, cfg.baseline_percentile))
        return stats

    if method == "uv_globalfit_percentile_session":
        accumulator = baseline.GlobalFitAccumulator()
        for rec in entries:
            uv_raw = rec["uv_raw"]
            sig_raw = rec["sig_raw"]
            fs_hz = rec["fs_hz"]
            uv_filt, _ = preprocessing.lowpass_filter_with_meta(uv_raw, fs_hz, cfg)
            sig_filt, _ = preprocessing.lowpass_filter_with_meta(sig_raw, fs_hz, cfg)
            accumulator.add(roi, uv_filt, sig_filt)

        solved = accumulator.solve()
        params = solved.get(roi, {"a": 1.0, "b": 0.0})
        stats.global_fit_params[roi] = {"a": float(params["a"]), "b": float(params["b"])}

        reservoir = baseline.DeterministicReservoir(seed=cfg.seed)
        a = float(params["a"])
        b = float(params["b"])
        for rec in entries:
            uv_est = a * rec["uv_raw"] + b
            reservoir.add(roi, uv_est)

        stats.method_used = method
        stats.f0_values[roi] = float(reservoir.get_percentile(roi, cfg.baseline_percentile))
        return stats

    raise RuntimeError(
        "Unsupported baseline_method for correction retune: "
        f"{method}."
    )


def _recompute_roi_chunks(
    entries: list[dict[str, Any]],
    roi: str,
    cfg: Config,
    stats: SessionStats,
) -> tuple[list[Chunk], pd.DataFrame, list[float]]:
    chunks: list[Chunk] = []
    features_rows: list[pd.DataFrame] = []
    durations: list[float] = []

    for rec in entries:
        chunk = Chunk(
            chunk_id=int(rec["chunk_id"]),
            source_file=str(rec["source_file"]),
            format="cache",
            time_sec=np.asarray(rec["time_sec"]),
            uv_raw=np.asarray(rec["uv_raw"]).reshape(-1, 1),
            sig_raw=np.asarray(rec["sig_raw"]).reshape(-1, 1),
            fs_hz=float(rec["fs_hz"]),
            channel_names=[roi],
            metadata={},
        )

        chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
        chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)
        uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode="phasic")
        chunk.uv_fit = uv_fit
        chunk.delta_f = delta_f
        chunk.dff = normalization.compute_dff(chunk, stats, cfg)

        feats_df = extract_features(chunk, cfg)
        features_rows.append(feats_df)
        chunks.append(chunk)

        if len(chunk.time_sec) >= 2:
            durations.append(float(chunk.time_sec[-1] - chunk.time_sec[0]))

    if not chunks:
        raise RuntimeError("No chunks were recomputed for selected ROI.")

    all_features = pd.concat(features_rows, ignore_index=True)
    all_features = all_features[all_features["roi"] == roi].copy()
    all_features.sort_values(["chunk_id", "roi"], inplace=True)
    return chunks, all_features, durations


def _write_features_artifacts(
    retune_dir: str,
    roi: str,
    features_df: pd.DataFrame,
    median_session_duration_s: float,
) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}

    features_csv = os.path.join(retune_dir, f"retuned_features_{roi}.csv")
    features_df.to_csv(features_csv, index=False)
    artifacts["retuned_features_csv"] = features_csv

    summary_df = (
        features_df.groupby("chunk_id", as_index=False)
        .agg({"peak_count": "sum", "auc": "sum"})
        .sort_values("chunk_id")
    )
    if median_session_duration_s > 0:
        summary_df["peak_rate_per_min"] = summary_df["peak_count"] / (median_session_duration_s / 60.0)
    else:
        summary_df["peak_rate_per_min"] = np.nan

    summary_csv = os.path.join(retune_dir, f"retuned_summary_{roi}.csv")
    summary_df.to_csv(summary_csv, index=False)
    artifacts["retuned_summary_csv"] = summary_csv

    return artifacts


def _write_correction_inspection(
    retune_dir: str,
    roi: str,
    chunk: Chunk,
) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    cid = int(chunk.chunk_id)
    suffix = f"{roi}_chunk_{cid:03d}"

    t = np.asarray(chunk.time_sec)
    sig = np.asarray(chunk.sig_raw[:, 0])
    uv = np.asarray(chunk.uv_raw[:, 0])
    fit = np.asarray(chunk.uv_fit[:, 0]) if chunk.uv_fit is not None else np.full_like(sig, np.nan)
    delta_f = np.asarray(chunk.delta_f[:, 0]) if chunk.delta_f is not None else np.full_like(sig, np.nan)
    dff = np.asarray(chunk.dff[:, 0]) if chunk.dff is not None else np.full_like(sig, np.nan)
    event_reject_info = {}
    adaptive_info = {}
    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        by_roi = chunk.metadata.get("dynamic_fit_event_reject", {})
        if isinstance(by_roi, dict):
            event_reject_info = by_roi.get(roi, {}) or {}
        adaptive_by_roi = chunk.metadata.get("dynamic_fit_adaptive_event_gated", {})
        if isinstance(adaptive_by_roi, dict):
            adaptive_info = adaptive_by_roi.get(roi, {}) or {}
    excluded_mask = np.asarray(event_reject_info.get("excluded_mask", []), dtype=bool)
    if excluded_mask.shape != sig.shape:
        excluded_mask = np.zeros(sig.shape, dtype=bool)
    final_coef = event_reject_info.get("final_coef", {}) if isinstance(event_reject_info, dict) else {}
    iter_summaries = (
        event_reject_info.get("iteration_summaries", [])
        if isinstance(event_reject_info, dict)
        else []
    )
    fit_mode_resolved = ""
    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
        fit_mode_resolved = str(chunk.metadata.get("dynamic_fit_mode_resolved", "") or "")
    robust_mode = (
        fit_mode_resolved == "robust_global_event_reject"
        or bool(event_reject_info)
    )
    adaptive_mode = (
        fit_mode_resolved == "adaptive_event_gated_regression"
        or bool(adaptive_info)
    )
    keep_fraction = (
        event_reject_info.get("final_keep_fraction", None)
        if isinstance(event_reject_info, dict)
        else None
    )
    if keep_fraction is None and isinstance(final_coef, dict):
        keep_fraction = final_coef.get("keep_fraction", None)
    try:
        keep_fraction_value = float(keep_fraction) if keep_fraction is not None else np.nan
    except Exception:
        keep_fraction_value = np.nan
    if robust_mode and not np.isfinite(keep_fraction_value):
        keep_fraction_value = 1.0
    n_iters = (
        int(event_reject_info.get("n_iterations_completed", 0))
        if isinstance(event_reject_info, dict)
        else 0
    )
    if n_iters <= 0:
        n_iters = len(iter_summaries)
    fallback_to_global = bool(event_reject_info.get("fallback_to_global_linear", False))
    fallback_failed = bool(event_reject_info.get("fallback_failed", False))
    fallback_status = "yes_failed" if fallback_failed else ("yes" if fallback_to_global else "no")
    robust_backend_used = str(event_reject_info.get("robust_fit_backend_used", "")) if isinstance(event_reject_info, dict) else ""
    excluded_count = int(np.sum(excluded_mask))
    finite_sig = np.isfinite(sig)
    excluded_fraction = (
        float(excluded_count) / float(max(1, int(np.sum(finite_sig))))
        if robust_mode
        else np.nan
    )
    if robust_mode:
        artifacts["retuned_correction_inspection_robust_diagnostics"] = {
            "fit_mode_resolved": fit_mode_resolved or "robust_global_event_reject",
            "iterations_completed": int(n_iters),
            "keep_fraction": float(keep_fraction_value),
            "fallback_to_global_linear": bool(fallback_to_global),
            "fallback_failed": bool(fallback_failed),
            "fallback_status": fallback_status,
            "robust_fit_backend_used": robust_backend_used,
            "excluded_count": int(excluded_count),
            "excluded_fraction": float(excluded_fraction),
        }
    trusted_mask = np.asarray(adaptive_info.get("trusted_mask", []), dtype=bool)
    if trusted_mask.shape != sig.shape:
        trusted_mask = np.zeros(sig.shape, dtype=bool)
    gated_mask = np.asarray(adaptive_info.get("gated_mask", []), dtype=bool)
    if gated_mask.shape != sig.shape:
        gated_mask = np.zeros(sig.shape, dtype=bool)
    trust_fraction = adaptive_info.get("trust_fraction", None)
    gated_fraction = adaptive_info.get("gated_fraction", None)
    try:
        trust_fraction_value = float(trust_fraction) if trust_fraction is not None else np.nan
    except Exception:
        trust_fraction_value = np.nan
    try:
        gated_fraction_value = float(gated_fraction) if gated_fraction is not None else np.nan
    except Exception:
        gated_fraction_value = np.nan
    if adaptive_mode:
        if not np.isfinite(trust_fraction_value):
            trust_fraction_value = float(np.mean(trusted_mask)) if trusted_mask.size else np.nan
        if not np.isfinite(gated_fraction_value):
            gated_fraction_value = float(np.mean(gated_mask)) if gated_mask.size else np.nan
    fallback_mode = str(adaptive_info.get("fallback_mode", "")).strip() if isinstance(adaptive_info, dict) else ""
    fallback_failed_adaptive = bool(adaptive_info.get("fallback_failed", False))
    fallback_status_adaptive = (
        "yes_failed"
        if fallback_failed_adaptive
        else ("yes" if fallback_mode and fallback_mode != "none" else "no")
    )
    if adaptive_mode:
        artifacts["retuned_correction_inspection_adaptive_diagnostics"] = {
            "fit_mode_resolved": fit_mode_resolved or "adaptive_event_gated_regression",
            "trust_fraction": float(trust_fraction_value) if np.isfinite(trust_fraction_value) else float("nan"),
            "gated_fraction": float(gated_fraction_value) if np.isfinite(gated_fraction_value) else float("nan"),
            "fallback_mode": fallback_mode or "none",
            "fallback_failed": bool(fallback_failed_adaptive),
            "fallback_status": fallback_status_adaptive,
            "n_trusted": int(np.sum(trusted_mask)),
            "n_gated": int(np.sum(gated_mask)),
        }

    csv_path = os.path.join(retune_dir, f"retuned_correction_session_{suffix}.csv")
    pd.DataFrame(
        {
            "chunk_id": cid,
            "source_file": str(chunk.source_file),
            "t_s": t,
            "sig_raw": sig,
            "uv_raw": uv,
            "fit_ref": fit,
            "delta_f": delta_f,
            "dff": dff,
        }
    ).to_csv(csv_path, index=False)
    artifacts["retuned_correction_session_csv"] = csv_path

    try:
        sig_centered, uv_centered = prepare_centered_common_gain(sig, uv)
    except ValueError:
        sig_centered = np.asarray(sig, dtype=np.float64).copy()
        uv_centered = np.asarray(uv, dtype=np.float64).copy()
        sig_finite = np.isfinite(sig_centered)
        uv_finite = np.isfinite(uv_centered)
        if np.any(sig_finite):
            sig_centered = sig_centered - float(np.median(sig_centered[sig_finite]))
        if np.any(uv_finite):
            uv_centered = uv_centered - float(np.median(uv_centered[uv_finite]))

    panel_specs: list[tuple[str, str]] = [
        ("raw", "Raw absolute sig/iso"),
        ("centered", "Centered common-gain sig/iso"),
        ("fit", "Dynamic fit"),
        ("dff", "Final corrected dF/F"),
    ]
    panel_paths: list[str] = []
    panel_labels: list[str] = []

    for panel_key, panel_label in panel_specs:
        panel_path = os.path.join(
            retune_dir, f"retuned_correction_inspection_{suffix}_{panel_key}.png"
        )
        fig, ax = plt.subplots(1, 1, figsize=_CORRECTION_INSPECTION_FIGSIZE)
        ax.set_title(
            f"Correction Retune Inspection ({roi}) | chunk={cid} | {panel_label} | source={chunk.source_file}"
        )
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.25)

        if panel_key == "raw":
            ax.plot(t, sig, color="forestgreen", linewidth=0.9, label="sig_raw")
            ax.plot(t, uv, color="purple", linewidth=0.8, alpha=0.8, label="uv_raw")
            ax.set_ylabel("Raw output (V)")
        elif panel_key == "centered":
            ax.plot(
                t,
                sig_centered,
                color="forestgreen",
                linewidth=0.9,
                label="sig_raw (centered)",
            )
            ax.plot(
                t,
                uv_centered,
                color="purple",
                linewidth=0.8,
                alpha=0.8,
                label="uv_raw (centered)",
            )
            ax.set_ylabel("Centered (V)")
        elif panel_key == "fit":
            ax.plot(t, sig, color="forestgreen", linewidth=0.9, label="sig_raw")
            ax.plot(
                t,
                fit,
                color="black",
                linewidth=0.9,
                linestyle="--",
                label="fit_ref",
            )
            if np.any(excluded_mask):
                ax.scatter(
                    t[excluded_mask],
                    sig[excluded_mask],
                    s=8,
                    c="crimson",
                    alpha=0.45,
                    linewidths=0.0,
                    label="excluded points",
                )
            keep_pct = 100.0 * keep_fraction_value if np.isfinite(keep_fraction_value) else np.nan
            if robust_mode:
                ax.text(
                    0.01,
                    0.99,
                    (
                        f"iters: {int(n_iters)}\n"
                        f"keep: {keep_pct:.1f}%\n"
                        f"fallback: {fallback_status}"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "0.7"},
                )
            if adaptive_mode and np.any(gated_mask):
                gated_spans = _true_spans(gated_mask)
                first_label = True
                for start_idx, end_idx in gated_spans:
                    if start_idx >= t.size or end_idx >= t.size:
                        continue
                    t0 = float(t[start_idx])
                    t1 = float(t[end_idx])
                    if not np.isfinite(t0) or not np.isfinite(t1):
                        continue
                    ax.axvspan(
                        t0,
                        t1,
                        facecolor="#FFD166",
                        alpha=0.22,
                        linewidth=0.0,
                        label="gated/frozen region" if first_label else None,
                    )
                    first_label = False
            if adaptive_mode:
                trust_pct = 100.0 * trust_fraction_value if np.isfinite(trust_fraction_value) else np.nan
                gated_pct = 100.0 * gated_fraction_value if np.isfinite(gated_fraction_value) else np.nan
                ax.text(
                    0.01,
                    0.99,
                    (
                        f"trust: {trust_pct:.1f}%\n"
                        f"gated: {gated_pct:.1f}%\n"
                        f"fallback: {fallback_status_adaptive}"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "0.7"},
                )
            if robust_mode and np.isfinite(keep_pct):
                ax.set_title(
                    f"Correction Retune Inspection ({roi}) | chunk={cid} | {panel_label} "
                    f"| keep={keep_pct:.1f}% | iters={int(n_iters)} | fallback={fallback_status} "
                    f"| source={chunk.source_file}"
                )
            elif adaptive_mode:
                trust_pct = 100.0 * trust_fraction_value if np.isfinite(trust_fraction_value) else np.nan
                gated_pct = 100.0 * gated_fraction_value if np.isfinite(gated_fraction_value) else np.nan
                ax.set_title(
                    f"Correction Retune Inspection ({roi}) | chunk={cid} | {panel_label} "
                    f"| trust={trust_pct:.1f}% | gated={gated_pct:.1f}% | fallback={fallback_status_adaptive} "
                    f"| source={chunk.source_file}"
                )
            elif np.isfinite(keep_pct):
                ax.set_title(
                    f"Correction Retune Inspection ({roi}) | chunk={cid} | {panel_label} "
                    f"| keep={keep_pct:.1f}% | iters={len(iter_summaries)} | source={chunk.source_file}"
                )
            ax.set_ylabel("Fit view (V)")
        else:
            ax.plot(t, dff, color="darkorange", linewidth=0.9, label="dff")
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
            ax.set_ylabel("dF/F")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(panel_path, dpi=_CORRECTION_INSPECTION_DPI)
        plt.close(fig)
        panel_paths.append(panel_path)
        panel_labels.append(panel_label)

    artifacts["retuned_correction_inspection_pngs"] = panel_paths
    artifacts["retuned_correction_inspection_panel_labels"] = panel_labels
    # Backward-compatibility key for existing consumers expecting a single image path.
    artifacts["retuned_correction_inspection_png"] = panel_paths[0]
    artifacts["retuned_correction_inspection_raw_png"] = panel_paths[0]
    artifacts["retuned_correction_inspection_centered_png"] = panel_paths[1]
    artifacts["retuned_correction_inspection_fit_png"] = panel_paths[2]
    artifacts["retuned_correction_inspection_dff_png"] = panel_paths[3]

    return artifacts


def _write_provenance(
    path: str,
    *,
    run_dir: str,
    phasic_out_dir: str,
    selected_roi: str,
    inspection_chunk_id_requested: int | None,
    inspection_chunk_id_used: int,
    completed_evidence: str,
    base_config_path: str,
    base_config: Config,
    overrides: Dict[str, Any],
    classes: Dict[str, list[str]],
) -> None:
    payload = {
        "schema_version": 1,
        "tool": "cache_correction_retune",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": os.path.abspath(run_dir),
        "source_phasic_out_dir": os.path.abspath(phasic_out_dir),
        "completed_run_evidence": completed_evidence,
        "selected_roi": selected_roi,
        "inspection_chunk_id_requested": inspection_chunk_id_requested,
        "inspection_chunk_id_used": inspection_chunk_id_used,
        "correction_overrides_applied": dict(overrides),
        "override_classification": classes,
        "base_config_source": os.path.abspath(base_config_path),
        "base_config_snapshot": dataclasses.asdict(base_config),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_cache_correction_retune(
    *,
    run_dir: str,
    roi: str,
    overrides: Dict[str, Any],
    chunk_id: int | None = None,
    out_dir: str | None = None,
) -> Dict[str, Any]:
    """Execute correction-sensitive cache retune for selected ROI across all chunks."""
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"Run directory does not exist: {run_dir}")
    if not roi or not roi.strip():
        raise RuntimeError("ROI must be provided.")

    classes = classify_overrides(overrides)
    if classes["downstream_only"]:
        raise ValueError(
            "Downstream-only override(s) are not allowed in correction retune: "
            f"{classes['downstream_only']}. "
            "Use the cache_downstream_retune backend for downstream event-detection knobs."
        )
    if classes["unsupported"]:
        raise ValueError(
            "Unsupported override key(s) for correction retune: "
            f"{classes['unsupported']}."
        )
    if classes["unknown"]:
        raise ValueError(f"Unknown override key(s): {classes['unknown']}")

    completed_evidence = _assert_successful_completed_run(run_dir)

    phasic_out_dir = os.path.join(run_dir, "_analysis", "phasic_out")
    if not os.path.isdir(phasic_out_dir):
        raise RuntimeError(f"Missing phasic analysis directory: {phasic_out_dir}")

    cache_path = os.path.join(phasic_out_dir, "phasic_trace_cache.h5")
    if not os.path.isfile(cache_path):
        raise RuntimeError(f"Missing phasic cache: {cache_path}")

    base_config, base_config_path = _resolve_base_config(phasic_out_dir)
    effective_config = _apply_correction_overrides(base_config, overrides)

    with open_phasic_cache(cache_path) as cache:
        resolved_roi = resolve_cache_roi(cache, roi)
        entries = _load_roi_raw_entries(cache, resolved_roi, effective_config)

    available_chunk_ids = sorted(int(rec["chunk_id"]) for rec in entries)
    if chunk_id is None:
        inspection_chunk_id = available_chunk_ids[0]
    else:
        inspection_chunk_id = int(chunk_id)
        if inspection_chunk_id not in available_chunk_ids:
            raise RuntimeError(
                f"Requested chunk_id={inspection_chunk_id} not present for ROI {resolved_roi}. "
                f"Available chunk_ids={available_chunk_ids}"
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
        completed_evidence=completed_evidence,
        base_config_path=base_config_path,
        base_config=base_config,
        overrides=overrides,
        classes=classes,
    )

    stats = _compute_roi_baseline_stats(entries, resolved_roi, effective_config)
    chunks, features_df, durations = _recompute_roi_chunks(entries, resolved_roi, effective_config, stats)

    retune_cache_path = os.path.join(retune_dir, f"retuned_correction_trace_cache_{resolved_roi}.h5")
    with Hdf5TraceCacheWriter(retune_cache_path, "phasic", effective_config) as writer:
        for chunk in chunks:
            writer.add_chunk(chunk, chunk_id=int(chunk.chunk_id), source_file=str(chunk.source_file))

    median_duration_s = float(np.median(durations)) if durations else 0.0
    artifacts: Dict[str, str] = {
        "retuned_correction_cache_h5": retune_cache_path,
    }
    artifacts.update(_write_features_artifacts(retune_dir, resolved_roi, features_df, median_duration_s))

    inspection_chunk = next((c for c in chunks if int(c.chunk_id) == int(inspection_chunk_id)), None)
    if inspection_chunk is None:
        raise RuntimeError(f"Internal error: inspection chunk {inspection_chunk_id} was not recomputed.")
    artifacts.update(_write_correction_inspection(retune_dir, resolved_roi, inspection_chunk))

    baseline_snapshot = {
        "method_used": stats.method_used,
        "f0_values": stats.f0_values,
        "global_fit_params": stats.global_fit_params,
    }
    baseline_path = os.path.join(retune_dir, "retuned_baseline_snapshot.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_snapshot, f, indent=2, sort_keys=True)
    artifacts["retuned_baseline_snapshot_json"] = baseline_path

    result = {
        "retune_dir": retune_dir,
        "selected_roi": resolved_roi,
        "inspection_chunk_id": int(inspection_chunk_id),
        "inspection_source_file": str(inspection_chunk.source_file),
        "n_chunks": int(len(chunks)),
        "n_rows": int(len(features_df)),
        "correction_overrides_applied": dict(overrides),
        "artifacts": artifacts,
    }
    with open(os.path.join(retune_dir, "retune_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    return result
