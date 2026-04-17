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
import hashlib
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
from photometry_pipeline.core.feature_extraction import (
    apply_peak_prefilter,
    compute_detection_threshold_bounds,
    extract_features,
    get_peak_indices_for_trace,
    normalize_signal_excursion_polarity,
)
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    open_phasic_cache,
    resolve_cache_roi,
)


DOWNSTREAM_RETUNABLE_KEYS = {
    "event_signal",
    "signal_excursion_polarity",
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
    "signal_excursion_polarity": str,
    "peak_threshold_method": str,
    "peak_threshold_k": float,
    "peak_threshold_percentile": float,
    "peak_threshold_abs": float,
    "peak_min_distance_sec": float,
    "peak_min_prominence_k": float,
    "peak_min_width_sec": float,
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


_RETUNE_DEBUG_ENV = "PHOTOMETRY_RETUNE_DEBUG"
_RETUNE_OVERLAY_FIGSIZE = (14.0, 5.0)
_RETUNE_OVERLAY_DPI = 220


def _retune_debug_enabled() -> bool:
    value = os.environ.get(_RETUNE_DEBUG_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return _sha256_bytes(f.read())


def _trace_summary(trace: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(trace, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    payload: Dict[str, Any] = {
        "n_samples": int(arr.size),
        "trace_sha256_f64": _sha256_bytes(np.ascontiguousarray(arr).tobytes()),
    }
    if finite.size:
        payload.update(
            {
                "trace_min": float(np.min(finite)),
                "trace_max": float(np.max(finite)),
                "trace_mean": float(np.mean(finite)),
                "trace_std": float(np.std(finite)),
            }
        )
    else:
        payload.update(
            {
                "trace_min": float("nan"),
                "trace_max": float("nan"),
                "trace_mean": float("nan"),
                "trace_std": float("nan"),
            }
        )
    return payload


def _normalize_retune_prefilter_mode(mode_raw: str) -> str:
    """
    Retune semantics intentionally collapse legacy `lowpass` into `smooth`.

    Post-run tuning is a visual tuning surface; this canonicalization keeps
    backward compatibility for old configs while enforcing honest UI semantics.
    """
    mode = str(mode_raw or "none").strip().lower()
    if mode == "none":
        return "none"
    if mode in {"smooth", "lowpass"}:
        return "smooth"
    return "none"


def _signal_excursion_polarity_interpretation(mode_raw: str) -> str:
    mode = normalize_signal_excursion_polarity(mode_raw)
    if mode == "negative":
        return "Downward excursions only."
    if mode == "both":
        return "Two-tailed: upward and downward excursions."
    return "Upward excursions only."


def _event_auc_semantics_text(mode_raw: str) -> str:
    mode = normalize_signal_excursion_polarity(mode_raw)
    if mode == "negative":
        return "Event AUC is integrated as negative area below baseline."
    if mode == "both":
        return "Event AUC is signed net area around baseline (two-tailed)."
    return "Event AUC is integrated as positive area above baseline."


def _write_backend_debug_record(retune_dir: str, payload: Dict[str, Any]) -> None:
    if not _retune_debug_enabled():
        return
    path = os.path.join(retune_dir, "retune_preview_debug_backend.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(
        "RETUNE_DEBUG BACKEND "
        f"chunk={payload.get('chunk_id')} "
        f"prefilter={payload.get('peak_pre_filter')} "
        f"trace_hash={payload.get('trace_sha256_f64')} "
        f"overlay_hash={payload.get('overlay_png_sha256')}"
    )


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


def _resolve_chunk_fs_hz(grp, time_sec: np.ndarray, fallback: float) -> float:
    """
    Resolve chunk sampling rate for retune preview/detection.

    Prefer cache metadata (`grp.attrs['fs_hz']`) when present and valid. This
    preserves the original analyzed fs even when legacy/coarse `time_sec`
    storage would under-estimate fs and weaken retune prefiltering effects.
    """
    fs_attr = grp.attrs.get("fs_hz")
    if fs_attr is not None:
        try:
            fs_attr_f = float(fs_attr)
            if np.isfinite(fs_attr_f) and fs_attr_f > 0:
                return fs_attr_f
        except (TypeError, ValueError):
            pass
    return _compute_chunk_fs_hz(time_sec, fallback)


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

    fs_hz = _resolve_chunk_fs_hz(grp, time_sec, cfg.target_fs_hz)
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
    signal_excursion_polarity_used: str,
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
        "signal_excursion_polarity_used": signal_excursion_polarity_used,
        "signal_excursion_polarity_interpretation": _signal_excursion_polarity_interpretation(
            signal_excursion_polarity_used
        ),
        "event_auc_semantics": _event_auc_semantics_text(signal_excursion_polarity_used),
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
    signal_excursion_polarity: str,
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
    per_chunk["auc_signed"] = per_chunk["auc"]
    per_chunk["signal_excursion_polarity"] = str(
        normalize_signal_excursion_polarity(signal_excursion_polarity)
    )
    per_chunk["event_auc_semantics"] = _event_auc_semantics_text(signal_excursion_polarity)

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
    ax.set_ylabel("Signed Event AUC")
    ax.set_title(
        f"Retuned Event AUC ({roi}) | polarity={normalize_signal_excursion_polarity(signal_excursion_polarity)}"
    )
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
    polarity = normalize_signal_excursion_polarity(
        str(getattr(cfg, "signal_excursion_polarity", "positive"))
    )
    bounds = compute_detection_threshold_bounds(clean, cfg)
    threshold_upper = float(bounds["upper"])
    threshold_lower = float(bounds["lower"])
    if polarity == "positive":
        threshold_active = threshold_upper
    elif polarity == "negative":
        threshold_active = threshold_lower
    else:
        threshold_active = threshold_upper
    return float(threshold_active), {
        "mean": float(bounds["mean"]),
        "median": float(bounds["median"]),
        "std": float(bounds["std"]),
        "mad": float(bounds["mad"]),
        "sigma_robust": float(bounds["sigma_robust"]),
        "threshold_upper": float(threshold_upper),
        "threshold_lower": float(threshold_lower),
        "threshold_active": float(threshold_active),
        "signal_excursion_polarity": str(polarity),
    }


def _resolve_prefilter_config_for_chunk(
    cfg: Config,
    fs_hz: float,
    n_samples: int | None = None,
) -> Config:
    """
    Resolve per-chunk prefilter config for downstream tuning preview/detection.
    """
    _ = fs_hz
    _ = n_samples
    mode = _normalize_retune_prefilter_mode(getattr(cfg, "peak_pre_filter", "none"))
    if mode == str(getattr(cfg, "peak_pre_filter", "none")).strip().lower():
        return cfg
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict["peak_pre_filter"] = mode
    return Config(**cfg_dict)


def _detect_events_for_chunk(chunk: Chunk, cfg: Config) -> Dict[str, Any]:
    cfg_for_chunk = _resolve_prefilter_config_for_chunk(
        cfg,
        chunk.fs_hz,
        n_samples=len(chunk.time_sec),
    )
    signal_field = "dff" if cfg.event_signal == "dff" else "delta_f"
    signal = chunk.dff[:, 0] if signal_field == "dff" else chunk.delta_f[:, 0]

    trace_use, prefilter_meta = apply_peak_prefilter(signal, chunk.fs_hz, cfg_for_chunk)
    peak_pre_filter = str(prefilter_meta.get("mode", getattr(cfg_for_chunk, "peak_pre_filter", "none")))

    polarity = normalize_signal_excursion_polarity(
        str(getattr(cfg_for_chunk, "signal_excursion_polarity", "positive"))
    )
    threshold, stats = _compute_threshold(trace_use, cfg_for_chunk)
    threshold_upper = float(stats.get("threshold_upper", threshold))
    threshold_lower = float(stats.get("threshold_lower", np.nan))
    threshold_active = float(stats.get("threshold_active", threshold))
    dist_samples = max(1, int(cfg_for_chunk.peak_min_distance_sec * chunk.fs_hz))
    event_idx, event_polarities_raw = get_peak_indices_for_trace(
        signal,
        chunk.fs_hz,
        cfg_for_chunk,
        trace_use=trace_use,
        threshold=threshold_upper,
        threshold_lower=threshold_lower,
        return_polarities=True,
    )
    event_polarities = (
        np.where(event_polarities_raw > 0, "positive", "negative").astype(object)
        if len(event_idx)
        else np.array([], dtype=object)
    )
    event_times = chunk.time_sec[event_idx] if len(event_idx) else np.array([], dtype=float)
    event_values = trace_use[event_idx] if len(event_idx) else np.array([], dtype=float)
    return {
        "chunk_id": int(chunk.chunk_id),
        "source_file": chunk.source_file,
        "fs_hz": float(chunk.fs_hz),
        "signal_field": signal_field,
        "trace_time_sec": chunk.time_sec,
        "trace_used_for_calling": trace_use,
        "event_indices": event_idx,
        "event_times_sec": event_times,
        "event_values": event_values,
        "event_polarities": event_polarities,
        "threshold": float(threshold_active),
        "threshold_upper": float(threshold_upper),
        "threshold_lower": float(threshold_lower),
        "threshold_stats": stats,
        "distance_samples": int(dist_samples),
        "signal_excursion_polarity": str(polarity),
        "signal_excursion_polarity_interpretation": _signal_excursion_polarity_interpretation(
            polarity
        ),
        "event_auc_semantics": _event_auc_semantics_text(polarity),
        "peak_pre_filter": peak_pre_filter,
        "prefilter_applied": bool(prefilter_meta.get("prefilter_applied", False)),
        "savgol_window_length": (
            int(prefilter_meta["savgol_window_length"])
            if prefilter_meta.get("savgol_window_length") is not None
            else np.nan
        ),
        "savgol_polyorder": (
            int(prefilter_meta["savgol_polyorder"])
            if prefilter_meta.get("savgol_polyorder") is not None
            else np.nan
        ),
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
            "peak_pre_filter": str(detection.get("peak_pre_filter", "none")),
            "prefilter_applied": bool(detection.get("prefilter_applied", False)),
            "savgol_window_length": float(detection.get("savgol_window_length", np.nan)),
            "savgol_polyorder": float(detection.get("savgol_polyorder", np.nan)),
            "event_index": detection["event_indices"],
            "event_time_sec": detection["event_times_sec"],
            "event_value": detection["event_values"],
            "event_polarity": detection.get("event_polarities", np.array([], dtype=object)),
            "threshold": float(detection["threshold"]),
            "threshold_upper": float(detection.get("threshold_upper", np.nan)),
            "threshold_lower": float(detection.get("threshold_lower", np.nan)),
            "signal_excursion_polarity": str(
                detection.get(
                    "signal_excursion_polarity",
                    getattr(cfg, "signal_excursion_polarity", "positive"),
                )
            ),
            "signal_excursion_polarity_interpretation": str(
                detection.get(
                    "signal_excursion_polarity_interpretation",
                    _signal_excursion_polarity_interpretation(
                        getattr(cfg, "signal_excursion_polarity", "positive")
                    ),
                )
            ),
            "event_auc_semantics": str(
                detection.get(
                    "event_auc_semantics",
                    _event_auc_semantics_text(getattr(cfg, "signal_excursion_polarity", "positive")),
                )
            ),
            "threshold_method": cfg.peak_threshold_method,
            "peak_min_distance_sec": float(cfg.peak_min_distance_sec),
        }
    )
    events_csv = os.path.join(retune_dir, f"retuned_events_{suffix}.csv")
    events_df.to_csv(events_csv, index=False)
    out["retuned_events_csv"] = events_csv

    overlay_png = os.path.join(retune_dir, f"retuned_overlay_{suffix}.png")
    # Keep the visible preview line faithful to the actual y-array used for detection.
    # Path simplification can collapse distinct high-density traces into the same pixels.
    with matplotlib.rc_context({"path.simplify": False, "path.simplify_threshold": 0.0}):
        fig, ax = plt.subplots(figsize=_RETUNE_OVERLAY_FIGSIZE)
        x = detection["trace_time_sec"]
        y = detection["trace_used_for_calling"]
        trace_label = f"{detection['signal_field']} trace"
        if str(detection.get("peak_pre_filter", "none")) == "smooth":
            win = detection.get("savgol_window_length", np.nan)
            poly = detection.get("savgol_polyorder", np.nan)
            if np.isfinite(win) and np.isfinite(poly):
                trace_label = (
                    f"{detection['signal_field']} trace (smooth SG window={int(win)}, poly={int(poly)})"
                )
            else:
                trace_label = f"{detection['signal_field']} trace (smooth)"
        ax.plot(
            x,
            y,
            linewidth=1.0,
            color="steelblue",
            label=trace_label,
            antialiased=False,
        )
        event_polarities = np.asarray(detection.get("event_polarities", []), dtype=object)
        if len(detection["event_indices"]):
            pos_mask = event_polarities == "positive"
            neg_mask = event_polarities == "negative"
            if np.any(pos_mask):
                ax.scatter(
                    np.asarray(detection["event_times_sec"])[pos_mask],
                    np.asarray(detection["event_values"])[pos_mask],
                    color="crimson",
                    marker="x",
                    s=28,
                    linewidths=1.0,
                    label="Detected events (+)",
                    zorder=3,
                )
            if np.any(neg_mask):
                ax.scatter(
                    np.asarray(detection["event_times_sec"])[neg_mask],
                    np.asarray(detection["event_values"])[neg_mask],
                    color="teal",
                    marker="x",
                    s=28,
                    linewidths=1.0,
                    label="Detected events (-)",
                    zorder=3,
                )
            if not np.any(pos_mask) and not np.any(neg_mask):
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
        polarity = str(
            detection.get(
                "signal_excursion_polarity",
                getattr(cfg, "signal_excursion_polarity", "positive"),
            )
        ).strip().lower()
        if polarity in {"positive", "both"}:
            ax.axhline(
                float(detection.get("threshold_upper", detection["threshold"])),
                color="darkorange",
                linestyle="--",
                linewidth=1.0,
                label="Upper threshold",
            )
        if polarity in {"negative", "both"}:
            ax.axhline(
                float(detection.get("threshold_lower", detection["threshold"])),
                color="slateblue",
                linestyle="--",
                linewidth=1.0,
                label="Lower threshold",
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(detection["signal_field"])
        ax.set_title(
            "Retuned Event Overlay "
            f"({roi}) | chunk={chunk_id} | source={detection['source_file']} | "
            f"polarity={str(polarity)}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(overlay_png, dpi=_RETUNE_OVERLAY_DPI)
        plt.close(fig)
    out["retuned_overlay_png"] = overlay_png
    if _retune_debug_enabled():
        out["retuned_overlay_png_sha256"] = _sha256_file(overlay_png)
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
            signal_excursion_polarity_used=str(
                getattr(effective_config, "signal_excursion_polarity", "positive")
            ),
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
            chunk_cfg = _resolve_prefilter_config_for_chunk(
                effective_config,
                chunk.fs_hz,
                n_samples=len(chunk.time_sec),
            )
            feats = extract_features(chunk, chunk_cfg)
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
        signal_excursion_polarity=str(
            getattr(effective_config, "signal_excursion_polarity", "positive")
        ),
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
    if _retune_debug_enabled():
        backend_debug = {
            "schema_version": 1,
            "debug_kind": "post_run_tuning_backend_overlay",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_run_dir": run_dir,
            "retune_dir": retune_dir,
            "roi": resolved_roi,
            "chunk_id": int(detection.get("chunk_id", inspection_chunk_id)),
            "source_file": str(detection.get("source_file", inspection_chunk.source_file)),
            "peak_pre_filter": str(detection.get("peak_pre_filter", "none")),
            "event_signal": str(detection.get("signal_field", effective_config.event_signal)),
            "signal_excursion_polarity": str(
                getattr(effective_config, "signal_excursion_polarity", "positive")
            ),
            "signal_excursion_polarity_interpretation": _signal_excursion_polarity_interpretation(
                getattr(effective_config, "signal_excursion_polarity", "positive")
            ),
            "event_auc_semantics": _event_auc_semantics_text(
                getattr(effective_config, "signal_excursion_polarity", "positive")
            ),
            "prefilter_applied": bool(detection.get("prefilter_applied", False)),
            "savgol_window_length": float(detection.get("savgol_window_length", np.nan)),
            "savgol_polyorder": float(detection.get("savgol_polyorder", np.nan)),
            "fs_hz": float(detection.get("fs_hz", inspection_chunk.fs_hz)),
            "threshold": float(detection.get("threshold", np.nan)),
            "threshold_upper": float(detection.get("threshold_upper", np.nan)),
            "threshold_lower": float(detection.get("threshold_lower", np.nan)),
            "threshold_method": str(effective_config.peak_threshold_method),
            "distance_samples": int(detection.get("distance_samples", 0)),
            "event_count": int(len(detection.get("event_indices", []))),
            "overlay_png_path": str(artifact_paths.get("retuned_overlay_png", "")),
            "overlay_png_sha256": str(artifact_paths.get("retuned_overlay_png_sha256", "")),
            "overrides": dict(overrides),
        }
        backend_debug.update(_trace_summary(np.asarray(detection.get("trace_used_for_calling", []), dtype=np.float64)))
        _write_backend_debug_record(retune_dir, backend_debug)

    result = {
        "retune_dir": retune_dir,
        "selected_roi": resolved_roi,
        "inspection_chunk_id": inspection_chunk_id,
        "inspection_source_file": inspection_chunk.source_file,
        "event_signal_used": effective_config.event_signal,
        "signal_excursion_polarity_used": str(
            getattr(effective_config, "signal_excursion_polarity", "positive")
        ),
        "signal_excursion_polarity_interpretation": _signal_excursion_polarity_interpretation(
            getattr(effective_config, "signal_excursion_polarity", "positive")
        ),
        "event_auc_semantics": _event_auc_semantics_text(
            getattr(effective_config, "signal_excursion_polarity", "positive")
        ),
        "n_chunks": int(features_df["chunk_id"].nunique()) if not features_df.empty else 0,
        "n_rows": int(len(features_df)),
        "downstream_overrides_applied": dict(overrides),
        "artifacts": artifact_paths,
    }
    with open(os.path.join(retune_dir, "retune_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result
