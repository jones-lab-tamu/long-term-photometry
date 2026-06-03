"""Continuous-mode summary table generation from production artifacts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from photometry_pipeline.io.hdf5_cache_reader import (
    load_cache_chunk_attrs,
    load_cache_chunk_fields,
    list_cache_chunk_ids,
    list_cache_rois,
    open_phasic_cache,
    open_tonic_cache,
)


PHASIC_SUMMARY_FILENAME = "continuous_phasic_window_summary.csv"
TONIC_SUMMARY_FILENAME = "continuous_tonic_window_summary.csv"
PHASIC_RATE_PLOT_FILENAME = "phasic_peak_rate_timeseries.png"
PHASIC_COUNT_PLOT_FILENAME = "phasic_peak_count_timeseries.png"
PHASIC_AUC_PLOT_FILENAME = "phasic_auc_timeseries.png"
TONIC_OVERVIEW_PLOT_FILENAME = "tonic_overview.png"
CONTINUOUS_TONIC_TRACE_OVERVIEW_FILENAME = "continuous_tonic_trace_overview.png"
CONTINUOUS_PHASIC_DFF_TRACE_OVERVIEW_FILENAME = "continuous_phasic_dff_trace_overview.png"
CONTINUOUS_TRACE_OVERVIEW_MAX_POINTS = 100_000
_RETUNED_PHASIC_SUMMARY_TEMPLATE = "{prefix}_continuous_phasic_window_summary_{roi}.csv"
_RETUNED_PHASIC_RATE_PLOT_TEMPLATE = "{prefix}_phasic_peak_rate_timeseries_{roi}.png"
_RETUNED_PHASIC_COUNT_PLOT_TEMPLATE = "{prefix}_phasic_peak_count_timeseries_{roi}.png"
_RETUNED_PHASIC_AUC_PLOT_TEMPLATE = "{prefix}_phasic_auc_timeseries_{roi}.png"
_AUC_SEMANTICS = (
    "aggregate finite-run AUC from feature_extraction output; not per-event AUC"
)
_CONTINUOUS_REQUIRED_ATTRS = (
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "window_duration_sec",
)


def _log(logger, message: str) -> None:
    if logger is not None:
        logger(message)


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def _roi_tables_dir(run_dir: str, roi: str) -> str:
    path = os.path.join(run_dir, str(roi), "tables")
    os.makedirs(path, exist_ok=True)
    return path


def _roi_summary_dir(run_dir: str, roi: str) -> str:
    path = os.path.join(run_dir, str(roi), "summary")
    os.makedirs(path, exist_ok=True)
    return path


def _empty_result(kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "generated_files": [],
        "skipped_outputs": [],
        "rois_processed": [],
        "row_counts": {},
        "source_artifacts": [],
    }


def _skip(result: dict[str, Any], output: str, reason: str) -> dict[str, Any]:
    result["skipped_outputs"].append({"output": output, "reason": reason})
    return result


def _require_continuous_attrs(attrs: dict[str, Any], *, roi: str, chunk_id: int) -> None:
    missing = [key for key in _CONTINUOUS_REQUIRED_ATTRS if key not in attrs]
    if missing:
        raise RuntimeError(
            "Continuous summary requires HDF5 window metadata. "
            f"Missing attrs for roi={roi} chunk_id={chunk_id}: {missing}"
        )
    mode = str(attrs.get("acquisition_mode", "")).strip().lower()
    if mode != "continuous":
        raise RuntimeError(
            "Continuous summary requested for non-continuous cache chunk: "
            f"roi={roi} chunk_id={chunk_id} acquisition_mode={attrs.get('acquisition_mode')!r}"
        )


def _window_metadata_row(attrs: dict[str, Any], *, roi: str, chunk_id: int) -> dict[str, Any]:
    _require_continuous_attrs(attrs, roi=roi, chunk_id=chunk_id)
    start = float(attrs["window_start_sec"])
    end = float(attrs["window_end_sec"])
    duration = float(attrs["window_duration_sec"])
    return {
        "roi": str(roi),
        "source_file": str(attrs.get("source_file", "")),
        "chunk_id": int(chunk_id),
        "window_index": int(round(float(attrs["window_index"]))),
        "window_start_sec": start,
        "window_end_sec": end,
        "window_duration_sec": duration,
        "elapsed_hour_start": start / 3600.0,
        "elapsed_hour_mid": ((start + end) / 2.0) / 3600.0,
        "is_partial_final_window": bool(attrs.get("is_partial_final_window", False)),
        "original_file_duration_sec": _float_or_nan(attrs.get("original_file_duration_sec")),
        "continuous_window_sec": _float_or_nan(attrs.get("continuous_window_sec")),
        "continuous_step_sec": _float_or_nan(attrs.get("continuous_step_sec")),
        "acquisition_mode": str(attrs.get("acquisition_mode", "")),
    }


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _write_roi_and_aggregate_tables(
    *,
    df: pd.DataFrame,
    run_dir: str,
    filename: str,
    result: dict[str, Any],
) -> None:
    if df.empty:
        return
    for roi, roi_df in df.groupby("roi", sort=True):
        out_path = os.path.join(_roi_tables_dir(run_dir, str(roi)), filename)
        roi_df.to_csv(out_path, index=False)
        result["generated_files"].append(out_path)
        result["rois_processed"].append(str(roi))
        result["row_counts"][str(roi)] = int(len(roi_df))
    result["rois_processed"] = sorted(set(result["rois_processed"]))
    result["row_counts"]["all_rois"] = int(len(df))


_PHASIC_SUMMARY_COLUMNS = [
    "roi",
    "source_file",
    "chunk_id",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "window_duration_sec",
    "elapsed_hour_start",
    "elapsed_hour_mid",
    "event_count",
    "event_rate_per_min",
    "event_rate_per_hour",
    "event_signal_auc",
    "event_signal_auc_semantics",
    "event_signal_mean",
    "event_signal_median",
    "event_signal_std",
    "event_signal_mad",
    "is_partial_final_window",
    "original_file_duration_sec",
    "continuous_window_sec",
    "continuous_step_sec",
    "acquisition_mode",
]


def _load_phasic_features(features_path: str, *, roi: str | None = None) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    required = {"chunk_id", "roi", "peak_count", "auc"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in phasic features.csv: {missing}")

    features["chunk_id"] = pd.to_numeric(features["chunk_id"], errors="raise").astype(int)
    if roi is not None:
        features = features[features["roi"].astype(str) == str(roi)].copy()
    return features


def _continuous_status_for_features(
    *,
    features: pd.DataFrame,
    cache_path: str,
    roi: str,
) -> tuple[bool, str]:
    if features.empty:
        return False, f"No feature rows found for ROI {roi}."

    first_chunk_id = int(features.sort_values("chunk_id").iloc[0]["chunk_id"])
    with open_phasic_cache(cache_path) as cache:
        attrs = load_cache_chunk_attrs(cache, roi, first_chunk_id)

    mode = str(attrs.get("acquisition_mode", "")).strip().lower()
    if mode != "continuous":
        return False, (
            "source cache is not continuous; retuned continuous outputs skipped: "
            f"roi={roi} chunk_id={first_chunk_id} acquisition_mode={attrs.get('acquisition_mode')!r}"
        )
    _require_continuous_attrs(attrs, roi=roi, chunk_id=first_chunk_id)
    return True, ""


def _build_continuous_phasic_summary_dataframe(
    *,
    features_path: str,
    cache_path: str,
    roi: str | None = None,
) -> pd.DataFrame:
    features = _load_phasic_features(features_path, roi=roi)
    rows: list[dict[str, Any]] = []
    with open_phasic_cache(cache_path) as cache:
        for record in features.to_dict(orient="records"):
            row_roi = str(record["roi"])
            chunk_id = int(record["chunk_id"])
            attrs = load_cache_chunk_attrs(cache, row_roi, chunk_id)
            meta = _window_metadata_row(attrs, roi=row_roi, chunk_id=chunk_id)
            duration_min = meta["window_duration_sec"] / 60.0
            duration_hour = meta["window_duration_sec"] / 3600.0
            event_count = int(record["peak_count"])
            rows.append(
                {
                    **meta,
                    "event_count": event_count,
                    "event_rate_per_min": (
                        event_count / duration_min if duration_min > 0 else np.nan
                    ),
                    "event_rate_per_hour": (
                        event_count / duration_hour if duration_hour > 0 else np.nan
                    ),
                    "event_signal_auc": _float_or_nan(record.get("auc")),
                    "event_signal_auc_semantics": _AUC_SEMANTICS,
                    "event_signal_mean": _float_or_nan(record.get("mean", np.nan)),
                    "event_signal_median": _float_or_nan(record.get("median", np.nan)),
                    "event_signal_std": _float_or_nan(record.get("std", np.nan)),
                    "event_signal_mad": _float_or_nan(record.get("mad", np.nan)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=_PHASIC_SUMMARY_COLUMNS)
    return pd.DataFrame(rows, columns=_PHASIC_SUMMARY_COLUMNS).sort_values(
        ["roi", "window_index", "chunk_id"]
    )


def generate_continuous_phasic_summary(
    phasic_out_dir: str,
    output_root_or_run_dir: str,
    *,
    logger=None,
) -> dict[str, Any]:
    """Generate per-window phasic summary CSVs from features.csv and cache attrs."""
    result = _empty_result("phasic")
    features_path = os.path.join(phasic_out_dir, "features", "features.csv")
    cache_path = os.path.join(phasic_out_dir, "phasic_trace_cache.h5")
    result["source_artifacts"] = [features_path, cache_path]

    if not os.path.exists(features_path):
        return _skip(result, PHASIC_SUMMARY_FILENAME, f"features.csv not found: {features_path}")
    if not os.path.exists(cache_path):
        return _skip(result, PHASIC_SUMMARY_FILENAME, f"phasic cache not found: {cache_path}")

    summary = _build_continuous_phasic_summary_dataframe(
        features_path=features_path,
        cache_path=cache_path,
    )
    _write_roi_and_aggregate_tables(
        df=summary,
        run_dir=output_root_or_run_dir,
        filename=PHASIC_SUMMARY_FILENAME,
        result=result,
    )
    _log(logger, f"Generated continuous phasic summary rows={len(summary)}")
    return result


def _retuned_continuous_skip(
    *,
    features_path: str,
    cache_path: str,
    output_dir: str,
    roi: str,
    reason: str,
    continuous_detected: bool = False,
) -> dict[str, Any]:
    return {
        "generated": False,
        "summary_csv": "",
        "plots": {},
        "skips": [{"output": "retuned_continuous_phasic_outputs", "reason": reason}],
        "source_features_path": features_path,
        "source_cache_path": cache_path,
        "output_dir": output_dir,
        "roi": str(roi),
        "reason": reason,
        "continuous_detected": bool(continuous_detected),
    }


def _generate_retuned_phasic_plots(
    *,
    summary_csv: str,
    output_dir: str,
    roi: str,
    output_prefix: str,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    df = pd.read_csv(summary_csv)
    plot_specs = [
        (
            "peak_rate",
            _RETUNED_PHASIC_RATE_PLOT_TEMPLATE.format(prefix=output_prefix, roi=roi),
            "event_rate_per_min",
            "Event rate (events/min)",
            f"{roi} retuned continuous event rate",
        ),
        (
            "peak_count",
            _RETUNED_PHASIC_COUNT_PLOT_TEMPLATE.format(prefix=output_prefix, roi=roi),
            "event_count",
            "Event count per window",
            f"{roi} retuned continuous event count",
        ),
        (
            "auc",
            _RETUNED_PHASIC_AUC_PLOT_TEMPLATE.format(prefix=output_prefix, roi=roi),
            "event_signal_auc",
            "Aggregate event-signal AUC per window",
            f"{roi} retuned continuous aggregate event-signal AUC",
        ),
    ]
    plots: dict[str, str] = {}
    skips: list[dict[str, str]] = []
    for key, filename, y_col, y_label, title in plot_specs:
        out_path = os.path.join(output_dir, filename)
        ok = _plot_xy_from_summary(
            df=df,
            x_col="elapsed_hour_mid",
            y_col=y_col,
            y_label=y_label,
            title=title,
            out_path=out_path,
        )
        if ok:
            plots[key] = out_path
        else:
            skips.append(
                {
                    "output": filename,
                    "reason": (
                        "No finite values available for required columns "
                        f"elapsed_hour_mid/{y_col}"
                    ),
                }
            )
    return plots, skips


def generate_retuned_continuous_phasic_outputs(
    *,
    features_path: str,
    cache_path: str,
    output_dir: str,
    roi: str,
    output_prefix: str = "retuned",
    logger=None,
) -> dict[str, Any]:
    """
    Generate flat retune-root continuous phasic summary/plot outputs.

    Missing source files and non-continuous/intermittent caches are treated as
    retune skips. Caches that explicitly claim continuous mode must include the
    required continuous window attrs.
    """
    output_dir = os.path.abspath(output_dir)
    features_path = os.path.abspath(features_path)
    cache_path = os.path.abspath(cache_path)
    resolved_roi = str(roi)
    if not os.path.exists(features_path):
        return _retuned_continuous_skip(
            features_path=features_path,
            cache_path=cache_path,
            output_dir=output_dir,
            roi=resolved_roi,
            reason=f"Retuned features CSV not found: {features_path}",
        )
    if not os.path.exists(cache_path):
        return _retuned_continuous_skip(
            features_path=features_path,
            cache_path=cache_path,
            output_dir=output_dir,
            roi=resolved_roi,
            reason=f"Retuned/source phasic cache not found: {cache_path}",
        )

    features = _load_phasic_features(features_path, roi=resolved_roi)
    continuous_detected, reason = _continuous_status_for_features(
        features=features,
        cache_path=cache_path,
        roi=resolved_roi,
    )
    if not continuous_detected:
        return _retuned_continuous_skip(
            features_path=features_path,
            cache_path=cache_path,
            output_dir=output_dir,
            roi=resolved_roi,
            reason=reason,
        )

    summary = _build_continuous_phasic_summary_dataframe(
        features_path=features_path,
        cache_path=cache_path,
        roi=resolved_roi,
    )
    if summary.empty:
        return _retuned_continuous_skip(
            features_path=features_path,
            cache_path=cache_path,
            output_dir=output_dir,
            roi=resolved_roi,
            reason=f"No retuned continuous summary rows generated for ROI {resolved_roi}.",
            continuous_detected=True,
        )

    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(
        output_dir,
        _RETUNED_PHASIC_SUMMARY_TEMPLATE.format(prefix=output_prefix, roi=resolved_roi),
    )
    summary.to_csv(summary_csv, index=False)
    plots, skips = _generate_retuned_phasic_plots(
        summary_csv=summary_csv,
        output_dir=output_dir,
        roi=resolved_roi,
        output_prefix=output_prefix,
    )
    result = {
        "generated": True,
        "summary_csv": summary_csv,
        "plots": plots,
        "skips": skips,
        "source_features_path": features_path,
        "source_cache_path": cache_path,
        "output_dir": output_dir,
        "roi": resolved_roi,
        "continuous_detected": True,
        "row_count": int(len(summary)),
    }
    _log(logger, f"Generated retuned continuous phasic rows={len(summary)} roi={resolved_roi}")
    return result


def generate_continuous_tonic_summary(
    tonic_out_dir: str,
    output_root_or_run_dir: str,
    *,
    logger=None,
) -> dict[str, Any]:
    """Generate per-window tonic summary CSVs from tonic cache deltaF traces."""
    result = _empty_result("tonic")
    cache_path = os.path.join(tonic_out_dir, "tonic_trace_cache.h5")
    result["source_artifacts"] = [cache_path]

    if not os.path.exists(cache_path):
        return _skip(result, TONIC_SUMMARY_FILENAME, f"tonic cache not found: {cache_path}")

    rows: list[dict[str, Any]] = []
    with open_tonic_cache(cache_path) as cache:
        rois = list_cache_rois(cache)
        chunk_ids = list_cache_chunk_ids(cache)
        for roi in rois:
            for chunk_id in chunk_ids:
                attrs = load_cache_chunk_attrs(cache, roi, int(chunk_id))
                meta = _window_metadata_row(attrs, roi=roi, chunk_id=int(chunk_id))
                delta_f, = load_cache_chunk_fields(cache, roi, int(chunk_id), ["deltaF"])
                arr = np.asarray(delta_f, dtype=float)
                finite = arr[np.isfinite(arr)]
                n_total = int(arr.size)
                n_finite = int(finite.size)
                if n_finite:
                    stats = {
                        "tonic_mean": float(np.mean(finite)),
                        "tonic_median": float(np.median(finite)),
                        "tonic_min": float(np.min(finite)),
                        "tonic_max": float(np.max(finite)),
                        "tonic_p05": float(np.percentile(finite, 5)),
                        "tonic_p95": float(np.percentile(finite, 95)),
                    }
                else:
                    stats = {
                        "tonic_mean": np.nan,
                        "tonic_median": np.nan,
                        "tonic_min": np.nan,
                        "tonic_max": np.nan,
                        "tonic_p05": np.nan,
                        "tonic_p95": np.nan,
                    }
                rows.append(
                    {
                        **meta,
                        **stats,
                        "tonic_n_finite": n_finite,
                        "tonic_nan_fraction": (
                            1.0 - (n_finite / float(n_total)) if n_total > 0 else np.nan
                        ),
                    }
                )

    columns = [
        "roi",
        "source_file",
        "chunk_id",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "window_duration_sec",
        "elapsed_hour_start",
        "elapsed_hour_mid",
        "tonic_mean",
        "tonic_median",
        "tonic_min",
        "tonic_max",
        "tonic_p05",
        "tonic_p95",
        "tonic_n_finite",
        "tonic_nan_fraction",
        "is_partial_final_window",
        "original_file_duration_sec",
        "continuous_window_sec",
        "continuous_step_sec",
        "acquisition_mode",
    ]
    summary = pd.DataFrame(rows, columns=columns).sort_values(["roi", "window_index", "chunk_id"])
    _write_roi_and_aggregate_tables(
        df=summary,
        run_dir=output_root_or_run_dir,
        filename=TONIC_SUMMARY_FILENAME,
        result=result,
    )
    _log(logger, f"Generated continuous tonic summary rows={len(summary)}")
    return result


def _candidate_roi_dirs(run_dir: str) -> list[tuple[str, str]]:
    if not os.path.isdir(run_dir):
        return []
    out = []
    for name in sorted(os.listdir(run_dir), key=lambda s: s.lower()):
        path = os.path.join(run_dir, name)
        if not os.path.isdir(path):
            continue
        if name.startswith(".") or name.startswith("_"):
            continue
        out.append((name, path))
    return out


def _plot_xy_from_summary(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_label: str,
    title: str,
    out_path: str,
) -> bool:
    import matplotlib.pyplot as plt

    if x_col not in df.columns or y_col not in df.columns:
        return False
    plot_df = df[[x_col, y_col]].copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df[x_col]) & np.isfinite(plot_df[y_col])]
    if plot_df.empty:
        return False
    plot_df = plot_df.sort_values(x_col)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df[x_col], plot_df[y_col], marker="o", linewidth=1.2)
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _evenly_spaced_indices(n_items: int, n_select: int) -> np.ndarray:
    """Return deterministic indices spanning a finite chunk."""
    n_items = int(n_items)
    n_select = int(n_select)
    if n_items <= 0 or n_select <= 0:
        return np.array([], dtype=int)
    if n_select >= n_items:
        return np.arange(n_items, dtype=int)
    return np.unique(np.linspace(0, n_items - 1, n_select, dtype=int))


def _select_finite_positions_for_chunk(
    n_finite: int,
    n_select: int,
    *,
    force_first: bool = False,
    force_last: bool = False,
) -> np.ndarray:
    """Select bounded positions within a finite chunk while preserving required endpoints."""
    n_finite = int(n_finite)
    n_select = int(n_select)
    if n_finite <= 0 or n_select <= 0:
        return np.array([], dtype=int)
    if n_select >= n_finite:
        return np.arange(n_finite, dtype=int)

    required: list[int] = []
    if force_first:
        required.append(0)
    if force_last:
        required.append(n_finite - 1)
    required = sorted(set(required))

    if len(required) >= n_select:
        # This only occurs when an endpoint chunk receives a single slot. Prefer
        # the global final endpoint for last-only chunks; otherwise keep first.
        if force_last and not force_first:
            return np.array([n_finite - 1], dtype=int)
        return np.array(required[:n_select], dtype=int)

    selected = set(required)
    for idx in _evenly_spaced_indices(n_finite, n_select):
        selected.add(int(idx))
        if len(selected) >= n_select:
            break

    if len(selected) < n_select:
        for idx in range(n_finite):
            selected.add(idx)
            if len(selected) >= n_select:
                break

    return np.array(sorted(selected), dtype=int)


def _allocate_trace_points(
    chunk_records: list[dict[str, Any]],
    *,
    max_points: int,
) -> dict[int, int]:
    """Allocate a bounded number of display points proportionally across chunks."""
    if max_points < 2:
        raise ValueError("max_points must be >= 2")
    finite_records = [r for r in chunk_records if int(r["n_finite"]) > 0]
    if not finite_records:
        return {}

    total_finite = int(sum(int(r["n_finite"]) for r in finite_records))
    if total_finite <= max_points:
        return {int(r["chunk_id"]): int(r["n_finite"]) for r in finite_records}

    if len(finite_records) <= max_points:
        allocations = {int(r["chunk_id"]): 1 for r in finite_records}
        remaining = int(max_points - len(finite_records))
    else:
        allocations = {int(r["chunk_id"]): 0 for r in finite_records}
        remaining = int(max_points)
        # Preserve the full recording span when there are more finite chunks than points.
        allocations[int(finite_records[0]["chunk_id"])] = 1
        allocations[int(finite_records[-1]["chunk_id"])] = 1
        remaining -= 2

    weighted: list[tuple[float, int, int]] = []
    for rec in finite_records:
        cid = int(rec["chunk_id"])
        n_finite = int(rec["n_finite"])
        target = (n_finite / float(total_finite)) * float(max_points)
        whole = int(np.floor(target))
        if allocations[cid] > 0:
            whole = max(0, whole - allocations[cid])
        add = min(max(0, whole), n_finite - allocations[cid], remaining)
        allocations[cid] += add
        remaining -= add
        weighted.append((target - np.floor(target), n_finite, cid))

    for _frac, _n_finite, cid in sorted(weighted, key=lambda item: (-item[0], -item[1], item[2])):
        if remaining <= 0:
            break
        capacity = int(next(r["n_finite"] for r in finite_records if int(r["chunk_id"]) == cid))
        if allocations[cid] < capacity:
            allocations[cid] += 1
            remaining -= 1

    # Defensive trim in case floor/min logic changes later.
    while sum(allocations.values()) > max_points:
        candidates = sorted(
            (cid for cid, count in allocations.items() if count > 1),
            key=lambda c: (-allocations[c], c),
        )
        if not candidates:
            break
        allocations[candidates[0]] -= 1

    return {cid: count for cid, count in allocations.items() if count > 0}


def _sample_elapsed_trace_from_cache(
    cache,
    roi: str,
    field: str,
    *,
    max_points: int = CONTINUOUS_TRACE_OVERVIEW_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Sample a bounded elapsed-time trace from continuous HDF5 chunks.

    This intentionally reads one chunk at a time and concatenates only selected
    display points, never the full recording.
    """
    chunk_records: list[dict[str, Any]] = []
    n_samples_seen = 0
    for chunk_id in list_cache_chunk_ids(cache):
        cid = int(chunk_id)
        attrs = load_cache_chunk_attrs(cache, roi, cid)
        _require_continuous_attrs(attrs, roi=roi, chunk_id=cid)
        time_sec, values = load_cache_chunk_fields(cache, roi, cid, ["time_sec", field])
        local_t = np.asarray(time_sec, dtype=float).reshape(-1)
        y = np.asarray(values, dtype=float).reshape(-1)
        if local_t.shape != y.shape:
            raise RuntimeError(
                f"Trace field shape mismatch for roi={roi} chunk_id={cid} field={field}: "
                f"time_sec={local_t.shape} values={y.shape}"
            )
        n_samples_seen += int(y.size)
        start = float(attrs["window_start_sec"])
        finite = np.isfinite(local_t) & np.isfinite(y)
        finite_idx = np.flatnonzero(finite)
        elapsed_finite = start + local_t[finite_idx] if finite_idx.size else np.array([], dtype=float)
        chunk_records.append(
            {
                "chunk_id": cid,
                "window_start_sec": start,
                "window_index": int(round(float(attrs.get("window_index", cid)))),
                "n_samples": int(y.size),
                "n_finite": int(finite_idx.size),
                "elapsed_start": (
                    float(elapsed_finite[0]) if elapsed_finite.size else float("nan")
                ),
                "elapsed_end": (
                    float(elapsed_finite[-1]) if elapsed_finite.size else float("nan")
                ),
            }
        )

    chunk_records.sort(key=lambda r: (float(r["window_start_sec"]), int(r["window_index"]), int(r["chunk_id"])))
    finite_records = [r for r in chunk_records if int(r["n_finite"]) > 0]
    first_finite_chunk_id = int(finite_records[0]["chunk_id"]) if finite_records else None
    last_finite_chunk_id = int(finite_records[-1]["chunk_id"]) if finite_records else None
    allocations = _allocate_trace_points(chunk_records, max_points=max_points)

    elapsed_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for rec in chunk_records:
        cid = int(rec["chunk_id"])
        n_select = int(allocations.get(cid, 0))
        if n_select <= 0:
            continue
        time_sec, values = load_cache_chunk_fields(cache, roi, cid, ["time_sec", field])
        local_t = np.asarray(time_sec, dtype=float).reshape(-1)
        y = np.asarray(values, dtype=float).reshape(-1)
        start = float(rec["window_start_sec"])
        finite = np.isfinite(local_t) & np.isfinite(y)
        finite_idx = np.flatnonzero(finite)
        selected_positions = _select_finite_positions_for_chunk(
            finite_idx.size,
            n_select,
            force_first=cid == first_finite_chunk_id,
            force_last=cid == last_finite_chunk_id,
        )
        selected = finite_idx[selected_positions]
        elapsed_parts.append(start + local_t[selected])
        value_parts.append(y[selected])

    if elapsed_parts:
        elapsed = np.concatenate(elapsed_parts)
        values = np.concatenate(value_parts)
        order = np.argsort(elapsed, kind="mergesort")
        elapsed = elapsed[order]
        values = values[order]
    else:
        elapsed = np.array([], dtype=float)
        values = np.array([], dtype=float)

    details = {
        "n_chunks": int(len(chunk_records)),
        "n_samples_seen": int(n_samples_seen),
        "n_finite_samples": int(sum(int(r["n_finite"]) for r in chunk_records)),
        "n_points_plotted": int(values.size),
        "max_plot_points": int(max_points),
        "chunk_ids": [int(r["chunk_id"]) for r in chunk_records],
        "finite_chunk_ids": [int(r["chunk_id"]) for r in finite_records],
        "elapsed_hour_start": (
            float(elapsed[0] / 3600.0) if elapsed.size else float("nan")
        ),
        "elapsed_hour_end": (
            float(elapsed[-1] / 3600.0) if elapsed.size else float("nan")
        ),
    }
    return elapsed, values, details


def _plot_continuous_trace_overview(
    *,
    elapsed_sec: np.ndarray,
    values: np.ndarray,
    roi: str,
    ylabel: str,
    title: str,
    out_path: str,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    x_plot = np.asarray(elapsed_sec, dtype=float).reshape(-1)
    y_plot = np.asarray(values, dtype=float).reshape(-1)
    if x_plot.size == 0 or y_plot.size == 0:
        return {
            "generated": False,
            "reason": "No finite values available for full continuous trace overview.",
        }

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(x_plot / 3600.0, y_plot, linewidth=0.8)
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{roi} {title}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "generated": True,
    }


def _generate_trace_overview_for_cache(
    *,
    run_dir: str,
    cache_path: str,
    cache_kind: str,
    field: str,
    filename: str,
    ylabel: str,
    title: str,
    opener,
    max_plot_points: int,
) -> dict[str, Any]:
    result = _empty_result(f"{cache_kind}_trace_overview")
    result["details"] = {}
    result["source_artifacts"] = [cache_path]
    if not os.path.exists(cache_path):
        return _skip(result, filename, f"{cache_kind} cache not found: {cache_path}")

    with opener(cache_path) as cache:
        for roi in list_cache_rois(cache):
            rel_output = f"{roi}/summary/{filename}"
            try:
                elapsed_sec, values, details = _sample_elapsed_trace_from_cache(
                    cache,
                    str(roi),
                    field,
                    max_points=max_plot_points,
                )
                out_path = os.path.join(run_dir, str(roi), "summary", filename)
                plot_details = _plot_continuous_trace_overview(
                    elapsed_sec=elapsed_sec,
                    values=values,
                    roi=str(roi),
                    ylabel=ylabel,
                    title=title,
                    out_path=out_path,
                )
                details.update(plot_details)
                details["field"] = field
                details["cache_kind"] = cache_kind
                result["details"][str(roi)] = details
                if plot_details.get("generated"):
                    result["generated_files"].append(out_path)
                    result["rois_processed"].append(str(roi))
                    result["row_counts"][str(roi)] = 1
                else:
                    _skip(result, rel_output, str(plot_details.get("reason", "plot skipped")))
            except Exception as exc:
                result["details"][str(roi)] = {
                    "generated": False,
                    "field": field,
                    "cache_kind": cache_kind,
                    "reason": str(exc),
                }
                _skip(result, rel_output, str(exc))

    result["rois_processed"] = sorted(set(result["rois_processed"]))
    return result


def generate_continuous_trace_overview_plots(
    run_dir: str,
    *,
    mode: str = "both",
    logger=None,
    max_plot_points: int = CONTINUOUS_TRACE_OVERVIEW_MAX_POINTS,
) -> dict[str, Any]:
    """Generate per-ROI full elapsed trace overview plots from continuous HDF5 caches."""
    requested = str(mode or "both").strip().lower()
    if requested not in {"both", "phasic", "tonic"}:
        raise ValueError(f"Unsupported continuous trace overview mode: {mode!r}")

    run_dir = os.path.abspath(run_dir)
    tonic_cache = os.path.join(run_dir, "_analysis", "tonic_out", "tonic_trace_cache.h5")
    phasic_cache = os.path.join(run_dir, "_analysis", "phasic_out", "phasic_trace_cache.h5")
    results: dict[str, Any] = {
        "generated": False,
        "plots": [],
        "skips": [],
        "details": {},
        "tonic": None,
        "phasic": None,
        "max_plot_points": int(max_plot_points),
    }

    if requested in {"both", "tonic"}:
        results["tonic"] = _generate_trace_overview_for_cache(
            run_dir=run_dir,
            cache_path=tonic_cache,
            cache_kind="tonic",
            field="deltaF",
            filename=CONTINUOUS_TONIC_TRACE_OVERVIEW_FILENAME,
            ylabel="Tonic deltaF",
            title="full continuous tonic trace overview",
            opener=open_tonic_cache,
            max_plot_points=max_plot_points,
        )
    else:
        results["tonic"] = _skip(
            _empty_result("tonic_trace_overview"),
            CONTINUOUS_TONIC_TRACE_OVERVIEW_FILENAME,
            "tonic mode not requested",
        )

    if requested in {"both", "phasic"}:
        results["phasic"] = _generate_trace_overview_for_cache(
            run_dir=run_dir,
            cache_path=phasic_cache,
            cache_kind="phasic",
            field="dff",
            filename=CONTINUOUS_PHASIC_DFF_TRACE_OVERVIEW_FILENAME,
            ylabel="Phasic dF/F",
            title="full continuous phasic dF/F trace overview",
            opener=open_phasic_cache,
            max_plot_points=max_plot_points,
        )
    else:
        results["phasic"] = _skip(
            _empty_result("phasic_trace_overview"),
            CONTINUOUS_PHASIC_DFF_TRACE_OVERVIEW_FILENAME,
            "phasic mode not requested",
        )

    details: dict[str, Any] = {}
    for key in ("tonic", "phasic"):
        sub = results[key] or {}
        for path in sub.get("generated_files", []):
            results["plots"].append(_rel(path, run_dir))
        for skip in sub.get("skipped_outputs", []):
            results["skips"].append(skip)
        details[key] = sub.get("details", {})
    results["details"] = details
    results["plots"] = sorted(set(results["plots"]))
    results["generated"] = bool(results["plots"])
    _log(logger, f"Generated continuous trace overview files={len(results['plots'])}")
    return results


def _record_generated_plot(
    result: dict[str, Any],
    *,
    run_dir: str,
    roi: str,
    path: str,
    source_csv: str,
) -> None:
    result["generated_files"].append(path)
    result["rois_processed"].append(str(roi))
    result.setdefault("source_artifacts", []).append(source_csv)
    result["row_counts"][str(roi)] = result["row_counts"].get(str(roi), 0) + 1


def generate_continuous_phasic_plots(run_dir: str, *, logger=None) -> dict[str, Any]:
    """Generate continuous phasic elapsed-time plots from Patch 3a summary CSVs."""
    result = _empty_result("phasic_plots")
    any_table = False
    for roi, roi_dir in _candidate_roi_dirs(run_dir):
        table_path = os.path.join(roi_dir, "tables", PHASIC_SUMMARY_FILENAME)
        if not os.path.exists(table_path):
            continue
        any_table = True
        df = pd.read_csv(table_path)
        summary_dir = os.path.join(run_dir, str(roi), "summary")
        plot_specs = [
            (
                PHASIC_RATE_PLOT_FILENAME,
                "event_rate_per_min",
                "Event rate (events/min)",
                f"{roi} continuous event rate",
            ),
            (
                PHASIC_COUNT_PLOT_FILENAME,
                "event_count",
                "Event count per window",
                f"{roi} continuous event count",
            ),
            (
                PHASIC_AUC_PLOT_FILENAME,
                "event_signal_auc",
                "Aggregate event-signal AUC per window",
                f"{roi} continuous aggregate event-signal AUC",
            ),
        ]
        for filename, y_col, y_label, title in plot_specs:
            out_path = os.path.join(summary_dir, filename)
            ok = _plot_xy_from_summary(
                df=df,
                x_col="elapsed_hour_mid",
                y_col=y_col,
                y_label=y_label,
                title=title,
                out_path=out_path,
            )
            if ok:
                _record_generated_plot(
                    result,
                    run_dir=run_dir,
                    roi=roi,
                    path=out_path,
                    source_csv=table_path,
                )
            else:
                _skip(
                    result,
                    f"{roi}/summary/{filename}",
                    f"No finite values available for required columns elapsed_hour_mid/{y_col}",
                )
    if not any_table:
        _skip(
            result,
            PHASIC_RATE_PLOT_FILENAME,
            f"No per-ROI {PHASIC_SUMMARY_FILENAME} tables found under {run_dir}",
        )
    result["rois_processed"] = sorted(set(result["rois_processed"]))
    result["source_artifacts"] = sorted(set(result.get("source_artifacts", [])))
    _log(logger, f"Generated continuous phasic plot files={len(result['generated_files'])}")
    return result


def generate_continuous_tonic_plots(run_dir: str, *, logger=None) -> dict[str, Any]:
    """Generate continuous tonic elapsed-time plots from Patch 3a summary CSVs."""
    result = _empty_result("tonic_plots")
    any_table = False
    for roi, roi_dir in _candidate_roi_dirs(run_dir):
        table_path = os.path.join(roi_dir, "tables", TONIC_SUMMARY_FILENAME)
        if not os.path.exists(table_path):
            continue
        any_table = True
        df = pd.read_csv(table_path)
        if "tonic_median" in df.columns:
            y_col = "tonic_median"
            y_label = "Tonic dF/F, median per window"
        else:
            y_col = "tonic_mean"
            y_label = "Tonic dF/F, mean per window"
        summary_dir = os.path.join(run_dir, str(roi), "summary")
        out_path = os.path.join(summary_dir, TONIC_OVERVIEW_PLOT_FILENAME)
        ok = _plot_xy_from_summary(
            df=df,
            x_col="elapsed_hour_mid",
            y_col=y_col,
            y_label=y_label,
            title=f"{roi} continuous tonic summary",
            out_path=out_path,
        )
        if ok:
            _record_generated_plot(
                result,
                run_dir=run_dir,
                roi=roi,
                path=out_path,
                source_csv=table_path,
            )
        else:
            _skip(
                result,
                f"{roi}/summary/{TONIC_OVERVIEW_PLOT_FILENAME}",
                f"No finite values available for required columns elapsed_hour_mid/{y_col}",
            )
    if not any_table:
        _skip(
            result,
            TONIC_OVERVIEW_PLOT_FILENAME,
            f"No per-ROI {TONIC_SUMMARY_FILENAME} tables found under {run_dir}",
        )
    result["rois_processed"] = sorted(set(result["rois_processed"]))
    result["source_artifacts"] = sorted(set(result.get("source_artifacts", [])))
    _log(logger, f"Generated continuous tonic plot files={len(result['generated_files'])}")
    return result


def generate_continuous_summary_plots(
    run_dir: str,
    *,
    mode: str = "both",
    logger=None,
) -> dict[str, Any]:
    """Generate continuous elapsed-time plots from existing summary tables."""
    requested = str(mode or "both").strip().lower()
    if requested not in {"both", "phasic", "tonic"}:
        raise ValueError(f"Unsupported continuous plot mode: {mode!r}")

    results: dict[str, Any] = {
        "summary_plots_generated": False,
        "summary_plots": [],
        "plot_skips": [],
        "phasic": None,
        "tonic": None,
    }
    if requested in {"both", "phasic"}:
        results["phasic"] = generate_continuous_phasic_plots(run_dir, logger=logger)
    else:
        results["phasic"] = _skip(_empty_result("phasic_plots"), PHASIC_RATE_PLOT_FILENAME, "phasic mode not requested")

    if requested in {"both", "tonic"}:
        results["tonic"] = generate_continuous_tonic_plots(run_dir, logger=logger)
    else:
        results["tonic"] = _skip(_empty_result("tonic_plots"), TONIC_OVERVIEW_PLOT_FILENAME, "tonic mode not requested")

    for key in ("phasic", "tonic"):
        sub = results[key] or {}
        for path in sub.get("generated_files", []):
            results["summary_plots"].append(_rel(path, run_dir))
        for skip in sub.get("skipped_outputs", []):
            results["plot_skips"].append(skip)

    results["summary_plots"] = sorted(set(results["summary_plots"]))
    results["summary_plots_generated"] = bool(results["summary_plots"])
    return results


def generate_continuous_summary_tables(
    run_dir: str,
    *,
    tonic_out_dir: str | None = None,
    phasic_out_dir: str | None = None,
    mode: str = "both",
    logger=None,
) -> dict[str, Any]:
    """Generate continuous summary tables for the requested analysis mode."""
    requested = str(mode or "both").strip().lower()
    if requested not in {"both", "phasic", "tonic"}:
        raise ValueError(f"Unsupported continuous summary mode: {mode!r}")

    results: dict[str, Any] = {
        "summary_tables_generated": False,
        "summary_tables": [],
        "summary_skips": [],
        "phasic": None,
        "tonic": None,
    }
    if requested in {"both", "phasic"}:
        if phasic_out_dir:
            phasic = generate_continuous_phasic_summary(
                phasic_out_dir, run_dir, logger=logger
            )
        else:
            phasic = _skip(_empty_result("phasic"), PHASIC_SUMMARY_FILENAME, "phasic mode not requested")
        results["phasic"] = phasic
    else:
        results["phasic"] = _skip(_empty_result("phasic"), PHASIC_SUMMARY_FILENAME, "phasic mode not requested")

    if requested in {"both", "tonic"}:
        if tonic_out_dir:
            tonic = generate_continuous_tonic_summary(
                tonic_out_dir, run_dir, logger=logger
            )
        else:
            tonic = _skip(_empty_result("tonic"), TONIC_SUMMARY_FILENAME, "tonic mode not requested")
        results["tonic"] = tonic
    else:
        results["tonic"] = _skip(_empty_result("tonic"), TONIC_SUMMARY_FILENAME, "tonic mode not requested")

    for key in ("phasic", "tonic"):
        sub = results[key] or {}
        for path in sub.get("generated_files", []):
            results["summary_tables"].append(_rel(path, run_dir))
        for skip in sub.get("skipped_outputs", []):
            results["summary_skips"].append(skip)

    results["summary_tables"] = sorted(set(results["summary_tables"]))
    results["summary_tables_generated"] = bool(results["summary_tables"])
    return results
