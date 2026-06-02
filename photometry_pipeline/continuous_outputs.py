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

    features = pd.read_csv(features_path)
    required = {"chunk_id", "roi", "peak_count", "auc"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in phasic features.csv: {missing}")

    features["chunk_id"] = pd.to_numeric(features["chunk_id"], errors="raise").astype(int)
    rows: list[dict[str, Any]] = []
    with open_phasic_cache(cache_path) as cache:
        for row in features.itertuples(index=False):
            roi = str(getattr(row, "roi"))
            chunk_id = int(getattr(row, "chunk_id"))
            attrs = load_cache_chunk_attrs(cache, roi, chunk_id)
            meta = _window_metadata_row(attrs, roi=roi, chunk_id=chunk_id)
            duration_min = meta["window_duration_sec"] / 60.0
            duration_hour = meta["window_duration_sec"] / 3600.0
            event_count = int(getattr(row, "peak_count"))
            out = {
                **meta,
                "event_count": event_count,
                "event_rate_per_min": (
                    event_count / duration_min if duration_min > 0 else np.nan
                ),
                "event_rate_per_hour": (
                    event_count / duration_hour if duration_hour > 0 else np.nan
                ),
                "event_signal_auc": _float_or_nan(getattr(row, "auc")),
                "event_signal_auc_semantics": _AUC_SEMANTICS,
                "event_signal_mean": _float_or_nan(getattr(row, "mean", np.nan)),
                "event_signal_median": _float_or_nan(getattr(row, "median", np.nan)),
                "event_signal_std": _float_or_nan(getattr(row, "std", np.nan)),
                "event_signal_mad": _float_or_nan(getattr(row, "mad", np.nan)),
            }
            rows.append(out)

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
    summary = pd.DataFrame(rows, columns=columns).sort_values(["roi", "window_index", "chunk_id"])
    _write_roi_and_aggregate_tables(
        df=summary,
        run_dir=output_root_or_run_dir,
        filename=PHASIC_SUMMARY_FILENAME,
        result=result,
    )
    _log(logger, f"Generated continuous phasic summary rows={len(summary)}")
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
