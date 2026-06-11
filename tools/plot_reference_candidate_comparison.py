#!/usr/bin/env python3
"""Plot diagnostic dynamic-vs-baseline reference candidate comparisons."""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photometry_pipeline.core.baseline_reference_candidate import (  # noqa: E402
    classify_baseline_fit_relationship,
    compute_baseline_reference_candidate,
)
from photometry_pipeline.io.hdf5_cache_reader import (  # noqa: E402
    CacheReadError,
    load_cache_chunk_fields,
    open_phasic_cache,
    resolve_cache_roi,
)


DEFAULT_CACHE_NAME = "phasic_trace_cache.h5"
DEFAULT_QC_CSV = os.path.join("qc", "baseline_reference_candidate_by_chunk.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot diagnostic dynamic-vs-baseline reference candidate comparisons."
    )
    parser.add_argument("--analysis-out", required=True, help="Path to _analysis/phasic_out")
    parser.add_argument("--roi", required=True, help="ROI name, for example CH3")
    parser.add_argument(
        "--chunk-id",
        type=int,
        action="append",
        default=[],
        help="Chunk ID to plot. May be repeated.",
    )
    parser.add_argument(
        "--chunks",
        default="",
        help="Comma-separated chunk IDs, for example 28,29,30.",
    )
    parser.add_argument("--out", default=None, help="Output PNG path or output directory")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--show-raw-reference",
        action="store_true",
        help="Also show the raw reference on a secondary y-axis in panel 1.",
    )
    parser.add_argument(
        "--include-normalized-overlay",
        action="store_true",
        help="Add a normalized overlay panel for shape comparison.",
    )
    parser.add_argument(
        "--include-reference-difference",
        action="store_true",
        help="Add a panel showing dynamic reference minus baseline candidate.",
    )
    parser.add_argument(
        "--include-baseline-fit-diagnostics",
        action="store_true",
        help="Add panels showing baseline-candidate fit inputs and smoothed fit relationship.",
    )
    parser.add_argument(
        "--no-metadata-box",
        action="store_true",
        help="Suppress the wrapped metadata box above the panels.",
    )
    parser.add_argument("--title-extra", default="", help="Additional title text")
    return parser.parse_args()


def _parse_chunk_ids(chunk_id: list[int] | None, chunks: str | None) -> list[int]:
    ids: list[int] = []
    for cid in chunk_id or []:
        ids.append(int(cid))
    for part in str(chunks or "").split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    seen = set()
    out = []
    for cid in ids:
        if cid not in seen:
            out.append(cid)
            seen.add(cid)
    if not out:
        raise ValueError("At least one --chunk-id or --chunks value is required.")
    return out


def _default_output_path(analysis_out: str, roi: str, chunk_id: int) -> str:
    safe_roi = str(roi).replace(os.sep, "_").replace("/", "_")
    return os.path.join(
        analysis_out,
        "qc",
        "reference_candidate_plots",
        f"{safe_roi}_chunk_{int(chunk_id)}_reference_candidate_comparison.png",
    )


def _resolve_output_path(
    analysis_out: str,
    roi: str,
    chunk_id: int,
    out: str | None,
    *,
    multiple_chunks: bool,
) -> str:
    if not out:
        return _default_output_path(analysis_out, roi, chunk_id)
    out_path = Path(out)
    if multiple_chunks or out_path.suffix.lower() != ".png":
        safe_roi = str(roi).replace(os.sep, "_").replace("/", "_")
        return str(
            out_path
            / f"{safe_roi}_chunk_{int(chunk_id)}_reference_candidate_comparison.png"
        )
    return str(out_path)


def _load_candidate_table(analysis_out: str) -> pd.DataFrame:
    path = os.path.join(analysis_out, DEFAULT_QC_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing baseline candidate QC table: {path}")
    return pd.read_csv(path)


def _find_candidate_row(table: pd.DataFrame, roi: str, chunk_id: int) -> dict[str, Any]:
    if "roi" not in table.columns or "chunk_id" not in table.columns:
        raise ValueError("Baseline candidate table must contain roi and chunk_id columns.")
    matches = table[
        (table["roi"].astype(str) == str(roi))
        & (table["chunk_id"].astype(int) == int(chunk_id))
    ]
    if matches.empty:
        raise ValueError(
            f"No baseline candidate row found for ROI '{roi}' chunk {int(chunk_id)}."
        )
    return matches.iloc[0].to_dict()


def _load_chunk_traces(analysis_out: str, roi: str, chunk_id: int) -> dict[str, np.ndarray]:
    cache_path = os.path.join(analysis_out, DEFAULT_CACHE_NAME)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Missing phasic trace cache: {cache_path}")
    try:
        with open_phasic_cache(cache_path) as cache:
            resolved_roi = resolve_cache_roi(cache, roi)
            time_sec, sig_raw, uv_raw, fit_ref = load_cache_chunk_fields(
                cache,
                resolved_roi,
                int(chunk_id),
                ["time_sec", "sig_raw", "uv_raw", "fit_ref"],
            )
            chunk_group = cache[f"roi/{resolved_roi}/chunk_{int(chunk_id)}"]
            stored_baseline = (
                np.asarray(chunk_group["baseline_ref_candidate"][()], dtype=float).reshape(-1)
                if "baseline_ref_candidate" in chunk_group
                else None
            )
    except CacheReadError as exc:
        raise RuntimeError(
            f"Unable to load ROI '{roi}' chunk {int(chunk_id)} from phasic cache: {exc}"
        ) from exc
    out = {
        "time_sec": np.asarray(time_sec, dtype=float).reshape(-1),
        "sig_raw": np.asarray(sig_raw, dtype=float).reshape(-1),
        "uv_raw": np.asarray(uv_raw, dtype=float).reshape(-1),
        "fit_ref": np.asarray(fit_ref, dtype=float).reshape(-1),
    }
    if stored_baseline is not None:
        if stored_baseline.shape != out["sig_raw"].shape:
            raise RuntimeError(
                "Stored baseline_ref_candidate length does not match sig_raw "
                f"for ROI '{roi}' chunk {int(chunk_id)}: "
                f"{stored_baseline.shape} vs {out['sig_raw'].shape}"
            )
        out["baseline_ref_candidate"] = stored_baseline
    return out


def _float_from_row(row: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _recompute_baseline_candidate_trace(
    traces: dict[str, np.ndarray],
    row: dict[str, Any],
) -> tuple[np.ndarray, str]:
    stored = traces.get("baseline_ref_candidate")
    if stored is not None:
        return np.asarray(stored, dtype=float).reshape(-1), "stored_hdf5"

    t = np.asarray(traces["time_sec"], dtype=float)
    if t.size >= 2:
        dt = float(np.nanmedian(np.diff(t)))
        fs = 1.0 / dt if np.isfinite(dt) and dt > 0.0 else 1.0
    else:
        fs = 1.0
    actual_window = _float_from_row(row, "baseline_ref_actual_smoothing_window_sec", 300.0)
    candidate = compute_baseline_reference_candidate(
        signal=traces["sig_raw"],
        reference=traces["uv_raw"],
        fs=fs,
        smoothing_window_sec=actual_window,
        default_smoothing_window_sec=actual_window,
        min_smoothing_window_sec=_float_from_row(
            row,
            "baseline_ref_min_smoothing_window_sec",
            60.0,
        ),
        max_window_fraction_of_chunk=_float_from_row(
            row,
            "baseline_ref_max_window_fraction_of_chunk",
            0.75,
        ),
        large_window_fraction_warning=_float_from_row(
            row,
            "baseline_ref_large_window_fraction_warning",
            0.50,
        ),
    )
    trace = candidate.get("baseline_ref_candidate")
    if trace is None:
        raise RuntimeError(
            f"Baseline candidate trace could not be recomputed: "
            f"{candidate.get('baseline_ref_warning') or candidate.get('baseline_ref_status')}"
        )
    return np.asarray(trace, dtype=float).reshape(-1), "recomputed_from_metadata"


def _compute_baseline_fit_diagnostics(
    traces: dict[str, np.ndarray],
    row: dict[str, Any],
    plotted_baseline: np.ndarray | None = None,
) -> dict[str, Any]:
    t = np.asarray(traces["time_sec"], dtype=float)
    if t.size >= 2:
        dt = float(np.nanmedian(np.diff(t)))
        fs = 1.0 / dt if np.isfinite(dt) and dt > 0.0 else 1.0
    else:
        fs = 1.0
    actual_window = _float_from_row(row, "baseline_ref_actual_smoothing_window_sec", 300.0)
    diagnostics = compute_baseline_reference_candidate(
        signal=traces["sig_raw"],
        reference=traces["uv_raw"],
        fs=fs,
        smoothing_window_sec=actual_window,
        default_smoothing_window_sec=actual_window,
        min_smoothing_window_sec=_float_from_row(
            row,
            "baseline_ref_min_smoothing_window_sec",
            60.0,
        ),
        max_window_fraction_of_chunk=_float_from_row(
            row,
            "baseline_ref_max_window_fraction_of_chunk",
            0.75,
        ),
        large_window_fraction_warning=_float_from_row(
            row,
            "baseline_ref_large_window_fraction_warning",
            0.50,
        ),
        return_diagnostics=True,
    )
    diagnostics["baseline_fit_diagnostics_source"] = "recomputed_from_metadata"
    recomputed = diagnostics.get("baseline_ref_candidate")
    if plotted_baseline is not None and recomputed is not None:
        plotted = np.asarray(plotted_baseline, dtype=float).reshape(-1)
        recomputed_arr = np.asarray(recomputed, dtype=float).reshape(-1)
        if plotted.shape == recomputed_arr.shape:
            diff = plotted - recomputed_arr
            finite = diff[np.isfinite(diff)]
            rms = float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")
            max_abs = float(np.max(np.abs(finite))) if finite.size else float("nan")
            diagnostics["baseline_ref_recomputed_vs_plotted_rms"] = rms
            diagnostics["baseline_ref_recomputed_vs_plotted_max_abs"] = max_abs
            diagnostics["baseline_ref_recomputed_diff_warning"] = bool(
                np.isfinite(rms) and rms > 1e-6
            )
    return diagnostics


def _scaled_for_display(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    finite_v = values[np.isfinite(values)]
    finite_t = target[np.isfinite(target)]
    if finite_v.size < 2 or finite_t.size < 2:
        return values.copy()
    v_range = float(np.percentile(finite_v, 95.0) - np.percentile(finite_v, 5.0))
    t_range = float(np.percentile(finite_t, 95.0) - np.percentile(finite_t, 5.0))
    if not np.isfinite(v_range) or abs(v_range) <= 1e-12:
        return values.copy()
    return (values - float(np.nanmedian(finite_v))) * (t_range / v_range) + float(
        np.nanmedian(finite_t)
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return values.copy()
    scale = float(np.nanstd(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        return values - float(np.nanmedian(finite))
    return (values - float(np.nanmedian(finite))) / scale


def _robust_limits(*series: np.ndarray) -> tuple[float, float] | None:
    values = []
    for arr in series:
        flat = np.asarray(arr, dtype=float).reshape(-1)
        finite = flat[np.isfinite(flat)]
        if finite.size:
            values.append(finite)
    if not values:
        return None
    combined = np.concatenate(values)
    if combined.size == 0:
        return None
    lo = float(np.percentile(combined, 1.0))
    hi = float(np.percentile(combined, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(combined))
        hi = float(np.nanmax(combined))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        center = float(lo)
        span = max(abs(center) * 0.05, 1.0)
        return center - span, center + span
    margin = max((hi - lo) * 0.08, 1e-9)
    return lo - margin, hi + margin


def _compact_field(row: dict[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return str(value)


def _wrapped_metadata_text(
    *,
    row: dict[str, Any],
    roi: str,
    chunk_id: int,
    baseline_source: str,
    title_extra: str = "",
    width: int = 130,
) -> str:
    fields = [
        f"ROI={roi}",
        f"chunk={int(chunk_id)}",
        f"dynamic_fit_qc_severity={_compact_field(row, 'dynamic_fit_qc_severity')}",
        f"reference_comparison_class={_compact_field(row, 'reference_comparison_class')}",
        f"soft_flags={_compact_field(row, 'dynamic_fit_qc_soft_flags')}",
        f"hard_flags={_compact_field(row, 'dynamic_fit_qc_hard_flags')}",
        f"comparison_flags={_compact_field(row, 'reference_comparison_flags')}",
        f"baseline_window_sec={_compact_field(row, 'baseline_ref_actual_smoothing_window_sec')}",
        f"baseline_source={baseline_source}",
    ]
    if row.get("baseline_fit_relationship_class"):
        fields.append(f"fit_relationship={row.get('baseline_fit_relationship_class')}")
    warning = _compact_field(row, "baseline_ref_smoothing_window_warning")
    if warning:
        fields.append(f"window_warning={warning}")
    if title_extra:
        fields.append(f"note={title_extra}")
    return textwrap.fill(" | ".join(fields), width=width)


def build_reference_candidate_comparison_figure(
    *,
    traces: dict[str, np.ndarray],
    baseline_candidate: np.ndarray,
    row: dict[str, Any],
    roi: str,
    chunk_id: int,
    baseline_source: str = "",
    show_raw_reference: bool = False,
    include_normalized_overlay: bool = False,
    include_reference_difference: bool = False,
    include_baseline_fit_diagnostics: bool = False,
    baseline_fit_diagnostics: dict[str, Any] | None = None,
    metadata_box: bool = True,
    title_extra: str = "",
):
    panels = (
        3
        + int(bool(include_baseline_fit_diagnostics)) * 2
        + int(bool(include_reference_difference))
        + int(bool(include_normalized_overlay))
    )
    fig_height = 9.2 + 1.7 * max(0, panels - 3)
    fig, axes = plt.subplots(panels, 1, figsize=(11.5, fig_height), sharex=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    next_axis = 3

    t = traces["time_sec"]
    sig = traces["sig_raw"]
    uv = traces["uv_raw"]
    fit_ref = traces["fit_ref"]
    baseline = np.asarray(baseline_candidate, dtype=float).reshape(-1)
    uv_scaled = _scaled_for_display(uv, sig)
    residual_dynamic = sig - fit_ref
    residual_baseline = sig - baseline
    reference_difference = fit_ref - baseline
    baseline_fit_diagnostics = (
        baseline_fit_diagnostics if isinstance(baseline_fit_diagnostics, dict) else {}
    )
    metadata_row = dict(row)
    if include_baseline_fit_diagnostics and baseline_fit_diagnostics:
        metadata_row["baseline_fit_relationship_class"] = baseline_fit_diagnostics.get(
            "baseline_fit_relationship_class", ""
        )

    if metadata_box:
        fig.text(
            0.012,
            0.985,
            _wrapped_metadata_text(
                row=metadata_row,
                roi=roi,
                chunk_id=chunk_id,
                baseline_source=baseline_source,
                title_extra=title_extra,
            ),
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "0.75"},
        )

    ax1.plot(t, sig, color="forestgreen", linewidth=0.8, label="raw signal")
    ax1.plot(
        t,
        uv_scaled,
        color="purple",
        linewidth=0.75,
        alpha=0.75,
        label="raw reference (scaled for display)",
    )
    ax1.plot(t, fit_ref, color="black", linestyle="--", linewidth=0.9, label="dynamic fitted reference")
    ax1.plot(t, baseline, color="#1f77b4", linestyle=":", linewidth=1.0, label="baseline-only candidate")
    ax1.set_title("Raw traces and candidate references")
    ax1.set_ylabel("Signal frame")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    if show_raw_reference:
        ax1b = ax1.twinx()
        ax1b.plot(t, uv, color="purple", linewidth=0.45, alpha=0.25, label="raw reference")
        ax1b.set_ylabel("Raw reference", color="purple")
        ax1b.tick_params(axis="y", colors="purple")

    ax2.plot(t, fit_ref, color="black", linestyle="--", linewidth=0.95, label="dynamic fitted reference")
    ax2.plot(t, baseline, color="#1f77b4", linestyle=":", linewidth=1.05, label="baseline-only candidate")
    ax2.set_title("Candidate reference traces")
    ax2.set_ylabel("Reference frame")
    limits = _robust_limits(fit_ref, baseline)
    if limits is not None:
        ax2.set_ylim(*limits)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    ax3.plot(
        t,
        residual_dynamic,
        color="darkorange",
        linewidth=0.8,
        alpha=0.85,
        label="signal - dynamic reference",
    )
    ax3.plot(
        t,
        residual_baseline,
        color="teal",
        linewidth=0.8,
        alpha=0.85,
        label="signal - baseline candidate",
    )
    ax3.set_title("Diagnostic residuals: signal minus candidate reference")
    ax3.set_ylabel("Residual")
    limits = _robust_limits(residual_dynamic, residual_baseline)
    if limits is not None:
        ax3.set_ylim(*limits)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best", fontsize=8)

    if include_reference_difference:
        ax_diff = axes[next_axis]
        next_axis += 1
        ax_diff.plot(
            t,
            reference_difference,
            color="crimson",
            linewidth=0.85,
            label="dynamic reference - baseline candidate",
        )
        ax_diff.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        ax_diff.set_title("Difference between candidate references")
        ax_diff.set_ylabel("Difference")
        limits = _robust_limits(reference_difference)
        if limits is not None:
            ax_diff.set_ylim(*limits)
        ax_diff.grid(True, alpha=0.25)
        ax_diff.legend(loc="best", fontsize=8)

    if include_baseline_fit_diagnostics:
        ax_inputs = axes[next_axis]
        next_axis += 1
        smoothed_signal = np.asarray(
            baseline_fit_diagnostics.get("baseline_ref_smoothed_signal", []),
            dtype=float,
        ).reshape(-1)
        smoothed_reference = np.asarray(
            baseline_fit_diagnostics.get("baseline_ref_smoothed_reference", []),
            dtype=float,
        ).reshape(-1)
        recomputed_candidate = np.asarray(
            baseline_fit_diagnostics.get("baseline_ref_candidate", baseline),
            dtype=float,
        ).reshape(-1)
        if smoothed_signal.shape == sig.shape:
            ax_inputs.plot(
                t,
                sig,
                color="forestgreen",
                linewidth=0.45,
                alpha=0.25,
                label="raw signal",
            )
            ax_inputs.plot(
                t,
                smoothed_signal,
                color="forestgreen",
                linewidth=0.95,
                label="smoothed signal fit target",
            )
        if recomputed_candidate.shape == sig.shape:
            ax_inputs.plot(
                t,
                recomputed_candidate,
                color="#1f77b4",
                linestyle=":",
                linewidth=1.0,
                label="recomputed baseline candidate",
            )
        if baseline.shape == sig.shape:
            ax_inputs.plot(
                t,
                baseline,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
                label="plotted baseline candidate",
            )
        ax_inputs.set_title("Baseline-candidate fit inputs")
        ax_inputs.set_ylabel("Signal units")
        limits = _robust_limits(smoothed_signal, recomputed_candidate, baseline)
        if limits is not None:
            ax_inputs.set_ylim(*limits)
        warning = bool(baseline_fit_diagnostics.get("baseline_ref_recomputed_diff_warning", False))
        source_note = (
            f"candidate trace source={baseline_source}; "
            "fit diagnostics recomputed_from_metadata"
        )
        if warning:
            source_note += "; WARNING: recomputed diagnostics differ from plotted candidate"
        ax_inputs.text(
            0.01,
            0.97,
            source_note,
            transform=ax_inputs.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.75"},
        )
        ax_inputs.grid(True, alpha=0.25)
        ax_inputs.legend(loc="best", fontsize=8)

        ax_relation = axes[next_axis]
        next_axis += 1
        finite = np.isfinite(smoothed_reference) & np.isfinite(smoothed_signal)
        included = np.asarray(
            baseline_fit_diagnostics.get("baseline_ref_fit_included_mask", finite),
            dtype=bool,
        ).reshape(-1)
        if included.shape != finite.shape:
            included = finite.copy()
        excluded = finite & ~included
        if np.any(included):
            ax_relation.scatter(
                smoothed_reference[included],
                smoothed_signal[included],
                s=8,
                alpha=0.45,
                color="steelblue",
                label="included fit samples",
            )
        if np.any(excluded):
            ax_relation.scatter(
                smoothed_reference[excluded],
                smoothed_signal[excluded],
                s=10,
                alpha=0.5,
                color="crimson",
                label="excluded residual samples",
            )
        x_finite = smoothed_reference[finite]
        if x_finite.size:
            x_line = np.linspace(float(np.nanmin(x_finite)), float(np.nanmax(x_finite)), 200)
            initial_slope = baseline_fit_diagnostics.get("baseline_ref_initial_slope")
            initial_intercept = baseline_fit_diagnostics.get("baseline_ref_initial_intercept")
            final_slope = baseline_fit_diagnostics.get("baseline_ref_final_slope")
            final_intercept = baseline_fit_diagnostics.get("baseline_ref_final_intercept")
            try:
                ax_relation.plot(
                    x_line,
                    float(initial_intercept) + float(initial_slope) * x_line,
                    color="0.45",
                    linestyle=":",
                    linewidth=1.0,
                    label="initial fit",
                )
            except Exception:
                pass
            try:
                ax_relation.plot(
                    x_line,
                    float(final_intercept) + float(final_slope) * x_line,
                    color="black",
                    linestyle="-",
                    linewidth=1.1,
                    label="final fit",
                )
            except Exception:
                pass
        relationship = baseline_fit_diagnostics.get(
            "baseline_fit_relationship_class",
            classify_baseline_fit_relationship(
                slope=baseline_fit_diagnostics.get("baseline_ref_final_slope"),
                corr=baseline_fit_diagnostics.get(
                    "baseline_ref_smoothed_signal_reference_corr"
                ),
            ),
        )
        annotation = (
            f"slope={_compact_field(baseline_fit_diagnostics, 'baseline_ref_final_slope')} | "
            f"intercept={_compact_field(baseline_fit_diagnostics, 'baseline_ref_final_intercept')} | "
            f"corr={_compact_field(baseline_fit_diagnostics, 'baseline_ref_smoothed_signal_reference_corr')} | "
            f"excluded={_compact_field(baseline_fit_diagnostics, 'baseline_ref_residual_exclusion_fraction')} | "
            f"stage={_compact_field(baseline_fit_diagnostics, 'baseline_ref_fit_stage')} | "
            f"status={_compact_field(baseline_fit_diagnostics, 'baseline_ref_status')} | "
            f"class={relationship}"
        )
        ax_relation.text(
            0.01,
            0.97,
            textwrap.fill(annotation, width=115),
            transform=ax_relation.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.75"},
        )
        ax_relation.set_title("Smoothed signal vs smoothed reference fit")
        ax_relation.set_xlabel("Smoothed reference")
        ax_relation.set_ylabel("Smoothed signal")
        ax_relation.grid(True, alpha=0.25)
        ax_relation.legend(loc="best", fontsize=8)

    if include_normalized_overlay:
        ax_norm = axes[next_axis]
        ax_norm.plot(t, _zscore(sig), color="forestgreen", linewidth=0.8, label="signal z")
        ax_norm.plot(t, _zscore(uv), color="purple", linewidth=0.75, alpha=0.8, label="reference z")
        ax_norm.plot(t, _zscore(fit_ref), color="black", linestyle="--", linewidth=0.9, label="dynamic ref z")
        ax_norm.plot(t, _zscore(baseline), color="#1f77b4", linestyle=":", linewidth=1.0, label="baseline candidate z")
        ax_norm.set_title("Normalized shape overlay")
        ax_norm.set_ylabel("Z-score")
        ax_norm.grid(True, alpha=0.25)
        ax_norm.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90 if metadata_box else 0.98))
    return fig


def plot_one(
    *,
    analysis_out: str,
    roi: str,
    chunk_id: int,
    output_path: str,
    show_raw_reference: bool = False,
    include_normalized_overlay: bool = False,
    include_reference_difference: bool = False,
    include_baseline_fit_diagnostics: bool = False,
    metadata_box: bool = True,
    title_extra: str = "",
    dpi: int = 150,
) -> str:
    table = _load_candidate_table(analysis_out)
    row = _find_candidate_row(table, roi, int(chunk_id))
    traces = _load_chunk_traces(analysis_out, roi, int(chunk_id))
    baseline, baseline_source = _recompute_baseline_candidate_trace(traces, row)
    baseline_fit_diagnostics = (
        _compute_baseline_fit_diagnostics(traces, row, plotted_baseline=baseline)
        if include_baseline_fit_diagnostics
        else {}
    )
    fig = build_reference_candidate_comparison_figure(
        traces=traces,
        baseline_candidate=baseline,
        row=row,
        roi=roi,
        chunk_id=int(chunk_id),
        baseline_source=baseline_source,
        show_raw_reference=show_raw_reference,
        include_normalized_overlay=include_normalized_overlay,
        include_reference_difference=include_reference_difference,
        include_baseline_fit_diagnostics=include_baseline_fit_diagnostics,
        baseline_fit_diagnostics=baseline_fit_diagnostics,
        metadata_box=metadata_box,
        title_extra=title_extra,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    try:
        chunk_ids = _parse_chunk_ids(args.chunk_id, args.chunks)
        multiple = len(chunk_ids) > 1
        outputs = []
        for chunk_id in chunk_ids:
            out_path = _resolve_output_path(
                args.analysis_out,
                args.roi,
                chunk_id,
                args.out,
                multiple_chunks=multiple,
            )
            outputs.append(
                plot_one(
                    analysis_out=args.analysis_out,
                    roi=args.roi,
                    chunk_id=chunk_id,
                    output_path=out_path,
                    show_raw_reference=bool(args.show_raw_reference),
                    include_normalized_overlay=bool(args.include_normalized_overlay),
                    include_reference_difference=bool(args.include_reference_difference),
                    include_baseline_fit_diagnostics=bool(args.include_baseline_fit_diagnostics),
                    metadata_box=not bool(args.no_metadata_box),
                    title_extra=args.title_extra,
                    dpi=int(args.dpi),
                )
            )
        for output in outputs:
            print(output)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
