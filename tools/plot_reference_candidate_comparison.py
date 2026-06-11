#!/usr/bin/env python3
"""Plot diagnostic dynamic-vs-baseline reference candidate comparisons."""

from __future__ import annotations

import argparse
import os
import sys
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


def _compact_field(row: dict[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return str(value)


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
    title_extra: str = "",
):
    panels = 3 if include_normalized_overlay else 2
    fig, axes = plt.subplots(panels, 1, figsize=(11, 7.5 if panels == 2 else 9.5), sharex=True)
    if panels == 2:
        ax1, ax2 = axes
        ax3 = None
    else:
        ax1, ax2, ax3 = axes

    t = traces["time_sec"]
    sig = traces["sig_raw"]
    uv = traces["uv_raw"]
    fit_ref = traces["fit_ref"]
    baseline = np.asarray(baseline_candidate, dtype=float).reshape(-1)
    uv_scaled = _scaled_for_display(uv, sig)

    title_parts = [
        f"ROI {roi}",
        f"chunk {int(chunk_id)}",
        f"dynamic severity={_compact_field(row, 'dynamic_fit_qc_severity')}",
        f"class={_compact_field(row, 'reference_comparison_class')}",
    ]
    if title_extra:
        title_parts.append(str(title_extra))
    ax1.set_title(" | ".join(title_parts))
    subtitle = (
        f"soft={_compact_field(row, 'dynamic_fit_qc_soft_flags')} | "
        f"hard={_compact_field(row, 'dynamic_fit_qc_hard_flags')} | "
        f"comparison_flags={_compact_field(row, 'reference_comparison_flags')} | "
        f"baseline_window={_compact_field(row, 'baseline_ref_actual_smoothing_window_sec')}s"
    )
    if baseline_source:
        subtitle += f" | baseline_source={baseline_source}"
    warning = _compact_field(row, "baseline_ref_smoothing_window_warning")
    if warning:
        subtitle += f" | window_warning={warning}"
    ax1.text(
        0.01,
        0.98,
        subtitle,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.75"},
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
    ax1.set_ylabel("Signal frame")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)

    if show_raw_reference:
        ax1b = ax1.twinx()
        ax1b.plot(t, uv, color="purple", linewidth=0.45, alpha=0.25, label="raw reference")
        ax1b.set_ylabel("Raw reference", color="purple")
        ax1b.tick_params(axis="y", colors="purple")

    ax2.plot(t, fit_ref, color="black", linestyle="--", linewidth=0.9, label="dynamic fitted reference")
    ax2.plot(t, baseline, color="#1f77b4", linestyle=":", linewidth=1.0, label="baseline-only candidate")
    ax2.plot(
        t,
        sig - fit_ref,
        color="darkorange",
        linewidth=0.7,
        alpha=0.75,
        label="signal - dynamic ref (diagnostic)",
    )
    ax2.plot(
        t,
        sig - baseline,
        color="teal",
        linewidth=0.7,
        alpha=0.75,
        label="signal - baseline candidate (diagnostic)",
    )
    ax2.set_ylabel("Diagnostic traces")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8)

    if ax3 is not None:
        ax3.plot(t, _zscore(sig), color="forestgreen", linewidth=0.8, label="signal z")
        ax3.plot(t, _zscore(uv), color="purple", linewidth=0.75, alpha=0.8, label="reference z")
        ax3.plot(t, _zscore(fit_ref), color="black", linestyle="--", linewidth=0.9, label="dynamic ref z")
        ax3.plot(t, _zscore(baseline), color="#1f77b4", linestyle=":", linewidth=1.0, label="baseline candidate z")
        ax3.set_ylabel("Robust shape overlay")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.set_xlabel("Time (s)")
    else:
        ax2.set_xlabel("Time (s)")

    fig.tight_layout()
    return fig


def plot_one(
    *,
    analysis_out: str,
    roi: str,
    chunk_id: int,
    output_path: str,
    show_raw_reference: bool = False,
    include_normalized_overlay: bool = False,
    title_extra: str = "",
    dpi: int = 150,
) -> str:
    table = _load_candidate_table(analysis_out)
    row = _find_candidate_row(table, roi, int(chunk_id))
    traces = _load_chunk_traces(analysis_out, roi, int(chunk_id))
    baseline, baseline_source = _recompute_baseline_candidate_trace(traces, row)
    fig = build_reference_candidate_comparison_figure(
        traces=traces,
        baseline_candidate=baseline,
        row=row,
        roi=roi,
        chunk_id=int(chunk_id),
        baseline_source=baseline_source,
        show_raw_reference=show_raw_reference,
        include_normalized_overlay=include_normalized_overlay,
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
