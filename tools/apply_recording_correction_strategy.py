#!/usr/bin/env python3
"""Preview/export explicit recording-level correction strategy outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photometry_pipeline.io.hdf5_cache_reader import (  # noqa: E402
    CacheReadError,
    list_cache_chunk_ids,
    list_cache_source_files,
    load_cache_chunk_fields,
    open_phasic_cache,
)
from tools.export_signal_only_f0_dff_diagnostics import (  # noqa: E402
    FLAG_EXTREME_SEVERE,
    FLAG_HIGH_EXTRAPOLATION,
    FLAG_INSUFFICIENT_ANCHORS,
    FLAG_INSUFFICIENT_LOW_SUPPORT,
    FLAG_LARGE_ANCHOR_GAP,
    FLAG_LOW_CONFIDENCE,
    FLAG_UNAVAILABLE,
    _as_flag_list,
    _find_phasic_cache,
    _json_safe,
    _load_or_compute_f0,
    _load_qc_records,
    _load_signal_only_f0_config,
    _safe_roi_name,
    compute_signal_only_f0_dff_diagnostic,
)

STRATEGIES = ("dynamic_fit", "signal_only_f0", "no_correction")
DEFAULT_DIR_NAME = "applied_correction_preview"
F0_SOURCE_SIGNAL_ONLY = "uncapped_core_state_aware_recompute"
FLAG_APPLIED_TRACE_PARTIAL = "APPLIED_TRACE_PARTIAL"

SUMMARY_FIELDS = [
    "roi",
    "recording_key",
    "requested_correction_strategy",
    "correction_strategy_selection",
    "applied_correction_strategy",
    "applied_trace_source",
    "applied_trace_units",
    "applied_trace_available",
    "applied_trace_complete",
    "reason_if_unavailable",
    "n_chunks",
    "n_chunks_available",
    "n_chunks_unavailable",
    "n_chunks_caution",
    "n_chunks_severe",
    "applied_trace_review_required",
    "applied_trace_warning_level",
    "applied_trace_flags",
    "hdf5_modified",
    "feature_detection_input",
    "output_dir",
    "f0_source_for_signal_only_f0",
    "n_chunks_signal_only_f0_low_confidence",
    "n_chunks_signal_only_f0_hard_inspect",
    "n_chunks_signal_only_f0_insufficient_anchors",
    "n_chunks_signal_only_f0_insufficient_low_support",
    "trace_csv",
    "preview_plots",
]


def _parse_chunk_ids(chunks: str | None) -> list[int] | None:
    if chunks is None or not str(chunks).strip():
        return None
    out: list[int] = []
    seen: set[int] = set()
    for part in str(chunks).split(","):
        text = part.strip()
        if not text:
            continue
        chunk_id = int(text)
        if chunk_id not in seen:
            out.append(chunk_id)
            seen.add(chunk_id)
    return out


def _select_output_dir(base_dir: Path, *, overwrite: bool, dry_run: bool) -> Path:
    if dry_run or overwrite or not base_dir.exists() or not any(base_dir.iterdir()):
        return base_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir.with_name(f"{base_dir.name}_{stamp}")


def _finite_level(levels: list[str]) -> str:
    if "severe" in levels:
        return "severe"
    if "caution" in levels:
        return "caution"
    if "info" in levels:
        return "info"
    return "none"


def _recording_key(cache, phasic_path: Path) -> str:
    try:
        sources = list_cache_source_files(cache)
    except Exception:
        sources = []
    if sources:
        return ";".join(str(x) for x in sources)
    return phasic_path.name


def _as_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _write_summary(output_dir: Path, summary: dict[str, Any], chunks: list[dict[str, Any]]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "applied_correction_summary.csv"
    json_path = output_dir / "applied_correction_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        row = dict(summary)
        for key, value in list(row.items()):
            if isinstance(value, (list, tuple)):
                row[key] = ";".join(str(x) for x in value)
            elif isinstance(value, bool):
                row[key] = str(value).lower()
            elif isinstance(value, float) and not math.isfinite(value):
                row[key] = ""
        writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})
    payload = {**summary, "chunks": chunks}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return csv_path, json_path


def _write_trace_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _append_trace_rows(
    out: list[dict[str, Any]],
    *,
    roi: str,
    chunk_id: int,
    time_sec: np.ndarray,
    applied_dff: np.ndarray,
    extra: dict[str, np.ndarray],
) -> None:
    t = np.asarray(time_sec, dtype=float).reshape(-1)
    y = np.asarray(applied_dff, dtype=float).reshape(-1)
    for idx, value in enumerate(y):
        row: dict[str, Any] = {
            "roi": roi,
            "chunk_id": int(chunk_id),
            "sample_index": int(idx),
            "time_sec": float(t[idx]) if idx < t.size and np.isfinite(t[idx]) else "",
            "applied_dff": float(value) if np.isfinite(value) else "",
        }
        for key, arr in extra.items():
            values = np.asarray(arr, dtype=float).reshape(-1)
            row[key] = float(values[idx]) if idx < values.size and np.isfinite(values[idx]) else ""
        out.append(row)


def _chunk_group_has(cache, roi: str, chunk_id: int, name: str) -> bool:
    group = cache.get(f"roi/{roi}/chunk_{int(chunk_id)}")
    return group is not None and name in group


def _plot_dynamic_fit(
    out_path: Path,
    *,
    roi: str,
    chunk_id: int,
    time_sec: np.ndarray,
    dff: np.ndarray,
    warning_level: str,
    flags: list[str],
    dpi: int,
) -> None:
    t = np.asarray(time_sec, dtype=float).reshape(-1)
    y = np.asarray(dff, dtype=float).reshape(-1)
    x = t - float(t[np.isfinite(t)][0]) if t.shape == y.shape and np.any(np.isfinite(t)) else np.arange(y.size)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(x, y, color="black", linewidth=0.8, label="dynamic_fit_dff")
    axes[0].axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    axes[0].set_ylabel("Applied dF/F")
    axes[0].set_title(f"ROI={roi} chunk={int(chunk_id)} | applied strategy = dynamic_fit")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)
    axes[1].axis("off")
    axes[1].text(
        0.01,
        0.9,
        f"source = dynamic_fit_dff | warning = {warning_level} | flags = {';'.join(flags)}",
        va="top",
        ha="left",
        fontsize=8,
        wrap=True,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _plot_signal_only_f0(
    out_path: Path,
    *,
    roi: str,
    chunk_id: int,
    time_sec: np.ndarray,
    signal: np.ndarray,
    f0: np.ndarray,
    dff: np.ndarray,
    warning_level: str,
    flags: list[str],
    dpi: int,
) -> None:
    t = np.asarray(time_sec, dtype=float).reshape(-1)
    sig = np.asarray(signal, dtype=float).reshape(-1)
    base = np.asarray(f0, dtype=float).reshape(-1)
    y = np.asarray(dff, dtype=float).reshape(-1)
    x = t - float(t[np.isfinite(t)][0]) if t.shape == sig.shape and np.any(np.isfinite(t)) else np.arange(sig.size)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 2, 1]})
    axes[0].plot(x, sig, color="forestgreen", linewidth=0.8, label="signal")
    axes[0].plot(x, base, color="black", linestyle="--", linewidth=0.9, label="signal_only_f0_uncapped_for_dff")
    axes[0].set_ylabel("Signal / F0")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, y, color="darkorange", linewidth=0.8, label="applied_dff")
    axes[1].axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    finite = y[np.isfinite(y)]
    if finite.size:
        ymin = min(float(np.min(finite)), float(np.percentile(finite, 1.0)), 0.0)
        ymax = max(float(np.max(finite)), float(np.percentile(finite, 99.0)), 0.0)
        pad = max((ymax - ymin) * 0.05, 1e-6)
        axes[1].set_ylim(ymin - pad, ymax + pad)
    axes[1].set_ylabel("Applied dF/F")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    axes[2].axis("off")
    axes[2].text(
        0.01,
        0.95,
        "applied strategy = signal_only_f0 | source = uncapped core state-aware F0\n"
        f"ROI={roi} chunk={int(chunk_id)} | warning = {warning_level}\n"
        f"flags = {';'.join(flags)}",
        va="top",
        ha="left",
        fontsize=8,
        wrap=True,
    )
    fig.suptitle(f"Explicit recording-level strategy selection | {roi} chunk {int(chunk_id)}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _base_summary(
    *,
    roi: str,
    recording_key: str,
    strategy: str,
    output_dir: Path,
) -> dict[str, Any]:
    trace_source = {
        "dynamic_fit": "dynamic_fit_dff",
        "signal_only_f0": "signal_only_f0_dff",
        "no_correction": "none",
    }[strategy]
    trace_units = "dff" if strategy in {"dynamic_fit", "signal_only_f0"} else "none"
    return {
        "roi": roi,
        "recording_key": recording_key,
        "requested_correction_strategy": strategy,
        "correction_strategy_selection": "explicit",
        "applied_correction_strategy": strategy,
        "applied_trace_source": trace_source,
        "applied_trace_units": trace_units,
        "applied_trace_available": False,
        "applied_trace_complete": False,
        "reason_if_unavailable": "",
        "n_chunks": 0,
        "n_chunks_available": 0,
        "n_chunks_unavailable": 0,
        "n_chunks_caution": 0,
        "n_chunks_severe": 0,
        "applied_trace_review_required": False,
        "applied_trace_warning_level": "none",
        "applied_trace_flags": "",
        "hdf5_modified": False,
        "feature_detection_input": False,
        "output_dir": str(output_dir),
        "f0_source_for_signal_only_f0": F0_SOURCE_SIGNAL_ONLY if strategy == "signal_only_f0" else "",
        "n_chunks_signal_only_f0_low_confidence": 0,
        "n_chunks_signal_only_f0_hard_inspect": 0,
        "n_chunks_signal_only_f0_insufficient_anchors": 0,
        "n_chunks_signal_only_f0_insufficient_low_support": 0,
        "trace_csv": "",
        "preview_plots": "",
    }


def apply_recording_correction_strategy(
    phasic_out: str | os.PathLike[str],
    *,
    roi: str,
    strategy: str,
    output_dir: str | os.PathLike[str] | None = None,
    chunks: list[int] | None = None,
    max_preview_chunks: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    dpi: int = 150,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")
    phasic_path = Path(phasic_out).resolve()
    cache_path = _find_phasic_cache(phasic_path)
    requested_output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else phasic_path / "qc" / DEFAULT_DIR_NAME
    )
    selected_output = _select_output_dir(requested_output, overwrite=overwrite, dry_run=dry_run)

    qc_records = _load_qc_records(phasic_path)
    config, using_defaults, config_source = _load_signal_only_f0_config(phasic_path)
    safe_roi = _safe_roi_name(roi)
    trace_path = selected_output / f"{safe_roi}_{strategy}_applied_trace.csv"

    chunk_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    preview_plots: list[str] = []
    warning_levels: list[str] = []
    all_flags: Counter[str] = Counter()
    preview_ids = set(int(x) for x in chunks) if chunks is not None else None
    preview_budget = None if max_preview_chunks is None else max(0, int(max_preview_chunks))

    with open_phasic_cache(str(cache_path)) as cache:
        recording_key = _recording_key(cache, phasic_path)
        summary = _base_summary(
            roi=roi,
            recording_key=recording_key,
            strategy=strategy,
            output_dir=selected_output,
        )
        all_chunk_ids = list_cache_chunk_ids(cache)

        if strategy == "no_correction":
            summary.update(
                {
                    "n_chunks": int(len(all_chunk_ids)),
                    "n_chunks_available": 0,
                    "n_chunks_unavailable": int(len(all_chunk_ids)),
                    "applied_trace_complete": False,
                    "reason_if_unavailable": "no_correction_selected_no_corrected_applied_dff_produced",
                    "applied_trace_review_required": True,
                    "applied_trace_warning_level": "info",
                    "applied_trace_flags": "NO_CORRECTION_SELECTED_NO_APPLIED_DFF",
                }
            )
            chunk_rows = [
                {
                    "roi": roi,
                    "chunk_id": int(chunk_id),
                    "available": False,
                    "reason_if_unavailable": "no_correction_selected_no_corrected_applied_dff_produced",
                    "warning_level": "info",
                    "flags": ["NO_CORRECTION_SELECTED_NO_APPLIED_DFF"],
                }
                for chunk_id in all_chunk_ids
            ]
        else:
            for chunk_id in all_chunk_ids:
                chunk: dict[str, Any] = {
                    "roi": roi,
                    "chunk_id": int(chunk_id),
                    "available": False,
                    "reason_if_unavailable": "",
                    "warning_level": "none",
                    "flags": [],
                }
                try:
                    time_sec, = load_cache_chunk_fields(cache, roi, int(chunk_id), ["time_sec"])
                    time_arr = np.asarray(time_sec, dtype=float).reshape(-1)
                    should_plot = (
                        (preview_ids is None or int(chunk_id) in preview_ids)
                        and (preview_budget is None or len(preview_plots) < preview_budget)
                    )
                    if strategy == "dynamic_fit":
                        if not _chunk_group_has(cache, roi, int(chunk_id), "dff"):
                            chunk["reason_if_unavailable"] = "dynamic_fit_dff_unavailable"
                            chunk["warning_level"] = "severe"
                            chunk["flags"] = ["DYNAMIC_FIT_DFF_UNAVAILABLE"]
                        else:
                            dff, = load_cache_chunk_fields(cache, roi, int(chunk_id), ["dff"])
                            dff_arr = np.asarray(dff, dtype=float).reshape(-1)
                            if dff_arr.size == 0 or not np.any(np.isfinite(dff_arr)):
                                chunk["reason_if_unavailable"] = "dynamic_fit_dff_no_finite_values"
                                chunk["warning_level"] = "severe"
                                chunk["flags"] = ["DYNAMIC_FIT_DFF_UNAVAILABLE"]
                            else:
                                chunk["available"] = True
                                _append_trace_rows(
                                    trace_rows,
                                    roi=roi,
                                    chunk_id=int(chunk_id),
                                    time_sec=time_arr,
                                    applied_dff=dff_arr,
                                    extra={"dynamic_fit_dff": dff_arr},
                                )
                                if should_plot:
                                    plot_path = selected_output / f"{safe_roi}_chunk_{int(chunk_id)}_{strategy}_applied_preview.png"
                                    chunk["plot_path"] = str(plot_path)
                                    if not dry_run:
                                        _plot_dynamic_fit(
                                            plot_path,
                                            roi=roi,
                                            chunk_id=int(chunk_id),
                                            time_sec=time_arr,
                                            dff=dff_arr,
                                            warning_level="none",
                                            flags=[],
                                            dpi=dpi,
                                        )
                                    preview_plots.append(str(plot_path))
                    elif strategy == "signal_only_f0":
                        signal, = load_cache_chunk_fields(cache, roi, int(chunk_id), ["sig_raw"])
                        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
                        record = qc_records.get((roi, int(chunk_id)), {})
                        f0, diag_record, f0_source, _h5_avail, _h5_candidate = _load_or_compute_f0(
                            cache,
                            roi,
                            int(chunk_id),
                            signal_arr,
                            time_arr,
                            record,
                            config,
                        )
                        chunk["f0_source_for_signal_only_f0"] = f0_source
                        if f0 is None or np.asarray(f0).reshape(-1).shape != signal_arr.shape:
                            chunk["reason_if_unavailable"] = "signal_only_f0_uncapped_unavailable"
                            chunk["warning_level"] = "severe"
                            chunk["flags"] = [FLAG_UNAVAILABLE]
                        else:
                            dff_diag = compute_signal_only_f0_dff_diagnostic(
                                signal_arr,
                                np.asarray(f0, dtype=float).reshape(-1),
                                qc_record=diag_record,
                            )
                            flags = _as_flag_list(dff_diag.get("diagnostic_flags"))
                            level = str(dff_diag.get("diagnostic_warning_level") or "severe")
                            chunk.update(
                                {
                                    "available": bool(dff_diag.get("available")),
                                    "reason_if_unavailable": dff_diag.get("reason_if_unavailable", ""),
                                    "warning_level": level,
                                    "flags": flags,
                                    "signal_only_f0_candidate_viability": diag_record.get("signal_only_f0_candidate_viability"),
                                    "signal_only_f0_candidate_confidence": diag_record.get("signal_only_f0_candidate_confidence"),
                                    "signal_only_f0_flags": _as_flag_list(diag_record.get("signal_only_f0_flags")),
                                }
                            )
                            dff_arr = dff_diag.get("dff")
                            if chunk["available"] and dff_arr is not None:
                                f0_arr = np.asarray(f0, dtype=float).reshape(-1)
                                dff_values = np.asarray(dff_arr, dtype=float).reshape(-1)
                                _append_trace_rows(
                                    trace_rows,
                                    roi=roi,
                                    chunk_id=int(chunk_id),
                                    time_sec=time_arr,
                                    applied_dff=dff_values,
                                    extra={
                                        "signal": signal_arr,
                                        "signal_only_f0_uncapped_for_dff": f0_arr,
                                    },
                                )
                                if should_plot:
                                    plot_path = selected_output / f"{safe_roi}_chunk_{int(chunk_id)}_{strategy}_applied_preview.png"
                                    chunk["plot_path"] = str(plot_path)
                                    if not dry_run:
                                        _plot_signal_only_f0(
                                            plot_path,
                                            roi=roi,
                                            chunk_id=int(chunk_id),
                                            time_sec=time_arr,
                                            signal=signal_arr,
                                            f0=f0_arr,
                                            dff=dff_values,
                                            warning_level=level,
                                            flags=flags,
                                            dpi=dpi,
                                        )
                                    preview_plots.append(str(plot_path))
                except CacheReadError as exc:
                    chunk["reason_if_unavailable"] = str(exc)
                    chunk["warning_level"] = "severe"
                    chunk["flags"] = [FLAG_UNAVAILABLE]
                except Exception as exc:
                    chunk["reason_if_unavailable"] = f"unexpected_error: {exc}"
                    chunk["warning_level"] = "severe"
                    chunk["flags"] = [FLAG_UNAVAILABLE]

                chunk_rows.append(chunk)
                warning_levels.append(str(chunk.get("warning_level") or "none"))
                all_flags.update(_as_flag_list(chunk.get("flags")))

            available_count = sum(1 for row in chunk_rows if bool(row.get("available")))
            unavailable_count = len(chunk_rows) - available_count
            levels = [str(row.get("warning_level") or "none") for row in chunk_rows]
            if available_count > 0 and unavailable_count > 0:
                all_flags.update([FLAG_APPLIED_TRACE_PARTIAL])
            summary.update(
                {
                    "n_chunks": int(len(chunk_rows)),
                    "n_chunks_available": int(available_count),
                    "n_chunks_unavailable": int(unavailable_count),
                    "n_chunks_caution": int(sum(1 for x in levels if x == "caution")),
                    "n_chunks_severe": int(sum(1 for x in levels if x == "severe")),
                    "applied_trace_available": bool(available_count > 0),
                    "applied_trace_complete": bool(
                        len(chunk_rows) > 0
                        and available_count == len(chunk_rows)
                        and unavailable_count == 0
                    ),
                    "reason_if_unavailable": "" if available_count > 0 else ";".join(
                        sorted({str(row.get("reason_if_unavailable") or "") for row in chunk_rows if row.get("reason_if_unavailable")})
                    ),
                    "applied_trace_review_required": any(x in {"caution", "severe"} for x in levels) or unavailable_count > 0,
                    "applied_trace_warning_level": _finite_level(levels),
                    "applied_trace_flags": ";".join(sorted(all_flags)),
                    "trace_csv": str(trace_path) if available_count > 0 else "",
                }
            )
            if FLAG_APPLIED_TRACE_PARTIAL in _as_flag_list(summary.get("applied_trace_flags")):
                summary["applied_trace_review_required"] = True
                if summary["applied_trace_warning_level"] == "none":
                    summary["applied_trace_warning_level"] = "caution"

            if strategy == "signal_only_f0":
                summary.update(
                    {
                        "n_chunks_signal_only_f0_low_confidence": sum(
                            1
                            for row in chunk_rows
                            if FLAG_LOW_CONFIDENCE in _as_flag_list(row.get("flags"))
                            or str(row.get("signal_only_f0_candidate_confidence", "")).lower() == "low"
                        ),
                        "n_chunks_signal_only_f0_hard_inspect": sum(
                            1
                            for row in chunk_rows
                            if str(row.get("signal_only_f0_candidate_viability", "")).lower() == "hard_inspect"
                            or "SIGNAL_ONLY_F0_HARD_INSPECT" in _as_flag_list(row.get("signal_only_f0_flags"))
                        ),
                        "n_chunks_signal_only_f0_insufficient_anchors": sum(
                            1 for row in chunk_rows if FLAG_INSUFFICIENT_ANCHORS in _as_flag_list(row.get("flags"))
                        ),
                        "n_chunks_signal_only_f0_insufficient_low_support": sum(
                            1 for row in chunk_rows if FLAG_INSUFFICIENT_LOW_SUPPORT in _as_flag_list(row.get("flags"))
                        ),
                    }
                )
                review_flags = set(_as_flag_list(summary.get("applied_trace_flags")))
                severe_or_review = {
                    FLAG_LOW_CONFIDENCE,
                    FLAG_HIGH_EXTRAPOLATION,
                    FLAG_LARGE_ANCHOR_GAP,
                    FLAG_INSUFFICIENT_ANCHORS,
                    FLAG_INSUFFICIENT_LOW_SUPPORT,
                    FLAG_EXTREME_SEVERE,
                }
                if review_flags.intersection(severe_or_review) or summary["n_chunks_signal_only_f0_hard_inspect"]:
                    summary["applied_trace_review_required"] = True
                    if summary["applied_trace_warning_level"] == "none":
                        summary["applied_trace_warning_level"] = "caution"

    if preview_plots:
        summary["preview_plots"] = ";".join(preview_plots)

    summary_csv = selected_output / "applied_correction_summary.csv"
    summary_json = selected_output / "applied_correction_summary.json"
    if not dry_run:
        if trace_rows and strategy in {"dynamic_fit", "signal_only_f0"}:
            fields = ["roi", "chunk_id", "sample_index", "time_sec", "applied_dff"]
            if strategy == "dynamic_fit":
                fields.append("dynamic_fit_dff")
            else:
                fields.extend(["signal", "signal_only_f0_uncapped_for_dff"])
            _write_trace_csv(trace_path, trace_rows, fields)
        _write_summary(selected_output, summary, chunk_rows)

    return {
        "dry_run": bool(dry_run),
        "phasic_out": str(phasic_path),
        "cache_path": str(cache_path),
        "output_dir": str(selected_output),
        "using_default_signal_only_f0_config": bool(using_defaults),
        "signal_only_f0_config_source": config_source,
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "trace_csv": str(trace_path) if trace_rows and strategy in {"dynamic_fit", "signal_only_f0"} else "",
        "preview_plots": preview_plots,
        "summary": summary,
        "chunks": chunk_rows,
        "trace_rows_written": 0 if dry_run else len(trace_rows),
    }


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"phasic_out: {report['phasic_out']}")
    print(f"output_dir: {report['output_dir']}")
    print(f"dry_run: {_as_bool_text(report['dry_run'])}")
    print(f"roi: {summary['roi']}")
    print(f"requested_correction_strategy: {summary['requested_correction_strategy']}")
    print(f"correction_strategy_selection: {summary['correction_strategy_selection']}")
    print(f"applied_correction_strategy: {summary['applied_correction_strategy']}")
    print(f"applied_trace_source: {summary['applied_trace_source']}")
    print(f"applied_trace_available: {_as_bool_text(summary['applied_trace_available'])}")
    print(f"applied_trace_complete: {_as_bool_text(summary['applied_trace_complete'])}")
    print(f"reason_if_unavailable: {summary['reason_if_unavailable']}")
    print(f"n_chunks: {summary['n_chunks']}")
    print(f"n_chunks_available: {summary['n_chunks_available']}")
    print(f"n_chunks_unavailable: {summary['n_chunks_unavailable']}")
    print(f"applied_trace_warning_level: {summary['applied_trace_warning_level']}")
    print(f"applied_trace_review_required: {_as_bool_text(summary['applied_trace_review_required'])}")
    print("hdf5_modified: false")
    print("feature_detection_input: false")
    if report.get("trace_csv"):
        print(f"trace_csv: {report['trace_csv']}")
    print(f"summary_csv: {report['summary_csv']}")
    print(f"summary_json: {report['summary_json']}")
    for path in report.get("preview_plots", [])[:10]:
        print(f"preview_plot: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview/export an explicitly selected recording-level correction strategy "
            "for one ROI without modifying HDF5 or rerouting feature detection."
        )
    )
    parser.add_argument("--phasic-out", required=True, help="Path to _analysis/phasic_out")
    parser.add_argument("--roi", required=True, help="ROI to export")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--chunks", default=None, help="Optional comma-separated preview chunk IDs")
    parser.add_argument("--max-preview-chunks", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = apply_recording_correction_strategy(
            args.phasic_out,
            roi=args.roi,
            strategy=args.strategy,
            output_dir=args.output_dir,
            chunks=_parse_chunk_ids(args.chunks),
            max_preview_chunks=args.max_preview_chunks,
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
            dpi=int(args.dpi),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
