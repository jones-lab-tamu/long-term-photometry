#!/usr/bin/env python3
"""Export diagnostic-only signal-only F0 dF/F traces for inspection."""

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

from photometry_pipeline.core.signal_only_f0_candidate import (  # noqa: E402
    compute_signal_only_f0_candidate,
)
from photometry_pipeline.io.hdf5_cache_reader import (  # noqa: E402
    CacheReadError,
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_fields,
    open_phasic_cache,
)
from tools.recompute_signal_only_f0_candidates import (  # noqa: E402
    _load_csv_records,
    _load_json_records,
    _load_signal_only_f0_config,
    _normalize_loaded_record,
)

F0_FLOOR_EPS = 1e-9
F0_FLOOR_FRACTION_SEVERE = 0.01
DFF_ABS_P99_CAUTION = 10.0
DFF_ABS_P99_SEVERE = 50.0

FLAG_AVAILABLE = "SIGNAL_ONLY_F0_DFF_AVAILABLE"
FLAG_UNAVAILABLE = "SIGNAL_ONLY_F0_DFF_UNAVAILABLE"
FLAG_F0_FLOOR_APPLIED = "SIGNAL_ONLY_F0_DFF_F0_FLOOR_APPLIED"
FLAG_F0_FLOOR_EXCESSIVE = "SIGNAL_ONLY_F0_DFF_F0_FLOOR_EXCESSIVE"
FLAG_NONFINITE = "SIGNAL_ONLY_F0_DFF_NONFINITE"
FLAG_EXTREME_CAUTION = "SIGNAL_ONLY_F0_DFF_EXTREME_RANGE_CAUTION"
FLAG_EXTREME_SEVERE = "SIGNAL_ONLY_F0_DFF_EXTREME_RANGE_SEVERE"
FLAG_LOW_CONFIDENCE = "SIGNAL_ONLY_F0_DFF_LOW_CONFIDENCE"
FLAG_HIGH_EXTRAPOLATION = "SIGNAL_ONLY_F0_DFF_HIGH_EXTRAPOLATION"
FLAG_LARGE_ANCHOR_GAP = "SIGNAL_ONLY_F0_DFF_LARGE_ANCHOR_GAP"
FLAG_INSUFFICIENT_ANCHORS = "SIGNAL_ONLY_F0_DFF_INSUFFICIENT_ANCHORS"
FLAG_INSUFFICIENT_LOW_SUPPORT = "SIGNAL_ONLY_F0_DFF_INSUFFICIENT_LOW_SUPPORT"

SUMMARY_FIELDS = [
    "roi",
    "chunk_id",
    "available",
    "reason_if_unavailable",
    "signal_n_samples",
    "signal_min",
    "signal_max",
    "signal_median",
    "f0_available",
    "f0_min",
    "f0_max",
    "f0_median",
    "f0_floor_applied",
    "f0_floor_value",
    "dff_available",
    "dff_min",
    "dff_max",
    "dff_median",
    "dff_p01",
    "dff_p99",
    "signal_only_f0_candidate_viability",
    "signal_only_f0_candidate_confidence",
    "signal_only_f0_anchor_count",
    "signal_only_f0_low_support_fraction",
    "signal_only_f0_extrapolated_fraction",
    "signal_only_f0_max_anchor_gap_fraction_observed",
    "signal_only_f0_flags",
    "signal_state_candidate_class",
    "signal_state_warning",
    "diagnostic_warning_level",
    "diagnostic_flags",
    "plot_path",
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
        cid = int(text)
        if cid not in seen:
            out.append(cid)
            seen.add(cid)
    return out


def _find_phasic_cache(phasic_path: Path) -> Path:
    direct = phasic_path / "phasic_trace_cache.h5"
    if direct.exists():
        return direct
    matches = sorted(phasic_path.rglob("phasic_trace_cache.h5"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing phasic_trace_cache.h5 under {phasic_path}")


def _load_qc_records(phasic_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    qc_dir = phasic_path / "qc"
    json_path = qc_dir / "baseline_reference_candidate_by_chunk.json"
    csv_path = qc_dir / "baseline_reference_candidate_by_chunk.csv"
    if json_path.exists():
        records = _load_json_records(json_path)
    elif csv_path.exists():
        records, _columns = _load_csv_records(csv_path)
    else:
        records = []
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in records:
        rec = _normalize_loaded_record(dict(raw))
        roi = str(rec.get("roi", "")).strip()
        try:
            chunk_id = int(float(rec.get("chunk_id")))
        except Exception:
            continue
        if roi:
            out[(roi, chunk_id)] = rec
    return out


def _safe_roi_name(roi: str) -> str:
    return str(roi).replace(os.sep, "_").replace("/", "_").replace("\\", "_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _as_flag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [x.strip() for x in text.split(";") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _finite_stats(arr: np.ndarray, prefix: str) -> dict[str, Any]:
    vals = np.asarray(arr, dtype=float).reshape(-1)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {f"{prefix}_min": None, f"{prefix}_max": None, f"{prefix}_median": None}
    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_median": float(np.median(finite)),
    }


def compute_signal_only_f0_dff_diagnostic(
    signal: np.ndarray,
    f0: np.ndarray,
    *,
    qc_record: dict[str, Any] | None = None,
    f0_floor_value: float = F0_FLOOR_EPS,
) -> dict[str, Any]:
    """Compute diagnostic-only signal-only F0 dF/F and guardrail metadata."""
    sig = np.asarray(signal, dtype=float).reshape(-1)
    base = np.asarray(f0, dtype=float).reshape(-1)
    record = qc_record or {}
    flags: list[str] = []
    if sig.shape != base.shape:
        return {
            "available": False,
            "reason_if_unavailable": "signal_f0_shape_mismatch",
            "dff": None,
            "diagnostic_warning_level": "severe",
            "diagnostic_flags": [FLAG_UNAVAILABLE],
            "f0_floor_applied": False,
            "f0_floor_fraction": 0.0,
            "f0_floor_value": float(f0_floor_value),
        }

    valid_signal = np.isfinite(sig)
    valid_f0 = np.isfinite(base)
    finite_pair = valid_signal & valid_f0
    if sig.size == 0 or int(np.sum(finite_pair)) == 0:
        return {
            "available": False,
            "reason_if_unavailable": "no_finite_signal_f0_pairs",
            "dff": None,
            "diagnostic_warning_level": "severe",
            "diagnostic_flags": [FLAG_UNAVAILABLE, FLAG_NONFINITE],
            "f0_floor_applied": False,
            "f0_floor_fraction": 0.0,
            "f0_floor_value": float(f0_floor_value),
        }

    floor = float(f0_floor_value)
    floor_mask = finite_pair & (base <= floor)
    floor_fraction = float(np.sum(floor_mask) / max(1, np.sum(finite_pair)))
    denom = base.copy()
    denom[floor_mask] = floor
    dff = np.full(sig.shape, np.nan, dtype=float)
    ok = valid_signal & np.isfinite(denom) & (denom > 0)
    dff[ok] = (sig[ok] - denom[ok]) / denom[ok]

    finite_dff = dff[np.isfinite(dff)]
    if floor_fraction > 0:
        flags.append(FLAG_F0_FLOOR_APPLIED)
    if floor_fraction > F0_FLOOR_FRACTION_SEVERE:
        flags.append(FLAG_F0_FLOOR_EXCESSIVE)
    if finite_dff.size == 0 or np.any(~np.isfinite(dff[ok])):
        flags.append(FLAG_NONFINITE)
    abs_p99 = float(np.percentile(np.abs(finite_dff), 99.0)) if finite_dff.size else math.nan
    if np.isfinite(abs_p99) and abs_p99 >= DFF_ABS_P99_SEVERE:
        flags.append(FLAG_EXTREME_SEVERE)
    elif np.isfinite(abs_p99) and abs_p99 >= DFF_ABS_P99_CAUTION:
        flags.append(FLAG_EXTREME_CAUTION)

    confidence = str(record.get("signal_only_f0_candidate_confidence", "")).strip().lower()
    if confidence == "low":
        flags.append(FLAG_LOW_CONFIDENCE)
    src_flags = set(_as_flag_list(record.get("signal_only_f0_flags")))
    if "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION" in src_flags:
        flags.append(FLAG_HIGH_EXTRAPOLATION)
    if "SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP" in src_flags or "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_LARGE_GAP" in src_flags:
        flags.append(FLAG_LARGE_ANCHOR_GAP)
    if "SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS" in src_flags:
        flags.append(FLAG_INSUFFICIENT_ANCHORS)
    if "SIGNAL_ONLY_F0_INSUFFICIENT_LOW_SUPPORT" in src_flags:
        flags.append(FLAG_INSUFFICIENT_LOW_SUPPORT)

    if finite_dff.size == 0:
        warning = "severe"
        flags.extend([FLAG_UNAVAILABLE, FLAG_NONFINITE])
        available = False
        reason = "no_finite_dff_after_flooring"
    else:
        available = True
        reason = ""
        if FLAG_EXTREME_SEVERE in flags or FLAG_F0_FLOOR_EXCESSIVE in flags or FLAG_NONFINITE in flags:
            warning = "severe"
        elif flags:
            warning = "caution"
        else:
            warning = "none"
        flags.append(FLAG_AVAILABLE)

    flags = list(dict.fromkeys(flags))
    out = {
        "available": bool(available),
        "reason_if_unavailable": reason,
        "dff": dff if available else None,
        "diagnostic_warning_level": warning,
        "diagnostic_flags": flags,
        "f0_floor_applied": bool(floor_fraction > 0),
        "f0_floor_fraction": floor_fraction,
        "f0_floor_value": floor,
    }
    if finite_dff.size:
        out.update(
            {
                "dff_min": float(np.min(finite_dff)),
                "dff_max": float(np.max(finite_dff)),
                "dff_median": float(np.median(finite_dff)),
                "dff_p01": float(np.percentile(finite_dff, 1.0)),
                "dff_p99": float(np.percentile(finite_dff, 99.0)),
            }
        )
    else:
        out.update({"dff_min": None, "dff_max": None, "dff_median": None, "dff_p01": None, "dff_p99": None})
    return out


def _chunk_has_dataset(cache, roi: str, chunk_id: int, name: str) -> bool:
    grp = cache.get(f"roi/{roi}/chunk_{int(chunk_id)}")
    return grp is not None and name in grp


def _load_or_compute_f0(cache, roi: str, chunk_id: int, signal: np.ndarray, time_sec: np.ndarray, record: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any], str]:
    for field in ("signal_only_f0_candidate", "signal_only_f0"):
        if _chunk_has_dataset(cache, roi, chunk_id, field):
            (candidate,) = load_cache_chunk_fields(cache, roi, chunk_id, [field])
            return np.asarray(candidate, dtype=float).reshape(-1), dict(record), f"hdf5:{field}"
    diagnostics = compute_signal_only_f0_candidate(
        signal=np.asarray(signal, dtype=float),
        time=np.asarray(time_sec, dtype=float),
        signal_state=record,
        config=config,
    )
    candidate = diagnostics.get("signal_only_f0_candidate")
    if candidate is None:
        return None, {**record, **{k: v for k, v in diagnostics.items() if k != "signal_only_f0_candidate"}}, "recomputed_unavailable"
    merged = {**record, **{k: v for k, v in diagnostics.items() if k != "signal_only_f0_candidate"}}
    return np.asarray(candidate, dtype=float).reshape(-1), merged, "recomputed_in_memory"


def _base_row(roi: str, chunk_id: int) -> dict[str, Any]:
    row = {key: None for key in SUMMARY_FIELDS}
    row.update(
        {
            "roi": str(roi),
            "chunk_id": int(chunk_id),
            "available": False,
            "reason_if_unavailable": "",
            "f0_available": False,
            "f0_floor_applied": False,
            "f0_floor_value": F0_FLOOR_EPS,
            "dff_available": False,
            "diagnostic_warning_level": "severe",
            "diagnostic_flags": FLAG_UNAVAILABLE,
            "plot_path": "",
        }
    )
    return row


def _plot_diagnostic(out_path: Path, *, roi: str, chunk_id: int, time_sec: np.ndarray, signal: np.ndarray, f0: np.ndarray, dff: np.ndarray, row: dict[str, Any], dpi: int) -> None:
    t = np.asarray(time_sec, dtype=float).reshape(-1)
    sig = np.asarray(signal, dtype=float).reshape(-1)
    base = np.asarray(f0, dtype=float).reshape(-1)
    trace = np.asarray(dff, dtype=float).reshape(-1)
    if t.shape != sig.shape or not np.any(np.isfinite(t)):
        t_plot = np.arange(sig.size, dtype=float)
        xlabel = "Sample"
    else:
        first = t[np.isfinite(t)][0]
        t_plot = t - float(first)
        xlabel = "Time within chunk (s)"

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False, gridspec_kw={"height_ratios": [2, 2, 1]})
    axes[0].plot(t_plot, sig, color="forestgreen", linewidth=0.8, label="sig_raw")
    axes[0].plot(t_plot, base, color="black", linestyle="--", linewidth=0.9, label="signal_only_f0")
    axes[0].set_ylabel("Signal / F0")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_plot, trace, color="darkorange", linewidth=0.8, label="signal_only_f0_dff diagnostic")
    axes[1].axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    axes[1].set_ylabel("Diagnostic dF/F")
    axes[1].set_xlabel(xlabel)
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    axes[2].axis("off")
    text = (
        "DIAGNOSTIC ONLY - not applied correction, not event/feature input\n"
        f"ROI={roi} chunk={int(chunk_id)} | viability={row.get('signal_only_f0_candidate_viability') or ''} | "
        f"confidence={row.get('signal_only_f0_candidate_confidence') or ''} | "
        f"anchors={row.get('signal_only_f0_anchor_count') or ''} | "
        f"low_support={row.get('signal_only_f0_low_support_fraction') or ''} | "
        f"extrapolated={row.get('signal_only_f0_extrapolated_fraction') or ''} | "
        f"max_gap={row.get('signal_only_f0_max_anchor_gap_fraction_observed') or ''} | "
        f"warning={row.get('diagnostic_warning_level') or ''}\n"
        f"flags={row.get('diagnostic_flags') or ''}"
    )
    axes[2].text(0.01, 0.95, text, va="top", ha="left", fontsize=8, wrap=True)
    fig.suptitle(f"Signal-only F0 dF/F diagnostic | {roi} chunk {int(chunk_id)}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> tuple[Path, Path]:
    csv_path = output_dir / "signal_only_f0_dff_diagnostic_summary.csv"
    json_path = output_dir / "signal_only_f0_dff_diagnostic_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if isinstance(value, (list, tuple)):
                    out[key] = ";".join(str(x) for x in value)
                elif isinstance(value, bool):
                    out[key] = str(value).lower()
                elif isinstance(value, float):
                    out[key] = value if math.isfinite(value) else ""
            writer.writerow({key: out.get(key, "") for key in SUMMARY_FIELDS})
    payload = {**metadata, "records": rows}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return csv_path, json_path


def _select_output_dir(base_dir: Path, *, overwrite: bool, dry_run: bool) -> Path:
    if dry_run or overwrite or not base_dir.exists() or not any(base_dir.iterdir()):
        return base_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir.with_name(f"{base_dir.name}_{stamp}")


def export_signal_only_f0_dff_diagnostics(
    phasic_out: str | os.PathLike[str],
    *,
    roi: str | None = None,
    chunks: list[int] | None = None,
    max_chunks: int | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    dpi: int = 150,
) -> dict[str, Any]:
    phasic_path = Path(phasic_out).resolve()
    cache_path = _find_phasic_cache(phasic_path)
    requested_output = Path(output_dir).resolve() if output_dir is not None else phasic_path / "qc" / "signal_only_f0_dff_diagnostics"
    selected_output = _select_output_dir(requested_output, overwrite=overwrite, dry_run=dry_run)
    records = _load_qc_records(phasic_path)
    config, using_defaults, config_source = _load_signal_only_f0_config(phasic_path)

    rows: list[dict[str, Any]] = []
    plots_written: list[str] = []
    rois_seen: set[str] = set()
    plot_budget = None if max_chunks is None else max(0, int(max_chunks))

    with open_phasic_cache(str(cache_path)) as cache:
        rois = [str(roi)] if roi is not None else list_cache_rois(cache)
        all_chunk_ids = list_cache_chunk_ids(cache)
        requested_chunks = [int(x) for x in chunks] if chunks is not None else all_chunk_ids
        for roi_name in rois:
            rois_seen.add(str(roi_name))
            for chunk_id in requested_chunks:
                row = _base_row(str(roi_name), int(chunk_id))
                record = records.get((str(roi_name), int(chunk_id)), {})
                try:
                    time_sec, sig_raw = load_cache_chunk_fields(
                        cache, str(roi_name), int(chunk_id), ["time_sec", "sig_raw"]
                    )
                    signal = np.asarray(sig_raw, dtype=float).reshape(-1)
                    time_arr = np.asarray(time_sec, dtype=float).reshape(-1)
                    row["signal_n_samples"] = int(signal.size)
                    row.update(_finite_stats(signal, "signal"))
                    f0, diag_record, f0_source = _load_or_compute_f0(
                        cache,
                        str(roi_name),
                        int(chunk_id),
                        signal,
                        time_arr,
                        record,
                        config,
                    )
                    row["signal_only_f0_candidate_viability"] = diag_record.get("signal_only_f0_candidate_viability")
                    row["signal_only_f0_candidate_confidence"] = diag_record.get("signal_only_f0_candidate_confidence")
                    row["signal_only_f0_anchor_count"] = diag_record.get("signal_only_f0_anchor_count")
                    row["signal_only_f0_low_support_fraction"] = diag_record.get("signal_only_f0_low_support_fraction")
                    row["signal_only_f0_extrapolated_fraction"] = diag_record.get("signal_only_f0_extrapolated_fraction")
                    row["signal_only_f0_max_anchor_gap_fraction_observed"] = diag_record.get("signal_only_f0_max_anchor_gap_fraction_observed")
                    row["signal_only_f0_flags"] = ";".join(_as_flag_list(diag_record.get("signal_only_f0_flags")))
                    row["signal_state_candidate_class"] = diag_record.get("signal_state_candidate_class")
                    row["signal_state_warning"] = diag_record.get("signal_state_warning")
                    if f0 is None or f0.shape != signal.shape:
                        row["reason_if_unavailable"] = f0_source
                        row["diagnostic_flags"] = FLAG_UNAVAILABLE
                        rows.append(row)
                        continue
                    row["f0_available"] = True
                    row.update(_finite_stats(f0, "f0"))
                    dff_diag = compute_signal_only_f0_dff_diagnostic(signal, f0, qc_record=diag_record)
                    row["available"] = bool(dff_diag["available"])
                    row["dff_available"] = bool(dff_diag["available"])
                    row["reason_if_unavailable"] = dff_diag.get("reason_if_unavailable", "")
                    row["f0_floor_applied"] = bool(dff_diag.get("f0_floor_applied", False))
                    row["f0_floor_value"] = dff_diag.get("f0_floor_value", F0_FLOOR_EPS)
                    row["diagnostic_warning_level"] = dff_diag.get("diagnostic_warning_level", "severe")
                    row["diagnostic_flags"] = ";".join(_as_flag_list(dff_diag.get("diagnostic_flags")))
                    for key in ("dff_min", "dff_max", "dff_median", "dff_p01", "dff_p99"):
                        row[key] = dff_diag.get(key)
                    dff = dff_diag.get("dff")
                    if row["available"] and dff is not None and (plot_budget is None or len(plots_written) < plot_budget):
                        out_path = selected_output / f"{_safe_roi_name(roi_name)}_chunk_{int(chunk_id)}_signal_only_f0_dff_diagnostic.png"
                        row["plot_path"] = str(out_path)
                        if not dry_run:
                            _plot_diagnostic(
                                out_path,
                                roi=str(roi_name),
                                chunk_id=int(chunk_id),
                                time_sec=time_arr,
                                signal=signal,
                                f0=f0,
                                dff=np.asarray(dff, dtype=float),
                                row=row,
                                dpi=int(dpi),
                            )
                        plots_written.append(str(out_path))
                except CacheReadError as exc:
                    row["reason_if_unavailable"] = str(exc)
                    row["diagnostic_flags"] = FLAG_UNAVAILABLE
                except Exception as exc:
                    row["reason_if_unavailable"] = f"unexpected_error: {exc}"
                    row["diagnostic_flags"] = FLAG_UNAVAILABLE
                rows.append(row)

    warning_counts = Counter(str(row.get("diagnostic_warning_level") or "") for row in rows)
    flag_counts: Counter[str] = Counter()
    for row in rows:
        flag_counts.update(_as_flag_list(row.get("diagnostic_flags")))
    metadata = {
        "diagnostic_only": True,
        "applied_correction": False,
        "feature_detection_input": False,
        "modifies_hdf5": False,
        "phasic_out": str(phasic_path),
        "cache_path": str(cache_path),
        "output_dir": str(selected_output),
        "using_default_signal_only_f0_config": bool(using_defaults),
        "signal_only_f0_config_source": config_source,
        "chunks_evaluated": int(len(rows)),
        "chunks_exported": int(len(plots_written)),
        "unavailable_chunks": int(sum(1 for row in rows if not bool(row.get("available")))),
        "warning_level_counts": dict(sorted(warning_counts.items())),
        "top_diagnostic_flags": dict(flag_counts.most_common(20)),
        "note": (
            "signal_only_f0_dff diagnostics are not applied correction outputs. They do not change "
            "dynamic-fit dF/F, applied_dff, event detection, or feature detection."
        ),
    }
    csv_path = selected_output / "signal_only_f0_dff_diagnostic_summary.csv"
    json_path = selected_output / "signal_only_f0_dff_diagnostic_summary.json"
    if not dry_run:
        _write_outputs(selected_output, rows, metadata)
    return {
        **metadata,
        "dry_run": bool(dry_run),
        "rois_processed": sorted(rois_seen),
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
        "plots_written": plots_written,
        "rows": rows,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"phasic_out: {report['phasic_out']}")
    print(f"output_dir: {report['output_dir']}")
    print(f"dry_run: {str(report['dry_run']).lower()}")
    print(f"rois processed: {report['rois_processed']}")
    print(f"chunks evaluated: {report['chunks_evaluated']}")
    print(f"chunks exported: {report['chunks_exported']}")
    print(f"unavailable chunks: {report['unavailable_chunks']}")
    print(f"warning level counts: {report['warning_level_counts']}")
    print(f"top diagnostic flags: {report['top_diagnostic_flags']}")
    print("applied_correction: false")
    print("feature_detection_input: false")
    print("modifies_hdf5: false")
    if report.get("summary_csv"):
        print(f"summary_csv: {report['summary_csv']}")
    if report.get("summary_json"):
        print(f"summary_json: {report['summary_json']}")
    for path in report.get("plots_written", [])[:10]:
        print(f"plot: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export diagnostic-only signal-only F0 dF/F summaries and plots."
    )
    parser.add_argument("--phasic-out", required=True, help="Path to _analysis/phasic_out")
    parser.add_argument("--roi", default=None, help="Optional ROI filter")
    parser.add_argument("--chunks", default=None, help="Optional comma-separated chunk IDs")
    parser.add_argument("--max-chunks", type=int, default=None, help="Maximum number of chunk PNGs to export")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = export_signal_only_f0_dff_diagnostics(
            args.phasic_out,
            roi=args.roi,
            chunks=_parse_chunk_ids(args.chunks),
            max_chunks=args.max_chunks,
            output_dir=args.output_dir,
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
