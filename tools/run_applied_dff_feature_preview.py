#!/usr/bin/env python3
"""Run feature/event preview from an explicit applied_dff export."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
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

from photometry_pipeline.config import Config  # noqa: E402
from photometry_pipeline.core.feature_extraction import (  # noqa: E402
    get_peak_indices_for_trace,
)

FLAG_PARTIAL_INPUT = "FEATURE_PREVIEW_PARTIAL_APPLIED_TRACE_INPUT"

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
    "applied_trace_review_required",
    "applied_trace_warning_level",
    "applied_trace_flags",
    "n_chunks",
    "n_samples",
    "n_events",
    "n_chunks_with_events",
    "feature_detection_input_trace",
    "feature_detection_input_strategy",
    "feature_detection_input_source",
    "feature_detection_input_units",
    "feature_detection_input_applied_trace_complete",
    "feature_detection_input_review_required",
    "feature_detection_input_warning_level",
    "feature_detection_input_warning_flags",
    "feature_detection_preview",
    "feature_preview_review_required",
    "feature_preview_warning_level",
    "feature_preview_warning_flags",
    "hdf5_modified",
    "replaces_existing_feature_outputs",
    "output_dir",
    "events_csv",
    "preview_plot",
]

EVENT_FIELDS = [
    "roi",
    "chunk_id",
    "event_id",
    "event_start_sample",
    "event_peak_sample",
    "event_end_sample",
    "event_start_time_sec",
    "event_peak_time_sec",
    "event_end_time_sec",
    "event_peak_value",
    "event_amplitude",
    "event_auc",
    "event_duration_sec",
    "detection_input_trace",
    "detection_input_strategy",
    "detection_input_source",
    "detection_preview",
    "event_boundary_mode",
    "event_metrics_available",
]


class AppliedFeaturePreviewError(RuntimeError):
    """Raised for invalid applied-preview inputs."""


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off", ""}:
        return False
    return bool(value)


def _as_flag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(";") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _warning_at_least(level: str, minimum: str) -> str:
    order = {"none": 0, "info": 1, "caution": 2, "severe": 3}
    current = str(level or "none").strip().lower()
    required = str(minimum or "none").strip().lower()
    return current if order.get(current, 0) >= order.get(required, 0) else required


def _select_output_dir(base_dir: Path, *, overwrite: bool, dry_run: bool) -> Path:
    if dry_run or overwrite or not base_dir.exists() or not any(base_dir.iterdir()):
        return base_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir.with_name(f"{base_dir.name}_{stamp}")


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AppliedFeaturePreviewError(f"applied correction summary not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if len(payload) != 1:
                raise AppliedFeaturePreviewError("applied summary must contain exactly one ROI/strategy")
            return dict(payload[0])
        if "records" in payload and isinstance(payload["records"], list):
            if len(payload["records"]) != 1:
                raise AppliedFeaturePreviewError("applied summary must contain exactly one ROI/strategy")
            merged = dict(payload)
            merged.update(dict(payload["records"][0]))
            return merged
        return dict(payload)

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise AppliedFeaturePreviewError("applied summary CSV must contain exactly one ROI/strategy row")
    return dict(rows[0])


def _resolve_summary_path(applied_preview_dir: Path, summary_csv: str | os.PathLike[str] | None) -> Path:
    if summary_csv is not None:
        return Path(summary_csv).resolve()
    csv_path = applied_preview_dir / "applied_correction_summary.csv"
    json_path = applied_preview_dir / "applied_correction_summary.json"
    if csv_path.exists():
        return csv_path.resolve()
    if json_path.exists():
        return json_path.resolve()
    raise AppliedFeaturePreviewError(
        f"missing applied_correction_summary.csv/json under {applied_preview_dir}"
    )


def _resolve_trace_path(
    applied_preview_dir: Path,
    summary: dict[str, Any],
    trace_csv: str | os.PathLike[str] | None,
) -> Path:
    if trace_csv is not None:
        path = Path(trace_csv)
        return path.resolve() if path.is_absolute() else (applied_preview_dir / path).resolve()

    summary_trace = str(summary.get("trace_csv") or "").strip()
    if summary_trace:
        path = Path(summary_trace)
        return path.resolve() if path.is_absolute() else (applied_preview_dir / path).resolve()

    matches = sorted(applied_preview_dir.glob("*_applied_trace.csv"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise AppliedFeaturePreviewError(
            "multiple *_applied_trace.csv files found; pass --trace-csv explicitly"
        )
    raise AppliedFeaturePreviewError("missing applied_dff trace CSV")


def _validate_applied_summary(
    summary: dict[str, Any],
    *,
    allow_partial_applied_trace: bool,
) -> None:
    strategy = str(summary.get("applied_correction_strategy") or "").strip()
    source = str(summary.get("applied_trace_source") or "").strip()
    units = str(summary.get("applied_trace_units") or "").strip()
    if strategy == "no_correction" or source == "none" or units == "none":
        raise AppliedFeaturePreviewError(
            "no_correction has no corrected applied_dff for feature/event detection preview"
        )
    if not _as_bool(summary.get("applied_trace_available")):
        raise AppliedFeaturePreviewError("applied trace is not available")
    if source == "":
        raise AppliedFeaturePreviewError("applied_trace_source is missing")
    if units != "dff":
        raise AppliedFeaturePreviewError("applied_trace_units must be dff")
    if _as_bool(summary.get("feature_detection_input")):
        raise AppliedFeaturePreviewError(
            "applied summary is already marked as feature_detection_input; refusing ambiguous input"
        )
    if not _as_bool(summary.get("applied_trace_complete")) and not allow_partial_applied_trace:
        raise AppliedFeaturePreviewError(
            "applied_trace_complete is false; rerun with --allow-partial-applied-trace to preview incomplete input"
        )


def _load_trace(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise AppliedFeaturePreviewError(f"applied trace CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"roi", "chunk_id", "sample_index", "time_sec", "applied_dff"}
    missing = required - set(df.columns)
    if missing:
        raise AppliedFeaturePreviewError(f"applied trace CSV missing required columns: {sorted(missing)}")
    rois = sorted(str(x) for x in df["roi"].dropna().unique())
    if len(rois) != 1:
        raise AppliedFeaturePreviewError("applied trace CSV must contain exactly one ROI")
    return df


def _infer_fs_hz(time_sec: np.ndarray) -> float:
    t = np.asarray(time_sec, dtype=float).reshape(-1)
    finite = t[np.isfinite(t)]
    if finite.size >= 2:
        diffs = np.diff(finite)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(1.0 / np.median(diffs))
    return float(Config().target_fs_hz)


def _detect_events(df: pd.DataFrame, summary: dict[str, Any], config: Config) -> list[dict[str, Any]]:
    """Detect peaks from applied_dff without inventing event segmentation bounds."""
    events: list[dict[str, Any]] = []
    event_id = 0
    strategy = str(summary.get("applied_correction_strategy") or "")
    source = str(summary.get("applied_trace_source") or "")
    for chunk_id, chunk_df in df.groupby("chunk_id", sort=True):
        chunk_sorted = chunk_df.sort_values("sample_index")
        t = chunk_sorted["time_sec"].to_numpy(dtype=float)
        y = chunk_sorted["applied_dff"].to_numpy(dtype=float)
        fs_hz = _infer_fs_hz(t)
        peaks = get_peak_indices_for_trace(y, fs_hz, config)
        for peak_idx in peaks.astype(int).tolist():
            peak_t = float(t[peak_idx]) if peak_idx < t.size and np.isfinite(t[peak_idx]) else math.nan
            event_id += 1
            events.append(
                {
                    "roi": str(chunk_sorted["roi"].iloc[0]),
                    "chunk_id": int(chunk_id),
                    "event_id": int(event_id),
                    "event_start_sample": "",
                    "event_peak_sample": int(peak_idx),
                    "event_end_sample": "",
                    "event_start_time_sec": "",
                    "event_peak_time_sec": peak_t,
                    "event_end_time_sec": "",
                    "event_peak_value": float(y[peak_idx]) if np.isfinite(y[peak_idx]) else math.nan,
                    "event_amplitude": float(y[peak_idx]) if np.isfinite(y[peak_idx]) else math.nan,
                    "event_auc": "",
                    "event_duration_sec": "",
                    "detection_input_trace": "applied_dff",
                    "detection_input_strategy": strategy,
                    "detection_input_source": source,
                    "detection_preview": True,
                    "event_boundary_mode": "peak_only_no_event_segmentation",
                    "event_metrics_available": False,
                }
            )
    return events


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, bool):
                    value = str(value).lower()
                elif isinstance(value, float) and not math.isfinite(value):
                    value = ""
                out[key] = value
            writer.writerow(out)


def _write_summary(output_dir: Path, summary: dict[str, Any], events: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = output_dir / "applied_dff_feature_preview_summary.csv"
    json_path = output_dir / "applied_dff_feature_preview_summary.json"
    _write_csv(csv_path, [summary], SUMMARY_FIELDS)
    payload = {**summary, "events": events[:1000]}
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return csv_path, json_path


def _plot_preview(path: Path, trace_df: pd.DataFrame, events: list[dict[str, Any]], max_events: int | None) -> None:
    chunks = sorted(trace_df["chunk_id"].dropna().unique().tolist())
    if not chunks:
        return
    chunk_id = chunks[0]
    chunk_df = trace_df[trace_df["chunk_id"] == chunk_id].sort_values("sample_index")
    t = chunk_df["time_sec"].to_numpy(dtype=float)
    y = chunk_df["applied_dff"].to_numpy(dtype=float)
    x = t - float(t[np.isfinite(t)][0]) if np.any(np.isfinite(t)) else np.arange(y.size)
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.plot(x, y, color="black", linewidth=0.8, label="applied_dff")
    ax.axhline(0.0, color="black", alpha=0.4, linewidth=0.5)
    chunk_events = [e for e in events if int(e["chunk_id"]) == int(chunk_id)]
    if max_events is not None:
        chunk_events = chunk_events[: max(0, int(max_events))]
    for event in chunk_events:
        peak = int(event["event_peak_sample"])
        if 0 <= peak < x.size:
            ax.plot(x[peak], y[peak], "o", color="darkorange", markersize=4)
    ax.set_title(f"Applied dF/F feature preview | chunk {int(chunk_id)}")
    ax.set_xlabel("Time within chunk (s)")
    ax.set_ylabel("Applied dF/F")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_applied_dff_feature_preview(
    applied_preview_dir: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    summary_csv: str | os.PathLike[str] | None = None,
    trace_csv: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    max_preview_events: int | None = None,
    plot_preview: bool = True,
    allow_partial_applied_trace: bool = False,
) -> dict[str, Any]:
    applied_dir = Path(applied_preview_dir).resolve()
    summary_path = _resolve_summary_path(applied_dir, summary_csv)
    applied_summary = _read_summary(summary_path)
    _validate_applied_summary(
        applied_summary,
        allow_partial_applied_trace=bool(allow_partial_applied_trace),
    )
    trace_path = _resolve_trace_path(applied_dir, applied_summary, trace_csv)
    trace_df = _load_trace(trace_path)

    output_base = Path(output_dir).resolve() if output_dir is not None else applied_dir / "feature_event_preview"
    selected_output = _select_output_dir(output_base, overwrite=overwrite, dry_run=dry_run)
    events_csv = selected_output / "applied_dff_feature_events.csv"
    summary_out_csv = selected_output / "applied_dff_feature_preview_summary.csv"
    summary_out_json = selected_output / "applied_dff_feature_preview_summary.json"
    plot_path = selected_output / "applied_dff_feature_preview_plot.png"

    cfg = Config(event_signal="dff")
    events = _detect_events(trace_df, applied_summary, cfg)
    flags = _as_flag_list(applied_summary.get("applied_trace_flags"))
    feature_warning = str(applied_summary.get("applied_trace_warning_level") or "none")
    feature_review = _as_bool(applied_summary.get("applied_trace_review_required"))
    if not _as_bool(applied_summary.get("applied_trace_complete")) and allow_partial_applied_trace:
        flags.append(FLAG_PARTIAL_INPUT)
        feature_review = True
        feature_warning = _warning_at_least(feature_warning, "caution")
    flags = list(dict.fromkeys(flags))
    roi_values = sorted(str(x) for x in trace_df["roi"].dropna().unique())
    chunk_values = sorted(trace_df["chunk_id"].dropna().unique().tolist())
    chunks_with_events = sorted({int(event["chunk_id"]) for event in events})

    summary = {
        "roi": roi_values[0] if roi_values else str(applied_summary.get("roi") or ""),
        "recording_key": applied_summary.get("recording_key", ""),
        "requested_correction_strategy": applied_summary.get("requested_correction_strategy", ""),
        "correction_strategy_selection": applied_summary.get("correction_strategy_selection", ""),
        "applied_correction_strategy": applied_summary.get("applied_correction_strategy", ""),
        "applied_trace_source": applied_summary.get("applied_trace_source", ""),
        "applied_trace_units": applied_summary.get("applied_trace_units", ""),
        "applied_trace_available": _as_bool(applied_summary.get("applied_trace_available")),
        "applied_trace_complete": _as_bool(applied_summary.get("applied_trace_complete")),
        "applied_trace_review_required": _as_bool(applied_summary.get("applied_trace_review_required")),
        "applied_trace_warning_level": applied_summary.get("applied_trace_warning_level", ""),
        "applied_trace_flags": ";".join(_as_flag_list(applied_summary.get("applied_trace_flags"))),
        "n_chunks": int(len(chunk_values)),
        "n_samples": int(len(trace_df)),
        "n_events": int(len(events)),
        "n_chunks_with_events": int(len(chunks_with_events)),
        "feature_detection_input_trace": "applied_dff",
        "feature_detection_input_strategy": applied_summary.get("applied_correction_strategy", ""),
        "feature_detection_input_source": applied_summary.get("applied_trace_source", ""),
        "feature_detection_input_units": "dff",
        "feature_detection_input_applied_trace_complete": _as_bool(applied_summary.get("applied_trace_complete")),
        "feature_detection_input_review_required": feature_review,
        "feature_detection_input_warning_level": feature_warning,
        "feature_detection_input_warning_flags": ";".join(flags),
        "feature_detection_preview": True,
        "feature_preview_review_required": feature_review,
        "feature_preview_warning_level": feature_warning,
        "feature_preview_warning_flags": ";".join(flags),
        "hdf5_modified": False,
        "replaces_existing_feature_outputs": False,
        "output_dir": str(selected_output),
        "events_csv": str(events_csv),
        "preview_plot": str(plot_path) if plot_preview else "",
    }

    if not dry_run:
        selected_output.mkdir(parents=True, exist_ok=True)
        _write_csv(events_csv, events, EVENT_FIELDS)
        _write_summary(selected_output, summary, events)
        if plot_preview:
            _plot_preview(plot_path, trace_df, events, max_preview_events)

    return {
        "dry_run": bool(dry_run),
        "applied_preview_dir": str(applied_dir),
        "summary_input": str(summary_path),
        "trace_input": str(trace_path),
        "output_dir": str(selected_output),
        "summary_csv": str(summary_out_csv),
        "summary_json": str(summary_out_json),
        "events_csv": str(events_csv),
        "preview_plot": str(plot_path) if plot_preview else "",
        "summary": summary,
        "events": events,
    }


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"applied_preview_dir: {report['applied_preview_dir']}")
    print(f"trace_input: {report['trace_input']}")
    print(f"output_dir: {report['output_dir']}")
    print(f"dry_run: {str(report['dry_run']).lower()}")
    print(f"roi: {summary['roi']}")
    print(f"feature_detection_input_trace: {summary['feature_detection_input_trace']}")
    print(f"feature_detection_input_strategy: {summary['feature_detection_input_strategy']}")
    print(f"feature_detection_input_source: {summary['feature_detection_input_source']}")
    print(f"feature_detection_preview: true")
    print(f"n_chunks: {summary['n_chunks']}")
    print(f"n_samples: {summary['n_samples']}")
    print(f"n_events: {summary['n_events']}")
    print(f"hdf5_modified: false")
    print(f"replaces_existing_feature_outputs: false")
    print(f"summary_csv: {report['summary_csv']}")
    print(f"summary_json: {report['summary_json']}")
    print(f"events_csv: {report['events_csv']}")
    if report.get("preview_plot"):
        print(f"preview_plot: {report['preview_plot']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run feature/event preview from an explicit applied_dff export."
    )
    parser.add_argument("--applied-preview-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--trace-csv", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-preview-events", type=int, default=None)
    parser.add_argument("--plot-preview", dest="plot_preview", action="store_true", default=True)
    parser.add_argument("--no-plot-preview", dest="plot_preview", action="store_false")
    parser.add_argument("--allow-partial-applied-trace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = run_applied_dff_feature_preview(
            args.applied_preview_dir,
            output_dir=args.output_dir,
            summary_csv=args.summary_csv,
            trace_csv=args.trace_csv,
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
            max_preview_events=args.max_preview_events,
            plot_preview=bool(args.plot_preview),
            allow_partial_applied_trace=bool(args.allow_partial_applied_trace),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
