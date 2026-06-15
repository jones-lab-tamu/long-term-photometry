#!/usr/bin/env python3
"""Verify the explicit applied_dff preview stack end to end."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.apply_recording_correction_strategy import (  # noqa: E402
    apply_recording_correction_strategy,
)
from tools.run_applied_dff_feature_preview import (  # noqa: E402
    AppliedFeaturePreviewError,
    run_applied_dff_feature_preview,
)

STRATEGIES = ("dynamic_fit", "signal_only_f0", "no_correction")
VERIFICATION_MODE = "applied_dff_preview_stack"
EXPECTED_NO_CORRECTION_REFUSAL = (
    "no_correction has no corrected applied_dff for feature/event detection preview"
)

SUMMARY_FIELDS = [
    "verification_passed",
    "verification_mode",
    "roi",
    "requested_strategy",
    "applied_correction_strategy",
    "applied_trace_source",
    "applied_trace_units",
    "applied_trace_available",
    "applied_trace_complete",
    "applied_trace_warning_level",
    "applied_trace_review_required",
    "n_chunks",
    "n_chunks_available",
    "n_chunks_unavailable",
    "feature_preview_expected",
    "feature_preview_ran",
    "feature_preview_refused_expectedly",
    "feature_preview_refusal_message",
    "feature_detection_input_trace",
    "feature_detection_input_strategy",
    "feature_detection_input_source",
    "feature_detection_preview",
    "peak_detector_source_function",
    "peak_detector_mode",
    "peak_detection_config_source",
    "peak_detection_config_path",
    "peak_detection_config_hash",
    "n_events",
    "n_event_rows_with_wrong_detection_input",
    "n_event_rows_with_wrong_boundary_mode",
    "n_event_rows_with_metrics_populated",
    "hdf5_modified",
    "replaces_existing_feature_outputs",
    "applied_preview_dir",
    "peak_preview_dir",
    "verification_output_dir",
    "failure_messages",
]


class PreviewStackVerificationError(RuntimeError):
    """Raised when verification fails."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report or {}


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _select_output_dir(base_dir: Path, *, overwrite: bool, dry_run: bool) -> Path:
    if dry_run or overwrite or not base_dir.exists() or not any(base_dir.iterdir()):
        return base_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir.with_name(f"{base_dir.name}_{stamp}")


def _find_phasic_cache(phasic_path: Path) -> Path | None:
    direct = phasic_path / "phasic_trace_cache.h5"
    if direct.exists():
        return direct
    matches = sorted(phasic_path.rglob("phasic_trace_cache.h5"))
    return matches[0] if matches else None


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _feature_outputs_snapshot(phasic_path: Path) -> dict[str, str]:
    features_dir = phasic_path / "features"
    if not features_dir.exists():
        return {}
    return {
        str(path.resolve()): _file_hash(path)
        for path in sorted(features_dir.rglob("*"))
        if path.is_file()
    }


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "applied_dff_preview_stack_verification_summary.csv"
    json_path = output_dir / "applied_dff_preview_stack_verification_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        row = {}
        for key in SUMMARY_FIELDS:
            value = summary.get(key, "")
            if isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, (list, tuple)):
                value = ";".join(str(x) for x in value)
            elif isinstance(value, float) and not math.isfinite(value):
                value = ""
            row[key] = value
        writer.writerow(row)
    json_path.write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return csv_path, json_path


def _empty_summary(
    *,
    roi: str,
    strategy: str,
    selected_output: Path,
    applied_dir: Path,
    peak_dir: Path,
) -> dict[str, Any]:
    return {
        "verification_passed": False,
        "verification_mode": VERIFICATION_MODE,
        "roi": roi,
        "requested_strategy": strategy,
        "applied_correction_strategy": "",
        "applied_trace_source": "",
        "applied_trace_units": "",
        "applied_trace_available": False,
        "applied_trace_complete": False,
        "applied_trace_warning_level": "",
        "applied_trace_review_required": False,
        "n_chunks": 0,
        "n_chunks_available": 0,
        "n_chunks_unavailable": 0,
        "feature_preview_expected": strategy != "no_correction",
        "feature_preview_ran": False,
        "feature_preview_refused_expectedly": False,
        "feature_preview_refusal_message": "",
        "feature_detection_input_trace": "",
        "feature_detection_input_strategy": "",
        "feature_detection_input_source": "",
        "feature_detection_preview": False,
        "peak_detector_source_function": "",
        "peak_detector_mode": "",
        "peak_detection_config_source": "",
        "peak_detection_config_path": "",
        "peak_detection_config_hash": "",
        "n_events": 0,
        "n_event_rows_with_wrong_detection_input": 0,
        "n_event_rows_with_wrong_boundary_mode": 0,
        "n_event_rows_with_metrics_populated": 0,
        "hdf5_modified": False,
        "replaces_existing_feature_outputs": False,
        "applied_preview_dir": str(applied_dir),
        "peak_preview_dir": str(peak_dir),
        "verification_output_dir": str(selected_output),
        "failure_messages": "",
    }


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _validate_applied_stage(
    applied_report: dict[str, Any],
    *,
    requested_strategy: str,
    failures: list[str],
) -> dict[str, Any]:
    summary = dict(applied_report.get("summary") or {})
    expected_source = {
        "dynamic_fit": "dynamic_fit_dff",
        "signal_only_f0": "signal_only_f0_dff",
        "no_correction": "none",
    }[requested_strategy]
    _require(
        summary.get("applied_correction_strategy") == requested_strategy,
        "applied strategy does not match requested strategy",
        failures,
    )
    _require(summary.get("correction_strategy_selection") == "explicit", "correction selection is not explicit", failures)
    _require(summary.get("applied_trace_source") == expected_source, "unexpected applied trace source", failures)
    _require(_as_bool(summary.get("hdf5_modified")) is False, "applied stage reports HDF5 modified", failures)
    _require(_as_bool(summary.get("feature_detection_input")) is False, "applied stage ran feature detection", failures)
    _require(Path(applied_report.get("summary_csv", "")).exists(), "applied summary CSV missing", failures)
    _require(Path(applied_report.get("summary_json", "")).exists(), "applied summary JSON missing", failures)
    if requested_strategy == "no_correction":
        _require(_as_bool(summary.get("applied_trace_available")) is False, "no_correction produced applied trace", failures)
        _require(summary.get("applied_trace_units") == "none", "no_correction applied units not none", failures)
    else:
        _require(_as_bool(summary.get("applied_trace_available")) is True, "applied trace unavailable", failures)
        _require(_as_bool(summary.get("applied_trace_complete")) is True, "applied trace incomplete", failures)
        _require(_as_int(summary.get("n_chunks_available")) == _as_int(summary.get("n_chunks"), -1), "not all chunks available", failures)
        _require(_as_int(summary.get("n_chunks_unavailable"), -1) == 0, "some chunks unavailable", failures)
        _require(summary.get("applied_trace_units") == "dff", "applied trace units not dff", failures)
        _require(Path(applied_report.get("trace_csv", "")).exists(), "applied trace CSV missing", failures)
    return summary


def _validate_peak_stage(
    peak_report: dict[str, Any],
    *,
    requested_strategy: str,
    applied_summary: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    summary = dict(peak_report.get("summary") or {})
    events_path = Path(peak_report.get("events_csv", ""))
    _require(summary.get("feature_detection_input_trace") == "applied_dff", "feature input trace is not applied_dff", failures)
    _require(summary.get("feature_detection_input_strategy") == requested_strategy, "feature input strategy mismatch", failures)
    _require(
        summary.get("feature_detection_input_source") == applied_summary.get("applied_trace_source"),
        "feature input source mismatch",
        failures,
    )
    _require(summary.get("feature_detection_input_units") == "dff", "feature input units not dff", failures)
    _require(_as_bool(summary.get("feature_detection_preview")) is True, "feature preview flag false", failures)
    _require(_as_bool(summary.get("hdf5_modified")) is False, "peak preview reports HDF5 modified", failures)
    _require(_as_bool(summary.get("replaces_existing_feature_outputs")) is False, "peak preview replaces existing outputs", failures)
    _require(summary.get("peak_detector_source_function") == "get_peak_indices_for_trace", "wrong peak detector function", failures)
    _require(summary.get("peak_detector_mode") == "peak_only_no_event_segmentation", "wrong peak detector mode", failures)
    _require(bool(str(summary.get("peak_detection_config_hash") or "")), "missing peak config hash", failures)
    _require(bool(str(summary.get("peak_detection_config_source") or "")), "missing peak config source", failures)
    _require(events_path.exists(), "peak event CSV missing", failures)

    wrong_detection = 0
    wrong_boundary = 0
    metrics_populated = 0
    if events_path.exists():
        events = pd.read_csv(events_path, dtype=str, keep_default_na=False)
        expected_hash = str(summary.get("peak_detection_config_hash") or "")
        if len(events) > 0:
            wrong_detection = int(
                (
                    (events.get("detection_input_trace", "") != "applied_dff")
                    | (events.get("detection_input_strategy", "") != requested_strategy)
                    | (events.get("detection_input_source", "") != str(applied_summary.get("applied_trace_source")))
                    | (events.get("detection_preview", "").astype(str).str.lower() != "true")
                    | (events.get("peak_detection_config_hash", "") != expected_hash)
                ).sum()
            )
            wrong_boundary = int(
                (events.get("event_boundary_mode", "") != "peak_only_no_event_segmentation").sum()
            )
            metrics_populated = int(
                (
                    (events.get("event_start_sample", "") != "")
                    | (events.get("event_end_sample", "") != "")
                    | (events.get("event_auc", "") != "")
                    | (events.get("event_duration_sec", "") != "")
                    | (events.get("event_metrics_available", "").astype(str).str.lower() != "false")
                ).sum()
            )
        _require(wrong_detection == 0, "event rows have wrong detection provenance", failures)
        _require(wrong_boundary == 0, "event rows have wrong boundary mode", failures)
        _require(metrics_populated == 0, "event rows have populated event metrics", failures)
    return {
        **summary,
        "n_event_rows_with_wrong_detection_input": wrong_detection,
        "n_event_rows_with_wrong_boundary_mode": wrong_boundary,
        "n_event_rows_with_metrics_populated": metrics_populated,
    }


def _path_inside(path_text: str, root: Path) -> bool:
    if not path_text:
        return True
    try:
        Path(path_text).resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def verify_applied_dff_preview_stack(
    phasic_out: str | os.PathLike[str],
    *,
    roi: str,
    strategy: str,
    output_dir: str | os.PathLike[str],
    peak_config_json: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    max_preview_chunks: int | None = None,
    max_preview_events: int | None = None,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}")

    phasic_path = Path(phasic_out).resolve()
    requested_output = Path(output_dir).resolve()
    selected_output = _select_output_dir(requested_output, overwrite=overwrite, dry_run=dry_run)
    applied_dir = selected_output / "applied_preview"
    peak_dir = selected_output / "peak_preview"
    summary = _empty_summary(
        roi=roi,
        strategy=strategy,
        selected_output=selected_output,
        applied_dir=applied_dir,
        peak_dir=peak_dir,
    )

    if dry_run:
        return {
            **summary,
            "dry_run": True,
            "would_run_applied_stage": True,
            "would_run_peak_preview": strategy != "no_correction",
        }

    failures: list[str] = []
    cache_path = _find_phasic_cache(phasic_path)
    cache_hash_before = _file_hash(cache_path) if cache_path is not None else ""
    features_before = _feature_outputs_snapshot(phasic_path)

    try:
        applied_report = apply_recording_correction_strategy(
            phasic_path,
            roi=roi,
            strategy=strategy,
            output_dir=applied_dir,
            overwrite=overwrite,
            max_preview_chunks=max_preview_chunks,
        )
        applied_summary = _validate_applied_stage(
            applied_report,
            requested_strategy=strategy,
            failures=failures,
        )
        summary.update(
            {
                "applied_correction_strategy": applied_summary.get("applied_correction_strategy", ""),
                "applied_trace_source": applied_summary.get("applied_trace_source", ""),
                "applied_trace_units": applied_summary.get("applied_trace_units", ""),
                "applied_trace_available": _as_bool(applied_summary.get("applied_trace_available")),
                "applied_trace_complete": _as_bool(applied_summary.get("applied_trace_complete")),
                "applied_trace_warning_level": applied_summary.get("applied_trace_warning_level", ""),
                "applied_trace_review_required": _as_bool(applied_summary.get("applied_trace_review_required")),
                "n_chunks": _as_int(applied_summary.get("n_chunks")),
                "n_chunks_available": _as_int(applied_summary.get("n_chunks_available")),
                "n_chunks_unavailable": _as_int(applied_summary.get("n_chunks_unavailable")),
            }
        )

        if strategy == "no_correction":
            try:
                run_applied_dff_feature_preview(
                    applied_dir,
                    output_dir=peak_dir,
                    peak_config_json=peak_config_json,
                    overwrite=overwrite,
                    max_preview_events=max_preview_events,
                )
                failures.append("no_correction peak preview did not refuse input")
            except AppliedFeaturePreviewError as exc:
                message = str(exc)
                expected = EXPECTED_NO_CORRECTION_REFUSAL in message
                summary.update(
                    {
                        "feature_preview_expected": False,
                        "feature_preview_ran": False,
                        "feature_preview_refused_expectedly": bool(expected),
                        "feature_preview_refusal_message": message,
                    }
                )
                _require(expected, "no_correction refusal message was unexpected", failures)
                _require(not (peak_dir / "applied_dff_feature_events.csv").exists(), "no_correction event output was created", failures)
        else:
            peak_report = run_applied_dff_feature_preview(
                applied_dir,
                output_dir=peak_dir,
                peak_config_json=peak_config_json,
                overwrite=overwrite,
                max_preview_events=max_preview_events,
            )
            peak_summary = _validate_peak_stage(
                peak_report,
                requested_strategy=strategy,
                applied_summary=applied_summary,
                failures=failures,
            )
            summary.update(
                {
                    "feature_preview_expected": True,
                    "feature_preview_ran": True,
                    "feature_detection_input_trace": peak_summary.get("feature_detection_input_trace", ""),
                    "feature_detection_input_strategy": peak_summary.get("feature_detection_input_strategy", ""),
                    "feature_detection_input_source": peak_summary.get("feature_detection_input_source", ""),
                    "feature_detection_preview": _as_bool(peak_summary.get("feature_detection_preview")),
                    "peak_detector_source_function": peak_summary.get("peak_detector_source_function", ""),
                    "peak_detector_mode": peak_summary.get("peak_detector_mode", ""),
                    "peak_detection_config_source": peak_summary.get("peak_detection_config_source", ""),
                    "peak_detection_config_path": peak_summary.get("peak_detection_config_path", ""),
                    "peak_detection_config_hash": peak_summary.get("peak_detection_config_hash", ""),
                    "n_events": int(peak_summary.get("n_events") or 0),
                    "n_event_rows_with_wrong_detection_input": int(peak_summary.get("n_event_rows_with_wrong_detection_input") or 0),
                    "n_event_rows_with_wrong_boundary_mode": int(peak_summary.get("n_event_rows_with_wrong_boundary_mode") or 0),
                    "n_event_rows_with_metrics_populated": int(peak_summary.get("n_event_rows_with_metrics_populated") or 0),
                    "replaces_existing_feature_outputs": _as_bool(peak_summary.get("replaces_existing_feature_outputs")),
                }
            )

        _require(_path_inside(str(applied_dir), selected_output), "applied preview dir outside verification output", failures)
        _require(_path_inside(str(peak_dir), selected_output), "peak preview dir outside verification output", failures)
        # Intermediate applied_preview/ and peak_preview/ outputs are intentionally
        # retained so verification results remain inspectable after failures.
    except Exception as exc:
        if not isinstance(exc, AppliedFeaturePreviewError):
            failures.append(str(exc))
        else:
            failures.append(str(exc))

    cache_hash_after = _file_hash(cache_path) if cache_path is not None else ""
    features_after = _feature_outputs_snapshot(phasic_path)
    hdf5_modified = bool(cache_hash_before and cache_hash_after and cache_hash_before != cache_hash_after)
    replaced_features = features_before != features_after
    if hdf5_modified:
        failures.append("HDF5 hash changed")
    if replaced_features:
        failures.append("existing feature outputs changed")
    summary["hdf5_modified"] = bool(hdf5_modified)
    summary["replaces_existing_feature_outputs"] = bool(
        summary.get("replaces_existing_feature_outputs") or replaced_features
    )
    summary["failure_messages"] = ";".join(str(x) for x in failures if str(x))
    summary["verification_passed"] = len(failures) == 0

    summary_csv, summary_json = _write_summary(selected_output, summary)
    report = {
        **summary,
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "dry_run": False,
    }
    if failures:
        raise PreviewStackVerificationError(summary["failure_messages"], report=report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"verification_passed: {str(report.get('verification_passed')).lower()}")
    print(f"verification_mode: {report.get('verification_mode')}")
    print(f"roi: {report.get('roi')}")
    print(f"requested_strategy: {report.get('requested_strategy')}")
    print(f"applied_correction_strategy: {report.get('applied_correction_strategy')}")
    print(f"applied_trace_source: {report.get('applied_trace_source')}")
    print(f"feature_preview_ran: {str(report.get('feature_preview_ran')).lower()}")
    print(f"feature_preview_refused_expectedly: {str(report.get('feature_preview_refused_expectedly')).lower()}")
    print(f"n_events: {report.get('n_events')}")
    print(f"hdf5_modified: {str(report.get('hdf5_modified')).lower()}")
    print(f"replaces_existing_feature_outputs: {str(report.get('replaces_existing_feature_outputs')).lower()}")
    if report.get("failure_messages"):
        print(f"failure_messages: {report.get('failure_messages')}")
    if report.get("summary_csv"):
        print(f"summary_csv: {report.get('summary_csv')}")
    if report.get("summary_json"):
        print(f"summary_json: {report.get('summary_json')}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the applied_dff preview stack.")
    parser.add_argument("--phasic-out", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--peak-config-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-preview-chunks", type=int, default=None)
    parser.add_argument("--max-preview-events", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = verify_applied_dff_preview_stack(
            args.phasic_out,
            roi=args.roi,
            strategy=args.strategy,
            output_dir=args.output_dir,
            peak_config_json=args.peak_config_json,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            max_preview_chunks=args.max_preview_chunks,
            max_preview_events=args.max_preview_events,
        )
    except PreviewStackVerificationError as exc:
        if exc.report:
            _print_report(exc.report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
