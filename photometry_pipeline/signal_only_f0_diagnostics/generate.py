"""Backend-only Signal-Only F0 diagnostic generation.

Stage 4D2 reads a completed-run phasic cache read-only and writes bounded,
diagnostic-only JSON/CSV artifacts under the Stage 4D diagnostic namespace. It
does not write manifests, route applied-dF/F, run feature extraction, validate
GUI setup, launch the pipeline, call correction retune, or generate plots.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from photometry_pipeline.core import signal_only_f0_candidate as signal_f0_core
from photometry_pipeline.io.hdf5_cache_reader import (
    CacheReadError,
    list_cache_chunk_ids,
    list_cache_rois,
    load_cache_chunk_fields,
    open_phasic_cache,
)
from photometry_pipeline.signal_only_f0_diagnostics.contract import (
    SIGNAL_ONLY_F0_DIAGNOSTIC_PROVENANCE_FILENAME,
    SIGNAL_ONLY_F0_DIAGNOSTIC_SUMMARY_FILENAME,
    SOURCE_TYPE_COMPLETED_RUN,
    SOURCE_TYPE_DIAGNOSTIC_CACHE,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_SUCCESS,
    build_default_signal_only_f0_diagnostic_cache_output_dir,
    build_default_signal_only_f0_diagnostic_output_dir,
    build_signal_only_f0_diagnostic_provenance,
    build_signal_only_f0_diagnostic_summary,
    make_signal_only_f0_diagnostic_id,
    resolve_completed_run_signal_only_f0_source,
    resolve_diagnostic_cache_signal_only_f0_source,
    validate_signal_only_f0_diagnostic_output_dir,
)


SIGNAL_ONLY_F0_DIAGNOSTIC_CHUNKS_FILENAME = "signal_only_f0_diagnostic_chunks.csv"

CHUNK_CSV_FIELDS = [
    "roi",
    "chunk_id",
    "status",
    "n_samples",
    "signal_finite_count",
    "signal_nonfinite_count",
    "signal_finite_fraction",
    "f0_finite_count",
    "f0_nonfinite_count",
    "f0_finite_fraction",
    "f0_min",
    "f0_median",
    "f0_max",
    "dff_finite_count",
    "dff_nonfinite_count",
    "dff_finite_fraction",
    "dff_min",
    "dff_median",
    "dff_max",
    "dff_p01",
    "dff_p99",
    "negative_dff_count",
    "signal_only_f0_candidate_available",
    "signal_only_f0_candidate_viability",
    "signal_only_f0_candidate_confidence",
    "signal_only_f0_status",
    "signal_only_f0_warning",
    "warning_flags",
    "error",
]


def _failed_result(
    *,
    diagnostic_id: str,
    output_dir: str = "",
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    chunk_statuses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "status": STATUS_FAILED,
        "diagnostic_id": diagnostic_id,
        "output_dir": output_dir,
        "provenance_path": "",
        "summary_path": "",
        "chunk_csv_path": "",
        "trace_csv_paths": [],
        "warnings": list(warnings or []),
        "errors": list(errors or []),
        "chunk_statuses": dict(chunk_statuses or {}),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _write_chunk_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHUNK_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CHUNK_CSV_FIELDS})


def _finite_summary(values: np.ndarray, prefix: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    out: dict[str, Any] = {
        f"{prefix}_finite_count": int(finite.size),
        f"{prefix}_nonfinite_count": int(arr.size - finite.size),
        f"{prefix}_finite_fraction": float(finite.size / arr.size) if arr.size else 0.0,
        f"{prefix}_min": "",
        f"{prefix}_median": "",
        f"{prefix}_max": "",
    }
    if finite.size:
        out.update(
            {
                f"{prefix}_min": float(np.min(finite)),
                f"{prefix}_median": float(np.median(finite)),
                f"{prefix}_max": float(np.max(finite)),
            }
        )
    return out


def _as_flags_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value if str(item))
    return str(value)


def _diagnose_chunk(cache: Any, *, roi: str, chunk_id: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "roi": roi,
        "chunk_id": int(chunk_id),
        "status": STATUS_FAILED,
        "n_samples": 0,
        "signal_finite_count": 0,
        "signal_nonfinite_count": 0,
        "signal_finite_fraction": 0.0,
        "f0_finite_count": "",
        "f0_nonfinite_count": "",
        "f0_finite_fraction": "",
        "f0_min": "",
        "f0_median": "",
        "f0_max": "",
        "dff_finite_count": "",
        "dff_nonfinite_count": "",
        "dff_finite_fraction": "",
        "dff_min": "",
        "dff_median": "",
        "dff_max": "",
        "dff_p01": "",
        "dff_p99": "",
        "negative_dff_count": "",
        "signal_only_f0_candidate_available": "",
        "signal_only_f0_candidate_viability": "",
        "signal_only_f0_candidate_confidence": "",
        "signal_only_f0_status": "",
        "signal_only_f0_warning": "",
        "warning_flags": "",
        "error": "",
    }
    try:
        time_sec, signal = load_cache_chunk_fields(cache, roi, int(chunk_id), ["time_sec", "sig_raw"])
        time_arr = np.asarray(time_sec, dtype=float).reshape(-1)
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
    except CacheReadError as exc:
        row["error"] = str(exc)
        return row
    except Exception as exc:
        row["error"] = f"failed to load source arrays: {exc}"
        return row

    row["n_samples"] = int(signal_arr.size)
    row.update(_finite_summary(signal_arr, "signal"))
    if time_arr.shape != signal_arr.shape:
        row["error"] = (
            f"time_sec/sig_raw length mismatch: time_sec={time_arr.size}, "
            f"sig_raw={signal_arr.size}"
        )
        return row
    if signal_arr.size == 0:
        row["error"] = "empty sig_raw array"
        return row

    try:
        diagnostics = signal_f0_core.compute_signal_only_f0_candidate(
            signal_arr,
            time_arr,
            return_uncapped_candidate=True,
        )
    except Exception as exc:
        row["error"] = f"signal-only F0 computation failed: {exc}"
        return row

    row["signal_only_f0_candidate_available"] = bool(
        diagnostics.get("signal_only_f0_candidate_available", False)
    )
    row["signal_only_f0_candidate_viability"] = str(
        diagnostics.get("signal_only_f0_candidate_viability") or ""
    )
    row["signal_only_f0_candidate_confidence"] = str(
        diagnostics.get("signal_only_f0_candidate_confidence") or ""
    )
    row["signal_only_f0_status"] = str(diagnostics.get("signal_only_f0_status") or "")
    row["signal_only_f0_warning"] = str(diagnostics.get("signal_only_f0_warning") or "")
    row["warning_flags"] = _as_flags_text(diagnostics.get("signal_only_f0_flags"))

    f0 = diagnostics.get("signal_only_f0_candidate_uncapped")
    if f0 is None:
        row["error"] = "signal-only F0 uncapped candidate was not computed"
        return row
    f0_arr = np.asarray(f0, dtype=float).reshape(-1)
    if f0_arr.shape != signal_arr.shape:
        row["error"] = (
            f"signal-only F0 length mismatch: f0={f0_arr.size}, sig_raw={signal_arr.size}"
        )
        return row
    row.update(_finite_summary(f0_arr, "f0"))

    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (signal_arr - f0_arr) / f0_arr
    row.update(_finite_summary(dff, "dff"))
    finite_dff = dff[np.isfinite(dff)]
    if finite_dff.size:
        row["dff_p01"] = float(np.percentile(finite_dff, 1.0))
        row["dff_p99"] = float(np.percentile(finite_dff, 99.0))
        row["negative_dff_count"] = int(np.sum(finite_dff < 0.0))
    else:
        row["negative_dff_count"] = 0

    if not bool(diagnostics.get("signal_only_f0_candidate_available", False)):
        row["error"] = row["signal_only_f0_warning"] or "signal-only F0 candidate unavailable"
        return row
    if row["dff_finite_count"] == 0:
        row["error"] = "computed signal-only dF/F contains no finite values"
        return row

    row["status"] = STATUS_SUCCESS
    row["error"] = ""
    return row


def _selected_chunk_ids(
    available_chunk_ids: list[int],
    *,
    chunk_ids: list[int] | tuple[int, ...] | None,
    max_chunks: int | None,
) -> list[int]:
    available = [int(chunk_id) for chunk_id in available_chunk_ids]
    if chunk_ids is None:
        limit = int(max_chunks) if max_chunks is not None else 1
        limit = max(1, limit)
        return available[:limit]
    requested = [int(chunk_id) for chunk_id in chunk_ids]
    if max_chunks is not None:
        requested = requested[: max(1, int(max_chunks))]
    return requested


def _overall_status(rows: list[dict[str, Any]]) -> str:
    n_success = sum(1 for row in rows if row.get("status") == STATUS_SUCCESS)
    if n_success == len(rows) and rows:
        return STATUS_SUCCESS
    if n_success > 0:
        return STATUS_PARTIAL
    return STATUS_FAILED


def run_signal_only_f0_diagnostic_review(
    completed_run_dir: str | os.PathLike[str],
    *,
    roi: str,
    chunk_ids: list[int] | tuple[int, ...] | None = None,
    max_chunks: int | None = None,
    diagnostic_id: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    allow_existing: bool = False,
    source_type: str = SOURCE_TYPE_COMPLETED_RUN,
) -> dict[str, Any]:
    """Generate bounded Signal-Only F0 diagnostic-review artifacts.

    By default it processes the first available chunk for the requested ROI.
    """
    did = diagnostic_id or make_signal_only_f0_diagnostic_id()
    if not str(roi or "").strip():
        return _failed_result(diagnostic_id=did, errors=["explicit ROI is required"])

    requested_source_type = str(source_type or SOURCE_TYPE_COMPLETED_RUN)
    if requested_source_type == SOURCE_TYPE_DIAGNOSTIC_CACHE:
        source = resolve_diagnostic_cache_signal_only_f0_source(completed_run_dir)
    elif requested_source_type == SOURCE_TYPE_COMPLETED_RUN:
        source = resolve_completed_run_signal_only_f0_source(completed_run_dir)
    else:
        return _failed_result(
            diagnostic_id=did,
            errors=[f"Unsupported Signal-Only F0 diagnostic source_type: {requested_source_type!r}"],
        )
    if not source.ok:
        return _failed_result(diagnostic_id=did, errors=[source.reason])

    if output_dir is not None:
        out_dir_path = Path(output_dir)
    elif requested_source_type == SOURCE_TYPE_DIAGNOSTIC_CACHE:
        cache_root = str((source.diagnostic_cache_metadata or {}).get("cache_root_path") or "")
        out_dir_path = build_default_signal_only_f0_diagnostic_cache_output_dir(cache_root, did)
    else:
        out_dir_path = build_default_signal_only_f0_diagnostic_output_dir(source.completed_run_dir, did)
    output_check = validate_signal_only_f0_diagnostic_output_dir(
        out_dir_path,
        completed_run_dir=source.completed_run_dir,
        phasic_out_dir=source.phasic_out_dir,
        diagnostic_id=did,
        allow_existing=allow_existing,
    )
    if not output_check.ok:
        return _failed_result(
            diagnostic_id=did,
            output_dir=output_check.resolved_output_dir,
            errors=[output_check.reason],
        )

    try:
        with open_phasic_cache(source.phasic_trace_cache_path) as cache:
            rois = list_cache_rois(cache)
            if roi not in rois:
                return _failed_result(
                    diagnostic_id=did,
                    output_dir=output_check.resolved_output_dir,
                    errors=[f"requested ROI '{roi}' not found in phasic cache"],
                )
            available_chunks = list_cache_chunk_ids(cache)
            selected_chunks = _selected_chunk_ids(
                available_chunks,
                chunk_ids=chunk_ids,
                max_chunks=max_chunks,
            )
            if not selected_chunks:
                return _failed_result(
                    diagnostic_id=did,
                    output_dir=output_check.resolved_output_dir,
                    errors=["no chunks selected for Signal-Only F0 diagnostic review"],
                )
            available_set = {int(chunk_id) for chunk_id in available_chunks}
            rows: list[dict[str, Any]] = []
            for chunk_id in selected_chunks:
                if int(chunk_id) not in available_set:
                    rows.append(
                        {
                            "roi": roi,
                            "chunk_id": int(chunk_id),
                            "status": STATUS_FAILED,
                            "n_samples": 0,
                            "signal_finite_count": 0,
                            "signal_nonfinite_count": 0,
                            "signal_finite_fraction": 0.0,
                            "error": f"requested chunk_id {int(chunk_id)} not found in phasic cache",
                        }
                    )
                    continue
                rows.append(_diagnose_chunk(cache, roi=roi, chunk_id=int(chunk_id)))
    except CacheReadError as exc:
        return _failed_result(
            diagnostic_id=did,
            output_dir=output_check.resolved_output_dir,
            errors=[f"failed to read phasic cache: {exc}"],
        )
    except Exception as exc:
        return _failed_result(
            diagnostic_id=did,
            output_dir=output_check.resolved_output_dir,
            errors=[f"unexpected diagnostic setup failure: {exc}"],
        )

    status = _overall_status(rows)
    warnings = sorted(
        {
            str(row.get("warning_flags") or "")
            for row in rows
            if row.get("warning_flags")
        }
    )
    errors = [
        f"chunk {row.get('chunk_id')}: {row.get('error')}"
        for row in rows
        if row.get("status") != STATUS_SUCCESS and row.get("error")
    ]
    chunk_statuses = {
        str(int(row["chunk_id"])): {
            "status": row.get("status", STATUS_FAILED),
            "error": row.get("error", ""),
        }
        for row in rows
    }
    roi_statuses = {
        roi: {
            "status": status,
            "n_chunks_requested": int(len(rows)),
            "n_chunks_success": int(sum(1 for row in rows if row.get("status") == STATUS_SUCCESS)),
            "n_chunks_failed": int(sum(1 for row in rows if row.get("status") != STATUS_SUCCESS)),
        }
    }

    out_dir = Path(output_check.resolved_output_dir)
    out_dir.mkdir(parents=True, exist_ok=bool(allow_existing))
    provenance_path = out_dir / SIGNAL_ONLY_F0_DIAGNOSTIC_PROVENANCE_FILENAME
    summary_path = out_dir / SIGNAL_ONLY_F0_DIAGNOSTIC_SUMMARY_FILENAME
    chunk_csv_path = out_dir / SIGNAL_ONLY_F0_DIAGNOSTIC_CHUNKS_FILENAME

    generated_artifacts = [
        str(provenance_path),
        str(summary_path),
        str(chunk_csv_path),
    ]
    provenance = build_signal_only_f0_diagnostic_provenance(
        diagnostic_id=did,
        source_type=source.source_type,
        completed_run_dir=source.completed_run_dir,
        phasic_out_dir=source.phasic_out_dir,
        phasic_trace_cache_path=source.phasic_trace_cache_path,
        config_source_path=source.config_source_path,
        selected_rois=[roi],
        selected_chunks=[int(row["chunk_id"]) for row in rows],
        selected_window={},
        source_artifact_hashes={},
        diagnostic_cache_metadata=source.diagnostic_cache_metadata or {},
    )
    summary = build_signal_only_f0_diagnostic_summary(
        diagnostic_id=did,
        status=status,
        roi_statuses=roi_statuses,
        chunk_statuses=chunk_statuses,
        warnings=warnings,
        errors=errors,
        generated_artifact_paths=generated_artifacts,
        diagnostic_metrics={
            "n_chunks_requested": int(len(rows)),
            "n_chunks_success": int(sum(1 for row in rows if row.get("status") == STATUS_SUCCESS)),
            "n_chunks_failed": int(sum(1 for row in rows if row.get("status") != STATUS_SUCCESS)),
            "default_chunk_selection": "first_available_chunk" if chunk_ids is None else "explicit_chunk_ids",
            "trace_csvs_written": False,
            "plots_written": False,
        },
    )

    _write_json(provenance_path, provenance)
    _write_json(summary_path, summary)
    _write_chunk_csv(chunk_csv_path, rows)

    return {
        "ok": status == STATUS_SUCCESS,
        "status": status,
        "diagnostic_id": did,
        "output_dir": str(out_dir),
        "provenance_path": str(provenance_path),
        "summary_path": str(summary_path),
        "chunk_csv_path": str(chunk_csv_path),
        "trace_csv_paths": [],
        "warnings": warnings,
        "errors": errors,
        "chunk_statuses": chunk_statuses,
        "source_type": source.source_type,
        "diagnostic_cache": dict(source.diagnostic_cache_metadata or {}),
    }
