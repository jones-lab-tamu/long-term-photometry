#!/usr/bin/env python3
"""Write production applied_dff cache outputs for explicit dynamic_fit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photometry_pipeline.io.hdf5_cache_reader import (  # noqa: E402
    CacheReadError,
    list_cache_chunk_ids,
    list_cache_rois,
    list_cache_source_files,
    load_cache_chunk_fields,
    open_phasic_cache,
)

SCHEMA_VERSION = "1.0"
CONTRACT_NAME = "applied_dff_production_output_contract"
CONTRACT_VERSION = "1.0"
TOOL_NAME = "write_applied_dff_cache"
DEFAULT_DIR_NAME = "applied_dff"
SUPPORTED_STRATEGY = "dynamic_fit"
FLAG_PARTIAL = "APPLIED_TRACE_PARTIAL"
FLAG_NONFINITE = "NONFINITE_APPLIED_DFF_VALUES"

OUTPUT_FILENAMES = (
    "applied_trace_cache.h5",
    "applied_correction_summary.csv",
    "applied_correction_summary.json",
    "applied_correction_chunks.csv",
    "applied_correction_chunks.json",
)

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
    "applied_trace_review_required",
    "applied_trace_warning_level",
    "applied_trace_flags",
    "source_phasic_cache_path",
    "source_phasic_cache_sha256",
    "applied_trace_cache_path",
    "applied_trace_cache_sha256",
    "applied_trace_cache_sha256_location",
    "hdf5_modified_source_phasic_cache",
    "feature_detection_input",
    "created_at_utc",
    "tool_name",
    "contract_name",
    "contract_version",
    "schema_version",
    "output_dir",
]

CHUNK_FIELDS = [
    "roi",
    "chunk_id",
    "source_file",
    "available",
    "applied_trace_source",
    "applied_trace_units",
    "n_samples",
    "warning_level",
    "review_required",
    "flags",
    "reason_if_unavailable",
]


class AppliedDffCacheWriteError(RuntimeError):
    """Raised when production applied_dff cache writing cannot proceed."""


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


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_phasic_cache(phasic_out: Path) -> Path:
    path = phasic_out / "phasic_trace_cache.h5"
    if not path.exists():
        raise AppliedDffCacheWriteError(f"missing source phasic cache: {path}")
    return path


def _check_supported_strategy(strategy: str) -> None:
    if strategy == SUPPORTED_STRATEGY:
        return
    if strategy == "signal_only_f0":
        raise AppliedDffCacheWriteError(
            "signal_only_f0 production applied cache writing is not implemented yet"
        )
    if strategy == "no_correction":
        raise AppliedDffCacheWriteError(
            "no_correction production applied cache writing is not implemented yet"
        )
    if strategy == "auto":
        raise AppliedDffCacheWriteError(
            "auto strategy selection is not implemented for production applied cache writing"
        )
    raise AppliedDffCacheWriteError(f"unsupported requested_correction_strategy: {strategy}")


def _source_file_for_chunk(source_files: list[str], chunk_ids: list[int], chunk_id: int) -> str:
    if len(source_files) == len(chunk_ids):
        try:
            return str(source_files[chunk_ids.index(int(chunk_id))])
        except ValueError:
            return ""
    if len(source_files) == 1:
        return str(source_files[0])
    return ";".join(str(x) for x in source_files)


def _recording_key(source_files: list[str], phasic_path: Path) -> str:
    if source_files:
        return ";".join(str(x) for x in source_files)
    return phasic_path.name


def _warning_at_least(level: str, minimum: str) -> str:
    order = {"none": 0, "info": 1, "caution": 2, "severe": 3}
    current = str(level or "none")
    required = str(minimum or "none")
    return current if order.get(current, 0) >= order.get(required, 0) else required


def _flags_text(flags: list[str]) -> str:
    return ";".join(dict.fromkeys(str(x) for x in flags if str(x)))


def _output_exists(output_dir: Path) -> bool:
    return any((output_dir / name).exists() for name in OUTPUT_FILENAMES)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if _output_exists(output_dir) and not overwrite:
        raise AppliedDffCacheWriteError(
            f"output already exists, refusing to overwrite without --overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in OUTPUT_FILENAMES:
            path = output_dir / name
            if path.exists():
                path.unlink()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, bool):
                    value = str(value).lower()
                elif isinstance(value, (list, tuple)):
                    value = _flags_text([str(x) for x in value])
                elif isinstance(value, float) and not math.isfinite(value):
                    value = ""
                out[key] = value
            writer.writerow(out)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _write_scalar_string(group: h5py.Group, name: str, value: Any) -> None:
    group.create_dataset(name, data=str(value), dtype=_string_dtype())


def _write_scalar_bool(group: h5py.Group, name: str, value: bool) -> None:
    group.create_dataset(name, data=np.asarray(bool(value), dtype=np.bool_))


def _write_recording_json(group: h5py.Group, name: str, payload: dict[str, Any]) -> None:
    _write_scalar_string(group, name, json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False))


def _load_dynamic_fit_chunks(
    cache: h5py.File,
    *,
    roi: str,
    chunk_ids: list[int],
    source_files: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    for chunk_id in chunk_ids:
        row: dict[str, Any] = {
            "roi": roi,
            "chunk_id": int(chunk_id),
            "source_file": _source_file_for_chunk(source_files, chunk_ids, int(chunk_id)),
            "available": False,
            "applied_trace_source": "dynamic_fit_dff",
            "applied_trace_units": "dff",
            "n_samples": 0,
            "warning_level": "none",
            "review_required": False,
            "flags": "",
            "reason_if_unavailable": "",
        }
        try:
            time_sec, dff = load_cache_chunk_fields(cache, roi, int(chunk_id), ["time_sec", "dff"])
            time_arr = np.asarray(time_sec).reshape(-1)
            dff_arr = np.asarray(dff).reshape(-1)
        except CacheReadError as exc:
            row.update(
                {
                    "reason_if_unavailable": str(exc),
                    "warning_level": "severe",
                    "review_required": True,
                    "flags": "DYNAMIC_FIT_DFF_UNAVAILABLE",
                }
            )
            chunk_rows.append(row)
            continue

        if time_arr.shape != dff_arr.shape:
            raise AppliedDffCacheWriteError(
                f"length mismatch for ROI {roi} chunk {int(chunk_id)}: "
                f"time_sec has {time_arr.size} samples, dff has {dff_arr.size} samples"
            )

        flags: list[str] = []
        warning_level = "none"
        review_required = False
        if dff_arr.size == 0:
            row.update(
                {
                    "reason_if_unavailable": "dynamic_fit_dff_empty",
                    "warning_level": "severe",
                    "review_required": True,
                    "flags": "DYNAMIC_FIT_DFF_UNAVAILABLE",
                }
            )
            chunk_rows.append(row)
            continue
        if not np.all(np.isfinite(dff_arr.astype(float, copy=False))):
            flags.append(FLAG_NONFINITE)
            warning_level = _warning_at_least(warning_level, "caution")
            review_required = True

        row.update(
            {
                "available": True,
                "n_samples": int(dff_arr.size),
                "warning_level": warning_level,
                "review_required": review_required,
                "flags": _flags_text(flags),
            }
        )
        data_rows.append(
            {
                "chunk_id": int(chunk_id),
                "source_file": row["source_file"],
                "time_sec": time_arr,
                "dff": dff_arr,
                "chunk_row": row,
            }
        )
        chunk_rows.append(row)
    return data_rows, chunk_rows


def _summary_from_chunks(
    *,
    roi: str,
    recording_key: str,
    requested_strategy: str,
    output_dir: Path,
    source_cache_path: Path,
    source_hash: str,
    cache_path: Path,
    created_at_utc: str,
    chunk_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    available_count = sum(1 for row in chunk_rows if bool(row.get("available")))
    unavailable_count = len(chunk_rows) - available_count
    levels = [str(row.get("warning_level") or "none") for row in chunk_rows]
    flags: list[str] = []
    for row in chunk_rows:
        flags.extend(str(row.get("flags") or "").split(";"))
    if available_count > 0 and unavailable_count > 0:
        flags.append(FLAG_PARTIAL)
    if "severe" in levels:
        warning_level = "severe"
    elif "caution" in levels:
        warning_level = "caution"
    elif "info" in levels:
        warning_level = "info"
    else:
        warning_level = "none"
    if FLAG_PARTIAL in flags:
        warning_level = _warning_at_least(warning_level, "caution")

    reasons = sorted(
        {
            str(row.get("reason_if_unavailable") or "")
            for row in chunk_rows
            if str(row.get("reason_if_unavailable") or "")
        }
    )
    complete = bool(len(chunk_rows) > 0 and available_count == len(chunk_rows) and unavailable_count == 0)
    review_required = any(str(row.get("review_required")).lower() == "true" or bool(row.get("review_required")) for row in chunk_rows)
    review_required = bool(review_required or unavailable_count > 0)
    return {
        "roi": roi,
        "recording_key": recording_key,
        "requested_correction_strategy": requested_strategy,
        "correction_strategy_selection": "explicit",
        "applied_correction_strategy": SUPPORTED_STRATEGY,
        "applied_trace_source": "dynamic_fit_dff",
        "applied_trace_units": "dff",
        "applied_trace_available": bool(available_count > 0),
        "applied_trace_complete": complete,
        "reason_if_unavailable": "" if available_count > 0 else ";".join(reasons),
        "n_chunks": int(len(chunk_rows)),
        "n_chunks_available": int(available_count),
        "n_chunks_unavailable": int(unavailable_count),
        "applied_trace_review_required": review_required,
        "applied_trace_warning_level": warning_level,
        "applied_trace_flags": _flags_text(flags),
        "source_phasic_cache_path": str(source_cache_path),
        "source_phasic_cache_sha256": source_hash,
        "applied_trace_cache_path": str(cache_path),
        "applied_trace_cache_sha256": "",
        "applied_trace_cache_sha256_location": "external_summary_after_cache_finalization",
        "hdf5_modified_source_phasic_cache": False,
        "feature_detection_input": False,
        "created_at_utc": created_at_utc,
        "tool_name": TOOL_NAME,
        "contract_name": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output_dir),
    }


def _write_hdf5_cache(
    path: Path,
    *,
    summary: dict[str, Any],
    chunk_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    roi: str,
    chunk_ids: list[int],
    source_files: list[str],
) -> None:
    dt = _string_dtype()
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("meta")
        _write_scalar_string(meta, "schema_version", SCHEMA_VERSION)
        _write_scalar_string(meta, "mode", "applied_dff")
        _write_scalar_string(meta, "source_phasic_cache_path", summary["source_phasic_cache_path"])
        _write_scalar_string(meta, "source_phasic_cache_sha256", summary["source_phasic_cache_sha256"])
        meta.create_dataset("rois", data=np.asarray([roi], dtype=object), dtype=dt)
        meta.create_dataset("chunk_ids", data=np.asarray(chunk_ids, dtype=int))
        meta.create_dataset("source_files", data=np.asarray(source_files, dtype=object), dtype=dt)
        _write_scalar_string(meta, "created_at_utc", summary["created_at_utc"])
        _write_scalar_string(meta, "tool_name", TOOL_NAME)
        _write_scalar_string(meta, "contract_name", CONTRACT_NAME)
        _write_scalar_string(meta, "contract_version", CONTRACT_VERSION)

        rec = h5.create_group(f"recording/{roi}")
        # The applied cache cannot contain its own final SHA256: writing that value
        # into this file would change the file hash. The authoritative final cache
        # hash is recorded in the external summary after HDF5 finalization.
        _write_recording_json(rec, "summary", summary)
        _write_recording_json(rec, "provenance_json", {**summary, "chunks": chunk_rows})

        data_by_chunk = {int(row["chunk_id"]): row for row in data_rows}
        chunk_by_id = {int(row["chunk_id"]): row for row in chunk_rows}
        roi_group = h5.create_group(f"roi/{roi}")
        for chunk_id in chunk_ids:
            grp = roi_group.create_group(f"chunk_{int(chunk_id)}")
            chunk_row = chunk_by_id[int(chunk_id)]
            data = data_by_chunk.get(int(chunk_id))
            if data is not None:
                grp.create_dataset("time_sec", data=np.asarray(data["time_sec"]))
                grp.create_dataset("applied_dff", data=np.asarray(data["dff"]))
                grp.create_dataset("dynamic_fit_dff", data=np.asarray(data["dff"]))
            _write_scalar_string(grp, "applied_trace_source", "dynamic_fit_dff")
            _write_scalar_string(grp, "source_file", chunk_row.get("source_file", ""))
            _write_scalar_bool(grp, "available", bool(chunk_row.get("available")))
            _write_scalar_string(grp, "warning_level", chunk_row.get("warning_level", "none"))
            _write_scalar_bool(grp, "review_required", bool(chunk_row.get("review_required")))
            _write_scalar_string(grp, "flags", chunk_row.get("flags", ""))


def write_applied_dff_cache(
    phasic_out: str | os.PathLike[str],
    *,
    roi: str,
    requested_correction_strategy: str,
    output_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    strategy = str(requested_correction_strategy or "").strip()
    _check_supported_strategy(strategy)
    if not str(roi or "").strip():
        raise AppliedDffCacheWriteError("ROI must be provided")

    phasic_path = Path(phasic_out).resolve()
    source_cache = _find_phasic_cache(phasic_path)
    selected_output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else phasic_path / DEFAULT_DIR_NAME
    )

    if dry_run:
        n_chunks_planned = 0
        if source_cache.exists():
            try:
                with open_phasic_cache(str(source_cache)) as cache:
                    n_chunks_planned = len(list_cache_chunk_ids(cache))
            except Exception:
                n_chunks_planned = 0
        return {
            "dry_run": True,
            "would_write_applied_trace_cache": str(selected_output / "applied_trace_cache.h5"),
            "would_write_summary": str(selected_output / "applied_correction_summary.csv"),
            "requested_correction_strategy": strategy,
            "roi": roi,
            "output_dir": str(selected_output),
            "source_phasic_cache_path": str(source_cache),
            "source_phasic_cache_exists": source_cache.exists(),
            "n_chunks_planned": int(n_chunks_planned),
        }

    _prepare_output_dir(selected_output, overwrite=overwrite)
    source_hash_before = _file_sha256(source_cache)
    created_at_utc = datetime.now(timezone.utc).isoformat()
    cache_path = selected_output / "applied_trace_cache.h5"

    with open_phasic_cache(str(source_cache)) as cache:
        rois = list_cache_rois(cache)
        if roi not in rois:
            raise AppliedDffCacheWriteError(f"requested ROI '{roi}' not found in source phasic cache")
        chunk_ids = list_cache_chunk_ids(cache)
        source_files = list_cache_source_files(cache)
        data_rows, chunk_rows = _load_dynamic_fit_chunks(
            cache,
            roi=roi,
            chunk_ids=chunk_ids,
            source_files=source_files,
        )

    source_hash_after_read = _file_sha256(source_cache)
    if source_hash_before != source_hash_after_read:
        raise AppliedDffCacheWriteError("source phasic cache changed while preparing applied cache")

    summary = _summary_from_chunks(
        roi=roi,
        recording_key=_recording_key(source_files, phasic_path),
        requested_strategy=strategy,
        output_dir=selected_output,
        source_cache_path=source_cache,
        source_hash=source_hash_before,
        cache_path=cache_path,
        created_at_utc=created_at_utc,
        chunk_rows=chunk_rows,
    )

    _write_hdf5_cache(
        cache_path,
        summary=summary,
        chunk_rows=chunk_rows,
        data_rows=data_rows,
        roi=roi,
        chunk_ids=chunk_ids,
        source_files=source_files,
    )
    applied_hash = _file_sha256(cache_path)
    source_hash_after = _file_sha256(source_cache)
    summary["applied_trace_cache_sha256"] = applied_hash
    summary["hdf5_modified_source_phasic_cache"] = bool(source_hash_before != source_hash_after)
    if summary["hdf5_modified_source_phasic_cache"]:
        raise AppliedDffCacheWriteError("source phasic cache hash changed after writing applied cache")

    _write_csv(selected_output / "applied_correction_summary.csv", [summary], SUMMARY_FIELDS)
    _write_json(selected_output / "applied_correction_summary.json", summary)
    _write_csv(selected_output / "applied_correction_chunks.csv", chunk_rows, CHUNK_FIELDS)
    _write_json(selected_output / "applied_correction_chunks.json", {"chunks": chunk_rows})

    return {
        "dry_run": False,
        "phasic_out": str(phasic_path),
        "output_dir": str(selected_output),
        "source_phasic_cache_path": str(source_cache),
        "applied_trace_cache_path": str(cache_path),
        "summary_csv": str(selected_output / "applied_correction_summary.csv"),
        "summary_json": str(selected_output / "applied_correction_summary.json"),
        "chunks_csv": str(selected_output / "applied_correction_chunks.csv"),
        "chunks_json": str(selected_output / "applied_correction_chunks.json"),
        "summary": summary,
        "chunks": chunk_rows,
    }


def _print_report(report: dict[str, Any]) -> None:
    if report.get("dry_run"):
        for key in (
            "dry_run",
            "would_write_applied_trace_cache",
            "would_write_summary",
            "requested_correction_strategy",
            "roi",
            "output_dir",
            "source_phasic_cache_path",
            "source_phasic_cache_exists",
            "n_chunks_planned",
        ):
            print(f"{key}: {report.get(key)}")
        return
    summary = report["summary"]
    for key in (
        "roi",
        "requested_correction_strategy",
        "applied_correction_strategy",
        "applied_trace_source",
        "applied_trace_units",
        "applied_trace_available",
        "applied_trace_complete",
        "n_chunks",
        "n_chunks_available",
        "n_chunks_unavailable",
        "applied_trace_warning_level",
        "applied_trace_review_required",
        "hdf5_modified_source_phasic_cache",
        "feature_detection_input",
        "applied_trace_cache_sha256",
    ):
        print(f"{key}: {summary.get(key)}")
    print(f"output_dir: {report['output_dir']}")
    print(f"applied_trace_cache: {report['applied_trace_cache_path']}")
    print(f"summary_csv: {report['summary_csv']}")
    print(f"summary_json: {report['summary_json']}")
    print(f"chunks_csv: {report['chunks_csv']}")
    print(f"chunks_json: {report['chunks_json']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write production applied_dff cache outputs.")
    parser.add_argument("--phasic-out", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--requested-correction-strategy", default=None)
    parser.add_argument("--strategy", default=None, help="Alias for --requested-correction-strategy")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    strategy = args.requested_correction_strategy or args.strategy
    if not strategy:
        print("ERROR: --requested-correction-strategy is required", file=sys.stderr)
        return 1
    try:
        report = write_applied_dff_cache(
            args.phasic_out,
            roi=args.roi,
            requested_correction_strategy=strategy,
            output_dir=args.output_dir,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
