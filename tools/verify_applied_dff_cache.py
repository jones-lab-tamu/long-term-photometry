#!/usr/bin/env python3
"""Verify production applied_dff cache artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STRATEGIES = ("dynamic_fit", "signal_only_f0")
VERIFICATION_MODE = "production_applied_dff_cache"
APPLIED_HASH_LOCATION = "external_summary_after_cache_finalization"
FLAG_PARTIAL = "APPLIED_TRACE_PARTIAL"
FLAG_NONFINITE = "NONFINITE_APPLIED_DFF_VALUES"


class AppliedDffCacheVerificationError(RuntimeError):
    """Raised when production applied_dff cache verification fails."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report or {}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _features_snapshot(phasic_out: Path) -> dict[str, str]:
    features = phasic_out / "features" / "features.csv"
    if not features.exists():
        return {}
    return {str(features.resolve()): _file_sha256(features)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_single_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise AppliedDffCacheVerificationError(f"summary CSV must contain exactly one row: {path}")
    return rows[0]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y", "on"}


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(";") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _warning_rank(level: Any) -> int:
    return {"none": 0, "info": 1, "caution": 2, "severe": 3}.get(str(level or "none").lower(), 0)


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_scalar(value.item())
        return [_decode_scalar(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return _decode_scalar(value.item())
    return value


def _h5_value(h5: h5py.File | h5py.Group, path: str) -> Any:
    return _decode_scalar(h5[path][()])


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _summary_values_match(csv_value: Any, json_value: Any) -> bool:
    if isinstance(json_value, bool):
        return _as_bool(csv_value) is json_value
    return str(csv_value) == str(json_value)


def _base_report(
    *,
    roi: str,
    strategy: str,
    write_summary: Path | None,
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
        "n_chunks": 0,
        "n_chunks_available": 0,
        "n_chunks_unavailable": 0,
        "applied_trace_warning_level": "",
        "applied_trace_review_required": False,
        "applied_trace_flags": "",
        "source_phasic_cache_sha256_matches": False,
        "applied_trace_cache_sha256_matches": False,
        "hdf5_modified_source_phasic_cache": False,
        "feature_detection_input": False,
        "n_available_chunks_checked": 0,
        "n_unavailable_chunks_checked": 0,
        "n_trace_formula_failures": 0,
        "n_missing_required_datasets": 0,
        "n_wrong_strategy_datasets": 0,
        "n_nonfinite_applied_dff_chunks": 0,
        "negative_dff_present": False,
        "source_features_modified": False,
        "verification_output_path": str(write_summary or ""),
        "failure_messages": "",
    }


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _check_path(h5: h5py.File | h5py.Group, path: str, failures: list[str], report: dict[str, Any]) -> bool:
    if path in h5:
        return True
    report["n_missing_required_datasets"] += 1
    failures.append(f"missing required dataset: {path}")
    return False


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.allclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True)


def _validate_available_dynamic_fit(
    *,
    src_grp: h5py.Group,
    app_grp: h5py.Group,
    failures: list[str],
    report: dict[str, Any],
    chunk_id: int,
    summary_flags: list[str],
    chunk_flags: list[str],
) -> None:
    for name in ("time_sec", "applied_dff", "dynamic_fit_dff"):
        _check_path(app_grp, name, failures, report)
    _check_path(src_grp, "dff", failures, report)
    _check_path(src_grp, "time_sec", failures, report)
    for forbidden in ("signal_raw_for_dff", "signal_only_f0_uncapped_for_dff", "signal_only_f0_dff", "denominator_trace"):
        if forbidden in app_grp:
            report["n_wrong_strategy_datasets"] += 1
            failures.append(f"wrong strategy dataset present for dynamic_fit chunk {chunk_id}: {forbidden}")
    if not all(name in app_grp for name in ("time_sec", "applied_dff", "dynamic_fit_dff")) or "dff" not in src_grp or "time_sec" not in src_grp:
        return
    src_dff = np.asarray(src_grp["dff"][()], dtype=float).reshape(-1)
    src_time = np.asarray(src_grp["time_sec"][()]).reshape(-1)
    applied = np.asarray(app_grp["applied_dff"][()], dtype=float).reshape(-1)
    dyn = np.asarray(app_grp["dynamic_fit_dff"][()], dtype=float).reshape(-1)
    app_time = np.asarray(app_grp["time_sec"][()]).reshape(-1)
    if not _allclose(applied, src_dff) or not _allclose(dyn, src_dff) or not _allclose(app_time, src_time):
        report["n_trace_formula_failures"] += 1
        failures.append(f"dynamic_fit trace mismatch for chunk {chunk_id}")
    _check_nonfinite(applied, summary_flags, chunk_flags, failures, report, chunk_id)


def _check_nonfinite(
    values: np.ndarray,
    summary_flags: list[str],
    chunk_flags: list[str],
    failures: list[str],
    report: dict[str, Any],
    chunk_id: int,
) -> None:
    if np.any(~np.isfinite(values)):
        report["n_nonfinite_applied_dff_chunks"] += 1
        if FLAG_NONFINITE not in set(summary_flags + chunk_flags):
            failures.append(f"non-finite applied_dff without {FLAG_NONFINITE} flag for chunk {chunk_id}")


def _validate_available_signal_only(
    *,
    src_grp: h5py.Group,
    app_grp: h5py.Group,
    failures: list[str],
    report: dict[str, Any],
    chunk_id: int,
    summary_flags: list[str],
    chunk_flags: list[str],
) -> None:
    for name in ("time_sec", "applied_dff", "signal_raw_for_dff", "signal_only_f0_uncapped_for_dff", "signal_only_f0_dff"):
        _check_path(app_grp, name, failures, report)
    _check_path(src_grp, "sig_raw", failures, report)
    _check_path(src_grp, "time_sec", failures, report)
    for forbidden in ("dynamic_fit_dff", "denominator_trace"):
        if forbidden in app_grp:
            report["n_wrong_strategy_datasets"] += 1
            failures.append(f"wrong strategy dataset present for signal_only_f0 chunk {chunk_id}: {forbidden}")
    required = ("time_sec", "applied_dff", "signal_raw_for_dff", "signal_only_f0_uncapped_for_dff", "signal_only_f0_dff")
    if not all(name in app_grp for name in required) or "sig_raw" not in src_grp or "time_sec" not in src_grp:
        return
    signal = np.asarray(app_grp["signal_raw_for_dff"][()], dtype=float).reshape(-1)
    denominator = np.asarray(app_grp["signal_only_f0_uncapped_for_dff"][()], dtype=float).reshape(-1)
    applied = np.asarray(app_grp["applied_dff"][()], dtype=float).reshape(-1)
    dff = np.asarray(app_grp["signal_only_f0_dff"][()], dtype=float).reshape(-1)
    src_signal = np.asarray(src_grp["sig_raw"][()], dtype=float).reshape(-1)
    src_time = np.asarray(src_grp["time_sec"][()]).reshape(-1)
    app_time = np.asarray(app_grp["time_sec"][()]).reshape(-1)
    if np.any(applied < 0):
        report["negative_dff_present"] = True
    if denominator.shape != signal.shape or not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
        report["n_trace_formula_failures"] += 1
        failures.append(f"invalid signal_only_f0 denominator for chunk {chunk_id}")
        return
    expected = (signal - denominator) / denominator
    if (
        not _allclose(signal, src_signal)
        or not _allclose(applied, dff)
        or not _allclose(applied, expected)
        or not _allclose(app_time, src_time)
    ):
        report["n_trace_formula_failures"] += 1
        failures.append(f"signal_only_f0 formula mismatch for chunk {chunk_id}")
    _check_nonfinite(applied, summary_flags, chunk_flags, failures, report, chunk_id)


def verify_applied_dff_cache(
    phasic_out: str | os.PathLike[str],
    *,
    roi: str,
    strategy: str,
    applied_output_dir: str | os.PathLike[str],
    write_summary: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    fail_on_warning: bool = False,
) -> dict[str, Any]:
    strategy = str(strategy or "").strip()
    if strategy not in STRATEGIES:
        raise AppliedDffCacheVerificationError(
            f"unsupported production applied_dff verification strategy: {strategy}"
        )
    phasic_path = Path(phasic_out).resolve()
    output_dir = Path(applied_output_dir).resolve()
    summary_path = Path(write_summary).resolve() if write_summary is not None else None
    report = _base_report(roi=roi, strategy=strategy, write_summary=summary_path)
    if dry_run:
        return {**report, "dry_run": True, "would_verify_applied_output_dir": str(output_dir)}

    failures: list[str] = []
    source_cache = phasic_path / "phasic_trace_cache.h5"
    applied_cache = output_dir / "applied_trace_cache.h5"
    summary_csv = output_dir / "applied_correction_summary.csv"
    summary_json = output_dir / "applied_correction_summary.json"
    chunks_csv = output_dir / "applied_correction_chunks.csv"
    chunks_json = output_dir / "applied_correction_chunks.json"
    required_files = (source_cache, applied_cache, summary_csv, summary_json, chunks_csv, chunks_json)
    for path in required_files:
        _require(path.exists(), f"missing required file: {path}", failures)
    if failures:
        report["failure_messages"] = ";".join(failures)
        _write_report(summary_path, report)
        raise AppliedDffCacheVerificationError(report["failure_messages"], report=report)

    source_hash_before = _file_sha256(source_cache)
    features_before = _features_snapshot(phasic_path)
    summary_csv_row = _read_single_csv(summary_csv)
    summary = _read_json(summary_json)
    chunks = _read_csv_rows(chunks_csv)
    chunks_payload = _read_json(chunks_json)
    if isinstance(chunks_payload, dict) and "chunks" in chunks_payload:
        _require(len(chunks_payload["chunks"]) == len(chunks), "chunk CSV/JSON row count mismatch", failures)

    actual_source_hash = _file_sha256(source_cache)
    actual_applied_hash = _file_sha256(applied_cache)
    report.update(
        {
            "applied_correction_strategy": summary.get("applied_correction_strategy", ""),
            "applied_trace_source": summary.get("applied_trace_source", ""),
            "applied_trace_units": summary.get("applied_trace_units", ""),
            "applied_trace_available": _as_bool(summary.get("applied_trace_available")),
            "applied_trace_complete": _as_bool(summary.get("applied_trace_complete")),
            "n_chunks": _as_int(summary.get("n_chunks")),
            "n_chunks_available": _as_int(summary.get("n_chunks_available")),
            "n_chunks_unavailable": _as_int(summary.get("n_chunks_unavailable")),
            "applied_trace_warning_level": summary.get("applied_trace_warning_level", ""),
            "applied_trace_review_required": _as_bool(summary.get("applied_trace_review_required")),
            "applied_trace_flags": summary.get("applied_trace_flags", ""),
            "source_phasic_cache_sha256_matches": summary.get("source_phasic_cache_sha256") == actual_source_hash,
            "applied_trace_cache_sha256_matches": summary.get("applied_trace_cache_sha256") == actual_applied_hash,
            "hdf5_modified_source_phasic_cache": _as_bool(summary.get("hdf5_modified_source_phasic_cache")),
            "feature_detection_input": _as_bool(summary.get("feature_detection_input")),
        }
    )
    expected_source = {"dynamic_fit": "dynamic_fit_dff", "signal_only_f0": "signal_only_f0_dff"}[strategy]
    _require(report["source_phasic_cache_sha256_matches"], "source_phasic_cache_sha256 mismatch", failures)
    _require(report["applied_trace_cache_sha256_matches"], "applied_trace_cache_sha256 mismatch", failures)
    _require(summary.get("applied_trace_cache_sha256_location") == APPLIED_HASH_LOCATION, "wrong applied_trace_cache_sha256_location", failures)
    _require(_as_bool(summary.get("hdf5_modified_source_phasic_cache")) is False, "summary reports source HDF5 modified", failures)
    _require(_as_bool(summary.get("feature_detection_input")) is False, "summary reports feature detection input", failures)
    _require(summary.get("requested_correction_strategy") == strategy, "requested strategy mismatch in summary", failures)
    _require(summary.get("correction_strategy_selection") == "explicit", "correction strategy selection is not explicit", failures)
    _require(summary.get("applied_correction_strategy") == strategy, "applied strategy mismatch in summary", failures)
    _require(summary.get("applied_trace_source") == expected_source, "applied trace source mismatch", failures)
    _require(summary.get("applied_trace_units") == "dff", "applied trace units mismatch", failures)
    for key in (
        "requested_correction_strategy",
        "applied_correction_strategy",
        "applied_trace_source",
        "source_phasic_cache_sha256",
        "applied_trace_cache_sha256",
        "applied_trace_cache_sha256_location",
        "hdf5_modified_source_phasic_cache",
        "feature_detection_input",
    ):
        _require(
            _summary_values_match(summary_csv_row.get(key, ""), summary.get(key, "")),
            f"summary CSV/JSON mismatch for {key}",
            failures,
        )

    available_rows = [row for row in chunks if _as_bool(row.get("available"))]
    unavailable_rows = [row for row in chunks if not _as_bool(row.get("available"))]
    expected_available = len(available_rows) > 0
    expected_complete = len(chunks) > 0 and len(available_rows) == len(chunks) and len(unavailable_rows) == 0
    _require(_as_bool(summary.get("applied_trace_available")) == expected_available, "applied_trace_available does not match chunks", failures)
    _require(_as_bool(summary.get("applied_trace_complete")) == expected_complete, "applied_trace_complete does not match chunks", failures)
    _require(_as_int(summary.get("n_chunks")) == len(chunks), "n_chunks does not match chunk summary", failures)
    _require(_as_int(summary.get("n_chunks_available")) == len(available_rows), "n_chunks_available does not match chunk summary", failures)
    _require(_as_int(summary.get("n_chunks_unavailable")) == len(unavailable_rows), "n_chunks_unavailable does not match chunk summary", failures)
    if available_rows and unavailable_rows:
        _require(FLAG_PARTIAL in _flags(summary.get("applied_trace_flags")), "partial output missing APPLIED_TRACE_PARTIAL", failures)
        _require(_as_bool(summary.get("applied_trace_complete")) is False, "partial output marked complete", failures)
        _require(_as_bool(summary.get("applied_trace_review_required")) is True, "partial output missing review_required", failures)
        _require(_warning_rank(summary.get("applied_trace_warning_level")) >= _warning_rank("caution"), "partial output warning below caution", failures)
    if fail_on_warning:
        _require(_warning_rank(summary.get("applied_trace_warning_level")) == 0, "verification warning present with fail_on_warning", failures)

    if strategy == "signal_only_f0":
        _require(summary.get("f0_source_for_signal_only_f0") == "core_uncapped_signal_only_f0_candidate", "wrong signal_only_f0 F0 source", failures)
        _require(summary.get("signal_only_f0_denominator_source") == "signal_only_f0_candidate_uncapped", "wrong signal_only_f0 denominator source", failures)
        _require(_as_bool(summary.get("signal_only_f0_negative_dff_preserved")) is True, "negative dff preservation not recorded", failures)

    chunk_by_id = {_as_int(row.get("chunk_id")): row for row in chunks}
    summary_flags = _flags(summary.get("applied_trace_flags"))
    with h5py.File(source_cache, "r") as src, h5py.File(applied_cache, "r") as app:
        for path in (
            "meta/schema_version",
            "meta/mode",
            "meta/source_phasic_cache_path",
            "meta/source_phasic_cache_sha256",
            "meta/rois",
            "meta/chunk_ids",
            "meta/source_files",
            "meta/created_at_utc",
            "meta/tool_name",
            "meta/contract_name",
            "meta/contract_version",
            f"recording/{roi}/summary",
            f"recording/{roi}/provenance_json",
        ):
            _check_path(app, path, failures, report)
        if "meta/source_phasic_cache_sha256" in app:
            _require(_h5_value(app, "meta/source_phasic_cache_sha256") == actual_source_hash, "HDF5 source hash mismatch", failures)
        expected_chunks = [int(x) for x in np.asarray(src["meta/chunk_ids"][()]).reshape(-1).tolist()]
        _require(set(expected_chunks) == set(chunk_by_id), "chunk summary does not cover expected chunks", failures)
        for chunk_id in expected_chunks:
            app_path = f"roi/{roi}/chunk_{chunk_id}"
            src_path = f"roi/{roi}/chunk_{chunk_id}"
            if not _check_path(app, app_path, failures, report):
                continue
            _check_path(src, src_path, failures, report)
            app_grp = app[app_path]
            src_grp = src[src_path] if src_path in src else None
            provenance_ok = True
            for name in ("applied_trace_source", "source_file", "available", "warning_level", "review_required", "flags"):
                provenance_ok = _check_path(app_grp, name, failures, report) and provenance_ok
            row = chunk_by_id.get(chunk_id, {})
            available = _as_bool(row.get("available"))
            h5_available = _as_bool(_h5_value(app_grp, "available")) if "available" in app_grp else available
            _require(h5_available == available, f"HDF5/summary availability mismatch for chunk {chunk_id}", failures)
            if "applied_trace_source" in app_grp:
                _require(_h5_value(app_grp, "applied_trace_source") == expected_source, f"HDF5 applied trace source mismatch for chunk {chunk_id}", failures)
            chunk_flags = _flags(row.get("flags")) + (_flags(_h5_value(app_grp, "flags")) if "flags" in app_grp else [])
            if not provenance_ok:
                continue
            if available:
                report["n_available_chunks_checked"] += 1
                if src_grp is None:
                    continue
                if strategy == "dynamic_fit":
                    _validate_available_dynamic_fit(
                        src_grp=src_grp,
                        app_grp=app_grp,
                        failures=failures,
                        report=report,
                        chunk_id=chunk_id,
                        summary_flags=summary_flags,
                        chunk_flags=chunk_flags,
                    )
                else:
                    _validate_available_signal_only(
                        src_grp=src_grp,
                        app_grp=app_grp,
                        failures=failures,
                        report=report,
                        chunk_id=chunk_id,
                        summary_flags=summary_flags,
                        chunk_flags=chunk_flags,
                    )
            else:
                report["n_unavailable_chunks_checked"] += 1
                if "applied_dff" in app_grp:
                    failures.append(f"unavailable chunk has applied_dff dataset: {chunk_id}")
                _require(str(row.get("warning_level", "none")).lower() != "none", f"unavailable chunk warning_level is none: {chunk_id}", failures)
                _require(_as_bool(row.get("review_required")) is True, f"unavailable chunk review_required false: {chunk_id}", failures)
                _require(bool(str(row.get("reason_if_unavailable") or "")), f"unavailable chunk missing reason_if_unavailable: {chunk_id}", failures)

    source_hash_after = _file_sha256(source_cache)
    features_after = _features_snapshot(phasic_path)
    report["hdf5_modified_source_phasic_cache"] = source_hash_before != source_hash_after
    report["source_features_modified"] = features_before != features_after
    if report["hdf5_modified_source_phasic_cache"]:
        failures.append("source phasic cache changed during verification")
    if report["source_features_modified"]:
        failures.append("existing features changed during verification")
    report["verification_passed"] = len(failures) == 0
    report["failure_messages"] = ";".join(str(x) for x in failures if str(x))
    _write_report(summary_path, report)
    if failures:
        raise AppliedDffCacheVerificationError(report["failure_messages"], report=report)
    return report


def _print_report(report: dict[str, Any]) -> None:
    for key in (
        "verification_passed",
        "verification_mode",
        "roi",
        "requested_strategy",
        "applied_trace_source",
        "applied_trace_complete",
        "n_chunks_available",
        "n_trace_formula_failures",
        "n_missing_required_datasets",
        "n_wrong_strategy_datasets",
        "n_nonfinite_applied_dff_chunks",
        "negative_dff_present",
        "hdf5_modified_source_phasic_cache",
        "feature_detection_input",
        "source_features_modified",
        "verification_output_path",
        "failure_messages",
    ):
        print(f"{key}: {report.get(key)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a production applied_dff cache.")
    parser.add_argument("--phasic-out", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--applied-output-dir", required=True)
    parser.add_argument("--write-summary", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = verify_applied_dff_cache(
            args.phasic_out,
            roi=args.roi,
            strategy=args.strategy,
            applied_output_dir=args.applied_output_dir,
            write_summary=args.write_summary,
            dry_run=bool(args.dry_run),
            fail_on_warning=bool(args.fail_on_warning),
        )
    except AppliedDffCacheVerificationError as exc:
        if exc.report:
            _print_report(exc.report)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
