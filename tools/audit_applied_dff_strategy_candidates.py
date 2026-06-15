#!/usr/bin/env python3
"""Advisory audit of explicit applied_dff strategy candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photometry_pipeline.core import signal_only_f0_candidate as signal_f0_core  # noqa: E402

TOOL_NAME = "audit_applied_dff_strategy_candidates"
AUDIT_MODE = "advisory_manual_strategy_candidate_audit"
STRATEGIES = ("dynamic_fit", "signal_only_f0")
OUTPUT_FILES = (
    "applied_dff_strategy_candidate_audit.csv",
    "applied_dff_strategy_candidate_audit.json",
    "applied_dff_strategy_candidate_audit_summary.json",
    "applied_dff_strategy_candidate_audit_provenance.json",
)
FORBIDDEN_FIELD_NAMES = {
    "recommended_strategy",
    "chosen_strategy",
    "selected_strategy",
    "best_strategy",
}

FIELDS = [
    "roi",
    "strategy_candidate",
    "strategy_candidate_status",
    "review_required",
    "manual_review_priority",
    "evidence_summary",
    "blocking_issues",
    "cautions",
    "source_phasic_cache_path",
    "source_phasic_cache_sha256",
    "n_chunks",
    "n_chunks_with_required_source_data",
    "n_chunks_missing_required_source_data",
    "n_nonfinite_source_values",
    "n_candidate_warnings",
    "n_candidate_cautions",
    "n_candidate_blockers",
    "min_value",
    "max_value",
    "mean_value",
    "median_value",
    "fraction_finite",
    "candidate_negative_dff_present",
    "viability_counts",
    "confidence_counts",
    "flag_counts",
    "viability_count_summary",
    "confidence_count_summary",
    "top_flag_counts",
    "n_viable_chunks",
    "n_contextual_chunks",
    "n_hard_inspect_chunks",
    "n_low_confidence_chunks",
    "n_medium_confidence_chunks",
    "n_high_confidence_chunks",
    "n_chunks_with_large_anchor_gap",
    "n_chunks_with_few_anchors",
    "n_chunks_with_capped_extrapolation",
    "n_chunks_with_above_signal_excessive",
    "example_problem_chunks",
    "hdf5_modified_source_phasic_cache",
    "legacy_features_modified",
]


class AppliedDffStrategyCandidateAuditError(RuntimeError):
    """Raised when advisory strategy candidate auditing cannot complete."""


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _legacy_features_hash(phasic_out: Path) -> str:
    path = phasic_out / "features" / "features.csv"
    return _file_sha256(path) if path.exists() else ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in FIELDS:
                value = row.get(key, "")
                if isinstance(value, bool):
                    value = str(value).lower()
                elif isinstance(value, (dict, list, tuple)):
                    value = json.dumps(_json_safe(value), sort_keys=True)
                elif isinstance(value, float) and not math.isfinite(value):
                    value = ""
                out[key] = value
            writer.writerow(out)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise AppliedDffStrategyCandidateAuditError(
            f"output exists, refusing without --overwrite: {output_dir}"
        )
    if output_dir.exists() and overwrite:
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _contains_path(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _assert_safe_audit_output_dir(output_dir: Path, phasic_out: Path, source_cache: Path) -> None:
    output_resolved = output_dir.resolve()
    phasic_resolved = phasic_out.resolve()
    source_resolved = source_cache.resolve()
    legacy_features = phasic_resolved / "features" / "features.csv"
    legacy_features_dir = phasic_resolved / "features"
    if output_resolved == phasic_resolved:
        raise AppliedDffStrategyCandidateAuditError("unsafe audit output_dir equals phasic_out")
    if _contains_path(output_resolved, source_resolved):
        raise AppliedDffStrategyCandidateAuditError("unsafe audit output_dir contains phasic_trace_cache.h5")
    if output_resolved == legacy_features_dir.resolve():
        raise AppliedDffStrategyCandidateAuditError("unsafe audit output_dir equals legacy features directory")
    if legacy_features.exists() and _contains_path(output_resolved, legacy_features):
        raise AppliedDffStrategyCandidateAuditError("unsafe audit output_dir contains legacy features.csv")


def _list_rois(cache: h5py.File) -> list[str]:
    if "meta/rois" in cache:
        out = []
        for value in np.asarray(cache["meta/rois"][()]).reshape(-1).tolist():
            if isinstance(value, bytes):
                out.append(value.decode("utf-8"))
            else:
                out.append(str(value))
        return out
    if "roi" in cache:
        return sorted(str(x) for x in cache["roi"].keys())
    return []


def _list_chunk_ids(cache: h5py.File) -> list[int]:
    if "meta/chunk_ids" not in cache:
        return []
    return [int(x) for x in np.asarray(cache["meta/chunk_ids"][()]).reshape(-1).tolist()]


def _flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(";") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _format_counter_summary(counter: Counter[str]) -> str:
    return "; ".join(f"{key}={counter[key]}" for key in sorted(counter) if counter[key])


def _format_top_flag_counts(counter: Counter[str], max_items: int = 8) -> str:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:max_items]
    return "; ".join(f"{key}={value}" for key, value in items if value)


def _stats(values: list[np.ndarray]) -> dict[str, Any]:
    if not values:
        return {
            "min_value": "",
            "max_value": "",
            "mean_value": "",
            "median_value": "",
            "fraction_finite": 0.0,
            "n_nonfinite": 0,
        }
    arr = np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in values])
    finite = arr[np.isfinite(arr)]
    return {
        "min_value": float(np.min(finite)) if finite.size else "",
        "max_value": float(np.max(finite)) if finite.size else "",
        "mean_value": float(np.mean(finite)) if finite.size else "",
        "median_value": float(np.median(finite)) if finite.size else "",
        "fraction_finite": float(finite.size / arr.size) if arr.size else 0.0,
        "n_nonfinite": int(arr.size - finite.size),
    }


def _base_row(roi: str, strategy: str, cache_path: Path, source_hash: str, n_chunks: int) -> dict[str, Any]:
    return {
        "roi": roi,
        "strategy_candidate": strategy,
        "strategy_candidate_status": "unknown",
        "review_required": False,
        "manual_review_priority": "compare_dynamic_fit_and_signal_only_f0",
        "evidence_summary": "",
        "blocking_issues": "",
        "cautions": "",
        "source_phasic_cache_path": str(cache_path),
        "source_phasic_cache_sha256": source_hash,
        "n_chunks": int(n_chunks),
        "n_chunks_with_required_source_data": 0,
        "n_chunks_missing_required_source_data": 0,
        "n_nonfinite_source_values": 0,
        "n_candidate_warnings": 0,
        "n_candidate_cautions": 0,
        "n_candidate_blockers": 0,
        "min_value": "",
        "max_value": "",
        "mean_value": "",
        "median_value": "",
        "fraction_finite": 0.0,
        "candidate_negative_dff_present": False,
        "viability_counts": {},
        "confidence_counts": {},
        "flag_counts": {},
        "viability_count_summary": "",
        "confidence_count_summary": "",
        "top_flag_counts": "",
        "n_viable_chunks": 0,
        "n_contextual_chunks": 0,
        "n_hard_inspect_chunks": 0,
        "n_low_confidence_chunks": 0,
        "n_medium_confidence_chunks": 0,
        "n_high_confidence_chunks": 0,
        "n_chunks_with_large_anchor_gap": 0,
        "n_chunks_with_few_anchors": 0,
        "n_chunks_with_capped_extrapolation": 0,
        "n_chunks_with_above_signal_excessive": 0,
        "example_problem_chunks": "",
        "hdf5_modified_source_phasic_cache": False,
        "legacy_features_modified": False,
    }


def _audit_dynamic_fit(cache: h5py.File, roi: str, chunk_ids: list[int], cache_path: Path, source_hash: str) -> dict[str, Any]:
    row = _base_row(roi, "dynamic_fit", cache_path, source_hash, len(chunk_ids))
    blockers: list[str] = []
    cautions: list[str] = []
    values: list[np.ndarray] = []
    missing = 0
    for chunk_id in chunk_ids:
        path = f"roi/{roi}/chunk_{chunk_id}"
        if path not in cache:
            blockers.append(f"missing chunk group {chunk_id}")
            missing += 1
            continue
        grp = cache[path]
        chunk_blockers = []
        if "dff" not in grp:
            chunk_blockers.append("missing dff")
        if "time_sec" not in grp:
            chunk_blockers.append("missing time_sec")
        if chunk_blockers:
            blockers.append(f"chunk {chunk_id}: {', '.join(chunk_blockers)}")
            missing += 1
            continue
        dff = np.asarray(grp["dff"][()], dtype=float).reshape(-1)
        time_sec = np.asarray(grp["time_sec"][()]).reshape(-1)
        if dff.shape != time_sec.shape:
            blockers.append(f"chunk {chunk_id}: dff/time_sec length mismatch")
            missing += 1
            continue
        values.append(dff)
    stats = _stats(values)
    row.update(stats)
    row["n_nonfinite_source_values"] = int(stats["n_nonfinite"])
    row["n_chunks_with_required_source_data"] = len(values)
    row["n_chunks_missing_required_source_data"] = missing
    if stats["n_nonfinite"]:
        cautions.append(f"non-finite source dff values: {stats['n_nonfinite']}")
    if blockers:
        row["strategy_candidate_status"] = "blocked"
        row["manual_review_priority"] = "blocked_until_source_data_fixed"
        row["n_candidate_blockers"] = len(blockers)
    elif cautions:
        row["strategy_candidate_status"] = "available_with_cautions"
        row["manual_review_priority"] = "inspect_dynamic_fit_quality"
        row["review_required"] = True
        row["n_candidate_cautions"] = len(cautions)
    else:
        row["strategy_candidate_status"] = "available"
        row["manual_review_priority"] = "no_obvious_candidate_blocker"
    row["blocking_issues"] = "; ".join(blockers)
    row["cautions"] = "; ".join(cautions)
    row["n_candidate_warnings"] = len(cautions)
    row["evidence_summary"] = (
        f"dynamic_fit source dff present for {len(values)}/{len(chunk_ids)} chunks; "
        f"missing required data for {missing}; finite fraction {row['fraction_finite']}; "
        "inspect correction quality before choosing."
    )
    return row


def _audit_signal_only_f0(cache: h5py.File, roi: str, chunk_ids: list[int], cache_path: Path, source_hash: str) -> dict[str, Any]:
    row = _base_row(roi, "signal_only_f0", cache_path, source_hash, len(chunk_ids))
    blockers: list[str] = []
    cautions: list[str] = []
    values: list[np.ndarray] = []
    missing = 0
    viability = Counter()
    confidence = Counter()
    flags = Counter()
    problem_chunk_ids: list[int] = []
    few_anchor_chunks = 0
    negative = False
    computed = 0
    for chunk_id in chunk_ids:
        path = f"roi/{roi}/chunk_{chunk_id}"
        if path not in cache:
            blockers.append(f"missing chunk group {chunk_id}")
            missing += 1
            continue
        grp = cache[path]
        chunk_blockers = []
        if "sig_raw" not in grp:
            chunk_blockers.append("missing signal/raw input")
        if "time_sec" not in grp:
            chunk_blockers.append("missing time_sec")
        if chunk_blockers:
            blockers.append(f"chunk {chunk_id}: {', '.join(chunk_blockers)}")
            missing += 1
            continue
        signal = np.asarray(grp["sig_raw"][()], dtype=float).reshape(-1)
        time_sec = np.asarray(grp["time_sec"][()]).reshape(-1)
        if signal.shape != time_sec.shape:
            blockers.append(f"chunk {chunk_id}: signal/time_sec length mismatch")
            missing += 1
            continue
        try:
            diagnostic = signal_f0_core.compute_signal_only_f0_candidate(
                signal,
                time_sec,
                return_uncapped_candidate=True,
            )
        except Exception as exc:
            blockers.append(f"chunk {chunk_id}: core candidate failed: {exc}")
            missing += 1
            continue
        f0 = diagnostic.get("signal_only_f0_candidate_uncapped")
        if f0 is None:
            blockers.append(f"chunk {chunk_id}: uncapped F0 missing")
            missing += 1
            continue
        f0_arr = np.asarray(f0, dtype=float).reshape(-1)
        if f0_arr.shape != signal.shape or np.any(~np.isfinite(f0_arr)) or np.any(f0_arr <= 0):
            blockers.append(f"chunk {chunk_id}: invalid uncapped F0")
            missing += 1
            continue
        dff = (signal - f0_arr) / f0_arr
        values.append(dff)
        computed += 1
        negative = negative or bool(np.any(dff < 0))
        v = str(diagnostic.get("signal_only_f0_candidate_viability") or "unknown")
        c = str(diagnostic.get("signal_only_f0_candidate_confidence") or "unknown")
        viability[v] += 1
        confidence[c] += 1
        chunk_flags = _flags(diagnostic.get("signal_only_f0_flags"))
        flags.update(chunk_flags)
        has_few_anchor_flag = (
            "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_FEW_ANCHORS" in chunk_flags
            or "SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS" in chunk_flags
        )
        if has_few_anchor_flag:
            few_anchor_chunks += 1
        if (
            v.lower() == "hard_inspect"
            or c.lower() == "low"
            or "SIGNAL_ONLY_F0_ABOVE_SIGNAL_EXCESSIVE" in chunk_flags
            or "SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP" in chunk_flags
            or has_few_anchor_flag
        ):
            problem_chunk_ids.append(int(chunk_id))
        if v.lower() not in {"viable", "ok", "acceptable"} or c.lower() == "low" or chunk_flags:
            cautions.append(f"chunk {chunk_id}: viability={v}, confidence={c}, flags={','.join(chunk_flags)}")
    stats = _stats(values)
    row.update(stats)
    row["n_nonfinite_source_values"] = int(stats["n_nonfinite"])
    row["n_chunks_with_required_source_data"] = computed
    row["n_chunks_missing_required_source_data"] = missing
    row["candidate_negative_dff_present"] = negative
    row["viability_counts"] = dict(viability)
    row["confidence_counts"] = dict(confidence)
    row["flag_counts"] = dict(flags)
    row["viability_count_summary"] = _format_counter_summary(viability)
    row["confidence_count_summary"] = _format_counter_summary(confidence)
    row["top_flag_counts"] = _format_top_flag_counts(flags)
    row["n_viable_chunks"] = int(viability.get("viable", 0))
    row["n_contextual_chunks"] = int(viability.get("contextual", 0))
    row["n_hard_inspect_chunks"] = int(viability.get("hard_inspect", 0))
    row["n_low_confidence_chunks"] = int(confidence.get("low", 0))
    row["n_medium_confidence_chunks"] = int(confidence.get("medium", 0))
    row["n_high_confidence_chunks"] = int(confidence.get("high", 0))
    row["n_chunks_with_large_anchor_gap"] = int(flags.get("SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP", 0))
    row["n_chunks_with_few_anchors"] = int(few_anchor_chunks)
    row["n_chunks_with_capped_extrapolation"] = int(flags.get("SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION", 0))
    row["n_chunks_with_above_signal_excessive"] = int(flags.get("SIGNAL_ONLY_F0_ABOVE_SIGNAL_EXCESSIVE", 0))
    row["example_problem_chunks"] = ",".join(str(x) for x in problem_chunk_ids[:20])
    row["n_candidate_warnings"] = len(cautions)
    row["n_candidate_cautions"] = len(cautions)
    if blockers:
        row["strategy_candidate_status"] = "blocked"
        row["manual_review_priority"] = "blocked_until_source_data_fixed"
        row["n_candidate_blockers"] = len(blockers)
    elif cautions:
        row["strategy_candidate_status"] = "viable_with_cautions"
        row["manual_review_priority"] = "inspect_signal_only_f0_candidate"
        row["review_required"] = True
    else:
        row["strategy_candidate_status"] = "viable"
        row["manual_review_priority"] = "no_obvious_candidate_blocker"
    row["blocking_issues"] = "; ".join(blockers)
    row["cautions"] = "; ".join(cautions)
    row["evidence_summary"] = (
        f"signal_only_f0 candidate computed for {computed}/{len(chunk_ids)} chunks; "
        f"cautions {len(cautions)}; negative dF/F present {negative}; "
        "review candidate diagnostics before choosing."
    )
    return row


def audit_applied_dff_strategy_candidates(
    phasic_out: str | os.PathLike[str],
    *,
    roi: str | None = None,
    output_dir: str | os.PathLike[str],
    overwrite: bool = False,
    dry_run: bool = False,
    fail_on_warning: bool = False,
) -> dict[str, Any]:
    phasic_path = Path(phasic_out).resolve()
    output_path = Path(output_dir).resolve()
    cache_path = phasic_path / "phasic_trace_cache.h5"
    if not cache_path.exists():
        raise AppliedDffStrategyCandidateAuditError(f"missing source phasic cache: {cache_path}")
    rois: list[str] = []
    try:
        with h5py.File(cache_path, "r") as h5:
            rois = _list_rois(h5)
            if roi:
                rois = [roi] if roi in rois else []
    except Exception:
        rois = []
    if dry_run:
        return {
            "dry_run": True,
            "phasic_out": str(phasic_path),
            "roi_filter": roi or "",
            "rois_that_would_be_audited": rois,
            "would_audit_dynamic_fit": True,
            "would_audit_signal_only_f0": True,
            "would_write_outputs": True,
            "output_paths": {name: str(output_path / name) for name in OUTPUT_FILES},
        }
    if roi and not rois:
        raise AppliedDffStrategyCandidateAuditError(f"requested ROI not found: {roi}")
    _assert_safe_audit_output_dir(output_path, phasic_path, cache_path)
    _prepare_output(output_path, overwrite=overwrite)
    source_before = _file_sha256(cache_path)
    legacy_before = _legacy_features_hash(phasic_path)
    rows: list[dict[str, Any]] = []
    with h5py.File(cache_path, "r") as h5:
        chunk_ids = _list_chunk_ids(h5)
        for roi_name in rois:
            rows.append(_audit_dynamic_fit(h5, roi_name, chunk_ids, cache_path, source_before))
            rows.append(_audit_signal_only_f0(h5, roi_name, chunk_ids, cache_path, source_before))
    source_after = _file_sha256(cache_path)
    legacy_after = _legacy_features_hash(phasic_path)
    hdf5_modified = source_before != source_after
    legacy_modified = legacy_before != legacy_after
    for row in rows:
        row["hdf5_modified_source_phasic_cache"] = hdf5_modified
        row["legacy_features_modified"] = legacy_modified
    if hdf5_modified or legacy_modified:
        raise AppliedDffStrategyCandidateAuditError("read-only guarantee failed during audit")

    audit_csv = output_path / OUTPUT_FILES[0]
    audit_json = output_path / OUTPUT_FILES[1]
    summary_json = output_path / OUTPUT_FILES[2]
    provenance_json = output_path / OUTPUT_FILES[3]
    _write_csv(audit_csv, rows)
    _write_json(audit_json, {"rows": rows})
    summary = {
        "audit_passed": True,
        "tool_name": TOOL_NAME,
        "phasic_out": str(phasic_path),
        "roi_filter": roi or "",
        "n_rois_audited": len(rois),
        "strategies_audited": list(STRATEGIES),
        "output_dir": str(output_path),
        "audit_csv": str(audit_csv),
        "audit_json": str(audit_json),
        "source_phasic_cache_sha256_before": source_before,
        "source_phasic_cache_sha256_after": source_after,
        "hdf5_modified_source_phasic_cache": hdf5_modified,
        "legacy_features_modified": legacy_modified,
        "n_blocked_candidates": sum(1 for row in rows if row["strategy_candidate_status"] == "blocked"),
        "n_candidates_with_cautions": sum(1 for row in rows if "caution" in str(row["strategy_candidate_status"])),
        "n_candidates_available_or_viable": sum(1 for row in rows if row["strategy_candidate_status"] in {"available", "viable", "available_with_cautions", "viable_with_cautions"}),
        "failure_messages": "",
    }
    if fail_on_warning and summary["n_candidates_with_cautions"]:
        summary["audit_passed"] = False
        summary["failure_messages"] = "candidate cautions present with fail_on_warning"
    provenance = {
        "tool_name": TOOL_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phasic_out": str(phasic_path),
        "roi_filter": roi or "",
        "strategies_audited": list(STRATEGIES),
        "audit_mode": AUDIT_MODE,
        "source_phasic_cache_sha256_before": source_before,
        "source_phasic_cache_sha256_after": source_after,
        "legacy_features_sha256_before": legacy_before,
        "legacy_features_sha256_after": legacy_after,
        "hdf5_modified_source_phasic_cache": hdf5_modified,
        "legacy_features_modified": legacy_modified,
        "output_paths": {
            "audit_csv": str(audit_csv),
            "audit_json": str(audit_json),
            "summary_json": str(summary_json),
            "provenance_json": str(provenance_json),
        },
        "no_auto_selection": True,
        "no_strategy_chosen": True,
        "no_pipeline_execution": True,
        "no_applied_outputs_written": True,
        "no_feature_outputs_written": True,
    }
    forbidden = FORBIDDEN_FIELD_NAMES.intersection(summary).union(FORBIDDEN_FIELD_NAMES.intersection(provenance))
    if forbidden:
        raise AppliedDffStrategyCandidateAuditError(f"forbidden strategy selection fields present: {sorted(forbidden)}")
    _write_json(summary_json, summary)
    _write_json(provenance_json, provenance)
    if not summary["audit_passed"]:
        raise AppliedDffStrategyCandidateAuditError(summary["failure_messages"])
    return {
        "dry_run": False,
        "rows": rows,
        "summary": summary,
        "provenance": provenance,
        "audit_csv": str(audit_csv),
        "audit_json": str(audit_json),
        "summary_json": str(summary_json),
        "provenance_json": str(provenance_json),
    }


def _print_report(report: dict[str, Any]) -> None:
    if report.get("dry_run"):
        for key, value in report.items():
            print(f"{key}: {value}")
        return
    summary = report["summary"]
    for key in (
        "audit_passed",
        "n_rois_audited",
        "n_blocked_candidates",
        "n_candidates_with_cautions",
        "n_candidates_available_or_viable",
        "hdf5_modified_source_phasic_cache",
        "legacy_features_modified",
        "audit_csv",
        "summary_json",
    ):
        print(f"{key}: {summary.get(key, report.get(key))}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit applied_dff explicit strategy candidates.")
    parser.add_argument("--phasic-out", required=True)
    parser.add_argument("--roi", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = audit_applied_dff_strategy_candidates(
            args.phasic_out,
            roi=args.roi,
            output_dir=args.output_dir,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            fail_on_warning=bool(args.fail_on_warning),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
