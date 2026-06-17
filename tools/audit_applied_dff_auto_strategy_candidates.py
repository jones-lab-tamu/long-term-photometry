#!/usr/bin/env python3
"""Read-only provisional auto-strategy candidate audit for applied_dff."""

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

TOOL_NAME = "audit_applied_dff_auto_strategy_candidates"
TOOL_VERSION = "1.0"
SCHEMA_VERSION = "1.0"
DECISION_CONTRACT_VERSION = "auto_strategy_candidate_audit_v1"
AUDIT_MODE = "read_only_provisional_auto_strategy_candidate_audit"
OUTPUT_FILES = (
    "auto_strategy_candidate_audit.csv",
    "auto_strategy_candidate_audit.json",
    "auto_strategy_candidate_audit_summary.json",
    "auto_strategy_candidate_audit_provenance.json",
)
DECISIONS = ("dynamic_fit", "signal_only_f0", "needs_review")
CONFIDENCE_LEVELS = ("high", "medium", "low", "none")
DECISION_STATUSES = ("decided", "blocked", "needs_review")

# Conservative first-pass thresholds. These are audit thresholds only; they are
# intentionally fail-closed and are not a production auto-selection policy.
MAX_DYNAMIC_HARD_FLAG_FRACTION = 0.01
MAX_DYNAMIC_NEGATIVE_OR_MIXED_FRACTION = 0.02
MAX_DYNAMIC_REFERENCE_LOW_OR_FLAT_FRACTION = 0.01
MAX_DYNAMIC_CRITICAL_SLOPE_WARNING_FRACTION = 0.02
MIN_DYNAMIC_SUPPORT_CORR_MEDIAN = 0.35

STRONG_REF_FAILURE_NEGATIVE_OR_MIXED_FRACTION = 0.25
STRONG_REF_FAILURE_CRITICAL_SLOPE_FRACTION = 0.25
STRONG_REF_FAILURE_LOW_OR_FLAT_FRACTION = 0.10
POOR_REF_COUPLING_CORR_MEDIAN = 0.20

MIN_SIGNAL_ONLY_AVAILABLE_FRACTION = 0.98
MIN_SIGNAL_ONLY_VIABLE_OR_CONTEXTUAL_FRACTION = 0.90
MAX_SIGNAL_ONLY_HARD_INSPECT_FRACTION = 0.02
MAX_SIGNAL_ONLY_LOW_CONFIDENCE_FRACTION_FOR_RESCUE = 0.65
MAX_SIGNAL_ONLY_LARGE_ANCHOR_GAP_FRACTION = 0.10
MAX_SIGNAL_ONLY_FEW_ANCHOR_FRACTION = 0.10

# A provisional signal_only_f0 label is intentionally stricter than "rescue
# candidate" evidence. Caution-heavy correction-reference failures should remain
# needs_review until manually validated.
MIN_SIGNAL_ONLY_VIABLE_FRACTION_FOR_DECISION = 0.95
MAX_SIGNAL_ONLY_CONTEXTUAL_FRACTION_FOR_DECISION = 0.05
MAX_SIGNAL_ONLY_HARD_INSPECT_FRACTION_FOR_DECISION = 0.0
MIN_SIGNAL_ONLY_HIGH_CONFIDENCE_FRACTION_FOR_DECISION = 0.90
MAX_SIGNAL_ONLY_LOW_CONFIDENCE_FRACTION_FOR_DECISION = 0.05
MAX_SIGNAL_ONLY_EDGE_EXTRAPOLATION_MEDIAN_FOR_DECISION = 0.25
MAX_SIGNAL_ONLY_LARGE_ANCHOR_GAP_FRACTION_FOR_DECISION = 0.0
MAX_SIGNAL_ONLY_FEW_ANCHOR_FRACTION_FOR_DECISION = 0.0

REQUIRED_DATASETS = ("time_sec", "sig_raw", "uv_raw", "fit_ref", "dff")
OPTIONAL_DATASETS = ("signal_only_f0_candidate", "baseline_ref_candidate")
NONFINITE_DATASETS = ("dff", "sig_raw", "uv_raw", "fit_ref")

DYNAMIC_NUMERIC_ATTRS = (
    "dynamic_fit_qc_signal_iso_corr",
    "dynamic_fit_qc_signal_fitted_ref_corr",
    "dynamic_fit_qc_iso_fitted_ref_corr",
    "dynamic_fit_qc_fitted_ref_to_signal_range_ratio",
    "dynamic_fit_qc_fitted_ref_to_iso_range_ratio",
    "dynamic_fit_qc_fitted_ref_total_variance",
    "dynamic_fit_qc_fitted_ref_response_scale_fraction",
    "dynamic_fit_qc_fitted_ref_baseline_scale_fraction",
    "dynamic_fit_qc_slope_fraction_negative",
    "dynamic_fit_slope_slope_min",
    "dynamic_fit_slope_slope_max",
    "dynamic_fit_slope_slope_median",
    "dynamic_fit_slope_slope_negative_fraction",
    "dynamic_fit_slope_clamped_fraction",
)
DYNAMIC_BOOL_ATTRS = (
    "dynamic_fit_qc_available",
    "dynamic_fit_qc_dynamic_fit_has_hard_flags",
    "dynamic_fit_qc_dynamic_fit_has_soft_flags",
    "dynamic_fit_qc_dynamic_fit_needs_inspection",
    "dynamic_fit_qc_dynamic_fit_negative_or_mixed_coupling",
    "dynamic_fit_qc_dynamic_fit_reference_flat_or_uninformative",
    "dynamic_fit_qc_dynamic_fit_reference_low_range",
    "dynamic_fit_qc_dynamic_fit_response_scale_rich",
    "dynamic_fit_slope_fallback_used",
    "dynamic_fit_slope_constraint_applied",
    "dynamic_fit_slope_nonnegative_support_insufficient",
)
DYNAMIC_STRING_ATTRS = (
    "dynamic_fit_qc_severity",
    "dynamic_fit_qc_flags",
    "dynamic_fit_qc_soft_flags",
    "dynamic_fit_qc_hard_flags",
    "dynamic_fit_slope_warning_level",
)

SIGNAL_NUMERIC_ATTRS = (
    "signal_only_f0_anchor_count",
    "signal_only_f0_low_support_fraction",
    "signal_only_f0_direct_support_fraction",
    "signal_only_f0_extrapolated_fraction",
    "signal_only_f0_edge_extrapolation_fraction",
    "signal_only_f0_max_anchor_gap_fraction_observed",
    "signal_only_f0_max_anchor_gap_sec_observed",
)
SIGNAL_BOOL_ATTRS = (
    "signal_only_f0_candidate_available",
    "signal_only_f0_high_state_context_applied",
    "signal_only_f0_state_aware_used",
)
SIGNAL_STRING_ATTRS = (
    "signal_only_f0_candidate_viability",
    "signal_only_f0_candidate_confidence",
    "signal_only_f0_flags",
    "signal_only_f0_anchor_status",
    "signal_only_f0_status",
    "signal_only_f0_warning",
)

CSV_FIELDS = [
    "roi",
    "auto_strategy_decision",
    "auto_strategy_confidence",
    "auto_strategy_decision_status",
    "auto_strategy_warning_level",
    "auto_strategy_review_required",
    "auto_strategy_flags",
    "decision_rationale",
    "decision_blockers",
    "decision_cautions",
    "n_chunks",
    "n_chunks_with_required_dynamic_fit_data",
    "n_chunks_with_required_signal_only_data",
    "dynamic_fit_hard_flag_fraction",
    "dynamic_fit_soft_flag_fraction",
    "dynamic_fit_needs_inspection_fraction",
    "dynamic_fit_negative_or_mixed_coupling_fraction",
    "dynamic_fit_reference_flat_or_uninformative_fraction",
    "dynamic_fit_reference_low_range_fraction",
    "dynamic_fit_critical_slope_warning_fraction",
    "dynamic_fit_signal_iso_corr_median",
    "dynamic_fit_signal_fitted_ref_corr_median",
    "dynamic_fit_slope_negative_fraction_median",
    "dynamic_fit_slope_clamped_fraction_median",
    "signal_only_f0_available_fraction",
    "signal_only_f0_viable_fraction",
    "signal_only_f0_contextual_fraction",
    "signal_only_f0_hard_inspect_fraction",
    "signal_only_f0_high_confidence_fraction",
    "signal_only_f0_medium_confidence_fraction",
    "signal_only_f0_low_confidence_fraction",
    "signal_only_f0_edge_extrapolation_fraction_median",
    "signal_only_f0_large_anchor_gap_fraction",
    "signal_only_f0_few_anchor_fraction",
    "source_phasic_cache_path",
    "source_phasic_cache_sha256",
    "hdf5_modified_source_phasic_cache",
    "legacy_features_modified",
    "no_pipeline_execution",
    "no_feature_routing",
]

FORBIDDEN_FIELD_NAMES = {
    "recommended_strategy",
    "chosen_strategy",
    "selected_strategy",
    "best_strategy",
}


class AppliedDffAutoStrategyCandidateAuditError(RuntimeError):
    """Raised when the read-only auto-strategy candidate audit cannot complete."""


def _thresholds() -> dict[str, Any]:
    return {
        "max_dynamic_hard_flag_fraction": MAX_DYNAMIC_HARD_FLAG_FRACTION,
        "max_dynamic_negative_or_mixed_fraction": MAX_DYNAMIC_NEGATIVE_OR_MIXED_FRACTION,
        "max_dynamic_reference_low_or_flat_fraction": MAX_DYNAMIC_REFERENCE_LOW_OR_FLAT_FRACTION,
        "max_dynamic_critical_slope_warning_fraction": MAX_DYNAMIC_CRITICAL_SLOPE_WARNING_FRACTION,
        "min_dynamic_support_corr_median": MIN_DYNAMIC_SUPPORT_CORR_MEDIAN,
        "strong_ref_failure_negative_or_mixed_fraction": STRONG_REF_FAILURE_NEGATIVE_OR_MIXED_FRACTION,
        "strong_ref_failure_critical_slope_fraction": STRONG_REF_FAILURE_CRITICAL_SLOPE_FRACTION,
        "strong_ref_failure_low_or_flat_fraction": STRONG_REF_FAILURE_LOW_OR_FLAT_FRACTION,
        "poor_ref_coupling_corr_median": POOR_REF_COUPLING_CORR_MEDIAN,
        "min_signal_only_available_fraction": MIN_SIGNAL_ONLY_AVAILABLE_FRACTION,
        "min_signal_only_viable_or_contextual_fraction": MIN_SIGNAL_ONLY_VIABLE_OR_CONTEXTUAL_FRACTION,
        "max_signal_only_hard_inspect_fraction": MAX_SIGNAL_ONLY_HARD_INSPECT_FRACTION,
        "max_signal_only_low_confidence_fraction_for_rescue": MAX_SIGNAL_ONLY_LOW_CONFIDENCE_FRACTION_FOR_RESCUE,
        "max_signal_only_large_anchor_gap_fraction": MAX_SIGNAL_ONLY_LARGE_ANCHOR_GAP_FRACTION,
        "max_signal_only_few_anchor_fraction": MAX_SIGNAL_ONLY_FEW_ANCHOR_FRACTION,
        "min_signal_only_viable_fraction_for_decision": MIN_SIGNAL_ONLY_VIABLE_FRACTION_FOR_DECISION,
        "max_signal_only_contextual_fraction_for_decision": MAX_SIGNAL_ONLY_CONTEXTUAL_FRACTION_FOR_DECISION,
        "max_signal_only_hard_inspect_fraction_for_decision": MAX_SIGNAL_ONLY_HARD_INSPECT_FRACTION_FOR_DECISION,
        "min_signal_only_high_confidence_fraction_for_decision": MIN_SIGNAL_ONLY_HIGH_CONFIDENCE_FRACTION_FOR_DECISION,
        "max_signal_only_low_confidence_fraction_for_decision": MAX_SIGNAL_ONLY_LOW_CONFIDENCE_FRACTION_FOR_DECISION,
        "max_signal_only_edge_extrapolation_median_for_decision": MAX_SIGNAL_ONLY_EDGE_EXTRAPOLATION_MEDIAN_FOR_DECISION,
        "max_signal_only_large_anchor_gap_fraction_for_decision": MAX_SIGNAL_ONLY_LARGE_ANCHOR_GAP_FRACTION_FOR_DECISION,
        "max_signal_only_few_anchor_fraction_for_decision": MAX_SIGNAL_ONLY_FEW_ANCHOR_FRACTION_FOR_DECISION,
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _legacy_features_hash(phasic_out: Path) -> str:
    path = phasic_out / "features" / "features.csv"
    return _file_sha256(path) if path.exists() else ""


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in CSV_FIELDS:
                value = row.get(key, "")
                if isinstance(value, bool):
                    value = str(value).lower()
                elif isinstance(value, (dict, list, tuple)):
                    value = json.dumps(_json_safe(value), sort_keys=True)
                elif isinstance(value, float) and not math.isfinite(value):
                    value = ""
                out[key] = value
            writer.writerow(out)


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise AppliedDffAutoStrategyCandidateAuditError(
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


def _path_equal_or_inside(path: Path, parent: Path) -> bool:
    path_resolved = path.resolve()
    parent_resolved = parent.resolve()
    return path_resolved == parent_resolved or _contains_path(parent_resolved, path_resolved)


def _assert_safe_output_dir(output_dir: Path, phasic_out: Path, source_cache: Path) -> None:
    output_resolved = output_dir.resolve()
    phasic_resolved = phasic_out.resolve()
    source_resolved = source_cache.resolve()
    legacy_features = phasic_resolved / "features" / "features.csv"
    legacy_features_dir = phasic_resolved / "features"
    message = "auto-strategy audit output_dir must be separate from source phasic_out and legacy features"
    if output_dir.exists() and not output_dir.is_dir():
        raise AppliedDffAutoStrategyCandidateAuditError(f"{message}: selected path is a file")
    if _path_equal_or_inside(output_resolved, phasic_resolved):
        raise AppliedDffAutoStrategyCandidateAuditError(message)
    if _path_equal_or_inside(output_resolved, legacy_features_dir):
        raise AppliedDffAutoStrategyCandidateAuditError(message)
    if output_resolved == source_resolved:
        raise AppliedDffAutoStrategyCandidateAuditError(f"{message}: selected path resolves to phasic_trace_cache.h5")
    if _contains_path(output_resolved, source_resolved):
        raise AppliedDffAutoStrategyCandidateAuditError(f"{message}: output_dir contains phasic_trace_cache.h5")
    if legacy_features.exists() and _contains_path(output_resolved, legacy_features):
        raise AppliedDffAutoStrategyCandidateAuditError(f"{message}: output_dir contains legacy features.csv")


def _list_rois(cache: h5py.File) -> list[str]:
    if "meta/rois" in cache:
        out = []
        for value in np.asarray(cache["meta/rois"][()]).reshape(-1).tolist():
            out.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
        return out
    if "roi" in cache:
        return sorted(str(x) for x in cache["roi"].keys())
    return []


def _list_chunk_ids(cache: h5py.File) -> list[int]:
    if "meta/chunk_ids" not in cache:
        return []
    return [int(x) for x in np.asarray(cache["meta/chunk_ids"][()]).reshape(-1).tolist()]


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y", "on"}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _flag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(";") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _counter_text(counter: Counter[str]) -> str:
    return "; ".join(f"{key}={counter[key]}" for key in sorted(counter) if key)


def _median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else float("nan")


def _fraction(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _dataset_stats(arr: np.ndarray) -> dict[str, Any]:
    values = np.asarray(arr, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return {
        "n_values": int(values.size),
        "n_nonfinite": int(values.size - finite.size),
        "finite_fraction": float(finite.size / values.size) if values.size else 0.0,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
    }


def _base_row(roi: str, cache_path: Path, source_hash: str, n_chunks: int) -> dict[str, Any]:
    return {
        "roi": roi,
        "auto_strategy_decision": "needs_review",
        "auto_strategy_confidence": "none",
        "auto_strategy_decision_status": "needs_review",
        "auto_strategy_warning_level": "caution",
        "auto_strategy_review_required": True,
        "auto_strategy_flags": "",
        "decision_rationale": "",
        "decision_blockers": "",
        "decision_cautions": "",
        "n_chunks": int(n_chunks),
        "n_chunks_with_required_dynamic_fit_data": 0,
        "n_chunks_with_required_signal_only_data": 0,
        "dynamic_fit_hard_flag_fraction": 0.0,
        "dynamic_fit_soft_flag_fraction": 0.0,
        "dynamic_fit_needs_inspection_fraction": 0.0,
        "dynamic_fit_negative_or_mixed_coupling_fraction": 0.0,
        "dynamic_fit_reference_flat_or_uninformative_fraction": 0.0,
        "dynamic_fit_reference_low_range_fraction": 0.0,
        "dynamic_fit_critical_slope_warning_fraction": 0.0,
        "dynamic_fit_signal_iso_corr_median": "",
        "dynamic_fit_signal_fitted_ref_corr_median": "",
        "dynamic_fit_slope_negative_fraction_median": "",
        "dynamic_fit_slope_clamped_fraction_median": "",
        "signal_only_f0_available_fraction": 0.0,
        "signal_only_f0_viable_fraction": 0.0,
        "signal_only_f0_contextual_fraction": 0.0,
        "signal_only_f0_hard_inspect_fraction": 0.0,
        "signal_only_f0_high_confidence_fraction": 0.0,
        "signal_only_f0_medium_confidence_fraction": 0.0,
        "signal_only_f0_low_confidence_fraction": 0.0,
        "signal_only_f0_edge_extrapolation_fraction_median": "",
        "signal_only_f0_large_anchor_gap_fraction": 0.0,
        "signal_only_f0_few_anchor_fraction": 0.0,
        "source_phasic_cache_path": str(cache_path),
        "source_phasic_cache_sha256": source_hash,
        "hdf5_modified_source_phasic_cache": False,
        "legacy_features_modified": False,
        "no_pipeline_execution": True,
        "no_feature_routing": True,
        "dynamic_fit_evidence": {},
        "signal_only_f0_evidence": {},
        "correction_reference_failure_evidence": {},
    }


def _audit_roi(cache: h5py.File, roi: str, chunk_ids: list[int], cache_path: Path, source_hash: str) -> dict[str, Any]:
    row = _base_row(roi, cache_path, source_hash, len(chunk_ids))
    blockers: list[str] = []
    cautions: list[str] = []
    flags: set[str] = set()
    dynamic_required = 0
    signal_required = 0
    missing_chunks: list[int] = []
    missing_required_datasets: list[str] = []
    length_mismatches: list[str] = []
    nonfinite_counts: Counter[str] = Counter()
    dataset_chunks: Counter[str] = Counter()
    optional_dataset_chunks: Counter[str] = Counter()

    dynamic_bool_counts = Counter()
    dynamic_severity = Counter()
    dynamic_flags = Counter()
    dynamic_soft_flags = Counter()
    dynamic_hard_flags = Counter()
    dynamic_slope_warnings = Counter()
    dynamic_numeric = {key: [] for key in DYNAMIC_NUMERIC_ATTRS}
    dynamic_missing_attrs = Counter()

    signal_bool_counts = Counter()
    signal_viability = Counter()
    signal_confidence = Counter()
    signal_flags = Counter()
    signal_anchor_status = Counter()
    signal_status = Counter()
    signal_numeric = {key: [] for key in SIGNAL_NUMERIC_ATTRS}
    signal_missing_attrs = Counter()

    for chunk_id in chunk_ids:
        path = f"roi/{roi}/chunk_{chunk_id}"
        if path not in cache:
            missing_chunks.append(int(chunk_id))
            continue
        grp = cache[path]
        arrays: dict[str, np.ndarray] = {}
        missing_for_chunk = []
        for name in REQUIRED_DATASETS:
            if name not in grp:
                missing_for_chunk.append(name)
                missing_required_datasets.append(f"chunk {chunk_id}: missing {name}")
            else:
                arr = np.asarray(grp[name][()])
                arrays[name] = arr
                dataset_chunks[name] += 1
                if name in NONFINITE_DATASETS:
                    stats = _dataset_stats(arr)
                    nonfinite_counts[name] += int(stats["n_nonfinite"])
        for name in OPTIONAL_DATASETS:
            if name in grp:
                optional_dataset_chunks[name] += 1
        if not missing_for_chunk:
            shapes = {name: np.asarray(arrays[name]).reshape(-1).shape for name in REQUIRED_DATASETS}
            if len(set(shapes.values())) == 1:
                dynamic_required += 1
                signal_required += 1
            else:
                length_mismatches.append(f"chunk {chunk_id}: required dataset length mismatch {shapes}")

        attrs = grp.attrs
        for key in DYNAMIC_BOOL_ATTRS:
            if key in attrs:
                if _as_bool(attrs.get(key)):
                    dynamic_bool_counts[key] += 1
            else:
                dynamic_missing_attrs[key] += 1
        for key in DYNAMIC_NUMERIC_ATTRS:
            if key in attrs:
                dynamic_numeric[key].append(_as_float(attrs.get(key)))
            else:
                dynamic_missing_attrs[key] += 1
        for key in DYNAMIC_STRING_ATTRS:
            if key not in attrs:
                dynamic_missing_attrs[key] += 1
                continue
            value = attrs.get(key)
            if key == "dynamic_fit_qc_severity":
                dynamic_severity[str(value or "")] += 1
            elif key == "dynamic_fit_qc_flags":
                dynamic_flags.update(_flag_list(value))
            elif key == "dynamic_fit_qc_soft_flags":
                dynamic_soft_flags.update(_flag_list(value))
            elif key == "dynamic_fit_qc_hard_flags":
                dynamic_hard_flags.update(_flag_list(value))
            elif key == "dynamic_fit_slope_warning_level":
                dynamic_slope_warnings[str(value or "")] += 1

        for key in SIGNAL_BOOL_ATTRS:
            if key in attrs:
                if _as_bool(attrs.get(key)):
                    signal_bool_counts[key] += 1
            else:
                signal_missing_attrs[key] += 1
        for key in SIGNAL_NUMERIC_ATTRS:
            if key in attrs:
                signal_numeric[key].append(_as_float(attrs.get(key)))
            else:
                signal_missing_attrs[key] += 1
        for key in SIGNAL_STRING_ATTRS:
            if key not in attrs:
                signal_missing_attrs[key] += 1
                continue
            value = attrs.get(key)
            if key == "signal_only_f0_candidate_viability":
                signal_viability[str(value or "").lower()] += 1
            elif key == "signal_only_f0_candidate_confidence":
                signal_confidence[str(value or "").lower()] += 1
            elif key == "signal_only_f0_flags":
                signal_flags.update(_flag_list(value))
            elif key == "signal_only_f0_anchor_status":
                signal_anchor_status[str(value or "")] += 1
            elif key == "signal_only_f0_status":
                signal_status[str(value or "")] += 1

    if missing_chunks:
        blockers.append(f"missing chunk groups: {missing_chunks[:20]}")
        flags.add("MISSING_CHUNK_GROUPS")
    if missing_required_datasets:
        blockers.append(f"missing required datasets: {len(missing_required_datasets)}")
        flags.add("MISSING_REQUIRED_DATASETS")
    if length_mismatches:
        blockers.append(f"trace length mismatches: {len(length_mismatches)}")
        flags.add("TRACE_LENGTH_MISMATCH")
    nonfinite_total = sum(nonfinite_counts.values())
    if nonfinite_total:
        blockers.append(f"non-finite required source values: {dict(nonfinite_counts)}")
        flags.add("NONFINITE_REQUIRED_SOURCE_VALUES")

    required_dynamic_missing = {
        key: value for key, value in dynamic_missing_attrs.items() if value == len(chunk_ids)
    }
    required_signal_missing = {
        key: value for key, value in signal_missing_attrs.items() if value == len(chunk_ids)
    }
    if required_dynamic_missing:
        blockers.append(f"missing dynamic-fit QC attributes: {sorted(required_dynamic_missing)}")
        flags.add("MISSING_DYNAMIC_FIT_QC_ATTRIBUTES")
    if required_signal_missing:
        blockers.append(f"missing signal-only F0 diagnostics: {sorted(required_signal_missing)}")
        flags.add("MISSING_SIGNAL_ONLY_F0_DIAGNOSTICS")

    n_chunks = len(chunk_ids)
    hard_fraction = _fraction(dynamic_bool_counts["dynamic_fit_qc_dynamic_fit_has_hard_flags"], n_chunks)
    soft_fraction = _fraction(dynamic_bool_counts["dynamic_fit_qc_dynamic_fit_has_soft_flags"], n_chunks)
    needs_inspection_fraction = _fraction(dynamic_bool_counts["dynamic_fit_qc_dynamic_fit_needs_inspection"], n_chunks)
    neg_mixed_fraction = _fraction(dynamic_bool_counts["dynamic_fit_qc_dynamic_fit_negative_or_mixed_coupling"], n_chunks)
    flat_fraction = _fraction(dynamic_bool_counts["dynamic_fit_qc_dynamic_fit_reference_flat_or_uninformative"], n_chunks)
    low_range_fraction = _fraction(dynamic_bool_counts["dynamic_fit_qc_dynamic_fit_reference_low_range"], n_chunks)
    critical_slope_fraction = _fraction(dynamic_slope_warnings["critical"], n_chunks)
    sig_iso_median = _median(dynamic_numeric["dynamic_fit_qc_signal_iso_corr"])
    sig_fit_median = _median(dynamic_numeric["dynamic_fit_qc_signal_fitted_ref_corr"])
    slope_negative_median = _median(dynamic_numeric["dynamic_fit_slope_slope_negative_fraction"])
    slope_clamped_median = _median(dynamic_numeric["dynamic_fit_slope_clamped_fraction"])

    available_fraction = _fraction(signal_bool_counts["signal_only_f0_candidate_available"], n_chunks)
    viable_fraction = _fraction(signal_viability["viable"], n_chunks)
    contextual_fraction = _fraction(signal_viability["contextual"], n_chunks)
    hard_inspect_fraction = _fraction(signal_viability["hard_inspect"], n_chunks)
    high_conf_fraction = _fraction(signal_confidence["high"], n_chunks)
    medium_conf_fraction = _fraction(signal_confidence["medium"], n_chunks)
    low_conf_fraction = _fraction(signal_confidence["low"], n_chunks)
    edge_extrap_median = _median(signal_numeric["signal_only_f0_edge_extrapolation_fraction"])
    large_anchor_gap_fraction = _fraction(signal_flags["SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP"], n_chunks)
    few_anchor_chunks = 0
    for chunk_id in chunk_ids:
        path = f"roi/{roi}/chunk_{chunk_id}"
        if path not in cache:
            continue
        chunk_flags = _flag_list(cache[path].attrs.get("signal_only_f0_flags"))
        if (
            "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_FEW_ANCHORS" in chunk_flags
            or "SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS" in chunk_flags
        ):
            few_anchor_chunks += 1
    few_anchor_fraction = _fraction(few_anchor_chunks, n_chunks)

    strong_reference_failure = (
        neg_mixed_fraction >= STRONG_REF_FAILURE_NEGATIVE_OR_MIXED_FRACTION
        or critical_slope_fraction >= STRONG_REF_FAILURE_CRITICAL_SLOPE_FRACTION
        or max(flat_fraction, low_range_fraction) >= STRONG_REF_FAILURE_LOW_OR_FLAT_FRACTION
        or (
            math.isfinite(sig_iso_median)
            and sig_iso_median < POOR_REF_COUPLING_CORR_MEDIAN
            and neg_mixed_fraction >= 0.05
        )
    )
    dynamic_clean = (
        dynamic_required == n_chunks
        and hard_fraction <= MAX_DYNAMIC_HARD_FLAG_FRACTION
        and neg_mixed_fraction <= MAX_DYNAMIC_NEGATIVE_OR_MIXED_FRACTION
        and max(flat_fraction, low_range_fraction) <= MAX_DYNAMIC_REFERENCE_LOW_OR_FLAT_FRACTION
        and critical_slope_fraction <= MAX_DYNAMIC_CRITICAL_SLOPE_WARNING_FRACTION
        and (
            (math.isfinite(sig_iso_median) and sig_iso_median >= MIN_DYNAMIC_SUPPORT_CORR_MEDIAN)
            or (math.isfinite(sig_fit_median) and sig_fit_median >= MIN_DYNAMIC_SUPPORT_CORR_MEDIAN)
        )
    )
    signal_rescue_viable = (
        signal_required == n_chunks
        and available_fraction >= MIN_SIGNAL_ONLY_AVAILABLE_FRACTION
        and (viable_fraction + contextual_fraction) >= MIN_SIGNAL_ONLY_VIABLE_OR_CONTEXTUAL_FRACTION
        and hard_inspect_fraction <= MAX_SIGNAL_ONLY_HARD_INSPECT_FRACTION
        and low_conf_fraction <= MAX_SIGNAL_ONLY_LOW_CONFIDENCE_FRACTION_FOR_RESCUE
        and large_anchor_gap_fraction <= MAX_SIGNAL_ONLY_LARGE_ANCHOR_GAP_FRACTION
        and few_anchor_fraction <= MAX_SIGNAL_ONLY_FEW_ANCHOR_FRACTION
    )
    signal_rescue_clean_for_decision = (
        signal_required == n_chunks
        and available_fraction >= MIN_SIGNAL_ONLY_AVAILABLE_FRACTION
        and viable_fraction >= MIN_SIGNAL_ONLY_VIABLE_FRACTION_FOR_DECISION
        and contextual_fraction <= MAX_SIGNAL_ONLY_CONTEXTUAL_FRACTION_FOR_DECISION
        and hard_inspect_fraction <= MAX_SIGNAL_ONLY_HARD_INSPECT_FRACTION_FOR_DECISION
        and high_conf_fraction >= MIN_SIGNAL_ONLY_HIGH_CONFIDENCE_FRACTION_FOR_DECISION
        and low_conf_fraction <= MAX_SIGNAL_ONLY_LOW_CONFIDENCE_FRACTION_FOR_DECISION
        and (
            not math.isfinite(edge_extrap_median)
            or edge_extrap_median <= MAX_SIGNAL_ONLY_EDGE_EXTRAPOLATION_MEDIAN_FOR_DECISION
        )
        and large_anchor_gap_fraction <= MAX_SIGNAL_ONLY_LARGE_ANCHOR_GAP_FRACTION_FOR_DECISION
        and few_anchor_fraction <= MAX_SIGNAL_ONLY_FEW_ANCHOR_FRACTION_FOR_DECISION
    )

    if soft_fraction:
        cautions.append(f"dynamic-fit soft flag fraction {soft_fraction:.3f}")
    if needs_inspection_fraction:
        cautions.append(f"dynamic-fit needs-inspection fraction {needs_inspection_fraction:.3f}")
    if low_conf_fraction:
        cautions.append(f"signal-only F0 low-confidence fraction {low_conf_fraction:.3f}")
    if hard_inspect_fraction:
        cautions.append(f"signal-only F0 hard-inspect fraction {hard_inspect_fraction:.3f}")

    if blockers:
        decision = "needs_review"
        confidence = "none"
        status = "blocked"
        warning_level = "severe"
        rationale = "Required evidence is missing or invalid; refusing provisional strategy decision."
    elif dynamic_clean and not strong_reference_failure:
        decision = "dynamic_fit"
        confidence = "high" if not cautions else "medium"
        status = "decided"
        warning_level = "none" if not cautions else "caution"
        rationale = "Dynamic-fit QC is clean enough and no strong correction-reference failure pattern is present."
    elif strong_reference_failure and signal_rescue_clean_for_decision and not dynamic_clean:
        decision = "signal_only_f0"
        confidence = "high"
        status = "decided"
        warning_level = "caution"
        rationale = "Correction-reference failure evidence is strong and signal-only F0 rescue evidence is clean enough for a provisional high-confidence rescue label."
    else:
        decision = "needs_review"
        confidence = "low"
        status = "needs_review"
        warning_level = "caution"
        rationale = "Evidence is mixed, caution-heavy, missing a validated threshold, or outside conservative first-pass decision limits."
        if strong_reference_failure:
            flags.add("SIGNAL_ONLY_F0_RESCUE_CANDIDATE")
            if signal_rescue_viable:
                cautions.append(
                    "strong correction-reference failure evidence present; signal-only F0 may be appropriate after manual review but is too caution-heavy for a provisional signal_only_f0 label"
                )
            else:
                cautions.append("strong correction-reference failure evidence present")
        elif not dynamic_clean:
            flags.add("DYNAMIC_FIT_NOT_CLEAN")
            cautions.append("dynamic-fit evidence is not clean enough for automatic dynamic_fit")

    if strong_reference_failure:
        flags.add("CORRECTION_REFERENCE_FAILURE_EVIDENCE")
    if signal_rescue_viable:
        flags.add("SIGNAL_ONLY_F0_RESCUE_EVIDENCE_VIABLE")
    if signal_rescue_clean_for_decision:
        flags.add("SIGNAL_ONLY_F0_RESCUE_EVIDENCE_CLEAN")
    if dynamic_clean:
        flags.add("DYNAMIC_FIT_EVIDENCE_CLEAN")

    dynamic_evidence = {
        "n_chunks_with_required_dynamic_fit_data": dynamic_required,
        "dataset_chunks": dict(dataset_chunks),
        "optional_dataset_chunks": dict(optional_dataset_chunks),
        "nonfinite_counts": dict(nonfinite_counts),
        "missing_chunks": missing_chunks,
        "missing_required_datasets": missing_required_datasets[:50],
        "length_mismatches": length_mismatches[:50],
        "bool_counts": dict(dynamic_bool_counts),
        "severity_counts": dict(dynamic_severity),
        "flag_counts": dict(dynamic_flags),
        "soft_flag_counts": dict(dynamic_soft_flags),
        "hard_flag_counts": dict(dynamic_hard_flags),
        "slope_warning_counts": dict(dynamic_slope_warnings),
        "missing_attr_counts": dict(dynamic_missing_attrs),
        "numeric_medians": {key: _median(values) for key, values in dynamic_numeric.items()},
        "dynamic_clean": bool(dynamic_clean),
    }
    signal_evidence = {
        "n_chunks_with_required_signal_only_data": signal_required,
        "bool_counts": dict(signal_bool_counts),
        "viability_counts": dict(signal_viability),
        "confidence_counts": dict(signal_confidence),
        "flag_counts": dict(signal_flags),
        "anchor_status_counts": dict(signal_anchor_status),
        "status_counts": dict(signal_status),
        "missing_attr_counts": dict(signal_missing_attrs),
        "numeric_medians": {key: _median(values) for key, values in signal_numeric.items()},
        "signal_rescue_viable": bool(signal_rescue_viable),
        "signal_rescue_clean_for_decision": bool(signal_rescue_clean_for_decision),
    }
    correction_failure = {
        "strong_reference_failure": bool(strong_reference_failure),
        "negative_or_mixed_coupling_fraction": neg_mixed_fraction,
        "critical_slope_warning_fraction": critical_slope_fraction,
        "reference_flat_or_uninformative_fraction": flat_fraction,
        "reference_low_range_fraction": low_range_fraction,
        "signal_iso_corr_median": sig_iso_median,
        "signal_fitted_ref_corr_median": sig_fit_median,
    }

    row.update(
        {
            "auto_strategy_decision": decision,
            "auto_strategy_confidence": confidence,
            "auto_strategy_decision_status": status,
            "auto_strategy_warning_level": warning_level,
            "auto_strategy_review_required": decision == "needs_review" or warning_level != "none",
            "auto_strategy_flags": ";".join(sorted(flags)),
            "decision_rationale": rationale,
            "decision_blockers": "; ".join(blockers),
            "decision_cautions": "; ".join(cautions),
            "n_chunks_with_required_dynamic_fit_data": dynamic_required,
            "n_chunks_with_required_signal_only_data": signal_required,
            "dynamic_fit_hard_flag_fraction": hard_fraction,
            "dynamic_fit_soft_flag_fraction": soft_fraction,
            "dynamic_fit_needs_inspection_fraction": needs_inspection_fraction,
            "dynamic_fit_negative_or_mixed_coupling_fraction": neg_mixed_fraction,
            "dynamic_fit_reference_flat_or_uninformative_fraction": flat_fraction,
            "dynamic_fit_reference_low_range_fraction": low_range_fraction,
            "dynamic_fit_critical_slope_warning_fraction": critical_slope_fraction,
            "dynamic_fit_signal_iso_corr_median": sig_iso_median,
            "dynamic_fit_signal_fitted_ref_corr_median": sig_fit_median,
            "dynamic_fit_slope_negative_fraction_median": slope_negative_median,
            "dynamic_fit_slope_clamped_fraction_median": slope_clamped_median,
            "signal_only_f0_available_fraction": available_fraction,
            "signal_only_f0_viable_fraction": viable_fraction,
            "signal_only_f0_contextual_fraction": contextual_fraction,
            "signal_only_f0_hard_inspect_fraction": hard_inspect_fraction,
            "signal_only_f0_high_confidence_fraction": high_conf_fraction,
            "signal_only_f0_medium_confidence_fraction": medium_conf_fraction,
            "signal_only_f0_low_confidence_fraction": low_conf_fraction,
            "signal_only_f0_edge_extrapolation_fraction_median": edge_extrap_median,
            "signal_only_f0_large_anchor_gap_fraction": large_anchor_gap_fraction,
            "signal_only_f0_few_anchor_fraction": few_anchor_fraction,
            "dynamic_fit_evidence": dynamic_evidence,
            "signal_only_f0_evidence": signal_evidence,
            "correction_reference_failure_evidence": correction_failure,
        }
    )
    return row


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    forbidden = set()
    for row in rows:
        forbidden.update(FORBIDDEN_FIELD_NAMES.intersection(row))
        if row.get("auto_strategy_decision") not in DECISIONS:
            raise AppliedDffAutoStrategyCandidateAuditError(
                f"invalid provisional decision: {row.get('auto_strategy_decision')}"
            )
        if row.get("auto_strategy_confidence") not in CONFIDENCE_LEVELS:
            raise AppliedDffAutoStrategyCandidateAuditError(
                f"invalid confidence: {row.get('auto_strategy_confidence')}"
            )
        if row.get("auto_strategy_decision_status") not in DECISION_STATUSES:
            raise AppliedDffAutoStrategyCandidateAuditError(
                f"invalid decision status: {row.get('auto_strategy_decision_status')}"
            )
    if forbidden:
        raise AppliedDffAutoStrategyCandidateAuditError(
            f"forbidden strategy selection fields present: {sorted(forbidden)}"
        )


def audit_applied_dff_auto_strategy_candidates(
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
        raise AppliedDffAutoStrategyCandidateAuditError(f"missing source phasic cache: {cache_path}")
    rois: list[str] = []
    try:
        with h5py.File(cache_path, "r") as h5:
            rois = _list_rois(h5)
            if roi:
                rois = [roi] if roi in rois else []
    except Exception as exc:
        raise AppliedDffAutoStrategyCandidateAuditError(f"source cache unreadable: {exc}") from exc
    if dry_run:
        return {
            "dry_run": True,
            "phasic_out": str(phasic_path),
            "roi_filter": roi or "",
            "rois_that_would_be_audited": rois,
            "would_write_outputs": True,
            "output_paths": {name: str(output_path / name) for name in OUTPUT_FILES},
            "no_pipeline_execution": True,
            "no_feature_routing": True,
            "no_manifest_written": True,
        }
    if roi and not rois:
        raise AppliedDffAutoStrategyCandidateAuditError(f"requested ROI not found: {roi}")
    _assert_safe_output_dir(output_path, phasic_path, cache_path)
    _prepare_output(output_path, overwrite=overwrite)

    source_before = _file_sha256(cache_path)
    legacy_before = _legacy_features_hash(phasic_path)
    rows: list[dict[str, Any]] = []
    with h5py.File(cache_path, "r") as h5:
        chunk_ids = _list_chunk_ids(h5)
        if not chunk_ids:
            raise AppliedDffAutoStrategyCandidateAuditError("source cache has no chunk IDs")
        for roi_name in rois:
            rows.append(_audit_roi(h5, roi_name, chunk_ids, cache_path, source_before))
    _validate_rows(rows)
    source_after = _file_sha256(cache_path)
    legacy_after = _legacy_features_hash(phasic_path)
    hdf5_modified = source_before != source_after
    legacy_modified = legacy_before != legacy_after
    for row in rows:
        row["hdf5_modified_source_phasic_cache"] = hdf5_modified
        row["legacy_features_modified"] = legacy_modified
    if hdf5_modified or legacy_modified:
        raise AppliedDffAutoStrategyCandidateAuditError("read-only guarantee failed during auto audit")

    audit_csv = output_path / OUTPUT_FILES[0]
    audit_json = output_path / OUTPUT_FILES[1]
    summary_json = output_path / OUTPUT_FILES[2]
    provenance_json = output_path / OUTPUT_FILES[3]
    _write_csv(audit_csv, rows)
    _write_json(audit_json, {"schema_version": SCHEMA_VERSION, "rows": rows})
    decision_counts = Counter(str(row["auto_strategy_decision"]) for row in rows)
    confidence_counts = Counter(str(row["auto_strategy_confidence"]) for row in rows)
    summary = {
        "audit_passed": True,
        "tool_name": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "decision_contract_version": DECISION_CONTRACT_VERSION,
        "phasic_out": str(phasic_path),
        "roi_filter": roi or "",
        "n_rois": len(rois),
        "decision_counts": dict(decision_counts),
        "confidence_counts": dict(confidence_counts),
        "n_review_required": sum(1 for row in rows if bool(row.get("auto_strategy_review_required"))),
        "source_hash_unchanged": source_before == source_after,
        "legacy_features_unchanged": legacy_before == legacy_after,
        "hdf5_modified_source_phasic_cache": hdf5_modified,
        "legacy_features_modified": legacy_modified,
        "thresholds": _thresholds(),
        "global_warnings": [],
        "output_dir": str(output_path),
        "audit_csv": str(audit_csv),
        "audit_json": str(audit_json),
        "summary_json": str(summary_json),
        "provenance_json": str(provenance_json),
        "failure_messages": "",
    }
    if fail_on_warning and summary["n_review_required"]:
        summary["audit_passed"] = False
        summary["failure_messages"] = "review-required provisional decisions present with fail_on_warning"
    provenance = {
        "tool_name": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "decision_contract_version": DECISION_CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phasic_out": str(phasic_path),
        "source_phasic_cache_path": str(cache_path),
        "source_phasic_cache_sha256_before": source_before,
        "source_phasic_cache_sha256_after": source_after,
        "hdf5_modified_source_phasic_cache": hdf5_modified,
        "legacy_features_path": str(phasic_path / "features" / "features.csv")
        if (phasic_path / "features" / "features.csv").exists()
        else "",
        "legacy_features_sha256_before": legacy_before,
        "legacy_features_sha256_after": legacy_after,
        "legacy_features_modified": legacy_modified,
        "output_dir": str(output_path),
        "argv": sys.argv[:],
        "thresholds": _thresholds(),
        "audit_mode": AUDIT_MODE,
        "no_pipeline_execution": True,
        "no_feature_routing": True,
        "no_manifest_written": True,
        "no_production_strategy_execution": True,
    }
    forbidden = (
        FORBIDDEN_FIELD_NAMES.intersection(summary)
        .union(FORBIDDEN_FIELD_NAMES.intersection(provenance))
    )
    if forbidden:
        raise AppliedDffAutoStrategyCandidateAuditError(
            f"forbidden strategy selection fields present: {sorted(forbidden)}"
        )
    _write_json(summary_json, summary)
    _write_json(provenance_json, provenance)
    if not summary["audit_passed"]:
        raise AppliedDffAutoStrategyCandidateAuditError(summary["failure_messages"])
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
    print(f"audit_passed: {summary.get('audit_passed')}")
    print(f"n_rois: {summary.get('n_rois')}")
    print(f"decision_counts: {summary.get('decision_counts')}")
    print(f"confidence_counts: {summary.get('confidence_counts')}")
    print(f"n_review_required: {summary.get('n_review_required')}")
    print(f"hdf5_modified_source_phasic_cache: {summary.get('hdf5_modified_source_phasic_cache')}")
    print(f"legacy_features_modified: {summary.get('legacy_features_modified')}")
    print(f"audit_csv: {summary.get('audit_csv')}")
    print(f"summary_json: {summary.get('summary_json')}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only provisional audit of future applied_dff auto-strategy candidates."
    )
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
        report = audit_applied_dff_auto_strategy_candidates(
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
