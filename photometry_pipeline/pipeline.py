
import os
import glob
import json
import yaml
import logging
import pathlib
import time
import pandas as pd
import numpy as np
from typing import List, Optional

from .config import Config
from .core.types import Chunk, SessionStats
from .io.adapters import (
    load_chunk,
    sniff_format,
    sort_npm_files,
    plan_continuous_windows_for_source,
    iter_continuous_custom_tabular_chunks,
    resolve_continuous_source_metadata,
)
from .core import preprocessing, regression, normalization, feature_extraction, baseline
from .core.baseline_reference_candidate import (
    compute_baseline_reference_candidate,
    compute_baseline_reference_candidate_metrics,
)
from .core.dynamic_fit_qc import compute_dynamic_fit_validity_metrics
from .core.reference_candidate_comparison import classify_reference_candidates
from .core.utils import natural_sort_key
from .core.reporting import generate_run_report, append_run_report_warnings
# from .viz import plots # Moved to run() to avoid side effects

TONIC_GLOBAL_FIT_SAMPLE_CAPACITY = 200_000

class _PairedDeterministicReservoir:
    """Bounded deterministic sampler for paired UV/SIG samples used by tonic fit."""

    def __init__(self, seed: int, capacity: int = 200_000):
        self.seed = int(seed)
        self.capacity = int(capacity)
        self.buffer = {}
        self.count = {}
        self._rng = np.random.default_rng(self.seed)

    def _ensure_channel(self, channel: str) -> None:
        if channel in self.buffer:
            return
        self.buffer[channel] = {
            "uv": np.zeros(self.capacity, dtype=np.float64),
            "sig": np.zeros(self.capacity, dtype=np.float64),
        }
        self.count[channel] = 0

    def add(self, channel: str, uv: np.ndarray, sig: np.ndarray) -> None:
        self._ensure_channel(channel)
        uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1)
        sig_arr = np.asarray(sig, dtype=np.float64).reshape(-1)
        if uv_arr.shape != sig_arr.shape:
            raise ValueError(f"Paired reservoir shape mismatch for {channel}: {uv_arr.shape} vs {sig_arr.shape}")

        valid = np.isfinite(uv_arr) & np.isfinite(sig_arr)
        if not np.any(valid):
            return
        u = uv_arr[valid]
        s = sig_arr[valid]
        n = int(u.size)
        current = int(self.count[channel])

        if current < self.capacity:
            take = min(n, self.capacity - current)
            if take > 0:
                self.buffer[channel]["uv"][current: current + take] = u[:take]
                self.buffer[channel]["sig"][current: current + take] = s[:take]
                self.count[channel] = current + take
                current = int(self.count[channel])
            if take < n:
                self._update_existing(channel, u[take:], s[take:], total_seen=current)
        else:
            self._update_existing(channel, u, s, total_seen=current)

    def _update_existing(self, channel: str, uv: np.ndarray, sig: np.ndarray, *, total_seen: int) -> None:
        n_new = int(uv.size)
        if n_new <= 0:
            return
        probs = self._rng.random(n_new)
        denominators = np.arange(total_seen + 1, total_seen + n_new + 1)
        mask = probs < (self.capacity / denominators)
        n_replace = int(np.sum(mask))
        if n_replace > 0:
            replace_indices = self._rng.integers(0, self.capacity, size=n_replace)
            self.buffer[channel]["uv"][replace_indices] = uv[mask]
            self.buffer[channel]["sig"][replace_indices] = sig[mask]
        self.count[channel] = total_seen + n_new

    def channels(self) -> List[str]:
        return sorted(self.buffer.keys())

    def arrays(self, channel: str) -> tuple[np.ndarray, np.ndarray]:
        n_valid = min(int(self.count.get(channel, 0)), self.capacity)
        if n_valid <= 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        return (
            self.buffer[channel]["uv"][:n_valid].copy(),
            self.buffer[channel]["sig"][:n_valid].copy(),
        )

def _sanitize_metadata(obj):
    """
    Recursively convert metadata to JSON-safe primitives.
    Handles numpy types explicitly so they become Python scalars/lists.
    Unknown types are converted to repr(obj).
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize_metadata(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_metadata(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _sanitize_strict_json(obj):
    """Recursively convert metadata to JSON-safe primitives with no NaN/Infinity."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_strict_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_strict_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_strict_json(obj.tolist())
    if isinstance(obj, np.generic):
        return _sanitize_strict_json(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return repr(obj)


def _append_run_report_section(output_dir: str, section: str, payload) -> None:
    """Best-effort run_report.json updater for analysis provenance generated after Pass 1."""
    path = os.path.join(output_dir, "run_report.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        derived = data.setdefault("derived_settings", {})
        derived[section] = _sanitize_metadata(payload)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        logging.warning("Failed to append %s to run_report.json", section, exc_info=True)

class Pipeline:
    def __init__(self, config: Config, mode: str = 'phasic'):
        self.config = config
        self.mode = mode
        self.file_list = []
        self._continuous_window_map = {}
        self._continuous_source_cache = {}
        self._continuous_plan_summary = None
        self.stats = SessionStats()
        self.stats.tonic_fit_params = {} # ROI -> {slope, intercept} (Ad-hoc extension)
        self.stats.tonic_global_fit_provenance = {}
        self.qc_summary = {
            'failed_chunks': [],
            'chunk_fail_fraction': 0.0,
            'roi_failures': {}
        }
        self.roi_map = {}
        self._pass1_manifest = []
        self._selected_rois = None
        self.roi_selection = None
        self.traces_only = False
        self._phasic_started_at = None
        self._phasic_phase_buckets = {}
        self._phasic_detail_buckets = {}
        self._phasic_metrics = {}
        self.dynamic_fit_slope_records = []
        self.dynamic_fit_slope_warning_records = []
        self.dynamic_fit_slope_constraint_records = []
        self.dynamic_fit_qc_records = []
        self.baseline_reference_candidate_records = []
        self.dynamic_fit_slope_warning_summary = {
            "roi_chunk_fits_with_any_negative_slope": 0,
            "roi_chunk_fits_by_warning_level": {
                "low": 0,
                "moderate": 0,
                "high": 0,
                "critical": 0,
            },
            "roi_chunk_fits_with_moderate_high_critical_warnings": 0,
            "max_slope_negative_fraction": 0.0,
            "min_slope_min_observed": None,
            "dynamic_fit_modes_affected": [],
            "rois_affected": [],
        }
        self.dynamic_fit_slope_constraint_summary = {
            "roi_chunk_fits_with_slope_constraint_applied": 0,
            "roi_chunk_fits_by_constraint_mode": {},
            "roi_chunk_fits_with_any_clamped_slope": 0,
            "max_slope_clamped_fraction": 0.0,
            "max_unconstrained_slope_negative_fraction_among_clamped": 0.0,
            "dynamic_fit_modes_with_slope_constraint_applied": [],
            "rois_with_slope_constraint_applied": [],
        }
        self._continuous_csv_reading = {
            "sequential_csv_reading_used": False,
            "source_csv_open_read_passes": 0,
            "windows_yielded_sequentially": 0,
            "bounded_loader_fallback_count": 0,
            "phases": {},
        }

    def _dynamic_fit_roi_metadata(self, chunk: Chunk, fit_mode: str, roi: str) -> dict:
        if not hasattr(chunk, "metadata") or not isinstance(chunk.metadata, dict):
            return {}
        if fit_mode == "global_linear_regression":
            by_roi = chunk.metadata.get("dynamic_fit_global_linear", {})
        elif fit_mode == "robust_global_event_reject":
            by_roi = chunk.metadata.get("dynamic_fit_event_reject", {})
        elif fit_mode == "adaptive_event_gated_regression":
            by_roi = chunk.metadata.get("dynamic_fit_adaptive_event_gated", {})
        else:
            by_roi = chunk.metadata.get("dynamic_fit_rolling_local", {})
        if not isinstance(by_roi, dict):
            return {}
        payload = by_roi.get(str(roi), {})
        return payload if isinstance(payload, dict) else {}

    def _record_dynamic_fit_validity_metrics(self, chunk: Chunk, chunk_id: int, source_file: str) -> None:
        if self.mode == "tonic" or chunk.uv_fit is None:
            return
        if not hasattr(chunk, "metadata") or not isinstance(chunk.metadata, dict):
            return
        fit_mode = str(chunk.metadata.get("dynamic_fit_mode_resolved", "") or "")
        slope_constraint = str(getattr(self.config, "dynamic_fit_slope_constraint", "unconstrained"))
        min_slope = float(getattr(self.config, "dynamic_fit_min_slope", 0.0))
        acquisition_mode = str(getattr(self.config, "acquisition_mode", "intermittent"))
        roi_metrics_by_name: dict[str, dict] = {}

        for r_idx, roi in enumerate(chunk.channel_names):
            roi_name = str(roi)
            payload = self._dynamic_fit_roi_metadata(chunk, fit_mode, roi_name)
            slope = payload.get("coef_slope")
            if slope is None:
                final_coef = payload.get("final_coef", {}) if isinstance(payload, dict) else {}
                if isinstance(final_coef, dict) and "slope" in final_coef:
                    slope = final_coef.get("slope")
            slope_unconstrained = payload.get("coef_slope_unconstrained")
            metrics = compute_dynamic_fit_validity_metrics(
                signal=chunk.sig_raw[:, r_idx],
                iso=chunk.uv_raw[:, r_idx],
                fitted_ref=chunk.uv_fit[:, r_idx],
                sample_rate_hz=float(chunk.fs_hz),
                slope=slope,
                local_slope_unconstrained=slope_unconstrained,
                local_slope_final=payload.get("coef_slope"),
                fit_mode=fit_mode,
                slope_constraint=slope_constraint,
                min_slope=min_slope,
            )
            record = {
                "roi": roi_name,
                "chunk_id": int(chunk_id),
                "source_file": str(source_file),
                "dynamic_fit_mode": fit_mode,
                "slope_constraint": slope_constraint,
                "acquisition_mode": acquisition_mode,
                **_sanitize_metadata(metrics),
            }
            self.dynamic_fit_qc_records.append(record)
            roi_metrics_by_name[roi_name] = record

        if roi_metrics_by_name:
            chunk.metadata.setdefault("dynamic_fit_validity_qc", {}).update(roi_metrics_by_name)

    def _record_baseline_reference_candidate_metrics(self, chunk: Chunk, chunk_id: int, source_file: str) -> None:
        if self.mode == "tonic" or chunk.uv_fit is None:
            return
        if not bool(getattr(self.config, "baseline_reference_candidate_enabled", True)):
            return
        if not hasattr(chunk, "metadata") or not isinstance(chunk.metadata, dict):
            return
        fit_mode = str(chunk.metadata.get("dynamic_fit_mode_resolved", "") or "")
        slope_constraint = str(getattr(self.config, "dynamic_fit_slope_constraint", "unconstrained"))
        acquisition_mode = str(getattr(self.config, "acquisition_mode", "intermittent"))
        dynamic_qc_by_roi = chunk.metadata.get("dynamic_fit_validity_qc", {})
        if not isinstance(dynamic_qc_by_roi, dict):
            dynamic_qc_by_roi = {}

        records_by_roi: dict[str, dict] = {}
        for r_idx, roi in enumerate(chunk.channel_names):
            roi_name = str(roi)
            dynamic_qc = dynamic_qc_by_roi.get(roi_name, {})
            if not isinstance(dynamic_qc, dict):
                dynamic_qc = {}
            candidate = compute_baseline_reference_candidate(
                signal=chunk.sig_raw[:, r_idx],
                reference=chunk.uv_raw[:, r_idx],
                fs=float(chunk.fs_hz),
                smoothing_window_sec=float(
                    getattr(self.config, "baseline_reference_smoothing_window_sec", 300.0)
                ),
                default_smoothing_window_sec=300.0,
                min_smoothing_window_sec=float(
                    getattr(self.config, "baseline_reference_min_smoothing_window_sec", 60.0)
                ),
                max_window_fraction_of_chunk=float(
                    getattr(self.config, "baseline_reference_max_window_fraction_of_chunk", 0.75)
                ),
                large_window_fraction_warning=float(
                    getattr(self.config, "baseline_reference_large_window_fraction_warning", 0.50)
                ),
            )
            candidate_trace = candidate.get("baseline_ref_candidate")
            candidate_meta = {
                key: value
                for key, value in candidate.items()
                if key != "baseline_ref_candidate"
            }
            metrics: dict = {}
            if candidate.get("baseline_ref_candidate_available") and candidate_trace is not None:
                try:
                    metrics = compute_baseline_reference_candidate_metrics(
                        signal=chunk.sig_raw[:, r_idx],
                        reference=chunk.uv_raw[:, r_idx],
                        dynamic_fitted_ref=chunk.uv_fit[:, r_idx],
                        baseline_candidate=np.asarray(candidate_trace, dtype=float),
                        fs=float(chunk.fs_hz),
                    )
                except Exception as exc:
                    candidate_meta["baseline_ref_candidate_available"] = False
                    candidate_meta["baseline_ref_warning"] = f"metric_failure:{exc}"
                    metrics = {}
            record = {
                "roi": roi_name,
                "chunk_id": int(chunk_id),
                "source_file": str(source_file),
                "recording_mode": acquisition_mode,
                "dynamic_fit_mode": fit_mode,
                "slope_constraint": slope_constraint,
                **candidate_meta,
                **metrics,
                "dynamic_fit_qc_severity": dynamic_qc.get("dynamic_fit_qc_severity", ""),
                "dynamic_fit_qc_hard_flags": dynamic_qc.get("dynamic_fit_qc_hard_flags", []),
                "dynamic_fit_qc_soft_flags": dynamic_qc.get("dynamic_fit_qc_soft_flags", []),
                "dynamic_fit_qc_flags": dynamic_qc.get("dynamic_fit_qc_flags", []),
            }
            record.update(
                classify_reference_candidates(
                    dynamic_qc=dynamic_qc,
                    baseline_record=record,
                )
            )
            clean_record = _sanitize_metadata(record)
            self.baseline_reference_candidate_records.append(clean_record)
            records_by_roi[roi_name] = clean_record
            if candidate.get("baseline_ref_candidate_available") and candidate_trace is not None:
                trace_arr = np.asarray(candidate_trace, dtype=float).reshape(-1)
                if trace_arr.shape == chunk.sig_raw[:, r_idx].shape and np.any(np.isfinite(trace_arr)):
                    chunk.metadata.setdefault("baseline_reference_candidate_trace", {})[
                        roi_name
                    ] = trace_arr

        if records_by_roi:
            chunk.metadata.setdefault("baseline_reference_candidate_qc", {}).update(records_by_roi)

    def _update_dynamic_fit_qc_summary(self) -> None:
        records = list(self.dynamic_fit_qc_records)
        flag_counts: dict[str, int] = {}
        severity_counts = {"ok": 0, "context": 0, "inspect": 0}
        hard_flag_fit_count = 0
        context_only_fit_count = 0
        for rec in records:
            flags = rec.get("dynamic_fit_qc_flags", [])
            if isinstance(flags, str):
                flags = [x for x in flags.split(";") if x]
            if isinstance(flags, (list, tuple)):
                for flag in flags:
                    flag_s = str(flag)
                    if flag_s:
                        flag_counts[flag_s] = flag_counts.get(flag_s, 0) + 1
            hard_flags = rec.get("dynamic_fit_qc_hard_flags", [])
            soft_flags = rec.get("dynamic_fit_qc_soft_flags", [])
            if isinstance(hard_flags, str):
                hard_flags = [x for x in hard_flags.split(";") if x]
            if isinstance(soft_flags, str):
                soft_flags = [x for x in soft_flags.split(";") if x]
            has_hard = isinstance(hard_flags, (list, tuple)) and bool(hard_flags)
            has_soft = isinstance(soft_flags, (list, tuple)) and bool(soft_flags)
            severity = str(rec.get("dynamic_fit_qc_severity", "") or "").strip().lower()
            if severity not in severity_counts:
                severity = "inspect" if has_hard else ("context" if has_soft else "ok")
            severity_counts[severity] += 1
            if has_hard:
                hard_flag_fit_count += 1
            elif has_soft:
                context_only_fit_count += 1
        summary = {
            "roi_chunk_fit_count": int(len(records)),
            "roi_chunk_fits_needing_inspection": int(
                sum(1 for rec in records if bool(rec.get("dynamic_fit_needs_inspection", False)))
            ),
            "flag_counts": {k: int(v) for k, v in sorted(flag_counts.items())},
            "severity_counts": {k: int(v) for k, v in sorted(severity_counts.items())},
            "roi_chunk_fits_with_hard_flags": int(hard_flag_fit_count),
            "roi_chunk_fits_with_context_only_flags": int(context_only_fit_count),
        }
        self.qc_summary["dynamic_fit_validity_qc_summary"] = _sanitize_metadata(summary)

    def _update_baseline_reference_candidate_summary(self) -> None:
        records = list(self.baseline_reference_candidate_records)

        def _finite_values(key: str) -> list[float]:
            out = []
            for rec in records:
                try:
                    val = float(rec.get(key, float("nan")))
                except Exception:
                    continue
                if np.isfinite(val):
                    out.append(val)
            return out

        def _quartiles(key: str) -> dict[str, float | None]:
            vals = np.asarray(_finite_values(key), dtype=float)
            if vals.size == 0:
                return {"median": None, "p25": None, "p75": None}
            return {
                "median": float(np.percentile(vals, 50.0)),
                "p25": float(np.percentile(vals, 25.0)),
                "p75": float(np.percentile(vals, 75.0)),
            }

        available = [
            rec for rec in records if bool(rec.get("baseline_ref_candidate_available", False))
        ]
        summary = {
            "roi_chunk_candidate_count": int(len(records)),
            "roi_chunk_candidate_available_count": int(len(available)),
            "roi_chunk_candidate_unavailable_count": int(len(records) - len(available)),
            "baseline_ref_response_scale_rich_count": int(
                sum(1 for rec in records if bool(rec.get("baseline_ref_response_scale_rich", False)))
            ),
            "baseline_ref_low_range_count": int(
                sum(1 for rec in records if bool(rec.get("baseline_ref_low_range", False)))
            ),
            "baseline_ref_flat_or_uninformative_count": int(
                sum(1 for rec in records if bool(rec.get("baseline_ref_flat_or_uninformative", False)))
            ),
            "baseline_ref_response_scale_fraction": _quartiles("baseline_ref_response_scale_fraction"),
            "baseline_ref_baseline_scale_fraction": _quartiles("baseline_ref_baseline_scale_fraction"),
            "dynamic_minus_baseline_ref_rms": _quartiles("dynamic_minus_baseline_ref_rms"),
            "dynamic_minus_baseline_ref_range": _quartiles("dynamic_minus_baseline_ref_range"),
        }
        self.qc_summary["baseline_reference_candidate_qc_summary"] = _sanitize_metadata(summary)

    def _update_reference_candidate_comparison_summary(self) -> None:
        records = list(self.baseline_reference_candidate_records)

        def _count_values(key: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for rec in records:
                val = str(rec.get(key, "") or "").strip()
                if val:
                    counts[val] = counts.get(val, 0) + 1
            return {k: int(v) for k, v in sorted(counts.items())}

        flag_counts: dict[str, int] = {}
        for rec in records:
            flags = rec.get("reference_comparison_flags", [])
            if isinstance(flags, str):
                flags = [x for x in flags.split(";") if x]
            if isinstance(flags, (list, tuple)):
                for flag in flags:
                    flag_s = str(flag).strip()
                    if flag_s:
                        flag_counts[flag_s] = flag_counts.get(flag_s, 0) + 1

        summary = {
            "roi_chunk_comparison_count": int(len(records)),
            "reference_comparison_class_counts": _count_values("reference_comparison_class"),
            "dynamic_reference_viability_counts": _count_values("dynamic_reference_viability"),
            "baseline_reference_viability_counts": _count_values("baseline_reference_viability"),
            "reference_comparison_review_level_counts": _count_values(
                "reference_comparison_review_level"
            ),
            "reference_comparison_flag_counts": {
                k: int(v) for k, v in sorted(flag_counts.items())
            },
        }
        self.qc_summary["reference_candidate_comparison_summary"] = _sanitize_metadata(summary)

    def _record_dynamic_fit_slope_summaries(self, chunk: Chunk, chunk_id: int, source_file: str) -> None:
        if self.mode == "tonic" or not hasattr(chunk, "metadata") or not isinstance(chunk.metadata, dict):
            return
        fit_mode = str(chunk.metadata.get("dynamic_fit_mode_resolved", "") or "")
        if fit_mode == "global_linear_regression":
            by_roi = chunk.metadata.get("dynamic_fit_global_linear", {})
        elif fit_mode == "robust_global_event_reject":
            by_roi = chunk.metadata.get("dynamic_fit_event_reject", {})
        elif fit_mode == "adaptive_event_gated_regression":
            by_roi = chunk.metadata.get("dynamic_fit_adaptive_event_gated", {})
        else:
            by_roi = chunk.metadata.get("dynamic_fit_rolling_local", {})
        if not isinstance(by_roi, dict):
            return

        acquisition_mode = str(getattr(self.config, "acquisition_mode", "intermittent"))
        for roi, payload in by_roi.items():
            if not isinstance(payload, dict):
                continue
            summary = payload.get("slope_summary", {})
            if not isinstance(summary, dict) or not summary:
                continue
            record = {
                "roi": str(roi),
                "chunk_id": int(chunk_id),
                "source_file": str(source_file),
                "dynamic_fit_mode": fit_mode,
                "acquisition_mode": acquisition_mode,
                **_sanitize_metadata(summary),
            }
            self.dynamic_fit_slope_records.append(record)

            level = str(summary.get("warning_level", "none"))
            if level != "none":
                warning_record = dict(record)
                self.dynamic_fit_slope_warning_records.append(warning_record)
                self.qc_summary.setdefault("dynamic_fit_slope_warnings", []).append(warning_record)

            constraint_summary = payload.get("slope_constraint_summary", {})
            if isinstance(constraint_summary, dict) and constraint_summary:
                constraint_record = {
                    "roi": str(roi),
                    "chunk_id": int(chunk_id),
                    "source_file": str(source_file),
                    "dynamic_fit_mode": fit_mode,
                    "acquisition_mode": acquisition_mode,
                    **_sanitize_metadata(constraint_summary),
                }
                if bool(constraint_summary.get("slope_constraint_applied", False)):
                    self.dynamic_fit_slope_constraint_records.append(constraint_record)
                    self.qc_summary.setdefault(
                        "dynamic_fit_slope_constraint_warnings", []
                    ).append(constraint_record)

        self._update_dynamic_fit_slope_warning_summary()
        self._update_dynamic_fit_slope_constraint_summary()

    def _update_dynamic_fit_slope_warning_summary(self) -> None:
        records = list(self.dynamic_fit_slope_warning_records)
        level_counts = {
            "low": 0,
            "moderate": 0,
            "high": 0,
            "critical": 0,
        }
        severe = {"moderate", "high", "critical"}
        max_neg = 0.0
        min_slope = None
        for rec in records:
            level = str(rec.get("warning_level", "none"))
            if level in level_counts:
                level_counts[level] += 1
            try:
                max_neg = max(max_neg, float(rec.get("slope_negative_fraction", 0.0)))
            except Exception:
                pass
            try:
                slope_min = float(rec.get("slope_min"))
            except Exception:
                continue
            if np.isfinite(slope_min):
                min_slope = slope_min if min_slope is None else min(min_slope, slope_min)
        summary = {
            "roi_chunk_fits_with_any_negative_slope": int(len(records)),
            "roi_chunk_fits_by_warning_level": {
                key: int(value) for key, value in level_counts.items()
            },
            "roi_chunk_fits_with_low_warnings": int(level_counts["low"]),
            "roi_chunk_fits_with_moderate_warnings": int(level_counts["moderate"]),
            "roi_chunk_fits_with_high_warnings": int(level_counts["high"]),
            "roi_chunk_fits_with_critical_warnings": int(level_counts["critical"]),
            "roi_chunk_fits_with_moderate_high_critical_warnings": int(
                sum(1 for rec in records if str(rec.get("warning_level", "none")) in severe)
            ),
            "max_slope_negative_fraction": float(max_neg),
            "min_slope_min_observed": min_slope,
            "dynamic_fit_modes_affected": sorted(
                {
                    str(rec.get("dynamic_fit_mode", ""))
                    for rec in records
                    if str(rec.get("dynamic_fit_mode", ""))
                }
            ),
            "rois_affected": sorted(
                {
                    str(rec.get("roi", ""))
                    for rec in records
                    if str(rec.get("roi", ""))
                }
            ),
        }
        self.dynamic_fit_slope_warning_summary = summary
        self.qc_summary["dynamic_fit_slope_warning_summary"] = _sanitize_metadata(summary)

    def _update_dynamic_fit_slope_constraint_summary(self) -> None:
        records = list(self.dynamic_fit_slope_constraint_records)
        mode_counts: dict[str, int] = {}
        max_clamped = 0.0
        max_unconstrained_neg = 0.0
        for rec in records:
            mode = str(rec.get("slope_constraint_mode", "unavailable"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            try:
                max_clamped = max(max_clamped, float(rec.get("slope_clamped_fraction", 0.0)))
            except Exception:
                pass
            unconstrained = rec.get("unconstrained_slope_summary", {})
            if isinstance(unconstrained, dict):
                try:
                    max_unconstrained_neg = max(
                        max_unconstrained_neg,
                        float(unconstrained.get("slope_negative_fraction", 0.0)),
                    )
                except Exception:
                    pass
        summary = {
            "roi_chunk_fits_with_slope_constraint_applied": int(len(records)),
            "roi_chunk_fits_by_constraint_mode": {
                str(key): int(value) for key, value in sorted(mode_counts.items())
            },
            "roi_chunk_fits_with_any_clamped_slope": int(
                sum(
                    1
                    for rec in records
                    if float(rec.get("n_clamped_slope_samples", 0) or 0) > 0
                )
            ),
            "max_slope_clamped_fraction": float(max_clamped),
            "max_unconstrained_slope_negative_fraction_among_clamped": float(
                max_unconstrained_neg
            ),
            "dynamic_fit_modes_with_slope_constraint_applied": sorted(
                {
                    str(rec.get("dynamic_fit_mode", ""))
                    for rec in records
                    if str(rec.get("dynamic_fit_mode", ""))
                }
            ),
            "rois_with_slope_constraint_applied": sorted(
                {
                    str(rec.get("roi", ""))
                    for rec in records
                    if str(rec.get("roi", ""))
                }
            ),
        }
        self.dynamic_fit_slope_constraint_summary = summary
        self.qc_summary["dynamic_fit_slope_constraint_summary"] = _sanitize_metadata(summary)

    def _is_phasic_timing_enabled(self) -> bool:
        return self.mode == 'phasic'

    def _is_continuous_mode_enabled(self) -> bool:
        return str(getattr(self.config, "acquisition_mode", "intermittent")).strip().lower() == "continuous"

    def _add_phasic_phase_bucket(self, bucket: str, elapsed_sec: float):
        if not self._is_phasic_timing_enabled():
            return
        self._phasic_phase_buckets[bucket] = self._phasic_phase_buckets.get(bucket, 0.0) + float(elapsed_sec)

    def _add_phasic_detail_bucket(self, bucket: str, elapsed_sec: float):
        if not self._is_phasic_timing_enabled():
            return
        self._phasic_detail_buckets[bucket] = self._phasic_detail_buckets.get(bucket, 0.0) + float(elapsed_sec)

    def _set_phasic_metric(self, name: str, value):
        if not self._is_phasic_timing_enabled():
            return
        self._phasic_metrics[name] = value

    def _add_phasic_metric(self, name: str, delta):
        if not self._is_phasic_timing_enabled():
            return
        self._phasic_metrics[name] = self._phasic_metrics.get(name, 0) + delta

    def _record_continuous_csv_reading(
        self,
        phase: str,
        *,
        sequential_passes: int = 0,
        windows_yielded: int = 0,
        fallback_windows: int = 0,
    ) -> None:
        if not self._is_continuous_mode_enabled():
            return
        phase_stats = self._continuous_csv_reading.setdefault("phases", {}).setdefault(
            str(phase),
            {
                "source_csv_open_read_passes": 0,
                "windows_yielded_sequentially": 0,
                "bounded_loader_fallback_count": 0,
            },
        )
        if sequential_passes:
            self._continuous_csv_reading["sequential_csv_reading_used"] = True
            self._continuous_csv_reading["source_csv_open_read_passes"] += int(sequential_passes)
            phase_stats["source_csv_open_read_passes"] += int(sequential_passes)
        if windows_yielded:
            self._continuous_csv_reading["windows_yielded_sequentially"] += int(windows_yielded)
            phase_stats["windows_yielded_sequentially"] += int(windows_yielded)
        if fallback_windows:
            self._continuous_csv_reading["bounded_loader_fallback_count"] += int(fallback_windows)
            phase_stats["bounded_loader_fallback_count"] += int(fallback_windows)

    def _emit_phasic_timing_details(self, total_elapsed_sec: float):
        if not self._is_phasic_timing_enabled():
            return

        for bucket in sorted(self._phasic_phase_buckets.keys()):
            elapsed = self._phasic_phase_buckets[bucket]
            print(f"TIMING DETAIL phase=phasic_analysis bucket={bucket} elapsed_sec={elapsed:.3f}", flush=True)

        for bucket in sorted(self._phasic_detail_buckets.keys()):
            elapsed = self._phasic_detail_buckets[bucket]
            print(f"TIMING DETAIL phase=phasic_analysis bucket={bucket} elapsed_sec={elapsed:.3f}", flush=True)

        phase_explicit_sum = float(sum(self._phasic_phase_buckets.values()))
        phase_remainder = max(0.0, float(total_elapsed_sec) - phase_explicit_sum)
        print(f"TIMING DETAIL phase=phasic_analysis bucket=phase.remainder elapsed_sec={phase_remainder:.3f}", flush=True)
        print(f"TIMING DETAIL phase=phasic_analysis bucket=phase.total elapsed_sec={float(total_elapsed_sec):.3f}", flush=True)

        for name in sorted(self._phasic_metrics.keys()):
            value = self._phasic_metrics[name]
            print(f"TIMING METRIC phase=phasic_analysis name={name} value={value}", flush=True)

    def discover_files(self, input_path: str, recursive: bool = False, file_glob: str = "*.csv", force_format: str = 'auto'):
        self._continuous_window_map = {}
        self._continuous_source_cache = {}
        self._continuous_plan_summary = None
        valid_formats = {"auto", "rwd", "npm", "custom_tabular"}
        if force_format not in valid_formats:
            raise ValueError(
                f"Unsupported format '{force_format}'. Allowed: {sorted(valid_formats)}"
            )
        if force_format == 'rwd':
            # RWD Discovery: Treat input_path as root containing timestamped subdirectories
            from .io.adapters import discover_rwd_chunks
            self.file_list = discover_rwd_chunks(input_path)
        elif os.path.isfile(input_path):
            self.file_list = [input_path]
        else:
            if recursive and force_format == 'auto':
                # In recursive auto mode, prefer canonical RWD targets first so
                # session artifact CSVs (events/outputs/unaligned) are not
                # treated as analyzable sessions.
                from .io.adapters import discover_rwd_chunks
                direct_fluorescence = os.path.join(input_path, "fluorescence.csv")
                if os.path.isfile(direct_fluorescence):
                    if sniff_format(direct_fluorescence, self.config) == 'rwd':
                        self.file_list = [direct_fluorescence]
                    else:
                        self.file_list = []
                else:
                    self.file_list = []

                if not self.file_list:
                    try:
                        self.file_list = discover_rwd_chunks(input_path)
                    except Exception:
                        self.file_list = []

            if recursive:
                if not self.file_list:
                    search_pattern = os.path.join(input_path, "**", file_glob)
                    self.file_list = glob.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(input_path, file_glob)
                self.file_list = glob.glob(search_pattern)
                if force_format == 'auto' and not self.file_list:
                    from .io.adapters import discover_csv_or_rwd_chunks
                    self.file_list = discover_csv_or_rwd_chunks(input_path, file_glob=file_glob)
        
        if not self.file_list:
            raise ValueError(f"No files found in {input_path}")

        resolved_for_order = force_format
        if resolved_for_order == 'auto' and self.file_list:
            sniffed = sniff_format(self.file_list[0], self.config)
            resolved_for_order = sniffed if sniffed is not None else 'auto'

        if resolved_for_order == 'npm':
            self.file_list = sort_npm_files(self.file_list)
        else:
            self.file_list.sort(key=natural_sort_key)
            
        print(f"Found {len(self.file_list)} files.")

        if self._is_continuous_mode_enabled():
            expanded_entries = []
            source_files = list(self.file_list)
            planned_window_total = 0
            partial_window_total = 0
            for src in source_files:
                fmt = self._get_format(src, force_format)
                if fmt == "npm":
                    raise ValueError(
                        "Continuous acquisition mode is not yet implemented for NPM/interleaved inputs."
                    )
                if fmt not in {"rwd", "custom_tabular"}:
                    raise ValueError(
                        f"Continuous acquisition mode is unsupported for format '{fmt}'."
                    )
                planned = plan_continuous_windows_for_source(
                    src,
                    fmt,
                    self.config,
                    source_cache=self._continuous_source_cache,
                )
                for win in planned:
                    entry_id = (
                        f"{src}::window_{int(win['window_index']):06d}"
                    )
                    record = dict(win)
                    record["format"] = fmt
                    record["entry_id"] = entry_id
                    self._continuous_window_map[entry_id] = record
                    expanded_entries.append(entry_id)
                    planned_window_total += 1
                    if bool(win.get("is_partial_final_window", False)):
                        partial_window_total += 1

            self.file_list = expanded_entries
            self._continuous_plan_summary = {
                "acquisition_mode": "continuous",
                "source_file_count": len(source_files),
                "planned_window_count": planned_window_total,
                "partial_window_count": partial_window_total,
                "continuous_window_sec": float(getattr(self.config, "continuous_window_sec", 600.0)),
                "continuous_step_sec": float(getattr(self.config, "continuous_step_sec", 600.0)),
                "allow_partial_final_window": bool(getattr(self.config, "allow_partial_final_window", False)),
            }
            print(
                f"Continuous mode: planned {planned_window_total} windows across "
                f"{len(source_files)} source file(s).",
                flush=True,
            )

    def _get_format(self, path: str, force_format: str) -> str:
        if force_format != 'auto':
            return force_format
        
        fmt = sniff_format(path, self.config)
        if fmt is None:
            raise ValueError(f"Could not automatically detect format for {path}. Use --format to specify.")
        return fmt

    def _entry_source_file(self, entry: str) -> str:
        record = self._continuous_window_map.get(entry)
        if record is not None:
            return str(record["source_file"])
        return str(entry)

    def _entry_window_info(self, entry: str) -> Optional[dict]:
        rec = self._continuous_window_map.get(entry)
        return dict(rec) if rec is not None else None

    def _entry_session_id(self, entry: str) -> str:
        rec = self._continuous_window_map.get(entry)
        if rec is None:
            return self._session_entry_to_id(entry)
        src_id = self._session_entry_to_id(str(rec["source_file"]))
        return f"{src_id}__window_{int(rec['window_index']):04d}"

    def _load_entry_chunk(self, entry: str, chunk_id: int, force_format: str) -> Chunk:
        rec = self._continuous_window_map.get(entry)
        if rec is not None:
            return load_chunk(
                str(rec["source_file"]),
                str(rec["format"]),
                self.config,
                chunk_id=chunk_id,
                continuous_window=rec,
                source_cache=self._continuous_source_cache,
            )
        fmt = self._get_format(entry, force_format)
        return load_chunk(
            entry,
            fmt,
            self.config,
            chunk_id=chunk_id,
            source_cache=self._continuous_source_cache,
        )

    def _continuous_entries_support_sequential_custom_tabular(self, entries) -> bool:
        if not self._is_continuous_mode_enabled() or not entries:
            return False
        last_key = None
        last_row_start = -1
        last_row_stop = -1
        last_window_end = None
        for entry in entries:
            rec = self._continuous_window_map.get(entry)
            if rec is None or str(rec.get("format", "")).lower() != "custom_tabular":
                return False
            if "row_start" not in rec or "row_stop" not in rec:
                return False
            key = os.path.abspath(str(rec["source_file"]))
            row_start = int(rec["row_start"])
            row_stop = int(rec["row_stop"])
            window_start = float(rec.get("window_start_sec", np.nan))
            if key != last_key:
                last_key = key
                last_row_start = -1
                last_row_stop = -1
                last_window_end = None
            if row_start < last_row_start:
                return False
            # Sequential custom_tabular optimization is intentionally limited to
            # non-overlapping windows. Adjacent planned windows may share support
            # rows for interpolation coverage; true sliding/overlapping windows
            # remain correct through the existing bounded random-access loader.
            if row_start < last_row_stop and (
                last_window_end is None
                or not np.isfinite(last_window_end)
                or not np.isfinite(window_start)
                or window_start < last_window_end
            ):
                return False
            last_row_start = row_start
            last_row_stop = row_stop
            last_window_end = float(rec.get("window_end_sec", np.nan))
        return True

    def _iter_entry_chunks_for_pass(self, entries, force_format: str, phase_name: str):
        if self._continuous_entries_support_sequential_custom_tabular(entries):
            group_entries = []
            group_chunk_ids = []
            group_windows = []
            group_source = None

            def _flush_group():
                if not group_entries:
                    return
                source_data = resolve_continuous_source_metadata(
                    str(group_source),
                    "custom_tabular",
                    self.config,
                    source_cache=self._continuous_source_cache,
                )
                self._record_continuous_csv_reading(phase_name, sequential_passes=1)
                yielded = 0
                iterator = iter_continuous_custom_tabular_chunks(
                    source_data,
                    group_windows,
                    self.config,
                    chunk_ids=group_chunk_ids,
                )
                while True:
                    t_load = time.perf_counter()
                    try:
                        chunk_id, _win, chunk = next(iterator)
                    except StopIteration:
                        break
                    yielded += 1
                    yield (
                        chunk_id,
                        group_entries[group_chunk_ids.index(chunk_id)],
                        chunk,
                        time.perf_counter() - t_load,
                    )
                self._record_continuous_csv_reading(phase_name, windows_yielded=yielded)

            for chunk_id, entry in enumerate(entries):
                rec = self._continuous_window_map[entry]
                src = str(rec["source_file"])
                if group_source is not None and os.path.abspath(src) != os.path.abspath(str(group_source)):
                    yield from _flush_group()
                    group_entries = []
                    group_chunk_ids = []
                    group_windows = []
                group_source = src
                group_entries.append(entry)
                group_chunk_ids.append(chunk_id)
                group_windows.append(rec)
            yield from _flush_group()
            return

        self._record_continuous_csv_reading(
            phase_name,
            fallback_windows=len(entries) if self._is_continuous_mode_enabled() else 0,
        )
        for chunk_id, entry in enumerate(entries):
            t_load = time.perf_counter()
            chunk = self._load_entry_chunk(entry, chunk_id, force_format)
            yield chunk_id, entry, chunk, time.perf_counter() - t_load

    def _continuous_metadata_channels_seen(self, entries):
        if not self._continuous_entries_support_sequential_custom_tabular(entries):
            return None
        channels_seen = []
        seen_sources = set()
        for entry in entries:
            rec = self._continuous_window_map.get(entry)
            if rec is None:
                return None
            src = os.path.abspath(str(rec["source_file"]))
            if src in seen_sources:
                continue
            source_data = resolve_continuous_source_metadata(
                str(rec["source_file"]),
                "custom_tabular",
                self.config,
                source_cache=self._continuous_source_cache,
            )
            names = list(source_data.get("channel_names", []))
            if not names:
                return None
            channels_seen.append(names)
            if not self.roi_map and source_data.get("roi_map"):
                self.roi_map = dict(source_data["roi_map"])
            seen_sources.add(src)
        return channels_seen or None

    def run_pass_1(self, force_format: str = 'auto'):
        """
        Baseline Computation.
        """
        print("Starting Pass 1: Baseline Computation...")
        pass1_started = time.perf_counter()
        pass1_chunk_load_sec = 0.0
        pass1_filter_sec = 0.0
        pass1_accumulate_sec = 0.0
        pass1_solve_sec = 0.0
        pass1_f0_compute_sec = 0.0
        
        method = self.config.baseline_method
        reservoir = baseline.DeterministicReservoir(seed=self.config.seed)
        
        if method == 'uv_raw_percentile_session':
            print("Pass 1 (Reservoir)...")
            for i, fpath, chunk, load_elapsed in self._iter_entry_chunks_for_pass(self.file_list, force_format, "pass1"):
                try:
                    t_load = time.perf_counter()
                    if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                    pass1_chunk_load_sec += load_elapsed + (time.perf_counter() - t_load)
                    
                    if not self.roi_map and chunk.metadata.get('roi_map'):
                        self.roi_map = chunk.metadata['roi_map']
                    
                    t_acc = time.perf_counter()
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        uv_data = chunk.uv_raw[:, ch_idx]
                        reservoir.add(ch_name, uv_data)
                    pass1_accumulate_sec += (time.perf_counter() - t_acc)
                        
                    if fpath not in self._pass1_manifest:
                        self._pass1_manifest.append(fpath)
                        
                except Exception as e:
                    logging.warning(f"Pass 1: Skipping {fpath} due to error: {e}")
                    if not any(x['file'] == fpath for x in self.qc_summary['failed_chunks']):
                        self.qc_summary['failed_chunks'].append({'file': fpath, 'error': str(e)})
                    continue
            
            self.stats.method_used = method
            t_f0 = time.perf_counter()
            for ch in reservoir.buffer.keys():
                f0 = reservoir.get_percentile(ch, self.config.baseline_percentile)
                self.stats.f0_values[ch] = f0
            pass1_f0_compute_sec += (time.perf_counter() - t_f0)
                
        elif method == 'uv_globalfit_percentile_session':
            accumulator = baseline.GlobalFitAccumulator()
            
            print("Pass 1a (Stats)...")
            for i, fpath, chunk, load_elapsed in self._iter_entry_chunks_for_pass(self.file_list, force_format, "pass1a"):
                try:
                    t_load = time.perf_counter()
                    if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                    pass1_chunk_load_sec += load_elapsed + (time.perf_counter() - t_load)
                    
                    if not self.roi_map and chunk.metadata.get('roi_map'):
                        self.roi_map = chunk.metadata['roi_map']
                    
                    # Compute filtered explicitly for fit accumulation
                    t_filter = time.perf_counter()
                    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, self.config)
                    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, self.config)
                    pass1_filter_sec += (time.perf_counter() - t_filter)
                    
                    t_acc = time.perf_counter()
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        accumulator.add(ch_name, chunk.uv_filt[:, ch_idx], chunk.sig_filt[:, ch_idx])
                    pass1_accumulate_sec += (time.perf_counter() - t_acc)
                        
                except Exception as e:
                    logging.warning(f"Pass 1a: Skipping {fpath}: {e}")
                    if not any(x['file'] == fpath for x in self.qc_summary['failed_chunks']):
                        self.qc_summary['failed_chunks'].append({'file': fpath, 'error': str(e)})
                    continue
            
            t_solve = time.perf_counter()
            self.stats.global_fit_params = accumulator.solve()
            pass1_solve_sec += (time.perf_counter() - t_solve)
            
            print("Pass 1b (Reservoir)...")
            for i, fpath, chunk, load_elapsed in self._iter_entry_chunks_for_pass(self.file_list, force_format, "pass1b"):
                try:
                    t_load = time.perf_counter()
                    if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                    pass1_chunk_load_sec += load_elapsed + (time.perf_counter() - t_load)
                    
                    t_acc = time.perf_counter()
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        params = self.stats.global_fit_params.get(ch_name)
                        if params:
                            uv_val = chunk.uv_raw[:, ch_idx]
                            uv_est = params['a'] * uv_val + params['b']
                            reservoir.add(ch_name, uv_est)
                    pass1_accumulate_sec += (time.perf_counter() - t_acc)
                            
                    if fpath not in self._pass1_manifest:
                        self._pass1_manifest.append(fpath)
                except Exception as e:
                    continue
            
            self.stats.method_used = method
            t_f0 = time.perf_counter()
            for ch in reservoir.buffer.keys():
                f0 = reservoir.get_percentile(ch, self.config.baseline_percentile)
                self.stats.f0_values[ch] = f0
            pass1_f0_compute_sec += (time.perf_counter() - t_f0)

        print(f"Pass 1 Complete. F0: {self.stats.f0_values}")
        if self._is_phasic_timing_enabled():
            pass1_total_sec = time.perf_counter() - pass1_started
            self._add_phasic_detail_bucket("pass1.total", pass1_total_sec)
            self._add_phasic_detail_bucket("pass1.chunk_load", pass1_chunk_load_sec)
            self._add_phasic_detail_bucket("pass1.filter", pass1_filter_sec)
            self._add_phasic_detail_bucket("pass1.accumulate", pass1_accumulate_sec)
            self._add_phasic_detail_bucket("pass1.solve", pass1_solve_sec)
            self._add_phasic_detail_bucket("pass1.f0_compute", pass1_f0_compute_sec)
            pass1_explicit = (
                pass1_chunk_load_sec
                + pass1_filter_sec
                + pass1_accumulate_sec
                + pass1_solve_sec
                + pass1_f0_compute_sec
            )
            self._add_phasic_detail_bucket("pass1.remainder", max(0.0, pass1_total_sec - pass1_explicit))
            self._set_phasic_metric("pass1.files_seen", len(self.file_list))
            self._set_phasic_metric("pass1.manifest_size", len(self._pass1_manifest))
            self._set_phasic_metric("pass1.baseline_method", method)
        
        # Robustness: Check for Missing/Invalid Baselines
        from .core.reporting import append_run_report_warnings
        # We need output_dir. Check if run_pass_1 has access. No.
        # So we must move this check to run() or pass output_dir to run_pass_1.
        # Constraint: "Do not change output filenames or directory structure... No changing regression math..."
        # But changing signature of run_pass_1 might be allowed as internal API improvement for robustness? 
        # Alternatively, do this in run().
        # "After Pass 1... Append warning to run_report.json".
        # I will do this in the `run()` method right after `run_pass_1` returns.
        # Wait, run_pass_1 computes baselines. So best place is `run()`.
        # TONIC MODE: PASS 1c (Global Robust Fit)
        if self.mode == 'tonic':
            print("Pass 1c (Tonic Global Fit accumulation)...")
            from .core.tonic_dff import compute_global_iso_fit_robust
            use_bounded_tonic_sampling = self._is_continuous_mode_enabled()
            
            if use_bounded_tonic_sampling:
                # Sample paired raw UV/SIG values per ROI for a single global tonic fit.
                # The robust fit itself caps at 200k points; this reservoir enforces
                # that bound before fitting so continuous recordings do not accumulate
                # every multi-day sample in memory.
                tonic_fit_sampler = _PairedDeterministicReservoir(
                    seed=self.config.seed,
                    capacity=TONIC_GLOBAL_FIT_SAMPLE_CAPACITY,
                )

                for i, fpath, chunk, _load_elapsed in self._iter_entry_chunks_for_pass(self.file_list, force_format, "tonic_pass1c"):
                    try:
                        if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                        for ch_idx, ch_name in enumerate(chunk.channel_names):
                            tonic_fit_sampler.add(
                                ch_name,
                                chunk.uv_raw[:, ch_idx],
                                chunk.sig_raw[:, ch_idx],
                            )

                        if fpath not in self._pass1_manifest:
                            self._pass1_manifest.append(fpath)
                    except Exception:
                        continue

                for ch in tonic_fit_sampler.channels():
                    uv_sample, sig_sample = tonic_fit_sampler.arrays(ch)
                    slope, intercept, ok, n_used = compute_global_iso_fit_robust(
                        uv_sample,
                        sig_sample,
                    )
                    total_seen = int(tonic_fit_sampler.count.get(ch, 0))
                    self.stats.tonic_global_fit_provenance[ch] = {
                        "channel": ch,
                        "tonic_global_fit_sampling_mode": "bounded_paired_reservoir",
                        "tonic_global_fit_sample_capacity": TONIC_GLOBAL_FIT_SAMPLE_CAPACITY,
                        "tonic_global_fit_seed": int(self.config.seed),
                        "tonic_global_fit_samples_seen": total_seen,
                        "tonic_global_fit_samples_used": int(n_used),
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "ok": bool(ok),
                        "n_used": int(n_used),
                    }
                    if ok:
                        self.stats.tonic_fit_params[ch] = {'slope': slope, 'intercept': intercept}
                        print(
                            f"  Tonic Fit ({ch}): slope={slope:.4f}, int={intercept:.4f} "
                            f"(N={n_used}, sampled_from={total_seen})"
                        )
                    else:
                        logging.warning(f"  Tonic Fit ({ch}) FAILED.")
            else:
                # Preserve intermittent tonic behavior: collect all raw UV/SIG
                # arrays per ROI, then fit once on the full accumulated recording.
                acc_uv = {}
                acc_sig = {}

                for i, fpath, chunk, _load_elapsed in self._iter_entry_chunks_for_pass(self.file_list, force_format, "tonic_pass1c"):
                    try:
                        if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                        for ch_idx, ch_name in enumerate(chunk.channel_names):
                            if ch_name not in acc_uv:
                                acc_uv[ch_name] = []
                                acc_sig[ch_name] = []
                            acc_uv[ch_name].append(chunk.uv_raw[:, ch_idx])
                            acc_sig[ch_name].append(chunk.sig_raw[:, ch_idx])

                        if fpath not in self._pass1_manifest:
                            self._pass1_manifest.append(fpath)
                    except Exception:
                        continue

                for ch in acc_uv.keys():
                    uv_full = np.concatenate(acc_uv[ch])
                    sig_full = np.concatenate(acc_sig[ch])
                    slope, intercept, ok, n_used = compute_global_iso_fit_robust(uv_full, sig_full)
                    self.stats.tonic_global_fit_provenance[ch] = {
                        "channel": ch,
                        "tonic_global_fit_sampling_mode": "full_accumulation",
                        "tonic_global_fit_samples_seen": int(np.sum(np.isfinite(uv_full) & np.isfinite(sig_full))),
                        "tonic_global_fit_samples_used": int(n_used),
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "ok": bool(ok),
                        "n_used": int(n_used),
                    }
                    if ok:
                        self.stats.tonic_fit_params[ch] = {'slope': slope, 'intercept': intercept}
                        print(f"  Tonic Fit ({ch}): slope={slope:.4f}, int={intercept:.4f} (N={n_used})")
                    else:
                        logging.warning(f"  Tonic Fit ({ch}) FAILED.")
            
        # End Pass 1

    def _apply_standard_analysis(self, chunk, chunk_id):
        """
        Shared source of truth for standard analysis steps (preprocessing -> regression -> normalization).
        Returns the processed chunk.
        """
        t_filter = time.perf_counter()
        uv_filt, uv_meta = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, self.config)
        sig_filt, sig_meta = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, self.config)
        if self._is_phasic_timing_enabled():
            self._add_phasic_detail_bucket("pass2.filter_lowpass", time.perf_counter() - t_filter)
        
        chunk.uv_filt = uv_filt
        chunk.sig_filt = sig_filt
        
        # Warning aggregation (NaN Safety Policy 2)
        for m in [uv_meta, sig_meta]:
             if m.get('rois_affected', 0) > 0:
                 msg = f"Chunk {chunk_id} Block-wise filtering active: {m['rois_affected']} ROIs, {m['samples_skipped']} samples skipped."
                 if 'scan_warnings' not in self.qc_summary: self.qc_summary['scan_warnings'] = []
                 self.qc_summary['scan_warnings'].append(msg)
        
        if self.mode == 'tonic':
             self._process_chunk_tonic(chunk, chunk_id)
        else:
             # PHASIC MODE (Dynamic)
             t_reg = time.perf_counter()
             uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, self.config, mode=self.mode)
             if self._is_phasic_timing_enabled():
                 self._add_phasic_detail_bucket("pass2.dynamic_regression", time.perf_counter() - t_reg)
                 dyn_timing = None
                 if hasattr(chunk, 'metadata') and chunk.metadata:
                     dyn_timing = chunk.metadata.get('dynamic_regression_timing')
                 if isinstance(dyn_timing, dict):
                     for sub_bucket, sub_elapsed in dyn_timing.get('buckets', {}).items():
                         self._add_phasic_detail_bucket(
                             f"pass2.dynamic_regression.{sub_bucket}",
                             float(sub_elapsed)
                         )
                     for metric_name, metric_value in dyn_timing.get('metrics', {}).items():
                         self._add_phasic_metric(
                             f"pass2.dynamic_regression.{metric_name}",
                             metric_value
                         )
             chunk.uv_fit = uv_fit
             chunk.delta_f = delta_f
             t_dff = time.perf_counter()
             chunk.dff = normalization.compute_dff(chunk, self.stats, self.config)
             if self._is_phasic_timing_enabled():
                 self._add_phasic_detail_bucket("pass2.dff_compute", time.perf_counter() - t_dff)
        
        return chunk

    def _resolve_representative_session(self, force_format: str, emitter=None):
        """
        Resolves the representative session only after file_list is finalized.
        Implements legacy fallback and emits audit event.
        """
        n_sessions_resolved = len(self.file_list)
        user_idx = self.config.representative_session_index
        rep_idx_effective = None
        rep_session_id = None
        user_provided = (user_idx is not None)

        if user_provided:
            if not isinstance(user_idx, int) or not (0 <= user_idx < n_sessions_resolved):
                raise ValueError(f"representative_session_index out of range: idx={user_idx}, n_sessions={n_sessions_resolved}")
            rep_idx_effective = user_idx
            rep_session_id = self._entry_session_id(self.file_list[user_idx])
        else:
            # Legacy Default: find first loadable
            for i, fpath in enumerate(self.file_list):
                try:
                    self._load_entry_chunk(fpath, i, force_format) # Validation load
                    rep_idx_effective = i
                    rep_session_id = self._entry_session_id(fpath)
                    break
                except Exception as e:
                    logging.warning(f"Resolution: Skipping session {fpath} for representative selection: {e}")
        
        self.representative_session_index = rep_idx_effective
        self.representative_session_id = rep_session_id
        self.n_sessions_resolved = n_sessions_resolved
        self.representative_user_provided = user_provided

        self.representative_session_info = {
            "representative_session_index": self.representative_session_index,
            "representative_session_id": self.representative_session_id,
            "n_sessions_resolved": self.n_sessions_resolved,
            "resolved_session_ids_preview": [self._entry_session_id(f) for f in self.file_list[:5]],
            "user_provided": self.representative_user_provided
        }

        if emitter:
            emitter.emit("inputs", "representative_session", "Representative session resolved",
                         payload=self.representative_session_info)


    # Helper for Unit Testing / Invariant Enforcement
    def _process_chunk_tonic(self, chunk: Chunk, i: int):

         # Explicit Global Fit Application
         from .core.tonic_dff import apply_global_fit, compute_session_tonic_df_from_global
         
         if not hasattr(self.stats, 'tonic_fit_params'):
             raise RuntimeError("Tonic mode active but tonic_fit_params missing!")
             
         chunk.uv_fit = np.full_like(chunk.uv_raw, np.nan)
         chunk.delta_f = np.full_like(chunk.sig_raw, np.nan)
         chunk.dff = np.full_like(chunk.sig_raw, np.nan) # Derived from delta_f/F0 below
         
         # Provenance: Print exactly once per run
         if not getattr(self.stats, '_provenance_printed_tonic', False):
             print(f"Tonic iso-fit source: global robust fit (entire recording). Dynamic uv_fit ignored.")
             self.stats._provenance_printed_tonic = True
         
         # Check for missing params (Invariant A: No silent NaNs)
         missing_rois = [r for r in chunk.channel_names if r not in self.stats.tonic_fit_params]
         if missing_rois:
             raise RuntimeError(f"Chunk {i}: Missing tonic fit params for {len(missing_rois)} ROIs: {missing_rois[:5]}. Cannot compute Tonic DF.")
             
         # Tonic ROI Loop
         for r_idx, roi in enumerate(chunk.channel_names):
             params = self.stats.tonic_fit_params.get(roi)
             if params:
                  # Apply Global Fit
                  iso_fit = apply_global_fit(chunk.uv_raw[:, r_idx], params['slope'], params['intercept'])
                  chunk.uv_fit[:, r_idx] = iso_fit
                  
                  # Compute Tonic DF (Additive)
                  res = compute_session_tonic_df_from_global(chunk.sig_raw[:, r_idx], chunk.uv_raw[:, r_idx], iso_fit)
                  if not res.get('success', False):
                      reason = res.get('reason', 'Unknown failure in compute_session_tonic_df_from_global')
                      raise RuntimeError(f"Chunk {i}, ROI {roi}: Tonic DF compute failed. Reason: {reason}")
                  
                  # NaN Fraction Check (Tolerance)
                  valid_mask = res.get('valid_mask')
                  if valid_mask is None:
                      raise RuntimeError(f"Chunk {i}, ROI {roi}: Missing valid_mask from tonic DF result.")

                  n_total = len(valid_mask)
                  n_valid = int(np.sum(valid_mask))
                  frac_invalid = 1.0 - (n_valid / float(n_total)) if n_total > 0 else 1.0
                       
                  allowed_raw = getattr(self.config, 'tonic_allowed_nan_frac', 0.0)
                  try:
                      allowed = float(allowed_raw)
                  except (ValueError, TypeError):
                      raise RuntimeError(f"Invalid tonic_allowed_nan_frac={allowed_raw!r} (type {type(allowed_raw).__name__}), must be a float")

                  if frac_invalid > allowed:
                      raise RuntimeError(f"Chunk {i}, ROI {roi}: Tonic NaN fraction ({frac_invalid:.4f}) exceeds allowed ({allowed}).")

                  chunk.delta_f[:, r_idx] = res['df']
         
         
         # Invariant Post-Check: Ensure no NaNs in explicitly computed ROIs
         # RELAXED for Nan-Tolerance Logic: normalization.compute_dff will handle excessive NaN checking.
         # for r_idx, roi in enumerate(chunk.channel_names):
         #     if np.any(np.isnan(chunk.delta_f[:, r_idx])):
         #         raise RuntimeError(f"Chunk {i}, ROI {roi}: Tonic delta_f contains NaNs after computation. Strict invariant violated.")

         # Compute dFF (using normalization.compute_dff which uses chunk.delta_f / F0)
         chunk.dff = normalization.compute_dff(chunk, self.stats, self.config)

    def run_pass_2(self, output_dir: str, force_format: str = 'auto'):
        # Lazy import for VIZ
        from .viz import plots
        pass2_started = time.perf_counter()
        pass2_manifest_check_sec = 0.0
        pass2_chunk_read_sec = 0.0
        pass2_feature_extract_sec = 0.0
        pass2_cache_write_sec = 0.0
        pass2_qc_scan_sec = 0.0
        pass2_features_csv_write_sec = 0.0
        pass2_qc_summary_write_sec = 0.0
        pass2_run_metadata_write_sec = 0.0
        pass2_rep_validation_sec = 0.0
        pass2_chunks_processed = 0
        pass2_sample_rows_processed = 0
        pass2_roi_samples_processed = 0
        pass2_features_rows = 0
        pass2_peak_count_total = 0
        
        # Robustness: strict directory creation
        os.makedirs(os.path.join(output_dir, 'qc'), exist_ok=True)
        
        all_features = []
        rep_chunk_for_plotting = None
        rep_idx = self.representative_session_index  # set by _resolve_representative_session
        
        # Freeze manifest to ensure it cannot be mutated after Pass 1
        frozen_manifest = tuple(self._pass1_manifest)
        
        # Check for new files not in pass 1 manifest
        t_manifest = time.perf_counter()
        new_files = [f for f in self.file_list if f not in frozen_manifest]
        for f in new_files:
            if not any(x['file'] == f for x in self.qc_summary['failed_chunks']):
                self.qc_summary['failed_chunks'].append({'file': f, 'error': 'Ignored (Not in Pass 1 manifest)'})
        pass2_manifest_check_sec += (time.perf_counter() - t_manifest)
                
        if new_files:
             logging.warning(f"Pass 2: Found {len(new_files)} new or skipped files not in Pass 1 manifest. First few: {new_files[:3]}. They will be ignored.")
        
        print("Pass 2 (Analysis)...")
        # Ensure we only iterate over files successfully processed in Pass 1
        for i, fpath, chunk, load_elapsed in self._iter_entry_chunks_for_pass(frozen_manifest, force_format, "pass2"):
            try:
                t_read = time.perf_counter()
                if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                pass2_chunk_read_sec += load_elapsed + (time.perf_counter() - t_read)
                pass2_chunks_processed += 1
                if hasattr(chunk, 'sig_raw') and chunk.sig_raw is not None:
                    pass2_sample_rows_processed += int(chunk.sig_raw.shape[0])
                    pass2_roi_samples_processed += int(chunk.sig_raw.shape[0] * chunk.sig_raw.shape[1])
                
                # Capture ROI map if missing (e.g. if Pass 1 failed or skipped)
                if not self.roi_map and chunk.metadata.get("roi_map"):
                    self.roi_map = chunk.metadata["roi_map"]

                
                # SHARED PROCESSING (single source of truth for filtering, regression, dff)
                chunk = self._apply_standard_analysis(chunk, i)
                self._record_dynamic_fit_slope_summaries(chunk, i, self._entry_source_file(fpath))
                self._record_dynamic_fit_validity_metrics(chunk, i, self._entry_source_file(fpath))
                self._record_baseline_reference_candidate_metrics(chunk, i, self._entry_source_file(fpath))
                
                # Retain for representative plotting
                if rep_idx is not None and i == rep_idx:
                    rep_chunk_for_plotting = chunk

                
                # traces-only: skip feature extraction and all feature-derived outputs.
                # NOTE: This pipeline does NOT perform event detection as a separate stage.
                # "events" in this codebase refers to NDJSON lifecycle logging (engine:start,
                # engine:context, etc.), not signal event detection.  The only analysis step
                # gated here is feature_extraction.extract_features(), which computes
                # per-chunk statistics (peak count, AUC, mean, etc.).
                if not self.traces_only:
                    t_feats = time.perf_counter()
                    feats_df = feature_extraction.extract_features(chunk, self.config)
                    pass2_feature_extract_sec += (time.perf_counter() - t_feats)
                    all_features.append(feats_df)
                    pass2_features_rows += int(len(feats_df))
                    if 'peak_count' in feats_df.columns:
                        peak_sum = pd.to_numeric(feats_df['peak_count'], errors='coerce').fillna(0).sum()
                        pass2_peak_count_total += int(peak_sum)
                
                
                if hasattr(self, '_cache_writer'):
                    t_cache = time.perf_counter()
                    self._cache_writer.add_chunk(chunk, i, self._entry_source_file(fpath))
                    pass2_cache_write_sec += (time.perf_counter() - t_cache)
                
                t_scan = time.perf_counter()
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    qc_warnings = chunk.metadata.get('qc_warnings', [])
                    if any("DEGENERATE" in w for w in qc_warnings):
                        if not any(x['file'] == fpath for x in self.qc_summary['failed_chunks']):
                            self.qc_summary['failed_chunks'].append({'file': fpath, 'error': 'QC: Degenerate data detected'})
                pass2_qc_scan_sec += (time.perf_counter() - t_scan)


                # Legacy VIZ was here, now moved out of loop for strict resolution/loading
                
            except Exception as e:
                # Requirement B4: Fail fast if a file in the manifest cannot be loaded in Pass 2
                raise RuntimeError(f"Pass 2: Cannot reliably read manifest file {fpath} successfully processed in Pass 1. Error: {e}")
                
        if all_features and self.mode != 'tonic':
            t_feats_write = time.perf_counter()
            full_feats = pd.concat(all_features, ignore_index=True)
            feats_dir = os.path.join(output_dir, 'features')
            os.makedirs(feats_dir, exist_ok=True)
            
            full_feats.to_csv(os.path.join(feats_dir, 'features.csv'), index=False)
            pass2_features_csv_write_sec += (time.perf_counter() - t_feats_write)

        if self.dynamic_fit_qc_records and self.mode != 'tonic':
            qc_rows = []
            for rec in self.dynamic_fit_qc_records:
                row = dict(rec)
                for list_key in (
                    "dynamic_fit_qc_flags",
                    "dynamic_fit_qc_hard_flags",
                    "dynamic_fit_qc_soft_flags",
                    "reference_comparison_flags",
                ):
                    flags = row.get(list_key, [])
                    if isinstance(flags, (list, tuple)):
                        row[list_key] = ";".join(str(x) for x in flags)
                    elif flags is None:
                        row[list_key] = ""
                qc_rows.append(row)
            pd.DataFrame(qc_rows).to_csv(
                os.path.join(output_dir, "qc", "dynamic_fit_qc_by_chunk.csv"),
                index=False,
            )
            with open(os.path.join(output_dir, "qc", "dynamic_fit_qc_by_chunk.json"), "w") as f:
                json.dump(
                    _sanitize_strict_json(self.dynamic_fit_qc_records),
                    f,
                    indent=2,
                    allow_nan=False,
                )

        if self.baseline_reference_candidate_records and self.mode != 'tonic':
            candidate_rows = []
            for rec in self.baseline_reference_candidate_records:
                row = dict(rec)
                for list_key in (
                    "dynamic_fit_qc_flags",
                    "dynamic_fit_qc_hard_flags",
                    "dynamic_fit_qc_soft_flags",
                ):
                    flags = row.get(list_key, [])
                    if isinstance(flags, (list, tuple)):
                        row[list_key] = ";".join(str(x) for x in flags)
                    elif flags is None:
                        row[list_key] = ""
                candidate_rows.append(row)
            pd.DataFrame(candidate_rows).to_csv(
                os.path.join(output_dir, "qc", "baseline_reference_candidate_by_chunk.csv"),
                index=False,
            )
            with open(
                os.path.join(output_dir, "qc", "baseline_reference_candidate_by_chunk.json"),
                "w",
            ) as f:
                json.dump(
                    _sanitize_strict_json(self.baseline_reference_candidate_records),
                    f,
                    indent=2,
                    allow_nan=False,
                )

        total_chunks = len(self.file_list)
        if total_chunks > 0:
            self.qc_summary['chunk_fail_fraction'] = len(self.qc_summary.get('failed_chunks', [])) / total_chunks
            
        # Robustness: Add baseline invalid counts if tracked
        if 'invalid_baseline_rois' in self.qc_summary:
            bad_rois = self.qc_summary['invalid_baseline_rois']
            # D3: Ensure explicit counts always present if key exists
            self.qc_summary['baseline_invalid_roi_count'] = len(bad_rois)
            total_affected = len(bad_rois) * total_chunks
            self.qc_summary['baseline_invalid_roi_chunk_pairs'] = total_affected
            if bad_rois:
                logging.warning(f"Baseline invalid for {len(bad_rois)} ROIs across {total_chunks} chunks ({total_affected} pairs).")
        self._update_dynamic_fit_slope_warning_summary()
        self._update_dynamic_fit_slope_constraint_summary()
        self._update_dynamic_fit_qc_summary()
        self._update_baseline_reference_candidate_summary()
        self._update_reference_candidate_comparison_summary()
            
        if self.mode != 'tonic':
            t_qc_write = time.perf_counter()
            with open(os.path.join(output_dir, 'qc', 'qc_summary.json'), 'w') as f:
                json.dump(_sanitize_metadata(self.qc_summary), f, indent=2)
            pass2_qc_summary_write_sec += (time.perf_counter() - t_qc_write)
            
        run_meta = {
            'target_fs_hz': self.config.target_fs_hz,
            'seed': self.config.seed,
            'allow_partial_final_chunk': self.config.allow_partial_final_chunk,
            'roi_map': self.roi_map,
            'baseline_method': self.stats.method_used,
            'f0_values': self.stats.f0_values,
            'global_fit_params': self.stats.global_fit_params,
            # Validation Metadata for Paper Alignment
            'f0_source': self.stats.method_used,
            'phasic_uv_fit_method': 'dynamic', # Strict requirement for this pipeline version
            'f0_is_from_uv_fit': False,        # Constraint: explicit separation
            'regression_window_sec': self.config.window_sec,
            'regression_step_sec': self.config.step_sec,
            'regression_mode': 'dynamic' if self.mode == 'phasic' else self.mode,
            # D1: Write invalid baseline ROIs
            'invalid_baseline_rois': self.qc_summary.get('invalid_baseline_rois', []),
            'dynamic_fit_slope_warning_summary': self.dynamic_fit_slope_warning_summary,
            'dynamic_fit_slope_constraint_summary': self.dynamic_fit_slope_constraint_summary,
        }
        if self.mode != 'tonic':
            run_meta['dynamic_fit_slope_warning_records'] = self.dynamic_fit_slope_warning_records
            run_meta['dynamic_fit_slope_constraint_records'] = self.dynamic_fit_slope_constraint_records
        if self._is_continuous_mode_enabled():
            run_meta.update(
                {
                    'acquisition_mode': 'continuous',
                    'continuous_window_sec': float(getattr(self.config, 'continuous_window_sec', 600.0)),
                    'continuous_step_sec': float(getattr(self.config, 'continuous_step_sec', 600.0)),
                    'allow_partial_final_window': bool(getattr(self.config, 'allow_partial_final_window', False)),
                    'continuous_plan_summary': self._continuous_plan_summary,
                    'continuous_source_file_count': int(
                        self._continuous_plan_summary.get('source_file_count', 0)
                    )
                    if isinstance(self._continuous_plan_summary, dict)
                    else 0,
                    'continuous_planned_window_count': int(
                        self._continuous_plan_summary.get('planned_window_count', len(self.file_list))
                    )
                    if isinstance(self._continuous_plan_summary, dict)
                    else int(len(self.file_list)),
                    'continuous_csv_reading': _sanitize_metadata(self._continuous_csv_reading),
                }
            )
        t_meta_write = time.perf_counter()
        with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
            json.dump(_sanitize_metadata(run_meta), f, indent=2)
        pass2_run_metadata_write_sec += (time.perf_counter() - t_meta_write)
            
        if self.qc_summary['chunk_fail_fraction'] > self.config.qc_max_chunk_fail_fraction:
            logging.error(f"High failure rate: {self.qc_summary['chunk_fail_fraction']:.2%}")

        # -----------------------------
        # Representative Session Validation
        # -----------------------------
        rep_fpath = self.file_list[rep_idx] if (rep_idx is not None and 0 <= rep_idx < len(self.file_list)) else None
        
        t_rep = time.perf_counter()
        if rep_chunk_for_plotting is None:
            if self.representative_user_provided:
                raise RuntimeError(
                    f"FAILED to process requested representative session "
                    f"(index={rep_idx}, file={rep_fpath}, stage=analysis/pass-2). "
                    f"Session was not successfully processed during Pass 2."
                )
        pass2_rep_validation_sec += (time.perf_counter() - t_rep)

        if self._is_phasic_timing_enabled():
            pass2_total_sec = time.perf_counter() - pass2_started
            self._add_phasic_detail_bucket("pass2.total", pass2_total_sec)
            self._add_phasic_detail_bucket("pass2.manifest_check", pass2_manifest_check_sec)
            self._add_phasic_detail_bucket("pass2.chunk_read", pass2_chunk_read_sec)
            self._add_phasic_detail_bucket("pass2.feature_extraction", pass2_feature_extract_sec)
            self._add_phasic_detail_bucket("pass2.cache_write", pass2_cache_write_sec)
            self._add_phasic_detail_bucket("pass2.qc_warning_scan", pass2_qc_scan_sec)
            self._add_phasic_detail_bucket("pass2.features_csv_write", pass2_features_csv_write_sec)
            self._add_phasic_detail_bucket("pass2.qc_summary_write", pass2_qc_summary_write_sec)
            self._add_phasic_detail_bucket("pass2.run_metadata_write", pass2_run_metadata_write_sec)
            self._add_phasic_detail_bucket("pass2.rep_validation", pass2_rep_validation_sec)
            pass2_explicit = (
                pass2_manifest_check_sec
                + pass2_chunk_read_sec
                + pass2_feature_extract_sec
                + pass2_cache_write_sec
                + pass2_qc_scan_sec
                + pass2_features_csv_write_sec
                + pass2_qc_summary_write_sec
                + pass2_run_metadata_write_sec
                + pass2_rep_validation_sec
                + self._phasic_detail_buckets.get("pass2.filter_lowpass", 0.0)
                + self._phasic_detail_buckets.get("pass2.dynamic_regression", 0.0)
                + self._phasic_detail_buckets.get("pass2.dff_compute", 0.0)
            )
            self._add_phasic_detail_bucket("pass2.remainder", max(0.0, pass2_total_sec - pass2_explicit))
            self._set_phasic_metric("pass2.chunks_processed", pass2_chunks_processed)
            self._set_phasic_metric("pass2.samples_processed_rows", pass2_sample_rows_processed)
            self._set_phasic_metric("pass2.samples_processed_roi_values", pass2_roi_samples_processed)
            self._set_phasic_metric("pass2.features_rows", pass2_features_rows)
            self._set_phasic_metric("pass2.peaks_detected_total", pass2_peak_count_total)
                
    def _apply_roi_filter(self, chunk):
        """Filter chunk data to only include channels in self._selected_rois."""
        selected = set(self._selected_rois)
        keep_idx = [i for i, name in enumerate(chunk.channel_names) if name in selected]
        chunk.channel_names = [chunk.channel_names[i] for i in keep_idx]
        chunk.uv_raw = chunk.uv_raw[:, keep_idx]
        chunk.sig_raw = chunk.sig_raw[:, keep_idx]
        if chunk.metadata and "roi_map" in chunk.metadata:
            chunk.metadata["roi_map"] = {k: v for k, v in chunk.metadata["roi_map"].items() if k in selected}
        return chunk

    def _session_entry_to_id(self, entry: str) -> str:
        """Returns a stable session ID for a file path or RWD folder."""
        p = pathlib.Path(entry)
        if p.name == "fluorescence.csv":
            return p.parent.name
        return p.stem

    def run(self, input_dir: str, output_dir: str, force_format: str = 'auto', recursive: bool = False, glob_pattern: str = "*.csv", include_rois: List[str] = None, exclude_rois: List[str] = None, traces_only: bool = False, emitter=None, sessions_per_hour: int = None):
        # Lazy import to avoid GUI side effects at module level
        from .viz import plots
        run_started = time.perf_counter()
        if self._is_phasic_timing_enabled():
            self._phasic_started_at = run_started
        
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'qc'), exist_ok=True)
        t_discovery = time.perf_counter()
        self.discover_files(input_dir, recursive, glob_pattern, force_format=force_format)
        self._add_phasic_phase_bucket("phase.input_discovery", time.perf_counter() - t_discovery)
        self._set_phasic_metric("files_discovered", len(self.file_list))
        
        # --- Preview Mode: limit to first N sessions ---
        n_total_discovered = len(self.file_list)
        preview_first_n = self.config.preview_first_n
        t_preview = time.perf_counter()
        if preview_first_n is not None:
            limit_n = min(preview_first_n, n_total_discovered)
            self.file_list = self.file_list[:limit_n]
            self.run_type = "preview"
            self.preview_info = {
                "selector": "first_n",
                "first_n": preview_first_n,
                "n_total_discovered": n_total_discovered,
                "n_sessions_resolved": len(self.file_list)
            }
            logging.info(f"Preview mode: processing {len(self.file_list)} of {n_total_discovered} discovered sessions (first_n={preview_first_n}).")
        else:
            self.run_type = "full"
            self.preview_info = None
        self._add_phasic_phase_bucket("phase.preview_selection", time.perf_counter() - t_preview)
        self._set_phasic_metric("files_after_preview", len(self.file_list))
        
        # Emit inputs:preview audit event if emitter provided
        if emitter and self.preview_info is not None:
            emitter.emit("inputs", "preview", "Preview selection resolved",
                         payload=self.preview_info)
        
        # --- ROI Discovery & Resolution ---
        roi_read_sec = 0.0
        channels_seen = self._continuous_metadata_channels_seen(self.file_list)
        if channels_seen is not None:
            self._record_continuous_csv_reading("roi_discovery", sequential_passes=0, windows_yielded=0)
            self._set_phasic_metric("roi_discovery_source", "continuous_metadata")
        else:
            channels_seen = []
            for i, fpath in enumerate(self.file_list):
                try:
                    t_roi_read = time.perf_counter()
                    chunk = self._load_entry_chunk(fpath, i, force_format)
                    roi_read_sec += (time.perf_counter() - t_roi_read)
                    channels_seen.append(chunk.channel_names)
                except Exception as e:
                    logging.warning(f"ROI Discovery: Failed to read {fpath}: {e}")
            if self._is_continuous_mode_enabled():
                self._record_continuous_csv_reading(
                    "roi_discovery",
                    fallback_windows=len(self.file_list),
                )
            self._set_phasic_metric("roi_discovery_source", "chunk_reads")
        self._add_phasic_phase_bucket("phase.roi_discovery_read_chunks", roi_read_sec)
        
        if not channels_seen:
            raise RuntimeError("No valid data files found for ROI discovery.")
            
        # Intersection over all valid chunks, preserving discovered order from first chunk
        t_roi_resolve = time.perf_counter()
        channel_sets = [set(cx) for cx in channels_seen]
        discovered_rois = [r for r in channels_seen[0] if all(r in cs for cs in channel_sets)]
                
        selected_rois = list(discovered_rois)
        
        if include_rois is not None:
             missing = [r for r in include_rois if r not in discovered_rois]
             if missing:
                 raise ValueError(f"Validation Error: Included ROIs not found in discovered ROIs: {missing}")
             # Preserve discovered order, filter by include_rois
             selected_rois = [r for r in discovered_rois if r in include_rois]
             
        if exclude_rois is not None:
             missing = [r for r in exclude_rois if r not in discovered_rois]
             if missing:
                 logging.warning(f"Excluded ROIs not found in discovered ROIs (ignoring): {missing}")
             selected_rois = [r for r in selected_rois if r not in exclude_rois]
             
        self.roi_selection = {
            "discovered_rois": discovered_rois,
            "include_rois": include_rois,
            "exclude_rois": exclude_rois,
            "selected_rois": selected_rois
        }
        self._add_phasic_phase_bucket("phase.roi_selection_resolution", time.perf_counter() - t_roi_resolve)
        self._set_phasic_metric("rois_discovered", len(discovered_rois))
        self._set_phasic_metric("rois_selected", len(selected_rois))
        
        self._selected_rois = selected_rois
        self.traces_only = traces_only

        # --- Representative Session Resolution ---
        t_rep = time.perf_counter()
        self._resolve_representative_session(force_format, emitter=emitter)
        self._add_phasic_phase_bucket("phase.representative_resolution", time.perf_counter() - t_rep)

        # 1. Run Report (Pre-Analysis)

        t_report = time.perf_counter()
        generate_run_report(
            self.config, output_dir, 
            roi_selection=self.roi_selection, 
            traces_only=traces_only,
            representative_info=self.representative_session_info,
            preview_info=self.preview_info,
            sessions_per_hour=sessions_per_hour,
            sessions_per_hour_source=None
        )
        self._add_phasic_phase_bucket("phase.run_report_write", time.perf_counter() - t_report)
        
        t_pass1 = time.perf_counter()
        self.run_pass_1(force_format)
        self._add_phasic_phase_bucket("phase.pass1_total", time.perf_counter() - t_pass1)
        if self.mode == "tonic":
            _append_run_report_section(
                output_dir,
                "tonic_global_fit_provenance",
                getattr(self.stats, "tonic_global_fit_provenance", {}),
            )
        
        baseline_warnings = []
        invalid_rois = []
        
        # Robustness: Always track these keys
        self.qc_summary['invalid_baseline_rois'] = []
        self.qc_summary['baseline_invalid_roi_count'] = 0
        
        # D2: ROI Union
        keys_map = list(self.roi_map.keys()) if self.roi_map else []
        keys_stats = list(self.stats.f0_values.keys())
        all_known_rois = sorted(list(set(keys_map) | set(keys_stats)))
        
        t_baseline_check = time.perf_counter()
        for roi in all_known_rois:
            f0 = self.stats.f0_values.get(roi, float('nan'))
            if np.isnan(f0) or np.isinf(f0) or f0 <= self.config.f0_min_value:
                invalid_rois.append(roi)
                baseline_warnings.append(f"Invalid F0 for ROI '{roi}': {f0}. (Min allowed: {self.config.f0_min_value})")
                
        if baseline_warnings:
             append_run_report_warnings(output_dir, baseline_warnings)
             self.qc_summary['invalid_baseline_rois'] = invalid_rois
             self.qc_summary['baseline_invalid_roi_count'] = len(invalid_rois)
        self._add_phasic_phase_bucket("phase.baseline_validation", time.perf_counter() - t_baseline_check)
        self._set_phasic_metric("baseline_invalid_roi_count", len(invalid_rois))

        from .io.hdf5_cache import Hdf5TraceCacheWriter
        t_cache_init = time.perf_counter()
        cache_path = os.path.join(output_dir, f"{self.mode}_trace_cache.h5")
        self._cache_writer = Hdf5TraceCacheWriter(cache_path, self.mode, self.config)
        self._add_phasic_phase_bucket("phase.cache_writer_init", time.perf_counter() - t_cache_init)
        
        try:
            t_pass2 = time.perf_counter()
            self.run_pass_2(output_dir, force_format)
            self._add_phasic_phase_bucket("phase.pass2_total", time.perf_counter() - t_pass2)
            t_finalize = time.perf_counter()
            self._cache_writer.finalize()
            self._add_phasic_phase_bucket("phase.cache_finalize", time.perf_counter() - t_finalize)
        except Exception:
            self._cache_writer.abort()
            raise

        if self._is_continuous_mode_enabled():
            _append_run_report_section(
                output_dir,
                "continuous_csv_reading",
                self._continuous_csv_reading,
            )
        if self.mode != "tonic":
            _append_run_report_section(
                output_dir,
                "dynamic_fit_slope_warning_summary",
                self.dynamic_fit_slope_warning_summary,
            )
            _append_run_report_section(
                output_dir,
                "dynamic_fit_slope_constraint_summary",
                self.dynamic_fit_slope_constraint_summary,
            )

        if self._is_phasic_timing_enabled():
            dyn_total = self._phasic_detail_buckets.get("pass2.dynamic_regression", 0.0)
            dyn_sub_sum = sum(
                value
                for key, value in self._phasic_detail_buckets.items()
                if key.startswith("pass2.dynamic_regression.")
                and not key.endswith(".remainder")
                and "." not in key[len("pass2.dynamic_regression."):]
            )
            self._add_phasic_detail_bucket(
                "pass2.dynamic_regression.remainder",
                max(0.0, dyn_total - dyn_sub_sum)
            )

            wp_total = self._phasic_detail_buckets.get("pass2.dynamic_regression.window_pearson_gating", 0.0)
            wp_sub_sum = sum(
                value
                for key, value in self._phasic_detail_buckets.items()
                if key.startswith("pass2.dynamic_regression.window_pearson_gating.")
                and not key.endswith(".remainder")
            )
            self._add_phasic_detail_bucket(
                "pass2.dynamic_regression.window_pearson_gating.remainder",
                max(0.0, wp_total - wp_sub_sum)
            )

            self._set_phasic_metric("traces_only", int(bool(self.traces_only)))
            if self._is_continuous_mode_enabled():
                self._set_phasic_metric(
                    "continuous_csv_reading.sequential_used",
                    int(bool(self._continuous_csv_reading.get("sequential_csv_reading_used", False))),
                )
                self._set_phasic_metric(
                    "continuous_csv_reading.source_csv_open_read_passes",
                    int(self._continuous_csv_reading.get("source_csv_open_read_passes", 0)),
                )
                self._set_phasic_metric(
                    "continuous_csv_reading.windows_yielded_sequentially",
                    int(self._continuous_csv_reading.get("windows_yielded_sequentially", 0)),
                )
                self._set_phasic_metric(
                    "continuous_csv_reading.bounded_loader_fallback_count",
                    int(self._continuous_csv_reading.get("bounded_loader_fallback_count", 0)),
                )
            self._emit_phasic_timing_details(time.perf_counter() - run_started)
        
        print("Pipeline Done.")
