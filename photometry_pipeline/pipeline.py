
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
from types import MappingProxyType

from .config import Config
from .core.types import Chunk, SessionStats, PerRoiCorrectionSpec
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
from .core.signal_state_diagnostics import (
    compute_signal_state_diagnostics,
    summarize_signal_state_diagnostics,
)
from .core.signal_only_f0_candidate import (
    compute_signal_only_f0_candidate,
    summarize_signal_only_f0_candidates,
)
from .core.correction_policy_proposal import (
    apply_correction_policy_proposals,
    summarize_correction_policy_proposals,
)
from .core.reference_candidate_comparison import classify_reference_candidates
from .core.utils import natural_sort_key
from .core.reporting import generate_run_report, append_run_report_warnings
from .input_processing_completeness import (
    InputProcessingAccountant,
    InputProcessingError,
    POLICY_INCOMPLETE_FINAL_RWD_CHUNK,
)
from .run_completion_contract import CORRECTION_PROVENANCE_SCHEMA_VERSION
from .guided_manifest_current_facts import build_guided_manifest_current_facts
from .guided_manifest_verification import (
    GuidedManifestCliContext,
    load_guided_candidate_manifest,
    verify_guided_candidate_manifest_consumption,
)
# from .viz import plots # Moved to run() to avoid side effects

TONIC_GLOBAL_FIT_SAMPLE_CAPACITY = 200_000


class CorrectionProcessingError(RuntimeError):
    """Typed failure raised when a selected correction cannot produce a trace."""

    def __init__(
        self,
        *,
        roi_id: str,
        chunk_id: int,
        source_file: str,
        selected_strategy: str,
        reason: str,
    ) -> None:
        self.roi_id = str(roi_id)
        self.chunk_id = int(chunk_id)
        self.source_file = str(source_file)
        self.selected_strategy = str(selected_strategy)
        self.reason = str(reason)
        super().__init__(
            "Correction failed for "
            f"ROI {self.roi_id!r}, chunk {self.chunk_id}, source {self.source_file!r}, "
            f"strategy {self.selected_strategy!r}: {self.reason}"
        )


_SIGNAL_STATE_CONFIG_KEYS = (
    "signal_state_smoothing_window_fraction",
    "signal_state_smoothing_window_sec",
    "signal_state_high_quantile",
    "signal_state_low_quantile",
    "signal_state_min_episode_fraction",
    "signal_state_min_episode_sec",
    "signal_state_edge_fraction",
    "signal_state_variability_window_fraction",
    "signal_state_variability_window_sec",
    "signal_state_low_variability_quantile",
    "signal_state_low_variability_ratio_threshold",
    "signal_state_partial_min_high_fraction",
    "signal_state_partial_min_longest_fraction",
    "signal_state_partial_max_variability_ratio",
    "signal_state_partial_min_variability_suppression",
    "signal_state_partial_requires_low_variability",
    "signal_state_step_window_fraction",
    "signal_state_step_window_sec",
    "signal_state_step_threshold_robust_z",
    "signal_state_min_robust_range",
)

_SIGNAL_ONLY_F0_CONFIG_KEYS = (
    "signal_only_f0_window_fraction",
    "signal_only_f0_window_sec",
    "signal_only_f0_low_quantile",
    "signal_only_f0_smoothing_window_fraction",
    "signal_only_f0_smoothing_window_sec",
    "signal_only_f0_min_window_samples",
    "signal_only_f0_max_window_fraction",
    "signal_only_f0_min_robust_range",
    "signal_only_f0_max_above_signal_fraction",
    "signal_only_f0_max_tracking_fraction",
    "signal_only_f0_min_coverage_fraction",
    "signal_only_f0_high_state_context_mode",
    "signal_only_f0_state_aware_enabled",
    "signal_only_f0_low_support_quantile",
    "signal_only_f0_low_support_buffer_fraction",
    "signal_only_f0_low_support_buffer_sec",
    "signal_only_f0_min_low_support_fraction",
    "signal_only_f0_min_anchor_count",
    "signal_only_f0_max_anchor_gap_fraction",
    "signal_only_f0_max_anchor_gap_sec",
    "signal_only_f0_edge_extrapolation_mode",
    "signal_only_f0_max_edge_extrapolation_fraction",
    "signal_only_f0_max_edge_extrapolation_sec",
    "signal_only_f0_medium_extrapolation_fraction",
    "signal_only_f0_high_extrapolation_fraction",
    "signal_only_f0_low_anchor_support_fraction",
    "signal_only_f0_low_anchor_count",
    "signal_only_f0_confidence_cap_on_large_gap",
)

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
    def __init__(
        self,
        config: Config,
        mode: str = 'phasic',
        per_roi_feature_config: dict | None = None,
        per_roi_feature_provenance: dict | None = None,
        per_roi_correction: dict[str, PerRoiCorrectionSpec] | None = None,
    ):
        self.config = config
        self.mode = str(mode)
        # The correction map is an execution input, not a live reference to
        # mutable caller state. PerRoiCorrectionSpec is frozen; copying the
        # mapping here ensures the same resolved selection is used for every
        # chunk and prevents external dict mutation during a run.
        if per_roi_correction is None:
            self.per_roi_correction = None
        else:
            if not isinstance(per_roi_correction, dict):
                raise TypeError("per_roi_correction must be a dict or None")
            self.per_roi_correction = MappingProxyType(dict(per_roi_correction))
        # Optional per-ROI feature-detection settings (4J16k32b). When None
        # (the default), feature extraction uses `self.config` for every ROI,
        # identical to today's single-config behavior.
        # per_roi_feature_config: dict[str, Config] | None -- each value is
        #   already a complete, resolved Config (see
        #   guided_new_analysis_plan.build_per_roi_feature_backend_config),
        #   not a sparse override; Pipeline never merges partial settings.
        # per_roi_feature_provenance: dict[str, dict] | None, written to
        #   <phasic_out>/features/feature_event_provenance.json only when
        #   set. Each entry should have "source", "feature_event_profile_id",
        #   "override_config_fields" (may be sparse), and
        #   "effective_config_fields" (the complete settings actually used;
        #   see guided_new_analysis_plan.build_per_roi_feature_event_provenance).
        self.per_roi_feature_config = per_roi_feature_config
        self.per_roi_feature_provenance = per_roi_feature_provenance
        # Optional authoritative production correction selection. A missing
        # map intentionally preserves the legacy uniform dynamic-fit path;
        # an explicit map is copied above and passed to every phasic chunk.
        self.file_list = []
        # The single authorized final-chunk exclusion (if any) and the frozen
        # per-chunk accountant, so every admitted chunk reaches exactly one
        # terminal disposition and none is silently omitted (4J16k41 / C8).
        self._authorized_exclusion = None
        self._admitted_accountant = None
        self._intermittent_chunk_id_by_source = {}
        self._continuous_window_map = {}
        self._continuous_source_cache = {}
        self._continuous_plan_summary = None
        self._guided_npm_authorized_runtime = None
        self._rwd_contract_validation = dict(
            getattr(config, "rwd_contract_validation", {}) or {}
        )
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
        # fit_chunk_dynamic always returns a full-width array now, even when
        # zero ROIs requested dynamic fitting (e.g. a future all-Signal-Only-F0
        # chunk); dynamic_fit_group_count, not uv_fit-is-None, is the explicit
        # signal for "no dynamic-fit computation actually happened here".
        if int(chunk.metadata.get("dynamic_fit_group_count", 1)) == 0:
            return
        # Per-ROI, not chunk-wide: a mixed-strategy chunk can have a
        # different resolved dynamic_fit_mode per ROI (grouped dispatch).
        # Three-state contract (regression.classify_per_roi_dynamic_fit_mode_
        # contract): "authoritative" -- a ROI absent from it did not undergo
        # dynamic fitting and must not get a fabricated validity record via
        # the chunk-wide fallback, even though that flat value is a real,
        # valid mode string for the ROI(s) that DID; "absent" -- legacy/pre-
        # grouping metadata, flat fallback allowed for every ROI; "malformed"
        # -- present but not a dict is corrupt current metadata, not legacy
        # data, and must fail closed rather than silently fall back. This
        # method runs in the production per-chunk processing path, so it
        # raises rather than quietly dropping records.
        contract_state = regression.classify_per_roi_dynamic_fit_mode_contract(chunk.metadata)
        if contract_state == "malformed":
            raise RuntimeError(
                "chunk.metadata['dynamic_fit_mode_resolved_by_roi'] is present but not a "
                f"dict (got {type(chunk.metadata['dynamic_fit_mode_resolved_by_roi'])!r}); "
                "refusing to record dynamic-fit validity metrics that could mislabel any ROI "
                f"(chunk_id={chunk_id}, source_file={source_file!r})"
            )
        has_per_roi_contract = contract_state == "authoritative"
        mode_by_roi = chunk.metadata.get("dynamic_fit_mode_resolved_by_roi", {}) if has_per_roi_contract else {}
        chunk_wide_fallback = str(chunk.metadata.get("dynamic_fit_mode_resolved", "") or "")
        slope_constraint = str(getattr(self.config, "dynamic_fit_slope_constraint", "unconstrained"))
        min_slope = float(getattr(self.config, "dynamic_fit_min_slope", 0.0))
        acquisition_mode = str(getattr(self.config, "acquisition_mode", "intermittent"))
        roi_metrics_by_name: dict[str, dict] = {}

        for r_idx, roi in enumerate(chunk.channel_names):
            roi_name = str(roi)
            if has_per_roi_contract:
                if roi_name not in mode_by_roi:
                    # Authoritative: this ROI never underwent dynamic fitting.
                    continue
                fit_mode = str(mode_by_roi[roi_name])
            else:
                fit_mode = chunk_wide_fallback
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
        # Per-ROI, not chunk-wide: this function runs for every ROI
        # regardless of strategy (it always computes the Signal-Only F0
        # diagnostic candidate too), so a single chunk-wide fit_mode would
        # mislabel a non-dynamic-fit ROI's record with a mode it never used.
        # Three-state contract (regression.classify_per_roi_dynamic_fit_mode_
        # contract), matching every other reader -- except this function's
        # chosen policy for "malformed": unlike the QC/slope recorders (which
        # raise), this record must still be produced for every ROI, since it
        # is also where the Signal-Only F0 diagnostic candidate is computed
        # and that computation does not depend on dynamic-fit metadata at
        # all. Raising here would kill an unrelated, unaffected diagnostic
        # for the whole chunk over a mislabeled string field. So: on
        # "malformed", leave every ROI's dynamic_fit_mode label empty (never
        # borrow the flat value) and record an explicit, inspectable
        # contract-error field instead of silently mislabeling anything.
        contract_state = regression.classify_per_roi_dynamic_fit_mode_contract(chunk.metadata)
        dynamic_fit_mode_contract_error = None
        has_per_roi_contract = contract_state == "authoritative"
        if contract_state == "malformed":
            dynamic_fit_mode_contract_error = (
                "dynamic_fit_mode_resolved_by_roi is present but not a dict (got "
                f"{type(chunk.metadata['dynamic_fit_mode_resolved_by_roi'])!r})"
            )
        mode_by_roi = chunk.metadata.get("dynamic_fit_mode_resolved_by_roi", {}) if has_per_roi_contract else {}
        chunk_wide_fallback = str(chunk.metadata.get("dynamic_fit_mode_resolved", "") or "")
        slope_constraint = str(getattr(self.config, "dynamic_fit_slope_constraint", "unconstrained"))
        acquisition_mode = str(getattr(self.config, "acquisition_mode", "intermittent"))
        dynamic_qc_by_roi = chunk.metadata.get("dynamic_fit_validity_qc", {})
        if not isinstance(dynamic_qc_by_roi, dict):
            dynamic_qc_by_roi = {}
        signal_state_config = self._signal_state_config()
        signal_only_f0_config = self._signal_only_f0_config()
        production_qc_by_roi = chunk.metadata.get(
            "signal_only_f0_production_qc", {}
        )
        if not isinstance(production_qc_by_roi, dict):
            production_qc_by_roi = {}
        production_baseline_by_roi = chunk.metadata.get(
            "signal_only_f0_production_baseline", {}
        )
        if not isinstance(production_baseline_by_roi, dict):
            production_baseline_by_roi = {}

        records_by_roi: dict[str, dict] = {}
        for r_idx, roi in enumerate(chunk.channel_names):
            roi_name = str(roi)
            # "" (not the chunk-wide value) when this exact ROI is absent
            # from a present, authoritative per-ROI contract, OR when the
            # contract is malformed: this ROI either never underwent dynamic
            # fitting, or its true dynamic-fit status cannot be determined,
            # so no mode string truthfully describes it. The record is still
            # produced (baseline-reference and Signal-Only F0 diagnostic
            # candidates are always computed for every ROI regardless of
            # strategy) -- only the label is corrected, not the record's
            # existence. Only genuine key absence gets the flat fallback.
            if has_per_roi_contract:
                fit_mode = str(mode_by_roi.get(roi_name, ""))
            elif contract_state == "malformed":
                fit_mode = ""
            else:
                fit_mode = chunk_wide_fallback
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
                "dynamic_fit_mode_contract_error": dynamic_fit_mode_contract_error,
                "slope_constraint": slope_constraint,
                **candidate_meta,
                **metrics,
                "dynamic_fit_qc_severity": dynamic_qc.get("dynamic_fit_qc_severity", ""),
                "dynamic_fit_qc_hard_flags": dynamic_qc.get("dynamic_fit_qc_hard_flags", []),
                "dynamic_fit_qc_soft_flags": dynamic_qc.get("dynamic_fit_qc_soft_flags", []),
                "dynamic_fit_qc_flags": dynamic_qc.get("dynamic_fit_qc_flags", []),
            }
            production_qc = production_qc_by_roi.get(roi_name)
            if isinstance(production_qc, dict):
                signal_state = production_qc.get("signal_state", {})
                if not isinstance(signal_state, dict):
                    signal_state = {}
                record.update(signal_state)
                signal_only_f0_meta = {
                    key: value
                    for key, value in production_qc.items()
                    if key != "signal_state"
                }
                signal_only_f0_trace = production_baseline_by_roi.get(roi_name)
            else:
                record.update(
                    compute_signal_state_diagnostics(
                        signal=chunk.sig_raw[:, r_idx],
                        time=chunk.time_sec,
                        config=signal_state_config,
                    )
                )
                signal_only_f0 = compute_signal_only_f0_candidate(
                    signal=chunk.sig_raw[:, r_idx],
                    time=chunk.time_sec,
                    signal_state=record,
                    config=signal_only_f0_config,
                )
                signal_only_f0_trace = signal_only_f0.get("signal_only_f0_candidate")
                signal_only_f0_meta = {
                    key: value
                    for key, value in signal_only_f0.items()
                    if key != "signal_only_f0_candidate"
                }
            record.update(signal_only_f0_meta)
            record.update(
                classify_reference_candidates(
                    dynamic_qc=dynamic_qc,
                    baseline_record=record,
                )
            )
            record = apply_correction_policy_proposals(record)
            clean_record = _sanitize_metadata(record)
            self.baseline_reference_candidate_records.append(clean_record)
            records_by_roi[roi_name] = clean_record
            if candidate.get("baseline_ref_candidate_available") and candidate_trace is not None:
                trace_arr = np.asarray(candidate_trace, dtype=float).reshape(-1)
                if trace_arr.shape == chunk.sig_raw[:, r_idx].shape and np.any(np.isfinite(trace_arr)):
                    chunk.metadata.setdefault("baseline_reference_candidate_trace", {})[
                        roi_name
                    ] = trace_arr
            if signal_only_f0_trace is not None:
                trace_arr = np.asarray(signal_only_f0_trace, dtype=float).reshape(-1)
                if trace_arr.shape == chunk.sig_raw[:, r_idx].shape and np.any(np.isfinite(trace_arr)):
                    chunk.metadata.setdefault("signal_only_f0_candidate_trace", {})[
                        roi_name
                    ] = trace_arr

        if records_by_roi:
            chunk.metadata.setdefault("baseline_reference_candidate_qc", {}).update(records_by_roi)
            chunk.metadata.setdefault("signal_only_f0_candidate_qc", {}).update(records_by_roi)

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

    def _update_signal_state_diagnostics_summary(self) -> None:
        summary = summarize_signal_state_diagnostics(
            list(self.baseline_reference_candidate_records)
        )
        self.qc_summary["signal_state_diagnostics_summary"] = _sanitize_metadata(summary)

    def _update_signal_only_f0_candidate_summary(self) -> None:
        summary = summarize_signal_only_f0_candidates(
            list(self.baseline_reference_candidate_records)
        )
        self.qc_summary["signal_only_f0_candidate_summary"] = _sanitize_metadata(summary)

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

    def _update_correction_policy_proposal_summary(self) -> None:
        out = summarize_correction_policy_proposals(
            list(self.baseline_reference_candidate_records)
        )
        self.qc_summary["correction_policy_proposal_summary"] = _sanitize_metadata(out)

    def _record_dynamic_fit_slope_summaries(self, chunk: Chunk, chunk_id: int, source_file: str) -> None:
        if self.mode == "tonic" or not hasattr(chunk, "metadata") or not isinstance(chunk.metadata, dict):
            return
        # Per-ROI, not chunk-wide: a mixed-strategy chunk populates multiple
        # of the four per-mode dicts at once (grouped dispatch), one per
        # resolved mode actually used -- each ROI's own resolved mode decides
        # which dict (and which recorded dynamic_fit_mode tag) applies to it.
        # Three-state contract (regression.classify_per_roi_dynamic_fit_mode_
        # contract): "authoritative" -- an all-non-dynamic chunk's
        # dynamic_fit_mode_resolved_by_roi is present but intentionally
        # EMPTY, and that empty dict is still authoritative, never treated
        # the same as the key being entirely absent; "absent" -- the only
        # case allowed to fall back to synthesizing chunk-wide entries for
        # every channel name; "malformed" -- present but not a dict is
        # corrupt current metadata, not legacy data, and must fail closed.
        # This method runs in the production per-chunk processing path.
        contract_state = regression.classify_per_roi_dynamic_fit_mode_contract(chunk.metadata)
        if contract_state == "malformed":
            raise RuntimeError(
                "chunk.metadata['dynamic_fit_mode_resolved_by_roi'] is present but not a "
                f"dict (got {type(chunk.metadata['dynamic_fit_mode_resolved_by_roi'])!r}); "
                "refusing to record dynamic-fit slope summaries that could mislabel any ROI "
                f"(chunk_id={chunk_id}, source_file={source_file!r})"
            )
        if contract_state == "authoritative":
            effective_mode_by_roi = dict(chunk.metadata["dynamic_fit_mode_resolved_by_roi"])
        else:
            chunk_wide_fallback = str(chunk.metadata.get("dynamic_fit_mode_resolved", "") or "")
            effective_mode_by_roi = {str(roi): chunk_wide_fallback for roi in chunk.channel_names}

        acquisition_mode = str(getattr(self.config, "acquisition_mode", "intermittent"))
        for roi_name, fit_mode in effective_mode_by_roi.items():
            payload = self._dynamic_fit_roi_metadata(chunk, fit_mode, roi_name)
            summary = payload.get("slope_summary", {}) if isinstance(payload, dict) else {}
            if not isinstance(summary, dict) or not summary:
                continue
            record = {
                "roi": str(roi_name),
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
                    "roi": str(roi_name),
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

        def _normalized_abs_path(path: str) -> str:
            return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))

        raw_excluded_source_files = [
            str(path)
            for path in (getattr(self.config, "rwd_excluded_source_files", []) or [])
            if str(path).strip()
        ]
        excluded_source_files = {
            _normalized_abs_path(path) for path in raw_excluded_source_files
        }
        if self._rwd_contract_validation:
            self.qc_summary["rwd_contract_validation"] = _sanitize_metadata(
                self._rwd_contract_validation
            )

        self._authorized_exclusion = None

        if excluded_source_files:
            if len(raw_excluded_source_files) != 1 or len(excluded_source_files) != 1:
                raise ValueError(
                    "Recorded RWD final-chunk exclusion did not match discovered source files; "
                    "analysis will not continue. Expected exactly one recorded "
                    f"rwd_excluded_source_files entry, got {len(raw_excluded_source_files)}."
                )
            before_count = len(self.file_list)
            before_files = list(self.file_list)
            # The incomplete-final-chunk policy may exclude ONLY the final
            # chronological chunk. A recorded exclusion pointing anywhere else is
            # not this policy and must not silently drop a mid-recording chunk.
            final_discovered = _normalized_abs_path(before_files[-1]) if before_files else None
            if final_discovered not in excluded_source_files:
                raise ValueError(
                    "Recorded RWD final-chunk exclusion does not name the final "
                    "chronological chunk; only the incomplete final chunk may be "
                    "excluded. Recorded exclusions: "
                    f"{raw_excluded_source_files}. Final discovered chunk: {before_files[-1] if before_files else None}."
                )
            self.file_list = [
                path
                for path in self.file_list
                if _normalized_abs_path(path) not in excluded_source_files
            ]
            excluded_count = before_count - len(self.file_list)
            if excluded_count != 1:
                raise ValueError(
                    "Recorded RWD final-chunk exclusion did not match discovered source files; "
                    "analysis will not continue. Expected exactly one discovered "
                    f"file to be removed, removed {excluded_count}. "
                    f"Recorded exclusions: {raw_excluded_source_files}. "
                    f"Discovered source files: {before_files}."
                )
            if not self.file_list:
                raise ValueError(
                    "RWD final-chunk exclusion removed all discovered source files; "
                    "no validated chunks remain for analysis."
                )
            self._authorized_exclusion = before_files[-1]
            print(
                "WARNING: Excluded incomplete final RWD chunk by explicit policy. "
                f"Analysis used {len(self.file_list)} valid chunks. "
                f"{excluded_count} final chunk was excluded. "
                "See provenance/status output for details.",
                flush=True,
            )
            
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

    def _build_admitted_accountant(self, force_format: str) -> None:
        """Freeze the admitted intermittent input set into a per-chunk accountant.

        Skipped for continuous runs (they carry a window index) and for preview
        runs (an explicit user subset). For a full intermittent run the ordered
        admitted set is the frozen expected set; the one authorized final-chunk
        exclusion is appended last so it is verifiably the final chronological
        chunk.
        """
        self._admitted_accountant = None
        if self._is_continuous_mode_enabled():
            return
        if getattr(self, "run_type", "full") != "full":
            return
        if not self.file_list:
            return

        from .input_processing_completeness import (
            InputProcessingError,
            build_session_index,
            load_frozen_input_manifest,
        )

        try:
            input_format = self._get_format(self.file_list[0], force_format)
        except Exception:
            input_format = str(force_format)

        def _norm(path):
            return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))

        excluded = self._authorized_exclusion

        # Scientist-approved corrupted/missing sessions stay in the ordered set as
        # explicit missing intervals; they are not loaded (removed from the
        # processing list) but keep their chronological slot in the session index.
        raw_missing = [
            str(p)
            for p in (getattr(self.config, "authorized_missing_sessions", []) or [])
            if str(p).strip()
        ]
        missing_norm = {_norm(p) for p in raw_missing}
        discovered_norm = {_norm(p) for p in self.file_list}
        unknown = missing_norm - discovered_norm
        if unknown:
            raise ValueError(
                "Approved missing sessions were not found among the discovered "
                f"sources: {sorted(unknown)}. Only identity-bound discovered "
                "sessions may be authorized missing."
            )

        # Full chronological set = admitted-for-processing (file_list) plus the
        # single final exclusion appended last. Approved-missing sources are
        # inside file_list and are marked (not removed from the index).
        full_ordered = list(self.file_list) + ([excluded] if excluded is not None else [])
        self._intermittent_chunk_id_by_source = {
            _norm(source): index for index, source in enumerate(full_ordered)
        }
        expected_duration = float(getattr(self.config, "chunk_duration_sec", 0.0)) or None

        local_index = build_session_index(
            acquisition_mode="intermittent",
            input_format=str(input_format),
            ordered_sources=full_ordered,
            missing_sources=sorted(missing_norm),
            excluded_source=excluded,
            exclusion_policy=(POLICY_INCOMPLETE_FINAL_RWD_CHUNK if excluded is not None else ""),
            expected_duration_sec=expected_duration,
        )

        # The approved-missing sessions are never opened.
        self.file_list = [p for p in self.file_list if _norm(p) not in missing_norm]

        frozen_path = getattr(self, "_frozen_input_manifest_path", None)
        if frozen_path:
            # This process independently computed the session index above; require
            # it to be byte-identical (by digest) to the one the wrapper froze, so
            # phasic and tonic are provably held to the same expected sessions.
            manifest, error = load_frozen_input_manifest(frozen_path)
            if manifest is None:
                raise ValueError(
                    f"Run-wide frozen input manifest could not be consumed: {error}"
                )
            if manifest["digest"] != local_index["digest"]:
                raise InputProcessingError(
                    chunk_index=None,
                    source=self.file_list[0] if self.file_list else "",
                    phase="frozen_manifest_verification",
                    category="frozen_manifest_mismatch",
                    reason=(
                        "this analysis resolved a different session index than the "
                        "run-wide frozen manifest"
                    ),
                )
            self._admitted_accountant = InputProcessingAccountant.from_frozen_manifest(manifest)
        else:
            self._admitted_accountant = InputProcessingAccountant.from_frozen_manifest(local_index)

    def _session_index_for_feature_rows(self, feats_df):
        """Authoritative chronological session index per features row, by source."""
        if self._admitted_accountant is None:
            return feats_df.get('chunk_id')
        src_to_index = self._admitted_accountant.session_index_by_source()

        def _norm(path):
            return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))

        return feats_df['source_file'].map(lambda s: src_to_index.get(_norm(s)))

    def _missing_session_feature_rows(self, analyzed_rois) -> list:
        """NaN ROI-session summary rows for each approved missing session.

        One row per analyzed ROI, at the missing session's chronological index and
        source, with every analytical value NaN (never zero) and an explicit
        status -- so a missing session is never coerced into a valid zero-event
        session downstream.
        """
        if self._admitted_accountant is None:
            return []
        rows = []
        for entry in self._admitted_accountant.missing_sessions():
            for roi in analyzed_rois:
                rows.append({
                    # No cache/storage contribution; the session number and timing
                    # come from the authoritative session_index.
                    "chunk_id": float("nan"),
                    "session_index": int(entry["index"]),
                    "source_file": str(entry.get("source", "")),
                    "roi": str(roi),
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                    "mad": float("nan"),
                    "peak_count": float("nan"),
                    "auc": float("nan"),
                    "status": "missing_corrupted",
                })
        return rows

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
        if self._guided_npm_authorized_runtime is not None:
            from .guided_npm_authorized_adapter import (
                load_guided_npm_authorized_chunk,
            )

            chunk = load_guided_npm_authorized_chunk(
                self._guided_npm_authorized_runtime.authorized_input,
                entry,
                self.config,
                chunk_id,
            )
            return self._bind_authorized_chunk_chronology(chunk, entry, chunk_id)
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
        # Fail closed if an admitted source drifted from its frozen identity
        # (disappeared, resized, or replaced) since it was validated.
        if self._admitted_accountant is not None:
            self._admitted_accountant.before_load(entry, phase="load")
        fmt = self._get_format(entry, force_format)
        return load_chunk(
            entry,
            fmt,
            self.config,
            chunk_id=chunk_id,
            source_cache=self._continuous_source_cache,
        )

    def _resolve_chunk_elapsed_offset_sec(self, entry: str, chunk_id: int) -> float:
        """Resolve cross-session placement without changing legacy formats."""
        runtime = self._guided_npm_authorized_runtime
        if runtime is None:
            return self._resolve_legacy_chunk_elapsed_offset_sec(entry, chunk_id)
        from .guided_npm_worker_prelaunch_claim import stored_paths_equal

        authorized = runtime.authorized_input
        if (
            isinstance(chunk_id, bool)
            or not isinstance(chunk_id, int)
            or chunk_id < 0
            or chunk_id >= len(authorized.ordered_session_paths)
            or not stored_paths_equal(
                entry,
                authorized.ordered_session_paths[chunk_id],
                authorized.source_path_style,
            )
        ):
            raise ValueError("guided_npm_authorized_chronology_binding_mismatch")
        # sessions_per_hour remains nominal cadence/QC authority only.  It is
        # deliberately absent from actual recording-relative placement.
        return float(authorized.actual_elapsed_sec_by_chunk[chunk_id])

    def _resolve_legacy_chunk_elapsed_offset_sec(
        self, entry: str, chunk_id: int
    ) -> float:
        """Preserve the existing Pipeline-local time axis for ordinary runs."""
        return 0.0

    def _bind_authorized_chunk_chronology(
        self, chunk: Chunk, entry: str, chunk_id: int
    ) -> Chunk:
        """Compose frozen actual elapsed time with parser-authorized local time."""
        offset = self._resolve_chunk_elapsed_offset_sec(entry, chunk_id)
        local_time = np.asarray(chunk.time_sec, dtype=float).reshape(-1)
        if local_time.size == 0 or not np.all(np.isfinite(local_time)):
            raise ValueError("guided_npm_authorized_within_session_time_invalid")
        metadata = chunk.metadata
        expected = self._guided_npm_authorized_runtime.authorized_input
        if (
            metadata.get("guided_npm_chronological_position") != chunk_id
            or metadata.get("guided_npm_actual_elapsed_sec") != offset
            or metadata.get("guided_npm_nominal_expected_elapsed_sec")
            != expected.nominal_expected_elapsed_sec_by_chunk[chunk_id]
            or metadata.get("guided_npm_authoritative_source_start_time")
            != expected.authoritative_source_start_times[chunk_id]
            or metadata.get("guided_npm_cross_session_time_authority")
            != "frozen_worker_projection"
        ):
            raise ValueError("guided_npm_authorized_chunk_chronology_mismatch")
        metadata["guided_npm_within_session_start_sec"] = float(local_time[0])
        metadata["guided_npm_within_session_end_sec"] = float(local_time[-1])
        metadata["guided_npm_recording_time_start_sec"] = float(offset + local_time[0])
        metadata["guided_npm_recording_time_end_sec"] = float(offset + local_time[-1])
        metadata["output_time_basis"] = (
            "recording_relative_seconds_from_frozen_actual_elapsed_plus_"
            "authorized_within_session_time"
        )
        chunk.time_sec = local_time + offset
        chunk.validate(tolerance_frac=self.config.timestamp_cv_max)
        return chunk

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
        for fallback_chunk_id, entry in enumerate(entries):
            chunk_id = self._intermittent_chunk_id_by_source.get(
                os.path.normcase(os.path.abspath(os.path.normpath(str(entry)))),
                fallback_chunk_id,
            )
            t_load = time.perf_counter()
            try:
                chunk = self._load_entry_chunk(entry, chunk_id, force_format)
            except Exception as e:
                self._handle_pass_chunk_exception(entry, phase_name, e)
                continue
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

    def _handle_pass_chunk_exception(self, fpath, phase: str, exc: Exception) -> None:
        """React to an exception while processing an admitted chunk.

        Fail closed for a frozen admitted intermittent run: any processing
        exception for an admitted, non-excluded chunk terminates the run, so a
        chunk can never be silently omitted from the outputs. When no admitted
        accountant is active (continuous or preview), preserve the prior
        record-and-continue behavior unchanged.
        """
        if isinstance(exc, InputProcessingError):
            raise exc
        if self._admitted_accountant is not None:
            raise self._admitted_accountant.fail(
                source=self._entry_source_file(fpath),
                phase=phase,
                category="processing_exception",
                reason=str(exc),
            ) from exc
        logging.warning(f"{phase}: Skipping {fpath} due to error: {exc}")
        if not any(x['file'] == fpath for x in self.qc_summary['failed_chunks']):
            self.qc_summary['failed_chunks'].append({'file': fpath, 'error': str(exc)})

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
                    self._handle_pass_chunk_exception(fpath, "pass1", e)
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
                    self._handle_pass_chunk_exception(fpath, "pass1a", e)
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
                    self._handle_pass_chunk_exception(fpath, "pass1b", e)
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
        if self.mode == 'tonic' and self.per_roi_correction is None:
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
                    except Exception as e:
                        self._handle_pass_chunk_exception(fpath, "tonic_pass1c", e)
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
                    except Exception as e:
                        self._handle_pass_chunk_exception(fpath, "tonic_pass1c", e)
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
        
        if self.mode == 'tonic' and self.per_roi_correction is None:
             self._process_chunk_tonic(chunk, chunk_id)
        else:
             # Native per-ROI correction is shared by phasic and tonic.
             strategy_map, dispatch_map = self._resolve_correction_map_for_chunk(
                 chunk.channel_names
             )
             t_reg = time.perf_counter()
             uv_fit, delta_f = regression.fit_chunk_dynamic(
                 chunk,
                 self.config,
                 # Native tonic consumes the same canonical per-session
                 # correction engine as phasic. ``mode='tonic'`` is reserved
                 # for the positive legacy recording-global tonic route.
                 mode="phasic" if self.per_roi_correction is not None else self.mode,
                 per_roi_correction=dispatch_map,
             )
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
             if chunk.dff is None:
                 chunk.dff = np.full_like(chunk.sig_raw, np.nan, dtype=float)

             # Assemble both correction families into one canonical, stable
             # original-column-order result before feature extraction. A
             # Signal-Only ROI never enters regression and never uses UV data.
             consumed_by_roi = {}
             production_baselines = {}
             production_qc_by_roi = {}
             for roi_index, roi_name_raw in enumerate(chunk.channel_names):
                 roi_name = str(roi_name_raw)
                 spec = strategy_map[roi_name]
                 consumed = {
                     "roi_id": roi_name,
                     "strategy_family": str(spec.strategy_family),
                     "selected_strategy": str(spec.selected_strategy),
                     "dynamic_fit_mode": (
                         str(spec.dynamic_fit_mode)
                         if spec.dynamic_fit_mode is not None
                         else None
                     ),
                     "parameter_identity": str(spec.parameter_identity),
                     "evidence_identity": str(spec.evidence_identity),
                     "execution_status": "consumed",
                 }
                 if spec.strategy_family == "signal_only_f0":
                     (
                         canonical_delta_f,
                         canonical_dff,
                         production_baseline,
                         state,
                         production_qc,
                     ) = (
                         self._compute_signal_only_f0_production(
                             chunk,
                             roi_index=roi_index,
                             roi_id=roi_name,
                             chunk_id=chunk_id,
                         )
                     )
                     chunk.delta_f[:, roi_index] = canonical_delta_f
                     chunk.dff[:, roi_index] = canonical_dff
                     # Keep the exact production baseline out of the JSON-like
                     # QC record; it is persisted as an aligned dataset below.
                     production_baselines[roi_name] = np.asarray(
                         production_baseline, dtype=float
                     )
                     production_qc_by_roi[roi_name] = dict(production_qc)
                     production_qc_by_roi[roi_name]["signal_state"] = dict(state)
                     consumed.update(
                         {
                             "production_baseline_dataset": "signal_only_f0_baseline",
                             "production_baseline_source": "signal_only_f0_candidate_uncapped",
                             "production_qc_key": "signal_only_f0_production_qc",
                         }
                     )
                 consumed_by_roi[roi_name] = consumed

             chunk.metadata["correction_strategy_consumed_by_roi"] = consumed_by_roi
             chunk.metadata["signal_only_f0_production_baseline"] = production_baselines
             chunk.metadata["signal_only_f0_production_qc"] = production_qc_by_roi
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


    def _resolve_correction_map_for_chunk(
        self, channel_names: list[str]
    ) -> tuple[dict[str, PerRoiCorrectionSpec], dict[str, PerRoiCorrectionSpec] | None]:
        """Resolve the immutable Pipeline correction selection for one chunk.

        The returned second value is the argument passed to grouped regression:
        ``None`` deliberately preserves the accepted legacy translation path;
        an explicit map is passed unchanged so Config.dynamic_fit_mode cannot
        override it.
        """
        names = [str(name) for name in channel_names]
        if self.per_roi_correction is None:
            frozen_map = getattr(self, "_requested_correction_strategy_map", None)
            if isinstance(frozen_map, dict) and set(frozen_map) == set(names):
                strategy_map = dict(frozen_map)
            else:
                strategy_map = regression.build_uniform_per_roi_correction_map(
                    names,
                    getattr(self.config, "dynamic_fit_mode", "rolling_local_regression"),
                )
            return strategy_map, None

        strategy_map = dict(self.per_roi_correction)
        regression.validate_per_roi_correction_map(names, strategy_map)
        return strategy_map, strategy_map

    def _build_requested_correction_provenance(
        self, included_rois: list[str] | tuple[str, ...]
    ) -> dict:
        """Freeze the exact correction selection requested for this run.

        The map is resolved once, after ROI inclusion/exclusion is final, and
        then copied into the report and run metadata.  Explicit maps are never
        reconstructed from ``Config.dynamic_fit_mode``; only legacy callers
        that supplied no map receive the documented uniform translation.
        """
        names = [str(roi) for roi in included_rois]
        if len(names) != len(set(names)):
            raise ValueError(f"included ROI identities are duplicated: {names}")

        if self.per_roi_correction is None:
            strategy_map = regression.build_uniform_per_roi_correction_map(
                names,
                getattr(self.config, "dynamic_fit_mode", "rolling_local_regression"),
            )
            source = "legacy_uniform_translation"
        else:
            strategy_map = dict(self.per_roi_correction)
            regression.validate_per_roi_correction_map(names, strategy_map)
            source = "explicit_per_roi_map"

        # Keep the resolved objects used by every subsequent chunk in this
        # run.  The objects are frozen dataclasses; retaining this private copy
        # prevents a legacy uniform translation from being recomputed from a
        # later-mutated Config.dynamic_fit_mode.
        self._requested_correction_strategy_map = dict(strategy_map)

        records = []
        for roi in names:
            spec = strategy_map[roi]
            record = {
                "roi_id": str(spec.roi_id),
                "strategy_family": str(spec.strategy_family),
                "selected_strategy": str(spec.selected_strategy),
                "dynamic_fit_mode": (
                    str(spec.dynamic_fit_mode)
                    if spec.dynamic_fit_mode is not None
                    else None
                ),
                "parameter_identity": str(spec.parameter_identity),
                "evidence_identity": str(spec.evidence_identity),
            }
            records.append(record)

        return {
            "schema_version": CORRECTION_PROVENANCE_SCHEMA_VERSION,
            "source": source,
            "analysis_mode": str(self.mode),
            "included_roi_ids": list(names),
            "requested_by_roi": records,
            "finite_coverage_fraction": float(
                getattr(self.config, "signal_only_f0_min_coverage_fraction", 0.80)
            ),
        }

    def _signal_state_config(self) -> dict:
        return {
            key: getattr(self.config, key)
            for key in _SIGNAL_STATE_CONFIG_KEYS
            if hasattr(self.config, key)
        }

    def _signal_only_f0_config(self) -> dict:
        return {
            key: getattr(self.config, key)
            for key in _SIGNAL_ONLY_F0_CONFIG_KEYS
            if hasattr(self.config, key)
        }

    def _compute_signal_only_f0_production(
        self,
        chunk: Chunk,
        *,
        roi_index: int,
        roi_id: str,
        chunk_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
        """Compute and validate one canonical native Signal-Only F0 trace.

        The production baseline is the exact uncapped candidate used by the
        accepted standalone Signal-Only F0 formula. The candidate is computed
        once here; the diagnostic recorder reuses the returned metadata and
        baseline rather than invoking the candidate a second time.
        """
        signal = np.asarray(chunk.sig_raw[:, roi_index], dtype=float).reshape(-1)
        state = compute_signal_state_diagnostics(
            signal=signal,
            time=chunk.time_sec,
            config=self._signal_state_config(),
        )
        result = compute_signal_only_f0_candidate(
            signal=signal,
            time=chunk.time_sec,
            signal_state=state,
            config=self._signal_only_f0_config(),
            return_uncapped_candidate=True,
        )

        baseline_raw = result.get("signal_only_f0_candidate_uncapped")
        expected_sample_count = int(signal.size)
        finite_signal = np.isfinite(signal)
        n_finite_signal = int(np.sum(finite_signal))
        coverage_fraction = float(
            getattr(self.config, "signal_only_f0_min_coverage_fraction", 0.80)
        )
        if not np.isfinite(coverage_fraction) or not 0.0 < coverage_fraction <= 1.0:
            raise CorrectionProcessingError(
                roi_id=roi_id,
                chunk_id=chunk_id,
                source_file=chunk.source_file,
                selected_strategy="signal_only_f0",
                reason=(
                    "invalid signal-only coverage policy: "
                    f"signal_only_f0_min_coverage_fraction={coverage_fraction!r}"
                ),
            )
        # Coverage is measured against the expected processed trace length,
        # never only against samples that happened to be finite. The explicit
        # policy allows up to (1 - coverage_fraction) missing samples,
        # including ordinary edge NaNs; there is no additional hidden edge
        # allowance.
        min_required = max(
            10, int(np.ceil(coverage_fraction * expected_sample_count))
        )
        if n_finite_signal < min_required:
            raise CorrectionProcessingError(
                roi_id=roi_id,
                chunk_id=chunk_id,
                source_file=chunk.source_file,
                selected_strategy="signal_only_f0",
                reason=(
                    "raw signal finite coverage is insufficient: "
                    f"{n_finite_signal}/{expected_sample_count} valid samples, "
                    f"{min_required} required"
                ),
            )
        if str(result.get("signal_only_f0_status", "")) != "ok":
            reason = str(
                result.get("signal_only_f0_warning")
                or result.get("signal_only_f0_status")
                or "candidate_unavailable"
            )
            raise CorrectionProcessingError(
                roi_id=roi_id,
                chunk_id=chunk_id,
                source_file=chunk.source_file,
                selected_strategy="signal_only_f0",
                reason=reason,
            )
        if baseline_raw is None:
            raise CorrectionProcessingError(
                roi_id=roi_id,
                chunk_id=chunk_id,
                source_file=chunk.source_file,
                selected_strategy="signal_only_f0",
                reason="candidate did not return the production F0 baseline",
            )

        baseline = np.asarray(baseline_raw, dtype=float).reshape(-1)
        if baseline.shape != signal.shape:
            raise CorrectionProcessingError(
                roi_id=roi_id,
                chunk_id=chunk_id,
                source_file=chunk.source_file,
                selected_strategy="signal_only_f0",
                reason=(
                    "production F0 baseline shape mismatch: "
                    f"{baseline.shape} versus signal {signal.shape}"
                ),
            )
        valid = finite_signal & np.isfinite(baseline)
        baseline_finite_count = int(np.sum(np.isfinite(baseline)))
        valid_count = int(np.sum(valid))
        if baseline_finite_count < min_required:
            raise CorrectionProcessingError(
                roi_id=roi_id,
                chunk_id=chunk_id,
                source_file=chunk.source_file,
                selected_strategy="signal_only_f0",
                reason=(
                    "production F0 finite coverage is insufficient: "
                    f"{baseline_finite_count}/{expected_sample_count} finite samples, "
                    f"{min_required} required"
                ),
            )
        if valid_count < min_required:
            raise CorrectionProcessingError(
                roi_id=roi_id,
                chunk_id=chunk_id,
                source_file=chunk.source_file,
                selected_strategy="signal_only_f0",
                reason=(
                    "production F0 coverage is insufficient: "
                    f"{valid_count} valid samples, {min_required} required"
                ),
            )
        min_f0 = float(getattr(self.config, "f0_min_value", 1e-9))
        if np.any(baseline[valid] <= min_f0):
            raise CorrectionProcessingError(
                roi_id=roi_id,
                chunk_id=chunk_id,
                source_file=chunk.source_file,
                selected_strategy="signal_only_f0",
                reason=(
                    "production F0 baseline contains non-positive or too-small "
                    f"values (minimum allowed {min_f0})"
                ),
            )

        canonical_delta_f = np.full_like(signal, np.nan, dtype=float)
        canonical_dff = np.full_like(signal, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            canonical_delta_f[valid] = signal[valid] - baseline[valid]
            canonical_dff[valid] = 100.0 * canonical_delta_f[valid] / baseline[valid]
        if int(np.sum(np.isfinite(canonical_dff))) < min_required:
            raise CorrectionProcessingError(
                roi_id=roi_id,
                chunk_id=chunk_id,
                source_file=chunk.source_file,
                selected_strategy="signal_only_f0",
                reason="canonical dF/F has insufficient finite coverage",
            )

        qc = {
            key: value
            for key, value in result.items()
            if key not in {
                "signal_only_f0_candidate",
                "signal_only_f0_candidate_uncapped",
            }
        }
        qc.update(
            {
                "signal_only_f0_production_available": True,
                "signal_only_f0_production_baseline_source": "signal_only_f0_candidate_uncapped",
                "signal_only_f0_production_formula": "100 * (signal - f0) / f0",
                "signal_only_f0_production_baseline_p05": float(
                    np.percentile(baseline[valid], 5.0)
                ),
                "signal_only_f0_production_baseline_p50": float(
                    np.percentile(baseline[valid], 50.0)
                ),
                "signal_only_f0_production_baseline_p95": float(
                    np.percentile(baseline[valid], 95.0)
                ),
                "signal_only_f0_production_valid_sample_count": valid_count,
                "signal_only_f0_production_expected_sample_count": expected_sample_count,
                "signal_only_f0_production_baseline_finite_count": baseline_finite_count,
                "signal_only_f0_production_dff_finite_count": int(
                    np.sum(np.isfinite(canonical_dff))
                ),
                "signal_only_f0_production_min_required_samples": min_required,
                "signal_only_f0_production_valid_fraction": float(
                    valid_count / max(1, expected_sample_count)
                ),
            }
        )
        return canonical_delta_f, canonical_dff, baseline, state, qc


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
             # For a frozen admitted intermittent run, an admitted chunk absent
             # from the Pass 1 manifest means it never processed. That is a silent
             # omission, so fail closed rather than "ignore" it.
             if self._admitted_accountant is not None:
                 raise self._admitted_accountant.fail(
                     source=self._entry_source_file(new_files[0]),
                     phase="pass2_manifest_check",
                     category="unprocessed_admitted_chunk",
                     reason="an admitted chunk was not processed in Pass 1",
                 )

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
                    feats_df = feature_extraction.extract_features(
                        chunk, self.config, per_roi_config=self.per_roi_feature_config
                    )
                    if self._guided_npm_authorized_runtime is not None:
                        feats_df["guided_npm_actual_elapsed_sec"] = chunk.metadata[
                            "guided_npm_actual_elapsed_sec"
                        ]
                        feats_df["guided_npm_nominal_expected_elapsed_sec"] = chunk.metadata[
                            "guided_npm_nominal_expected_elapsed_sec"
                        ]
                        feats_df["recording_time_start_sec"] = chunk.metadata[
                            "guided_npm_recording_time_start_sec"
                        ]
                        feats_df["recording_time_end_sec"] = chunk.metadata[
                            "guided_npm_recording_time_end_sec"
                        ]
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

                # This admitted chunk reached a successful terminal disposition:
                # its per-ROI features and its trace-cache contribution are done.
                if self._admitted_accountant is not None:
                    self._admitted_accountant.mark_processed(
                        self._entry_source_file(fpath), cache_chunk_id=i
                    )

                # Legacy VIZ was here, now moved out of loop for strict resolution/loading

            except InputProcessingError:
                # Already identity-bound (e.g. source drift at load); fail closed.
                raise
            except Exception as e:
                # Fail fast: an admitted chunk that cannot be processed in Pass 2
                # terminates the run rather than being omitted from the outputs.
                if self._admitted_accountant is not None:
                    raise self._admitted_accountant.fail(
                        source=self._entry_source_file(fpath),
                        phase="pass2",
                        category="processing_exception",
                        reason=str(e),
                    ) from e
                raise RuntimeError(f"Pass 2: Cannot reliably read manifest file {fpath} successfully processed in Pass 1. Error: {e}")

        # Write the input-processing completeness record fail-closed: finalize()
        # raises if any admitted, non-excluded chunk lacks a processing record, so
        # a record on disk means the admitted set was fully accounted for.
        if self._admitted_accountant is not None:
            self._admitted_accountant.write(output_dir)

        if all_features and self.mode != 'tonic':
            t_feats_write = time.perf_counter()
            full_feats = pd.concat(all_features, ignore_index=True)
            # A processed session is valid even with zero detected events; the
            # status column keeps that distinct from an approved missing session.
            full_feats['status'] = 'valid'
            # chunk_id is a cache/storage position, not the session number. The
            # authoritative chronological session number comes from the session
            # index; carry it so consumers never treat a storage id as a session.
            full_feats['session_index'] = self._session_index_for_feature_rows(full_feats)
            feats_dir = os.path.join(output_dir, 'features')
            os.makedirs(feats_dir, exist_ok=True)

            # Every feature-extracting run records the settings actually
            # consumed for every analyzed ROI, so a missing file can never be
            # misread as "this run had no per-ROI settings" (4J16k39b).
            analyzed_rois = sorted(set(full_feats['roi'].astype(str)))

            # Approved missing sessions keep a chronological row per analyzed ROI
            # with NaN analytical values (never zero) and an explicit status, so a
            # missing session is never read as a valid zero-event session.
            missing_rows = self._missing_session_feature_rows(analyzed_rois)
            if missing_rows:
                full_feats = pd.concat(
                    [full_feats, pd.DataFrame(missing_rows)], ignore_index=True
                )

            full_feats.to_csv(os.path.join(feats_dir, 'features.csv'), index=False)
            pass2_features_csv_write_sec += (time.perf_counter() - t_feats_write)
            self._write_feature_event_provenance(feats_dir, analyzed_rois)
            self._stamp_feature_event_provenance_contract(output_dir)

        if self.dynamic_fit_qc_records and self.mode != 'tonic':
            qc_rows = []
            for rec in self.dynamic_fit_qc_records:
                row = dict(rec)
                for list_key in (
                    "dynamic_fit_qc_flags",
                    "dynamic_fit_qc_hard_flags",
                    "dynamic_fit_qc_soft_flags",
                    "reference_comparison_flags",
                    "signal_state_flags",
                    "proposal_flags_conservative",
                    "proposal_flags_balanced",
                    "proposal_flags_liberal",
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
                    "reference_comparison_flags",
                    "signal_state_flags",
                    "signal_only_f0_flags",
                    "proposal_flags_conservative",
                    "proposal_flags_balanced",
                    "proposal_flags_liberal",
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
        self._update_signal_state_diagnostics_summary()
        self._update_signal_only_f0_candidate_summary()
        self._update_reference_candidate_comparison_summary()
        self._update_correction_policy_proposal_summary()
            
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
            'rwd_contract_validation': _sanitize_metadata(self._rwd_contract_validation),
            # Versioned, run-level requested correction authority.  This is the
            # exact map frozen before Pass 1/Pass 2, including legacy uniform
            # translation provenance when no explicit map was supplied.
            'correction_provenance': getattr(
                self,
                '_requested_correction_provenance',
                None,
            ),
        }
        if self._guided_npm_authorized_runtime is not None:
            authorized = self._guided_npm_authorized_runtime.authorized_input
            run_meta["guided_npm_cross_session_chronology"] = {
                "authority": "frozen_worker_projection",
                "within_session_output_time_basis": authorized.output_time_basis,
                "combined_output_time_basis": (
                    "recording_relative_seconds_from_frozen_actual_elapsed_plus_"
                    "authorized_within_session_time"
                ),
                "chronological_positions": authorized.chronological_positions,
                "ordered_session_paths": authorized.ordered_session_paths,
                "actual_elapsed_sec_by_chunk": authorized.actual_elapsed_sec_by_chunk,
                "nominal_expected_elapsed_sec_by_chunk": (
                    authorized.nominal_expected_elapsed_sec_by_chunk
                ),
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
                
    def _write_feature_event_provenance(self, feats_dir: str, analyzed_rois) -> None:
        """Write the feature-detection settings actually consumed per ROI.

        Written by EVERY run that extracts features, including runs with no
        per-ROI override: a Default-only run records one Default entry per
        analyzed ROI (4J16k39b). Absence of this file therefore never means
        "Default-only" -- it means the run predates this contract.

        The payload is built from the Config objects this Pipeline actually
        used (`self.config` and `self.per_roi_feature_config`), never
        reconstructed from the Guided plan or startup artifact, so Default-only
        and Custom runs follow the same evidence path. `override_config_fields`
        and the profile id are carried through from
        `self.per_roi_feature_provenance` for description only; they never
        determine the effective settings.
        """
        from photometry_pipeline.feature_event_provenance import (
            FEATURE_EVENT_PROVENANCE_FILENAME,
            build_feature_event_provenance_payload,
        )

        payload = build_feature_event_provenance_payload(
            base_config=self.config,
            analyzed_rois=list(analyzed_rois),
            per_roi_feature_config=self.per_roi_feature_config or {},
            per_roi_source_details=dict(self.per_roi_feature_provenance or {}),
        )
        path = os.path.join(feats_dir, FEATURE_EVENT_PROVENANCE_FILENAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._feature_event_provenance_payload = payload

    def _stamp_feature_event_provenance_contract(self, output_dir: str) -> None:
        """Record the explicit contract-version signal in run_report.json.

        Consumers classify a run as current-or-legacy from THIS signal, never
        from whether the provenance file happens to exist. Failing to stamp it
        would let a current run be misclassified as legacy and silently verified
        against global settings, so this raises rather than degrading.
        """
        from photometry_pipeline.feature_event_provenance import (
            FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION,
            FEATURE_EVENT_PROVENANCE_FILENAME,
        )

        payload = getattr(self, "_feature_event_provenance_payload", None) or {}
        report_path = os.path.join(output_dir, "run_report.json")
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["feature_event_provenance"] = {
            "contract_version": FEATURE_EVENT_PROVENANCE_CONTRACT_VERSION,
            "schema_version": payload.get("schema_version", ""),
            "relative_path": os.path.join("features", FEATURE_EVENT_PROVENANCE_FILENAME),
            "global_default_config_digest": payload.get(
                "global_default_config_digest", ""
            ),
            "roi_count": len(payload.get("rois", [])),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

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

    def run_guided_npm_authorized(
        self,
        authorized_runtime,
        output_dir: str,
        *,
        traces_only: bool = False,
    ):
        """Run exact Guided NPM authority without legacy discovery or inference."""
        from .guided_npm_authorized_adapter import (
            GuidedNpmAuthorizedRuntime,
            verify_guided_npm_authorized_input,
        )

        if type(authorized_runtime) is not GuidedNpmAuthorizedRuntime:
            raise TypeError("guided_npm_authorized_runtime_invalid")
        if self._guided_npm_authorized_runtime is not None:
            raise RuntimeError("guided_npm_authorized_runtime_already_active")
        verify_guided_npm_authorized_input(authorized_runtime.authorized_input)
        if (
            self.config != authorized_runtime.config
            or self.mode != authorized_runtime.mode
            or dict(self.per_roi_correction or {})
            != authorized_runtime.per_roi_correction
            or self.per_roi_feature_config
            != authorized_runtime.per_roi_feature_config
            or self.per_roi_feature_provenance
            != authorized_runtime.per_roi_feature_provenance
            or output_dir
            != authorized_runtime.authorized_input.run_directory_path
        ):
            raise ValueError("guided_npm_authorized_pipeline_binding_mismatch")
        self._guided_npm_authorized_runtime = authorized_runtime
        try:
            return self.run(
                authorized_runtime.authorized_input.source_root_path,
                output_dir,
                force_format="npm",
                recursive=False,
                glob_pattern="__guided_npm_authorized_no_discovery__",
                include_rois=None,
                exclude_rois=None,
                traces_only=traces_only,
                sessions_per_hour=authorized_runtime.authorized_input.sessions_per_hour,
            )
        finally:
            self._guided_npm_authorized_runtime = None

    def run(
        self,
        input_dir: str,
        output_dir: str,
        force_format: str = 'auto',
        recursive: bool = False,
        glob_pattern: str = "*.csv",
        include_rois: List[str] = None,
        exclude_rois: List[str] = None,
        traces_only: bool = False,
        emitter=None,
        sessions_per_hour: int = None,
        guided_manifest_path: str | None = None,
        frozen_input_manifest_path: str | None = None,
    ):
        # Lazy import to avoid GUI side effects at module level
        from .viz import plots
        authorized_npm_runtime = self._guided_npm_authorized_runtime
        if authorized_npm_runtime is not None:
            authorized = authorized_npm_runtime.authorized_input
            if (
                force_format != "npm"
                or recursive
                or glob_pattern != "__guided_npm_authorized_no_discovery__"
                or include_rois is not None
                or exclude_rois is not None
                or guided_manifest_path is not None
                or frozen_input_manifest_path is not None
                or input_dir != authorized.source_root_path
                or self.config != authorized_npm_runtime.config
                or sessions_per_hour != authorized.sessions_per_hour
            ):
                raise ValueError("guided_npm_authorized_route_conflict")
        self._frozen_input_manifest_path = frozen_input_manifest_path
        run_started = time.perf_counter()
        if self._is_phasic_timing_enabled():
            self._phasic_started_at = run_started

        guided_verification = None
        guided_facts = None
        if guided_manifest_path is not None:
            loaded_manifest = load_guided_candidate_manifest(guided_manifest_path)
            if not loaded_manifest.accepted or loaded_manifest.manifest is None:
                detail = (
                    loaded_manifest.blocking_issues[0].category
                    if loaded_manifest.blocking_issues
                    else "guided_manifest_load_failed"
                )
                raise RuntimeError(f"Guided manifest verification refused: {detail}")
            manifest = loaded_manifest.manifest
            guided_facts = build_guided_manifest_current_facts(
                source_root=input_dir,
                config=self.config,
                manifest_included_roi_ids=manifest.included_roi_ids,
            )
            guided_verification = verify_guided_candidate_manifest_consumption(
                manifest=manifest,
                source_root=input_dir,
                current_candidates=guided_facts.current_candidates,
                current_roi_inventory=guided_facts.current_roi_inventory,
                cli_context=GuidedManifestCliContext(
                    input_format=force_format,
                    mode=self.mode,
                    run_type="full",
                    traces_only=bool(traces_only),
                    discover=False,
                    validate_only=False,
                    overwrite=False,
                    preview_first_n=self.config.preview_first_n,
                    requested_include_rois=(
                        tuple(include_rois) if include_rois is not None else None
                    ),
                    requested_exclude_rois=(
                        tuple(exclude_rois) if exclude_rois is not None else ()
                    ),
                ),
            )
            if not guided_verification.accepted:
                detail = (
                    guided_verification.blocking_issues[0].category
                    if guided_verification.blocking_issues
                    else "guided_manifest_verification_failed"
                )
                raise RuntimeError(f"Guided manifest verification refused: {detail}")
            self.file_list = [
                item.absolute_path
                for item in guided_verification.verified_candidates
            ]
            # Guided verification restores the authoritative full candidate
            # order. Reapply the already validated final exclusion so the
            # excluded source remains in session accounting but is never
            # reopened by either analysis branch.
            guided_exclusions = [
                str(path)
                for path in (
                    getattr(self.config, "rwd_excluded_source_files", []) or []
                )
                if str(path).strip()
            ]
            if guided_exclusions:
                if len(guided_exclusions) != 1 or not self.file_list:
                    raise ValueError(
                        "Guided final exclusion must identify exactly one final source."
                    )
                excluded_norm = os.path.normcase(
                    os.path.abspath(os.path.normpath(guided_exclusions[0]))
                )
                final_norm = os.path.normcase(
                    os.path.abspath(os.path.normpath(self.file_list[-1]))
                )
                if excluded_norm != final_norm:
                    raise ValueError(
                        "Guided final exclusion does not identify the final chronological source."
                    )
                self._authorized_exclusion = self.file_list[-1]
                self.file_list = [
                    path
                    for path in self.file_list
                    if os.path.normcase(
                        os.path.abspath(os.path.normpath(path))
                    )
                    != excluded_norm
                ]
            self._selected_rois = list(
                guided_verification.verified_included_roi_ids
            )
            self._guided_manifest_verification = guided_verification

        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'qc'), exist_ok=True)
        if authorized_npm_runtime is not None:
            self.file_list = list(
                authorized_npm_runtime.authorized_input.ordered_session_paths
            )
            self._add_phasic_phase_bucket("phase.input_discovery", 0.0)
        elif guided_verification is None:
            t_discovery = time.perf_counter()
            self.discover_files(input_dir, recursive, glob_pattern, force_format=force_format)
            self._add_phasic_phase_bucket("phase.input_discovery", time.perf_counter() - t_discovery)
        else:
            self._add_phasic_phase_bucket("phase.input_discovery", 0.0)
        self._set_phasic_metric("files_discovered", len(self.file_list))
        
        # --- Preview Mode: limit to first N sessions ---
        n_total_discovered = len(self.file_list)
        preview_first_n = self.config.preview_first_n
        t_preview = time.perf_counter()
        if authorized_npm_runtime is not None:
            self.run_type = "full"
            self.preview_info = None
        elif guided_verification is not None:
            self.run_type = "full"
            self.preview_info = None
        elif preview_first_n is not None:
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

        # Freeze the admitted intermittent input set with per-chunk identity, so
        # every admitted chunk must reach one terminal disposition and no chunk
        # can be silently omitted (4J16k41 / C8). Continuous runs carry their own
        # window index (4J16k40); preview runs are an explicit subset selection.
        self._build_admitted_accountant(force_format)
        
        t_roi_resolve = time.perf_counter()
        if authorized_npm_runtime is not None:
            authorized = authorized_npm_runtime.authorized_input
            discovered_rois = list(authorized.canonical_roi_ids)
            selected_rois = list(authorized.selected_canonical_roi_ids)
            selected_set = set(selected_rois)
            include_rois = list(selected_rois)
            exclude_rois = [
                roi for roi in discovered_rois if roi not in selected_set
            ]
            self._add_phasic_phase_bucket("phase.roi_discovery_read_chunks", 0.0)
            self._set_phasic_metric("roi_discovery_source", "guided_npm_authorized")
        elif guided_verification is not None:
            assert guided_facts is not None
            discovered_rois = list(
                guided_facts.current_roi_inventory.discovered_roi_ids
            )
            selected_rois = list(guided_verification.verified_included_roi_ids)
            include_rois = list(guided_verification.verified_included_roi_ids)
            exclude_rois = list(guided_verification.verified_excluded_roi_ids)
            self._add_phasic_phase_bucket("phase.roi_discovery_read_chunks", 0.0)
            self._set_phasic_metric("roi_discovery_source", "guided_manifest")
        else:
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

            # Intersection over all valid chunks, preserving first-chunk order.
            channel_sets = [set(cx) for cx in channels_seen]
            discovered_rois = [
                r for r in channels_seen[0] if all(r in cs for cs in channel_sets)
            ]
            selected_rois = list(discovered_rois)
            if include_rois is not None:
                missing = [r for r in include_rois if r not in discovered_rois]
                if missing:
                    raise ValueError(
                        "Validation Error: Included ROIs not found in "
                        f"discovered ROIs: {missing}"
                    )
                selected_rois = [r for r in discovered_rois if r in include_rois]
            if exclude_rois is not None:
                missing = [r for r in exclude_rois if r not in discovered_rois]
                if missing:
                    logging.warning(
                        "Excluded ROIs not found in discovered ROIs "
                        f"(ignoring): {missing}"
                    )
                selected_rois = [
                    r for r in selected_rois if r not in exclude_rois
                ]
             
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
        # Freeze the requested correction authority before any production pass
        # begins.  The same immutable snapshot is written to the report and to
        # run_metadata; downstream completion verification never infers it from
        # Config.dynamic_fit_mode or from consumed cache attributes.
        self._requested_correction_provenance = (
            self._build_requested_correction_provenance(selected_rois)
        )

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
        _append_run_report_section(
            output_dir,
            "correction_provenance",
            self._requested_correction_provenance,
        )
        if authorized_npm_runtime is not None:
            authorized = authorized_npm_runtime.authorized_input
            _append_run_report_section(
                output_dir,
                "guided_npm_cross_session_chronology",
                {
                    "authority": "frozen_worker_projection",
                    "within_session_output_time_basis": authorized.output_time_basis,
                    "combined_output_time_basis": (
                        "recording_relative_seconds_from_frozen_actual_elapsed_plus_"
                        "authorized_within_session_time"
                    ),
                    "sessions_per_hour_role": "nominal_cadence_only",
                    "sessions": [
                        {
                            "chunk_id": position,
                            "chronological_position": position,
                            "source_path": path,
                            "authoritative_source_start_time": start,
                            "actual_elapsed_sec": actual,
                            "nominal_expected_elapsed_sec": nominal,
                        }
                        for position, path, start, actual, nominal in zip(
                            authorized.chronological_positions,
                            authorized.ordered_session_paths,
                            authorized.authoritative_source_start_times,
                            authorized.actual_elapsed_sec_by_chunk,
                            authorized.nominal_expected_elapsed_sec_by_chunk,
                        )
                    ],
                },
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
