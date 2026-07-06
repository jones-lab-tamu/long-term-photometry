from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Optional
import numpy as np

from photometry_pipeline.feature_event_config import (
    validate_feature_event_config_fields,
)
from photometry_pipeline.core.feature_extraction import (
    apply_peak_prefilter,
    compute_detection_threshold_bounds,
    get_peak_indices_for_trace,
)


class GuidedFeaturePreviewUnsupportedError(ValueError):
    """Raised when a requested trace/event-signal/strategy combination is not supported."""
    pass


@dataclass(frozen=True)
class GuidedFeaturePreviewTraceRequest:
    roi_id: str
    event_signal: str  # allowed: "dff", "delta_f"
    correction_strategy: str
    feature_profile_id: str
    feature_settings: dict[str, Any]
    dynamic_fit_mode: Optional[str] = None
    segment_start_sec: Optional[float] = None
    segment_duration_sec: Optional[float] = None
    setup_signature: Optional[str] = None
    correction_signature: Optional[str] = None


@dataclass(frozen=True)
class GuidedFeaturePreviewTrace:
    roi_id: str
    time_sec: np.ndarray
    trace: np.ndarray
    fs_hz: float
    event_signal: str
    trace_kind: str
    correction_strategy: str
    trace_identity: dict[str, Any]
    correction_identity: dict[str, Any]
    source_kind: str
    dynamic_fit_mode: Optional[str] = None
    preview_only: bool = True
    production_analysis: bool = False


@dataclass(frozen=True)
class GuidedFeatureDetectionPreviewResult:
    roi_id: str
    event_signal: str
    time_sec: np.ndarray
    trace: np.ndarray
    prefiltered_trace: Optional[np.ndarray]
    threshold_upper: Optional[float]
    threshold_lower: Optional[float]
    positive_peak_indices: np.ndarray | list[int]
    negative_peak_indices: np.ndarray | list[int]
    positive_peak_times_sec: np.ndarray | list[float]
    negative_peak_times_sec: np.ndarray | list[float]
    feature_profile_id: str
    feature_settings_digest: str
    trace_identity: dict[str, Any]
    correction_identity: dict[str, Any]
    detector_identity: dict[str, Any]
    warnings: list[str]
    preview_only: bool = True
    production_analysis: bool = False
    feature_extraction_run: bool = False


class SettingsConfigAdapter:
    """Adapts a raw feature settings dictionary to look like a Config object."""
    def __init__(self, settings: dict[str, Any]):
        self._settings = settings

    def __getattr__(self, name: str) -> Any:
        if name in self._settings:
            return self._settings[name]
        # Core defaults matching Config definitions where not provided
        defaults = {
            "peak_pre_filter": "none",
            "signal_excursion_polarity": "positive",
            "peak_min_prominence_k": 0.0,
            "peak_min_width_sec": 0.0,
            "event_auc_baseline": "zero",
            "lowpass_hz": 1.0,
            "filter_order": 3,
        }
        if name in defaults:
            return defaults[name]
        raise AttributeError(f"SettingsConfigAdapter has no attribute '{name}'")


def compute_settings_digest(settings: dict[str, Any]) -> str:
    """Generate a deterministic sha256 hex digest of settings."""
    canonical_json = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def build_feature_detection_preview_from_trace(
    *,
    roi_id: str,
    time_sec: np.ndarray,
    trace: np.ndarray,
    fs_hz: float,
    event_signal: str,
    feature_settings: dict[str, Any],
    feature_profile_id: str,
    trace_identity: dict[str, Any],
    correction_identity: dict[str, Any],
) -> GuidedFeatureDetectionPreviewResult:
    """Pure, no-write peak detection preview runner."""
    # 1. Validation
    if len(time_sec) != len(trace):
        raise ValueError(
            f"Length mismatch between time_sec ({len(time_sec)}) and trace ({len(trace)})"
        )
    if len(trace) == 0:
        raise ValueError("Cannot perform feature detection preview on empty trace.")
    if fs_hz <= 0 or not np.isfinite(fs_hz):
        raise ValueError(f"fs_hz must be finite and positive, found {fs_hz}")
    if event_signal not in {"dff", "delta_f"}:
        raise ValueError(
            f"event_signal must be 'dff' or 'delta_f', found '{event_signal}'"
        )

    # Validate settings dict using config fields check
    validation_errors = validate_feature_event_config_fields(feature_settings)
    if validation_errors:
        raise ValueError(f"Invalid feature_settings: {validation_errors}")

    # Check for missing required detector fields
    required_fields = {"peak_threshold_method", "peak_min_distance_sec"}
    missing = required_fields - set(feature_settings)
    if missing:
        raise ValueError(f"Missing required detector settings fields: {sorted(missing)}")

    method = feature_settings.get("peak_threshold_method")
    if method == "absolute" and "peak_threshold_abs" not in feature_settings:
        raise ValueError("Missing 'peak_threshold_abs' for absolute threshold method.")
    if method in {"mean_std", "median_mad"} and "peak_threshold_k" not in feature_settings:
        raise ValueError(f"Missing 'peak_threshold_k' for '{method}' threshold method.")
    if method == "percentile" and "peak_threshold_percentile" not in feature_settings:
        raise ValueError("Missing 'peak_threshold_percentile' for percentile threshold method.")

    # Check polarity is valid
    polarity = feature_settings.get("signal_excursion_polarity", "positive")
    if polarity not in {"positive", "negative", "both"}:
        raise ValueError(f"Unknown signal excursion polarity: '{polarity}'")

    # Check for too few finite samples
    finite_mask = np.isfinite(trace)
    if np.count_nonzero(finite_mask) < 2:
        raise ValueError(
            "Need at least 2 finite samples to compute detection threshold bounds."
        )

    # 2. Adaptation and Execution
    config_adapter = SettingsConfigAdapter(feature_settings)
    prefiltered, _ = apply_peak_prefilter(trace, fs_hz, config_adapter)

    # Compute threshold bounds
    clean_finite = prefiltered[np.isfinite(prefiltered)]
    if len(clean_finite) < 2:
        raise ValueError(
            "Need at least 2 finite samples in prefiltered trace to compute detection."
        )

    threshold_bounds = compute_detection_threshold_bounds(clean_finite, config_adapter)
    thresh_upper = float(threshold_bounds["upper"])
    thresh_lower = float(threshold_bounds["lower"])

    peak_idx, pol = get_peak_indices_for_trace(
        trace,
        fs_hz,
        config_adapter,
        trace_use=prefiltered,
        threshold=thresh_upper,
        threshold_lower=thresh_lower,
        return_polarities=True,
    )

    # Separate positive and negative peaks
    pos_mask = pol == 1
    neg_mask = pol == -1

    pos_idx = peak_idx[pos_mask]
    neg_idx = peak_idx[neg_mask]

    pos_times = time_sec[pos_idx]
    neg_times = time_sec[neg_idx]

    # Deterministic metadata digests and version mappings
    settings_digest = compute_settings_digest(feature_settings)
    detector_identity = {
        "function": "get_peak_indices_for_trace",
        "module": "photometry_pipeline.core.feature_extraction",
        "version": "1.0",
    }

    # Populate warnings list
    warnings_list = []
    # Production DD4 check
    if method == "median_mad" and threshold_bounds.get("sigma_robust", 0.0) == 0.0:
        warnings_list.append(f"Zero robust variance in ROI '{roi_id}'")

    return GuidedFeatureDetectionPreviewResult(
        roi_id=roi_id,
        event_signal=event_signal,
        time_sec=time_sec,
        trace=trace,
        prefiltered_trace=prefiltered,
        threshold_upper=thresh_upper,
        threshold_lower=thresh_lower,
        positive_peak_indices=pos_idx,
        negative_peak_indices=neg_idx,
        positive_peak_times_sec=pos_times,
        negative_peak_times_sec=neg_times,
        feature_profile_id=feature_profile_id,
        feature_settings_digest=settings_digest,
        trace_identity=trace_identity,
        correction_identity=correction_identity,
        detector_identity=detector_identity,
        warnings=warnings_list,
    )


def resolve_guided_feature_preview_trace(
    request: GuidedFeaturePreviewTraceRequest,
    available_trace_context: dict[str, Any],
) -> GuidedFeaturePreviewTrace:
    """Resolve the preview trace request using context according to the support matrix."""
    roi_id = request.roi_id
    event_signal = request.event_signal
    strategy = request.correction_strategy
    fit_mode = request.dynamic_fit_mode

    # Normalize strategy and fit_mode if strategy is one of the specific dynamic fit modes
    from photometry_pipeline.guided_new_analysis_plan import FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES
    if strategy in FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES:
        fit_mode = strategy
        strategy = "dynamic_fit"

    if strategy == "dynamic_fit":
        if event_signal == "dff":
            mode_map = available_trace_context.get("dynamic_dff", {})
            trace_key = (roi_id, fit_mode)
            if trace_key not in mode_map:
                raise GuidedFeaturePreviewUnsupportedError(
                    "No matching local correction-preview dynamic-fit dF/F "
                    f"trace found for ROI '{roi_id}' and mode '{fit_mode}'"
                )
            trace_data = mode_map[trace_key]
            trace_identity = trace_data.get("trace_identity", {})
            correction_identity = trace_data.get(
                "correction_identity", {}
            )
            if trace_data.get("stale", False) or not trace_data.get(
                "current", True
            ):
                raise GuidedFeaturePreviewUnsupportedError(
                    f"Dynamic-fit dF/F trace for ROI '{roi_id}' is stale."
                )
            if not (
                isinstance(trace_identity, dict)
                and trace_identity.get("roi_id") == roi_id
                and trace_identity.get("trace_source")
                == "local_correction_preview_dff"
                and trace_identity.get("dff_scale") == "fractional_ratio"
                and trace_identity.get("preview_only") is True
                and trace_identity.get("production_analysis") is False
                and isinstance(correction_identity, dict)
                and correction_identity.get("correction_strategy")
                == "dynamic_fit"
                and correction_identity.get("dynamic_fit_mode") == fit_mode
            ):
                raise GuidedFeaturePreviewUnsupportedError(
                    "Dynamic-fit dF/F trace identity does not match the "
                    "requested local preview strategy."
                )
            return GuidedFeaturePreviewTrace(
                roi_id=roi_id,
                time_sec=trace_data["time_sec"],
                trace=trace_data["trace"],
                fs_hz=trace_data["fs_hz"],
                event_signal="dff",
                trace_kind="local_correction_preview_dff",
                correction_strategy=strategy,
                dynamic_fit_mode=fit_mode,
                trace_identity=trace_identity,
                correction_identity=correction_identity,
                source_kind="local_correction_preview",
            )
        elif event_signal == "delta_f":
            # Lookup in dynamic_delta_f
            mode_map = available_trace_context.get("dynamic_delta_f", {})
            trace_key = (roi_id, fit_mode)
            if trace_key not in mode_map:
                raise GuidedFeaturePreviewUnsupportedError(
                    f"No matching dynamic delta_f trace found in context for ROI '{roi_id}' and mode '{fit_mode}'"
                )
            trace_data = mode_map[trace_key]
            if trace_data.get("stale", False) or not trace_data.get("current", True):
                raise GuidedFeaturePreviewUnsupportedError(
                    f"Dynamic delta_f trace for ROI '{roi_id}' is stale or not current."
                )

            return GuidedFeaturePreviewTrace(
                roi_id=roi_id,
                time_sec=trace_data["time_sec"],
                trace=trace_data["trace"],
                fs_hz=trace_data["fs_hz"],
                event_signal="delta_f",
                trace_kind="delta_f",
                correction_strategy=strategy,
                dynamic_fit_mode=fit_mode,
                trace_identity=trace_data["trace_identity"],
                correction_identity=trace_data["correction_identity"],
                source_kind=trace_data.get("source_kind", "preview_derived"),
            )
        else:
            raise GuidedFeaturePreviewUnsupportedError(
                f"Unsupported event signal '{event_signal}' for dynamic_fit strategy."
            )

    elif strategy == "signal_only_f0":
        if event_signal == "dff":
            # Lookup in signal_only_dff
            roi_map = available_trace_context.get("signal_only_dff", {})
            if roi_id not in roi_map:
                raise GuidedFeaturePreviewUnsupportedError(
                    f"No matching Signal-Only F0 dF/F trace found in context for ROI '{roi_id}'"
                )
            trace_data = roi_map[roi_id]
            if trace_data.get("stale", False) or not trace_data.get("current", True):
                raise GuidedFeaturePreviewUnsupportedError(
                    f"Signal-Only F0 dF/F trace for ROI '{roi_id}' is stale or not current."
                )

            return GuidedFeaturePreviewTrace(
                roi_id=roi_id,
                time_sec=trace_data["time_sec"],
                trace=trace_data["trace"],
                fs_hz=trace_data["fs_hz"],
                event_signal="dff",
                trace_kind="dff",
                correction_strategy=strategy,
                trace_identity=trace_data["trace_identity"],
                correction_identity=trace_data["correction_identity"],
                source_kind=trace_data.get("source_kind", "preview_derived"),
            )
        elif event_signal == "delta_f":
            raise GuidedFeaturePreviewUnsupportedError(
                "Signal-Only delta_f is unsupported"
            )
        else:
            raise GuidedFeaturePreviewUnsupportedError(
                f"Unsupported event signal '{event_signal}' for signal_only_f0 strategy."
            )

    else:
        raise GuidedFeaturePreviewUnsupportedError(
            f"Unsupported correction strategy '{strategy}' for feature detection preview."
        )


def build_guided_feature_detection_preview(
    *,
    trace_request: GuidedFeaturePreviewTraceRequest,
    available_trace_context: dict[str, Any],
) -> GuidedFeatureDetectionPreviewResult:
    """Compose trace resolution and detector preview logic."""
    # 1. Resolve the trace
    preview_trace = resolve_guided_feature_preview_trace(
        trace_request, available_trace_context
    )

    # 2. Run detection preview
    return build_feature_detection_preview_from_trace(
        roi_id=preview_trace.roi_id,
        time_sec=preview_trace.time_sec,
        trace=preview_trace.trace,
        fs_hz=preview_trace.fs_hz,
        event_signal=preview_trace.event_signal,
        feature_settings=trace_request.feature_settings,
        feature_profile_id=trace_request.feature_profile_id,
        trace_identity=preview_trace.trace_identity,
        correction_identity=preview_trace.correction_identity,
    )
