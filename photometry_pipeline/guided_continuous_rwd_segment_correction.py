"""Pure CR1-C4b correction of one accepted continuous-RWD segment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Callable, Mapping

import numpy as np

from photometry_pipeline.config import Config
from photometry_pipeline.core import preprocessing, regression
from photometry_pipeline.core.signal_only_f0_candidate import (
    DEFAULTS as _SIGNAL_ONLY_F0_CANDIDATE_DEFAULTS,
)
from photometry_pipeline.core.signal_state_diagnostics import (
    DEFAULTS as _SIGNAL_STATE_DIAGNOSTIC_DEFAULTS,
)
from photometry_pipeline.core.types import Chunk, PerRoiCorrectionSpec
from photometry_pipeline.guided_continuous_rwd_correction_segments import (
    DYNAMIC_STRATEGIES,
    RAW_SEGMENT_SCHEMA_NAME,
    RAW_SEGMENT_SCHEMA_VERSION,
    SIGNAL_ONLY_STRATEGY,
    GuidedContinuousRwdCorrectionSegmentPlan,
    GuidedContinuousRwdDynamicF0Authority,
    GuidedContinuousRwdRawCorrectionSegment,
    _resolve_accepted_correction_context,
    _validate_dynamic_f0_authority,
    _validate_segment_plan,
)
from photometry_pipeline.guided_continuous_rwd_block_plan import (
    build_guided_continuous_rwd_block_plan,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBinding,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    GuidedContinuousRwdTargetGridDescription,
)
from photometry_pipeline.guided_continuous_rwd_projection import _target_coordinates
from photometry_pipeline.guided_execution_payloads import (
    GUIDED_CONFIG_DEFAULT_OVERRIDES,
    GuidedExecutionStartupMappingContract,
    build_guided_execution_startup_mapping_contract,
)
from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.io.rwd_continuous_projection_reader import _validate_authorities
from photometry_pipeline.signal_only_f0 import (
    SignalOnlyF0ProductionError,
    compute_signal_only_f0_production,
)


SCHEMA_NAME = "guided_continuous_rwd_corrected_segment"
SCHEMA_VERSION = "v1"
POLICY_NAME = "segment-local-mixed-correction"
POLICY_VERSION = "v1"
RESULT_IDENTITY_DOMAIN = "guided-continuous-rwd-corrected-segment:v1"

SETTINGS_SCHEMA_NAME = "guided_continuous_rwd_segment_correction_settings"
SETTINGS_SCHEMA_VERSION = "v1"
SETTINGS_POLICY_NAME = "c4b-segment-correction-effective-settings"
SETTINGS_POLICY_VERSION = "v1"
SETTINGS_IDENTITY_DOMAIN = "guided-continuous-rwd-segment-correction-settings:v1"

# Config fields consumed directly by preprocessing.lowpass_filter_with_meta,
# regression.fit_chunk_dynamic (global-linear/robust-event-reject/adaptive-
# event-gated dispatch), and the Signal-Only F0 production path. Deliberately
# excludes fields that exist on Config but are never read on those code paths
# for the accepted DYNAMIC_STRATEGIES/SIGNAL_ONLY_STRATEGY subset (e.g.
# baseline_subtract_before_fit only applies to the rolling_* dynamic-fit
# modes, config.dynamic_fit_mode is never consulted because C4b always
# supplies an explicit per_roi_correction map, and baseline_method/
# baseline_percentile/seed only feed C4a's F0 reservoir, already bound via
# fixed_correction_settings_identity and dynamic_f0_authority_identity).
_C4B_DYNAMIC_FIT_SETTING_NAMES = (
    "lowpass_hz",
    "filter_order",
    "bleach_correction_mode",
    "dynamic_fit_slope_constraint",
    "dynamic_fit_min_slope",
    "robust_event_reject_max_iters",
    "robust_event_reject_residual_z_thresh",
    "robust_event_reject_local_var_window_sec",
    "robust_event_reject_local_var_ratio_thresh",
    "robust_event_reject_min_keep_fraction",
    "adaptive_event_gate_residual_z_thresh",
    "adaptive_event_gate_local_var_window_sec",
    "adaptive_event_gate_local_var_ratio_thresh",
    "adaptive_event_gate_smooth_window_sec",
    "adaptive_event_gate_min_trust_fraction",
    "adaptive_event_gate_freeze_interp_method",
    "f0_min_value",
)
# Every key compute_signal_state_diagnostics/compute_signal_only_f0_candidate
# actually reads is sourced directly from their own DEFAULTS dicts, rather
# than transcribed by hand, so this list cannot silently drift from what the
# engines consume.
_C4B_SIGNAL_ONLY_SETTING_NAMES = tuple(
    sorted(set(_SIGNAL_STATE_DIAGNOSTIC_DEFAULTS) | set(_SIGNAL_ONLY_F0_CANDIDATE_DEFAULTS))
)
_C4B_LOAD_BEARING_SETTING_NAMES = (
    _C4B_DYNAMIC_FIT_SETTING_NAMES + _C4B_SIGNAL_ONLY_SETTING_NAMES
)

# signal_excursion_polarity is read by the robust/adaptive dynamic-fit engines
# (regression._compute_dynamic_fit_ref_robust_global_event_reject and
# _compute_dynamic_fit_ref_adaptive_event_gated_regression) but is classified
# as a Guided CONFIG_DISPOSITION_CONFIRMED_FEATURE field scoped to feature/
# event detection (guided_production_mapping.FEATURE_EVENT_TYPED_FIELD_CONFIG_MAP),
# not a correction setting, and is therefore absent from
# GUIDED_CONFIG_DEFAULT_OVERRIDES/fixed_config_overrides. Nothing in the
# accepted correction context ever sources it from anywhere but the bare
# Config() dataclass default, so it is pinned here as an explicit immutable
# policy constant (matching that default) rather than left as an unbound
# Config() fallback.
_C4B_FIXED_POLICY_CONSTANTS: Mapping[str, Any] = {
    "signal_excursion_polarity": "positive",
}


def _resolve_segment_correction_settings(
    startup_mapping_contract: object,
) -> tuple[Config, str]:
    canonical = build_guided_execution_startup_mapping_contract()
    if not isinstance(startup_mapping_contract, GuidedExecutionStartupMappingContract):
        _raise(
            "accepted_correction_binding_mismatch",
            "startup_mapping_contract has the wrong type.",
        )
    if startup_mapping_contract != canonical:
        _raise(
            "accepted_correction_binding_mismatch",
            "startup_mapping_contract does not match the canonical accepted "
            "Guided contract.",
        )
    overrides = {
        item.name: item.value for item in startup_mapping_contract.fixed_config_overrides
    }
    settings: dict[str, Any] = {}
    for name in _C4B_LOAD_BEARING_SETTING_NAMES:
        if name not in overrides or overrides[name] != GUIDED_CONFIG_DEFAULT_OVERRIDES[name]:
            _raise(
                "accepted_correction_binding_mismatch",
                f"{name} does not match the accepted Guided correction authority.",
                field=name,
            )
        settings[name] = overrides[name]
    config = Config()
    for name, value in settings.items():
        setattr(config, name, value)
    for name, value in _C4B_FIXED_POLICY_CONSTANTS.items():
        setattr(config, name, value)
    payload = {
        "schema_name": SETTINGS_SCHEMA_NAME,
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "policy_name": SETTINGS_POLICY_NAME,
        "policy_version": SETTINGS_POLICY_VERSION,
        "contract_version": startup_mapping_contract.contract_version,
        "config_mapping_contract_version": (
            startup_mapping_contract.config_mapping_contract_version
        ),
        "settings": settings,
        "fixed_policy_constants": dict(_C4B_FIXED_POLICY_CONSTANTS),
    }
    identity = hashlib.sha256(
        SETTINGS_IDENTITY_DOMAIN.encode("utf-8") + b"\x00" + encode_canonical_value(payload)
    ).hexdigest()
    return config, identity


REFERENCE_FITTED_CONTROL = "fitted_control"
REFERENCE_SIGNAL_DERIVED_F0 = "signal_derived_time_varying_f0"

ERROR_CATEGORIES = frozenset(
    {
        "accepted_correction_binding_mismatch",
        "invalid_raw_segment_binding",
        "invalid_dynamic_f0_authority",
        "segment_strategy_mismatch",
        "missing_dynamic_f0",
        "invalid_dynamic_f0",
        "dynamic_filter_failure",
        "dynamic_fit_failure",
        "signal_only_failure",
        "correction_output_shape_mismatch",
        "nonfinite_correction_output",
        "segment_correction_interrupted",
        "result_identity_mismatch",
    }
)


class GuidedContinuousRwdSegmentCorrectionError(ValueError):
    """A narrow refusal while correcting one accepted C4a segment."""

    def __init__(self, category: str, message: str, **context: Any) -> None:
        if category not in ERROR_CATEGORIES:
            raise ValueError(f"Unsupported CR1-C4b error category: {category!r}")
        super().__init__(message)
        self.category = category
        self.context = dict(context)


@dataclass(frozen=True)
class GuidedContinuousRwdPerRoiCorrectionResult:
    roi_id: str
    strategy_family: str
    selected_strategy: str
    dynamic_fit_mode: str | None
    parameter_identity: str
    evidence_identity: str
    reference_kind: str
    applied_strategy: str
    fallback_path: tuple[str, ...]
    qc_json: str
    scalar_f0: float | None


@dataclass(frozen=True)
class GuidedContinuousRwdCorrectedSegment:
    schema_name: str
    schema_version: str
    policy_name: str
    policy_version: str
    recording_identity: str
    source_content_identity: str
    target_grid_identity: str
    correction_segment_plan_identity: str
    dynamic_f0_authority_identity: str
    accepted_guided_plan_identity: str
    correction_payload_identity: str
    fixed_correction_settings_identity: str
    segment_correction_settings_identity: str
    segment_index: int
    start_target_index: int
    stop_target_index: int
    included_roi_ids: tuple[str, ...]
    target_elapsed_seconds: np.ndarray
    raw_control_values: np.ndarray
    raw_signal_values: np.ndarray
    correction_reference_values: np.ndarray
    delta_f_values: np.ndarray
    dff_values: np.ndarray
    per_roi_results: tuple[GuidedContinuousRwdPerRoiCorrectionResult, ...]
    result_identity: str


def _raise(category: str, message: str, **context: Any) -> None:
    raise GuidedContinuousRwdSegmentCorrectionError(category, message, **context)


def _check_cancellation(callback: Callable[[], bool] | None) -> None:
    if callback is None:
        return
    if not callable(callback):
        _raise(
            "segment_correction_interrupted",
            "cancellation_requested must be callable or None.",
        )
    try:
        interrupted = callback()
    except Exception as exc:
        raise GuidedContinuousRwdSegmentCorrectionError(
            "segment_correction_interrupted",
            "Cancellation callback failed.",
            reason=str(exc),
        ) from exc
    if type(interrupted) is not bool:
        _raise(
            "segment_correction_interrupted",
            "Cancellation callback must return bool.",
        )
    if interrupted:
        _raise("segment_correction_interrupted", "Segment correction was cancelled.")


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(b"\x00float64\x00")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite_count": int(np.sum(np.isfinite(arr))) if np.issubdtype(arr.dtype, np.number) else None,
            "true_count": int(np.sum(arr)) if arr.dtype == np.bool_ else None,
        }
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _qc_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _identity_payload(result: GuidedContinuousRwdCorrectedSegment) -> dict[str, Any]:
    return {
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "policy_name": result.policy_name,
        "policy_version": result.policy_version,
        "recording_identity": result.recording_identity,
        "source_content_identity": result.source_content_identity,
        "target_grid_identity": result.target_grid_identity,
        "correction_segment_plan_identity": result.correction_segment_plan_identity,
        "dynamic_f0_authority_identity": result.dynamic_f0_authority_identity,
        "accepted_guided_plan_identity": result.accepted_guided_plan_identity,
        "correction_payload_identity": result.correction_payload_identity,
        "fixed_correction_settings_identity": result.fixed_correction_settings_identity,
        "segment_correction_settings_identity": result.segment_correction_settings_identity,
        "segment_index": result.segment_index,
        "start_target_index": result.start_target_index,
        "stop_target_index": result.stop_target_index,
        "included_roi_ids": result.included_roi_ids,
        "per_roi_results": [asdict(item) for item in result.per_roi_results],
        "arrays": {
            "target_elapsed_seconds": [list(result.target_elapsed_seconds.shape), _array_digest(result.target_elapsed_seconds)],
            "raw_control_values": [list(result.raw_control_values.shape), _array_digest(result.raw_control_values)],
            "raw_signal_values": [list(result.raw_signal_values.shape), _array_digest(result.raw_signal_values)],
            "correction_reference_values": [list(result.correction_reference_values.shape), _array_digest(result.correction_reference_values)],
            "delta_f_values": [list(result.delta_f_values.shape), _array_digest(result.delta_f_values)],
            "dff_values": [list(result.dff_values.shape), _array_digest(result.dff_values)],
        },
    }


def _compute_result_identity(result: GuidedContinuousRwdCorrectedSegment) -> str:
    return hashlib.sha256(
        RESULT_IDENTITY_DOMAIN.encode("utf-8")
        + b"\x00"
        + encode_canonical_value(_identity_payload(result))
    ).hexdigest()


def _validate_raw_segment(
    raw_segment: object,
    *,
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    segment_plan: GuidedContinuousRwdCorrectionSegmentPlan,
) -> GuidedContinuousRwdRawCorrectionSegment:
    if not isinstance(raw_segment, GuidedContinuousRwdRawCorrectionSegment):
        _raise("invalid_raw_segment_binding", "Raw segment has the wrong type.")
    if not isinstance(raw_segment.segment_index, int) or isinstance(raw_segment.segment_index, bool):
        _raise("invalid_raw_segment_binding", "Raw segment index is invalid.")
    if not 0 <= raw_segment.segment_index < segment_plan.segment_count:
        _raise("invalid_raw_segment_binding", "Raw segment index is outside the plan.")
    descriptor = segment_plan.descriptors[raw_segment.segment_index]
    recording = review_binding.recording
    expected_roi_order = recording.roi.included_roi_ids
    if (
        raw_segment.schema_name != RAW_SEGMENT_SCHEMA_NAME
        or raw_segment.schema_version != RAW_SEGMENT_SCHEMA_VERSION
        or raw_segment.recording_identity != recording.recording_identity
        or raw_segment.source_content_identity != recording.source.source_content_identity
        or raw_segment.target_grid_identity != target_grid.target_grid_identity
        or raw_segment.correction_segment_plan_identity != segment_plan.plan_identity
        or raw_segment.start_target_index != descriptor.start_target_index
        or raw_segment.stop_target_index != descriptor.stop_target_index
        or raw_segment.included_roi_ids != expected_roi_order
    ):
        _raise("invalid_raw_segment_binding", "Raw segment provenance or range is invalid.")
    expected_shape = (descriptor.sample_count, len(expected_roi_order))
    expected_time = _target_coordinates(
        target_grid, descriptor.start_target_index, descriptor.stop_target_index
    )
    for name, value, shape in (
        ("target_elapsed_seconds", raw_segment.target_elapsed_seconds, (descriptor.sample_count,)),
        ("control_values", raw_segment.control_values, expected_shape),
        ("signal_values", raw_segment.signal_values, expected_shape),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(np.float64)
            or value.shape != shape
            or value.flags.writeable
            or not np.all(np.isfinite(value))
        ):
            _raise("invalid_raw_segment_binding", f"Raw segment {name} is invalid.")
    if not np.array_equal(raw_segment.target_elapsed_seconds, expected_time):
        _raise("invalid_raw_segment_binding", "Raw segment target coordinates are invalid.")
    return raw_segment


def _dynamic_qc_and_fallback(
    chunk: Chunk,
    roi_id: str,
    requested: str,
    raw_segment: GuidedContinuousRwdRawCorrectionSegment,
) -> tuple[str, tuple[str, ...], str]:
    metadata = chunk.metadata or {}
    warnings = tuple(str(item) for item in metadata.get("qc_warnings", ()) or ())
    if requested == "global_linear_regression":
        detail = metadata.get("dynamic_fit_global_linear", {}).get(roi_id, {})
        applied = requested
        path = (requested,)
    elif requested == "robust_global_event_reject":
        detail = metadata.get("dynamic_fit_event_reject", {}).get(roi_id, {})
        if detail.get("fallback_failed"):
            _raise(
                "dynamic_fit_failure",
                "Dynamic fallback chain was exhausted.",
                roi=roi_id,
                segment_index=raw_segment.segment_index,
                start_target_index=raw_segment.start_target_index,
                stop_target_index=raw_segment.stop_target_index,
                requested_strategy=requested,
                attempted_fallback_chain=(requested, "global_linear_regression"),
                failed_stage="global_linear_regression",
                reason=str(detail.get("stop_reason", "fallback_failed")),
            )
        used_fallback = bool(detail.get("fallback_to_global_linear", False))
        applied = "global_linear_regression" if used_fallback else requested
        path = (requested, applied) if used_fallback else (requested,)
    else:
        detail = metadata.get("dynamic_fit_adaptive_event_gated", {}).get(roi_id, {})
        fallback = str(detail.get("fallback_mode", "none"))
        if detail.get("fallback_failed") or fallback.endswith("_failed"):
            _raise(
                "dynamic_fit_failure",
                "Dynamic fallback chain was exhausted.",
                roi=roi_id,
                segment_index=raw_segment.segment_index,
                start_target_index=raw_segment.start_target_index,
                stop_target_index=raw_segment.stop_target_index,
                requested_strategy=requested,
                attempted_fallback_chain=(requested, "robust_global_event_reject", "global_linear_regression"),
                failed_stage="global_linear_regression",
                reason=fallback,
            )
        if fallback == "robust_global_event_reject":
            applied = fallback
            path = (requested, fallback)
        elif fallback == "global_linear_regression":
            applied = fallback
            path = (requested, "robust_global_event_reject", fallback)
        else:
            applied = requested
            path = (requested,)
    return applied, path, _qc_json({"detail": detail, "warnings": warnings})


def _correct_dynamic_roi(
    raw_segment: GuidedContinuousRwdRawCorrectionSegment,
    roi_index: int,
    spec: PerRoiCorrectionSpec,
    config: object,
    scalar_f0: float,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, tuple[str, ...], str]:
    local_time = raw_segment.target_elapsed_seconds - raw_segment.target_elapsed_seconds[0]
    control = raw_segment.control_values[:, roi_index : roi_index + 1]
    signal = raw_segment.signal_values[:, roi_index : roi_index + 1]
    chunk = Chunk(
        chunk_id=raw_segment.segment_index,
        source_file="continuous-rwd-accepted-segment",
        format="rwd",
        time_sec=local_time,
        uv_raw=control.copy(),
        sig_raw=signal.copy(),
        fs_hz=sampling_rate_hz,
        channel_names=[spec.roi_id],
        metadata={},
    )
    try:
        chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(
            chunk.uv_raw, chunk.fs_hz, config
        )
        chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(
            chunk.sig_raw, chunk.fs_hz, config
        )
    except Exception as exc:
        raise GuidedContinuousRwdSegmentCorrectionError(
            "dynamic_filter_failure",
            "Segment-local low-pass filtering failed.",
            roi=spec.roi_id,
            segment_index=raw_segment.segment_index,
            start_target_index=raw_segment.start_target_index,
            stop_target_index=raw_segment.stop_target_index,
            requested_strategy=spec.selected_strategy,
            failed_stage="lowpass_filter",
            reason=str(exc),
        ) from exc
    try:
        fitted, delta = regression.fit_chunk_dynamic(
            chunk,
            config,
            mode="phasic",
            per_roi_correction={spec.roi_id: spec},
        )
    except Exception as exc:
        raise GuidedContinuousRwdSegmentCorrectionError(
            "dynamic_fit_failure",
            "Segment-local dynamic fit failed.",
            roi=spec.roi_id,
            segment_index=raw_segment.segment_index,
            start_target_index=raw_segment.start_target_index,
            stop_target_index=raw_segment.stop_target_index,
            requested_strategy=spec.selected_strategy,
            attempted_fallback_chain={
                "global_linear_regression": ("global_linear_regression",),
                "robust_global_event_reject": ("robust_global_event_reject", "global_linear_regression"),
                "adaptive_event_gated_regression": ("adaptive_event_gated_regression", "robust_global_event_reject", "global_linear_regression"),
            }[spec.selected_strategy],
            failed_stage="dynamic_fit",
            reason=str(exc),
        ) from exc
    reference = np.asarray(fitted[:, 0], dtype=np.float64)
    delta_f = np.asarray(delta[:, 0], dtype=np.float64)
    dff = np.asarray(100.0 * delta_f / scalar_f0, dtype=np.float64)
    applied, fallback, qc = _dynamic_qc_and_fallback(
        chunk, spec.roi_id, spec.selected_strategy, raw_segment
    )
    if not (
        np.all(np.isfinite(reference))
        and np.all(np.isfinite(delta_f))
        and np.all(np.isfinite(dff))
    ):
        _raise(
            "dynamic_fit_failure",
            "Dynamic fallback chain produced unusable output.",
            roi=spec.roi_id,
            segment_index=raw_segment.segment_index,
            start_target_index=raw_segment.start_target_index,
            stop_target_index=raw_segment.stop_target_index,
            requested_strategy=spec.selected_strategy,
            attempted_fallback_chain=fallback,
            failed_stage=applied,
            reason="nonfinite_dynamic_output",
        )
    return reference, delta_f, dff, applied, fallback, qc


def _validate_result(
    result: object,
    *,
    raw_segment: GuidedContinuousRwdRawCorrectionSegment,
    dynamic_f0_authority: GuidedContinuousRwdDynamicF0Authority,
    accepted: object,
    segment_correction_settings_identity: str,
) -> None:
    if not isinstance(result, GuidedContinuousRwdCorrectedSegment):
        _raise("result_identity_mismatch", "Corrected segment has the wrong type.")
    if (
        result.schema_name != SCHEMA_NAME
        or result.schema_version != SCHEMA_VERSION
        or result.policy_name != POLICY_NAME
        or result.policy_version != POLICY_VERSION
        or result.recording_identity != raw_segment.recording_identity
        or result.source_content_identity != raw_segment.source_content_identity
        or result.target_grid_identity != raw_segment.target_grid_identity
        or result.correction_segment_plan_identity != raw_segment.correction_segment_plan_identity
        or result.dynamic_f0_authority_identity != dynamic_f0_authority.authority_identity
        or result.accepted_guided_plan_identity != accepted.accepted_guided_plan_identity
        or result.correction_payload_identity != accepted.correction_payload_identity
        or result.fixed_correction_settings_identity != accepted.fixed_correction_settings_identity
        or result.segment_correction_settings_identity != segment_correction_settings_identity
        or result.segment_index != raw_segment.segment_index
        or result.start_target_index != raw_segment.start_target_index
        or result.stop_target_index != raw_segment.stop_target_index
        or result.included_roi_ids != raw_segment.included_roi_ids
    ):
        _raise("result_identity_mismatch", "Corrected segment provenance is invalid.")
    shape = raw_segment.signal_values.shape
    for name, value, expected in (
        ("target_elapsed_seconds", result.target_elapsed_seconds, raw_segment.target_elapsed_seconds),
        ("raw_control_values", result.raw_control_values, raw_segment.control_values),
        ("raw_signal_values", result.raw_signal_values, raw_segment.signal_values),
    ):
        if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.flags.writeable or not np.array_equal(value, expected):
            _raise("correction_output_shape_mismatch", f"{name} does not preserve the raw segment.")
    for name, value in (
        ("correction_reference_values", result.correction_reference_values),
        ("delta_f_values", result.delta_f_values),
        ("dff_values", result.dff_values),
    ):
        if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.shape != shape or value.flags.writeable:
            _raise("correction_output_shape_mismatch", f"{name} is not a canonical segment matrix.")
        if not np.all(np.isfinite(value)):
            _raise("nonfinite_correction_output", f"{name} contains nonfinite values.")
    if (
        not isinstance(result.per_roi_results, tuple)
        or len(result.per_roi_results) != len(result.included_roi_ids)
        or any(
            not isinstance(item, GuidedContinuousRwdPerRoiCorrectionResult)
            for item in result.per_roi_results
        )
    ):
        _raise("segment_strategy_mismatch", "Per-ROI result contract is invalid.")
    if tuple(item.roi_id for item in result.per_roi_results) != result.included_roi_ids:
        _raise("segment_strategy_mismatch", "Per-ROI result order is not canonical.")
    value_by_roi = {item.roi_id: item.scalar_f0 for item in dynamic_f0_authority.values}
    binding_by_roi = {item.roi_id: item for item in accepted.bindings}
    for index, item in enumerate(result.per_roi_results):
        binding = binding_by_roi[item.roi_id]
        if (
            item.strategy_family != binding.strategy_family
            or item.selected_strategy != binding.selected_strategy
            or item.dynamic_fit_mode != binding.dynamic_fit_mode
            or item.parameter_identity != binding.parameter_identity
            or item.evidence_identity != binding.evidence_identity
        ):
            _raise("segment_strategy_mismatch", "Per-ROI correction metadata was substituted.", roi=item.roi_id)
        if (
            not isinstance(item.fallback_path, tuple)
            or not item.fallback_path
            or item.fallback_path[0] != item.selected_strategy
            or item.applied_strategy != item.fallback_path[-1]
            or not isinstance(item.qc_json, str)
        ):
            _raise("segment_strategy_mismatch", "Per-ROI execution evidence is invalid.", roi=item.roi_id)
        try:
            json.loads(item.qc_json)
        except (TypeError, ValueError) as exc:
            raise GuidedContinuousRwdSegmentCorrectionError(
                "segment_strategy_mismatch",
                "Per-ROI QC evidence is invalid.",
                roi=item.roi_id,
            ) from exc
        expected_delta = result.raw_signal_values[:, index] - result.correction_reference_values[:, index]
        if not np.allclose(result.delta_f_values[:, index], expected_delta, rtol=1e-12, atol=1e-12):
            _raise("result_identity_mismatch", "Delta-F formula is inconsistent.", roi=item.roi_id)
        if item.strategy_family == "dynamic_fit":
            scalar = value_by_roi.get(item.roi_id)
            if item.reference_kind != REFERENCE_FITTED_CONTROL or item.scalar_f0 != scalar:
                _raise("invalid_dynamic_f0", "Dynamic ROI denominator or reference kind is invalid.", roi=item.roi_id)
            expected_dff = 100.0 * result.delta_f_values[:, index] / scalar
        else:
            if item.reference_kind != REFERENCE_SIGNAL_DERIVED_F0 or item.scalar_f0 is not None:
                _raise("segment_strategy_mismatch", "Signal-Only reference metadata is invalid.", roi=item.roi_id)
            expected_dff = 100.0 * result.delta_f_values[:, index] / result.correction_reference_values[:, index]
        if not np.allclose(result.dff_values[:, index], expected_dff, rtol=1e-12, atol=1e-12):
            _raise("result_identity_mismatch", "dF/F formula is inconsistent.", roi=item.roi_id)
    if result.result_identity != _compute_result_identity(result):
        _raise("result_identity_mismatch", "Corrected-segment identity mismatch.")


def correct_guided_continuous_rwd_segment(
    review_binding: GuidedContinuousRwdReviewBinding,
    target_grid: GuidedContinuousRwdTargetGridDescription,
    segment_plan: GuidedContinuousRwdCorrectionSegmentPlan,
    dynamic_f0_authority: GuidedContinuousRwdDynamicF0Authority,
    raw_segment: GuidedContinuousRwdRawCorrectionSegment,
    *,
    accepted_draft: GuidedNewAnalysisDraftPlan,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
    cancellation_requested: Callable[[], bool] | None = None,
) -> GuidedContinuousRwdCorrectedSegment:
    """Correct exactly one complete accepted C4a raw correction segment."""
    _check_cancellation(cancellation_requested)
    if not isinstance(review_binding, GuidedContinuousRwdReviewBinding):
        _raise("accepted_correction_binding_mismatch", "B3 Review binding has the wrong type.")
    try:
        _validate_authorities(
            review_binding,
            target_grid,
            build_guided_continuous_rwd_block_plan(target_grid),
        )
        accepted = _resolve_accepted_correction_context(
            review_binding, accepted_draft, startup_mapping_contract
        )
        _validate_segment_plan(segment_plan, target_grid, review_binding, accepted)
        segment_correction_config, segment_correction_settings_identity = (
            _resolve_segment_correction_settings(startup_mapping_contract)
        )
    except Exception as exc:
        raise GuidedContinuousRwdSegmentCorrectionError(
            "accepted_correction_binding_mismatch",
            "Accepted correction authorities are incompatible.",
            reason=str(exc),
        ) from exc
    raw = _validate_raw_segment(
        raw_segment,
        review_binding=review_binding,
        target_grid=target_grid,
        segment_plan=segment_plan,
    )
    try:
        _validate_dynamic_f0_authority(
            dynamic_f0_authority,
            review_binding=review_binding,
            target_grid=target_grid,
            segment_plan=segment_plan,
            accepted_context=accepted,
        )
    except Exception as exc:
        raise GuidedContinuousRwdSegmentCorrectionError(
            "invalid_dynamic_f0_authority",
            "Finalized dynamic-F0 authority is invalid.",
            reason=str(exc),
        ) from exc
    _check_cancellation(cancellation_requested)

    f0_by_roi = {item.roi_id: item.scalar_f0 for item in dynamic_f0_authority.values}
    dynamic_expected = tuple(
        item.roi_id for item in accepted.bindings if item.strategy_family == "dynamic_fit"
    )
    if tuple(f0_by_roi) != dynamic_expected:
        _raise("missing_dynamic_f0", "Dynamic F0 lookup is incomplete or misordered.")

    rows, columns = raw.signal_values.shape
    reference = np.empty((rows, columns), dtype=np.float64)
    delta_f = np.empty((rows, columns), dtype=np.float64)
    dff = np.empty((rows, columns), dtype=np.float64)
    per_roi: list[GuidedContinuousRwdPerRoiCorrectionResult] = []
    local_time = raw.target_elapsed_seconds - raw.target_elapsed_seconds[0]
    for roi_index, binding in enumerate(accepted.bindings):
        _check_cancellation(cancellation_requested)
        spec = accepted.correction_specs[binding.roi_id]
        if binding.strategy_family == "dynamic_fit":
            scalar = f0_by_roi.get(binding.roi_id)
            if scalar is None:
                _raise("missing_dynamic_f0", "Dynamic ROI lacks finalized scalar F0.", roi=binding.roi_id)
            if not math.isfinite(scalar) or scalar <= dynamic_f0_authority.f0_min_value:
                _raise("invalid_dynamic_f0", "Dynamic ROI scalar F0 is invalid.", roi=binding.roi_id)
            ref, delta, normalized, applied, path, qc = _correct_dynamic_roi(
                raw,
                roi_index,
                spec,
                segment_correction_config,
                scalar,
                1.0 / float(target_grid.cadence_fraction),
            )
            reference_kind = REFERENCE_FITTED_CONTROL
            scalar_result: float | None = scalar
        elif binding.selected_strategy == SIGNAL_ONLY_STRATEGY:
            try:
                signal_only = compute_signal_only_f0_production(
                    raw.signal_values[:, roi_index],
                    local_time,
                    signal_state_config=dict(vars(segment_correction_config)),
                    signal_only_f0_config=dict(vars(segment_correction_config)),
                    coverage_fraction=float(
                        segment_correction_config.signal_only_f0_min_coverage_fraction
                    ),
                    f0_min_value=float(segment_correction_config.f0_min_value),
                )
            except SignalOnlyF0ProductionError as exc:
                raise GuidedContinuousRwdSegmentCorrectionError(
                    "signal_only_failure",
                    "Segment-local Signal-Only correction failed.",
                    roi=binding.roi_id,
                    segment_index=raw.segment_index,
                    start_target_index=raw.start_target_index,
                    stop_target_index=raw.stop_target_index,
                    requested_strategy=binding.selected_strategy,
                    attempted_fallback_chain=(SIGNAL_ONLY_STRATEGY,),
                    failed_stage="signal_only_f0",
                    reason=str(exc),
                ) from exc
            ref = signal_only.baseline
            delta = signal_only.delta_f
            normalized = signal_only.dff
            applied = SIGNAL_ONLY_STRATEGY
            path = (SIGNAL_ONLY_STRATEGY,)
            qc = _qc_json(
                {"production": signal_only.qc, "signal_state": signal_only.signal_state}
            )
            reference_kind = REFERENCE_SIGNAL_DERIVED_F0
            scalar_result = None
        else:
            _raise("segment_strategy_mismatch", "Accepted ROI strategy is unsupported.", roi=binding.roi_id)
        reference[:, roi_index] = ref
        delta_f[:, roi_index] = delta
        dff[:, roi_index] = normalized
        per_roi.append(
            GuidedContinuousRwdPerRoiCorrectionResult(
                roi_id=binding.roi_id,
                strategy_family=binding.strategy_family,
                selected_strategy=binding.selected_strategy,
                dynamic_fit_mode=binding.dynamic_fit_mode,
                parameter_identity=binding.parameter_identity,
                evidence_identity=binding.evidence_identity,
                reference_kind=reference_kind,
                applied_strategy=applied,
                fallback_path=path,
                qc_json=qc,
                scalar_f0=scalar_result,
            )
        )
        _check_cancellation(cancellation_requested)

    _check_cancellation(cancellation_requested)
    arrays = (
        raw.target_elapsed_seconds.copy(),
        raw.control_values.copy(),
        raw.signal_values.copy(),
        reference,
        delta_f,
        dff,
    )
    for array in arrays:
        array.setflags(write=False)
    draft = GuidedContinuousRwdCorrectedSegment(
        schema_name=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        policy_name=POLICY_NAME,
        policy_version=POLICY_VERSION,
        recording_identity=raw.recording_identity,
        source_content_identity=raw.source_content_identity,
        target_grid_identity=raw.target_grid_identity,
        correction_segment_plan_identity=raw.correction_segment_plan_identity,
        dynamic_f0_authority_identity=dynamic_f0_authority.authority_identity,
        accepted_guided_plan_identity=accepted.accepted_guided_plan_identity,
        correction_payload_identity=accepted.correction_payload_identity,
        fixed_correction_settings_identity=accepted.fixed_correction_settings_identity,
        segment_correction_settings_identity=segment_correction_settings_identity,
        segment_index=raw.segment_index,
        start_target_index=raw.start_target_index,
        stop_target_index=raw.stop_target_index,
        included_roi_ids=raw.included_roi_ids,
        target_elapsed_seconds=arrays[0],
        raw_control_values=arrays[1],
        raw_signal_values=arrays[2],
        correction_reference_values=arrays[3],
        delta_f_values=arrays[4],
        dff_values=arrays[5],
        per_roi_results=tuple(per_roi),
        result_identity="",
    )
    result = GuidedContinuousRwdCorrectedSegment(
        **{**draft.__dict__, "result_identity": _compute_result_identity(draft)}
    )
    _check_cancellation(cancellation_requested)
    _validate_result(
        result,
        raw_segment=raw,
        dynamic_f0_authority=dynamic_f0_authority,
        accepted=accepted,
        segment_correction_settings_identity=segment_correction_settings_identity,
    )
    return result
