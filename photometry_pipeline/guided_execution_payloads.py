"""Pure, in-memory Guided execution payload models and derivation function."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import math

from photometry_pipeline.config import Config
from photometry_pipeline.feature_event_config import (
    FEATURE_EVENT_CONFIG_FIELDS,
    validate_feature_event_config_fields,
)
from photometry_pipeline.guided_identity import encode_canonical_value
from photometry_pipeline.guided_run_authorization import (
    GuidedRunAuthorizationResult,
    compute_guided_run_authorization_identity,
)
from photometry_pipeline.guided_production_mapping import (
    GuidedProductionExecutionIntent,
    compute_guided_production_execution_intent_identity,
)
from photometry_pipeline.guided_execution_preflight import (
    GuidedCandidateManifestExecutionPreflightResult,
    GuidedRoiExecutionPreflightResult,
    compute_guided_candidate_preflight_identity,
    compute_guided_roi_preflight_identity,
)


GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE = "payloads_derived_nonrunnable"
GUIDED_EXECUTION_PAYLOAD_STATUS_RUNNABLE = "runnable_payloads_derived"
GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED = "refused"

GUIDED_EXECUTION_STARTUP_MAPPING_CONTRACT_SCHEMA_NAME = "guided_execution_startup_mapping_contract"
GUIDED_EXECUTION_STARTUP_MAPPING_CONTRACT_SCHEMA_VERSION = "v1"
GUIDED_EXECUTION_STARTUP_MAPPING_CONTRACT_VERSION = (
    "guided_execution_startup_mapping.post_4J14l.v2"
)


@dataclass(frozen=True)
class GuidedExecutionStartupMappingContract:
    schema_name: str
    schema_version: str
    contract_version: str
    supported_intent_schema_name: str
    supported_intent_schema_version: str
    supported_authorization_contract_version: str
    supported_mapping_contract_version: str
    supported_runner_contract_version: str
    config_mapping_contract_version: str
    candidate_manifest_schema_version: str
    candidate_consumption_contract_version: str
    roi_consumption_contract_version: str
    startup_provenance_schema_version: str
    runner_request_schema_version: str
    command_record_version: str
    exact_candidate_manifest_consumption_capable: bool
    exact_roi_consumption_capable: bool
    fixed_config_overrides: tuple[GuidedConfigFieldValue, ...]


@dataclass(frozen=True)
class GuidedConfigFieldValue:
    name: str
    value: Any


@dataclass(frozen=True)
class GuidedExecutionConfigPayload:
    config_schema_name: str
    config_mapping_contract_version: str
    values: tuple[GuidedConfigFieldValue, ...]
    canonical_config_payload_identity: str


@dataclass(frozen=True)
class GuidedRunnerCandidateManifestEntry:
    canonical_relative_path: str
    size_bytes: int
    sha256_content_digest: str


@dataclass(frozen=True)
class GuidedRunnerCandidateManifestPayload:
    manifest_schema_name: str
    manifest_schema_version: str
    candidate_consumption_contract_version: str
    source_root_canonical: str
    source_candidate_set_digest: str
    source_candidate_content_digest: str
    candidate_files: tuple[GuidedRunnerCandidateManifestEntry, ...]
    parser_contract_digest: str
    discovered_roi_ids: tuple[str, ...]
    included_roi_ids: tuple[str, ...]
    excluded_roi_ids: tuple[str, ...]
    strict_roi_inventory_digest: str
    candidate_preflight_identity: str
    roi_preflight_identity: str
    canonical_candidate_manifest_payload_identity: str


@dataclass(frozen=True)
class GuidedRunnerExecutionRequest:
    runner_request_schema_name: str
    runner_request_schema_version: str
    runner_contract_version: str
    runner_entrypoint: str
    input_source_root: str
    output_base_canonical: str
    run_type: str
    mode: str
    input_format: str
    include_rois: tuple[str, ...]
    traces_only: bool
    overwrite: bool
    config_filename: str
    candidate_manifest_filename: str
    required_future_argv_flags: tuple[str, ...]
    prohibited_argv_flags: tuple[str, ...]
    canonical_runner_request_identity: str


@dataclass(frozen=True)
class GuidedStartupProvenanceSeed:
    provenance_schema_name: str
    provenance_schema_version: str
    startup_mapping_contract_version: str
    validation_request_identity: str
    authorization_identity: str
    production_intent_identity: str
    application_build_identity: str
    production_mapping_contract_version: str
    runner_contract_version: str
    candidate_preflight_identity: str
    roi_preflight_identity: str
    config_payload_identity: str
    candidate_manifest_payload_identity: str
    runner_request_identity: str | None
    runnable: bool
    canonical_provenance_seed_identity: str


@dataclass(frozen=True)
class GuidedExecutionPayloadIssue:
    category: str
    section: str
    message: str
    detail_code: str = ""


@dataclass(frozen=True)
class GuidedExecutionPayloadDerivationResult:
    status: str
    ok: bool
    runnable: bool
    config_payload: GuidedExecutionConfigPayload | None
    candidate_manifest_payload: GuidedRunnerCandidateManifestPayload | None
    runner_request: GuidedRunnerExecutionRequest | None
    provenance_seed: GuidedStartupProvenanceSeed | None
    config_payload_identity: str | None
    candidate_manifest_payload_identity: str | None
    runner_request_identity: str | None
    provenance_seed_identity: str | None
    limiting_issues: tuple[GuidedExecutionPayloadIssue, ...]
    blocking_issues: tuple[GuidedExecutionPayloadIssue, ...]
    no_files_written: bool = True
    no_directories_created: bool = True
    no_artifacts_created: bool = True
    no_output_allocated: bool = True
    no_run_id_allocated: bool = True
    no_config_file_generated: bool = True
    no_argv_generated: bool = True
    no_runner_invoked: bool = True


CONFIG_DISPOSITION_INTENT = "mapped_from_intent"
CONFIG_DISPOSITION_FIXED = "fixed_by_contract"
CONFIG_DISPOSITION_FIXED_FALSE_EMPTY = "fixed_false_or_empty"
CONFIG_DISPOSITION_UNSUPPORTED_FUTURE = "unsupported_future"
CONFIG_DISPOSITION_PROHIBITED_FIRST_SUBSET = "prohibited_first_subset"
CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED = "not_applicable_serialized_fixed"
# 4J16k39a: feature-detection fields are sourced from the feature/event
# settings the user explicitly confirmed in Guided Step 5
# (intent.feature_event.effective_values), never from a baked contract
# override. They are therefore neither "mapped_from_intent" (which covers the
# acquisition/correction intent fields) nor "fixed_by_contract".
CONFIG_DISPOSITION_CONFIRMED_FEATURE = "confirmed_feature_settings"

GUIDED_CONFIG_FIELD_DISPOSITIONS = {
    "chunk_duration_sec": CONFIG_DISPOSITION_FIXED,
    "trim_samples_start": CONFIG_DISPOSITION_FIXED,
    "trim_samples_end": CONFIG_DISPOSITION_FIXED,
    "seed": CONFIG_DISPOSITION_FIXED,
    "allow_partial_final_chunk": CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
    "exclude_incomplete_final_rwd_chunk": CONFIG_DISPOSITION_INTENT,
    "rwd_excluded_source_files": CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
    "rwd_contract_validation": CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
    "target_fs_hz": CONFIG_DISPOSITION_INTENT,
    "lowpass_hz": CONFIG_DISPOSITION_FIXED,
    "filter_order": CONFIG_DISPOSITION_FIXED,
    "window_sec": CONFIG_DISPOSITION_FIXED,
    "step_sec": CONFIG_DISPOSITION_FIXED,
    "r_low": CONFIG_DISPOSITION_FIXED,
    "r_high": CONFIG_DISPOSITION_FIXED,
    "g_min": CONFIG_DISPOSITION_FIXED,
    "min_samples_per_window": CONFIG_DISPOSITION_FIXED,
    "min_valid_windows": CONFIG_DISPOSITION_FIXED,
    "baseline_subtract_before_fit": CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
    "bleach_correction_mode": CONFIG_DISPOSITION_FIXED,
    "dynamic_fit_mode": CONFIG_DISPOSITION_INTENT,
    "dynamic_fit_slope_constraint": CONFIG_DISPOSITION_FIXED,
    "dynamic_fit_min_slope": CONFIG_DISPOSITION_FIXED,
    "robust_event_reject_max_iters": CONFIG_DISPOSITION_FIXED,
    "robust_event_reject_residual_z_thresh": CONFIG_DISPOSITION_FIXED,
    "robust_event_reject_local_var_window_sec": CONFIG_DISPOSITION_FIXED,
    "robust_event_reject_local_var_ratio_thresh": CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
    "robust_event_reject_min_keep_fraction": CONFIG_DISPOSITION_FIXED,
    "adaptive_event_gate_residual_z_thresh": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "adaptive_event_gate_local_var_window_sec": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "adaptive_event_gate_local_var_ratio_thresh": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "adaptive_event_gate_smooth_window_sec": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "adaptive_event_gate_min_trust_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "adaptive_event_gate_freeze_interp_method": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "baseline_reference_smoothing_window_sec": CONFIG_DISPOSITION_FIXED,
    "baseline_reference_min_smoothing_window_sec": CONFIG_DISPOSITION_FIXED,
    "baseline_reference_max_window_fraction_of_chunk": CONFIG_DISPOSITION_FIXED,
    "baseline_reference_large_window_fraction_warning": CONFIG_DISPOSITION_FIXED,
    "baseline_reference_candidate_enabled": CONFIG_DISPOSITION_FIXED,
    "signal_state_smoothing_window_fraction": CONFIG_DISPOSITION_FIXED,
    "signal_state_smoothing_window_sec": CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
    "signal_state_high_quantile": CONFIG_DISPOSITION_FIXED,
    "signal_state_low_quantile": CONFIG_DISPOSITION_FIXED,
    "signal_state_min_episode_fraction": CONFIG_DISPOSITION_FIXED,
    "signal_state_min_episode_sec": CONFIG_DISPOSITION_FIXED,
    "signal_state_edge_fraction": CONFIG_DISPOSITION_FIXED,
    "signal_state_variability_window_fraction": CONFIG_DISPOSITION_FIXED,
    "signal_state_variability_window_sec": CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
    "signal_state_low_variability_quantile": CONFIG_DISPOSITION_FIXED,
    "signal_state_low_variability_ratio_threshold": CONFIG_DISPOSITION_FIXED,
    "signal_state_partial_min_high_fraction": CONFIG_DISPOSITION_FIXED,
    "signal_state_partial_min_longest_fraction": CONFIG_DISPOSITION_FIXED,
    "signal_state_partial_max_variability_ratio": CONFIG_DISPOSITION_FIXED,
    "signal_state_partial_min_variability_suppression": CONFIG_DISPOSITION_FIXED,
    "signal_state_partial_requires_low_variability": CONFIG_DISPOSITION_FIXED,
    "signal_state_step_window_fraction": CONFIG_DISPOSITION_FIXED,
    "signal_state_step_window_sec": CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
    "signal_state_step_threshold_robust_z": CONFIG_DISPOSITION_FIXED,
    "signal_state_min_robust_range": CONFIG_DISPOSITION_FIXED,
    "signal_only_f0_window_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_window_sec": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_low_quantile": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_smoothing_window_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_smoothing_window_sec": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_min_window_samples": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_max_window_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_min_robust_range": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_max_above_signal_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_max_tracking_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_min_coverage_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_high_state_context_mode": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_state_aware_enabled": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_low_support_quantile": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_low_support_buffer_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_low_support_buffer_sec": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_min_low_support_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_min_anchor_count": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_max_anchor_gap_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_max_anchor_gap_sec": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_edge_extrapolation_mode": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_max_edge_extrapolation_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_max_edge_extrapolation_sec": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_medium_extrapolation_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_high_extrapolation_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_low_anchor_support_fraction": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_low_anchor_count": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "signal_only_f0_confidence_cap_on_large_gap": CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
    "baseline_method": CONFIG_DISPOSITION_FIXED,
    "baseline_percentile": CONFIG_DISPOSITION_FIXED,
    "f0_min_value": CONFIG_DISPOSITION_FIXED,
    "npm_time_axis": CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
    "sampling_rate_hz_fallback": CONFIG_DISPOSITION_FIXED,
    "timestamp_cv_max": CONFIG_DISPOSITION_FIXED,
    "duration_tolerance_frac": CONFIG_DISPOSITION_FIXED,
    "qc_max_chunk_fail_fraction": CONFIG_DISPOSITION_FIXED,
    "peak_threshold_method": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "peak_threshold_k": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "peak_threshold_percentile": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "peak_threshold_abs": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "peak_min_distance_sec": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "peak_min_prominence_k": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "peak_min_width_sec": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "peak_pre_filter": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "event_auc_baseline": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "event_signal": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "signal_excursion_polarity": CONFIG_DISPOSITION_CONFIRMED_FEATURE,
    "representative_session_index": CONFIG_DISPOSITION_PROHIBITED_FIRST_SUBSET,
    "preview_first_n": CONFIG_DISPOSITION_PROHIBITED_FIRST_SUBSET,
    "adapter_value_nan_policy": CONFIG_DISPOSITION_FIXED,
    "tonic_allowed_nan_frac": CONFIG_DISPOSITION_FIXED,
    "tonic_output_mode": CONFIG_DISPOSITION_FIXED,
    "tonic_timeline_mode": CONFIG_DISPOSITION_FIXED,
    "export_display_series_csv": CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
    "rwd_time_col": CONFIG_DISPOSITION_INTENT,
    "uv_suffix": CONFIG_DISPOSITION_INTENT,
    "sig_suffix": CONFIG_DISPOSITION_INTENT,
    "npm_frame_col": CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
    "npm_system_ts_col": CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
    "npm_computer_ts_col": CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
    "npm_led_col": CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
    "npm_region_prefix": CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
    "npm_region_suffix": CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
    "custom_tabular_time_col": CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
    "custom_tabular_uv_suffix": CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
    "custom_tabular_sig_suffix": CONFIG_DISPOSITION_UNSUPPORTED_FUTURE,
    "acquisition_mode": CONFIG_DISPOSITION_INTENT,
    "continuous_window_sec": CONFIG_DISPOSITION_PROHIBITED_FIRST_SUBSET,
    "continuous_step_sec": CONFIG_DISPOSITION_PROHIBITED_FIRST_SUBSET,
    "allow_partial_final_window": CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
}

GUIDED_CONFIG_DEFAULT_OVERRIDES = {
    "chunk_duration_sec": 600.0,
    "trim_samples_start": 0,
    "trim_samples_end": 0,
    "seed": 0,
    "allow_partial_final_chunk": False,
    "rwd_excluded_source_files": [],
    "rwd_contract_validation": {},
    "lowpass_hz": 1.0,
    "filter_order": 3,
    "window_sec": 60.0,
    "step_sec": 10.0,
    "r_low": 0.2,
    "r_high": 0.8,
    "g_min": 0.2,
    "min_samples_per_window": 0,
    "min_valid_windows": 5,
    "baseline_subtract_before_fit": False,
    "bleach_correction_mode": "none",
    "dynamic_fit_slope_constraint": "unconstrained",
    "dynamic_fit_min_slope": 0.0,
    "robust_event_reject_max_iters": 3,
    "robust_event_reject_residual_z_thresh": 3.5,
    "robust_event_reject_local_var_window_sec": 10.0,
    "robust_event_reject_local_var_ratio_thresh": None,
    "robust_event_reject_min_keep_fraction": 0.5,
    "adaptive_event_gate_residual_z_thresh": 3.5,
    "adaptive_event_gate_local_var_window_sec": 10.0,
    "adaptive_event_gate_local_var_ratio_thresh": None,
    "adaptive_event_gate_smooth_window_sec": 60.0,
    "adaptive_event_gate_min_trust_fraction": 0.5,
    "adaptive_event_gate_freeze_interp_method": "linear_hold",
    "baseline_reference_smoothing_window_sec": 300.0,
    "baseline_reference_min_smoothing_window_sec": 60.0,
    "baseline_reference_max_window_fraction_of_chunk": 0.75,
    "baseline_reference_large_window_fraction_warning": 0.50,
    "baseline_reference_candidate_enabled": True,
    "signal_state_smoothing_window_fraction": 0.05,
    "signal_state_smoothing_window_sec": None,
    "signal_state_high_quantile": 0.80,
    "signal_state_low_quantile": 0.20,
    "signal_state_min_episode_fraction": 0.20,
    "signal_state_min_episode_sec": 0.0,
    "signal_state_edge_fraction": 0.10,
    "signal_state_variability_window_fraction": 0.05,
    "signal_state_variability_window_sec": None,
    "signal_state_low_variability_quantile": 0.35,
    "signal_state_low_variability_ratio_threshold": 0.75,
    "signal_state_partial_min_high_fraction": 0.10,
    "signal_state_partial_min_longest_fraction": 0.075,
    "signal_state_partial_max_variability_ratio": 0.60,
    "signal_state_partial_min_variability_suppression": 0.35,
    "signal_state_partial_requires_low_variability": True,
    "signal_state_step_window_fraction": 0.03,
    "signal_state_step_window_sec": None,
    "signal_state_step_threshold_robust_z": 3.5,
    "signal_state_min_robust_range": 1e-6,
    "signal_only_f0_window_fraction": 0.20,
    "signal_only_f0_window_sec": None,
    "signal_only_f0_low_quantile": 0.10,
    "signal_only_f0_smoothing_window_fraction": 0.10,
    "signal_only_f0_smoothing_window_sec": None,
    "signal_only_f0_min_window_samples": 21,
    "signal_only_f0_max_window_fraction": 0.50,
    "signal_only_f0_min_robust_range": 1e-6,
    "signal_only_f0_max_above_signal_fraction": 0.20,
    "signal_only_f0_max_tracking_fraction": 0.85,
    "signal_only_f0_min_coverage_fraction": 0.80,
    "signal_only_f0_high_state_context_mode": "contextual_cap",
    "signal_only_f0_state_aware_enabled": True,
    "signal_only_f0_low_support_quantile": 0.35,
    "signal_only_f0_low_support_buffer_fraction": 0.02,
    "signal_only_f0_low_support_buffer_sec": None,
    "signal_only_f0_min_low_support_fraction": 0.10,
    "signal_only_f0_min_anchor_count": 3,
    "signal_only_f0_max_anchor_gap_fraction": 0.50,
    "signal_only_f0_max_anchor_gap_sec": None,
    "signal_only_f0_edge_extrapolation_mode": "hold_nearest_anchor",
    "signal_only_f0_max_edge_extrapolation_fraction": 0.50,
    "signal_only_f0_max_edge_extrapolation_sec": None,
    "signal_only_f0_medium_extrapolation_fraction": 0.25,
    "signal_only_f0_high_extrapolation_fraction": 0.50,
    "signal_only_f0_low_anchor_support_fraction": 0.10,
    "signal_only_f0_low_anchor_count": 5,
    "signal_only_f0_confidence_cap_on_large_gap": True,
    "baseline_method": "uv_raw_percentile_session",
    "baseline_percentile": 10.0,
    "f0_min_value": 1e-9,
    "sampling_rate_hz_fallback": 40.0,
    "timestamp_cv_max": 0.02,
    "duration_tolerance_frac": 0.02,
    "qc_max_chunk_fail_fraction": 0.20,
    # Feature-detection fields are intentionally ABSENT here (4J16k39a):
    # they carry CONFIG_DISPOSITION_CONFIRMED_FEATURE and are serialized from
    # the settings the user confirmed in Guided Step 5, never from a baked
    # default. Adding them back would silently override the confirmed values.
    "representative_session_index": None,
    "preview_first_n": None,
    "adapter_value_nan_policy": "strict",
    "tonic_allowed_nan_frac": 0.0,
    "tonic_output_mode": "preserve_raw_session_shape",
    "tonic_timeline_mode": "real_elapsed_time",
    "export_display_series_csv": False,
    "continuous_window_sec": 600.0,
    "continuous_step_sec": 600.0,
    "allow_partial_final_window": False,
    "npm_time_axis": "system_timestamp",
    "npm_frame_col": "Frame",
    "npm_system_ts_col": "SystemTimestamp",
    "npm_computer_ts_col": "ComputerTimestamp",
    "npm_led_col": "LED",
    "npm_region_prefix": "Region",
    "npm_region_suffix": "",
    "custom_tabular_time_col": "Time(s)",
    "custom_tabular_uv_suffix": "-410",
    "custom_tabular_sig_suffix": "-470",
}


class FrozenDict(tuple):
    """An immutable, frozen dictionary representation built on tuple."""
    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self:
            if k == key:
                return v
        return default

    def keys(self) -> list[str]:
        return [k for k, v in self]

    def values(self) -> list[Any]:
        return [v for k, v in self]

    def items(self) -> FrozenDict:
        return self


def _freeze_override_value(value: Any) -> Any:
    """Recursively freeze list, tuple, dict, set, and reject custom/mutable objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return tuple(_freeze_override_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_override_value(v) for v in value)
    if isinstance(value, dict):
        return FrozenDict(
            (k, _freeze_override_value(value[k]))
            for k in sorted(value.keys())
        )
    if isinstance(value, set):
        return tuple(
            _freeze_override_value(v)
            for v in sorted(list(value))
        )
    raise TypeError(f"Unsupported mutable or custom object in contract overrides: {type(value)}")


def build_guided_execution_startup_mapping_contract(
    *,
    exact_candidate_manifest_consumption_capable: bool = True,
    exact_roi_consumption_capable: bool = True,
    fixed_config_overrides: Mapping[str, Any] | None = None,
) -> GuidedExecutionStartupMappingContract:
    """Build a frozen GuidedExecutionStartupMappingContract instance."""
    overrides_dict = (
        dict(fixed_config_overrides)
        if fixed_config_overrides is not None
        else dict(GUIDED_CONFIG_DEFAULT_OVERRIDES)
    )
    overrides_tuple = tuple(
        GuidedConfigFieldValue(name, _freeze_override_value(overrides_dict[name]))
        for name in sorted(overrides_dict.keys())
    )
    return GuidedExecutionStartupMappingContract(
        schema_name=GUIDED_EXECUTION_STARTUP_MAPPING_CONTRACT_SCHEMA_NAME,
        schema_version=GUIDED_EXECUTION_STARTUP_MAPPING_CONTRACT_SCHEMA_VERSION,
        contract_version=GUIDED_EXECUTION_STARTUP_MAPPING_CONTRACT_VERSION,
        supported_intent_schema_name="guided_production_execution_intent",
        supported_intent_schema_version="v1",
        supported_authorization_contract_version="guided_run_authorization.v1",
        supported_mapping_contract_version="guided_production_mapping.v1",
        supported_runner_contract_version="run_full_pipeline_deliverables.v1",
        config_mapping_contract_version="guided_execution_config_mapping.v1",
        candidate_manifest_schema_version="v1",
        candidate_consumption_contract_version="exact_candidate_manifest_consumption.v1",
        roi_consumption_contract_version="exact_included_roi_tuple_consumption.v1",
        startup_provenance_schema_version="v1",
        runner_request_schema_version="v1",
        command_record_version="v1",
        exact_candidate_manifest_consumption_capable=exact_candidate_manifest_consumption_capable,
        exact_roi_consumption_capable=exact_roi_consumption_capable,
        fixed_config_overrides=overrides_tuple,
    )


def compute_guided_execution_config_payload_identity(
    payload: GuidedExecutionConfigPayload,
) -> str:
    """Compute deterministic identity for GuidedExecutionConfigPayload."""
    data = {
        "config_schema_name": payload.config_schema_name,
        "config_mapping_contract_version": payload.config_mapping_contract_version,
        "values": [{"name": v.name, "value": v.value} for v in payload.values],
    }
    domain = b"guided-execution-config-payload:v1"
    payload_bytes = encode_canonical_value(data)
    return hashlib.sha256(domain + b"\x00" + payload_bytes).hexdigest()


def compute_guided_runner_candidate_manifest_payload_identity(
    payload: GuidedRunnerCandidateManifestPayload,
) -> str:
    """Compute deterministic identity for GuidedRunnerCandidateManifestPayload."""
    data = {
        "manifest_schema_name": payload.manifest_schema_name,
        "manifest_schema_version": payload.manifest_schema_version,
        "candidate_consumption_contract_version": payload.candidate_consumption_contract_version,
        "source_root_canonical": payload.source_root_canonical,
        "source_candidate_set_digest": payload.source_candidate_set_digest,
        "source_candidate_content_digest": payload.source_candidate_content_digest,
        "candidate_files": [
            {
                "canonical_relative_path": f.canonical_relative_path,
                "size_bytes": f.size_bytes,
                "sha256_content_digest": f.sha256_content_digest,
            }
            for f in payload.candidate_files
        ],
        "parser_contract_digest": payload.parser_contract_digest,
        "discovered_roi_ids": payload.discovered_roi_ids,
        "included_roi_ids": payload.included_roi_ids,
        "excluded_roi_ids": payload.excluded_roi_ids,
        "strict_roi_inventory_digest": payload.strict_roi_inventory_digest,
        "candidate_preflight_identity": payload.candidate_preflight_identity,
        "roi_preflight_identity": payload.roi_preflight_identity,
    }
    domain = b"guided-runner-candidate-manifest-payload:v1"
    payload_bytes = encode_canonical_value(data)
    return hashlib.sha256(domain + b"\x00" + payload_bytes).hexdigest()


def compute_guided_runner_execution_request_identity(
    request: GuidedRunnerExecutionRequest,
) -> str:
    """Compute deterministic identity for GuidedRunnerExecutionRequest."""
    data = {
        "runner_request_schema_name": request.runner_request_schema_name,
        "runner_request_schema_version": request.runner_request_schema_version,
        "runner_contract_version": request.runner_contract_version,
        "runner_entrypoint": request.runner_entrypoint,
        "input_source_root": request.input_source_root,
        "output_base_canonical": request.output_base_canonical,
        "run_type": request.run_type,
        "mode": request.mode,
        "input_format": request.input_format,
        "include_rois": request.include_rois,
        "traces_only": request.traces_only,
        "overwrite": request.overwrite,
        "config_filename": request.config_filename,
        "candidate_manifest_filename": request.candidate_manifest_filename,
        "required_future_argv_flags": request.required_future_argv_flags,
        "prohibited_argv_flags": request.prohibited_argv_flags,
    }
    domain = b"guided-runner-execution-request:v1"
    payload_bytes = encode_canonical_value(data)
    return hashlib.sha256(domain + b"\x00" + payload_bytes).hexdigest()


def compute_guided_startup_provenance_seed_identity(
    seed: GuidedStartupProvenanceSeed,
) -> str:
    """Compute deterministic identity for GuidedStartupProvenanceSeed."""
    data = {
        "provenance_schema_name": seed.provenance_schema_name,
        "provenance_schema_version": seed.provenance_schema_version,
        "startup_mapping_contract_version": seed.startup_mapping_contract_version,
        "validation_request_identity": seed.validation_request_identity,
        "authorization_identity": seed.authorization_identity,
        "production_intent_identity": seed.production_intent_identity,
        "application_build_identity": seed.application_build_identity,
        "production_mapping_contract_version": seed.production_mapping_contract_version,
        "runner_contract_version": seed.runner_contract_version,
        "candidate_preflight_identity": seed.candidate_preflight_identity,
        "roi_preflight_identity": seed.roi_preflight_identity,
        "config_payload_identity": seed.config_payload_identity,
        "candidate_manifest_payload_identity": seed.candidate_manifest_payload_identity,
        "runner_request_identity": seed.runner_request_identity,
        "runnable": seed.runnable,
    }
    domain = b"guided-startup-provenance-seed:v1"
    payload_bytes = encode_canonical_value(data)
    return hashlib.sha256(domain + b"\x00" + payload_bytes).hexdigest()


def _unresolved(
    category: str, message: str, detail_code: str = ""
) -> GuidedExecutionPayloadDerivationResult:
    issue = GuidedExecutionPayloadIssue(
        category=category,
        section="guided_execution_payload",
        message=message,
        detail_code=detail_code,
    )
    return GuidedExecutionPayloadDerivationResult(
        status=GUIDED_EXECUTION_PAYLOAD_STATUS_REFUSED,
        ok=False,
        runnable=False,
        config_payload=None,
        candidate_manifest_payload=None,
        runner_request=None,
        provenance_seed=None,
        config_payload_identity=None,
        candidate_manifest_payload_identity=None,
        runner_request_identity=None,
        provenance_seed_identity=None,
        limiting_issues=(),
        blocking_issues=(issue,),
    )


def _find_semantic_value(semantic_values: tuple[Any, ...], name: str) -> Any:
    for item in semantic_values:
        if getattr(item, "field_name", None) == name:
            return getattr(item, "value", None)
    return None


def resolve_confirmed_feature_config_fields(
    intent: GuidedProductionExecutionIntent,
) -> tuple[dict[str, Any] | None, str]:
    """Extract the COMPLETE feature-detection settings the user confirmed in
    Guided Step 5 from a mapped production intent (4J16k39a).

    These become the production base configuration consumed by Pipeline. They
    are never reconstructed from GUIDED_CONFIG_DEFAULT_OVERRIDES, which no
    longer carries feature-detection fields at all.

    Returns (fields, "") when the confirmed settings are complete and valid, or
    (None, reason) when they are missing, incomplete, non-finite, or
    semantically invalid. Fails closed: a Guided plan that claims settings were
    confirmed must produce a complete, valid base configuration.
    """
    feature_event = getattr(intent, "feature_event", None)
    if feature_event is None:
        return None, "Intent carries no confirmed feature/event settings."
    if not getattr(feature_event, "explicitly_applied", False):
        return None, "Feature-detection settings were never confirmed."
    if not getattr(feature_event, "current", False):
        return None, "Confirmed feature-detection settings are stale."

    fields: dict[str, Any] = {}
    for item in getattr(feature_event, "effective_values", ()) or ():
        name = getattr(item, "field_name", None)
        if name in FEATURE_EVENT_CONFIG_FIELDS:
            fields[name] = getattr(item, "value", None)

    missing = set(FEATURE_EVENT_CONFIG_FIELDS) - set(fields)
    if missing:
        return None, f"Confirmed feature settings are incomplete: {sorted(missing)}"

    for name, value in fields.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return None, f"Confirmed feature setting {name} is not finite."

    # Validate only the fields ACTIVE for the confirmed threshold method, using
    # the same activity rule the Guided effective-value preview already applies.
    # A complete profile legitimately carries a dormant value for the threshold
    # fields the selected method does not use (e.g. peak_threshold_abs=0.0 under
    # mean_std); those must be serialized into Config but never validated as if
    # they were in use.
    from photometry_pipeline.guided_new_analysis_plan import (
        _feature_event_field_activity,
    )

    method = str(fields.get("peak_threshold_method", ""))
    active_fields = {
        name: value
        for name, value in fields.items()
        if _feature_event_field_activity(name, method) == "active"
    }
    # peak_threshold_method itself is always "active", so an unrecognized method
    # is still rejected below by validate_feature_event_config_fields.
    semantic_errors = validate_feature_event_config_fields(active_fields)
    if semantic_errors:
        return None, f"Confirmed feature settings are invalid: {semantic_errors[0]}"

    return fields, ""


def derive_guided_execution_payloads(
    authorized_result: GuidedRunAuthorizationResult,
    *,
    startup_mapping_contract: GuidedExecutionStartupMappingContract,
) -> GuidedExecutionPayloadDerivationResult:
    """Derive execution payloads from an accepted GuidedRunAuthorizationResult."""
    try:
        # 1. Validate authorized_result type and structure
        if not isinstance(authorized_result, GuidedRunAuthorizationResult):
            return _unresolved("payload_request_invalid", "Invalid authorization result type.")

        # 2. Gate validations
        if authorized_result.status != "authorized":
            return _unresolved("authorization_not_accepted", "Authorization outcome is not authorized.")
        if authorized_result.authorized is not True or authorized_result.run_authorization is not True:
            return _unresolved("authorization_not_accepted", "Authorization outcome boolean gates are not True.")
        if not authorized_result.canonical_authorization_identity:
            return _unresolved("authorization_not_accepted", "Canonical authorization identity is missing.")
        if not authorized_result.production_intent:
            return _unresolved("authorization_not_accepted", "Production intent is missing.")
        
        # Verify preflights
        cand_pre = authorized_result.candidate_preflight_result
        if not cand_pre or cand_pre.status != "accepted" or cand_pre.accepted is not True:
            return _unresolved("authorization_not_accepted", "Candidate preflight result is not accepted.")
        roi_pre = authorized_result.roi_preflight_result
        if not roi_pre or roi_pre.status != "accepted" or roi_pre.accepted is not True:
            return _unresolved("authorization_not_accepted", "ROI preflight result is not accepted.")

        # 3. Recompute identities
        recomputed_auth_id = compute_guided_run_authorization_identity(authorized_result)
        if recomputed_auth_id != authorized_result.canonical_authorization_identity:
            return _unresolved("authorization_identity_mismatch", "Recomputed authorization identity mismatch.")

        intent = authorized_result.production_intent
        recomputed_intent_id = compute_guided_production_execution_intent_identity(intent)
        if recomputed_intent_id != authorized_result.production_intent_identity:
            return _unresolved("production_intent_identity_mismatch", "Recomputed intent identity mismatch.")

        recomputed_cand_id = compute_guided_candidate_preflight_identity(cand_pre)
        if recomputed_cand_id != authorized_result.candidate_preflight_identity:
            return _unresolved("candidate_preflight_identity_mismatch", "Recomputed candidate preflight identity mismatch.")

        recomputed_roi_id = compute_guided_roi_preflight_identity(roi_pre)
        if recomputed_roi_id != authorized_result.roi_preflight_identity:
            return _unresolved("roi_preflight_identity_mismatch", "Recomputed ROI preflight identity mismatch.")

        # 8. Startup mapping contract type and compatibility checks
        if not isinstance(startup_mapping_contract, GuidedExecutionStartupMappingContract):
            return _unresolved("payload_request_invalid", "Invalid startup mapping contract type.")
        if startup_mapping_contract.supported_intent_schema_name != intent.intent_schema_name:
            return _unresolved("startup_mapping_contract_unsupported", "Unsupported intent schema name.")
        if startup_mapping_contract.supported_intent_schema_version != intent.intent_schema_version:
            return _unresolved("startup_mapping_contract_unsupported", "Unsupported intent schema version.")

        # 9. Config field coverage validations
        # Get all Config dataclass field names
        from dataclasses import fields as dataclass_fields
        config_fields = {f.name for f in dataclass_fields(Config)}
        disposition_keys = set(GUIDED_CONFIG_FIELD_DISPOSITIONS.keys())

        if config_fields != disposition_keys:
            return _unresolved("config_mapping_incomplete", "Disposition map fields do not match Config fields.")
        
        # Non-intent fields must be explicitly specified in startup mapping contract overrides.
        # Confirmed-feature fields are excluded from that requirement: they come from
        # intent.feature_event.effective_values, not from the contract overrides.
        intent_keys = {k for k, v in GUIDED_CONFIG_FIELD_DISPOSITIONS.items() if v == CONFIG_DISPOSITION_INTENT}
        confirmed_feature_keys = {
            k for k, v in GUIDED_CONFIG_FIELD_DISPOSITIONS.items()
            if v == CONFIG_DISPOSITION_CONFIRMED_FEATURE
        }
        non_intent_keys = {
            k for k, v in GUIDED_CONFIG_FIELD_DISPOSITIONS.items()
            if v not in (CONFIG_DISPOSITION_INTENT, CONFIG_DISPOSITION_CONFIRMED_FEATURE)
        }
        serialized_non_intent_keys = {k for k, v in GUIDED_CONFIG_FIELD_DISPOSITIONS.items() if v in (
            CONFIG_DISPOSITION_FIXED,
            CONFIG_DISPOSITION_FIXED_FALSE_EMPTY,
            CONFIG_DISPOSITION_NOT_APPLICABLE_FIXED,
        )}

        # The confirmed-feature key set must exactly equal the shared
        # feature-detection field set: a drift in either direction would mean
        # some feature field silently keeps a baked default.
        if confirmed_feature_keys != set(FEATURE_EVENT_CONFIG_FIELDS):
            return _unresolved(
                "config_mapping_incomplete",
                "Confirmed feature-setting fields do not match FEATURE_EVENT_CONFIG_FIELDS.",
            )

        overrides_dict = {item.name: item.value for item in startup_mapping_contract.fixed_config_overrides}
        if overrides_keys_overlap := (set(overrides_dict) & confirmed_feature_keys):
            return _unresolved(
                "config_field_unsupported",
                "Contract overrides must not bake confirmed feature settings: "
                f"{sorted(overrides_keys_overlap)}",
            )
        overrides_keys = set(overrides_dict.keys())
        missing_overrides = non_intent_keys - overrides_keys
        extra_overrides = overrides_keys - non_intent_keys
        duplicate_mapped = overrides_keys & intent_keys

        if missing_overrides or extra_overrides or duplicate_mapped:
            return _unresolved("config_mapping_incomplete", "Overrides mapping contract is incomplete or invalid.")

        # Validate intent boolean gates fail-closed
        if intent.acquisition.exclude_incomplete_final_rwd_chunk is not False:
            return _unresolved("config_field_unsupported", "exclude_incomplete_final_rwd_chunk must be False.")
        if intent.acquisition.acquisition_mode != "intermittent":
            return _unresolved("config_field_unsupported", "acquisition_mode must be intermittent.")
        if intent.execution_profile.traces_only is not False:
            return _unresolved("config_field_unsupported", "traces_only must be False.")
        if intent.output_policy.overwrite is not False:
            return _unresolved("config_field_unsupported", "output_overwrite must be False.")

        # Check for non-default values in prohibited or unsupported configuration fields
        if overrides_dict.get("allow_partial_final_window") is not False:
            return _unresolved("config_field_unsupported", "allow_partial_final_window must be False.")
        if overrides_dict.get("allow_partial_final_chunk") is not False:
            return _unresolved("config_field_unsupported", "allow_partial_final_chunk must be False.")
        if overrides_dict.get("export_display_series_csv") is not False:
            return _unresolved("config_field_unsupported", "export_display_series_csv must be False.")
        if overrides_dict.get("representative_session_index") is not None:
            return _unresolved("config_field_unsupported", "representative_session_index must be None.")
        if overrides_dict.get("preview_first_n") is not None:
            return _unresolved("config_field_unsupported", "preview_first_n must be None.")
        if overrides_dict.get("continuous_window_sec") != 600.0:
            return _unresolved("config_field_unsupported", "continuous_window_sec must be 600.0.")
        if overrides_dict.get("continuous_step_sec") != 600.0:
            return _unresolved("config_field_unsupported", "continuous_step_sec must be 600.0.")

        # Check other unsupported future fields to ensure they match default clean values
        unsupported_defaults = {
            "npm_time_axis": "system_timestamp",
            "npm_frame_col": "Frame",
            "npm_system_ts_col": "SystemTimestamp",
            "npm_computer_ts_col": "ComputerTimestamp",
            "npm_led_col": "LED",
            "npm_region_prefix": "Region",
            "npm_region_suffix": "",
            "custom_tabular_time_col": "Time(s)",
            "custom_tabular_uv_suffix": "-410",
            "custom_tabular_sig_suffix": "-470",
        }
        for field_name, expected_val in unsupported_defaults.items():
            if overrides_dict.get(field_name) != expected_val:
                return _unresolved("config_field_unsupported", f"{field_name} must be {expected_val!r}.")

        # Resolve target_fs_hz from intent semantic values
        target_fs_hz_value = _find_semantic_value(intent.acquisition.semantic_values, "target_fs_hz")
        if target_fs_hz_value is None:
            target_fs_hz_value = _find_semantic_value(intent.correction.dynamic_fit_parameter_values, "target_fs_hz")
        if target_fs_hz_value is None:
            target_fs_hz_value = _find_semantic_value(intent.feature_event.effective_values, "target_fs_hz")
        if target_fs_hz_value is None:
            return _unresolved("config_field_unsupported", "target_fs_hz must be provided in intent semantic values.")

        # 10. Derive config payload
        payload_values = []
        # Populate mapped intent fields
        payload_values.append(GuidedConfigFieldValue("exclude_incomplete_final_rwd_chunk", intent.acquisition.exclude_incomplete_final_rwd_chunk))
        payload_values.append(GuidedConfigFieldValue("target_fs_hz", target_fs_hz_value))
        payload_values.append(GuidedConfigFieldValue("rwd_time_col", intent.acquisition.rwd_time_col))
        payload_values.append(GuidedConfigFieldValue("uv_suffix", intent.acquisition.uv_suffix))
        payload_values.append(GuidedConfigFieldValue("sig_suffix", intent.acquisition.sig_suffix))
        payload_values.append(GuidedConfigFieldValue("dynamic_fit_mode", intent.correction.global_dynamic_fit_mode))
        payload_values.append(GuidedConfigFieldValue("acquisition_mode", intent.acquisition.acquisition_mode))
        
        # Populate overrides
        for name in sorted(serialized_non_intent_keys):
            payload_values.append(GuidedConfigFieldValue(name, overrides_dict[name]))

        # Populate the confirmed Step 5 feature-detection settings. These are the
        # production base feature configuration; a Default-only run analyzes with
        # exactly these values, and a Custom ROI's sparse override is layered on
        # top of them (never on a baked default).
        confirmed_feature_fields, feature_reason = resolve_confirmed_feature_config_fields(intent)
        if confirmed_feature_fields is None:
            return _unresolved("config_field_unsupported", feature_reason)
        for name in sorted(confirmed_feature_keys):
            payload_values.append(
                GuidedConfigFieldValue(name, confirmed_feature_fields[name])
            )

        # Sort values canonical
        sorted_values = tuple(sorted(payload_values, key=lambda x: x.name))

        # Invariant check: ensure config payload keys equal the full set of intended serialized fields
        config_payload_keys = {v.name for v in sorted_values}
        expected_keys = intent_keys | serialized_non_intent_keys | confirmed_feature_keys
        if config_payload_keys != expected_keys:
            return _unresolved("config_mapping_incomplete", "Config payload fields mismatch.")

        provisional_config = GuidedExecutionConfigPayload(
            config_schema_name="photometry_pipeline_config",
            config_mapping_contract_version=startup_mapping_contract.config_mapping_contract_version,
            values=sorted_values,
            canonical_config_payload_identity="0" * 64,
        )
        config_id = compute_guided_execution_config_payload_identity(provisional_config)
        config_payload = replace_config_payload_identity(provisional_config, config_id)

        # 12. Derive candidate manifest payload
        manifest_files = tuple(
            GuidedRunnerCandidateManifestEntry(
                canonical_relative_path=f.canonical_relative_path,
                size_bytes=f.size_bytes,
                sha256_content_digest=f.sha256_content_digest,
            )
            for f in cand_pre.actual_candidates
        )
        provisional_manifest = GuidedRunnerCandidateManifestPayload(
            manifest_schema_name="guided_runner_candidate_manifest",
            manifest_schema_version=startup_mapping_contract.candidate_manifest_schema_version,
            candidate_consumption_contract_version=startup_mapping_contract.candidate_consumption_contract_version,
            source_root_canonical=cand_pre.source_root_canonical if hasattr(cand_pre, "source_root_canonical") else intent.input_source.source_root_canonical,
            source_candidate_set_digest=cand_pre.actual_candidate_set_digest,
            source_candidate_content_digest=cand_pre.actual_candidate_content_digest,
            candidate_files=manifest_files,
            parser_contract_digest=roi_pre.parser_contract_digest,
            discovered_roi_ids=roi_pre.actual_discovered_roi_ids,
            included_roi_ids=roi_pre.actual_included_roi_ids,
            excluded_roi_ids=roi_pre.actual_excluded_roi_ids,
            strict_roi_inventory_digest=roi_pre.actual_strict_roi_inventory_digest,
            candidate_preflight_identity=authorized_result.candidate_preflight_identity,
            roi_preflight_identity=authorized_result.roi_preflight_identity,
            canonical_candidate_manifest_payload_identity="0" * 64,
        )
        manifest_id = compute_guided_runner_candidate_manifest_payload_identity(provisional_manifest)
        candidate_manifest_payload = replace_manifest_payload_identity(provisional_manifest, manifest_id)

        # Runner exact consumption exists post-4J14l, but no startup
        # transaction serializes these payloads or launches execution.
        # Derive provenance seed with runner_request_identity=None and runnable=False
        provisional_seed = GuidedStartupProvenanceSeed(
            provenance_schema_name="guided_startup_provenance_seed",
            provenance_schema_version=startup_mapping_contract.startup_provenance_schema_version,
            startup_mapping_contract_version=startup_mapping_contract.contract_version,
            validation_request_identity=authorized_result.stored_request_identity,
            authorization_identity=authorized_result.canonical_authorization_identity,
            production_intent_identity=authorized_result.production_intent_identity,
            application_build_identity=authorized_result.application_build_identity,
            production_mapping_contract_version=intent.mapping_contract_version,
            runner_contract_version=intent.runner_contract_version,
            candidate_preflight_identity=authorized_result.candidate_preflight_identity,
            roi_preflight_identity=authorized_result.roi_preflight_identity,
            config_payload_identity=config_id,
            candidate_manifest_payload_identity=manifest_id,
            runner_request_identity=None,
            runnable=False,
            canonical_provenance_seed_identity="0" * 64,
        )
        seed_id = compute_guided_startup_provenance_seed_identity(provisional_seed)
        provenance_seed = replace_provenance_seed_identity(provisional_seed, seed_id)

        limiting_issue = GuidedExecutionPayloadIssue(
            category="startup_transaction_unavailable",
            section="guided_execution_payload",
            message=(
                "Exact runner manifest/ROI consumption is available, but no "
                "startup transaction exists to serialize payloads, allocate "
                "output, build an invocation, or launch the runner."
            ),
        )

        return GuidedExecutionPayloadDerivationResult(
            status=GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE,
            ok=True,
            runnable=False,
            config_payload=config_payload,
            candidate_manifest_payload=candidate_manifest_payload,
            runner_request=None,
            provenance_seed=provenance_seed,
            config_payload_identity=config_id,
            candidate_manifest_payload_identity=manifest_id,
            runner_request_identity=None,
            provenance_seed_identity=seed_id,
            limiting_issues=(limiting_issue,),
            blocking_issues=(),
        )

    except Exception:
        return _unresolved(
            "payload_internal_error",
            "An unexpected error occurred during execution payload derivation.",
            detail_code="payload_derivation_exception",
        )


# Internal dataclass replace helper utilities
def replace_config_payload_identity(payload: GuidedExecutionConfigPayload, identity: str) -> GuidedExecutionConfigPayload:
    from dataclasses import replace
    return replace(payload, canonical_config_payload_identity=identity)


def replace_manifest_payload_identity(payload: GuidedRunnerCandidateManifestPayload, identity: str) -> GuidedRunnerCandidateManifestPayload:
    from dataclasses import replace
    return replace(payload, canonical_candidate_manifest_payload_identity=identity)


def replace_runner_request_identity(request: GuidedRunnerExecutionRequest, identity: str) -> GuidedRunnerExecutionRequest:
    from dataclasses import replace
    return replace(request, canonical_runner_request_identity=identity)


def replace_provenance_seed_identity(seed: GuidedStartupProvenanceSeed, identity: str) -> GuidedStartupProvenanceSeed:
    from dataclasses import replace
    return replace(seed, canonical_provenance_seed_identity=identity)
