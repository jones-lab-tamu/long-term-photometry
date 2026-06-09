from dataclasses import dataclass, field
import dataclasses
from typing import List, Optional, Dict, Literal
import yaml
import os
import math

@dataclass
class Config:
    # chunking & timing
    chunk_duration_sec: float = 600.0
    trim_samples_start: int = 0
    trim_samples_end: int = 0
    
    seed: int = 0
    
    # Config Parameters Used by Strict Mode
    allow_partial_final_chunk: bool = False # If False, enforces strict grid coverage & reciprocity
    # chunk_duration_sec : Defines strict grid length
    # target_fs_hz       : Defines strict grid spacing
    # npm_*              : Used for strict time axis parsing
    
    # filters
    target_fs_hz: float = 40.0
    lowpass_hz: float = 1.0
    filter_order: int = 3
    
    # regression
    window_sec: float = 60.0
    step_sec: float = 10.0
    r_low: float = 0.2
    r_high: float = 0.8
    g_min: float = 0.2
    min_samples_per_window: int = 0  # 0 implies dynamic 80%
    min_valid_windows: int = 5
    baseline_subtract_before_fit: bool = False
    bleach_correction_mode: Literal[
        'none',
        'single_exponential',
        'double_exponential',
    ] = 'none'
    dynamic_fit_mode: Literal[
        'rolling_local_regression',
        'rolling_filtered_to_raw',
        'rolling_filtered_to_filtered',
        'global_linear_regression',
        'robust_global_event_reject',
        'adaptive_event_gated_regression',
    ] = 'robust_global_event_reject'
    dynamic_fit_slope_constraint: Literal['unconstrained', 'nonnegative'] = 'unconstrained'
    dynamic_fit_min_slope: float = 0.0
    robust_event_reject_max_iters: int = 3
    robust_event_reject_residual_z_thresh: float = 3.5
    robust_event_reject_local_var_window_sec: Optional[float] = 10.0
    robust_event_reject_local_var_ratio_thresh: Optional[float] = None
    robust_event_reject_min_keep_fraction: float = 0.5
    adaptive_event_gate_residual_z_thresh: float = 3.5
    adaptive_event_gate_local_var_window_sec: Optional[float] = 10.0
    adaptive_event_gate_local_var_ratio_thresh: Optional[float] = None
    adaptive_event_gate_smooth_window_sec: float = 60.0
    adaptive_event_gate_min_trust_fraction: float = 0.5
    adaptive_event_gate_freeze_interp_method: Literal['linear_hold'] = 'linear_hold'
    
    # baseline
    baseline_method: Literal['uv_raw_percentile_session', 'uv_globalfit_percentile_session'] = 'uv_raw_percentile_session'
    baseline_percentile: float = 10.0
    f0_min_value: float = 1e-9
    
    # npm specific
    npm_time_axis: Literal['system_timestamp', 'computer_timestamp'] = 'system_timestamp'
    
    # system
    sampling_rate_hz_fallback: float = 40.0
    timestamp_cv_max: float = 0.02
    duration_tolerance_frac: float = 0.02
    qc_max_chunk_fail_fraction: float = 0.20
    
    # peak detection (exposed params per user request)
    peak_threshold_method: str = 'mean_std'
    peak_threshold_k: float = 2.5
    peak_threshold_percentile: float = 95.0
    peak_threshold_abs: float = 0.0 # Used only when method is 'absolute'
    peak_min_distance_sec: float = 1.0
    peak_min_prominence_k: float = 2.0
    peak_min_width_sec: float = 0.3
    peak_pre_filter: str = 'none'
    event_auc_baseline: str = 'zero'
    event_signal: Literal['dff', 'delta_f'] = 'dff'
    signal_excursion_polarity: Literal['positive', 'negative', 'both'] = 'positive'
    representative_session_index: Optional[int] = None
    preview_first_n: Optional[int] = None
    
    # adapters
    adapter_value_nan_policy: str = 'strict'
    tonic_allowed_nan_frac: float = 0.0
    tonic_output_mode: Literal[
        'preserve_raw_session_shape',
        'flatten_session_bleach_preserve_session_baseline',
    ] = 'preserve_raw_session_shape'
    tonic_timeline_mode: Literal[
        'real_elapsed_time',
        'gap_free_elapsed_time',
        'compressed_recording_time',
    ] = 'real_elapsed_time'
    export_display_series_csv: bool = False

    # channel identifiers - MUST be provided in config (no defaults for these essentially)
    rwd_time_col: str = "Time(s)" # Default often seen, but user should override
    uv_suffix: str = "-410"
    sig_suffix: str = "-470"
    
    npm_frame_col: str = "FrameCounter"
    npm_system_ts_col: str = "SystemTimestamp"
    npm_computer_ts_col: str = "ComputerTimestamp"
    npm_led_col: str = "LedState"
    npm_region_prefix: str = "Region"
    npm_region_suffix: str = "G"
    custom_tabular_time_col: str = "time_sec"
    custom_tabular_uv_suffix: str = "_iso"
    custom_tabular_sig_suffix: str = "_sig"

    # acquisition planning (phase 1 continuous-mode plumbing only)
    acquisition_mode: Literal['intermittent', 'continuous'] = 'intermittent'
    continuous_window_sec: float = 600.0
    continuous_step_sec: float = 600.0
    # Safer default is False so any trailing undersized window is not silently
    # admitted before explicit continuous-window ingestion is implemented.
    allow_partial_final_window: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("Config YAML must be a mapping/dict.")
            
        # Strict Validation
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
            
        # Enum Validation
        if 'baseline_method' in data:
            if data['baseline_method'] not in {'uv_raw_percentile_session', 'uv_globalfit_percentile_session'}:
                raise ValueError(f"Invalid baseline_method: {data['baseline_method']}. Allowed: {{'uv_raw_percentile_session', 'uv_globalfit_percentile_session'}}")
                
        if 'npm_time_axis' in data:
            if data['npm_time_axis'] not in {'system_timestamp', 'computer_timestamp'}:
                raise ValueError(f"Invalid npm_time_axis: {data['npm_time_axis']}. Allowed: {{'system_timestamp', 'computer_timestamp'}}")
                
        if 'peak_threshold_method' in data:
            if data['peak_threshold_method'] not in {'mean_std', 'percentile', 'median_mad', 'absolute'}:
                raise ValueError(f"Invalid peak_threshold_method: {data['peak_threshold_method']}. Allowed: {{'mean_std', 'percentile', 'median_mad', 'absolute'}}")

        if 'peak_pre_filter' in data:
            if data['peak_pre_filter'] not in {'none', 'lowpass'}:
                raise ValueError(f"Invalid peak_pre_filter: {data['peak_pre_filter']}. Allowed: {{'none', 'lowpass'}}")
                
        if 'event_auc_baseline' in data:
             if data['event_auc_baseline'] not in {'zero', 'median'}:
                raise ValueError(f"Invalid event_auc_baseline: {data['event_auc_baseline']}. Allowed: {{'zero', 'median'}}")
                
        if 'event_signal' in data:
             if data['event_signal'] not in {'dff', 'delta_f'}:
                raise ValueError(f"Invalid event_signal: {data['event_signal']}. Allowed: {{'dff', 'delta_f'}}")
        if 'signal_excursion_polarity' in data:
             if data['signal_excursion_polarity'] not in {'positive', 'negative', 'both'}:
                raise ValueError(
                    f"Invalid signal_excursion_polarity: {data['signal_excursion_polarity']}. "
                    "Allowed: {'positive', 'negative', 'both'}"
                )

        if 'dynamic_fit_mode' in data:
            if data['dynamic_fit_mode'] not in {
                'rolling_local_regression',
                'rolling_filtered_to_raw',
                'rolling_filtered_to_filtered',
                'global_linear_regression',
                'robust_global_event_reject',
                'adaptive_event_gated_regression',
            }:
                raise ValueError(
                    f"Invalid dynamic_fit_mode: {data['dynamic_fit_mode']}. "
                    "Allowed: {'rolling_local_regression', 'rolling_filtered_to_raw', "
                    "'rolling_filtered_to_filtered', 'global_linear_regression', "
                    "'robust_global_event_reject', 'adaptive_event_gated_regression'}"
                )
        if 'dynamic_fit_slope_constraint' in data:
            if data['dynamic_fit_slope_constraint'] not in {'unconstrained', 'nonnegative'}:
                raise ValueError(
                    f"Invalid dynamic_fit_slope_constraint: {data['dynamic_fit_slope_constraint']}. "
                    "Allowed: {'unconstrained', 'nonnegative'}"
                )
        if 'bleach_correction_mode' in data:
            if data['bleach_correction_mode'] not in {
                'none',
                'single_exponential',
                'double_exponential',
            }:
                raise ValueError(
                    f"Invalid bleach_correction_mode: {data['bleach_correction_mode']}. "
                    "Allowed: {'none', 'single_exponential', 'double_exponential'}"
                )
                
        if 'adapter_value_nan_policy' in data:
             if data['adapter_value_nan_policy'] not in {'strict', 'mask'}:
                raise ValueError(f"Invalid adapter_value_nan_policy: {data['adapter_value_nan_policy']}. Allowed: {{'strict', 'mask'}}")

        if 'tonic_output_mode' in data:
             if data['tonic_output_mode'] not in {
                 'preserve_raw_session_shape',
                 'flatten_session_bleach_preserve_session_baseline',
             }:
                raise ValueError(
                    f"Invalid tonic_output_mode: {data['tonic_output_mode']}. "
                    "Allowed: {'preserve_raw_session_shape', "
                    "'flatten_session_bleach_preserve_session_baseline'}"
                )

        if 'tonic_timeline_mode' in data:
            if data['tonic_timeline_mode'] not in {
                'real_elapsed_time',
                'gap_free_elapsed_time',
                'compressed_recording_time',
            }:
                raise ValueError(
                    f"Invalid tonic_timeline_mode: {data['tonic_timeline_mode']}. "
                    "Allowed: {'real_elapsed_time', 'gap_free_elapsed_time'} "
                    "(legacy alias 'compressed_recording_time' is also accepted)"
                )
        if 'export_display_series_csv' in data:
            if not isinstance(data['export_display_series_csv'], bool):
                raise ValueError(
                    "export_display_series_csv must be a boolean (true/false)"
                )
        if 'acquisition_mode' in data:
            if data['acquisition_mode'] not in {'intermittent', 'continuous'}:
                raise ValueError(
                    f"Invalid acquisition_mode: {data['acquisition_mode']}. "
                    "Allowed: {'intermittent', 'continuous'}"
                )
        
        obj = cls(**data)
        
        if obj.peak_threshold_method == 'absolute':
            if obj.peak_threshold_abs <= 0.0:
                raise ValueError("peak_threshold_abs must be > 0 when peak_threshold_method='absolute'")
        if obj.signal_excursion_polarity not in {'positive', 'negative', 'both'}:
            raise ValueError(
                "signal_excursion_polarity must be one of {'positive', 'negative', 'both'}"
            )
        if obj.peak_min_prominence_k < 0.0:
            raise ValueError("peak_min_prominence_k must be >= 0")
        if obj.peak_min_width_sec < 0.0:
            raise ValueError("peak_min_width_sec must be >= 0")

        if obj.preview_first_n is not None:
            if not isinstance(obj.preview_first_n, int) or obj.preview_first_n <= 0:
                raise ValueError("preview_first_n must be an int > 0")

        if obj.robust_event_reject_max_iters < 1:
            raise ValueError("robust_event_reject_max_iters must be >= 1")
        if obj.robust_event_reject_residual_z_thresh <= 0.0:
            raise ValueError("robust_event_reject_residual_z_thresh must be > 0")
        if obj.robust_event_reject_local_var_window_sec is not None:
            if obj.robust_event_reject_local_var_window_sec <= 0.0:
                raise ValueError("robust_event_reject_local_var_window_sec must be > 0 when provided")
        if obj.robust_event_reject_local_var_ratio_thresh is not None:
            if obj.robust_event_reject_local_var_ratio_thresh <= 0.0:
                raise ValueError("robust_event_reject_local_var_ratio_thresh must be > 0 when provided")
        if not (0.0 < obj.robust_event_reject_min_keep_fraction <= 1.0):
            raise ValueError("robust_event_reject_min_keep_fraction must be in (0, 1]")
        if obj.adaptive_event_gate_residual_z_thresh <= 0.0:
            raise ValueError("adaptive_event_gate_residual_z_thresh must be > 0")
        if obj.adaptive_event_gate_local_var_window_sec is not None:
            if obj.adaptive_event_gate_local_var_window_sec <= 0.0:
                raise ValueError("adaptive_event_gate_local_var_window_sec must be > 0 when provided")
        if obj.adaptive_event_gate_local_var_ratio_thresh is not None:
            if obj.adaptive_event_gate_local_var_ratio_thresh <= 0.0:
                raise ValueError("adaptive_event_gate_local_var_ratio_thresh must be > 0 when provided")
        if obj.adaptive_event_gate_smooth_window_sec <= 0.0:
            raise ValueError("adaptive_event_gate_smooth_window_sec must be > 0")
        if not (0.0 < obj.adaptive_event_gate_min_trust_fraction <= 1.0):
            raise ValueError("adaptive_event_gate_min_trust_fraction must be in (0, 1]")
        if obj.adaptive_event_gate_freeze_interp_method not in {"linear_hold"}:
            raise ValueError("adaptive_event_gate_freeze_interp_method must be 'linear_hold'")
        if obj.dynamic_fit_slope_constraint not in {"unconstrained", "nonnegative"}:
            raise ValueError(
                "dynamic_fit_slope_constraint must be one of {'unconstrained', 'nonnegative'}"
            )
        try:
            obj.dynamic_fit_min_slope = float(obj.dynamic_fit_min_slope)
        except Exception as exc:
            raise ValueError("dynamic_fit_min_slope must be a finite float") from exc
        if not math.isfinite(obj.dynamic_fit_min_slope):
            raise ValueError("dynamic_fit_min_slope must be a finite float")
        if (
            obj.dynamic_fit_slope_constraint == "nonnegative"
            and obj.dynamic_fit_min_slope < 0.0
        ):
            raise ValueError(
                "dynamic_fit_min_slope must be >= 0 when "
                "dynamic_fit_slope_constraint is 'nonnegative'"
            )
        if obj.bleach_correction_mode not in {"none", "single_exponential", "double_exponential"}:
            raise ValueError(
                "bleach_correction_mode must be one of {'none', 'single_exponential', 'double_exponential'}"
            )
        for field_name in (
            "custom_tabular_time_col",
            "custom_tabular_uv_suffix",
            "custom_tabular_sig_suffix",
        ):
            value = str(getattr(obj, field_name, "")).strip()
            if not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if str(obj.custom_tabular_uv_suffix) == str(obj.custom_tabular_sig_suffix):
            raise ValueError(
                "custom_tabular_uv_suffix and custom_tabular_sig_suffix must be different"
            )
        if obj.acquisition_mode not in {'intermittent', 'continuous'}:
            raise ValueError(
                "acquisition_mode must be one of {'intermittent', 'continuous'}"
            )
        if obj.continuous_window_sec <= 0.0:
            raise ValueError("continuous_window_sec must be > 0")
        if obj.continuous_step_sec <= 0.0:
            raise ValueError("continuous_step_sec must be > 0")
        if abs(float(obj.continuous_step_sec) - float(obj.continuous_window_sec)) > 1e-9:
            raise ValueError(
                "continuous_step_sec must equal continuous_window_sec in this version; "
                "overlapping/sliding windows are not yet supported."
            )
                
        return obj
