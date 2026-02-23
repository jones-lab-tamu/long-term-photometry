from dataclasses import dataclass, field
import dataclasses
from typing import List, Optional, Dict, Literal
import yaml
import os

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
    peak_threshold_k: float = 2.0
    peak_threshold_percentile: float = 95.0
    peak_min_distance_sec: float = 0.5 # Default kept at 0.5 per user request (was 1.0 in previous file but user asked to keep default 0.5)
    peak_pre_filter: str = 'none'
    event_auc_baseline: str = 'zero'
    
    # adapters
    adapter_value_nan_policy: str = 'strict'
    tonic_allowed_nan_frac: float = 0.0

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
            if data['peak_threshold_method'] not in {'mean_std', 'percentile', 'median_mad'}:
                raise ValueError(f"Invalid peak_threshold_method: {data['peak_threshold_method']}. Allowed: {{'mean_std', 'percentile', 'median_mad'}}")

        if 'peak_pre_filter' in data:
            if data['peak_pre_filter'] not in {'none', 'lowpass'}:
                raise ValueError(f"Invalid peak_pre_filter: {data['peak_pre_filter']}. Allowed: {{'none', 'lowpass'}}")
                
        if 'event_auc_baseline' in data:
             if data['event_auc_baseline'] not in {'zero', 'median'}:
                raise ValueError(f"Invalid event_auc_baseline: {data['event_auc_baseline']}. Allowed: {{'zero', 'median'}}")
                
        if 'adapter_value_nan_policy' in data:
             if data['adapter_value_nan_policy'] not in {'strict', 'mask'}:
                raise ValueError(f"Invalid adapter_value_nan_policy: {data['adapter_value_nan_policy']}. Allowed: {{'strict', 'mask'}}")
        
        return cls(**data)
