import json
import yaml
import os
import pathlib
import numpy as np
from typing import Dict, Any
from ..config import Config

def make_json_safe(obj: Any) -> Any:
    """
    Recursively converts objects to JSON-safe primitives.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isfinite(obj):
            return float(obj)
        return str(obj) # NaN/Inf to string
    if isinstance(obj, np.generic):
        return make_json_safe(obj.item())
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, (list, tuple, np.ndarray)):
        # Handle np scalar edge case if array is 0-d? No, usually list.
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    # Fallback
    return str(obj)

def generate_run_report(config: Config, output_dir: str, roi_selection: Dict = None, traces_only: bool = False, representative_info: Dict = None, preview_info: Dict = None, sessions_per_hour: int = None, sessions_per_hour_source: str = None):
    """
    Generates the mandatory run-report artifact.
    Freezes analytical assumptions and flags tonic-attenuation risk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Configuration Snapshot
    # Extract, copy, and clean
    raw_config = config.__dict__.copy()
    config_snapshot = make_json_safe(raw_config)
    
    # 2. Derived Settings
    tonic_period_interest = getattr(config, 'tonic_period_of_interest_sec', 86400.0)
    tonic_warning_ratio = getattr(config, 'tonic_warning_ratio_threshold', 0.5)
    
    strict_enabled = not config.allow_partial_final_chunk
    tonic_attenuation_warning = (config.window_sec < (tonic_warning_ratio * tonic_period_interest))
    
    # Preserve resolved sessions_per_hour (source of truth from runner/GUI)
    # If not provided, it remains NULL in derived_settings
    resolved_sph = sessions_per_hour
    resolved_sph_source = sessions_per_hour_source
    
    warnings_list = []
    
    # Nyquist Check
    nyquist = config.target_fs_hz / 2.0
    if config.lowpass_hz >= nyquist:
        warnings_list.append(
            f"Lowpass filter disabled: cutoff {config.lowpass_hz}Hz >= Nyquist {nyquist}Hz (fs={config.target_fs_hz}Hz). Raw passed."
        )
        
    if tonic_attenuation_warning:
        warnings_list.append("Regression window may attenuate slow tonic structure.")

    derived_settings_raw = {
        "regression_window_sec": config.window_sec,
        "regression_step_sec": config.step_sec,
        "target_fs_hz": config.target_fs_hz,
        "chunk_duration_sec": config.chunk_duration_sec,
        "sessions_per_hour": resolved_sph,
        "sessions_per_hour_source": resolved_sph_source,
        "strict_mode_enabled": strict_enabled,
        "tonic_period_of_interest_sec": tonic_period_interest,
        "tonic_warning_ratio_threshold": tonic_warning_ratio,
        "tonic_attenuation_warning": tonic_attenuation_warning,
        "warnings": warnings_list
    }
    derived_settings = make_json_safe(derived_settings_raw)
    
    # 3. Declared Analytical Contract (Verbatim)
    # Hardcoded as strictly requested
    contract = {
        "strict_mode_guarantees": [
            "timestamps are strictly increasing (per channel where relevant)",
            "resampling onto a fixed grid of length round(chunk_duration_sec * target_fs_hz)",
            "no extrapolation beyond available data (strict start/end coverage enforced)",
            "NPM UV and SIG are validated independently (per-channel checks)"
        ],
        "pipeline_limitations": [
            "no tonic-phasic decomposition",
            "no circadian fitting",
            "no Bayesian correction"
        ],
        "signal_semantics": {
            "uv_raw": "raw isosbestic channel on the canonical grid",
            "sig_raw": "raw calcium-dependent channel on the canonical grid",
            "uv_fit": "estimated artifact component derived from uv_filt fit to sig_filt",
            "delta_f": "sig_raw - uv_fit (artifact-corrected numerator only)",
            "dff": "100 * delta_f / F0 (if computed)"
        }
    }
    
    # 4. Baseline Semantics (Dynamic)
    baseline_semantics = {}
    
    if config.baseline_method == 'uv_raw_percentile_session':
        baseline_semantics = {
            "method": "uv_raw_percentile_session",
            "f0_source": "uv_raw",
            "f0_units": "uv-scale",
            "dff_formula": "100 * (sig_raw - uv_fit) / F0",
            "interpretation_note": "F0 is the isosbestic baseline. dFF represents signal change relative to isosbestic baseline scale."
        }
    elif config.baseline_method == 'uv_globalfit_percentile_session':
        baseline_semantics = {
            "method": "uv_globalfit_percentile_session",
            "f0_source": "uv_est",
            "f0_units": "signal-scale",
            "dff_formula": "100 * (sig_raw - uv_fit) / F0",
            "interpretation_note": "F0 is the mapped isosbestic baseline (scaled to signal). dFF is relative to signal scale."
        }
    else:
        # Strict contract: unknown method is an error
        raise ValueError(f"Unknown baseline_method: {config.baseline_method}")
        
    contract["baseline_semantics"] = baseline_semantics
 
    # 5. Output
    report = {
        "run_context": {
            "run_type": "preview" if preview_info is not None else "full",
            "features_extracted": False if traces_only else None,
            "preview": preview_info,
            "traces_only": traces_only,
            "event_signal": getattr(config, 'event_signal', 'dff'),
            "signal_excursion_polarity": getattr(config, "signal_excursion_polarity", "positive"),
            "bleach_correction_mode": getattr(config, "bleach_correction_mode", "none"),
            "custom_tabular_contract": {
                "session_model": "one_csv_per_session",
                "time_col": getattr(config, "custom_tabular_time_col", "time_sec"),
                "uv_suffix": getattr(config, "custom_tabular_uv_suffix", "_iso"),
                "sig_suffix": getattr(config, "custom_tabular_sig_suffix", "_sig"),
            },
            "sessions_per_hour": resolved_sph,
            "sessions_per_hour_source": resolved_sph_source,
            "representative_session_index": representative_info.get("representative_session_index") if representative_info else None,
            "representative_session_id": representative_info.get("representative_session_id") if representative_info else None,
            "n_sessions_resolved": representative_info.get("n_sessions_resolved", 0) if representative_info else 0,
            "user_provided_representative_session_index": representative_info.get("user_provided", False) if representative_info else False
        },
        "configuration": config_snapshot,
        "derived_settings": derived_settings,
        "analytical_contract": contract
    }

    
    if roi_selection is not None:
        report["roi_selection"] = make_json_safe(roi_selection)
    
    with open(os.path.join(output_dir, "run_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(config_snapshot, f)

def append_run_report_warnings(output_dir: str, new_warnings: list):
    """
    Read-Modify-Write run_report.json to append warnings without regeneration.
    """
    path = os.path.join(output_dir, "run_report.json")
    if not os.path.exists(path):
        return
        
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        current = data["derived_settings"].get("warnings", [])
        # Deduplicate
        uniq = list(current)
        for w in new_warnings:
            if w not in uniq:
                uniq.append(w)
                
        data["derived_settings"]["warnings"] = uniq
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass # Best effort
