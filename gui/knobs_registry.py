"""
GUI knob registry: allowlists, metadata, and filtering.

Defines which Config keys are exposed in the GUI (categorized into
NORMAL, ADVANCED, and DEVELOPER tiers), provides human-readable metadata
for each, and ensures GUI can never write unknown config keys.

No PySide6 dependency.
"""

from typing import Dict, Set

from gui.knobs_schema import get_config_field_specs, is_config_key


# ======================================================================
# Allowlists - every name MUST be a real Config field.
# ======================================================================

GUI_KNOBS_NORMAL: Set[str] = set()

GUI_KNOBS_ADVANCED: Set[str] = {
    # Preprocessing + Baseline
    "lowpass_hz",
    "baseline_method",
    "baseline_percentile",
    "f0_min_value",
    # Dynammic Correction
    "dynamic_fit_mode",
    "baseline_subtract_before_fit",
    "window_sec",
    "step_sec",
    "min_valid_windows",
    "r_low",
    "r_high",
    "g_min",
    "min_samples_per_window",
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
    # Peak/Event
    "event_signal",
    "peak_threshold_method",
    "peak_threshold_k",
    "peak_threshold_percentile",
    "peak_threshold_abs",
    "peak_min_distance_sec",
    "peak_min_prominence_k",
    "peak_min_width_sec",
    "peak_pre_filter",
    "event_auc_baseline",
}

GUI_KNOBS_DEVELOPER: Set[str] = {
    "allow_partial_final_chunk",
    "timestamp_cv_max",
    "duration_tolerance_frac",
    "qc_max_chunk_fail_fraction",
    "adapter_value_nan_policy",
    "tonic_allowed_nan_frac",
}


# ======================================================================
# Metadata — every allowlisted key must have at least label + help.
# ======================================================================

KNOB_META: Dict[str, dict] = {
    # --- NORMAL ---
    # --- ADVANCED ---
    "event_signal": {
        "label": "Event Signal",
        "help": "Signal type for event detection: 'dff' or 'delta_f'.",
    },
    "lowpass_hz": {
        "label": "Lowpass Filter (Hz)",
        "help": "Cutoff frequency for lowpass filtering.",
        "range": {"min": 1e-9},
    },
    "baseline_method": {
        "label": "Baseline Method",
        "help": "Method used to compute the baseline.",
    },
    "baseline_percentile": {
        "label": "Baseline Percentile",
        "help": "Percentile threshold, used when the baseline method uses a percentile.",
        "range": {"min": 0.0, "max": 100.0},
    },
    "f0_min_value": {
        "label": "F0 Min Value",
        "help": "Minimum allowed value for the baseline F0.",
        "range": {"min": 0.0},
    },
    "peak_threshold_method": {
        "label": "Peak Threshold Method",
        "help": "Algorithm for computing the peak detection threshold.",
    },
    "peak_threshold_k": {
        "label": "Peak Threshold K",
        "help": "Number of standard deviations for mean_std method.",
        "range": {"min": 0.0, "max": 10.0},
    },
    "peak_threshold_percentile": {
        "label": "Peak Threshold Percentile",
        "help": "Percentile for percentile-based threshold method.",
        "range": {"min": 0.0, "max": 100.0},
    },
    "peak_threshold_abs": {
        "label": "Peak Threshold Absolute",
        "help": "Absolute threshold value (used when method is 'absolute').",
        "visible_when": [{"key": "peak_threshold_method", "equals": "absolute"}],
    },
    "peak_min_distance_sec": {
        "label": "Peak Min Distance (sec)",
        "help": "Minimum time between detected peaks.",
        "range": {"min": 0.0, "max": 60.0},
    },
    "peak_min_prominence_k": {
        "label": "Peak Min Prominence K",
        "help": "Minimum peak prominence relative to robust noise (MAD-based sigma).",
        "range": {"min": 0.0},
    },
    "peak_min_width_sec": {
        "label": "Peak Min Width (sec)",
        "help": "Minimum event width in seconds.",
        "range": {"min": 0.0},
    },
    "peak_pre_filter": {
        "label": "Peak Pre-Filter",
        "help": "Optional lowpass filter before peak detection.",
    },
    "event_auc_baseline": {
        "label": "Event AUC Baseline",
        "help": "Baseline method for AUC calculation: 'zero' or 'median'.",
    },
    "window_sec": {
        "label": "Regression Window (sec)",
        "help": "Sliding window size for regression analysis.",
    },
    "dynamic_fit_mode": {
        "label": "Dynamic Fit Mode",
        "help": "Isosbestic fit engine selector for rolling, global linear, or robust global event-reject fitting.",
    },
    "baseline_subtract_before_fit": {
        "label": "Baseline Subtract Before Fit",
        "help": "If enabled, subtract a moving baseline from fit-input traces before dynamic-fit estimation in rolling modes.",
    },
    "step_sec": {
        "label": "Regression Step (sec)",
        "help": "Legacy/inactive under rolling local regression; retained for compatibility metadata.",
    },
    "min_valid_windows": {
        "label": "Min Valid Windows",
        "help": "Legacy/inactive under rolling local regression; retained for compatibility metadata.",
    },
    "r_low": {
        "label": "R-Low Threshold",
        "help": "Legacy/inactive under rolling local regression; retained for compatibility metadata.",
        "range": {"min": 0.0, "max": 1.0},
    },
    "r_high": {
        "label": "R-High Threshold",
        "help": "Legacy/inactive under rolling local regression; retained for compatibility metadata.",
        "range": {"min": 0.0, "max": 1.0},
    },
    "g_min": {
        "label": "G-Min Threshold",
        "help": "Legacy/inactive under rolling local regression; retained for compatibility metadata.",
        "range": {"min": 0.0},
    },
    "min_samples_per_window": {
        "label": "Min Samples per Window",
        "help": "Minimum valid samples per window.",
    },
    "robust_event_reject_max_iters": {
        "label": "Robust Event-Reject Max Iters",
        "help": "Maximum rejection/refit passes for robust global event-reject mode.",
        "range": {"min": 1.0},
    },
    "robust_event_reject_residual_z_thresh": {
        "label": "Robust Event-Reject Residual Z",
        "help": "Positive residual robust-z threshold used to exclude likely event-dominated samples.",
        "range": {"min": 0.0},
    },
    "robust_event_reject_local_var_window_sec": {
        "label": "Robust Event-Reject Local Var Window (s)",
        "help": "Centered window used for optional local-variance asymmetry screening.",
        "range": {"min": 0.0},
    },
    "robust_event_reject_local_var_ratio_thresh": {
        "label": "Robust Event-Reject Local Var Ratio",
        "help": "Optional local variance ratio threshold (signal/iso). Disabled when unset.",
        "range": {"min": 0.0},
    },
    "robust_event_reject_min_keep_fraction": {
        "label": "Robust Event-Reject Min Keep Fraction",
        "help": "Lower keep-fraction guardrail for robust sample exclusion.",
        "range": {"min": 0.0, "max": 1.0},
    },
    "adaptive_event_gate_residual_z_thresh": {
        "label": "Adaptive Event-Gate Residual Z",
        "help": "Positive residual robust-z threshold used for adaptive trust gating.",
        "range": {"min": 0.0},
    },
    "adaptive_event_gate_local_var_window_sec": {
        "label": "Adaptive Event-Gate Local Var Window (s)",
        "help": "Centered local-variance window used by adaptive trust gating.",
        "range": {"min": 0.0},
    },
    "adaptive_event_gate_local_var_ratio_thresh": {
        "label": "Adaptive Event-Gate Local Var Ratio",
        "help": "Optional local variance ratio threshold (signal/iso) for adaptive gating.",
        "range": {"min": 0.0},
    },
    "adaptive_event_gate_smooth_window_sec": {
        "label": "Adaptive Event-Gate Smooth Window (s)",
        "help": "Smoothing window for adaptive local coefficient traces.",
        "range": {"min": 0.0},
    },
    "adaptive_event_gate_min_trust_fraction": {
        "label": "Adaptive Event-Gate Min Trust Fraction",
        "help": "Lower trusted-data fraction guardrail for adaptive mode.",
        "range": {"min": 0.0, "max": 1.0},
    },
    "adaptive_event_gate_freeze_interp_method": {
        "label": "Adaptive Event-Gate Freeze Interp Method",
        "help": "Interpolation/freezing strategy used through gated spans.",
    },
    # --- DEVELOPER ---
    "allow_partial_final_chunk": {
        "label": "Allow Partial Final Chunk",
        "help": "If True, the last chunk may be shorter than chunk_duration_sec.",
    },
    "timestamp_cv_max": {
        "label": "Timestamp CV Max",
        "help": "Maximum allowed coefficient of variation for timestamp intervals.",
    },
    "duration_tolerance_frac": {
        "label": "Duration Tolerance Fraction",
        "help": "Fractional tolerance for session duration validation.",
    },
    "qc_max_chunk_fail_fraction": {
        "label": "QC Max Chunk Fail Fraction",
        "help": "Maximum fraction of chunks allowed to fail QC.",
        "range": {"min": 0.0, "max": 1.0},
    },
    "adapter_value_nan_policy": {
        "label": "Adapter NaN Policy",
        "help": "How adapters handle NaN values: 'strict' or 'mask'.",
    },
    "tonic_allowed_nan_frac": {
        "label": "Tonic Allowed NaN Fraction",
        "help": "Maximum fraction of NaN values allowed in tonic computation.",
        "range": {"min": 0.0, "max": 1.0},
    },
}


# ======================================================================
# Validation
# ======================================================================

def validate_registry_against_schema() -> None:
    """Assert registry is consistent with Config schema.

    Checks:
    - Every allowlisted key exists in Config
    - No overlap between NORMAL / ADVANCED / DEVELOPER
    - Every allowlisted key has label + help in KNOB_META
    """
    schema = get_config_field_specs()
    all_knobs = GUI_KNOBS_NORMAL | GUI_KNOBS_ADVANCED | GUI_KNOBS_DEVELOPER

    # Every key must be a real Config field
    unknown = all_knobs - set(schema.keys())
    assert not unknown, f"Registry contains non-Config keys: {unknown}"

    # No overlap
    na = GUI_KNOBS_NORMAL & GUI_KNOBS_ADVANCED
    nd = GUI_KNOBS_NORMAL & GUI_KNOBS_DEVELOPER
    ad = GUI_KNOBS_ADVANCED & GUI_KNOBS_DEVELOPER
    assert not na, f"NORMAL/ADVANCED overlap: {na}"
    assert not nd, f"NORMAL/DEVELOPER overlap: {nd}"
    assert not ad, f"ADVANCED/DEVELOPER overlap: {ad}"

    # Every key has metadata
    for key in all_knobs:
        assert key in KNOB_META, f"Missing KNOB_META for: {key}"
        meta = KNOB_META[key]
        assert "label" in meta, f"Missing 'label' in KNOB_META[{key!r}]"
        assert "help" in meta, f"Missing 'help' in KNOB_META[{key!r}]"


# ======================================================================
# Filtering
# ======================================================================

def filter_config_overrides(
    overrides: dict,
    allow_developer: bool = False,
) -> dict:
    """Filter config overrides to only allowlisted + valid Config keys.

    Args:
        overrides: Dict of config key -> value from GUI.
        allow_developer: If True, DEVELOPER keys are allowed.

    Returns:
        Shallow copy containing only permitted keys.

    Raises:
        ValueError: If overrides contain unknown Config keys or
            developer-only keys when allow_developer is False.
    """
    # Determine which keys are allowed
    allowed = GUI_KNOBS_NORMAL | GUI_KNOBS_ADVANCED
    if allow_developer:
        allowed = allowed | GUI_KNOBS_DEVELOPER

    # Check for keys not in Config at all
    unknown = set()
    for key in overrides:
        if not is_config_key(key):
            unknown.add(key)
    if unknown:
        raise ValueError(
            f"Unknown config keys (not in Config schema): {sorted(unknown)}"
        )

    # Check for developer keys when not allowed
    if not allow_developer:
        dev_blocked = set(overrides.keys()) & GUI_KNOBS_DEVELOPER
        if dev_blocked:
            raise ValueError(
                f"Developer-only config keys not allowed: {sorted(dev_blocked)}"
            )

    # Check for keys that are valid Config but not in any allowlist
    all_knobs = GUI_KNOBS_NORMAL | GUI_KNOBS_ADVANCED | GUI_KNOBS_DEVELOPER
    unregistered = set(overrides.keys()) - all_knobs
    # Allow unregistered valid Config keys to pass through (they are valid
    # per schema, just not in the GUI registry yet). This is intentional:
    # the base YAML may contain keys the GUI doesn't manage.
    # Actually, for config_overrides (GUI-originated), we should be strict:
    valid_config_but_not_allowed = set()
    for key in overrides:
        if key not in unknown and key not in allowed:
            valid_config_but_not_allowed.add(key)
    if valid_config_but_not_allowed:
        raise ValueError(
            f"Config keys not in GUI allowlist: {sorted(valid_config_but_not_allowed)}"
        )

    return dict(overrides)


# ======================================================================
# Meta-Validation Hook (Import-Time)
# ======================================================================
try:
    from photometry_pipeline.config import Config
    import dataclasses
    required_step6 = ["lowpass_hz", "baseline_method", "baseline_percentile", "f0_min_value"]
    required_step7 = [
        "event_signal", "peak_threshold_method", "peak_threshold_k",
        "peak_threshold_percentile", "peak_threshold_abs",
        "peak_min_distance_sec", "peak_min_prominence_k",
        "peak_min_width_sec", "event_auc_baseline"
    ]
    cfg_fields = {f.name for f in dataclasses.fields(Config)}
    
    missing_step6 = [k for k in required_step6 if k not in cfg_fields]
    if missing_step6:
        raise RuntimeError(f"Config missing expected Step 6 key: {missing_step6[0]}")
        
    missing_step7 = [k for k in required_step7 if k not in cfg_fields]
    if missing_step7:
        raise RuntimeError(f"Config missing expected Step 7 key: {missing_step7[0]}")
except RuntimeError:
    raise
except Exception as e:
    raise RuntimeError(f"Step 6/7 schema validation failure: {type(e).__name__}: {e}") from e
