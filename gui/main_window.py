"""
MainWindow for the Long-Term Photometry Analysis GUI.

UI shell:
  - Top status strip (status, phase, elapsed, progress)
  - Left pane (run configuration, plotting, advanced, live log)
  - Right pane (results workspace)

State model:
  - IDLE / VALIDATING / RUNNING / terminal states
  - Successful full runs can enter complete-state workspace mode
  - "New Run" returns to editable idle controls while keeping prior values
"""

import json
import hashlib
import sys
import os
import csv
import re
import math
import secrets
import subprocess as _subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from statistics import median

from PySide6.QtCore import Qt, QSettings, QTimer, QSize, QEventLoop, QByteArray, QBuffer, QIODevice, Signal
from PySide6.QtGui import QColor, QFont, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QPlainTextEdit, QScrollArea,
    QFileDialog, QMessageBox, QSizePolicy, QListWidget, QListWidgetItem, QToolButton, QStackedWidget,
    QProgressBar, QLayout, QSplitter, QGridLayout, QStyle, QToolTip,
)

from gui.process_runner import PipelineRunner, RunnerState
from gui.interactive_image import InteractiveImageLabel, InteractiveImageController
from gui.run_spec import RunSpec, FORMAT_CHOICES
from gui.status_follower import StatusFollower
from gui.log_follower import LogFollower
from gui.run_report_viewer import RunReportViewer
from gui.run_report_parser import is_successful_completed_run_dir
from gui.validate_run_policy import (
    compute_run_signature,
    is_validation_current
)
from photometry_pipeline.config import Config
from photometry_pipeline.core.tonic_output import (
    TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
    TONIC_OUTPUT_MODE_PRESERVE_RAW,
)
from photometry_pipeline.core.tonic_timeline import (
    TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
    TONIC_TIMELINE_MODE_REAL_ELAPSED,
)
from photometry_pipeline.io.hdf5_cache_reader import (
    open_phasic_cache,
)
from photometry_pipeline.tuning.cache_downstream_retune import run_cache_downstream_retune
from photometry_pipeline.tuning.cache_correction_retune import run_cache_correction_retune
import dataclasses
from typing import get_args


_SETTINGS_GROUP = "run_config"
_ANCHOR_CLOCK_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")


def _generate_run_id():
    """Generate a run_id: run_YYYYMMDD_HHMMSS_<8hex>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{secrets.token_hex(4)}"


def _env_flag_enabled(name: str) -> bool:
    """Return True if an environment flag is set to a truthy value."""
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pixmap_sha256_png(pix: QPixmap) -> str:
    if pix.isNull():
        return ""
    payload = QByteArray()
    buffer = QBuffer(payload)
    if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
        return ""
    try:
        if not pix.save(buffer, "PNG"):
            return ""
    finally:
        buffer.close()
    return _sha256_bytes(bytes(payload))


def _open_folder(path: str) -> None:
    """Cross-platform open a folder in the file manager."""
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        _subprocess.run(["open", path], check=False)
    else:
        _subprocess.run(["xdg-open", path], check=False)


def _open_file(path: str) -> None:
    """Cross-platform open a file with the default associated app."""
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        _subprocess.run(["open", path], check=False)
    else:
        _subprocess.run(["xdg-open", path], check=False)


def parse_and_validate_isosbestic_knobs(
    window_sec_str: str,
    step_sec_str: str,
    min_valid_windows_val: int,
    r_low_str: str,
    r_high_str: str,
    g_min_str: str,
    min_samples_val: int,
    defaults: dict,
) -> tuple[dict | None, str | None]:
    """
    Parses and validates isosbestic correction knobs for the active rolling-fit engine.
    Returns (dict_of_overrides, None) if valid.
    Returns (None, error_message) if invalid.

    Legacy dynamic-fit knobs are retained in emitted overrides for compatibility
    but are sourced from defaults and treated as inactive in the active engine.
    """
    ws_text = window_sec_str.strip()
    try:
        ws = float(ws_text if ws_text else defaults["window_sec"])
        if ws <= 0:
            return None, "Regression Window must be > 0."
    except ValueError:
        return None, "Regression Window must be a number."
    if min_samples_val < 1:
        return None, "Min Samples per Window must be >= 1."

    # Legacy/inactive under rolling fit; preserve defaults for backward compatibility.
    try:
        ss = float(defaults["step_sec"])
        r_low = float(defaults["r_low"])
        r_high = float(defaults["r_high"])
        g_min = float(defaults["g_min"])
        min_valid_windows = int(defaults["min_valid_windows"])
    except (KeyError, TypeError, ValueError):
        cfg0 = Config()
        ss = float(cfg0.step_sec)
        r_low = float(cfg0.r_low)
        r_high = float(cfg0.r_high)
        g_min = float(cfg0.g_min)
        min_valid_windows = int(cfg0.min_valid_windows)

    overrides = {
        "window_sec": ws,
        "step_sec": ss,
        "min_valid_windows": min_valid_windows,
        "r_low": r_low,
        "r_high": r_high,
        "g_min": g_min,
        "min_samples_per_window": min_samples_val,
    }
    return overrides, None


def parse_and_validate_robust_event_reject_knobs(
    max_iters_val: int,
    residual_z_thresh_val: float,
    local_var_window_sec_val: float,
    local_var_ratio_thresh_enabled: bool,
    local_var_ratio_thresh_val: float,
    min_keep_fraction_val: float,
    defaults: dict,
) -> tuple[dict | None, str | None]:
    """
    Parse and validate robust_global_event_reject controls.
    """
    max_iters = int(max_iters_val)
    if max_iters < 1:
        return None, "Max iterations must be >= 1."

    residual_z = float(residual_z_thresh_val)
    if residual_z <= 0.0:
        return None, "Residual z-threshold must be > 0."

    local_window = float(local_var_window_sec_val)
    if local_window <= 0.0:
        return None, "Local variance window must be > 0."

    min_keep_fraction = float(min_keep_fraction_val)
    if not (0.0 < min_keep_fraction <= 1.0):
        return None, "Minimum keep fraction must be in (0, 1]."

    ratio_thresh = None
    if bool(local_var_ratio_thresh_enabled):
        ratio_thresh = float(local_var_ratio_thresh_val)
        if ratio_thresh <= 0.0:
            return None, "Local variance ratio threshold must be > 0 when enabled."

    overrides = {
        "robust_event_reject_max_iters": max_iters,
        "robust_event_reject_residual_z_thresh": residual_z,
        "robust_event_reject_local_var_window_sec": local_window,
        "robust_event_reject_local_var_ratio_thresh": ratio_thresh,
        "robust_event_reject_min_keep_fraction": min_keep_fraction,
    }
    return overrides, None


def parse_and_validate_adaptive_event_gated_knobs(
    residual_z_thresh_val: float,
    local_var_window_sec_val: float,
    local_var_ratio_thresh_enabled: bool,
    local_var_ratio_thresh_val: float,
    smooth_window_sec_val: float,
    min_trust_fraction_val: float,
    freeze_interp_method_text: str,
    defaults: dict,
) -> tuple[dict | None, str | None]:
    """
    Parse and validate adaptive_event_gated_regression controls.
    """
    residual_z = float(residual_z_thresh_val)
    if residual_z <= 0.0:
        return None, "Residual z-threshold must be > 0."

    local_window = float(local_var_window_sec_val)
    if local_window <= 0.0:
        return None, "Local variance window must be > 0."

    smooth_window = float(smooth_window_sec_val)
    if smooth_window <= 0.0:
        return None, "Smooth window must be > 0."

    min_trust = float(min_trust_fraction_val)
    if not (0.0 < min_trust <= 1.0):
        return None, "Minimum trust fraction must be in (0, 1]."

    freeze_interp_method = str(freeze_interp_method_text or "").strip() or "linear_hold"
    if freeze_interp_method not in {"linear_hold"}:
        return None, "Freeze interpolation method must be 'linear_hold'."

    ratio_thresh = None
    if bool(local_var_ratio_thresh_enabled):
        ratio_thresh = float(local_var_ratio_thresh_val)
        if ratio_thresh <= 0.0:
            return None, "Local variance ratio threshold must be > 0 when enabled."

    overrides = {
        "adaptive_event_gate_residual_z_thresh": residual_z,
        "adaptive_event_gate_local_var_window_sec": local_window,
        "adaptive_event_gate_local_var_ratio_thresh": ratio_thresh,
        "adaptive_event_gate_smooth_window_sec": smooth_window,
        "adaptive_event_gate_min_trust_fraction": min_trust,
        "adaptive_event_gate_freeze_interp_method": freeze_interp_method,
    }
    return overrides, None


def is_isosbestic_active(mode_text: str) -> bool:
    """Return True if the mode implies phasic analysis will run."""
    return mode_text in ("both", "phasic")


def get_isosbestic_overrides_if_active(mode_text: str, parsed_overrides: dict) -> dict:
    """Return the overrides only if the mode implies phasic analysis will run."""
    if is_isosbestic_active(mode_text):
        return parsed_overrides
    return {}


def compute_isosbestic_overrides_user_changed(parsed: dict, defaults: dict) -> dict:
    """
    Returns only the key/value pairs in parsed that differ from the defaults.
    """
    changed = {}
    for k, v in parsed.items():
        if k in defaults and v != defaults[k]:
            changed[k] = v
    return changed

_cached_allowed_fields = {}
_KNOWN_ALLOWED_VALUES = {
    "baseline_method": ["uv_raw_percentile_session", "uv_globalfit_percentile_session"],
    "event_signal": ["dff", "delta_f"],
    "peak_threshold_method": ["mean_std", "percentile", "median_mad", "absolute"],
    "event_auc_baseline": ["zero", "median"],
    "peak_pre_filter": ["none", "lowpass"],
    "dynamic_fit_mode": [
        "rolling_filtered_to_raw",
        "rolling_filtered_to_filtered",
        "global_linear_regression",
        "robust_global_event_reject",
        "adaptive_event_gated_regression",
        "rolling_local_regression",
    ],
    "tonic_output_mode": [
        "preserve_raw_session_shape",
        "flatten_session_bleach_preserve_session_baseline",
    ],
    "tonic_timeline_mode": [
        "real_elapsed_time",
        "gap_free_elapsed_time",
        "compressed_recording_time",
    ],
}
_RETUNE_PEAK_PRE_FILTER_OPTIONS = ["none", "smooth"]
_DYNAMIC_FIT_MODE_ALIAS_TO_CANONICAL = {
    "rolling_local_regression": "rolling_filtered_to_raw",
}
_DYNAMIC_FIT_MODE_CANONICAL_ORDER = [
    "rolling_filtered_to_raw",
    "rolling_filtered_to_filtered",
    "global_linear_regression",
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
]
_DYNAMIC_FIT_MODE_LABELS = {
    "rolling_filtered_to_raw": "Rolling regression (filtered→raw)",
    "rolling_filtered_to_filtered": "Rolling regression (filtered→filtered)",
    "global_linear_regression": "Global linear regression",
    "robust_global_event_reject": "Robust global fit + event rejection",
    "adaptive_event_gated_regression": "Adaptive event-gated regression",
}
_TONIC_OUTPUT_MODE_LABELS = {
    TONIC_OUTPUT_MODE_PRESERVE_RAW: "Preserve raw session shape",
    TONIC_OUTPUT_MODE_FLATTEN_BLEACH: "Flatten session bleach, preserve session baseline",
}
_TONIC_TIMELINE_MODE_LABELS = {
    TONIC_TIMELINE_MODE_REAL_ELAPSED: "Real elapsed time",
    TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED: "Gap-free elapsed time",
}

_DYNAMIC_FIT_TOOLTIPS = {
    "dynamic_fit_mode": (
        "Chooses how the isosbestic reference is fit. Rolling modes adapt over time, "
        "global linear fits one chunk-wide line, and robust/adaptive modes resist large "
        "event-driven distortions. Switch modes when correction is under- or over-responsive."
    ),
    "baseline_subtract_before_fit": (
        "Rolling modes only. Subtracts a slow baseline from fit inputs before estimating "
        "coefficients. Turn on if slow drift biases rolling fits; leave off for standard behavior."
    ),
    "regression_window_sec": (
        "Rolling modes only. Local regression window length in seconds. Larger values make the fit "
        "smoother and less reactive; smaller values track local changes more aggressively. Increase "
        "if the fit still rides large events."
    ),
    "min_samples_per_window": (
        "Rolling modes only. Minimum samples required in each local regression window. Higher values "
        "stabilize estimates but can disable adaptation in sparse windows; lower values allow more "
        "local fits but can be noisier."
    ),
    "robust_max_iters": (
        "Robust event-reject mode only. Number of reject-and-refit passes. Higher values can remove "
        "more event contamination but run longer; lower values are faster and more conservative."
    ),
    "robust_residual_z_thresh": (
        "Threshold for treating large positive residuals as likely biology instead of artifact. "
        "Lower values reject more points; higher values reject fewer. Reduce if the fit still "
        "rides large events."
    ),
    "robust_local_var_window_sec": (
        "Window used to estimate local variance for the optional variance-ratio rule. Larger values "
        "smooth variance estimates; smaller values react to shorter bursts. Increase to avoid jittery gating."
    ),
    "robust_local_var_ratio_thresh": (
        "Optional extra rejection rule based on local variance(signal)/variance(iso). Lower values "
        "reject more aggressively; higher values reject less. Enable if residual gating alone still "
        "lets event-heavy regions steer the fit."
    ),
    "robust_min_keep_fraction": (
        "Guardrail on retained samples after rejection. Higher values are safer and trigger fallback sooner; "
        "lower values allow more aggressive rejection. Raise if too much data is being dropped."
    ),
    "adaptive_residual_z_thresh": (
        "Threshold for marking event-like positive residuals as untrusted for adaptation. Lower values gate "
        "more regions; higher values gate fewer. Reduce if adaptation still follows event peaks."
    ),
    "adaptive_local_var_window_sec": (
        "Window used to estimate local variance for optional trust gating. Larger values smooth trust scoring; "
        "smaller values respond to brief variance bursts."
    ),
    "adaptive_local_var_ratio_thresh": (
        "Optional trust gate based on local variance(signal)/variance(iso). Lower values mark more regions "
        "as gated/frozen; higher values gate fewer. Enable when residual-only gating misses eventful spans."
    ),
    "adaptive_smooth_window_sec": (
        "Long-timescale smoothing window for adaptive coefficients. Larger values make adaptation steadier and "
        "less responsive; smaller values allow more local adaptation. Increase if coefficients still move too much."
    ),
    "adaptive_min_trust_fraction": (
        "Guardrail on trusted-data coverage. Higher values require more trusted samples and fall back sooner; "
        "lower values allow adaptation with less trusted support."
    ),
    "adaptive_freeze_interp_method": (
        "How coefficients bridge gated spans. 'linear_hold' keeps coefficients stable through gated regions and "
        "linearly reconnects across trusted boundaries to avoid event-driven jumps."
    ),
}

_TONIC_OUTPUT_MODE_TOOLTIP = (
    "Controls tonic output rendering/export mode. 'Preserve raw session shape' keeps current behavior. "
    "'Flatten session bleach, preserve session baseline' removes within-session bleach trend shape while "
    "retaining each session's baseline placement for cross-session comparison."
)
_TONIC_TIMELINE_MODE_TOOLTIP = (
    "Controls tonic timeline representation. 'Real elapsed time' preserves OFF gaps between sessions. "
    "'Gap-free elapsed time' removes OFF-gap whitespace while preserving full elapsed-duration span and "
    "session order."
)


def _get_allowed_from_config_field(field_name: str) -> list[str]:
    """
    Derives allowed strings from the Config schema dynamically for a given field.
    Falls back to a single default list if introspection fails.
    """
    global _cached_allowed_fields
    if field_name in _cached_allowed_fields:
        return _cached_allowed_fields[field_name]

    try:
        cfg_fields = {f.name: f for f in dataclasses.fields(Config)}
        field_type = cfg_fields[field_name].type
        
        # Literal extraction
        args = get_args(field_type)
        if args:
            methods = [a for a in args if isinstance(a, str)]
            if methods:
                _cached_allowed_fields[field_name] = sorted(set(methods))
                return _cached_allowed_fields[field_name]

        # Enum extraction
        import enum
        if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
            methods = [e.value for e in field_type if isinstance(e.value, str)]
            if methods:
                _cached_allowed_fields[field_name] = sorted(set(methods))
                return _cached_allowed_fields[field_name]
            
    except Exception:
        pass

    if field_name in _KNOWN_ALLOWED_VALUES:
        _cached_allowed_fields[field_name] = list(_KNOWN_ALLOWED_VALUES[field_name])
        return _cached_allowed_fields[field_name]
        
    default_val = getattr(Config(), field_name)
    _cached_allowed_fields[field_name] = [str(default_val)]
    return _cached_allowed_fields[field_name]

def get_allowed_baseline_methods_from_config() -> list[str]:
    return _get_allowed_from_config_field("baseline_method")

def get_allowed_event_signals_from_config() -> list[str]:
    return _get_allowed_from_config_field("event_signal")

def get_allowed_peak_threshold_methods_from_config() -> list[str]:
    return _get_allowed_from_config_field("peak_threshold_method")

def get_allowed_event_auc_baselines_from_config() -> list[str]:
    return _get_allowed_from_config_field("event_auc_baseline")

def get_allowed_peak_pre_filters_from_config() -> list[str]:
    return _get_allowed_from_config_field("peak_pre_filter")


def get_allowed_dynamic_fit_modes_from_config() -> list[str]:
    raw_modes = _get_allowed_from_config_field("dynamic_fit_mode")
    normalized_from_config: list[str] = []
    for mode_raw in raw_modes:
        mode = normalize_dynamic_fit_mode(mode_raw)
        if mode and mode not in normalized_from_config:
            normalized_from_config.append(mode)

    ordered: list[str] = []
    for mode in _DYNAMIC_FIT_MODE_CANONICAL_ORDER:
        if mode not in ordered:
            ordered.append(mode)
    for mode in normalized_from_config:
        if mode not in ordered:
            ordered.append(mode)
    return ordered


def get_allowed_tonic_output_modes_from_config() -> list[str]:
    raw_modes = _get_allowed_from_config_field("tonic_output_mode")
    ordered: list[str] = [
        TONIC_OUTPUT_MODE_PRESERVE_RAW,
        TONIC_OUTPUT_MODE_FLATTEN_BLEACH,
    ]
    for mode in raw_modes:
        key = str(mode or "").strip()
        if key and key not in ordered:
            ordered.append(key)
    return ordered


def get_allowed_tonic_timeline_modes_from_config() -> list[str]:
    raw_modes = _get_allowed_from_config_field("tonic_timeline_mode")
    ordered: list[str] = [
        TONIC_TIMELINE_MODE_REAL_ELAPSED,
        TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED,
    ]
    for mode in raw_modes:
        key = normalize_tonic_timeline_mode(str(mode or "").strip())
        if key and key not in ordered:
            ordered.append(key)
    return ordered


def normalize_tonic_output_mode(mode_raw: str) -> str:
    mode = str(mode_raw or "").strip()
    if not mode:
        return TONIC_OUTPUT_MODE_PRESERVE_RAW
    if mode == "flatten_session_bleach":
        return TONIC_OUTPUT_MODE_FLATTEN_BLEACH
    if mode == "preserve_raw":
        return TONIC_OUTPUT_MODE_PRESERVE_RAW
    return mode


def tonic_output_mode_label(mode_raw: str) -> str:
    mode = normalize_tonic_output_mode(mode_raw)
    return _TONIC_OUTPUT_MODE_LABELS.get(mode, mode or TONIC_OUTPUT_MODE_PRESERVE_RAW)


def normalize_tonic_timeline_mode(mode_raw: str) -> str:
    mode = str(mode_raw or "").strip()
    if not mode:
        return TONIC_TIMELINE_MODE_REAL_ELAPSED
    if mode == "elapsed":
        return TONIC_TIMELINE_MODE_REAL_ELAPSED
    if mode in {"compressed", "compressed_recording_time", "gap_free", "gap_free_elapsed_time"}:
        return TONIC_TIMELINE_MODE_GAP_FREE_ELAPSED
    return mode


def tonic_timeline_mode_label(mode_raw: str) -> str:
    mode = normalize_tonic_timeline_mode(mode_raw)
    return _TONIC_TIMELINE_MODE_LABELS.get(mode, mode or TONIC_TIMELINE_MODE_REAL_ELAPSED)


def normalize_dynamic_fit_mode(mode_raw: str) -> str:
    mode = str(mode_raw or "").strip()
    if not mode:
        return "rolling_filtered_to_raw"
    return _DYNAMIC_FIT_MODE_ALIAS_TO_CANONICAL.get(mode, mode)


def is_rolling_dynamic_fit_mode(mode_raw: str) -> bool:
    mode = normalize_dynamic_fit_mode(mode_raw)
    return mode in {"rolling_filtered_to_raw", "rolling_filtered_to_filtered"}


def is_robust_event_reject_mode(mode_raw: str) -> bool:
    mode = normalize_dynamic_fit_mode(mode_raw)
    return mode == "robust_global_event_reject"


def is_adaptive_event_gated_mode(mode_raw: str) -> bool:
    mode = normalize_dynamic_fit_mode(mode_raw)
    return mode == "adaptive_event_gated_regression"


def dynamic_fit_mode_label(mode_raw: str) -> str:
    mode = normalize_dynamic_fit_mode(mode_raw)
    return _DYNAMIC_FIT_MODE_LABELS.get(mode, mode or "rolling_filtered_to_raw")


def format_robust_event_reject_settings_summary(
    max_iters: int,
    residual_z_thresh: float,
    local_var_window_sec: float,
    local_var_ratio_thresh: float | None,
    min_keep_fraction: float,
) -> str:
    ratio_text = (
        "disabled"
        if local_var_ratio_thresh is None
        else f"{float(local_var_ratio_thresh):.4g}"
    )
    return (
        "Robust event-reject settings: "
        f"max_iters={int(max_iters)}, "
        f"residual_z={float(residual_z_thresh):.4g}, "
        f"local_var_window_s={float(local_var_window_sec):.4g}, "
        f"local_var_ratio={ratio_text}, "
        f"min_keep={float(min_keep_fraction):.4g}"
    )


def format_adaptive_event_gated_settings_summary(
    residual_z_thresh: float,
    local_var_window_sec: float,
    local_var_ratio_thresh: float | None,
    smooth_window_sec: float,
    min_trust_fraction: float,
    freeze_interp_method: str,
) -> str:
    ratio_text = (
        "disabled"
        if local_var_ratio_thresh is None
        else f"{float(local_var_ratio_thresh):.4g}"
    )
    return (
        "Adaptive event-gated settings: "
        f"residual_z={float(residual_z_thresh):.4g}, "
        f"local_var_window_s={float(local_var_window_sec):.4g}, "
        f"local_var_ratio={ratio_text}, "
        f"smooth_window_s={float(smooth_window_sec):.4g}, "
        f"min_trust={float(min_trust_fraction):.4g}, "
        f"freeze_interp={str(freeze_interp_method or 'linear_hold')}"
    )


def get_retune_peak_pre_filters() -> list[str]:
    return list(_RETUNE_PEAK_PRE_FILTER_OPTIONS)


def normalize_retune_peak_pre_filter(mode_raw: str) -> str:
    mode = str(mode_raw or "none").strip().lower()
    if mode == "none":
        return "none"
    if mode in {"smooth", "lowpass"}:
        return "smooth"
    return "none"


def map_retune_peak_pre_filter_to_run_setting(mode_raw: str) -> str:
    # Main run settings keep legacy config semantics for full-pipeline compatibility.
    mode = normalize_retune_peak_pre_filter(mode_raw)
    return "lowpass" if mode == "smooth" else "none"

_cached_percentile_reqs = {}

def baseline_method_requires_percentile(method_str: str) -> bool:
    """
    Returns True if the given baseline method requires a percentile parameter.
    Determined deterministically from the Config schema behavior, without heuristics.
    """
    global _cached_percentile_reqs
    if method_str in _cached_percentile_reqs:
        return _cached_percentile_reqs[method_str]

    allowed = get_allowed_baseline_methods_from_config()
    if method_str not in allowed:
        _cached_percentile_reqs[method_str] = False
        return False

    try:
        # Construct two configs differing only in baseline_percentile
        cfg1 = Config(baseline_method=method_str, baseline_percentile=10.0)
        cfg2 = Config(baseline_method=method_str, baseline_percentile=50.0)
        
        # If changing the input percentile changes the effective percentile, it is required.
        req = (cfg1.baseline_percentile != cfg2.baseline_percentile)
        _cached_percentile_reqs[method_str] = req
        return req
    except Exception:
        # Fallback if instantiation fails for some reason
        _cached_percentile_reqs[method_str] = False
        return False

def parse_and_validate_preproc_baseline_knobs(
    lowpass_hz_str: str,
    baseline_method_text: str,
    baseline_percentile_str: str,
    f0_min_value_str: str,
    defaults: dict,
) -> tuple[dict | None, str | None]:
    """
    Parses and validates the preprocessing and baseline advanced knobs.
    Returns (dict_of_overrides, None) if valid.
    Returns (None, error_message) if invalid.
    """
    try:
        lp_val = lowpass_hz_str.strip()
        lowpass_hz = float(lp_val if lp_val else defaults["lowpass_hz"])
        if lowpass_hz <= 0:
            return None, "Lowpass Filter (Hz) must be > 0."
    except ValueError:
        return None, "Lowpass Filter (Hz) must be a number."

    method = baseline_method_text.strip()
    allowed_methods = get_allowed_baseline_methods_from_config()
    if method not in allowed_methods:
        return None, "Invalid Baseline Method."

    try:
        f0_val = f0_min_value_str.strip()
        f0_min = float(f0_val if f0_val else defaults["f0_min_value"])
        if f0_min < 0:
            return None, "F0 Min Value must be >= 0."
    except ValueError:
        return None, "F0 Min Value must be a number."

    overrides = {
        "lowpass_hz": lowpass_hz,
        "baseline_method": method,
        "f0_min_value": f0_min,
    }

    if baseline_method_requires_percentile(method):
        try:
            pct_val = baseline_percentile_str.strip()
            # Careful not to accidentally validate if we shouldn't have been parsing
            pct = float(pct_val if pct_val else defaults["baseline_percentile"])
            if not (0 <= pct <= 100):
                return None, "Baseline Percentile must be between 0 and 100."
            overrides["baseline_percentile"] = pct
        except ValueError:
            return None, "Baseline Percentile must be a number."

    return overrides, None

def peak_threshold_method_requires_k(method_str: str) -> bool:
    """K-threshold applies to std/mad-style methods only."""
    return method_str in {"mean_std", "median_mad"}

def peak_threshold_method_requires_percentile(method_str: str) -> bool:
    """Percentile threshold applies only to percentile method."""
    return method_str == "percentile"

def peak_threshold_method_requires_abs(method_str: str) -> bool:
    """Absolute threshold applies only to absolute method."""
    return method_str == "absolute"


def validate_representative_index_preview_compatibility(
    representative_session_index: int | None,
    preview_first_n: int | None,
) -> str | None:
    """Validate representative-session selection against preview truncation."""
    if representative_session_index is None or preview_first_n is None:
        return None
    if representative_session_index >= preview_first_n:
        return (
            f"Representative Session index {representative_session_index} is out of range for "
            f"Preview first N={preview_first_n}. Choose (auto) or a lower session index."
        )
    return None

def parse_and_validate_event_feature_knobs(
    event_signal_text: str,
    peak_method_text: str,
    peak_k_str: str,
    peak_pct_str: str,
    peak_abs_str: str,
    peak_dist_str: str,
    event_auc_text: str,
    defaults: dict,
    peak_pre_filter_text: str | None = None,
    peak_prominence_k_str: str | None = None,
    peak_width_sec_str: str | None = None,
) -> tuple[dict | None, str | None]:
    """
    Parses and validates the event + feature advanced knobs.
    Returns (dict_of_overrides, None) if valid.
    Returns (None, error_message) if invalid.
    """
    event_sig_method = event_signal_text.strip()
    allowed_event_signals = get_allowed_event_signals_from_config()
    if event_sig_method not in allowed_event_signals:
        return None, "Invalid Event Signal."

    peak_method = peak_method_text.strip()
    allowed_peak_methods = get_allowed_peak_threshold_methods_from_config()
    if peak_method not in allowed_peak_methods:
        return None, "Invalid Peak Threshold Method."
        
    auc_baseline = event_auc_text.strip()
    allowed_auc_baselines = get_allowed_event_auc_baselines_from_config()
    if auc_baseline not in allowed_auc_baselines:
        return None, "Invalid Event AUC Baseline."

    if peak_pre_filter_text is None:
        peak_pre_filter = str(defaults.get("peak_pre_filter", "none"))
    else:
        peak_pre_filter = peak_pre_filter_text.strip()
    allowed_peak_pre_filters = get_allowed_peak_pre_filters_from_config()
    if peak_pre_filter not in allowed_peak_pre_filters:
        return None, "Invalid Peak Pre-Filter."

    try:
        dist_val = peak_dist_str.strip()
        peak_dist = float(dist_val if dist_val else defaults["peak_min_distance_sec"])
        if peak_dist < 0:
            return None, "Peak Min Distance (sec) must be >= 0."
    except ValueError:
        return None, "Peak Min Distance (sec) must be a number."

    try:
        prom_val = peak_prominence_k_str.strip() if peak_prominence_k_str is not None else ""
        peak_prominence_k = float(prom_val if prom_val else defaults.get("peak_min_prominence_k", 0.0))
        if peak_prominence_k < 0:
            return None, "Peak Min Prominence K must be >= 0."
    except ValueError:
        return None, "Peak Min Prominence K must be a number."

    try:
        width_val = peak_width_sec_str.strip() if peak_width_sec_str is not None else ""
        peak_width_sec = float(width_val if width_val else defaults.get("peak_min_width_sec", 0.0))
        if peak_width_sec < 0:
            return None, "Peak Min Width (s) must be >= 0."
    except ValueError:
        return None, "Peak Min Width (s) must be a number."

    overrides = {
        "event_signal": event_sig_method,
        "peak_threshold_method": peak_method,
        "peak_min_distance_sec": peak_dist,
        "peak_min_prominence_k": peak_prominence_k,
        "peak_min_width_sec": peak_width_sec,
        "peak_pre_filter": peak_pre_filter,
        "event_auc_baseline": auc_baseline,
    }

    if peak_threshold_method_requires_k(peak_method):
        try:
            k_val = peak_k_str.strip()
            k = float(k_val if k_val else defaults["peak_threshold_k"])
            if k <= 0:
                return None, "Peak Threshold K must be > 0."
            overrides["peak_threshold_k"] = k
        except ValueError:
            return None, "Peak Threshold K must be a number."
            
    if peak_threshold_method_requires_percentile(peak_method):
        try:
            pct_val = peak_pct_str.strip()
            pct = float(pct_val if pct_val else defaults["peak_threshold_percentile"])
            if not (0 <= pct <= 100):
                return None, "Peak Threshold Percentile must be between 0 and 100."
            overrides["peak_threshold_percentile"] = pct
        except ValueError:
            return None, "Peak Threshold Percentile must be a number."
            
    if peak_threshold_method_requires_abs(peak_method):
        try:
            abs_val = peak_abs_str.strip()
            abs_v = float(abs_val if abs_val else defaults["peak_threshold_abs"])
            if abs_v <= 0:
                return None, "Peak Threshold Absolute must be > 0."
            overrides["peak_threshold_abs"] = abs_v
        except ValueError:
            return None, "Peak Threshold Absolute must be a number."

    return overrides, None



def compute_overrides_user_changed(parsed: dict, defaults: dict) -> dict:
    """
    Generic helper: returns only key/value pairs in parsed that differ from defaults.
    """
    changed = {}
    for k, v in parsed.items():
        if k in defaults and v != defaults[k]:
            changed[k] = v
    return changed


_ClickableImageLabel = InteractiveImageLabel


class MainWindow(QMainWindow):
    """Long-Term Photometry Analysis GUI."""

    WINDOW_TITLE_BASE = "Long-Term Photometry Analysis"

    def __init__(self, parent=None, settings: QSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.WINDOW_TITLE_BASE)
        self.resize(1100, 850)

        # Settings (injectable for testing)
        self._settings = settings if settings is not None else QSettings()

        from photometry_pipeline.config import Config
        self._default_cfg = Config()
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._lab_default_config_path = os.path.normpath(
            os.path.join(repo_root, "config", "qc_universal_config.yaml")
        )

        # Validate->Run reuse tracking (Fix B1)
        self._validated_run_dir = None
        self._validated_gui_run_spec_json_path = None
        self._validated_config_effective_yaml_path = None
        self._validated_run_signature = None
        self._status_follower = None
        self._log_follower = None
        self._runner = PipelineRunner()
        self._runner.started.connect(self._on_run_started)
        self._runner.error.connect(self._on_run_error)
        self._runner.state_changed.connect(self._on_state_changed)
        self._runner.finished.connect(self._on_run_finished_failsafe)

        # Current run directory (set before each run)
        self._current_run_dir = ""
        self._is_validate_only = False
        self._validation_passed = False
        self._did_finalize_run_ui = False

        # Accumulated stdout for validate-only result checking
        self._validate_stdout = []

        # Discovery cache (result of RunSpec.run_discovery)
        self._discovery_cache = None

        # Status label fields
        self._state_str = "IDLE"
        self._ui_state = RunnerState.IDLE
        self._last_status_phase = "\u2014"
        self._last_status_state = "\u2014"
        self._last_status_duration = ""
        self._last_status_duration_sec = None
        self._last_status_errors = []
        self._last_status_msg = ""
        self._last_status_pct = None
        self._ui_progress_pct = 0
        self._run_started_monotonic = None
        self._last_elapsed_sec = 0.0
        self._saw_cancel_status = False
        self._is_complete_workspace_active = False
        self._shell_left_width_floor = 420
        self._shell_workspace_left_floor = 420
        self._shell_setup_left_min = 460
        self._shell_setup_left_ratio = 0.45
        self._shell_workspace_left_ratio = 0.32
        self._shell_setup_right_min = 380
        self._shell_workspace_right_min = 500
        self._tuning_workspace_available = False
        self._tuning_last_result = None
        self._tuning_last_changed_fields: list[str] = []
        self._tuning_applyback_applied = False
        self._tuning_applyback_timestamp = ""
        self._tuning_active_overlay_path = ""
        self._tuning_active_overlay_pixmap = QPixmap()
        self._tuning_last_loaded_overlay_sha256 = ""
        self._tuning_last_loaded_overlay_size = 0
        self._tuning_overlay_zoom_mode = False
        self._tuning_overlay_interaction: InteractiveImageController | None = None
        self._roi_chunk_ids_cache: dict[str, list[int]] = {}
        self._correction_tuning_workspace_available = False
        self._correction_tuning_last_result = None
        self._correction_tuning_applyback_applied = False
        self._correction_tuning_applyback_timestamp = ""
        self._correction_tuning_inspection_paths: list[str] = []
        self._correction_tuning_inspection_panel_labels: list[str] = []
        self._correction_tuning_inspection_index = 0
        self._correction_tuning_active_inspection_path = ""
        self._correction_tuning_active_inspection_pixmap = QPixmap()
        self._correction_tuning_zoom_mode = False
        self._correction_tuning_overlay_interaction: InteractiveImageController | None = None
        # Debug-only GUI preflight timing. Disabled by default.
        # Enable with PHOTOMETRY_GUI_TIMING=1.
        self._gui_timing_enabled = _env_flag_enabled("PHOTOMETRY_GUI_TIMING")
        self._retune_preview_debug_enabled = _env_flag_enabled("PHOTOMETRY_RETUNE_DEBUG")
        self._timing_action = ""
        self._timing_click_monotonic = None
        self._elapsed_first_tick_logged = False
        self._rwd_contract_cache = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(250)
        self._elapsed_timer.timeout.connect(self._on_elapsed_timer_tick)

        # Build UI
        central = QWidget()
        central.setObjectName("appShellRoot")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        main_layout.addWidget(self._build_status_strip(), 0)
        main_layout.addWidget(self._build_main_body(), 1)
        self._apply_shell_chrome_styles()
        self._apply_results_idle_placeholder()
        self._refresh_tuning_workspace_availability()
        self._update_button_states()
        self._refresh_splitter_workspace_policy()

        # Restore persisted settings
        self._load_settings_into_widgets()

    # ==================================================================
    # Config Panel
    # ==================================================================

    def _build_status_strip(self) -> QWidget:
        """Top status strip with status/phase labels and progress."""
        strip = QWidget()
        outer = QVBoxLayout(strip)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._status_header_card = QWidget()
        self._status_header_card.setObjectName("statusHeaderCard")
        card = QVBoxLayout(self._status_header_card)
        card.setContentsMargins(12, 10, 12, 10)
        card.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(10)

        self._status_label = QLabel("Run status: IDLE")
        self._status_label.setStyleSheet("font-weight: bold;")
        self._status_label.setToolTip("Overall run state.")
        top_row.addWidget(self._status_label, 0)

        self._phase_label = QLabel("Run phase: \u2014")
        self._phase_label.setToolTip("Current pipeline phase.")
        top_row.addWidget(self._phase_label, 0)

        self._elapsed_label = QLabel("Elapsed: \u2014")
        self._elapsed_label.setToolTip("Elapsed wall-clock time for active validation/run.")
        top_row.addWidget(self._elapsed_label, 0)

        top_row.addStretch(1)

        self._preview_badge = QLabel("PREVIEW")
        self._preview_badge.setStyleSheet(
            "font-weight: bold; color: white; background: #d9534f; "
            "padding: 2px 8px; border-radius: 4px;"
        )
        self._preview_badge.hide()
        top_row.addWidget(self._preview_badge, 0)
        card.addLayout(top_row)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(8)
        self._progress_caption_label = QLabel("Progress")
        self._progress_caption_label.setStyleSheet("font-size: 11px; color: #555;")
        progress_row.addWidget(self._progress_caption_label, 0)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setMinimumHeight(24)
        self._progress_bar.setMaximumHeight(24)
        self._progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._progress_bar.setStyleSheet(
            "QProgressBar {"
            " min-height: 24px;"
            " max-height: 24px;"
            " border: 1px solid #7f7f7f;"
            " border-radius: 3px;"
            " text-align: center;"
            " background-color: #f2f2f2;"
            "}"
            "QProgressBar::chunk {"
            " background-color: #2d89ef;"
            " margin: 0px;"
            "}"
        )
        self._progress_bar.setToolTip("Milestone-based run progress.")
        progress_row.addWidget(self._progress_bar, 1)
        card.addLayout(progress_row)
        outer.addWidget(self._status_header_card)
        return strip

    def _build_main_body(self) -> QWidget:
        """Fixed major panes: upper-left controls, lower-left log, right results."""
        body = QWidget()
        row = QVBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)
        splitter.setFocusPolicy(Qt.NoFocus)
        self._main_splitter = splitter

        left_pane = self._build_left_pane()
        self._left_pane = left_pane
        left_pane.setMinimumWidth(self._shell_left_width_floor)
        left_pane.setMaximumWidth(900)
        splitter.addWidget(left_pane)
        self._results_pane = self._build_results_pane()
        splitter.addWidget(self._results_pane)
        self._lock_main_splitter_handle()
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self._apply_splitter_setup_mode()
        row.addWidget(splitter, 1)
        return body

    def _lock_main_splitter_handle(self) -> None:
        """Keep shell widths mode-controlled; divider is not user-adjustable."""
        splitter = getattr(self, "_main_splitter", None)
        if splitter is None or splitter.count() < 2:
            return
        handle = splitter.handle(1)
        if handle is None:
            return
        handle.setEnabled(False)
        handle.setFocusPolicy(Qt.NoFocus)
        handle.setCursor(Qt.ArrowCursor)
        handle.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def _splitter_total_width(self) -> int:
        splitter = getattr(self, "_main_splitter", None)
        if splitter is None:
            return 0
        total = splitter.width()
        if total > 0:
            return total
        total = sum(splitter.sizes())
        if total > 0:
            return total
        return max(self.width() - 24, 1100)

    def _target_splitter_left_width(self, *, setup_mode: bool) -> int:
        left_pane = getattr(self, "_left_pane", None)
        if left_pane is None:
            return 0
        total = self._splitter_total_width()
        left_floor = left_pane.minimumWidth()
        if setup_mode:
            left = max(
                int(total * self._shell_setup_left_ratio),
                self._shell_setup_left_min,
                left_floor,
            )
            right_floor = self._shell_setup_right_min
        else:
            left = max(int(total * self._shell_workspace_left_ratio), left_floor)
            right_floor = self._shell_workspace_right_min
        left = min(left, left_pane.maximumWidth())
        max_left_for_right_floor = total - right_floor
        if max_left_for_right_floor >= left_floor:
            left = min(left, max_left_for_right_floor)
        return left

    def _set_shell_left_floor(self, *, setup_mode: bool) -> None:
        left_pane = getattr(self, "_left_pane", None)
        if left_pane is None:
            return
        floor = self._shell_left_width_floor if setup_mode else self._shell_workspace_left_floor
        if left_pane.minimumWidth() != floor:
            left_pane.setMinimumWidth(floor)

    def _apply_splitter_setup_mode(self) -> None:
        splitter = getattr(self, "_main_splitter", None)
        left_pane = getattr(self, "_left_pane", None)
        if splitter is None or left_pane is None:
            return
        self._set_shell_left_floor(setup_mode=True)
        total = self._splitter_total_width()
        left = self._target_splitter_left_width(setup_mode=True)
        right = max(1, total - left)
        splitter.setSizes([left, right])

    def _apply_splitter_workspace_mode(self) -> None:
        splitter = getattr(self, "_main_splitter", None)
        left_pane = getattr(self, "_left_pane", None)
        if splitter is None or left_pane is None:
            return
        self._set_shell_left_floor(setup_mode=False)
        total = self._splitter_total_width()
        left = self._target_splitter_left_width(setup_mode=False)
        right = max(1, total - left)
        splitter.setSizes([left, right])

    def _is_workspace_splitter_mode(self) -> bool:
        if self._is_complete_workspace_active:
            return True
        if self._validation_passed:
            return True
        if self._ui_state in (RunnerState.RUNNING, RunnerState.VALIDATING):
            return True
        if self._runner.is_running():
            return True
        return False

    def _apply_splitter_workspace_policy(self) -> None:
        if self._is_workspace_splitter_mode():
            self._apply_splitter_workspace_mode()
        else:
            self._apply_splitter_setup_mode()

    def _refresh_splitter_workspace_policy(self) -> None:
        self._apply_splitter_workspace_policy()
        if hasattr(self, "_main_splitter"):
            QTimer.singleShot(0, self._apply_splitter_workspace_policy)

    def _build_left_pane(self) -> QWidget:
        """Fixed left column with control area above and log pane below."""
        pane = QWidget()
        pane.setObjectName("workflowColumn")
        layout = QVBoxLayout(pane)
        self._left_pane_layout = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        workflow_title = QLabel("Workflow Setup")
        workflow_title.setObjectName("workflowColumnTitle")
        layout.addWidget(workflow_title)

        workflow_subtitle = QLabel(
            "Configure inputs, validate settings, and launch runs from top to bottom."
        )
        workflow_subtitle.setWordWrap(True)
        workflow_subtitle.setObjectName("workflowColumnSubtitle")
        layout.addWidget(workflow_subtitle)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QScrollArea.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._controls_scroll = controls_scroll
        self._controls_stack = QStackedWidget()
        self._controls_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._controls_stack.setMinimumWidth(0)
        self._config_panel = self._build_config_panel()
        self._complete_state_panel = self._build_complete_state_panel()
        self._complete_state_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._complete_state_panel.setMinimumWidth(0)
        self._controls_stack.addWidget(self._config_panel)
        self._controls_stack.addWidget(self._complete_state_panel)
        self._controls_stack.setCurrentWidget(self._config_panel)
        controls_scroll.setWidget(self._controls_stack)
        controls_scroll.setObjectName("workflowControlsScroll")

        log_group = self._build_log_panel()
        self._log_group = log_group

        layout.addWidget(controls_scroll, 1)
        layout.addWidget(log_group, 0)
        self._sync_live_log_disclosure_layout(
            expanded=bool(
                hasattr(self, "_live_log_disclosure_btn")
                and self._live_log_disclosure_btn.isChecked()
            )
        )
        return pane

    def _refresh_left_column_width_clamp(self) -> None:
        """Refresh geometry so controls track left-pane viewport width after disclosure toggles."""
        controls_scroll = getattr(self, "_controls_scroll", None)
        controls_stack = getattr(self, "_controls_stack", None)
        if controls_scroll is None or controls_stack is None:
            return
        controls_stack.updateGeometry()
        config_panel = getattr(self, "_config_panel", None)
        if config_panel is not None:
            config_panel.updateGeometry()
        controls_scroll.updateGeometry()
        controls_scroll.viewport().updateGeometry()

    def _build_results_pane(self) -> QGroupBox:
        """Right pane with the large results viewer."""
        results_group = QGroupBox("Results")
        results_group.setObjectName("resultsPaneShell")
        results_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_lay = QVBoxLayout(results_group)
        results_lay.setContentsMargins(8, 8, 8, 8)
        results_lay.setSpacing(8)
        self._results_layout = results_lay

        self._results_summary_group = QGroupBox("Run Summary")
        self._results_summary_group.setProperty("workflowSection", True)
        self._results_summary_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self._results_summary_group.setMaximumHeight(124)
        summary_layout = QVBoxLayout(self._results_summary_group)
        summary_layout.setContentsMargins(8, 3, 8, 4)
        summary_layout.setSpacing(2)

        self._results_summary_title_label = QLabel("No completed run loaded.")
        self._results_summary_title_label.setObjectName("resultsSummaryHeadline")
        self._results_summary_title_label.setWordWrap(False)
        summary_layout.addWidget(self._results_summary_title_label)

        self._results_summary_compact_label = QLabel("Idle")
        self._results_summary_compact_label.setObjectName("resultsSummaryHint")
        self._results_summary_compact_label.setWordWrap(False)
        self._results_summary_compact_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self._results_summary_compact_label.setVisible(False)
        summary_layout.addWidget(self._results_summary_compact_label)

        self._results_summary_details_widget = QWidget()
        summary_details_layout = QVBoxLayout(self._results_summary_details_widget)
        summary_details_layout.setContentsMargins(0, 0, 0, 0)
        summary_details_layout.setSpacing(1)

        summary_form = QFormLayout()
        summary_form.setContentsMargins(0, 0, 0, 0)
        summary_form.setHorizontalSpacing(8)
        summary_form.setVerticalSpacing(2)
        summary_form.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        summary_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        summary_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._results_summary_run_value = QLabel("(none)")
        self._results_summary_run_value.setWordWrap(False)
        self._results_summary_run_value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._results_summary_run_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_form.addRow("Run:", self._results_summary_run_value)

        self._results_summary_state_value = QLabel("Idle")
        self._results_summary_state_value.setWordWrap(False)
        self._results_summary_state_value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        summary_form.addRow("State:", self._results_summary_state_value)
        summary_details_layout.addLayout(summary_form)

        self._results_summary_hint_label = QLabel(
            "Run the pipeline or open completed results to populate this workspace."
        )
        self._results_summary_hint_label.setObjectName("resultsSummaryHint")
        self._results_summary_hint_label.setWordWrap(True)
        self._results_summary_hint_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        summary_details_layout.addWidget(self._results_summary_hint_label)
        summary_layout.addWidget(self._results_summary_details_widget)
        results_lay.addWidget(self._results_summary_group, 0)

        self._results_views_group = QGroupBox("Analysis Outputs")
        self._results_views_group.setProperty("workflowSection", True)
        self._results_views_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        views_layout = QVBoxLayout(self._results_views_group)
        views_layout.setContentsMargins(10, 8, 10, 10)
        views_layout.setSpacing(8)
        self._results_views_hint_label = QLabel(
            "Use the region selector and tabs below to inspect deliverables."
        )
        self._results_views_hint_label.setObjectName("resultsSummaryHint")
        self._results_views_hint_label.setWordWrap(True)
        views_layout.addWidget(self._results_views_hint_label)

        self._report_viewer = RunReportViewer()
        if hasattr(self._report_viewer, "_status_label"):
            self._report_viewer._status_label.setWordWrap(True)
        self._report_viewer.region_changed.connect(self._on_results_region_changed)
        views_layout.addWidget(self._report_viewer, 1)
        results_lay.addWidget(self._results_views_group, 1)

        self._tuning_group = self._build_tuning_workspace_group()
        self._tuning_group.setProperty("workflowSection", True)
        self._tuning_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        results_lay.addWidget(self._tuning_group)
        results_lay.setStretch(0, 0)
        results_lay.setStretch(1, 1)
        results_lay.setStretch(2, 0)
        self._refresh_results_workspace_summary()
        self._update_results_pane_mode_for_tuning()
        return results_group

    def _build_complete_state_panel(self) -> QWidget:
        """Compact completion-state summary card shown after successful full runs."""
        panel = QWidget()
        panel.setObjectName("completeModePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self._complete_mode_title_label = QLabel("Results Mode")
        self._complete_mode_title_label.setObjectName("workflowColumnTitle")
        layout.addWidget(self._complete_mode_title_label)

        self._complete_mode_subtitle_label = QLabel(
            "Completed outputs are loaded on the right. Post-run tuning is available below."
        )
        self._complete_mode_subtitle_label.setObjectName("workflowColumnSubtitle")
        self._complete_mode_subtitle_label.setWordWrap(True)
        layout.addWidget(self._complete_mode_subtitle_label)

        card = QGroupBox("Completed Run")
        card.setProperty("workflowSection", True)
        card.setObjectName("completeModeContextCard")
        card_layout = QVBoxLayout(card)
        self._complete_summary_label = QLabel(
            "No completed run selected. Run the pipeline or open completed results."
        )
        self._complete_summary_label.setWordWrap(True)
        self._complete_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card_layout.addWidget(self._complete_summary_label)
        layout.addWidget(card)

        self._complete_mode_next_steps_label = QLabel(
            "Next: inspect outputs, retune as needed, then apply back to next-run settings if needed."
        )
        self._complete_mode_next_steps_label.setWordWrap(True)
        self._complete_mode_next_steps_label.setObjectName("resultsSummaryHint")
        layout.addWidget(self._complete_mode_next_steps_label)

        action_row = QHBoxLayout()
        self._new_run_btn = QPushButton("Start New Run")
        self._new_run_btn.setToolTip(
            "Exit results mode and return to editable setup controls for a new run."
        )
        self._new_run_btn.clicked.connect(self._on_new_run)
        action_row.addWidget(self._new_run_btn)
        action_row.addStretch()
        layout.addLayout(action_row)
        layout.addStretch()
        return panel

    def _build_tuning_workspace_group(self) -> QGroupBox:
        """Bounded post-run tuning surface for downstream and correction retune workflows."""
        group = QGroupBox("Post-Run Tuning")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._tuning_phase_note = QLabel(
            "Post-run tuning tools. Completed outputs remain unchanged unless you run a new analysis."
        )
        self._tuning_phase_note.setWordWrap(True)
        self._tuning_phase_note.setObjectName("resultsSummaryHint")
        layout.addWidget(self._tuning_phase_note)

        disclosure_row = QHBoxLayout()
        self._tuning_disclosure_btn = QToolButton()
        self._tuning_disclosure_btn.setText("Primary: Event-Detection Tuning")
        self._tuning_disclosure_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._tuning_disclosure_btn.setArrowType(Qt.RightArrow)
        self._tuning_disclosure_btn.setCheckable(True)
        self._tuning_disclosure_btn.setChecked(False)
        self._tuning_disclosure_btn.setAutoRaise(True)
        self._tuning_disclosure_btn.setToolTip(
            "Expand downstream tuning controls for event-detection retuning from cache."
        )
        self._tuning_disclosure_btn.toggled.connect(self._on_tuning_disclosure_toggled)
        disclosure_row.addWidget(self._tuning_disclosure_btn)
        disclosure_row.addStretch()
        layout.addLayout(disclosure_row)

        self._tuning_collapsed_status_label = QLabel(
            "Tuning is available only after a successful completed run is loaded."
        )
        self._tuning_collapsed_status_label.setWordWrap(True)
        self._tuning_collapsed_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._tuning_collapsed_status_label.setMinimumWidth(0)
        self._set_status_label_style(self._tuning_collapsed_status_label, "warn")
        layout.addWidget(self._tuning_collapsed_status_label)

        self._tuning_content = QWidget()
        tuning_content_outer = QVBoxLayout(self._tuning_content)
        tuning_content_outer.setContentsMargins(0, 0, 0, 0)
        tuning_content_outer.setSpacing(0)
        self._tuning_scroll = QScrollArea()
        self._tuning_scroll.setWidgetResizable(True)
        self._tuning_scroll.setFrameShape(QScrollArea.NoFrame)
        self._tuning_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._tuning_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._tuning_scroll_content = QWidget()
        content_layout = QVBoxLayout(self._tuning_scroll_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        self._tuning_scroll.setWidget(self._tuning_scroll_content)
        tuning_content_outer.addWidget(self._tuning_scroll)
        layout.addWidget(self._tuning_content)
        self._tuning_content.setVisible(False)

        self._tuning_scope_note = QLabel(
            "Retunes downstream event detection from cached phasic traces only. "
            "Use this for quick threshold/signal iteration before deciding whether to rerun."
        )
        self._tuning_scope_note.setWordWrap(True)
        self._tuning_scope_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._tuning_scope_note.setMinimumWidth(0)
        self._tuning_scope_note.setStyleSheet("font-size: 11px; color: #555;")
        content_layout.addWidget(self._tuning_scope_note)

        self._tuning_availability_label = QLabel("Tuning is available only after a successful completed run is loaded.")
        self._tuning_availability_label.setWordWrap(True)
        self._tuning_availability_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._tuning_availability_label.setMinimumWidth(0)
        self._set_status_label_style(self._tuning_availability_label, "warn")
        content_layout.addWidget(self._tuning_availability_label)

        self._tuning_controls_container = QWidget()
        self._tuning_controls_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._tuning_controls_container.setMaximumWidth(640)
        form_row = QHBoxLayout()
        form_row.setContentsMargins(0, 0, 0, 0)
        form_row.setSpacing(0)
        form_row.addWidget(self._tuning_controls_container, 0, Qt.AlignLeft | Qt.AlignTop)
        form_row.addStretch(1)
        content_layout.addLayout(form_row)

        tuning_form = QFormLayout(self._tuning_controls_container)
        tuning_form.setContentsMargins(0, 0, 0, 0)
        tuning_form.setSpacing(6)
        tuning_form.setHorizontalSpacing(12)
        tuning_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        tuning_form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        tuning_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        def _tuning_row_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setMinimumWidth(170)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return lbl

        self._tuning_roi_combo = QComboBox()
        self._tuning_roi_combo.setMinimumWidth(220)
        self._tuning_roi_combo.setToolTip("ROI shown in the downstream tuning workspace.")
        self._tuning_roi_combo.currentIndexChanged.connect(self._on_tuning_roi_changed)
        tuning_form.addRow(_tuning_row_label("ROI:"), self._tuning_roi_combo)

        self._tuning_chunk_combo = QComboBox()
        self._tuning_chunk_combo.setMinimumWidth(220)
        self._tuning_chunk_combo.setToolTip(
            "Used for preview/inspection in the tuning workspace."
        )
        tuning_form.addRow(_tuning_row_label("Preview session:"), self._tuning_chunk_combo)

        self._tuning_event_signal_combo = QComboBox()
        self._tuning_event_signal_combo.setMinimumWidth(220)
        self._tuning_event_signal_combo.addItems(get_allowed_event_signals_from_config())
        self._tuning_event_signal_combo.setToolTip(
            "Signal used for event detection during downstream retuning."
        )
        tuning_form.addRow(_tuning_row_label("Event Signal:"), self._tuning_event_signal_combo)

        self._tuning_peak_method_combo = QComboBox()
        self._tuning_peak_method_combo.setMinimumWidth(220)
        self._tuning_peak_method_combo.addItems(get_allowed_peak_threshold_methods_from_config())
        self._tuning_peak_method_combo.setToolTip(
            "Method used to decide which peaks count as events."
        )
        self._tuning_peak_method_combo.currentIndexChanged.connect(self._on_tuning_peak_method_changed)
        tuning_form.addRow(_tuning_row_label("Peak Threshold Method:"), self._tuning_peak_method_combo)

        self._tuning_peak_k_spin = QDoubleSpinBox()
        self._tuning_peak_k_spin.setMinimumWidth(140)
        self._tuning_peak_k_spin.setRange(0.000001, 1_000_000.0)
        self._tuning_peak_k_spin.setDecimals(6)
        self._tuning_peak_k_spin.setSingleStep(0.1)
        self._tuning_peak_k_spin.setToolTip(
            "Scale factor used by threshold methods that require a K parameter."
        )
        self._tuning_peak_k_label = _tuning_row_label("Peak Threshold K:")
        tuning_form.addRow(self._tuning_peak_k_label, self._tuning_peak_k_spin)

        self._tuning_peak_pct_spin = QDoubleSpinBox()
        self._tuning_peak_pct_spin.setMinimumWidth(140)
        self._tuning_peak_pct_spin.setRange(0.0, 100.0)
        self._tuning_peak_pct_spin.setDecimals(3)
        self._tuning_peak_pct_spin.setSingleStep(1.0)
        self._tuning_peak_pct_spin.setToolTip(
            "Percentile cutoff used by percentile-based threshold methods."
        )
        self._tuning_peak_pct_label = _tuning_row_label("Peak Threshold Percentile:")
        tuning_form.addRow(self._tuning_peak_pct_label, self._tuning_peak_pct_spin)

        self._tuning_peak_abs_spin = QDoubleSpinBox()
        self._tuning_peak_abs_spin.setMinimumWidth(140)
        self._tuning_peak_abs_spin.setRange(0.0, 1_000_000.0)
        self._tuning_peak_abs_spin.setDecimals(6)
        self._tuning_peak_abs_spin.setSingleStep(0.05)
        self._tuning_peak_abs_spin.setToolTip(
            "Absolute threshold used when the selected method uses a fixed cutoff."
        )
        self._tuning_peak_abs_label = _tuning_row_label("Peak Threshold Absolute:")
        tuning_form.addRow(self._tuning_peak_abs_label, self._tuning_peak_abs_spin)

        self._tuning_peak_dist_spin = QDoubleSpinBox()
        self._tuning_peak_dist_spin.setMinimumWidth(140)
        self._tuning_peak_dist_spin.setRange(0.0, 10_000.0)
        self._tuning_peak_dist_spin.setDecimals(3)
        self._tuning_peak_dist_spin.setSingleStep(0.1)
        self._tuning_peak_dist_spin.setToolTip(
            "Minimum spacing between detected peaks, in seconds."
        )
        tuning_form.addRow(_tuning_row_label("Peak Min Distance (s):"), self._tuning_peak_dist_spin)

        self._tuning_peak_prominence_k_spin = QDoubleSpinBox()
        self._tuning_peak_prominence_k_spin.setMinimumWidth(140)
        self._tuning_peak_prominence_k_spin.setRange(0.0, 1_000_000.0)
        self._tuning_peak_prominence_k_spin.setDecimals(6)
        self._tuning_peak_prominence_k_spin.setSingleStep(0.1)
        self._tuning_peak_prominence_k_spin.setToolTip(
            "Minimum required peak prominence relative to robust noise. "
            "Higher values reject small fluctuations. Set to 0 to disable."
        )
        tuning_form.addRow(_tuning_row_label("Peak Min Prominence K:"), self._tuning_peak_prominence_k_spin)

        self._tuning_peak_width_sec_spin = QDoubleSpinBox()
        self._tuning_peak_width_sec_spin.setMinimumWidth(140)
        self._tuning_peak_width_sec_spin.setRange(0.0, 10_000.0)
        self._tuning_peak_width_sec_spin.setDecimals(6)
        self._tuning_peak_width_sec_spin.setSingleStep(0.1)
        self._tuning_peak_width_sec_spin.setToolTip(
            "Minimum required peak width in seconds. "
            "Higher values reject very narrow excursions. Set to 0 to disable."
        )
        tuning_form.addRow(_tuning_row_label("Peak Min Width (s):"), self._tuning_peak_width_sec_spin)

        self._tuning_peak_pre_filter_combo = QComboBox()
        self._tuning_peak_pre_filter_combo.setMinimumWidth(220)
        self._tuning_peak_pre_filter_combo.addItems(get_retune_peak_pre_filters())
        self._tuning_peak_pre_filter_combo.setToolTip(
            "Optional pre-filter applied before event detection. "
            "'smooth' uses phase-preserving Savitzky-Golay smoothing."
        )
        tuning_form.addRow(_tuning_row_label("Peak Pre-Filter:"), self._tuning_peak_pre_filter_combo)

        self._tuning_event_auc_combo = QComboBox()
        self._tuning_event_auc_combo.setMinimumWidth(220)
        self._tuning_event_auc_combo.addItems(get_allowed_event_auc_baselines_from_config())
        self._tuning_event_auc_combo.setToolTip(
            "Reference baseline convention used when integrating event AUC."
        )
        tuning_form.addRow(_tuning_row_label("Event AUC Baseline:"), self._tuning_event_auc_combo)
        self._apply_form_row_tooltips(tuning_form)

        btn_row = QHBoxLayout()
        self._run_tuning_btn = QPushButton("Run Tuning")
        self._run_tuning_btn.setToolTip(
            "Run downstream cache retuning for the selected ROI and preview session."
        )
        self._run_tuning_btn.clicked.connect(self._on_run_tuning)
        btn_row.addWidget(self._run_tuning_btn)
        self._open_tuning_dir_btn = QPushButton("Open Tuning Output")
        self._open_tuning_dir_btn.setToolTip(
            "Open the output folder from the most recent downstream retune."
        )
        self._open_tuning_dir_btn.clicked.connect(self._on_open_tuning_output)
        btn_row.addWidget(self._open_tuning_dir_btn)
        btn_row.addStretch()
        content_layout.addLayout(btn_row)

        apply_row = QHBoxLayout()
        self._apply_tuning_btn = QPushButton("Apply to Next-Run Settings")
        self._apply_tuning_btn.setToolTip(
            "Copy tuned downstream values into main setup controls for the next run. "
            "This does not modify completed outputs; rerun to generate updated artifacts."
        )
        self._apply_tuning_btn.clicked.connect(self._on_apply_tuning_values_to_run_settings)
        apply_row.addWidget(self._apply_tuning_btn)
        self._save_tuned_config_btn = QPushButton("Save Tuned Settings as Config...")
        self._save_tuned_config_btn.setToolTip(
            "Durable action: write the current tuned effective settings to a reusable YAML config file."
        )
        self._save_tuned_config_btn.clicked.connect(self._on_save_tuned_settings_as_config)
        apply_row.addWidget(self._save_tuned_config_btn)
        apply_row.addStretch()
        content_layout.addLayout(apply_row)

        self._tuning_applyback_scope_label = QLabel(
            "Apply-back is temporary for this session's next run. "
            "Use 'Save Tuned Settings as Config...' for durable YAML reuse."
        )
        self._tuning_applyback_scope_label.setWordWrap(True)
        self._tuning_applyback_scope_label.setObjectName("resultsSummaryHint")
        content_layout.addWidget(self._tuning_applyback_scope_label)

        self._tuning_summary_label = QLabel("No tuning result yet.")
        self._tuning_summary_label.setWordWrap(True)
        self._tuning_summary_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._tuning_summary_label.setMinimumWidth(0)
        self._tuning_summary_label.setToolTip(
            "Summary of the most recent downstream tuning run."
        )
        self._tuning_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(self._tuning_summary_label)

        self._tuning_applyback_status_label = QLabel("Apply-back status: not applied.")
        self._tuning_applyback_status_label.setWordWrap(True)
        self._tuning_applyback_status_label.setObjectName("resultsSummaryHint")
        content_layout.addWidget(self._tuning_applyback_status_label)

        self._tuning_overlay_title = QLabel("No tuning overlay loaded.")
        self._tuning_overlay_title.setAlignment(Qt.AlignCenter)
        self._tuning_overlay_title.setToolTip(
            "Filename of the currently displayed downstream tuning overlay."
        )
        content_layout.addWidget(self._tuning_overlay_title)

        self._tuning_overlay_label = _ClickableImageLabel(
            "Run tuning to generate an ROI preview-session event overlay."
        )
        self._tuning_overlay_label.setAlignment(Qt.AlignCenter)
        self._tuning_overlay_label.setToolTip(
            "Preview overlay for the selected downstream ROI and preview session. "
            "Click image to toggle fit/full-size inspection."
        )
        self._tuning_overlay_label.setStyleSheet(
            "QLabel { background: #111; color: #ddd; border: 1px solid #444; }"
        )
        self._tuning_overlay_label.clicked.connect(self._on_tuning_overlay_clicked)
        self._tuning_overlay_scroll = QScrollArea()
        self._tuning_overlay_scroll.setWidgetResizable(False)
        self._tuning_overlay_scroll.setAlignment(Qt.AlignCenter)
        self._tuning_overlay_scroll.setFrameShape(QScrollArea.NoFrame)
        self._tuning_overlay_scroll.setMinimumHeight(180)
        self._tuning_overlay_scroll.setMaximumHeight(520)
        self._tuning_overlay_scroll.setWidget(self._tuning_overlay_label)
        content_layout.addWidget(self._tuning_overlay_scroll)
        self._tuning_overlay_zoom_hint_label = QLabel(
            "Scroll wheel to zoom. Click image to toggle fit/full size."
        )
        self._tuning_overlay_zoom_hint_label.setAlignment(Qt.AlignCenter)
        self._tuning_overlay_zoom_hint_label.setStyleSheet("font-size: 11px; color: #666;")
        content_layout.addWidget(self._tuning_overlay_zoom_hint_label)
        self._tuning_overlay_interaction = InteractiveImageController(
            label=self._tuning_overlay_label,
            scroll_area=self._tuning_overlay_scroll,
            set_hint_text=self._tuning_overlay_zoom_hint_label.setText,
            fit_hint="Scroll wheel to zoom. Click image to toggle fit/full size.",
            zoom_hint="Scroll wheel to zoom in/out. Drag to pan. Click image to return to fit.",
            allow_upscale_in_fit=False,
            on_zoom_mode_changed=self._on_tuning_overlay_zoom_mode_changed,
        )

        self._build_correction_tuning_subsection(layout)

        self._on_tuning_peak_method_changed()
        self._set_tuning_workspace_unavailable(
            "Tuning is available only after a successful completed run is loaded."
        )
        self._set_correction_tuning_workspace_unavailable(
            "Correction retune is available only after a successful completed run is loaded."
        )
        self._set_tuning_disclosure_expanded(False)
        self._set_correction_tuning_disclosure_expanded(False)
        group.setVisible(False)
        return group

    def _build_correction_tuning_subsection(self, parent_layout: QVBoxLayout) -> None:
        """Build correction-sensitive retune subsection (separate from downstream tuning)."""
        section = QWidget()
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(6)

        disclosure_row = QHBoxLayout()
        self._correction_tuning_disclosure_btn = QToolButton()
        self._correction_tuning_disclosure_btn.setText("Secondary: Correction Retune (Advanced)")
        self._correction_tuning_disclosure_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._correction_tuning_disclosure_btn.setArrowType(Qt.RightArrow)
        self._correction_tuning_disclosure_btn.setCheckable(True)
        self._correction_tuning_disclosure_btn.setChecked(False)
        self._correction_tuning_disclosure_btn.setAutoRaise(True)
        self._correction_tuning_disclosure_btn.setToolTip(
            "Expand correction-sensitive retune controls."
        )
        self._correction_tuning_disclosure_btn.toggled.connect(
            self._on_correction_tuning_disclosure_toggled
        )
        disclosure_row.addWidget(self._correction_tuning_disclosure_btn)
        disclosure_row.addStretch()
        section_layout.addLayout(disclosure_row)

        self._correction_tuning_role_note = QLabel(
            "Advanced path. Use when baseline/correction assumptions need deliberate recomputation."
        )
        self._correction_tuning_role_note.setWordWrap(True)
        self._correction_tuning_role_note.setObjectName("resultsSummaryHint")
        section_layout.addWidget(self._correction_tuning_role_note)

        self._correction_tuning_collapsed_status_label = QLabel(
            "Correction retune is available only after a successful completed run is loaded."
        )
        self._correction_tuning_collapsed_status_label.setWordWrap(True)
        self._correction_tuning_collapsed_status_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        self._correction_tuning_collapsed_status_label.setMinimumWidth(0)
        self._set_status_label_style(self._correction_tuning_collapsed_status_label, "warn")
        section_layout.addWidget(self._correction_tuning_collapsed_status_label)

        self._correction_tuning_content = QWidget()
        content_outer = QVBoxLayout(self._correction_tuning_content)
        content_outer.setContentsMargins(0, 0, 0, 0)
        content_outer.setSpacing(0)

        self._correction_tuning_scroll = QScrollArea()
        self._correction_tuning_scroll.setWidgetResizable(True)
        self._correction_tuning_scroll.setFrameShape(QScrollArea.NoFrame)
        self._correction_tuning_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._correction_tuning_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._correction_tuning_scroll_content = QWidget()
        content_layout = QVBoxLayout(self._correction_tuning_scroll_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        self._correction_tuning_scroll.setWidget(self._correction_tuning_scroll_content)
        content_outer.addWidget(self._correction_tuning_scroll)
        section_layout.addWidget(self._correction_tuning_content)
        self._correction_tuning_content.setVisible(False)

        self._correction_tuning_scope_note = QLabel(
            "Recomputes baseline and correction for the selected ROI across all sessions available for that ROI. "
            "The preview session is used only for the inspection figure. "
            "Outputs are written to an isolated retune directory. "
            "Production run artifacts are not modified."
        )
        self._correction_tuning_scope_note.setWordWrap(True)
        self._correction_tuning_scope_note.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self._correction_tuning_scope_note.setMinimumWidth(0)
        self._correction_tuning_scope_note.setStyleSheet("font-size: 11px; color: #555;")
        content_layout.addWidget(self._correction_tuning_scope_note)

        self._correction_tuning_availability_label = QLabel(
            "Correction retune is available only after a successful completed run is loaded."
        )
        self._correction_tuning_availability_label.setWordWrap(True)
        self._correction_tuning_availability_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        self._correction_tuning_availability_label.setMinimumWidth(0)
        self._set_status_label_style(self._correction_tuning_availability_label, "warn")
        content_layout.addWidget(self._correction_tuning_availability_label)

        self._correction_tuning_controls_container = QWidget()
        self._correction_tuning_controls_container.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        self._correction_tuning_controls_container.setMaximumWidth(700)
        form_row = QHBoxLayout()
        form_row.setContentsMargins(0, 0, 0, 0)
        form_row.setSpacing(0)
        form_row.addWidget(
            self._correction_tuning_controls_container, 0, Qt.AlignLeft | Qt.AlignTop
        )
        form_row.addStretch(1)
        content_layout.addLayout(form_row)

        form = QFormLayout(self._correction_tuning_controls_container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        form.setHorizontalSpacing(12)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        def _corr_row_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setMinimumWidth(190)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return lbl

        self._correction_tuning_roi_combo = QComboBox()
        self._correction_tuning_roi_combo.setMinimumWidth(220)
        self._correction_tuning_roi_combo.setToolTip(
            "ROI whose baseline and correction will be recomputed across all available sessions."
        )
        self._correction_tuning_roi_combo.currentIndexChanged.connect(
            self._on_correction_tuning_roi_changed
        )
        form.addRow(_corr_row_label("ROI:"), self._correction_tuning_roi_combo)

        self._correction_tuning_chunk_combo = QComboBox()
        self._correction_tuning_chunk_combo.setMinimumWidth(220)
        self._correction_tuning_chunk_combo.setToolTip(
            "Used only for the inspection figure. Does not limit recomputation."
        )
        form.addRow(_corr_row_label("Preview session:"), self._correction_tuning_chunk_combo)

        self._correction_tuning_baseline_method_combo = QComboBox()
        self._correction_tuning_baseline_method_combo.setMinimumWidth(240)
        self._correction_tuning_baseline_method_combo.addItems(
            get_allowed_baseline_methods_from_config()
        )
        self._correction_tuning_baseline_method_combo.setToolTip(
            "Method used to estimate baseline context before normalization."
        )
        form.addRow(
            _corr_row_label("Baseline Method:"),
            self._correction_tuning_baseline_method_combo,
        )

        self._correction_tuning_baseline_pct_spin = QDoubleSpinBox()
        self._correction_tuning_baseline_pct_spin.setMinimumWidth(140)
        self._correction_tuning_baseline_pct_spin.setRange(0.0, 100.0)
        self._correction_tuning_baseline_pct_spin.setDecimals(3)
        self._correction_tuning_baseline_pct_spin.setSingleStep(1.0)
        self._correction_tuning_baseline_pct_spin.setToolTip(
            "Percentile used when estimating the session baseline."
        )
        form.addRow(
            _corr_row_label("Baseline Percentile:"),
            self._correction_tuning_baseline_pct_spin,
        )

        self._correction_tuning_lowpass_spin = QDoubleSpinBox()
        self._correction_tuning_lowpass_spin.setMinimumWidth(140)
        self._correction_tuning_lowpass_spin.setRange(0.000001, 10_000.0)
        self._correction_tuning_lowpass_spin.setDecimals(6)
        self._correction_tuning_lowpass_spin.setSingleStep(0.1)
        self._correction_tuning_lowpass_spin.setToolTip(
            "Lowpass cutoff applied before correction-related computations."
        )
        form.addRow(_corr_row_label("Lowpass Filter (Hz):"), self._correction_tuning_lowpass_spin)

        self._correction_tuning_fit_mode_combo = QComboBox()
        self._correction_tuning_fit_mode_combo.setMinimumWidth(260)
        self._correction_tuning_fit_mode_combo.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["dynamic_fit_mode"]
        )
        for mode_name in get_allowed_dynamic_fit_modes_from_config():
            self._correction_tuning_fit_mode_combo.addItem(dynamic_fit_mode_label(mode_name), mode_name)
        self._correction_tuning_fit_mode_combo.currentIndexChanged.connect(
            self._on_correction_tuning_fit_mode_changed
        )
        form.addRow(_corr_row_label("Dynamic Fit Mode:"), self._correction_tuning_fit_mode_combo)

        self._correction_tuning_fit_mode_note = QLabel("")
        self._correction_tuning_fit_mode_note.setWordWrap(True)
        self._correction_tuning_fit_mode_note.setStyleSheet("font-size: 11px; color: #666;")
        form.addRow(self._correction_tuning_fit_mode_note)

        self._correction_tuning_baseline_subtract_cb = QCheckBox("")
        self._correction_tuning_baseline_subtract_cb.setChecked(
            bool(getattr(self._default_cfg, "baseline_subtract_before_fit", False))
        )
        self._correction_tuning_baseline_subtract_cb.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["baseline_subtract_before_fit"]
        )
        self._correction_tuning_baseline_subtract_cb.stateChanged.connect(self._on_config_changed)
        form.addRow(
            _corr_row_label("Baseline subtract before fit:"),
            self._correction_tuning_baseline_subtract_cb,
        )

        self._correction_tuning_window_spin = QDoubleSpinBox()
        self._correction_tuning_window_spin.setMinimumWidth(140)
        self._correction_tuning_window_spin.setRange(0.000001, 1_000_000.0)
        self._correction_tuning_window_spin.setDecimals(6)
        self._correction_tuning_window_spin.setSingleStep(1.0)
        self._correction_tuning_window_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["regression_window_sec"]
        )
        form.addRow(_corr_row_label("Regression Window (s):"), self._correction_tuning_window_spin)

        self._correction_tuning_min_samples_spin = QSpinBox()
        self._correction_tuning_min_samples_spin.setMinimumWidth(120)
        self._correction_tuning_min_samples_spin.setRange(1, 1_000_000)
        self._correction_tuning_min_samples_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["min_samples_per_window"]
        )
        form.addRow(
            _corr_row_label("Min Samples/Window:"),
            self._correction_tuning_min_samples_spin,
        )

        self._correction_tuning_robust_max_iters_spin = QSpinBox()
        self._correction_tuning_robust_max_iters_spin.setMinimumWidth(120)
        self._correction_tuning_robust_max_iters_spin.setRange(1, 1000)
        self._correction_tuning_robust_max_iters_spin.setValue(
            int(getattr(self._default_cfg, "robust_event_reject_max_iters", 3))
        )
        self._correction_tuning_robust_max_iters_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_max_iters"]
        )
        form.addRow(
            _corr_row_label("Max iterations:"),
            self._correction_tuning_robust_max_iters_spin,
        )

        self._correction_tuning_robust_residual_z_spin = QDoubleSpinBox()
        self._correction_tuning_robust_residual_z_spin.setMinimumWidth(140)
        self._correction_tuning_robust_residual_z_spin.setRange(0.000001, 1_000_000.0)
        self._correction_tuning_robust_residual_z_spin.setDecimals(4)
        self._correction_tuning_robust_residual_z_spin.setSingleStep(0.1)
        self._correction_tuning_robust_residual_z_spin.setValue(
            float(getattr(self._default_cfg, "robust_event_reject_residual_z_thresh", 3.5))
        )
        self._correction_tuning_robust_residual_z_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_residual_z_thresh"]
        )
        form.addRow(
            _corr_row_label("Residual z-threshold:"),
            self._correction_tuning_robust_residual_z_spin,
        )

        self._correction_tuning_robust_local_var_window_spin = QDoubleSpinBox()
        self._correction_tuning_robust_local_var_window_spin.setMinimumWidth(140)
        self._correction_tuning_robust_local_var_window_spin.setRange(0.000001, 1_000_000.0)
        self._correction_tuning_robust_local_var_window_spin.setDecimals(4)
        self._correction_tuning_robust_local_var_window_spin.setSingleStep(0.5)
        self._correction_tuning_robust_local_var_window_spin.setValue(
            float(getattr(self._default_cfg, "robust_event_reject_local_var_window_sec", 10.0) or 10.0)
        )
        self._correction_tuning_robust_local_var_window_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_local_var_window_sec"]
        )
        form.addRow(
            _corr_row_label("Local variance window (s):"),
            self._correction_tuning_robust_local_var_window_spin,
        )

        self._correction_tuning_robust_local_var_ratio_enable_cb = QCheckBox("Enable")
        self._correction_tuning_robust_local_var_ratio_enable_cb.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_local_var_ratio_thresh"]
        )
        self._correction_tuning_robust_local_var_ratio_spin = QDoubleSpinBox()
        self._correction_tuning_robust_local_var_ratio_spin.setMinimumWidth(140)
        self._correction_tuning_robust_local_var_ratio_spin.setRange(0.000001, 1_000_000.0)
        self._correction_tuning_robust_local_var_ratio_spin.setDecimals(4)
        self._correction_tuning_robust_local_var_ratio_spin.setSingleStep(0.1)
        ratio_default = getattr(self._default_cfg, "robust_event_reject_local_var_ratio_thresh", None)
        self._correction_tuning_robust_local_var_ratio_enable_cb.setChecked(
            ratio_default is not None
        )
        self._correction_tuning_robust_local_var_ratio_spin.setValue(
            float(ratio_default if ratio_default is not None else 3.0)
        )
        self._correction_tuning_robust_local_var_ratio_spin.setEnabled(bool(ratio_default is not None))
        self._correction_tuning_robust_local_var_ratio_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_local_var_ratio_thresh"]
        )
        ratio_row = QWidget()
        ratio_row.setToolTip(_DYNAMIC_FIT_TOOLTIPS["robust_local_var_ratio_thresh"])
        ratio_row_layout = QHBoxLayout(ratio_row)
        ratio_row_layout.setContentsMargins(0, 0, 0, 0)
        ratio_row_layout.setSpacing(8)
        ratio_row_layout.addWidget(self._correction_tuning_robust_local_var_ratio_enable_cb)
        ratio_row_layout.addWidget(self._correction_tuning_robust_local_var_ratio_spin)
        form.addRow(
            _corr_row_label("Local variance ratio threshold:"),
            ratio_row,
        )

        self._correction_tuning_robust_min_keep_fraction_spin = QDoubleSpinBox()
        self._correction_tuning_robust_min_keep_fraction_spin.setMinimumWidth(140)
        self._correction_tuning_robust_min_keep_fraction_spin.setRange(0.000001, 1.0)
        self._correction_tuning_robust_min_keep_fraction_spin.setDecimals(4)
        self._correction_tuning_robust_min_keep_fraction_spin.setSingleStep(0.05)
        self._correction_tuning_robust_min_keep_fraction_spin.setValue(
            float(getattr(self._default_cfg, "robust_event_reject_min_keep_fraction", 0.5))
        )
        self._correction_tuning_robust_min_keep_fraction_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_min_keep_fraction"]
        )
        form.addRow(
            _corr_row_label("Minimum keep fraction:"),
            self._correction_tuning_robust_min_keep_fraction_spin,
        )

        self._correction_tuning_adaptive_residual_z_spin = QDoubleSpinBox()
        self._correction_tuning_adaptive_residual_z_spin.setMinimumWidth(140)
        self._correction_tuning_adaptive_residual_z_spin.setRange(0.000001, 1_000_000.0)
        self._correction_tuning_adaptive_residual_z_spin.setDecimals(4)
        self._correction_tuning_adaptive_residual_z_spin.setSingleStep(0.1)
        self._correction_tuning_adaptive_residual_z_spin.setValue(
            float(getattr(self._default_cfg, "adaptive_event_gate_residual_z_thresh", 3.5))
        )
        self._correction_tuning_adaptive_residual_z_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_residual_z_thresh"]
        )
        form.addRow(
            _corr_row_label("Adaptive residual z-threshold:"),
            self._correction_tuning_adaptive_residual_z_spin,
        )

        self._correction_tuning_adaptive_local_var_window_spin = QDoubleSpinBox()
        self._correction_tuning_adaptive_local_var_window_spin.setMinimumWidth(140)
        self._correction_tuning_adaptive_local_var_window_spin.setRange(0.000001, 1_000_000.0)
        self._correction_tuning_adaptive_local_var_window_spin.setDecimals(4)
        self._correction_tuning_adaptive_local_var_window_spin.setSingleStep(0.5)
        self._correction_tuning_adaptive_local_var_window_spin.setValue(
            float(getattr(self._default_cfg, "adaptive_event_gate_local_var_window_sec", 10.0) or 10.0)
        )
        self._correction_tuning_adaptive_local_var_window_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_local_var_window_sec"]
        )
        form.addRow(
            _corr_row_label("Adaptive local variance window (s):"),
            self._correction_tuning_adaptive_local_var_window_spin,
        )

        self._correction_tuning_adaptive_local_var_ratio_enable_cb = QCheckBox("Enable")
        self._correction_tuning_adaptive_local_var_ratio_enable_cb.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_local_var_ratio_thresh"]
        )
        self._correction_tuning_adaptive_local_var_ratio_spin = QDoubleSpinBox()
        self._correction_tuning_adaptive_local_var_ratio_spin.setMinimumWidth(140)
        self._correction_tuning_adaptive_local_var_ratio_spin.setRange(0.000001, 1_000_000.0)
        self._correction_tuning_adaptive_local_var_ratio_spin.setDecimals(4)
        self._correction_tuning_adaptive_local_var_ratio_spin.setSingleStep(0.1)
        self._correction_tuning_adaptive_local_var_ratio_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_local_var_ratio_thresh"]
        )
        adaptive_ratio_default = getattr(
            self._default_cfg, "adaptive_event_gate_local_var_ratio_thresh", None
        )
        self._correction_tuning_adaptive_local_var_ratio_enable_cb.setChecked(
            adaptive_ratio_default is not None
        )
        self._correction_tuning_adaptive_local_var_ratio_spin.setValue(
            float(adaptive_ratio_default if adaptive_ratio_default is not None else 3.0)
        )
        self._correction_tuning_adaptive_local_var_ratio_spin.setEnabled(
            bool(adaptive_ratio_default is not None)
        )
        adaptive_ratio_row = QWidget()
        adaptive_ratio_row.setToolTip(_DYNAMIC_FIT_TOOLTIPS["adaptive_local_var_ratio_thresh"])
        adaptive_ratio_row_layout = QHBoxLayout(adaptive_ratio_row)
        adaptive_ratio_row_layout.setContentsMargins(0, 0, 0, 0)
        adaptive_ratio_row_layout.setSpacing(8)
        adaptive_ratio_row_layout.addWidget(self._correction_tuning_adaptive_local_var_ratio_enable_cb)
        adaptive_ratio_row_layout.addWidget(self._correction_tuning_adaptive_local_var_ratio_spin)
        form.addRow(
            _corr_row_label("Adaptive local variance ratio threshold:"),
            adaptive_ratio_row,
        )

        self._correction_tuning_adaptive_smooth_window_spin = QDoubleSpinBox()
        self._correction_tuning_adaptive_smooth_window_spin.setMinimumWidth(140)
        self._correction_tuning_adaptive_smooth_window_spin.setRange(0.000001, 1_000_000.0)
        self._correction_tuning_adaptive_smooth_window_spin.setDecimals(4)
        self._correction_tuning_adaptive_smooth_window_spin.setSingleStep(1.0)
        self._correction_tuning_adaptive_smooth_window_spin.setValue(
            float(getattr(self._default_cfg, "adaptive_event_gate_smooth_window_sec", 60.0))
        )
        self._correction_tuning_adaptive_smooth_window_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_smooth_window_sec"]
        )
        form.addRow(
            _corr_row_label("Adaptive smoothing window duration (s):"),
            self._correction_tuning_adaptive_smooth_window_spin,
        )

        self._correction_tuning_adaptive_min_trust_fraction_spin = QDoubleSpinBox()
        self._correction_tuning_adaptive_min_trust_fraction_spin.setMinimumWidth(140)
        self._correction_tuning_adaptive_min_trust_fraction_spin.setRange(0.000001, 1.0)
        self._correction_tuning_adaptive_min_trust_fraction_spin.setDecimals(4)
        self._correction_tuning_adaptive_min_trust_fraction_spin.setSingleStep(0.05)
        self._correction_tuning_adaptive_min_trust_fraction_spin.setValue(
            float(getattr(self._default_cfg, "adaptive_event_gate_min_trust_fraction", 0.5))
        )
        self._correction_tuning_adaptive_min_trust_fraction_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_min_trust_fraction"]
        )
        form.addRow(
            _corr_row_label("Adaptive minimum trust fraction:"),
            self._correction_tuning_adaptive_min_trust_fraction_spin,
        )

        self._correction_tuning_adaptive_freeze_interp_combo = QComboBox()
        self._correction_tuning_adaptive_freeze_interp_combo.setMinimumWidth(180)
        self._correction_tuning_adaptive_freeze_interp_combo.addItem("linear_hold", "linear_hold")
        default_freeze_interp = str(
            getattr(self._default_cfg, "adaptive_event_gate_freeze_interp_method", "linear_hold")
        ).strip() or "linear_hold"
        idx_freeze_interp = self._correction_tuning_adaptive_freeze_interp_combo.findData(default_freeze_interp)
        if idx_freeze_interp >= 0:
            self._correction_tuning_adaptive_freeze_interp_combo.setCurrentIndex(idx_freeze_interp)
        self._correction_tuning_adaptive_freeze_interp_combo.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_freeze_interp_method"]
        )
        form.addRow(
            _corr_row_label("Adaptive freeze interpolation method:"),
            self._correction_tuning_adaptive_freeze_interp_combo,
        )

        self._correction_tuning_robust_max_iters_spin.valueChanged.connect(self._on_config_changed)
        self._correction_tuning_robust_residual_z_spin.valueChanged.connect(self._on_config_changed)
        self._correction_tuning_robust_local_var_window_spin.valueChanged.connect(self._on_config_changed)
        self._correction_tuning_robust_local_var_ratio_enable_cb.toggled.connect(
            self._correction_tuning_robust_local_var_ratio_spin.setEnabled
        )
        self._correction_tuning_robust_local_var_ratio_enable_cb.toggled.connect(self._on_config_changed)
        self._correction_tuning_robust_local_var_ratio_spin.valueChanged.connect(self._on_config_changed)
        self._correction_tuning_robust_min_keep_fraction_spin.valueChanged.connect(self._on_config_changed)
        self._correction_tuning_adaptive_residual_z_spin.valueChanged.connect(self._on_config_changed)
        self._correction_tuning_adaptive_local_var_window_spin.valueChanged.connect(self._on_config_changed)
        self._correction_tuning_adaptive_local_var_ratio_enable_cb.toggled.connect(
            self._correction_tuning_adaptive_local_var_ratio_spin.setEnabled
        )
        self._correction_tuning_adaptive_local_var_ratio_enable_cb.toggled.connect(self._on_config_changed)
        self._correction_tuning_adaptive_local_var_ratio_spin.valueChanged.connect(self._on_config_changed)
        self._correction_tuning_adaptive_smooth_window_spin.valueChanged.connect(self._on_config_changed)
        self._correction_tuning_adaptive_min_trust_fraction_spin.valueChanged.connect(self._on_config_changed)
        self._correction_tuning_adaptive_freeze_interp_combo.currentIndexChanged.connect(self._on_config_changed)
        self._apply_form_row_tooltips(form)
        self._apply_correction_tuning_fit_mode_ui_state()

        btn_row = QHBoxLayout()
        self._run_correction_tuning_btn = QPushButton("Run Correction Retune")
        self._run_correction_tuning_btn.setToolTip(
            "Run correction-sensitive recomputation for the selected ROI."
        )
        self._run_correction_tuning_btn.clicked.connect(self._on_run_correction_tuning)
        btn_row.addWidget(self._run_correction_tuning_btn)
        self._open_correction_tuning_dir_btn = QPushButton("Open Correction Output")
        self._open_correction_tuning_dir_btn.setToolTip(
            "Open the output folder from the most recent correction retune."
        )
        self._open_correction_tuning_dir_btn.clicked.connect(
            self._on_open_correction_tuning_output
        )
        btn_row.addWidget(self._open_correction_tuning_dir_btn)
        btn_row.addStretch()
        content_layout.addLayout(btn_row)

        apply_row = QHBoxLayout()
        self._apply_correction_tuning_btn = QPushButton(
            "Use These Correction Settings for Next Run"
        )
        self._apply_correction_tuning_btn.setToolTip(
            "Copy current correction-retune settings into main Advanced correction controls "
            "for the next run. This does not auto-run analysis and does not change completed outputs."
        )
        self._apply_correction_tuning_btn.clicked.connect(
            self._on_apply_correction_tuning_values_to_run_settings
        )
        apply_row.addWidget(self._apply_correction_tuning_btn)
        self._save_correction_tuned_config_btn = QPushButton(
            "Save Tuned Settings as Config..."
        )
        self._save_correction_tuned_config_btn.setToolTip(
            "Durable action: write the current tuned effective settings to a reusable YAML config file."
        )
        self._save_correction_tuned_config_btn.clicked.connect(
            self._on_save_tuned_settings_as_config
        )
        apply_row.addWidget(self._save_correction_tuned_config_btn)
        apply_row.addStretch()
        content_layout.addLayout(apply_row)

        self._correction_tuning_applyback_scope_label = QLabel(
            "Copy action is one-way (Correction Retune -> main run configuration) "
            "and temporary for the next run. Use 'Save Tuned Settings as Config...' "
            "for durable YAML reuse."
        )
        self._correction_tuning_applyback_scope_label.setWordWrap(True)
        self._correction_tuning_applyback_scope_label.setObjectName("resultsSummaryHint")
        content_layout.addWidget(self._correction_tuning_applyback_scope_label)

        self._correction_tuning_applyback_status_label = QLabel("Copy status: not copied.")
        self._correction_tuning_applyback_status_label.setWordWrap(True)
        self._correction_tuning_applyback_status_label.setObjectName("resultsSummaryHint")
        content_layout.addWidget(self._correction_tuning_applyback_status_label)

        self._correction_tuning_summary_label = QLabel("No correction retune result yet.")
        self._correction_tuning_summary_label.setWordWrap(True)
        self._correction_tuning_summary_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self._correction_tuning_summary_label.setMinimumWidth(0)
        self._correction_tuning_summary_label.setToolTip(
            "Summary of the most recent correction retune run."
        )
        self._correction_tuning_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(self._correction_tuning_summary_label)

        self._correction_tuning_inspection_title = QLabel(
            "No correction inspection artifact loaded."
        )
        self._correction_tuning_inspection_title.setAlignment(Qt.AlignCenter)
        self._correction_tuning_inspection_title.setToolTip(
            "Current correction inspection panel and file."
        )
        content_layout.addWidget(self._correction_tuning_inspection_title)

        self._correction_tuning_inspection_label = _ClickableImageLabel(
            "Run correction retune to generate a correction inspection artifact."
        )
        self._correction_tuning_inspection_label.setAlignment(Qt.AlignCenter)
        self._correction_tuning_inspection_label.setToolTip(
            "Inspection panel for the selected preview session after correction retune. "
            "Use previous/next to switch Raw/Centered/Fit/dF/F views. "
            "Click image to toggle fit/full-size inspection."
        )
        self._correction_tuning_inspection_label.setStyleSheet(
            "QLabel { background: #111; color: #ddd; border: 1px solid #444; }"
        )
        self._correction_tuning_inspection_label.clicked.connect(
            self._on_correction_tuning_inspection_clicked
        )
        self._correction_tuning_inspection_scroll = QScrollArea()
        self._correction_tuning_inspection_scroll.setWidgetResizable(False)
        self._correction_tuning_inspection_scroll.setAlignment(Qt.AlignCenter)
        self._correction_tuning_inspection_scroll.setFrameShape(QScrollArea.NoFrame)
        self._correction_tuning_inspection_scroll.setMinimumHeight(220)
        self._correction_tuning_inspection_scroll.setMaximumHeight(680)
        self._correction_tuning_inspection_scroll.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self._correction_tuning_inspection_scroll.setWidget(
            self._correction_tuning_inspection_label
        )
        content_layout.addWidget(self._correction_tuning_inspection_scroll, 1)
        nav_row = QHBoxLayout()
        self._correction_tuning_prev_btn = QPushButton("<")
        self._correction_tuning_prev_btn.setToolTip(
            "Previous correction inspection panel image."
        )
        self._correction_tuning_prev_btn.clicked.connect(
            self._on_correction_tuning_prev_image
        )
        nav_row.addWidget(self._correction_tuning_prev_btn)
        self._correction_tuning_inspection_counter_label = QLabel("")
        self._correction_tuning_inspection_counter_label.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self._correction_tuning_inspection_counter_label, 1)
        self._correction_tuning_next_btn = QPushButton(">")
        self._correction_tuning_next_btn.setToolTip(
            "Next correction inspection panel image."
        )
        self._correction_tuning_next_btn.clicked.connect(
            self._on_correction_tuning_next_image
        )
        nav_row.addWidget(self._correction_tuning_next_btn)
        content_layout.addLayout(nav_row)
        self._correction_tuning_prev_btn.setEnabled(False)
        self._correction_tuning_next_btn.setEnabled(False)
        self._correction_tuning_zoom_hint_label = QLabel(
            "Scroll wheel to zoom. Click image to toggle fit/full size."
        )
        self._correction_tuning_zoom_hint_label.setAlignment(Qt.AlignCenter)
        self._correction_tuning_zoom_hint_label.setStyleSheet(
            "font-size: 11px; color: #666;"
        )
        content_layout.addWidget(self._correction_tuning_zoom_hint_label)
        self._correction_tuning_overlay_interaction = InteractiveImageController(
            label=self._correction_tuning_inspection_label,
            scroll_area=self._correction_tuning_inspection_scroll,
            set_hint_text=self._correction_tuning_zoom_hint_label.setText,
            fit_hint="Scroll wheel to zoom. Click image to toggle fit/full size.",
            zoom_hint="Scroll wheel to zoom in/out. Drag to pan. Click image to return to fit.",
            allow_upscale_in_fit=True,
            on_zoom_mode_changed=self._on_correction_tuning_zoom_mode_changed,
        )

        parent_layout.addWidget(section)

    # ==================================================================
    # Log Panel
    # ==================================================================

    def _build_log_panel(self) -> QGroupBox:
        group = QGroupBox("Live Log")
        group.setObjectName("liveLogSection")
        group.setProperty("workflowSection", True)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        disclosure_row = QHBoxLayout()
        self._live_log_disclosure_btn = QToolButton()
        self._live_log_disclosure_btn.setText("Log output")
        self._live_log_disclosure_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._live_log_disclosure_btn.setArrowType(Qt.RightArrow)
        self._live_log_disclosure_btn.setCheckable(True)
        self._live_log_disclosure_btn.setChecked(False)
        self._live_log_disclosure_btn.setAutoRaise(True)
        self._live_log_disclosure_btn.setToolTip(
            "Expand to inspect accumulated run output and diagnostics."
        )
        self._live_log_disclosure_btn.toggled.connect(self._on_live_log_disclosure_toggled)
        disclosure_row.addWidget(self._live_log_disclosure_btn)
        disclosure_row.addStretch()
        layout.addLayout(disclosure_row)

        self._live_log_collapsed_hint = QLabel(
            "Hidden during normal use. Expand to inspect run output."
        )
        self._live_log_collapsed_hint.setWordWrap(True)
        self._live_log_collapsed_hint.setStyleSheet("font-size: 11px; color: #5f6b7a;")
        layout.addWidget(self._live_log_collapsed_hint)

        self._live_log_content_container = QWidget()
        content_layout = QVBoxLayout(self._live_log_content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        layout.addWidget(self._live_log_content_container)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Consolas", 9))
        self._log_view.setMaximumBlockCount(10000)
        self._log_view.setMinimumHeight(180)
        content_layout.addWidget(self._log_view)
        self._on_live_log_disclosure_toggled(False)

        return group

    def _sync_live_log_disclosure_layout(self, *, expanded: bool) -> None:
        left_layout = getattr(self, "_left_pane_layout", None)
        if left_layout is not None:
            left_layout.setStretch(2, 3 if expanded else 1)
            left_layout.setStretch(3, 2 if expanded else 0)
            left_layout.invalidate()
            left_layout.activate()
        log_group = getattr(self, "_log_group", None)
        if log_group is not None:
            if expanded:
                log_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
                log_group.setMaximumHeight(16777215)
            else:
                log_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
                log_group.setMaximumHeight(120)

    def _on_live_log_disclosure_toggled(self, expanded: bool) -> None:
        if hasattr(self, "_live_log_content_container"):
            self._live_log_content_container.setVisible(expanded)
        if hasattr(self, "_live_log_collapsed_hint"):
            self._live_log_collapsed_hint.setVisible(not expanded)
        if hasattr(self, "_live_log_disclosure_btn"):
            self._live_log_disclosure_btn.setArrowType(
                Qt.DownArrow if expanded else Qt.RightArrow
            )
        self._sync_live_log_disclosure_layout(expanded=expanded)

    def _refresh_results_workspace_summary(self) -> None:
        if not hasattr(self, "_results_summary_title_label"):
            return

        run_dir = (self._current_run_dir or "").strip()
        run_name = os.path.basename(run_dir) if run_dir else ""
        run_display = run_name or "(none)"
        compact_text = ""
        show_compact = False
        show_hint = True
        show_details = True
        summary_max_height = 116

        has_loaded_results = bool(
            hasattr(self, "_report_viewer") and self._report_viewer.has_loaded_results()
        )
        if has_loaded_results and run_dir:
            self._results_summary_title_label.setText("Completed results loaded.")
            state_text = "Complete-state workspace"
            if "[PREVIEW]" in self.windowTitle():
                state_text += " [PREVIEW]"
            if self._is_any_tuning_subsection_expanded():
                state_text += " (Post-Run Tuning focused)"
            hint_text = "Use Analysis Outputs to inspect deliverables."
            compact_text = f"Run: {run_display} | {state_text}"
            show_compact = True
            show_hint = False
            show_details = False
            summary_max_height = 68
        elif self._ui_state in (RunnerState.RUNNING, RunnerState.VALIDATING) or self._runner.is_running():
            self._results_summary_title_label.setText("Run in progress.")
            state_text = (
                "Validating inputs"
                if self._ui_state == RunnerState.VALIDATING
                else "Pipeline running"
            )
            hint_text = "Monitor progress in the status strip and live log while this run is active."
        else:
            self._results_summary_title_label.setText("No completed run loaded.")
            state_text = "Idle"
            hint_text = "Run the pipeline or open completed results."

        self._results_summary_run_value.setText(run_display)
        self._results_summary_run_value.setToolTip(run_dir or "No run directory selected.")
        self._results_summary_state_value.setText(state_text)
        self._results_summary_compact_label.setText(compact_text)
        self._results_summary_compact_label.setToolTip(run_dir or state_text)
        self._results_summary_compact_label.setVisible(show_compact)
        self._results_summary_hint_label.setText(hint_text)
        self._results_summary_hint_label.setVisible(show_hint)
        self._results_summary_details_widget.setVisible(show_details)
        self._results_summary_group.setMaximumHeight(summary_max_height)
        if hasattr(self, "_results_views_hint_label"):
            self._results_views_hint_label.setVisible(not has_loaded_results)

    def _results_idle_placeholder_text(self) -> str:
        return (
            "Results workspace\n"
            "No completed run is loaded yet.\n"
            "Run the pipeline or open a completed run folder.\n"
            "Results and plots appear here after completion."
        )

    def _results_running_placeholder_text(self, message: str) -> str:
        return (
            "Results workspace\n"
            f"{message}\n"
            "Results and plots will appear here after completion."
        )

    def _apply_results_idle_placeholder(self) -> None:
        self._report_viewer.clear()
        if hasattr(self._report_viewer, "_set_status_message"):
            self._report_viewer._set_status_message(
                self._results_idle_placeholder_text(),
                level="idle",
            )
        self._refresh_results_workspace_summary()

    def _apply_shell_chrome_styles(self) -> None:
        """Shell-level framing and spacing styles (workflow/header modernization only)."""
        self.setStyleSheet(
            """
            QWidget#statusHeaderCard {
                border: 1px solid palette(mid);
                border-radius: 10px;
            }
            QScrollArea#workflowControlsScroll {
                border: none;
            }
            QGroupBox[workflowSection="true"] {
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox[workflowSection="true"]::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QGroupBox#resultsPaneShell {
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox#resultsPaneShell::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QToolButton#formRowHelpIcon {
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QToolButton#formRowHelpIcon:hover {
                background: palette(alternate-base);
                border-radius: 7px;
            }
            """
        )

    def _configure_sidebar_form_layout(self, form: QFormLayout) -> None:
        """Use wrapping-friendly form behavior for the fixed-width workflow pane."""
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def _status_color_for_severity(self, severity: str) -> QColor:
        """Palette-aware color mapping for true status labels."""
        pal = self.palette()
        text = pal.color(QPalette.WindowText)
        if severity in {"neutral", "info"}:
            return text
        if severity == "ready":
            link = pal.color(QPalette.Link)
            return link if link.isValid() else text
        bg_is_light = pal.color(QPalette.Window).lightness() >= 128
        if severity == "warn":
            return QColor.fromHsv(36, 180 if bg_is_light else 140, 120 if bg_is_light else 232)
        if severity == "error":
            return QColor.fromHsv(0, 210 if bg_is_light else 150, 120 if bg_is_light else 240)
        return text

    def _set_status_label_style(self, label: QLabel, severity: str) -> None:
        """Centralized severity styling for status-bearing labels."""
        if label is None:
            return
        normalized = severity if severity in {"neutral", "info", "ready", "warn", "error"} else "neutral"
        label.setProperty("statusSeverity", normalized)
        is_emphasis = normalized in {"ready", "warn", "error"}
        label.setStyleSheet(f"font-size: 11px; font-weight: {'600' if is_emphasis else '400'};")
        pal = label.palette()
        pal.setColor(QPalette.WindowText, self._status_color_for_severity(normalized))
        label.setPalette(pal)
        label.update()

    # ==================================================================
    # Post-run tuning workspace (Patch D)
    # ==================================================================

    def _is_downstream_tuning_expanded(self) -> bool:
        return bool(
            hasattr(self, "_tuning_disclosure_btn") and self._tuning_disclosure_btn.isChecked()
        )

    def _is_correction_tuning_expanded(self) -> bool:
        return bool(
            hasattr(self, "_correction_tuning_disclosure_btn")
            and self._correction_tuning_disclosure_btn.isChecked()
        )

    def _is_any_tuning_subsection_expanded(self) -> bool:
        return self._is_downstream_tuning_expanded() or self._is_correction_tuning_expanded()

    def _sync_tuning_status_visibility(self) -> None:
        expanded = self._is_downstream_tuning_expanded()
        if hasattr(self, "_tuning_collapsed_status_label"):
            self._tuning_collapsed_status_label.setVisible(not expanded)
        if hasattr(self, "_tuning_availability_label"):
            self._tuning_availability_label.setVisible(expanded)

    def _sync_correction_tuning_status_visibility(self) -> None:
        expanded = self._is_correction_tuning_expanded()
        if hasattr(self, "_correction_tuning_collapsed_status_label"):
            self._correction_tuning_collapsed_status_label.setVisible(not expanded)
        if hasattr(self, "_correction_tuning_availability_label"):
            self._correction_tuning_availability_label.setVisible(expanded)

    def _update_results_pane_mode_for_tuning(self) -> None:
        if not hasattr(self, "_report_viewer") or not hasattr(self, "_tuning_group"):
            return
        expanded_tuning = bool(
            self._is_complete_workspace_active
            and not self._tuning_group.isHidden()
            and self._is_any_tuning_subsection_expanded()
        )
        collapsed_tuning_max_height = 188
        if hasattr(self, "_results_views_group"):
            self._results_views_group.setVisible(not expanded_tuning)
        self._report_viewer.setVisible(not expanded_tuning)
        if expanded_tuning:
            self._tuning_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._tuning_group.setMaximumHeight(16777215)
        else:
            self._tuning_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            self._tuning_group.setMaximumHeight(collapsed_tuning_max_height)
        if hasattr(self, "_results_layout"):
            self._results_layout.invalidate()
            self._results_layout.activate()
        self._refresh_results_workspace_summary()

    def _on_tuning_disclosure_toggled(self, expanded: bool) -> None:
        if expanded and hasattr(self, "_report_viewer"):
            selected_region = self._report_viewer.selected_region().strip()
            if selected_region and self._tuning_roi_combo.findText(selected_region) >= 0:
                self._tuning_roi_combo.setCurrentText(selected_region)
        self._tuning_content.setVisible(expanded)
        self._sync_tuning_status_visibility()
        self._tuning_disclosure_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._update_results_pane_mode_for_tuning()
        QTimer.singleShot(0, self._render_tuning_overlay)

    def _set_tuning_disclosure_expanded(self, expanded: bool) -> None:
        if not hasattr(self, "_tuning_disclosure_btn"):
            return
        self._tuning_disclosure_btn.blockSignals(True)
        self._tuning_disclosure_btn.setChecked(expanded)
        self._tuning_disclosure_btn.blockSignals(False)
        self._on_tuning_disclosure_toggled(expanded)

    def _set_tuning_collapsed_status(self, message: str, *, ready: bool) -> None:
        self._tuning_collapsed_status_label.setText(message)
        self._set_status_label_style(
            self._tuning_collapsed_status_label,
            "ready" if ready else "warn",
        )

    def _set_tuning_workspace_unavailable(self, reason: str) -> None:
        self._tuning_workspace_available = False
        self._tuning_controls_container.setEnabled(False)
        self._run_tuning_btn.setEnabled(False)
        self._open_tuning_dir_btn.setEnabled(False)
        self._apply_tuning_btn.setEnabled(False)
        if hasattr(self, "_save_tuned_config_btn"):
            self._save_tuned_config_btn.setEnabled(False)
        self._tuning_availability_label.setText(reason)
        self._set_status_label_style(self._tuning_availability_label, "warn")
        self._set_tuning_collapsed_status(reason, ready=False)
        if hasattr(self, "_tuning_applyback_status_label"):
            self._tuning_applyback_status_label.setText("Apply-back status: not applied.")
        self._sync_tuning_status_visibility()
        self._update_results_pane_mode_for_tuning()

    def _set_tuning_workspace_available(self, message: str) -> None:
        self._tuning_workspace_available = True
        self._tuning_controls_container.setEnabled(True)
        self._run_tuning_btn.setEnabled(True)
        self._open_tuning_dir_btn.setEnabled(bool(self._tuning_last_result))
        self._apply_tuning_btn.setEnabled(True)
        if hasattr(self, "_save_tuned_config_btn"):
            self._save_tuned_config_btn.setEnabled(True)
        self._tuning_availability_label.setText(message)
        self._set_status_label_style(self._tuning_availability_label, "ready")
        self._set_tuning_collapsed_status(message, ready=True)
        self._refresh_tuning_feedback_summary()
        self._sync_tuning_status_visibility()
        self._update_results_pane_mode_for_tuning()

    def _current_phasic_out_dir(self) -> str:
        return os.path.join(self._current_run_dir, "_analysis", "phasic_out")

    def _current_phasic_cache_path(self) -> str:
        return os.path.join(self._current_phasic_out_dir(), "phasic_trace_cache.h5")

    def _current_phasic_config_path(self) -> str:
        return os.path.join(self._current_phasic_out_dir(), "config_used.yaml")

    def _set_tuning_overlay_message(self, text: str) -> None:
        self._tuning_active_overlay_path = ""
        self._tuning_active_overlay_pixmap = QPixmap()
        self._tuning_last_loaded_overlay_sha256 = ""
        self._tuning_last_loaded_overlay_size = 0
        if self._tuning_overlay_interaction is not None:
            self._tuning_overlay_interaction.clear(
                text,
                fallback_width=640,
                fallback_height=300,
            )
            self._tuning_overlay_zoom_mode = self._tuning_overlay_interaction.zoom_mode
        else:
            self._set_tuning_overlay_zoom_mode(False)
            self._tuning_overlay_label.setPixmap(QPixmap())
            self._tuning_overlay_label.setText(text)
            viewport = self._tuning_overlay_scroll.viewport().size()
            if viewport.width() < 10 or viewport.height() < 10:
                viewport = QSize(640, 300)
            target = QSize(max(10, viewport.width() - 8), max(10, viewport.height() - 8))
            self._tuning_overlay_label.resize(target)
        self._tuning_overlay_title.setText("No tuning overlay loaded.")

    def _render_tuning_overlay(self) -> None:
        """Render tuning overlay with shared interactive zoom/pan behavior."""
        if self._tuning_active_overlay_pixmap.isNull():
            return
        if self._tuning_overlay_interaction is not None:
            self._tuning_overlay_interaction.set_pixmap(
                self._tuning_active_overlay_pixmap,
                reset_zoom=False,
            )
            self._tuning_overlay_zoom_mode = self._tuning_overlay_interaction.zoom_mode

    def _set_tuning_overlay_image(self, image_path: str) -> None:
        if not image_path or not os.path.isfile(image_path):
            self._set_tuning_overlay_message("Tuning overlay image is missing.")
            return
        try:
            with open(image_path, "rb") as f:
                raw = f.read()
        except OSError:
            self._set_tuning_overlay_message("Unable to read tuning overlay image.")
            return
        self._tuning_last_loaded_overlay_size = int(len(raw))
        self._tuning_last_loaded_overlay_sha256 = _sha256_bytes(raw)
        pix = QPixmap()
        if not pix.loadFromData(raw):
            self._set_tuning_overlay_message("Unable to render tuning overlay image.")
            return
        if pix.isNull():
            self._set_tuning_overlay_message("Unable to render tuning overlay image.")
            return
        self._tuning_active_overlay_path = image_path
        self._tuning_active_overlay_pixmap = pix
        self._tuning_overlay_title.setText(os.path.basename(image_path))
        if self._tuning_overlay_interaction is not None:
            self._tuning_overlay_interaction.set_pixmap(pix, reset_zoom=True)
            self._tuning_overlay_zoom_mode = self._tuning_overlay_interaction.zoom_mode
        else:
            self._set_tuning_overlay_zoom_mode(False)
            self._render_tuning_overlay()

    def _on_tuning_overlay_clicked(self) -> None:
        if self._tuning_active_overlay_pixmap.isNull():
            return
        if self._tuning_overlay_interaction is not None:
            self._tuning_overlay_interaction.toggle_zoom()
            self._tuning_overlay_zoom_mode = self._tuning_overlay_interaction.zoom_mode
        else:
            self._set_tuning_overlay_zoom_mode(not self._tuning_overlay_zoom_mode)
            self._render_tuning_overlay()

    def _set_tuning_overlay_zoom_mode(self, enabled: bool) -> None:
        if self._tuning_overlay_interaction is not None:
            self._tuning_overlay_interaction.set_zoom_mode(bool(enabled))
            self._tuning_overlay_zoom_mode = self._tuning_overlay_interaction.zoom_mode
            return
        self._tuning_overlay_zoom_mode = bool(enabled)
        if hasattr(self, "_tuning_overlay_zoom_hint_label"):
            self._tuning_overlay_zoom_hint_label.setText(
                "Scroll wheel to zoom in/out. Drag to pan. Click image to return to fit."
                if self._tuning_overlay_zoom_mode
                else "Scroll wheel to zoom. Click image to toggle fit/full size."
            )

    def _on_tuning_overlay_zoom_mode_changed(self, enabled: bool) -> None:
        self._tuning_overlay_zoom_mode = bool(enabled)

    def _write_tuning_display_debug_record(
        self,
        *,
        result: dict,
        overrides: dict,
        previous_overlay_path: str,
    ) -> None:
        if not self._retune_preview_debug_enabled:
            return
        retune_dir = str(result.get("retune_dir", "")).strip()
        if not retune_dir or not os.path.isdir(retune_dir):
            return
        shown = self._tuning_overlay_label.pixmap()
        shown_hash = _pixmap_sha256_png(shown) if shown is not None else ""
        active_hash = _pixmap_sha256_png(self._tuning_active_overlay_pixmap)
        overlay_file_hash = ""
        if self._tuning_active_overlay_path and os.path.isfile(self._tuning_active_overlay_path):
            try:
                with open(self._tuning_active_overlay_path, "rb") as f:
                    overlay_file_hash = _sha256_bytes(f.read())
            except OSError:
                overlay_file_hash = ""
        record = {
            "schema_version": 1,
            "debug_kind": "post_run_tuning_display_overlay",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "retune_dir": retune_dir,
            "run_dir": str(self._current_run_dir),
            "preview_slot_id": "post_run_tuning_overlay",
            "roi": str(result.get("selected_roi", self._tuning_roi_combo.currentText().strip())),
            "chunk_id": int(result.get("inspection_chunk_id", self._tuning_chunk_combo.currentData() or -1)),
            "peak_pre_filter": str(overrides.get("peak_pre_filter", "")),
            "overlay_loaded_path": str(self._tuning_active_overlay_path),
            "overlay_loaded_basename": os.path.basename(str(self._tuning_active_overlay_path)),
            "overlay_previous_path": str(previous_overlay_path),
            "same_path_reused": bool(previous_overlay_path and previous_overlay_path == self._tuning_active_overlay_path),
            "overlay_loaded_bytes_sha256": str(self._tuning_last_loaded_overlay_sha256),
            "overlay_loaded_bytes_size": int(self._tuning_last_loaded_overlay_size),
            "overlay_file_sha256": str(overlay_file_hash),
            "active_pixmap_sha256_png": str(active_hash),
            "displayed_pixmap_sha256_png": str(shown_hash),
        }
        debug_path = os.path.join(retune_dir, "retune_preview_debug_display.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        print(
            "RETUNE_DEBUG DISPLAY "
            f"chunk={record['chunk_id']} "
            f"prefilter={record['peak_pre_filter']} "
            f"loaded_hash={record['overlay_loaded_bytes_sha256']} "
            f"display_hash={record['displayed_pixmap_sha256_png']}"
        )

    def _load_tuning_base_config(self) -> Config:
        cfg_path = self._current_phasic_config_path()
        if os.path.isfile(cfg_path):
            try:
                return Config.from_yaml(cfg_path)
            except Exception:
                pass
        return self._default_cfg

    def _roi_chunk_ids_map(self, cache_path: str) -> dict[str, list[int]]:
        """
        Return ROI->chunk_id map using ROI-group membership, not global chunk ids.
        """
        roi_chunk_map: dict[str, list[int]] = {}
        with open_phasic_cache(cache_path) as cache:
            roi_group = cache.get("roi")
            if roi_group is None:
                return roi_chunk_map
            try:
                meta_chunk_ids = [int(cid) for cid in cache["meta"]["chunk_ids"]]
            except Exception:
                meta_chunk_ids = []
            for roi_name in roi_group.keys():
                grp = roi_group.get(str(roi_name))
                if grp is None:
                    continue
                chunk_ids: list[int] = []
                if meta_chunk_ids:
                    for cid in meta_chunk_ids:
                        if f"chunk_{cid}" in grp:
                            chunk_ids.append(cid)
                if not chunk_ids:
                    for key in grp.keys():
                        if not str(key).startswith("chunk_"):
                            continue
                        try:
                            cid = int(str(key).split("_", 1)[1])
                        except (ValueError, IndexError):
                            continue
                        chunk_ids.append(cid)
                if chunk_ids:
                    roi_chunk_map[str(roi_name)] = sorted(set(chunk_ids))
        return roi_chunk_map

    def _chunk_ids_for_roi(self, roi: str) -> list[int]:
        roi = (roi or "").strip()
        if not roi:
            return []
        if self._roi_chunk_ids_cache:
            if roi in self._roi_chunk_ids_cache:
                return list(self._roi_chunk_ids_cache[roi])
            roi_l = roi.lower()
            for key, ids in self._roi_chunk_ids_cache.items():
                if str(key).lower() == roi_l:
                    return list(ids)
            return []
        cache_path = self._current_phasic_cache_path()
        if not os.path.isfile(cache_path):
            return []
        try:
            self._roi_chunk_ids_cache = self._roi_chunk_ids_map(cache_path)
        except Exception:
            return []
        return list(self._roi_chunk_ids_cache.get(roi, []))

    def _populate_tuning_chunk_choices(self, roi: str) -> None:
        self._tuning_chunk_combo.blockSignals(True)
        self._tuning_chunk_combo.clear()
        chunk_ids = self._chunk_ids_for_roi(roi)
        for cid in chunk_ids:
            self._tuning_chunk_combo.addItem(str(cid), cid)
        self._tuning_chunk_combo.blockSignals(False)

    def _populate_tuning_roi_choices(
        self,
        *,
        cache_rois_with_chunks: list[str],
        prefer_roi: str | None = None,
    ) -> None:
        cache_rois = list(cache_rois_with_chunks)
        viewer_rois = self._report_viewer.available_regions()
        if viewer_rois:
            merged = [r for r in viewer_rois if r in cache_rois]
            if not merged:
                merged = cache_rois
        else:
            merged = cache_rois

        current = self._tuning_roi_combo.currentText().strip()
        target = (prefer_roi or current or "").strip()

        self._tuning_roi_combo.blockSignals(True)
        self._tuning_roi_combo.clear()
        self._tuning_roi_combo.addItems(merged)
        self._tuning_roi_combo.blockSignals(False)

        if merged:
            if target in merged:
                self._tuning_roi_combo.setCurrentText(target)
            else:
                self._tuning_roi_combo.setCurrentIndex(0)
        self._populate_tuning_chunk_choices(self._tuning_roi_combo.currentText().strip())

    def _apply_tuning_defaults_from_config(self, cfg: Config) -> None:
        if self._tuning_event_signal_combo.findText(str(cfg.event_signal)) >= 0:
            self._tuning_event_signal_combo.setCurrentText(str(cfg.event_signal))
        if self._tuning_peak_method_combo.findText(str(cfg.peak_threshold_method)) >= 0:
            self._tuning_peak_method_combo.setCurrentText(str(cfg.peak_threshold_method))
        self._tuning_peak_k_spin.setValue(float(cfg.peak_threshold_k))
        self._tuning_peak_pct_spin.setValue(float(cfg.peak_threshold_percentile))
        self._tuning_peak_abs_spin.setValue(float(getattr(cfg, "peak_threshold_abs", 0.0)))
        self._tuning_peak_dist_spin.setValue(float(cfg.peak_min_distance_sec))
        self._tuning_peak_prominence_k_spin.setValue(float(getattr(cfg, "peak_min_prominence_k", 0.0)))
        self._tuning_peak_width_sec_spin.setValue(float(getattr(cfg, "peak_min_width_sec", 0.0)))
        pre_filter = normalize_retune_peak_pre_filter(str(getattr(cfg, "peak_pre_filter", "none")))
        if self._tuning_peak_pre_filter_combo.findText(pre_filter) >= 0:
            self._tuning_peak_pre_filter_combo.setCurrentText(pre_filter)
        auc_base = str(cfg.event_auc_baseline)
        if self._tuning_event_auc_combo.findText(auc_base) >= 0:
            self._tuning_event_auc_combo.setCurrentText(auc_base)
        self._on_tuning_peak_method_changed()

    def _on_tuning_peak_method_changed(self) -> None:
        method = self._tuning_peak_method_combo.currentText().strip()
        show_k = peak_threshold_method_requires_k(method)
        show_pct = peak_threshold_method_requires_percentile(method)
        show_abs = peak_threshold_method_requires_abs(method)
        self._tuning_peak_k_label.setVisible(show_k)
        self._tuning_peak_k_spin.setVisible(show_k)
        self._tuning_peak_pct_label.setVisible(show_pct)
        self._tuning_peak_pct_spin.setVisible(show_pct)
        self._tuning_peak_abs_label.setVisible(show_abs)
        self._tuning_peak_abs_spin.setVisible(show_abs)

    def _on_results_region_changed(self, region: str) -> None:
        if not self._is_complete_workspace_active or not region:
            self._refresh_results_workspace_summary()
            return
        if self._tuning_roi_combo.findText(region) >= 0:
            self._tuning_roi_combo.setCurrentText(region)
        if (
            hasattr(self, "_correction_tuning_roi_combo")
            and self._correction_tuning_roi_combo.findText(region) >= 0
        ):
            self._correction_tuning_roi_combo.setCurrentText(region)
        self._refresh_results_workspace_summary()

    def _on_tuning_roi_changed(self, _index: int) -> None:
        roi = self._tuning_roi_combo.currentText().strip()
        self._populate_tuning_chunk_choices(roi)

    def _refresh_tuning_workspace_availability(self) -> None:
        if not hasattr(self, "_tuning_group"):
            return
        self._roi_chunk_ids_cache = {}

        # Keep complete-state flag synchronized with a loaded results workspace.
        # This prevents stale "no completed run loaded" tuning gating when the
        # report viewer has already loaded a completed run directory.
        if (
            not self._is_complete_workspace_active
            and hasattr(self, "_report_viewer")
            and self._report_viewer.has_loaded_results()
            and bool(self._current_run_dir and os.path.isdir(self._current_run_dir))
        ):
            self._enter_complete_state_workspace()
            return

        self._tuning_group.setVisible(bool(self._is_complete_workspace_active))
        if not self._is_complete_workspace_active:
            self._set_tuning_workspace_unavailable(
                "Tuning is available only after a successful completed run is loaded."
            )
            self._set_correction_tuning_workspace_unavailable(
                "Correction retune is available only after a successful completed run is loaded."
            )
            return

        run_dir = self._current_run_dir
        if not run_dir or not os.path.isdir(run_dir):
            self._set_tuning_workspace_unavailable("No completed run directory is active.")
            self._set_correction_tuning_workspace_unavailable(
                "No completed run directory is active."
            )
            return
        is_successful_complete, evidence = is_successful_completed_run_dir(run_dir)
        if not is_successful_complete:
            reason = (
                "Tuning unavailable: selected run directory is not confirmed as a successful completed run."
            )
            self._set_tuning_workspace_unavailable(reason)
            self._set_correction_tuning_workspace_unavailable(
                f"Correction retune unavailable: selected run directory is not confirmed as a successful completed run. ({evidence})"
            )
            return

        phasic_out_dir = self._current_phasic_out_dir()
        if not os.path.isdir(phasic_out_dir):
            self._set_tuning_workspace_unavailable(
                "Tuning unavailable: missing phasic output directory at _analysis/phasic_out."
            )
            self._set_correction_tuning_workspace_unavailable(
                "Correction retune unavailable: missing phasic output directory at _analysis/phasic_out."
            )
            return

        cache_path = self._current_phasic_cache_path()
        if not os.path.isfile(cache_path):
            self._set_tuning_workspace_unavailable(
                "Tuning unavailable: phasic cache is missing for this completed run."
            )
            self._set_correction_tuning_workspace_unavailable(
                "Correction retune unavailable: phasic cache is missing for this completed run."
            )
            return

        cfg_path = self._current_phasic_config_path()
        if not os.path.isfile(cfg_path):
            self._set_tuning_workspace_unavailable(
                "Tuning unavailable: missing config snapshot _analysis/phasic_out/config_used.yaml."
            )
            self._set_correction_tuning_workspace_unavailable(
                "Correction retune unavailable: missing config snapshot _analysis/phasic_out/config_used.yaml."
            )
            return

        try:
            roi_chunk_map = self._roi_chunk_ids_map(cache_path)
        except Exception as exc:
            self._set_tuning_workspace_unavailable(
                f"Tuning unavailable: unable to read ROI/session targets from phasic cache ({exc})."
            )
            self._set_correction_tuning_workspace_unavailable(
                f"Correction retune unavailable: unable to read ROI/session targets from phasic cache ({exc})."
            )
            return
        self._roi_chunk_ids_cache = roi_chunk_map
        valid_rois = sorted(roi_chunk_map.keys(), key=lambda s: s.lower())
        if not valid_rois:
            self._set_tuning_workspace_unavailable(
                "Tuning unavailable: no valid ROI groups with session data found in phasic cache."
            )
            self._set_correction_tuning_workspace_unavailable(
                "Correction retune unavailable: no valid ROI groups with session data found in phasic cache."
            )
            return

        prefer_roi = self._report_viewer.selected_region()
        self._populate_tuning_roi_choices(
            cache_rois_with_chunks=valid_rois,
            prefer_roi=prefer_roi,
        )
        selected_roi = self._tuning_roi_combo.currentText().strip()
        if not selected_roi:
            self._set_tuning_workspace_unavailable(
                "Tuning unavailable: no valid ROI target is available for this run."
            )
            self._set_correction_tuning_workspace_unavailable(
                "Correction retune unavailable: no valid ROI target is available for this run."
            )
            return
        if self._tuning_chunk_combo.count() == 0:
            self._set_tuning_workspace_unavailable(
                f"Tuning unavailable: selected ROI '{selected_roi}' has no available sessions in phasic cache."
            )
            self._set_correction_tuning_workspace_unavailable(
                f"Correction retune unavailable: selected ROI '{selected_roi}' has no available sessions in phasic cache."
            )
            return

        self._apply_tuning_defaults_from_config(self._load_tuning_base_config())
        self._set_tuning_workspace_available(
            "Ready: downstream-only tuning from cache. Correction-sensitive parameters are not part of this workspace."
        )

        self._populate_correction_tuning_roi_choices(
            cache_rois_with_chunks=valid_rois,
            prefer_roi=prefer_roi,
        )
        selected_corr_roi = self._correction_tuning_roi_combo.currentText().strip()
        if not selected_corr_roi:
            self._set_correction_tuning_workspace_unavailable(
                "Correction retune unavailable: no valid ROI target is available for this run."
            )
            return
        if self._correction_tuning_chunk_combo.count() == 0:
            self._set_correction_tuning_workspace_unavailable(
                f"Correction retune unavailable: selected ROI '{selected_corr_roi}' has no available sessions in phasic cache."
            )
            return

        self._apply_correction_tuning_defaults_from_config(self._load_tuning_base_config())
        self._set_correction_tuning_workspace_available(
            "Ready: correction retune recomputes the selected ROI across all available sessions. "
            "The preview session is used only for inspection."
        )

    def _collect_tuning_overrides(self) -> dict:
        method = self._tuning_peak_method_combo.currentText().strip()
        overrides = {
            "event_signal": self._tuning_event_signal_combo.currentText().strip(),
            "peak_threshold_method": method,
            "peak_min_distance_sec": float(self._tuning_peak_dist_spin.value()),
            "peak_min_prominence_k": float(self._tuning_peak_prominence_k_spin.value()),
            "peak_min_width_sec": float(self._tuning_peak_width_sec_spin.value()),
            "peak_pre_filter": normalize_retune_peak_pre_filter(
                self._tuning_peak_pre_filter_combo.currentText().strip()
            ),
            "event_auc_baseline": self._tuning_event_auc_combo.currentText().strip(),
        }
        if peak_threshold_method_requires_k(method):
            overrides["peak_threshold_k"] = float(self._tuning_peak_k_spin.value())
        if peak_threshold_method_requires_percentile(method):
            overrides["peak_threshold_percentile"] = float(self._tuning_peak_pct_spin.value())
        if peak_threshold_method_requires_abs(method):
            overrides["peak_threshold_abs"] = float(self._tuning_peak_abs_spin.value())
        return overrides

    @staticmethod
    def _floats_differ(a: float, b: float, *, tol: float = 1e-9) -> bool:
        return abs(float(a) - float(b)) > tol

    def _summarize_tuning_changes_from_baseline(self, overrides: dict, baseline_cfg: Config) -> list[str]:
        changed: list[str] = []
        if str(overrides.get("event_signal", "")) != str(baseline_cfg.event_signal):
            changed.append("event signal")
        if str(overrides.get("peak_threshold_method", "")) != str(baseline_cfg.peak_threshold_method):
            changed.append("threshold method")
        if self._floats_differ(
            float(overrides.get("peak_min_distance_sec", 0.0)),
            float(baseline_cfg.peak_min_distance_sec),
            tol=1e-6,
        ):
            changed.append("min distance")
        if self._floats_differ(
            float(overrides.get("peak_min_prominence_k", 0.0)),
            float(getattr(baseline_cfg, "peak_min_prominence_k", 0.0)),
            tol=1e-6,
        ):
            changed.append("min prominence")
        if self._floats_differ(
            float(overrides.get("peak_min_width_sec", 0.0)),
            float(getattr(baseline_cfg, "peak_min_width_sec", 0.0)),
            tol=1e-6,
        ):
            changed.append("min width")
        if self._floats_differ(
            float(overrides.get("peak_threshold_k", getattr(baseline_cfg, "peak_threshold_k", 0.0))),
            float(getattr(baseline_cfg, "peak_threshold_k", 0.0)),
            tol=1e-6,
        ):
            changed.append("threshold K")
        if self._floats_differ(
            float(
                overrides.get(
                    "peak_threshold_percentile",
                    getattr(baseline_cfg, "peak_threshold_percentile", 0.0),
                )
            ),
            float(getattr(baseline_cfg, "peak_threshold_percentile", 0.0)),
            tol=1e-6,
        ):
            changed.append("threshold percentile")
        if self._floats_differ(
            float(
                overrides.get("peak_threshold_abs", getattr(baseline_cfg, "peak_threshold_abs", 0.0))
            ),
            float(getattr(baseline_cfg, "peak_threshold_abs", 0.0)),
            tol=1e-6,
        ):
            changed.append("threshold absolute")
        base_pre = normalize_retune_peak_pre_filter(str(getattr(baseline_cfg, "peak_pre_filter", "none")))
        if normalize_retune_peak_pre_filter(str(overrides.get("peak_pre_filter", "none"))) != base_pre:
            changed.append("pre-filter mode")
        if str(overrides.get("event_auc_baseline", "")) != str(baseline_cfg.event_auc_baseline):
            changed.append("AUC baseline")
        return changed

    def _refresh_tuning_feedback_summary(self) -> None:
        if not isinstance(self._tuning_last_result, dict):
            self._tuning_summary_label.setText("No tuning result yet.")
            if hasattr(self, "_tuning_applyback_status_label"):
                self._tuning_applyback_status_label.setText("Apply-back status: not applied.")
            return

        result = self._tuning_last_result
        roi = result.get("selected_roi", self._tuning_roi_combo.currentText().strip() or "(unknown)")
        chunk = result.get("inspection_chunk_id", self._tuning_chunk_combo.currentData() or "(unknown)")
        event_signal = result.get("event_signal_used", "(unknown)")
        retune_dir = result.get("retune_dir", "(unknown)")
        changed = self._tuning_last_changed_fields
        changed_text = ", ".join(changed[:5]) if changed else "none detected"
        if len(changed) > 5:
            changed_text += f", +{len(changed) - 5} more"
        apply_text = (
            f"applied ({self._tuning_applyback_timestamp})"
            if self._tuning_applyback_applied else "not applied"
        )
        lines = [
            f"ROI: {roi}",
            f"Preview session: {chunk}",
            f"Event signal: {event_signal}",
            f"Changed vs baseline: {changed_text}",
            f"Apply-back to next run: {apply_text}",
            f"Retune output: {retune_dir}",
        ]
        self._tuning_summary_label.setText("\n".join(lines))
        if hasattr(self, "_tuning_applyback_status_label"):
            if self._tuning_applyback_applied:
                self._tuning_applyback_status_label.setText(
                    "Apply-back status: applied to setup controls. Completed run is unchanged; rerun to regenerate outputs."
                )
            else:
                self._tuning_applyback_status_label.setText(
                    "Apply-back status: not applied yet. Completed run remains unchanged."
                )

    def _on_run_tuning(self) -> None:
        if not self._tuning_workspace_available:
            QMessageBox.information(self, "Tuning Unavailable", self._tuning_availability_label.text())
            return

        roi = self._tuning_roi_combo.currentText().strip()
        if not roi:
            QMessageBox.warning(self, "Tuning Error", "Select an ROI before running tuning.")
            return
        if self._tuning_chunk_combo.currentIndex() < 0:
            QMessageBox.warning(
                self,
                "Tuning Error",
                "Select a preview session before running tuning.",
            )
            return
        chunk_id = int(self._tuning_chunk_combo.currentData())
        overrides = self._collect_tuning_overrides()

        self._run_tuning_btn.setEnabled(False)
        self._run_tuning_btn.setText("Running...")
        try:
            result = run_cache_downstream_retune(
                run_dir=self._current_run_dir,
                roi=roi,
                chunk_id=chunk_id,
                overrides=overrides,
                out_dir=None,
            )
        except Exception as exc:
            self._append_run_log(f"Tuning retune failed: {exc}")
            QMessageBox.critical(
                self,
                "Tuning Failed",
                f"Cache-driven downstream retune failed.\n\n{exc}",
            )
            return
        finally:
            self._run_tuning_btn.setEnabled(self._tuning_workspace_available)
            self._run_tuning_btn.setText("Run Tuning")

        self._tuning_last_result = result
        self._tuning_applyback_applied = False
        self._tuning_applyback_timestamp = ""
        self._open_tuning_dir_btn.setEnabled(True)
        artifacts = result.get("artifacts", {}) if isinstance(result, dict) else {}
        overlay_path = str(artifacts.get("retuned_overlay_png", "")).strip()
        previous_overlay_path = str(self._tuning_active_overlay_path)
        self._set_tuning_overlay_image(overlay_path)
        baseline_cfg = self._load_tuning_base_config()
        self._tuning_last_changed_fields = self._summarize_tuning_changes_from_baseline(
            overrides,
            baseline_cfg,
        )
        self._write_tuning_display_debug_record(
            result=result if isinstance(result, dict) else {},
            overrides=overrides,
            previous_overlay_path=previous_overlay_path,
        )
        self._refresh_tuning_feedback_summary()
        self._append_run_log(
            f"Tuning completed for ROI={result.get('selected_roi', roi)} preview_session={result.get('inspection_chunk_id', chunk_id)} "
            f"-> {result.get('retune_dir', '(unknown)')}"
        )
        QTimer.singleShot(0, self._finalize_tuning_result_layout)

    def _finalize_tuning_result_layout(self) -> None:
        """Finalize tuning-result geometry before fit-to-view overlay render."""
        self._tuning_group.updateGeometry()
        self._tuning_scroll_content.updateGeometry()
        self._tuning_scroll.viewport().updateGeometry()
        self._tuning_scroll.updateGeometry()
        self._render_tuning_overlay()

    def _on_open_tuning_output(self) -> None:
        if not isinstance(self._tuning_last_result, dict):
            QMessageBox.information(self, "No Tuning Output", "Run tuning first to create an output directory.")
            return
        retune_dir = str(self._tuning_last_result.get("retune_dir", "")).strip()
        if not retune_dir or not os.path.isdir(retune_dir):
            QMessageBox.information(self, "No Tuning Output", "Tuning output directory is not available.")
            return
        _open_folder(retune_dir)

    def _on_apply_tuning_values_to_run_settings(self) -> None:
        if not self._tuning_workspace_available:
            QMessageBox.information(self, "Tuning Unavailable", self._tuning_availability_label.text())
            return

        def _set_combo_if_allowed(combo: QComboBox, value: str) -> None:
            idx = combo.findText(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        def _fmt(v: float, decimals: int = 6) -> str:
            text = f"{float(v):.{decimals}f}"
            return text.rstrip("0").rstrip(".") if "." in text else text

        _set_combo_if_allowed(self._event_signal_combo, self._tuning_event_signal_combo.currentText().strip())

        method = self._tuning_peak_method_combo.currentText().strip()
        _set_combo_if_allowed(self._peak_method_combo, method)

        self._peak_k_edit.setText(_fmt(self._tuning_peak_k_spin.value(), decimals=6))
        self._peak_pct_edit.setText(_fmt(self._tuning_peak_pct_spin.value(), decimals=3))
        self._peak_abs_edit.setText(_fmt(self._tuning_peak_abs_spin.value(), decimals=6))
        self._peak_dist_edit.setText(_fmt(self._tuning_peak_dist_spin.value(), decimals=3))
        self._peak_min_prominence_k_edit.setText(
            _fmt(self._tuning_peak_prominence_k_spin.value(), decimals=6)
        )
        self._peak_min_width_sec_edit.setText(
            _fmt(self._tuning_peak_width_sec_spin.value(), decimals=3)
        )

        _set_combo_if_allowed(
            self._peak_pre_filter_combo,
            map_retune_peak_pre_filter_to_run_setting(
                self._tuning_peak_pre_filter_combo.currentText().strip()
            ),
        )
        _set_combo_if_allowed(self._event_auc_combo, self._tuning_event_auc_combo.currentText().strip())

        self._update_adv_ev_visibility()
        # Ensure validate->run reuse state is invalidated even if any set* call
        # is a no-op for the current value.
        self._on_config_changed()
        self._tuning_applyback_applied = True
        self._tuning_applyback_timestamp = datetime.now().strftime("%H:%M:%S")
        self._refresh_tuning_feedback_summary()
        self._append_run_log(
            "Applied downstream tuning values to next-run setup controls. "
            "Completed run outputs are unchanged until rerun."
        )

    def _on_correction_tuning_disclosure_toggled(self, expanded: bool) -> None:
        if expanded and hasattr(self, "_report_viewer"):
            selected_region = self._report_viewer.selected_region().strip()
            if (
                selected_region
                and self._correction_tuning_roi_combo.findText(selected_region) >= 0
            ):
                self._correction_tuning_roi_combo.setCurrentText(selected_region)
        self._correction_tuning_content.setVisible(expanded)
        self._sync_correction_tuning_status_visibility()
        self._correction_tuning_disclosure_btn.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        self._update_results_pane_mode_for_tuning()
        QTimer.singleShot(0, self._render_correction_tuning_overlay)

    def _set_correction_tuning_disclosure_expanded(self, expanded: bool) -> None:
        if not hasattr(self, "_correction_tuning_disclosure_btn"):
            return
        self._correction_tuning_disclosure_btn.blockSignals(True)
        self._correction_tuning_disclosure_btn.setChecked(expanded)
        self._correction_tuning_disclosure_btn.blockSignals(False)
        self._on_correction_tuning_disclosure_toggled(expanded)

    def _set_correction_tuning_collapsed_status(self, message: str, *, ready: bool) -> None:
        self._correction_tuning_collapsed_status_label.setText(message)
        self._set_status_label_style(
            self._correction_tuning_collapsed_status_label,
            "ready" if ready else "warn",
        )

    def _set_correction_tuning_workspace_unavailable(self, reason: str) -> None:
        self._correction_tuning_workspace_available = False
        self._correction_tuning_controls_container.setEnabled(False)
        self._run_correction_tuning_btn.setEnabled(False)
        self._open_correction_tuning_dir_btn.setEnabled(False)
        if hasattr(self, "_apply_correction_tuning_btn"):
            self._apply_correction_tuning_btn.setEnabled(False)
        if hasattr(self, "_save_correction_tuned_config_btn"):
            self._save_correction_tuned_config_btn.setEnabled(False)
        self._correction_tuning_availability_label.setText(reason)
        self._set_status_label_style(self._correction_tuning_availability_label, "warn")
        self._correction_tuning_applyback_applied = False
        self._correction_tuning_applyback_timestamp = ""
        self._refresh_correction_tuning_applyback_status()
        self._set_correction_tuning_collapsed_status(reason, ready=False)
        self._sync_correction_tuning_status_visibility()
        self._update_results_pane_mode_for_tuning()

    def _set_correction_tuning_workspace_available(self, message: str) -> None:
        self._correction_tuning_workspace_available = True
        self._correction_tuning_controls_container.setEnabled(True)
        self._run_correction_tuning_btn.setEnabled(True)
        self._open_correction_tuning_dir_btn.setEnabled(
            bool(self._correction_tuning_last_result)
        )
        if hasattr(self, "_apply_correction_tuning_btn"):
            self._apply_correction_tuning_btn.setEnabled(True)
        if hasattr(self, "_save_correction_tuned_config_btn"):
            self._save_correction_tuned_config_btn.setEnabled(True)
        self._correction_tuning_availability_label.setText(message)
        self._set_status_label_style(self._correction_tuning_availability_label, "ready")
        self._refresh_correction_tuning_applyback_status()
        self._set_correction_tuning_collapsed_status(message, ready=True)
        self._sync_correction_tuning_status_visibility()
        self._update_results_pane_mode_for_tuning()

    def _set_correction_tuning_overlay_message(self, text: str) -> None:
        self._correction_tuning_inspection_paths = []
        self._correction_tuning_inspection_panel_labels = []
        self._correction_tuning_inspection_index = 0
        self._correction_tuning_active_inspection_path = ""
        self._correction_tuning_active_inspection_pixmap = QPixmap()
        if self._correction_tuning_overlay_interaction is not None:
            self._correction_tuning_overlay_interaction.clear(
                text,
                fallback_width=640,
                fallback_height=300,
            )
            self._correction_tuning_zoom_mode = (
                self._correction_tuning_overlay_interaction.zoom_mode
            )
        else:
            self._set_correction_tuning_zoom_mode(False)
            self._correction_tuning_inspection_label.setPixmap(QPixmap())
            self._correction_tuning_inspection_label.setText(text)
            viewport = self._correction_tuning_inspection_scroll.viewport().size()
            if viewport.width() < 10 or viewport.height() < 10:
                viewport = QSize(640, 300)
            target = QSize(max(10, viewport.width() - 8), max(10, viewport.height() - 8))
            self._correction_tuning_inspection_label.resize(target)
        self._correction_tuning_inspection_title.setText(
            "No correction inspection artifact loaded."
        )
        self._refresh_correction_tuning_inspection_nav()

    def _render_correction_tuning_overlay(self) -> None:
        if self._correction_tuning_active_inspection_pixmap.isNull():
            return
        if self._correction_tuning_overlay_interaction is not None:
            self._correction_tuning_overlay_interaction.set_pixmap(
                self._correction_tuning_active_inspection_pixmap,
                reset_zoom=False,
            )
            self._correction_tuning_zoom_mode = (
                self._correction_tuning_overlay_interaction.zoom_mode
            )

    def _refresh_correction_tuning_inspection_nav(self) -> None:
        count = len(self._correction_tuning_inspection_paths)
        idx = int(self._correction_tuning_inspection_index)
        if hasattr(self, "_correction_tuning_prev_btn"):
            self._correction_tuning_prev_btn.setEnabled(count > 1)
        if hasattr(self, "_correction_tuning_next_btn"):
            self._correction_tuning_next_btn.setEnabled(count > 1)
        if hasattr(self, "_correction_tuning_inspection_counter_label"):
            if count <= 0:
                self._correction_tuning_inspection_counter_label.setText("")
            else:
                self._correction_tuning_inspection_counter_label.setText(
                    f"{idx + 1}/{count}"
                )

    def _panel_label_from_inspection_path(self, image_path: str) -> str:
        base = os.path.basename(str(image_path)).lower()
        if base.endswith("_raw.png"):
            return "Raw absolute sig/iso"
        if base.endswith("_centered.png"):
            return "Centered common-gain sig/iso"
        if base.endswith("_fit.png"):
            return "Dynamic fit"
        if base.endswith("_dff.png"):
            return "Final corrected dF/F"
        return "Correction inspection"

    def _correction_tuning_panel_priority(self, image_path: str, panel_label: str) -> int:
        """Preferred correction-retune carousel order: fit, dF/F, centered, raw."""
        base = os.path.basename(str(image_path)).lower()
        label = str(panel_label or "").strip().lower()
        if base.endswith("_fit.png") or "dynamic fit" in label:
            return 0
        if base.endswith("_dff.png") or "final corrected" in label:
            return 1
        if base.endswith("_centered.png") or "centered" in label:
            return 2
        if base.endswith("_raw.png") or "raw absolute" in label:
            return 3
        return 99

    def _set_correction_tuning_overlay_images(
        self,
        image_paths: list[str],
        panel_labels: list[str] | None = None,
    ) -> None:
        valid_paths = [str(p).strip() for p in image_paths if str(p).strip()]
        valid_paths = [p for p in valid_paths if os.path.isfile(p)]
        if not valid_paths:
            self._set_correction_tuning_overlay_message(
                "Correction inspection artifact is missing."
            )
            return
        self._correction_tuning_inspection_paths = valid_paths
        labels = list(panel_labels or [])
        if len(labels) != len(valid_paths):
            labels = [self._panel_label_from_inspection_path(p) for p in valid_paths]
        ordered = sorted(
            enumerate(zip(valid_paths, labels)),
            key=lambda item: (
                self._correction_tuning_panel_priority(item[1][0], item[1][1]),
                item[0],
            ),
        )
        valid_paths = [entry[1][0] for entry in ordered]
        labels = [entry[1][1] for entry in ordered]
        self._correction_tuning_inspection_paths = valid_paths
        self._correction_tuning_inspection_panel_labels = labels
        self._correction_tuning_inspection_index = 0
        self._show_correction_tuning_overlay_index(0)

    def _show_correction_tuning_overlay_index(self, index: int) -> None:
        if not self._correction_tuning_inspection_paths:
            self._set_correction_tuning_overlay_message(
                "Correction inspection artifact is missing."
            )
            return
        count = len(self._correction_tuning_inspection_paths)
        idx = int(index) % count
        image_path = self._correction_tuning_inspection_paths[idx]
        pix = QPixmap(image_path)
        if pix.isNull():
            self._set_correction_tuning_overlay_message(
                "Unable to render correction inspection artifact."
            )
            return
        self._correction_tuning_inspection_index = idx
        self._correction_tuning_active_inspection_path = image_path
        self._correction_tuning_active_inspection_pixmap = pix
        label = self._correction_tuning_inspection_panel_labels[idx]
        self._correction_tuning_inspection_title.setText(
            f"[{idx + 1}/{count}] {label} - {os.path.basename(image_path)}"
        )
        self._refresh_correction_tuning_inspection_nav()
        if self._correction_tuning_overlay_interaction is not None:
            self._correction_tuning_overlay_interaction.set_pixmap(pix, reset_zoom=True)
            self._correction_tuning_zoom_mode = (
                self._correction_tuning_overlay_interaction.zoom_mode
            )
        else:
            self._set_correction_tuning_zoom_mode(False)
            self._render_correction_tuning_overlay()

    def _set_correction_tuning_overlay_image(self, image_path: str) -> None:
        self._set_correction_tuning_overlay_images([image_path], panel_labels=None)

    def _on_correction_tuning_prev_image(self) -> None:
        if len(self._correction_tuning_inspection_paths) <= 1:
            return
        self._show_correction_tuning_overlay_index(
            self._correction_tuning_inspection_index - 1
        )

    def _on_correction_tuning_next_image(self) -> None:
        if len(self._correction_tuning_inspection_paths) <= 1:
            return
        self._show_correction_tuning_overlay_index(
            self._correction_tuning_inspection_index + 1
        )

    def _on_correction_tuning_inspection_clicked(self) -> None:
        if self._correction_tuning_active_inspection_pixmap.isNull():
            return
        if self._correction_tuning_overlay_interaction is not None:
            self._correction_tuning_overlay_interaction.toggle_zoom()
            self._correction_tuning_zoom_mode = (
                self._correction_tuning_overlay_interaction.zoom_mode
            )
        else:
            self._set_correction_tuning_zoom_mode(not self._correction_tuning_zoom_mode)
            self._render_correction_tuning_overlay()

    def _set_correction_tuning_zoom_mode(self, enabled: bool) -> None:
        if self._correction_tuning_overlay_interaction is not None:
            self._correction_tuning_overlay_interaction.set_zoom_mode(bool(enabled))
            self._correction_tuning_zoom_mode = (
                self._correction_tuning_overlay_interaction.zoom_mode
            )
            return
        self._correction_tuning_zoom_mode = bool(enabled)
        if hasattr(self, "_correction_tuning_zoom_hint_label"):
            self._correction_tuning_zoom_hint_label.setText(
                "Scroll wheel to zoom in/out. Drag to pan. Click image to return to fit."
                if self._correction_tuning_zoom_mode
                else "Scroll wheel to zoom. Click image to toggle fit/full size."
            )

    def _on_correction_tuning_zoom_mode_changed(self, enabled: bool) -> None:
        self._correction_tuning_zoom_mode = bool(enabled)

    def _populate_correction_tuning_chunk_choices(self, roi: str) -> None:
        self._correction_tuning_chunk_combo.blockSignals(True)
        self._correction_tuning_chunk_combo.clear()
        chunk_ids = self._chunk_ids_for_roi(roi)
        for cid in chunk_ids:
            self._correction_tuning_chunk_combo.addItem(str(cid), cid)
        self._correction_tuning_chunk_combo.blockSignals(False)

    def _populate_correction_tuning_roi_choices(
        self,
        *,
        cache_rois_with_chunks: list[str],
        prefer_roi: str | None = None,
    ) -> None:
        cache_rois = list(cache_rois_with_chunks)
        viewer_rois = self._report_viewer.available_regions()
        if viewer_rois:
            merged = [r for r in viewer_rois if r in cache_rois]
            if not merged:
                merged = cache_rois
        else:
            merged = cache_rois

        current = self._correction_tuning_roi_combo.currentText().strip()
        target = (prefer_roi or current or "").strip()

        self._correction_tuning_roi_combo.blockSignals(True)
        self._correction_tuning_roi_combo.clear()
        self._correction_tuning_roi_combo.addItems(merged)
        self._correction_tuning_roi_combo.blockSignals(False)

        if merged:
            if target in merged:
                self._correction_tuning_roi_combo.setCurrentText(target)
            else:
                self._correction_tuning_roi_combo.setCurrentIndex(0)
        self._populate_correction_tuning_chunk_choices(
            self._correction_tuning_roi_combo.currentText().strip()
        )

    def _apply_correction_tuning_defaults_from_config(self, cfg: Config) -> None:
        method = str(cfg.baseline_method)
        if self._correction_tuning_baseline_method_combo.findText(method) >= 0:
            self._correction_tuning_baseline_method_combo.setCurrentText(method)
        fit_mode = normalize_dynamic_fit_mode(
            str(getattr(cfg, "dynamic_fit_mode", "rolling_local_regression"))
        )
        idx_fit = self._correction_tuning_fit_mode_combo.findData(fit_mode)
        if idx_fit >= 0:
            self._correction_tuning_fit_mode_combo.setCurrentIndex(idx_fit)
        self._correction_tuning_baseline_pct_spin.setValue(float(cfg.baseline_percentile))
        self._correction_tuning_lowpass_spin.setValue(float(cfg.lowpass_hz))
        self._correction_tuning_baseline_subtract_cb.setChecked(
            bool(getattr(cfg, "baseline_subtract_before_fit", False))
        )
        self._correction_tuning_window_spin.setValue(float(cfg.window_sec))
        self._correction_tuning_min_samples_spin.setValue(int(cfg.min_samples_per_window))
        self._correction_tuning_robust_max_iters_spin.setValue(
            int(getattr(cfg, "robust_event_reject_max_iters", 3))
        )
        self._correction_tuning_robust_residual_z_spin.setValue(
            float(getattr(cfg, "robust_event_reject_residual_z_thresh", 3.5))
        )
        self._correction_tuning_robust_local_var_window_spin.setValue(
            float(getattr(cfg, "robust_event_reject_local_var_window_sec", 10.0) or 10.0)
        )
        ratio_thresh = getattr(cfg, "robust_event_reject_local_var_ratio_thresh", None)
        self._correction_tuning_robust_local_var_ratio_enable_cb.setChecked(
            ratio_thresh is not None
        )
        self._correction_tuning_robust_local_var_ratio_spin.setValue(
            float(ratio_thresh if ratio_thresh is not None else 3.0)
        )
        self._correction_tuning_robust_min_keep_fraction_spin.setValue(
            float(getattr(cfg, "robust_event_reject_min_keep_fraction", 0.5))
        )
        self._correction_tuning_adaptive_residual_z_spin.setValue(
            float(getattr(cfg, "adaptive_event_gate_residual_z_thresh", 3.5))
        )
        self._correction_tuning_adaptive_local_var_window_spin.setValue(
            float(getattr(cfg, "adaptive_event_gate_local_var_window_sec", 10.0) or 10.0)
        )
        adaptive_ratio_thresh = getattr(cfg, "adaptive_event_gate_local_var_ratio_thresh", None)
        self._correction_tuning_adaptive_local_var_ratio_enable_cb.setChecked(
            adaptive_ratio_thresh is not None
        )
        self._correction_tuning_adaptive_local_var_ratio_spin.setValue(
            float(adaptive_ratio_thresh if adaptive_ratio_thresh is not None else 3.0)
        )
        self._correction_tuning_adaptive_smooth_window_spin.setValue(
            float(getattr(cfg, "adaptive_event_gate_smooth_window_sec", 60.0))
        )
        self._correction_tuning_adaptive_min_trust_fraction_spin.setValue(
            float(getattr(cfg, "adaptive_event_gate_min_trust_fraction", 0.5))
        )
        freeze_interp = str(
            getattr(cfg, "adaptive_event_gate_freeze_interp_method", "linear_hold")
        ).strip() or "linear_hold"
        idx_freeze = self._correction_tuning_adaptive_freeze_interp_combo.findData(freeze_interp)
        if idx_freeze >= 0:
            self._correction_tuning_adaptive_freeze_interp_combo.setCurrentIndex(idx_freeze)
        self._apply_correction_tuning_fit_mode_ui_state()

    def _on_correction_tuning_roi_changed(self, _index: int) -> None:
        roi = self._correction_tuning_roi_combo.currentText().strip()
        self._populate_correction_tuning_chunk_choices(roi)

    def _collect_correction_tuning_overrides(self) -> dict:
        fit_mode = self._selected_correction_tuning_fit_mode()
        overrides = {
            "dynamic_fit_mode": fit_mode,
            "baseline_method": self._correction_tuning_baseline_method_combo.currentText().strip(),
            "baseline_percentile": float(self._correction_tuning_baseline_pct_spin.value()),
            "lowpass_hz": float(self._correction_tuning_lowpass_spin.value()),
        }
        if is_rolling_dynamic_fit_mode(fit_mode):
            overrides["baseline_subtract_before_fit"] = bool(
                self._correction_tuning_baseline_subtract_cb.isChecked()
            )
            overrides["window_sec"] = float(self._correction_tuning_window_spin.value())
            overrides["min_samples_per_window"] = int(self._correction_tuning_min_samples_spin.value())
        elif is_robust_event_reject_mode(fit_mode):
            overrides["robust_event_reject_max_iters"] = int(
                self._correction_tuning_robust_max_iters_spin.value()
            )
            overrides["robust_event_reject_residual_z_thresh"] = float(
                self._correction_tuning_robust_residual_z_spin.value()
            )
            overrides["robust_event_reject_local_var_window_sec"] = float(
                self._correction_tuning_robust_local_var_window_spin.value()
            )
            if self._correction_tuning_robust_local_var_ratio_enable_cb.isChecked():
                overrides["robust_event_reject_local_var_ratio_thresh"] = float(
                    self._correction_tuning_robust_local_var_ratio_spin.value()
                )
            else:
                overrides["robust_event_reject_local_var_ratio_thresh"] = None
            overrides["robust_event_reject_min_keep_fraction"] = float(
                self._correction_tuning_robust_min_keep_fraction_spin.value()
            )
        elif is_adaptive_event_gated_mode(fit_mode):
            overrides["adaptive_event_gate_residual_z_thresh"] = float(
                self._correction_tuning_adaptive_residual_z_spin.value()
            )
            overrides["adaptive_event_gate_local_var_window_sec"] = float(
                self._correction_tuning_adaptive_local_var_window_spin.value()
            )
            if self._correction_tuning_adaptive_local_var_ratio_enable_cb.isChecked():
                overrides["adaptive_event_gate_local_var_ratio_thresh"] = float(
                    self._correction_tuning_adaptive_local_var_ratio_spin.value()
                )
            else:
                overrides["adaptive_event_gate_local_var_ratio_thresh"] = None
            overrides["adaptive_event_gate_smooth_window_sec"] = float(
                self._correction_tuning_adaptive_smooth_window_spin.value()
            )
            overrides["adaptive_event_gate_min_trust_fraction"] = float(
                self._correction_tuning_adaptive_min_trust_fraction_spin.value()
            )
            overrides["adaptive_event_gate_freeze_interp_method"] = str(
                self._correction_tuning_adaptive_freeze_interp_combo.currentData()
                or self._correction_tuning_adaptive_freeze_interp_combo.currentText()
                or "linear_hold"
            ).strip() or "linear_hold"
        return overrides

    def _refresh_correction_tuning_applyback_status(self) -> None:
        if not hasattr(self, "_correction_tuning_applyback_status_label"):
            return
        if self._correction_tuning_applyback_applied:
            timestamp = self._correction_tuning_applyback_timestamp or datetime.now().strftime(
                "%H:%M:%S"
            )
            self._correction_tuning_applyback_status_label.setText(
                f"Copy status: copied to main run controls ({timestamp})."
            )
        else:
            self._correction_tuning_applyback_status_label.setText("Copy status: not copied.")

    def _on_apply_correction_tuning_values_to_run_settings(self) -> None:
        if not self._correction_tuning_workspace_available:
            QMessageBox.information(
                self,
                "Correction Retune Unavailable",
                self._correction_tuning_availability_label.text(),
            )
            return

        overrides = self._collect_correction_tuning_overrides()

        def _fmt(value: float, decimals: int = 6) -> str:
            text = f"{float(value):.{decimals}f}"
            return text.rstrip("0").rstrip(".") if "." in text else text

        fit_mode = normalize_dynamic_fit_mode(
            str(overrides.get("dynamic_fit_mode", "rolling_filtered_to_raw"))
        )
        idx_fit = self._dynamic_fit_mode_combo.findData(fit_mode)
        if idx_fit >= 0:
            self._dynamic_fit_mode_combo.setCurrentIndex(idx_fit)

        method = str(overrides.get("baseline_method", "")).strip()
        if method and self._baseline_method_combo.findText(method) >= 0:
            self._baseline_method_combo.setCurrentText(method)

        if "baseline_percentile" in overrides:
            self._baseline_percentile_edit.setText(
                _fmt(float(overrides["baseline_percentile"]), decimals=6)
            )
        if "lowpass_hz" in overrides:
            self._lowpass_hz_edit.setText(_fmt(float(overrides["lowpass_hz"]), decimals=6))

        if is_rolling_dynamic_fit_mode(fit_mode):
            self._baseline_subtract_before_fit_cb.setChecked(
                bool(overrides.get("baseline_subtract_before_fit", False))
            )
            if "window_sec" in overrides:
                self._window_sec_edit.setText(_fmt(float(overrides["window_sec"]), decimals=6))
            if "min_samples_per_window" in overrides:
                self._min_samples_per_window_spin.setValue(
                    int(overrides["min_samples_per_window"])
                )
        elif is_robust_event_reject_mode(fit_mode):
            if "robust_event_reject_max_iters" in overrides:
                self._robust_event_reject_max_iters_spin.setValue(
                    int(overrides["robust_event_reject_max_iters"])
                )
            if "robust_event_reject_residual_z_thresh" in overrides:
                self._robust_event_reject_residual_z_spin.setValue(
                    float(overrides["robust_event_reject_residual_z_thresh"])
                )
            if "robust_event_reject_local_var_window_sec" in overrides:
                self._robust_event_reject_local_var_window_spin.setValue(
                    float(overrides["robust_event_reject_local_var_window_sec"])
                )
            robust_ratio = overrides.get("robust_event_reject_local_var_ratio_thresh", None)
            self._robust_event_reject_local_var_ratio_enable_cb.setChecked(
                robust_ratio is not None
            )
            if robust_ratio is not None:
                self._robust_event_reject_local_var_ratio_spin.setValue(float(robust_ratio))
            if "robust_event_reject_min_keep_fraction" in overrides:
                self._robust_event_reject_min_keep_fraction_spin.setValue(
                    float(overrides["robust_event_reject_min_keep_fraction"])
                )
        elif is_adaptive_event_gated_mode(fit_mode):
            if "adaptive_event_gate_residual_z_thresh" in overrides:
                self._adaptive_event_gate_residual_z_spin.setValue(
                    float(overrides["adaptive_event_gate_residual_z_thresh"])
                )
            if "adaptive_event_gate_local_var_window_sec" in overrides:
                self._adaptive_event_gate_local_var_window_spin.setValue(
                    float(overrides["adaptive_event_gate_local_var_window_sec"])
                )
            adaptive_ratio = overrides.get("adaptive_event_gate_local_var_ratio_thresh", None)
            self._adaptive_event_gate_local_var_ratio_enable_cb.setChecked(
                adaptive_ratio is not None
            )
            if adaptive_ratio is not None:
                self._adaptive_event_gate_local_var_ratio_spin.setValue(float(adaptive_ratio))
            if "adaptive_event_gate_smooth_window_sec" in overrides:
                self._adaptive_event_gate_smooth_window_spin.setValue(
                    float(overrides["adaptive_event_gate_smooth_window_sec"])
                )
            if "adaptive_event_gate_min_trust_fraction" in overrides:
                self._adaptive_event_gate_min_trust_fraction_spin.setValue(
                    float(overrides["adaptive_event_gate_min_trust_fraction"])
                )
            freeze_method = str(
                overrides.get("adaptive_event_gate_freeze_interp_method", "linear_hold")
            ).strip() or "linear_hold"
            idx_freeze = self._adaptive_event_gate_freeze_interp_combo.findData(freeze_method)
            if idx_freeze >= 0:
                self._adaptive_event_gate_freeze_interp_combo.setCurrentIndex(idx_freeze)

        self._apply_dynamic_fit_mode_ui_state()
        self._on_config_changed()
        self._correction_tuning_applyback_applied = True
        self._correction_tuning_applyback_timestamp = datetime.now().strftime("%H:%M:%S")
        self._refresh_correction_tuning_applyback_status()
        self._append_run_log("Copied correction tuning settings to main run configuration.")

    def _on_run_correction_tuning(self) -> None:
        self._run_correction_tuning_btn.setEnabled(False)
        self._run_correction_tuning_btn.setText("Running...")
        self._push_busy_cursor()
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        try:
            if not self._correction_tuning_workspace_available:
                QMessageBox.information(
                    self,
                    "Correction Retune Unavailable",
                    self._correction_tuning_availability_label.text(),
                )
                return

            roi = self._correction_tuning_roi_combo.currentText().strip()
            if not roi:
                QMessageBox.warning(
                    self,
                    "Correction Retune Error",
                    "Select an ROI before running correction retune.",
                )
                return
            if self._correction_tuning_chunk_combo.currentIndex() < 0:
                QMessageBox.warning(
                    self,
                    "Correction Retune Error",
                    "Select a preview session before running correction retune.",
                )
                return

            chunk_id = int(self._correction_tuning_chunk_combo.currentData())
            overrides = self._collect_correction_tuning_overrides()

            try:
                result = run_cache_correction_retune(
                    run_dir=self._current_run_dir,
                    roi=roi,
                    chunk_id=chunk_id,
                    overrides=overrides,
                    out_dir=None,
                )
            except Exception as exc:
                self._append_run_log(f"Correction retune failed: {exc}")
                QMessageBox.critical(
                    self,
                    "Correction Retune Failed",
                    f"Correction-sensitive cache retune failed.\n\n{exc}",
                )
                return

            self._correction_tuning_last_result = result
            self._correction_tuning_applyback_applied = False
            self._correction_tuning_applyback_timestamp = ""
            self._refresh_correction_tuning_applyback_status()
            self._open_correction_tuning_dir_btn.setEnabled(True)
            artifacts = result.get("artifacts", {}) if isinstance(result, dict) else {}
            inspect_paths = artifacts.get("retuned_correction_inspection_pngs", [])
            panel_labels = artifacts.get("retuned_correction_inspection_panel_labels", [])
            if isinstance(inspect_paths, list) and inspect_paths:
                self._set_correction_tuning_overlay_images(
                    [str(p) for p in inspect_paths],
                    [str(x) for x in panel_labels] if isinstance(panel_labels, list) else None,
                )
            else:
                inspect_path = str(
                    artifacts.get("retuned_correction_inspection_png", "")
                ).strip()
                self._set_correction_tuning_overlay_image(inspect_path)

            lines = [
                f"ROI: {result.get('selected_roi', roi)}",
                f"Dynamic fit mode: {dynamic_fit_mode_label(overrides.get('dynamic_fit_mode', 'rolling_local_regression'))}",
                (
                    "Baseline subtract before fit: "
                    + (
                        "enabled"
                        if bool(overrides.get("baseline_subtract_before_fit", False))
                        else "disabled"
                    )
                    if is_rolling_dynamic_fit_mode(overrides.get("dynamic_fit_mode", ""))
                    else (
                        "Baseline subtract before fit: inactive in robust global event-reject mode"
                        if is_robust_event_reject_mode(overrides.get("dynamic_fit_mode", ""))
                        else (
                            "Baseline subtract before fit: inactive in adaptive event-gated mode"
                            if is_adaptive_event_gated_mode(overrides.get("dynamic_fit_mode", ""))
                            else "Baseline subtract before fit: inactive in global linear regression mode"
                        )
                    )
                ),
                "Recomputed across: all available sessions for this ROI",
                f"Preview session: {result.get('inspection_chunk_id', chunk_id)}",
                f"Retune output: {result.get('retune_dir', '(unknown)')}",
            ]
            if is_robust_event_reject_mode(overrides.get("dynamic_fit_mode", "")):
                lines.insert(
                    3,
                    format_robust_event_reject_settings_summary(
                        int(overrides.get("robust_event_reject_max_iters", 3)),
                        float(overrides.get("robust_event_reject_residual_z_thresh", 3.5)),
                        float(overrides.get("robust_event_reject_local_var_window_sec", 10.0)),
                        (
                            None
                            if overrides.get("robust_event_reject_local_var_ratio_thresh", None) is None
                            else float(overrides.get("robust_event_reject_local_var_ratio_thresh"))
                        ),
                        float(overrides.get("robust_event_reject_min_keep_fraction", 0.5)),
                    ),
                )
                robust_diag = artifacts.get("retuned_correction_inspection_robust_diagnostics", {})
                if isinstance(robust_diag, dict) and robust_diag:
                    fallback_status = str(robust_diag.get("fallback_status", "")).strip()
                    if not fallback_status:
                        fallback_status = (
                            "yes"
                            if bool(robust_diag.get("fallback_to_global_linear", False))
                            else "no"
                        )
                    keep_fraction_text = "n/a"
                    try:
                        keep_fraction = float(robust_diag.get("keep_fraction"))
                        if math.isfinite(keep_fraction):
                            keep_fraction_text = f"{keep_fraction:.3f}"
                    except Exception:
                        pass
                    lines.insert(
                        4,
                        "Robust diagnostics: "
                        f"keep fraction={keep_fraction_text}, "
                        f"iterations={int(robust_diag.get('iterations_completed', 0))}, "
                        f"fallback={fallback_status}",
                    )
            elif is_adaptive_event_gated_mode(overrides.get("dynamic_fit_mode", "")):
                lines.insert(
                    3,
                    format_adaptive_event_gated_settings_summary(
                        float(overrides.get("adaptive_event_gate_residual_z_thresh", 3.5)),
                        float(overrides.get("adaptive_event_gate_local_var_window_sec", 10.0)),
                        (
                            None
                            if overrides.get("adaptive_event_gate_local_var_ratio_thresh", None) is None
                            else float(overrides.get("adaptive_event_gate_local_var_ratio_thresh"))
                        ),
                        float(overrides.get("adaptive_event_gate_smooth_window_sec", 60.0)),
                        float(overrides.get("adaptive_event_gate_min_trust_fraction", 0.5)),
                        str(overrides.get("adaptive_event_gate_freeze_interp_method", "linear_hold")),
                    ),
                )
                adaptive_diag = artifacts.get("retuned_correction_inspection_adaptive_diagnostics", {})
                if isinstance(adaptive_diag, dict) and adaptive_diag:
                    fallback_status = str(adaptive_diag.get("fallback_status", "")).strip() or "no"
                    trust_fraction_text = "n/a"
                    gated_fraction_text = "n/a"
                    try:
                        trust_fraction = float(adaptive_diag.get("trust_fraction"))
                        if math.isfinite(trust_fraction):
                            trust_fraction_text = f"{trust_fraction:.3f}"
                    except Exception:
                        pass
                    try:
                        gated_fraction = float(adaptive_diag.get("gated_fraction"))
                        if math.isfinite(gated_fraction):
                            gated_fraction_text = f"{gated_fraction:.3f}"
                    except Exception:
                        pass
                    lines.insert(
                        4,
                        "Adaptive diagnostics: "
                        f"trust fraction={trust_fraction_text}, "
                        f"gated fraction={gated_fraction_text}, "
                        f"fallback={fallback_status}",
                    )
            self._correction_tuning_summary_label.setText("\n".join(lines))
            self._append_run_log(
                f"Correction retune completed for ROI={result.get('selected_roi', roi)} "
                f"preview_session={result.get('inspection_chunk_id', chunk_id)} "
                f"-> {result.get('retune_dir', '(unknown)')}"
            )
            QTimer.singleShot(0, self._finalize_correction_tuning_result_layout)
        finally:
            self._pop_busy_cursor()
            self._run_correction_tuning_btn.setEnabled(
                self._correction_tuning_workspace_available
            )
            self._run_correction_tuning_btn.setText("Run Correction Retune")

    def _on_open_correction_tuning_output(self) -> None:
        if not isinstance(self._correction_tuning_last_result, dict):
            QMessageBox.information(
                self,
                "No Correction Output",
                "Run correction retune first to create an output directory.",
            )
            return
        retune_dir = str(self._correction_tuning_last_result.get("retune_dir", "")).strip()
        if not retune_dir or not os.path.isdir(retune_dir):
            QMessageBox.information(
                self,
                "No Correction Output",
                "Correction retune output directory is not available.",
            )
            return
        _open_folder(retune_dir)

    def _finalize_correction_tuning_result_layout(self) -> None:
        self._tuning_group.updateGeometry()
        self._correction_tuning_scroll_content.updateGeometry()
        self._correction_tuning_scroll.viewport().updateGeometry()
        self._correction_tuning_scroll.updateGeometry()
        self._render_correction_tuning_overlay()

    def _reset_correction_tuning_state(self) -> None:
        self._correction_tuning_last_result = None
        self._correction_tuning_applyback_applied = False
        self._correction_tuning_applyback_timestamp = ""
        if hasattr(self, "_open_correction_tuning_dir_btn"):
            self._open_correction_tuning_dir_btn.setEnabled(False)
        if hasattr(self, "_apply_correction_tuning_btn"):
            self._apply_correction_tuning_btn.setEnabled(
                bool(self._correction_tuning_workspace_available)
            )
        if hasattr(self, "_correction_tuning_summary_label"):
            self._correction_tuning_summary_label.setText(
                "No correction retune result yet."
            )
        self._refresh_correction_tuning_applyback_status()
        if hasattr(self, "_set_correction_tuning_overlay_message"):
            self._set_correction_tuning_overlay_message(
                "Run correction retune to generate a correction inspection artifact."
            )
        self._set_correction_tuning_disclosure_expanded(False)

    def _collect_tuned_export_overrides(self) -> dict:
        """
        Collect tuned workspace overrides for durable config export.

        Uses the same typed override sources already used by retune/apply-back
        paths, then lets RunSpec's effective-config machinery merge them with
        the active baseline config source.
        """
        overrides: dict = {}

        include_downstream = bool(
            self._tuning_applyback_applied or isinstance(self._tuning_last_result, dict)
        )
        if include_downstream:
            downstream = dict(self._collect_tuning_overrides())
            if "peak_pre_filter" in downstream:
                downstream["peak_pre_filter"] = map_retune_peak_pre_filter_to_run_setting(
                    str(downstream["peak_pre_filter"])
                )
            overrides.update(downstream)

        include_correction = bool(
            self._correction_tuning_applyback_applied
            or isinstance(self._correction_tuning_last_result, dict)
        )
        if include_correction:
            overrides.update(self._collect_correction_tuning_overrides())

        return overrides

    def _build_tuned_config_export_spec(self) -> RunSpec:
        """
        Build a RunSpec-backed effective config export without mutating run state.
        """
        previous_run_dir = self._current_run_dir
        try:
            spec = self._build_run_spec(validate_only=False)
        finally:
            self._current_run_dir = previous_run_dir

        tuned_overrides = self._collect_tuned_export_overrides()
        if tuned_overrides:
            spec.config_overrides.update(tuned_overrides)

        # Export-only path; caller writes chosen destination file.
        spec.run_dir = ""
        return spec

    def _on_save_tuned_settings_as_config(self) -> None:
        if not (self._tuning_workspace_available or self._correction_tuning_workspace_available):
            QMessageBox.information(
                self,
                "Tuning Unavailable",
                "Tuned settings can be saved only when post-run tuning is available.",
            )
            return

        try:
            spec = self._build_tuned_config_export_spec()
            preview_yaml = spec.get_derived_config_preview()
        except Exception as exc:
            self._append_run_log(f"Save tuned config failed: {exc}")
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Could not build tuned effective config.\n\n{exc}",
            )
            return

        preferred_dir = ""
        cfg_path = self._config_path.text().strip()
        if cfg_path:
            preferred_dir = os.path.dirname(cfg_path)
        if not preferred_dir and self._current_run_dir and os.path.isdir(self._current_run_dir):
            preferred_dir = self._current_run_dir
        out_base = self._output_dir.text().strip()
        if not preferred_dir and out_base and os.path.isdir(out_base):
            preferred_dir = out_base
        if not preferred_dir:
            preferred_dir = os.getcwd()

        default_name = f"tuned_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        default_path = os.path.join(preferred_dir, default_name)

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Tuned Settings as Config",
            default_path,
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not save_path:
            return
        if not save_path.lower().endswith((".yaml", ".yml")):
            save_path = f"{save_path}.yaml"

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(preview_yaml)
            RunSpec.validate_effective_config(save_path)
        except Exception as exc:
            self._append_run_log(f"Save tuned config failed: {exc}")
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Could not write a valid config YAML.\n\n{exc}",
            )
            return

        self._append_run_log(f"Saved tuned config YAML: {save_path}")
        activate = QMessageBox.question(
            self,
            "Activate Saved Config?",
            "Saved tuned settings as YAML.\n\n"
            "Use this file as the active custom config source now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if activate == QMessageBox.StandardButton.Yes:
            self._use_custom_config_cb.setChecked(True)
            self._config_path.setText(save_path)
            self._update_config_source_ui()
            self._on_config_changed()
            self._append_run_log(
                f"Activated custom config source: {save_path}"
            )
            QMessageBox.information(
                self,
                "Saved and Activated",
                f"Saved tuned settings to:\n{save_path}\n\n"
                "Custom config source is now active.",
            )
        else:
            QMessageBox.information(
                self,
                "Saved",
                f"Saved tuned settings to:\n{save_path}",
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_splitter_workspace_policy()
        self._render_tuning_overlay()
        self._render_correction_tuning_overlay()

    def _update_adv_group_visibility(self):
        # In our GUI contract, "both" maps to mode_val=None (which implies phasic runs)
        # Show if (mode == "both" or mode == "phasic")
        mode_text = self._mode_combo.currentText()
        if is_isosbestic_active(mode_text):
            self._adv_group.show()
        else:
            self._adv_group.hide()

    def _update_adv_prep_visibility(self):
        method_text = self._baseline_method_combo.currentText()
        if baseline_method_requires_percentile(method_text):
            self._baseline_percentile_edit.show()
            self._baseline_percentile_label.show()
        else:
            self._baseline_percentile_edit.hide()
            self._baseline_percentile_label.hide()

    def _update_adv_ev_visibility(self):
        method_text = self._peak_method_combo.currentText()
        
        if peak_threshold_method_requires_k(method_text):
            self._peak_k_edit.show()
            self._peak_k_label.show()
        else:
            self._peak_k_edit.hide()
            self._peak_k_label.hide()
            
        if peak_threshold_method_requires_percentile(method_text):
            self._peak_pct_edit.show()
            self._peak_pct_label.show()
        else:
            self._peak_pct_edit.hide()
            self._peak_pct_label.hide()
            
        if peak_threshold_method_requires_abs(method_text):
            self._peak_abs_edit.show()
            self._peak_abs_label.show()
        else:
            self._peak_abs_edit.hide()
            self._peak_abs_label.hide()

    # ==================================================================
    # RunSpec construction + argv (GUI mode: --out <explicit_run_dir>)
    # ==================================================================

    @staticmethod
    def _track_if_nonempty(field_name: str, text: str, out: list) -> None:
        """Append field_name to out if text is non-empty."""
        if text:
            out.append(field_name)

    @staticmethod
    def _track_if_changed(field_name: str, value, default, out: list) -> None:
        """Append field_name to out if value != default."""
        if value != default:
            out.append(field_name)

    def _timeline_anchor_mode_value(self) -> str:
        """Return normalized timeline anchor mode from plotting controls."""
        combo = getattr(self, "_timeline_anchor_mode_combo", None)
        if combo is None:
            return "civil"
        data = combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        text = combo.currentText().strip().lower()
        if text.startswith("elapsed"):
            return "elapsed"
        if text.startswith("fixed"):
            return "fixed_daily_anchor"
        return "civil"

    def _timeline_anchor_summary(self) -> str:
        """Human-readable timeline anchor summary for UI text."""
        mode = self._timeline_anchor_mode_value()
        if mode == "elapsed":
            return "Elapsed from first session"
        if mode == "fixed_daily_anchor":
            clock_edit = getattr(self, "_fixed_daily_anchor_time_edit", None)
            clock = clock_edit.text().strip() if clock_edit is not None else ""
            return f"Fixed daily anchor ({clock or 'unset'})"
        return "Civil clock"

    def _is_custom_config_enabled(self) -> bool:
        """True when user explicitly opted into custom baseline YAML."""
        return bool(self._use_custom_config_cb.isChecked())

    def _active_config_source_path(self) -> str:
        """Resolved config source path currently used to build effective config."""
        if self._is_custom_config_enabled():
            return self._config_path.text().strip()
        return self._lab_default_config_path

    def _active_config_source_summary(self) -> str:
        """Human-readable active config source descriptor."""
        if self._is_custom_config_enabled():
            cfg = self._config_path.text().strip() or "(not set)"
            return f"Custom YAML: {cfg}"
        return f"Lab standard default: {self._lab_default_config_path}"

    @staticmethod
    def _normalize_csv_cells(row: list[str]) -> list[str]:
        return [str(cell).strip().lstrip("\ufeff") for cell in row]

    @staticmethod
    def _discover_rwd_csv_files(input_path: str) -> list[str]:
        if os.path.isfile(input_path) and input_path.lower().endswith(".csv"):
            return [input_path]
        if not os.path.isdir(input_path):
            return []

        from photometry_pipeline.io.adapters import discover_rwd_chunks

        try:
            chunk_files = discover_rwd_chunks(input_path)
            if chunk_files:
                return chunk_files
        except Exception:
            pass

        return sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.lower().endswith(".csv")
        )

    @staticmethod
    def _discover_npm_csv_files(input_path: str) -> list[str]:
        if os.path.isfile(input_path) and input_path.lower().endswith(".csv"):
            return [input_path]
        if not os.path.isdir(input_path):
            return []
        files = sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.lower().endswith(".csv")
        )
        from photometry_pipeline.io.adapters import sort_npm_files
        return sort_npm_files(files)

    @staticmethod
    def _looks_like_npm_header_row(columns: list[str], cfg: Config) -> bool:
        has_led = cfg.npm_led_col in columns
        has_roi = any(
            col.startswith(cfg.npm_region_prefix) and col.endswith(cfg.npm_region_suffix)
            for col in columns
        )
        has_time = (
            "Timestamp" in columns
            or cfg.npm_system_ts_col in columns
            or cfg.npm_computer_ts_col in columns
            or "SystemTimestamp" in columns
            or "ComputerTimestamp" in columns
        )
        return has_time and has_led and has_roi

    def _infer_npm_chunk_contract(self, csv_path: str, cfg: Config) -> dict:
        with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(f"Empty NPM CSV: {csv_path}")
        cols = self._normalize_csv_cells(header)
        if not self._looks_like_npm_header_row(cols, cfg):
            raise ValueError(f"No recognizable NPM header row found in: {csv_path}")

        if "Timestamp" in cols:
            time_col = "Timestamp"
            time_axis = "system_timestamp"
        elif cfg.npm_system_ts_col in cols:
            time_col = cfg.npm_system_ts_col
            time_axis = "system_timestamp"
        elif "SystemTimestamp" in cols:
            time_col = "SystemTimestamp"
            time_axis = "system_timestamp"
        elif cfg.npm_computer_ts_col in cols:
            time_col = cfg.npm_computer_ts_col
            time_axis = "computer_timestamp"
        elif "ComputerTimestamp" in cols:
            time_col = "ComputerTimestamp"
            time_axis = "computer_timestamp"
        else:
            raise ValueError(f"No recognizable NPM time column found in: {csv_path}")

        if cfg.npm_led_col not in cols:
            raise ValueError(f"Missing required NPM LED column '{cfg.npm_led_col}' in: {csv_path}")
        led_idx = cols.index(cfg.npm_led_col)
        time_idx = cols.index(time_col)

        t_uv: list[float] = []
        t_sig: list[float] = []
        with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                norm = self._normalize_csv_cells(row)
                if time_idx >= len(norm) or led_idx >= len(norm):
                    continue
                t_cell = norm[time_idx]
                led_cell = norm[led_idx]
                if not t_cell or not led_cell:
                    continue
                try:
                    t_val = float(t_cell)
                    led_val = int(float(led_cell))
                except Exception:
                    continue
                if led_val == 1:
                    t_uv.append(t_val)
                elif led_val == 2:
                    t_sig.append(t_val)

        def _finite_strict(values: list[float], label: str) -> list[float]:
            out = [v for v in values if math.isfinite(v)]
            if len(out) < 2:
                raise ValueError(
                    f"NPM contract inference: insufficient finite {label} samples in: {csv_path}"
                )
            if any((out[i + 1] - out[i]) <= 0 for i in range(len(out) - 1)):
                raise ValueError(
                    f"NPM contract inference: non-increasing {label} timestamps in: {csv_path}"
                )
            return out

        t_uv_f = _finite_strict(t_uv, "UV (LedState=1)")
        t_sig_f = _finite_strict(t_sig, "SIG (LedState=2)")

        # Match strict loader alignment semantics:
        # t0 = max(min(t_uv), min(t_sig)), then derive relative streams.
        t0 = max(float(t_uv_f[0]), float(t_sig_f[0]))
        t_uv_rel = [t - t0 for t in t_uv_f if (t - t0) >= 0.0]
        t_sig_rel = [t - t0 for t in t_sig_f if (t - t0) >= 0.0]

        t_uv_rel = _finite_strict(t_uv_rel, "UV relative")
        t_sig_rel = _finite_strict(t_sig_rel, "SIG relative")

        def _infer_fs(rel_values: list[float], label: str) -> float:
            dt = [rel_values[i + 1] - rel_values[i] for i in range(len(rel_values) - 1)]
            if any(x <= 0 for x in dt):
                raise ValueError(
                    f"NPM contract inference: invalid {label} dt (non-positive) in: {csv_path}"
                )
            med_dt = float(median(dt))
            if med_dt <= 0:
                raise ValueError(
                    f"NPM contract inference: invalid {label} median dt in: {csv_path}"
                )
            return 1.0 / med_dt

        fs_uv = _infer_fs(t_uv_rel, "UV")
        fs_sig = _infer_fs(t_sig_rel, "SIG")
        fs_tol = max(1e-6, max(fs_uv, fs_sig) * 1e-3)
        if not math.isclose(fs_uv, fs_sig, rel_tol=0.0, abs_tol=fs_tol):
            raise ValueError(
                "NPM contract inference: UV/SIG cadence mismatch in "
                f"{csv_path}. UV fs={fs_uv:.9f}, SIG fs={fs_sig:.9f}."
            )
        inferred_fs = float((fs_uv + fs_sig) / 2.0)

        # Align chunk_duration to strict loader grid semantics:
        # n_target = round(chunk_duration_sec * target_fs_hz),
        # grid = arange(n_target) / target_fs_hz.
        sample_count = min(len(t_uv_rel), len(t_sig_rel))
        if sample_count < 2:
            raise ValueError(
                f"NPM contract inference: insufficient aligned UV/SIG samples in: {csv_path}"
            )
        chunk_duration_sec = sample_count / inferred_fs
        n_target = int(round(chunk_duration_sec * inferred_fs))
        if n_target != sample_count:
            raise ValueError(
                "NPM contract inference failed strict grid compatibility for "
                f"{csv_path}: sample_count={sample_count}, fs={inferred_fs:.9f}, "
                f"chunk_duration_sec={chunk_duration_sec:.9f}, n_target={n_target}."
            )

        return {
            "csv_path": csv_path,
            "time_col": time_col,
            "time_axis": time_axis,
            "led_col": cfg.npm_led_col,
            "region_prefix": cfg.npm_region_prefix,
            "region_suffix": cfg.npm_region_suffix,
            "fs_hz": float(inferred_fs),
            "chunk_duration_sec": float(chunk_duration_sec),
            "sample_count": int(sample_count),
        }

    def _infer_npm_dataset_contract_overrides(self, fmt_text: str) -> dict:
        if fmt_text not in ("auto", "npm"):
            return {}

        input_path = self._input_dir.text().strip()
        csv_paths = self._discover_npm_csv_files(input_path)
        if not csv_paths:
            return {}

        cfg = self._active_baseline_config()
        try:
            base = self._infer_npm_chunk_contract(csv_paths[0], cfg)
        except ValueError:
            if fmt_text == "npm":
                raise
            return {}
        contracts = [base]
        for path in csv_paths[1:]:
            contracts.append(self._infer_npm_chunk_contract(path, cfg))

        fs_vals = [float(c["fs_hz"]) for c in contracts]
        dur_vals = [float(c["chunk_duration_sec"]) for c in contracts]
        rep_fs = float(median(fs_vals))
        rep_duration = float(median(dur_vals))
        fs_tol = max(1e-6, rep_fs * 5e-3)
        # NPM long runs are known to show per-file duration drift; enforce a
        # bounded spread guard instead of exact equality.
        dur_tol = max(1.0 / max(rep_fs, 1e-9), rep_duration * 0.10)
        for idx, contract in enumerate(contracts):
            if contract["time_col"] != base["time_col"] or contract["time_axis"] != base["time_axis"]:
                raise ValueError(
                    "Inconsistent NPM contract across files: "
                    f"time column mismatch at file {idx} ({contract['csv_path']}). "
                    f"Expected axis/col ({base['time_axis']}, {base['time_col']}), "
                    f"got ({contract['time_axis']}, {contract['time_col']})."
                )
            if (
                contract["led_col"] != base["led_col"]
                or contract["region_prefix"] != base["region_prefix"]
                or contract["region_suffix"] != base["region_suffix"]
            ):
                raise ValueError(
                    "Inconsistent NPM contract across files: "
                    f"header semantics mismatch at file {idx} ({contract['csv_path']})."
                )
            if not math.isclose(contract["fs_hz"], rep_fs, rel_tol=0.0, abs_tol=fs_tol):
                raise ValueError(
                    "Inconsistent NPM contract across files: "
                    f"timing mismatch at file {idx} ({contract['csv_path']}). "
                    f"Representative fs {rep_fs:.9f}, got {contract['fs_hz']:.9f}. "
                    f"Observed fs range [{min(fs_vals):.9f}, {max(fs_vals):.9f}]."
                )
            if not math.isclose(
                contract["chunk_duration_sec"],
                rep_duration,
                rel_tol=0.0,
                abs_tol=dur_tol,
            ):
                raise ValueError(
                    "Inconsistent NPM contract across files: "
                    f"chunk duration mismatch at file {idx} ({contract['csv_path']}). "
                    f"Representative {rep_duration:.9f}, got {contract['chunk_duration_sec']:.9f}. "
                    f"Observed duration range [{min(dur_vals):.9f}, {max(dur_vals):.9f}]."
                )

        out = {
            "npm_time_axis": str(base["time_axis"]),
            "npm_led_col": str(base["led_col"]),
            "npm_region_prefix": str(base["region_prefix"]),
            "npm_region_suffix": str(base["region_suffix"]),
            "target_fs_hz": float(round(rep_fs, 9)),
            "chunk_duration_sec": float(round(rep_duration, 9)),
        }
        if base["time_axis"] == "system_timestamp":
            out["npm_system_ts_col"] = str(base["time_col"])
        else:
            out["npm_computer_ts_col"] = str(base["time_col"])
        return out

    def _infer_dataset_contract_overrides(self, fmt_text: str) -> dict:
        """
        Resolve dataset-derived overrides for NPM/RWD without broad guessing.

        For auto format, prefer NPM contract detection first, then fall back to
        existing RWD contract inference.
        """
        npm_overrides = self._infer_npm_dataset_contract_overrides(fmt_text)
        if npm_overrides:
            return npm_overrides
        return self._infer_rwd_dataset_contract_overrides(fmt_text)

    @staticmethod
    def _infer_rwd_suffixes_from_header(columns: list[str]) -> tuple[str, str] | None:
        base_to_suffixes: dict[str, set[str]] = {}
        for col in columns:
            m = re.match(r"^(.*?)(-\d+)$", col)
            if not m:
                continue
            base = m.group(1)
            suffix = m.group(2)
            if not base:
                continue
            base_to_suffixes.setdefault(base, set()).add(suffix)

        pair_counts: dict[tuple[str, str], int] = {}
        for suffixes in base_to_suffixes.values():
            if "-470" in suffixes:
                for uv_candidate in ("-410", "-415"):
                    if uv_candidate in suffixes:
                        key = (uv_candidate, "-470")
                        pair_counts[key] = pair_counts.get(key, 0) + 1

        if not pair_counts:
            return None

        return max(pair_counts.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _extract_metadata_fps_from_row(row: list[str]) -> float | None:
        if not row:
            return None
        meta_text = ",".join(str(cell).strip() for cell in row if str(cell).strip())
        if not meta_text:
            return None
        m = re.search(r'"?Fps"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)', meta_text, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            fps = float(m.group(1))
        except Exception:
            return None
        if fps <= 0:
            return None
        return fps

    @staticmethod
    def _extract_enabled_excitation_count_from_row(row: list[str]) -> int | None:
        if not row:
            return None
        meta_text = ",".join(str(cell).strip() for cell in row if str(cell).strip())
        if not meta_text:
            return None

        enabled_count = 0
        seen_any = False
        for led_key in ("Led410Enable", "Led470Enable", "Led560Enable"):
            m = re.search(
                rf'"?{re.escape(led_key)}"?\s*[:=]\s*(true|false|1|0)',
                meta_text,
                flags=re.IGNORECASE,
            )
            if not m:
                continue
            seen_any = True
            token = str(m.group(1)).strip().lower()
            if token in {"true", "1"}:
                enabled_count += 1

        if not seen_any:
            return None
        return enabled_count

    @staticmethod
    def _resolve_timestamp_unit_and_fs(
        median_dt: float,
        metadata_fps: float | None,
        enabled_excitation_count: int | None = None,
    ) -> tuple[str, float]:
        """
        Determine timestamp units deterministically.

        Rules:
        - If metadata FPS is available, require timestamp-derived FPS to match
          either seconds (1/dt) or milliseconds (1000/dt) semantics within a
          tight tolerance. If both or neither match, fail explicitly.
        - If metadata FPS is unavailable, only accept clearly separated dt
          ranges to avoid silent unit guessing:
            dt < 0.5   -> seconds
            dt > 2.0   -> milliseconds
            else       -> ambiguous (error)
        """
        if median_dt <= 0:
            raise ValueError(f"Invalid non-positive median timestamp delta: {median_dt}")

        fs_seconds = 1.0 / median_dt
        fs_milliseconds = 1000.0 / median_dt

        if metadata_fps is not None and metadata_fps > 0:
            rel_tol = 0.03
            if enabled_excitation_count is not None and enabled_excitation_count <= 0:
                raise ValueError(
                    "RWD metadata declares no enabled excitation LEDs, cannot reconcile metadata FPS."
                )

            multiplier = (
                enabled_excitation_count
                if (enabled_excitation_count is not None and enabled_excitation_count > 1)
                else 1
            )

            def _matches_metadata(candidate_row_fs: float) -> bool:
                direct_ok = abs(candidate_row_fs - metadata_fps) / metadata_fps <= rel_tol
                multiplex_ok = abs((candidate_row_fs * multiplier) - metadata_fps) / metadata_fps <= rel_tol
                return direct_ok or multiplex_ok

            sec_match = _matches_metadata(fs_seconds)
            ms_match = _matches_metadata(fs_milliseconds)
            if sec_match and not ms_match:
                return "seconds", fs_seconds
            if ms_match and not sec_match:
                return "milliseconds", fs_milliseconds
            if sec_match and ms_match:
                raise ValueError(
                    "Ambiguous RWD timestamp units: both seconds and milliseconds "
                    f"fit metadata FPS={metadata_fps:.6f} (dt={median_dt:.6f}, "
                    f"enabled_excitation_count={enabled_excitation_count})."
                )
            raise ValueError(
                "RWD timestamps are incompatible with metadata FPS: "
                f"dt={median_dt:.6f}, fs_seconds={fs_seconds:.6f}, "
                f"fs_milliseconds={fs_milliseconds:.6f}, metadata_fps={metadata_fps:.6f}, "
                f"enabled_excitation_count={enabled_excitation_count}."
            )

        if median_dt < 0.5:
            return "seconds", fs_seconds
        if median_dt > 2.0:
            return "milliseconds", fs_milliseconds
        raise ValueError(
            "Ambiguous RWD timestamp units: metadata FPS unavailable and median dt "
            f"{median_dt:.6f} is not clearly seconds (<0.5) or milliseconds (>2.0)."
        )

    def _infer_rwd_chunk_contract(self, csv_path: str) -> dict:
        t_chunk = self._timing_start("rwd_chunk_contract", extra=f"path={csv_path}")
        time_candidates = ("TimeStamp", "Time(s)", "Timestamp", "Time")
        header_row_idx = None
        header_cols: list[str] = []
        time_col = None
        rows: list[list[str]] = []

        with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        for idx, row in enumerate(rows[:60]):
            cols = self._normalize_csv_cells(row)
            candidate_time = next((c for c in time_candidates if c in cols), None)
            if candidate_time:
                header_row_idx = idx
                header_cols = cols
                time_col = candidate_time
                break

        if header_row_idx is None or not time_col:
            raise ValueError(f"No recognizable RWD header row found in: {csv_path}")

        suffix_pair = self._infer_rwd_suffixes_from_header(header_cols)
        if suffix_pair is None:
            raise ValueError(f"No valid UV/SIG channel suffix pair found in header: {csv_path}")
        uv_suffix, sig_suffix = suffix_pair

        metadata_fps = None
        enabled_excitation_count = None
        if header_row_idx > 0:
            meta_row = self._normalize_csv_cells(rows[header_row_idx - 1])
            metadata_fps = self._extract_metadata_fps_from_row(meta_row)
            enabled_excitation_count = self._extract_enabled_excitation_count_from_row(meta_row)

        cols = self._normalize_csv_cells(rows[header_row_idx])
        if time_col not in cols:
            raise ValueError(f"Configured RWD time column '{time_col}' not found in: {csv_path}")
        time_idx = cols.index(time_col)

        t_vals: list[float] = []
        for row in rows[header_row_idx + 1:]:
            norm = self._normalize_csv_cells(row)
            if time_idx >= len(norm):
                continue
            cell = norm[time_idx]
            if not cell:
                continue
            try:
                t_vals.append(float(cell))
            except Exception:
                continue

        if len(t_vals) < 2:
            raise ValueError(f"Insufficient numeric timestamps in RWD file: {csv_path}")

        dt_all = [t_vals[i + 1] - t_vals[i] for i in range(len(t_vals) - 1)]
        if any(x <= 0 for x in dt_all):
            raise ValueError(f"Non-monotonic or repeated timestamps in RWD file: {csv_path}")

        median_dt = float(median(dt_all))
        unit, inferred_fs = self._resolve_timestamp_unit_and_fs(
            median_dt,
            metadata_fps,
            enabled_excitation_count,
        )
        if inferred_fs <= 0:
            raise ValueError(f"Inferred non-positive fs in RWD file: {csv_path}")

        sample_count = len(t_vals)
        # Align with strict reader semantics:
        # n_target = round(chunk_duration_sec * target_fs_hz), grid = arange(n_target)/fs.
        # Setting chunk_duration_sec = sample_count / fs guarantees n_target equals
        # observed sample_count for full chunks.
        chunk_duration_sec = sample_count / inferred_fs
        n_target = int(round(chunk_duration_sec * inferred_fs))
        if n_target != sample_count:
            raise ValueError(
                "RWD contract inference failed strict grid compatibility for "
                f"{csv_path}: sample_count={sample_count}, fs={inferred_fs:.9f}, "
                f"chunk_duration_sec={chunk_duration_sec:.9f}, n_target={n_target}."
            )

        out = {
            "csv_path": csv_path,
            "time_col": time_col,
            "uv_suffix": uv_suffix,
            "sig_suffix": sig_suffix,
            "timestamp_unit": unit,
            "fs_hz": float(inferred_fs),
            "sample_count": int(sample_count),
            "chunk_duration_sec": float(chunk_duration_sec),
        }
        self._timing_end(
            "rwd_chunk_contract",
            t_chunk,
            extra=(
                f"time_col={time_col} uv={uv_suffix} sig={sig_suffix} "
                f"fs={out['fs_hz']:.6f} samples={sample_count}"
            ),
        )
        return out

    def _infer_rwd_dataset_contract_overrides(self, fmt_text: str) -> dict:
        """
        Infer RWD acquisition contract from selected input data.

        Returns overrides for config_effective.yaml when format is auto/rwd:
        target_fs_hz, chunk_duration_sec, rwd_time_col, uv_suffix, sig_suffix.
        """
        if fmt_text not in ("auto", "rwd"):
            return {}

        t_total = self._timing_start("rwd_contract_inference_total", extra=f"format={fmt_text}")
        input_path = self._input_dir.text().strip()
        t_inspect = self._timing_start("selected_input_inspection", extra=f"input={input_path}")
        csv_paths = self._discover_rwd_csv_files(input_path)
        self._timing_end("selected_input_inspection", t_inspect, extra=f"csv_files={len(csv_paths)}")
        if not csv_paths:
            if self._rwd_contract_cache is not None:
                self._emit_gui_timing("CACHE_INVALIDATE", "rwd_contract", extra="reason=no_csv_files")
                self._rwd_contract_cache = None
            self._timing_end("rwd_contract_inference_total", t_total, extra="no_csv_files")
            return {}

        # Cheap signature for cache invalidation:
        # path + format + ordered csv path list + per-file (size, mtime_ns).
        # This detects ordinary file additions/removals/edits while remaining much
        # cheaper than reparsing every chunk on every click.
        t_sig = self._timing_start("rwd_contract_signature")
        file_sigs = []
        for path in csv_paths:
            st = os.stat(path)
            file_sigs.append((path, int(st.st_size), int(st.st_mtime_ns)))
        signature = (input_path, fmt_text, tuple(file_sigs))
        self._timing_end("rwd_contract_signature", t_sig)

        cache = self._rwd_contract_cache
        if cache is not None and cache.get("signature") == signature:
            self._emit_gui_timing("CACHE_HIT", "rwd_contract", extra=f"chunks={len(csv_paths)}")
            self._timing_end("rwd_contract_inference_total", t_total, extra="cache_hit")
            return dict(cache["overrides"])
        if cache is None:
            self._emit_gui_timing("CACHE_MISS", "rwd_contract", extra="reason=no_cache")
        else:
            reason = "signature_changed"
            if cache.get("input_path") != input_path:
                reason = "input_path_changed"
            elif cache.get("format") != fmt_text:
                reason = "format_changed"
            self._emit_gui_timing("CACHE_INVALIDATE", "rwd_contract", extra=f"reason={reason}")

        t_chunks = self._timing_start("rwd_contract_scan_chunks", extra=f"chunk_count={len(csv_paths)}")
        contracts = [self._infer_rwd_chunk_contract(path) for path in csv_paths]
        self._timing_end("rwd_contract_scan_chunks", t_chunks)
        base = contracts[0]
        fs_tol = max(1e-6, base["fs_hz"] * 1e-4)
        dur_tol = max(1e-6, base["chunk_duration_sec"] * 1e-4)
        endpoint_sample_tolerance = 1
        t_cross = self._timing_start("rwd_contract_cross_chunk_validation")
        for idx, contract in enumerate(contracts[1:], start=1):
            path = contract["csv_path"]
            if contract["time_col"] != base["time_col"]:
                raise ValueError(
                    "Inconsistent RWD contract across chunks: "
                    f"time column mismatch at chunk {idx} ({path}). "
                    f"Expected '{base['time_col']}', got '{contract['time_col']}'."
                )
            if contract["uv_suffix"] != base["uv_suffix"] or contract["sig_suffix"] != base["sig_suffix"]:
                raise ValueError(
                    "Inconsistent RWD contract across chunks: "
                    f"channel suffix mismatch at chunk {idx} ({path}). "
                    f"Expected ({base['uv_suffix']}, {base['sig_suffix']}), "
                    f"got ({contract['uv_suffix']}, {contract['sig_suffix']})."
                )
            if not math.isclose(contract["fs_hz"], base["fs_hz"], rel_tol=0.0, abs_tol=fs_tol):
                raise ValueError(
                    "Inconsistent RWD contract across chunks: "
                    f"fs mismatch at chunk {idx} ({path}). "
                    f"Expected {base['fs_hz']:.9f}, got {contract['fs_hz']:.9f}."
                )

            sample_delta = abs(int(contract["sample_count"]) - int(base["sample_count"]))
            if sample_delta > endpoint_sample_tolerance:
                raise ValueError(
                    "Inconsistent RWD contract across chunks: "
                    f"sample_count mismatch exceeds endpoint tolerance at chunk {idx} ({path}). "
                    f"Expected {base['sample_count']}, got {contract['sample_count']} "
                    f"(allowed delta <= {endpoint_sample_tolerance})."
                )

            fs_ref = max(1e-9, 0.5 * (float(base["fs_hz"]) + float(contract["fs_hz"])))
            endpoint_duration_tol = 1.05 / fs_ref
            duration_abs_tol = max(dur_tol, endpoint_duration_tol)
            if not math.isclose(
                contract["chunk_duration_sec"],
                base["chunk_duration_sec"],
                rel_tol=0.0,
                abs_tol=duration_abs_tol,
            ):
                raise ValueError(
                    "Inconsistent RWD contract across chunks: "
                    f"chunk_duration mismatch at chunk {idx} ({path}). "
                    f"Expected {base['chunk_duration_sec']:.9f}, got {contract['chunk_duration_sec']:.9f}."
                )
        self._timing_end("rwd_contract_cross_chunk_validation", t_cross)

        out = {
            "target_fs_hz": float(round(base["fs_hz"], 9)),
            "chunk_duration_sec": float(round(base["chunk_duration_sec"], 9)),
            "rwd_time_col": str(base["time_col"]),
            "uv_suffix": str(base["uv_suffix"]),
            "sig_suffix": str(base["sig_suffix"]),
        }
        self._rwd_contract_cache = {
            "input_path": input_path,
            "format": fmt_text,
            "signature": signature,
            "overrides": dict(out),
        }
        self._timing_end("rwd_contract_inference_total", t_total)
        return out

    def _active_baseline_config(self) -> Config:
        """
        Resolve baseline config from the active source path for current run intent.

        This keeps GUI override diffing aligned with the same baseline that is
        written into config_effective.yaml.
        """
        cfg_path = self._active_config_source_path()
        if cfg_path and os.path.isfile(cfg_path):
            try:
                return Config.from_yaml(cfg_path)
            except Exception:
                pass
        return self._default_cfg

    def _event_feature_defaults_from_active_baseline(self) -> dict:
        """Event-feature default values sourced from active baseline config."""
        base_cfg = self._active_baseline_config()
        return {
            "event_signal": base_cfg.event_signal,
            "peak_threshold_method": base_cfg.peak_threshold_method,
            "peak_threshold_k": base_cfg.peak_threshold_k,
            "peak_threshold_percentile": base_cfg.peak_threshold_percentile,
            "peak_threshold_abs": base_cfg.peak_threshold_abs,
            "peak_min_distance_sec": base_cfg.peak_min_distance_sec,
            "peak_min_prominence_k": getattr(base_cfg, "peak_min_prominence_k", 0.0),
            "peak_min_width_sec": getattr(base_cfg, "peak_min_width_sec", 0.0),
            "peak_pre_filter": getattr(base_cfg, "peak_pre_filter", "none"),
            "event_auc_baseline": base_cfg.event_auc_baseline,
        }

    def _sync_event_feature_controls_from_active_baseline(self) -> None:
        """Sync the full main-run event-feature subsection to active baseline defaults."""
        required_attrs = (
            "_event_signal_combo",
            "_peak_method_combo",
            "_peak_k_edit",
            "_peak_pct_edit",
            "_peak_abs_edit",
            "_peak_dist_edit",
            "_peak_min_prominence_k_edit",
            "_peak_min_width_sec_edit",
            "_peak_pre_filter_combo",
            "_event_auc_combo",
        )
        if any(not hasattr(self, attr) for attr in required_attrs):
            return

        base_cfg = self._active_baseline_config()
        defaults = {
            "event_signal": str(base_cfg.event_signal),
            "peak_threshold_method": str(base_cfg.peak_threshold_method),
            "peak_threshold_k": float(base_cfg.peak_threshold_k),
            "peak_threshold_percentile": float(base_cfg.peak_threshold_percentile),
            "peak_threshold_abs": float(base_cfg.peak_threshold_abs),
            "peak_min_distance_sec": float(base_cfg.peak_min_distance_sec),
            "peak_min_prominence_k": float(getattr(base_cfg, "peak_min_prominence_k", 0.0)),
            "peak_min_width_sec": float(getattr(base_cfg, "peak_min_width_sec", 0.0)),
            "peak_pre_filter": str(getattr(base_cfg, "peak_pre_filter", "none")),
            "event_auc_baseline": str(base_cfg.event_auc_baseline),
        }

        def _sync_combo(combo: QComboBox, value_text: str) -> None:
            if combo.currentText().strip() == value_text:
                return
            idx = combo.findText(value_text)
            if idx < 0:
                return
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(blocked)

        def _sync_numeric(edit: QLineEdit, value: float) -> None:
            current_text = edit.text().strip()
            try:
                if current_text and float(current_text) == float(value):
                    return
            except ValueError:
                pass

            target = str(float(value))
            if current_text == target:
                return

            blocked = edit.blockSignals(True)
            edit.setText(target)
            edit.blockSignals(blocked)

        _sync_combo(self._event_signal_combo, defaults["event_signal"])
        _sync_combo(self._peak_method_combo, defaults["peak_threshold_method"])
        _sync_numeric(self._peak_k_edit, defaults["peak_threshold_k"])
        _sync_numeric(self._peak_pct_edit, defaults["peak_threshold_percentile"])
        _sync_numeric(self._peak_abs_edit, defaults["peak_threshold_abs"])
        _sync_numeric(self._peak_dist_edit, defaults["peak_min_distance_sec"])
        _sync_numeric(self._peak_min_prominence_k_edit, defaults["peak_min_prominence_k"])
        _sync_numeric(self._peak_min_width_sec_edit, defaults["peak_min_width_sec"])
        _sync_combo(self._peak_pre_filter_combo, defaults["peak_pre_filter"])
        _sync_combo(self._event_auc_combo, defaults["event_auc_baseline"])
        self._update_adv_ev_visibility()

    def _sync_main_advanced_controls_from_active_baseline(self) -> None:
        """Sync main Advanced controls to the currently active baseline config source."""
        base_cfg = self._active_baseline_config()

        def _sync_combo_by_data_or_text(combo: QComboBox, target_value: str) -> None:
            idx = combo.findData(target_value)
            if idx < 0:
                idx = combo.findText(str(target_value))
            if idx < 0 or combo.currentIndex() == idx:
                return
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(blocked)

        def _sync_checkbox(box: QCheckBox, target_value: bool) -> None:
            target = bool(target_value)
            if box.isChecked() == target:
                return
            blocked = box.blockSignals(True)
            box.setChecked(target)
            box.blockSignals(blocked)

        def _sync_spin_int(spin: QSpinBox, target_value: int) -> None:
            target = int(target_value)
            if spin.value() == target:
                return
            blocked = spin.blockSignals(True)
            spin.setValue(target)
            spin.blockSignals(blocked)

        def _sync_spin_float(spin: QDoubleSpinBox, target_value: float) -> None:
            target = float(target_value)
            if float(spin.value()) == target:
                return
            blocked = spin.blockSignals(True)
            spin.setValue(target)
            spin.blockSignals(blocked)

        def _sync_line_float(edit: QLineEdit, target_value: float) -> None:
            current_text = edit.text().strip()
            try:
                if current_text and float(current_text) == float(target_value):
                    return
            except ValueError:
                pass
            target_text = str(float(target_value))
            if current_text == target_text:
                return
            blocked = edit.blockSignals(True)
            edit.setText(target_text)
            edit.blockSignals(blocked)

        _sync_combo_by_data_or_text(
            self._dynamic_fit_mode_combo,
            normalize_dynamic_fit_mode(
                str(getattr(base_cfg, "dynamic_fit_mode", "rolling_local_regression"))
            ),
        )
        _sync_checkbox(
            self._baseline_subtract_before_fit_cb,
            bool(getattr(base_cfg, "baseline_subtract_before_fit", False)),
        )
        _sync_line_float(self._window_sec_edit, float(getattr(base_cfg, "window_sec", 60.0)))
        _sync_spin_int(
            self._min_samples_per_window_spin,
            max(1, int(getattr(base_cfg, "min_samples_per_window", 1))),
        )
        _sync_spin_int(
            self._robust_event_reject_max_iters_spin,
            int(getattr(base_cfg, "robust_event_reject_max_iters", 3)),
        )
        _sync_spin_float(
            self._robust_event_reject_residual_z_spin,
            float(getattr(base_cfg, "robust_event_reject_residual_z_thresh", 3.5)),
        )
        _sync_spin_float(
            self._robust_event_reject_local_var_window_spin,
            float(getattr(base_cfg, "robust_event_reject_local_var_window_sec", 10.0) or 10.0),
        )
        robust_ratio = getattr(base_cfg, "robust_event_reject_local_var_ratio_thresh", None)
        _sync_checkbox(
            self._robust_event_reject_local_var_ratio_enable_cb,
            bool(robust_ratio is not None),
        )
        _sync_spin_float(
            self._robust_event_reject_local_var_ratio_spin,
            float(robust_ratio if robust_ratio is not None else 3.0),
        )
        _sync_spin_float(
            self._robust_event_reject_min_keep_fraction_spin,
            float(getattr(base_cfg, "robust_event_reject_min_keep_fraction", 0.5)),
        )
        _sync_spin_float(
            self._adaptive_event_gate_residual_z_spin,
            float(getattr(base_cfg, "adaptive_event_gate_residual_z_thresh", 3.5)),
        )
        _sync_spin_float(
            self._adaptive_event_gate_local_var_window_spin,
            float(getattr(base_cfg, "adaptive_event_gate_local_var_window_sec", 10.0) or 10.0),
        )
        adaptive_ratio = getattr(base_cfg, "adaptive_event_gate_local_var_ratio_thresh", None)
        _sync_checkbox(
            self._adaptive_event_gate_local_var_ratio_enable_cb,
            bool(adaptive_ratio is not None),
        )
        _sync_spin_float(
            self._adaptive_event_gate_local_var_ratio_spin,
            float(adaptive_ratio if adaptive_ratio is not None else 3.0),
        )
        _sync_spin_float(
            self._adaptive_event_gate_smooth_window_spin,
            float(getattr(base_cfg, "adaptive_event_gate_smooth_window_sec", 60.0)),
        )
        _sync_spin_float(
            self._adaptive_event_gate_min_trust_fraction_spin,
            float(getattr(base_cfg, "adaptive_event_gate_min_trust_fraction", 0.5)),
        )
        _sync_combo_by_data_or_text(
            self._adaptive_event_gate_freeze_interp_combo,
            str(getattr(base_cfg, "adaptive_event_gate_freeze_interp_method", "linear_hold"))
            .strip()
            or "linear_hold",
        )
        _sync_line_float(self._lowpass_hz_edit, float(getattr(base_cfg, "lowpass_hz", 1.0)))
        _sync_combo_by_data_or_text(self._baseline_method_combo, str(base_cfg.baseline_method))
        _sync_line_float(
            self._baseline_percentile_edit,
            float(getattr(base_cfg, "baseline_percentile", 10.0)),
        )
        _sync_line_float(
            self._f0_min_value_edit,
            float(getattr(base_cfg, "f0_min_value", 1e-9)),
        )
        _sync_combo_by_data_or_text(
            self._tonic_output_mode_combo,
            normalize_tonic_output_mode(
                str(getattr(base_cfg, "tonic_output_mode", TONIC_OUTPUT_MODE_PRESERVE_RAW))
            ),
        )
        _sync_combo_by_data_or_text(
            self._tonic_timeline_mode_combo,
            normalize_tonic_timeline_mode(
                str(getattr(base_cfg, "tonic_timeline_mode", TONIC_TIMELINE_MODE_REAL_ELAPSED))
            ),
        )

        self._sync_event_feature_controls_from_active_baseline()
        self._on_dynamic_fit_mode_changed()
        self._update_adv_prep_visibility()

    def _update_config_source_ui(self) -> None:
        """Enable custom config widgets only when advanced custom mode is active."""
        use_custom = self._is_custom_config_enabled()
        self._config_path.setEnabled(use_custom)
        self._config_browse_btn.setEnabled(use_custom)

        if use_custom:
            cfg = self._config_path.text().strip() or "(not set)"
            self._active_config_source_label.setText(
                f"Active baseline source: custom YAML ({cfg})"
            )
            self._set_status_label_style(self._active_config_source_label, "info")
        else:
            self._active_config_source_label.setText(
                f"Active baseline source: lab standard default ({self._lab_default_config_path})"
            )
            self._set_status_label_style(self._active_config_source_label, "ready")
        should_sync_main_advanced = True
        if use_custom:
            cfg = self._config_path.text().strip()
            should_sync_main_advanced = bool(cfg and os.path.isfile(cfg))
        if should_sync_main_advanced:
            self._sync_main_advanced_controls_from_active_baseline()
        if hasattr(self, "_effective_summary_label"):
            self._refresh_effective_run_summary()

    def _build_run_spec(self, validate_only: bool = False) -> RunSpec:
        """Create a RunSpec from current widget values for a real run.

        Computes run_dir from out_base + run_id and sets
        self._current_run_dir.  Used exclusively by _build_argv()
        for validate and run operations.

        Discovery and preview use _build_discovery_spec() instead.
        """
        t_spec = self._timing_start("build_run_spec", extra=f"validate_only={validate_only}")
        out_base = self._output_dir.text().strip()
        run_id = _generate_run_id()
        run_dir = os.path.join(out_base, run_id)
        self._current_run_dir = run_dir

        user_set = []

        # Parse optional numeric fields
        sph_text = self._sph_edit.text().strip()
        sph_val = int(sph_text) if sph_text else None
        self._track_if_nonempty("sessions_per_hour", sph_text, user_set)

        dur_text = self._duration_edit.text().strip()
        dur_val = float(dur_text) if dur_text else None
        self._track_if_nonempty("session_duration_s", dur_text, user_set)

        timeline_anchor_mode_val = self._timeline_anchor_mode_value()
        self._track_if_changed("timeline_anchor_mode", timeline_anchor_mode_val, "civil", user_set)
        fixed_daily_anchor_clock_val = None
        fixed_anchor_edit = getattr(self, "_fixed_daily_anchor_time_edit", None)
        if timeline_anchor_mode_val == "fixed_daily_anchor" and fixed_anchor_edit is not None:
            fixed_daily_anchor_clock_val = fixed_anchor_edit.text().strip() or None
            self._track_if_nonempty(
                "fixed_daily_anchor_clock",
                fixed_daily_anchor_clock_val or "",
                user_set,
            )

        smooth = self._smooth_spin.value()
        self._track_if_changed("smooth_window_s", smooth, 1.0, user_set)

        plot_mode_full = self._plotting_mode_combo.currentText() == "Full"
        sig_iso_render_mode_text = "full" if plot_mode_full else "qc"
        sig_iso_render_mode_val = None if sig_iso_render_mode_text == "qc" else sig_iso_render_mode_text
        self._track_if_changed("sig_iso_render_mode", sig_iso_render_mode_text, "qc", user_set)

        dff_render_mode_text = "full" if plot_mode_full else "qc"
        dff_render_mode_val = None if dff_render_mode_text == "qc" else dff_render_mode_text
        self._track_if_changed("dff_render_mode", dff_render_mode_text, "qc", user_set)

        stacked_render_mode_text = "full" if plot_mode_full else "qc"
        stacked_render_mode_val = None if stacked_render_mode_text == "qc" else stacked_render_mode_text
        self._track_if_changed("stacked_render_mode", stacked_render_mode_text, "qc", user_set)

        fmt = self._format_combo.currentText()
        self._track_if_changed("format", fmt, "auto", user_set)

        if validate_only:
            user_set.append("validate_only")

        # --- Mode ---
        mode_text = self._mode_combo.currentText()
        if mode_text == "both":
            mode_val = None
        else:
            mode_val = mode_text
            user_set.append("mode")

        # --- Traces-only (--traces-only CLI flag) ---
        traces_only = self._traces_only_cb.isChecked()
        if traces_only:
            user_set.append("traces_only")

        # --- Preview first N (--preview-first-n CLI flag) ---
        preview_first_n = None
        if self._preview_enabled_cb.isChecked():
            preview_first_n = self._preview_n_spin.value()
            user_set.append("preview_first_n")


        # --- Representative session index (0-based, from discovery) ---
        rep_session_idx = None
        if self._rep_session_combo.currentIndex() > 0:
            # Index 0 is "(auto)"; session indices start at 1 in combo
            rep_session_idx = self._rep_session_combo.currentIndex() - 1
            user_set.append("representative_session_index")

        # --- ROI selection (include vs exclude) ---
        include_roi_ids = None
        exclude_roi_ids = None
        if self._discovery_cache is not None:
            all_rois = [r["roi_id"] for r in self._discovery_cache.get("rois", [])]
            checked = []
            for i in range(self._roi_list.count()):
                item = self._roi_list.item(i)
                if item.checkState() == Qt.Checked:
                    checked.append(item.text())
            if len(checked) == len(all_rois):
                include_roi_ids = None
            elif len(checked) == 0:
                include_roi_ids = []
                user_set.append("include_roi_ids")
            else:
                include_roi_ids = checked
                user_set.append("include_roi_ids")

        # --- Config Overrides ---
        base_cfg = self._active_baseline_config()
        config_overrides = {}
        if is_isosbestic_active(mode_text):
            fit_mode = self._selected_dynamic_fit_mode()
            default_fit_mode = normalize_dynamic_fit_mode(
                str(getattr(base_cfg, "dynamic_fit_mode", "rolling_local_regression"))
            )
            if fit_mode != default_fit_mode:
                config_overrides["dynamic_fit_mode"] = fit_mode

            default_dict = {
                "window_sec": base_cfg.window_sec,
                "step_sec": base_cfg.step_sec,
                "min_valid_windows": base_cfg.min_valid_windows,
                "r_low": base_cfg.r_low,
                "r_high": base_cfg.r_high,
                "g_min": base_cfg.g_min,
                # Enforce dynamic minimum mapping matching GUI constraint default
                "min_samples_per_window": max(1, base_cfg.min_samples_per_window),
            }
            if is_rolling_dynamic_fit_mode(fit_mode):
                overrides, _ = parse_and_validate_isosbestic_knobs(
                    self._window_sec_edit.text(),
                    "",
                    int(base_cfg.min_valid_windows),
                    "",
                    "",
                    "",
                    self._min_samples_per_window_spin.value(),
                    defaults=default_dict,
                )
                if overrides is not None:
                    changed_overrides = compute_isosbestic_overrides_user_changed(overrides, default_dict)
                    config_overrides.update(changed_overrides)
                baseline_subtract = bool(
                    getattr(self, "_baseline_subtract_before_fit_cb", None)
                    and self._baseline_subtract_before_fit_cb.isChecked()
                )
                default_baseline_subtract = bool(
                    getattr(base_cfg, "baseline_subtract_before_fit", False)
                )
                if baseline_subtract != default_baseline_subtract:
                    config_overrides["baseline_subtract_before_fit"] = baseline_subtract
            elif is_robust_event_reject_mode(fit_mode):
                robust_defaults = {
                    "robust_event_reject_max_iters": int(
                        getattr(base_cfg, "robust_event_reject_max_iters", 3)
                    ),
                    "robust_event_reject_residual_z_thresh": float(
                        getattr(base_cfg, "robust_event_reject_residual_z_thresh", 3.5)
                    ),
                    "robust_event_reject_local_var_window_sec": float(
                        getattr(base_cfg, "robust_event_reject_local_var_window_sec", 10.0) or 10.0
                    ),
                    "robust_event_reject_local_var_ratio_thresh": getattr(
                        base_cfg, "robust_event_reject_local_var_ratio_thresh", None
                    ),
                    "robust_event_reject_min_keep_fraction": float(
                        getattr(base_cfg, "robust_event_reject_min_keep_fraction", 0.5)
                    ),
                }
                robust_overrides, err = parse_and_validate_robust_event_reject_knobs(
                    self._robust_event_reject_max_iters_spin.value(),
                    self._robust_event_reject_residual_z_spin.value(),
                    self._robust_event_reject_local_var_window_spin.value(),
                    self._robust_event_reject_local_var_ratio_enable_cb.isChecked(),
                    self._robust_event_reject_local_var_ratio_spin.value(),
                    self._robust_event_reject_min_keep_fraction_spin.value(),
                    defaults=robust_defaults,
                )
                if robust_overrides is not None and err is None:
                    for key, value in robust_overrides.items():
                        if key in robust_defaults and value != robust_defaults[key]:
                            config_overrides[key] = value
            elif is_adaptive_event_gated_mode(fit_mode):
                adaptive_defaults = {
                    "adaptive_event_gate_residual_z_thresh": float(
                        getattr(base_cfg, "adaptive_event_gate_residual_z_thresh", 3.5)
                    ),
                    "adaptive_event_gate_local_var_window_sec": float(
                        getattr(base_cfg, "adaptive_event_gate_local_var_window_sec", 10.0) or 10.0
                    ),
                    "adaptive_event_gate_local_var_ratio_thresh": getattr(
                        base_cfg, "adaptive_event_gate_local_var_ratio_thresh", None
                    ),
                    "adaptive_event_gate_smooth_window_sec": float(
                        getattr(base_cfg, "adaptive_event_gate_smooth_window_sec", 60.0)
                    ),
                    "adaptive_event_gate_min_trust_fraction": float(
                        getattr(base_cfg, "adaptive_event_gate_min_trust_fraction", 0.5)
                    ),
                    "adaptive_event_gate_freeze_interp_method": str(
                        getattr(base_cfg, "adaptive_event_gate_freeze_interp_method", "linear_hold")
                    ).strip()
                    or "linear_hold",
                }
                adaptive_overrides, err = parse_and_validate_adaptive_event_gated_knobs(
                    self._adaptive_event_gate_residual_z_spin.value(),
                    self._adaptive_event_gate_local_var_window_spin.value(),
                    self._adaptive_event_gate_local_var_ratio_enable_cb.isChecked(),
                    self._adaptive_event_gate_local_var_ratio_spin.value(),
                    self._adaptive_event_gate_smooth_window_spin.value(),
                    self._adaptive_event_gate_min_trust_fraction_spin.value(),
                    str(
                        self._adaptive_event_gate_freeze_interp_combo.currentData()
                        or self._adaptive_event_gate_freeze_interp_combo.currentText()
                        or "linear_hold"
                    ),
                    defaults=adaptive_defaults,
                )
                if adaptive_overrides is not None and err is None:
                    for key, value in adaptive_overrides.items():
                        if key == "adaptive_event_gate_freeze_interp_method":
                            config_overrides[key] = str(value)
                        elif key in adaptive_defaults and value != adaptive_defaults[key]:
                            config_overrides[key] = value

        selected_tonic_output_mode = normalize_tonic_output_mode(
            str(
                getattr(self, "_tonic_output_mode_combo", None).currentData()
                if hasattr(self, "_tonic_output_mode_combo")
                else getattr(base_cfg, "tonic_output_mode", TONIC_OUTPUT_MODE_PRESERVE_RAW)
            )
        )
        default_tonic_output_mode = normalize_tonic_output_mode(
            str(getattr(base_cfg, "tonic_output_mode", TONIC_OUTPUT_MODE_PRESERVE_RAW))
        )
        if selected_tonic_output_mode != default_tonic_output_mode:
            config_overrides["tonic_output_mode"] = selected_tonic_output_mode

        selected_tonic_timeline_mode = normalize_tonic_timeline_mode(
            str(
                getattr(self, "_tonic_timeline_mode_combo", None).currentData()
                if hasattr(self, "_tonic_timeline_mode_combo")
                else getattr(base_cfg, "tonic_timeline_mode", TONIC_TIMELINE_MODE_REAL_ELAPSED)
            )
        )
        default_tonic_timeline_mode = normalize_tonic_timeline_mode(
            str(getattr(base_cfg, "tonic_timeline_mode", TONIC_TIMELINE_MODE_REAL_ELAPSED))
        )
        if selected_tonic_timeline_mode != default_tonic_timeline_mode:
            config_overrides["tonic_timeline_mode"] = selected_tonic_timeline_mode

        # Preprocessing + Baseline overrides
        default_prep_dict = {
            "lowpass_hz": base_cfg.lowpass_hz,
            "baseline_method": base_cfg.baseline_method,
            "baseline_percentile": base_cfg.baseline_percentile,
            "f0_min_value": base_cfg.f0_min_value,
        }
        prep_overrides, _ = parse_and_validate_preproc_baseline_knobs(
            self._lowpass_hz_edit.text(),
            self._baseline_method_combo.currentText(),
            self._baseline_percentile_edit.text(),
            self._f0_min_value_edit.text(),
            defaults=default_prep_dict,
        )
        if prep_overrides is not None:
            changed_prep_overrides = compute_overrides_user_changed(prep_overrides, default_prep_dict)
            config_overrides.update(changed_prep_overrides)

        # Event + Feature overrides
        default_ev_dict = self._event_feature_defaults_from_active_baseline()
        ev_overrides, _ = parse_and_validate_event_feature_knobs(
            self._event_signal_combo.currentText(),
            self._peak_method_combo.currentText(),
            self._peak_k_edit.text(),
            self._peak_pct_edit.text(),
            self._peak_abs_edit.text(),
            self._peak_dist_edit.text(),
            self._event_auc_combo.currentText(),
            defaults=default_ev_dict,
            peak_pre_filter_text=self._peak_pre_filter_combo.currentText(),
            peak_prominence_k_str=self._peak_min_prominence_k_edit.text(),
            peak_width_sec_str=self._peak_min_width_sec_edit.text(),
        )
        if ev_overrides is not None:
            changed_ev_overrides = compute_overrides_user_changed(ev_overrides, default_ev_dict)
            config_overrides.update(changed_ev_overrides)

        # Ensure effective config tracks the selected RWD dataset contract.
        # This prevents stale/default baseline config values (e.g., fs/suffix/time-col)
        # from conflicting with the currently selected input data.
        t_contract = self._timing_start("dataset_contract_inference")
        data_contract_overrides = self._infer_dataset_contract_overrides(fmt)
        self._timing_end("dataset_contract_inference", t_contract)
        if not isinstance(data_contract_overrides, dict):
            data_contract_overrides = {}

        if self._is_custom_config_enabled():
            user_set.append("config_source_path")

        spec = RunSpec(
            input_dir=self._input_dir.text().strip(),
            run_dir=run_dir,
            format=fmt,
            validate_only=validate_only,
            sessions_per_hour=sph_val,
            session_duration_s=dur_val,
            timeline_anchor_mode=timeline_anchor_mode_val,
            fixed_daily_anchor_clock=fixed_daily_anchor_clock_val,
            smooth_window_s=smooth,
            sig_iso_render_mode=sig_iso_render_mode_val,
            dff_render_mode=dff_render_mode_val,
            stacked_render_mode=stacked_render_mode_val,
            traces_only=traces_only,
            preview_first_n=preview_first_n,
            representative_session_index=rep_session_idx,
            include_roi_ids=include_roi_ids,
            exclude_roi_ids=exclude_roi_ids,
            config_source_path=self._active_config_source_path(),
            config_overrides=config_overrides,
            data_contract_overrides=data_contract_overrides,
            gui_version="1.0.0",
            timestamp_local=datetime.now().isoformat(),
            mode=mode_val,
            user_set_fields=user_set,
        )
        self._timing_end("build_run_spec", t_spec)
        return spec

    def _build_argv(self, validate_only: bool = False, overwrite: bool = False) -> list:
        """Construct argv via RunSpec.

        Writes into run_dir:
          1. config_effective.yaml (derived config)
          2. gui_run_spec.json (intent record)
          3. command_invoked.txt (exact argv)

        Validates derived config before returning.
        Uses --out <run_dir> mode.
        """
        t_argv = self._timing_start("build_argv", extra=f"validate_only={validate_only} overwrite={overwrite}")
        t_spec = self._timing_start("build_run_spec_from_build_argv")
        spec = self._build_run_spec(validate_only=validate_only)
        self._timing_end("build_run_spec_from_build_argv", t_spec)
        spec.overwrite = overwrite
        run_dir = self._current_run_dir
        os.makedirs(run_dir, exist_ok=True)

        # Write derived config and validate it
        t_cfg = self._timing_start("generate_derived_config")
        config_path = spec.generate_derived_config(run_dir)
        self._timing_end("generate_derived_config", t_cfg)
        t_cfg_valid = self._timing_start("validate_effective_config")
        RunSpec.validate_effective_config(config_path)
        self._timing_end("validate_effective_config", t_cfg_valid)

        # Build argv
        t_build_runner = self._timing_start("build_runner_argv")
        argv = spec.build_runner_argv()
        self._timing_end("build_runner_argv", t_build_runner)

        # Write intent record and command log
        t_write_spec = self._timing_start("write_gui_run_spec")
        spec.write_gui_run_spec(run_dir)
        self._timing_end("write_gui_run_spec", t_write_spec)
        t_write_cmd = self._timing_start("write_command_invoked")
        spec.write_command_invoked(run_dir, argv)
        self._timing_end("write_command_invoked", t_write_cmd)

        self._timing_end("build_argv", t_argv)
        return argv

    # ==================================================================
    # Input validation (cheap, GUI-side only)
    # ==================================================================

    def _validate_gui_inputs(self) -> str | None:
        """Return error message if inputs are obviously wrong, else None."""
        input_dir = self._input_dir.text().strip()
        if not input_dir:
            return "Input directory is required."
        if not os.path.isdir(input_dir):
            return f"Input directory does not exist: {input_dir}"

        if self._is_custom_config_enabled():
            config = self._config_path.text().strip()
            if not config:
                return "Custom Config YAML is enabled, but no path was provided."
            if not os.path.isfile(config):
                return f"Custom config file does not exist: {config}"
        else:
            if not os.path.isfile(self._lab_default_config_path):
                return (
                    "Default lab baseline config is missing: "
                    f"{self._lab_default_config_path}"
                )

        out_dir = self._output_dir.text().strip()
        if not out_dir:
            return "Output directory path is required."

        sph = self._sph_edit.text().strip()
        if sph:
            try:
                v = int(sph)
                if v < 1:
                    return "Sessions/Hour must be >= 1."
            except ValueError:
                return f"Sessions/Hour must be an integer, got: '{sph}'"

        dur = self._duration_edit.text().strip()
        if dur:
            try:
                v = float(dur)
                if v <= 0:
                    return "Session Duration must be > 0."
            except ValueError:
                return f"Session Duration must be a number, got: '{dur}'"

        anchor_mode = self._timeline_anchor_mode_value()
        if anchor_mode == "fixed_daily_anchor":
            clock = self._fixed_daily_anchor_time_edit.text().strip()
            if not clock:
                return (
                    "Fixed Daily Anchor Time is required when timeline anchor mode is "
                    "Fixed daily anchor."
                )
            if not _ANCHOR_CLOCK_RE.fullmatch(clock):
                return (
                    "Fixed Daily Anchor Time must be HH:MM or HH:MM:SS "
                    f"(got '{clock}')."
                )

        # ROI selection: include-only semantics (checked == included)
        if self._discovery_cache is not None:
            total_rois = self._roi_list.count()
            checked_count = sum(
                1 for i in range(total_rois)
                if self._roi_list.item(i).checkState() == Qt.Checked
            )
            if checked_count == 0:
                return "No ROIs selected. Check at least one ROI."

        # Representative session must remain valid under preview truncation.
        rep_session_idx = None
        if self._rep_session_combo.currentIndex() > 0:
            rep_session_idx = self._rep_session_combo.currentIndex() - 1
        preview_n = self._preview_n_spin.value() if self._preview_enabled_cb.isChecked() else None
        rep_preview_err = validate_representative_index_preview_compatibility(rep_session_idx, preview_n)
        if rep_preview_err:
            return rep_preview_err

        fmt = self._format_combo.currentText()
        if not fmt or fmt not in FORMAT_CHOICES:
            return f"Invalid Format: '{fmt}'. Must be one of {FORMAT_CHOICES}."

        base_cfg = self._active_baseline_config()

        if is_isosbestic_active(self._mode_combo.currentText()):
            fit_mode = self._selected_dynamic_fit_mode()
            if is_rolling_dynamic_fit_mode(fit_mode):
                default_dict = {
                    "window_sec": base_cfg.window_sec,
                    "step_sec": base_cfg.step_sec,
                    "min_valid_windows": base_cfg.min_valid_windows,
                    "r_low": base_cfg.r_low,
                    "r_high": base_cfg.r_high,
                    "g_min": base_cfg.g_min,
                    # Enforce dynamic minimum mapping matching GUI constraint default
                    "min_samples_per_window": max(1, base_cfg.min_samples_per_window),
                }
                _, err = parse_and_validate_isosbestic_knobs(
                    self._window_sec_edit.text(),
                    "",
                    int(base_cfg.min_valid_windows),
                    "",
                    "",
                    "",
                    self._min_samples_per_window_spin.value(),
                    defaults=default_dict,
                )
                if err:
                    return err
            elif is_robust_event_reject_mode(fit_mode):
                robust_defaults = {
                    "robust_event_reject_max_iters": int(
                        getattr(base_cfg, "robust_event_reject_max_iters", 3)
                    ),
                    "robust_event_reject_residual_z_thresh": float(
                        getattr(base_cfg, "robust_event_reject_residual_z_thresh", 3.5)
                    ),
                    "robust_event_reject_local_var_window_sec": float(
                        getattr(base_cfg, "robust_event_reject_local_var_window_sec", 10.0) or 10.0
                    ),
                    "robust_event_reject_local_var_ratio_thresh": getattr(
                        base_cfg, "robust_event_reject_local_var_ratio_thresh", None
                    ),
                    "robust_event_reject_min_keep_fraction": float(
                        getattr(base_cfg, "robust_event_reject_min_keep_fraction", 0.5)
                    ),
                }
                _, err = parse_and_validate_robust_event_reject_knobs(
                    self._robust_event_reject_max_iters_spin.value(),
                    self._robust_event_reject_residual_z_spin.value(),
                    self._robust_event_reject_local_var_window_spin.value(),
                    self._robust_event_reject_local_var_ratio_enable_cb.isChecked(),
                    self._robust_event_reject_local_var_ratio_spin.value(),
                    self._robust_event_reject_min_keep_fraction_spin.value(),
                    defaults=robust_defaults,
                )
                if err:
                    return err
            elif is_adaptive_event_gated_mode(fit_mode):
                adaptive_defaults = {
                    "adaptive_event_gate_residual_z_thresh": float(
                        getattr(base_cfg, "adaptive_event_gate_residual_z_thresh", 3.5)
                    ),
                    "adaptive_event_gate_local_var_window_sec": float(
                        getattr(base_cfg, "adaptive_event_gate_local_var_window_sec", 10.0) or 10.0
                    ),
                    "adaptive_event_gate_local_var_ratio_thresh": getattr(
                        base_cfg, "adaptive_event_gate_local_var_ratio_thresh", None
                    ),
                    "adaptive_event_gate_smooth_window_sec": float(
                        getattr(base_cfg, "adaptive_event_gate_smooth_window_sec", 60.0)
                    ),
                    "adaptive_event_gate_min_trust_fraction": float(
                        getattr(base_cfg, "adaptive_event_gate_min_trust_fraction", 0.5)
                    ),
                    "adaptive_event_gate_freeze_interp_method": str(
                        getattr(base_cfg, "adaptive_event_gate_freeze_interp_method", "linear_hold")
                    ).strip()
                    or "linear_hold",
                }
                _, err = parse_and_validate_adaptive_event_gated_knobs(
                    self._adaptive_event_gate_residual_z_spin.value(),
                    self._adaptive_event_gate_local_var_window_spin.value(),
                    self._adaptive_event_gate_local_var_ratio_enable_cb.isChecked(),
                    self._adaptive_event_gate_local_var_ratio_spin.value(),
                    self._adaptive_event_gate_smooth_window_spin.value(),
                    self._adaptive_event_gate_min_trust_fraction_spin.value(),
                    str(
                        self._adaptive_event_gate_freeze_interp_combo.currentData()
                        or self._adaptive_event_gate_freeze_interp_combo.currentText()
                        or "linear_hold"
                    ),
                    defaults=adaptive_defaults,
                )
                if err:
                    return err

        # Preprocessing + Baseline validation
        default_prep_dict = {
            "lowpass_hz": base_cfg.lowpass_hz,
            "baseline_method": base_cfg.baseline_method,
            "baseline_percentile": base_cfg.baseline_percentile,
            "f0_min_value": base_cfg.f0_min_value,
        }
        _, err = parse_and_validate_preproc_baseline_knobs(
            self._lowpass_hz_edit.text(),
            self._baseline_method_combo.currentText(),
            self._baseline_percentile_edit.text(),
            self._f0_min_value_edit.text(),
            defaults=default_prep_dict,
        )
        if err:
            return err

        # Event + Feature validation
        default_ev_dict = self._event_feature_defaults_from_active_baseline()
        _, err = parse_and_validate_event_feature_knobs(
            self._event_signal_combo.currentText(),
            self._peak_method_combo.currentText(),
            self._peak_k_edit.text(),
            self._peak_pct_edit.text(),
            self._peak_abs_edit.text(),
            self._peak_dist_edit.text(),
            self._event_auc_combo.currentText(),
            defaults=default_ev_dict,
            peak_pre_filter_text=self._peak_pre_filter_combo.currentText(),
            peak_prominence_k_str=self._peak_min_prominence_k_edit.text(),
            peak_width_sec_str=self._peak_min_width_sec_edit.text(),
        )
        if err:
            return err

        return None
    # ==================================================================
    # Config change handler — resets validation
    # ==================================================================

    def _on_config_changed(self):
        """Any config widget change invalidates prior validation."""
        self._validation_passed = False
        self._validated_run_signature = None
        self._refresh_splitter_workspace_policy()
        self._update_button_states()

    # ==================================================================
    # Button handlers
    # ==================================================================

    def _build_discovery_spec(self) -> RunSpec:
        """Build a lightweight RunSpec for discovery/preview only.

        Does NOT compute a run_id, does NOT touch _current_run_dir,
        and does NOT depend on the output directory widget.
        run_dir is left empty since discovery never writes to it.
        """
        t_discovery_spec = self._timing_start("build_discovery_spec")
        fmt = self._format_combo.currentText()
        t_contract = self._timing_start("dataset_contract_inference_discovery")
        data_contract_overrides = self._infer_dataset_contract_overrides(fmt)
        self._timing_end("dataset_contract_inference_discovery", t_contract)
        if not isinstance(data_contract_overrides, dict):
            data_contract_overrides = {}
        spec = RunSpec(
            input_dir=self._input_dir.text().strip(),
            run_dir="",
            format=fmt,
            config_source_path=self._active_config_source_path(),
            data_contract_overrides=data_contract_overrides,
        )
        self._timing_end("build_discovery_spec", t_discovery_spec)
        return spec

    def _on_preview_config(self):
        """Show a read-only dialog with the derived config YAML preview."""
        config_path = self._active_config_source_path()
        if not config_path or not os.path.isfile(config_path):
            if self._is_custom_config_enabled():
                msg = "Select a valid custom Config YAML first."
            else:
                msg = f"Default lab baseline config is missing:\n{config_path}"
            QMessageBox.warning(self, "No Config", msg)
            return

        spec = self._build_discovery_spec()
        preview_text = spec.get_derived_config_preview()

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Derived Config Preview")
        dlg.setText("This is the exact config YAML that will be passed to the runner:")
        dlg.setDetailedText(preview_text)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.exec()

    def _on_discover(self):
        """Resolve available ROIs via the runner backend and populate ROI selection."""
        input_dir = self._input_dir.text().strip()
        config_path = self._active_config_source_path()

        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "ROI Selection Error",
                                "Select a valid input directory first.")
            return
        if not config_path or not os.path.isfile(config_path):
            if self._is_custom_config_enabled():
                msg = "Select a valid custom config YAML first."
            else:
                msg = f"Default lab baseline config is missing:\n{config_path}"
            QMessageBox.warning(self, "ROI Selection Error", msg)
            return

        spec = self._build_discovery_spec()
        try:
            result = spec.run_discovery()
        except Exception as e:
            self._discovery_cache = None
            self._discovery_summary.setText("Discovery failed.")
            self._sessions_list.clear()
            self._roi_list.clear()
            if hasattr(self, "_roi_selection_container"):
                self._roi_selection_container.setVisible(False)
            self._rep_session_combo.clear()
            self._rep_session_combo.addItem("(auto)")
            self._append_log(f"Discovery error: {e}")
            QMessageBox.critical(self, "ROI Selection Failed", str(e))
            return

        self._discovery_cache = result
        self._populate_discovery_ui(result)
        self._append_log(
            f"ROI selection ready: {len(result.get('rois', []))} ROIs found "
            f"(format={result.get('resolved_format', '?')})."
        )

    def _populate_discovery_ui(self, disco: dict):
        """Fill ROI checklist and compatibility session data from discovery JSON."""
        if hasattr(self, "_roi_selection_container"):
            self._roi_selection_container.setVisible(True)

        # Summary
        n_total = disco.get("n_total_discovered", 0)
        n_preview = disco.get("n_preview", 0)
        fmt = disco.get("resolved_format", "?")
        self._discovery_summary.setText(
            f"Format: {fmt}  |  Sessions: {n_total}  |  Preview: {n_preview}"
        )

        # Sessions list (read-only, exact order from discovery)
        self._sessions_list.clear()
        sessions = disco.get("sessions", [])
        for sess in sessions:
            sid = sess.get("session_id", "?")
            preview_flag = sess.get("included_in_preview", False)
            label = f"{sid}  {'[preview]' if preview_flag else ''}"
            self._sessions_list.addItem(label)

        # ROIs checklist (all checked by default, exact order)
        self._roi_list.clear()
        rois = disco.get("rois", [])
        self._roi_list.blockSignals(True)
        for roi in rois:
            roi_id = roi.get("roi_id", "?")
            item = QListWidgetItem(roi_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._roi_list.addItem(item)
        self._roi_list.blockSignals(False)

        # Representative session dropdown
        self._rep_session_combo.blockSignals(True)
        self._rep_session_combo.clear()
        self._rep_session_combo.addItem("(auto)")
        for sess in sessions:
            self._rep_session_combo.addItem(sess.get("session_id", "?"))
        self._rep_session_combo.blockSignals(False)

        # Discovery outcome changes run intent; force readiness/summary refresh.
        self._on_config_changed()

    def _on_roi_select_all(self):
        """Check all ROI items in the checklist."""
        for i in range(self._roi_list.count()):
            self._roi_list.item(i).setCheckState(Qt.Checked)

    def _on_roi_select_none(self):
        """Uncheck all ROI items in the checklist."""
        for i in range(self._roi_list.count()):
            self._roi_list.item(i).setCheckState(Qt.Unchecked)

    def _on_validate(self):
        self._timing_action = "validate"
        self._timing_click_monotonic = time.perf_counter()
        with self._busy_cursor_scope():
            t_handler = self._timing_start("button_validate_handler")
            t_validate_inputs = self._timing_start("validate_gui_inputs")
            err = self._validate_gui_inputs()
            self._timing_end("validate_gui_inputs", t_validate_inputs)
            if err:
                self._timing_end("button_validate_handler", t_handler, extra="aborted=invalid_inputs")
                QMessageBox.warning(self, "Validation Error", err)
                return

            t_preflight = self._timing_start("shared_preflight_setup")
            self._exit_complete_state_workspace()
            self._save_widgets_to_settings()
            self._apply_results_idle_placeholder()
            self._log_view.clear()
            self._timing_end("shared_preflight_setup", t_preflight)

            # Narrative start (Fix Readability 2: Append BEFORE starting followers/runner)
            self._append_run_log("--- Validate Only ---")
            t_build_argv = self._timing_start("build_argv_validate")
            argv = self._build_argv(validate_only=True)
            self._timing_end("build_argv_validate", t_build_argv)
            self._append_run_log(f"Run directory: {self._current_run_dir}")
            self._append_run_log(f"Config (temp): {os.path.join(self._current_run_dir, 'config_effective.yaml')}")

            self._validation_passed = False
            self._is_validate_only = True
            self._validate_stdout = []
            self._reset_status_flags(next_state=RunnerState.VALIDATING)

            # Validate->Run reuse tracking (Fix B1)
            self._validated_run_dir = None
            self._validated_gui_run_spec_json_path = None
            self._validated_config_effective_yaml_path = None
            self._validated_run_signature = None

            # run_dir already created by _build_argv -> _build_run_spec

            self._runner.set_run_dir(self._current_run_dir)
            t_status_follow = self._timing_start("start_status_follower")
            self._start_status_follower()
            self._timing_end("start_status_follower", t_status_follow)
            t_log_follow = self._timing_start("start_log_follower")
            self._start_log_follower(self._current_run_dir)
            self._timing_end("start_log_follower", t_log_follow)
            t_runner_start = self._timing_start("subprocess_launch_validate")
            self._runner.start(argv, state=RunnerState.VALIDATING)
            self._timing_end("subprocess_launch_validate", t_runner_start)
            self._timing_end("button_validate_handler", t_handler)

    def _on_run(self):
        self._timing_action = "run"
        self._timing_click_monotonic = time.perf_counter()
        with self._busy_cursor_scope():
            t_handler = self._timing_start("button_run_handler")
            t_validate_inputs = self._timing_start("validate_gui_inputs")
            err = self._validate_gui_inputs()
            self._timing_end("validate_gui_inputs", t_validate_inputs)
            if err:
                self._timing_end("button_run_handler", t_handler, extra="aborted=invalid_inputs")
                QMessageBox.warning(self, "Validation Error", err)
                return

            t_preflight = self._timing_start("shared_preflight_setup")
            self._exit_complete_state_workspace()
            self._save_widgets_to_settings()
            self._timing_end("shared_preflight_setup", t_preflight)

            # Build argv (also generates derived config + gui_run_spec.json)
            t_build_argv = self._timing_start("build_argv_run")
            argv = self._build_argv(validate_only=False, overwrite=True)
            self._timing_end("build_argv_run", t_build_argv)
            run_dir = self._current_run_dir

            # --- Fix B1v4: Validate->Run Consistency Policy ---
            # Same-directory reuse is abandoned due to handle conflicts on Windows and
            # destructive runner semantics. Instead, we use distinct directories but
            # ensure the user intent is still validly validated.
            try:
                t_sig = self._timing_start("compute_run_signature")
                current_sig = compute_run_signature(run_dir)
                self._timing_end("compute_run_signature", t_sig)
                t_validate_current = self._timing_start("is_validation_current")
                if not is_validation_current(self._validated_run_signature, current_sig):
                    self._timing_end("is_validation_current", t_validate_current, extra="result=False")
                    self._timing_end("button_run_handler", t_handler, extra="aborted=needs_revalidation")
                    QMessageBox.warning(
                        self,
                        "Re-validation Required",
                        "The current settings differ from the last successful validation. "
                        "Please run 'Validate Only' again before proceeding.",
                    )
                    return
                self._timing_end("is_validation_current", t_validate_current, extra="result=True")
            except Exception as e:
                self._timing_end("button_run_handler", t_handler, extra="aborted=signature_error")
                QMessageBox.warning(
                    self,
                    "Validation State Error",
                    f"Could not verify consistency with prior validation: {e}",
                )
                return

            # Clear past results
            self._apply_results_idle_placeholder()
            self._log_view.clear()

            # Narrative start (Fix Readability 2: Append BEFORE starting followers/runner)
            self._append_run_log("--- Starting Pipeline ---")
            self._append_run_log(f"Run directory: {run_dir}")
            self._append_run_log(f"Config: {os.path.join(run_dir, 'config_effective.yaml')}")

            self._is_validate_only = False
            self._validation_passed = False
            self._reset_status_flags(next_state=RunnerState.RUNNING)

            self._runner.set_run_dir(run_dir)
            t_status_follow = self._timing_start("start_status_follower")
            self._start_status_follower()
            self._timing_end("start_status_follower", t_status_follow)
            t_log_follow = self._timing_start("start_log_follower")
            self._start_log_follower(run_dir)
            self._timing_end("start_log_follower", t_log_follow)
            t_runner_start = self._timing_start("subprocess_launch_run")
            self._runner.start(argv, state=RunnerState.RUNNING)
            self._timing_end("subprocess_launch_run", t_runner_start)
            self._timing_end("button_run_handler", t_handler)

    def _on_cancel(self):
        self._runner.cancel()

    def _on_open_results(self):
        """Open a completed successful output directory into complete-state workspace."""
        selected = QFileDialog.getExistingDirectory(
            self, "Select Output Directory with MANIFEST.json",
            self._output_dir.text().strip(),
        )
        if not selected:
            return

        self._current_run_dir = selected
        self._output_dir.setText(selected)
        self._save_widgets_to_settings()

        is_successful_complete, evidence = is_successful_completed_run_dir(selected)
        if not is_successful_complete:
            self._append_run_log(
                f"Open Results blocked: selected directory is not a confirmed successful completed run. {evidence}"
            )
            QMessageBox.information(
                self,
                "Results Not Opened",
                "The selected directory is not confirmed as a successfully completed run.\n\n"
                "Choose a run folder with final-success metadata (status.json or MANIFEST).\n\n"
                f"Details: {evidence}",
            )
            self._apply_results_idle_placeholder()
            self._exit_complete_state_workspace()
            self._refresh_tuning_workspace_availability()
            self._state_str = RunnerState.IDLE.value
            self._ui_state = RunnerState.IDLE
            self._render_status_label()
            self._update_button_states()
            return

        self._append_run_log(f"--- Opening results from {selected} ---")

        loaded = self._report_viewer.load_report(selected)
        if loaded:
            self._is_validate_only = False
            self._validation_passed = False
            self._elapsed_timer.stop()
            self._run_started_monotonic = None
            self._state_str = RunnerState.SUCCESS.value
            self._ui_state = RunnerState.SUCCESS
            self._last_status_phase = "final"
            self._last_status_state = "success"
            self._last_status_msg = ""
            self._last_status_errors = []
            self._last_status_pct = 100
            self._refresh_status_from_disk_final()
            self._enter_complete_state_workspace()
            self._refresh_tuning_workspace_availability()
            self._append_run_log("Complete-state results workspace loaded.")
        else:
            self._append_run_log("Could not load complete-state workspace from selected directory.")
            self._exit_complete_state_workspace()
            self._refresh_tuning_workspace_availability()
        self._update_button_states()

    def _on_open_folder(self):
        """Open the current run_dir in the system file manager."""
        run_dir = self._current_run_dir
        if not run_dir or not os.path.isdir(run_dir):
            QMessageBox.information(
                self, "No Run Folder",
                "No run directory available to open."
            )
            return

        # Show MANIFEST summary if available
        manifest_path = os.path.join(run_dir, "MANIFEST.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    m = json.loads(fh.read())
                status = m.get("status", "unknown")
                run_id = m.get("run_id", "unknown")
                n_commands = len(m.get("commands", []))
                self._append_run_log(
                    f"Manifest status={status}, run_id={run_id}, "
                    f"commands={n_commands}"
                )
            except Exception:
                self._append_run_log("Could not parse MANIFEST.json")
        else:
            self._append_run_log("Manifest not created")

        _open_folder(run_dir)

    def _artifact_path_in_current_run(self, filename: str) -> str:
        """Absolute path for a key artifact in the current run directory."""
        if not self._current_run_dir:
            return ""
        return os.path.join(self._current_run_dir, filename)

    def _on_open_key_artifact(self, filename: str):
        """Open a key provenance artifact from the current run directory."""
        run_dir = self._current_run_dir
        if not run_dir or not os.path.isdir(run_dir):
            QMessageBox.information(
                self, "No Run Folder",
                "No run directory available. Run or open results first."
            )
            return

        artifact_path = self._artifact_path_in_current_run(filename)
        if not os.path.isfile(artifact_path):
            QMessageBox.information(
                self,
                "Artifact Not Available",
                f"{filename} is not present in the current run directory.\n\n{run_dir}",
            )
            return

        self._append_run_log(f"Opening {filename}: {artifact_path}")
        _open_file(artifact_path)

    # ==================================================================
    # Status follower integration
    # ==================================================================

    def _start_status_follower(self):
        """Create and start a StatusFollower for the current run_dir."""
        self._stop_status_follower()
        status_path = os.path.join(self._current_run_dir, "status.json")
        from gui.status_follower import StatusFollower
        self._status_follower = StatusFollower(status_path, poll_ms=500, parent=self)
        self._status_follower.status_received.connect(self._on_status)
        self._status_follower.parse_error.connect(self._on_status_parse_error)
        self._status_follower.status_warning.connect(self._on_status_warning)
        self._status_follower.start()

    def _stop_status_follower(self):
        """Stop and discard the status follower."""
        if hasattr(self, '_status_follower') and self._status_follower is not None:
            self._status_follower.stop()
            try:
                self._status_follower.status_received.disconnect(self._on_status)
                self._status_follower.status_warning.disconnect(self._on_status_warning)
            except (RuntimeError, TypeError):
                pass
            self._status_follower.deleteLater()
            self._status_follower = None

    def _start_log_follower(self, run_dir: str):
        """Create and start a LogFollower for the current run_dir."""
        self._stop_log_follower()
        self._log_follower = LogFollower(run_dir, poll_ms=500, parent=self)
        self._log_follower.line_received.connect(self._append_log)
        self._log_follower.start()

    def _stop_log_follower(self):
        """Stop and discard the log follower."""
        if hasattr(self, '_log_follower') and self._log_follower is not None:
            self._log_follower.stop()
            try:
                self._log_follower.line_received.disconnect(self._append_log)
            except (RuntimeError, TypeError):
                pass
            self._log_follower.deleteLater()
            self._log_follower = None

    def _reset_status_flags(self, next_state: RunnerState | None = None):
        """Reset status-derived fields at the start of each validate/run."""
        self._saw_cancel_status = False
        self._last_status_phase = "\u2014"
        self._last_status_state = "\u2014"
        self._last_status_duration = ""
        self._last_status_duration_sec = None
        self._last_status_errors = []
        self._last_status_msg = ""
        self._last_status_pct = None
        self._ui_progress_pct = 0
        self._run_started_monotonic = None
        self._last_elapsed_sec = 0.0
        self._elapsed_timer.stop()
        if next_state is not None:
            self._ui_state = next_state
            self._state_str = next_state.value
        self._refresh_splitter_workspace_policy()
        self._render_status_label()

    def _on_status_parse_error(self, msg: str):
        """Update UI to show partial write / reading state (non-critical)."""
        self._last_status_msg = msg
        self._render_status_label(is_updating=True)

    def _on_status_warning(self, msg: str):
        """Update UI to show unknown status warning from follower."""
        self._last_status_msg = msg
        self._render_status_label(is_updating=False)

    # ==================================================================
    # Runner signal handlers
    # ==================================================================

    def _on_run_started(self):
        t_started = self._timing_start("runner_started_signal")
        self._did_finalize_run_ui = False
        self._run_started_monotonic = time.monotonic()
        self._last_elapsed_sec = 0.0
        self._elapsed_first_tick_logged = False
        self._elapsed_timer.start()
        if self._gui_timing_enabled and self._timing_click_monotonic is not None:
            delay = time.perf_counter() - self._timing_click_monotonic
            self._emit_gui_timing(
                "END",
                "click_to_runner_started",
                elapsed_sec=delay,
                extra=f"action={self._timing_action or 'unknown'}",
            )
        if self._ui_state == RunnerState.VALIDATING:
            msg = "Validation in progress..."
        else:
            msg = "Run in progress..."
        self._report_viewer.set_running_message(self._results_running_placeholder_text(msg))
        self._refresh_results_workspace_summary()
        self._refresh_tuning_workspace_availability()
        self._update_button_states()
        self._timing_end("runner_started_signal", t_started)

    def _on_state_changed(self, state_str: str):
        """Update status label and button states on state transitions."""
        self._state_str = state_str
        try:
            self._ui_state = RunnerState(state_str)
            if self._ui_state in (RunnerState.RUNNING, RunnerState.VALIDATING):
                if self._run_started_monotonic is None:
                    self._run_started_monotonic = time.monotonic()
                if not self._elapsed_timer.isActive():
                    self._elapsed_timer.start()
            
            # Terminal state transition
            terminal_states = (
                RunnerState.SUCCESS, RunnerState.FAILED, 
                RunnerState.CANCELLED, RunnerState.FAIL_CLOSED
            )
            if self._ui_state in terminal_states:
                self._elapsed_timer.stop()
                self._stop_status_follower()
                self._stop_log_follower()
                self._finalize_run_ui()
                
        except ValueError:
            pass
        self._render_status_label()
        self._update_button_states()

    def _on_run_finished_failsafe(self, exit_code: int):
        """Backup handler to ensure finalization if state_changed was missed."""
        # Ensure finalization exactly once
        self._elapsed_timer.stop()
        self._stop_status_follower()
        self._stop_log_follower()
        self._finalize_run_ui()

    def _refresh_status_from_disk_final(self):
        """Authoritative read of status.json at run completion."""
        if not self._current_run_dir:
            return
            
        status_path = os.path.join(self._current_run_dir, "status.json")
        if not os.path.exists(status_path):
            return
            
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Map fields (mirrors _on_status)
            self._last_status_phase = str(data.get("phase", self._last_status_phase))
            self._last_status_state = str(data.get("status", self._last_status_state))
            
            dur = data.get("duration_sec")
            if isinstance(dur, (int, float)):
                self._last_status_duration_sec = float(dur)
                self._last_status_duration = f"{dur:.1f}s"

            for key in ("progress_pct", "progress_percent", "pct", "percent_complete"):
                value = data.get(key)
                if isinstance(value, (int, float)):
                    self._last_status_pct = value
                    break

            self._last_status_errors = data.get("errors", self._last_status_errors)
            self._last_status_msg = "" # Clear warnings on final valid read
            
        except Exception as e:
            # Safely ignore read errors during finalization to avoid crashing
            self._append_log(f"DEBUG: Final status sync skipped: {e}")

    def _finalize_run_ui(self):
        """Update UI based on final runner state (authoritative)."""
        if self._did_finalize_run_ui:
            return
        self._did_finalize_run_ui = True
        self._elapsed_timer.stop()
        if self._run_started_monotonic is not None:
            self._last_elapsed_sec = max(0.0, time.monotonic() - self._run_started_monotonic)
            self._run_started_monotonic = None
        
        # Sync terminal values from disk before rendering (Fix stale top-status)
        self._refresh_status_from_disk_final()
        
        # Determine final outcome from runner state (authoritative)
        state = self._runner.state
        code = self._runner.final_status_code
        
        if state == RunnerState.SUCCESS:
            self._append_run_log(f"--- Finished (status: {code}) ---")
            self._last_status_msg = ""
            if self._is_validate_only:
                self._validation_passed = True
                self._append_run_log("Validation PASSED (per status.json). Run is now enabled.")
        elif state == RunnerState.FAILED:
            self._append_run_log(f"--- Run FAILED (status: {code}) ---")
            self._append_run_log("Run failed during execution. Inspect Live Log and open the run folder for artifacts and status details.")
            self._last_status_msg = "Run failed; inspect logs and run folder."
            if self._runner.final_errors:
                self._append_run_log("ERRORS from status.json:")
                for e in self._runner.final_errors:
                    self._append_run_log(f"  \u2022 {e}")
        elif state == RunnerState.FAIL_CLOSED:
            class_id = self._runner.fail_closed_code or "FAIL_CLOSED"
            detail = self._runner.fail_closed_detail or "Status contract check failed."
            remediation = self._runner.fail_closed_remediation
            self._append_run_log(f"Run failed (FAIL_CLOSED): {class_id}")
            self._append_run_log(f"Reason: {detail}")
            if remediation:
                self._append_run_log(f"Next step: {remediation}")
            self._last_status_msg = f"FAIL_CLOSED: {class_id}"
        elif state == RunnerState.CANCELLED:
            self._append_run_log("--- Run CANCELLED ---")
            self._append_run_log("Run was cancelled before normal completion. Partial outputs may exist; inspect the run folder.")
            self._last_status_msg = "Run cancelled; partial outputs may exist."

        # Complete-state workspace is shown only for successful full runs.
        workspace_loaded = False
        if state == RunnerState.SUCCESS and not self._is_validate_only:
            workspace_loaded = self._report_viewer.load_report(self._current_run_dir)
            if workspace_loaded:
                self._append_run_log(f"Analysis completed successfully in {self._current_run_dir}")
            else:
                self._append_run_log(
                    "Run succeeded, but complete-state artifacts were not found. "
                    "Inspect the run folder for available outputs."
                )
        else:
            self._apply_results_idle_placeholder()

        if state == RunnerState.SUCCESS and not self._is_validate_only and workspace_loaded:
            self._enter_complete_state_workspace()
        else:
            self._exit_complete_state_workspace()
        self._refresh_tuning_workspace_availability()

        # Step 8 Preview Mode labeling (Requirement)
        self._apply_preview_labeling()

        # After any full run: require re-validation
        if self._is_validate_only and self._runner.state == RunnerState.SUCCESS:
            self._validation_passed = True
            # Store validated state for future reuse (Fix B1)
            try:
                self._validated_run_dir = self._current_run_dir
                self._validated_gui_run_spec_json_path = os.path.join(self._validated_run_dir, "gui_run_spec.json")
                self._validated_config_effective_yaml_path = os.path.join(self._validated_run_dir, "config_effective.yaml")
                self._validated_run_signature = compute_run_signature(self._validated_run_dir)
            except Exception as e:
                self._append_log(f"DEBUG: Failed to record validation signature: {e}")
                self._validated_run_dir = None
                self._validated_gui_run_spec_json_path = None
                self._validated_config_effective_yaml_path = None
                self._validated_run_signature = None

        self._refresh_splitter_workspace_policy()
        self._render_status_label()
        # Ensure _is_validate_only is False before updating buttons so we are no longer "validating"
        self._is_validate_only = False
        self._update_button_states()

    def _apply_preview_labeling(self):
        """Source preview state from run_report.json and update window/badge."""
        # Preview controls are demoted from idle layout, but preview mode may still
        # be active via compatibility state; keep explicit top-strip indication.
        self.setWindowTitle(self.WINDOW_TITLE_BASE)
        self._preview_badge.hide()
        
        if not self._current_run_dir:
            return
            
        report_path = os.path.join(self._current_run_dir, "run_report.json")
        if not os.path.exists(report_path):
            return
            
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            from gui.run_report_parser import get_preview_mode
            if get_preview_mode(data):
                self.setWindowTitle(f"{self.WINDOW_TITLE_BASE} [PREVIEW]")
                self._preview_badge.show()
        except Exception:
            pass

    def _update_complete_state_summary(self) -> None:
        """Populate compact completion-state summary card from current GUI/run context."""
        run_dir = self._current_run_dir or "(not set)"
        run_name = os.path.basename(run_dir) if run_dir and run_dir != "(not set)" else "(not set)"
        input_dir = self._input_dir.text().strip() or "(not set)"
        mode_text = self._mode_combo.currentText().strip() or "(not set)"
        plotting_mode = self._plotting_mode_combo.currentText().strip() or "(not set)"
        roi_summary = self._compute_roi_filter_summary()
        summary_lines = [
            f"Run: {run_name}",
            f"Run directory: {run_dir}",
            f"Input source: {input_dir}",
            f"Setup profile: mode={mode_text}, plotting={plotting_mode}",
            f"ROI filter used: {roi_summary}",
            "Completed outputs are read-only in this phase.",
        ]
        self._complete_summary_label.setText("\n".join(summary_lines))
        if hasattr(self, "_complete_mode_next_steps_label"):
            self._complete_mode_next_steps_label.setText(
                "Next actions: inspect outputs, run post-run tuning as needed, then apply back to next-run settings. "
                "Apply-back does not mutate the completed run; rerun to produce updated outputs."
            )

    def _enter_complete_state_workspace(self) -> None:
        """Switch left pane to compact completion card after successful full runs."""
        self._is_complete_workspace_active = True
        self._update_complete_state_summary()
        self._refresh_results_workspace_summary()
        if hasattr(self, "_controls_stack"):
            self._controls_stack.setCurrentWidget(self._complete_state_panel)
        self._refresh_splitter_workspace_policy()
        self._set_tuning_disclosure_expanded(False)
        self._set_correction_tuning_disclosure_expanded(False)
        self._refresh_tuning_workspace_availability()
        self._render_status_label()

    def _exit_complete_state_workspace(self) -> None:
        """Return left pane to editable run controls."""
        self._is_complete_workspace_active = False
        self._refresh_results_workspace_summary()
        if hasattr(self, "_controls_stack"):
            self._controls_stack.setCurrentWidget(self._config_panel)
        self._refresh_splitter_workspace_policy()
        self._reset_correction_tuning_state()
        self._refresh_tuning_workspace_availability()

    def _on_new_run(self) -> None:
        """Exit complete-state workspace and restore idle editable controls."""
        self._exit_complete_state_workspace()
        self._apply_results_idle_placeholder()
        self._tuning_last_result = None
        self._tuning_last_changed_fields = []
        self._tuning_applyback_applied = False
        self._tuning_applyback_timestamp = ""
        self._set_tuning_overlay_message(
            "Run tuning to generate an ROI preview-session event overlay."
        )
        self._refresh_tuning_feedback_summary()
        self._reset_correction_tuning_state()
        self._refresh_tuning_workspace_availability()
        self.setWindowTitle(self.WINDOW_TITLE_BASE)
        self._preview_badge.hide()
        self._is_validate_only = False
        self._validation_passed = False
        self._state_str = RunnerState.IDLE.value
        self._ui_state = RunnerState.IDLE
        self._last_status_phase = "\u2014"
        self._last_status_state = "\u2014"
        self._last_status_duration = ""
        self._last_status_duration_sec = None
        self._last_status_errors = []
        self._last_status_msg = ""
        self._last_status_pct = None
        self._ui_progress_pct = 0
        self._run_started_monotonic = None
        self._last_elapsed_sec = 0.0
        self._elapsed_timer.stop()
        self._render_status_label()
        self._update_button_states()

    def _on_run_error(self, msg: str):
        self._elapsed_timer.stop()
        if self._run_started_monotonic is not None:
            self._last_elapsed_sec = max(0.0, time.monotonic() - self._run_started_monotonic)
            self._run_started_monotonic = None
        self._stop_status_follower()
        self._stop_log_follower()
        self._update_button_states()
        self._append_log(f"ERR: {msg}")
        QMessageBox.critical(self, "Process Error", msg)

    # ==================================================================
    # QSettings persistence
    # ==================================================================

    def _load_settings_into_widgets(self):
        """Restore widget values from QSettings. Safe if keys are absent."""
        self._settings.beginGroup(_SETTINGS_GROUP)
        self._input_dir.setText(self._settings.value("input_dir", "", str))
        self._output_dir.setText(self._settings.value("output_dir", "", str))
        self._use_custom_config_cb.setChecked(
            self._settings.value("use_custom_config", False, bool)
        )
        self._config_path.setText(self._settings.value("config_path", "", str))

        fmt = self._settings.value("format", "auto", str)
        idx = self._format_combo.findText(fmt)
        if idx >= 0:
            self._format_combo.setCurrentIndex(idx)

        self._sph_edit.setText(self._settings.value("sessions_per_hour", "", str))
        self._duration_edit.setText(self._settings.value("session_duration_s", "", str))

        smooth = self._settings.value("smooth_window_s", 1.0, float)
        self._smooth_spin.setValue(smooth)

        plotting_mode = self._settings.value("plotting_mode", "", str).strip()
        if not plotting_mode:
            legacy_modes = [
                self._settings.value("sig_iso_render_mode", "qc", str),
                self._settings.value("dff_render_mode", "qc", str),
                self._settings.value("stacked_render_mode", "qc", str),
            ]
            plotting_mode = "Full" if any(m == "full" for m in legacy_modes) else "Standard"
        if self._plotting_mode_combo.findText(plotting_mode) >= 0:
            self._plotting_mode_combo.setCurrentText(plotting_mode)
        self._on_plotting_mode_changed()

        anchor_mode = self._settings.value("timeline_anchor_mode", "civil", str).strip()
        idx_anchor = self._timeline_anchor_mode_combo.findData(anchor_mode)
        if idx_anchor < 0:
            idx_anchor = self._timeline_anchor_mode_combo.findData("civil")
        if idx_anchor >= 0:
            self._timeline_anchor_mode_combo.setCurrentIndex(idx_anchor)
        anchor_clock = self._settings.value("fixed_daily_anchor_clock", "07:00", str).strip()
        self._fixed_daily_anchor_time_edit.setText(anchor_clock or "07:00")
        self._on_timeline_anchor_mode_changed()
        fit_mode = self._settings.value(
            "dynamic_fit_mode",
            str(getattr(self._default_cfg, "dynamic_fit_mode", "rolling_local_regression")),
            str,
        ).strip()
        fit_mode = normalize_dynamic_fit_mode(fit_mode)
        idx_fit = self._dynamic_fit_mode_combo.findData(fit_mode)
        if idx_fit >= 0:
            self._dynamic_fit_mode_combo.setCurrentIndex(idx_fit)
        tonic_output_mode = normalize_tonic_output_mode(
            self._settings.value(
                "tonic_output_mode",
                str(getattr(self._default_cfg, "tonic_output_mode", TONIC_OUTPUT_MODE_PRESERVE_RAW)),
                str,
            )
        )
        idx_tonic_mode = self._tonic_output_mode_combo.findData(tonic_output_mode)
        if idx_tonic_mode >= 0:
            self._tonic_output_mode_combo.setCurrentIndex(idx_tonic_mode)
        tonic_timeline_mode = normalize_tonic_timeline_mode(
            self._settings.value(
                "tonic_timeline_mode",
                str(getattr(self._default_cfg, "tonic_timeline_mode", TONIC_TIMELINE_MODE_REAL_ELAPSED)),
                str,
            )
        )
        idx_tonic_timeline_mode = self._tonic_timeline_mode_combo.findData(tonic_timeline_mode)
        if idx_tonic_timeline_mode >= 0:
            self._tonic_timeline_mode_combo.setCurrentIndex(idx_tonic_timeline_mode)
        self._baseline_subtract_before_fit_cb.setChecked(
            self._settings.value(
                "baseline_subtract_before_fit",
                bool(getattr(self._default_cfg, "baseline_subtract_before_fit", False)),
                bool,
            )
        )
        self._robust_event_reject_max_iters_spin.setValue(
            self._settings.value(
                "robust_event_reject_max_iters",
                int(getattr(self._default_cfg, "robust_event_reject_max_iters", 3)),
                int,
            )
        )
        self._robust_event_reject_residual_z_spin.setValue(
            self._settings.value(
                "robust_event_reject_residual_z_thresh",
                float(getattr(self._default_cfg, "robust_event_reject_residual_z_thresh", 3.5)),
                float,
            )
        )
        self._robust_event_reject_local_var_window_spin.setValue(
            self._settings.value(
                "robust_event_reject_local_var_window_sec",
                float(getattr(self._default_cfg, "robust_event_reject_local_var_window_sec", 10.0) or 10.0),
                float,
            )
        )
        robust_ratio_enabled = self._settings.value(
            "robust_event_reject_local_var_ratio_enabled",
            bool(getattr(self._default_cfg, "robust_event_reject_local_var_ratio_thresh", None) is not None),
            bool,
        )
        self._robust_event_reject_local_var_ratio_enable_cb.setChecked(bool(robust_ratio_enabled))
        self._robust_event_reject_local_var_ratio_spin.setValue(
            self._settings.value(
                "robust_event_reject_local_var_ratio_thresh",
                float(
                    getattr(self._default_cfg, "robust_event_reject_local_var_ratio_thresh", 3.0)
                    if getattr(self._default_cfg, "robust_event_reject_local_var_ratio_thresh", None) is not None
                    else 3.0
                ),
                float,
            )
        )
        self._robust_event_reject_min_keep_fraction_spin.setValue(
            self._settings.value(
                "robust_event_reject_min_keep_fraction",
                float(getattr(self._default_cfg, "robust_event_reject_min_keep_fraction", 0.5)),
                float,
            )
        )
        self._adaptive_event_gate_residual_z_spin.setValue(
            self._settings.value(
                "adaptive_event_gate_residual_z_thresh",
                float(getattr(self._default_cfg, "adaptive_event_gate_residual_z_thresh", 3.5)),
                float,
            )
        )
        self._adaptive_event_gate_local_var_window_spin.setValue(
            self._settings.value(
                "adaptive_event_gate_local_var_window_sec",
                float(getattr(self._default_cfg, "adaptive_event_gate_local_var_window_sec", 10.0) or 10.0),
                float,
            )
        )
        adaptive_ratio_enabled = self._settings.value(
            "adaptive_event_gate_local_var_ratio_enabled",
            bool(getattr(self._default_cfg, "adaptive_event_gate_local_var_ratio_thresh", None) is not None),
            bool,
        )
        self._adaptive_event_gate_local_var_ratio_enable_cb.setChecked(bool(adaptive_ratio_enabled))
        self._adaptive_event_gate_local_var_ratio_spin.setValue(
            self._settings.value(
                "adaptive_event_gate_local_var_ratio_thresh",
                float(
                    getattr(self._default_cfg, "adaptive_event_gate_local_var_ratio_thresh", 3.0)
                    if getattr(self._default_cfg, "adaptive_event_gate_local_var_ratio_thresh", None) is not None
                    else 3.0
                ),
                float,
            )
        )
        self._adaptive_event_gate_smooth_window_spin.setValue(
            self._settings.value(
                "adaptive_event_gate_smooth_window_sec",
                float(getattr(self._default_cfg, "adaptive_event_gate_smooth_window_sec", 60.0)),
                float,
            )
        )
        self._adaptive_event_gate_min_trust_fraction_spin.setValue(
            self._settings.value(
                "adaptive_event_gate_min_trust_fraction",
                float(getattr(self._default_cfg, "adaptive_event_gate_min_trust_fraction", 0.5)),
                float,
            )
        )
        adaptive_freeze_interp = str(
            self._settings.value(
                "adaptive_event_gate_freeze_interp_method",
                str(getattr(self._default_cfg, "adaptive_event_gate_freeze_interp_method", "linear_hold")),
                str,
            )
            or "linear_hold"
        ).strip()
        idx_adaptive_freeze = self._adaptive_event_gate_freeze_interp_combo.findData(adaptive_freeze_interp)
        if idx_adaptive_freeze >= 0:
            self._adaptive_event_gate_freeze_interp_combo.setCurrentIndex(idx_adaptive_freeze)
        self._on_dynamic_fit_mode_changed()

        sig_iso_render_mode = self._settings.value("sig_iso_render_mode", "qc", str)
        if self._sig_iso_render_mode_combo.findText(sig_iso_render_mode) >= 0:
            self._sig_iso_render_mode_combo.setCurrentText(sig_iso_render_mode)

        dff_render_mode = self._settings.value("dff_render_mode", "qc", str)
        if self._dff_render_mode_combo.findText(dff_render_mode) >= 0:
            self._dff_render_mode_combo.setCurrentText(dff_render_mode)

        stacked_render_mode = self._settings.value("stacked_render_mode", "qc", str)
        if self._stacked_render_mode_combo.findText(stacked_render_mode) >= 0:
            self._stacked_render_mode_combo.setCurrentText(stacked_render_mode)

        peak_pre_filter = self._settings.value(
            "peak_pre_filter",
            str(getattr(self._default_cfg, "peak_pre_filter", "none")),
            str,
        )
        if self._peak_pre_filter_combo.findText(peak_pre_filter) >= 0:
            self._peak_pre_filter_combo.setCurrentText(peak_pre_filter)

        overwrite = self._settings.value("overwrite", False, bool)
        self._overwrite_cb.setChecked(overwrite)
        self._settings.endGroup()
        self._update_config_source_ui()

    def _save_widgets_to_settings(self):
        """Persist current widget values to QSettings."""
        self._settings.beginGroup(_SETTINGS_GROUP)
        self._settings.setValue("input_dir", self._input_dir.text().strip())
        self._settings.setValue("output_dir", self._output_dir.text().strip())
        self._settings.setValue("use_custom_config", self._use_custom_config_cb.isChecked())
        self._settings.setValue("config_path", self._config_path.text().strip())
        self._settings.setValue("format", self._format_combo.currentText())
        self._settings.setValue("sessions_per_hour", self._sph_edit.text().strip())
        self._settings.setValue("session_duration_s", self._duration_edit.text().strip())
        self._settings.setValue("smooth_window_s", self._smooth_spin.value())
        self._settings.setValue("plotting_mode", self._plotting_mode_combo.currentText())
        self._settings.setValue("timeline_anchor_mode", self._timeline_anchor_mode_value())
        self._settings.setValue("fixed_daily_anchor_clock", self._fixed_daily_anchor_time_edit.text().strip())
        self._settings.setValue("dynamic_fit_mode", self._selected_dynamic_fit_mode())
        self._settings.setValue(
            "baseline_subtract_before_fit",
            bool(getattr(self, "_baseline_subtract_before_fit_cb", None) and self._baseline_subtract_before_fit_cb.isChecked()),
        )
        self._settings.setValue(
            "robust_event_reject_max_iters",
            int(getattr(self, "_robust_event_reject_max_iters_spin", None).value() if hasattr(self, "_robust_event_reject_max_iters_spin") else 3),
        )
        self._settings.setValue(
            "robust_event_reject_residual_z_thresh",
            float(getattr(self, "_robust_event_reject_residual_z_spin", None).value() if hasattr(self, "_robust_event_reject_residual_z_spin") else 3.5),
        )
        self._settings.setValue(
            "robust_event_reject_local_var_window_sec",
            float(getattr(self, "_robust_event_reject_local_var_window_spin", None).value() if hasattr(self, "_robust_event_reject_local_var_window_spin") else 10.0),
        )
        self._settings.setValue(
            "robust_event_reject_local_var_ratio_enabled",
            bool(
                getattr(self, "_robust_event_reject_local_var_ratio_enable_cb", None)
                and self._robust_event_reject_local_var_ratio_enable_cb.isChecked()
            ),
        )
        self._settings.setValue(
            "robust_event_reject_local_var_ratio_thresh",
            float(getattr(self, "_robust_event_reject_local_var_ratio_spin", None).value() if hasattr(self, "_robust_event_reject_local_var_ratio_spin") else 3.0),
        )
        self._settings.setValue(
            "robust_event_reject_min_keep_fraction",
            float(getattr(self, "_robust_event_reject_min_keep_fraction_spin", None).value() if hasattr(self, "_robust_event_reject_min_keep_fraction_spin") else 0.5),
        )
        self._settings.setValue(
            "adaptive_event_gate_residual_z_thresh",
            float(getattr(self, "_adaptive_event_gate_residual_z_spin", None).value() if hasattr(self, "_adaptive_event_gate_residual_z_spin") else 3.5),
        )
        self._settings.setValue(
            "adaptive_event_gate_local_var_window_sec",
            float(getattr(self, "_adaptive_event_gate_local_var_window_spin", None).value() if hasattr(self, "_adaptive_event_gate_local_var_window_spin") else 10.0),
        )
        self._settings.setValue(
            "adaptive_event_gate_local_var_ratio_enabled",
            bool(
                getattr(self, "_adaptive_event_gate_local_var_ratio_enable_cb", None)
                and self._adaptive_event_gate_local_var_ratio_enable_cb.isChecked()
            ),
        )
        self._settings.setValue(
            "adaptive_event_gate_local_var_ratio_thresh",
            float(getattr(self, "_adaptive_event_gate_local_var_ratio_spin", None).value() if hasattr(self, "_adaptive_event_gate_local_var_ratio_spin") else 3.0),
        )
        self._settings.setValue(
            "adaptive_event_gate_smooth_window_sec",
            float(getattr(self, "_adaptive_event_gate_smooth_window_spin", None).value() if hasattr(self, "_adaptive_event_gate_smooth_window_spin") else 60.0),
        )
        self._settings.setValue(
            "adaptive_event_gate_min_trust_fraction",
            float(getattr(self, "_adaptive_event_gate_min_trust_fraction_spin", None).value() if hasattr(self, "_adaptive_event_gate_min_trust_fraction_spin") else 0.5),
        )
        self._settings.setValue(
            "adaptive_event_gate_freeze_interp_method",
            str(
                getattr(self, "_adaptive_event_gate_freeze_interp_combo", None).currentData()
                if hasattr(self, "_adaptive_event_gate_freeze_interp_combo")
                else "linear_hold"
            ),
        )
        self._settings.setValue(
            "tonic_output_mode",
            str(
                getattr(self, "_tonic_output_mode_combo", None).currentData()
                if hasattr(self, "_tonic_output_mode_combo")
                else TONIC_OUTPUT_MODE_PRESERVE_RAW
            ),
        )
        self._settings.setValue(
            "tonic_timeline_mode",
            str(
                getattr(self, "_tonic_timeline_mode_combo", None).currentData()
                if hasattr(self, "_tonic_timeline_mode_combo")
                else TONIC_TIMELINE_MODE_REAL_ELAPSED
            ),
        )
        self._settings.setValue("sig_iso_render_mode", self._sig_iso_render_mode_combo.currentText())
        self._settings.setValue("dff_render_mode", self._dff_render_mode_combo.currentText())
        self._settings.setValue("stacked_render_mode", self._stacked_render_mode_combo.currentText())
        self._settings.setValue("peak_pre_filter", self._peak_pre_filter_combo.currentText())
        self._settings.setValue("overwrite", self._overwrite_cb.isChecked())
        self._settings.endGroup()
        self._settings.sync()

    # ==================================================================
    # Helpers
    # ==================================================================

    def _push_busy_cursor(self) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)

    def _pop_busy_cursor(self) -> None:
        if QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

    @contextmanager
    def _busy_cursor_scope(self):
        """
        Brief non-modal busy cursor scope for synchronous preflight/start work.
        Restores reliably on success, early return, and exceptions.
        """
        self._push_busy_cursor()
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        try:
            yield
        finally:
            self._pop_busy_cursor()

    def _append_log(self, text: str):
        self._log_view.appendPlainText(text)

    def _append_run_log(self, text: str):
        """Append a message from the wrapper/GUI with a 'RUN: ' prefix."""
        self._append_log(f"RUN: {text}")

    def _emit_gui_timing(self, event: str, step: str, elapsed_sec: float | None = None, extra: str = "") -> None:
        """Emit grep-friendly GUI preflight timing lines."""
        if not self._gui_timing_enabled:
            return
        action = self._timing_action or "unknown"
        if elapsed_sec is None:
            msg = f"GUI_TIMING {event} action={action} step={step}"
        else:
            msg = f"GUI_TIMING {event} action={action} step={step} elapsed_sec={elapsed_sec:.6f}"
        if extra:
            msg = f"{msg} {extra}"
        self._append_run_log(msg)
        try:
            print(msg, flush=True)
        except Exception:
            pass

    def _timing_start(self, step: str, extra: str = "") -> float:
        if not self._gui_timing_enabled:
            return 0.0
        t0 = time.perf_counter()
        self._emit_gui_timing("START", step, extra=extra)
        return t0

    def _timing_end(self, step: str, t0: float, extra: str = "") -> None:
        if not self._gui_timing_enabled:
            return
        elapsed = time.perf_counter() - t0
        self._emit_gui_timing("END", step, elapsed_sec=elapsed, extra=extra)

    def _compute_roi_filter_summary(self) -> str:
        """Human-readable summary of current ROI filtering intent."""
        if self._discovery_cache is None:
            return "Unknown (run Select ROIs...)"

        total = self._roi_list.count()
        checked = sum(
            1 for i in range(total)
            if self._roi_list.item(i).checkState() == Qt.Checked
        )
        if checked == total:
            return f"Include all discovered ROIs ({total})"
        if checked == 0:
            return "Include none (invalid)"
        return f"Include subset ({checked}/{total})"

    def _compute_representative_summary(self) -> str:
        """Human-readable summary of representative-session behavior."""
        idx = self._rep_session_combo.currentIndex()
        if idx <= 0:
            if self._discovery_cache is None:
                return "Auto (runtime selection; use Select ROIs... for explicit choice)"
            return "Auto"

        session_idx = idx - 1
        session_label = self._rep_session_combo.currentText().strip() or "?"
        return f"Session index {session_idx} ({session_label})"

    def _selected_dynamic_fit_mode(self) -> str:
        combo = getattr(self, "_dynamic_fit_mode_combo", None)
        if combo is not None:
            data = combo.currentData()
            if data:
                return normalize_dynamic_fit_mode(str(data))
        return normalize_dynamic_fit_mode(
            str(getattr(self._default_cfg, "dynamic_fit_mode", "rolling_local_regression"))
        )

    def _selected_tonic_output_mode(self) -> str:
        combo = getattr(self, "_tonic_output_mode_combo", None)
        if combo is not None:
            data = combo.currentData()
            if data:
                return normalize_tonic_output_mode(str(data))
        return normalize_tonic_output_mode(
            str(getattr(self._default_cfg, "tonic_output_mode", TONIC_OUTPUT_MODE_PRESERVE_RAW))
        )

    def _selected_tonic_timeline_mode(self) -> str:
        combo = getattr(self, "_tonic_timeline_mode_combo", None)
        if combo is not None:
            data = combo.currentData()
            if data:
                return normalize_tonic_timeline_mode(str(data))
        return normalize_tonic_timeline_mode(
            str(getattr(self._default_cfg, "tonic_timeline_mode", TONIC_TIMELINE_MODE_REAL_ELAPSED))
        )

    def _selected_correction_tuning_fit_mode(self) -> str:
        combo = getattr(self, "_correction_tuning_fit_mode_combo", None)
        if combo is not None:
            data = combo.currentData()
            if data:
                return normalize_dynamic_fit_mode(str(data))
        return normalize_dynamic_fit_mode(
            str(getattr(self._default_cfg, "dynamic_fit_mode", "rolling_local_regression"))
        )

    def _on_dynamic_fit_mode_changed(self) -> None:
        self._apply_dynamic_fit_mode_ui_state()

    def _apply_dynamic_fit_mode_ui_state(self) -> None:
        fit_mode = self._selected_dynamic_fit_mode()
        rolling = is_rolling_dynamic_fit_mode(fit_mode)
        robust = is_robust_event_reject_mode(fit_mode)
        adaptive = is_adaptive_event_gated_mode(fit_mode)
        self._window_sec_edit.setEnabled(rolling)
        self._min_samples_per_window_spin.setEnabled(rolling)
        if hasattr(self, "_baseline_subtract_before_fit_cb"):
            self._baseline_subtract_before_fit_cb.setEnabled(rolling)
        if hasattr(self, "_robust_event_reject_max_iters_spin"):
            self._robust_event_reject_max_iters_spin.setEnabled(robust)
        if hasattr(self, "_robust_event_reject_residual_z_spin"):
            self._robust_event_reject_residual_z_spin.setEnabled(robust)
        if hasattr(self, "_robust_event_reject_local_var_window_spin"):
            self._robust_event_reject_local_var_window_spin.setEnabled(robust)
        if hasattr(self, "_robust_event_reject_local_var_ratio_enable_cb"):
            self._robust_event_reject_local_var_ratio_enable_cb.setEnabled(robust)
        if hasattr(self, "_robust_event_reject_local_var_ratio_spin"):
            ratio_enabled = (
                robust
                and bool(self._robust_event_reject_local_var_ratio_enable_cb.isChecked())
            )
            self._robust_event_reject_local_var_ratio_spin.setEnabled(ratio_enabled)
        if hasattr(self, "_robust_event_reject_min_keep_fraction_spin"):
            self._robust_event_reject_min_keep_fraction_spin.setEnabled(robust)
        if hasattr(self, "_adaptive_event_gate_residual_z_spin"):
            self._adaptive_event_gate_residual_z_spin.setEnabled(adaptive)
        if hasattr(self, "_adaptive_event_gate_local_var_window_spin"):
            self._adaptive_event_gate_local_var_window_spin.setEnabled(adaptive)
        if hasattr(self, "_adaptive_event_gate_local_var_ratio_enable_cb"):
            self._adaptive_event_gate_local_var_ratio_enable_cb.setEnabled(adaptive)
        if hasattr(self, "_adaptive_event_gate_local_var_ratio_spin"):
            adaptive_ratio_enabled = (
                adaptive
                and bool(self._adaptive_event_gate_local_var_ratio_enable_cb.isChecked())
            )
            self._adaptive_event_gate_local_var_ratio_spin.setEnabled(adaptive_ratio_enabled)
        if hasattr(self, "_adaptive_event_gate_smooth_window_spin"):
            self._adaptive_event_gate_smooth_window_spin.setEnabled(adaptive)
        if hasattr(self, "_adaptive_event_gate_min_trust_fraction_spin"):
            self._adaptive_event_gate_min_trust_fraction_spin.setEnabled(adaptive)
        if hasattr(self, "_adaptive_event_gate_freeze_interp_combo"):
            self._adaptive_event_gate_freeze_interp_combo.setEnabled(adaptive)
        if rolling:
            msg = (
                f"{dynamic_fit_mode_label(fit_mode)} is active. "
                "Regression window, min samples per window, and baseline-subtract toggle are active."
            )
        elif robust:
            msg = (
                "Robust global fit + event rejection is active. "
                "Robust exclusion controls are active; rolling-only and adaptive controls are inactive."
            )
        elif adaptive:
            msg = (
                "Adaptive event-gated regression is active. "
                "Adaptive controls are active; rolling-only, robust, and baseline-subtract controls are inactive."
            )
        else:
            msg = (
                "Global linear regression is active. "
                "Rolling, robust event-rejection, and adaptive controls are inactive."
            )
        self._dynamic_fit_mode_note.setText(msg)

    def _on_correction_tuning_fit_mode_changed(self) -> None:
        self._apply_correction_tuning_fit_mode_ui_state()

    def _apply_correction_tuning_fit_mode_ui_state(self) -> None:
        fit_mode = self._selected_correction_tuning_fit_mode()
        rolling = is_rolling_dynamic_fit_mode(fit_mode)
        robust = is_robust_event_reject_mode(fit_mode)
        adaptive = is_adaptive_event_gated_mode(fit_mode)
        self._correction_tuning_window_spin.setEnabled(rolling)
        self._correction_tuning_min_samples_spin.setEnabled(rolling)
        if hasattr(self, "_correction_tuning_baseline_subtract_cb"):
            self._correction_tuning_baseline_subtract_cb.setEnabled(rolling)
        if hasattr(self, "_correction_tuning_robust_max_iters_spin"):
            self._correction_tuning_robust_max_iters_spin.setEnabled(robust)
        if hasattr(self, "_correction_tuning_robust_residual_z_spin"):
            self._correction_tuning_robust_residual_z_spin.setEnabled(robust)
        if hasattr(self, "_correction_tuning_robust_local_var_window_spin"):
            self._correction_tuning_robust_local_var_window_spin.setEnabled(robust)
        if hasattr(self, "_correction_tuning_robust_local_var_ratio_enable_cb"):
            self._correction_tuning_robust_local_var_ratio_enable_cb.setEnabled(robust)
        if hasattr(self, "_correction_tuning_robust_local_var_ratio_spin"):
            ratio_enabled = (
                robust
                and bool(self._correction_tuning_robust_local_var_ratio_enable_cb.isChecked())
            )
            self._correction_tuning_robust_local_var_ratio_spin.setEnabled(ratio_enabled)
        if hasattr(self, "_correction_tuning_robust_min_keep_fraction_spin"):
            self._correction_tuning_robust_min_keep_fraction_spin.setEnabled(robust)
        if hasattr(self, "_correction_tuning_adaptive_residual_z_spin"):
            self._correction_tuning_adaptive_residual_z_spin.setEnabled(adaptive)
        if hasattr(self, "_correction_tuning_adaptive_local_var_window_spin"):
            self._correction_tuning_adaptive_local_var_window_spin.setEnabled(adaptive)
        if hasattr(self, "_correction_tuning_adaptive_local_var_ratio_enable_cb"):
            self._correction_tuning_adaptive_local_var_ratio_enable_cb.setEnabled(adaptive)
        if hasattr(self, "_correction_tuning_adaptive_local_var_ratio_spin"):
            adaptive_ratio_enabled = (
                adaptive
                and bool(self._correction_tuning_adaptive_local_var_ratio_enable_cb.isChecked())
            )
            self._correction_tuning_adaptive_local_var_ratio_spin.setEnabled(adaptive_ratio_enabled)
        if hasattr(self, "_correction_tuning_adaptive_smooth_window_spin"):
            self._correction_tuning_adaptive_smooth_window_spin.setEnabled(adaptive)
        if hasattr(self, "_correction_tuning_adaptive_min_trust_fraction_spin"):
            self._correction_tuning_adaptive_min_trust_fraction_spin.setEnabled(adaptive)
        if hasattr(self, "_correction_tuning_adaptive_freeze_interp_combo"):
            self._correction_tuning_adaptive_freeze_interp_combo.setEnabled(adaptive)
        if rolling:
            msg = (
                f"{dynamic_fit_mode_label(fit_mode)} is active. "
                "Regression window, min samples per window, and baseline-subtract toggle are active."
            )
        elif robust:
            msg = (
                "Robust global fit + event rejection is active. "
                "Robust exclusion controls are active; rolling-only and adaptive controls are inactive."
            )
        elif adaptive:
            msg = (
                "Adaptive event-gated regression is active. "
                "Adaptive controls are active; rolling-only, robust, and baseline-subtract controls are inactive."
            )
        else:
            msg = (
                "Global linear regression is active. "
                "Rolling, robust event-rejection, and adaptive controls are inactive."
            )
        self._correction_tuning_fit_mode_note.setText(msg)

    def _refresh_effective_run_summary(self) -> None:
        """Refresh compact pre-run intent summary from current GUI state."""
        mode_text = self._mode_combo.currentText()
        phasic_active = is_isosbestic_active(mode_text)
        analysis_scope = (
            "Traces-only (feature extraction skipped)"
            if self._traces_only_cb.isChecked()
            else "Full analysis"
        )
        preview_text = (
            f"first N = {self._preview_n_spin.value()}"
            if self._preview_enabled_cb.isChecked()
            else "off"
        )
        render_text = self._plotting_mode_combo.currentText()
        if not phasic_active:
            render_text += " (inactive in tonic mode)"
        roi_text = self._compute_roi_filter_summary()
        rep_text = self._compute_representative_summary()

        out_base = self._output_dir.text().strip()
        if out_base:
            out_text = (
                f"{out_base} "
                "(a timestamped run folder will be created for Validate/Run)"
            )
        else:
            out_text = "(not set)"

        summary_lines = [
            f"Mode: {mode_text}",
            f"Analysis: {analysis_scope}",
            (
                f"Tonic Output Mode: {tonic_output_mode_label(self._selected_tonic_output_mode())}"
                if mode_text in ("both", "tonic")
                else "Tonic Output Mode: (inactive in phasic mode)"
            ),
            (
                f"Tonic Timeline Mode: {tonic_timeline_mode_label(self._selected_tonic_timeline_mode())}"
                if mode_text in ("both", "tonic")
                else "Tonic Timeline Mode: (inactive in phasic mode)"
            ),
            (
                f"Dynamic Fit Mode: {dynamic_fit_mode_label(self._selected_dynamic_fit_mode())}"
                if phasic_active
                else "Dynamic Fit Mode: (inactive in tonic mode)"
            ),
            (
                (
                    "Baseline subtract before fit: "
                    + (
                        "on"
                        if bool(getattr(self, "_baseline_subtract_before_fit_cb", None) and self._baseline_subtract_before_fit_cb.isChecked())
                        else "off"
                    )
                )
                if phasic_active and is_rolling_dynamic_fit_mode(self._selected_dynamic_fit_mode())
                else (
                    (
                        "Baseline subtract before fit: inactive in robust global event-reject mode"
                        if phasic_active and is_robust_event_reject_mode(self._selected_dynamic_fit_mode())
                        else (
                            "Baseline subtract before fit: inactive in adaptive event-gated mode"
                            if phasic_active and is_adaptive_event_gated_mode(self._selected_dynamic_fit_mode())
                            else (
                                "Baseline subtract before fit: inactive in global linear mode"
                                if phasic_active
                                else "Baseline subtract before fit: (inactive in tonic mode)"
                            )
                        )
                    )
                )
            ),
            f"Baseline Config Source: {self._active_config_source_summary()}",
            f"Preview: {preview_text}",
            f"Plotting Mode: {render_text}",
            f"Timeline Anchor: {self._timeline_anchor_summary()}",
            f"ROI Filter: {roi_text}",
            f"Representative Session: {rep_text}",
            f"Output Destination: {out_text}",
        ]
        if phasic_active and is_robust_event_reject_mode(self._selected_dynamic_fit_mode()):
            robust_ratio_thresh = (
                float(self._robust_event_reject_local_var_ratio_spin.value())
                if bool(self._robust_event_reject_local_var_ratio_enable_cb.isChecked())
                else None
            )
            summary_lines.insert(
                4,
                format_robust_event_reject_settings_summary(
                    int(self._robust_event_reject_max_iters_spin.value()),
                    float(self._robust_event_reject_residual_z_spin.value()),
                    float(self._robust_event_reject_local_var_window_spin.value()),
                    robust_ratio_thresh,
                    float(self._robust_event_reject_min_keep_fraction_spin.value()),
                ),
            )
        elif phasic_active and is_adaptive_event_gated_mode(self._selected_dynamic_fit_mode()):
            adaptive_ratio_thresh = (
                float(self._adaptive_event_gate_local_var_ratio_spin.value())
                if bool(self._adaptive_event_gate_local_var_ratio_enable_cb.isChecked())
                else None
            )
            summary_lines.insert(
                4,
                format_adaptive_event_gated_settings_summary(
                    float(self._adaptive_event_gate_residual_z_spin.value()),
                    float(self._adaptive_event_gate_local_var_window_spin.value()),
                    adaptive_ratio_thresh,
                    float(self._adaptive_event_gate_smooth_window_spin.value()),
                    float(self._adaptive_event_gate_min_trust_fraction_spin.value()),
                    str(
                        self._adaptive_event_gate_freeze_interp_combo.currentData()
                        or self._adaptive_event_gate_freeze_interp_combo.currentText()
                        or "linear_hold"
                    ),
                ),
            )
        self._effective_summary_label.setText("\n".join(summary_lines))

    def _update_context_sensitive_controls(self) -> None:
        """Enable/disable controls that are irrelevant in current mode/context."""
        mode_text = self._mode_combo.currentText()
        phasic_active = is_isosbestic_active(mode_text)
        tonic_active = mode_text in ("both", "tonic")
        self._apply_dynamic_fit_mode_ui_state()
        if hasattr(self, "_dynamic_fit_mode_combo"):
            self._dynamic_fit_mode_combo.setEnabled(phasic_active)
        if hasattr(self, "_tonic_output_mode_combo"):
            self._tonic_output_mode_combo.setEnabled(tonic_active)
        if hasattr(self, "_tonic_timeline_mode_combo"):
            self._tonic_timeline_mode_combo.setEnabled(tonic_active)

        # Phasic-only controls: render family selectors + event/features group.
        self._plotting_mode_combo.setEnabled(phasic_active)
        if hasattr(self, "_timeline_anchor_mode_combo"):
            self._timeline_anchor_mode_combo.setEnabled(phasic_active)
        for combo in (
            self._sig_iso_render_mode_combo,
            self._dff_render_mode_combo,
            self._stacked_render_mode_combo,
        ):
            combo.setEnabled(phasic_active)
        self._adv_ev_group.setEnabled(phasic_active)
        if hasattr(self, "_fixed_daily_anchor_time_edit"):
            self._on_timeline_anchor_mode_changed()
        if phasic_active:
            self._mode_context_label.setText("Phasic controls are active for this mode.")
            self._set_status_label_style(self._mode_context_label, "info")
        else:
            self._mode_context_label.setText(
                "Phasic-only controls are disabled in tonic mode."
            )
            self._set_status_label_style(self._mode_context_label, "warn")

        # Discovery-dependent ROI/representative controls.
        discovery_ready = self._discovery_cache is not None and self._roi_list.count() > 0
        self._roi_filter_combo.setEnabled(discovery_ready)
        self._roi_list.setEnabled(discovery_ready)
        self._rep_session_combo.setEnabled(discovery_ready)

        if not discovery_ready:
            self._discovery_controls_hint.setText(
                "ROI choices are unresolved. Click 'Select ROIs...' to populate ROI choices."
            )
            self._set_status_label_style(self._discovery_controls_hint, "warn")
            self._rep_preview_hint.setText(
                "Representative selection requires discovery session ordering."
            )
            self._set_status_label_style(self._rep_preview_hint, "warn")
            return

        self._discovery_controls_hint.setText("ROI choices loaded. Checked ROIs will be included.")
        self._set_status_label_style(self._discovery_controls_hint, "ready")

        if not self._preview_enabled_cb.isChecked():
            self._rep_preview_hint.setText(
                "Representative selection applies to the full discovered session list."
            )
            self._set_status_label_style(self._rep_preview_hint, "info")
            return

        preview_n = self._preview_n_spin.value()
        rep_idx = self._rep_session_combo.currentIndex() - 1
        rep_preview_err = validate_representative_index_preview_compatibility(rep_idx, preview_n)
        if rep_preview_err:
            self._rep_preview_hint.setText(rep_preview_err)
            self._set_status_label_style(self._rep_preview_hint, "error")
            return

        if rep_idx >= 0:
            self._rep_preview_hint.setText(
                f"Representative session index {rep_idx} is within preview first N={preview_n}."
            )
        else:
            self._rep_preview_hint.setText(
                f"Preview first N={preview_n}; representative session remains auto."
            )
        self._set_status_label_style(self._rep_preview_hint, "info")

    def _update_key_artifact_buttons(self, running: bool) -> None:
        """Enable post-run key-file shortcuts when files exist."""
        run_dir_ok = bool(self._current_run_dir and os.path.isdir(self._current_run_dir))

        def has_file(name: str) -> bool:
            if not run_dir_ok:
                return False
            return os.path.isfile(self._artifact_path_in_current_run(name))

        can_open = not running
        self._open_cmd_file_btn.setEnabled(can_open and has_file("command_invoked.txt"))
        self._open_spec_file_btn.setEnabled(can_open and has_file("gui_run_spec.json"))
        self._open_cfg_file_btn.setEnabled(can_open and has_file("config_effective.yaml"))
        self._open_manifest_file_btn.setEnabled(can_open and has_file("MANIFEST.json"))
        self._open_report_file_btn.setEnabled(can_open and has_file("run_report.json"))

    def _compute_run_readiness_reason(self) -> tuple[str, str]:
        """
        Returns (reason_text, severity) for run-state guidance.
        severity in {"ready", "info", "warn", "error"}.
        """
        running = self._runner.is_running()
        if running and self._ui_state == RunnerState.RUNNING:
            return "Run unavailable while a pipeline run is in progress.", "info"
        if running and self._ui_state == RunnerState.VALIDATING:
            return "Run unavailable while validation is in progress.", "info"

        err = self._validate_gui_inputs()
        if err:
            return f"Invalid setting combination: {err}", "error"

        if not self._validation_passed:
            if self._discovery_cache is None:
                return (
                    "Validation required before Run. ROI selection is optional but recommended.",
                    "warn",
                )
            return "Validation required after config change. Click 'Validate Only'.", "warn"

        if self._discovery_cache is None:
            return (
                "Ready to run. ROI choices will auto-resolve unless 'Select ROIs...' is used.",
                "ready",
            )

        return "Ready to run.", "ready"

    def _update_run_reason_label(self) -> None:
        """Render concise run-state reason text near the Run button."""
        reason, severity = self._compute_run_readiness_reason()
        self._set_status_label_style(self._run_reason_label, severity)
        self._run_reason_label.setText(f"Run status: {reason}")

    def _update_button_states(self):
        state = self._ui_state
        running = self._runner.is_running()
        editing_enabled = not running
        
        is_done = state in (RunnerState.SUCCESS, RunnerState.FAILED,
                            RunnerState.CANCELLED, RunnerState.FAIL_CLOSED)
        is_idle_or_done = state == RunnerState.IDLE or is_done

        # Validate: enabled only when idle or done (not running/validating)
        self._validate_btn.setEnabled(is_idle_or_done and not running)

        # Run: enabled only when idle/done, not running, AND validated
        self._run_btn.setEnabled(
            bool(is_idle_or_done and not running and self._validation_passed)
        )

        # Cancel: enabled only when RUNNING (not VALIDATING per rule A)
        self._cancel_btn.setEnabled(state == RunnerState.RUNNING and running)

        # Open Results: enabled only when SUCCESS and status code is "success" (Requirement 5.2)
        is_success = bool(state == RunnerState.SUCCESS)
        has_success_code = bool(self._runner.final_status_code == "success")
        self._open_results_btn.setEnabled(bool(is_success and has_success_code))

        # Open Run Folder: enabled when done and run_dir exists
        has_run_dir = bool(self._current_run_dir and os.path.isdir(self._current_run_dir))
        self._open_folder_btn.setEnabled(bool(is_done and has_run_dir and not running))

        # Keep upper-left controls visible but disable editing during active validate/run.
        self._run_config_inputs_container.setEnabled(editing_enabled)
        self._plotting_group.setEnabled(editing_enabled)
        self._advanced_group.setEnabled(editing_enabled)

        self._update_context_sensitive_controls()
        self._update_key_artifact_buttons(running)
        self._update_run_reason_label()
        self._refresh_effective_run_summary()

    def _browse_dir(self, line_edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title, line_edit.text())
        if path:
            line_edit.setText(path)
            self._save_widgets_to_settings()

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Config YAML",
            self._config_path.text(),
            "YAML files (*.yaml *.yml);;All files (*)"
        )
        if path:
            self._config_path.setText(path)
            self._save_widgets_to_settings()

    # ==================================================================
    # Patch 1: Structural GUI shell rebuild (layout/grouping only)
    # ==================================================================

    def _build_config_panel(self) -> QWidget:
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        panel.setMinimumWidth(0)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        intro = QLabel("Primary workflow stages")
        intro.setStyleSheet("font-size: 11px; font-weight: 600;")
        outer.addWidget(intro)

        self._run_config_group = self._build_run_configuration_group()
        self._plotting_group = self._build_plotting_group()
        self._advanced_group = self._build_advanced_group()
        for section in (self._run_config_group, self._plotting_group, self._advanced_group):
            section.setProperty("workflowSection", True)
            section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            section.setMinimumWidth(0)
        outer.addWidget(self._run_config_group)
        outer.addWidget(self._plotting_group)
        outer.addWidget(self._advanced_group)
        # Demoted controls are kept for behavior compatibility but hidden from idle layout.
        outer.addWidget(self._build_hidden_compatibility_group())
        return panel

    def _first_layout_tooltip(self, layout: QLayout) -> str:
        """Return first non-empty tooltip found in a layout tree."""
        for idx in range(layout.count()):
            item = layout.itemAt(idx)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                tip = widget.toolTip().strip()
                if tip:
                    return tip
            child_layout = item.layout()
            if child_layout is not None:
                tip = self._first_layout_tooltip(child_layout)
                if tip:
                    return tip
        return ""

    def _resolve_form_row_tooltip(self, form_layout: QFormLayout, row: int, label_widget: QLabel) -> str:
        tooltip = label_widget.toolTip().strip()
        if tooltip:
            return tooltip
        field_item = form_layout.itemAt(row, QFormLayout.FieldRole)
        if field_item is None:
            return ""
        field_widget = field_item.widget()
        if field_widget is not None:
            tooltip = field_widget.toolTip().strip()
            if tooltip:
                return tooltip
        field_layout = field_item.layout()
        if field_layout is None:
            return ""
        return self._first_layout_tooltip(field_layout)

    def _attach_form_row_help_icon(
        self,
        form_layout: QFormLayout,
        row: int,
        label_widget: QLabel,
        tooltip: str,
    ) -> None:
        if not tooltip:
            return
        label_text = label_widget.text().strip()
        if not label_text:
            return

        container = QWidget(form_layout.parentWidget())
        container.setObjectName("formRowHelpContainer")
        container.setProperty("formRowHelpContainer", True)
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        label_widget.setProperty("formRowHelpText", True)
        label_widget.setToolTip(tooltip)

        help_btn = QToolButton(container)
        help_btn.setObjectName("formRowHelpIcon")
        help_btn.setAutoRaise(True)
        help_btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        help_btn.setIconSize(QSize(12, 12))
        help_btn.setFixedSize(QSize(16, 16))
        help_btn.setFocusPolicy(Qt.NoFocus)
        help_btn.setToolTip(tooltip)
        help_btn.setCursor(Qt.WhatsThisCursor)
        help_btn.setProperty("formRowHelpText", label_text.rstrip(":"))
        help_btn.clicked.connect(
            lambda _=False, btn=help_btn, tip=tooltip: QToolTip.showText(
                btn.mapToGlobal(btn.rect().center()),
                tip,
                btn,
            )
        )

        # Keep the original QLabel instance alive so any stored references remain
        # valid for dynamic visibility updates (e.g., tuning peak-method handlers).
        form_layout.removeWidget(label_widget)
        label_widget.setParent(container)
        row_layout.addWidget(label_widget)
        row_layout.addWidget(help_btn, 0, Qt.AlignVCenter)
        form_layout.setWidget(row, QFormLayout.LabelRole, container)

    @staticmethod
    def _form_row_help_text_label(container: QWidget) -> QLabel | None:
        labels = container.findChildren(QLabel)
        for lbl in labels:
            if bool(lbl.property("formRowHelpText")):
                return lbl
        return labels[0] if labels else None

    def _set_form_row_help_tooltip(self, container: QWidget, tooltip: str) -> None:
        text_label = self._form_row_help_text_label(container)
        if text_label is not None:
            text_label.setToolTip(tooltip)
        help_btn = container.findChild(QToolButton, "formRowHelpIcon")
        if help_btn is not None:
            help_btn.setToolTip(tooltip)

    def _apply_form_row_tooltips(self, form_layout: QFormLayout) -> None:
        """Mirror tooltip text onto visible row labels for hover usability."""
        for row in range(form_layout.rowCount()):
            label_item = form_layout.itemAt(row, QFormLayout.LabelRole)
            if label_item is None:
                continue
            label_widget = label_item.widget()
            if label_widget is None:
                continue

            if isinstance(label_widget, QLabel):
                tooltip = self._resolve_form_row_tooltip(form_layout, row, label_widget)
                if tooltip:
                    label_widget.setToolTip(tooltip)
                    self._attach_form_row_help_icon(form_layout, row, label_widget, tooltip)
                continue

            if bool(label_widget.property("formRowHelpContainer")):
                text_label = self._form_row_help_text_label(label_widget)
                if text_label is None:
                    continue
                tooltip = self._resolve_form_row_tooltip(form_layout, row, text_label)
                if tooltip:
                    self._set_form_row_help_tooltip(label_widget, tooltip)


    def _build_run_configuration_group(self) -> QGroupBox:
        group = QGroupBox("Run Configuration")
        layout = QVBoxLayout(group)

        form = QFormLayout()
        self._configure_sidebar_form_layout(form)

        self._input_dir = QLineEdit()
        self._input_dir.setToolTip("The source recording/session folder to analyze.")
        self._input_dir.textChanged.connect(self._on_config_changed)
        input_row = QHBoxLayout()
        input_row.addWidget(self._input_dir)
        input_browse = QPushButton("Browse...")
        input_browse.clicked.connect(lambda: self._browse_dir(self._input_dir, "Select Input Directory"))
        input_row.addWidget(input_browse)
        form.addRow("Input Directory:", input_row)

        self._output_dir = QLineEdit()
        self._output_dir.setToolTip(
            "Where the run folder and deliverables will be created. Each run generates a unique timestamped subfolder."
        )
        self._output_dir.textChanged.connect(self._on_config_changed)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_dir)
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(lambda: self._browse_dir(self._output_dir, "Select Output Base Directory"))
        output_row.addWidget(output_browse)
        form.addRow("Output Directory:", output_row)

        self._format_combo = QComboBox()
        self._format_combo.addItems(list(FORMAT_CHOICES))
        self._format_combo.setToolTip(
            "Input format hint. Use auto to detect format from the selected input directory."
        )
        self._format_combo.currentIndexChanged.connect(self._on_config_changed)
        form.addRow("Format:", self._format_combo)

        self._sph_edit = QLineEdit()
        self._sph_edit.setPlaceholderText("(optional, integer >= 1)")
        self._sph_edit.setToolTip(
            "Sessions per hour used for duty-cycled/sessionized data when timestamps are incomplete. "
            "Required for duty-cycled data unless timestamps are available."
        )
        self._sph_edit.textChanged.connect(self._on_config_changed)
        self._sph_warning = QLabel(
            "Required for duty-cycled data unless timestamps are available."
        )
        self._sph_warning.setObjectName("sessionsPerHourWarning")
        self._sph_warning.setWordWrap(True)
        self._sph_warning.setStyleSheet("font-size: 11px;")
        self._sph_warning.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._sph_field_container = QWidget()
        self._sph_field_container.setObjectName("sessionsPerHourField")
        self._sph_field_container.setToolTip(self._sph_edit.toolTip())
        sph_layout = QVBoxLayout(self._sph_field_container)
        sph_layout.setContentsMargins(0, 0, 0, 0)
        sph_layout.setSpacing(2)
        sph_layout.addWidget(self._sph_edit)
        sph_layout.addWidget(self._sph_warning)
        form.addRow("Sessions/Hour:", self._sph_field_container)

        self._duration_edit = QLineEdit()
        self._duration_edit.setPlaceholderText("(optional, seconds > 0)")
        self._duration_edit.setToolTip(
            "Session duration in seconds. Leave blank to infer from timestamps where supported."
        )
        self._duration_edit.textChanged.connect(self._on_config_changed)
        form.addRow("Session Duration (s):", self._duration_edit)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["both", "phasic", "tonic"])
        self._mode_combo.setToolTip(
            "Select which analysis family to run: tonic, phasic, or both."
        )
        self._mode_combo.currentIndexChanged.connect(self._on_config_changed)
        form.addRow("Mode:", self._mode_combo)

        self._run_config_inputs_container = QWidget()
        inputs_layout = QVBoxLayout(self._run_config_inputs_container)
        inputs_layout.setContentsMargins(0, 0, 0, 0)
        inputs_layout.addLayout(form)
        self._apply_form_row_tooltips(form)

        roi_group = QGroupBox("ROI Selection")
        roi_layout = QVBoxLayout(roi_group)

        discover_row = QHBoxLayout()
        self._discover_btn = QPushButton("Select ROIs...")
        self._discover_btn.setToolTip("Discover and populate ROI choices from the selected input directory.")
        self._discover_btn.clicked.connect(self._on_discover)
        discover_row.addWidget(self._discover_btn)
        discover_row.addStretch()
        roi_layout.addLayout(discover_row)

        self._discovery_controls_hint = QLabel(
            "Click 'Select ROIs...' to populate ROI choices from the input directory."
        )
        self._discovery_controls_hint.setWordWrap(True)
        self._discovery_controls_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._discovery_controls_hint.setMinimumWidth(0)
        self._set_status_label_style(self._discovery_controls_hint, "warn")
        roi_layout.addWidget(self._discovery_controls_hint)

        self._roi_selection_container = QWidget()
        self._roi_selection_container.setVisible(False)
        roi_selection_layout = QVBoxLayout(self._roi_selection_container)
        roi_selection_layout.setContentsMargins(0, 0, 0, 0)
        self._roi_checked_label = QLabel("ROIs (checked = included):")
        self._roi_checked_label.setToolTip(
            "Checked ROIs are included in analysis and deliverables; unchecked ROIs are excluded."
        )
        roi_selection_layout.addWidget(self._roi_checked_label)
        self._roi_list = QListWidget()
        self._roi_list.setMaximumHeight(120)
        self._roi_list.setToolTip(
            "Select which ROIs to include in this run. Checked means included."
        )
        self._roi_list.itemChanged.connect(lambda _item: self._on_config_changed())
        roi_selection_layout.addWidget(self._roi_list)
        roi_layout.addWidget(self._roi_selection_container)

        # Keep include/exclude mode and bulk buttons for compatibility plumbing only.
        self._roi_filter_combo = QComboBox()
        self._roi_filter_combo.addItems(["Include selected", "Exclude selected"])
        self._roi_filter_combo.setCurrentIndex(0)
        self._roi_filter_combo.setVisible(False)
        self._roi_filter_combo.currentIndexChanged.connect(self._on_config_changed)
        roi_layout.addWidget(self._roi_filter_combo)

        self._roi_select_all_btn = QPushButton("Select all")
        self._roi_select_all_btn.clicked.connect(self._on_roi_select_all)
        self._roi_select_all_btn.setVisible(False)
        roi_layout.addWidget(self._roi_select_all_btn)
        self._roi_select_none_btn = QPushButton("Select none")
        self._roi_select_none_btn.clicked.connect(self._on_roi_select_none)
        self._roi_select_none_btn.setVisible(False)
        roi_layout.addWidget(self._roi_select_none_btn)

        # Discovery/session details are retained for plumbing but demoted from idle layout.
        hidden_discovery_details = QWidget()
        hidden_discovery_details.setVisible(False)
        hidden_discovery_layout = QVBoxLayout(hidden_discovery_details)
        self._discovery_summary = QLabel("No discovery run yet.")
        hidden_discovery_layout.addWidget(self._discovery_summary)
        self._sessions_list = QListWidget()
        self._sessions_list.setSelectionMode(QListWidget.NoSelection)
        hidden_discovery_layout.addWidget(self._sessions_list)
        self._rep_session_combo = QComboBox()
        self._rep_session_combo.addItem("(auto)")
        self._rep_session_combo.currentIndexChanged.connect(self._on_config_changed)
        hidden_discovery_layout.addWidget(self._rep_session_combo)
        self._rep_preview_hint = QLabel("")
        self._rep_preview_hint.setWordWrap(True)
        self._rep_preview_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._rep_preview_hint.setMinimumWidth(0)
        self._set_status_label_style(self._rep_preview_hint, "info")
        hidden_discovery_layout.addWidget(self._rep_preview_hint)
        roi_layout.addWidget(hidden_discovery_details)

        inputs_layout.addWidget(roi_group)
        layout.addWidget(self._run_config_inputs_container)

        self._run_actions_container = QWidget()
        self._run_actions_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        actions = QGridLayout(self._run_actions_container)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setHorizontalSpacing(8)
        actions.setVerticalSpacing(6)
        actions.setColumnStretch(0, 1)
        actions.setColumnStretch(1, 1)
        self._validate_btn = QPushButton("Validate Only")
        self._validate_btn.clicked.connect(self._on_validate)
        self._validate_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        actions.addWidget(self._validate_btn, 0, 0)
        self._run_btn = QPushButton("Run Pipeline")
        self._run_btn.setStyleSheet("font-weight: bold;")
        self._run_btn.clicked.connect(self._on_run)
        self._run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        actions.addWidget(self._run_btn, 0, 1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        actions.addWidget(self._cancel_btn, 1, 0)
        self._open_results_btn = QPushButton("Open Results...")
        self._open_results_btn.clicked.connect(self._on_open_results)
        self._open_results_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        actions.addWidget(self._open_results_btn, 1, 1)
        self._open_folder_btn = QPushButton("Open Run Folder")
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        self._open_folder_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        actions.addWidget(self._open_folder_btn, 2, 0, 1, 2)
        layout.addWidget(self._run_actions_container)

        self._run_reason_label = QLabel("Run status: Validation required before first run.")
        self._run_reason_label.setWordWrap(True)
        self._run_reason_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._run_reason_label.setMinimumWidth(0)
        self._set_status_label_style(self._run_reason_label, "warn")
        layout.addWidget(self._run_reason_label)
        return group

    def _build_plotting_group(self) -> QGroupBox:
        group = QGroupBox("Plotting")
        layout = QFormLayout(group)
        self._configure_sidebar_form_layout(layout)

        self._plotting_mode_combo = QComboBox()
        self._plotting_mode_combo.addItems(["Standard", "Full"])
        self._plotting_mode_combo.setCurrentText("Standard")
        self._plotting_mode_combo.setToolTip(
            "Standard uses QC-oriented rendering. Full generates all detailed plot variants."
        )
        self._plotting_mode_combo.currentIndexChanged.connect(self._on_plotting_mode_changed)
        self._plotting_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        layout.addRow("Plotting Mode:", self._plotting_mode_combo)

        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.01, 100.0)
        self._smooth_spin.setValue(1.0)
        self._smooth_spin.setDecimals(2)
        self._smooth_spin.setSingleStep(0.1)
        self._smooth_spin.setToolTip(
            "Duration in seconds for plot smoothing when smoothing is supported."
        )
        self._smooth_spin.valueChanged.connect(self._on_config_changed)
        layout.addRow("Smoothing Window Duration (s):", self._smooth_spin)

        self._timeline_anchor_mode_combo = QComboBox()
        self._timeline_anchor_mode_combo.addItem("Civil clock", "civil")
        self._timeline_anchor_mode_combo.addItem("Elapsed from first session", "elapsed")
        self._timeline_anchor_mode_combo.addItem("Fixed daily anchor", "fixed_daily_anchor")
        self._timeline_anchor_mode_combo.setCurrentIndex(0)
        self._timeline_anchor_mode_combo.setToolTip(
            "Choose how timeline x-positions are anchored:\n"
            "Civil clock: keeps real time-of-day placement (including OFF gaps).\n"
            "Elapsed from first session: starts at t=0 from the first session.\n"
            "Fixed daily anchor: aligns each day/session to one clock anchor."
        )
        self._timeline_anchor_mode_combo.currentIndexChanged.connect(self._on_timeline_anchor_mode_changed)
        self._timeline_anchor_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        layout.addRow("Timeline Anchor:", self._timeline_anchor_mode_combo)

        self._fixed_daily_anchor_time_edit = QLineEdit("07:00")
        self._fixed_daily_anchor_time_edit.setPlaceholderText("HH:MM or HH:MM:SS")
        self._fixed_daily_anchor_time_edit.setToolTip(
            "Clock anchor used only when Timeline Anchor is Fixed daily anchor."
        )
        self._fixed_daily_anchor_time_edit.textChanged.connect(self._on_config_changed)
        layout.addRow("Fixed Anchor Time:", self._fixed_daily_anchor_time_edit)
        self._timeline_anchor_help_label = QLabel(
            "Civil clock: real time-of-day\n"
            "Elapsed from first session: starts at 0\n"
            "Fixed daily anchor: align each day to one clock time"
        )
        self._timeline_anchor_help_label.setWordWrap(True)
        self._timeline_anchor_help_label.setObjectName("resultsSummaryHint")
        layout.addRow("", self._timeline_anchor_help_label)
        self._apply_form_row_tooltips(layout)
        self._on_timeline_anchor_mode_changed()

        self._mode_context_label = QLabel("")
        self._mode_context_label.setWordWrap(True)
        self._mode_context_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._mode_context_label.setMinimumWidth(0)
        self._set_status_label_style(self._mode_context_label, "info")
        layout.addRow("", self._mode_context_label)
        return group

    def _on_timeline_anchor_mode_changed(self) -> None:
        """Toggle fixed-anchor time input visibility/enabled state."""
        if not hasattr(self, "_fixed_daily_anchor_time_edit"):
            return
        fixed_mode = self._timeline_anchor_mode_value() == "fixed_daily_anchor"
        phasic_active = is_isosbestic_active(self._mode_combo.currentText())
        plotting_layout = self._plotting_group.layout() if hasattr(self, "_plotting_group") else None
        if isinstance(plotting_layout, QFormLayout):
            fixed_label = plotting_layout.labelForField(self._fixed_daily_anchor_time_edit)
            if fixed_label is not None:
                fixed_label.setVisible(fixed_mode)
        self._fixed_daily_anchor_time_edit.setVisible(fixed_mode)
        self._fixed_daily_anchor_time_edit.setEnabled(bool(fixed_mode and phasic_active))

    def _on_plotting_mode_changed(self) -> None:
        """Map single plotting mode control to legacy per-family render settings."""
        target = "full" if self._plotting_mode_combo.currentText() == "Full" else "qc"
        for attr in (
            "_sig_iso_render_mode_combo",
            "_dff_render_mode_combo",
            "_stacked_render_mode_combo",
        ):
            combo = getattr(self, attr, None)
            if combo is not None and combo.findText(target) >= 0:
                combo.blockSignals(True)
                combo.setCurrentText(target)
                combo.blockSignals(False)

    def _build_advanced_group(self) -> QGroupBox:
        group = QGroupBox("Advanced")
        outer = QVBoxLayout(group)

        disclosure_row = QHBoxLayout()
        self._advanced_disclosure_btn = QToolButton()
        self._advanced_disclosure_btn.setText("Advanced controls")
        self._advanced_disclosure_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._advanced_disclosure_btn.setArrowType(Qt.RightArrow)
        self._advanced_disclosure_btn.setCheckable(True)
        self._advanced_disclosure_btn.setChecked(False)
        self._advanced_disclosure_btn.setAutoRaise(True)
        self._advanced_disclosure_btn.toggled.connect(self._on_advanced_disclosure_toggled)
        disclosure_row.addWidget(self._advanced_disclosure_btn)
        disclosure_row.addStretch()
        outer.addLayout(disclosure_row)

        self._advanced_content = QWidget()
        content_layout = QVBoxLayout(self._advanced_content)
        outer.addWidget(self._advanced_content)
        self._advanced_content.setVisible(False)

        self._adv_group = QGroupBox("Isosbestic Correction")
        iso_layout = QVBoxLayout(self._adv_group)

        iso_sampling = QGroupBox("Sampling Geometry")
        iso_sampling_form = QFormLayout(iso_sampling)
        self._configure_sidebar_form_layout(iso_sampling_form)
        self._dynamic_fit_mode_combo = QComboBox()
        self._dynamic_fit_mode_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._dynamic_fit_mode_combo.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["dynamic_fit_mode"]
        )
        for mode_name in get_allowed_dynamic_fit_modes_from_config():
            self._dynamic_fit_mode_combo.addItem(dynamic_fit_mode_label(mode_name), mode_name)
        default_fit_mode = normalize_dynamic_fit_mode(
            str(getattr(self._default_cfg, "dynamic_fit_mode", "rolling_local_regression"))
        )
        idx_mode = self._dynamic_fit_mode_combo.findData(default_fit_mode)
        if idx_mode < 0 and self._dynamic_fit_mode_combo.count() > 0:
            idx_mode = 0
        if idx_mode >= 0:
            self._dynamic_fit_mode_combo.setCurrentIndex(idx_mode)
        self._dynamic_fit_mode_combo.currentIndexChanged.connect(self._on_dynamic_fit_mode_changed)
        self._dynamic_fit_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        iso_sampling_form.addRow("Dynamic Fit Mode:", self._dynamic_fit_mode_combo)

        self._dynamic_fit_mode_note = QLabel("")
        self._dynamic_fit_mode_note.setWordWrap(True)
        self._dynamic_fit_mode_note.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._dynamic_fit_mode_note.setMinimumWidth(0)
        self._dynamic_fit_mode_note.setStyleSheet("font-size: 11px;")
        iso_sampling_form.addRow(self._dynamic_fit_mode_note)

        self._baseline_subtract_before_fit_cb = QCheckBox("")
        self._baseline_subtract_before_fit_cb.setChecked(
            bool(getattr(self._default_cfg, "baseline_subtract_before_fit", False))
        )
        self._baseline_subtract_before_fit_cb.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["baseline_subtract_before_fit"]
        )
        self._baseline_subtract_before_fit_cb.stateChanged.connect(self._on_config_changed)
        iso_sampling_form.addRow(
            "Baseline subtract before fit:",
            self._baseline_subtract_before_fit_cb,
        )

        self._window_sec_edit = QLineEdit(str(self._default_cfg.window_sec))
        self._window_sec_edit.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["regression_window_sec"]
        )
        self._window_sec_edit.textChanged.connect(self._on_config_changed)
        iso_sampling_form.addRow("Regression Window:", self._window_sec_edit)
        self._apply_form_row_tooltips(iso_sampling_form)
        iso_layout.addWidget(iso_sampling)

        iso_accept = QGroupBox("Window Constraints")
        iso_accept_form = QFormLayout(iso_accept)
        self._configure_sidebar_form_layout(iso_accept_form)
        self._min_samples_per_window_spin = QSpinBox()
        self._min_samples_per_window_spin.setRange(1, 100000)
        self._min_samples_per_window_spin.setValue(max(1, self._default_cfg.min_samples_per_window))
        self._min_samples_per_window_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["min_samples_per_window"]
        )
        self._min_samples_per_window_spin.valueChanged.connect(self._on_config_changed)
        iso_accept_form.addRow("Min Samples per Window:", self._min_samples_per_window_spin)
        self._apply_form_row_tooltips(iso_accept_form)
        iso_layout.addWidget(iso_accept)

        robust_group = QGroupBox("Robust Event Rejection")
        robust_form = QFormLayout(robust_group)
        self._configure_sidebar_form_layout(robust_form)

        self._robust_event_reject_max_iters_spin = QSpinBox()
        self._robust_event_reject_max_iters_spin.setRange(1, 1000)
        self._robust_event_reject_max_iters_spin.setValue(
            int(getattr(self._default_cfg, "robust_event_reject_max_iters", 3))
        )
        self._robust_event_reject_max_iters_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_max_iters"]
        )
        self._robust_event_reject_max_iters_spin.valueChanged.connect(self._on_config_changed)
        robust_form.addRow("Max iterations:", self._robust_event_reject_max_iters_spin)

        self._robust_event_reject_residual_z_spin = QDoubleSpinBox()
        self._robust_event_reject_residual_z_spin.setRange(0.000001, 1_000_000.0)
        self._robust_event_reject_residual_z_spin.setDecimals(4)
        self._robust_event_reject_residual_z_spin.setSingleStep(0.1)
        self._robust_event_reject_residual_z_spin.setValue(
            float(getattr(self._default_cfg, "robust_event_reject_residual_z_thresh", 3.5))
        )
        self._robust_event_reject_residual_z_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_residual_z_thresh"]
        )
        self._robust_event_reject_residual_z_spin.valueChanged.connect(self._on_config_changed)
        robust_form.addRow("Residual z-threshold:", self._robust_event_reject_residual_z_spin)

        self._robust_event_reject_local_var_window_spin = QDoubleSpinBox()
        self._robust_event_reject_local_var_window_spin.setRange(0.000001, 1_000_000.0)
        self._robust_event_reject_local_var_window_spin.setDecimals(4)
        self._robust_event_reject_local_var_window_spin.setSingleStep(0.5)
        self._robust_event_reject_local_var_window_spin.setValue(
            float(getattr(self._default_cfg, "robust_event_reject_local_var_window_sec", 10.0) or 10.0)
        )
        self._robust_event_reject_local_var_window_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_local_var_window_sec"]
        )
        self._robust_event_reject_local_var_window_spin.valueChanged.connect(self._on_config_changed)
        robust_form.addRow("Local variance window (s):", self._robust_event_reject_local_var_window_spin)

        ratio_enabled = (
            getattr(self._default_cfg, "robust_event_reject_local_var_ratio_thresh", None)
            is not None
        )
        ratio_value = getattr(self._default_cfg, "robust_event_reject_local_var_ratio_thresh", None)
        self._robust_event_reject_local_var_ratio_enable_cb = QCheckBox("Enable")
        self._robust_event_reject_local_var_ratio_enable_cb.setChecked(bool(ratio_enabled))
        self._robust_event_reject_local_var_ratio_enable_cb.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_local_var_ratio_thresh"]
        )
        self._robust_event_reject_local_var_ratio_spin = QDoubleSpinBox()
        self._robust_event_reject_local_var_ratio_spin.setRange(0.000001, 1_000_000.0)
        self._robust_event_reject_local_var_ratio_spin.setDecimals(4)
        self._robust_event_reject_local_var_ratio_spin.setSingleStep(0.1)
        self._robust_event_reject_local_var_ratio_spin.setValue(
            float(ratio_value if ratio_value is not None else 3.0)
        )
        self._robust_event_reject_local_var_ratio_spin.setEnabled(bool(ratio_enabled))
        self._robust_event_reject_local_var_ratio_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_local_var_ratio_thresh"]
        )
        self._robust_event_reject_local_var_ratio_enable_cb.toggled.connect(
            self._robust_event_reject_local_var_ratio_spin.setEnabled
        )
        self._robust_event_reject_local_var_ratio_enable_cb.toggled.connect(self._on_config_changed)
        self._robust_event_reject_local_var_ratio_spin.valueChanged.connect(self._on_config_changed)
        ratio_row = QWidget()
        ratio_row_layout = QHBoxLayout(ratio_row)
        ratio_row_layout.setContentsMargins(0, 0, 0, 0)
        ratio_row_layout.setSpacing(8)
        ratio_row.setToolTip(_DYNAMIC_FIT_TOOLTIPS["robust_local_var_ratio_thresh"])
        ratio_row_layout.addWidget(self._robust_event_reject_local_var_ratio_enable_cb)
        ratio_row_layout.addWidget(self._robust_event_reject_local_var_ratio_spin)
        robust_form.addRow("Local variance ratio threshold:", ratio_row)

        self._robust_event_reject_min_keep_fraction_spin = QDoubleSpinBox()
        self._robust_event_reject_min_keep_fraction_spin.setRange(0.000001, 1.0)
        self._robust_event_reject_min_keep_fraction_spin.setDecimals(4)
        self._robust_event_reject_min_keep_fraction_spin.setSingleStep(0.05)
        self._robust_event_reject_min_keep_fraction_spin.setValue(
            float(getattr(self._default_cfg, "robust_event_reject_min_keep_fraction", 0.5))
        )
        self._robust_event_reject_min_keep_fraction_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["robust_min_keep_fraction"]
        )
        self._robust_event_reject_min_keep_fraction_spin.valueChanged.connect(self._on_config_changed)
        robust_form.addRow("Minimum keep fraction:", self._robust_event_reject_min_keep_fraction_spin)
        self._apply_form_row_tooltips(robust_form)
        iso_layout.addWidget(robust_group)

        adaptive_group = QGroupBox("Adaptive Event-Gated Regression")
        adaptive_form = QFormLayout(adaptive_group)
        self._configure_sidebar_form_layout(adaptive_form)

        self._adaptive_event_gate_residual_z_spin = QDoubleSpinBox()
        self._adaptive_event_gate_residual_z_spin.setRange(0.000001, 1_000_000.0)
        self._adaptive_event_gate_residual_z_spin.setDecimals(4)
        self._adaptive_event_gate_residual_z_spin.setSingleStep(0.1)
        self._adaptive_event_gate_residual_z_spin.setValue(
            float(getattr(self._default_cfg, "adaptive_event_gate_residual_z_thresh", 3.5))
        )
        self._adaptive_event_gate_residual_z_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_residual_z_thresh"]
        )
        self._adaptive_event_gate_residual_z_spin.valueChanged.connect(self._on_config_changed)
        adaptive_form.addRow("Residual z-threshold:", self._adaptive_event_gate_residual_z_spin)

        self._adaptive_event_gate_local_var_window_spin = QDoubleSpinBox()
        self._adaptive_event_gate_local_var_window_spin.setRange(0.000001, 1_000_000.0)
        self._adaptive_event_gate_local_var_window_spin.setDecimals(4)
        self._adaptive_event_gate_local_var_window_spin.setSingleStep(0.5)
        self._adaptive_event_gate_local_var_window_spin.setValue(
            float(getattr(self._default_cfg, "adaptive_event_gate_local_var_window_sec", 10.0) or 10.0)
        )
        self._adaptive_event_gate_local_var_window_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_local_var_window_sec"]
        )
        self._adaptive_event_gate_local_var_window_spin.valueChanged.connect(self._on_config_changed)
        adaptive_form.addRow("Local variance window (s):", self._adaptive_event_gate_local_var_window_spin)

        adaptive_ratio_enabled = (
            getattr(self._default_cfg, "adaptive_event_gate_local_var_ratio_thresh", None)
            is not None
        )
        adaptive_ratio_value = getattr(
            self._default_cfg, "adaptive_event_gate_local_var_ratio_thresh", None
        )
        self._adaptive_event_gate_local_var_ratio_enable_cb = QCheckBox("Enable")
        self._adaptive_event_gate_local_var_ratio_enable_cb.setChecked(bool(adaptive_ratio_enabled))
        self._adaptive_event_gate_local_var_ratio_enable_cb.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_local_var_ratio_thresh"]
        )
        self._adaptive_event_gate_local_var_ratio_spin = QDoubleSpinBox()
        self._adaptive_event_gate_local_var_ratio_spin.setRange(0.000001, 1_000_000.0)
        self._adaptive_event_gate_local_var_ratio_spin.setDecimals(4)
        self._adaptive_event_gate_local_var_ratio_spin.setSingleStep(0.1)
        self._adaptive_event_gate_local_var_ratio_spin.setValue(
            float(adaptive_ratio_value if adaptive_ratio_value is not None else 3.0)
        )
        self._adaptive_event_gate_local_var_ratio_spin.setEnabled(bool(adaptive_ratio_enabled))
        self._adaptive_event_gate_local_var_ratio_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_local_var_ratio_thresh"]
        )
        self._adaptive_event_gate_local_var_ratio_enable_cb.toggled.connect(
            self._adaptive_event_gate_local_var_ratio_spin.setEnabled
        )
        self._adaptive_event_gate_local_var_ratio_enable_cb.toggled.connect(self._on_config_changed)
        self._adaptive_event_gate_local_var_ratio_spin.valueChanged.connect(self._on_config_changed)
        adaptive_ratio_row = QWidget()
        adaptive_ratio_row.setToolTip(_DYNAMIC_FIT_TOOLTIPS["adaptive_local_var_ratio_thresh"])
        adaptive_ratio_row_layout = QHBoxLayout(adaptive_ratio_row)
        adaptive_ratio_row_layout.setContentsMargins(0, 0, 0, 0)
        adaptive_ratio_row_layout.setSpacing(8)
        adaptive_ratio_row_layout.addWidget(self._adaptive_event_gate_local_var_ratio_enable_cb)
        adaptive_ratio_row_layout.addWidget(self._adaptive_event_gate_local_var_ratio_spin)
        adaptive_form.addRow("Local variance ratio threshold:", adaptive_ratio_row)

        self._adaptive_event_gate_smooth_window_spin = QDoubleSpinBox()
        self._adaptive_event_gate_smooth_window_spin.setRange(0.000001, 1_000_000.0)
        self._adaptive_event_gate_smooth_window_spin.setDecimals(4)
        self._adaptive_event_gate_smooth_window_spin.setSingleStep(1.0)
        self._adaptive_event_gate_smooth_window_spin.setValue(
            float(getattr(self._default_cfg, "adaptive_event_gate_smooth_window_sec", 60.0))
        )
        self._adaptive_event_gate_smooth_window_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_smooth_window_sec"]
        )
        self._adaptive_event_gate_smooth_window_spin.valueChanged.connect(self._on_config_changed)
        adaptive_form.addRow(
            "Smoothing window duration (s):",
            self._adaptive_event_gate_smooth_window_spin,
        )

        self._adaptive_event_gate_min_trust_fraction_spin = QDoubleSpinBox()
        self._adaptive_event_gate_min_trust_fraction_spin.setRange(0.000001, 1.0)
        self._adaptive_event_gate_min_trust_fraction_spin.setDecimals(4)
        self._adaptive_event_gate_min_trust_fraction_spin.setSingleStep(0.05)
        self._adaptive_event_gate_min_trust_fraction_spin.setValue(
            float(getattr(self._default_cfg, "adaptive_event_gate_min_trust_fraction", 0.5))
        )
        self._adaptive_event_gate_min_trust_fraction_spin.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_min_trust_fraction"]
        )
        self._adaptive_event_gate_min_trust_fraction_spin.valueChanged.connect(self._on_config_changed)
        adaptive_form.addRow("Minimum trust fraction:", self._adaptive_event_gate_min_trust_fraction_spin)

        self._adaptive_event_gate_freeze_interp_combo = QComboBox()
        self._adaptive_event_gate_freeze_interp_combo.addItem("linear_hold", "linear_hold")
        default_adaptive_freeze = str(
            getattr(self._default_cfg, "adaptive_event_gate_freeze_interp_method", "linear_hold")
        ).strip() or "linear_hold"
        idx_adaptive_freeze = self._adaptive_event_gate_freeze_interp_combo.findData(default_adaptive_freeze)
        if idx_adaptive_freeze >= 0:
            self._adaptive_event_gate_freeze_interp_combo.setCurrentIndex(idx_adaptive_freeze)
        self._adaptive_event_gate_freeze_interp_combo.setToolTip(
            _DYNAMIC_FIT_TOOLTIPS["adaptive_freeze_interp_method"]
        )
        self._adaptive_event_gate_freeze_interp_combo.currentIndexChanged.connect(self._on_config_changed)
        adaptive_form.addRow("Freeze interpolation method:", self._adaptive_event_gate_freeze_interp_combo)
        self._apply_form_row_tooltips(adaptive_form)
        iso_layout.addWidget(adaptive_group)

        content_layout.addWidget(self._adv_group)

        self._adv_prep_group = QGroupBox("Preprocessing")
        prep_layout = QFormLayout(self._adv_prep_group)
        self._configure_sidebar_form_layout(prep_layout)
        self._lowpass_hz_edit = QLineEdit(str(self._default_cfg.lowpass_hz))
        self._lowpass_hz_edit.setToolTip(
            "Lowpass cutoff (Hz) applied before feature-oriented computations."
        )
        self._lowpass_hz_edit.textChanged.connect(self._on_config_changed)
        prep_layout.addRow("Lowpass Filter:", self._lowpass_hz_edit)

        self._tonic_output_mode_combo = QComboBox()
        self._tonic_output_mode_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tonic_output_mode_combo.setToolTip(_TONIC_OUTPUT_MODE_TOOLTIP)
        for mode_name in get_allowed_tonic_output_modes_from_config():
            self._tonic_output_mode_combo.addItem(
                tonic_output_mode_label(mode_name),
                mode_name,
            )
        default_tonic_mode = normalize_tonic_output_mode(
            str(getattr(self._default_cfg, "tonic_output_mode", TONIC_OUTPUT_MODE_PRESERVE_RAW))
        )
        idx_tonic_mode = self._tonic_output_mode_combo.findData(default_tonic_mode)
        if idx_tonic_mode < 0 and self._tonic_output_mode_combo.count() > 0:
            idx_tonic_mode = 0
        if idx_tonic_mode >= 0:
            self._tonic_output_mode_combo.setCurrentIndex(idx_tonic_mode)
        self._tonic_output_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        prep_layout.addRow("Tonic Output Mode:", self._tonic_output_mode_combo)

        self._tonic_timeline_mode_combo = QComboBox()
        self._tonic_timeline_mode_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tonic_timeline_mode_combo.setToolTip(_TONIC_TIMELINE_MODE_TOOLTIP)
        for mode_name in get_allowed_tonic_timeline_modes_from_config():
            self._tonic_timeline_mode_combo.addItem(
                tonic_timeline_mode_label(mode_name),
                mode_name,
            )
        default_tonic_timeline_mode = normalize_tonic_timeline_mode(
            str(getattr(self._default_cfg, "tonic_timeline_mode", TONIC_TIMELINE_MODE_REAL_ELAPSED))
        )
        idx_tonic_timeline_mode = self._tonic_timeline_mode_combo.findData(default_tonic_timeline_mode)
        if idx_tonic_timeline_mode < 0 and self._tonic_timeline_mode_combo.count() > 0:
            idx_tonic_timeline_mode = 0
        if idx_tonic_timeline_mode >= 0:
            self._tonic_timeline_mode_combo.setCurrentIndex(idx_tonic_timeline_mode)
        self._tonic_timeline_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        prep_layout.addRow("Tonic Timeline Mode:", self._tonic_timeline_mode_combo)

        self._apply_form_row_tooltips(prep_layout)
        content_layout.addWidget(self._adv_prep_group)

        baseline_group = QGroupBox("Baseline / Normalization")
        baseline_layout = QFormLayout(baseline_group)
        self._configure_sidebar_form_layout(baseline_layout)
        self._baseline_method_combo = QComboBox()
        allowed_methods = get_allowed_baseline_methods_from_config()
        if self._default_cfg.baseline_method not in allowed_methods:
            allowed_methods.append(self._default_cfg.baseline_method)
        self._baseline_method_combo.addItems(sorted(allowed_methods))
        idx = self._baseline_method_combo.findText(self._default_cfg.baseline_method)
        if idx >= 0:
            self._baseline_method_combo.setCurrentIndex(idx)
        self._baseline_method_combo.setToolTip(
            "Method used to estimate baseline F0 for normalization."
        )
        self._baseline_method_combo.currentIndexChanged.connect(self._on_config_changed)
        baseline_layout.addRow("Baseline Method:", self._baseline_method_combo)
        self._baseline_percentile_label = QLabel("Baseline Percentile:")
        self._baseline_percentile_edit = QLineEdit(str(self._default_cfg.baseline_percentile))
        self._baseline_percentile_edit.setToolTip(
            "Percentile value used when the selected baseline method is percentile-based."
        )
        self._baseline_percentile_edit.textChanged.connect(self._on_config_changed)
        baseline_layout.addRow(self._baseline_percentile_label, self._baseline_percentile_edit)
        self._f0_min_value_edit = QLineEdit(str(self._default_cfg.f0_min_value))
        self._f0_min_value_edit.setToolTip(
            "Lower bound applied to F0 to avoid unstable normalization near zero."
        )
        self._f0_min_value_edit.textChanged.connect(self._on_config_changed)
        baseline_layout.addRow("F0 Min Value:", self._f0_min_value_edit)
        self._apply_form_row_tooltips(baseline_layout)
        content_layout.addWidget(baseline_group)

        self._adv_ev_group = QGroupBox("Feature Detection")
        ev_layout = QFormLayout(self._adv_ev_group)
        self._configure_sidebar_form_layout(ev_layout)
        self._event_signal_combo = QComboBox()
        allowed_sigs = get_allowed_event_signals_from_config()
        if self._default_cfg.event_signal not in allowed_sigs:
            allowed_sigs.append(self._default_cfg.event_signal)
        self._event_signal_combo.addItems(sorted(allowed_sigs))
        idx = self._event_signal_combo.findText(self._default_cfg.event_signal)
        if idx >= 0:
            self._event_signal_combo.setCurrentIndex(idx)
        self._event_signal_combo.setToolTip(
            "Signal channel used for peak detection and event-derived metrics."
        )
        self._event_signal_combo.currentIndexChanged.connect(self._on_config_changed)
        ev_layout.addRow("Event Signal:", self._event_signal_combo)

        self._peak_method_combo = QComboBox()
        allowed_peak_methods = get_allowed_peak_threshold_methods_from_config()
        if self._default_cfg.peak_threshold_method not in allowed_peak_methods:
            allowed_peak_methods.append(self._default_cfg.peak_threshold_method)
        self._peak_method_combo.addItems(sorted(allowed_peak_methods))
        idx = self._peak_method_combo.findText(self._default_cfg.peak_threshold_method)
        if idx >= 0:
            self._peak_method_combo.setCurrentIndex(idx)
        self._peak_method_combo.setToolTip(
            "Thresholding method used to determine which peaks are counted as events."
        )
        self._peak_method_combo.currentIndexChanged.connect(self._on_config_changed)
        ev_layout.addRow("Peak Threshold Method:", self._peak_method_combo)

        self._peak_k_label = QLabel("Peak Threshold K:")
        self._peak_k_edit = QLineEdit(str(self._default_cfg.peak_threshold_k))
        self._peak_k_edit.setToolTip(
            "Scale factor used by threshold methods that require a K parameter."
        )
        self._peak_k_edit.textChanged.connect(self._on_config_changed)
        ev_layout.addRow(self._peak_k_label, self._peak_k_edit)
        self._peak_pct_label = QLabel("Peak Threshold Percentile:")
        self._peak_pct_edit = QLineEdit(str(self._default_cfg.peak_threshold_percentile))
        self._peak_pct_edit.setToolTip(
            "Percentile cutoff used by percentile-based threshold methods."
        )
        self._peak_pct_edit.textChanged.connect(self._on_config_changed)
        ev_layout.addRow(self._peak_pct_label, self._peak_pct_edit)
        self._peak_abs_label = QLabel("Peak Threshold Absolute:")
        self._peak_abs_edit = QLineEdit(str(self._default_cfg.peak_threshold_abs))
        self._peak_abs_edit.setToolTip(
            "Absolute threshold used only when the selected method requires a fixed cutoff."
        )
        self._peak_abs_edit.textChanged.connect(self._on_config_changed)
        ev_layout.addRow(self._peak_abs_label, self._peak_abs_edit)
        self._peak_dist_edit = QLineEdit(str(self._default_cfg.peak_min_distance_sec))
        self._peak_dist_edit.setToolTip(
            "Minimum spacing between detected peaks, in seconds."
        )
        self._peak_dist_edit.textChanged.connect(self._on_config_changed)
        ev_layout.addRow("Peak Min Distance:", self._peak_dist_edit)

        self._peak_min_prominence_k_edit = QLineEdit(
            str(float(getattr(self._default_cfg, "peak_min_prominence_k", 0.0)))
        )
        self._peak_min_prominence_k_edit.setToolTip(
            "Minimum required peak prominence relative to robust noise. "
            "Higher values reject small fluctuations. Set to 0 to disable."
        )
        self._peak_min_prominence_k_edit.textChanged.connect(self._on_config_changed)
        ev_layout.addRow("Peak Min Prominence K:", self._peak_min_prominence_k_edit)

        self._peak_min_width_sec_edit = QLineEdit(
            str(float(getattr(self._default_cfg, "peak_min_width_sec", 0.0)))
        )
        self._peak_min_width_sec_edit.setToolTip(
            "Minimum required peak width in seconds. "
            "Higher values reject very narrow excursions. Set to 0 to disable."
        )
        self._peak_min_width_sec_edit.textChanged.connect(self._on_config_changed)
        ev_layout.addRow("Peak Min Width (s):", self._peak_min_width_sec_edit)

        self._peak_pre_filter_combo = QComboBox()
        allowed_pre_filters = get_allowed_peak_pre_filters_from_config()
        default_pre_filter = str(getattr(self._default_cfg, "peak_pre_filter", "none"))
        if default_pre_filter not in allowed_pre_filters:
            allowed_pre_filters.append(default_pre_filter)
        self._peak_pre_filter_combo.addItems(sorted(allowed_pre_filters))
        idx = self._peak_pre_filter_combo.findText(default_pre_filter)
        if idx >= 0:
            self._peak_pre_filter_combo.setCurrentIndex(idx)
        self._peak_pre_filter_combo.setToolTip(
            "Optional pre-filter applied before running peak detection."
        )
        self._peak_pre_filter_combo.currentIndexChanged.connect(self._on_config_changed)
        ev_layout.addRow("Peak Pre-Filter:", self._peak_pre_filter_combo)

        self._event_auc_combo = QComboBox()
        allowed_auc = get_allowed_event_auc_baselines_from_config()
        if self._default_cfg.event_auc_baseline not in allowed_auc:
            allowed_auc.append(self._default_cfg.event_auc_baseline)
        self._event_auc_combo.addItems(sorted(allowed_auc))
        idx = self._event_auc_combo.findText(self._default_cfg.event_auc_baseline)
        if idx >= 0:
            self._event_auc_combo.setCurrentIndex(idx)
        self._event_auc_combo.setToolTip(
            "Reference baseline convention used when integrating event AUC."
        )
        self._event_auc_combo.currentIndexChanged.connect(self._on_config_changed)
        ev_layout.addRow("Event AUC Baseline:", self._event_auc_combo)
        self._apply_form_row_tooltips(ev_layout)
        content_layout.addWidget(self._adv_ev_group)

        cfg_group = QGroupBox("Config Source (Advanced)")
        cfg_layout = QFormLayout(cfg_group)
        self._configure_sidebar_form_layout(cfg_layout)
        self._use_custom_config_cb = QCheckBox("Use custom config YAML")
        self._use_custom_config_cb.setToolTip(
            "Enable only if you need a non-standard baseline YAML instead of the lab default."
        )
        self._use_custom_config_cb.stateChanged.connect(self._on_config_changed)
        self._use_custom_config_cb.toggled.connect(self._update_config_source_ui)
        cfg_layout.addRow("Baseline Source:", self._use_custom_config_cb)
        self._config_path = QLineEdit()
        self._config_path.setPlaceholderText("(optional) custom baseline config path")
        self._config_path.setToolTip(
            "Path to custom baseline YAML used when custom config mode is enabled."
        )
        self._config_path.textChanged.connect(self._on_config_changed)
        self._config_path.textChanged.connect(self._update_config_source_ui)
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(self._config_path)
        self._config_browse_btn = QPushButton("Browse...")
        self._config_browse_btn.setToolTip("Browse for a custom baseline YAML file.")
        self._config_browse_btn.clicked.connect(self._browse_config)
        cfg_row.addWidget(self._config_browse_btn)
        cfg_layout.addRow("Custom Config YAML:", cfg_row)
        self._active_config_source_label = QLabel("")
        self._active_config_source_label.setWordWrap(True)
        self._active_config_source_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._active_config_source_label.setMinimumWidth(0)
        self._set_status_label_style(self._active_config_source_label, "info")
        cfg_layout.addRow("", self._active_config_source_label)
        self._apply_form_row_tooltips(cfg_layout)
        content_layout.addWidget(cfg_group)

        content_layout.addStretch()

        self._baseline_method_combo.currentIndexChanged.connect(self._update_adv_prep_visibility)
        self._peak_method_combo.currentIndexChanged.connect(self._update_adv_ev_visibility)
        self._mode_combo.currentIndexChanged.connect(self._update_adv_group_visibility)
        self._on_dynamic_fit_mode_changed()
        self._update_adv_prep_visibility()
        self._update_adv_ev_visibility()
        self._update_adv_group_visibility()
        self._update_config_source_ui()
        return group

    def _on_advanced_disclosure_toggled(self, expanded: bool) -> None:
        """Toggle advanced section content via disclosure-style header."""
        self._advanced_content.setVisible(expanded)
        self._advanced_disclosure_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._refresh_left_column_width_clamp()
        QTimer.singleShot(0, self._refresh_left_column_width_clamp)

    def _build_hidden_compatibility_group(self) -> QWidget:
        group = QGroupBox("Compatibility Controls")
        group.setVisible(False)
        layout = QVBoxLayout(group)

        self._traces_only_cb = QCheckBox("Skip feature extraction (traces and QC only)")
        self._traces_only_cb.stateChanged.connect(self._on_config_changed)
        layout.addWidget(self._traces_only_cb)

        # Legacy per-family render controls retained for compatibility plumbing.
        self._sig_iso_render_mode_combo = QComboBox()
        self._sig_iso_render_mode_combo.addItems(["qc", "full"])
        self._sig_iso_render_mode_combo.setVisible(False)
        self._dff_render_mode_combo = QComboBox()
        self._dff_render_mode_combo.addItems(["qc", "full"])
        self._dff_render_mode_combo.setVisible(False)
        self._stacked_render_mode_combo = QComboBox()
        self._stacked_render_mode_combo.addItems(["qc", "full"])
        self._stacked_render_mode_combo.setVisible(False)
        layout.addWidget(self._sig_iso_render_mode_combo)
        layout.addWidget(self._dff_render_mode_combo)
        layout.addWidget(self._stacked_render_mode_combo)
        self._on_plotting_mode_changed()

        preview_row = QHBoxLayout()
        self._preview_enabled_cb = QCheckBox("Limit sessions")
        self._preview_enabled_cb.stateChanged.connect(self._on_config_changed)
        preview_row.addWidget(self._preview_enabled_cb)
        self._preview_n_spin = QSpinBox()
        self._preview_n_spin.setRange(1, 100000)
        self._preview_n_spin.setValue(5)
        self._preview_n_spin.setEnabled(False)
        self._preview_enabled_cb.toggled.connect(self._preview_n_spin.setEnabled)
        self._preview_n_spin.valueChanged.connect(self._on_config_changed)
        preview_row.addWidget(self._preview_n_spin)
        preview_row.addStretch()
        layout.addLayout(preview_row)

        self._recursive_cb = QCheckBox("Always enabled by runner")
        self._recursive_cb.setChecked(True)
        self._recursive_cb.setEnabled(False)
        layout.addWidget(self._recursive_cb)

        self._overwrite_cb = QCheckBox("Overwrite existing output (legacy CLI only)")
        self._overwrite_cb.setEnabled(False)
        layout.addWidget(self._overwrite_cb)

        self._preview_config_btn = QPushButton("Preview Config")
        self._preview_config_btn.clicked.connect(self._on_preview_config)
        self._preview_config_btn.setVisible(False)
        layout.addWidget(self._preview_config_btn)

        artifact_row = QHBoxLayout()
        self._open_cmd_file_btn = QPushButton("Command")
        self._open_cmd_file_btn.clicked.connect(lambda: self._on_open_key_artifact("command_invoked.txt"))
        artifact_row.addWidget(self._open_cmd_file_btn)
        self._open_spec_file_btn = QPushButton("Run Spec")
        self._open_spec_file_btn.clicked.connect(lambda: self._on_open_key_artifact("gui_run_spec.json"))
        artifact_row.addWidget(self._open_spec_file_btn)
        self._open_cfg_file_btn = QPushButton("Effective Config")
        self._open_cfg_file_btn.clicked.connect(lambda: self._on_open_key_artifact("config_effective.yaml"))
        artifact_row.addWidget(self._open_cfg_file_btn)
        self._open_manifest_file_btn = QPushButton("Manifest")
        self._open_manifest_file_btn.clicked.connect(lambda: self._on_open_key_artifact("MANIFEST.json"))
        artifact_row.addWidget(self._open_manifest_file_btn)
        self._open_report_file_btn = QPushButton("Run Report")
        self._open_report_file_btn.clicked.connect(lambda: self._on_open_key_artifact("run_report.json"))
        artifact_row.addWidget(self._open_report_file_btn)
        layout.addLayout(artifact_row)

        summary_group = QGroupBox("Effective Run Summary")
        summary_group.setVisible(False)
        summary_layout = QVBoxLayout(summary_group)
        self._effective_summary_label = QLabel()
        self._effective_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._effective_summary_label.setWordWrap(True)
        self._effective_summary_label.setStyleSheet("font-size: 11px;")
        summary_layout.addWidget(self._effective_summary_label)
        layout.addWidget(summary_group)
        return group

    def _on_event(self, data: dict):
        """Compatibility event hook used by tests and optional event integrations."""
        if not isinstance(data, dict):
            return
        self._last_event_stage = str(data.get("stage", "?"))
        self._last_event_type = str(data.get("type", "?"))
        self._last_event_message = str(data.get("message", "")).strip()
        self._render_status_label()

    def _on_elapsed_timer_tick(self) -> None:
        """Refresh elapsed wall-clock display while validate/run is active."""
        if self._ui_state not in (RunnerState.RUNNING, RunnerState.VALIDATING):
            self._elapsed_timer.stop()
            return
        if self._gui_timing_enabled and not self._elapsed_first_tick_logged:
            self._elapsed_first_tick_logged = True
            if self._run_started_monotonic is not None:
                first_tick_delay = max(0.0, time.monotonic() - self._run_started_monotonic)
                self._emit_gui_timing(
                    "END",
                    "elapsed_timer_first_tick",
                    elapsed_sec=first_tick_delay,
                )
        self._render_status_label()

    def _friendly_run_status(self) -> str:
        """Map internal runner state to user-facing run status text."""
        state_map = {
            RunnerState.IDLE: "IDLE",
            RunnerState.VALIDATING: "VALIDATING",
            RunnerState.RUNNING: "RUNNING",
            RunnerState.SUCCESS: "COMPLETE",
            RunnerState.FAILED: "FAILED",
            RunnerState.FAIL_CLOSED: "FAILED",
            RunnerState.CANCELLED: "CANCELLED",
        }
        return state_map.get(self._ui_state, str(self._state_str or "IDLE"))

    @staticmethod
    def _normalize_phase_token(raw: str) -> str:
        token = (raw or "").strip().lower()
        return token.replace("-", "_").replace(" ", "_")

    def _friendly_run_phase(self) -> str:
        """Return user-facing phase label from status tokens and runner state."""
        if self._ui_state == RunnerState.VALIDATING:
            return "Validation"
        if self._ui_state == RunnerState.SUCCESS:
            return "Complete"
        if self._ui_state == RunnerState.CANCELLED:
            return "Cancelled"
        if self._ui_state in (RunnerState.FAILED, RunnerState.FAIL_CLOSED):
            return "Failed"
        if self._ui_state == RunnerState.IDLE:
            return "\u2014"

        phase_token = self._normalize_phase_token(self._last_status_phase)
        status_token = self._normalize_phase_token(self._last_status_state)
        token = phase_token or status_token
        if not token or token in {"?", "\u2014", "unknown"}:
            return "Setup"
        if "valid" in token:
            return "Validation"
        if "setup" in token or "init" in token or "bootstrap" in token:
            return "Setup"
        if "tonic" in token:
            return "Tonic analysis"
        if "phasic" in token:
            return "Phasic analysis"
        if "plot" in token or "render" in token or "figure" in token:
            return "Plotting"
        if (
            "final" in token
            or "manifest" in token
            or "artifact" in token
            or "package" in token
            or "write" in token
        ):
            return "Finalizing"
        if "cancel" in token:
            return "Cancelled"
        if "fail" in token or "error" in token:
            return "Failed"
        if "success" in token or "complete" in token or "done" in token:
            return "Complete"
        return token.replace("_", " ").strip().title()

    def _milestone_progress_pct(self) -> int:
        """Fallback milestone-based progress when no explicit percent is available."""
        if self._ui_state == RunnerState.IDLE:
            return 0
        if self._ui_state == RunnerState.VALIDATING:
            return 20
        if self._ui_state in (
            RunnerState.SUCCESS,
            RunnerState.FAILED,
            RunnerState.FAIL_CLOSED,
            RunnerState.CANCELLED,
        ):
            return 100

        phase = self._friendly_run_phase()
        if phase == "Setup":
            return 10
        if phase == "Validation":
            return 20
        if phase == "Tonic analysis":
            return 35
        if phase == "Phasic analysis":
            return 60
        if phase == "Plotting":
            return 82
        if phase == "Finalizing":
            return 94
        if phase == "Complete":
            return 100
        return 12

    def _effective_progress_pct(self) -> int:
        """Compute stable user-facing progress percentage for the status strip."""
        if self._ui_state == RunnerState.IDLE:
            self._ui_progress_pct = 0
            return 0

        if self._ui_state in (
            RunnerState.SUCCESS,
            RunnerState.FAILED,
            RunnerState.FAIL_CLOSED,
            RunnerState.CANCELLED,
        ):
            self._ui_progress_pct = 100
            return 100

        milestone_pct = self._milestone_progress_pct()
        explicit_pct = None
        if isinstance(self._last_status_pct, (int, float)):
            explicit_pct = max(0, min(100, int(round(float(self._last_status_pct)))))

        candidate = milestone_pct if explicit_pct is None else max(milestone_pct, explicit_pct)
        # Never show 100% while still actively validating/running.
        if self._ui_state in (RunnerState.VALIDATING, RunnerState.RUNNING):
            candidate = min(candidate, 99)
        self._ui_progress_pct = max(self._ui_progress_pct, candidate)
        return self._ui_progress_pct

    def _effective_elapsed_seconds(self) -> float | None:
        """Elapsed wall-clock seconds since validate/run start."""
        if self._run_started_monotonic is not None:
            self._last_elapsed_sec = max(0.0, time.monotonic() - self._run_started_monotonic)
            return self._last_elapsed_sec
        if isinstance(self._last_status_duration_sec, (int, float)):
            return float(self._last_status_duration_sec)
        if self._last_elapsed_sec > 0:
            return self._last_elapsed_sec
        return None

    def _render_status_label(self, is_updating: bool = False):
        """Compose top-strip status and phase/progress labels."""
        status_text = f"Run status: {self._friendly_run_status()}"
        if is_updating and not self._last_status_msg:
            status_text += " | updating..."
        if self._last_status_msg:
            status_text += f" | {self._last_status_msg}"
        self._status_label.setText(status_text)

        phase_text = self._friendly_run_phase()
        if self._last_status_errors:
            phase_text += f" | errors: {len(self._last_status_errors)}"
        self._phase_label.setText(f"Run phase: {phase_text}")

        elapsed_sec = self._effective_elapsed_seconds()
        if elapsed_sec is None:
            self._elapsed_label.setText("Elapsed: \u2014")
        else:
            self._elapsed_label.setText(f"Elapsed: {elapsed_sec:.1f}s")

        pct_i = self._effective_progress_pct()
        if self._ui_state in (RunnerState.VALIDATING, RunnerState.RUNNING):
            pct_i = min(pct_i, 99)
            self._ui_progress_pct = min(self._ui_progress_pct, 99)
        self._progress_bar.setValue(pct_i)
        if self._ui_state == RunnerState.IDLE:
            self._progress_bar.setFormat("idle")
        elif self._ui_state == RunnerState.VALIDATING:
            self._progress_bar.setFormat(f"validating {pct_i}%")
        elif self._ui_state == RunnerState.RUNNING:
            self._progress_bar.setFormat(f"running {pct_i}%")
        else:
            self._progress_bar.setFormat(f"{pct_i}%")

    def _on_status(self, data: dict):
        """Handle parsed status.json updates and refresh top-strip progress."""
        self._last_status_phase = str(data.get("phase", "?"))
        self._last_status_state = str(data.get("status", "?"))
        dur = data.get("duration_sec")
        if isinstance(dur, (int, float)):
            self._last_status_duration_sec = float(dur)
            self._last_status_duration = f"{dur:.1f}s"
        else:
            self._last_status_duration_sec = None
            self._last_status_duration = ""
        self._last_status_errors = data.get("errors", [])
        self._last_status_msg = ""

        explicit_pct = None
        for key in ("progress_pct", "progress_percent", "pct", "percent_complete"):
            value = data.get(key)
            if isinstance(value, (int, float)):
                explicit_pct = value
                break
        self._last_status_pct = explicit_pct
        self._render_status_label(is_updating=False)
        if self._last_status_state.lower() == "cancelled":
            self._saw_cancel_status = True
