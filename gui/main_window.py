"""
MainWindow, primary GUI window for the Photometry Pipeline Deliverables runner.

Three zones:
  1) Config panel (top) -- run parameters, Validate/Run/Cancel/Open Results buttons
  2) Log panel (middle) -- live stdout/stderr from pipeline
  3) Results panel (bottom) -- ManifestViewer, populated on successful run

State machine:
  IDLE -> VALIDATING -> (SUCCESS -> IDLE with _validation_passed)
  IDLE -> RUNNING -> SUCCESS / FAILED / CANCELLED
  Any DONE state allows re-validate.

Button gating:
  IDLE: Validate YES, Run YES (only if _validation_passed), Cancel NO, Open Folder NO
  VALIDATING: Validate NO, Run NO, Cancel NO, Open Folder NO
  RUNNING: Validate NO, Run NO, Cancel YES, Open Folder NO
  DONE: Validate YES, Run NO (re-validate required), Cancel NO, Open Folder YES
"""

import json
import sys
import os
import secrets
import subprocess as _subprocess
from datetime import datetime

from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QPlainTextEdit, QSplitter, QScrollArea,
    QFileDialog, QMessageBox, QSizePolicy, QListWidget, QListWidgetItem,
)

from gui.process_runner import PipelineRunner, RunnerState
from gui.run_spec import RunSpec, FORMAT_CHOICES
from gui.status_follower import StatusFollower
from gui.log_follower import LogFollower
from gui.run_report_viewer import RunReportViewer
from gui.validate_run_policy import (
    compute_run_signature,
    is_validation_current
)
from photometry_pipeline.config import Config
import dataclasses
from typing import get_args


_SETTINGS_GROUP = "run_config"


def _generate_run_id():
    """Generate a run_id: run_YYYYMMDD_HHMMSS_<8hex>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{secrets.token_hex(4)}"


def _open_folder(path: str) -> None:
    """Cross-platform open a folder in the file manager."""
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
    Parses and validates the isosbestic advanced knobs.
    Returns (dict_of_overrides, None) if valid.
    Returns (None, error_message) if invalid.
    """
    ws_text = window_sec_str.strip()
    try:
        ws = float(ws_text if ws_text else defaults["window_sec"])
        if ws <= 0:
            return None, "Regression Window must be > 0."
    except ValueError:
        return None, "Regression Window must be a number."

    ss_text = step_sec_str.strip()
    try:
        ss = float(ss_text if ss_text else defaults["step_sec"])
        if ss <= 0:
            return None, "Regression Step must be > 0."
    except ValueError:
        return None, "Regression Step must be a number."

    if ss > ws:
        return None, "Regression Step cannot be greater than Regression Window."

    try:
        r_low_val = r_low_str.strip()
        r_high_val = r_high_str.strip()
        r_low = float(r_low_val if r_low_val else defaults["r_low"])
        r_high = float(r_high_val if r_high_val else defaults["r_high"])
        if not (0 <= r_low <= r_high <= 1):
            return None, "R-Low and R-High must be between 0 and 1, and R-Low <= R-High."
    except ValueError:
        return None, "R-Low and R-High must be numbers."

    try:
        g_min_val = g_min_str.strip()
        g_min = float(g_min_val if g_min_val else defaults["g_min"])
        if g_min < 0:
            return None, "G-Min must be >= 0."
    except ValueError:
        return None, "G-Min must be a number."

    if min_valid_windows_val < 1:
        return None, "Min Valid Windows must be >= 1."
    if min_samples_val < 1:
        return None, "Min Samples per Window must be >= 1."

    overrides = {
        "window_sec": ws,
        "step_sec": ss,
        "min_valid_windows": min_valid_windows_val,
        "r_low": r_low,
        "r_high": r_high,
        "g_min": g_min,
        "min_samples_per_window": min_samples_val,
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

_cached_peak_reqs = {}  # type: dict[tuple[str, str], bool]

def _peak_threshold_method_requires_param(method_str: str, param_name: str, val1: float, val2: float) -> bool:
    global _cached_peak_reqs
    cache_key = (method_str, param_name)
    if cache_key in _cached_peak_reqs:
        return _cached_peak_reqs[cache_key]

    try:
        from photometry_pipeline.config import Config
    except Exception:
        # If import fails entirely, we cannot determine requirement. Fail-closed: assume required.
        _cached_peak_reqs[cache_key] = True
        return True

    try:
        allowed = get_allowed_peak_threshold_methods_from_config()
        if method_str not in allowed:
            _cached_peak_reqs[cache_key] = False
            return False

        kwargs1 = {"peak_threshold_method": method_str, param_name: val1}
        kwargs2 = {"peak_threshold_method": method_str, param_name: val2}
        cfg1 = Config(**kwargs1)
        cfg2 = Config(**kwargs2)
        req = (getattr(cfg1, param_name) != getattr(cfg2, param_name))
        _cached_peak_reqs[cache_key] = req
        return req
    except Exception:
        # Fallback approach if construction fails without validation
        try:
            Config(peak_threshold_method=method_str)
            base_succeeds = True
        except Exception:
            base_succeeds = False
            
        if not base_succeeds:
            # Cannot determine. Fail-closed: assume required.
            _cached_peak_reqs[cache_key] = True
            return True
            
        # Base succeeds. Does adding the param fail?
        try:
            Config(peak_threshold_method=method_str, **{param_name: val1})
            param_succeeds = True
        except Exception:
            param_succeeds = False
            
        if param_succeeds:
            # Both succeed. Cannot determine. Fail-closed: assume required. (Conservative)
            _cached_peak_reqs[cache_key] = True
            return True
        else:
            # Base succeeds but adding param explicitly rejected -> deterministic False
            _cached_peak_reqs[cache_key] = False
            return False

def peak_threshold_method_requires_k(method_str: str) -> bool:
    return _peak_threshold_method_requires_param(method_str, "peak_threshold_k", 1.0, 2.0)

def peak_threshold_method_requires_percentile(method_str: str) -> bool:
    return _peak_threshold_method_requires_param(method_str, "peak_threshold_percentile", 10.0, 50.0)

def peak_threshold_method_requires_abs(method_str: str) -> bool:
    return _peak_threshold_method_requires_param(method_str, "peak_threshold_abs", 0.5, 1.0)

def parse_and_validate_event_feature_knobs(
    event_signal_text: str,
    peak_method_text: str,
    peak_k_str: str,
    peak_pct_str: str,
    peak_abs_str: str,
    peak_dist_str: str,
    event_auc_text: str,
    defaults: dict,
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

    try:
        dist_val = peak_dist_str.strip()
        peak_dist = float(dist_val if dist_val else defaults["peak_min_distance_sec"])
        if peak_dist < 0:
            return None, "Peak Min Distance (sec) must be >= 0."
    except ValueError:
        return None, "Peak Min Distance (sec) must be a number."

    overrides = {
        "event_signal": event_sig_method,
        "peak_threshold_method": peak_method,
        "peak_min_distance_sec": peak_dist,
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

class MainWindow(QMainWindow):
    """Photometry Pipeline Deliverables GUI."""

    def __init__(self, parent=None, settings: QSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle("Photometry Pipeline Deliverables")
        self.resize(1100, 850)

        # Settings (injectable for testing)
        self._settings = settings if settings is not None else QSettings()

        from photometry_pipeline.config import Config
        self._default_cfg = Config()

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
        self._last_status_errors = []
        self._last_status_msg = ""
        self._saw_cancel_status = False

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        main_splitter = QSplitter(Qt.Horizontal)

        # --- Zone A: Config Panel (Scrollable Sidebar) ---
        config_group = self._build_config_panel()
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setWidget(config_group)
        config_scroll.setFrameShape(QScrollArea.NoFrame)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        config_scroll.setMinimumWidth(480)
        main_splitter.addWidget(config_scroll)

        # Right side: Content (Log + Results)
        content_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(content_splitter)

        # --- Status label & Preview Badge ---
        status_row = QHBoxLayout()
        self._status_label = QLabel("Runner State: IDLE")
        self._status_label.setStyleSheet(
            "font-weight: bold; padding: 4px; background: #f0f0f0;"
        )
        status_row.addWidget(self._status_label, 1)

        self._preview_badge = QLabel("PREVIEW")
        self._preview_badge.setStyleSheet(
            "font-weight: bold; color: white; background: #d9534f; "
            "padding: 2px 8px; border-radius: 4px;"
        )
        self._preview_badge.hide()
        status_row.addWidget(self._preview_badge)

        main_layout.addLayout(status_row, 0)

        # --- Zone B: Log Panel ---
        log_group = self._build_log_panel()
        content_splitter.addWidget(log_group)

        # --- Zone C: Results Panel ---
        results_group = QGroupBox("Results")
        results_lay = QVBoxLayout(results_group)
        self._report_viewer = RunReportViewer()
        results_lay.addWidget(self._report_viewer)
        content_splitter.addWidget(results_group)

        # Sidebar vs Content stretch
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        # Inner Content stretch (Log vs Results)
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 2)

        main_splitter.setSizes([500, 1000])
        main_layout.addWidget(main_splitter, 1)
        self._update_button_states()

        # Restore persisted settings
        self._load_settings_into_widgets()

    # ==================================================================
    # Config Panel
    # ==================================================================

    def _build_config_panel(self) -> QGroupBox:
        group = QGroupBox("Run Configuration")
        # Relaxed size policy returned to Preferred to allow scroll area to function correctly
        outer = QVBoxLayout(group)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Input directory
        self._input_dir = QLineEdit()
        self._input_dir.setToolTip("The source recording/session folder to analyze.")
        self._input_dir.textChanged.connect(self._on_config_changed)
        input_row = QHBoxLayout()
        input_row.addWidget(self._input_dir)
        btn = QPushButton("Browse...")
        btn.setToolTip("Browse for the input directory.")
        btn.clicked.connect(lambda: self._browse_dir(self._input_dir, "Select Input Directory"))
        input_row.addWidget(btn)
        form.addRow("Input Directory:", input_row)

        # Output base directory (run_dir = out_base / run_id)
        self._output_dir = QLineEdit()
        self._output_dir.setToolTip("Where the run folder and deliverables will be created. Each run generates a unique timestamped subfolder.")
        self._output_dir.textChanged.connect(self._on_config_changed)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_dir)
        btn2 = QPushButton("Browse...")
        btn2.setToolTip("Browse for the output base directory.")
        btn2.clicked.connect(lambda: self._browse_dir(self._output_dir, "Select Output Base Directory"))
        output_row.addWidget(btn2)
        form.addRow("Output Directory:", output_row)

        # Config YAML
        self._config_path = QLineEdit()
        self._config_path.setToolTip("The analysis settings used to build the effective run configuration.")
        self._config_path.textChanged.connect(self._on_config_changed)
        config_row = QHBoxLayout()
        config_row.addWidget(self._config_path)
        btn3 = QPushButton("Browse...")
        btn3.setToolTip("Browse for a configuration YAML file.")
        btn3.clicked.connect(self._browse_config)
        config_row.addWidget(btn3)
        form.addRow("Config YAML:", config_row)

        # Format
        self._format_combo = QComboBox()
        self._format_combo.addItems(list(FORMAT_CHOICES))
        self._format_combo.setToolTip("Tells the pipeline how to interpret the input data layout.")
        self._format_combo.currentIndexChanged.connect(self._on_config_changed)
        form.addRow("Format:", self._format_combo)

        # sessions_per_hour (optional)
        self._sph_edit = QLineEdit()
        self._sph_edit.setPlaceholderText("(optional, integer >= 1)")
        self._sph_edit.setMaximumWidth(200)
        self._sph_edit.setToolTip("Required for duty-cycled recordings when session spacing cannot be inferred from timestamps. Leave blank only if the pipeline can determine the spacing on its own.")
        self._sph_edit.textChanged.connect(self._on_config_changed)
        form.addRow("Sessions/Hour:", self._sph_edit)

        # SPH warning
        self._sph_warning = QLabel(
            "Warning: Duty-cycled data requires sessions_per_hour unless timestamps exist."
        )
        self._sph_warning.setStyleSheet("color: #cc6600; font-size: 11px;")
        form.addRow("", self._sph_warning)

        # session_duration_s (optional)
        self._duration_edit = QLineEdit()
        self._duration_edit.setPlaceholderText("(optional, seconds > 0)")
        self._duration_edit.setMaximumWidth(200)
        self._duration_edit.setToolTip("The expected duration of each session. Mainly matters when timing cannot be inferred reliably from the data.")
        self._duration_edit.textChanged.connect(self._on_config_changed)
        form.addRow("Session Duration (s):", self._duration_edit)

        # smooth_window_s
        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.01, 100.0)
        self._smooth_spin.setValue(1.0)
        self._smooth_spin.setDecimals(2)
        self._smooth_spin.setSingleStep(0.1)
        self._smooth_spin.setMaximumWidth(200)
        self._smooth_spin.setToolTip("Affects smoothing of plotted and output time-series. Higher values result in smoother traces but may mask fast transients.")
        self._smooth_spin.valueChanged.connect(self._on_config_changed)
        form.addRow("Smooth Window (s):", self._smooth_spin)

        # Mode
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["both", "phasic", "tonic"])
        self._mode_combo.setToolTip("Select whether to run tonic analysis, phasic analysis, or both.")
        form.addRow("Mode:", self._mode_combo)
        self._mode_combo.currentIndexChanged.connect(self._on_config_changed)

        # Render modes (phasic day-plot families)
        self._sig_iso_render_mode_combo = QComboBox()
        self._sig_iso_render_mode_combo.addItems(["qc", "full"])
        self._sig_iso_render_mode_combo.setCurrentText("qc")
        self._sig_iso_render_mode_combo.setToolTip(
            "Render mode for sig/iso day plots. 'qc' is the fast default; 'full' is higher-fidelity rendering."
        )
        self._sig_iso_render_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        form.addRow("Sig/Iso Render Mode:", self._sig_iso_render_mode_combo)

        self._dff_render_mode_combo = QComboBox()
        self._dff_render_mode_combo.addItems(["qc", "full"])
        self._dff_render_mode_combo.setCurrentText("qc")
        self._dff_render_mode_combo.setToolTip(
            "Render mode for dFF day plots. 'qc' is the fast default; 'full' is higher-fidelity rendering."
        )
        self._dff_render_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        form.addRow("dFF Render Mode:", self._dff_render_mode_combo)

        self._stacked_render_mode_combo = QComboBox()
        self._stacked_render_mode_combo.addItems(["qc", "full"])
        self._stacked_render_mode_combo.setCurrentText("qc")
        self._stacked_render_mode_combo.setToolTip(
            "Render mode for stacked day plots. 'qc' is the fast default; 'full' is higher-fidelity rendering."
        )
        self._stacked_render_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        form.addRow("Stacked Render Mode:", self._stacked_render_mode_combo)

        # Traces-only (--traces-only CLI flag)
        self._traces_only_cb = QCheckBox("Skip feature extraction (traces and QC only)")
        self._traces_only_cb.setToolTip("Run preprocessing and QC only. Skip feature extraction and downstream deliverable generation.")
        self._traces_only_cb.stateChanged.connect(self._on_config_changed)
        form.addRow("Traces-only:", self._traces_only_cb)

        # Preview first N (--preview-first-n CLI flag)
        self._preview_enabled_cb = QCheckBox("Limit sessions")
        self._preview_enabled_cb.setToolTip("Limit processing to the first N sessions for a quick test run. Useful for checking settings before a full run.")
        self._preview_enabled_cb.stateChanged.connect(self._on_config_changed)
        self._preview_n_spin = QSpinBox()
        self._preview_n_spin.setRange(1, 100000)
        self._preview_n_spin.setValue(5)
        self._preview_n_spin.setMaximumWidth(120)
        self._preview_n_spin.setEnabled(False)
        self._preview_enabled_cb.toggled.connect(self._preview_n_spin.setEnabled)
        preview_row = QHBoxLayout()
        preview_row.addWidget(self._preview_enabled_cb)
        preview_row.addWidget(self._preview_n_spin)
        preview_row.addStretch()
        form.addRow("Preview first N:", preview_row)

        # Recursive (always enabled by runner; informational only)
        self._recursive_cb = QCheckBox("Always enabled by runner")
        self._recursive_cb.setChecked(True)
        self._recursive_cb.setEnabled(False)
        self._recursive_cb.setToolTip(
            "Always enabled by the runner currently "
            "(it always passes --recursive)."
        )
        form.addRow("Recursive:", self._recursive_cb)



        # Overwrite (legacy CLI only; disabled in GUI mode)
        self._overwrite_cb = QCheckBox("Overwrite existing output (legacy CLI only)")
        self._overwrite_cb.setEnabled(False)
        self._overwrite_cb.setToolTip(
            "Not applicable in GUI mode. Each run gets a unique run directory."
        )
        form.addRow("", self._overwrite_cb)

        outer.addLayout(form)

        # --- Advanced: Isosbestic Correction ---
        adv_group = QGroupBox("Advanced: Isosbestic Correction")
        adv_layout = QFormLayout(adv_group)
        adv_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._window_sec_edit = QLineEdit(str(self._default_cfg.window_sec))
        self._window_sec_edit.setToolTip("The duration of the sliding window used for isosbestic regression fitting.")
        adv_layout.addRow("Regression Window (sec):", self._window_sec_edit)

        self._step_sec_edit = QLineEdit(str(self._default_cfg.step_sec))
        self._step_sec_edit.setToolTip("How far the sliding window moves between regression fits.")
        adv_layout.addRow("Regression Step (sec):", self._step_sec_edit)

        self._min_valid_windows_spin = QSpinBox()
        self._min_valid_windows_spin.setRange(1, 1000)
        self._min_valid_windows_spin.setValue(self._default_cfg.min_valid_windows)
        self._min_valid_windows_spin.setToolTip("Minimum number of valid regression windows required to compute a stable fit.")
        adv_layout.addRow("Min Valid Windows:", self._min_valid_windows_spin)

        self._r_low_edit = QLineEdit(str(self._default_cfg.r_low))
        self._r_low_edit.setToolTip("Lower threshold for the correlation coefficient (R) to consider a regression window valid.")
        adv_layout.addRow("R-Low Threshold:", self._r_low_edit)

        self._r_high_edit = QLineEdit(str(self._default_cfg.r_high))
        self._r_high_edit.setToolTip("Upper threshold for the correlation coefficient (R) to consider a regression window valid.")
        adv_layout.addRow("R-High Threshold:", self._r_high_edit)

        self._g_min_edit = QLineEdit(str(self._default_cfg.g_min))
        self._g_min_edit.setToolTip("Minimum green channel intensity required for a window to be included in the regression.")
        adv_layout.addRow("G-Min Threshold:", self._g_min_edit)

        self._min_samples_per_window_spin = QSpinBox()
        self._min_samples_per_window_spin.setRange(1, 100000)
        
        # In our schema, min_samples_per_window defaults to 0 (dynamic).
        # But the strict GUI spec for Step 5 says to enforce >= 1.
        default_min_samples = self._default_cfg.min_samples_per_window
        if default_min_samples < 1:
            default_min_samples = 1
        
        self._min_samples_per_window_spin.setValue(default_min_samples)
        self._min_samples_per_window_spin.setToolTip("Minimum valid samples per window.")
        adv_layout.addRow("Min Samples per Window:", self._min_samples_per_window_spin)

        self._adv_group = adv_group
        outer.addWidget(adv_group)

        # Step 6 baseline and preprocessing knobs apply to all modes, only percentile is method-dependent.
        # --- Advanced: Preprocessing + Baseline ---
        adv_prep_group = QGroupBox("Advanced: Preprocessing + Baseline")
        adv_prep_layout = QFormLayout(adv_prep_group)
        adv_prep_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._lowpass_hz_edit = QLineEdit(str(self._default_cfg.lowpass_hz))
        self._lowpass_hz_edit.setToolTip("Frequency cutoff for the low-pass filter applied to raw traces to remove high-frequency noise.")
        adv_prep_layout.addRow("Lowpass Filter (Hz):", self._lowpass_hz_edit)

        self._baseline_method_combo = QComboBox()
        allowed_methods = get_allowed_baseline_methods_from_config()
        if self._default_cfg.baseline_method not in allowed_methods:
            allowed_methods.append(self._default_cfg.baseline_method)
            
        self._baseline_method_combo.addItems(sorted(allowed_methods))
        
        # Set combo box to the default value
        idx = self._baseline_method_combo.findText(self._default_cfg.baseline_method)
        if idx >= 0:
            self._baseline_method_combo.setCurrentIndex(idx)
            
        self._baseline_method_combo.setToolTip("Controls how the baseline reference is computed before dFF/event calculations. Leave at the default unless you have a specific analysis reason to change it.")
        adv_prep_layout.addRow("Baseline Method:", self._baseline_method_combo)

        self._baseline_percentile_edit = QLineEdit(str(self._default_cfg.baseline_percentile))
        self._baseline_percentile_edit.setToolTip("The percentile used for baseline estimation. Only applies to percentile-based methods.")
        self._baseline_percentile_label = QLabel("Baseline Percentile:")
        adv_prep_layout.addRow(self._baseline_percentile_label, self._baseline_percentile_edit)

        self._f0_min_value_edit = QLineEdit(str(self._default_cfg.f0_min_value))
        self._f0_min_value_edit.setToolTip("Minimum allowed value for the calculated baseline (F0) to prevent division by zero or extremely low values.")
        adv_prep_layout.addRow("F0 Min Value:", self._f0_min_value_edit)

        self._adv_prep_group = adv_prep_group
        outer.addWidget(adv_prep_group)

        # --- Advanced: Events + Features ---
        adv_ev_group = QGroupBox("Advanced: Events + Features")
        adv_ev_layout = QFormLayout(adv_ev_group)
        adv_ev_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._event_signal_combo = QComboBox()
        allowed_sigs = get_allowed_event_signals_from_config()
        if self._default_cfg.event_signal not in allowed_sigs:
            allowed_sigs.append(self._default_cfg.event_signal)
        self._event_signal_combo.addItems(sorted(allowed_sigs))
        idx = self._event_signal_combo.findText(self._default_cfg.event_signal)
        if idx >= 0: self._event_signal_combo.setCurrentIndex(idx)
        self._event_signal_combo.setToolTip("The signal type used for event detection (e.g., dFF or detrended delta_f).")
        adv_ev_layout.addRow("Event Signal:", self._event_signal_combo)

        self._peak_method_combo = QComboBox()
        allowed_peak_methods = get_allowed_peak_threshold_methods_from_config()
        if self._default_cfg.peak_threshold_method not in allowed_peak_methods:
            allowed_peak_methods.append(self._default_cfg.peak_threshold_method)
        self._peak_method_combo.addItems(sorted(allowed_peak_methods))
        idx = self._peak_method_combo.findText(self._default_cfg.peak_threshold_method)
        if idx >= 0: self._peak_method_combo.setCurrentIndex(idx)
        self._peak_method_combo.setToolTip("Algorithm used to determine the threshold for identifying significant peaks/events.")
        adv_ev_layout.addRow("Peak Threshold Method:", self._peak_method_combo)

        self._peak_k_edit = QLineEdit(str(self._default_cfg.peak_threshold_k))
        self._peak_k_edit.setToolTip("Multiplier for Standard Deviation or MAD based thresholding methods.")
        self._peak_k_label = QLabel("Peak Threshold K:")
        adv_ev_layout.addRow(self._peak_k_label, self._peak_k_edit)

        self._peak_pct_edit = QLineEdit(str(self._default_cfg.peak_threshold_percentile))
        self._peak_pct_edit.setToolTip("Percentile used for identifying peaks if using a percentile-based threshold method.")
        self._peak_pct_label = QLabel("Peak Threshold Percentile:")
        adv_ev_layout.addRow(self._peak_pct_label, self._peak_pct_edit)

        self._peak_abs_edit = QLineEdit(str(self._default_cfg.peak_threshold_abs))
        self._peak_abs_edit.setToolTip("Fixed absolute value threshold for identifying peaks.")
        self._peak_abs_label = QLabel("Peak Threshold Absolute:")
        adv_ev_layout.addRow(self._peak_abs_label, self._peak_abs_edit)

        self._peak_dist_edit = QLineEdit(str(self._default_cfg.peak_min_distance_sec))
        self._peak_dist_edit.setToolTip("Minimum time separation required between adjacent peaks to be counted as distinct events.")
        adv_ev_layout.addRow("Peak Min Distance (sec):", self._peak_dist_edit)

        self._event_auc_combo = QComboBox()
        allowed_auc = get_allowed_event_auc_baselines_from_config()
        if self._default_cfg.event_auc_baseline not in allowed_auc:
            allowed_auc.append(self._default_cfg.event_auc_baseline)
        self._event_auc_combo.addItems(sorted(allowed_auc))
        idx = self._event_auc_combo.findText(self._default_cfg.event_auc_baseline)
        if idx >= 0: self._event_auc_combo.setCurrentIndex(idx)
        self._event_auc_combo.setToolTip("Determines the baseline reference used for calculating the Area Under the Curve (AUC) for detected events.")
        adv_ev_layout.addRow("Event AUC Baseline:", self._event_auc_combo)

        self._adv_ev_group = adv_ev_group
        outer.addWidget(adv_ev_group)

        # Wire visibility based on baseline method
        self._baseline_method_combo.currentIndexChanged.connect(self._update_adv_prep_visibility)
        self._update_adv_prep_visibility()

        # Wire visibility based on peak threshold method
        self._peak_method_combo.currentIndexChanged.connect(self._update_adv_ev_visibility)
        self._update_adv_ev_visibility()

        # Wire visibility based on mode
        self._mode_combo.currentIndexChanged.connect(self._update_adv_group_visibility)
        self._update_adv_group_visibility()

        # --- Discovery Section ---
        disc_group = QGroupBox("Session / ROI Discovery")
        disc_layout = QVBoxLayout(disc_group)

        disc_btn_row = QHBoxLayout()
        self._discover_btn = QPushButton("Discover Sessions / ROIs")
        self._discover_btn.setToolTip("Search the input directory for sessions and ROIs based on the selected format.")
        self._discover_btn.clicked.connect(self._on_discover)
        disc_btn_row.addWidget(self._discover_btn)
        disc_btn_row.addStretch()
        disc_layout.addLayout(disc_btn_row)

        self._discovery_summary = QLabel("No discovery run yet.")
        self._discovery_summary.setStyleSheet("color: #666; font-size: 11px;")
        disc_layout.addWidget(self._discovery_summary)

        # Sessions list (read-only, exact discovery order)
        disc_lists = QHBoxLayout()

        sess_col = QVBoxLayout()
        sess_col.addWidget(QLabel("Sessions (discovery order):"))
        self._sessions_list = QListWidget()
        self._sessions_list.setMaximumHeight(120)
        self._sessions_list.setSelectionMode(QListWidget.NoSelection)
        sess_col.addWidget(self._sessions_list)
        disc_lists.addLayout(sess_col)

        # ROI filter mode (Include vs Exclude)
        roi_filter_row = QHBoxLayout()
        roi_filter_row.addWidget(QLabel("ROI Filter Mode:"))
        self._roi_filter_combo = QComboBox()
        self._roi_filter_combo.addItems(["Include selected", "Exclude selected"])
        self._roi_filter_combo.setMaximumWidth(200)
        self._roi_filter_combo.currentIndexChanged.connect(self._on_config_changed)
        roi_filter_row.addWidget(self._roi_filter_combo)
        roi_filter_row.addStretch()
        disc_layout.addLayout(roi_filter_row)

        # ROIs checklist (checkable, exact discovery order)
        roi_col = QVBoxLayout()
        roi_col.addWidget(QLabel("ROIs (check to include/exclude):"))
        self._roi_list = QListWidget()
        self._roi_list.setMaximumHeight(120)
        roi_col.addWidget(self._roi_list)
        roi_btn_row = QHBoxLayout()
        self._roi_select_all_btn = QPushButton("Select all")
        self._roi_select_all_btn.clicked.connect(self._on_roi_select_all)
        roi_btn_row.addWidget(self._roi_select_all_btn)
        self._roi_select_none_btn = QPushButton("Select none")
        self._roi_select_none_btn.clicked.connect(self._on_roi_select_none)
        roi_btn_row.addWidget(self._roi_select_none_btn)
        roi_btn_row.addStretch()
        roi_col.addLayout(roi_btn_row)
        disc_lists.addLayout(roi_col)

        disc_layout.addLayout(disc_lists)

        # Representative session dropdown
        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel("Representative Session:"))
        self._rep_session_combo = QComboBox()
        self._rep_session_combo.addItem("(auto)")
        self._rep_session_combo.setMinimumWidth(200)
        self._rep_session_combo.setToolTip("Select a specific session to use for representative summary plots. Leave at (auto) for default selection.")
        rep_row.addWidget(self._rep_session_combo)
        rep_row.addStretch()
        disc_layout.addLayout(rep_row)

        outer.addWidget(disc_group)

        # Buttons row
        btn_row = QHBoxLayout()
        self._validate_btn = QPushButton("Validate Only")
        self._validate_btn.setToolTip("Check settings and directory structure without running the full analysis.")
        self._validate_btn.clicked.connect(self._on_validate)
        btn_row.addWidget(self._validate_btn)

        self._run_btn = QPushButton("Run Pipeline")
        self._run_btn.setStyleSheet("font-weight: bold;")
        self._run_btn.setToolTip("Start the full analysis pipeline using the current settings.")
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setToolTip("Stop the currently running pipeline.")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

        self._preview_config_btn = QPushButton("Preview Config")
        self._preview_config_btn.setToolTip("Show the exact configuration that will be used for the run, combining the YAML file and GUI overrides.")
        self._preview_config_btn.clicked.connect(self._on_preview_config)
        btn_row.addWidget(self._preview_config_btn)

        self._open_results_btn = QPushButton("Open Results...")
        self._open_results_btn.clicked.connect(self._on_open_results)
        btn_row.addWidget(self._open_results_btn)

        self._open_folder_btn = QPushButton("Open Run Folder")
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        btn_row.addWidget(self._open_folder_btn)

        btn_row.addStretch()
        outer.addLayout(btn_row)

        return group

    # ==================================================================
    # Log Panel
    # ==================================================================

    def _build_log_panel(self) -> QGroupBox:
        group = QGroupBox("Live Log")
        layout = QVBoxLayout(group)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Consolas", 9))
        self._log_view.setMaximumBlockCount(10000)
        layout.addWidget(self._log_view)

        return group

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

    def _build_run_spec(self, validate_only: bool = False) -> RunSpec:
        """Create a RunSpec from current widget values for a real run.

        Computes run_dir from out_base + run_id and sets
        self._current_run_dir.  Used exclusively by _build_argv()
        for validate and run operations.

        Discovery and preview use _build_discovery_spec() instead.
        """
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

        smooth = self._smooth_spin.value()
        self._track_if_changed("smooth_window_s", smooth, 1.0, user_set)

        sig_iso_render_mode_text = self._sig_iso_render_mode_combo.currentText()
        sig_iso_render_mode_val = None if sig_iso_render_mode_text == "qc" else sig_iso_render_mode_text
        self._track_if_changed("sig_iso_render_mode", sig_iso_render_mode_text, "qc", user_set)

        dff_render_mode_text = self._dff_render_mode_combo.currentText()
        dff_render_mode_val = None if dff_render_mode_text == "qc" else dff_render_mode_text
        self._track_if_changed("dff_render_mode", dff_render_mode_text, "qc", user_set)

        stacked_render_mode_text = self._stacked_render_mode_combo.currentText()
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
        is_exclude_mode = (self._roi_filter_combo.currentIndex() == 1)
        if self._discovery_cache is not None:
            all_rois = [r["roi_id"] for r in self._discovery_cache.get("rois", [])]
            checked = []
            for i in range(self._roi_list.count()):
                item = self._roi_list.item(i)
                if item.checkState() == Qt.Checked:
                    checked.append(item.text())
            if is_exclude_mode:
                # Exclude mode: checked = excluded ROIs
                if len(checked) == 0:
                    exclude_roi_ids = None  # exclude nothing
                elif len(checked) == len(all_rois):
                    exclude_roi_ids = []  # all excluded (blocked by GUI)
                    user_set.append("exclude_roi_ids")
                else:
                    exclude_roi_ids = checked
                    user_set.append("exclude_roi_ids")
            else:
                # Include mode: checked = included ROIs
                if len(checked) == len(all_rois):
                    include_roi_ids = None  # all included (default)
                elif len(checked) == 0:
                    include_roi_ids = []  # none included (blocked by GUI)
                    user_set.append("include_roi_ids")
                else:
                    include_roi_ids = checked
                    user_set.append("include_roi_ids")

        # --- Config Overrides ---
        config_overrides = {}
        if is_isosbestic_active(mode_text):
            default_dict = {
                "window_sec": self._default_cfg.window_sec,
                "step_sec": self._default_cfg.step_sec,
                "min_valid_windows": self._default_cfg.min_valid_windows,
                "r_low": self._default_cfg.r_low,
                "r_high": self._default_cfg.r_high,
                "g_min": self._default_cfg.g_min,
                # Enforce dynamic minimum mapping matching GUI constraint default
                "min_samples_per_window": max(1, self._default_cfg.min_samples_per_window),
            }
            overrides, _ = parse_and_validate_isosbestic_knobs(
                self._window_sec_edit.text(),
                self._step_sec_edit.text(),
                self._min_valid_windows_spin.value(),
                self._r_low_edit.text(),
                self._r_high_edit.text(),
                self._g_min_edit.text(),
                self._min_samples_per_window_spin.value(),
                defaults=default_dict,
            )
            if overrides is not None:
                changed_overrides = compute_isosbestic_overrides_user_changed(overrides, default_dict)
                config_overrides.update(changed_overrides)

        # Preprocessing + Baseline overrides
        default_prep_dict = {
            "lowpass_hz": self._default_cfg.lowpass_hz,
            "baseline_method": self._default_cfg.baseline_method,
            "baseline_percentile": self._default_cfg.baseline_percentile,
            "f0_min_value": self._default_cfg.f0_min_value,
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
        default_ev_dict = {
            "event_signal": self._default_cfg.event_signal,
            "peak_threshold_method": self._default_cfg.peak_threshold_method,
            "peak_threshold_k": self._default_cfg.peak_threshold_k,
            "peak_threshold_percentile": self._default_cfg.peak_threshold_percentile,
            "peak_threshold_abs": self._default_cfg.peak_threshold_abs,
            "peak_min_distance_sec": self._default_cfg.peak_min_distance_sec,
            "event_auc_baseline": self._default_cfg.event_auc_baseline,
        }
        ev_overrides, _ = parse_and_validate_event_feature_knobs(
            self._event_signal_combo.currentText(),
            self._peak_method_combo.currentText(),
            self._peak_k_edit.text(),
            self._peak_pct_edit.text(),
            self._peak_abs_edit.text(),
            self._peak_dist_edit.text(),
            self._event_auc_combo.currentText(),
            defaults=default_ev_dict,
        )
        if ev_overrides is not None:
            changed_ev_overrides = compute_overrides_user_changed(ev_overrides, default_ev_dict)
            config_overrides.update(changed_ev_overrides)

        spec = RunSpec(
            input_dir=self._input_dir.text().strip(),
            run_dir=run_dir,
            format=fmt,
            validate_only=validate_only,
            sessions_per_hour=sph_val,
            session_duration_s=dur_val,
            smooth_window_s=smooth,
            sig_iso_render_mode=sig_iso_render_mode_val,
            dff_render_mode=dff_render_mode_val,
            stacked_render_mode=stacked_render_mode_val,
            traces_only=traces_only,
            preview_first_n=preview_first_n,
            representative_session_index=rep_session_idx,
            include_roi_ids=include_roi_ids,
            exclude_roi_ids=exclude_roi_ids,
            config_source_path=self._config_path.text().strip(),
            config_overrides=config_overrides,
            gui_version="1.0.0",
            timestamp_local=datetime.now().isoformat(),
            mode=mode_val,
            user_set_fields=user_set,
        )
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
        spec = self._build_run_spec(validate_only=validate_only)
        spec.overwrite = overwrite
        run_dir = self._current_run_dir
        os.makedirs(run_dir, exist_ok=True)

        # Write derived config and validate it
        config_path = spec.generate_derived_config(run_dir)
        RunSpec.validate_effective_config(config_path)

        # Build argv
        argv = spec.build_runner_argv()

        # Write intent record and command log
        spec.write_gui_run_spec(run_dir)
        spec.write_command_invoked(run_dir, argv)

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

        config = self._config_path.text().strip()
        if not config:
            return "Config YAML path is required."
        if not os.path.isfile(config):
            return f"Config file does not exist: {config}"

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

        # ROI selection: block "process nothing" states
        if self._discovery_cache is not None:
            total_rois = self._roi_list.count()
            checked_count = sum(
                1 for i in range(total_rois)
                if self._roi_list.item(i).checkState() == Qt.Checked
            )
            is_exclude_mode = (self._roi_filter_combo.currentIndex() == 1)
            if not is_exclude_mode and checked_count == 0:
                return ("No ROIs selected. Select at least one ROI "
                        "or click Select all.")
            if is_exclude_mode and checked_count == total_rois:
                return ("All ROIs excluded. Uncheck at least one ROI "
                        "or click Select none.")

        fmt = self._format_combo.currentText()
        if not fmt or fmt not in FORMAT_CHOICES:
            return f"Invalid Format: '{fmt}'. Must be one of {FORMAT_CHOICES}."

        if is_isosbestic_active(self._mode_combo.currentText()):
            default_dict = {
                "window_sec": self._default_cfg.window_sec,
                "step_sec": self._default_cfg.step_sec,
                "min_valid_windows": self._default_cfg.min_valid_windows,
                "r_low": self._default_cfg.r_low,
                "r_high": self._default_cfg.r_high,
                "g_min": self._default_cfg.g_min,
                # Enforce dynamic minimum mapping matching GUI constraint default
                "min_samples_per_window": max(1, self._default_cfg.min_samples_per_window),
            }
            _, err = parse_and_validate_isosbestic_knobs(
                self._window_sec_edit.text(),
                self._step_sec_edit.text(),
                self._min_valid_windows_spin.value(),
                self._r_low_edit.text(),
                self._r_high_edit.text(),
                self._g_min_edit.text(),
                self._min_samples_per_window_spin.value(),
                defaults=default_dict,
            )
            if err:
                return err

        # Preprocessing + Baseline validation
        default_prep_dict = {
            "lowpass_hz": self._default_cfg.lowpass_hz,
            "baseline_method": self._default_cfg.baseline_method,
            "baseline_percentile": self._default_cfg.baseline_percentile,
            "f0_min_value": self._default_cfg.f0_min_value,
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
        default_ev_dict = {
            "event_signal": self._default_cfg.event_signal,
            "peak_threshold_method": self._default_cfg.peak_threshold_method,
            "peak_threshold_k": self._default_cfg.peak_threshold_k,
            "peak_threshold_percentile": self._default_cfg.peak_threshold_percentile,
            "peak_threshold_abs": self._default_cfg.peak_threshold_abs,
            "peak_min_distance_sec": self._default_cfg.peak_min_distance_sec,
            "event_auc_baseline": self._default_cfg.event_auc_baseline,
        }
        _, err = parse_and_validate_event_feature_knobs(
            self._event_signal_combo.currentText(),
            self._peak_method_combo.currentText(),
            self._peak_k_edit.text(),
            self._peak_pct_edit.text(),
            self._peak_abs_edit.text(),
            self._peak_dist_edit.text(),
            self._event_auc_combo.currentText(),
            defaults=default_ev_dict,
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
        fmt = self._format_combo.currentText()
        return RunSpec(
            input_dir=self._input_dir.text().strip(),
            run_dir="",
            format=fmt,
            config_source_path=self._config_path.text().strip(),
            config_overrides={},
        )

    def _on_preview_config(self):
        """Show a read-only dialog with the derived config YAML preview."""
        config_path = self._config_path.text().strip()
        if not config_path or not os.path.isfile(config_path):
            QMessageBox.warning(self, "No Config", "Select a valid Config YAML first.")
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
        """Run discovery via the runner backend and populate the UI."""
        input_dir = self._input_dir.text().strip()
        config_path = self._config_path.text().strip()

        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Discovery Error",
                                "Select a valid input directory first.")
            return
        if not config_path or not os.path.isfile(config_path):
            QMessageBox.warning(self, "Discovery Error",
                                "Select a valid config YAML first.")
            return

        spec = self._build_discovery_spec()
        try:
            result = spec.run_discovery()
        except Exception as e:
            self._discovery_cache = None
            self._discovery_summary.setText("Discovery failed.")
            self._sessions_list.clear()
            self._roi_list.clear()
            self._rep_session_combo.clear()
            self._rep_session_combo.addItem("(auto)")
            self._append_log(f"Discovery error: {e}")
            QMessageBox.critical(self, "Discovery Failed", str(e))
            return

        self._discovery_cache = result
        self._populate_discovery_ui(result)
        self._append_log(
            f"Discovery complete: {result.get('n_total_discovered', 0)} sessions, "
            f"{len(result.get('rois', []))} ROIs, format={result.get('resolved_format', '?')}"
        )

    def _populate_discovery_ui(self, disco: dict):
        """Fill sessions list, ROI checklist, and rep session combo from discovery JSON."""
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
        for roi in rois:
            roi_id = roi.get("roi_id", "?")
            item = QListWidgetItem(roi_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._roi_list.addItem(item)

        # Representative session dropdown
        self._rep_session_combo.clear()
        self._rep_session_combo.addItem("(auto)")
        for sess in sessions:
            self._rep_session_combo.addItem(sess.get("session_id", "?"))

    def _on_roi_select_all(self):
        """Check all ROI items in the checklist."""
        for i in range(self._roi_list.count()):
            self._roi_list.item(i).setCheckState(Qt.Checked)

    def _on_roi_select_none(self):
        """Uncheck all ROI items in the checklist."""
        for i in range(self._roi_list.count()):
            self._roi_list.item(i).setCheckState(Qt.Unchecked)

    def _on_validate(self):
        err = self._validate_gui_inputs()
        if err:
            QMessageBox.warning(self, "Validation Error", err)
            return

        self._save_widgets_to_settings()
        self._report_viewer.clear()
        self._log_view.clear()
        
        # Narrative start (Fix Readability 2: Append BEFORE starting followers/runner)
        self._append_run_log("--- Validate Only ---")
        argv = self._build_argv(validate_only=True)
        self._append_run_log(f"Run directory: {self._current_run_dir}")
        self._append_run_log(f"Config (temp): {os.path.join(self._current_run_dir, 'config_effective.yaml')}")

        self._validation_passed = False
        self._is_validate_only = True
        self._validate_stdout = []

        # Validate->Run reuse tracking (Fix B1)
        self._validated_run_dir = None
        self._validated_gui_run_spec_json_path = None
        self._validated_config_effective_yaml_path = None
        self._validated_run_signature = None

        # run_dir already created by _build_argv -> _build_run_spec

        self._runner.set_run_dir(self._current_run_dir)
        self._start_status_follower()
        self._start_log_follower(self._current_run_dir)
        self._runner.start(argv, state=RunnerState.VALIDATING)

    def _on_run(self):
        err = self._validate_gui_inputs()
        if err:
            QMessageBox.warning(self, "Validation Error", err)
            return

        self._save_widgets_to_settings()

        # Build argv (also generates derived config + gui_run_spec.json)
        argv = self._build_argv(validate_only=False, overwrite=True)
        run_dir = self._current_run_dir
        
        # --- Fix B1v4: Validate->Run Consistency Policy ---
        # Same-directory reuse is abandoned due to handle conflicts on Windows and
        # destructive runner semantics. Instead, we use distinct directories but 
        # ensure the user intent is still validly validated.
        try:
            current_sig = compute_run_signature(run_dir)
            if not is_validation_current(self._validated_run_signature, current_sig):
                QMessageBox.warning(self, "Re-validation Required", 
                                    "The current settings differ from the last successful validation. "
                                    "Please run 'Validate Only' again before proceeding.")
                return
        except Exception as e:
            QMessageBox.warning(self, "Validation State Error", 
                                f"Could not verify consistency with prior validation: {e}")
            return

        # Clear past results
        self._report_viewer.clear()
        self._log_view.clear()
        
        # Narrative start (Fix Readability 2: Append BEFORE starting followers/runner)
        self._append_run_log("--- Starting Pipeline ---")
        self._append_run_log(f"Run directory: {run_dir}")
        self._append_run_log(f"Config: {os.path.join(run_dir, 'config_effective.yaml')}")

        self._is_validate_only = False
        self._validation_passed = False
        self._reset_status_flags()

        self._runner.set_run_dir(run_dir)
        self._start_status_follower()
        self._start_log_follower(run_dir)
        self._runner.start(argv, state=RunnerState.RUNNING)

    def _on_cancel(self):
        self._runner.cancel()

    def _on_open_results(self):
        """Open a previously-completed output directory and load its MANIFEST."""
        selected = QFileDialog.getExistingDirectory(
            self, "Select Output Directory with MANIFEST.json",
            self._output_dir.text().strip(),
        )
        if not selected:
            return

        self._current_run_dir = selected
        self._output_dir.setText(selected)
        self._save_widgets_to_settings()

        manifest_path = os.path.join(selected, "MANIFEST.json")
        if not os.path.isfile(manifest_path):
            self._append_run_log(f"No MANIFEST.json found in {selected}")
        else:
            self._append_run_log(f"--- Opening results from {selected} ---")

        # ManifestViewer.load_manifest handles missing/invalid file gracefully
        self._report_viewer.load_report(selected)

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

    def _reset_status_flags(self):
        """Reset status-derived fields at the start of each validate/run."""
        self._saw_cancel_status = False
        self._last_status_phase = "\u2014"
        self._last_status_state = "\u2014"
        self._last_status_duration = ""
        self._last_status_errors = []
        self._last_status_msg = ""
        self._render_status_label()

    def _render_status_label(self, is_updating: bool = False):
        """Compose status label from Runner state + status.json fields."""
        parts = [f"Runner State: {self._state_str}"]
        parts.append(f"Phase: {self._last_status_phase}")
        parts.append(f"Status: {self._last_status_state}")
        
        if self._last_status_duration:
            parts.append(f"Duration: {self._last_status_duration}")
            
        if self._last_status_errors:
            parts.append(f"({len(self._last_status_errors)} Error(s))")

        if self._last_status_msg:
            parts.append(f"[{self._last_status_msg}]")

        if is_updating and not self._last_status_msg:
            parts.append("[updating...]")

        self._status_label.setText(" | ".join(parts))

    def _on_status(self, data: dict):
        """Handle a successfully parsed status.json dictionary."""
        self._last_status_phase = str(data.get("phase", "?"))
        self._last_status_state = str(data.get("status", "?"))
        
        # Format duration to 1 decimal place if available
        dur = data.get("duration_sec")
        if isinstance(dur, (int, float)):
            self._last_status_duration = f"{dur:.1f}s"
        else:
            self._last_status_duration = ""
            
        self._last_status_errors = data.get("errors", [])
        self._last_status_msg = "" # Clear warnings on valid parse
        
        self._render_status_label(is_updating=False)

        # Detect cancellation via status.json
        if self._last_status_state.lower() == "cancelled":
            self._saw_cancel_status = True

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
        self._did_finalize_run_ui = False
        self._update_button_states()

    def _on_state_changed(self, state_str: str):
        """Update status label and button states on state transitions."""
        self._state_str = state_str
        try:
            self._ui_state = RunnerState(state_str)
            
            # Terminal state transition
            terminal_states = (
                RunnerState.SUCCESS, RunnerState.FAILED, 
                RunnerState.CANCELLED, RunnerState.FAIL_CLOSED
            )
            if self._ui_state in terminal_states:
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
                self._last_status_duration = f"{dur:.1f}s"
            
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
        
        # Sync terminal values from disk before rendering (Fix stale top-status)
        self._refresh_status_from_disk_final()
        
        # Determine final outcome from runner state (authoritative)
        state = self._runner.state
        code = self._runner.final_status_code
        
        if state == RunnerState.SUCCESS:
            self._append_run_log(f"--- Finished (status: {code}) ---")
            if self._is_validate_only:
                self._validation_passed = True
                self._append_run_log("Validation PASSED (per status.json). Run is now enabled.")
        elif state == RunnerState.FAILED:
            self._append_run_log(f"--- Run FAILED (status: {code}) ---")
            if self._runner.final_errors:
                self._append_run_log("ERRORS from status.json:")
                for e in self._runner.final_errors:
                    self._append_run_log(f"  \u2022 {e}")
        elif state == RunnerState.FAIL_CLOSED:
            class_id = self._runner.fail_closed_code or "FAIL_CLOSED"
            self._append_run_log(f"Run failed (FAIL_CLOSED): {class_id}")
        elif state == RunnerState.CANCELLED:
            self._append_run_log("--- Run CANCELLED ---")

        # Step 8 Rendering Hardening:
        # Load report if it exists on disk, regardless of runner state or flag.
        report_on_disk = os.path.join(self._current_run_dir, "run_report.json")
        if os.path.exists(report_on_disk):
            if state != RunnerState.SUCCESS:
                 self._append_run_log(f"Report present, runner state = {state.name}")
            
            if not self._is_validate_only:
                 self._report_viewer.load_report(self._current_run_dir)
                 if state == RunnerState.SUCCESS:
                      self._append_run_log(f"Analysis completed successfully in {self._current_run_dir}")

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

        self._render_status_label()
        # Ensure _is_validate_only is False before updating buttons so we are no longer "validating"
        self._is_validate_only = False
        self._update_button_states()

    def _apply_preview_labeling(self):
        """Source preview state from run_report.json and update window/badge."""
        self.setWindowTitle("Photometry Pipeline Deliverables")
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
                self.setWindowTitle("Photometry Pipeline Deliverables [PREVIEW]")
                self._preview_badge.show()
        except Exception:
            pass

    def _on_run_error(self, msg: str):
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
        self._config_path.setText(self._settings.value("config_path", "", str))

        fmt = self._settings.value("format", "auto", str)
        idx = self._format_combo.findText(fmt)
        if idx >= 0:
            self._format_combo.setCurrentIndex(idx)

        self._sph_edit.setText(self._settings.value("sessions_per_hour", "", str))
        self._duration_edit.setText(self._settings.value("session_duration_s", "", str))

        smooth = self._settings.value("smooth_window_s", 1.0, float)
        self._smooth_spin.setValue(smooth)

        sig_iso_render_mode = self._settings.value("sig_iso_render_mode", "qc", str)
        if self._sig_iso_render_mode_combo.findText(sig_iso_render_mode) >= 0:
            self._sig_iso_render_mode_combo.setCurrentText(sig_iso_render_mode)

        dff_render_mode = self._settings.value("dff_render_mode", "qc", str)
        if self._dff_render_mode_combo.findText(dff_render_mode) >= 0:
            self._dff_render_mode_combo.setCurrentText(dff_render_mode)

        stacked_render_mode = self._settings.value("stacked_render_mode", "qc", str)
        if self._stacked_render_mode_combo.findText(stacked_render_mode) >= 0:
            self._stacked_render_mode_combo.setCurrentText(stacked_render_mode)

        overwrite = self._settings.value("overwrite", False, bool)
        self._overwrite_cb.setChecked(overwrite)
        self._settings.endGroup()

    def _save_widgets_to_settings(self):
        """Persist current widget values to QSettings."""
        self._settings.beginGroup(_SETTINGS_GROUP)
        self._settings.setValue("input_dir", self._input_dir.text().strip())
        self._settings.setValue("output_dir", self._output_dir.text().strip())
        self._settings.setValue("config_path", self._config_path.text().strip())
        self._settings.setValue("format", self._format_combo.currentText())
        self._settings.setValue("sessions_per_hour", self._sph_edit.text().strip())
        self._settings.setValue("session_duration_s", self._duration_edit.text().strip())
        self._settings.setValue("smooth_window_s", self._smooth_spin.value())
        self._settings.setValue("sig_iso_render_mode", self._sig_iso_render_mode_combo.currentText())
        self._settings.setValue("dff_render_mode", self._dff_render_mode_combo.currentText())
        self._settings.setValue("stacked_render_mode", self._stacked_render_mode_combo.currentText())
        self._settings.setValue("overwrite", self._overwrite_cb.isChecked())
        self._settings.endGroup()
        self._settings.sync()

    # ==================================================================
    # Helpers
    # ==================================================================

    def _append_log(self, text: str):
        self._log_view.appendPlainText(text)

    def _append_run_log(self, text: str):
        """Append a message from the wrapper/GUI with a 'RUN: ' prefix."""
        self._append_log(f"RUN: {text}")

    def _update_button_states(self):
        state = self._ui_state
        running = self._runner.is_running()
        
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
