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
import time
from datetime import datetime

from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QPlainTextEdit, QScrollArea,
    QFileDialog, QMessageBox, QSizePolicy, QListWidget, QListWidgetItem, QToolButton,
    QProgressBar,
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

def get_allowed_peak_pre_filters_from_config() -> list[str]:
    return _get_allowed_from_config_field("peak_pre_filter")

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

    overrides = {
        "event_signal": event_sig_method,
        "peak_threshold_method": peak_method,
        "peak_min_distance_sec": peak_dist,
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
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(250)
        self._elapsed_timer.timeout.connect(self._on_elapsed_timer_tick)

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addWidget(self._build_status_strip(), 0)
        main_layout.addWidget(self._build_main_body(), 1)
        self._update_button_states()

        # Restore persisted settings
        self._load_settings_into_widgets()

    # ==================================================================
    # Config Panel
    # ==================================================================

    def _build_status_strip(self) -> QWidget:
        """Top status strip with status/phase labels and progress."""
        strip = QWidget()
        row = QHBoxLayout(strip)
        row.setContentsMargins(0, 0, 0, 0)

        self._status_label = QLabel("Run status: IDLE")
        self._status_label.setStyleSheet("font-weight: bold;")
        row.addWidget(self._status_label, 0)

        self._phase_label = QLabel("Run phase: \u2014")
        row.addWidget(self._phase_label, 0)

        self._elapsed_label = QLabel("Elapsed: \u2014")
        row.addWidget(self._elapsed_label, 0)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        row.addWidget(self._progress_bar, 1)

        self._preview_badge = QLabel("PREVIEW")
        self._preview_badge.setStyleSheet(
            "font-weight: bold; color: white; background: #d9534f; "
            "padding: 2px 8px; border-radius: 4px;"
        )
        self._preview_badge.hide()
        row.addWidget(self._preview_badge, 0)
        return strip

    def _build_main_body(self) -> QWidget:
        """Fixed major panes: upper-left controls, lower-left log, right results."""
        body = QWidget()
        row = QHBoxLayout(body)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        left_pane = self._build_left_pane()
        left_pane.setMinimumWidth(520)
        row.addWidget(left_pane, 0)
        row.addWidget(self._build_results_pane(), 1)
        return body

    def _build_left_pane(self) -> QWidget:
        """Fixed left column with control area above and log pane below."""
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QScrollArea.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        controls_scroll.setWidget(self._build_config_panel())

        log_group = self._build_log_panel()
        log_group.setMinimumHeight(180)

        layout.addWidget(controls_scroll, 3)
        layout.addWidget(log_group, 2)
        return pane

    def _build_results_pane(self) -> QGroupBox:
        """Right pane with the large results viewer."""
        results_group = QGroupBox("Results")
        results_lay = QVBoxLayout(results_group)
        self._report_viewer = RunReportViewer()
        results_lay.addWidget(self._report_viewer)
        return results_group

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
            self._active_config_source_label.setStyleSheet("font-size: 11px; color: #8a6d3b;")
        else:
            self._active_config_source_label.setText(
                f"Active baseline source: lab standard default ({self._lab_default_config_path})"
            )
            self._active_config_source_label.setStyleSheet("font-size: 11px; color: #2d7d2d;")

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
            "peak_pre_filter": getattr(self._default_cfg, "peak_pre_filter", "none"),
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
            peak_pre_filter_text=self._peak_pre_filter_combo.currentText(),
        )
        if ev_overrides is not None:
            changed_ev_overrides = compute_overrides_user_changed(ev_overrides, default_ev_dict)
            config_overrides.update(changed_ev_overrides)

        if self._is_custom_config_enabled():
            user_set.append("config_source_path")

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
            config_source_path=self._active_config_source_path(),
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
            "peak_pre_filter": getattr(self._default_cfg, "peak_pre_filter", "none"),
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
            peak_pre_filter_text=self._peak_pre_filter_combo.currentText(),
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
            config_source_path=self._active_config_source_path(),
            config_overrides={},
        )

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

    def _reset_status_flags(self):
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
        self._did_finalize_run_ui = False
        self._run_started_monotonic = time.monotonic()
        self._last_elapsed_sec = 0.0
        self._elapsed_timer.start()
        if hasattr(self._report_viewer, "_title_label"):
            if self._ui_state == RunnerState.VALIDATING:
                msg = "Validation in progress..."
            else:
                msg = "Run in progress..."
            self._report_viewer._title_label.setText(msg)
            self._report_viewer._title_label.setStyleSheet("color: #666; font-size: 14px;")
        self._update_button_states()

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

        # Step 8 Rendering Hardening:
        # Load report if it exists on disk, regardless of runner state or flag.
        report_on_disk = os.path.join(self._current_run_dir, "run_report.json")
        if os.path.exists(report_on_disk):
            if state != RunnerState.SUCCESS:
                 self._append_run_log(f"Report present, runner state = {state.name}. You can inspect available outputs via Open Run Folder.")
            
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
        # Preview controls are demoted from idle layout, but preview mode may still
        # be active via compatibility state; keep explicit top-strip indication.
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

    def _append_log(self, text: str):
        self._log_view.appendPlainText(text)

    def _append_run_log(self, text: str):
        """Append a message from the wrapper/GUI with a 'RUN: ' prefix."""
        self._append_log(f"RUN: {text}")

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
            f"Baseline Config Source: {self._active_config_source_summary()}",
            f"Preview: {preview_text}",
            f"Plotting Mode: {render_text}",
            f"ROI Filter: {roi_text}",
            f"Representative Session: {rep_text}",
            f"Output Destination: {out_text}",
        ]
        self._effective_summary_label.setText("\n".join(summary_lines))

    def _update_context_sensitive_controls(self) -> None:
        """Enable/disable controls that are irrelevant in current mode/context."""
        mode_text = self._mode_combo.currentText()
        phasic_active = is_isosbestic_active(mode_text)

        # Phasic-only controls: render family selectors + event/features group.
        self._plotting_mode_combo.setEnabled(phasic_active)
        for combo in (
            self._sig_iso_render_mode_combo,
            self._dff_render_mode_combo,
            self._stacked_render_mode_combo,
        ):
            combo.setEnabled(phasic_active)
        self._adv_ev_group.setEnabled(phasic_active)
        if phasic_active:
            self._mode_context_label.setText("Phasic controls are active for this mode.")
            self._mode_context_label.setStyleSheet("font-size: 11px; color: #666;")
        else:
            self._mode_context_label.setText(
                "Phasic-only controls are disabled in tonic mode."
            )
            self._mode_context_label.setStyleSheet("font-size: 11px; color: #8a6d3b;")

        # Discovery-dependent ROI/representative controls.
        discovery_ready = self._discovery_cache is not None and self._roi_list.count() > 0
        self._roi_filter_combo.setEnabled(discovery_ready)
        self._roi_list.setEnabled(discovery_ready)
        self._rep_session_combo.setEnabled(discovery_ready)

        if not discovery_ready:
            self._discovery_controls_hint.setText(
                "ROI choices are unresolved. Click 'Select ROIs...' to populate ROI choices."
            )
            self._discovery_controls_hint.setStyleSheet("color: #8a6d3b; font-size: 11px;")
            self._rep_preview_hint.setText(
                "Representative selection requires discovery session ordering."
            )
            self._rep_preview_hint.setStyleSheet("color: #8a6d3b; font-size: 11px;")
            return

        self._discovery_controls_hint.setText("ROI choices loaded. Checked ROIs will be included.")
        self._discovery_controls_hint.setStyleSheet("color: #2d7d2d; font-size: 11px;")

        if not self._preview_enabled_cb.isChecked():
            self._rep_preview_hint.setText(
                "Representative selection applies to the full discovered session list."
            )
            self._rep_preview_hint.setStyleSheet("color: #666; font-size: 11px;")
            return

        preview_n = self._preview_n_spin.value()
        rep_idx = self._rep_session_combo.currentIndex() - 1
        rep_preview_err = validate_representative_index_preview_compatibility(rep_idx, preview_n)
        if rep_preview_err:
            self._rep_preview_hint.setText(rep_preview_err)
            self._rep_preview_hint.setStyleSheet("color: #a94442; font-size: 11px;")
            return

        if rep_idx >= 0:
            self._rep_preview_hint.setText(
                f"Representative session index {rep_idx} is within preview first N={preview_n}."
            )
        else:
            self._rep_preview_hint.setText(
                f"Preview first N={preview_n}; representative session remains auto."
            )
        self._rep_preview_hint.setStyleSheet("color: #666; font-size: 11px;")

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
        color_map = {
            "ready": "#2d7d2d",
            "info": "#555555",
            "warn": "#8a6d3b",
            "error": "#a94442",
        }
        color = color_map.get(severity, "#555555")
        self._run_reason_label.setStyleSheet(f"color: {color}; font-size: 11px;")
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
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        self._run_config_group = self._build_run_configuration_group()
        self._plotting_group = self._build_plotting_group()
        self._advanced_group = self._build_advanced_group()
        outer.addWidget(self._run_config_group)
        outer.addWidget(self._plotting_group)
        outer.addWidget(self._advanced_group)
        # Demoted controls are kept for behavior compatibility but hidden from idle layout.
        outer.addWidget(self._build_hidden_compatibility_group())
        return panel

    def _build_run_configuration_group(self) -> QGroupBox:
        group = QGroupBox("Run Configuration")
        layout = QVBoxLayout(group)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

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
        self._format_combo.currentIndexChanged.connect(self._on_config_changed)
        form.addRow("Format:", self._format_combo)

        self._sph_edit = QLineEdit()
        self._sph_edit.setPlaceholderText("(optional, integer >= 1)")
        self._sph_edit.setMaximumWidth(200)
        self._sph_edit.textChanged.connect(self._on_config_changed)
        form.addRow("Sessions/Hour:", self._sph_edit)

        self._duration_edit = QLineEdit()
        self._duration_edit.setPlaceholderText("(optional, seconds > 0)")
        self._duration_edit.setMaximumWidth(200)
        self._duration_edit.textChanged.connect(self._on_config_changed)
        form.addRow("Session Duration (s):", self._duration_edit)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["both", "phasic", "tonic"])
        self._mode_combo.currentIndexChanged.connect(self._on_config_changed)
        form.addRow("Mode:", self._mode_combo)

        self._sph_warning = QLabel(
            "Warning: Duty-cycled data requires sessions_per_hour unless timestamps exist."
        )
        self._sph_warning.setStyleSheet("color: #cc6600; font-size: 11px;")
        form.addRow("", self._sph_warning)
        self._run_config_inputs_container = QWidget()
        inputs_layout = QVBoxLayout(self._run_config_inputs_container)
        inputs_layout.setContentsMargins(0, 0, 0, 0)
        inputs_layout.addLayout(form)

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
        self._discovery_controls_hint.setStyleSheet("color: #8a6d3b; font-size: 11px;")
        roi_layout.addWidget(self._discovery_controls_hint)

        self._roi_selection_container = QWidget()
        self._roi_selection_container.setVisible(False)
        roi_selection_layout = QVBoxLayout(self._roi_selection_container)
        roi_selection_layout.setContentsMargins(0, 0, 0, 0)
        roi_selection_layout.addWidget(QLabel("ROIs (checked = included):"))
        self._roi_list = QListWidget()
        self._roi_list.setMaximumHeight(120)
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
        hidden_discovery_layout.addWidget(self._rep_preview_hint)
        roi_layout.addWidget(hidden_discovery_details)

        inputs_layout.addWidget(roi_group)
        layout.addWidget(self._run_config_inputs_container)

        actions = QHBoxLayout()
        self._validate_btn = QPushButton("Validate Only")
        self._validate_btn.clicked.connect(self._on_validate)
        actions.addWidget(self._validate_btn)
        self._run_btn = QPushButton("Run Pipeline")
        self._run_btn.setStyleSheet("font-weight: bold;")
        self._run_btn.clicked.connect(self._on_run)
        actions.addWidget(self._run_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        actions.addWidget(self._cancel_btn)
        self._open_results_btn = QPushButton("Open Results...")
        self._open_results_btn.clicked.connect(self._on_open_results)
        actions.addWidget(self._open_results_btn)
        self._open_folder_btn = QPushButton("Open Run Folder")
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        actions.addWidget(self._open_folder_btn)
        actions.addStretch()
        layout.addLayout(actions)

        self._run_reason_label = QLabel("Run status: Validation required before first run.")
        self._run_reason_label.setWordWrap(True)
        self._run_reason_label.setStyleSheet("color: #8a6d3b; font-size: 11px;")
        layout.addWidget(self._run_reason_label)
        return group

    def _build_plotting_group(self) -> QGroupBox:
        group = QGroupBox("Plotting")
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._plotting_mode_combo = QComboBox()
        self._plotting_mode_combo.addItems(["Standard", "Full"])
        self._plotting_mode_combo.setCurrentText("Standard")
        self._plotting_mode_combo.currentIndexChanged.connect(self._on_plotting_mode_changed)
        self._plotting_mode_combo.currentIndexChanged.connect(self._on_config_changed)
        layout.addRow("Plotting Mode:", self._plotting_mode_combo)

        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.01, 100.0)
        self._smooth_spin.setValue(1.0)
        self._smooth_spin.setDecimals(2)
        self._smooth_spin.setSingleStep(0.1)
        self._smooth_spin.setMaximumWidth(200)
        self._smooth_spin.valueChanged.connect(self._on_config_changed)
        layout.addRow("Smooth Window (s):", self._smooth_spin)

        self._mode_context_label = QLabel("")
        self._mode_context_label.setWordWrap(True)
        self._mode_context_label.setStyleSheet("font-size: 11px; color: #666;")
        layout.addRow("", self._mode_context_label)
        return group

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
        self._window_sec_edit = QLineEdit(str(self._default_cfg.window_sec))
        self._window_sec_edit.textChanged.connect(self._on_config_changed)
        iso_sampling_form.addRow("Regression Window:", self._window_sec_edit)
        self._step_sec_edit = QLineEdit(str(self._default_cfg.step_sec))
        self._step_sec_edit.textChanged.connect(self._on_config_changed)
        iso_sampling_form.addRow("Regression Step:", self._step_sec_edit)
        iso_layout.addWidget(iso_sampling)

        iso_accept = QGroupBox("Window Acceptance")
        iso_accept_form = QFormLayout(iso_accept)
        self._min_valid_windows_spin = QSpinBox()
        self._min_valid_windows_spin.setRange(1, 1000)
        self._min_valid_windows_spin.setValue(self._default_cfg.min_valid_windows)
        self._min_valid_windows_spin.valueChanged.connect(self._on_config_changed)
        iso_accept_form.addRow("Min Valid Windows:", self._min_valid_windows_spin)
        self._min_samples_per_window_spin = QSpinBox()
        self._min_samples_per_window_spin.setRange(1, 100000)
        self._min_samples_per_window_spin.setValue(max(1, self._default_cfg.min_samples_per_window))
        self._min_samples_per_window_spin.valueChanged.connect(self._on_config_changed)
        iso_accept_form.addRow("Min Samples per Window:", self._min_samples_per_window_spin)
        iso_layout.addWidget(iso_accept)

        iso_trust = QGroupBox("Correlation-Based Trust of Slope")
        iso_trust_form = QFormLayout(iso_trust)
        self._r_low_edit = QLineEdit(str(self._default_cfg.r_low))
        self._r_low_edit.textChanged.connect(self._on_config_changed)
        iso_trust_form.addRow("R-Low Threshold:", self._r_low_edit)
        self._r_high_edit = QLineEdit(str(self._default_cfg.r_high))
        self._r_high_edit.textChanged.connect(self._on_config_changed)
        iso_trust_form.addRow("R-High Threshold:", self._r_high_edit)
        self._g_min_edit = QLineEdit(str(self._default_cfg.g_min))
        self._g_min_edit.textChanged.connect(self._on_config_changed)
        iso_trust_form.addRow("G-Min Threshold:", self._g_min_edit)
        iso_layout.addWidget(iso_trust)
        content_layout.addWidget(self._adv_group)

        self._adv_prep_group = QGroupBox("Preprocessing")
        prep_layout = QFormLayout(self._adv_prep_group)
        self._lowpass_hz_edit = QLineEdit(str(self._default_cfg.lowpass_hz))
        self._lowpass_hz_edit.textChanged.connect(self._on_config_changed)
        prep_layout.addRow("Lowpass Filter:", self._lowpass_hz_edit)
        content_layout.addWidget(self._adv_prep_group)

        baseline_group = QGroupBox("Baseline / Normalization")
        baseline_layout = QFormLayout(baseline_group)
        self._baseline_method_combo = QComboBox()
        allowed_methods = get_allowed_baseline_methods_from_config()
        if self._default_cfg.baseline_method not in allowed_methods:
            allowed_methods.append(self._default_cfg.baseline_method)
        self._baseline_method_combo.addItems(sorted(allowed_methods))
        idx = self._baseline_method_combo.findText(self._default_cfg.baseline_method)
        if idx >= 0:
            self._baseline_method_combo.setCurrentIndex(idx)
        self._baseline_method_combo.currentIndexChanged.connect(self._on_config_changed)
        baseline_layout.addRow("Baseline Method:", self._baseline_method_combo)
        self._baseline_percentile_label = QLabel("Baseline Percentile:")
        self._baseline_percentile_edit = QLineEdit(str(self._default_cfg.baseline_percentile))
        self._baseline_percentile_edit.textChanged.connect(self._on_config_changed)
        baseline_layout.addRow(self._baseline_percentile_label, self._baseline_percentile_edit)
        self._f0_min_value_edit = QLineEdit(str(self._default_cfg.f0_min_value))
        self._f0_min_value_edit.textChanged.connect(self._on_config_changed)
        baseline_layout.addRow("F0 Min Value:", self._f0_min_value_edit)
        content_layout.addWidget(baseline_group)

        self._adv_ev_group = QGroupBox("Feature Detection")
        ev_layout = QFormLayout(self._adv_ev_group)
        self._event_signal_combo = QComboBox()
        allowed_sigs = get_allowed_event_signals_from_config()
        if self._default_cfg.event_signal not in allowed_sigs:
            allowed_sigs.append(self._default_cfg.event_signal)
        self._event_signal_combo.addItems(sorted(allowed_sigs))
        idx = self._event_signal_combo.findText(self._default_cfg.event_signal)
        if idx >= 0:
            self._event_signal_combo.setCurrentIndex(idx)
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
        self._peak_method_combo.currentIndexChanged.connect(self._on_config_changed)
        ev_layout.addRow("Peak Threshold Method:", self._peak_method_combo)

        self._peak_k_label = QLabel("Peak Threshold K:")
        self._peak_k_edit = QLineEdit(str(self._default_cfg.peak_threshold_k))
        self._peak_k_edit.textChanged.connect(self._on_config_changed)
        ev_layout.addRow(self._peak_k_label, self._peak_k_edit)
        self._peak_pct_label = QLabel("Peak Threshold Percentile:")
        self._peak_pct_edit = QLineEdit(str(self._default_cfg.peak_threshold_percentile))
        self._peak_pct_edit.textChanged.connect(self._on_config_changed)
        ev_layout.addRow(self._peak_pct_label, self._peak_pct_edit)
        self._peak_abs_label = QLabel("Peak Threshold Absolute:")
        self._peak_abs_edit = QLineEdit(str(self._default_cfg.peak_threshold_abs))
        self._peak_abs_edit.textChanged.connect(self._on_config_changed)
        ev_layout.addRow(self._peak_abs_label, self._peak_abs_edit)
        self._peak_dist_edit = QLineEdit(str(self._default_cfg.peak_min_distance_sec))
        self._peak_dist_edit.textChanged.connect(self._on_config_changed)
        ev_layout.addRow("Peak Min Distance:", self._peak_dist_edit)

        self._peak_pre_filter_combo = QComboBox()
        allowed_pre_filters = get_allowed_peak_pre_filters_from_config()
        default_pre_filter = str(getattr(self._default_cfg, "peak_pre_filter", "none"))
        if default_pre_filter not in allowed_pre_filters:
            allowed_pre_filters.append(default_pre_filter)
        self._peak_pre_filter_combo.addItems(sorted(allowed_pre_filters))
        idx = self._peak_pre_filter_combo.findText(default_pre_filter)
        if idx >= 0:
            self._peak_pre_filter_combo.setCurrentIndex(idx)
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
        self._event_auc_combo.currentIndexChanged.connect(self._on_config_changed)
        ev_layout.addRow("Event AUC Baseline:", self._event_auc_combo)
        content_layout.addWidget(self._adv_ev_group)

        cfg_group = QGroupBox("Config Source (Advanced)")
        cfg_layout = QFormLayout(cfg_group)
        cfg_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._use_custom_config_cb = QCheckBox("Use custom config YAML")
        self._use_custom_config_cb.stateChanged.connect(self._on_config_changed)
        self._use_custom_config_cb.toggled.connect(self._update_config_source_ui)
        cfg_layout.addRow("Baseline Source:", self._use_custom_config_cb)
        self._config_path = QLineEdit()
        self._config_path.setPlaceholderText("(optional) custom baseline config path")
        self._config_path.textChanged.connect(self._on_config_changed)
        self._config_path.textChanged.connect(self._update_config_source_ui)
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(self._config_path)
        self._config_browse_btn = QPushButton("Browse...")
        self._config_browse_btn.clicked.connect(self._browse_config)
        cfg_row.addWidget(self._config_browse_btn)
        cfg_layout.addRow("Custom Config YAML:", cfg_row)
        self._active_config_source_label = QLabel("")
        self._active_config_source_label.setWordWrap(True)
        self._active_config_source_label.setStyleSheet("font-size: 11px; color: #666;")
        cfg_layout.addRow("", self._active_config_source_label)
        content_layout.addWidget(cfg_group)

        content_layout.addStretch()

        self._baseline_method_combo.currentIndexChanged.connect(self._update_adv_prep_visibility)
        self._peak_method_combo.currentIndexChanged.connect(self._update_adv_ev_visibility)
        self._mode_combo.currentIndexChanged.connect(self._update_adv_group_visibility)
        self._update_adv_prep_visibility()
        self._update_adv_ev_visibility()
        self._update_adv_group_visibility()
        self._update_config_source_ui()
        return group

    def _on_advanced_disclosure_toggled(self, expanded: bool) -> None:
        """Toggle advanced section content via disclosure-style header."""
        self._advanced_content.setVisible(expanded)
        self._advanced_disclosure_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

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
