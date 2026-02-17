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
    QDoubleSpinBox, QPushButton, QPlainTextEdit, QSplitter,
    QFileDialog, QMessageBox, QSizePolicy,
)

from gui.process_runner import PipelineRunner, RunnerState
from gui.events_follower import EventsFollower
from gui.manifest_viewer import ManifestViewer


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_SCRIPT = os.path.join(_REPO_ROOT, "tools", "run_full_pipeline_deliverables.py")

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


class MainWindow(QMainWindow):
    """Photometry Pipeline Deliverables GUI."""

    def __init__(self, parent=None, settings: QSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle("Photometry Pipeline Deliverables")
        self.resize(1100, 850)

        # Settings (injectable for testing)
        self._settings = settings if settings is not None else QSettings()

        # Pipeline runner
        self._runner = PipelineRunner(self)
        self._runner.log_line.connect(self._append_log)
        self._runner.started.connect(self._on_run_started)
        self._runner.finished.connect(self._on_run_finished)
        self._runner.error.connect(self._on_run_error)
        self._runner.state_changed.connect(self._on_state_changed)

        # Events follower (created per run)
        self._events_follower = None

        # Current run directory (set before each run)
        self._current_run_dir = ""
        self._is_validate_only = False
        self._validation_passed = False

        # Status label fields (state + last event, shown together)
        self._state_str = "IDLE"
        self._ui_state = RunnerState.IDLE
        self._last_event_stage = "\u2014"
        self._last_event_type = "\u2014"
        self._last_event_msg = ""

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Vertical)

        # --- Zone A: Config Panel ---
        config_group = self._build_config_panel()
        splitter.addWidget(config_group)

        # --- Status label ---
        self._status_label = QLabel("State: IDLE")
        self._status_label.setStyleSheet(
            "font-weight: bold; padding: 4px; background: #f0f0f0;"
        )
        main_layout.addWidget(self._status_label)

        # --- Zone B: Log Panel ---
        log_group = self._build_log_panel()
        splitter.addWidget(log_group)

        # --- Zone C: Results Panel ---
        results_group = QGroupBox("Results")
        results_lay = QVBoxLayout(results_group)
        self._manifest_viewer = ManifestViewer()
        results_lay.addWidget(self._manifest_viewer)
        splitter.addWidget(results_group)

        splitter.setStretchFactor(0, 0)  # config: natural size
        splitter.setStretchFactor(1, 1)  # log: grows
        splitter.setStretchFactor(2, 2)  # results: grows more

        main_layout.addWidget(splitter)
        self._update_button_states()

        # Restore persisted settings
        self._load_settings_into_widgets()

    # ==================================================================
    # Config Panel
    # ==================================================================

    def _build_config_panel(self) -> QGroupBox:
        group = QGroupBox("Run Configuration")
        outer = QVBoxLayout(group)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Input directory
        self._input_dir = QLineEdit()
        self._input_dir.textChanged.connect(self._on_config_changed)
        input_row = QHBoxLayout()
        input_row.addWidget(self._input_dir)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_dir(self._input_dir, "Select Input Directory"))
        input_row.addWidget(btn)
        form.addRow("Input Directory:", input_row)

        # Output directory (used as --out-base in GUI mode)
        self._output_dir = QLineEdit()
        self._output_dir.textChanged.connect(self._on_config_changed)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_dir)
        btn2 = QPushButton("Browse...")
        btn2.clicked.connect(lambda: self._browse_dir(self._output_dir, "Select Output Base Directory"))
        output_row.addWidget(btn2)
        form.addRow("Output Directory:", output_row)

        # Config YAML
        self._config_path = QLineEdit()
        self._config_path.textChanged.connect(self._on_config_changed)
        config_row = QHBoxLayout()
        config_row.addWidget(self._config_path)
        btn3 = QPushButton("Browse...")
        btn3.clicked.connect(self._browse_config)
        config_row.addWidget(btn3)
        form.addRow("Config YAML:", config_row)

        # Format
        self._format_combo = QComboBox()
        self._format_combo.addItems(["auto", "rwd", "npm"])
        self._format_combo.currentIndexChanged.connect(self._on_config_changed)
        form.addRow("Format:", self._format_combo)

        # sessions_per_hour (optional)
        self._sph_edit = QLineEdit()
        self._sph_edit.setPlaceholderText("(optional, integer >= 1)")
        self._sph_edit.setMaximumWidth(200)
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
        self._duration_edit.textChanged.connect(self._on_config_changed)
        form.addRow("Session Duration (s):", self._duration_edit)

        # smooth_window_s
        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.01, 100.0)
        self._smooth_spin.setValue(1.0)
        self._smooth_spin.setDecimals(2)
        self._smooth_spin.setSingleStep(0.1)
        self._smooth_spin.setMaximumWidth(200)
        self._smooth_spin.valueChanged.connect(self._on_config_changed)
        form.addRow("Smooth Window (s):", self._smooth_spin)

        # Overwrite (legacy CLI only; disabled in GUI which uses --out-base)
        self._overwrite_cb = QCheckBox("Overwrite existing output (legacy CLI only)")
        self._overwrite_cb.setEnabled(False)
        self._overwrite_cb.setToolTip(
            "Not applicable in GUI mode. Each run gets a unique run directory."
        )
        form.addRow("", self._overwrite_cb)

        outer.addLayout(form)

        # Buttons row
        btn_row = QHBoxLayout()
        self._validate_btn = QPushButton("Validate Only")
        self._validate_btn.clicked.connect(self._on_validate)
        btn_row.addWidget(self._validate_btn)

        self._run_btn = QPushButton("Run Pipeline")
        self._run_btn.setStyleSheet("font-weight: bold;")
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)

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

    # ==================================================================
    # Argv construction (GUI mode: --out-base + --run-id)
    # ==================================================================

    def _build_argv(self, validate_only: bool = False) -> list:
        """Construct the argv list for the pipeline runner.

        Uses --out-base mode: Output Directory field is the base directory,
        a unique run_id is generated per invocation.
        """
        out_base = self._output_dir.text().strip()
        run_id = _generate_run_id()
        self._current_run_dir = os.path.join(out_base, run_id)

        argv = [
            sys.executable,
            PIPELINE_SCRIPT,
            "--input", self._input_dir.text().strip(),
            "--out-base", out_base,
            "--run-id", run_id,
            "--config", self._config_path.text().strip(),
            "--format", self._format_combo.currentText(),
            "--events", "auto",
            "--cancel-flag", "auto",
        ]

        # GUI always uses --out-base; --overwrite is not passed.

        sph = self._sph_edit.text().strip()
        if sph:
            argv.extend(["--sessions-per-hour", sph])

        dur = self._duration_edit.text().strip()
        if dur:
            argv.extend(["--session-duration-s", dur])

        argv.extend(["--smooth-window-s", str(self._smooth_spin.value())])

        if validate_only:
            argv.append("--validate-only")

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

        return None

    # ==================================================================
    # Config change handler — resets validation
    # ==================================================================

    def _on_config_changed(self):
        """Any config widget change invalidates prior validation."""
        self._validation_passed = False
        self._update_button_states()

    # ==================================================================
    # Button handlers
    # ==================================================================

    def _on_validate(self):
        err = self._validate_gui_inputs()
        if err:
            QMessageBox.warning(self, "Validation Error", err)
            return

        self._save_widgets_to_settings()
        self._log_view.clear()
        self._append_log("--- Validate Only ---")
        argv = self._build_argv(validate_only=True)
        self._is_validate_only = True
        self._validation_passed = False
        self._reset_event_flags()

        # Pre-create run_dir so events/logs are stable (design rule D)
        os.makedirs(self._current_run_dir, exist_ok=True)

        self._runner.set_run_dir(self._current_run_dir)
        self._start_events_follower()
        self._runner.start(argv, state=RunnerState.VALIDATING)

    def _on_run(self):
        err = self._validate_gui_inputs()
        if err:
            QMessageBox.warning(self, "Validation Error", err)
            return

        self._save_widgets_to_settings()

        # Build argv first so _current_run_dir is computed
        argv = self._build_argv(validate_only=False)
        run_dir = self._current_run_dir

        # Warn if the computed run_dir already exists and is non-empty
        if os.path.isdir(run_dir) and os.listdir(run_dir):
            reply = QMessageBox.warning(
                self,
                "Run Directory Exists",
                f"Run directory already exists and is non-empty:\n{run_dir}\n\n"
                "Each GUI run normally gets a unique directory.\n"
                "Proceed anyway?",
                QMessageBox.Cancel | QMessageBox.Yes,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return

        # Pre-create run_dir (design rule D)
        os.makedirs(run_dir, exist_ok=True)

        self._log_view.clear()
        self._manifest_viewer.clear()
        self._append_log("--- Starting Pipeline ---")
        self._append_log(f"Run directory: {run_dir}")
        self._is_validate_only = False
        self._validation_passed = False
        self._reset_event_flags()

        self._runner.set_run_dir(run_dir)
        self._start_events_follower()
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
            self._append_log(f"ERR: No MANIFEST.json found in {selected}")
        else:
            self._append_log(f"--- Opening results from {selected} ---")

        # ManifestViewer.load_manifest handles missing/invalid file gracefully
        self._manifest_viewer.load_manifest(selected)

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
                self._append_log(
                    f"MANIFEST: status={status}, run_id={run_id}, "
                    f"commands={n_commands}"
                )
            except Exception:
                self._append_log("MANIFEST: could not parse MANIFEST.json")
        else:
            self._append_log("MANIFEST not created")

        _open_folder(run_dir)

    # ==================================================================
    # Events follower integration
    # ==================================================================

    def _start_events_follower(self):
        """Create and start an EventsFollower for the current run_dir."""
        self._stop_events_follower()
        events_path = os.path.join(self._current_run_dir, "events.ndjson")
        self._events_follower = EventsFollower(events_path, poll_ms=300, parent=self)
        self._events_follower.event_received.connect(self._on_event)
        self._events_follower.parse_error.connect(self._on_event_parse_error)
        self._events_follower.start()

    def _stop_events_follower(self):
        """Stop and discard the events follower."""
        if self._events_follower is not None:
            self._events_follower.stop()
            self._events_follower.deleteLater()
            self._events_follower = None

    def _reset_event_flags(self):
        """Reset event-derived fields at the start of each validate/run."""
        self._last_event_stage = "\u2014"
        self._last_event_type = "\u2014"
        self._last_event_msg = ""
        self._render_status_label()

    def _render_status_label(self):
        """Compose status label from state + last event fields."""
        parts = [f"State: {self._state_str}"]
        parts.append(f"Stage: {self._last_event_stage}")
        parts.append(f"Type: {self._last_event_type}")
        if self._last_event_msg:
            parts.append(self._last_event_msg)
        self._status_label.setText(" | ".join(parts))

    def _on_event(self, evt: dict):
        """Handle a parsed event from events.ndjson."""
        self._last_event_stage = evt.get("stage", "?")
        self._last_event_type = evt.get("type", "?")
        self._last_event_msg = evt.get("message", "")
        self._render_status_label()

    def _on_event_parse_error(self, msg: str):
        """Non-fatal warning for malformed event lines."""
        self._append_log(f"WARN(events): {msg}")

    # ==================================================================
    # Runner signal handlers
    # ==================================================================

    def _on_run_started(self):
        self._update_button_states()

    def _on_state_changed(self, state_str: str):
        """Update status label on state transitions."""
        self._state_str = state_str
        try:
            self._ui_state = RunnerState(state_str)
        except ValueError:
            pass  # leave _ui_state unchanged for unknown strings
        self._render_status_label()

    def _on_run_finished(self, exit_code: int):
        # Drain events follower before updating UI
        if self._events_follower is not None:
            self._events_follower.begin_drain()
            # Schedule final UI update after a brief drain window
            QTimer.singleShot(700, lambda: self._finalize_run(exit_code))
        else:
            self._finalize_run(exit_code)

    def _finalize_run(self, exit_code: int):
        """Called after events drain completes.

        Final state is taken from PipelineRunner.state (hardened precedence);
        events are only for progress display.
        """
        self._stop_events_follower()

        final_state = self._runner.state
        # redundant logic removed

        # Update UI-owned state for button gating
        self._ui_state = final_state
        self._state_str = final_state.value
        self._render_status_label()
        self._update_button_states()

        if final_state == RunnerState.SUCCESS:
            self._append_log(f"--- Finished (exit code {exit_code}) ---")
            if self._is_validate_only:
                self._check_validation_result()
            elif self._current_run_dir:
                self._manifest_viewer.load_manifest(self._current_run_dir)
        elif final_state == RunnerState.CANCELLED:
            self._append_log("--- Run CANCELLED ---")
            if self._current_run_dir:
                manifest_path = os.path.join(self._current_run_dir, "MANIFEST.json")
                if os.path.exists(manifest_path):
                    self._append_log("Loading cancelled run manifest...")
                    self._manifest_viewer.load_manifest(self._current_run_dir)
        else:
            self._append_log(f"--- Run FAILED (exit code {exit_code}) ---")
            if self._current_run_dir:
                status_path = os.path.join(self._current_run_dir, "status.json")
                if os.path.isfile(status_path):
                    try:
                        with open(status_path, "r", encoding="utf-8") as fh:
                            s = json.load(fh)
                        
                        phase = s.get("phase")
                        status = s.get("status")

                        if phase == "final" and status in ("error", "cancelled"):
                            errors = s.get("errors", [])
                            if errors:
                                self._append_log("ERRORS from status.json:")
                                for e in errors:
                                    self._append_log(f"  - {e}")
                    except Exception:
                        self._append_log("Could not parse status.json for errors.")

                manifest_path = os.path.join(self._current_run_dir, "MANIFEST.json")
                if os.path.exists(manifest_path):
                    self._append_log("Attempting to load partial results...")
                    self._manifest_viewer.load_manifest(self._current_run_dir)

        # After any full run: require re-validation
        if not self._is_validate_only:
            self._validation_passed = False
            self._update_button_states()

    def _check_validation_result(self):
        """Check status.json for validation success, avoiding log parsing (Requirement 2)."""
        if self._runner.state == RunnerState.SUCCESS:
            self._validation_passed = True
            self._append_log("Validation PASSED (per status.json). Run is now enabled.")
        else:
            self._validation_passed = False
            self._append_log("Validation FAILED (per status.json). Run remains disabled.")
        self._update_button_states()

    def _on_run_error(self, msg: str):
        self._stop_events_follower()
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
        self._settings.setValue("overwrite", self._overwrite_cb.isChecked())
        self._settings.endGroup()
        self._settings.sync()

    # ==================================================================
    # Helpers
    # ==================================================================

    def _append_log(self, text: str):
        self._log_view.appendPlainText(text)

    def _update_button_states(self):
        state = self._ui_state
        running = self._runner.is_running()

        is_done = state in (RunnerState.SUCCESS, RunnerState.FAILED,
                            RunnerState.CANCELLED)
        is_idle_or_done = state == RunnerState.IDLE or is_done

        # Validate: enabled only when idle or done (not running/validating)
        self._validate_btn.setEnabled(is_idle_or_done and not running)

        # Run: enabled only when idle/done, not running, AND validated
        self._run_btn.setEnabled(
            is_idle_or_done and not running and self._validation_passed
        )

        # Cancel: enabled only when RUNNING (not VALIDATING per rule A)
        self._cancel_btn.setEnabled(state == RunnerState.RUNNING and running)

        # Open Results: always enabled when not running
        self._open_results_btn.setEnabled(not running)

        # Open Run Folder: enabled when done and run_dir exists
        has_run_dir = (self._current_run_dir
                       and os.path.isdir(self._current_run_dir))
        self._open_folder_btn.setEnabled(is_done and has_run_dir and not running)

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
