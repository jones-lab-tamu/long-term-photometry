"""
MainWindow, primary GUI window for the Photometry Pipeline Deliverables runner.

Three zones:
  1) Config panel (top) -- run parameters, Validate/Run/Cancel/Open Results buttons
  2) Log panel (middle) -- live stdout/stderr from pipeline
  3) Results panel (bottom) -- ManifestViewer, populated on successful run
"""

import sys
import os
import secrets
from datetime import datetime

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QPlainTextEdit, QSplitter,
    QFileDialog, QMessageBox, QSizePolicy,
)

from gui.process_runner import PipelineRunner
from gui.manifest_viewer import ManifestViewer


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_SCRIPT = os.path.join(_REPO_ROOT, "tools", "run_full_pipeline_deliverables.py")

_SETTINGS_GROUP = "run_config"


def _generate_run_id():
    """Generate a run_id: run_YYYYMMDD_HHMMSS_<8hex>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{secrets.token_hex(4)}"


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

        # Current run directory (set before each run)
        self._current_run_dir = ""
        self._is_validate_only = False

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Vertical)

        # --- Zone A: Config Panel ---
        config_group = self._build_config_panel()
        splitter.addWidget(config_group)

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
        input_row = QHBoxLayout()
        input_row.addWidget(self._input_dir)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_dir(self._input_dir, "Select Input Directory"))
        input_row.addWidget(btn)
        form.addRow("Input Directory:", input_row)

        # Output directory (used as --out-base in GUI mode)
        self._output_dir = QLineEdit()
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_dir)
        btn2 = QPushButton("Browse...")
        btn2.clicked.connect(lambda: self._browse_dir(self._output_dir, "Select Output Base Directory"))
        output_row.addWidget(btn2)
        form.addRow("Output Directory:", output_row)

        # Config YAML
        self._config_path = QLineEdit()
        config_row = QHBoxLayout()
        config_row.addWidget(self._config_path)
        btn3 = QPushButton("Browse...")
        btn3.clicked.connect(self._browse_config)
        config_row.addWidget(btn3)
        form.addRow("Config YAML:", config_row)

        # Format
        self._format_combo = QComboBox()
        self._format_combo.addItems(["auto", "rwd", "npm"])
        form.addRow("Format:", self._format_combo)

        # sessions_per_hour (optional)
        self._sph_edit = QLineEdit()
        self._sph_edit.setPlaceholderText("(optional, integer >= 1)")
        self._sph_edit.setMaximumWidth(200)
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
        form.addRow("Session Duration (s):", self._duration_edit)

        # smooth_window_s
        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.01, 100.0)
        self._smooth_spin.setValue(1.0)
        self._smooth_spin.setDecimals(2)
        self._smooth_spin.setSingleStep(0.1)
        self._smooth_spin.setMaximumWidth(200)
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
        self._runner.set_run_dir(self._current_run_dir)
        self._runner.start(argv)

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

        self._log_view.clear()
        self._manifest_viewer.clear()
        self._append_log("--- Starting Pipeline ---")
        self._append_log(f"Run directory: {run_dir}")
        self._is_validate_only = False
        self._runner.set_run_dir(run_dir)
        self._runner.start(argv)

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

    # ==================================================================
    # Runner signal handlers
    # ==================================================================

    def _on_run_started(self):
        self._update_button_states()

    def _on_run_finished(self, exit_code: int):
        self._update_button_states()

        if exit_code == 0:
            self._append_log(f"--- Finished (exit code {exit_code}) ---")
            if not self._is_validate_only and self._current_run_dir:
                self._manifest_viewer.load_manifest(self._current_run_dir)
        elif exit_code == 130:
            self._append_log("--- Run CANCELLED (exit code 130) ---")
            if self._current_run_dir:
                manifest_path = os.path.join(self._current_run_dir, "MANIFEST.json")
                if os.path.exists(manifest_path):
                    self._append_log("Loading cancelled run manifest...")
                    self._manifest_viewer.load_manifest(self._current_run_dir)
        else:
            self._append_log(f"--- Run FAILED (exit code {exit_code}) ---")
            # Try to load partial results if MANIFEST exists
            if self._current_run_dir:
                manifest_path = os.path.join(self._current_run_dir, "MANIFEST.json")
                if os.path.exists(manifest_path):
                    self._append_log("Attempting to load partial results from MANIFEST.json...")
                    self._manifest_viewer.load_manifest(self._current_run_dir)

    def _on_run_error(self, msg: str):
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
        running = self._runner.is_running()
        self._validate_btn.setEnabled(not running)
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._open_results_btn.setEnabled(not running)

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
