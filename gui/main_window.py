"""
MainWindow — Primary GUI window for the Photometry Pipeline Deliverables runner.

Three zones:
  1) Config panel (top) — run parameters, Validate/Run/Cancel buttons
  2) Log panel (middle) — live stdout/stderr from pipeline
  3) Results panel (bottom) — ManifestViewer, populated on successful run
"""

import sys
import os

from PySide6.QtCore import Qt
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


class MainWindow(QMainWindow):
    """Photometry Pipeline Deliverables — GUI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Photometry Pipeline Deliverables")
        self.resize(1100, 850)

        # Pipeline runner
        self._runner = PipelineRunner(self)
        self._runner.log_line.connect(self._append_log)
        self._runner.started.connect(self._on_run_started)
        self._runner.finished.connect(self._on_run_finished)
        self._runner.error.connect(self._on_run_error)

        # Current output directory (set before each run)
        self._current_out_dir = ""

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
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse_dir(self._input_dir, "Select Input Directory"))
        input_row.addWidget(btn)
        form.addRow("Input Directory:", input_row)

        # Output directory
        self._output_dir = QLineEdit()
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_dir)
        btn2 = QPushButton("Browse…")
        btn2.clicked.connect(lambda: self._browse_dir(self._output_dir, "Select Output Package Directory"))
        output_row.addWidget(btn2)
        form.addRow("Output Directory:", output_row)

        # Config YAML
        self._config_path = QLineEdit()
        config_row = QHBoxLayout()
        config_row.addWidget(self._config_path)
        btn3 = QPushButton("Browse…")
        btn3.clicked.connect(self._browse_config)
        config_row.addWidget(btn3)
        form.addRow("Config YAML:", config_row)

        # Format
        self._format_combo = QComboBox()
        self._format_combo.addItems(["auto", "rwd", "npm"])
        form.addRow("Format:", self._format_combo)

        # sessions_per_hour (optional)
        self._sph_edit = QLineEdit()
        self._sph_edit.setPlaceholderText("(optional, integer ≥ 1)")
        self._sph_edit.setMaximumWidth(200)
        form.addRow("Sessions/Hour:", self._sph_edit)

        # SPH warning
        self._sph_warning = QLabel(
            "⚠ Duty-cycled data requires sessions_per_hour unless timestamps exist."
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

        # Overwrite
        self._overwrite_cb = QCheckBox("Overwrite existing output")
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
    # Argv construction
    # ==================================================================

    def _build_argv(self, validate_only: bool = False) -> list:
        """Construct the argv list for the pipeline runner."""
        argv = [
            sys.executable,
            PIPELINE_SCRIPT,
            "--input", self._input_dir.text().strip(),
            "--out", self._output_dir.text().strip(),
            "--config", self._config_path.text().strip(),
            "--format", self._format_combo.currentText(),
        ]

        if self._overwrite_cb.isChecked():
            argv.append("--overwrite")

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
                    return "Sessions/Hour must be ≥ 1."
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

        self._log_view.clear()
        self._append_log("--- Validate Only ---")
        argv = self._build_argv(validate_only=True)
        self._current_out_dir = self._output_dir.text().strip()
        self._is_validate_only = True
        self._runner.start(argv)

    def _on_run(self):
        err = self._validate_gui_inputs()
        if err:
            QMessageBox.warning(self, "Validation Error", err)
            return

        self._log_view.clear()
        self._manifest_viewer.clear()
        self._append_log("--- Starting Pipeline ---")
        argv = self._build_argv(validate_only=False)
        self._current_out_dir = self._output_dir.text().strip()
        self._is_validate_only = False
        self._runner.start(argv)

    def _on_cancel(self):
        self._runner.cancel()

    # ==================================================================
    # Runner signal handlers
    # ==================================================================

    def _on_run_started(self):
        self._update_button_states()

    def _on_run_finished(self, exit_code: int):
        self._update_button_states()

        if exit_code == 0:
            self._append_log(f"--- Finished (exit code {exit_code}) ---")
            if not self._is_validate_only and self._current_out_dir:
                self._manifest_viewer.load_manifest(self._current_out_dir)
        else:
            self._append_log(f"--- Run FAILED (exit code {exit_code}) ---")
            # Try to load partial results if MANIFEST exists
            if self._current_out_dir:
                manifest_path = os.path.join(self._current_out_dir, "MANIFEST.json")
                if os.path.exists(manifest_path):
                    self._append_log("Attempting to load partial results from MANIFEST.json...")
                    self._manifest_viewer.load_manifest(self._current_out_dir)

    def _on_run_error(self, msg: str):
        self._update_button_states()
        self._append_log(f"ERR: {msg}")

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

    def _browse_dir(self, line_edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title, line_edit.text())
        if path:
            line_edit.setText(path)

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Config YAML",
            self._config_path.text(),
            "YAML files (*.yaml *.yml);;All files (*)"
        )
        if path:
            self._config_path.setText(path)
