"""GUI dialog for curated synthetic demo dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from gui.synthetic_demo_generator import (
    DemoGenerationResult,
    FAST_DEMO_TYPE,
    LONG_DEMO_TYPE,
    copy_fast_quickstart_demo,
    run_long_duration_demo,
)


class _LongDemoWorker(QObject):
    finished = Signal(object)

    def __init__(self, destination: Path, *, overwrite: bool):
        super().__init__()
        self.destination = Path(destination)
        self.overwrite = bool(overwrite)

    def run(self) -> None:
        result = run_long_duration_demo(self.destination, overwrite=self.overwrite)
        self.finished.emit(result)


class GenerateSyntheticDemoDatasetDialog(QDialog):
    """Modal dialog exposing only curated synthetic demo dataset presets."""

    DEMO_CHOICES = [FAST_DEMO_TYPE, LONG_DEMO_TYPE]

    def __init__(
        self,
        *,
        apply_result_callback: Callable[[DemoGenerationResult], None] | None = None,
        open_folder: Callable[[str], None] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Generate Synthetic Demo Dataset")
        self.setModal(True)
        self._apply_result_callback = apply_result_callback
        self._open_folder = open_folder
        self._last_result: DemoGenerationResult | None = None
        self._thread: QThread | None = None
        self._worker: _LongDemoWorker | None = None

        layout = QVBoxLayout(self)

        layout.addWidget(
            QLabel(
                "Choose one curated demo. Advanced synthetic generator parameters "
                "remain CLI-only."
            )
        )

        self._demo_choice_combo = QComboBox()
        self._demo_choice_combo.setObjectName("demo_choice_combo")
        self._demo_choice_combo.addItems(self.DEMO_CHOICES)
        layout.addWidget(QLabel("Demo preset:"))
        layout.addWidget(self._demo_choice_combo)

        output_row = QHBoxLayout()
        self._output_folder_edit = QLineEdit()
        self._output_folder_edit.setObjectName("output_folder_edit")
        self._output_folder_edit.setPlaceholderText("Select output dataset folder")
        output_row.addWidget(self._output_folder_edit, 1)
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse_output_folder)
        output_row.addWidget(self._browse_btn)
        layout.addWidget(QLabel("Output folder:"))
        layout.addLayout(output_row)

        self._status_text = QPlainTextEdit()
        self._status_text.setObjectName("status_text")
        self._status_text.setReadOnly(True)
        self._status_text.setMinimumHeight(120)
        self._status_text.setPlainText("Select a demo preset and output folder.")
        layout.addWidget(self._status_text)

        action_row = QHBoxLayout()
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.clicked.connect(self._on_generate)
        action_row.addWidget(self._generate_btn)
        self._close_btn = QPushButton("Cancel/Close")
        self._close_btn.clicked.connect(self.reject)
        action_row.addWidget(self._close_btn)
        layout.addLayout(action_row)

        post_row = QHBoxLayout()
        self._open_folder_btn = QPushButton("Open Folder")
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        post_row.addWidget(self._open_folder_btn)
        self._set_current_input_btn = QPushButton("Set as Current Input")
        self._set_current_input_btn.setEnabled(False)
        self._set_current_input_btn.clicked.connect(self._on_set_current_input)
        post_row.addWidget(self._set_current_input_btn)
        layout.addLayout(post_row)

    def demo_choices(self) -> list[str]:
        return [self._demo_choice_combo.itemText(i) for i in range(self._demo_choice_combo.count())]

    def _append_status(self, text: str) -> None:
        current = self._status_text.toPlainText().strip()
        self._status_text.setPlainText((current + "\n" + text).strip() if current else text)

    def _browse_output_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Synthetic Demo Output Folder",
            self._output_folder_edit.text().strip(),
        )
        if selected:
            self._output_folder_edit.setText(selected)

    def _confirm_overwrite_if_needed(self, destination: Path) -> bool:
        if not destination.exists() or not any(destination.iterdir()):
            return True
        response = QMessageBox.question(
            self,
            "Replace Existing Folder?",
            "The selected output folder exists and is not empty. Replace only this folder?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return response == QMessageBox.Yes

    def _set_busy(self, busy: bool) -> None:
        self._generate_btn.setEnabled(not busy)
        self._browse_btn.setEnabled(not busy)
        self._demo_choice_combo.setEnabled(not busy)
        self._output_folder_edit.setEnabled(not busy)

    def _on_generate(self) -> None:
        destination_text = self._output_folder_edit.text().strip()
        if not destination_text:
            QMessageBox.warning(self, "Output Folder Required", "Select an output folder first.")
            return
        destination = Path(destination_text)
        overwrite = False
        if destination.exists() and any(destination.iterdir()):
            if not self._confirm_overwrite_if_needed(destination):
                self._append_status("Generation cancelled; existing folder was not changed.")
                return
            overwrite = True

        self._last_result = None
        self._open_folder_btn.setEnabled(False)
        self._set_current_input_btn.setEnabled(False)
        choice = self._demo_choice_combo.currentText()
        self._append_status(f"Starting: {choice}")

        if choice == FAST_DEMO_TYPE:
            result = copy_fast_quickstart_demo(destination, overwrite=overwrite)
            self._handle_result(result)
            return

        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = _LongDemoWorker(destination, overwrite=overwrite)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._handle_result)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(lambda: self._set_busy(False))
        self._thread.start()

    def _handle_result(self, result: DemoGenerationResult) -> None:
        self._last_result = result
        self._append_status(result.message)
        if result.stdout_path is not None:
            self._append_status(f"stdout: {result.stdout_path}")
        if result.stderr_path is not None:
            self._append_status(f"stderr: {result.stderr_path}")
        self._open_folder_btn.setEnabled(bool(result.success))
        self._set_current_input_btn.setEnabled(bool(result.success))
        self._set_busy(False)
        if result.success:
            self._append_status(
                "Recommended GUI settings: format=rwd, mode=both, sessions/hour=2."
            )
        else:
            QMessageBox.warning(self, "Synthetic Demo Generation Failed", result.message)

    def _on_open_folder(self) -> None:
        if self._last_result is None or not self._last_result.success:
            return
        if self._open_folder is not None:
            self._open_folder(str(self._last_result.input_dir))

    def _on_set_current_input(self) -> None:
        if self._last_result is None or not self._last_result.success:
            return
        if self._apply_result_callback is not None:
            self._apply_result_callback(self._last_result)
            self._append_status("Current GUI input settings updated. Validation was not started.")

    def closeEvent(self, event):  # noqa: N802 - Qt override name
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(
                self,
                "Generation Running",
                "Synthetic demo generation is still running. Wait for it to finish before closing.",
            )
            event.ignore()
            return
        super().closeEvent(event)
