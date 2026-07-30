"""GUI dialog for curated synthetic demo dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
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
    generate_guided_csv_demo,
)


class _GuidedDemoWorker(QObject):
    finished = Signal(object)
    progress = Signal(int, int)

    def __init__(self, destination_parent: Path):
        super().__init__()
        self.destination_parent = Path(destination_parent)

    def run(self) -> None:
        result = generate_guided_csv_demo(
            self.destination_parent,
            progress=lambda current, total: self.progress.emit(current, total),
        )
        self.finished.emit(result)


class GenerateSyntheticDemoDatasetDialog(QDialog):
    """Generate the one fixed synthetic recording supported by Guided Mode."""

    def __init__(
        self,
        *,
        open_folder: Callable[[str], None] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Generate Guided Demo Dataset")
        self.setModal(True)
        self._open_folder = open_folder
        self._last_result: DemoGenerationResult | None = None
        self._thread: QThread | None = None
        self._worker: _GuidedDemoWorker | None = None

        layout = QVBoxLayout(self)

        layout.addWidget(
            QLabel(
                "Create one synthetic CSV recording for the ordinary Guided "
                "Workflow. The demo has fixed, reviewer-friendly settings."
            )
        )

        output_row = QHBoxLayout()
        self._output_folder_edit = QLineEdit()
        self._output_folder_edit.setObjectName("output_folder_edit")
        self._output_folder_edit.setPlaceholderText("Select a destination folder")
        output_row.addWidget(self._output_folder_edit, 1)
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._browse_output_folder)
        output_row.addWidget(self._browse_btn)
        layout.addWidget(QLabel("Destination:"))
        layout.addLayout(output_row)

        self._status_text = QPlainTextEdit()
        self._status_text.setObjectName("status_text")
        self._status_text.setReadOnly(True)
        self._status_text.setMinimumHeight(120)
        self._status_text.setPlainText(
            "Select a destination. A folder named "
            "long_term_photometry_guided_demo will be created inside it."
        )
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
        layout.addLayout(post_row)

    def _append_status(self, text: str) -> None:
        current = self._status_text.toPlainText().strip()
        self._status_text.setPlainText((current + "\n" + text).strip() if current else text)

    def _browse_output_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Guided Demo Destination",
            self._output_folder_edit.text().strip(),
        )
        if selected:
            self._output_folder_edit.setText(selected)

    def _set_busy(self, busy: bool) -> None:
        self._generate_btn.setEnabled(not busy)
        self._browse_btn.setEnabled(not busy)
        self._output_folder_edit.setEnabled(not busy)

    def _on_generate(self) -> None:
        destination_text = self._output_folder_edit.text().strip()
        if not destination_text:
            QMessageBox.warning(self, "Output Folder Required", "Select an output folder first.")
            return
        self._last_result = None
        self._open_folder_btn.setEnabled(False)
        self._append_status("Starting synthetic Guided CSV generation.")

        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = _GuidedDemoWorker(Path(destination_text))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._handle_result)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(lambda: self._set_busy(False))
        self._thread.start()

    def _on_progress(self, current: int, total: int) -> None:
        self._status_text.setPlainText(
            f"Generating session {int(current)} of {int(total)}..."
        )

    def _handle_result(self, result: DemoGenerationResult) -> None:
        self._last_result = result
        self._append_status(result.message)
        if result.stdout_path is not None:
            self._append_status(f"stdout: {result.stdout_path}")
        if result.stderr_path is not None:
            self._append_status(f"stderr: {result.stderr_path}")
        self._open_folder_btn.setEnabled(bool(result.success))
        self._set_busy(False)
        if not result.success:
            QMessageBox.warning(self, "Synthetic Demo Generation Failed", result.message)

    def _on_open_folder(self) -> None:
        if self._last_result is None or not self._last_result.success:
            return
        if self._open_folder is not None:
            self._open_folder(str(self._last_result.input_dir))

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
