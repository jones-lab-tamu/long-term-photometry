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
    GUIDED_CONTINUOUS_DEMO_BLOCK_SEC,
    GUIDED_CONTINUOUS_DEMO_FOLDER_NAME,
    GUIDED_DEMO_FOLDER_NAME,
    DemoGenerationResult,
    generate_guided_continuous_demo,
    generate_guided_csv_demo,
)


INTERMITTENT_CHOICE = "intermittent"
CONTINUOUS_CHOICE = "continuous"
DEMO_CHOICES = (
    ("Intermittent recording, 48 hours", INTERMITTENT_CHOICE),
    ("Continuous recording, 48 hours", CONTINUOUS_CHOICE),
)
_DEMO_FOLDER_NAMES = {
    INTERMITTENT_CHOICE: GUIDED_DEMO_FOLDER_NAME,
    CONTINUOUS_CHOICE: GUIDED_CONTINUOUS_DEMO_FOLDER_NAME,
}


class _GuidedDemoWorker(QObject):
    finished = Signal(object)
    progress = Signal(int, int)

    def __init__(self, destination_parent: Path, recording_structure: str):
        super().__init__()
        self.destination_parent = Path(destination_parent)
        self.recording_structure = str(recording_structure)

    def run(self) -> None:
        generate = (
            generate_guided_continuous_demo
            if self.recording_structure == CONTINUOUS_CHOICE
            else generate_guided_csv_demo
        )
        result = generate(
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
                "Create one synthetic recording for the ordinary Guided "
                "Workflow. The demo has fixed, friendly settings."
            )
        )

        layout.addWidget(QLabel("Recording structure:"))
        self._structure_combo = QComboBox()
        self._structure_combo.setObjectName("recording_structure_combo")
        for label, value in DEMO_CHOICES:
            self._structure_combo.addItem(label, value)
        self._structure_combo.setCurrentIndex(0)
        self._structure_combo.currentIndexChanged.connect(self._on_structure_changed)
        layout.addWidget(self._structure_combo)

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
        self._status_text.setPlainText(self._destination_hint())
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

    def selected_recording_structure(self) -> str:
        return str(self._structure_combo.currentData())

    def _destination_hint(self) -> str:
        folder = _DEMO_FOLDER_NAMES[self.selected_recording_structure()]
        return f"Select a destination. A folder named {folder} will be created inside it."

    def _on_structure_changed(self, _index: int) -> None:
        self._status_text.setPlainText(self._destination_hint())

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
        self._structure_combo.setEnabled(not busy)

    def _on_generate(self) -> None:
        destination_text = self._output_folder_edit.text().strip()
        if not destination_text:
            QMessageBox.warning(self, "Output Folder Required", "Select an output folder first.")
            return
        structure = self.selected_recording_structure()
        self._last_result = None
        self._open_folder_btn.setEnabled(False)
        self._append_status(
            "Starting synthetic continuous generation."
            if structure == CONTINUOUS_CHOICE
            else "Starting synthetic intermittent generation."
        )

        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = _GuidedDemoWorker(Path(destination_text), structure)
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
        if self.selected_recording_structure() == CONTINUOUS_CHOICE:
            hours_done = int(current) * GUIDED_CONTINUOUS_DEMO_BLOCK_SEC / 3600.0
            hours_total = max(1, int(total)) * GUIDED_CONTINUOUS_DEMO_BLOCK_SEC / 3600.0
            self._status_text.setPlainText(
                f"Generating continuous recording: hour {hours_done:.1f} "
                f"of {hours_total:.0f}..."
            )
            return
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
