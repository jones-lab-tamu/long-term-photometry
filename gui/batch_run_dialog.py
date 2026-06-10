"""
Minimal GUI dialog for sequential batch execution.

The dialog is intentionally a thin GUI wrapper around BatchRunner.  It does not
create a separate analysis path; each dataset still runs through the normal
RunSpec/build_runner_argv wrapper flow.
"""

from __future__ import annotations

import copy
import os
from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from gui.batch_runner import BatchCancelToken, BatchRunner
from gui.batch_spec import (
    BatchDatasetRow,
    BatchRunSpec,
    compute_batch_summary_counts,
    discover_batch_datasets,
    make_batch_run_spec,
    plan_batch_outputs,
    utc_now_iso,
)
from gui.run_report_parser import is_successful_completed_run_dir, resolve_region_deliverables
from gui.run_spec import RunSpec


BuildBaseRunSpecCallback = Callable[..., RunSpec]
OpenCompletedRunCallback = Callable[[str], bool]
OpenFolderCallback = Callable[[str], None]


class BatchRunWorker(QObject):
    """QObject worker that runs BatchRunner off the UI thread."""

    row_updated = Signal(object)
    batch_updated = Signal(object)
    message = Signal(str)
    finished = Signal(object)

    def __init__(self, batch_spec: BatchRunSpec, validate_only: bool, parent=None):
        super().__init__(parent)
        self.batch_spec = batch_spec
        self.validate_only = bool(validate_only)
        self.runner: BatchRunner | None = None
        self.cancel_token = BatchCancelToken()

    def run(self) -> None:
        final_spec = None

        def _store_final_spec(spec: BatchRunSpec) -> None:
            nonlocal final_spec
            final_spec = spec

        try:
            self.runner = BatchRunner(
                self.batch_spec,
                cancel_requested=self.cancel_token.is_cancel_requested,
                on_row_update=lambda row: self.row_updated.emit(row),
                on_batch_update=lambda spec: self.batch_updated.emit(spec),
                on_message=lambda msg: self.message.emit(msg),
                on_finished=_store_final_spec,
            )
            result = self.runner.run(validate_only=self.validate_only)
            final_spec = final_spec or result
        except Exception as exc:
            self.message.emit(f"Batch worker failed: {exc}")
            final_spec = self.batch_spec
        finally:
            self.finished.emit(final_spec or self.batch_spec)

    def request_cancel(self) -> None:
        # Thread-safe: this may be called from the GUI thread while the worker
        # thread is synchronously inside BatchRunner.run().
        self.cancel_token.request_cancel()


class BatchRunDialog(QDialog):
    """Minimal user-facing batch workflow for independent dataset folders."""

    cancel_requested = Signal()

    HEADERS = [
        "dataset_id",
        "dataset_name",
        "input_path",
        "output_path",
        "status",
        "message",
        "elapsed_sec",
    ]

    def __init__(
        self,
        *,
        build_base_run_spec: BuildBaseRunSpecCallback,
        open_completed_run: OpenCompletedRunCallback,
        open_folder: OpenFolderCallback | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Run")
        self.resize(1100, 640)
        self._build_base_run_spec = build_base_run_spec
        self._open_completed_run = open_completed_run
        self._open_folder = open_folder
        self._rows: list[BatchDatasetRow] = []
        self._batch_spec: BatchRunSpec | None = None
        self._worker: BatchRunWorker | None = None
        self._thread: QThread | None = None
        self._batch_active = False
        self._last_batch_output_root = ""
        self._planning_key: tuple[str, str, bool] | None = None
        self._suppress_planning_change = False
        self._build_ui()
        self._set_batch_active(False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._scope_label = QLabel(
            "Batch mode applies the same current analysis settings to each dataset folder.\n"
            "Each immediate subfolder of the batch input root is treated as one independent dataset.\n"
            "Each dataset produces a normal completed run.\n"
            "Batch mode does not perform group statistics, averaging, or multi-recording visualization. "
            "Use exported tables for downstream statistics and plotting."
        )
        self._scope_label.setWordWrap(True)
        layout.addWidget(self._scope_label)

        paths_group = QGroupBox("Batch Folders")
        paths_layout = QGridLayout(paths_group)
        self._input_root_edit = QLineEdit()
        self._input_root_edit.textChanged.connect(self._on_planning_inputs_changed)
        self._input_root_browse_btn = QPushButton("Browse...")
        self._input_root_browse_btn.clicked.connect(self._browse_input_root)
        self._output_root_edit = QLineEdit()
        self._output_root_edit.textChanged.connect(self._on_planning_inputs_changed)
        self._output_root_browse_btn = QPushButton("Browse...")
        self._output_root_browse_btn.clicked.connect(self._browse_output_root)
        paths_layout.addWidget(QLabel("Batch Input Root:"), 0, 0)
        paths_layout.addWidget(self._input_root_edit, 0, 1)
        paths_layout.addWidget(self._input_root_browse_btn, 0, 2)
        paths_layout.addWidget(QLabel("Batch Output Root:"), 1, 0)
        paths_layout.addWidget(self._output_root_edit, 1, 1)
        paths_layout.addWidget(self._output_root_browse_btn, 1, 2)
        layout.addWidget(paths_group)

        options_row = QHBoxLayout()
        self._overwrite_cb = QCheckBox("Overwrite existing outputs")
        self._overwrite_cb.stateChanged.connect(self._on_planning_inputs_changed)
        self._stop_on_failure_cb = QCheckBox("Stop on first failure")
        self._detect_btn = QPushButton("Detect Datasets")
        self._detect_btn.clicked.connect(self._on_detect_datasets)
        options_row.addWidget(self._overwrite_cb)
        options_row.addWidget(self._stop_on_failure_cb)
        options_row.addStretch(1)
        options_row.addWidget(self._detect_btn)
        layout.addLayout(options_row)

        self._message_label = QLabel("Select batch folders, then click Detect Datasets.")
        self._message_label.setWordWrap(True)
        layout.addWidget(self._message_label)

        self._table = QTableWidget(0, len(self.HEADERS))
        self._table.setHorizontalHeaderLabels(self.HEADERS)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.itemSelectionChanged.connect(self._refresh_open_selected_state)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self._table, stretch=1)

        self._summary_label = QLabel("Batch summary: total=0")
        layout.addWidget(self._summary_label)

        actions = QHBoxLayout()
        self._validate_batch_btn = QPushButton("Validate Batch")
        self._validate_batch_btn.clicked.connect(lambda: self._start_batch(validate_only=True))
        self._run_batch_btn = QPushButton("Run Batch")
        self._run_batch_btn.clicked.connect(lambda: self._start_batch(validate_only=False))
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._open_selected_btn = QPushButton("Open Selected Run")
        self._open_selected_btn.clicked.connect(self._on_open_selected_run)
        self._open_output_folder_btn = QPushButton("Open Batch Output Folder")
        self._open_output_folder_btn.clicked.connect(self._on_open_batch_output_folder)
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.close)
        for button in (
            self._validate_batch_btn,
            self._run_batch_btn,
            self._cancel_btn,
            self._open_selected_btn,
            self._open_output_folder_btn,
            self._close_btn,
        ):
            actions.addWidget(button)
        layout.addLayout(actions)

    def _browse_input_root(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Batch Input Root", self._input_root_edit.text()
        )
        if path:
            self._input_root_edit.setText(path)

    def _browse_output_root(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Batch Output Root", self._output_root_edit.text()
        )
        if path:
            self._output_root_edit.setText(path)

    def _on_detect_datasets(self) -> None:
        input_root = self._input_root_edit.text().strip()
        output_root = self._output_root_edit.text().strip()
        if not input_root:
            self._show_message("Batch Input Root is required.")
            return
        if not output_root:
            self._show_message("Batch Output Root is required.")
            return
        try:
            discovery = discover_batch_datasets(input_root)
            rows = plan_batch_outputs(
                discovery,
                output_root,
                overwrite=self._overwrite_cb.isChecked(),
            )
        except Exception as exc:
            self._show_message(f"Dataset discovery failed: {exc}")
            return

        self._suppress_planning_change = True
        try:
            self._rows = rows
            self._planning_key = self._current_planning_key()
            self._last_batch_output_root = os.path.abspath(output_root)
            self._populate_table()
        finally:
            self._suppress_planning_change = False
        parts = [f"Detected {len(rows)} dataset folder(s)."]
        if discovery.ignored_root_files:
            parts.append(f"Ignored {len(discovery.ignored_root_files)} root-level file(s).")
        if discovery.ignored_hidden_dirs:
            parts.append(f"Ignored {len(discovery.ignored_hidden_dirs)} hidden folder(s).")
        if not rows:
            parts.append("No immediate subfolder datasets were found.")
        self._show_message(" ".join(parts))
        self._refresh_summary()
        self._update_buttons()

    def _current_planning_key(self) -> tuple[str, str, bool]:
        return (
            os.path.abspath(self._input_root_edit.text().strip()),
            os.path.abspath(self._output_root_edit.text().strip()),
            bool(self._overwrite_cb.isChecked()),
        )

    def _on_planning_inputs_changed(self, *_args) -> None:
        if self._suppress_planning_change or self._batch_active:
            return
        if not self._rows:
            self._planning_key = None
            return
        if self._planning_key == self._current_planning_key():
            return
        self._rows = []
        self._batch_spec = None
        self._planning_key = None
        self._populate_table()
        self._refresh_summary()
        self._show_message("Batch folders or overwrite setting changed. Click Detect Datasets again.")
        self._update_buttons()

    def _start_batch(self, *, validate_only: bool) -> None:
        if self._batch_active:
            self._show_message("A batch is already running.")
            return
        if not self._rows:
            self._show_message("Detect datasets before starting a batch.")
            return
        try:
            batch_spec = self.build_batch_run_spec(validate_only=validate_only)
        except Exception as exc:
            QMessageBox.warning(self, "Batch Run Error", str(exc))
            return

        self._batch_spec = batch_spec
        self._thread = QThread(self)
        self._worker = BatchRunWorker(batch_spec, validate_only)
        self._worker.moveToThread(self._thread)
        self.cancel_requested.connect(self._worker.request_cancel, Qt.QueuedConnection)
        self._thread.started.connect(self._worker.run)
        self._worker.row_updated.connect(self._on_worker_row_updated)
        self._worker.batch_updated.connect(self._on_worker_batch_updated)
        self._worker.message.connect(self._show_message)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._set_batch_active(True)
        self._show_message("Batch started.")
        self._thread.start()

    def build_batch_run_spec(self, *, validate_only: bool) -> BatchRunSpec:
        input_root = self._input_root_edit.text().strip()
        output_root = self._output_root_edit.text().strip()
        if not input_root:
            raise ValueError("Batch Input Root is required.")
        if not output_root:
            raise ValueError("Batch Output Root is required.")
        if not self._rows:
            raise ValueError("No datasets are planned. Click Detect Datasets first.")
        if self._planning_key != self._current_planning_key():
            raise ValueError(
                "Batch folders or overwrite setting changed. Click Detect Datasets again."
            )

        input_placeholder = os.path.join(
            os.path.abspath(output_root), "_batch_placeholders", "input"
        )
        output_placeholder = os.path.join(
            os.path.abspath(output_root), "_batch_placeholders", "output"
        )
        base_spec = self._build_base_run_spec(
            validate_only,
            input_placeholder=input_placeholder,
            output_placeholder=output_placeholder,
        )
        shared_settings = self._shared_settings_from_run_spec(base_spec)
        batch_id = "batch_" + utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
        return make_batch_run_spec(
            batch_id=batch_id,
            created_at=utc_now_iso(),
            batch_input_root=input_root,
            batch_output_root=output_root,
            base_run_spec=base_spec.to_dict(),
            shared_settings=shared_settings,
            datasets=copy.deepcopy(self._rows),
            overwrite=self._overwrite_cb.isChecked(),
            stop_on_failure=self._stop_on_failure_cb.isChecked(),
        )

    def _shared_settings_from_run_spec(self, spec: RunSpec) -> dict:
        return {
            "format": spec.format,
            "mode": spec.mode or "both",
            "acquisition_mode": spec.acquisition_mode,
            "continuous_window_sec": spec.continuous_window_sec,
            "continuous_step_sec": spec.continuous_step_sec,
            "allow_partial_final_window": spec.allow_partial_final_window,
            "run_profile": spec.run_profile,
            "config_source_path": spec.config_source_path,
            "config_overrides": copy.deepcopy(spec.config_overrides),
            "data_contract_overrides": copy.deepcopy(spec.data_contract_overrides),
            "include_roi_ids": copy.deepcopy(spec.include_roi_ids),
            "exclude_roi_ids": copy.deepcopy(spec.exclude_roi_ids),
            "sig_iso_render_mode": spec.sig_iso_render_mode,
            "dff_render_mode": spec.dff_render_mode,
            "stacked_render_mode": spec.stacked_render_mode,
            "traces_only": spec.traces_only,
            "preview_first_n": spec.preview_first_n,
        }

    def _on_cancel(self) -> None:
        if self._batch_active and self._worker is not None:
            self._worker.request_cancel()
            self.cancel_requested.emit()
            self._show_message("Batch cancellation requested.")
            self._cancel_btn.setEnabled(False)
        else:
            self.close()

    def _on_worker_row_updated(self, row: BatchDatasetRow) -> None:
        for idx, existing in enumerate(self._rows):
            if existing.dataset_id == row.dataset_id:
                self._rows[idx] = row
                break
        else:
            self._rows.append(row)
        self._populate_table()
        self._refresh_summary()

    def _on_worker_batch_updated(self, batch_spec: BatchRunSpec) -> None:
        self._batch_spec = batch_spec
        self._rows = copy.deepcopy(batch_spec.datasets)
        self._populate_table()
        self._refresh_summary()

    def _on_worker_finished(self, batch_spec: BatchRunSpec) -> None:
        self._batch_spec = batch_spec
        self._rows = copy.deepcopy(batch_spec.datasets)
        self._last_batch_output_root = batch_spec.batch_output_root
        self._populate_table()
        self._refresh_summary()
        self._show_message("Batch finished.")
        self._set_batch_active(False)

    def _on_thread_finished(self) -> None:
        if self._worker is not None:
            try:
                self.cancel_requested.disconnect(self._worker.request_cancel)
            except Exception:
                pass
        self._thread = None
        self._worker = None
        self._set_batch_active(False)

    def closeEvent(self, event) -> None:
        thread_running = bool(self._thread is not None and self._thread.isRunning())
        if self._batch_active or thread_running:
            event.ignore()
            self._show_message("Batch is running. Cancel the batch before closing this window.")
            return
        super().closeEvent(event)

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._rows))
        for row_idx, row in enumerate(self._rows):
            values = [
                row.dataset_id,
                row.dataset_name,
                row.input_path,
                row.output_path,
                row.status,
                row.message,
                "" if row.elapsed_sec is None else f"{row.elapsed_sec:.3f}",
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                self._table.setItem(row_idx, col_idx, item)
        self._refresh_open_selected_state()

    def _selected_row(self) -> BatchDatasetRow | None:
        selected = self._table.selectionModel().selectedRows()
        if not selected:
            return None
        row_index = selected[0].row()
        if row_index < 0 or row_index >= len(self._rows):
            return None
        return self._rows[row_index]

    def _on_open_selected_run(self) -> None:
        row = self._selected_row()
        if row is None:
            self._show_message("Select a successful dataset row first.")
            return
        ok, reason = self._is_row_openable(row)
        if not ok:
            self._show_message(f"Selected run is not openable: {reason}")
            return
        if not self._open_completed_run(row.output_path):
            self._show_message("Selected completed run could not be loaded.")

    def _is_row_openable(self, row: BatchDatasetRow) -> tuple[bool, str]:
        if row.status != "success":
            return False, "row status is not success"
        ok, evidence = is_successful_completed_run_dir(row.output_path)
        if not ok:
            return False, evidence
        if not resolve_region_deliverables(row.output_path):
            return False, "no region deliverables were found"
        return True, evidence

    def _refresh_open_selected_state(self) -> None:
        row = self._selected_row()
        ok = False
        if row is not None:
            ok, _reason = self._is_row_openable(row)
        self._open_selected_btn.setEnabled(bool(ok and not self._batch_active))

    def _on_open_batch_output_folder(self) -> None:
        root = self._output_root_edit.text().strip() or self._last_batch_output_root
        if not root:
            self._show_message("Batch Output Root is not set.")
            return
        if self._open_folder is None:
            self._show_message(f"Batch output folder: {root}")
            return
        self._open_folder(root)

    def _refresh_summary(self) -> None:
        counts = compute_batch_summary_counts(self._rows)
        self._summary_label.setText(
            "Batch summary: "
            f"total={counts['total']} "
            f"success={counts['success']} "
            f"failed={counts['failed']} "
            f"skipped={counts['skipped']} "
            f"cancelled={counts['cancelled']} "
            f"pending={counts['pending']} "
            f"running={counts['running']} "
            f"validating={counts['validating']}"
        )

    def _show_message(self, message: str) -> None:
        self._message_label.setText(str(message))

    def _set_batch_active(self, active: bool) -> None:
        self._batch_active = bool(active)
        self._update_buttons()

    def _update_buttons(self) -> None:
        active = self._batch_active
        has_rows = bool(self._rows)
        self._input_root_edit.setEnabled(not active)
        self._input_root_browse_btn.setEnabled(not active)
        self._output_root_edit.setEnabled(not active)
        self._output_root_browse_btn.setEnabled(not active)
        self._overwrite_cb.setEnabled(not active)
        self._stop_on_failure_cb.setEnabled(not active)
        self._detect_btn.setEnabled(not active)
        self._validate_batch_btn.setEnabled(not active and has_rows)
        self._run_batch_btn.setEnabled(not active and has_rows)
        self._cancel_btn.setEnabled(active)
        self._open_output_folder_btn.setEnabled(not active)
        self._close_btn.setEnabled(not active)
        self._refresh_open_selected_state()
