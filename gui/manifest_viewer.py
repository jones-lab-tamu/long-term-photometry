"""
ManifestViewer, MANIFEST.json-driven results browser.

Loads MANIFEST.json from a pipeline output directory, renders a Summary tab
and one tab per ROI with PNG thumbnails, day-plot selector, and CSV table viewer.
All file paths are resolved exclusively from manifest['deliverables'][roi]['files'].
"""

import os
import json

import pandas as pd

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel,
    QScrollArea, QDialog, QGridLayout, QComboBox, QTableView,
    QSplitter, QGroupBox, QPushButton, QSizePolicy, QHeaderView,
)


# ======================================================================
# CSV Table Model
# ======================================================================

class CsvTableModel(QAbstractTableModel):
    """Pandas DataFrame-backed table model (read-only, max 200 rows)."""

    MAX_ROWS = 200

    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df.head(self.MAX_ROWS).reset_index(drop=True)

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            if isinstance(val, float):
                return f"{val:.6g}"
            return str(val)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section)


# ======================================================================
# Image Preview Dialog
# ======================================================================

class ImageDialog(QDialog):
    """Modal dialog showing a full-size (fit-to-window) image."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(image_path))
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)

        pix = QPixmap(image_path)
        if not pix.isNull():
            scaled = pix.scaled(860, 660, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)
        else:
            label.setText(f"Failed to load image:\n{image_path}")

        layout.addWidget(label)


# ======================================================================
# ROI Tab
# ======================================================================

class RoiTab(QWidget):
    """Tab for a single ROI: image thumbnails, day-plot selector, CSV table."""

    STATIC_PNGS = [
        "tonic_overview.png",
        "phasic_correction_impact.png",
        "phasic_peak_rate_timeseries.png",
        "phasic_auc_timeseries.png",
    ]
    CSV_FILES = [
        "tonic_df_timeseries.csv",
        "phasic_peak_rate_timeseries.csv",
        "phasic_auc_timeseries.csv",
    ]

    def __init__(self, roi_name: str, out_dir: str, deliverables: dict, parent=None):
        super().__init__(parent)
        self._roi_name = roi_name
        self._roi_dir = os.path.join(out_dir, roi_name)
        self._files_list = deliverables.get("files", [])
        self._days = deliverables.get("days_generated", [])

        layout = QVBoxLayout(self)

        # --- Image thumbnails ---
        img_group = QGroupBox("Deliverable Images")
        img_layout = QGridLayout(img_group)

        row, col = 0, 0
        for png in self.STATIC_PNGS:
            widget = self._make_thumbnail(png)
            img_layout.addWidget(widget, row, col)
            col += 1
            if col >= 4:
                col = 0
                row += 1

        layout.addWidget(img_group)

        # --- Day plot selector ---
        if self._days:
            day_group = QGroupBox("Day Plots")
            day_layout = QVBoxLayout(day_group)

            selector_row = QHBoxLayout()
            selector_row.addWidget(QLabel("Day:"))
            self._day_combo = QComboBox()
            self._day_combo.addItems(sorted(self._days))
            self._day_combo.currentTextChanged.connect(self._on_day_changed)
            selector_row.addWidget(self._day_combo)
            selector_row.addStretch()
            day_layout.addLayout(selector_row)

            self._day_thumb_layout = QHBoxLayout()
            day_layout.addLayout(self._day_thumb_layout)

            layout.addWidget(day_group)

            # Trigger initial display
            if self._days:
                self._on_day_changed(sorted(self._days)[0])

        # --- CSV table viewer ---
        csv_group = QGroupBox("CSV Data")
        csv_layout = QVBoxLayout(csv_group)

        csv_selector = QHBoxLayout()
        csv_selector.addWidget(QLabel("CSV:"))
        self._csv_combo = QComboBox()

        available_csvs = [c for c in self.CSV_FILES if c in self._files_list]
        self._csv_combo.addItems(available_csvs)
        self._csv_combo.currentTextChanged.connect(self._on_csv_changed)
        csv_selector.addWidget(self._csv_combo)
        csv_selector.addStretch()
        csv_layout.addLayout(csv_selector)

        self._table_view = QTableView()
        self._table_view.setSortingEnabled(True)
        self._table_view.setAlternatingRowColors(True)
        csv_layout.addWidget(self._table_view)

        self._csv_warning = QLabel("")
        self._csv_warning.setStyleSheet("color: red; font-weight: bold;")
        self._csv_warning.hide()
        csv_layout.addWidget(self._csv_warning)

        layout.addWidget(csv_group)

        # Load first CSV if available
        if available_csvs:
            self._on_csv_changed(available_csvs[0])

    # ------------------------------------------------------------------
    # Thumbnails
    # ------------------------------------------------------------------

    def _make_thumbnail(self, filename: str) -> QWidget:
        """Create a clickable thumbnail label for a PNG file."""
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(4, 4, 4, 4)

        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setFixedSize(200, 150)

        fpath = os.path.join(self._roi_dir, filename)

        if filename not in self._files_list:
            label.setText(f"Not in manifest:\n{filename}")
            label.setStyleSheet("color: gray;")
        elif not os.path.exists(fpath):
            label.setText(f"MISSING on disk:\n{filename}")
            label.setStyleSheet("color: red; font-weight: bold;")
        else:
            pix = QPixmap(fpath)
            if not pix.isNull():
                scaled = pix.scaled(192, 142, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)
                label.setCursor(Qt.PointingHandCursor)
                label.mousePressEvent = lambda ev, p=fpath: self._show_full_image(p)
            else:
                label.setText(f"Load error:\n{filename}")
                label.setStyleSheet("color: orange;")

        name_label = QLabel(filename)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setFont(QFont("", 8))
        name_label.setWordWrap(True)

        vbox.addWidget(label)
        vbox.addWidget(name_label)
        return container

    def _show_full_image(self, path: str):
        dlg = ImageDialog(path, self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Day plots
    # ------------------------------------------------------------------

    def _on_day_changed(self, day: str):
        # Clear existing thumbnails
        while self._day_thumb_layout.count():
            item = self._day_thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        day_files = [
            f"phasic_sig_iso_day_{day}.png",
            f"phasic_dFF_day_{day}.png",
            f"phasic_stacked_day_{day}.png",
        ]
        for fname in day_files:
            widget = self._make_thumbnail(fname)
            self._day_thumb_layout.addWidget(widget)

    # ------------------------------------------------------------------
    # CSV viewer
    # ------------------------------------------------------------------

    def _on_csv_changed(self, csv_name: str):
        self._csv_warning.hide()
        fpath = os.path.join(self._roi_dir, csv_name)

        if csv_name not in self._files_list:
            self._csv_warning.setText(f"⚠ {csv_name} not listed in manifest files")
            self._csv_warning.show()
            return

        if not os.path.exists(fpath):
            self._csv_warning.setText(f"⚠ {csv_name} listed in manifest but MISSING on disk")
            self._csv_warning.show()
            return

        try:
            df = pd.read_csv(fpath)
            model = CsvTableModel(df, self)
            self._table_view.setModel(model)
            self._table_view.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeToContents
            )
        except Exception as e:
            self._csv_warning.setText(f"⚠ Error reading {csv_name}: {e}")
            self._csv_warning.show()


# ======================================================================
# ManifestViewer
# ======================================================================

class ManifestViewer(QWidget):
    """Manifest-driven results browser. Shows Summary + per-ROI tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        self._layout.addWidget(self._tabs)

        self._status_label = QLabel("No results loaded.")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("color: gray; font-size: 12px;")
        self._layout.addWidget(self._status_label)

    def clear(self):
        """Remove all tabs."""
        self._tabs.clear()
        self._status_label.setText("No results loaded.")
        self._status_label.show()

    def load_manifest(self, out_dir: str):
        """Load MANIFEST.json from out_dir and populate tabs."""
        self._tabs.clear()
        self._status_label.hide()

        manifest_path = os.path.join(out_dir, "MANIFEST.json")
        if not os.path.exists(manifest_path):
            self._status_label.setText(
                f"⚠ MANIFEST.json not found in:\n{out_dir}"
            )
            self._status_label.setStyleSheet("color: red; font-weight: bold; font-size: 12px;")
            self._status_label.show()
            return

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except Exception as e:
            self._status_label.setText(f"⚠ Error reading MANIFEST.json: {e}")
            self._status_label.setStyleSheet("color: red; font-weight: bold; font-size: 12px;")
            self._status_label.show()
            return

        # --- Summary tab ---
        summary_widget = self._build_summary_tab(manifest)
        self._tabs.addTab(summary_widget, "Summary")

        # --- Per-ROI tabs ---
        regions = manifest.get("regions", [])
        deliverables = manifest.get("deliverables", {})

        for roi in regions:
            roi_deliverables = deliverables.get(roi, {})
            roi_tab = RoiTab(roi, out_dir, roi_deliverables, self)

            scroll = QScrollArea()
            scroll.setWidget(roi_tab)
            scroll.setWidgetResizable(True)

            self._tabs.addTab(scroll, roi)

    def _build_summary_tab(self, manifest: dict) -> QWidget:
        """Build the Summary tab from manifest metadata."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        fields = [
            ("Tool", manifest.get("tool", "--")),
            ("Timestamp", manifest.get("timestamp", "--")),
            ("Sessions/Hour", str(manifest.get("sessions_per_hour", "--"))),
            ("Session Duration (s)", str(manifest.get("session_duration_s", "--"))),
            ("Session Stride (s)", str(manifest.get("session_stride_s", "--"))),
            ("Regions", ", ".join(manifest.get("regions", []))),
        ]

        grid = QGridLayout()
        for i, (label_text, value_text) in enumerate(fields):
            key_label = QLabel(f"<b>{label_text}:</b>")
            val_label = QLabel(value_text)
            val_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(key_label, i, 0, Qt.AlignTop | Qt.AlignRight)
            grid.addWidget(val_label, i, 1, Qt.AlignTop | Qt.AlignLeft)

        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch()

        return widget
