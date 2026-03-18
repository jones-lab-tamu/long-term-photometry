"""
RunReportViewer

Complete-state results workspace driven by run outputs under <run_dir>.
"""

import os
import re
from typing import Dict, List, Tuple

from PySide6.QtCore import Qt, QSize, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QScrollArea,
    QFrame,
    QPushButton,
    QSizePolicy,
    QComboBox,
    QTabWidget,
)

from gui.run_report_parser import parse_run_report, get_preview_mode, resolve_region_deliverables


TAB_VERIFICATION = "Verification"
TAB_TONIC = "Tonic"
TAB_PHASIC_RAW = "Phasic Raw"
TAB_PHASIC_DFF = "Phasic dFF"
TAB_PHASIC_STACKED = "Phasic Stacked"
TAB_PHASIC_SUMMARY = "Phasic Summary"

TAB_ORDER = [
    TAB_VERIFICATION,
    TAB_TONIC,
    TAB_PHASIC_RAW,
    TAB_PHASIC_DFF,
    TAB_PHASIC_STACKED,
    TAB_PHASIC_SUMMARY,
]


class RunReportViewer(QWidget):
    """Results workspace for completed runs."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_run_dir = ""
        self._run_summary_path = ""
        self._region_paths: Dict[str, str] = {}
        self._region_tab_images: Dict[str, Dict[str, List[str]]] = {}
        self._tab_indices: Dict[Tuple[str, str], int] = {}
        self._active_image_path = ""
        self._active_pixmap = QPixmap()
        self._zoom_mode = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._status_label = QLabel("No completed results loaded.")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("color: gray; font-size: 14px;")
        root.addWidget(self._status_label)

        self._workspace = QWidget()
        ws = QVBoxLayout(self._workspace)
        ws.setContentsMargins(0, 0, 0, 0)
        ws.setSpacing(8)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Region:"))
        self._region_combo = QComboBox()
        self._region_combo.setToolTip("Select the ROI region shown in the results viewer.")
        self._region_combo.currentIndexChanged.connect(self._on_region_changed)
        selector_row.addWidget(self._region_combo, 1)
        selector_row.addStretch()
        ws.addLayout(selector_row)

        self._tabs = QTabWidget()
        self._tabs.setToolTip("Available result views for the selected region.")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        ws.addWidget(self._tabs)

        viewer_col = QVBoxLayout()
        viewer_col.setSpacing(8)
        self._image_title_label = QLabel("No image selected.")
        self._image_title_label.setAlignment(Qt.AlignCenter)
        viewer_col.addWidget(self._image_title_label)

        self._image_label = _ClickableImageLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet(
            "QLabel { background: #111; color: #ddd; border: 1px solid #444; }"
        )
        self._image_label.setText("No image available.")
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setToolTip("Click image to toggle fit/full-size inspection.")
        self._image_label.clicked.connect(self._on_image_clicked)

        self._image_scroll = QScrollArea()
        self._image_scroll.setWidgetResizable(False)
        self._image_scroll.setAlignment(Qt.AlignCenter)
        self._image_scroll.setFrameShape(QFrame.NoFrame)
        self._image_scroll.setWidget(self._image_label)
        self._image_scroll.setMinimumHeight(400)
        viewer_col.addWidget(self._image_scroll, 1)

        self._zoom_hint_label = QLabel("Click image to toggle fit/full size.")
        self._zoom_hint_label.setAlignment(Qt.AlignCenter)
        self._zoom_hint_label.setStyleSheet("font-size: 11px; color: #666;")
        viewer_col.addWidget(self._zoom_hint_label)

        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("<")
        self._prev_btn.setToolTip("Previous image in this tab.")
        self._prev_btn.clicked.connect(self._on_prev_image)
        nav_row.addWidget(self._prev_btn)
        self._image_counter_label = QLabel("")
        self._image_counter_label.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self._image_counter_label, 1)
        self._next_btn = QPushButton(">")
        self._next_btn.setToolTip("Next image in this tab.")
        self._next_btn.clicked.connect(self._on_next_image)
        nav_row.addWidget(self._next_btn)
        viewer_col.addLayout(nav_row)
        ws.addLayout(viewer_col, 1)

        action_group = QGroupBox("Selected Region Actions")
        action_group_layout = QVBoxLayout(action_group)
        self._actions_scroll = QScrollArea()
        self._actions_scroll.setWidgetResizable(True)
        self._actions_scroll.setFrameShape(QFrame.NoFrame)
        self._actions_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._actions_container = QWidget()
        self._actions_layout = QVBoxLayout(self._actions_container)
        self._actions_layout.setContentsMargins(0, 0, 0, 0)
        self._actions_layout.setSpacing(6)
        self._actions_layout.addStretch()
        self._actions_scroll.setWidget(self._actions_container)
        action_group_layout.addWidget(self._actions_scroll)
        action_group.setMaximumHeight(220)
        ws.addWidget(action_group, 0)
        root.addWidget(self._workspace, 1)

        self.clear()

    def clear(self):
        """Reset to idle/placeholder state."""
        self._current_run_dir = ""
        self._run_summary_path = ""
        self._region_paths = {}
        self._region_tab_images = {}
        self._tab_indices = {}
        self._active_image_path = ""
        self._active_pixmap = QPixmap()
        self._set_zoom_mode(False)

        self._set_status_message(
            "No completed results loaded. Run the pipeline or open a completed run folder.",
            level="idle",
        )

        self._region_combo.blockSignals(True)
        self._region_combo.clear()
        self._region_combo.blockSignals(False)

        self._tabs.blockSignals(True)
        while self._tabs.count() > 0:
            self._tabs.removeTab(0)
        self._tabs.blockSignals(False)

        self._show_no_image("No image available in the current results workspace.")
        self._clear_action_rows()
        self._workspace.hide()

    def set_running_message(self, message: str):
        """Show a minimal running-state message in the Results pane."""
        self.clear()
        self._set_status_message(message, level="running")

    def load_report(self, out_dir: str) -> bool:
        """Load complete-state workspace from a run directory."""
        self.clear()
        self._current_run_dir = out_dir

        run_report_path = os.path.join(out_dir, "run_report.json")
        data, parse_err = parse_run_report(run_report_path)
        is_preview = get_preview_mode(data) if data else False

        self._run_summary_path = ""
        if os.path.isfile(run_report_path):
            self._run_summary_path = run_report_path
        else:
            manifest_path = os.path.join(out_dir, "MANIFEST.json")
            if os.path.isfile(manifest_path):
                self._run_summary_path = manifest_path

        regions = resolve_region_deliverables(out_dir)
        for reg in regions:
            name = str(reg.get("name", "")).strip()
            reg_path = str(reg.get("path", "")).strip()
            if not name or not reg_path:
                continue
            self._region_paths[name] = reg_path
            self._region_tab_images[name] = self._discover_region_tab_images(reg_path)

        if not self._region_paths:
            if parse_err:
                self._set_status_message(
                    f"Could not load results metadata ({parse_err}). No region deliverables were found.",
                    level="error",
                )
            else:
                self._set_status_message(
                    "No region deliverables were found in the selected run directory.",
                    level="error",
                )
            self._workspace.hide()
            return False

        title = "Results workspace"
        if is_preview:
            title += " [PREVIEW]"
        self._set_status_message(title, level="ready")
        self._workspace.show()

        region_names = sorted(self._region_paths.keys(), key=lambda s: s.lower())
        self._region_combo.blockSignals(True)
        self._region_combo.clear()
        self._region_combo.addItems(region_names)
        self._region_combo.blockSignals(False)

        if self._region_combo.count() > 0:
            self._region_combo.setCurrentIndex(0)
            self._on_region_changed(0)
        else:
            self._show_no_image("No region available.")
            self._clear_action_rows()

        return True

    def _discover_region_tab_images(self, region_path: str) -> Dict[str, List[str]]:
        """Discover per-tab image lists for one region directory."""
        summary_dir = os.path.join(region_path, "summary")
        day_plots_dir = os.path.join(region_path, "day_plots")

        verification = self._discover_verification_images(summary_dir)
        tonic = self._discover_tonic_images(summary_dir)
        phasic_raw = self._discover_day_series_images(
            day_plots_dir,
            r"^phasic_sig_iso_day_\d{3,}\.png$",
            ignore_case=False,
        )
        phasic_dff = self._discover_day_series_images(
            day_plots_dir,
            r"^phasic_dff_day_\d{3,}\.png$",
            ignore_case=True,
        )
        phasic_stacked = self._discover_day_series_images(
            day_plots_dir,
            r"^phasic_stacked_day_\d{3,}\.png$",
            ignore_case=False,
        )
        phasic_summary = self._discover_phasic_summary_images(summary_dir)

        return {
            TAB_VERIFICATION: verification,
            TAB_TONIC: tonic,
            TAB_PHASIC_RAW: phasic_raw,
            TAB_PHASIC_DFF: phasic_dff,
            TAB_PHASIC_STACKED: phasic_stacked,
            TAB_PHASIC_SUMMARY: phasic_summary,
        }

    @staticmethod
    def _dedupe_sorted_existing(paths: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for p in sorted(paths, key=lambda x: os.path.basename(x).lower()):
            norm = os.path.normcase(os.path.normpath(p))
            if norm in seen:
                continue
            if not os.path.isfile(p):
                continue
            seen.add(norm)
            out.append(p)
        return out

    def _discover_verification_images(self, summary_dir: str) -> List[str]:
        """
        Verification tab contract.
        Anchor on phasic_correction_impact.png with narrowly bounded expansion.
        """
        if not os.path.isdir(summary_dir):
            return []
        candidates = [
            os.path.join(summary_dir, "phasic_correction_impact.png"),
        ]
        for name in os.listdir(summary_dir):
            if name.startswith("phasic_correction_impact_") and name.endswith(".png"):
                candidates.append(os.path.join(summary_dir, name))
        return self._dedupe_sorted_existing(candidates)

    def _discover_tonic_images(self, summary_dir: str) -> List[str]:
        """
        Tonic tab contract.
        Anchor on tonic_overview.png with narrowly bounded expansion.
        """
        if not os.path.isdir(summary_dir):
            return []
        candidates = [
            os.path.join(summary_dir, "tonic_overview.png"),
        ]
        for name in os.listdir(summary_dir):
            if name.startswith("tonic_overview_") and name.endswith(".png"):
                candidates.append(os.path.join(summary_dir, name))
        return self._dedupe_sorted_existing(candidates)

    def _discover_phasic_summary_images(self, summary_dir: str) -> List[str]:
        """
        Phasic Summary tab contract.
        Anchored on auc/peak-rate timeseries images with bounded optional variants.
        """
        if not os.path.isdir(summary_dir):
            return []
        candidates = [
            os.path.join(summary_dir, "phasic_auc_timeseries.png"),
            os.path.join(summary_dir, "phasic_peak_rate_timeseries.png"),
        ]
        for name in os.listdir(summary_dir):
            if name.startswith("phasic_auc_timeseries_") and name.endswith(".png"):
                candidates.append(os.path.join(summary_dir, name))
            if name.startswith("phasic_peak_rate_timeseries_") and name.endswith(".png"):
                candidates.append(os.path.join(summary_dir, name))
        return self._dedupe_sorted_existing(candidates)

    def _discover_day_series_images(self, directory: str, pattern: str, ignore_case: bool) -> List[str]:
        """Discover day-series images using explicit filename regex contract."""
        if not os.path.isdir(directory):
            return []
        flags = re.IGNORECASE if ignore_case else 0
        rx = re.compile(pattern, flags)
        matches: List[str] = []
        for name in os.listdir(directory):
            if rx.fullmatch(name):
                p = os.path.join(directory, name)
                if os.path.isfile(p):
                    matches.append(p)
        return self._dedupe_sorted_existing(matches)

    def _on_region_changed(self, _index: int):
        """Refresh tabs, viewer, and actions for selected region."""
        self._rebuild_tabs_for_selected_region()
        self._refresh_action_rows()

    def _on_tab_changed(self, _index: int):
        """Refresh image viewer when tab changes."""
        self._refresh_active_image(reset_index=True)

    def _selected_region(self) -> str:
        return self._region_combo.currentText().strip()

    def _selected_tab(self) -> str:
        idx = self._tabs.currentIndex()
        if idx < 0:
            return ""
        return self._tabs.tabText(idx).strip()

    def _rebuild_tabs_for_selected_region(self):
        region = self._selected_region()
        tab_map = self._region_tab_images.get(region, {})
        available_tabs = [t for t in TAB_ORDER if tab_map.get(t)]

        current_tab = self._selected_tab()
        self._tabs.blockSignals(True)
        while self._tabs.count() > 0:
            self._tabs.removeTab(0)
        for tab_name in available_tabs:
            self._tabs.addTab(QWidget(), tab_name)
        self._tabs.blockSignals(False)

        if not available_tabs:
            self._show_no_image("No images available for the selected region.")
            return

        if current_tab in available_tabs:
            idx = available_tabs.index(current_tab)
        else:
            idx = 0
        self._tabs.setCurrentIndex(idx)
        self._refresh_active_image(reset_index=True)

    def _current_tab_images(self) -> List[str]:
        region = self._selected_region()
        tab = self._selected_tab()
        if not region or not tab:
            return []
        return list(self._region_tab_images.get(region, {}).get(tab, []))

    def _tab_key(self) -> Tuple[str, str]:
        return (self._selected_region(), self._selected_tab())

    def _refresh_active_image(self, reset_index: bool):
        images = self._current_tab_images()
        if not images:
            self._show_no_image("No images available for this tab.")
            return

        key = self._tab_key()
        if reset_index:
            idx = 0
        else:
            idx = self._tab_indices.get(key, 0)
        idx = max(0, min(idx, len(images) - 1))
        self._tab_indices[key] = idx

        path = images[idx]
        self._active_image_path = path
        self._image_title_label.setText(os.path.basename(path))
        self._set_zoom_mode(False)
        self._set_image(path)

        n = len(images)
        multi = n > 1
        self._prev_btn.setVisible(multi)
        self._next_btn.setVisible(multi)
        self._image_counter_label.setText(f"{idx + 1}/{n}" if multi else "")

    def _set_image(self, path: str):
        if not path or not os.path.isfile(path):
            self._show_no_image("Image file missing.")
            return
        pix = QPixmap(path)
        if pix.isNull():
            self._show_no_image(f"Unable to render image:\n{os.path.basename(path)}")
            return

        self._active_pixmap = pix
        self._render_image()

    def _on_image_clicked(self) -> None:
        """Toggle between fit-to-view and full-size inspection mode."""
        if self._active_pixmap.isNull():
            return
        self._set_zoom_mode(not self._zoom_mode)
        self._render_image()

    def _set_zoom_mode(self, enabled: bool) -> None:
        self._zoom_mode = bool(enabled)
        self._zoom_hint_label.setText(
            "Click image to return to fit mode." if self._zoom_mode
            else "Click image to toggle fit/full size."
        )

    def _render_image(self) -> None:
        """Render active image in fit or full-size mode."""
        if self._active_pixmap.isNull():
            return
        if self._zoom_mode:
            self._image_label.setPixmap(self._active_pixmap)
            self._image_label.resize(self._active_pixmap.size())
            self._image_label.setText("")
            return

        viewport = self._image_scroll.viewport().size()
        if viewport.width() < 10 or viewport.height() < 10:
            viewport = QSize(1000, 700)
        scaled = self._active_pixmap.scaled(viewport, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image_label.setPixmap(scaled)
        self._image_label.resize(scaled.size())
        self._image_label.setText("")

    def _show_no_image(self, message: str):
        self._active_image_path = ""
        self._active_pixmap = QPixmap()
        self._set_zoom_mode(False)
        self._image_label.setPixmap(QPixmap())
        self._image_label.setText(message)
        viewport = self._image_scroll.viewport().size()
        if viewport.width() < 10 or viewport.height() < 10:
            viewport = QSize(640, 400)
        self._image_label.resize(viewport)
        self._image_title_label.setText("No image selected.")
        self._prev_btn.setVisible(False)
        self._next_btn.setVisible(False)
        self._image_counter_label.setText("")

    def _on_prev_image(self):
        images = self._current_tab_images()
        if len(images) <= 1:
            return
        key = self._tab_key()
        cur = self._tab_indices.get(key, 0)
        self._tab_indices[key] = (cur - 1) % len(images)
        self._refresh_active_image(reset_index=False)

    def _on_next_image(self):
        images = self._current_tab_images()
        if len(images) <= 1:
            return
        key = self._tab_key()
        cur = self._tab_indices.get(key, 0)
        self._tab_indices[key] = (cur + 1) % len(images)
        self._refresh_active_image(reset_index=False)

    def _refresh_action_rows(self):
        self._clear_action_rows()

        # Run-level summary action
        self._add_action_row("Run summary", self._run_summary_path)

        region = self._selected_region()
        reg_path = self._region_paths.get(region, "")

        region_lbl = QLabel(f"Selected region: {region or '(none)'}")
        region_lbl.setWordWrap(True)
        self._actions_layout.addWidget(region_lbl)

        self._add_action_row("Open Summary", os.path.join(reg_path, "summary"))
        self._add_action_row("Open Day Plots", os.path.join(reg_path, "day_plots"))
        self._add_action_row("Open Tables", os.path.join(reg_path, "tables"))
        self._actions_layout.addStretch()

    def _add_action_row(self, label: str, path: str):
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        name = QLabel(label)
        lay.addWidget(name)
        btn = QPushButton("Open")
        btn.setFixedWidth(64)
        exists = bool(path and os.path.exists(path))
        btn.setEnabled(exists)
        if path:
            btn.setToolTip(path if exists else f"Not available: {path}")
        btn.clicked.connect(lambda _checked=False, p=path: self._open_path(p))
        lay.addWidget(btn)
        lay.addStretch()
        self._actions_layout.addWidget(row)

    def _open_path(self, path: str):
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _clear_action_rows(self):
        self._clear_layout(self._actions_layout)

    def _clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                child = item.layout()
                if child is not None:
                    self._clear_layout(child)

    def _set_status_message(self, text: str, level: str) -> None:
        """Centralized status-label styling for workspace states."""
        style_map = {
            "idle": "color: gray; font-size: 13px;",
            "running": "color: #666; font-size: 13px;",
            "ready": "font-weight: bold; font-size: 14px;",
            "error": "color: #a94442; font-size: 12px;",
        }
        self._status_label.setText(text)
        self._status_label.setStyleSheet(style_map.get(level, style_map["idle"]))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._active_image_path and os.path.isfile(self._active_image_path):
            self._render_image()


class _ClickableImageLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
