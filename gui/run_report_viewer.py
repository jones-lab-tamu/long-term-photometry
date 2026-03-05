"""
RunReportViewer

Results browser driven strictly by <run_dir>/run_report.json.
Replaces ManifestViewer to conform to artifact-first rendering requirements.
"""

import os
import json

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QGroupBox, QScrollArea, QFrame, QPushButton, QSizePolicy,
    QTextEdit
)

from gui.run_report_parser import parse_run_report, get_summary_fields, resolve_quick_links


class CollapsibleRawViewer(QWidget):
    """A collapsible text edit section for displaying raw JSON properties."""
    def __init__(self, title: str, json_data: dict, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toggle button
        self._toggle_btn = QPushButton(f"▶ Viewer: Raw {title}")
        self._toggle_btn.setStyleSheet("text-align: left; padding: 4px;")
        self._toggle_btn.clicked.connect(self._toggle)
        layout.addWidget(self._toggle_btn)
        
        # Text area
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Consolas", 9))
        self._text_edit.setPlainText(json.dumps(json_data, indent=2))
        self._text_edit.hide() # Hidden by default
        
        # Determine fixed height based on content size, cap at 200px
        doc_height = int(self._text_edit.document().size().height()) + 10
        self._text_edit.setFixedHeight(min(doc_height, 200))
        
        layout.addWidget(self._text_edit)
        
        self._is_expanded = False

    def _toggle(self):
        self._is_expanded = not self._is_expanded
        if self._is_expanded:
            self._text_edit.show()
            self._toggle_btn.setText(self._toggle_btn.text().replace("▶", "▼"))
        else:
            self._text_edit.hide()
            self._toggle_btn.setText(self._toggle_btn.text().replace("▼", "▶"))


class RunReportViewer(QWidget):
    """Run Summary interface built exclusively from run_report.json."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        
        # Scroll area for arbitrary length content
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._layout.addWidget(self._scroll)
        
        # Main container inside scroll
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._scroll.setWidget(self._container)
        
        # Title Header
        self._title_label = QLabel("No run report loaded.")
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setStyleSheet("color: gray; font-size: 14px;")
        self._container_layout.addWidget(self._title_label)
        
        # Summary Box
        self._summary_box = QGroupBox("Run Summary")
        self._summary_layout = QGridLayout(self._summary_box)
        self._container_layout.addWidget(self._summary_box)
        self._summary_box.hide()
        
        # Quick Links Box
        self._links_box = QGroupBox("Quick Links")
        self._links_layout = QVBoxLayout(self._links_box)
        self._container_layout.addWidget(self._links_box)
        self._links_box.hide()
        
        # Raw Viewers Container
        self._raw_viewers_layout = QVBoxLayout()
        self._container_layout.addLayout(self._raw_viewers_layout)
        
        self._container_layout.addStretch()
        
        self._current_run_dir = None

    def clear(self):
        """Reset the UI to its empty state."""
        self._current_run_dir = None
        self._title_label.setText("No run report loaded.")
        self._title_label.setStyleSheet("color: gray; font-size: 14px;")
        
        self._clear_layout(self._summary_layout)
        self._summary_box.hide()
        
        self._clear_layout(self._links_layout)
        self._links_box.hide()
        
        self._clear_layout(self._raw_viewers_layout)

    def load_report(self, out_dir: str):
        """Load and display data from run_report.json in the specified directory."""
        self.clear()
        self._current_run_dir = out_dir
        
        report_path = os.path.join(out_dir, "run_report.json")
        data, parse_err = parse_run_report(report_path)
        
        if parse_err:
            self._title_label.setText(f"⚠ Missing or invalid run_report.json.\nReason: {parse_err}")
            self._title_label.setStyleSheet("color: red; font-weight: bold; font-size: 12px;")
            # Everything else remains hidden
            return
            
        # Determine Title / Preview Labeling
        is_preview = data.get("run_context", {}).get("run_type") == "preview"
        title_text = "Run Report"
        if is_preview:
            title_text += " [PREVIEW]"
        self._title_label.setText(title_text)
        self._title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Populate allowlisted Summary fields
        fields = get_summary_fields(data)
        if fields:
            self._summary_box.show()
            for row, (k, v) in enumerate(fields):
                lbl_k = QLabel(f"<b>{k}:</b>")
                lbl_v = QLabel(v)
                lbl_v.setTextInteractionFlags(Qt.TextSelectableByMouse)
                self._summary_layout.addWidget(lbl_k, row, 0, Qt.AlignRight)
                self._summary_layout.addWidget(lbl_v, row, 1, Qt.AlignLeft)
        
        # Populate Quick Links
        links = resolve_quick_links(out_dir, data)
        if links:
            self._links_box.show()
            for label, full_path, status in links:
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 2, 0, 2)
                
                name_lbl = QLabel(f"<b>{label}:</b>")
                row_layout.addWidget(name_lbl)
                
                if status == "ok":
                    btn = QPushButton("Open")
                    btn.setCursor(Qt.PointingHandCursor)
                    # Use default parameter binding for lambda closure
                    btn.clicked.connect(lambda checked=False, p=full_path: self._open_path(p))
                    row_layout.addWidget(btn)
                    
                    path_lbl = QLabel(full_path)
                    path_lbl.setStyleSheet("color: gray; font-size: 11px;")
                    row_layout.addWidget(path_lbl)
                else:
                    err_lbl = QLabel(f"[{status}]")
                    err_lbl.setStyleSheet("color: red;")
                    row_layout.addWidget(err_lbl)
                    
                    path_lbl = QLabel(full_path)
                    path_lbl.setStyleSheet("color: gray; font-size: 11px; text-decoration: line-through;")
                    row_layout.addWidget(path_lbl)
                    
                row_layout.addStretch()
                self._links_layout.addWidget(row_widget)
                
        # Populate Raw Viewers
        run_ctx = data.get("run_context", {})
        config_obj = data.get("configuration", {})
        
        if run_ctx:
            self._raw_viewers_layout.addWidget(CollapsibleRawViewer("run_context", run_ctx))
        if config_obj:
            self._raw_viewers_layout.addWidget(CollapsibleRawViewer("configuration", config_obj))
            
    def _open_path(self, path: str):
        """Use default OS handler to open files or folders."""
        if os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _clear_layout(self, layout):
        """Recursively clear widgets from a layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self._clear_layout(item.layout())
