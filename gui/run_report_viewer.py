"""
RunReportViewer

Complete-state results workspace driven by run outputs under <run_dir>.
"""

import csv
import io
import math
import os
import re
from typing import Any, Dict, List, Tuple

from PySide6.QtCore import Qt, QSize, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
    QPushButton,
    QSizePolicy,
    QComboBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QApplication,
)

from gui.run_report_parser import (
    parse_run_report,
    get_preview_mode,
    get_run_type,
    resolve_region_deliverables,
    classify_completed_run_terminal_state,
    get_scientist_completion_summary,
    is_continuous_rwd_run_mode,
    declared_completed_run_mode,
    is_guided_continuous_saved_artifact_run_mode,
    build_guided_continuous_saved_artifact_index,
    SavedArtifactIndexError,
)
from gui.interactive_image import InteractiveImageLabel, InteractiveImageController
from photometry_pipeline.guided_completed_applied_dff_reload import (
    load_guided_completed_applied_dff_state,
    GuidedCompletedAppliedDffState,
    format_guided_completed_applied_dff_summary,
    format_guided_completed_applied_dff_technical_details,
)
from photometry_pipeline.guided_display_labels import (
    feature_event_signal_display_label,
    feature_threshold_method_display_label,
    format_display_label,
)
from photometry_pipeline.guided_completed_feature_event_reload import (
    load_guided_completed_feature_event_state,
    GuidedCompletedFeatureEventState,
    format_guided_completed_feature_event_summary,
    format_guided_completed_feature_event_technical_details,
)
from photometry_pipeline.completed_run_review import (
    CompletedRunReviewError,
    CompletedRunReviewModel,
    format_tonic_settings_summary,
    load_completed_phasic_review,
)
from photometry_pipeline.completed_continuous_rwd_review import (
    CompletedContinuousRwdReviewError,
    ContinuousRunOverview,
    load_continuous_phasic_events,
    load_continuous_roi_trace,
    load_continuous_run_overview,
    load_continuous_window_summary,
)
from photometry_pipeline.continuous_outputs import CONTINUOUS_TRACE_OVERVIEW_MAX_POINTS
from photometry_pipeline.tonic_session_plot import (
    TONIC_FALLBACK_NOTE,
    TONIC_SESSION_PLOT_FILENAME,
    METHOD_SIGNAL_ONLY,
    tonic_method_by_roi,
)


NATIVE_CSV_PREVIEW_ROW_LIMIT = 5000

TAB_VERIFICATION = "Verification"
TAB_TONIC = "Tonic"
TAB_PHASIC_RAW = "Phasic Sig/Iso"
TAB_PHASIC_DYNAMIC_FIT = "Dynamic Fit"
TAB_PHASIC_CORRECTION_REFERENCE = "Correction Reference"
TAB_PHASIC_DFF = "Phasic dFF"
TAB_PHASIC_STACKED = "Phasic Stacked"
TAB_PHASIC_SUMMARY = "Phasic Summary"
TAB_CONTINUOUS_TRACE = "Continuous Trace"

_RESULT_TAB_TOOLTIPS = {
    TAB_PHASIC_RAW: "Raw signal and reference traces for each plotted day.",
    TAB_PHASIC_DYNAMIC_FIT: "Dynamic fitted reference used for phasic correction.",
    TAB_PHASIC_CORRECTION_REFERENCE: "Reference trace used for phasic correction.",
    TAB_PHASIC_DFF: "Reference-corrected phasic dF/F trace for each plotted day.",
    TAB_PHASIC_STACKED: "Stacked phasic traces comparing the plotted days.",
    TAB_PHASIC_SUMMARY: "Summary plots of phasic event activity across plotted days.",
}

TAB_ORDER = [
    TAB_VERIFICATION,
    TAB_TONIC,
    TAB_PHASIC_RAW,
    TAB_PHASIC_DYNAMIC_FIT,
    TAB_PHASIC_DFF,
    TAB_PHASIC_STACKED,
    TAB_PHASIC_SUMMARY,
    TAB_CONTINUOUS_TRACE,
]

_NATIVE_PHASIC_SUMMARY_IMAGE_ORDER = {
    "phasic_auc_timeseries.png": 0,
    "phasic_peak_rate_timeseries.png": 1,
}
_NATIVE_CONTINUOUS_IMAGE_TABS = {
    "phasic_correction_impact.png": TAB_VERIFICATION,
    "tonic_overview.png": TAB_TONIC,
    "phasic_auc_timeseries.png": TAB_PHASIC_SUMMARY,
    "phasic_peak_rate_timeseries.png": TAB_PHASIC_SUMMARY,
}
_NATIVE_CONTINUOUS_DAY_PLOT_FAMILY_TABS = {
    "sampled_signal_reference": TAB_PHASIC_RAW,
    "sampled_correction_reference": TAB_PHASIC_CORRECTION_REFERENCE,
    "sampled_phasic_dff": TAB_PHASIC_DFF,
    "sampled_stacked": TAB_PHASIC_STACKED,
}
_NATIVE_CONTINUOUS_TAB_ORDER = (
    TAB_VERIFICATION,
    TAB_TONIC,
    TAB_PHASIC_RAW,
    TAB_PHASIC_CORRECTION_REFERENCE,
    TAB_PHASIC_DFF,
    TAB_PHASIC_STACKED,
    TAB_PHASIC_SUMMARY,
)


class RunReportViewer(QWidget):
    """Results workspace for completed runs."""
    region_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_run_dir = ""
        self._run_summary_path = ""
        self._continuous_overview: ContinuousRunOverview | None = None
        self._continuous_selected_roi = ""
        self._native_continuous_artifact_index: Dict[str, Any] | None = None
        self._native_continuous_artifacts_by_roi: Dict[str, List[Dict[str, Any]]] = {}
        self._native_continuous_artifacts_by_tab: Dict[
            Tuple[str, str], List[Dict[str, Any]]
        ] = {}
        self._native_continuous_artifact_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._native_continuous_context: Dict[str, Any] = {}
        self._native_continuous_mode = False
        self._applied_dff_state = GuidedCompletedAppliedDffState.absent()
        self._feature_event_state = GuidedCompletedFeatureEventState.absent()
        self._phasic_review_model: CompletedRunReviewModel | None = None
        self._completed_review_overview: dict = {}
        self._phasic_review_error = ""
        self._region_paths: Dict[str, str] = {}
        self._region_tab_images: Dict[str, Dict[str, List[str]]] = {}
        self._tonic_method_by_roi: Dict[str, Dict[str, str]] = {}
        self._tab_indices: Dict[Tuple[str, str], int] = {}
        self._external_tab_image_overrides: Dict[Tuple[str, str], List[str]] = {}
        self._active_image_path = ""
        self._active_pixmap = QPixmap()
        self._zoom_mode = False
        self._image_interaction: InteractiveImageController | None = None

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._status_label = QLabel("No completed results loaded.")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: gray; font-size: 14px;")
        # Ignored horizontally: the label never forces the viewer wider than
        # its allocated width (it just wraps at whatever width the layout
        # gives it). Preferred vertically, combined with word wrap, makes
        # Qt use QLabel's height-for-width sizing so a two- or three-line
        # warning grows the label instead of being clipped to one line.
        self._status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        root.addWidget(self._status_label)

        self._correction_summary_label = QLabel("")
        self._correction_summary_label.setObjectName("completedRunCorrectionSummary")
        self._correction_summary_label.setWordWrap(True)
        self._correction_summary_label.setVisible(False)
        root.addWidget(self._correction_summary_label)

        self._selected_feature_settings_label = QLabel("")
        self._selected_feature_settings_label.setObjectName(
            "completedRunSelectedFeatureSettings"
        )
        self._selected_feature_settings_label.setWordWrap(True)
        self._selected_feature_settings_label.setVisible(False)
        root.addWidget(self._selected_feature_settings_label)

        # Shown only for an ROI whose tonic result had to use the signal-only
        # fallback, so the scientist is never left reading raw-fluorescence
        # tonic values as if they were dF/F0. One line, once per ROI.
        self._tonic_method_note_label = QLabel("")
        self._tonic_method_note_label.setObjectName("completedRunTonicMethodNote")
        self._tonic_method_note_label.setWordWrap(True)
        self._tonic_method_note_label.setVisible(False)
        root.addWidget(self._tonic_method_note_label)

        self._tonic_settings_summary_label = QLabel("")
        self._tonic_settings_summary_label.setObjectName(
            "completedRunTonicSettingsSummary"
        )
        self._tonic_settings_summary_label.setWordWrap(True)
        self._tonic_settings_summary_label.setVisible(False)
        root.addWidget(self._tonic_settings_summary_label)

        self._workspace = QWidget()
        self._workspace.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._workspace.setMinimumHeight(0)
        ws = QVBoxLayout(self._workspace)
        ws.setContentsMargins(0, 0, 0, 0)
        ws.setSpacing(6)

        self._applied_dff_summary_label = QLabel(
            format_guided_completed_applied_dff_summary(
                self._applied_dff_state
            )
        )
        self._applied_dff_summary_label.setObjectName(
            "completedRunAppliedDffSummary"
        )
        # Not every run type produces a separate applied-dF/F routing
        # artifact. That is normal, not a missing result, so this line is
        # hidden unless the run actually has scientist-facing applied-dF/F
        # content to show.
        self._applied_dff_summary_label.setVisible(self._applied_dff_state.present)
        self._applied_dff_summary_label.setWordWrap(True)
        self._applied_dff_summary_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        self._applied_dff_summary_label.setFrameShape(QFrame.StyledPanel)
        self._applied_dff_summary_label.setContentsMargins(8, 6, 8, 6)
        ws.addWidget(self._applied_dff_summary_label)

        self._applied_dff_details_toggle = QPushButton("Show technical details")
        self._applied_dff_details_toggle.setObjectName(
            "completedRunAppliedDffDetailsToggle"
        )
        self._applied_dff_details_toggle.setCheckable(True)
        self._applied_dff_details_toggle.setVisible(self._applied_dff_state.present)
        self._applied_dff_details_toggle.toggled.connect(
            self._on_applied_dff_details_toggled
        )
        ws.addWidget(self._applied_dff_details_toggle, 0, Qt.AlignLeft)

        self._applied_dff_details_label = QLabel(
            format_guided_completed_applied_dff_technical_details(
                self._applied_dff_state
            )
        )
        self._applied_dff_details_label.setObjectName(
            "completedRunAppliedDffTechnicalDetails"
        )
        self._applied_dff_details_label.setWordWrap(True)
        self._applied_dff_details_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        self._applied_dff_details_label.setFrameShape(QFrame.StyledPanel)
        self._applied_dff_details_label.setContentsMargins(8, 6, 8, 6)
        self._applied_dff_details_label.setVisible(False)
        ws.addWidget(self._applied_dff_details_label)

        self._feature_event_summary_label = QLabel(
            format_guided_completed_feature_event_summary(self._feature_event_state)
        )
        self._feature_event_summary_label.setObjectName(
            "completedRunFeatureEventSummary"
        )
        # A run with no per-ROI feature settings (the common case) writes no
        # feature-detection settings file at all, so this line is hidden
        # rather than shown as an empty or reassuring-sounding default state.
        self._feature_event_summary_label.setVisible(self._feature_event_state.present)
        self._feature_event_summary_label.setWordWrap(True)
        self._feature_event_summary_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        self._feature_event_summary_label.setFrameShape(QFrame.StyledPanel)
        self._feature_event_summary_label.setContentsMargins(8, 6, 8, 6)
        ws.addWidget(self._feature_event_summary_label)

        self._feature_event_details_toggle = QPushButton("Show technical details")
        self._feature_event_details_toggle.setObjectName(
            "completedRunFeatureEventDetailsToggle"
        )
        self._feature_event_details_toggle.setCheckable(True)
        self._feature_event_details_toggle.setVisible(self._feature_event_state.present)
        self._feature_event_details_toggle.toggled.connect(
            self._on_feature_event_details_toggled
        )
        ws.addWidget(self._feature_event_details_toggle, 0, Qt.AlignLeft)

        self._feature_event_details_label = QLabel(
            format_guided_completed_feature_event_technical_details(
                self._feature_event_state
            )
        )
        self._feature_event_details_label.setObjectName(
            "completedRunFeatureEventTechnicalDetails"
        )
        self._feature_event_details_label.setWordWrap(True)
        self._feature_event_details_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )
        self._feature_event_details_label.setFrameShape(QFrame.StyledPanel)
        self._feature_event_details_label.setContentsMargins(8, 6, 8, 6)
        self._feature_event_details_label.setVisible(False)
        ws.addWidget(self._feature_event_details_label)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Region:"))
        self._region_combo = QComboBox()
        self._region_combo.setToolTip("Select the ROI region shown in the results viewer.")
        self._region_combo.currentIndexChanged.connect(self._on_region_changed)
        selector_row.addWidget(self._region_combo, 1)
        self._open_run_report_btn = QPushButton("Run Report")
        self._open_run_report_btn.setToolTip(
            "Open the saved analysis report for this run."
        )
        self._open_run_report_btn.clicked.connect(
            lambda _checked=False: self._open_path(self._run_summary_path)
        )
        selector_row.addWidget(self._open_run_report_btn)
        self._open_region_summary_btn = QPushButton("Summary")
        self._open_region_summary_btn.setToolTip("Open the selected region summary folder.")
        self._open_region_summary_btn.clicked.connect(
            lambda _checked=False: self._open_selected_region_subpath("summary")
        )
        selector_row.addWidget(self._open_region_summary_btn)
        self._open_region_day_plots_btn = QPushButton("Day Plots")
        self._open_region_day_plots_btn.setToolTip("Open the selected region day_plots folder.")
        self._open_region_day_plots_btn.clicked.connect(
            lambda _checked=False: self._open_selected_region_subpath("day_plots")
        )
        selector_row.addWidget(self._open_region_day_plots_btn)
        self._open_region_tables_btn = QPushButton("Tables")
        self._open_region_tables_btn.setToolTip("Open the selected region tables folder.")
        self._open_region_tables_btn.clicked.connect(
            lambda _checked=False: self._open_selected_table_or_region_tables()
        )
        selector_row.addWidget(self._open_region_tables_btn)
        selector_row.addStretch()
        ws.addLayout(selector_row)

        self._tabs = QTabWidget()
        self._tabs.setToolTip("Available result views for the selected region.")
        self._tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self._tabs.setMaximumHeight(44)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        ws.addWidget(self._tabs)

        viewer_col = QVBoxLayout()
        viewer_col.setSpacing(8)
        self._image_title_label = QLabel("No image selected.")
        self._image_title_label.setAlignment(Qt.AlignCenter)
        self._image_title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        viewer_col.addWidget(self._image_title_label)

        self._artifact_metadata_label = QLabel("")
        self._artifact_metadata_label.setObjectName("completedRunArtifactMetadata")
        self._artifact_metadata_label.setAlignment(Qt.AlignCenter)
        self._artifact_metadata_label.setWordWrap(True)
        self._artifact_metadata_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._artifact_metadata_label.setStyleSheet("font-size: 11px; color: #666;")
        self._artifact_metadata_label.setVisible(False)
        viewer_col.addWidget(self._artifact_metadata_label)

        self._image_label = InteractiveImageLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet(
            "QLabel { background: #111; color: #ddd; border: 1px solid #444; }"
        )
        self._image_label.setText("No image available.")
        self._image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._image_label.setToolTip("Click image to toggle fit/full-size inspection.")
        self._image_label.clicked.connect(self._on_image_clicked)

        self._image_scroll = QScrollArea()
        self._image_scroll.setWidgetResizable(False)
        self._image_scroll.setAlignment(Qt.AlignCenter)
        self._image_scroll.setFrameShape(QFrame.NoFrame)
        self._image_scroll.setWidget(self._image_label)
        self._image_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_scroll.setMinimumHeight(260)
        viewer_col.addWidget(self._image_scroll, 1)

        self._zoom_hint_label = QLabel("Scroll wheel to zoom. Click image to toggle fit/full size.")
        self._zoom_hint_label.setAlignment(Qt.AlignCenter)
        self._zoom_hint_label.setStyleSheet("font-size: 11px; color: #666;")
        viewer_col.addWidget(self._zoom_hint_label)
        self._image_interaction = InteractiveImageController(
            label=self._image_label,
            scroll_area=self._image_scroll,
            set_hint_text=self._zoom_hint_label.setText,
            fit_hint="Scroll wheel to zoom. Click image to toggle fit/full size.",
            zoom_hint="Scroll wheel to zoom in/out. Drag to pan. Click image to return to fit.",
            allow_upscale_in_fit=True,
            on_zoom_mode_changed=self._on_zoom_mode_changed,
        )

        self._artifact_table = QTableWidget(0, 0)
        self._artifact_table.setObjectName("completedRunArtifactTable")
        self._artifact_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._artifact_table.setSortingEnabled(False)
        self._artifact_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._artifact_table.setWordWrap(False)
        self._artifact_table_scroll = QScrollArea()
        self._artifact_table_scroll.setWidgetResizable(True)
        self._artifact_table_scroll.setFrameShape(QFrame.NoFrame)
        self._artifact_table_scroll.setWidget(self._artifact_table)
        self._artifact_table_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._artifact_table_scroll.setMinimumHeight(260)
        self._artifact_table_scroll.setVisible(False)
        viewer_col.addWidget(self._artifact_table_scroll, 1)

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
        ws.setStretch(3, 1)
        root.addWidget(self._workspace, 1)

        self._continuous_workspace = self._build_continuous_workspace()
        root.addWidget(self._continuous_workspace, 1)

        self.clear()

    def _build_continuous_workspace(self) -> QWidget:
        """Build the continuous-recording Results presentation (CR1-E1-B).

        One recording, one shared ROI selector, and up to two analysis tabs
        (Tonic/Phasic) added only for analyses that actually ran -- never one
        tab per ROI, never a fabricated empty tab. See the CR1-E1-B handoff,
        section 9.
        """
        workspace = QWidget()
        workspace.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cw = QVBoxLayout(workspace)
        cw.setContentsMargins(0, 0, 0, 0)
        cw.setSpacing(6)

        self._continuous_overview_label = QLabel("")
        self._continuous_overview_label.setObjectName("continuousRunOverviewLabel")
        self._continuous_overview_label.setWordWrap(True)
        self._continuous_overview_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._continuous_overview_label.setFrameShape(QFrame.StyledPanel)
        self._continuous_overview_label.setContentsMargins(8, 6, 8, 6)
        cw.addWidget(self._continuous_overview_label)

        self._continuous_roi_row = QWidget()
        roi_row = QHBoxLayout(self._continuous_roi_row)
        roi_row.setContentsMargins(0, 0, 0, 0)
        roi_row.addWidget(QLabel("Region:"))
        self._continuous_roi_combo = QComboBox()
        self._continuous_roi_combo.setToolTip(
            "Select the region of interest shown in the continuous results viewer."
        )
        self._continuous_roi_combo.currentIndexChanged.connect(
            self._on_continuous_roi_changed
        )
        roi_row.addWidget(self._continuous_roi_combo, 1)
        cw.addWidget(self._continuous_roi_row)

        self._continuous_tabs = QTabWidget()
        self._continuous_tabs.setToolTip(
            "Available continuous analysis views for the selected region."
        )
        cw.addWidget(self._continuous_tabs, 1)

        self._continuous_tonic_page = QWidget()
        (
            self._continuous_tonic_image_label,
            self._continuous_tonic_scroll,
            self._continuous_tonic_interaction,
            self._continuous_tonic_summary_table,
        ) = self._build_continuous_analysis_page(
            self._continuous_tonic_page,
            "No tonic trace loaded.",
            "continuousTonicSummaryTable",
        )

        self._continuous_phasic_page = QWidget()
        (
            self._continuous_phasic_image_label,
            self._continuous_phasic_scroll,
            self._continuous_phasic_interaction,
            self._continuous_phasic_summary_table,
        ) = self._build_continuous_analysis_page(
            self._continuous_phasic_page,
            "No phasic trace loaded.",
            "continuousPhasicSummaryTable",
            event_count_label_name="continuousPhasicEventCountLabel",
        )
        self._continuous_phasic_event_count_label = self._continuous_phasic_page.findChild(
            QLabel, "continuousPhasicEventCountLabel"
        )

        return workspace

    def _build_continuous_analysis_page(
        self,
        page: QWidget,
        placeholder_text: str,
        table_object_name: str,
        *,
        event_count_label_name: str | None = None,
    ) -> Tuple[InteractiveImageLabel, QScrollArea, InteractiveImageController, QTableWidget]:
        """Build one Tonic/Phasic analysis tab page: an in-memory-rendered
        trace image plus the persisted per-window summary table."""
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)

        image_label = InteractiveImageLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet(
            "QLabel { background: #111; color: #ddd; border: 1px solid #444; }"
        )
        image_label.setText(placeholder_text)
        image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(image_label)
        scroll.setMinimumHeight(220)
        layout.addWidget(scroll, 1)

        interaction = InteractiveImageController(
            label=image_label,
            scroll_area=scroll,
            set_hint_text=lambda _text: None,
            fit_hint="",
            zoom_hint="",
            allow_upscale_in_fit=True,
        )

        if event_count_label_name:
            event_count_label = QLabel("")
            event_count_label.setObjectName(event_count_label_name)
            layout.addWidget(event_count_label)

        table = QTableWidget(0, 0)
        table.setObjectName(table_object_name)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setMaximumHeight(220)
        layout.addWidget(table)

        return image_label, scroll, interaction, table

    def _on_applied_dff_details_toggled(self, checked: bool) -> None:
        self._applied_dff_details_label.setVisible(bool(checked))
        self._applied_dff_details_toggle.setText(
            "Hide technical details" if checked else "Show technical details"
        )

    def _refresh_applied_dff_display(self) -> None:
        self._applied_dff_summary_label.setText(
            format_guided_completed_applied_dff_summary(
                self._applied_dff_state
            )
        )
        self._applied_dff_summary_label.setVisible(self._applied_dff_state.present)
        self._applied_dff_details_label.setText(
            format_guided_completed_applied_dff_technical_details(
                self._applied_dff_state
            )
        )
        self._applied_dff_details_toggle.setVisible(self._applied_dff_state.present)
        self._applied_dff_details_toggle.setChecked(False)

    def _on_feature_event_details_toggled(self, checked: bool) -> None:
        self._feature_event_details_label.setVisible(bool(checked))
        self._feature_event_details_toggle.setText(
            "Hide technical details" if checked else "Show technical details"
        )

    def _refresh_feature_event_display(self) -> None:
        self._feature_event_summary_label.setText(
            format_guided_completed_feature_event_summary(self._feature_event_state)
        )
        self._feature_event_summary_label.setVisible(self._feature_event_state.present)
        self._feature_event_details_label.setText(
            format_guided_completed_feature_event_technical_details(
                self._feature_event_state
            )
        )
        self._feature_event_details_toggle.setVisible(self._feature_event_state.present)
        self._feature_event_details_toggle.setChecked(False)

    def clear(self):
        """Reset to idle/placeholder state."""
        self._current_run_dir = ""
        self._run_summary_path = ""
        self._continuous_overview = None
        self._continuous_selected_roi = ""
        self._native_continuous_artifact_index = None
        self._native_continuous_artifacts_by_roi = {}
        self._native_continuous_artifacts_by_tab = {}
        self._native_continuous_artifact_by_key = {}
        self._native_continuous_context = {}
        self._native_continuous_mode = False
        self._continuous_overview_label.setText("")
        self._continuous_roi_row.setVisible(False)
        self._continuous_roi_combo.blockSignals(True)
        self._continuous_roi_combo.clear()
        self._continuous_roi_combo.blockSignals(False)
        self._continuous_tabs.setVisible(False)
        self._clear_continuous_tonic_display()
        self._clear_continuous_phasic_display()
        self._continuous_workspace.hide()
        self._applied_dff_state = GuidedCompletedAppliedDffState.absent()
        self._refresh_applied_dff_display()
        self._feature_event_state = GuidedCompletedFeatureEventState.absent()
        self._refresh_feature_event_display()
        self._phasic_review_model = None
        self._completed_review_overview = {}
        self._phasic_review_error = ""
        self._correction_summary_label.setText("")
        self._correction_summary_label.setVisible(False)
        self._selected_feature_settings_label.setText("")
        self._selected_feature_settings_label.setVisible(False)
        self._tonic_settings_summary_label.setText("")
        self._tonic_settings_summary_label.setVisible(False)
        self._tonic_method_note_label.setText("")
        self._tonic_method_note_label.setVisible(False)
        self._region_paths = {}
        self._region_tab_images = {}
        self._tonic_method_by_roi = {}
        self._tab_indices = {}
        self._external_tab_image_overrides = {}
        self._active_image_path = ""
        self._active_pixmap = QPixmap()
        self._set_zoom_mode(False)
        self._artifact_metadata_label.setText("")
        self._artifact_metadata_label.setVisible(False)
        self._artifact_table.setRowCount(0)
        self._artifact_table.setColumnCount(0)
        self._artifact_table_scroll.setVisible(False)
        self._image_scroll.setVisible(True)

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
        self._refresh_inline_actions()
        self._workspace.hide()

    def set_running_message(self, message: str):
        """Show a minimal running-state message in the Results pane."""
        self.clear()
        self._set_status_message(message, level="running")

    def load_report(
        self,
        out_dir: str,
        *,
        review_overview: dict | None = None,
    ) -> bool:
        """Load complete-state workspace from a run directory."""
        self.clear()
        self._current_run_dir = out_dir

        # The Guided worker supplies the prepared native index. Honor it
        # directly so the GUI does not repeat terminal classification,
        # manifest parsing, or image validation. Direct callers that provide
        # only the marker retain the bounded synchronous fallback below.
        if (
            isinstance(review_overview, dict)
            and review_overview.get("native_saved_artifacts")
        ):
            prepared_index = review_overview.get("artifact_index")
            if isinstance(prepared_index, dict):
                return self._install_native_continuous_artifact_index(
                    out_dir, prepared_index
                )
            return self._load_native_continuous_saved_artifacts(
                out_dir
            )
        routing_classification = None
        if review_overview is None:
            routing_classification = classify_completed_run_terminal_state(out_dir)
            routing_run_mode = (
                routing_classification.run_mode
                or declared_completed_run_mode(out_dir)
                or {}
            )
            if is_guided_continuous_saved_artifact_run_mode(routing_run_mode):
                return self._load_native_continuous_saved_artifacts(
                    out_dir, classification=routing_classification
                )
            if routing_classification.is_success and is_continuous_rwd_run_mode(
                routing_run_mode
            ):
                return self.load_continuous_results(out_dir)

        run_report_path = os.path.join(out_dir, "run_report.json")
        data, parse_err = parse_run_report(run_report_path)
        run_type = get_run_type(data) if data else "full"
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

        # Which tonic method each ROI actually used. Empty for a run that
        # predates the session-level tonic summary; that run then simply has no
        # tonic plot to discover and needs no migration handling.
        try:
            self._tonic_method_by_roi = tonic_method_by_roi(out_dir)
        except Exception:
            self._tonic_method_by_roi = {}

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

        self._phasic_review_model = None
        self._completed_review_overview = dict(review_overview or {})
        self._refresh_tonic_settings_summary()
        self._phasic_review_error = ""
        if self._completed_review_overview:
            classification = None
            enabled_branches = tuple(
                str(branch)
                for branch in self._completed_review_overview.get(
                    "analysis_branches", ()
                )
            )
        else:
            self._applied_dff_state = load_guided_completed_applied_dff_state(out_dir)
            self._refresh_applied_dff_display()
            self._feature_event_state = load_guided_completed_feature_event_state(out_dir)
            self._refresh_feature_event_display()
            classification = classify_completed_run_terminal_state(out_dir)
            run_mode = getattr(classification, "run_mode", {}) or {}
            enabled_branches = tuple(
                branch
                for branch, key in (
                    ("tonic", "tonic_analysis"),
                    ("phasic", "phasic_analysis"),
                )
                if bool(run_mode.get(key))
            )
        branch_evidence = {
            branch: (
                os.path.join(out_dir, "_analysis", f"{branch}_out", "run_metadata.json"),
                os.path.join(out_dir, "_analysis", f"{branch}_out", "run_report.json"),
                os.path.join(out_dir, "_analysis", f"{branch}_out", f"{branch}_trace_cache.h5"),
            )
            for branch in ("tonic", "phasic")
        }
        historical_caches = tuple(
            branch
            for branch, paths in branch_evidence.items()
            if os.path.isfile(paths[2])
        )
        if classification is not None and enabled_branches and not classification.is_success:
            self._phasic_review_error = str(classification.reason)
            self._set_status_message(
                f"This completed result cannot be verified: {classification.reason}",
                level="error",
            )
            self._workspace.hide()
            return False
        if classification is not None and classification.is_current and enabled_branches:
            for branch in enabled_branches:
                missing = [path for path in branch_evidence[branch] if not os.path.isfile(path)]
                if missing:
                    self._phasic_review_error = (
                        f"The current {branch} result is missing persisted Review evidence."
                    )
                    self._set_status_message(
                        f"This completed result cannot be verified: {self._phasic_review_error}",
                        level="error",
                    )
                    self._workspace.hide()
                    return False
        should_load_review = (
            bool(enabled_branches)
            if classification is None or classification.is_current
            else bool(historical_caches)
        )
        if should_load_review and not self._completed_review_overview:
            try:
                self._phasic_review_model = load_completed_phasic_review(out_dir)
            except CompletedRunReviewError as exc:
                self._phasic_review_error = str(exc)
                self._set_status_message(
                    f"This completed result cannot be verified: {exc}",
                    level="error",
                )
                self._correction_summary_label.setVisible(False)
                self._workspace.hide()
                return False
        if (
            self._phasic_review_model is not None
            and self._phasic_review_model.current_native
        ) or self._completed_review_overview:
            # Native current-run canonical dF/F is authoritative.  The older
            # Guided post-hoc applied-dF/F tree remains available to its
            # standalone tools, but must not appear as a competing completed
            # Review result.
            self._applied_dff_state = GuidedCompletedAppliedDffState.absent()
            self._refresh_applied_dff_display()
        self._refresh_correction_summary()
        self._refresh_tonic_method_note()
        title = "Results workspace"
        if self._completed_review_overview:
            # Use the scientist-facing format name (CSV files, RWD, NPM), not
            # the upper-cased internal token.
            raw_format = str(
                self._completed_review_overview.get("format") or ""
            ).strip()
            overview_format = (
                format_display_label(raw_format) if raw_format else "analysis"
            )
            overview_rois = self._completed_review_overview.get(
                "included_rois", ()
            )
            title = (
                f"Results workspace — completed {overview_format} analysis "
                f"({len(overview_rois)} included ROI(s))"
            )
        if is_preview:
            title += " [PREVIEW]"
        elif run_type == "tuning_prep":
            title += " [TUNING PREP]"
        if (
            self._completed_review_overview.get("review_status")
            == "reviewable_with_warning"
        ):
            warning_title = str(
                self._completed_review_overview.get("validation_warning_title")
                or "Analysis completed with a validation warning"
            )
            warning_message = str(
                self._completed_review_overview.get("validation_warning_message") or ""
            )
            self._set_status_message(
                f"{warning_title}\n{warning_message}" if warning_message else warning_title,
                level="warning",
            )
        elif classification is not None and classification.completed_with_missing:
            completion_summary = get_scientist_completion_summary(out_dir, classification)
            self._set_status_message(completion_summary, level="ready")
        else:
            self._set_status_message(title, level="ready")
        self._workspace.show()

        region_names = sorted(self._region_paths.keys(), key=lambda s: s.lower())
        self._region_combo.blockSignals(True)
        self._region_combo.clear()
        self._region_combo.addItems(region_names)
        self._region_combo.blockSignals(False)

        if self._region_combo.count() > 0 and not self._completed_review_overview:
            self._region_combo.setCurrentIndex(0)
            self._on_region_changed(0)
        elif self._completed_review_overview:
            self._region_combo.blockSignals(True)
            self._region_combo.insertItem(0, "Choose an ROI to inspect")
            self._region_combo.setCurrentIndex(0)
            self._region_combo.blockSignals(False)
            self._show_no_image(
                "Choose an ROI to inspect its available review images."
            )
            requested = self._completed_review_overview.get(
                "requested_by_roi", {}
            )
            self._correction_summary_label.setText(
                "Correction settings are available for "
                f"{len(requested)} included ROI(s). Choose an ROI for details."
            )
            self._correction_summary_label.setVisible(True)
            features = self._completed_review_overview.get(
                "feature_settings_by_roi", {}
            )
            self._selected_feature_settings_label.setText(
                "Feature detection settings are available for "
                f"{len(features)} included ROI(s)."
            )
            self._selected_feature_settings_label.setVisible(True)
        else:
            self._show_no_image("No region available.")
            self._refresh_inline_actions()

        return True

    def _load_native_continuous_saved_artifacts(
        self, run_dir: str, *, classification=None
    ) -> bool:
        """Load the I3 native Guided continuous package into this viewer.

        Membership, ROI order, paths, and saved-image provenance come from the
        completion manifest.  This branch intentionally has no cache reader,
        trace loader, coordinate mapper, detector, or plotting call.
        """
        try:
            artifact_index = build_guided_continuous_saved_artifact_index(
                run_dir, classification=classification
            )
        except SavedArtifactIndexError as exc:
            self._set_status_message(str(exc), level="error")
            self._workspace.hide()
            self._continuous_workspace.hide()
            return False

        return self._install_native_continuous_artifact_index(
            run_dir, artifact_index
        )

    def install_prepared_native_continuous_artifacts(
        self, run_dir: str, artifact_index: Dict[str, Any]
    ) -> bool:
        """Install the already-validated native package from the Guided worker."""
        self.clear()
        return self._install_native_continuous_artifact_index(
            run_dir, artifact_index
        )

    def _install_native_continuous_artifact_index(
        self, run_dir: str, artifact_index: Dict[str, Any]
    ) -> bool:
        """Install a prepared index without rereading or revalidating files."""
        resolved_run_dir = os.path.realpath(str(run_dir))
        indexed_run_dir = os.path.realpath(
            str(artifact_index.get("run_dir") or "")
        )
        if not indexed_run_dir or indexed_run_dir != resolved_run_dir:
            self._set_status_message(
                "The prepared completed-analysis package belongs to a different run.",
                level="error",
            )
            self._workspace.hide()
            self._continuous_workspace.hide()
            return False
        try:
            roi_order = [str(roi) for roi in artifact_index.get("roi_order", ())]
            all_artifacts = list(artifact_index.get("artifacts") or [])
            if not roi_order or not all_artifacts:
                raise ValueError("the prepared artifact index is empty")
        except (AttributeError, TypeError, ValueError) as exc:
            self._set_status_message(
                f"The prepared completed-analysis package is invalid ({exc}).",
                level="error",
            )
            self._workspace.hide()
            self._continuous_workspace.hide()
            return False

        self._native_continuous_mode = True
        self._native_continuous_artifact_index = artifact_index
        self._native_continuous_context = {
            "timeline": dict(artifact_index.get("timeline") or {}),
            "window_timing": dict(artifact_index.get("window_timing") or {}),
            "run_mode": dict(artifact_index.get("run_mode") or {}),
        }
        roi_order = [str(roi) for roi in artifact_index.get("roi_order", ())]
        all_artifacts = list(artifact_index.get("artifacts") or [])
        self._native_continuous_artifacts_by_roi = {roi: [] for roi in roi_order}
        self._native_continuous_artifact_by_key = {}
        self._region_paths = {
            roi: os.path.join(indexed_run_dir, roi)
            for roi in roi_order
        }
        self._region_tab_images = {roi: {} for roi in roi_order}

        run_level_artifacts = [
            record for record in all_artifacts if record.get("scope") == "run"
        ]
        self._native_continuous_artifacts_by_tab = {}
        for roi in roi_order:
            records = [
                record for record in all_artifacts if record.get("roi") == roi
            ]
            # The run-level event table is available from every ROI selection,
            # but the manifest-backed package itself retains one record only.
            records.extend(run_level_artifacts)
            records.sort(key=lambda record: int(record.get("order", 0)))
            self._native_continuous_artifacts_by_roi[roi] = records
            for record in records:
                label = str(record.get("label") or "")
                if label:
                    self._native_continuous_artifact_by_key[(roi, label)] = record
                    if record.get("artifact_type") == "image":
                        tab_label = self._native_continuous_tab_label(record)
                        self._native_continuous_artifacts_by_tab.setdefault(
                            (roi, tab_label), []
                        ).append(record)
                        self._region_tab_images[roi].setdefault(tab_label, []).append(
                            str(record["path"])
                        )
                    else:
                        self._region_tab_images[roi].setdefault(label, [])

        for key, records in self._native_continuous_artifacts_by_tab.items():
            if key[1] != TAB_PHASIC_SUMMARY:
                if key[1] in _NATIVE_CONTINUOUS_DAY_PLOT_FAMILY_TABS.values():
                    records.sort(
                        key=lambda record: (
                            int(record.get("day_index", 0)),
                            str(record.get("relative_path") or ""),
                        )
                    )
                continue
            records.sort(
                key=lambda record: (
                    _NATIVE_PHASIC_SUMMARY_IMAGE_ORDER[
                        os.path.basename(str(record.get("relative_path") or ""))
                    ]
                    if os.path.basename(str(record.get("relative_path") or ""))
                    in _NATIVE_PHASIC_SUMMARY_IMAGE_ORDER
                    else int(record.get("order", 0))
                )
            )

        self._current_run_dir = indexed_run_dir
        self._run_summary_path = os.path.join(self._current_run_dir, "run_report.json")
        self._set_status_message("Results workspace — continuous analysis", level="ready")
        self._workspace.show()
        self._continuous_workspace.hide()
        self._region_combo.blockSignals(True)
        self._region_combo.clear()
        self._region_combo.addItems(roi_order)
        self._region_combo.blockSignals(False)
        if self._region_combo.count() > 0:
            self._region_combo.setCurrentIndex(0)
            self._on_region_changed(0)
        else:
            self._show_no_image("No region is available in the completed analysis.")
            self._refresh_inline_actions()
        return True

    def load_continuous_results(
        self,
        run_dir: str,
        overview: ContinuousRunOverview | None = None,
    ) -> bool:
        """Load the retained legacy/Full-Control continuous presentation.

        Native Guided continuous runs are diverted to the manifest-backed
        generic artifact viewer before this retained compatibility path is
        entered.  The old trace workspace remains only for explicit
        non-native callers whose existing behavior is outside CR1-F1-I4.
        """
        self.clear()
        self._current_run_dir = run_dir

        if overview is None:
            try:
                overview = load_continuous_run_overview(run_dir)
            except CompletedContinuousRwdReviewError as exc:
                self._set_status_message(str(exc), level="error")
                return False

        self._continuous_overview = overview
        self._run_summary_path = os.path.join(overview.run_dir, "run_report.json")
        self._set_status_message(
            "Results workspace — continuous recording", level="ready"
        )
        self._populate_continuous_overview(overview)
        self._rebuild_continuous_tabs(overview)

        self._continuous_roi_combo.blockSignals(True)
        self._continuous_roi_combo.clear()
        self._continuous_roi_combo.addItems(list(overview.included_roi_ids))
        self._continuous_roi_combo.blockSignals(False)
        self._continuous_roi_row.setVisible(
            bool(overview.tonic_analysis or overview.phasic_analysis)
        )

        if self._continuous_roi_combo.count() > 0 and (
            overview.tonic_analysis or overview.phasic_analysis
        ):
            self._continuous_roi_combo.blockSignals(True)
            self._continuous_roi_combo.setCurrentIndex(0)
            self._continuous_roi_combo.blockSignals(False)
            initial_roi_id = self._continuous_roi_combo.currentText().strip()
            if not self._select_continuous_roi(overview, initial_roi_id, is_initial=True):
                # The scientist-facing CompletedContinuousRwdReviewError text
                # is already in the status label (set by
                # _select_continuous_roi); neither workspace may be shown,
                # and there is no fallback to the intermittent loader.
                return False

        self._workspace.hide()
        self._continuous_workspace.show()
        return True

    @staticmethod
    def _format_recording_duration(total_sec: float) -> str:
        """Render a total recording duration as simple scientist-facing
        text -- never only a sample/second count (see CR1-E1-B handoff
        section 14)."""
        total_sec = max(0.0, float(total_sec))
        days, remainder_sec = divmod(total_sec, 86400.0)
        hours, remainder_sec = divmod(remainder_sec, 3600.0)
        minutes, _seconds = divmod(remainder_sec, 60.0)
        days, hours, minutes = int(days), int(hours), int(minutes)
        if days > 0:
            return f"{days} d {hours} h {minutes} min"
        return f"{hours} h {minutes} min"

    def _populate_continuous_overview(self, overview: ContinuousRunOverview) -> None:
        total_duration_sec = (
            overview.final_window.end_sec if overview.final_window is not None else 0.0
        )
        lines = [
            "Continuous recording",
            "Regions of interest: " + ", ".join(overview.included_roi_ids),
            "Correction: Completed" if overview.correction_completed else "Correction: Not completed",
            "Tonic analysis: " + ("Completed" if overview.tonic_analysis else "Not run"),
            "Phasic event analysis: " + ("Completed" if overview.phasic_analysis else "Not run"),
            "Recording duration: " + self._format_recording_duration(total_duration_sec),
            "Analysis windows: " + str(overview.corrected_segment_count),
        ]
        self._continuous_overview_label.setText("\n".join(lines))

    def _rebuild_continuous_tabs(self, overview: ContinuousRunOverview) -> None:
        """Show only the Tonic/Phasic tabs for analyses that actually
        completed -- never a fabricated empty tab for an analysis that did
        not run (see CR1-E1-B handoff section 9)."""
        while self._continuous_tabs.count() > 0:
            self._continuous_tabs.removeTab(0)
        if overview.tonic_analysis:
            self._continuous_tabs.addTab(self._continuous_tonic_page, "Tonic")
        if overview.phasic_analysis:
            self._continuous_tabs.addTab(self._continuous_phasic_page, "Phasic")
        has_any = overview.tonic_analysis or overview.phasic_analysis
        self._continuous_tabs.setVisible(has_any)
        if self._continuous_tabs.count() > 0:
            self._continuous_tabs.setCurrentIndex(0)

    def _clear_continuous_tonic_display(self) -> None:
        self._continuous_tonic_interaction.clear(
            "No tonic trace loaded.", fallback_width=640, fallback_height=320
        )
        self._continuous_tonic_summary_table.setRowCount(0)
        self._continuous_tonic_summary_table.setColumnCount(0)

    def _clear_continuous_phasic_display(self) -> None:
        self._continuous_phasic_interaction.clear(
            "No phasic trace loaded.", fallback_width=640, fallback_height=320
        )
        self._continuous_phasic_summary_table.setRowCount(0)
        self._continuous_phasic_summary_table.setColumnCount(0)
        self._continuous_phasic_event_count_label.setText("")

    def _on_continuous_roi_changed(self, _index: int) -> None:
        overview = self._continuous_overview
        if overview is None:
            return
        roi_id = self._continuous_roi_combo.currentText().strip()
        if not roi_id:
            return
        self._select_continuous_roi(overview, roi_id, is_initial=False)

    def _select_continuous_roi(
        self,
        overview: ContinuousRunOverview,
        roi_id: str,
        *,
        is_initial: bool,
    ) -> bool:
        """Atomically load and display one ROI's requested continuous
        analyses: every requested family (tonic and/or phasic) is loaded and
        rendered to an in-memory pixmap/table first, and the visible display
        is only replaced once every requested load has succeeded. A single
        family failing never leaves one section showing the new ROI while
        another still shows the old one, and never leaves a partial
        trace/table/event-count visible.

        On the initial load (``is_initial=True``) there is no previously
        displayed ROI to fall back to, so failure clears every continuous
        display instead of restoring a selection. On a later ROI switch,
        failure restores the ROI selector and every display to the last
        successfully shown ROI. Returns whether ``roi_id`` is now fully and
        successfully displayed.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            tonic_data, phasic_data = self._load_continuous_roi_data(overview, roi_id)
        except CompletedContinuousRwdReviewError as exc:
            QApplication.restoreOverrideCursor()
            self._set_status_message(str(exc), level="error")
            if is_initial or not self._continuous_selected_roi:
                self._clear_continuous_tonic_display()
                self._clear_continuous_phasic_display()
            else:
                self._restore_continuous_roi_selector(self._continuous_selected_roi)
            return False
        QApplication.restoreOverrideCursor()

        self._apply_continuous_roi_data(roi_id, tonic_data, phasic_data)
        self._continuous_selected_roi = roi_id
        return True

    def _restore_continuous_roi_selector(self, roi_id: str) -> None:
        """Restore the ROI selector to ``roi_id`` without re-triggering
        another load attempt (the last successful display for that ROI is
        already showing and must not be touched)."""
        if not roi_id:
            return
        idx = self._continuous_roi_combo.findText(roi_id)
        if idx < 0:
            return
        self._continuous_roi_combo.blockSignals(True)
        self._continuous_roi_combo.setCurrentIndex(idx)
        self._continuous_roi_combo.blockSignals(False)

    def _load_continuous_roi_data(
        self, overview: ContinuousRunOverview, roi_id: str
    ):
        """Load and render every analysis family requested for ``roi_id``
        without touching any visible widget. Raises
        ``CompletedContinuousRwdReviewError`` on the first failure, leaving
        no widget mutated."""
        tonic_data = None
        phasic_data = None
        if overview.tonic_analysis:
            trace = load_continuous_roi_trace(
                overview.run_dir, family="tonic", roi_id=roi_id
            )
            pixmap = self._render_continuous_trace_pixmap(
                trace.time_sec,
                trace.primary_trace,
                title=f"{roi_id} — {trace.primary_trace_label}",
                y_label=trace.primary_trace_label,
            )
            summary = load_continuous_window_summary(
                overview.run_dir, family="tonic", roi_id=roi_id
            )
            tonic_data = (pixmap, summary)
        if overview.phasic_analysis:
            trace = load_continuous_roi_trace(
                overview.run_dir, family="phasic", roi_id=roi_id
            )
            events = load_continuous_phasic_events(overview.run_dir, roi_id=roi_id)
            event_times = (
                events["global_time_sec"].to_numpy(dtype=float) if len(events) else None
            )
            event_polarities = (
                events["polarity"].to_numpy(dtype=float) if len(events) else None
            )
            pixmap = self._render_continuous_trace_pixmap(
                trace.time_sec,
                trace.primary_trace,
                title=f"{roi_id} — {trace.primary_trace_label}",
                y_label=trace.primary_trace_label,
                event_times=event_times,
                event_polarities=event_polarities,
            )
            persisted_total = overview.phasic_event_counts_by_roi.get(roi_id, len(events))
            summary = load_continuous_window_summary(
                overview.run_dir, family="phasic", roi_id=roi_id
            )
            phasic_data = (pixmap, persisted_total, summary)
        return tonic_data, phasic_data

    def _apply_continuous_roi_data(
        self,
        roi_id: str,
        tonic_data,
        phasic_data,
    ) -> None:
        """Replace visible continuous displays with already-successfully-
        loaded data. Never called unless every requested family for
        ``roi_id`` has already loaded without error."""
        if tonic_data is not None:
            pixmap, summary = tonic_data
            self._continuous_tonic_interaction.set_pixmap(pixmap, reset_zoom=True)
            self._populate_summary_table(self._continuous_tonic_summary_table, summary)
        if phasic_data is not None:
            pixmap, persisted_total, summary = phasic_data
            self._continuous_phasic_interaction.set_pixmap(pixmap, reset_zoom=True)
            self._continuous_phasic_event_count_label.setText(
                f"Saved phasic events for {roi_id}: {persisted_total}"
            )
            self._populate_summary_table(self._continuous_phasic_summary_table, summary)

    @staticmethod
    def _render_continuous_trace_pixmap(
        time_sec,
        primary_trace,
        *,
        title: str,
        y_label: str,
        event_times=None,
        event_polarities=None,
    ) -> QPixmap:
        """Render one selected ROI's full continuous trace to an in-memory
        PNG for display -- never persisted to disk, never a new plotting
        engine (matplotlib's non-interactive Agg backend is already used
        elsewhere in this GUI for on-the-fly preview rendering). Display-only
        decimation bounds the plotted line to
        ``CONTINUOUS_TRACE_OVERVIEW_MAX_POINTS`` samples; saved event markers
        are always drawn in full and are never counted from what is drawn
        (see CR1-E1-B handoff sections 10-12)."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        n_points = len(time_sec)
        stride = max(1, math.ceil(n_points / CONTINUOUS_TRACE_OVERVIEW_MAX_POINTS))
        plot_time = time_sec[::stride]
        plot_trace = primary_trace[::stride]

        with matplotlib.rc_context({"figure.dpi": 90}):
            figure, axis = plt.subplots(figsize=(9.5, 4.2))
            try:
                axis.plot(plot_time, plot_trace, linewidth=0.6, color="#3b6ea5")
                if event_times is not None and len(event_times):
                    positive = event_polarities > 0 if event_polarities is not None else None
                    if positive is not None:
                        axis.scatter(
                            event_times[positive],
                            [axis.get_ylim()[1]] * int(positive.sum()),
                            marker="v",
                            s=10,
                            color="#c0392b",
                            label="Positive event",
                        )
                        negative = ~positive
                        axis.scatter(
                            event_times[negative],
                            [axis.get_ylim()[0]] * int(negative.sum()),
                            marker="^",
                            s=10,
                            color="#2e7d32",
                            label="Negative event",
                        )
                        axis.legend(loc="upper right", fontsize="small")
                    else:
                        axis.scatter(
                            event_times,
                            [axis.get_ylim()[1]] * len(event_times),
                            marker="v",
                            s=10,
                            color="#c0392b",
                        )
                axis.set_title(title)
                axis.set_xlabel("Recording time (seconds)")
                axis.set_ylabel(y_label)
                figure.tight_layout()

                buffer = io.BytesIO()
                figure.savefig(buffer, format="png")
                buffer.seek(0)
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.getvalue(), "PNG")
                return pixmap
            finally:
                plt.close(figure)

    @staticmethod
    def _populate_summary_table(table: QTableWidget, summary) -> None:
        """Display an already-persisted per-window summary table verbatim --
        never recomputed. The storage-window partition column is relabeled
        for the scientist-facing header only; the underlying data is
        unchanged (see CR1-E1-B handoff section 13)."""
        columns = list(summary.columns)
        table.setColumnCount(len(columns))
        header_labels = [
            "Analysis window" if col == "window_index" else col for col in columns
        ]
        table.setHorizontalHeaderLabels(header_labels)
        table.setRowCount(len(summary))
        for row_idx, row in enumerate(summary.itertuples(index=False)):
            for col_idx, value in enumerate(row):
                table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

    @property
    def applied_dff_state(self) -> GuidedCompletedAppliedDffState:
        """Return the loaded completed run applied dF/F state."""
        return self._applied_dff_state

    @property
    def applied_dff_summary_text(self) -> str:
        """Return the read-only completed-run applied dF/F summary text."""
        return self._applied_dff_summary_label.text()

    @property
    def applied_dff_technical_details_text(self) -> str:
        """Return the optional-disclosure applied dF/F technical details text."""
        return self._applied_dff_details_label.text()

    @property
    def feature_event_state(self) -> GuidedCompletedFeatureEventState:
        """Return the loaded completed-run per-ROI feature-detection state."""
        return self._feature_event_state

    @property
    def feature_event_summary_text(self) -> str:
        """Return the read-only completed-run feature-detection summary text."""
        return self._feature_event_summary_label.text()

    @property
    def feature_event_technical_details_text(self) -> str:
        """Return the optional-disclosure feature-detection technical details text."""
        return self._feature_event_details_label.text()

    def _discover_region_tab_images(self, region_path: str) -> Dict[str, List[str]]:
        """Discover per-tab image lists for one region directory."""
        summary_dir = os.path.join(region_path, "summary")
        day_plots_dir = os.path.join(region_path, "day_plots")

        verification = self._discover_verification_images(summary_dir)
        tonic = self._discover_tonic_images(summary_dir)
        continuous_trace = self._discover_continuous_trace_images(summary_dir)
        phasic_raw = self._discover_day_series_images(
            day_plots_dir,
            r"^phasic_sig_iso_day_\d{3,}\.png$",
            ignore_case=False,
        )
        phasic_dynamic_fit = self._discover_day_series_images(
            day_plots_dir,
            r"^phasic_dynamic_fit_day_\d{3,}\.png$",
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
            TAB_PHASIC_DYNAMIC_FIT: phasic_dynamic_fit,
            TAB_PHASIC_DFF: phasic_dff,
            TAB_PHASIC_STACKED: phasic_stacked,
            TAB_PHASIC_SUMMARY: phasic_summary,
            TAB_CONTINUOUS_TRACE: continuous_trace,
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

        Anchored on the session-level tonic plot (one point per session, in the
        ROI's own tonic method and units). tonic_overview.png is deliberately
        NOT discovered: it renders the full phasic-correction residual trace,
        which is not a tonic result. That file is still produced because the
        completion contract requires it, but no scientist-facing tab shows it.
        """
        if not os.path.isdir(summary_dir):
            return []
        candidates = [
            os.path.join(summary_dir, TONIC_SESSION_PLOT_FILENAME),
        ]
        for name in os.listdir(summary_dir):
            if name.startswith("tonic_session_summary_") and name.endswith(".png"):
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

    def _discover_continuous_trace_images(self, summary_dir: str) -> List[str]:
        """
        Continuous Trace tab contract.
        Full elapsed trace overviews are separate from per-window summary plots.
        """
        if not os.path.isdir(summary_dir):
            return []
        candidates = [
            os.path.join(summary_dir, "continuous_tonic_trace_overview.png"),
            os.path.join(summary_dir, "continuous_phasic_dff_trace_overview.png"),
        ]
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

    @staticmethod
    def _day_plot_caption(filename: str) -> str:
        """Return a one-based plotted-day caption when the filename supports it."""
        name = os.path.basename(str(filename or ""))
        match = re.search(r"_day_(\d+)\.png$", name, re.IGNORECASE)
        if match is None:
            return name
        return f"Plotted day {int(match.group(1)) + 1}"

    def _on_region_changed(self, _index: int):
        """Refresh tabs, viewer, and actions for selected region."""
        self._rebuild_tabs_for_selected_region()
        self._refresh_correction_summary()
        self._refresh_tonic_method_note()
        self._refresh_inline_actions()
        self.region_changed.emit(self._selected_region())

    def _refresh_tonic_method_note(self) -> None:
        """Explain the signal-only tonic fallback for the selected ROI only.

        Shown once per affected ROI, never per session, and never for an ROI
        whose tonic used the primary global-isosbestic method.
        """
        record = self._tonic_method_by_roi.get(self._selected_region() or "", {})
        if str(record.get("tonic_method", "")) != METHOD_SIGNAL_ONLY:
            self._tonic_method_note_label.setText("")
            self._tonic_method_note_label.setVisible(False)
            return
        text = TONIC_FALLBACK_NOTE
        reason = str(record.get("fallback_reason", "") or "")
        if reason:
            text += f" (reason: {reason})"
        self._tonic_method_note_label.setText(text)
        self._tonic_method_note_label.setVisible(True)

    def _refresh_tonic_settings_summary(self) -> None:
        """Show the consumed tonic-settings summary whenever the compact
        Review overview carries valid evidence for it.

        Deliberately independent of ROI selection, feature settings, and
        feature extraction: this only reads
        self._completed_review_overview["tonic_settings"] (already sourced
        from the run's own consumed configuration, never the live GUI
        selection -- see completed_run_review.load_completed_review_overview).
        Called once per load_report(), before any ROI is chosen, so it is
        unaffected by later ROI/image changes handled elsewhere
        (_on_region_changed / _refresh_correction_summary never touch this
        label).
        """
        tonic_settings = self._completed_review_overview.get("tonic_settings", {})
        tonic_summary_text = format_tonic_settings_summary(tonic_settings)
        if tonic_summary_text:
            self._tonic_settings_summary_label.setText(tonic_summary_text)
            self._tonic_settings_summary_label.setVisible(True)
        else:
            self._tonic_settings_summary_label.setText("")
            self._tonic_settings_summary_label.setVisible(False)

    def _refresh_correction_summary(self) -> None:
        model = self._phasic_review_model
        roi = self._selected_region()
        if model is None and self._completed_review_overview and roi:
            requested = self._completed_review_overview.get(
                "requested_by_roi", {}
            )
            features = self._completed_review_overview.get(
                "feature_settings_by_roi", {}
            )
            record = requested.get(roi, {}) if isinstance(requested, dict) else {}
            selected = str(
                record.get("dynamic_fit_mode")
                or record.get("selected_strategy")
                or ""
            )
            labels = {
                "global_linear_regression": "Global linear regression",
                "robust_global_event_reject": (
                    "Robust global fit with event rejection"
                ),
                "adaptive_event_gated_regression": (
                    "Adaptive event-gated regression"
                ),
                "signal_only_f0": "Signal-Only F0",
            }
            self._correction_summary_label.setText(
                f"Correction approach for {roi}: "
                f"{labels.get(selected, selected or 'available')}."
            )
            self._correction_summary_label.setVisible(True)
            row = features.get(roi, {}) if isinstance(features, dict) else {}
            fields = row.get("effective_config_fields", {}) if isinstance(row, dict) else {}
            method = str(fields.get("peak_threshold_method") or "")
            source = str(row.get("source") or "default").lower()
            source_label = "Custom" if source == "override" else "Default"
            summary = f"{source_label} feature settings"
            if method:
                summary += (
                    f": {feature_threshold_method_display_label(method)} threshold"
                )
            signal = str(fields.get("event_signal") or "").strip().lower()
            if signal:
                summary += f"; event signal {feature_event_signal_display_label(signal)}"
            self._selected_feature_settings_label.setText(summary + ".")
            self._selected_feature_settings_label.setVisible(True)
            return
        if model is None or not roi:
            self._correction_summary_label.setText("")
            self._correction_summary_label.setVisible(False)
            self._selected_feature_settings_label.setText("")
            self._selected_feature_settings_label.setVisible(False)
            return
        label = model.strategy_label_for_roi(roi)
        if model.heterogeneous_correction:
            text = f"Correction approaches vary by ROI.  {roi}: {label}."
        else:
            text = f"Correction approach: {label}."
        if set(model.analysis_branches) == {"tonic", "phasic"}:
            text += " Used for tonic and phasic analyses."
        elif model.analysis_branches == ("tonic",):
            text += " Used for tonic analysis."
        sessions = model.sessions_for_roi(roi)
        processed_count = sum(1 for session in sessions if session.processed)
        absent_count = max(0, len(sessions) - processed_count)
        if sessions:
            text += f" {processed_count} session(s) processed"
            if absent_count:
                text += f"; {absent_count} absent"
            text += "."
        if label == "Signal-Only F0":
            qc = model.signal_only_qc_for_roi(roi)
            viability = qc.get("signal_only_f0_candidate_viability")
            confidence = qc.get("signal_only_f0_candidate_confidence")
            warning = qc.get("signal_only_f0_warning")
            if viability:
                text += f" Baseline support: {viability}."
            elif confidence:
                text += f" Baseline confidence: {confidence}."
            elif warning:
                text += f" Baseline note: {warning}."
            else:
                text += " Signal-only baseline evidence recorded."
        self._correction_summary_label.setText(text)
        self._correction_summary_label.setVisible(True)
        self._selected_feature_settings_label.setText(
            model.feature_settings_summary_for_roi(roi)
        )
        self._selected_feature_settings_label.setVisible(True)

    def _on_tab_changed(self, _index: int):
        """Refresh image viewer when tab changes."""
        self._refresh_active_image(reset_index=True)
        self._refresh_inline_actions()

    def _selected_region(self) -> str:
        return self._region_combo.currentText().strip()

    def selected_region(self) -> str:
        """Public selected-region getter for parent workspace integrations."""
        return self._selected_region()

    @property
    def phasic_review_model(self) -> CompletedRunReviewModel | None:
        """Return the persisted strategy-aware phasic Review model, if loaded."""
        return self._phasic_review_model

    @property
    def phasic_review_error(self) -> str:
        return str(self._phasic_review_error or "")

    def active_image_path(self) -> str:
        """Public getter for the currently displayed image path."""
        return str(self._active_image_path or "")

    def show_external_image(self, path: str, *, prefer_tab: str = TAB_PHASIC_DFF) -> bool:
        """
        Display an explicit image path in the viewer without mutating report artifacts.

        Used for optional display-variant previews (e.g., rerendered dF/F figures)
        that live outside the default tab discovery contract.
        """
        if not path or not os.path.isfile(path):
            return False

        return self.show_external_image_sequence(
            [path],
            initial_path=path,
            prefer_tab=prefer_tab,
        )

    def show_external_image_sequence(
        self,
        paths: List[str],
        *,
        initial_path: str = "",
        prefer_tab: str = TAB_PHASIC_DFF,
    ) -> bool:
        """
        Display a temporary external image sequence inside the existing tab/region
        navigation model.

        This preserves prev/next browsing behavior while keeping the underlying
        discovered artifact lists untouched.
        """
        cleaned = self._dedupe_sorted_existing(list(paths or []))
        if not cleaned:
            return False

        # Keep tab framing coherent by preferring the dF/F tab when present.
        if prefer_tab:
            for idx in range(self._tabs.count()):
                if self._tabs.tabText(idx).strip() == prefer_tab:
                    self._tabs.setCurrentIndex(idx)
                    break

        key = self._tab_key()
        if not key[0] or not key[1]:
            return False

        self._external_tab_image_overrides[key] = cleaned

        initial_idx = 0
        normalized = {os.path.normcase(os.path.normpath(p)): i for i, p in enumerate(cleaned)}
        if initial_path:
            idx = normalized.get(os.path.normcase(os.path.normpath(initial_path)))
            if idx is not None:
                initial_idx = int(idx)
        if initial_idx == 0 and self._active_image_path:
            idx = normalized.get(os.path.normcase(os.path.normpath(self._active_image_path)))
            if idx is not None:
                initial_idx = int(idx)

        self._tab_indices[key] = initial_idx
        self._refresh_active_image(reset_index=False)
        return True

    def available_regions(self) -> List[str]:
        """Public region list for parent workspace integrations."""
        if self._native_continuous_mode and self._native_continuous_artifact_index:
            return [
                str(roi)
                for roi in self._native_continuous_artifact_index.get("roi_order", ())
            ]
        return sorted(self._region_paths.keys(), key=lambda s: s.lower())

    def has_loaded_results(self) -> bool:
        """Return True when a completed run workspace is currently loaded."""
        return bool(self._region_paths)

    def available_artifacts(self, roi: str | None = None) -> List[Dict[str, Any]]:
        """Return the current native saved-artifact records for inspection/tests."""
        if not self._native_continuous_mode:
            return []
        selected = str(roi or self._selected_region()).strip()
        return [
            dict(record)
            for record in self._native_continuous_artifacts_by_roi.get(selected, [])
        ]

    def active_artifact_path(self) -> str:
        """Return the selected saved image or table path, if any."""
        if not self._native_continuous_mode:
            return self.active_image_path()
        region = self._selected_region()
        label = self._selected_tab()
        records = self._native_continuous_artifacts_by_tab.get((region, label), [])
        if not records:
            return ""
        idx = max(0, min(self._tab_indices.get((region, label), 0), len(records) - 1))
        return str(records[idx].get("path") or "")

    def available_view_tabs(self) -> List[str]:
        """Return currently visible tab labels for the selected region."""
        labels: List[str] = []
        for i in range(self._tabs.count()):
            text = self._tabs.tabText(i).strip()
            if text:
                labels.append(text)
        return labels

    def _selected_tab(self) -> str:
        idx = self._tabs.currentIndex()
        if idx < 0:
            return ""
        return self._tabs.tabText(idx).strip()

    def _rebuild_tabs_for_selected_region(self):
        region = self._selected_region()
        if self._native_continuous_mode:
            self._rebuild_native_tabs_for_selected_region(region)
            return
        tab_map = self._region_tab_images.get(region, {})
        available_tabs = [t for t in TAB_ORDER if tab_map.get(t)]
        model = self._phasic_review_model
        overview_requested = self._completed_review_overview.get(
            "requested_by_roi", {}
        )
        overview_record = (
            overview_requested.get(region, {})
            if isinstance(overview_requested, dict)
            else {}
        )
        signal_only = bool(
            model is not None
            and model.strategy_label_for_roi(region) == "Signal-Only F0"
        ) or str(overview_record.get("strategy_family") or "") == "signal_only_f0"
        if signal_only:
            if TAB_PHASIC_DYNAMIC_FIT in available_tabs:
                dynamic_index = available_tabs.index(TAB_PHASIC_DYNAMIC_FIT)
                available_tabs[dynamic_index] = TAB_PHASIC_CORRECTION_REFERENCE

        current_tab = self._selected_tab()
        self._tabs.blockSignals(True)
        while self._tabs.count() > 0:
            self._tabs.removeTab(0)
        for tab_name in available_tabs:
            tab_index = self._tabs.addTab(QWidget(), tab_name)
            tooltip = _RESULT_TAB_TOOLTIPS.get(tab_name)
            if tooltip:
                self._tabs.setTabToolTip(tab_index, tooltip)
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

    @staticmethod
    def _native_continuous_tab_label(record: Dict[str, Any]) -> str:
        """Return the scientist-facing tab for one saved native image."""
        family_tab = _NATIVE_CONTINUOUS_DAY_PLOT_FAMILY_TABS.get(
            str(record.get("family") or "")
        )
        if family_tab:
            return family_tab
        filename = os.path.basename(str(record.get("relative_path") or ""))
        tab_label = _NATIVE_CONTINUOUS_IMAGE_TABS.get(filename)
        if tab_label:
            return tab_label
        return str(record.get("label") or "")

    def _rebuild_native_tabs_for_selected_region(self, region: str) -> None:
        """Build the native scientist-facing tabs from saved image records."""
        labels = [
            label
            for label in _NATIVE_CONTINUOUS_TAB_ORDER
            if self._native_continuous_artifacts_by_tab.get((region, label))
        ]
        current_tab = self._selected_tab()
        self._tabs.blockSignals(True)
        while self._tabs.count() > 0:
            self._tabs.removeTab(0)
        for label in labels:
            tab_index = self._tabs.addTab(QWidget(), label)
            tooltip = _RESULT_TAB_TOOLTIPS.get(label)
            if tooltip:
                self._tabs.setTabToolTip(tab_index, tooltip)
        self._tabs.blockSignals(False)
        if not labels:
            self._show_no_image("No saved artifacts are available for the selected region.")
            return
        idx = labels.index(current_tab) if current_tab in labels else 0
        self._tabs.setCurrentIndex(idx)
        self._refresh_active_image(reset_index=True)

    def _current_tab_images(self) -> List[str]:
        key = self._tab_key()
        region, tab = key
        if not region or not tab:
            return []
        if self._native_continuous_mode:
            return [
                str(record.get("path") or "")
                for record in self._native_continuous_artifacts_by_tab.get(
                    (region, tab), []
                )
            ]
        override = self._external_tab_image_overrides.get(key, [])
        if override:
            return list(override)
        tab_map = self._region_tab_images.get(region, {})
        if tab == TAB_PHASIC_CORRECTION_REFERENCE:
            tab = TAB_PHASIC_DYNAMIC_FIT
        return list(tab_map.get(tab, []))

    def _tab_key(self) -> Tuple[str, str]:
        return (self._selected_region(), self._selected_tab())

    def _refresh_active_image(self, reset_index: bool):
        if self._native_continuous_mode:
            if reset_index:
                self._tab_indices[self._tab_key()] = 0
            self._refresh_native_artifact()
            return
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
        filename = os.path.basename(path)
        self._image_title_label.setText(self._day_plot_caption(filename))
        self._image_title_label.setToolTip(filename)
        self._set_zoom_mode(False)
        self._set_image(path)

        n = len(images)
        multi = n > 1
        self._prev_btn.setVisible(multi)
        self._next_btn.setVisible(multi)
        self._image_counter_label.setText(f"{idx + 1}/{n}" if multi else "")

    def _refresh_native_artifact(self) -> None:
        region = self._selected_region()
        label = self._selected_tab()
        key = (region, label)
        records = self._native_continuous_artifacts_by_tab.get(key, [])
        if not records:
            self._show_no_image("No saved artifact is available for this selection.")
            return

        idx = max(0, min(self._tab_indices.get(key, 0), len(records) - 1))
        self._tab_indices[key] = idx
        record = records[idx]
        self._set_native_artifact_metadata(record, region, tab=label)
        multi = len(records) > 1
        self._prev_btn.setVisible(multi)
        self._next_btn.setVisible(multi)
        self._image_counter_label.setText(f"{idx + 1}/{len(records)}" if multi else "")
        artifact_type = str(record.get("artifact_type") or "")
        path = str(record.get("path") or "")
        if artifact_type == "image":
            self._artifact_table_scroll.setVisible(False)
            self._image_scroll.setVisible(True)
            self._active_image_path = path
            self._active_pixmap = QPixmap()
            title = (
                str(record.get("label") or "Saved figure")
                if label == TAB_PHASIC_SUMMARY
                else label
            )
            self._image_title_label.setText(title)
            self._image_title_label.setToolTip(os.path.basename(path))
            self._set_zoom_mode(False)
            self._set_image(path)
            return

        if artifact_type == "table":
            self._active_image_path = ""
            self._active_pixmap = QPixmap()
            self._set_zoom_mode(False)
            self._image_scroll.setVisible(False)
            self._artifact_table_scroll.setVisible(True)
            self._image_title_label.setText(str(record.get("label") or "Saved table"))
            self._image_title_label.setToolTip(os.path.basename(path))
            try:
                rows, total_rows = self._read_saved_csv_table(path)
            except (OSError, csv.Error, ValueError) as exc:
                self._artifact_table.setRowCount(0)
                self._artifact_table.setColumnCount(0)
                self._set_status_message(
                    f"The saved {record.get('label', 'table')} could not be displayed "
                    f"because it is missing or invalid ({exc}).",
                    level="error",
                )
                return
            self._populate_saved_csv_table(rows)
            preview_message = None
            displayed_rows = len(rows) - 1
            if total_rows > displayed_rows:
                preview_message = (
                    f"Showing first {displayed_rows:,} of {total_rows:,} rows. "
                    "Open the CSV to inspect the complete table."
                )
            self._set_native_artifact_metadata(
                record, region, table_preview=preview_message
            )
            return

        self._show_no_image("The selected saved artifact has an unsupported type.")

    @staticmethod
    def _read_saved_csv_table(
        path: str,
        *,
        preview_limit: int = NATIVE_CSV_PREVIEW_ROW_LIMIT,
    ) -> Tuple[List[List[str]], int]:
        if not path or not os.path.isfile(path):
            raise OSError("file is missing")
        if preview_limit < 1:
            raise ValueError("the CSV preview limit must be positive")
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                headers = next(reader)
            except StopIteration as exc:
                raise ValueError("the table has no column header") from exc
            if not headers:
                raise ValueError("the table has no column header")
            rows = [headers]
            width = len(headers)
            total_rows = 0
            for row in reader:
                if len(row) != width:
                    raise ValueError(
                        "the table rows do not match the stored column count"
                    )
                total_rows += 1
                if total_rows <= preview_limit:
                    rows.append(row)
        return rows, total_rows

    def _populate_saved_csv_table(self, rows: List[List[str]]) -> None:
        headers = list(rows[0])
        body = rows[1:]
        self._artifact_table.setColumnCount(len(headers))
        self._artifact_table.setHorizontalHeaderLabels(headers)
        self._artifact_table.setRowCount(len(body))
        for row_idx, row in enumerate(body):
            for col_idx, value in enumerate(row):
                self._artifact_table.setItem(row_idx, col_idx, QTableWidgetItem(value))

    def _set_native_artifact_metadata(
        self,
        record: Dict[str, Any],
        region: str,
        *,
        tab: str | None = None,
        table_preview: str | None = None,
    ) -> None:
        selected_tab = str(tab or self._selected_tab()).strip()
        lines = [f"Artifact: {record.get('label', '')}"]
        if record.get("scope") == "run":
            lines.append("Scope: Run-level (all included ROIs)")
        else:
            lines.insert(0, f"ROI: {region}")
        timeline = self._native_continuous_context.get("timeline", {})
        timeline_mode = str(timeline.get("timeline_mode") or "").strip().lower()
        timeline_labels = {
            "elapsed": "Elapsed time",
            "fixed_daily_anchor": "Fixed daily anchor",
            "civil": "Civil clock",
        }
        if timeline_mode:
            lines.append(f"Timeline: {timeline_labels.get(timeline_mode, timeline_mode)}")
        if timeline.get("fixed_daily_anchor_clock"):
            lines.append(f"Plotted-day start: {timeline['fixed_daily_anchor_clock']}")
        if timeline.get("recording_start_clock"):
            lines.append(f"Recording start: {timeline['recording_start_clock']}")
        if record.get("correction_strategy_label"):
            lines.append(f"Correction: {record['correction_strategy_label']}")
        representative = record.get("representative_window")
        if isinstance(representative, dict):
            index = representative.get("window_index")
            start = representative.get("elapsed_start_sec")
            end = representative.get("elapsed_end_sec")
            if index is not None:
                text = f"Representative window: {index}"
                if start is not None and end is not None:
                    text += f" ({float(start):g}–{float(end):g}s)"
                lines.append(text)
        if record.get("auc_units"):
            lines.append(f"AUC units: {record['auc_units']}")
        if record.get("day_index") is not None:
            lines.append(f"Plotted day: {int(record['day_index'])}")
        sampled_labels = record.get("sampled_column_labels")
        if isinstance(sampled_labels, (list, tuple)) and sampled_labels:
            lines.append("Columns: " + "; ".join(str(label) for label in sampled_labels))
        if record.get("display_explanation"):
            lines.append(str(record["display_explanation"]))
        window_timing = self._native_continuous_context.get("window_timing", {})
        if record.get("artifact_type") == "table" and window_timing:
            length = window_timing.get("window_length_sec")
            step = window_timing.get("window_step_sec")
            if length is not None and step is not None:
                lines.append(f"Window length: {float(length):g}s; step: {float(step):g}s")
        if table_preview:
            lines.append(table_preview)
        if selected_tab == TAB_VERIFICATION:
            allowed_prefixes = ("ROI:", "Correction:", "Representative window:")
        elif selected_tab in {TAB_TONIC, TAB_PHASIC_SUMMARY}:
            allowed_prefixes = (
                "ROI:",
                "Timeline:",
                "Plotted-day start:",
                "Recording start:",
            )
            if selected_tab == TAB_PHASIC_SUMMARY:
                allowed_prefixes += ("AUC units:",)
        elif selected_tab in _NATIVE_CONTINUOUS_DAY_PLOT_FAMILY_TABS.values():
            allowed_prefixes = (
                "ROI:",
                "Timeline:",
                "Plotted-day start:",
                "Recording start:",
                "Plotted day:",
                "Columns:",
                "Correction:",
                "Continuous Day Plots sample ",
            )
        else:
            allowed_prefixes = (
                "ROI:",
                "Timeline:",
                "Plotted-day start:",
                "Recording start:",
                "Window length:",
                "Showing first ",
            )
        lines = [line for line in lines if line.startswith(allowed_prefixes)]
        self._artifact_metadata_label.setText("  •  ".join(lines))
        self._artifact_metadata_label.setVisible(True)

    def _set_image(self, path: str):
        if not path or not os.path.isfile(path):
            self._show_no_image("Image file missing.")
            return
        pix = QPixmap(path)
        if pix.isNull():
            self._show_no_image(f"Unable to render image:\n{os.path.basename(path)}")
            return

        self._active_pixmap = pix
        if self._image_interaction is not None:
            self._image_interaction.set_pixmap(pix, reset_zoom=True)
            self._zoom_mode = self._image_interaction.zoom_mode
        else:
            self._render_image()

    def _on_image_clicked(self) -> None:
        """Toggle between fit-to-view and full-size inspection mode."""
        if self._active_pixmap.isNull():
            return
        self._set_zoom_mode(not self._zoom_mode)
        self._render_image()

    def _set_zoom_mode(self, enabled: bool) -> None:
        if self._image_interaction is not None:
            self._image_interaction.set_zoom_mode(bool(enabled))
        else:
            self._zoom_mode = bool(enabled)

    def _render_image(self) -> None:
        """Render active image with shared interactive zoom/pan behavior."""
        if self._image_interaction is not None:
            self._image_interaction.render()
            self._zoom_mode = self._image_interaction.zoom_mode

    def _show_no_image(self, message: str):
        self._active_image_path = ""
        self._active_pixmap = QPixmap()
        self._artifact_metadata_label.setText("")
        self._artifact_metadata_label.setVisible(False)
        self._image_title_label.setToolTip("")
        self._artifact_table_scroll.setVisible(False)
        self._image_scroll.setVisible(True)
        self._artifact_table.setRowCount(0)
        self._artifact_table.setColumnCount(0)
        if self._image_interaction is not None:
            self._image_interaction.clear(message, fallback_width=640, fallback_height=320)
            self._zoom_mode = self._image_interaction.zoom_mode
        else:
            self._set_zoom_mode(False)
            self._image_label.setPixmap(QPixmap())
            self._image_label.setText(message)
            viewport = self._image_scroll.viewport().size()
            if viewport.width() < 10 or viewport.height() < 10:
                viewport = QSize(640, 320)
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

    def _refresh_inline_actions(self) -> None:
        self._set_open_button_state(self._open_run_report_btn, self._run_summary_path)
        self._open_run_report_btn.setToolTip(
            "Open the saved analysis report for this run."
        )
        region = self._selected_region()
        region_path = self._region_paths.get(region, "")
        self._set_open_button_state(
            self._open_region_summary_btn,
            os.path.join(region_path, "summary") if region_path else "",
        )
        self._set_open_button_state(
            self._open_region_day_plots_btn,
            (
                os.path.join(region_path, "day_plots")
                if (
                    region_path
                    and (
                        not self._native_continuous_mode
                        or any(
                            record.get("artifact_type") == "image"
                            and str(record.get("family") or "")
                            in _NATIVE_CONTINUOUS_DAY_PLOT_FAMILY_TABS
                            for record in self._native_continuous_artifacts_by_roi.get(
                                region, []
                            )
                        )
                    )
                )
                else ""
            ),
        )
        table_target = os.path.join(region_path, "tables") if region_path else ""
        self._open_region_tables_btn.setText("Tables")
        self._open_region_tables_btn.setToolTip(
            "Open the selected region tables folder."
        )
        self._set_open_button_state(
            self._open_region_tables_btn,
            table_target,
        )

    def _set_open_button_state(self, button: QPushButton, path: str) -> None:
        exists = bool(path and os.path.exists(path))
        button.setEnabled(exists)
        if path:
            button.setToolTip(path if exists else f"Not available: {path}")
        else:
            button.setToolTip("Not available for the current selection.")

    def _open_selected_region_subpath(self, child: str) -> None:
        region = self._selected_region()
        region_path = self._region_paths.get(region, "")
        target = os.path.join(region_path, child) if region_path else ""
        self._open_path(target)

    def _open_selected_table_or_region_tables(self) -> None:
        self._open_selected_region_subpath("tables")

    def _open_path(self, path: str):
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _set_status_message(self, text: str, level: str) -> None:
        """Centralized status-label styling for workspace states."""
        style_map = {
            "idle": "color: gray; font-size: 13px;",
            "running": "color: #666; font-size: 13px;",
            "ready": "font-weight: bold; font-size: 14px;",
            "error": "color: #a94442; font-size: 12px;",
            "warning": "color: #8a6d3b; font-weight: bold; font-size: 13px;",
        }
        self._status_label.setText(text)
        self._status_label.setStyleSheet(style_map.get(level, style_map["idle"]))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._active_image_path and os.path.isfile(self._active_image_path):
            self._render_image()

    def _on_zoom_mode_changed(self, enabled: bool) -> None:
        self._zoom_mode = bool(enabled)
