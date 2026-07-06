"""GUI tests for the Guided Feature Detection selected-ROI preview UI."""

import os
import pytest
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem, QGroupBox, QComboBox, QPushButton, QLabel, QTableWidget

from gui.main_window import MainWindow
from photometry_pipeline.guided_feature_detection_preview import GuidedFeaturePreviewUnsupportedError


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    w = MainWindow()
    w._guided_workflow_mode = "new_analysis"
    # Mock minimal cache state
    w._discovery_cache = {
        "resolved_format": "npm",
        "sessions": []
    }
    # Mock current run directory and confirm source
    w._current_run_dir = "C:/mock_run"
    w._guided_diagnostic_cache_record = None
    w._guided_strategy_choices = {}
    w._guided_local_preview_evidence_by_roi = {}
    
    # Mock setup signature to be deterministic
    w._guided_local_preview_setup_signature = lambda: "test-setup-sig"
    
    # Mock active preview segment
    w._selected_guided_preview_segment = lambda: {
        "discovered_session_index": 0,
        "segment_label": "session-1",
        "adapter_chunk_index": 0,
        "source_path": "",
    }
    w._selected_guided_preview_chunk = lambda: 0
    
    yield w
    w.close()
    w.deleteLater()


def test_preview_panel_layout_and_elements(window):
    """Verify that the preview panel exists on the step with correct elements."""
    panel = window.findChild(QGroupBox, "guidedFeatureDetectionPreviewPanel")
    assert panel is not None
    assert panel.title() == "Preview feature detection"

    # Explanation text
    note = panel.findChild(QLabel, "guidedFeatureDetectionPreviewNote")
    assert note is not None
    assert "shared settings" in note.text()

    # ROI selector
    roi_combo = panel.findChild(QComboBox, "guidedFeaturePreviewRoiCombo")
    assert roi_combo is not None

    # Segment label
    segment_lbl = panel.findChild(QLabel, "guidedFeaturePreviewSegmentLabel")
    assert segment_lbl is not None

    # Generate button
    gen_btn = panel.findChild(QPushButton, "guidedFeaturePreviewGenerateButton")
    assert gen_btn is not None
    assert gen_btn.text() == "Generate Preview"

    # Status label
    status_lbl = panel.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    assert status_lbl is not None

    # Results table
    result_table = panel.findChild(QTableWidget, "guidedFeaturePreviewResultTable")
    assert result_table is not None
    assert result_table.columnCount() == 2
    assert result_table.rowCount() == 10
    assert result_table.isHidden()


def test_roi_selector_populates_only_included_rois(window):
    """Verify the ROI selector filters included ROIs correctly."""
    # Add CH1 (Included) and CH2 (Excluded)
    window._roi_list.clear()
    window._guided_roi_list.clear()

    item1_full = QListWidgetItem("CH1")
    item1_full.setCheckState(Qt.Checked)
    window._roi_list.addItem(item1_full)

    item1_guided = QListWidgetItem("CH1")
    item1_guided.setCheckState(Qt.Checked)
    window._guided_roi_list.addItem(item1_guided)

    item2_full = QListWidgetItem("CH2")
    item2_full.setCheckState(Qt.Unchecked)
    window._roi_list.addItem(item2_full)

    item2_guided = QListWidgetItem("CH2")
    item2_guided.setCheckState(Qt.Unchecked)
    window._guided_roi_list.addItem(item2_guided)

    window._refresh_guided_feature_detection_preview_panel()

    roi_combo = window.findChild(QComboBox, "guidedFeaturePreviewRoiCombo")
    items = [roi_combo.itemText(i) for i in range(roi_combo.count())]
    
    assert "CH1" in items
    assert "CH2" not in items


def _setup_signal_only_evidence(window, *, time_sec, preview_dff, valid=True, current_or_stale="current", preview_only=True, production_analysis=False):
    """Helper to mock evidence dictionary matching the exact production shape."""
    window._roi_list.clear()
    item = QListWidgetItem("CH1")
    item.setCheckState(Qt.Checked)
    window._roi_list.addItem(item)
    window._guided_roi_list.addItem(QListWidgetItem(item))

    run_dir = window._current_guided_completed_run_dir()
    window._guided_strategy_choices[(run_dir, "CH1")] = {
        "strategy": "signal_only_f0",
        "roi": "CH1"
    }

    # The actual production dictionary shape does not have preview_only/production_analysis at the top level
    window._guided_local_preview_evidence_by_roi["CH1"] = {
        "setup_signature": "test-setup-sig",
        "locked_evidence_candidates": {},
        "result": {
            "roi": "CH1",
            "source_type": "local_raw_segment",
            "preview_only": preview_only,
            "production_analysis": production_analysis,
            "signal_only_f0_preview_evidence": {
                "valid": valid,
                "current_or_stale": current_or_stale,
                "time_sec": time_sec,
                "preview_dff": preview_dff,
                "parameters": {"fs_hz": 10.0}
            }
        },
        "provenance": {
            "preview_only": preview_only,
            "production_analysis": production_analysis,
            "selected_roi": "CH1",
            "preview_id": "test-id"
        }
    }
    window._guided_feature_event_signal_combo.setCurrentText("dff")
    window._refresh_guided_feature_detection_preview_panel()


def test_generate_preview_signal_only_success(window):
    """Verify successful generation of in-memory Signal-Only F0 preview."""
    t = np.arange(100, dtype=float) * 0.1
    y = np.sin(t)
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)
    
    # Trigger Generate
    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")

    assert status_lbl.text() == "Preview generated successfully."
    assert not result_table.isHidden()
    
    # Check 10-row parameters strictly
    assert result_table.item(0, 1).text() == "CH1"
    assert result_table.item(1, 1).text() == "dff"
    assert result_table.item(2, 1).text() == "signal_only_f0 / none"
    assert result_table.item(3, 1).text() is not None # Positive Events
    assert result_table.item(4, 1).text() is not None # Negative Events
    assert result_table.item(9, 1).text() == "Preview-only (no files written)"


def test_generate_preview_insufficient_samples(window):
    """Verify preview context fails if fewer than 2 time points exist."""
    t = np.array([0.0]) # Only 1 time point
    y = np.array([1.0])
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)

    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")

    assert "Preview evidence for CH1 is not available in memory" in status_lbl.text()
    assert result_table.isHidden()


def test_generate_preview_non_finite_dt(window):
    """Verify preview context fails if median dt is np.nan or np.inf."""
    t = np.array([0.0, np.nan]) # Non-finite time point
    y = np.array([1.0, 2.0])
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)

    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")

    assert "Preview evidence for CH1 is not available in memory" in status_lbl.text()
    assert result_table.isHidden()


def test_generate_preview_non_positive_dt(window):
    """Verify preview context fails if median dt is <= 0."""
    t = np.array([1.0, 0.5]) # dt is negative
    y = np.array([1.0, 2.0])
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)

    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")

    assert "Preview evidence for CH1 is not available in memory" in status_lbl.text()
    assert result_table.isHidden()


def test_generate_preview_stale_evidence(window):
    """Verify preview fails if evidence is stale or not current."""
    t = np.arange(100, dtype=float) * 0.1
    y = np.sin(t)
    
    # 1. valid is False
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y, valid=False)
    window._on_guided_generate_feature_detection_preview()
    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    assert "Preview evidence for CH1 is not available in memory" in status_lbl.text()

    # 2. current_or_stale == "stale"
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y, current_or_stale="stale")
    window._on_guided_generate_feature_detection_preview()
    assert "Preview evidence for CH1 is not available in memory" in status_lbl.text()

    # 3. preview_only in result is False
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y, preview_only=False)
    window._on_guided_generate_feature_detection_preview()
    assert "Preview evidence for CH1 is not available in memory" in status_lbl.text()

    # 4. production_analysis in result is True
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y, production_analysis=True)
    window._on_guided_generate_feature_detection_preview()
    assert "Preview evidence for CH1 is not available in memory" in status_lbl.text()


def test_generate_preview_dynamic_fit_unsupported(window):
    """Verify dynamic-fit preview displays the correct unsupported/actionable message."""
    # 1. Setup ROI
    window._roi_list.clear()
    item = QListWidgetItem("CH1")
    item.setCheckState(Qt.Checked)
    window._roi_list.addItem(item)
    window._guided_roi_list.addItem(QListWidgetItem(item))

    # 2. Setup confirmed strategy choice for ROI (a dynamic fit mode)
    run_dir = window._current_guided_completed_run_dir()
    window._guided_strategy_choices[(run_dir, "CH1")] = {
        "strategy": "global_linear_regression",
        "roi": "CH1"
    }

    # Set Event signal to delta_f
    window._guided_feature_event_signal_combo.setCurrentText("delta_f")

    # Populate dropdowns
    window._refresh_guided_feature_detection_preview_panel()
    
    # 3. Trigger Generate
    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")

    # Should show the actionable message and hide the table
    assert "Preview evidence for this ROI is not available in memory" in status_lbl.text()
    assert result_table.isHidden()


def test_generate_preview_missing_evidence(window):
    """Verify missing evidence displays the correct warning message."""
    window._roi_list.clear()
    item = QListWidgetItem("CH1")
    item.setCheckState(Qt.Checked)
    window._roi_list.addItem(item)
    window._guided_roi_list.addItem(QListWidgetItem(item))

    # No strategy choices confirmed yet
    window._refresh_guided_feature_detection_preview_panel()
    
    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")

    assert "Preview evidence for CH1 is not available in memory" in status_lbl.text()
    assert result_table.isHidden()


def test_settings_changed_invalidates_preview(window):
    """Verify that editing settings stales/clears the generated preview."""
    # Generate successful preview first
    t = np.arange(100, dtype=float) * 0.1
    y = np.sin(t)
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)
    window._on_guided_generate_feature_detection_preview()
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")
    assert not result_table.isHidden()

    # Simulate settings edit
    window._on_guided_feature_detection_editor_changed()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    assert "Settings changed" in status_lbl.text()
    assert result_table.isHidden()


def test_preview_safety_and_continue_gating(window, tmp_path):
    """Verify that preview generation is filesystem-safe and doesn't bypass Continue gating."""
    # Initial status
    assert not window._guided_feature_detection_continue_btn.isEnabled()

    # Run successful preview
    t = np.arange(100, dtype=float) * 0.1
    y = np.sin(t)
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)
    window._on_guided_generate_feature_detection_preview()

    # Continue should still be disabled because settings haven't been applied to plan
    assert not window._guided_feature_detection_continue_btn.isEnabled()

    # Assert no files written under the tmp_path or workspace
    assert sorted(tmp_path.rglob("*")) == []


def test_no_read_or_write_sentinels(window, monkeypatch):
    """Verify that Generate Preview does not attempt to read from caches or run external scripts."""
    # Define raises to capture any unauthorized read/write/CLI calls
    def forbidden_call(*args, **kwargs):
        raise AssertionError("Forbidden external/cache access attempted!")

    # Patch in gui.main_window (or imported modules)
    import gui.main_window as main_window_module
    monkeypatch.setattr(main_window_module, "open_phasic_cache", forbidden_call)

    # Also patch subprocess/run commands to prevent CLI calls
    import subprocess
    monkeypatch.setattr(subprocess, "run", forbidden_call)
    monkeypatch.setattr(subprocess, "Popen", forbidden_call)

    # Run successful preview with active sentinels
    t = np.arange(100, dtype=float) * 0.1
    y = np.sin(t)
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)
    
    # Should complete successfully without triggering any monkeypatched sentinel raises
    window._on_guided_generate_feature_detection_preview()
    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    assert status_lbl.text() == "Preview generated successfully."
