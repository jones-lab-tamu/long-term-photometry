"""GUI tests for the Guided Feature Detection selected-ROI preview UI."""

import os
from types import SimpleNamespace
import pytest
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem, QGroupBox, QComboBox, QPushButton, QLabel, QTableWidget, QWidget, QToolButton

from gui.main_window import MainWindow, LOCAL_CORRECTION_PREVIEW_SOURCE_TYPE
from photometry_pipeline.guided_feature_detection_preview import GuidedFeaturePreviewUnsupportedError


MISSING_TRACE_MESSAGE = "The selected preview segment is unavailable."


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

    # Assert combo box is not visually collapsed and has minimum width configured
    assert roi_combo.minimumWidth() >= 80
    assert roi_combo.sizeAdjustPolicy() == QComboBox.AdjustToMinimumContentsLengthWithIcon

    # Segment label
    segment_lbl = panel.findChild(QLabel, "guidedFeaturePreviewSegmentLabel")
    assert segment_lbl is not None
    segment_combo = panel.findChild(
        QComboBox, "guidedFeaturePreviewSegmentCombo"
    )
    assert segment_combo is not None

    plot = panel.findChild(QWidget, "guidedFeaturePreviewPlot")
    assert plot is not None
    assert plot.isHidden()

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
    assert roi_combo.count() == 1
    assert roi_combo.currentText() == "CH1"


def test_roi_selector_empty_state(window):
    """Verify selector and button behavior when no ROIs are included."""
    window._roi_list.clear()
    window._guided_roi_list.clear()

    window._refresh_guided_feature_detection_preview_panel()

    roi_combo = window.findChild(QComboBox, "guidedFeaturePreviewRoiCombo")
    gen_btn = window.findChild(QPushButton, "guidedFeaturePreviewGenerateButton")
    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")

    assert roi_combo.count() == 0
    assert not gen_btn.isEnabled()
    assert status_lbl.text() == "No included ROIs available."


def _setup_signal_only_evidence(window, *, time_sec, preview_dff, valid=True, current_or_stale="current", preview_only=True, production_analysis=False):
    """Helper to mock evidence dictionary matching the exact production shape."""
    window._roi_list.clear()
    window._guided_roi_list.clear()

    item = QListWidgetItem("CH1")
    item.setCheckState(Qt.Checked)
    window._roi_list.addItem(item)
    window._guided_roi_list.addItem(QListWidgetItem(item))

    # Real Step 4 strategy confirmed choice key lookup matching
    choice_key = window._guided_confirm_choice_key(LOCAL_CORRECTION_PREVIEW_SOURCE_TYPE, None, "CH1")
    window._guided_strategy_choices[choice_key] = {
        "strategy": "signal_only_f0",
        "roi": "CH1",
        "strategy_family": "signal_only_f0",
        "confirmed": True,
        "current": True,
        "stale": False,
        "preview_only": True,
        "production_analysis": False,
        "source_type": LOCAL_CORRECTION_PREVIEW_SOURCE_TYPE,
        "setup_signature": "test-setup-sig",
    }

    # Registry evidence structure
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


def _setup_dynamic_evidence(
    window,
    *,
    time_sec,
    preview_dff=None,
    mode="global_linear_regression",
):
    window._roi_list.clear()
    window._guided_roi_list.clear()
    for roi_list in (window._roi_list, window._guided_roi_list):
        item = QListWidgetItem("CH1")
        item.setCheckState(Qt.Checked)
        roi_list.addItem(item)
    choice_key = window._guided_confirm_choice_key(
        LOCAL_CORRECTION_PREVIEW_SOURCE_TYPE, None, "CH1"
    )
    window._guided_strategy_choices[choice_key] = {
        "strategy": mode,
        "roi": "CH1",
        "strategy_family": "dynamic_fit",
        "dynamic_fit_mode": mode,
        "confirmed": True,
        "current": True,
        "stale": False,
        "preview_only": True,
        "production_analysis": False,
        "source_type": LOCAL_CORRECTION_PREVIEW_SOURCE_TYPE,
        "setup_signature": "test-setup-sig",
    }
    method_status = {"status": "success"}
    if preview_dff is not None:
        method_status["local_preview_dff_evidence"] = {
            "roi_id": "CH1",
            "strategy_family": "dynamic_fit",
            "selected_strategy": mode,
            "dynamic_fit_mode": mode,
            "trace_source": "local_correction_preview_dff",
            "dff_scale": "fractional_ratio",
            "preview_only": True,
            "production_analysis": False,
            "current_or_stale": "current",
            "valid": True,
            "time_sec": np.asarray(time_sec, dtype=float),
            "preview_dff": np.asarray(preview_dff, dtype=float),
            "baseline_scope": "selected_local_preview_segment",
        }
    window._guided_local_preview_evidence_by_roi["CH1"] = {
        "setup_signature": "test-setup-sig",
        "locked_evidence_candidates": {},
        "result": {
            "preview_id": "dynamic-preview",
            "roi": "CH1",
            "source_type": "local_raw_segment",
            "preview_only": True,
            "production_analysis": False,
            "preview_segment_label": "session-1",
            "method_statuses": {mode: method_status},
        },
        "provenance": {
            "preview_only": True,
            "production_analysis": False,
            "selected_roi": "CH1",
            "selected_segment_label": "session-1",
            "preview_id": "dynamic-preview",
        },
    }
    window._guided_feature_event_signal_combo.setCurrentText("dff")
    window._refresh_guided_feature_detection_preview_panel()


def test_evidence_lookup_resolution(window):
    """Verify that _guided_local_preview_evidence_for_roi resolves evidence correctly."""
    t = np.arange(10) * 0.1
    y = np.zeros(10)
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)

    resolved = window._guided_local_preview_evidence_for_roi("CH1", require_current=True)
    assert resolved is not None
    assert resolved["setup_signature"] == "test-setup-sig"
    assert resolved["result"]["roi"] == "CH1"


def test_build_context_with_real_registry_shape(window):
    """Verify context builder creates signal_only_dff keys with proper sampling rate."""
    t = np.arange(10) * 0.1
    y = np.zeros(10)
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)

    context = window._build_guided_feature_detection_preview_context("CH1")
    assert "CH1" in context["signal_only_dff"]

    trace_info = context["signal_only_dff"]["CH1"]
    assert trace_info["fs_hz"] == pytest.approx(10.0)
    assert np.array_equal(trace_info["time_sec"], t)
    assert np.array_equal(trace_info["trace"], y)


def test_manual_regression_success_path(window):
    """Perform regression check for the exact manual workflow success path (Signal-Only + dF/F)."""
    t = np.arange(100, dtype=float) * 0.1
    y = np.sin(t)
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)

    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")
    plot = window.findChild(QWidget, "guidedFeaturePreviewPlot")

    assert status_lbl.text() == "Preview generated successfully."
    assert not result_table.isHidden()
    assert not plot.isHidden()
    assert np.array_equal(plot.time_sec, t)
    assert np.array_equal(plot.trace, y)

    # 10-row parameters mapped correctly
    assert result_table.item(0, 1).text() == "CH1"
    assert result_table.item(1, 1).text() == "dff"
    assert result_table.item(2, 1).text() == "signal_only_f0 / none"
    assert result_table.item(9, 1).text() == "Preview-only (no files written)"
    details_toggle = window.findChild(
        QToolButton, "guidedFeaturePreviewDetailsToggle"
    )
    assert not details_toggle.isHidden()
    assert window._guided_feature_preview_details_content.isHidden()
    details_toggle.click()
    assert not window._guided_feature_preview_details_content.isHidden()


def test_visual_plot_renders_events_and_thresholds(
    window, monkeypatch
):
    t = np.arange(8, dtype=float)
    y = np.array([0.0, 2.0, 0.0, -2.0, 0.0, 3.0, 0.0, -3.0])
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)
    expected = SimpleNamespace(
        roi_id="CH1",
        event_signal="dff",
        time_sec=t,
        trace=y,
        prefiltered_trace=y.copy(),
        threshold_upper=1.5,
        threshold_lower=-1.5,
        positive_peak_indices=np.array([1, 5]),
        negative_peak_indices=np.array([3, 7]),
        positive_peak_times_sec=np.array([1.0, 5.0]),
        negative_peak_times_sec=np.array([3.0, 7.0]),
        feature_settings_digest="digest",
    )
    import photometry_pipeline.guided_feature_detection_preview as preview_module
    monkeypatch.setattr(
        preview_module,
        "build_guided_feature_detection_preview",
        lambda **_kwargs: expected,
    )

    window._on_guided_generate_feature_detection_preview()

    plot = window.findChild(QWidget, "guidedFeaturePreviewPlot")
    assert not plot.isHidden()
    assert np.array_equal(plot.positive_peak_indices, [1, 5])
    assert np.array_equal(plot.negative_peak_indices, [3, 7])
    assert plot.threshold_upper == pytest.approx(1.5)
    assert plot.threshold_lower == pytest.approx(-1.5)
    assert window._guided_feature_preview_last_result is expected


def test_only_retained_in_memory_segment_is_selectable(window):
    t = np.arange(10, dtype=float) * 0.1
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=np.sin(t))

    segment_combo = window.findChild(
        QComboBox, "guidedFeaturePreviewSegmentCombo"
    )
    assert segment_combo.count() == 1
    assert segment_combo.currentText() == "retained preview segment"


def test_segment_selector_lists_all_discovered_previewable_sessions(window):
    _setup_signal_only_evidence(
        window,
        time_sec=np.arange(10, dtype=float),
        preview_dff=np.zeros(10),
    )
    window._discovery_cache = {
        "resolved_format": "rwd",
        "sessions": [
            {
                "index": 0,
                "session_id": "session-1",
                "path": "C:/raw/session-1.csv",
                "included_in_preview": True,
            },
            {
                "index": 1,
                "session_id": "session-2",
                "path": "C:/raw/session-2.csv",
                "included_in_preview": True,
            },
            {
                "index": 2,
                "session_id": "excluded",
                "path": "C:/raw/excluded.csv",
                "included_in_preview": False,
            },
        ],
    }
    window._refresh_guided_feature_detection_preview_panel()

    combo = window._guided_feature_preview_segment_combo
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "session-1",
        "session-2",
    ]
    assert combo.itemData(1)["discovered_session_index"] == 1
    assert combo.itemData(1)["adapter_chunk_index"] == 0


@pytest.mark.parametrize(
    ("family", "strategy"),
    [
        ("signal_only_f0", "signal_only_f0"),
        ("dynamic_fit", "global_linear_regression"),
    ],
)
def test_generate_preview_computes_selected_nonretained_segment_on_demand(
    window, monkeypatch, family, strategy
):
    if family == "dynamic_fit":
        _setup_dynamic_evidence(
            window,
            time_sec=np.arange(10, dtype=float),
            preview_dff=np.zeros(10),
            mode=strategy,
        )
    else:
        _setup_signal_only_evidence(
            window,
            time_sec=np.arange(10, dtype=float),
            preview_dff=np.zeros(10),
        )
    window._guided_local_preview_evidence_by_roi.clear()
    window._discovery_cache = {
        "resolved_format": "rwd",
        "sessions": [
            {
                "index": 0,
                "session_id": "session-1",
                "path": "C:/raw/session-1.csv",
            },
            {
                "index": 1,
                "session_id": "session-2",
                "path": "C:/raw/session-2.csv",
            },
        ],
    }
    window._guided_feature_preview_config_overrides = (
        lambda _segment, _fmt: {}
    )
    window._active_config_source_path = lambda: "C:/config.yaml"
    calls = []
    t = np.arange(100, dtype=float) * 0.1
    y = np.sin(t)

    def compute(source_file, **kwargs):
        calls.append((source_file, kwargs))
        return {
            "valid": True,
            "time_sec": t,
            "preview_dff": y,
            "fs_hz": 10.0,
            "segment_label": kwargs["segment_label"],
            "issues": [],
        }

    import gui.main_window as main_window_module
    monkeypatch.setattr(
        main_window_module,
        "compute_guided_local_preview_dff_trace_in_memory",
        compute,
    )
    window._refresh_guided_feature_detection_preview_panel()
    window._guided_feature_preview_segment_combo.setCurrentIndex(1)

    window._on_guided_generate_feature_detection_preview()

    assert window._guided_feature_preview_status_label.text() == (
        "Preview generated successfully."
    )
    assert len(calls) == 1
    source_file, kwargs = calls[0]
    assert source_file.endswith("session-2.csv")
    assert kwargs["chunk_index"] == 1
    assert kwargs["strategy_family"] == family
    assert kwargs["strategy"] == strategy
    assert kwargs["dynamic_fit_mode"] == (
        strategy if family == "dynamic_fit" else None
    )
    assert np.array_equal(
        window._guided_feature_preview_plot.trace, y
    )
    assert window._guided_feature_preview_on_demand_trace["valid"] is True


def test_on_demand_preview_refuses_stale_confirmed_choice(
    window, monkeypatch
):
    _setup_signal_only_evidence(
        window,
        time_sec=np.arange(10, dtype=float),
        preview_dff=np.zeros(10),
    )
    choice = next(iter(window._guided_strategy_choices.values()))
    choice.update(current=False, stale=True)
    window._discovery_cache = {
        "resolved_format": "rwd",
        "sessions": [
            {"index": 0, "session_id": "s1", "path": "C:/raw/s1.csv"}
        ],
    }
    window._refresh_guided_feature_detection_preview_panel()
    import gui.main_window as main_window_module
    monkeypatch.setattr(
        main_window_module,
        "compute_guided_local_preview_dff_trace_in_memory",
        lambda *_args, **_kwargs: pytest.fail(
            "stale choice must not compute"
        ),
    )

    window._on_guided_generate_feature_detection_preview()

    assert "missing or stale" in (
        window._guided_feature_preview_status_label.text()
    )


def test_roi_and_segment_changes_clear_visual_plot(window):
    t = np.arange(100, dtype=float) * 0.1
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=np.sin(t))
    window._on_guided_generate_feature_detection_preview()
    plot = window.findChild(QWidget, "guidedFeaturePreviewPlot")
    assert not plot.isHidden()

    segment_combo = window.findChild(
        QComboBox, "guidedFeaturePreviewSegmentCombo"
    )
    segment_combo.addItem(
        "future retained segment",
        {"segment_label": "future retained segment", "preview_id": "two"},
    )
    segment_combo.setCurrentIndex(1)
    assert plot.isHidden()
    assert plot.time_sec.size == 0

    segment_combo.setCurrentIndex(0)
    window._on_guided_generate_feature_detection_preview()
    assert not plot.isHidden()
    roi_combo = window.findChild(QComboBox, "guidedFeaturePreviewRoiCombo")
    roi_combo.addItem("CH2", "CH2")
    roi_combo.setCurrentIndex(roi_combo.findText("CH2"))
    assert plot.isHidden()
    assert plot.time_sec.size == 0


def test_generate_preview_insufficient_samples(window):
    """Verify preview context fails if fewer than 2 time points exist."""
    t = np.array([0.0])
    y = np.array([1.0])
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)

    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")

    assert status_lbl.text() == MISSING_TRACE_MESSAGE
    assert result_table.isHidden()


def test_generate_preview_non_finite_dt(window):
    """Verify preview context fails if median dt is np.nan or np.inf."""
    t = np.array([0.0, np.nan])
    y = np.array([1.0, 2.0])
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)

    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")

    assert status_lbl.text() == MISSING_TRACE_MESSAGE
    assert result_table.isHidden()


def test_generate_preview_non_positive_dt(window):
    """Verify preview context fails if median dt is <= 0."""
    t = np.array([1.0, 0.5])
    y = np.array([1.0, 2.0])
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)

    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")

    assert status_lbl.text() == MISSING_TRACE_MESSAGE
    assert result_table.isHidden()


def test_generate_preview_stale_evidence(window):
    """Verify preview fails if evidence is stale or not current."""
    t = np.arange(100, dtype=float) * 0.1
    y = np.sin(t)

    # 1. valid is False
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y, valid=False)
    window._on_guided_generate_feature_detection_preview()
    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    assert status_lbl.text() == MISSING_TRACE_MESSAGE

    # 2. current_or_stale == "stale"
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y, current_or_stale="stale")
    window._on_guided_generate_feature_detection_preview()
    assert status_lbl.text() == MISSING_TRACE_MESSAGE

    # 3. preview_only in result is False
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y, preview_only=False)
    window._on_guided_generate_feature_detection_preview()
    assert status_lbl.text() == "No valid preview segment is selected."

    # 4. production_analysis in result is True
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y, production_analysis=True)
    window._on_guided_generate_feature_detection_preview()
    assert status_lbl.text() == "No valid preview segment is selected."


def test_generate_preview_dynamic_fit_dff_success(window):
    t = np.arange(100, dtype=float) * 0.1
    y = np.zeros(100)
    y[20] = 5.0
    y[60] = -5.0
    _setup_dynamic_evidence(window, time_sec=t, preview_dff=y)

    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")
    source_note = window.findChild(
        QLabel, "guidedFeaturePreviewSourceNote"
    )

    assert status_lbl.text() == "Preview generated successfully."
    assert not window._guided_feature_preview_plot.isHidden()
    assert np.array_equal(window._guided_feature_preview_plot.time_sec, t)
    assert np.array_equal(window._guided_feature_preview_plot.trace, y)
    assert not result_table.isHidden()
    assert not source_note.isHidden()
    assert "local correction-preview dF/F" in source_note.text()
    assert "Final run outputs may differ" in source_note.text()
    assert not window._guided_feature_detection_continue_btn.isEnabled()


def test_generate_preview_dynamic_fit_missing_dff(window):
    t = np.arange(100, dtype=float) * 0.1
    _setup_dynamic_evidence(window, time_sec=t, preview_dff=None)

    window._on_guided_generate_feature_detection_preview()

    assert (
        window._guided_feature_preview_status_label.text()
        == MISSING_TRACE_MESSAGE
    )
    assert window._guided_feature_preview_plot.isHidden()
    assert window._guided_feature_preview_result_table.isHidden()


def test_generate_preview_dynamic_fit_delta_f_does_not_fallback(window):
    t = np.arange(100, dtype=float) * 0.1
    _setup_dynamic_evidence(window, time_sec=t, preview_dff=np.sin(t))
    window._guided_feature_event_signal_combo.setCurrentText("delta_f")

    window._on_guided_generate_feature_detection_preview()

    assert "supports only local correction-preview dF/F" in (
        window._guided_feature_preview_status_label.text()
    )
    assert window._guided_feature_preview_plot.isHidden()


def test_generate_preview_dynamic_fit_no_evidence(window):
    window._roi_list.clear()
    window._guided_roi_list.clear()
    for roi_list in (window._roi_list, window._guided_roi_list):
        item = QListWidgetItem("CH1")
        item.setCheckState(Qt.Checked)
        roi_list.addItem(item)
    choice_key = window._guided_confirm_choice_key(
        LOCAL_CORRECTION_PREVIEW_SOURCE_TYPE, None, "CH1"
    )
    window._guided_strategy_choices[choice_key] = {
        "strategy": "global_linear_regression",
        "roi": "CH1",
        "strategy_family": "dynamic_fit",
        "dynamic_fit_mode": "global_linear_regression",
    }
    window._guided_local_preview_evidence_by_roi.clear()
    window._guided_feature_event_signal_combo.setCurrentText("dff")
    window._refresh_guided_feature_detection_preview_panel()

    window._on_guided_generate_feature_detection_preview()

    assert "Correction strategy for CH1 is missing or stale" in (
        window._guided_feature_preview_status_label.text()
    )
    assert window._guided_feature_preview_plot.isHidden()


def test_generate_preview_missing_evidence(window):
    """Verify missing evidence displays the correct warning message."""
    window._roi_list.clear()
    window._guided_roi_list.clear()
    item = QListWidgetItem("CH1")
    item.setCheckState(Qt.Checked)
    window._roi_list.addItem(item)
    window._guided_roi_list.addItem(QListWidgetItem(item))

    # No strategy choices confirmed yet
    window._refresh_guided_feature_detection_preview_panel()

    window._on_guided_generate_feature_detection_preview()

    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    result_table = window.findChild(QTableWidget, "guidedFeaturePreviewResultTable")

    assert "Correction strategy for CH1 is missing or stale" in status_lbl.text()
    assert result_table.isHidden()


def test_settings_changed_invalidates_preview(window):
    """Verify that editing settings stales/clears the generated preview."""
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
    assert window._guided_feature_preview_plot.isHidden()


def test_preview_safety_and_continue_gating(window, tmp_path):
    """Verify that preview generation is filesystem-safe and doesn't bypass Continue gating."""
    assert not window._guided_feature_detection_continue_btn.isEnabled()

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
    def forbidden_call(*args, **kwargs):
        raise AssertionError("Forbidden external/cache access attempted!")

    import gui.main_window as main_window_module
    monkeypatch.setattr(main_window_module, "open_phasic_cache", forbidden_call)

    import subprocess
    monkeypatch.setattr(subprocess, "run", forbidden_call)
    monkeypatch.setattr(subprocess, "Popen", forbidden_call)

    t = np.arange(100, dtype=float) * 0.1
    y = np.sin(t)
    _setup_signal_only_evidence(window, time_sec=t, preview_dff=y)

    window._on_guided_generate_feature_detection_preview()
    status_lbl = window.findChild(QLabel, "guidedFeaturePreviewStatusLabel")
    assert status_lbl.text() == "Preview generated successfully."


def test_dynamic_preview_no_read_or_write_sentinels(window, monkeypatch):
    def forbidden_call(*args, **kwargs):
        raise AssertionError("Forbidden external/cache access attempted!")

    import gui.main_window as main_window_module
    monkeypatch.setattr(main_window_module, "open_phasic_cache", forbidden_call)
    import subprocess
    monkeypatch.setattr(subprocess, "run", forbidden_call)
    monkeypatch.setattr(subprocess, "Popen", forbidden_call)

    t = np.arange(100, dtype=float) * 0.1
    _setup_dynamic_evidence(window, time_sec=t, preview_dff=np.sin(t))
    window._on_guided_generate_feature_detection_preview()

    assert (
        window._guided_feature_preview_status_label.text()
        == "Preview generated successfully."
    )
