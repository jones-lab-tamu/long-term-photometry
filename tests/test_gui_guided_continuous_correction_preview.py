"""CR1-F1-D: the Guided correction preview works for one continuous recording.

The first real scientist-facing attempt reached Correction approach and stopped
there. The page said "Complete Select data, Recording structure, and ROI
selection" even though all three were complete, and no correction evidence
could be built.

The cause was not recording-structure resolution: the correction preview built
its list of preview segments from the discovered *sessions*, and a continuous
recording correctly has none. Explicit Continuous failed identically, and both
Intermittent paths were unaffected.

These tests drive the real Select data -> Recording structure -> Correction
approach boundary and assert what a scientist can actually do there: that a
continuous recording offers analysis windows as preview evidence, that only the
selected window's rows are read, that repeated sessions still behave exactly as
before, and that changing the setup invalidates evidence built from the old one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QDeadlineTimer, Qt
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
import photometry_pipeline.preview.correction_preview as correction_preview_module
from gui.main_window import (
    GUIDED_PREVIEW_MISSING_PREREQUISITES_TEXT,
    GUIDED_STRUCTURE_CHOICE_AUTO,
    MainWindow,
)

from tests.test_guided_continuous_rwd_correction_pass_persistence import _values


pytestmark = pytest.mark.usefixtures("no_real_modals")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance.close()
    instance.deleteLater()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _fluorescence_rows(count):
    indices = np.arange(count, dtype=float)
    time_s, control, signal = _values(indices)
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(count):
        lines.append(
            f"{time_s[index]:.4f},{control[index, 0]:.12f},{signal[index, 0]:.12f},"
            f"{control[index, 1]:.12f},{signal[index, 1]:.12f}\n"
        )
    return lines


def _continuous_folder(folder, *, samples=12_000):
    """One uninterrupted 10 Hz RWD recording, long enough for whole windows."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Fluorescence.csv").write_text(
        "".join(_fluorescence_rows(samples)), encoding="utf-8", newline=""
    )
    (folder / "Events.csv").write_text(
        "Time(s),Event\n0.0,start\n", encoding="utf-8", newline=""
    )
    return folder


def _intermittent_folder(folder, *, sessions=2):
    """Repeated RWD session folders, the shape the session path expects."""
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(sessions):
        session = folder / f"2026_03_1{index}-10_00_00"
        session.mkdir()
        (session / "Fluorescence.csv").write_text(
            "".join(_fluorescence_rows(600)), encoding="utf-8", newline=""
        )
    return folder


# ---------------------------------------------------------------------------
# Driving the visible workflow
# ---------------------------------------------------------------------------


def _pump(qapp, predicate, timeout_ms=180_000):
    deadline = QDeadlineTimer(timeout_ms)
    while predicate() and not deadline.hasExpired():
        qapp.processEvents()


def _select_data(window, qapp, folder, *, fmt="auto", structure=GUIDED_STRUCTURE_CHOICE_AUTO):
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText(fmt)
    index = window._guided_acquisition_mode_combo.findData(structure)
    assert index >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(index)
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(folder.parent / "output"))
    window._on_guided_discover_rois()
    _pump(qapp, lambda: window._guided_roi_discovery_running)
    _pump(
        qapp,
        lambda: getattr(window, "_guided_roi_discovery_thread", None) is not None,
        20_000,
    )
    for row in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(row).setCheckState(Qt.Checked)
    qapp.processEvents()


def _check_recording(window, qapp):
    """The recording check forward navigation performs for continuous data."""
    assert window._maybe_start_guided_continuous_rwd_recording_check() is True
    _pump(
        qapp,
        lambda: getattr(window, "_guided_continuous_rwd_check_worker", None)
        is not None,
    )
    _pump(
        qapp,
        lambda: getattr(window, "_guided_continuous_rwd_check_thread", None)
        is not None,
        20_000,
    )


def _open_correction_approach(window):
    window._refresh_guided_diagnostics_panel()
    window._refresh_guided_correction_next_action()


def _generate_preview(window, qapp):
    window._on_generate_guided_correction_preview()
    _pump(
        qapp,
        lambda: getattr(window, "_guided_correction_preview_running", False),
        300_000,
    )
    _pump(
        qapp,
        lambda: getattr(window, "_guided_correction_preview_thread", None)
        is not None,
        20_000,
    )
    return getattr(window, "_guided_preview_last_result", {}) or {}


def _segments(window):
    combo = window._guided_preview_chunk_combo
    return [combo.itemData(index) for index in range(combo.count())]


def _select_all_preview_methods(window):
    for checkbox in window._guided_preview_method_checkboxes.values():
        checkbox.setChecked(True)


def _generate_and_confirm_corrections(window, qapp):
    """Drive the existing Step 4 evidence/confirmation path to Step 5."""
    _select_all_preview_methods(window)
    for roi_index in range(window._guided_preview_roi_combo.count()):
        window._guided_preview_roi_combo.setCurrentIndex(roi_index)
        _generate_preview(window, qapp)
    window._refresh_guided_diagnostics_panel()
    window._refresh_guided_correction_next_action()
    for row in dict(
        getattr(window, "_guided_local_preview_confirmation_rows", {})
    ).values():
        combo = row["strategy_combo"]
        for index in range(combo.count()):
            combo.setCurrentIndex(index)
            if combo.currentData():
                break
        row["action_button"].click()
        qapp.processEvents()


def _is_inside(widget, ancestor):
    parent = widget.parentWidget()
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parentWidget()
    return False


# ---------------------------------------------------------------------------
# The exact reported regression
# ---------------------------------------------------------------------------


def test_auto_detected_continuous_reaches_a_correction_preview(
    window, qapp, tmp_path
):
    """The reported failure: untouched defaults, real continuous folder."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)

    assert window._guided_selected_acquisition_mode() == GUIDED_STRUCTURE_CHOICE_AUTO
    assert window._guided_effective_acquisition_mode() == "continuous"
    assert window._guided_select_data_readiness()[0] is True
    assert window._guided_recording_structure_readiness()[0] is True
    assert list(window._guided_selected_roi_ids()[1]) == ["ROI1", "ROI2"]

    _check_recording(window, qapp)
    _open_correction_approach(window)

    # The false prerequisite message is gone and evidence can be built.
    assert (
        window._guided_preview_locked_label.text()
        != GUIDED_PREVIEW_MISSING_PREREQUISITES_TEXT
    )
    assert window._guided_preview_locked_label.isHidden()
    assert window._guided_preview_source_ok is True
    assert window._guided_preview_source_type == "local_raw_segment"
    assert window._guided_preview_generate_btn.isEnabled() is True

    # Every included ROI can be previewed.
    roi_combo = window._guided_preview_roi_combo
    assert [
        roi_combo.itemData(index) for index in range(roi_combo.count())
    ] == ["ROI1", "ROI2"]


def test_continuous_preview_request_carries_the_resolved_structure(
    window, qapp, tmp_path, monkeypatch
):
    """The raw "auto" choice never reaches the preview request."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    _select_all_preview_methods(window)

    captured: list[dict] = []

    def spy(*args, **kwargs):
        captured.append(dict(kwargs))
        return real(*args, **kwargs)

    real = correction_preview_module.run_guided_local_correction_preview
    monkeypatch.setattr(
        "gui.main_window.run_guided_local_correction_preview", spy
    )

    result = _generate_preview(window, qapp)

    assert captured, "no correction-preview request was built"
    request = captured[0]
    assert request["input_format"] == "rwd"
    assert request["continuous_window_index"] == 0
    overrides = request["config_overrides"]
    assert "auto" not in {str(value).lower() for value in overrides.values()}
    assert overrides["continuous_window_sec"] == pytest.approx(
        float(window._continuous_window_sec_spin.value())
    )
    assert result["status"] == "success"
    assert result["continuous_analysis_window"]["window_index"] == 0


def test_continuous_preview_reads_only_the_selected_window(
    window, qapp, tmp_path
):
    """The full recording is never materialized as one preview chunk."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    _select_all_preview_methods(window)

    result = _generate_preview(window, qapp)

    window_facts = result["continuous_analysis_window"]
    window_sec = float(window._continuous_window_sec_spin.value())
    assert window_facts["window_duration_sec"] == pytest.approx(window_sec)
    assert window_facts["original_file_duration_sec"] > window_sec
    # Only the window's own rows were read, not the whole 12 000-row file.
    read_rows = window_facts["row_stop"] - window_facts["row_start"]
    assert read_rows < 12_000
    assert read_rows == pytest.approx(window_sec * 10.0, abs=2)


def test_continuous_preview_never_infers_the_contract_from_the_whole_file(
    window, qapp, tmp_path, monkeypatch
):
    """``_infer_rwd_chunk_contract`` reads an entire CSV; it must not run here."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    _select_all_preview_methods(window)

    monkeypatch.setattr(
        MainWindow,
        "_infer_rwd_chunk_contract",
        lambda self, path: pytest.fail(
            "the whole continuous recording was parsed to infer its contract"
        ),
    )

    assert _generate_preview(window, qapp)["status"] == "success"


def test_continuous_preview_requires_no_session_settings(
    window, qapp, tmp_path, monkeypatch
):
    """No sessions-per-hour, session duration, or missing-session approval."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    _select_all_preview_methods(window)

    # Session-only settings are genuinely absent for this recording.
    assert window._guided_sessions_per_hour_edit.text().strip() == ""
    assert window._guided_session_duration_edit.text().strip() == ""
    assert list(getattr(window, "_guided_approved_missing_sessions", [])) == []

    # And no session-shaped discovery entry is consulted to build evidence.
    for segment in _segments(window):
        assert "continuous_window_index" in segment
        assert "session_id" not in segment
    assert list((window._discovery_cache or {}).get("sessions") or []) == []

    monkeypatch.setattr(
        MainWindow,
        "_guided_tonic_gap_free_blocked_by_missing_sessions",
        lambda self: pytest.fail(
            "continuous correction evidence consulted missing-session state"
        ),
    )

    assert _generate_preview(window, qapp)["status"] == "success"


def test_explicit_continuous_matches_automatic_continuous(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder, fmt="rwd", structure="continuous")
    _check_recording(window, qapp)
    _open_correction_approach(window)

    assert window._guided_selected_acquisition_mode() == "continuous"
    assert window._guided_preview_source_ok is True
    assert window._guided_preview_generate_btn.isEnabled() is True
    assert all("continuous_window_index" in s for s in _segments(window))


# ---------------------------------------------------------------------------
# Preview windows
# ---------------------------------------------------------------------------


def test_windows_are_equal_non_overlapping_and_whole(window, qapp, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    window._continuous_window_sec_spin.setValue(300.0)
    _check_recording(window, qapp)
    _open_correction_approach(window)

    segments = _segments(window)
    assert len(segments) >= 2
    for index, segment in enumerate(segments):
        assert segment["continuous_window_index"] == index
        assert segment["window_start_sec"] == pytest.approx(index * 300.0)
        assert segment["window_end_sec"] == pytest.approx((index + 1) * 300.0)
    # Non-overlapping and contiguous.
    for earlier, later in zip(segments, segments[1:]):
        assert earlier["window_end_sec"] == pytest.approx(
            later["window_start_sec"]
        )


def test_window_longer_than_the_recording_is_explained(window, qapp, tmp_path):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    window._continuous_window_sec_spin.setValue(7200.0)
    _check_recording(window, qapp)
    _open_correction_approach(window)

    assert window._guided_preview_source_ok is False
    assert window._guided_preview_generate_btn.isEnabled() is False
    message = window._guided_preview_locked_label.text()
    assert message != GUIDED_PREVIEW_MISSING_PREREQUISITES_TEXT
    assert "shorter analysis window" in message


def test_feature_detection_reuses_continuous_window_selector_and_bounded_preview(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec", samples=12_000)
    _select_data(window, qapp, folder)
    window._continuous_window_sec_spin.setValue(200.0)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    _generate_and_confirm_corrections(window, qapp)

    window._on_guided_continue_to_feature_detection()
    qapp.processEvents()

    combo = window._guided_feature_preview_segment_combo
    native_plan = window._guided_continuous_rwd_native_segment_plan()
    assert native_plan is not None
    target_grid, segment_plan = native_plan
    draft = window._guided_continuous_rwd_live_draft()
    assert draft is not None
    assert draft.continuous_step_sec == pytest.approx(
        draft.continuous_window_sec
    )
    assert window._continuous_step_sec_spin.isEnabled() is False
    expected_descriptors = segment_plan.descriptors
    selector_data = [combo.itemData(index) for index in range(combo.count())]
    assert [
        int(item["continuous_window_index"]) for item in selector_data
    ] == [int(item.segment_index) for item in expected_descriptors]
    cadence = target_grid.cadence_fraction
    np.testing.assert_allclose(
        [float(item["window_start_sec"]) for item in selector_data],
        [float(item.start_target_index * cadence) for item in expected_descriptors],
    )
    np.testing.assert_allclose(
        [float(item["window_end_sec"]) for item in selector_data],
        [float(item.stop_target_index * cadence) for item in expected_descriptors],
    )
    assert combo.count() == 6
    assert combo.currentIndex() == 0
    assert combo.itemText(0).startswith("Window 1 (0:00:00")
    assert "0:03:20" in combo.itemText(0)
    assert combo.itemText(5).startswith("Window 6 (0:16:40")
    assert "0:20:00" in combo.itemText(5)

    window._guided_feature_event_apply_btn.click()
    qapp.processEvents()
    window._guided_feature_preview_segment_combo.setCurrentIndex(4)

    real_compute = main_window_module.compute_guided_local_preview_dff_trace_in_memory
    captured = []

    def counted_compute(*args, **kwargs):
        captured.append(dict(kwargs))
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(
        main_window_module,
        "compute_guided_local_preview_dff_trace_in_memory",
        counted_compute,
    )
    window._on_guided_generate_feature_detection_preview()

    assert captured
    selected_window = window._guided_feature_preview_on_demand_trace[
        "continuous_analysis_window"
    ]
    assert selected_window["window_index"] == 4
    assert selected_window["window_start_sec"] == pytest.approx(800.0)
    assert selected_window["window_end_sec"] == pytest.approx(1000.0)
    assert (
        selected_window["row_stop"] - selected_window["row_start"]
    ) < 12_000
    assert window._guided_feature_preview_last_result is not None

    # The existing shared ROI-change refresh preserves the selected window.
    window._guided_feature_preview_roi_combo.setCurrentIndex(1)
    qapp.processEvents()
    assert window._guided_feature_preview_segment_combo.currentIndex() == 4

# ---------------------------------------------------------------------------
# One generic CSV file: the same windows, from the same authority
# ---------------------------------------------------------------------------


def _continuous_csv_folder(folder):
    """One uninterrupted 8 Hz generic CSV recording, via the demo writer."""
    from gui.synthetic_demo_generator import generate_guided_continuous_demo

    result = generate_guided_continuous_demo(folder, _duration_sec=1200.0)
    assert result.success, result.message
    return Path(result.input_dir)


def _select_csv_data(window, qapp, folder):
    """The ordinary Select-data controls for one continuous CSV recording."""
    window._on_guided_start_setup_new_analysis()
    window._guided_format_combo.setCurrentText("custom_tabular")
    index = window._guided_acquisition_mode_combo.findData("continuous")
    assert index >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(index)
    window._guided_input_dir_edit.setText(str(folder))
    window._guided_output_dir_edit.setText(str(folder.parent / "output"))
    window._refresh_guided_csv_source_interpretation()
    time_combo = window._guided_csv_time_column_combo
    time_combo.setCurrentIndex(time_combo.findData("ElapsedSeconds"))
    units = window._guided_csv_time_units_combo
    units.setCurrentIndex(units.findData("seconds"))
    window._guided_csv_order_confirm_cb.setChecked(True)
    while len(window._guided_csv_mapping_rows) < 2:
        window._add_guided_csv_mapping_row()
    for row, roi in zip(window._guided_csv_mapping_rows, ("ROI1", "ROI2")):
        row["name"].setText(roi)
        row["signal"].setCurrentIndex(row["signal"].findData(f"{roi}_Signal"))
        row["reference"].setCurrentIndex(
            row["reference"].findData(f"{roi}_Reference")
        )
    # A generic CSV carries only elapsed seconds, so the clock is stated here.
    window._guided_fixed_daily_anchor_clock_edit.setText("07:00")
    window._guided_recording_start_clock_edit.setText("12:00:00")
    window._on_guided_discover_rois()
    _pump(qapp, lambda: window._guided_roi_discovery_running)
    _pump(
        qapp,
        lambda: getattr(window, "_guided_roi_discovery_thread", None) is not None,
        20_000,
    )
    for row in range(window._guided_roi_list.count()):
        window._guided_roi_list.item(row).setCheckState(Qt.Checked)
    qapp.processEvents()


def _identities(combo):
    return [
        (combo.itemText(index), combo.itemData(index)["continuous_window_index"])
        for index in range(combo.count())
    ]


def test_continuous_csv_offers_the_same_windows_as_the_segment_authority(
    window, qapp, tmp_path
):
    """Correction Preview is not limited to a single whole-recording entry."""
    folder = _continuous_csv_folder(tmp_path / "csv")
    _select_csv_data(window, qapp, folder)
    window._continuous_window_sec_spin.setValue(200.0)
    _check_recording(window, qapp)
    _open_correction_approach(window)

    # Before any correction choice exists, only whole windows are offered --
    # the same pre-plan rule the RWD path uses.
    segments = _segments(window)
    assert len(segments) == 5
    for index, segment in enumerate(segments):
        assert segment["continuous_window_index"] == index
        assert segment["window_start_sec"] == pytest.approx(index * 200.0)
        assert segment["window_end_sec"] == pytest.approx((index + 1) * 200.0)
    for earlier, later in zip(segments, segments[1:]):
        assert earlier["window_end_sec"] == pytest.approx(
            later["window_start_sec"]
        )
    labels = [
        window._guided_preview_chunk_combo.itemText(index)
        for index in range(window._guided_preview_chunk_combo.count())
    ]
    assert labels[0].startswith("Window 1 (0:00:00")
    assert labels[-1].startswith("Window 5 (0:13:20")
    # The whole-recording name is never the only preview segment.
    assert len(labels) > 1
    assert not any(label == "continuous_recording" for label in labels)


def test_continuous_csv_feature_detection_uses_the_same_ordered_windows(
    window, qapp, tmp_path, monkeypatch
):
    """Both preview stages resolve identical segment identities and bounds."""
    folder = _continuous_csv_folder(tmp_path / "csv")
    _select_csv_data(window, qapp, folder)
    window._continuous_window_sec_spin.setValue(200.0)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    _generate_and_confirm_corrections(window, qapp)

    window._on_guided_continue_to_feature_detection()
    qapp.processEvents()
    _open_correction_approach(window)
    qapp.processEvents()

    correction = window._guided_preview_chunk_combo
    feature = window._guided_feature_preview_segment_combo
    assert feature.count() > 1
    assert _identities(feature) == _identities(correction)

    native_plan = window._guided_continuous_rwd_native_segment_plan()
    assert native_plan is not None
    target_grid, segment_plan = native_plan
    cadence = target_grid.cadence_fraction
    data = [feature.itemData(index) for index in range(feature.count())]
    assert [int(item["continuous_window_index"]) for item in data] == [
        int(item.segment_index) for item in segment_plan.descriptors
    ]
    np.testing.assert_allclose(
        [float(item["window_start_sec"]) for item in data],
        [
            float(item.start_target_index * cadence)
            for item in segment_plan.descriptors
        ],
    )
    np.testing.assert_allclose(
        [float(item["window_end_sec"]) for item in data],
        [
            float(item.stop_target_index * cadence)
            for item in segment_plan.descriptors
        ],
    )

    # Selecting a middle window really moves the rows that are read.
    window._guided_feature_event_apply_btn.click()
    qapp.processEvents()
    real_compute = main_window_module.compute_guided_local_preview_dff_trace_in_memory
    captured = []

    def counted_compute(*args, **kwargs):
        captured.append(dict(kwargs))
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(
        main_window_module,
        "compute_guided_local_preview_dff_trace_in_memory",
        counted_compute,
    )

    # Window 0 already has retained Step 4 evidence, so these three force the
    # real bounded read rather than reuse.
    seen = []
    for index in (1, 3, feature.count() - 1):
        feature.setCurrentIndex(index)
        window._on_guided_generate_feature_detection_preview()
        qapp.processEvents()
        assert (
            window._guided_feature_preview_last_result is not None
        ), window._guided_feature_preview_status_label.text()
        selected = window._guided_feature_preview_on_demand_trace[
            "continuous_analysis_window"
        ]
        # What was read is the interval the chosen segment declares. The
        # declared end is exclusive, so the last sample read sits within one
        # cadence of it -- which is how the final segment ends at the
        # recording's last sample rather than a nominal multiple.
        declared = feature.itemData(index)
        cadence_sec = float(cadence)
        assert selected["window_index"] == index
        assert selected["window_start_sec"] == pytest.approx(
            declared["window_start_sec"]
        )
        assert (
            declared["window_end_sec"] - cadence_sec
            <= selected["window_end_sec"]
            <= declared["window_end_sec"]
        )
        assert selected["window_start_sec"] == pytest.approx(index * 200.0)
        seen.append((selected["row_start"], selected["row_stop"]))

    # Distinct, non-overlapping row ranges: each preview read its own interval.
    assert len(set(seen)) == 3
    assert all(stop > start for start, stop in seen)
    assert seen == sorted(seen)
    assert captured

    # Both ROIs stay mapped, and the ROI change keeps the selected window.
    assert [
        window._guided_feature_preview_roi_combo.itemText(index)
        for index in range(window._guided_feature_preview_roi_combo.count())
    ] == ["ROI1", "ROI2"]
    current = feature.currentIndex()
    window._guided_feature_preview_roi_combo.setCurrentIndex(1)
    qapp.processEvents()
    assert feature.currentIndex() == current


# ---------------------------------------------------------------------------
# The intermittent path is untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,structure",
    [
        ("auto", GUIDED_STRUCTURE_CHOICE_AUTO),
        ("rwd", "intermittent"),
    ],
)
def test_intermittent_still_previews_discovered_sessions(
    window, qapp, tmp_path, fmt, structure
):
    folder = _intermittent_folder(tmp_path / "sessions")
    _select_data(window, qapp, folder, fmt=fmt, structure=structure)
    _open_correction_approach(window)

    assert window._guided_effective_acquisition_mode() == "intermittent"
    assert window._guided_preview_source_ok is True
    assert window._guided_preview_source_type == "local_raw_segment"
    assert window._guided_preview_generate_btn.isEnabled() is True

    segments = _segments(window)
    assert len(segments) == 2
    for segment in segments:
        # Still one discovered session file per segment, with no window identity.
        assert "continuous_window_index" not in segment
        assert segment["adapter_chunk_index"] == 0
        assert segment["source_path"].endswith("Fluorescence.csv")


# ---------------------------------------------------------------------------
# Step 3 visibility
# ---------------------------------------------------------------------------


def test_continuous_recording_structure_hides_session_display_controls(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    window._sync_guided_recording_visibility()

    # The whole session-display group is hidden, so every control and
    # explanation inside it is off the page too.
    group = window._guided_tonic_settings_group
    assert group.isHidden()
    for control in (
        window._guided_tonic_output_mode_combo,
        window._guided_tonic_output_mode_help_label,
        window._guided_tonic_timeline_mode_combo,
        window._guided_tonic_timeline_mode_help_label,
        window._guided_tonic_gap_free_note_label,
        window._guided_tonic_gap_free_blocked_label,
    ):
        assert _is_inside(control, group)
        assert not control.isVisible()

    assert window._guided_sessions_per_hour_edit.isHidden()
    assert window._guided_session_duration_edit.isHidden()

    # The controls that do describe a continuous recording stay visible.
    assert not window._guided_continuous_window_sec_spin.isHidden()
    assert not window._guided_allow_partial_final_window_cb.isHidden()


def test_intermittent_recording_structure_keeps_session_display_controls(
    window, qapp, tmp_path
):
    folder = _intermittent_folder(tmp_path / "sessions")
    _select_data(window, qapp, folder)
    window._sync_guided_recording_visibility()

    assert not window._guided_tonic_settings_group.isHidden()
    assert not window._guided_tonic_output_mode_combo.isHidden()
    assert not window._guided_tonic_timeline_mode_combo.isHidden()
    assert not window._guided_sessions_per_hour_edit.isHidden()
    assert not window._guided_session_duration_edit.isHidden()
    assert window._guided_continuous_window_sec_spin.isHidden()


def test_hidden_session_display_controls_keep_their_defaults(
    window, qapp, tmp_path
):
    """Hiding them must not change what the plan carries."""
    folder = _continuous_folder(tmp_path / "rec")
    before_output = window._guided_tonic_output_mode_combo.currentData()
    before_timeline = window._guided_tonic_timeline_mode_combo.currentData()

    _select_data(window, qapp, folder)
    window._sync_guided_recording_visibility()

    assert window._guided_tonic_output_mode_combo.currentData() == before_output
    assert (
        window._guided_tonic_timeline_mode_combo.currentData() == before_timeline
    )


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_failed_continuous_preview_is_explained_and_retryable(
    window, qapp, tmp_path, monkeypatch
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    _select_all_preview_methods(window)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated window read failure")

    monkeypatch.setattr(
        main_window_module, "run_guided_local_correction_preview", boom
    )

    _generate_preview(window, qapp)

    # The failure is explained in the scientist's terms, and never blamed on
    # steps that are complete.
    message = window._guided_preview_status_label.text()
    assert message != GUIDED_PREVIEW_MISSING_PREREQUISITES_TEXT
    assert "Complete Select data" not in message
    assert "simulated window read failure" not in message

    # Controls are restored, nothing was confirmed, and it can be retried.
    assert window._guided_correction_preview_running is False
    assert window._guided_preview_generate_btn.isEnabled() is True
    assert window._guided_preview_has_result is False
    assert window._guided_confirm_locked_label.isHidden() is False
    assert getattr(window, "_guided_correction_preview_thread", None) is None

    monkeypatch.undo()
    assert _generate_preview(window, qapp)["status"] == "success"


def test_stale_continuous_preview_worker_cannot_reinstall_evidence(
    window, qapp, tmp_path
):
    """A worker started for an older setup is not allowed to report back."""
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    _select_all_preview_methods(window)
    assert _generate_preview(window, qapp)["status"] == "success"

    window._continuous_window_sec_spin.setValue(300.0)
    _open_correction_approach(window)

    # A second request while the setup no longer has an accepted recording is
    # refused outright rather than producing evidence for the old window.
    window._on_generate_guided_correction_preview()
    assert window._guided_correction_preview_running is False
    assert window._guided_preview_source_ok is False


def test_changing_the_window_length_invalidates_continuous_evidence(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    _select_all_preview_methods(window)
    assert _generate_preview(window, qapp)["status"] == "success"
    before = window._guided_local_preview_setup_signature()

    window._continuous_window_sec_spin.setValue(300.0)
    _open_correction_approach(window)

    assert window._guided_local_preview_setup_signature() != before
    # The accepted recording no longer covers the plan on screen, so no
    # evidence can be reused until the recording is checked again.
    assert window._guided_continuous_rwd_accepted_plan() is None
    assert window._guided_preview_source_ok is False
    # And the step cannot be left on the strength of the old window.
    assert window._guided_correction_approach_readiness()[0] is False


def test_changing_roi_inclusion_invalidates_continuous_evidence(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    _select_all_preview_methods(window)
    assert _generate_preview(window, qapp)["status"] == "success"
    before = window._guided_local_preview_setup_signature()

    window._guided_roi_list.item(1).setCheckState(Qt.Unchecked)
    qapp.processEvents()
    _open_correction_approach(window)

    assert window._guided_local_preview_setup_signature() != before
    assert window._guided_continuous_rwd_accepted_plan() is None
    assert window._guided_preview_source_ok is False
    assert window._guided_correction_approach_readiness()[0] is False


def test_changing_structure_clears_continuous_preview_segments(
    window, qapp, tmp_path
):
    folder = _continuous_folder(tmp_path / "rec")
    _select_data(window, qapp, folder)
    _check_recording(window, qapp)
    _open_correction_approach(window)
    assert _segments(window)

    index = window._guided_acquisition_mode_combo.findData("intermittent")
    window._guided_acquisition_mode_combo.setCurrentIndex(index)
    qapp.processEvents()
    _open_correction_approach(window)

    assert window._guided_preview_source_ok is False
    assert _segments(window) == []
