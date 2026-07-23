"""CR1-E1-B: focused GUI tests for continuous-RWD Results presentation.

Reuses the same accepted-case fixtures as
``tests/test_completed_continuous_rwd_review.py`` (CR1-E1-A) to build small
synthetic correction-only/tonic-only/phasic-only/combined completed runs,
then drives them through the actual `RunReportViewer` continuous branch --
never through a hand-built model, since the whole point of CR1-E1-B is that
the GUI only ever calls the CR1-E1-A loader.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
from PySide6.QtWidgets import QApplication

from gui.run_report_viewer import RunReportViewer
from photometry_pipeline import completed_continuous_rwd_review as subject
from photometry_pipeline.core import feature_extraction as feature_extraction_module
from photometry_pipeline.guided_continuous_rwd_combined_run import (
    execute_guided_continuous_rwd_combined_run,
)
from photometry_pipeline.guided_continuous_rwd_correction_run import (
    execute_guided_continuous_rwd_correction_run,
)
from photometry_pipeline.guided_continuous_rwd_phasic_run import (
    execute_guided_continuous_rwd_phasic_run,
)
from photometry_pipeline.guided_continuous_rwd_tonic_run import (
    execute_guided_continuous_rwd_tonic_run,
)

from tests.test_guided_continuous_rwd_correction_pass_persistence import (
    _build_case,
    _pass_inputs,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_e1b") / "recording"
    return _build_case(folder, continuous_window_sec=90.0)


@pytest.fixture(scope="module")
def real_config(accepted_case):
    from photometry_pipeline.guided_continuous_rwd_segment_correction import (
        _resolve_segment_correction_settings,
    )

    _binding, _grid, _draft, contract, _source = accepted_case
    config, _identity = _resolve_segment_correction_settings(contract)
    return dataclasses.replace(
        config,
        peak_threshold_method="percentile",
        peak_threshold_percentile=50.0,
        peak_min_distance_sec=1.0,
    )


def _positional_call(func, inputs, real_config, output_base):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    return func(
        binding,
        grid,
        block_plan,
        segment_plan,
        f0,
        accepted_draft=draft,
        startup_mapping_contract=contract,
        output_base=str(output_base),
        config=real_config,
    )


@pytest.fixture(scope="module")
def correction_only_run(accepted_case, real_config, tmp_path_factory):
    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e1b_correction")
    return _positional_call(
        execute_guided_continuous_rwd_correction_run, inputs, real_config, output_base
    )


@pytest.fixture(scope="module")
def tonic_only_run(accepted_case, real_config, tmp_path_factory):
    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e1b_tonic")
    return _positional_call(
        execute_guided_continuous_rwd_tonic_run, inputs, real_config, output_base
    )


@pytest.fixture(scope="module")
def phasic_only_run(accepted_case, real_config, tmp_path_factory):
    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e1b_phasic")
    return _positional_call(
        execute_guided_continuous_rwd_phasic_run, inputs, real_config, output_base
    )


@pytest.fixture(scope="module")
def combined_run(accepted_case, real_config, tmp_path_factory):
    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e1b_combined")
    return _positional_call(
        execute_guided_continuous_rwd_combined_run, inputs, real_config, output_base
    )


@pytest.fixture(scope="module")
def included_roi_ids(accepted_case):
    binding = accepted_case[0]
    return tuple(binding.recording.roi.included_roi_ids)


def _viewer(qapp) -> RunReportViewer:
    return RunReportViewer()


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_full_control_load_report_routes_continuous_runs_to_continuous_branch(
    qapp, combined_run
):
    viewer = _viewer(qapp)
    loaded = viewer.load_report(combined_run.run_dir)
    assert loaded is True
    assert not viewer._continuous_workspace.isHidden()
    assert viewer._workspace.isHidden()
    assert viewer._continuous_overview is not None


def test_guided_entry_point_uses_same_continuous_branch_via_preloaded_overview(
    qapp, combined_run
):
    """Mirrors the Guided worker: overview is loaded once (off the GUI
    thread in production) and handed to load_continuous_results directly,
    without ever calling the intermittent loader."""
    overview = subject.load_continuous_run_overview(combined_run.run_dir)
    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(combined_run.run_dir, overview)
    assert loaded is True
    assert not viewer._continuous_workspace.isHidden()
    assert viewer._workspace.isHidden()


def test_continuous_loader_refusal_shows_error_and_does_not_fall_back(
    qapp, tonic_only_run, tmp_path_factory
):
    import shutil

    broken = tmp_path_factory.mktemp("cr1_e1b_broken_tonic")
    shutil.copytree(tonic_only_run.run_dir, str(broken), dirs_exist_ok=True)
    os.remove(
        os.path.join(str(broken), "_analysis", "tonic_out", "tonic_trace_cache.h5")
    )

    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(str(broken))
    assert loaded is False
    assert viewer._continuous_workspace.isHidden()
    assert viewer._workspace.isHidden()
    assert "tonic" in viewer._status_label.text().lower()
    # No traceback as the primary message.
    assert "Traceback" not in viewer._status_label.text()


# ---------------------------------------------------------------------------
# Correction-only presentation
# ---------------------------------------------------------------------------


def test_correction_only_presentation(qapp, correction_only_run, included_roi_ids):
    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(correction_only_run.run_dir)
    assert loaded is True

    overview_text = viewer._continuous_overview_label.text()
    assert "Continuous recording" in overview_text
    assert "Correction: Completed" in overview_text
    assert "Tonic analysis: Not run" in overview_text
    assert "Phasic event analysis: Not run" in overview_text
    for roi_id in included_roi_ids:
        assert roi_id in overview_text

    assert viewer._continuous_tabs.count() == 0
    assert viewer._continuous_tabs.isHidden()
    assert viewer._continuous_roi_row.isHidden()
    assert "session" not in overview_text.lower()


# ---------------------------------------------------------------------------
# Tonic-only presentation
# ---------------------------------------------------------------------------


def test_tonic_only_presentation(qapp, tonic_only_run, included_roi_ids):
    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(tonic_only_run.run_dir)
    assert loaded is True

    assert not viewer._continuous_roi_row.isHidden()
    combo_items = [
        viewer._continuous_roi_combo.itemText(i)
        for i in range(viewer._continuous_roi_combo.count())
    ]
    assert tuple(combo_items) == included_roi_ids

    tab_labels = [
        viewer._continuous_tabs.tabText(i) for i in range(viewer._continuous_tabs.count())
    ]
    assert tab_labels == ["Tonic"]

    assert not viewer._continuous_tonic_interaction._pixmap.isNull()
    assert viewer._continuous_tonic_summary_table.rowCount() > 0

    overview = subject.load_continuous_run_overview(tonic_only_run.run_dir)
    roi_id = included_roi_ids[0]
    assert viewer._continuous_tonic_summary_table.rowCount() == (
        overview.tonic_window_row_counts[roi_id]
    )


def test_tonic_only_final_short_window_present(qapp, tonic_only_run):
    viewer = _viewer(qapp)
    viewer.load_continuous_results(tonic_only_run.run_dir)
    table = viewer._continuous_tonic_summary_table
    headers = [
        table.horizontalHeaderItem(i).text() for i in range(table.columnCount())
    ]
    duration_col = headers.index("window_duration_sec")
    first_duration = float(table.item(0, duration_col).text())
    last_duration = float(table.item(table.rowCount() - 1, duration_col).text())
    assert last_duration < first_duration


def test_tonic_only_roi_switch_replaces_trace_and_summary(
    qapp, tonic_only_run, included_roi_ids
):
    assert len(included_roi_ids) >= 2
    viewer = _viewer(qapp)
    viewer.load_continuous_results(tonic_only_run.run_dir)

    first_pixmap = viewer._continuous_tonic_interaction._pixmap
    first_bytes = bytes(first_pixmap.toImage().constBits())

    viewer._continuous_roi_combo.setCurrentIndex(1)
    second_pixmap = viewer._continuous_tonic_interaction._pixmap
    second_bytes = bytes(second_pixmap.toImage().constBits())

    assert viewer._continuous_roi_combo.currentText() == included_roi_ids[1]
    assert first_bytes != second_bytes


def test_tonic_only_never_calls_phasic_loader(qapp, tonic_only_run, monkeypatch):
    import gui.run_report_viewer as rrv

    calls = {"count": 0}
    real_fn = rrv.load_continuous_phasic_events

    def guarded(*args, **kwargs):
        calls["count"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(rrv, "load_continuous_phasic_events", guarded)
    viewer = _viewer(qapp)
    viewer.load_continuous_results(tonic_only_run.run_dir)
    assert calls["count"] == 0


# ---------------------------------------------------------------------------
# Phasic-only presentation
# ---------------------------------------------------------------------------


def test_phasic_only_presentation(qapp, phasic_only_run, included_roi_ids):
    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(phasic_only_run.run_dir)
    assert loaded is True

    tab_labels = [
        viewer._continuous_tabs.tabText(i) for i in range(viewer._continuous_tabs.count())
    ]
    assert tab_labels == ["Phasic"]
    assert not viewer._continuous_roi_row.isHidden()

    overview = subject.load_continuous_run_overview(phasic_only_run.run_dir)
    roi_id = included_roi_ids[0]

    assert not viewer._continuous_phasic_interaction._pixmap.isNull()
    count_text = viewer._continuous_phasic_event_count_label.text()
    assert str(overview.phasic_event_counts_by_roi[roi_id]) in count_text

    summary_rows = viewer._continuous_phasic_summary_table.rowCount()
    assert summary_rows == overview.phasic_window_row_counts[roi_id]


def test_phasic_only_final_short_window_present(qapp, phasic_only_run):
    viewer = _viewer(qapp)
    viewer.load_continuous_results(phasic_only_run.run_dir)
    table = viewer._continuous_phasic_summary_table
    headers = [
        table.horizontalHeaderItem(i).text() for i in range(table.columnCount())
    ]
    duration_col = headers.index("window_duration_sec")
    first_duration = float(table.item(0, duration_col).text())
    last_duration = float(table.item(table.rowCount() - 1, duration_col).text())
    assert last_duration < first_duration


def test_phasic_only_never_invokes_event_detector(qapp, phasic_only_run, monkeypatch, included_roi_ids):
    calls = {"count": 0}
    real_fn = feature_extraction_module.get_peak_indices_for_trace

    def guarded(*args, **kwargs):
        calls["count"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(feature_extraction_module, "get_peak_indices_for_trace", guarded)

    viewer = _viewer(qapp)
    viewer.load_continuous_results(phasic_only_run.run_dir)
    for idx in range(len(included_roi_ids)):
        viewer._continuous_roi_combo.setCurrentIndex(idx)

    assert calls["count"] == 0


def test_phasic_only_never_calls_tonic_loader(qapp, phasic_only_run, monkeypatch):
    import gui.run_report_viewer as rrv

    calls = {"count": 0}
    real_fn = rrv.load_continuous_roi_trace

    def guarded(run_dir, *, family, roi_id):
        if family == "tonic":
            calls["count"] += 1
        return real_fn(run_dir, family=family, roi_id=roi_id)

    monkeypatch.setattr(rrv, "load_continuous_roi_trace", guarded)
    viewer = _viewer(qapp)
    viewer.load_continuous_results(phasic_only_run.run_dir)
    assert calls["count"] == 0


# ---------------------------------------------------------------------------
# Combined presentation
# ---------------------------------------------------------------------------


def test_combined_presentation(qapp, combined_run, included_roi_ids):
    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(combined_run.run_dir)
    assert loaded is True

    tab_labels = [
        viewer._continuous_tabs.tabText(i) for i in range(viewer._continuous_tabs.count())
    ]
    assert tab_labels == ["Tonic", "Phasic"]

    combo_items = [
        viewer._continuous_roi_combo.itemText(i)
        for i in range(viewer._continuous_roi_combo.count())
    ]
    assert tuple(combo_items) == included_roi_ids

    assert not viewer._continuous_tonic_interaction._pixmap.isNull()
    assert not viewer._continuous_phasic_interaction._pixmap.isNull()
    assert viewer._continuous_tonic_summary_table.rowCount() > 0
    assert viewer._continuous_phasic_summary_table.rowCount() > 0


def test_combined_final_short_window_in_both_families(qapp, combined_run):
    viewer = _viewer(qapp)
    viewer.load_continuous_results(combined_run.run_dir)
    for table in (
        viewer._continuous_tonic_summary_table,
        viewer._continuous_phasic_summary_table,
    ):
        headers = [
            table.horizontalHeaderItem(i).text() for i in range(table.columnCount())
        ]
        duration_col = headers.index("window_duration_sec")
        first_duration = float(table.item(0, duration_col).text())
        last_duration = float(table.item(table.rowCount() - 1, duration_col).text())
        assert last_duration < first_duration


def test_combined_roi_switch_refreshes_both_families(qapp, combined_run, included_roi_ids):
    assert len(included_roi_ids) >= 2
    viewer = _viewer(qapp)
    viewer.load_continuous_results(combined_run.run_dir)

    tonic_before = bytes(
        viewer._continuous_tonic_interaction._pixmap.toImage().constBits()
    )
    phasic_before = bytes(
        viewer._continuous_phasic_interaction._pixmap.toImage().constBits()
    )

    viewer._continuous_roi_combo.setCurrentIndex(1)

    tonic_after = bytes(
        viewer._continuous_tonic_interaction._pixmap.toImage().constBits()
    )
    phasic_after = bytes(
        viewer._continuous_phasic_interaction._pixmap.toImage().constBits()
    )
    assert tonic_before != tonic_after
    assert phasic_before != phasic_after


def test_combined_never_reruns_correction_or_detection(qapp, combined_run, monkeypatch):
    from photometry_pipeline import guided_continuous_rwd_correction_pass as c4c_module
    from photometry_pipeline import guided_continuous_rwd_phasic_detection as detection_module

    def flaky_traversal(*args, **kwargs):
        raise AssertionError("continuous Results must never rerun correction")

    def flaky_detection(*args, **kwargs):
        raise AssertionError("continuous Results must never rerun detection")

    monkeypatch.setattr(
        c4c_module, "iterate_guided_continuous_rwd_corrected_segments", flaky_traversal
    )
    monkeypatch.setattr(
        detection_module, "detect_guided_continuous_rwd_phasic_features", flaky_detection
    )

    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(combined_run.run_dir)
    assert loaded is True


# ---------------------------------------------------------------------------
# No false session semantics
# ---------------------------------------------------------------------------


def test_no_false_session_semantics_anywhere_in_continuous_presentation(
    qapp, combined_run
):
    viewer = _viewer(qapp)
    viewer.load_continuous_results(combined_run.run_dir)

    forbidden = ("session", "579 session")
    texts = [viewer._continuous_overview_label.text()]
    for table in (
        viewer._continuous_tonic_summary_table,
        viewer._continuous_phasic_summary_table,
    ):
        for col in range(table.columnCount()):
            header = table.horizontalHeaderItem(col)
            if header is not None:
                texts.append(header.text())
    combined_text = "\n".join(texts).lower()
    for word in forbidden:
        assert word not in combined_text


# ---------------------------------------------------------------------------
# Malformed artifacts / clear errors
# ---------------------------------------------------------------------------


def test_missing_event_csv_refuses_without_redetecting(
    qapp, phasic_only_run, tmp_path_factory, monkeypatch
):
    import shutil

    broken = tmp_path_factory.mktemp("cr1_e1b_broken_events")
    shutil.copytree(phasic_only_run.run_dir, str(broken), dirs_exist_ok=True)
    os.remove(
        os.path.join(
            str(broken), "_analysis", "phasic_out", "features", "continuous_phasic_events.csv"
        )
    )

    calls = {"count": 0}
    real_fn = feature_extraction_module.get_peak_indices_for_trace

    def guarded(*args, **kwargs):
        calls["count"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(feature_extraction_module, "get_peak_indices_for_trace", guarded)

    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(str(broken))
    assert loaded is False
    assert viewer._continuous_workspace.isHidden()
    assert calls["count"] == 0
    assert "event" in viewer._status_label.text().lower()


def test_failed_run_is_never_presented_as_completed(qapp, tmp_path):
    import json

    run_dir = tmp_path / "not_a_real_run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "bogus",
                "phase": "final",
                "status": "error",
                "errors": ["simulated failure"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(str(run_dir))
    assert loaded is False
    assert viewer._continuous_workspace.isHidden()


# ---------------------------------------------------------------------------
# Routing predicate: accepted CR1 profiles vs. the legacy continuous-
# acquisition profile that shares acquisition_mode/deliverable_profile
# ---------------------------------------------------------------------------


def test_is_continuous_rwd_run_mode_predicate_distinguishes_legacy_profile():
    from gui.run_report_parser import is_continuous_rwd_run_mode

    for profile in (
        "guided_continuous_rwd_correction",
        "guided_continuous_rwd_tonic",
        "guided_continuous_rwd_phasic",
        "guided_continuous_rwd_combined",
    ):
        assert is_continuous_rwd_run_mode({"run_profile": profile}) is True

    # The legacy chunked "continuous acquisition" full-pipeline mode shares
    # acquisition_mode == "continuous" and deliverable_profile == "continuous"
    # with the accepted CR1 producers, but uses run_profile == "full" -- it
    # must NOT route into the continuous branch.
    legacy_run_mode = {
        "run_profile": "full",
        "acquisition_mode": "continuous",
        "deliverable_profile": "continuous",
        "continuous_outputs_ran": True,
    }
    assert is_continuous_rwd_run_mode(legacy_run_mode) is False
    assert is_continuous_rwd_run_mode({}) is False


# ---------------------------------------------------------------------------
# Atomic selected-ROI loading: initial-load failures
# ---------------------------------------------------------------------------


def test_initial_tonic_trace_read_failure_fails_closed(qapp, tonic_only_run, monkeypatch):
    import gui.run_report_viewer as rrv

    def failing_trace(*args, **kwargs):
        raise subject.CompletedContinuousRwdReviewError(
            "This completed analysis's tonic trace could not be read."
        )

    monkeypatch.setattr(rrv, "load_continuous_roi_trace", failing_trace)

    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(tonic_only_run.run_dir)

    assert loaded is False
    assert viewer._continuous_workspace.isHidden()
    assert viewer._workspace.isHidden()
    assert "tonic trace could not be read" in viewer._status_label.text()
    assert "Traceback" not in viewer._status_label.text()
    # No partial pixmap or table left over from the failed attempt.
    assert viewer._continuous_tonic_interaction._pixmap.isNull()
    assert viewer._continuous_tonic_summary_table.rowCount() == 0


def test_initial_phasic_failure_fails_closed_with_zero_redetection(
    qapp, phasic_only_run, monkeypatch
):
    import gui.run_report_viewer as rrv

    def failing_summary(*args, **kwargs):
        raise subject.CompletedContinuousRwdReviewError(
            "This completed analysis's phasic summary could not be read."
        )

    monkeypatch.setattr(rrv, "load_continuous_window_summary", failing_summary)

    calls = {"count": 0}
    real_fn = feature_extraction_module.get_peak_indices_for_trace

    def guarded(*args, **kwargs):
        calls["count"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(feature_extraction_module, "get_peak_indices_for_trace", guarded)

    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(phasic_only_run.run_dir)

    assert loaded is False
    assert viewer._continuous_workspace.isHidden()
    assert viewer._workspace.isHidden()
    assert "phasic summary could not be read" in viewer._status_label.text()
    assert calls["count"] == 0
    assert viewer._continuous_phasic_interaction._pixmap.isNull()
    assert viewer._continuous_phasic_summary_table.rowCount() == 0
    assert viewer._continuous_phasic_event_count_label.text() == ""


# ---------------------------------------------------------------------------
# Atomic selected-ROI loading: combined ROI-switch failure
# ---------------------------------------------------------------------------


def test_combined_roi_switch_failure_never_shows_mixed_roi_state(
    qapp, combined_run, included_roi_ids, monkeypatch
):
    import gui.run_report_viewer as rrv

    assert len(included_roi_ids) >= 2
    first_roi, second_roi = included_roi_ids[0], included_roi_ids[1]

    viewer = _viewer(qapp)
    loaded = viewer.load_continuous_results(combined_run.run_dir)
    assert loaded is True

    tonic_before = bytes(
        viewer._continuous_tonic_interaction._pixmap.toImage().constBits()
    )
    phasic_before = bytes(
        viewer._continuous_phasic_interaction._pixmap.toImage().constBits()
    )
    event_count_before = viewer._continuous_phasic_event_count_label.text()

    real_trace_loader = rrv.load_continuous_roi_trace

    def selective_failure(run_dir, *, family, roi_id):
        # Tonic succeeds for the newly selected ROI; phasic fails -- the UI
        # must not end up showing ROI2's tonic trace next to ROI1's phasic
        # trace/events.
        if family == "phasic" and roi_id == second_roi:
            raise subject.CompletedContinuousRwdReviewError(
                "This completed analysis's phasic trace could not be read."
            )
        return real_trace_loader(run_dir, family=family, roi_id=roi_id)

    monkeypatch.setattr(rrv, "load_continuous_roi_trace", selective_failure)

    viewer._continuous_roi_combo.setCurrentIndex(1)

    # Selector must be restored to the last fully-successful ROI, and both
    # displays must remain exactly what they were before the failed switch
    # -- never ROI2's tonic trace paired with ROI1's phasic trace/events.
    assert viewer._continuous_roi_combo.currentText() == first_roi
    tonic_after = bytes(
        viewer._continuous_tonic_interaction._pixmap.toImage().constBits()
    )
    phasic_after = bytes(
        viewer._continuous_phasic_interaction._pixmap.toImage().constBits()
    )
    assert tonic_after == tonic_before
    assert phasic_after == phasic_before
    assert viewer._continuous_phasic_event_count_label.text() == event_count_before
    assert first_roi in event_count_before
    assert second_roi not in viewer._continuous_phasic_event_count_label.text()
    assert "phasic trace could not be read" in viewer._status_label.text()


def test_recovery_after_roi_switch_failure_allows_later_valid_switch(
    qapp, combined_run, included_roi_ids, monkeypatch
):
    import gui.run_report_viewer as rrv

    assert len(included_roi_ids) >= 2
    second_roi = included_roi_ids[1]

    viewer = _viewer(qapp)
    viewer.load_continuous_results(combined_run.run_dir)

    real_trace_loader = rrv.load_continuous_roi_trace

    def selective_failure(run_dir, *, family, roi_id):
        if family == "phasic" and roi_id == second_roi:
            raise subject.CompletedContinuousRwdReviewError("simulated failure")
        return real_trace_loader(run_dir, family=family, roi_id=roi_id)

    monkeypatch.setattr(rrv, "load_continuous_roi_trace", selective_failure)
    viewer._continuous_roi_combo.setCurrentIndex(1)
    assert viewer._continuous_roi_combo.currentText() == included_roi_ids[0]

    monkeypatch.setattr(rrv, "load_continuous_roi_trace", real_trace_loader)
    viewer._continuous_roi_combo.setCurrentIndex(1)
    assert viewer._continuous_roi_combo.currentText() == second_roi
    assert not viewer._continuous_tonic_interaction._pixmap.isNull()
    assert not viewer._continuous_phasic_interaction._pixmap.isNull()
    assert second_roi in viewer._continuous_phasic_event_count_label.text()
