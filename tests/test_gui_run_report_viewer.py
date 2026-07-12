"""
Tests for RunReportViewer and run_report_parser dynamic discovery.
"""

import os
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap
from gui.run_report_parser import (
    resolve_region_deliverables,
    resolve_internal_artifacts,
    resolve_primary_artifacts,
    is_successful_completed_run_dir,
    get_scientist_completion_summary,
)

from PySide6.QtWidgets import QApplication
from gui.run_report_viewer import RunReportViewer
from tests.terminal_run_fixtures import legacy_run_report, write_current_run
from types import SimpleNamespace


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])

def test_discover_region_deliverables_dynamic():
    """Verify that the parser finds arbitrary region folders with semantic subfolders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create Region A with 3 subfolders
        reg_a = os.path.join(tmpdir, "RegionA")
        os.makedirs(os.path.join(reg_a, "summary"))
        os.makedirs(os.path.join(reg_a, "day_plots"))
        os.makedirs(os.path.join(reg_a, "tables"))
        
        # Create Region B with 2 subfolders
        reg_b = os.path.join(tmpdir, "RegionB")
        os.makedirs(os.path.join(reg_b, "summary"))
        os.makedirs(os.path.join(reg_b, "tables"))
        
        # Create a non-region folder
        os.makedirs(os.path.join(tmpdir, "some_other_folder"))
        
        # Create an internal folder (should be skipped by region discovery)
        os.makedirs(os.path.join(tmpdir, "_analysis"))
        
        regions = resolve_region_deliverables(tmpdir)
        
        assert len(regions) == 2
        names = [r['name'] for r in regions]
        assert "RegionA" in names
        assert "RegionB" in names
        
        reg_a_data = next(r for r in regions if r['name'] == "RegionA")
        assert len(reg_a_data['subfolders']) == 3
        
        reg_b_data = next(r for r in regions if r['name'] == "RegionB")
        assert len(reg_b_data['subfolders']) == 2
        labels_b = [f[0] for f in reg_b_data['subfolders']]
        assert "Summary" in labels_b
        assert "Tables" in labels_b
        assert "Day Plots" not in labels_b

def test_internal_analysis_links_optional():
    """Verify that _analysis subfolders are discovered."""
    with tempfile.TemporaryDirectory() as tmpdir:
        analysis_dir = os.path.join(tmpdir, "_analysis")
        os.makedirs(os.path.join(analysis_dir, "phasic_out"))
        
        internal = resolve_internal_artifacts(tmpdir)
        assert len(internal) == 1
        assert internal[0][0] == "Phasic Analysis (Internal)"
        
        os.makedirs(os.path.join(analysis_dir, "tonic_out"))
        internal = resolve_internal_artifacts(tmpdir)
        assert len(internal) == 2

def test_no_obsolete_primary_quick_links():
    """Ensure we don't return hardcoded traces/features/etc anymore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an old-style traces folder
        os.makedirs(os.path.join(tmpdir, "traces"))
        
        # The new primary artifacts resolver should NOT find it
        primary = resolve_primary_artifacts(tmpdir, {})
        labels = [p[0] for p in primary]
        assert "Traces Folder" not in labels
        
        # And it shouldn't show up in regions if it doesn't have the subfolders
        regions = resolve_region_deliverables(tmpdir)
        assert len(regions) == 0

def test_primary_artifacts_resolver():
    """Verify root-level artifacts are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "status.json"), 'w') as f: f.write('{}')
        with open(os.path.join(tmpdir, "MANIFEST.json"), 'w') as f: f.write('{}')
        
        primary = resolve_primary_artifacts(tmpdir, {})
        assert len(primary) == 2
        labels = [p[0] for p in primary]
        assert "Run Status" in labels
        assert "Output Manifest" in labels


def test_successful_completed_run_dir_rejects_artifacts_only():
    """Artifact presence alone must not be treated as completed success."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reg = os.path.join(tmpdir, "Region0")
        os.makedirs(os.path.join(reg, "summary"))
        os.makedirs(os.path.join(reg, "day_plots"))
        os.makedirs(os.path.join(reg, "tables"))
        # No status.json success, no manifest success, no explicit run_report success fields
        with open(os.path.join(tmpdir, "run_report.json"), "w", encoding="utf-8") as f:
            json.dump({"run_context": {"run_type": "full"}}, f)
        ok, reason = is_successful_completed_run_dir(tmpdir)
        assert ok is False
        assert "complete record of a finished run" in reason.lower()


def test_successful_completed_run_dir_accepts_verified_terminal_set():
    """A coherent current terminal set allows complete-state entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        write_current_run(Path(tmpdir) / "run")
        ok, reason = is_successful_completed_run_dir(os.path.join(tmpdir, "run"))
        assert ok is True
        assert "verified" in reason


def test_successful_completed_run_dir_rejects_status_success_alone():
    """A success status with no report and no manifest verifies nothing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "status.json"), "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "phase": "final", "status": "success"}, f)
        ok, _reason = is_successful_completed_run_dir(tmpdir)
        assert ok is False


def test_run_report_viewer_tab_discovery_is_explicit(qapp):
    """Tab discovery should not pick unrelated wildcard files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = os.path.join(tmpdir, "summary")
        day_plots = os.path.join(tmpdir, "day_plots")
        os.makedirs(summary)
        os.makedirs(day_plots)

        # Canonical expected
        open(os.path.join(summary, "phasic_correction_impact.png"), "wb").close()
        open(os.path.join(summary, "tonic_overview.png"), "wb").close()
        open(os.path.join(summary, "phasic_auc_timeseries.png"), "wb").close()
        open(os.path.join(summary, "phasic_peak_rate_timeseries.png"), "wb").close()
        open(os.path.join(summary, "continuous_tonic_trace_overview.png"), "wb").close()
        open(os.path.join(summary, "continuous_phasic_dff_trace_overview.png"), "wb").close()
        open(os.path.join(day_plots, "phasic_sig_iso_day_000.png"), "wb").close()
        open(os.path.join(day_plots, "phasic_dynamic_fit_day_000.png"), "wb").close()
        open(os.path.join(day_plots, "phasic_dFF_day_000.png"), "wb").close()
        open(os.path.join(day_plots, "phasic_stacked_day_000.png"), "wb").close()

        # Should be ignored by tightened rules
        open(os.path.join(summary, "my_verification_notes.png"), "wb").close()
        open(os.path.join(summary, "tonic_anything.png"), "wb").close()
        open(os.path.join(summary, "phasic_random_timeseries_plot.png"), "wb").close()
        open(os.path.join(day_plots, "phasic_sig_iso_day_extra.png"), "wb").close()

        viewer = RunReportViewer()
        tab_map = viewer._discover_region_tab_images(tmpdir)

        assert [os.path.basename(p) for p in tab_map["Verification"]] == ["phasic_correction_impact.png"]
        assert [os.path.basename(p) for p in tab_map["Tonic"]] == ["tonic_overview.png"]
        assert [os.path.basename(p) for p in tab_map["Phasic Summary"]] == [
            "phasic_auc_timeseries.png",
            "phasic_peak_rate_timeseries.png",
        ]
        assert [os.path.basename(p) for p in tab_map["Continuous Trace"]] == [
            "continuous_phasic_dff_trace_overview.png",
            "continuous_tonic_trace_overview.png",
        ]
        assert [os.path.basename(p) for p in tab_map["Phasic Sig/Iso"]] == ["phasic_sig_iso_day_000.png"]
        assert [os.path.basename(p) for p in tab_map["Dynamic Fit"]] == ["phasic_dynamic_fit_day_000.png"]
        assert [os.path.basename(p) for p in tab_map["Phasic dFF"]] == ["phasic_dFF_day_000.png"]
        assert [os.path.basename(p) for p in tab_map["Phasic Stacked"]] == ["phasic_stacked_day_000.png"]
        assert "Phasic Raw" not in tab_map


def test_run_report_viewer_status_labels_tuning_prep(qapp):
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "run_report.json"), "w", encoding="utf-8") as f:
            json.dump({"run_context": {"run_type": "tuning_prep"}}, f)
        reg = os.path.join(tmpdir, "Region0")
        os.makedirs(os.path.join(reg, "summary"))
        os.makedirs(os.path.join(reg, "day_plots"))
        os.makedirs(os.path.join(reg, "tables"))

        viewer = RunReportViewer()
        assert viewer.load_report(tmpdir) is True
        assert "[TUNING PREP]" in viewer._status_label.text()
        viewer.close()


def test_positive_legacy_phasic_review_loads_and_viewer_opens(qapp, tmp_path):
    run_dir = tmp_path / "legacy_phasic"
    run_dir.mkdir()
    (run_dir / "run_report.json").write_text(
        json.dumps(
            legacy_run_report(
                run_context={"run_type": "full", "status": "success", "phase": "final"}
            )
        ),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"phase": "final", "status": "success"}), encoding="utf-8"
    )
    (run_dir / "Region0" / "summary").mkdir(parents=True)
    (run_dir / "Region0" / "summary" / "phasic_correction_impact.png").write_bytes(b"")
    phasic_dir = run_dir / "_analysis" / "phasic_out"
    phasic_dir.mkdir(parents=True)
    (phasic_dir / "config_used.yaml").write_text(
        "dynamic_fit_mode: global_linear_regression\n", encoding="utf-8"
    )
    with h5py.File(phasic_dir / "phasic_trace_cache.h5", "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray(["Region0"], dtype="S"))
        meta.create_dataset("chunk_ids", data=np.asarray([0], dtype=int))
        group = handle.create_group("roi/Region0/chunk_0")
        time_sec = np.arange(10, dtype=float)
        group.create_dataset("time_sec", data=time_sec)
        group.create_dataset("sig_raw", data=np.linspace(1.0, 2.0, 10))
        group.create_dataset("dff", data=np.linspace(0.0, 1.0, 10))
        group.create_dataset("fit_ref", data=np.linspace(1.0, 1.5, 10))

    from photometry_pipeline.completed_run_review import load_completed_phasic_review

    model = load_completed_phasic_review(run_dir)
    assert model.current_native is False
    assert model.sessions_for_roi("Region0")[0].strategy_family == "dynamic_fit"
    assert model.sessions_for_roi("Region0")[0].fitted_reference is not None
    assert model.sessions_for_roi("Region0")[0].production_f0_baseline is None
    assert model.strategy_label_for_roi("Region0") == "Global linear regression"

    viewer = RunReportViewer()
    assert viewer.load_report(str(run_dir)) is True
    qapp.processEvents()
    assert viewer.phasic_review_model is not None
    assert viewer.phasic_review_model.current_native is False
    assert "Global linear regression" in viewer._correction_summary_label.text()
    assert "legacy result" in viewer._selected_feature_settings_label.text()
    assert "Verification" in viewer._region_tab_images["Region0"]
    viewer.close()


def test_mixed_native_review_switch_updates_strategy_and_feature_settings(qapp, tmp_path):
    from photometry_pipeline.config import Config
    from photometry_pipeline.pipeline import Pipeline
    from photometry_pipeline.run_completion_contract import (
        PROFILE_TUNING_PREP,
        normalize_run_mode,
    )
    from tests.test_run_completion_correction_provenance import (
        _mixed_map,
        _root_for_case,
        _write_source,
        _write_terminal_set,
    )

    source = tmp_path / "input" / "2024_01_01-00_00_00" / "fluorescence.csv"
    _write_source(source)
    analysis = tmp_path / "mixed_analysis"
    config = Config(
        target_fs_hz=10.0,
        chunk_duration_sec=20.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        lowpass_hz=2.0,
        filter_order=2,
        signal_only_f0_min_window_samples=21,
    )
    Pipeline(config, mode="phasic", per_roi_correction=_mixed_map()).run(
        str(source.parent.parent), str(analysis), force_format="rwd", recursive=True
    )
    root = _root_for_case(tmp_path, analysis, "mixed_native_review")
    for roi in ("Region0", "Region1"):
        (root / roi / "summary").mkdir(parents=True, exist_ok=True)
        (root / roi / "summary" / "phasic_correction_impact.png").write_bytes(b"")
        (root / roi / "day_plots").mkdir(parents=True, exist_ok=True)
        (root / roi / "day_plots" / "phasic_dynamic_fit_day_000.png").write_bytes(b"")
    feature_path = root / "_analysis" / "phasic_out" / "features" / "feature_event_provenance.json"
    feature_path.write_text(
        json.dumps(
            {
                "schema_version": "guided_feature_event_provenance.v3",
                "rois": [
                    {
                        "roi": "Region0",
                        "source": "default",
                        "feature_event_profile_id": "default",
                        "effective_config_fields": {
                            "event_signal": "dff",
                            "peak_threshold_method": "mean_std",
                            "peak_threshold_k": 2.0,
                        },
                    },
                    {
                        "roi": "Region1",
                        "source": "override",
                        "feature_event_profile_id": "custom",
                        "effective_config_fields": {
                            "event_signal": "delta_f",
                            "peak_threshold_method": "absolute",
                            "peak_threshold_abs": 0.5,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    mode = normalize_run_mode(
        run_profile="tuning_prep",
        run_type="full",
        acquisition_mode="intermittent",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_TUNING_PREP,
        expected_rois=["Region0", "Region1"],
        chunked_input_processing=True,
        shared_input_manifest=False,
    )
    _write_terminal_set(root, mode)

    viewer = RunReportViewer()
    assert viewer.load_report(str(root)) is True
    qapp.processEvents()
    combo_names = [viewer._region_combo.itemText(i) for i in range(viewer._region_combo.count())]
    assert combo_names == ["Region0", "Region1"]

    viewer._region_combo.setCurrentIndex(0)
    qapp.processEvents()
    assert "Global linear regression" in viewer._correction_summary_label.text()
    assert "mean" in viewer._selected_feature_settings_label.text()
    assert "threshold" in viewer._selected_feature_settings_label.text()
    assert viewer._region_tab_images["Region0"]["Verification"]
    assert "Dynamic Fit" in viewer.available_view_tabs()
    assert viewer.phasic_review_model.sessions_for_roi("Region0")[0].fitted_reference is not None

    viewer._region_combo.setCurrentIndex(1)
    qapp.processEvents()
    assert "Signal-Only F0" in viewer._correction_summary_label.text()
    assert "absolute threshold" in viewer._selected_feature_settings_label.text()
    assert "Baseline support:" in viewer._correction_summary_label.text()
    assert viewer._region_tab_images["Region1"]["Verification"]
    assert "Correction Reference" in viewer.available_view_tabs()
    assert "Dynamic Fit" not in viewer.available_view_tabs()
    signal_only = viewer.phasic_review_model.sessions_for_roi("Region1")[0]
    assert signal_only.production_f0_baseline is not None
    assert signal_only.fitted_reference is None
    assert viewer.applied_dff_state.present is False
    viewer.close()


def test_run_report_viewer_click_to_zoom_toggle(qapp):
    """Clicking image toggles fit mode and full-size inspection mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "tall_plot.png")
        pix = QPixmap(900, 1800)
        pix.fill(Qt.red)
        assert pix.save(img_path)

        viewer = RunReportViewer()
        viewer.resize(1200, 800)
        viewer.show()
        qapp.processEvents()

        viewer._active_image_path = img_path
        viewer._set_image(img_path)
        qapp.processEvents()

        assert viewer._zoom_mode is False
        fit_h = viewer._image_label.pixmap().height()
        assert fit_h <= viewer._image_scroll.viewport().height()

        viewer._on_image_clicked()
        qapp.processEvents()
        assert viewer._zoom_mode is True
        assert viewer._image_label.pixmap().height() == 1800
        assert viewer._image_label.height() > viewer._image_scroll.viewport().height()

        viewer._on_image_clicked()
        qapp.processEvents()
        assert viewer._zoom_mode is False
        assert viewer._image_label.pixmap().height() <= viewer._image_scroll.viewport().height()

        viewer.close()


def test_run_report_viewer_wheel_zoom_is_incremental_and_pan_is_usable(qapp):
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "inspect_plot.png")
        pix = QPixmap(2200, 1200)
        pix.fill(Qt.blue)
        assert pix.save(img_path)

        viewer = RunReportViewer()
        viewer.resize(1200, 800)
        viewer.show()
        qapp.processEvents()

        viewer._active_image_path = img_path
        viewer._set_image(img_path)
        qapp.processEvents()

        fit_pix = viewer._image_label.pixmap()
        assert fit_pix is not None and not fit_pix.isNull()
        fit_w = fit_pix.width()

        viewer._image_label.wheel_zoom.emit(1, QPoint(100, 100))
        qapp.processEvents()
        zoomed_pix = viewer._image_label.pixmap()
        assert zoomed_pix is not None and not zoomed_pix.isNull()
        assert viewer._zoom_mode is True
        assert fit_w < zoomed_pix.width() < 2200

        assert viewer._image_label.cursor().shape() == Qt.OpenHandCursor
        viewer._image_label.drag_started.emit(QPoint(200, 160))
        assert viewer._image_label.cursor().shape() == Qt.ClosedHandCursor
        viewer._image_label.drag_moved.emit(QPoint(140, 120))
        viewer._image_label.drag_finished.emit()
        qapp.processEvents()
        assert viewer._image_label.cursor().shape() == Qt.OpenHandCursor

        viewer._image_label.wheel_zoom.emit(-50, QPoint(100, 100))
        qapp.processEvents()
        assert viewer._zoom_mode is False
        fit_again = viewer._image_label.pixmap()
        assert fit_again is not None and not fit_again.isNull()
        assert fit_again.width() <= viewer._image_scroll.viewport().width()
        assert fit_again.height() <= viewer._image_scroll.viewport().height()

        viewer.close()


def _make_multi_ch_completed_run(tmpdir: str) -> None:
    """Build a completed-run-style fixture with three CH-style regions."""
    with open(os.path.join(tmpdir, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_context": {"run_type": "full"},
                "status": "success",
            },
            f,
        )
    for region in ("CH1", "CH2", "CH3"):
        summary = os.path.join(tmpdir, region, "summary")
        day_plots = os.path.join(tmpdir, region, "day_plots")
        tables = os.path.join(tmpdir, region, "tables")
        os.makedirs(summary)
        os.makedirs(day_plots)
        os.makedirs(tables)
        open(os.path.join(summary, "phasic_correction_impact.png"), "wb").close()
        open(os.path.join(summary, "phasic_auc_timeseries.png"), "wb").close()
        open(os.path.join(day_plots, f"phasic_sig_iso_day_000.png"), "wb").close()
        open(os.path.join(day_plots, f"phasic_dff_day_000.png"), "wb").close()
        open(os.path.join(tables, f"{region}_phasic_summary.csv"), "wb").close()
    # Give CH2 a real, decodable image so a region switch can be checked at
    # the widget-state level (pixmap present / not null), not just file
    # existence.
    real_img = os.path.join(tmpdir, "CH2", "day_plots", "phasic_sig_iso_day_000.png")
    pix = QPixmap(400, 300)
    pix.fill(Qt.red)
    assert pix.save(real_img)


def test_run_report_viewer_multi_region_load_and_buttons(qapp):
    """Targeted regression for the completed-run Review workflow: a
    multi-CH-region run must populate the region selector, repopulate tabs
    per region, and the quick-open buttons must resolve to the right path
    for the selected region without launching a real file browser."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_multi_ch_completed_run(tmpdir)

        viewer = RunReportViewer()
        viewer.resize(1000, 700)
        viewer.show()
        qapp.processEvents()

        opened_paths = []
        viewer._open_path = lambda path: opened_paths.append(path)

        try:
            assert viewer.load_report(tmpdir) is True

            region_names = [
                viewer._region_combo.itemText(i)
                for i in range(viewer._region_combo.count())
            ]
            assert region_names == ["CH1", "CH2", "CH3"]

            for index, region in enumerate(region_names):
                viewer._region_combo.setCurrentIndex(index)
                qapp.processEvents()
                assert viewer._selected_region() == region
                assert viewer._tabs.count() > 0
                tab_images = viewer._region_tab_images[region]
                assert any(tab_images.values())

                opened_paths.clear()
                viewer._open_run_report_btn.click()
                assert opened_paths == [
                    os.path.join(tmpdir, "run_report.json")
                ]

                opened_paths.clear()
                viewer._open_region_summary_btn.click()
                assert opened_paths == [
                    os.path.join(tmpdir, region, "summary")
                ]

                opened_paths.clear()
                viewer._open_region_day_plots_btn.click()
                assert opened_paths == [
                    os.path.join(tmpdir, region, "day_plots")
                ]

                opened_paths.clear()
                viewer._open_region_tables_btn.click()
                assert opened_paths == [
                    os.path.join(tmpdir, region, "tables")
                ]

            # CH2 has a real decodable image; switching to it must produce a
            # usable, non-null pixmap in the viewer (widget-state level
            # zoom/scroll readiness check).
            ch2_index = region_names.index("CH2")
            viewer._region_combo.setCurrentIndex(ch2_index)
            qapp.processEvents()
            for tab_index in range(viewer._tabs.count()):
                viewer._tabs.setCurrentIndex(tab_index)
                qapp.processEvents()
            pix = viewer._image_label.pixmap()
            if viewer._active_image_path:
                assert pix is not None and not pix.isNull()
        finally:
            viewer.close()


def test_scientist_completion_summary_names_missing_session_without_internal_terms(tmp_path):
    (tmp_path / "input_manifest.json").write_text(
        json.dumps(
            {
                "expected": [
                    {"index": 0, "disposition": "process"},
                    {
                        "index": 1,
                        "disposition": "authorized_missing_corrupted",
                        "expected_start_time": "2024-01-01T01:00:00",
                        "expected_duration_sec": 60.0,
                        "reason": "approved damaged session",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    classification = SimpleNamespace(
        completed_with_missing=True,
        is_success=True,
        missing_session_count=1,
        reason="internal reason",
    )
    text = get_scientist_completion_summary(str(tmp_path), classification)
    assert "Completed with missing sessions" in text
    assert "Session 2" in text
    assert "2024-01-01 01:00:00" in text
    assert "expected duration 60s" in text
    for forbidden in ("manifest", "schema", "cache", "digest", "contract", "JSON"):
        assert forbidden.lower() not in text.lower()
