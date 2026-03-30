"""
Tests for RunReportViewer and run_report_parser dynamic discovery.
"""

import os
import json
import tempfile
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from gui.run_report_parser import (
    resolve_region_deliverables,
    resolve_internal_artifacts,
    resolve_primary_artifacts,
    is_successful_completed_run_dir,
)

from PySide6.QtWidgets import QApplication
from gui.run_report_viewer import RunReportViewer


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
        assert "does not explicitly report successful completion" in reason.lower()


def test_successful_completed_run_dir_accepts_status_success():
    """status.json final success should allow complete-state entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "status.json"), "w", encoding="utf-8") as f:
            json.dump({"schema_version": 1, "phase": "final", "status": "success"}, f)
        ok, reason = is_successful_completed_run_dir(tmpdir)
        assert ok is True
        assert "status.json" in reason


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
        assert [os.path.basename(p) for p in tab_map["Phasic Sig/Iso"]] == ["phasic_sig_iso_day_000.png"]
        assert [os.path.basename(p) for p in tab_map["Dynamic Fit"]] == ["phasic_dynamic_fit_day_000.png"]
        assert [os.path.basename(p) for p in tab_map["Phasic dFF"]] == ["phasic_dFF_day_000.png"]
        assert [os.path.basename(p) for p in tab_map["Phasic Stacked"]] == ["phasic_stacked_day_000.png"]
        assert "Phasic Raw" not in tab_map


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
