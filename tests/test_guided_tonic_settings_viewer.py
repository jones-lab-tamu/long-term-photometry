"""Viewer-level (real GUI-display) proof that the tonic-settings Results
summary is populated independently of feature settings/extraction/ROI
selection -- not merely a formatter unit test.

Builds real compact-overview dictionaries via
photometry_pipeline.completed_run_review.load_completed_review_overview
against small on-disk fixtures, then drives the actual
gui.run_report_viewer.RunReportViewer.load_report(...) method, exactly like
the existing persistent-warning-banner tests in
tests/test_completed_run_review_overview.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from gui.run_report_viewer import RunReportViewer
from photometry_pipeline.completed_run_review import load_completed_review_overview
from photometry_pipeline.run_completion_contract import (
    CORRECTION_PROVENANCE_SCHEMA_VERSION,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _branch_provenance(analysis_kind: str) -> dict:
    return {
        "schema_version": CORRECTION_PROVENANCE_SCHEMA_VERSION,
        "analysis_mode": analysis_kind,
        "source": "explicit_per_roi_map",
        "included_roi_ids": ["CH1"],
        "requested_by_roi": [
            {
                "roi_id": "CH1",
                "strategy_family": "dynamic_fit",
                "selected_strategy": "global_linear_regression",
                "dynamic_fit_mode": "global_linear_regression",
                "parameter_identity": "parameters-1",
                "evidence_identity": "evidence-1",
            }
        ],
    }


def _write_branch(root: Path, analysis_kind: str) -> None:
    branch_dir = root / "_analysis" / f"{analysis_kind}_out"
    provenance = _branch_provenance(analysis_kind)
    _write_json(branch_dir / "run_metadata.json", {"correction_provenance": provenance})
    _write_json(
        branch_dir / "run_report.json",
        {"derived_settings": {"correction_provenance": provenance}},
    )
    (branch_dir / f"{analysis_kind}_trace_cache.h5").write_bytes(b"cache-bytes")


def _build_run(
    root: Path,
    *,
    include_phasic: bool,
    include_tonic: bool,
    tonic_output_mode: str = "preserve_raw_session_shape",
    tonic_timeline_mode: str = "real_elapsed_time",
) -> Path:
    run_id = "guided-run-tonic-viewer"
    run_mode = {"phasic_analysis": include_phasic, "tonic_analysis": include_tonic}
    if include_phasic:
        _write_branch(root, "phasic")
    if include_tonic:
        _write_branch(root, "tonic")
        tonic_dir = root / "_analysis" / "tonic_out"
        tonic_dir.mkdir(parents=True, exist_ok=True)
        (tonic_dir / "config_used.yaml").write_text(
            f"tonic_output_mode: {tonic_output_mode}\n"
            f"tonic_timeline_mode: {tonic_timeline_mode}\n",
            encoding="utf-8",
        )
    manifest = {
        "completion": {
            "completion_contract_version": "run_completion.v1",
            "final": True,
            "run_id": run_id,
            "run_mode": run_mode,
        }
    }
    _write_json(root / "MANIFEST.json", manifest)
    digest = hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest()
    _write_json(
        root / "status.json",
        {
            "run_id": run_id,
            "phase": "final",
            "status": "success",
            "completion": {
                "completion_contract_version": "run_completion.v1",
                "manifest_sha256": digest,
            },
        },
    )
    _write_json(
        root / "run_report.json",
        {
            "completion_contract": {
                "contract_version": "run_completion.v1",
                "run_id": run_id,
            }
        },
    )
    summary = root / "CH1" / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    pixmap = QPixmap(4, 4)
    pixmap.fill()
    assert pixmap.save(str(summary / "phasic_correction_impact.png"), "PNG")
    return root


# ---------------------------------------------------------------------------
# 1. Tonic run with no feature settings still shows the summary.
# ---------------------------------------------------------------------------


def test_tonic_only_run_with_no_feature_settings_shows_tonic_summary(qapp, tmp_path):
    run = _build_run(
        tmp_path / "run",
        include_phasic=False,
        include_tonic=True,
        tonic_output_mode="flatten_session_bleach_preserve_session_baseline",
        tonic_timeline_mode="gap_free_elapsed_time",
    )
    overview = load_completed_review_overview(run)
    assert overview["feature_settings_by_roi"] == {}
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run), review_overview=overview) is True
        assert viewer._tonic_settings_summary_label.isHidden() is False
        assert viewer._tonic_settings_summary_label.text() == (
            "Tonic timeline: Gap-free elapsed time\n"
            "Session shape: Within-session bleaching trend removed"
        )
    finally:
        viewer.close()
        viewer.deleteLater()


# ---------------------------------------------------------------------------
# 2. Tonic run with feature settings shows the tonic summary once, alongside
# (not instead of) the feature-settings label.
# ---------------------------------------------------------------------------


def test_tonic_and_phasic_run_with_feature_settings_shows_both(qapp, tmp_path):
    run = _build_run(
        tmp_path / "run", include_phasic=True, include_tonic=True
    )
    overview = load_completed_review_overview(run)
    overview.update(
        {
            "feature_settings_by_roi": {
                "CH1": {
                    "roi": "CH1",
                    "source": "default",
                    "effective_config_fields": {"peak_threshold_method": "mean_std"},
                }
            }
        }
    )
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run), review_overview=overview) is True
        assert viewer._tonic_settings_summary_label.isHidden() is False
        assert viewer._tonic_settings_summary_label.text() == (
            "Tonic timeline: Real elapsed time\nSession shape: Preserved"
        )
        assert "1 included ROI(s)" in (
            viewer._selected_feature_settings_label.text()
        )
        # Shown exactly once -- setText is idempotent, not appended.
        assert viewer._tonic_settings_summary_label.text().count(
            "Tonic timeline"
        ) == 1
    finally:
        viewer.close()
        viewer.deleteLater()


# ---------------------------------------------------------------------------
# 3. Phasic-only run hides the tonic summary.
# ---------------------------------------------------------------------------


def test_phasic_only_run_hides_tonic_summary(qapp, tmp_path):
    run = _build_run(
        tmp_path / "run", include_phasic=True, include_tonic=False
    )
    overview = load_completed_review_overview(run)
    assert overview["tonic_settings"] == {}
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run), review_overview=overview) is True
        assert viewer._tonic_settings_summary_label.isHidden() is True
        assert viewer._tonic_settings_summary_label.text() == ""
    finally:
        viewer.close()
        viewer.deleteLater()


# ---------------------------------------------------------------------------
# 4. Warning-review tonic run still shows the tonic summary.
# ---------------------------------------------------------------------------


def test_warning_review_tonic_run_shows_tonic_summary(qapp, tmp_path):
    run = _build_run(
        tmp_path / "run", include_phasic=True, include_tonic=True
    )
    overview = load_completed_review_overview(run)
    overview.update(
        {
            "review_status": "reviewable_with_warning",
            "validation_warning_title": "Analysis completed with a validation warning",
            "validation_warning_message": (
                "Your plots and tables were generated and are available "
                "below. Some recording sessions were shorter than the "
                "expected 10-minute session length. 1 session was "
                "affected. Review those sessions before relying on the "
                "results."
            ),
        }
    )
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run), review_overview=overview) is True
        assert viewer._tonic_settings_summary_label.isHidden() is False
        assert viewer._tonic_settings_summary_label.text() == (
            "Tonic timeline: Real elapsed time\nSession shape: Preserved"
        )
    finally:
        viewer.close()
        viewer.deleteLater()


# ---------------------------------------------------------------------------
# 5. ROI/image changes do not clear the tonic summary.
# ---------------------------------------------------------------------------


def test_roi_and_image_changes_do_not_clear_tonic_summary(qapp, tmp_path):
    run = _build_run(
        tmp_path / "run", include_phasic=True, include_tonic=True
    )
    overview = load_completed_review_overview(run)
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run), review_overview=overview) is True
        before = viewer._tonic_settings_summary_label.text()
        assert before != ""
        viewer._region_combo.setCurrentIndex(1)
        qapp.processEvents()
        assert viewer._tonic_settings_summary_label.text() == before
        assert viewer._tonic_settings_summary_label.isHidden() is False
    finally:
        viewer.close()
        viewer.deleteLater()
