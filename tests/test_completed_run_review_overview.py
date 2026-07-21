from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

import photometry_pipeline.completed_run_review as review_module
from gui.run_report_viewer import RunReportViewer
from photometry_pipeline.completed_run_review import (
    CompletedRunReviewError,
    load_completed_review_overview,
)
from photometry_pipeline.run_completion_contract import (
    CORRECTION_PROVENANCE_SCHEMA_VERSION,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _compact_completed_run(root: Path) -> Path:
    run_id = "guided-run-overview"
    provenance = {
        "schema_version": CORRECTION_PROVENANCE_SCHEMA_VERSION,
        "analysis_mode": "phasic",
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
    analysis = root / "_analysis" / "phasic_out"
    _write_json(analysis / "run_metadata.json", {"correction_provenance": provenance})
    _write_json(
        analysis / "run_report.json",
        {"derived_settings": {"correction_provenance": provenance}},
    )
    (analysis / "phasic_trace_cache.h5").write_bytes(b"must-not-be-read")
    manifest = {
        "completion": {
            "completion_contract_version": "run_completion.v1",
            "final": True,
            "run_id": run_id,
            "run_mode": {
                "phasic_analysis": True,
                "tonic_analysis": False,
            },
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
    return root


def test_compact_overview_reads_exact_metadata_set_and_never_opens_hdf5(
    tmp_path, monkeypatch
):
    run = _compact_completed_run(tmp_path / "run")
    opened: list[tuple[str, str]] = []
    real_read_text = Path.read_text
    real_read_bytes = Path.read_bytes

    def tracked_text(path, *args, **kwargs):
        opened.append((str(Path(path).resolve()), "text"))
        return real_read_text(path, *args, **kwargs)

    def tracked_bytes(path, *args, **kwargs):
        opened.append((str(Path(path).resolve()), "bytes"))
        return real_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_text)
    monkeypatch.setattr(Path, "read_bytes", tracked_bytes)
    monkeypatch.setattr(
        review_module,
        "open_phasic_cache",
        lambda *_args, **_kwargs: pytest.fail("initial overview opened HDF5"),
    )
    monkeypatch.setattr(
        review_module,
        "open_tonic_cache",
        lambda *_args, **_kwargs: pytest.fail("initial overview opened HDF5"),
    )

    overview = load_completed_review_overview(run)

    expected = {
        (str((run / "status.json").resolve()), "text"),
        (str((run / "MANIFEST.json").resolve()), "text"),
        (str((run / "MANIFEST.json").resolve()), "bytes"),
        (str((run / "run_report.json").resolve()), "text"),
        (str((run / "_analysis/phasic_out/run_metadata.json").resolve()), "text"),
        (str((run / "_analysis/phasic_out/run_report.json").resolve()), "text"),
    }
    assert set(opened) == expected
    assert overview["terminal_state"] == "success"
    assert overview["included_rois"] == ["CH1"]
    assert overview["analysis_branches"] == ["phasic"]
    assert overview["requested_by_roi"]["CH1"]["selected_strategy"] == (
        "global_linear_regression"
    )
    assert overview["full_resolution_traces_loaded"] is False


def test_compact_overview_refuses_inconsistent_terminal_metadata(tmp_path):
    run = _compact_completed_run(tmp_path / "run")
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    status["status"] = "running"
    _write_json(run / "status.json", status)

    with pytest.raises(CompletedRunReviewError, match="incomplete or inconsistent"):
        load_completed_review_overview(run)


def test_viewer_orients_first_then_loads_one_selected_review_image(
    qapp, tmp_path
):
    run = _compact_completed_run(tmp_path / "run")
    summary = run / "CH1" / "summary"
    summary.mkdir(parents=True)
    image_path = summary / "phasic_correction_impact.png"
    pixmap = QPixmap(4, 4)
    pixmap.fill()
    assert pixmap.save(str(image_path), "PNG")
    overview = load_completed_review_overview(run)
    overview.update(
        {
            "format": "rwd",
            "included_rois": ["CH1"],
            "feature_settings_by_roi": {
                "CH1": {
                    "source": "default",
                    "effective_config_fields": {
                        "peak_threshold_method": "mean_std"
                    },
                }
            },
        }
    )
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run), review_overview=overview) is True
        assert viewer._current_run_dir == str(run)
        assert viewer.active_image_path() == ""
        assert "completed RWD analysis" in viewer._status_label.text()
        assert "Feature detection settings are available" in (
            viewer._selected_feature_settings_label.text()
        )

        viewer._region_combo.setCurrentIndex(1)
        qapp.processEvents()
        assert Path(viewer.active_image_path()) == image_path
        assert "Global linear regression" in (
            viewer._correction_summary_label.text()
        )
        assert "Default feature settings" in (
            viewer._selected_feature_settings_label.text()
        )
    finally:
        viewer.close()
        viewer.deleteLater()


def test_viewer_shows_persistent_warning_banner_for_reviewable_with_warning(
    qapp, tmp_path
):
    run = _compact_completed_run(tmp_path / "run")
    summary = run / "CH1" / "summary"
    summary.mkdir(parents=True)
    image_path = summary / "phasic_correction_impact.png"
    pixmap = QPixmap(4, 4)
    pixmap.fill()
    assert pixmap.save(str(image_path), "PNG")
    overview = load_completed_review_overview(run)
    overview.update(
        {
            "review_status": "reviewable_with_warning",
            "validation_warning_title": "Analysis completed with a validation warning",
            "validation_warning_message": (
                "Your plots and tables were generated and are available below. "
                "Later recordings became progressively shorter than the "
                "expected 10-minute session length. 50 sessions were "
                "affected, beginning partway through the recording. Review "
                "those sessions before relying on the results."
            ),
            "included_rois": ["CH1"],
        }
    )
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run), review_overview=overview) is True
        assert viewer._status_label.wordWrap() is True
        status_text = viewer._status_label.text()
        assert "validation warning" in status_text
        assert "50 sessions" in status_text
        # Never expose internal completion vocabulary to the scientist.
        for internal_term in (
            "C8",
            "terminal validation",
            "HDF5",
            "manifest",
            "schema",
            "contract",
            "artifact identity",
        ):
            assert internal_term.lower() not in status_text.lower()
        style_before = viewer._status_label.styleSheet()

        # The warning must stay visible across ROI/image changes, not just
        # at initial load.
        viewer._region_combo.setCurrentIndex(1)
        qapp.processEvents()
        assert Path(viewer.active_image_path()) == image_path
        assert viewer._status_label.text() == status_text
        assert viewer._status_label.styleSheet() == style_before
    finally:
        viewer.close()
        viewer.deleteLater()


def test_status_label_has_word_wrap_enabled(qapp):
    viewer = RunReportViewer()
    try:
        assert viewer._status_label.wordWrap() is True
    finally:
        viewer.close()
        viewer.deleteLater()


def test_long_warning_displays_in_full_and_grows_beyond_one_line(qapp, tmp_path):
    run = _compact_completed_run(tmp_path / "run")
    summary = run / "CH1" / "summary"
    summary.mkdir(parents=True)
    pixmap = QPixmap(4, 4)
    pixmap.fill()
    assert pixmap.save(str(summary / "phasic_correction_impact.png"), "PNG")
    long_message = (
        "Your plots and tables were generated and are available below. "
        "Some recording sessions were shorter than the expected 10-minute "
        "length. 50 sessions were affected. Review those sessions before "
        "relying on the results. It is a good idea to inspect each affected "
        "session's plots individually before drawing any conclusions from "
        "this analysis."
    )
    overview = load_completed_review_overview(run)
    overview.update(
        {
            "review_status": "reviewable_with_warning",
            "validation_warning_title": "Analysis completed with a validation warning",
            "validation_warning_message": long_message,
            "included_rois": ["CH1"],
        }
    )
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run), review_overview=overview) is True
        status_text = viewer._status_label.text()
        # The full message is present -- nothing truncated or elided.
        assert long_message in status_text

        # A realistic Results-pane width forces multiple wrapped lines
        # instead of a single clipped one.
        realistic_width = 420
        one_line_height = viewer._status_label.fontMetrics().height()
        wrapped_height = viewer._status_label.heightForWidth(realistic_width)
        assert wrapped_height > one_line_height * 1.5
    finally:
        viewer.close()
        viewer.deleteLater()


def test_ordinary_short_status_still_displays_normally(qapp, tmp_path):
    run = _compact_completed_run(tmp_path / "run")
    summary = run / "CH1" / "summary"
    summary.mkdir(parents=True)
    pixmap = QPixmap(4, 4)
    pixmap.fill()
    assert pixmap.save(str(summary / "phasic_correction_impact.png"), "PNG")
    overview = load_completed_review_overview(run)
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(run), review_overview=overview) is True
        assert viewer._status_label.wordWrap() is True
        status_text = viewer._status_label.text()
        assert status_text
        assert "results workspace" in status_text.lower()

        # A short, single-line-ish message must not be forced multiple
        # lines' worth of height at a generously wide viewer width.
        wide_width = 2000
        one_line_height = viewer._status_label.fontMetrics().height()
        assert viewer._status_label.heightForWidth(wide_width) <= one_line_height * 1.5
    finally:
        viewer.close()
        viewer.deleteLater()
