"""CR1-F1-I4: native Guided continuous Results use saved artifacts only."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from gui.run_report_viewer import RunReportViewer
from photometry_pipeline.guided_continuous_rwd_combined_run import (
    execute_guided_continuous_rwd_combined_run,
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
    return _build_case(tmp_path_factory.mktemp("cr1_f1_i4") / "recording", continuous_window_sec=90.0)


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


def _run(func, accepted_case, real_config, output_base):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = _pass_inputs(
        accepted_case
    )
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
def combined_run(accepted_case, real_config, tmp_path_factory):
    return _run(
        execute_guided_continuous_rwd_combined_run,
        accepted_case,
        real_config,
        tmp_path_factory.mktemp("combined"),
    )


@pytest.fixture(scope="module")
def phasic_run(accepted_case, real_config, tmp_path_factory):
    return _run(
        execute_guided_continuous_rwd_phasic_run,
        accepted_case,
        real_config,
        tmp_path_factory.mktemp("phasic"),
    )


@pytest.fixture(scope="module")
def tonic_run(accepted_case, real_config, tmp_path_factory):
    return _run(
        execute_guided_continuous_rwd_tonic_run,
        accepted_case,
        real_config,
        tmp_path_factory.mktemp("tonic"),
    )


def _forbid_scientific_continuous_work(monkeypatch):
    import gui.run_report_viewer as viewer_module
    import photometry_pipeline.completed_continuous_rwd_review as review_module
    import photometry_pipeline.continuous_outputs as outputs_module
    import photometry_pipeline.guided_continuous_saved_artifacts as saved_module
    import photometry_pipeline.guided_continuous_rwd_phasic_detection as detection_module
    import photometry_pipeline.core.feature_extraction as feature_module
    import photometry_pipeline.io.hdf5_cache_reader as cache_module

    def forbidden(name):
        return lambda *args, _name=name, **kwargs: pytest.fail(
            f"native saved-artifact Results called {_name}"
        )

    monkeypatch.setattr(
        viewer_module,
        "resolve_region_deliverables",
        forbidden("resolve_region_deliverables"),
    )
    for name in (
        "load_continuous_run_overview",
        "load_continuous_roi_trace",
        "load_continuous_window_summary",
        "load_continuous_phasic_events",
    ):
        monkeypatch.setattr(viewer_module, name, forbidden(name))
    for module, names in (
        (
            review_module,
            (
                "open_phasic_cache",
                "open_tonic_cache",
                "load_cache_chunk_fields",
                "load_cache_chunk_attrs",
                "list_cache_chunk_ids",
            ),
        ),
        (
            outputs_module,
            (
                "open_phasic_cache",
                "open_tonic_cache",
                "_allocate_trace_points",
                "_sample_elapsed_trace_from_cache",
                "_plot_continuous_trace_overview",
                "_plot_xy_from_summary",
            ),
        ),
        (
            saved_module,
            (
                "open_phasic_cache",
                "open_tonic_cache",
                "continuous_plot_coordinates",
                "build_window_plot_data",
                "_write_window_plot",
                "_publish_correction_impact",
                "_publish_tonic_overview",
                "_sample_tonic_trace",
            ),
        ),
        (
            detection_module,
            (
                "detect_guided_continuous_rwd_phasic_features",
                "_reconstruct_roi_trace",
                "_detect_roi",
            ),
        ),
        (
            feature_module,
            (
                "get_peak_indices_for_trace",
                "compute_auc_over_finite_runs",
                "compute_auc_above_threshold",
                "apply_peak_prefilter",
            ),
        ),
        (
            cache_module,
            (
                "open_phasic_cache",
                "open_tonic_cache",
                "load_cache_chunk_fields",
                "load_cache_chunk_attrs",
                "list_cache_chunk_ids",
                "list_cache_rois",
            ),
        ),
    ):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden(name))
    monkeypatch.setattr(
        viewer_module.RunReportViewer,
        "_render_continuous_trace_pixmap",
        staticmethod(forbidden("_render_continuous_trace_pixmap")),
    )
    import matplotlib.pyplot as pyplot

    monkeypatch.setattr(pyplot, "subplots", forbidden("matplotlib.pyplot.subplots"))


def _tab_labels(viewer: RunReportViewer) -> list[str]:
    return [viewer._tabs.tabText(i) for i in range(viewer._tabs.count())]


class _PreparedSignalSource(QObject):
    succeeded = Signal(object)
    failed = Signal(str)


def _copy_native_run(source: str, destination: Path) -> tuple[Path, dict, str]:
    shutil.copytree(source, destination)
    manifest_path = destination / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roi = str(manifest["completion"]["run_mode"]["expected_rois"][0])
    return destination, manifest, roi


def _write_manifest(path: Path, manifest: dict) -> None:
    manifest_path = path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    status_path = path / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["completion"]["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    status_path.write_text(json.dumps(status), encoding="utf-8")


def _copy_with_large_event_table(
    source: str, destination: Path, data_rows: int
) -> tuple[Path, Path]:
    broken, manifest, _roi = _copy_native_run(source, destination)
    relative_path = "_analysis/phasic_out/features/continuous_phasic_events.csv"
    source_path = Path(source) / relative_path
    event_path = broken / relative_path
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        headers = next(csv.reader(handle))
    with event_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row_index in range(data_rows):
            writer.writerow([str(row_index)] + ["" for _ in headers[1:]])

    event_record = next(
        record
        for record in manifest["completion"]["artifacts"]
        if record["relative_path"] == relative_path
    )
    event_record["size_bytes"] = event_path.stat().st_size
    event_record["sha256"] = hashlib.sha256(event_path.read_bytes()).hexdigest()
    _write_manifest(broken, manifest)
    return broken, event_path


def test_combined_native_results_index_and_switching_use_saved_files(
    qapp, combined_run, monkeypatch
):
    _forbid_scientific_continuous_work(monkeypatch)
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(combined_run.run_dir) is True
        assert viewer._continuous_workspace.isHidden()
        assert not viewer._workspace.isHidden()
        assert _tab_labels(viewer) == [
            "Correction impact",
            "Tonic overview",
            "Phasic signal AUC",
            "Peak rate",
            "Phasic window summary",
            "Tonic window summary",
            "Detected events",
        ]
        indexed = viewer._native_continuous_artifact_index["artifacts"]
        assert len(indexed) == 13
        assert sum(record.get("scope") == "run" for record in indexed) == 1
        assert all(
            sum(record.get("roi") == roi for record in indexed) == 6
            for roi in viewer.available_regions()
        )
        records = viewer.available_artifacts()
        assert sum(record["label"] == "Detected events" for record in records) == 1
        assert records[0]["path"].endswith("phasic_correction_impact.png")
        assert viewer.active_artifact_path().endswith("phasic_correction_impact.png")

        viewer._tabs.setCurrentIndex(_tab_labels(viewer).index("Phasic window summary"))
        headers = [
            viewer._artifact_table.horizontalHeaderItem(i).text()
            for i in range(viewer._artifact_table.columnCount())
        ]
        assert headers[0] == "roi"
        assert viewer.active_artifact_path().endswith(
            "continuous_phasic_window_summary.csv"
        )
        with open(viewer.active_artifact_path(), "r", encoding="utf-8-sig", newline="") as handle:
            stored_rows = list(csv.reader(handle))
        displayed_rows = [
            [
                viewer._artifact_table.item(row_idx, col_idx).text()
                for col_idx in range(viewer._artifact_table.columnCount())
            ]
            for row_idx in range(viewer._artifact_table.rowCount())
        ]
        assert stored_rows[0] == headers
        assert displayed_rows == stored_rows[1:]

        viewer._tabs.setCurrentIndex(_tab_labels(viewer).index("Detected events"))
        assert "Run-level" in viewer._artifact_metadata_label.text()
        assert f"ROI: {viewer.selected_region()}" not in viewer._artifact_metadata_label.text()

        viewer._tabs.setCurrentIndex(_tab_labels(viewer).index("Phasic window summary"))
        viewer._region_combo.setCurrentIndex(1)
        assert viewer.active_artifact_path().endswith(
            f"{viewer.selected_region()}\\tables\\continuous_phasic_window_summary.csv"
        )
        assert viewer._artifact_table.rowCount() > 0
        assert "Showing first" not in viewer._artifact_metadata_label.text()
    finally:
        viewer.close()


def test_native_completion_marker_reaches_the_same_generic_viewer(
    qapp, combined_run, monkeypatch
):
    _forbid_scientific_continuous_work(monkeypatch)
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(
            combined_run.run_dir,
            review_overview={
                "native_saved_artifacts": True,
                "run_dir": combined_run.run_dir,
            },
        ) is True
        assert not viewer._workspace.isHidden()
        assert viewer._continuous_workspace.isHidden()
        assert viewer.active_artifact_path().endswith("phasic_correction_impact.png")
    finally:
        viewer.close()


def test_guided_worker_prepares_index_and_callback_installs_without_rebuild(
    qapp, combined_run, monkeypatch
):
    import gui.main_window as main_window_module
    import gui.run_report_parser as parser_module
    import gui.run_report_viewer as viewer_module
    from gui.main_window import MainWindow

    classify_calls = []
    real_classify = main_window_module.classify_run_terminal_state

    def tracked_classify(path):
        classify_calls.append(os.path.realpath(path))
        return real_classify(path)

    monkeypatch.setattr(
        main_window_module, "classify_run_terminal_state", tracked_classify
    )
    worker = main_window_module._GuidedCompletedReviewLoadWorker(
        combined_run.run_dir
    )
    payloads = []
    worker.succeeded.connect(payloads.append)
    worker.run()

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["native_saved_artifacts"] is True
    assert payload["run_dir"] == os.path.realpath(combined_run.run_dir)
    assert payload["artifact_index"]["run_dir"] == payload["run_dir"]
    assert classify_calls == [payload["run_dir"]]

    forbidden = lambda *args, **kwargs: pytest.fail(
        "the prepared Guided package was rebuilt in the GUI callback"
    )
    monkeypatch.setattr(
        viewer_module, "build_guided_continuous_saved_artifact_index", forbidden
    )
    monkeypatch.setattr(
        viewer_module, "classify_completed_run_terminal_state", forbidden
    )
    monkeypatch.setattr(parser_module, "_read_json_dict", forbidden)
    monkeypatch.setattr(parser_module, "_validate_continuous_saved_image", forbidden)

    window = MainWindow()
    source = _PreparedSignalSource()
    try:
        viewer = window._guided_report_viewer
        monkeypatch.setattr(
            viewer,
            "load_report",
            lambda *args, **kwargs: pytest.fail(
                "the prepared Guided callback must not call load_report"
            ),
        )
        window._guided_completed_review_load_path = payload["run_dir"]
        window._guided_completed_review_load_worker = source
        window._guided_completed_review_loading = True
        source.succeeded.connect(window._on_guided_completed_review_load_succeeded)
        source.succeeded.emit(payload)
        QApplication.processEvents()

        assert window._guided_completed_review_loading is False
        assert viewer._native_continuous_artifact_index is payload["artifact_index"]
        assert viewer.active_artifact_path().endswith(
            "phasic_correction_impact.png"
        )
        assert not viewer._active_pixmap.isNull()
    finally:
        window.close()
        window.deleteLater()


def test_large_detected_events_preview_is_bounded_and_openable(
    qapp, combined_run, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QTableWidgetItem
    from gui.run_report_viewer import NATIVE_CSV_PREVIEW_ROW_LIMIT

    data_rows = NATIVE_CSV_PREVIEW_ROW_LIMIT + 123
    broken, event_path = _copy_with_large_event_table(
        combined_run.run_dir, tmp_path / "large_events", data_rows
    )
    _forbid_scientific_continuous_work(monkeypatch)
    viewer = RunReportViewer()
    opened = []
    try:
        assert viewer.load_report(str(broken)) is True
        viewer._tabs.setCurrentIndex(_tab_labels(viewer).index("Detected events"))
        QApplication.processEvents()

        assert viewer._artifact_table.rowCount() == NATIVE_CSV_PREVIEW_ROW_LIMIT
        assert viewer._artifact_table.columnCount() > 0
        assert isinstance(viewer._artifact_table.item(0, 0), QTableWidgetItem)
        metadata = viewer._artifact_metadata_label.text()
        assert (
            f"Showing first {NATIVE_CSV_PREVIEW_ROW_LIMIT:,} of {data_rows:,} rows."
            in metadata
        )
        assert "Run-level (all included ROIs)" in metadata
        assert viewer.active_artifact_path() == str(event_path)
        assert viewer._open_region_tables_btn.text() == "Open CSV"

        monkeypatch.setattr(viewer, "_open_path", opened.append)
        viewer._open_region_tables_btn.click()
        assert opened == [str(event_path)]

        viewer._tabs.setCurrentIndex(_tab_labels(viewer).index("Phasic signal AUC"))
        viewer._tabs.setCurrentIndex(_tab_labels(viewer).index("Detected events"))
        QApplication.processEvents()
        assert viewer._artifact_table.rowCount() == NATIVE_CSV_PREVIEW_ROW_LIMIT
    finally:
        viewer.close()


@pytest.mark.parametrize(
    "fixture_name, forbidden, expected",
    [
        (
            "phasic_run",
            {"Tonic overview", "Tonic window summary"},
            {
                "Correction impact",
                "Phasic signal AUC",
                "Peak rate",
                "Phasic window summary",
                "Detected events",
            },
        ),
        ("tonic_run", {"Correction impact", "Phasic signal AUC", "Peak rate", "Phasic window summary", "Detected events"}, {"Tonic overview", "Tonic window summary"}),
    ],
)
def test_native_results_are_run_type_aware(
    request, qapp, monkeypatch, fixture_name, forbidden, expected
):
    run = request.getfixturevalue(fixture_name)
    _forbid_scientific_continuous_work(monkeypatch)
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(run.run_dir) is True
        labels = set(_tab_labels(viewer))
        assert labels == expected
        assert labels.isdisjoint(forbidden)
    finally:
        viewer.close()


def test_missing_native_png_is_actionable_and_never_falls_back(
    qapp, combined_run, tmp_path, monkeypatch
):
    _forbid_scientific_continuous_work(monkeypatch)
    broken = tmp_path / "missing_auc"
    shutil.copytree(combined_run.run_dir, broken)
    manifest = json.loads(
        (Path(combined_run.run_dir) / "MANIFEST.json").read_text(encoding="utf-8")
    )
    roi = manifest["completion"]["run_mode"]["expected_rois"][0]
    os.remove(broken / roi / "summary" / "phasic_auc_timeseries.png")

    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(broken)) is False
        assert viewer._workspace.isHidden()
        assert viewer._continuous_workspace.isHidden()
        status = viewer._status_label.text().lower()
        assert "auc" in status or "phasic_auc_timeseries" in status
        assert "traceback" not in status
    finally:
        viewer.close()


@pytest.mark.parametrize(
    ("mutation", "expected_text"),
    [
        ("missing_saved_record", "saved-artifact index"),
        ("wrong_roi", "saved-artifact index"),
        ("wrong_analysis_family", "saved-artifact index"),
        ("malformed_table_path", "continuous window index"),
    ],
)
def test_invalid_native_manifest_provenance_fails_closed(
    qapp, combined_run, tmp_path, monkeypatch, mutation, expected_text
):
    broken, manifest, roi = _copy_native_run(
        combined_run.run_dir, tmp_path / mutation
    )
    completion = manifest["completion"]
    index = completion["deliverables"]["continuous_window_index"]
    if mutation == "missing_saved_record":
        index["saved_artifacts"] = [
            record
            for record in index["saved_artifacts"]
            if not str(record["relative_path"]).endswith(
                "phasic_auc_timeseries.png"
            )
        ]
    elif mutation == "wrong_roi":
        index["saved_artifacts"][0]["roi"] = "not-the-declared-roi"
    elif mutation == "wrong_analysis_family":
        index["saved_artifacts"][0]["analysis_family"] = "tonic"
    else:
        index["families"]["continuous_phasic_window_summary"][
            "relative_paths"
        ][roi] = "not-a-declared-table.csv"
    _write_manifest(broken, manifest)

    _forbid_scientific_continuous_work(monkeypatch)
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(broken)) is False
        assert viewer._workspace.isHidden()
        assert viewer._continuous_workspace.isHidden()
        status = viewer._status_label.text().lower()
        assert expected_text.lower() in status
        assert "traceback" not in status
    finally:
        viewer.close()


def test_invalid_native_png_fails_closed_without_reconstruction(
    qapp, combined_run, tmp_path, monkeypatch
):
    broken, manifest, roi = _copy_native_run(
        combined_run.run_dir, tmp_path / "invalid_png"
    )
    invalid_bytes = b"not a png"
    image_path = broken / roi / "summary" / "phasic_auc_timeseries.png"
    image_path.write_bytes(invalid_bytes)
    image_record = next(
        record
        for record in manifest["completion"]["artifacts"]
        if record["relative_path"] == f"{roi}/summary/phasic_auc_timeseries.png"
    )
    image_record["size_bytes"] = len(invalid_bytes)
    image_record["sha256"] = hashlib.sha256(invalid_bytes).hexdigest()
    _write_manifest(broken, manifest)

    _forbid_scientific_continuous_work(monkeypatch)
    viewer = RunReportViewer()
    try:
        assert viewer.load_report(str(broken)) is False
        assert viewer._workspace.isHidden()
        assert viewer._continuous_workspace.isHidden()
        status = viewer._status_label.text().lower()
        assert "invalid" in status or "cannot" in status
        assert "traceback" not in status
    finally:
        viewer.close()


def test_guided_completed_loader_routes_native_runs_without_cache_overview(
    combined_run, monkeypatch
):
    import gui.main_window as main_window_module

    monkeypatch.setattr(
        main_window_module,
        "load_continuous_run_overview",
        lambda *args, **kwargs: pytest.fail(
            "native Guided loader must not construct the cache-backed overview"
        ),
    )
    worker = main_window_module._GuidedCompletedReviewLoadWorker(combined_run.run_dir)
    results = []
    worker.succeeded.connect(results.append)
    worker.run()
    assert len(results) == 1
    assert results[0]["native_saved_artifacts"] is True
    assert results[0]["run_dir"] == os.path.realpath(combined_run.run_dir)
    assert results[0]["artifact_index"]["run_dir"] == results[0]["run_dir"]
