"""Display-only marker-on copies of native continuous dF/F day plots.

The action under test creates noncanonical QC copies from already-saved
corrected dF/F data and already-saved continuous event times. It must never
rerun correction or detection, and must never disturb a canonical artifact.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
import photometry_pipeline.guided_continuous_saved_artifacts as saved_module
from gui.main_window import MainWindow
from gui.run_report_viewer import TAB_PHASIC_DFF, RunReportViewer
from photometry_pipeline.core import feature_extraction as feature_extraction_module
from photometry_pipeline.guided_continuous_rwd_combined_run import (
    execute_guided_continuous_rwd_combined_run,
)
from photometry_pipeline.guided_continuous_rwd_phasic_run import (
    execute_guided_continuous_rwd_phasic_run,
)
from photometry_pipeline.guided_continuous_rwd_tonic_run import (
    execute_guided_continuous_rwd_tonic_run,
)
from photometry_pipeline.guided_continuous_saved_artifacts import (
    CONTINUOUS_MARKER_ON_VARIANT_RELATIVE_DIR,
    _extract_continuous_day_plot_panels,
    _publish_continuous_day_plots,
    build_continuous_marker_on_dff_dayplots,
    load_continuous_marker_event_times,
    map_continuous_event_times_to_panel_indices,
)
from tests.test_guided_continuous_rwd_correction_pass_persistence import (
    _build_case,
    _pass_inputs,
)

pytestmark = pytest.mark.usefixtures("no_real_modals")

MARKER_RGB = (220, 0, 0)


# ---------------------------------------------------------------------------
# Lightweight saved-cache fixtures (no production run required)
# ---------------------------------------------------------------------------


def _write_cache(
    path: Path,
    *,
    time_sec: np.ndarray,
    roi: str = "ROI1",
    strategy_family: str | None = None,
    phasic: bool = False,
    dff: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_sec = np.asarray(time_sec, dtype=float)
    dt = float(np.median(np.diff(time_sec))) if time_sec.size > 1 else 1.0
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([roi], dtype="S"))
        meta.create_dataset("chunk_ids", data=np.asarray([0], dtype=int))
        group = handle.create_group(f"roi/{roi}/chunk_0")
        group.attrs.update(
            {
                "window_index": 0,
                "window_start_sec": 0.0,
                "window_end_sec": float(time_sec[-1] + dt),
                "window_duration_sec": float(time_sec[-1] + dt),
                "acquisition_mode": "continuous",
                "fs_hz": 1.0 / dt if dt > 0 else 1.0,
            }
        )
        group.create_dataset("time_sec", data=time_sec)
        if strategy_family is not None:
            group.attrs["correction_strategy_family"] = strategy_family
            group.attrs["correction_selected_strategy"] = strategy_family
            group.create_dataset("fit_ref", data=20.0 + time_sec * 0.01)
            group.create_dataset("sig_raw", data=100.0 + time_sec)
            group.create_dataset("uv_raw", data=50.0 + time_sec * 0.5)
        if phasic:
            group.create_dataset(
                "dff", data=np.sin(time_sec / 30.0) if dff is None else dff
            )


def _build_saved_cache_run(tmp_path: Path, *, time_sec: np.ndarray, dff=None) -> Path:
    run_dir = tmp_path / "run"
    _write_cache(
        run_dir / "continuous_corrected_trace_cache.h5",
        time_sec=time_sec,
        strategy_family="dynamic_fit",
    )
    _write_cache(
        run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5",
        time_sec=time_sec,
        phasic=True,
        dff=dff,
    )
    return run_dir


def _elapsed_timeline() -> dict:
    return {
        "timeline_mode": "elapsed",
        "recording_start_clock": None,
        "fixed_daily_anchor_clock": None,
    }


def _dff_panels(run_dir: Path) -> list[dict]:
    extracted = _extract_continuous_day_plot_panels(
        run_dir=str(run_dir), roi="ROI1", timeline_contract=_elapsed_timeline()
    )
    return [panel for panels in extracted["phasic_dff"].values() for panel in panels]


def _marker_pixel_count(path: str) -> int:
    from PIL import Image

    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"))
    return int(
        np.sum(
            (arr[:, :, 0] == MARKER_RGB[0])
            & (arr[:, :, 1] == MARKER_RGB[1])
            & (arr[:, :, 2] == MARKER_RGB[2])
        )
    )


def _digest_tree(root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return digests


# ---------------------------------------------------------------------------
# 1. Event-to-panel mapping
# ---------------------------------------------------------------------------


def _panel(start: float, end: float, fs: float = 10.0) -> dict:
    n = int(round((end - start) * fs))
    return {
        "t": np.arange(n) / fs,
        "dff": np.zeros(n),
        "panel_start_sec": start,
        "panel_end_sec": end,
    }


def test_event_exactly_at_panel_start_is_included_at_index_zero():
    panel = _panel(1800.0, 2400.0)
    indices = map_continuous_event_times_to_panel_indices(panel, [1800.0])
    assert indices.tolist() == [0]


def test_event_just_before_panel_end_is_included_at_the_last_sample():
    panel = _panel(1800.0, 2400.0, fs=10.0)
    last_sample_time = 1800.0 + float(panel["t"][-1])
    indices = map_continuous_event_times_to_panel_indices(panel, [last_sample_time])
    assert indices.tolist() == [panel["t"].size - 1]


def test_event_exactly_at_panel_end_is_excluded():
    panel = _panel(1800.0, 2400.0)
    assert map_continuous_event_times_to_panel_indices(panel, [2400.0]).size == 0


def test_events_outside_the_panel_span_are_excluded():
    panel = _panel(1800.0, 2400.0)
    indices = map_continuous_event_times_to_panel_indices(
        panel, [0.0, 1799.9, 2400.0, 9999.0]
    )
    assert indices.size == 0


def test_mapped_indices_match_the_panel_sample_grid():
    panel = _panel(1800.0, 2400.0, fs=10.0)
    expected = [3, 137, 4211, 5999]
    event_times = [1800.0 + float(panel["t"][i]) for i in expected]
    indices = map_continuous_event_times_to_panel_indices(panel, event_times)
    assert indices.tolist() == expected


def test_mapped_indices_are_always_in_range_and_unique():
    panel = _panel(0.0, 600.0, fs=10.0)
    # Deliberately off-grid times inside the admitted span, including one in the
    # sub-sample sliver after the final stored sample.
    event_times = [0.03, 0.04, 12.37, 599.999]
    indices = map_continuous_event_times_to_panel_indices(panel, event_times)
    assert indices.size == len(set(indices.tolist()))
    assert indices.min() >= 0
    assert indices.max() <= panel["t"].size - 1


def test_non_finite_event_times_are_dropped_without_error():
    panel = _panel(0.0, 600.0)
    indices = map_continuous_event_times_to_panel_indices(
        panel, [np.nan, np.inf, -np.inf, 10.0]
    )
    assert indices.tolist() == [100]


def test_panels_carry_their_own_recording_global_bounds(tmp_path):
    run_dir = _build_saved_cache_run(
        tmp_path, time_sec=np.arange(0.0, 3600.0, 1.0)
    )
    panels = _dff_panels(run_dir)
    assert panels
    for panel in panels:
        assert panel["panel_end_sec"] - panel["panel_start_sec"] == pytest.approx(600.0)
        assert panel["panel_start_sec"] + panel["t"][0] == pytest.approx(
            panel["panel_start_sec"]
        )


# ---------------------------------------------------------------------------
# 2. No detector rerun
# ---------------------------------------------------------------------------


def test_marker_on_build_never_calls_the_event_detector(tmp_path, monkeypatch):
    run_dir = _build_saved_cache_run(tmp_path, time_sec=np.arange(0.0, 3600.0, 1.0))
    panels = _dff_panels(run_dir)
    event_times = [panels[0]["panel_start_sec"] + 5.0]

    monkeypatch.setattr(
        feature_extraction_module,
        "get_peak_indices_for_trace",
        lambda *a, **k: pytest.fail(
            "marker-on day-plot copies must never run the detector"
        ),
    )
    import tools.plot_phasic_dayplot_bundle as bundle_module

    monkeypatch.setattr(
        bundle_module,
        "get_peak_indices_for_trace",
        lambda *a, **k: pytest.fail(
            "marker-on day-plot copies must never run the detector"
        ),
    )

    result = build_continuous_marker_on_dff_dayplots(
        str(run_dir),
        roi="ROI1",
        timeline_contract=_elapsed_timeline(),
        event_times_sec=event_times,
    )
    assert result["paths"]


# ---------------------------------------------------------------------------
# 3. Rendering
# ---------------------------------------------------------------------------


def test_marker_on_copy_contains_marker_pixels_and_marker_free_copy_does_not(tmp_path):
    times = np.arange(0.0, 3600.0, 1.0)
    dff = np.zeros_like(times)
    dff[[10, 20, 30]] = 5.0
    run_dir = _build_saved_cache_run(tmp_path, time_sec=times, dff=dff)

    _publish_continuous_day_plots(
        run_dir=str(run_dir), roi="ROI1", timeline_contract=_elapsed_timeline()
    )
    canonical = run_dir / "ROI1" / "day_plots" / "phasic_dFF_day_000.png"
    assert _marker_pixel_count(str(canonical)) == 0

    result = build_continuous_marker_on_dff_dayplots(
        str(run_dir),
        roi="ROI1",
        timeline_contract=_elapsed_timeline(),
        event_times_sec=[10.0, 20.0, 30.0],
    )
    assert len(result["paths"]) == 1
    assert _marker_pixel_count(result["paths"][0]) > 0


def test_marker_counts_match_saved_events_inside_displayed_panels(tmp_path):
    run_dir = _build_saved_cache_run(tmp_path, time_sec=np.arange(0.0, 3600.0, 1.0))
    panels = _dff_panels(run_dir)
    assert len(panels) >= 2

    inside_first = panels[0]["panel_start_sec"] + 5.0
    inside_second = panels[1]["panel_start_sec"] + 5.0
    # 1200.0 falls in the 20-30 minute gap between two sampled display windows.
    outside_every_panel = 1200.0
    assert all(
        not (p["panel_start_sec"] <= outside_every_panel < p["panel_end_sec"])
        for p in panels
    )

    result = build_continuous_marker_on_dff_dayplots(
        str(run_dir),
        roi="ROI1",
        timeline_contract=_elapsed_timeline(),
        event_times_sec=[inside_first, inside_second, outside_every_panel],
    )
    assert result["total_markers"] == 2
    assert result["marker_counts_by_day"] == {0: 2}


def test_marker_on_copy_uses_the_same_display_scale_as_the_canonical_plot(tmp_path):
    run_dir = _build_saved_cache_run(tmp_path, time_sec=np.arange(0.0, 3600.0, 1.0))
    extracted = _extract_continuous_day_plot_panels(
        run_dir=str(run_dir), roi="ROI1", timeline_contract=_elapsed_timeline()
    )
    limits = saved_module._continuous_dff_display_limits(extracted["phasic_dff"])

    seen = {}
    original = saved_module._continuous_dff_display_limits

    def _record(panels):
        result = original(panels)
        seen["limits"] = result
        return result

    saved_module._continuous_dff_display_limits = _record
    try:
        build_continuous_marker_on_dff_dayplots(
            str(run_dir),
            roi="ROI1",
            timeline_contract=_elapsed_timeline(),
            event_times_sec=[5.0],
        )
    finally:
        saved_module._continuous_dff_display_limits = original
    assert seen["limits"] == limits


# ---------------------------------------------------------------------------
# 4. Artifact isolation
# ---------------------------------------------------------------------------


def test_marker_on_build_writes_only_into_the_variant_directory(tmp_path):
    run_dir = _build_saved_cache_run(tmp_path, time_sec=np.arange(0.0, 3600.0, 1.0))
    _publish_continuous_day_plots(
        run_dir=str(run_dir), roi="ROI1", timeline_contract=_elapsed_timeline()
    )
    before = _digest_tree(run_dir)

    result = build_continuous_marker_on_dff_dayplots(
        str(run_dir),
        roi="ROI1",
        timeline_contract=_elapsed_timeline(),
        event_times_sec=[5.0, 65.0],
    )
    after = _digest_tree(run_dir)

    variant_rel = (
        f"ROI1/{CONTINUOUS_MARKER_ON_VARIANT_RELATIVE_DIR}".replace("\\", "/")
    )
    new_files = set(after) - set(before)
    assert new_files
    assert all(name.startswith(variant_rel + "/") for name in new_files)
    # Nothing that already existed changed, including the canonical day plots.
    assert {name: after[name] for name in before} == before
    assert all(
        os.path.realpath(path).startswith(os.path.realpath(result["output_dir"]))
        for path in result["paths"]
    )


def test_marker_on_build_refuses_to_write_into_the_canonical_day_plots_root(tmp_path):
    run_dir = _build_saved_cache_run(tmp_path, time_sec=np.arange(0.0, 3600.0, 1.0))
    with pytest.raises(saved_module.GuidedContinuousSavedArtifactError) as excinfo:
        build_continuous_marker_on_dff_dayplots(
            str(run_dir),
            roi="ROI1",
            timeline_contract=_elapsed_timeline(),
            event_times_sec=[5.0],
            output_dir=str(run_dir / "ROI1" / "day_plots"),
        )
    assert "canonical" in str(excinfo.value)


def test_marker_on_copy_uses_the_canonical_day_plot_filenames(tmp_path):
    run_dir = _build_saved_cache_run(tmp_path, time_sec=np.arange(0.0, 3600.0, 1.0))
    canonical = _publish_continuous_day_plots(
        run_dir=str(run_dir), roi="ROI1", timeline_contract=_elapsed_timeline()
    )
    canonical_dff = {
        Path(record["relative_path"]).name
        for record in canonical
        if record["family"] == "sampled_phasic_dff"
    }
    result = build_continuous_marker_on_dff_dayplots(
        str(run_dir),
        roi="ROI1",
        timeline_contract=_elapsed_timeline(),
        event_times_sec=[5.0],
    )
    assert {Path(path).name for path in result["paths"]} == canonical_dff


# ---------------------------------------------------------------------------
# Real completed continuous runs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    return _build_case(
        tmp_path_factory.mktemp("marker_dayplots") / "recording",
        continuous_window_sec=90.0,
    )


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
        tmp_path_factory.mktemp("marker_combined"),
    )


@pytest.fixture(scope="module")
def phasic_only_run(accepted_case, real_config, tmp_path_factory):
    return _run(
        execute_guided_continuous_rwd_phasic_run,
        accepted_case,
        real_config,
        tmp_path_factory.mktemp("marker_phasic"),
    )


@pytest.fixture(scope="module")
def tonic_only_run(accepted_case, real_config, tmp_path_factory):
    return _run(
        execute_guided_continuous_rwd_tonic_run,
        accepted_case,
        real_config,
        tmp_path_factory.mktemp("marker_tonic"),
    )


@pytest.fixture
def window(qapp):
    w = MainWindow()
    yield w
    w.close()
    w.deleteLater()


@pytest.fixture
def copied_combined_run(combined_run, tmp_path):
    """A private copy of the combined run, so one test's copies cannot leak
    into another test's before/after comparison."""
    import shutil

    copied = tmp_path / "combined_copy"
    shutil.copytree(combined_run.run_dir, copied)
    return copied


def _load_guided_results(window, run_dir: str) -> None:
    window._current_run_dir = run_dir
    assert window._guided_report_viewer.load_report(run_dir) is True
    window._refresh_guided_continuous_marker_dayplot_availability()


def test_saved_event_times_load_through_the_existing_loader_scoped_to_one_roi(
    combined_run,
):
    import pandas as pd

    events_path = os.path.join(
        combined_run.run_dir,
        "_analysis",
        "phasic_out",
        "features",
        "continuous_phasic_events.csv",
    )
    raw = pd.read_csv(events_path)
    for roi in combined_run.detection.included_roi_ids:
        loaded = load_continuous_marker_event_times(combined_run.run_dir, roi)
        expected = raw[raw["roi"] == roi]["global_time_sec"].to_numpy(dtype=float)
        assert np.allclose(loaded, expected)


# ---------------------------------------------------------------------------
# 5. Eligibility
# ---------------------------------------------------------------------------


def test_action_is_enabled_for_a_completed_continuous_combined_run(window, combined_run):
    _load_guided_results(window, combined_run.run_dir)
    ready, message = window._guided_continuous_marker_dayplot_readiness()
    assert ready is True
    assert window._guided_continuous_marker_dayplot_btn.isEnabled()
    assert not window._guided_continuous_marker_dayplot_group.isHidden()
    # Only the action that applies to this recording type is offered.
    assert window._guided_marker_free_dayplot_group.isHidden()
    assert "unchanged" in message


def test_action_is_unavailable_for_continuous_phasic_only_with_a_specific_reason(
    window, phasic_only_run
):
    _load_guided_results(window, phasic_only_run.run_dir)
    ready, message = window._guided_continuous_marker_dayplot_readiness()
    assert ready is False
    assert not window._guided_continuous_marker_dayplot_btn.isEnabled()
    assert "phasic analysis only" in message
    # Visible but disabled, so the reason stays readable.
    assert not window._guided_continuous_marker_dayplot_group.isHidden()
    assert window._guided_marker_free_dayplot_group.isHidden()


def test_action_is_unavailable_for_continuous_tonic_only_with_a_specific_reason(
    window, tonic_only_run
):
    _load_guided_results(window, tonic_only_run.run_dir)
    ready, message = window._guided_continuous_marker_dayplot_readiness()
    assert ready is False
    assert not window._guided_continuous_marker_dayplot_btn.isEnabled()
    assert "tonic analysis only" in message
    # Visible but disabled, so the reason stays readable.
    assert not window._guided_continuous_marker_dayplot_group.isHidden()
    assert window._guided_marker_free_dayplot_group.isHidden()


def test_action_is_unavailable_when_no_results_are_loaded(window):
    ready, message = window._guided_continuous_marker_dayplot_readiness()
    assert ready is False
    assert "Open a completed analysis" in message
    assert window._guided_continuous_marker_dayplot_group.isHidden()
    assert window._guided_marker_free_dayplot_group.isHidden()


def test_action_is_unavailable_for_intermittent_runs_with_a_specific_reason(
    window, monkeypatch
):
    viewer = window._guided_report_viewer
    monkeypatch.setattr(viewer, "has_loaded_results", lambda: True)
    monkeypatch.setattr(viewer, "is_native_continuous_results", lambda: False)
    ready, message = window._guided_continuous_marker_dayplot_readiness()
    assert ready is False
    assert "only for continuous recordings" in message


def test_action_is_unavailable_when_the_saved_event_table_disappears_after_load(
    window, combined_run, tmp_path
):
    """The viewer refuses to open a run whose event table is already missing, so
    this branch guards the file being removed while Results stay open."""
    import shutil

    copied = tmp_path / "no_events"
    shutil.copytree(combined_run.run_dir, copied)
    _load_guided_results(window, str(copied))
    assert window._guided_continuous_marker_dayplot_readiness()[0] is True

    os.remove(
        copied
        / "_analysis"
        / "phasic_out"
        / "features"
        / "continuous_phasic_events.csv"
    )
    window._refresh_guided_continuous_marker_dayplot_availability()
    ready, message = window._guided_continuous_marker_dayplot_readiness()
    assert ready is False
    assert "detected-events table" in message
    assert not window._guided_continuous_marker_dayplot_btn.isEnabled()


# ---------------------------------------------------------------------------
# 6. Viewer behavior
# ---------------------------------------------------------------------------


def test_native_manifest_backed_images_still_display_when_no_override_is_active(
    qapp, combined_run
):
    viewer = RunReportViewer()
    assert viewer.load_report(combined_run.run_dir) is True
    tabs = viewer.available_view_tabs()
    assert TAB_PHASIC_DFF in tabs
    viewer._tabs.setCurrentIndex(tabs.index(TAB_PHASIC_DFF))
    active = viewer.active_image_path()
    assert active.endswith(".png")
    assert os.path.isfile(active)
    assert "rerendered_display_variants" not in active
    assert viewer.active_artifact_path() == active


def test_external_override_displays_in_native_continuous_results(
    qapp, combined_run, tmp_path
):
    viewer = RunReportViewer()
    assert viewer.load_report(combined_run.run_dir) is True
    roi = viewer.selected_region()
    events = load_continuous_marker_event_times(combined_run.run_dir, roi)
    result = build_continuous_marker_on_dff_dayplots(
        combined_run.run_dir,
        roi=roi,
        timeline_contract=dict(viewer.native_continuous_context().get("timeline") or {}),
        event_times_sec=events,
        # Keep the shared run directory pristine for other tests.
        output_dir=str(tmp_path / "variant"),
    )

    assert (
        viewer.show_external_image_sequence(
            result["paths"], initial_path=result["paths"][0]
        )
        is True
    )
    assert os.path.realpath(viewer.active_image_path()) == os.path.realpath(
        result["paths"][0]
    )
    assert viewer._artifact_table_scroll.isHidden()
    assert not viewer._artifact_metadata_label.isVisible()
    # The manifest-backed records for that tab are untouched.
    assert any(
        record.get("family") == "sampled_phasic_dff"
        for record in viewer.available_artifacts(roi)
    )


def test_guided_action_creates_displays_and_reports_marker_on_copies(
    window, copied_combined_run
):
    _load_guided_results(window, str(copied_combined_run))
    roi = window._guided_report_viewer.selected_region()

    window._on_guided_create_continuous_marker_dayplots()

    variant_dir = os.path.join(
        str(copied_combined_run), roi, CONTINUOUS_MARKER_ON_VARIANT_RELATIVE_DIR
    )
    written = sorted(
        name for name in os.listdir(variant_dir) if name.lower().endswith(".png")
    )
    assert written
    status = window._guided_continuous_marker_dayplot_status_label.text()
    assert f"Showing {len(written)}" in status
    assert variant_dir in status
    assert "unchanged" in status
    assert os.path.realpath(
        window._guided_report_viewer.active_image_path()
    ).startswith(os.path.realpath(variant_dir))


def test_guided_action_does_not_claim_display_when_the_viewer_cannot_switch(
    window, copied_combined_run, monkeypatch
):
    _load_guided_results(window, str(copied_combined_run))
    monkeypatch.setattr(
        window._guided_report_viewer, "show_external_image_sequence", lambda *a, **k: False
    )
    window._on_guided_create_continuous_marker_dayplots()
    status = window._guided_continuous_marker_dayplot_status_label.text()
    assert "Showing" not in status
    assert "could not switch" in status
    assert "Created" in status


def test_guided_action_leaves_analysis_outputs_unchanged(
    window, copied_combined_run, monkeypatch
):
    run_dir = copied_combined_run
    _load_guided_results(window, str(run_dir))
    roi = window._guided_report_viewer.selected_region()
    variant_rel = f"{roi}/{CONTINUOUS_MARKER_ON_VARIANT_RELATIVE_DIR}".replace("\\", "/")

    def _outside_variant(tree: dict[str, str]) -> dict[str, str]:
        return {
            name: digest
            for name, digest in tree.items()
            if not name.startswith(variant_rel + "/")
        }

    before = _digest_tree(run_dir)
    assert not any(name.startswith(variant_rel + "/") for name in before)

    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(
            main_window_module.QMessageBox,
            name,
            lambda _p, title, text, *a, **k: pytest.fail(f"{title}: {text}"),
        )
    window._on_guided_create_continuous_marker_dayplots()

    after = _digest_tree(run_dir)
    # Copies were written, and nothing outside the variant directory moved.
    assert any(name.startswith(variant_rel + "/") for name in after)
    assert _outside_variant(after) == _outside_variant(before)
    for guarded in (
        "MANIFEST.json",
        "run_report.json",
        "_analysis/phasic_out/features/continuous_phasic_events.csv",
        "_analysis/phasic_out/phasic_trace_cache.h5",
        "continuous_corrected_trace_cache.h5",
        f"{roi}/day_plots/phasic_dFF_day_000.png",
    ):
        assert after[guarded] == before[guarded]


def test_guided_action_refuses_and_explains_when_not_eligible(
    window, tonic_only_run, monkeypatch
):
    _load_guided_results(window, tonic_only_run.run_dir)
    shown = {}
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, text, *a, **k: shown.update(title=title, text=text),
    )
    window._on_guided_create_continuous_marker_dayplots()
    assert "tonic analysis only" in shown["text"]
    assert not os.path.isdir(
        os.path.join(
            tonic_only_run.run_dir,
            window._guided_report_viewer.selected_region(),
            CONTINUOUS_MARKER_ON_VARIANT_RELATIVE_DIR,
        )
    )
