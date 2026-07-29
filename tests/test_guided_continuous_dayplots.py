from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from photometry_pipeline.guided_continuous_saved_artifacts import (
    CONTINUOUS_DAY_PLOT_EXPLANATION,
    _extract_continuous_day_plot_panels,
    _publish_continuous_day_plots,
)


def _write_cache(
    path: Path,
    *,
    time_sec: np.ndarray,
    roi: str = "ROI1",
    strategy_family: str | None = None,
    phasic: bool = False,
    window_start_sec: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_sec = np.asarray(time_sec, dtype=float)
    if time_sec.size > 1:
        dt = float(np.median(np.diff(time_sec)))
    else:
        dt = 1.0
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
                "window_start_sec": float(window_start_sec),
                "window_end_sec": (
                    float(window_start_sec + time_sec[-1] + dt)
                    if time_sec.size
                    else float(window_start_sec)
                ),
                "window_duration_sec": (
                    float(time_sec[-1] + dt)
                    if time_sec.size
                    else 0.0
                ),
                "acquisition_mode": "continuous",
                "fs_hz": 1.0 / dt if dt > 0 else 1.0,
            }
        )
        group.create_dataset("time_sec", data=time_sec)
        if strategy_family is not None:
            group.attrs["correction_strategy_family"] = strategy_family
            group.attrs["correction_selected_strategy"] = strategy_family
            if strategy_family == "dynamic_fit":
                group.create_dataset("fit_ref", data=20.0 + time_sec * 0.01)
            else:
                group.create_dataset(
                    "signal_only_f0_baseline", data=10.0 + time_sec * 0.01
                )
            group.create_dataset("sig_raw", data=100.0 + time_sec)
            group.create_dataset("uv_raw", data=50.0 + time_sec * 0.5)
        if phasic:
            group.create_dataset("dff", data=np.sin(time_sec / 30.0))


def _build_saved_cache_run(
    tmp_path: Path,
    *,
    corrected_time: np.ndarray,
    phasic_time: np.ndarray | None = None,
    strategy_family: str = "dynamic_fit",
) -> Path:
    run_dir = tmp_path / "run"
    _write_cache(
        run_dir / "continuous_corrected_trace_cache.h5",
        time_sec=corrected_time,
        strategy_family=strategy_family,
    )
    _write_cache(
        run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5",
        time_sec=corrected_time if phasic_time is None else phasic_time,
        phasic=True,
    )
    return run_dir


def _timeline(mode: str, *, start: str | None, fixed: str | None) -> dict:
    return {
        "timeline_mode": mode,
        "recording_start_clock": start,
        "fixed_daily_anchor_clock": fixed,
    }


def _panels(result: dict, family: str) -> list[dict]:
    return [panel for panels in result[family].values() for panel in panels]


def test_civil_sampling_uses_clock_boundaries_not_recording_relative_windows(tmp_path):
    times = np.arange(0.0, 3600.0 + 60.0, 60.0)
    result = _extract_continuous_day_plot_panels(
        run_dir=str(_build_saved_cache_run(tmp_path, corrected_time=times)),
        roi="ROI1",
        timeline_contract=_timeline("civil", start="11:23", fixed=None),
    )

    placements = {
        (panel["hour"], panel["col"]): panel
        for panel in _panels(result, "signal_reference")
    }
    assert (11, 0) not in placements
    assert (11, 1) in placements
    assert np.isclose(placements[(11, 1)]["t"][0], 0.0)
    assert np.isclose(placements[(11, 1)]["t"][-1], 540.0)


def test_fixed_sampling_uses_the_existing_fixed_daily_anchor(tmp_path):
    times = np.arange(0.0, 3600.0 + 60.0, 60.0)
    result = _extract_continuous_day_plot_panels(
        run_dir=str(_build_saved_cache_run(tmp_path, corrected_time=times)),
        roi="ROI1",
        timeline_contract=_timeline(
            "fixed_daily_anchor", start="11:23", fixed="07:00"
        ),
    )
    placements = {
        (panel["hour"], panel["col"]): panel
        for panel in _panels(result, "signal_reference")
    }
    assert (4, 0) not in placements
    assert (4, 1) in placements


def test_elapsed_sampling_starts_at_elapsed_zero_and_ignores_civil_clock(tmp_path):
    times = np.arange(0.0, 3600.0 + 60.0, 60.0)
    result = _extract_continuous_day_plot_panels(
        run_dir=str(_build_saved_cache_run(tmp_path, corrected_time=times)),
        roi="ROI1",
        timeline_contract=_timeline("elapsed", start=None, fixed=None),
    )
    placements = {
        (panel["hour"], panel["col"]): panel
        for panel in _panels(result, "signal_reference")
    }
    assert (0, 0) in placements
    assert (0, 1) in placements
    assert all(panel["hour"] != 11 for panel in placements.values())


def test_incomplete_interval_is_a_blank_without_shift_or_padding(tmp_path):
    times = np.arange(0.0, 900.0 + 60.0, 60.0)
    result = _extract_continuous_day_plot_panels(
        run_dir=str(_build_saved_cache_run(tmp_path, corrected_time=times)),
        roi="ROI1",
        timeline_contract=_timeline("civil", start="11:23", fixed=None),
    )
    assert {(p["hour"], p["col"]) for p in _panels(result, "signal_reference")} == set()


def test_internal_timestamp_gap_blanks_only_the_affected_slot(tmp_path):
    times = np.arange(0.0, 3600.0 + 60.0, 60.0)
    times = times[times != 600.0]
    result = _extract_continuous_day_plot_panels(
        run_dir=str(_build_saved_cache_run(tmp_path, corrected_time=times)),
        roi="ROI1",
        timeline_contract=_timeline("civil", start="11:23", fixed=None),
    )
    placements = {
        (panel["hour"], panel["col"]): panel
        for panel in _panels(result, "signal_reference")
    }
    assert (11, 1) not in placements
    assert (12, 0) in placements


def test_each_trace_family_uses_its_own_saved_timestamps_and_strategy_field(tmp_path):
    corrected = np.arange(0.0, 3600.0 + 60.0, 60.0)
    phasic = corrected + 0.5
    run_dir = _build_saved_cache_run(
        tmp_path,
        corrected_time=corrected,
        phasic_time=phasic,
        strategy_family="signal_only_f0",
    )
    result = _extract_continuous_day_plot_panels(
        run_dir=str(run_dir),
        roi="ROI1",
        timeline_contract=_timeline("civil", start="11:23", fixed=None),
    )
    assert result["reference_field"] == "signal_only_f0_baseline"
    assert np.isclose(_panels(result, "signal_reference")[0]["t"][0], 0.0)
    assert np.isclose(_panels(result, "phasic_dff")[0]["t"][0], 0.5)
    assert result["correction_reference_label"] == (
        "Correction Reference (signal-only F0 baseline)"
    )


def test_publication_writes_only_manifest_ready_dayplot_images_and_not_features(
    tmp_path,
):
    times = np.arange(0.0, 3600.0 + 60.0, 60.0)
    run_dir = _build_saved_cache_run(tmp_path, corrected_time=times)
    features_path = run_dir / "_analysis" / "phasic_out" / "features.csv"
    features_path.write_text("sentinel\n", encoding="utf-8")
    before = features_path.read_bytes()
    artifacts = _publish_continuous_day_plots(
        run_dir=str(run_dir),
        roi="ROI1",
        timeline_contract=_timeline("civil", start="11:23", fixed=None),
    )
    assert len(artifacts) == 4
    names = {Path(record["relative_path"]).name for record in artifacts}
    assert names == {
        "phasic_sig_iso_day_000.png",
        "phasic_correction_reference_day_000.png",
        "phasic_dFF_day_000.png",
        "phasic_stacked_day_000.png",
    }
    assert all("dynamic_fit" not in name for name in names)
    assert all((run_dir / record["relative_path"]).is_file() for record in artifacts)
    assert all(record["display_explanation"] == CONTINUOUS_DAY_PLOT_EXPLANATION for record in artifacts)
    assert features_path.read_bytes() == before
    assert not any("features.csv" in record["relative_path"] for record in artifacts)
