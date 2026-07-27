from __future__ import annotations

import numpy as np
import pandas as pd

from photometry_pipeline import guided_continuous_saved_artifacts as subject
from photometry_pipeline.guided_continuous_saved_artifacts import (
    build_window_plot_data,
    continuous_plot_coordinates,
)
from photometry_pipeline.run_completion_contract import (
    PROFILE_CONTINUOUS,
    required_deliverables_for_run_mode,
    normalize_run_mode,
)


def _timeline(mode: str, *, start: str | None, fixed: str | None) -> dict:
    return {
        "timeline_mode": mode,
        "fixed_daily_anchor_clock": fixed,
        "recording_start_clock": start,
        "recording_start_clock_source": "user_confirmed"
        if start is not None
        else "not_applicable",
    }


def test_continuous_coordinates_preserve_accepted_leading_blank_and_unwrap():
    fixed_at_seven = continuous_plot_coordinates(
        [4.0 * 3600.0],
        _timeline("fixed_daily_anchor", start="07:00", fixed="07:00"),
    )
    fixed_starting_at_three = continuous_plot_coordinates(
        [0.0],
        _timeline("fixed_daily_anchor", start="03:00", fixed="07:00"),
    )
    civil_starting_at_eleven = continuous_plot_coordinates(
        [0.0], _timeline("civil", start="11:00", fixed=None)
    )
    elapsed = continuous_plot_coordinates(
        [0.0, 25.0 * 3600.0], _timeline("elapsed", start=None, fixed=None)
    )

    assert fixed_at_seven.tolist() == [4.0 * 3600.0]
    assert fixed_starting_at_three.tolist() == [20.0 * 3600.0]
    assert civil_starting_at_eleven.tolist() == [11.0 * 3600.0]
    assert elapsed.tolist() == [0.0, 25.0 * 3600.0]


def test_window_plot_data_keeps_nan_window_values_missing():
    summary = pd.DataFrame(
        {
            "window_index": [0, 1, 2],
            "window_midpoint_sec": [0.0, 10.0, 20.0],
            "phasic_signal_auc": [1.0, np.nan, 3.0],
        }
    )
    data = build_window_plot_data(
        summary,
        timeline_contract=_timeline("elapsed", start=None, fixed=None),
        value_column="phasic_signal_auc",
    )
    assert data["x_sec"].tolist() == [0.0, 10.0, 20.0]
    assert np.isnan(data["values"][1])
    assert data["values"][0] == 1.0
    assert data["values"][2] == 3.0


def test_tonic_sampling_is_bounded_and_preserves_gap_after_empty_chunk(monkeypatch):
    attrs = {
        chunk_id: {
            "window_index": chunk_id,
            "window_start_sec": start,
            "window_end_sec": end,
            "window_duration_sec": end - start,
            "acquisition_mode": "continuous",
            "fs_hz": 1.0,
        }
        for chunk_id, start, end in (
            (0, 0.0, 1.0),
            (1, 2.0, 3.0),
            (2, 10.0, 11.0),
        )
    }
    fields = {
        0: (
            np.array([0.0, 1.0]),
            np.array([10.0, 11.0]),
            np.array([5.0, 5.5]),
            np.array([1.0, 2.0]),
        ),
        1: (
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
        ),
        2: (
            np.array([0.0, 1.0]),
            np.array([12.0, 13.0]),
            np.array([6.0, 6.5]),
            np.array([3.0, 4.0]),
        ),
    }

    monkeypatch.setattr(subject, "list_cache_chunk_ids", lambda _cache: [0, 1, 2])
    monkeypatch.setattr(
        subject,
        "load_cache_chunk_attrs",
        lambda _cache, _roi, chunk_id: attrs[int(chunk_id)],
    )
    monkeypatch.setattr(
        subject,
        "load_cache_chunk_fields",
        lambda _cache, _roi, chunk_id, _names: fields[int(chunk_id)],
    )

    elapsed, traces, details = subject._sample_tonic_trace(
        object(), "ROI1", max_points=6
    )
    values = traces["tonic_signal"]

    assert elapsed.size <= 6
    assert values.size == elapsed.size
    assert traces["raw_signal"].size == elapsed.size
    assert traces["raw_reference"].size == elapsed.size
    assert np.any(~np.isfinite(values))
    assert details["contains_gap_markers"] is True
    assert details["max_plot_points"] == 6


def test_native_guided_continuous_profiles_require_only_their_analysis_images():
    phasic = normalize_run_mode(
        run_profile="guided_continuous_rwd_phasic",
        run_type="phasic_only",
        acquisition_mode="continuous",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_CONTINUOUS,
        expected_rois=["ROI1"],
        continuous_outputs_ran=True,
    )
    tonic = normalize_run_mode(
        run_profile="guided_continuous_rwd_tonic",
        run_type="tonic_only",
        acquisition_mode="continuous",
        traces_only=False,
        phasic_analysis=False,
        tonic_analysis=True,
        feature_extraction_ran=False,
        deliverable_profile=PROFILE_CONTINUOUS,
        expected_rois=["ROI1"],
        continuous_outputs_ran=True,
    )
    combined = normalize_run_mode(
        run_profile="guided_continuous_rwd_combined",
        run_type="tonic_and_phasic",
        acquisition_mode="continuous",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=True,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_CONTINUOUS,
        expected_rois=["ROI1"],
        continuous_outputs_ran=True,
    )

    assert set(required_deliverables_for_run_mode(phasic)) == {
        "ROI1/tables/continuous_phasic_window_summary.csv",
        "ROI1/summary/phasic_correction_impact.png",
        "ROI1/summary/phasic_auc_timeseries.png",
        "ROI1/summary/phasic_peak_rate_timeseries.png",
    }
    assert set(required_deliverables_for_run_mode(tonic)) == {
        "ROI1/tables/continuous_tonic_window_summary.csv",
        "ROI1/summary/tonic_overview.png",
    }
    assert set(required_deliverables_for_run_mode(combined)) == {
        "ROI1/tables/continuous_phasic_window_summary.csv",
        "ROI1/tables/continuous_tonic_window_summary.csv",
        "ROI1/summary/phasic_correction_impact.png",
        "ROI1/summary/phasic_auc_timeseries.png",
        "ROI1/summary/phasic_peak_rate_timeseries.png",
        "ROI1/summary/tonic_overview.png",
    }
