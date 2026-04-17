import numpy as np
import pandas as pd

from photometry_pipeline.config import Config
from photometry_pipeline.core.feature_extraction import (
    compute_auc_above_threshold,
    compute_detection_threshold_bounds,
    extract_features,
    get_peak_indices_for_trace,
)
from photometry_pipeline.core.types import Chunk


def _make_single_roi_chunk(trace: np.ndarray, fs_hz: float = 20.0) -> Chunk:
    arr = np.asarray(trace, dtype=float).reshape(-1)
    t = np.arange(arr.size, dtype=float) / float(fs_hz)
    z = np.zeros((arr.size, 1), dtype=float)
    chunk = Chunk(
        chunk_id=0,
        source_file="synthetic.csv",
        format="synthetic",
        time_sec=t,
        uv_raw=z.copy(),
        sig_raw=z.copy(),
        fs_hz=float(fs_hz),
        channel_names=["Region0"],
        metadata={},
    )
    chunk.dff = arr.reshape(-1, 1)
    chunk.delta_f = arr.reshape(-1, 1)
    return chunk


def test_threshold_bounds_semantics_for_mean_std_percentile_absolute():
    clean = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)

    cfg_mean = Config(peak_threshold_method="mean_std", peak_threshold_k=1.0)
    mean_bounds = compute_detection_threshold_bounds(clean, cfg_mean)
    assert mean_bounds["upper"] == np.mean(clean) + np.std(clean)
    assert mean_bounds["lower"] == np.mean(clean) - np.std(clean)

    cfg_pct = Config(peak_threshold_method="percentile", peak_threshold_percentile=80.0)
    pct_bounds = compute_detection_threshold_bounds(clean, cfg_pct)
    assert pct_bounds["upper"] == np.percentile(clean, 80.0)
    assert pct_bounds["lower"] == np.percentile(clean, 20.0)

    cfg_abs = Config(peak_threshold_method="absolute", peak_threshold_abs=1.5)
    abs_bounds = compute_detection_threshold_bounds(clean, cfg_abs)
    assert abs_bounds["upper"] == 1.5
    assert abs_bounds["lower"] == -1.5


def test_event_selection_respects_positive_negative_and_both_modes():
    fs = 20.0
    t = np.arange(0.0, 30.0, 1.0 / fs)
    trace = (
        1.8 * np.exp(-0.5 * ((t - 10.0) / 0.25) ** 2)
        - 1.9 * np.exp(-0.5 * ((t - 20.0) / 0.25) ** 2)
    )
    base_cfg = dict(
        peak_threshold_method="absolute",
        peak_threshold_abs=1.0,
        peak_min_distance_sec=0.5,
        peak_min_prominence_k=0.0,
        peak_min_width_sec=0.0,
        peak_pre_filter="none",
    )

    cfg_pos = Config(signal_excursion_polarity="positive", **base_cfg)
    idx_pos, pol_pos = get_peak_indices_for_trace(
        trace, fs, cfg_pos, return_polarities=True
    )
    assert len(idx_pos) == 1
    assert int(pol_pos[0]) == 1

    cfg_neg = Config(signal_excursion_polarity="negative", **base_cfg)
    idx_neg, pol_neg = get_peak_indices_for_trace(
        trace, fs, cfg_neg, return_polarities=True
    )
    assert len(idx_neg) == 1
    assert int(pol_neg[0]) == -1

    cfg_both = Config(signal_excursion_polarity="both", **base_cfg)
    idx_both, pol_both = get_peak_indices_for_trace(
        trace, fs, cfg_both, return_polarities=True
    )
    assert len(idx_both) == 2
    assert set(int(x) for x in pol_both.tolist()) == {-1, 1}


def test_auc_semantics_are_signed_by_selected_polarity():
    trace = np.array([-2.0, -2.0, 0.0, 2.0, 2.0], dtype=float)
    auc_pos = compute_auc_above_threshold(
        trace,
        baseline_value=0.0,
        fs_hz=1.0,
        signal_excursion_polarity="positive",
    )
    auc_neg = compute_auc_above_threshold(
        trace,
        baseline_value=0.0,
        fs_hz=1.0,
        signal_excursion_polarity="negative",
    )
    auc_both = compute_auc_above_threshold(
        trace,
        baseline_value=0.0,
        fs_hz=1.0,
        signal_excursion_polarity="both",
    )
    assert auc_pos == 3.0
    assert auc_neg == -3.0
    assert auc_both == 0.0


def test_extract_features_default_matches_explicit_positive_mode():
    fs = 20.0
    t = np.arange(0.0, 30.0, 1.0 / fs)
    trace = (
        1.8 * np.exp(-0.5 * ((t - 10.0) / 0.25) ** 2)
        - 1.9 * np.exp(-0.5 * ((t - 20.0) / 0.25) ** 2)
    )
    chunk = _make_single_roi_chunk(trace, fs_hz=fs)
    common = dict(
        event_signal="dff",
        peak_threshold_method="absolute",
        peak_threshold_abs=1.0,
        peak_min_distance_sec=0.5,
        peak_min_prominence_k=0.0,
        peak_min_width_sec=0.0,
        peak_pre_filter="none",
        event_auc_baseline="zero",
    )
    cfg_default = Config(**common)
    cfg_positive = Config(signal_excursion_polarity="positive", **common)
    df_default = extract_features(chunk, cfg_default)
    df_positive = extract_features(chunk, cfg_positive)
    pd.testing.assert_series_equal(
        df_default.loc[0, ["peak_count", "auc"]],
        df_positive.loc[0, ["peak_count", "auc"]],
        check_dtype=False,
        check_names=False,
    )
