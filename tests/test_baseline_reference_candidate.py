import json

import numpy as np

from photometry_pipeline.core.baseline_reference_candidate import (
    classify_baseline_fit_relationship,
    compute_baseline_reference_candidate,
    compute_baseline_reference_candidate_metrics,
    _nan_aware_moving_average,
)
from photometry_pipeline.core.dynamic_fit_qc import compute_dynamic_fit_validity_metrics


def _metadata_without_trace(payload):
    return {k: v for k, v in payload.items() if k != "baseline_ref_candidate"}


def _shared_slow_trace(duration_sec=900.0, fs=10.0):
    t = np.arange(int(duration_sec * fs), dtype=float) / fs
    slow = np.sin(2.0 * np.pi * (1.0 / 700.0) * t)
    reference = 1.0 + 0.8 * slow
    signal = 2.0 + 1.2 * slow
    return signal, reference, fs


def test_candidate_suppresses_response_scale_structure():
    fs = 10.0
    t = np.arange(7200, dtype=float) / fs
    slow = np.sin(2.0 * np.pi * (1.0 / 600.0) * t)
    response = np.sin(2.0 * np.pi * (1.0 / 30.0) * t)
    reference = 1.0 + 0.8 * slow + 0.3 * response
    signal = 2.0 + 1.1 * slow + 2.5 * response
    dynamic_ref_surrogate = 2.0 + 1.1 * slow + 2.0 * response

    candidate = compute_baseline_reference_candidate(signal, reference, fs)
    baseline_ref = candidate["baseline_ref_candidate"]
    metrics = compute_baseline_reference_candidate_metrics(
        signal,
        reference,
        dynamic_ref_surrogate,
        baseline_ref,
        fs,
    )
    dynamic_metrics = compute_dynamic_fit_validity_metrics(
        signal,
        reference,
        dynamic_ref_surrogate,
        fs,
    )

    assert candidate["baseline_ref_candidate_available"] is True
    assert baseline_ref.shape == signal.shape
    assert (
        metrics["baseline_ref_response_scale_fraction"]
        < dynamic_metrics["fitted_ref_response_scale_fraction"]
    )
    assert abs(np.corrcoef(baseline_ref[np.isfinite(baseline_ref)], response[np.isfinite(baseline_ref)])[0, 1]) < 0.35


def test_configurable_smoothing_window_records_requested_and_actual_values():
    signal, reference, fs = _shared_slow_trace(duration_sec=900.0)

    cand_120 = compute_baseline_reference_candidate(signal, reference, fs, smoothing_window_sec=120.0)
    cand_300 = compute_baseline_reference_candidate(signal, reference, fs, smoothing_window_sec=300.0)

    assert cand_120["baseline_ref_requested_smoothing_window_sec"] == 120.0
    assert cand_120["baseline_ref_actual_smoothing_window_sec"] == 120.0
    assert cand_120["baseline_ref_smoothing_window_sec"] == 120.0
    assert cand_120["baseline_ref_smoothing_window_adjusted"] is False
    assert cand_300["baseline_ref_requested_smoothing_window_sec"] == 300.0
    assert cand_300["baseline_ref_actual_smoothing_window_sec"] == 300.0
    assert cand_300["baseline_ref_smoothing_window_sec"] == 300.0


def test_large_window_warning_without_adjustment():
    signal, reference, fs = _shared_slow_trace(duration_sec=600.0)

    candidate = compute_baseline_reference_candidate(signal, reference, fs, smoothing_window_sec=300.0)

    assert candidate["baseline_ref_requested_smoothing_window_sec"] == 300.0
    assert candidate["baseline_ref_actual_smoothing_window_sec"] == 300.0
    assert candidate["baseline_ref_smoothing_window_fraction_of_chunk"] == 0.5
    assert candidate["baseline_ref_smoothing_window_adjusted"] is False
    assert "large_fraction" in candidate["baseline_ref_smoothing_window_warning"]


def test_too_large_window_is_adjusted_to_max_chunk_fraction():
    signal, reference, fs = _shared_slow_trace(duration_sec=600.0)

    candidate = compute_baseline_reference_candidate(signal, reference, fs, smoothing_window_sec=500.0)

    assert candidate["baseline_ref_requested_smoothing_window_sec"] == 500.0
    assert candidate["baseline_ref_actual_smoothing_window_sec"] == 450.0
    assert candidate["baseline_ref_smoothing_window_fraction_of_chunk"] == 0.75
    assert candidate["baseline_ref_smoothing_window_adjusted"] is True
    assert "too_large_for_chunk_adjusted" in candidate["baseline_ref_smoothing_window_warning"]


def test_max_fraction_is_not_violated_when_min_window_conflicts():
    signal, reference, fs = _shared_slow_trace(duration_sec=70.0)

    candidate = compute_baseline_reference_candidate(signal, reference, fs, smoothing_window_sec=500.0)

    assert candidate["baseline_ref_candidate_available"] is True
    assert candidate["baseline_ref_requested_smoothing_window_sec"] == 500.0
    assert candidate["baseline_ref_actual_smoothing_window_sec"] <= 52.5
    assert candidate["baseline_ref_actual_smoothing_window_sec"] != 60.0
    assert candidate["baseline_ref_smoothing_window_fraction_of_chunk"] <= 0.75
    assert candidate["baseline_ref_smoothing_window_adjusted"] is True
    assert "max_fraction_smaller_than_min_window" in candidate["baseline_ref_smoothing_window_warning"]


def test_invalid_requested_window_uses_default_and_records_warning():
    signal, reference, fs = _shared_slow_trace(duration_sec=900.0)

    candidate = compute_baseline_reference_candidate(signal, reference, fs, smoothing_window_sec=0.0)

    assert candidate["baseline_ref_requested_smoothing_window_sec"] == 300.0
    assert candidate["baseline_ref_actual_smoothing_window_sec"] == 300.0
    assert candidate["baseline_ref_default_smoothing_window_sec"] == 300.0
    assert candidate["baseline_ref_smoothing_window_adjusted"] is False
    assert "invalid_requested_smoothing_window" in candidate["baseline_ref_smoothing_window_warning"]


def test_reflected_smoothing_avoids_large_edge_deviation():
    fs = 10.0
    t = np.arange(7200, dtype=float) / fs
    slow = np.sin(2.0 * np.pi * (1.0 / 1800.0) * t)
    window_samples = int(round(300.0 * fs))

    def _shrinking_window_smooth(values, window):
        arr = np.asarray(values, dtype=float)
        kernel = np.ones(window, dtype=float)
        finite = np.isfinite(arr)
        filled = np.where(finite, arr, 0.0)
        sums = np.convolve(filled, kernel, mode="same")
        counts = np.convolve(finite.astype(float), kernel, mode="same")
        out = np.full(arr.shape, np.nan, dtype=float)
        valid = counts > 0.0
        out[valid] = sums[valid] / counts[valid]
        return out

    smooth = _nan_aware_moving_average(slow, window_samples)
    shrinking = _shrinking_window_smooth(slow, window_samples)

    assert smooth.shape == slow.shape
    assert np.all(np.isfinite(smooth))
    assert abs(smooth[0] - smooth[200]) < 0.25 * abs(shrinking[0] - shrinking[200])
    assert abs(smooth[-1] - smooth[-201]) < 0.25 * abs(shrinking[-1] - shrinking[-201])


def test_broad_signal_only_event_is_excluded_from_baseline_fit():
    fs = 10.0
    t = np.arange(18000, dtype=float) / fs
    slow = np.sin(2.0 * np.pi * (1.0 / 700.0) * t)
    reference = 1.0 + 0.8 * slow
    signal_control = 2.0 + 1.2 * slow
    event = 4.0 * np.exp(-0.5 * ((t - 900.0) / 25.0) ** 2)
    signal_event = signal_control + event

    control = compute_baseline_reference_candidate(signal_control, reference, fs)
    event_candidate = compute_baseline_reference_candidate(signal_event, reference, fs)
    control_trace = control["baseline_ref_candidate"]
    event_trace = event_candidate["baseline_ref_candidate"]
    event_window = (t >= 810.0) & (t <= 990.0)

    assert control["baseline_ref_candidate_available"] is True
    assert event_candidate["baseline_ref_candidate_available"] is True
    assert event_candidate["baseline_ref_fit_stage"] == "residual_refit"
    assert event_candidate["baseline_ref_residual_exclusion_fraction"] > 0.01
    assert abs(event_candidate["baseline_ref_slope"] - control["baseline_ref_slope"]) < 0.15
    assert abs(event_candidate["baseline_ref_intercept"] - control["baseline_ref_intercept"]) < 0.15
    assert float(np.nanmax(np.abs(event_trace[event_window] - control_trace[event_window]))) < 0.25
    assert np.corrcoef(event_trace[np.isfinite(event_trace)], slow[np.isfinite(event_trace)])[0, 1] > 0.7


def test_candidate_preserves_slow_baseline_structure():
    fs = 10.0
    t = np.arange(7200, dtype=float) / fs
    slow = np.sin(2.0 * np.pi * (1.0 / 600.0) * t)
    reference = 1.0 + 0.8 * slow + 0.05 * np.sin(2.0 * np.pi * 0.1 * t)
    signal = 2.0 + 1.2 * slow + 0.03 * np.cos(2.0 * np.pi * 0.12 * t)

    candidate = compute_baseline_reference_candidate(signal, reference, fs)
    baseline_ref = candidate["baseline_ref_candidate"]
    metrics = compute_baseline_reference_candidate_metrics(
        signal,
        reference,
        baseline_ref,
        baseline_ref,
        fs,
    )

    assert candidate["baseline_ref_candidate_available"] is True
    assert np.sum(np.isfinite(baseline_ref)) == signal.size
    assert float(np.nanstd(baseline_ref)) > 1e-3
    assert metrics["signal_baseline_ref_corr"] > 0.5
    assert metrics["baseline_ref_baseline_scale_fraction"] > 0.05


def test_candidate_insufficient_samples_is_safe():
    signal = np.array([1.0, np.nan, 1.2])
    reference = np.array([0.5, 0.6, np.nan])

    candidate = compute_baseline_reference_candidate(signal, reference, 10.0, min_samples=10)

    assert candidate["baseline_ref_candidate_available"] is False
    assert candidate["baseline_ref_candidate"] is None
    assert candidate["baseline_ref_warning"] == "insufficient_finite_samples"
    assert candidate["baseline_ref_fit_stage"] is None


def test_candidate_metadata_is_json_safe():
    signal = np.array([1.0, np.nan, 1.2])
    reference = np.array([0.5, 0.6, np.nan])

    candidate = compute_baseline_reference_candidate(signal, reference, 10.0, min_samples=10)

    text = json.dumps(_metadata_without_trace(candidate), allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text


def test_candidate_diagnostics_return_fit_intermediates():
    signal, reference, fs = _shared_slow_trace(duration_sec=300.0)

    candidate = compute_baseline_reference_candidate(
        signal,
        reference,
        fs,
        smoothing_window_sec=60.0,
        return_diagnostics=True,
    )

    assert candidate["baseline_ref_candidate_available"] is True
    assert candidate["baseline_ref_smoothed_signal"].shape == signal.shape
    assert candidate["baseline_ref_smoothed_reference"].shape == reference.shape
    assert candidate["baseline_ref_candidate"].shape == signal.shape
    assert candidate["baseline_ref_fit_included_mask"].shape == signal.shape
    assert candidate["baseline_ref_final_slope"] is not None
    assert candidate["baseline_ref_final_intercept"] is not None
    assert candidate["baseline_ref_initial_slope"] is not None
    assert candidate["baseline_ref_initial_intercept"] is not None
    assert candidate["baseline_ref_residual_exclusion_fraction"] is not None


def test_baseline_fit_relationship_class_negative_case():
    result = classify_baseline_fit_relationship(slope=-1.0, corr=-0.8)

    assert result == "negative_reference_relationship"


def test_baseline_fit_relationship_class_weak_case():
    result = classify_baseline_fit_relationship(slope=1.0, corr=0.05)

    assert result == "weak_reference_relationship"
