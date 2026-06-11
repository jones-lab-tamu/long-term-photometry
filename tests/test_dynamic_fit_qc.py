import json

import numpy as np
import pytest

from photometry_pipeline.core.dynamic_fit_qc import (
    compute_dynamic_fit_validity_metrics,
)


def test_flat_fitted_reference_is_flagged():
    fs = 10.0
    t = np.arange(1000, dtype=float) / fs
    signal = 2.0 + np.sin(2.0 * np.pi * 0.02 * t)
    iso = 1.0 + 0.5 * np.sin(2.0 * np.pi * 0.02 * t + 0.2)
    fitted_ref = np.full_like(signal, 2.1)

    metrics = compute_dynamic_fit_validity_metrics(signal, iso, fitted_ref, fs)

    assert metrics["dynamic_fit_reference_flat_or_uninformative"] is True
    assert "FITTED_REFERENCE_FLAT_OR_UNINFORMATIVE" in metrics["dynamic_fit_qc_flags"]
    assert metrics["fitted_ref_to_signal_range_ratio"] < 0.05


def test_informative_fitted_reference_is_not_flat():
    fs = 10.0
    t = np.arange(1200, dtype=float) / fs
    signal = 2.0 + np.sin(2.0 * np.pi * 0.02 * t)
    iso = 1.0 + 0.8 * np.sin(2.0 * np.pi * 0.02 * t + 0.1)
    fitted_ref = 1.8 + 0.7 * np.sin(2.0 * np.pi * 0.02 * t + 0.05)

    metrics = compute_dynamic_fit_validity_metrics(signal, iso, fitted_ref, fs)

    assert metrics["dynamic_fit_reference_flat_or_uninformative"] is False
    assert metrics["fitted_ref_to_signal_range_ratio"] > 0.2
    assert metrics["fitted_ref_to_iso_range_ratio"] > 0.2


def test_negative_or_mixed_slope_is_flagged():
    fs = 10.0
    t = np.arange(600, dtype=float) / fs
    signal = np.sin(2.0 * np.pi * 0.03 * t)
    iso = np.cos(2.0 * np.pi * 0.03 * t)
    fitted_ref = 0.5 * iso
    slope = np.r_[np.full(250, -0.5), np.full(350, 0.8)]

    metrics = compute_dynamic_fit_validity_metrics(signal, iso, fitted_ref, fs, slope=slope)

    assert metrics["dynamic_fit_negative_or_mixed_coupling"] is True
    assert "NEGATIVE_OR_MIXED_REFERENCE_COUPLING" in metrics["dynamic_fit_qc_flags"]
    assert metrics["slope_fraction_negative"] > 0.25


def test_response_scale_power_is_flagged():
    fs = 10.0
    t = np.arange(3600, dtype=float) / fs
    fitted_ref = np.sin(2.0 * np.pi * (1.0 / 30.0) * t)
    signal = fitted_ref + 0.2 * np.sin(2.0 * np.pi * 0.005 * t)
    iso = fitted_ref + 0.1 * np.cos(2.0 * np.pi * 0.01 * t)

    metrics = compute_dynamic_fit_validity_metrics(signal, iso, fitted_ref, fs)

    assert metrics["dynamic_fit_response_scale_rich"] is True
    assert "FITTED_REFERENCE_RESPONSE_SCALE_RICH" in metrics["dynamic_fit_qc_flags"]
    assert metrics["fitted_ref_response_scale_fraction"] > 0.35


def test_baseline_scale_power_is_not_response_rich():
    fs = 10.0
    t = np.arange(7200, dtype=float) / fs
    fitted_ref = np.sin(2.0 * np.pi * (1.0 / 600.0) * t)
    signal = fitted_ref + 0.1 * np.sin(2.0 * np.pi * 0.05 * t)
    iso = fitted_ref + 0.1 * np.cos(2.0 * np.pi * 0.04 * t)

    metrics = compute_dynamic_fit_validity_metrics(signal, iso, fitted_ref, fs)

    assert metrics["fitted_ref_baseline_scale_fraction"] > 0.5
    assert metrics["dynamic_fit_response_scale_rich"] is False


def test_nan_and_degenerate_inputs_do_not_crash_and_are_json_serializable():
    signal = np.array([1.0, np.nan, 1.0, 1.0])
    iso = np.array([2.0, 2.0, np.nan, 2.0])
    fitted_ref = np.array([0.5, 0.5, 0.5, np.nan])

    metrics = compute_dynamic_fit_validity_metrics(signal, iso, fitted_ref, 10.0)

    assert metrics["n_samples"] == 4
    assert metrics["signal_iso_corr_reason"] is not None
    json.dumps(metrics)


def test_dynamic_fit_qc_rejects_incompatible_lengths():
    with pytest.raises(ValueError, match="Input length mismatch"):
        compute_dynamic_fit_validity_metrics([1, 2], [1, 2], [1], 10.0)
