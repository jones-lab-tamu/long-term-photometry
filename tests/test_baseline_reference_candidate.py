import json

import numpy as np

from photometry_pipeline.core.baseline_reference_candidate import (
    classify_baseline_fit_relationship,
    compute_baseline_reference_candidate,
    compute_baseline_reference_candidate_metrics,
    _nan_aware_moving_average,
    _window_sum,
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


# ---------------------------------------------------------------------------
# Numerical equivalence: _nan_aware_moving_average is now an O(n) cumulative-
# sum implementation instead of a direct O(n * window) np.convolve. This
# section retains the previous direct-convolution semantics as a private
# reference implementation and proves the two agree, to a tight floating-
# point tolerance, across the full documented contract (centered window,
# forced-odd window <= n, reflect padding, NaN/Inf treated as missing with
# zero weight, NaN output where a window has zero finite support).
# ---------------------------------------------------------------------------


def _reference_nan_aware_moving_average(values, window_samples):
    """The previous (pre-optimization) direct np.convolve implementation,
    kept only as a numerical-equivalence reference -- not production code."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    window = int(max(1, window_samples))
    window = min(window, int(arr.size))
    if window % 2 == 0 and window > 1:
        window -= 1
    kernel = np.ones(window, dtype=float)
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    pad = window // 2
    if pad > 0:
        filled = np.pad(filled, pad_width=pad, mode="reflect")
        finite_weights = np.pad(finite.astype(float), pad_width=pad, mode="reflect")
    else:
        finite_weights = finite.astype(float)
    sums = np.convolve(filled, kernel, mode="valid")
    counts = np.convolve(finite_weights, kernel, mode="valid")
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = counts > 0.0
    out[valid] = sums[valid] / counts[valid]
    return out


def _assert_equivalent(old, new, *, max_abs_tol=1e-9, max_rel_tol=1e-9):
    assert old.shape == new.shape
    nan_old = np.isnan(old)
    nan_new = np.isnan(new)
    assert np.array_equal(nan_old, nan_new), "NaN-location mismatch between old and new implementations"
    both_finite = ~nan_old & ~nan_new
    if not both_finite.any():
        return 0.0, 0.0
    abs_diff = np.abs(old[both_finite] - new[both_finite])
    denom = np.maximum(np.abs(old[both_finite]), 1e-300)
    rel_diff = abs_diff / denom
    max_abs = float(abs_diff.max())
    max_rel = float(rel_diff.max())
    assert max_abs <= max_abs_tol, f"max abs diff {max_abs} exceeds tolerance {max_abs_tol}"
    assert max_rel <= max_rel_tol, f"max rel diff {max_rel} exceeds tolerance {max_rel_tol}"
    return max_abs, max_rel


def test_moving_average_equivalence_random_finite():
    rng = np.random.default_rng(42)
    arr = rng.normal(size=1000)
    old = _reference_nan_aware_moving_average(arr, 51)
    new = _nan_aware_moving_average(arr, 51)
    _assert_equivalent(old, new)


def test_moving_average_equivalence_random_with_nans():
    rng = np.random.default_rng(43)
    arr = rng.normal(size=1000)
    arr[rng.choice(1000, 100, replace=False)] = np.nan
    old = _reference_nan_aware_moving_average(arr, 51)
    new = _nan_aware_moving_average(arr, 51)
    _assert_equivalent(old, new)


def test_moving_average_equivalence_all_nan():
    arr = np.full(500, np.nan)
    old = _reference_nan_aware_moving_average(arr, 51)
    new = _nan_aware_moving_average(arr, 51)
    _assert_equivalent(old, new)
    assert np.all(np.isnan(new))


def test_moving_average_equivalence_isolated_finite_value():
    arr = np.full(200, np.nan)
    arr[100] = 5.0
    old = _reference_nan_aware_moving_average(arr, 21)
    new = _nan_aware_moving_average(arr, 21)
    _assert_equivalent(old, new)
    assert new[100] == 5.0


def test_moving_average_equivalence_long_nan_run():
    rng = np.random.default_rng(44)
    arr = rng.normal(size=500)
    arr[100:400] = np.nan
    old = _reference_nan_aware_moving_average(arr, 51)
    new = _nan_aware_moving_average(arr, 51)
    _assert_equivalent(old, new)


def test_moving_average_equivalence_window_one():
    rng = np.random.default_rng(45)
    arr = rng.normal(size=100)
    old = _reference_nan_aware_moving_average(arr, 1)
    new = _nan_aware_moving_average(arr, 1)
    # window=1 is conceptually an identity pass-through, but the new
    # cumulative-sum implementation still derives it as csum[i+1]-csum[i],
    # which is not bit-identical to the old direct value in floating point
    # (a machine-epsilon-scale cancellation artifact, not a semantic
    # difference) -- a tight machine-precision tolerance is the correct,
    # justified bound here, not exact equality.
    _assert_equivalent(old, new, max_abs_tol=1e-12, max_rel_tol=1e-12)
    np.testing.assert_allclose(new, arr, atol=1e-12, rtol=1e-12)


def test_moving_average_equivalence_small_odd_window():
    rng = np.random.default_rng(46)
    arr = rng.normal(size=100)
    old = _reference_nan_aware_moving_average(arr, 7)
    new = _nan_aware_moving_average(arr, 7)
    _assert_equivalent(old, new)


def test_moving_average_equivalence_small_even_window():
    rng = np.random.default_rng(47)
    arr = rng.normal(size=100)
    old = _reference_nan_aware_moving_average(arr, 8)
    new = _nan_aware_moving_average(arr, 8)
    _assert_equivalent(old, new)


def test_moving_average_equivalence_window_near_input_length():
    rng = np.random.default_rng(48)
    arr = rng.normal(size=100)
    old = _reference_nan_aware_moving_average(arr, 97)
    new = _nan_aware_moving_average(arr, 97)
    _assert_equivalent(old, new)


def test_moving_average_equivalence_window_equal_input_length():
    rng = np.random.default_rng(49)
    arr = rng.normal(size=100)
    old = _reference_nan_aware_moving_average(arr, 100)
    new = _nan_aware_moving_average(arr, 100)
    _assert_equivalent(old, new)


def test_moving_average_equivalence_window_larger_than_input():
    rng = np.random.default_rng(50)
    arr = rng.normal(size=100)
    old = _reference_nan_aware_moving_average(arr, 500)
    new = _nan_aware_moving_average(arr, 500)
    _assert_equivalent(old, new)


def test_moving_average_equivalence_with_infinities():
    rng = np.random.default_rng(51)
    arr = rng.normal(size=300)
    arr[10] = np.inf
    arr[20] = -np.inf
    old = _reference_nan_aware_moving_average(arr, 25)
    new = _nan_aware_moving_average(arr, 25)
    _assert_equivalent(old, new)
    # Infinities are treated as missing (excluded), exactly like NaN, by both
    # implementations -- neither the old nor new output should ever be inf.
    assert not np.any(np.isinf(new))


def test_moving_average_equivalence_empty_array():
    old = _reference_nan_aware_moving_average(np.array([]), 10)
    new = _nan_aware_moving_average(np.array([]), 10)
    assert old.shape == new.shape == (0,)


def test_moving_average_equivalence_large_window_relative_to_n():
    """A scaled-down proxy for the real NPM production shape (n~30000,
    window~15000, ~50% window fraction): same window-large-relative-to-n
    ratio, same reflected-padding-plus-cumulative-sum-subtraction-over-many-
    samples code path, with scattered NaNs, but sized to complete quickly in
    routine test runs. The actual full production-shape equivalence (n=30000,
    window=15000) was already measured directly in the development report's
    benchmark and full real-run comparison and does not need to be re-paid on
    every automated test invocation, since the old reference implementation
    costs ~9.4s at that exact shape."""
    rng = np.random.default_rng(52)
    arr = rng.normal(loc=5.0, scale=0.3, size=3000)
    arr[rng.choice(3000, 150, replace=False)] = np.nan
    old = _reference_nan_aware_moving_average(arr, 1500)
    new = _nan_aware_moving_average(arr, 1500)
    max_abs, max_rel = _assert_equivalent(old, new, max_abs_tol=1e-8, max_rel_tol=1e-8)
    print(f"large_window_relative_to_n: max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}")


def test_window_sum_matches_reference_convolution():
    """Direct correctness test for _window_sum against a small reference
    np.convolve(..., mode="valid") -- the exact primitive
    _nan_aware_moving_average was changed to use."""
    rng = np.random.default_rng(54)
    padded = rng.normal(size=47)
    window = 9
    expected = np.convolve(padded, np.ones(window, dtype=float), mode="valid")
    actual = _window_sum(padded, window)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_nan_aware_moving_average_never_calls_convolve(monkeypatch):
    """Structural regression: production _nan_aware_moving_average must use
    the O(n) _window_sum primitive for both value sums and finite counts,
    and must never fall back to (or reintroduce) the O(n * window) direct
    np.convolve implementation."""
    import photometry_pipeline.core.baseline_reference_candidate as module

    def _forbidden_convolve(*_args, **_kwargs):
        raise AssertionError("np.convolve must not be called by _nan_aware_moving_average")

    monkeypatch.setattr(module.np, "convolve", _forbidden_convolve)

    calls = []
    real_window_sum = module._window_sum

    def _spy_window_sum(padded, window):
        calls.append((len(padded), window))
        return real_window_sum(padded, window)

    monkeypatch.setattr(module, "_window_sum", _spy_window_sum)

    rng = np.random.default_rng(55)
    arr = rng.normal(size=200)
    arr[5:15] = np.nan
    result = _nan_aware_moving_average(arr, 21)

    assert result.shape == arr.shape
    # Exactly two _window_sum calls: one for the value sums, one for the
    # finite-weight counts.
    assert len(calls) == 2
