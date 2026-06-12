import numpy as np

from photometry_pipeline.core.signal_only_f0_candidate import compute_signal_only_f0_candidate
from photometry_pipeline.core.signal_state_diagnostics import compute_signal_state_diagnostics


def _ordinary_signal(seed=1, n=1200):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) / 20.0
    slow = 1.0 + 0.25 * np.sin(2.0 * np.pi * 0.01 * t)
    signal = slow + 0.1 * np.sin(2.0 * np.pi * 0.3 * t) + 0.03 * rng.normal(size=n)
    for start in (200, 700, 950):
        signal[start:start + 20] += 0.5
    return t, signal, slow


def test_ordinary_dynamic_signal_with_slow_drift_has_usable_lower_envelope():
    t, signal, slow = _ordinary_signal()

    out = compute_signal_only_f0_candidate(signal, t)
    candidate = out["signal_only_f0_candidate"]

    assert out["signal_only_f0_candidate_available"] is True
    assert out["signal_only_f0_candidate_viability"] in {"viable", "contextual"}
    assert out["signal_only_f0_above_signal_fraction"] <= 0.01
    assert out["signal_only_f0_support_fraction"] >= 0.95
    assert np.nanmedian(signal - candidate) > 0.0
    assert np.corrcoef(candidate[np.isfinite(candidate)], slow[np.isfinite(candidate)])[0, 1] > 0.5
    assert "SIGNAL_ONLY_F0_AVAILABLE" in out["signal_only_f0_flags"]


def test_sustained_high_state_adds_context_and_does_not_chase_plateau():
    rng = np.random.default_rng(2)
    t, signal, _slow = _ordinary_signal(seed=2)
    signal[500:] = 2.5 + 0.005 * rng.normal(size=signal.size - 500)
    signal_state = compute_signal_state_diagnostics(signal, t)

    out = compute_signal_only_f0_candidate(signal, t, signal_state=signal_state)
    candidate = out["signal_only_f0_candidate"]

    assert out["signal_only_f0_candidate_available"] is True
    assert "SIGNAL_ONLY_F0_HIGH_STATE_PRESENT" in out["signal_only_f0_flags"]
    assert out["signal_only_f0_high_state_context_mode"] == "contextual_cap"
    assert out["signal_only_f0_high_state_context_cap"] is not None
    assert out["signal_only_f0_high_state_context_applied"] is True
    assert "signal_only_f0_high_state_exclusion_fraction" not in out
    assert out["signal_only_f0_candidate_viability"] == "contextual"
    assert np.nanmedian(candidate[700:]) < np.nanmedian(signal[700:]) - 0.25


def test_partial_mixed_high_state_adds_contextual_flag():
    rng = np.random.default_rng(3)
    n = 1000
    t = np.arange(n, dtype=float) / 20.0
    signal = 1.0 + 0.12 * np.sin(2.0 * np.pi * 0.23 * t) + 0.05 * rng.normal(size=n)
    signal[455:570] = 2.3 + 0.006 * rng.normal(size=115)
    signal_state = compute_signal_state_diagnostics(
        signal,
        t,
        config={
            "signal_state_smoothing_window_fraction": 0.025,
            "signal_state_variability_window_fraction": 0.025,
        },
    )

    out = compute_signal_only_f0_candidate(signal, t, signal_state=signal_state)

    assert out["signal_only_f0_candidate_available"] is True
    assert "SIGNAL_ONLY_F0_PARTIAL_HIGH_STATE_PRESENT" in out["signal_only_f0_flags"]
    assert out["signal_only_f0_candidate_viability"] in {"viable", "contextual"}


def test_flat_or_insufficient_signal_is_unavailable():
    out = compute_signal_only_f0_candidate(np.ones(100, dtype=float))

    assert out["signal_only_f0_candidate_available"] is False
    assert out["signal_only_f0_candidate_viability"] == "unavailable"
    assert "SIGNAL_ONLY_F0_INSUFFICIENT_RANGE" in out["signal_only_f0_flags"]


def test_excessive_tracking_metric_can_trigger_contextual_status():
    t, signal, _slow = _ordinary_signal(seed=4)

    out = compute_signal_only_f0_candidate(
        signal,
        t,
        config={
            "signal_only_f0_low_quantile": 0.5,
            "signal_only_f0_window_fraction": 0.02,
            "signal_only_f0_smoothing_window_fraction": 0.01,
            "signal_only_f0_max_tracking_fraction": 0.2,
        },
    )

    assert out["signal_only_f0_tracking_score"] > out["signal_only_f0_max_tracking_fraction"]
    assert out["signal_only_f0_candidate_viability"] in {"contextual", "hard_inspect"}
    assert "SIGNAL_ONLY_F0_EXCESSIVE_TRACKING" in out["signal_only_f0_flags"]


def test_above_signal_qc_uses_pre_cap_fraction():
    rng = np.random.default_rng(7)
    n = 1200
    t = np.arange(n, dtype=float) / 20.0
    slow = 1.0 + 0.25 * np.sin(2.0 * np.pi * 0.01 * t)
    signal = slow + 0.2 * np.sin(2.0 * np.pi * 0.4 * t) + 0.01 * rng.normal(size=n)

    out = compute_signal_only_f0_candidate(
        signal,
        t,
        config={
            "signal_only_f0_low_quantile": 0.5,
            "signal_only_f0_window_fraction": 0.02,
            "signal_only_f0_smoothing_window_fraction": 0.10,
            "signal_only_f0_max_above_signal_fraction": 0.05,
        },
    )

    assert out["signal_only_f0_above_signal_fraction_pre_cap"] > 0.05
    assert out["signal_only_f0_above_signal_fraction"] <= 0.01
    assert "SIGNAL_ONLY_F0_ABOVE_SIGNAL_EXCESSIVE" in out["signal_only_f0_flags"]
    assert out["signal_only_f0_candidate_viability"] == "hard_inspect"


def test_configurable_windows_and_quantiles_are_recorded_and_affect_candidate():
    t, signal, _slow = _ordinary_signal(seed=5)

    low_q = compute_signal_only_f0_candidate(
        signal,
        t,
        config={"signal_only_f0_low_quantile": 0.05, "signal_only_f0_window_fraction": 0.10},
    )
    high_q = compute_signal_only_f0_candidate(
        signal,
        t,
        config={"signal_only_f0_low_quantile": 0.30, "signal_only_f0_window_fraction": 0.10},
    )

    assert low_q["signal_only_f0_low_quantile"] == 0.05
    assert high_q["signal_only_f0_low_quantile"] == 0.30
    assert low_q["signal_only_f0_window_fraction"] == 0.10
    assert high_q["signal_only_f0_window_fraction"] == 0.10
    assert np.nanmedian(high_q["signal_only_f0_candidate"]) > np.nanmedian(
        low_q["signal_only_f0_candidate"]
    )
