import numpy as np

from photometry_pipeline.core.signal_state_diagnostics import (
    compute_signal_state_diagnostics,
)


def test_ordinary_dynamic_signal_is_not_high_state_candidate():
    rng = np.random.default_rng(1)
    n = 1000
    t = np.arange(n, dtype=float) / 20.0
    signal = (
        1.0
        + 0.15 * np.sin(2.0 * np.pi * 0.2 * t)
        + 0.05 * rng.normal(size=n)
    )
    signal[200:205] += 1.0
    signal[700:705] += 0.8

    out = compute_signal_state_diagnostics(signal, t)

    assert out["signal_state_diagnostics_available"] is True
    assert out["signal_high_state_candidate_present"] is False
    assert out["signal_state_candidate_class"] in {
        "ordinary_dynamic_candidate",
        "uncertain_signal_state",
    }


def test_sustained_high_state_signal_is_detected():
    rng = np.random.default_rng(2)
    n = 1200
    t = np.arange(n, dtype=float) / 20.0
    signal = 1.0 + 0.15 * np.sin(2.0 * np.pi * 0.3 * t) + 0.05 * rng.normal(size=n)
    signal[500:] = 2.8 + 0.005 * rng.normal(size=n - 500)

    out = compute_signal_state_diagnostics(signal, t)

    assert out["signal_high_state_candidate_present"] is True
    assert out["signal_high_state_fraction"] > 0.25
    assert out["signal_variability_suppression_score"] > 0.2
    assert out["signal_state_candidate_class"] in {
        "candidate_sustained_high_state",
        "candidate_mixed_dynamic_high_state",
        "candidate_edge_high_state",
    }
    assert "SIGNAL_HIGH_STATE_CANDIDATE" in out["signal_state_flags"]


def test_edge_high_state_signal_is_detected():
    rng = np.random.default_rng(3)
    n = 1000
    t = np.arange(n, dtype=float) / 20.0
    signal = 1.0 + 0.08 * rng.normal(size=n)
    signal[:300] = 2.5 + 0.004 * rng.normal(size=300)

    out = compute_signal_state_diagnostics(signal, t)

    assert out["signal_edge_high_state_present"] is True
    assert out["signal_start_high_state_candidate"] is True
    assert out["signal_state_candidate_class"] == "candidate_edge_high_state"
    assert "SIGNAL_EDGE_HIGH_STATE_CANDIDATE" in out["signal_state_flags"]


def test_step_like_transitions_are_counted():
    n = 1500
    t = np.arange(n, dtype=float) / 20.0
    signal = np.ones(n, dtype=float)
    signal[400:1000] = 2.5
    signal += 0.01 * np.sin(2.0 * np.pi * 0.4 * t)

    out = compute_signal_state_diagnostics(signal, t)

    assert out["signal_step_like_transition_present"] is True
    assert out["signal_step_transition_count"] == 2
    assert out["signal_step_up_count"] == 1
    assert out["signal_step_down_count"] == 1
    assert out["signal_max_step_robust_z"] >= out["signal_state_step_threshold_robust_z"]


def test_low_variability_ratio_threshold_is_configurable():
    rng = np.random.default_rng(4)
    n = 1200
    t = np.arange(n, dtype=float) / 20.0
    signal = 1.0 + 0.06 * rng.normal(size=n)
    signal[500:] = 2.8 + 0.02 * rng.normal(size=n - 500)

    permissive = compute_signal_state_diagnostics(
        signal,
        t,
        config={"signal_state_low_variability_ratio_threshold": 0.90},
    )
    strict = compute_signal_state_diagnostics(
        signal,
        t,
        config={"signal_state_low_variability_ratio_threshold": 0.01},
    )

    assert permissive["signal_state_low_variability_ratio_threshold"] == 0.90
    assert strict["signal_state_low_variability_ratio_threshold"] == 0.01
    assert "SIGNAL_LOW_VARIABILITY_HIGH_STATE" in permissive["signal_state_flags"]
    assert "SIGNAL_LOW_VARIABILITY_HIGH_STATE" not in strict["signal_state_flags"]


def test_insufficient_or_flat_signal_reports_insufficient_information():
    flat = np.ones(100, dtype=float)

    out = compute_signal_state_diagnostics(flat)

    assert out["signal_state_diagnostics_available"] is False
    assert out["signal_state_candidate_class"] == "insufficient_signal_state_information"
    assert "SIGNAL_INSUFFICIENT_RANGE" in out["signal_state_flags"]


def test_fractional_episode_threshold_does_not_require_fixed_duration():
    for n in (200, 2000):
        signal = np.ones(n, dtype=float)
        high_start = int(0.60 * n)
        signal[high_start:] = 3.0
        t = np.arange(n, dtype=float)

        out = compute_signal_state_diagnostics(
            signal,
            t,
            config={
                "signal_state_min_episode_fraction": 0.25,
                "signal_state_min_episode_sec": 0.0,
                "signal_state_smoothing_window_fraction": 0.03,
            },
        )

        assert out["signal_high_state_candidate_present"] is True
        assert out["signal_longest_high_state_fraction"] >= 0.25
        assert out["signal_state_min_episode_fraction"] == 0.25
        assert out["signal_state_min_episode_sec"] == 0.0
