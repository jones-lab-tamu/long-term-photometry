import subprocess
import sys

import numpy as np
import pytest

from photometry_pipeline.core import regression
from photometry_pipeline.core.dynamic_fitting import (
    fit_adaptive_event_gated_regression,
    fit_robust_global_event_reject,
)


def _robust_kwargs(fs_hz: float) -> dict:
    return {
        "max_iters": 4,
        "residual_z_thresh": 3.0,
        "local_var_window_sec": None,
        "local_var_ratio_thresh": None,
        "min_keep_fraction": 0.5,
        "sample_rate_hz": fs_hz,
        "use_intercept": True,
        "signal_excursion_polarity": "positive",
    }


def _adaptive_kwargs(fs_hz: float) -> dict:
    return {
        "sample_rate_hz": fs_hz,
        "residual_z_thresh": 3.0,
        "local_var_window_sec": None,
        "local_var_ratio_thresh": None,
        "smooth_window_sec": 30.0,
        "min_trust_fraction": 0.5,
        "freeze_interp_method": "linear_hold",
        "use_intercept": True,
        "signal_excursion_polarity": "positive",
    }


def test_dynamic_fitting_import_does_not_import_pipeline_or_gui():
    code = (
        "import sys; "
        "import photometry_pipeline.core.dynamic_fitting; "
        "assert 'photometry_pipeline.pipeline' not in sys.modules; "
        "assert not any(name == 'gui' or name.startswith('gui.') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_robust_global_event_reject_recovers_background_and_contract():
    n = 3200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.3 * np.sin(2.0 * np.pi * 0.025 * t)
    true_slope = 1.65
    true_intercept = 0.75
    background = true_slope * iso + true_intercept
    event = 3.2 * np.exp(-0.5 * ((t - 36.0) / 2.4) ** 2)
    signal = background + event

    result = fit_robust_global_event_reject(
        signal_raw=signal,
        iso_raw=iso,
        **_robust_kwargs(fs),
    )

    unmasked_slope, unmasked_intercept = np.polyfit(iso, signal, 1)
    coef = result["final_coef"]
    robust_error = abs(float(coef["slope"]) - true_slope) + abs(
        float(coef["intercept"]) - true_intercept
    )
    unmasked_error = abs(float(unmasked_slope) - true_slope) + abs(
        float(unmasked_intercept) - true_intercept
    )
    event_mask = np.abs(t - 36.0) <= 2.0

    assert robust_error < unmasked_error
    assert float(np.mean(result["excluded_mask"][event_mask])) > 0.2
    assert int(coef["n_kept"]) < int(coef["n_finite"])
    assert result["iso_fit_signal_units"].shape == signal.shape
    assert result["keep_mask"].shape == signal.shape


def test_adaptive_event_gated_tracks_changing_slope_and_gates_event():
    n = 3600
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.35 * np.sin(2.0 * np.pi * 0.02 * t)
    slope_true = 1.3 + 0.35 * (t / t[-1])
    background = slope_true * iso + 0.6
    event = 2.8 * np.exp(-0.5 * ((t - 45.0) / 2.0) ** 2)
    signal = background + event

    result = fit_adaptive_event_gated_regression(
        signal_raw=signal,
        iso_raw=iso,
        signal_fit_input=signal,
        iso_fit_input=iso,
        **_adaptive_kwargs(fs),
    )

    global_slope = float(np.polyfit(iso, signal, 1)[0])
    adaptive_mae = float(np.mean(np.abs(result["coef_slope"] - slope_true)))
    global_mae = float(np.mean(np.abs(global_slope - slope_true)))
    event_mask = np.abs(t - 45.0) <= 1.8

    assert adaptive_mae < global_mae
    assert float(np.mean(result["gated_mask"][event_mask])) > 0.2
    assert result["iso_fit_signal_units"].shape == signal.shape
    assert result["coef_slope"].shape == signal.shape
    assert result["n_trusted"] < result["n_finite"]


@pytest.mark.parametrize(
    "standalone,production,kwargs",
    [
        (
            fit_robust_global_event_reject,
            regression.fit_robust_global_event_reject,
            _robust_kwargs(40.0),
        ),
        (
            fit_adaptive_event_gated_regression,
            regression.fit_adaptive_event_gated_regression,
            _adaptive_kwargs(40.0),
        ),
    ],
)
def test_standalone_functions_match_production_helpers(standalone, production, kwargs):
    n = 2400
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.1 + 0.3 * np.sin(2.0 * np.pi * 0.025 * t)
    signal = 1.55 * iso + 0.7 + 2.5 * np.exp(-0.5 * ((t - 28.0) / 2.0) ** 2)
    if standalone is fit_adaptive_event_gated_regression:
        kwargs = {**kwargs, "signal_fit_input": signal, "iso_fit_input": iso}

    extracted = standalone(signal_raw=signal, iso_raw=iso, **kwargs)
    current = production(signal_raw=signal, iso_raw=iso, **kwargs)

    assert extracted.keys() == current.keys()
    for key in extracted:
        left = extracted[key]
        right = current[key]
        if isinstance(left, np.ndarray):
            if left.dtype == bool:
                np.testing.assert_array_equal(left, right)
            else:
                np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True)
        elif isinstance(left, dict):
            assert left == right
        elif isinstance(left, list):
            assert left == right
        elif isinstance(left, float):
            assert left == pytest.approx(right, abs=1e-12, nan_ok=True)
        else:
            assert left == right


@pytest.mark.parametrize(
    "fit_fn,kwargs",
    [
        (fit_robust_global_event_reject, _robust_kwargs(40.0)),
        (fit_adaptive_event_gated_regression, _adaptive_kwargs(40.0)),
    ],
)
def test_dynamic_fitting_rejects_mismatched_and_all_nan_inputs(fit_fn, kwargs):
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_fn(signal_raw=np.ones(10), iso_raw=np.ones(9), **kwargs)

    with pytest.raises(RuntimeError, match="at least 3 finite samples"):
        fit_fn(signal_raw=np.full(10, np.nan), iso_raw=np.full(10, np.nan), **kwargs)


def test_adaptive_event_gated_sparse_trust_fails_clearly():
    n = 2000
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.2 * np.sin(2.0 * np.pi * 0.02 * t)
    signal = 1.6 * iso + 0.5 + 2.5 * np.exp(-0.5 * ((t - 25.0) / 3.0) ** 2)

    with pytest.raises(RuntimeError, match="trust_fraction_below_min"):
        fit_adaptive_event_gated_regression(
            signal_raw=signal,
            iso_raw=iso,
            sample_rate_hz=fs,
            residual_z_thresh=0.2,
            local_var_window_sec=8.0,
            local_var_ratio_thresh=1.2,
            smooth_window_sec=30.0,
            min_trust_fraction=0.99,
        )
