import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from photometry_pipeline.core.dynamic_fitting import (
    fit_adaptive_event_gated_regression as package_adaptive,
    fit_robust_global_event_reject as package_robust,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STANDALONE_PATH = REPO_ROOT / "examples" / "dynamic_fitting_standalone.py"


def _load_standalone_module():
    spec = importlib.util.spec_from_file_location(
        "student_dynamic_fitting_standalone",
        STANDALONE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "residual_z_thresh": 50.0,
        "local_var_window_sec": None,
        "local_var_ratio_thresh": None,
        "smooth_window_sec": 1.0,
        "min_trust_fraction": 0.1,
        "freeze_interp_method": "linear_hold",
        "use_intercept": True,
        "signal_excursion_polarity": "both",
    }


def _broad_opposite_deflection_trace() -> tuple[np.ndarray, np.ndarray, float]:
    fs = 40.0
    t = np.arange(2400, dtype=float) / fs
    uv = 3.0 + 0.25 * np.sin(2.0 * np.pi * 0.08 * t)
    signal = 1.2 * uv + 0.7
    event = (t >= 22.0) & (t <= 36.0)
    uv[event] -= 0.7 * np.sin(np.pi * (t[event] - 22.0) / 14.0)
    signal[event] += 1.3 * np.sin(np.pi * (t[event] - 22.0) / 14.0)
    return signal, uv, fs


def test_standalone_file_imports_by_path_without_package_relative_imports():
    module = _load_standalone_module()
    assert hasattr(module, "fit_robust_global_event_reject")
    assert hasattr(module, "fit_adaptive_event_gated_regression")

    text = STANDALONE_PATH.read_text(encoding="utf-8")
    assert "from .slope_qc" not in text
    assert "from photometry_pipeline" not in text
    assert "import photometry_pipeline" not in text


def test_standalone_robust_default_matches_package_and_nonnegative_clamps():
    standalone = _load_standalone_module()
    n = 1200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.3 * np.sin(2.0 * np.pi * 0.08 * t)
    signal = 7.5 - 1.4 * iso

    package = package_robust(signal_raw=signal, iso_raw=iso, **_robust_kwargs(fs))
    copied = standalone.fit_robust_global_event_reject(
        signal_raw=signal,
        iso_raw=iso,
        **_robust_kwargs(fs),
    )

    np.testing.assert_allclose(
        copied["iso_fit_signal_units"],
        package["iso_fit_signal_units"],
        rtol=0.0,
        atol=1e-12,
    )
    assert copied["final_coef"] == package["final_coef"]

    constrained = standalone.fit_robust_global_event_reject(
        signal_raw=signal,
        iso_raw=iso,
        slope_constraint="nonnegative",
        min_slope=0.0,
        **_robust_kwargs(fs),
    )
    assert constrained["final_coef"]["unconstrained_slope"] < 0.0
    assert constrained["final_coef"]["slope"] >= 0.0
    assert constrained["slope_constraint_summary"]["slope_constraint_applied"] is True
    np.testing.assert_allclose(
        constrained["iso_fit_signal_units"],
        constrained["final_coef"]["slope"] * iso + constrained["final_coef"]["intercept"],
        rtol=0.0,
        atol=1e-12,
    )


def test_standalone_adaptive_default_matches_package_and_nonnegative_clamps():
    standalone = _load_standalone_module()
    signal, uv, fs = _broad_opposite_deflection_trace()
    kwargs = {
        **_adaptive_kwargs(fs),
        "signal_fit_input": signal,
        "iso_fit_input": uv,
    }

    package = package_adaptive(signal_raw=signal, iso_raw=uv, **kwargs)
    copied = standalone.fit_adaptive_event_gated_regression(
        signal_raw=signal,
        iso_raw=uv,
        **kwargs,
    )

    np.testing.assert_allclose(
        copied["iso_fit_signal_units"],
        package["iso_fit_signal_units"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        copied["coef_slope"],
        package["coef_slope"],
        rtol=0.0,
        atol=1e-12,
    )

    constrained = standalone.fit_adaptive_event_gated_regression(
        signal_raw=signal,
        iso_raw=uv,
        slope_constraint="nonnegative",
        min_slope=0.0,
        **kwargs,
    )
    finite_slope = constrained["coef_slope"][np.isfinite(constrained["coef_slope"])]
    assert finite_slope.size > 0
    assert np.min(finite_slope) >= 0.0
    assert np.nanmin(constrained["coef_slope_unconstrained"]) < 0.0
    assert constrained["slope_constraint_summary"]["slope_constraint_applied"] is True
    assert constrained["slope_constraint_summary"]["slope_clamped_fraction"] > 0.0
    assert constrained["unconstrained_slope_summary"]["slope_negative_fraction"] > 0.0
    assert constrained["slope_summary"]["slope_negative_fraction"] == pytest.approx(0.0)


def test_standalone_nonnegative_rejects_negative_min_slope_but_unconstrained_allows_it():
    standalone = _load_standalone_module()
    n = 800
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 3.0 + 0.2 * np.sin(2.0 * np.pi * 0.11 * t)
    signal = 5.0 - 0.8 * iso

    with pytest.raises(ValueError, match="dynamic_fit_min_slope must be >= 0"):
        standalone.fit_robust_global_event_reject(
            signal_raw=signal,
            iso_raw=iso,
            slope_constraint="nonnegative",
            min_slope=-0.1,
            **_robust_kwargs(fs),
        )

    unconstrained = standalone.fit_robust_global_event_reject(
        signal_raw=signal,
        iso_raw=iso,
        slope_constraint="unconstrained",
        min_slope=-10.0,
        **_robust_kwargs(fs),
    )
    assert unconstrained["final_coef"]["slope"] < 0.0
    assert unconstrained["slope_constraint_summary"]["slope_constraint_applied"] is False


def test_standalone_slope_constraint_example_runs_successfully():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "standalone_dynamic_fit_slope_constraint.py")],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert "Constraint applied" in result.stdout
    assert "Clamped slope fraction" in result.stdout
