import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core import preprocessing, regression as regression_module
from photometry_pipeline.core.normalization import compute_dff
from photometry_pipeline.core.regression import (
    fit_adaptive_event_gated_regression,
    fit_chunk_dynamic,
    fit_robust_global_event_reject,
)
from photometry_pipeline.core.types import Chunk, SessionStats


def _make_chunk(uv_raw: np.ndarray, sig_raw: np.ndarray, fs_hz: float = 40.0) -> Chunk:
    n = int(uv_raw.shape[0])
    return Chunk(
        chunk_id=0,
        source_file="synth.csv",
        format="npm",
        time_sec=np.arange(n, dtype=float) / float(fs_hz),
        uv_raw=np.asarray(uv_raw, dtype=float).reshape(-1, 1),
        sig_raw=np.asarray(sig_raw, dtype=float).reshape(-1, 1),
        fs_hz=float(fs_hz),
        channel_names=["Region0"],
        metadata={},
    )


def _prepare_filtered(chunk: Chunk, cfg: Config) -> None:
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)


def test_dynamic_fit_mode_default_matches_explicit_rolling_filtered_to_raw():
    rng = np.random.default_rng(123)
    n = 3200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.5 * np.sin(2.0 * np.pi * 0.2 * t) + 0.03 * rng.standard_normal(n)
    sig = 1.4 * uv + 2.0 + 0.1 * np.sin(2.0 * np.pi * 0.8 * t + 0.5)

    cfg_default = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=3.5, filter_order=2)
    cfg_explicit = Config(
        window_sec=45.0,
        min_samples_per_window=20,
        lowpass_hz=3.5,
        filter_order=2,
        dynamic_fit_mode="rolling_filtered_to_raw",
    )

    c_default = _make_chunk(uv, sig, fs)
    c_explicit = _make_chunk(uv, sig, fs)
    _prepare_filtered(c_default, cfg_default)
    _prepare_filtered(c_explicit, cfg_explicit)

    uv_fit_default, delta_default = fit_chunk_dynamic(c_default, cfg_default, mode="phasic")
    uv_fit_explicit, delta_explicit = fit_chunk_dynamic(c_explicit, cfg_explicit, mode="phasic")

    np.testing.assert_allclose(uv_fit_default, uv_fit_explicit, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(delta_default, delta_explicit, rtol=0.0, atol=1e-12)
    assert c_default.metadata["dynamic_fit_mode_resolved"] == "rolling_filtered_to_raw"
    assert c_default.metadata["dynamic_fit_mode_alias_applied"] is True
    assert c_default.metadata["dynamic_fit_engine"] == "rolling_local_ols_v1"


def test_rolling_filtered_to_filtered_changes_reconstruction_domain_only():
    rng = np.random.default_rng(321)
    n = 2600
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 5.0 + 0.8 * np.sin(2.0 * np.pi * 0.22 * t) + 0.25 * rng.standard_normal(n)
    sig = 1.3 * uv + 0.7 + 0.1 * np.sin(2.0 * np.pi * 0.9 * t + 0.2)

    cfg_raw = Config(
        dynamic_fit_mode="rolling_filtered_to_raw",
        window_sec=50.0,
        min_samples_per_window=40,
        lowpass_hz=2.5,
        filter_order=2,
    )
    cfg_filt = Config(
        dynamic_fit_mode="rolling_filtered_to_filtered",
        window_sec=50.0,
        min_samples_per_window=40,
        lowpass_hz=2.5,
        filter_order=2,
    )

    c_raw = _make_chunk(uv, sig, fs)
    c_filt = _make_chunk(uv, sig, fs)
    _prepare_filtered(c_raw, cfg_raw)
    _prepare_filtered(c_filt, cfg_filt)

    uv_fit_raw, delta_raw = fit_chunk_dynamic(c_raw, cfg_raw, mode="phasic")
    uv_fit_filt, delta_filt = fit_chunk_dynamic(c_filt, cfg_filt, mode="phasic")

    assert uv_fit_raw.shape == uv_fit_filt.shape == c_raw.sig_raw.shape
    assert not np.allclose(uv_fit_raw, uv_fit_filt, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(delta_raw, c_raw.sig_raw - uv_fit_raw, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(delta_filt, c_filt.sig_raw - uv_fit_filt, rtol=0.0, atol=1e-12)

    stats = SessionStats(f0_values={"Region0": float(np.nanpercentile(c_raw.uv_raw[:, 0], 10.0))})
    c_filt.delta_f = delta_filt
    dff = compute_dff(c_filt, stats, cfg_filt)
    assert dff is not None
    assert dff.shape == c_filt.sig_raw.shape

    info = c_filt.metadata.get("dynamic_fit_engine_info", {})
    assert c_filt.metadata["dynamic_fit_mode_resolved"] == "rolling_filtered_to_filtered"
    assert info.get("reconstruction_signal") == "uv_filt"


def test_baseline_subtract_before_fit_toggle_changes_fit_inputs_but_not_dff_contract():
    n = 3000
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    slow = 0.8 * np.sin(2.0 * np.pi * 0.01 * t)
    uv = 4.0 + slow + 0.15 * np.sin(2.0 * np.pi * 0.25 * t)
    sig = 1.6 * uv + 0.4 + 0.2 * np.sin(2.0 * np.pi * 0.18 * t + 0.4)

    cfg_off = Config(
        dynamic_fit_mode="rolling_filtered_to_filtered",
        baseline_subtract_before_fit=False,
        window_sec=60.0,
        min_samples_per_window=30,
        lowpass_hz=2.0,
        filter_order=2,
    )
    cfg_on = Config(
        dynamic_fit_mode="rolling_filtered_to_filtered",
        baseline_subtract_before_fit=True,
        window_sec=60.0,
        min_samples_per_window=30,
        lowpass_hz=2.0,
        filter_order=2,
    )

    c_off = _make_chunk(uv, sig, fs)
    c_on = _make_chunk(uv, sig, fs)
    _prepare_filtered(c_off, cfg_off)
    _prepare_filtered(c_on, cfg_on)

    uv_fit_off, delta_off = fit_chunk_dynamic(c_off, cfg_off, mode="phasic")
    uv_fit_on, delta_on = fit_chunk_dynamic(c_on, cfg_on, mode="phasic")

    assert not np.allclose(uv_fit_off, uv_fit_on, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(delta_off, c_off.sig_raw - uv_fit_off, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(delta_on, c_on.sig_raw - uv_fit_on, rtol=0.0, atol=1e-12)

    f0 = float(np.nanpercentile(c_on.uv_raw[:, 0], 10.0))
    stats = SessionStats(f0_values={"Region0": f0})
    c_on.delta_f = delta_on
    dff_on = compute_dff(c_on, stats, cfg_on)
    np.testing.assert_allclose(dff_on[:, 0], 100.0 * delta_on[:, 0] / f0, rtol=0.0, atol=1e-12)
    assert c_on.metadata["baseline_subtract_before_fit_requested"] is True
    assert c_on.metadata["baseline_subtract_before_fit_applied"] is True


def test_rolling_filtered_to_filtered_reconstruction_is_domain_consistent_with_baseline_toggle():
    n = 2400
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 120.0 + 1.5 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 1.8 * uv + 25.0

    cfg_off = Config(
        dynamic_fit_mode="rolling_filtered_to_filtered",
        baseline_subtract_before_fit=False,
        window_sec=45.0,
        min_samples_per_window=30,
    )
    cfg_on = Config(
        dynamic_fit_mode="rolling_filtered_to_filtered",
        baseline_subtract_before_fit=True,
        window_sec=45.0,
        min_samples_per_window=30,
    )

    c_off = _make_chunk(uv, sig, fs)
    c_on = _make_chunk(uv, sig, fs)

    # Use identical filtered/raw traces to isolate reconstruction-domain math.
    c_off.uv_filt = c_off.uv_raw.copy()
    c_off.sig_filt = c_off.sig_raw.copy()
    c_on.uv_filt = c_on.uv_raw.copy()
    c_on.sig_filt = c_on.sig_raw.copy()

    uv_fit_off, delta_off = fit_chunk_dynamic(c_off, cfg_off, mode="phasic")
    uv_fit_on, delta_on = fit_chunk_dynamic(c_on, cfg_on, mode="phasic")

    err_off = float(np.nanmax(np.abs(uv_fit_off[:, 0] - sig)))
    err_on = float(np.nanmax(np.abs(uv_fit_on[:, 0] - sig)))
    assert err_off < 1e-9
    assert err_on < 1e-9

    np.testing.assert_allclose(delta_off[:, 0], c_off.sig_raw[:, 0] - uv_fit_off[:, 0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(delta_on[:, 0], c_on.sig_raw[:, 0] - uv_fit_on[:, 0], rtol=0.0, atol=1e-12)

    p_sig = float(np.nanpercentile(np.abs(sig), 99.0))
    p_fit = float(np.nanpercentile(np.abs(uv_fit_on[:, 0]), 99.0))
    scale_ratio = p_fit / max(p_sig, 1e-12)
    assert 0.9 <= scale_ratio <= 1.1

    info_on = c_on.metadata.get("dynamic_fit_engine_info", {})
    assert info_on.get("reconstruction_signal") == "uv_filt"
    assert info_on.get("reconstruction_domain_consistency") == "baseline_mapped"


def test_rolling_filtered_to_raw_baseline_subtract_reconstruction_uses_consistent_mapping():
    n = 2200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 90.0 + 1.2 * np.sin(2.0 * np.pi * 0.18 * t)
    sig = 1.5 * uv + 18.0

    cfg_off = Config(
        dynamic_fit_mode="rolling_filtered_to_raw",
        baseline_subtract_before_fit=False,
        window_sec=40.0,
        min_samples_per_window=30,
    )
    cfg_on = Config(
        dynamic_fit_mode="rolling_filtered_to_raw",
        baseline_subtract_before_fit=True,
        window_sec=40.0,
        min_samples_per_window=30,
    )

    c_off = _make_chunk(uv, sig, fs)
    c_on = _make_chunk(uv, sig, fs)
    c_off.uv_filt = c_off.uv_raw.copy()
    c_off.sig_filt = c_off.sig_raw.copy()
    c_on.uv_filt = c_on.uv_raw.copy()
    c_on.sig_filt = c_on.sig_raw.copy()

    uv_fit_off, delta_off = fit_chunk_dynamic(c_off, cfg_off, mode="phasic")
    uv_fit_on, delta_on = fit_chunk_dynamic(c_on, cfg_on, mode="phasic")

    err_off = float(np.nanmax(np.abs(uv_fit_off[:, 0] - sig)))
    err_on = float(np.nanmax(np.abs(uv_fit_on[:, 0] - sig)))
    assert err_off < 1e-9
    assert err_on < 1e-9

    np.testing.assert_allclose(delta_off[:, 0], c_off.sig_raw[:, 0] - uv_fit_off[:, 0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(delta_on[:, 0], c_on.sig_raw[:, 0] - uv_fit_on[:, 0], rtol=0.0, atol=1e-12)

    info_on = c_on.metadata.get("dynamic_fit_engine_info", {})
    assert info_on.get("reconstruction_signal") == "uv_raw"
    assert info_on.get("reconstruction_domain_consistency") == "baseline_mapped"


def test_global_linear_mode_recovers_expected_reference_and_preserves_dff_path():
    n = 600
    fs = 30.0
    t = np.arange(n, dtype=float) / fs
    uv = 6.0 + 0.25 * np.sin(2.0 * np.pi * 0.12 * t)
    slope = 2.25
    intercept = 1.75
    sig = slope * uv + intercept

    cfg = Config(dynamic_fit_mode="global_linear_regression")
    chunk = _make_chunk(uv, sig, fs)
    # Global mode fits filtered traces but reconstructs against raw UV.
    chunk.uv_filt = chunk.uv_raw.copy()
    chunk.sig_filt = chunk.sig_raw.copy()

    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")
    expected = (slope * chunk.uv_raw[:, 0]) + intercept

    np.testing.assert_allclose(uv_fit[:, 0], expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(delta_f[:, 0], chunk.sig_raw[:, 0] - uv_fit[:, 0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(delta_f[:, 0], 0.0, rtol=0.0, atol=1e-12)

    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f
    stats = SessionStats(f0_values={"Region0": float(np.nanpercentile(chunk.uv_raw[:, 0], 10.0))})
    dff = compute_dff(chunk, stats, cfg)
    assert dff is not None
    assert dff.shape == chunk.sig_raw.shape
    np.testing.assert_allclose(dff[:, 0], 0.0, rtol=0.0, atol=1e-12)
    assert chunk.metadata["dynamic_fit_mode_resolved"] == "global_linear_regression"
    assert chunk.metadata["dynamic_fit_engine"] == "global_linear_ols_v1"


def test_global_linear_mode_ignores_baseline_subtract_toggle_for_non_regression():
    n = 500
    fs = 30.0
    t = np.arange(n, dtype=float) / fs
    uv = 3.0 + 0.4 * np.sin(2.0 * np.pi * 0.2 * t)
    sig = 1.9 * uv + 0.6

    cfg_base = Config(dynamic_fit_mode="global_linear_regression", baseline_subtract_before_fit=False)
    cfg_toggle = Config(dynamic_fit_mode="global_linear_regression", baseline_subtract_before_fit=True)

    c0 = _make_chunk(uv, sig, fs)
    c1 = _make_chunk(uv, sig, fs)
    c0.uv_filt = c0.uv_raw.copy()
    c0.sig_filt = c0.sig_raw.copy()
    c1.uv_filt = c1.uv_raw.copy()
    c1.sig_filt = c1.sig_raw.copy()

    uv0, df0 = fit_chunk_dynamic(c0, cfg_base, mode="phasic")
    uv1, df1 = fit_chunk_dynamic(c1, cfg_toggle, mode="phasic")

    np.testing.assert_allclose(uv0, uv1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(df0, df1, rtol=0.0, atol=1e-12)
    assert c1.metadata["baseline_subtract_before_fit_requested"] is True
    assert c1.metadata["baseline_subtract_before_fit_applied"] is False


def test_global_linear_mode_degenerate_constant_uv_yields_nan_fit_and_dd2_warning():
    n = 200
    uv = np.ones(n, dtype=float) * 4.2
    sig = np.linspace(0.0, 1.0, n, dtype=float)
    cfg = Config(dynamic_fit_mode="global_linear_regression")
    chunk = _make_chunk(uv, sig, 20.0)
    chunk.uv_filt = chunk.uv_raw.copy()
    chunk.sig_filt = chunk.sig_raw.copy()

    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")
    assert np.isnan(uv_fit).all()
    assert np.isnan(delta_f).all()
    warns = chunk.metadata.get("qc_warnings", [])
    assert any("DEGENERATE[DD2]" in str(w) for w in warns)


def test_global_linear_mode_degenerate_nan_inputs_yield_nan_fit_and_dd1_warning():
    n = 160
    uv = np.linspace(1.0, 2.0, n, dtype=float)
    sig = np.linspace(3.0, 4.0, n, dtype=float)
    cfg = Config(dynamic_fit_mode="global_linear_regression")
    chunk = _make_chunk(uv, sig, 20.0)
    chunk.uv_filt = np.full_like(chunk.uv_raw, np.nan, dtype=float)
    chunk.sig_filt = np.full_like(chunk.sig_raw, np.nan, dtype=float)

    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")
    assert np.isnan(uv_fit).all()
    assert np.isnan(delta_f).all()
    warns = chunk.metadata.get("qc_warnings", [])
    assert any("DEGENERATE[DD1]" in str(w) for w in warns)


def test_bleach_correction_mode_default_none_matches_explicit_none():
    n = 1200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 3.0 + 0.4 * np.sin(2.0 * np.pi * 0.1 * t)
    sig = 1.8 * uv + 0.5 + 0.2 * np.sin(2.0 * np.pi * 0.22 * t + 0.3)

    cfg_default = Config(dynamic_fit_mode="global_linear_regression")
    cfg_none = Config(
        dynamic_fit_mode="global_linear_regression",
        bleach_correction_mode="none",
    )

    c0 = _make_chunk(uv, sig, fs)
    c1 = _make_chunk(uv, sig, fs)
    c0.uv_filt = c0.uv_raw.copy()
    c0.sig_filt = c0.sig_raw.copy()
    c1.uv_filt = c1.uv_raw.copy()
    c1.sig_filt = c1.sig_raw.copy()

    uv0, df0 = fit_chunk_dynamic(c0, cfg_default, mode="phasic")
    uv1, df1 = fit_chunk_dynamic(c1, cfg_none, mode="phasic")

    np.testing.assert_allclose(uv0, uv1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(df0, df1, rtol=0.0, atol=1e-12)
    assert c0.metadata.get("bleach_correction_mode_resolved") == "none"
    assert c0.metadata.get("bleach_correction_applied") is False


def test_bleach_single_exponential_reduces_large_exponential_drift_component():
    n = 3000
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    decay = 1.5 * np.exp(-t / 18.0)
    uv = 4.5 + decay + 0.08 * np.sin(2.0 * np.pi * 0.09 * t)
    sig = 1.7 * uv + 0.9 + 0.9 * np.exp(-t / 16.0) + 0.06 * np.sin(2.0 * np.pi * 0.24 * t + 0.4)

    cfg_none = Config(dynamic_fit_mode="global_linear_regression", bleach_correction_mode="none")
    cfg_bleach = Config(
        dynamic_fit_mode="global_linear_regression",
        bleach_correction_mode="single_exponential",
    )

    c_none = _make_chunk(uv, sig, fs)
    c_bleach = _make_chunk(uv, sig, fs)
    c_none.uv_filt = c_none.uv_raw.copy()
    c_none.sig_filt = c_none.sig_raw.copy()
    c_bleach.uv_filt = c_bleach.uv_raw.copy()
    c_bleach.sig_filt = c_bleach.sig_raw.copy()

    _, df_none = fit_chunk_dynamic(c_none, cfg_none, mode="phasic")
    _, df_bleach = fit_chunk_dynamic(c_bleach, cfg_bleach, mode="phasic")

    early_none = float(np.nanmean(df_none[:400, 0]))
    late_none = float(np.nanmean(df_none[-400:, 0]))
    early_bleach = float(np.nanmean(df_bleach[:400, 0]))
    late_bleach = float(np.nanmean(df_bleach[-400:, 0]))
    drift_none = abs(early_none - late_none)
    drift_bleach = abs(early_bleach - late_bleach)

    assert drift_bleach < drift_none
    assert c_bleach.metadata.get("bleach_correction_mode_resolved") == "single_exponential"
    assert c_bleach.metadata.get("bleach_correction_applied") is True
    roi_meta = c_bleach.metadata.get("bleach_correction", {}).get("Region0", {})
    assert isinstance(roi_meta, dict)
    assert roi_meta.get("signal", {}).get("fit_succeeded", False) is True
    assert roi_meta.get("isosbestic", {}).get("fit_succeeded", False) is True


def test_bleach_single_exponential_fits_signal_and_isosbestic_independently():
    n = 3200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    # Deliberately different decay constants to prove independent fits.
    uv = 5.0 + 1.8 * np.exp(-t / 45.0) + 0.03 * np.sin(2.0 * np.pi * 0.06 * t)
    sig = 2.0 + 2.1 * np.exp(-t / 8.0) + 0.02 * np.sin(2.0 * np.pi * 0.11 * t + 0.2)

    cfg_bleach = Config(
        dynamic_fit_mode="global_linear_regression",
        bleach_correction_mode="single_exponential",
    )
    chunk = _make_chunk(uv, sig, fs)
    chunk.uv_filt = chunk.uv_raw.copy()
    chunk.sig_filt = chunk.sig_raw.copy()

    fit_chunk_dynamic(chunk, cfg_bleach, mode="phasic")

    roi_meta = chunk.metadata.get("bleach_correction", {}).get("Region0", {})
    assert roi_meta.get("target") == "signal_and_isosbestic_independent"
    sig_meta = roi_meta.get("signal", {})
    iso_meta = roi_meta.get("isosbestic", {})
    assert sig_meta.get("fit_succeeded", False) is True
    assert iso_meta.get("fit_succeeded", False) is True
    sig_tau = float(sig_meta.get("tau_sec", np.nan))
    iso_tau = float(iso_meta.get("tau_sec", np.nan))
    assert np.isfinite(sig_tau) and np.isfinite(iso_tau)
    assert abs(sig_tau - iso_tau) > 5.0


def test_bleach_double_exponential_improves_two_timescale_fit_rmse_vs_single():
    n = 3600
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = (
        5.0
        + 1.6 * np.exp(-t / 5.0)
        + 0.9 * np.exp(-t / 65.0)
        + 0.02 * np.sin(2.0 * np.pi * 0.05 * t)
    )
    sig = (
        2.1
        + 2.2 * np.exp(-t / 4.5)
        + 1.4 * np.exp(-t / 52.0)
        + 0.03 * np.sin(2.0 * np.pi * 0.09 * t + 0.2)
    )

    cfg_single = Config(
        dynamic_fit_mode="global_linear_regression",
        bleach_correction_mode="single_exponential",
    )
    cfg_double = Config(
        dynamic_fit_mode="global_linear_regression",
        bleach_correction_mode="double_exponential",
    )

    c_single = _make_chunk(uv, sig, fs)
    c_double = _make_chunk(uv, sig, fs)
    c_single.uv_filt = c_single.uv_raw.copy()
    c_single.sig_filt = c_single.sig_raw.copy()
    c_double.uv_filt = c_double.uv_raw.copy()
    c_double.sig_filt = c_double.sig_raw.copy()

    fit_chunk_dynamic(c_single, cfg_single, mode="phasic")
    fit_chunk_dynamic(c_double, cfg_double, mode="phasic")

    single_meta = c_single.metadata.get("bleach_correction", {}).get("Region0", {}).get("signal", {})
    double_meta = c_double.metadata.get("bleach_correction", {}).get("Region0", {}).get("signal", {})
    assert single_meta.get("fit_succeeded", False) is True
    assert double_meta.get("fit_succeeded", False) is True
    rmse_single = float(single_meta.get("fit_rmse", np.nan))
    rmse_double = float(double_meta.get("fit_rmse", np.nan))
    assert np.isfinite(rmse_single) and np.isfinite(rmse_double)
    assert rmse_double < rmse_single
    assert c_double.metadata.get("bleach_correction_mode_resolved") == "double_exponential"


def test_bleach_double_exponential_fits_signal_and_isosbestic_independently():
    n = 3600
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = (
        4.0
        + 1.5 * np.exp(-t / 7.0)
        + 1.3 * np.exp(-t / 90.0)
        + 0.02 * np.sin(2.0 * np.pi * 0.05 * t)
    )
    sig = (
        3.0
        + 1.9 * np.exp(-t / 3.5)
        + 1.0 * np.exp(-t / 40.0)
        + 0.03 * np.sin(2.0 * np.pi * 0.08 * t + 0.1)
    )

    cfg_double = Config(
        dynamic_fit_mode="global_linear_regression",
        bleach_correction_mode="double_exponential",
    )
    chunk = _make_chunk(uv, sig, fs)
    chunk.uv_filt = chunk.uv_raw.copy()
    chunk.sig_filt = chunk.sig_raw.copy()

    fit_chunk_dynamic(chunk, cfg_double, mode="phasic")

    roi_meta = chunk.metadata.get("bleach_correction", {}).get("Region0", {})
    assert roi_meta.get("target") == "signal_and_isosbestic_independent"
    sig_meta = roi_meta.get("signal", {})
    iso_meta = roi_meta.get("isosbestic", {})
    assert sig_meta.get("fit_model") == "double_exponential"
    assert iso_meta.get("fit_model") == "double_exponential"
    assert sig_meta.get("fit_succeeded", False) is True
    assert iso_meta.get("fit_succeeded", False) is True

    sig_tau_fast = float(sig_meta.get("tau_fast_sec", np.nan))
    sig_tau_slow = float(sig_meta.get("tau_slow_sec", np.nan))
    iso_tau_fast = float(iso_meta.get("tau_fast_sec", np.nan))
    iso_tau_slow = float(iso_meta.get("tau_slow_sec", np.nan))
    assert np.isfinite(sig_tau_fast) and np.isfinite(sig_tau_slow)
    assert np.isfinite(iso_tau_fast) and np.isfinite(iso_tau_slow)
    assert sig_tau_fast < sig_tau_slow
    assert iso_tau_fast < iso_tau_slow
    assert abs(sig_tau_fast - iso_tau_fast) > 1.0
    assert abs(sig_tau_slow - iso_tau_slow) > 5.0


def test_bleach_single_exponential_fit_failure_falls_back_honestly(monkeypatch):
    n = 1600
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.3 * np.sin(2.0 * np.pi * 0.1 * t)
    sig = 1.5 * uv + 0.8 + 0.1 * np.sin(2.0 * np.pi * 0.22 * t)

    cfg_none = Config(dynamic_fit_mode="global_linear_regression", bleach_correction_mode="none")
    cfg_bleach = Config(
        dynamic_fit_mode="global_linear_regression",
        bleach_correction_mode="single_exponential",
    )

    c_none = _make_chunk(uv, sig, fs)
    c_fail = _make_chunk(uv, sig, fs)
    c_none.uv_filt = c_none.uv_raw.copy()
    c_none.sig_filt = c_none.sig_raw.copy()
    c_fail.uv_filt = c_fail.uv_raw.copy()
    c_fail.sig_filt = c_fail.sig_raw.copy()

    def _forced_fail(_trace, _sample_rate_hz):
        return {
            "fit_succeeded": False,
            "fit_failure_reason": "forced_failure",
            "n_finite_samples": 0,
            "amplitude": float("nan"),
            "tau_sec": float("nan"),
            "offset": float("nan"),
            "fit_rmse": float("nan"),
        }

    monkeypatch.setattr(regression_module, "_fit_single_exponential_with_offset", _forced_fail)
    uv_none, df_none = fit_chunk_dynamic(c_none, cfg_none, mode="phasic")
    uv_fail, df_fail = fit_chunk_dynamic(c_fail, cfg_bleach, mode="phasic")

    np.testing.assert_allclose(uv_none, uv_fail, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(df_none, df_fail, rtol=0.0, atol=1e-12)
    assert c_fail.metadata.get("bleach_correction_mode_resolved") == "single_exponential"
    assert c_fail.metadata.get("bleach_correction_applied") is False
    roi_meta = c_fail.metadata.get("bleach_correction", {}).get("Region0", {})
    assert roi_meta.get("signal", {}).get("fit_failure_reason") == "forced_failure"
    assert roi_meta.get("isosbestic", {}).get("fit_failure_reason") == "forced_failure"


def test_bleach_double_exponential_fit_failure_falls_back_honestly(monkeypatch):
    n = 1600
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.3 * np.sin(2.0 * np.pi * 0.1 * t)
    sig = 1.5 * uv + 0.8 + 0.1 * np.sin(2.0 * np.pi * 0.22 * t)

    cfg_none = Config(dynamic_fit_mode="global_linear_regression", bleach_correction_mode="none")
    cfg_double = Config(
        dynamic_fit_mode="global_linear_regression",
        bleach_correction_mode="double_exponential",
    )

    c_none = _make_chunk(uv, sig, fs)
    c_fail = _make_chunk(uv, sig, fs)
    c_none.uv_filt = c_none.uv_raw.copy()
    c_none.sig_filt = c_none.sig_raw.copy()
    c_fail.uv_filt = c_fail.uv_raw.copy()
    c_fail.sig_filt = c_fail.sig_raw.copy()

    def _forced_fail(_trace, _sample_rate_hz):
        return {
            "fit_model": "double_exponential",
            "fit_succeeded": False,
            "fit_failure_reason": "forced_double_failure",
            "n_finite_samples": 0,
            "amplitude_fast": float("nan"),
            "tau_fast_sec": float("nan"),
            "amplitude_slow": float("nan"),
            "tau_slow_sec": float("nan"),
            "offset": float("nan"),
            "fit_rmse": float("nan"),
        }

    monkeypatch.setattr(regression_module, "_fit_double_exponential_with_offset", _forced_fail)
    uv_none, df_none = fit_chunk_dynamic(c_none, cfg_none, mode="phasic")
    uv_fail, df_fail = fit_chunk_dynamic(c_fail, cfg_double, mode="phasic")

    np.testing.assert_allclose(uv_none, uv_fail, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(df_none, df_fail, rtol=0.0, atol=1e-12)
    assert c_fail.metadata.get("bleach_correction_mode_resolved") == "double_exponential"
    assert c_fail.metadata.get("bleach_correction_applied") is False
    roi_meta = c_fail.metadata.get("bleach_correction", {}).get("Region0", {})
    assert roi_meta.get("signal", {}).get("fit_failure_reason") == "forced_double_failure"
    assert roi_meta.get("isosbestic", {}).get("fit_failure_reason") == "forced_double_failure"


def test_robust_global_event_reject_excludes_large_event_and_preserves_background_fit():
    n = 3200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(42)

    uv = 4.0 + 0.25 * np.sin(2.0 * np.pi * 0.03 * t) + 0.05 * rng.standard_normal(n)
    background = 1.7 * uv + 0.8
    event = 2.4 * np.exp(-0.5 * ((t - 35.0) / 2.0) ** 2)
    sig = background + event + 0.04 * rng.standard_normal(n)

    cfg_roll = Config(
        dynamic_fit_mode="rolling_filtered_to_raw",
        window_sec=30.0,
        min_samples_per_window=60,
        lowpass_hz=2.0,
        filter_order=2,
    )
    cfg_robust = Config(
        dynamic_fit_mode="robust_global_event_reject",
        lowpass_hz=2.0,
        filter_order=2,
        robust_event_reject_max_iters=4,
        robust_event_reject_residual_z_thresh=3.0,
        robust_event_reject_local_var_window_sec=8.0,
        robust_event_reject_local_var_ratio_thresh=4.0,
        robust_event_reject_min_keep_fraction=0.55,
    )

    c_roll = _make_chunk(uv, sig, fs)
    c_robust = _make_chunk(uv, sig, fs)
    _prepare_filtered(c_roll, cfg_roll)
    _prepare_filtered(c_robust, cfg_robust)

    uv_fit_roll, _ = fit_chunk_dynamic(c_roll, cfg_roll, mode="phasic")
    uv_fit_robust, delta_robust = fit_chunk_dynamic(c_robust, cfg_robust, mode="phasic")

    np.testing.assert_allclose(
        delta_robust[:, 0],
        c_robust.sig_raw[:, 0] - uv_fit_robust[:, 0],
        rtol=0.0,
        atol=1e-12,
    )
    assert c_robust.metadata["dynamic_fit_mode_resolved"] == "robust_global_event_reject"
    assert c_robust.metadata["dynamic_fit_engine"] == "robust_global_event_reject_v1"

    event_mask = np.abs(t - 35.0) <= 2.0
    event_mae_roll = float(np.nanmean(np.abs(uv_fit_roll[event_mask, 0] - background[event_mask])))
    event_mae_robust = float(np.nanmean(np.abs(uv_fit_robust[event_mask, 0] - background[event_mask])))
    assert event_mae_robust < event_mae_roll

    roi_meta = c_robust.metadata.get("dynamic_fit_event_reject", {}).get("Region0", {})
    excluded = np.asarray(roi_meta.get("excluded_mask", []), dtype=bool)
    assert excluded.shape == (n,)
    assert float(np.mean(excluded[event_mask])) > 0.2
    assert roi_meta.get("n_iterations_completed", 0) >= 1
    assert np.isfinite(float(roi_meta.get("final_keep_fraction", np.nan)))
    assert str(roi_meta.get("robust_fit_backend_used", "")) != ""
    engine_info = c_robust.metadata.get("dynamic_fit_engine_info", {})
    assert "robust_backend_used_counts" in engine_info
    assert int(engine_info.get("success_roi_count", 0)) == 1


def test_fit_robust_global_event_reject_reports_backend_and_mad_zero_guard():
    n = 1200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.2 * np.sin(2.0 * np.pi * 0.05 * t)
    sig = 1.7 * iso + 0.8

    result = fit_robust_global_event_reject(
        signal_raw=sig,
        iso_raw=iso,
        max_iters=3,
        residual_z_thresh=3.5,
        local_var_window_sec=10.0,
        local_var_ratio_thresh=None,
        min_keep_fraction=0.5,
        sample_rate_hz=fs,
        use_intercept=True,
    )

    np.testing.assert_allclose(result["iso_fit_signal_units"], sig, rtol=0.0, atol=1e-8)
    assert result["n_iterations_completed"] >= 1
    assert result["stop_reason"] in {"mad_zero_or_nonfinite", "converged_keep_mask"}
    assert result["final_keep_fraction"] == pytest.approx(1.0)
    assert result["robust_fit_backend_used"] in {"sklearn_huber", "statsmodels_rlm", "unknown"}


def test_fit_robust_global_event_reject_default_matches_explicit_positive_polarity():
    n = 1400
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.2 * np.sin(2.0 * np.pi * 0.05 * t)
    sig = 1.7 * iso + 0.8 + 2.2 * np.exp(-0.5 * ((t - 15.0) / 2.0) ** 2)

    kwargs = dict(
        signal_raw=sig,
        iso_raw=iso,
        max_iters=3,
        residual_z_thresh=3.5,
        local_var_window_sec=None,
        local_var_ratio_thresh=None,
        min_keep_fraction=0.5,
        sample_rate_hz=fs,
        use_intercept=True,
    )
    result_default = fit_robust_global_event_reject(**kwargs)
    result_positive = fit_robust_global_event_reject(
        **kwargs,
        signal_excursion_polarity="positive",
    )

    np.testing.assert_allclose(
        np.asarray(result_default["iso_fit_signal_units"], dtype=float),
        np.asarray(result_positive["iso_fit_signal_units"], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        np.asarray(result_default["excluded_mask"], dtype=bool),
        np.asarray(result_positive["excluded_mask"], dtype=bool),
    )
    assert result_default["signal_excursion_polarity_applied"] == "positive"
    assert result_positive["signal_excursion_polarity_applied"] == "positive"


def test_fit_robust_global_event_reject_negative_and_both_protect_negative_excursions():
    n = 3200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.2 + 0.25 * np.sin(2.0 * np.pi * 0.03 * t)
    background = 1.6 * iso + 0.7
    neg_event = -2.8 * np.exp(-0.5 * ((t - 35.0) / 2.0) ** 2)
    sig = background + neg_event

    kwargs = dict(
        signal_raw=sig,
        iso_raw=iso,
        max_iters=4,
        residual_z_thresh=3.0,
        local_var_window_sec=None,
        local_var_ratio_thresh=None,
        min_keep_fraction=0.5,
        sample_rate_hz=fs,
        use_intercept=True,
    )
    result_positive = fit_robust_global_event_reject(**kwargs, signal_excursion_polarity="positive")
    result_negative = fit_robust_global_event_reject(**kwargs, signal_excursion_polarity="negative")
    result_both = fit_robust_global_event_reject(**kwargs, signal_excursion_polarity="both")

    event_mask = np.abs(t - 35.0) <= 2.0
    excluded_pos = float(np.mean(np.asarray(result_positive["excluded_mask"], dtype=bool)[event_mask]))
    excluded_neg = float(np.mean(np.asarray(result_negative["excluded_mask"], dtype=bool)[event_mask]))
    excluded_both = float(np.mean(np.asarray(result_both["excluded_mask"], dtype=bool)[event_mask]))
    assert excluded_pos < 0.05
    assert excluded_neg > 0.2
    assert excluded_both > 0.2

    fit_pos = np.asarray(result_positive["iso_fit_signal_units"], dtype=float)
    fit_neg = np.asarray(result_negative["iso_fit_signal_units"], dtype=float)
    fit_both = np.asarray(result_both["iso_fit_signal_units"], dtype=float)
    mae_pos = float(np.mean(np.abs(fit_pos[event_mask] - background[event_mask])))
    mae_neg = float(np.mean(np.abs(fit_neg[event_mask] - background[event_mask])))
    mae_both = float(np.mean(np.abs(fit_both[event_mask] - background[event_mask])))
    assert mae_neg < mae_pos
    assert mae_both < mae_pos

    assert result_negative["signal_excursion_polarity_applied"] == "negative"
    assert result_both["signal_excursion_polarity_applied"] == "both"


def test_fit_robust_global_event_reject_both_polarity_covers_upper_and_lower_tails():
    n = 3200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.25 * np.sin(2.0 * np.pi * 0.03 * t)
    background = 1.65 * iso + 0.75
    pos_event = 2.6 * np.exp(-0.5 * ((t - 24.0) / 1.8) ** 2)
    neg_event = -2.6 * np.exp(-0.5 * ((t - 50.0) / 1.8) ** 2)
    sig = background + pos_event + neg_event

    kwargs = dict(
        signal_raw=sig,
        iso_raw=iso,
        max_iters=4,
        residual_z_thresh=2.8,
        local_var_window_sec=None,
        local_var_ratio_thresh=None,
        min_keep_fraction=0.5,
        sample_rate_hz=fs,
        use_intercept=True,
    )
    excluded_pos = np.asarray(
        fit_robust_global_event_reject(**kwargs, signal_excursion_polarity="positive")["excluded_mask"],
        dtype=bool,
    )
    excluded_neg = np.asarray(
        fit_robust_global_event_reject(**kwargs, signal_excursion_polarity="negative")["excluded_mask"],
        dtype=bool,
    )
    excluded_both = np.asarray(
        fit_robust_global_event_reject(**kwargs, signal_excursion_polarity="both")["excluded_mask"],
        dtype=bool,
    )

    pos_mask = np.abs(t - 24.0) <= 1.5
    neg_mask = np.abs(t - 50.0) <= 1.5
    assert float(np.mean(excluded_pos[pos_mask])) > 0.2
    assert float(np.mean(excluded_pos[neg_mask])) < 0.05
    assert float(np.mean(excluded_neg[pos_mask])) < 0.05
    assert float(np.mean(excluded_neg[neg_mask])) > 0.2
    assert float(np.mean(excluded_both[pos_mask])) > 0.2
    assert float(np.mean(excluded_both[neg_mask])) > 0.2


def test_fit_robust_global_event_reject_min_keep_guard_stops_exclusion():
    n = 2000
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 5.0 + 0.2 * np.sin(2.0 * np.pi * 0.04 * t)
    sig = 1.6 * iso + 0.4
    sig += 2.8 * np.exp(-0.5 * ((t - 24.0) / 2.5) ** 2)

    result = fit_robust_global_event_reject(
        signal_raw=sig,
        iso_raw=iso,
        max_iters=3,
        residual_z_thresh=0.15,
        local_var_window_sec=10.0,
        local_var_ratio_thresh=None,
        min_keep_fraction=0.95,
        sample_rate_hz=fs,
        use_intercept=True,
    )

    reasons = [str(x.get("stop_reason", "")) for x in result.get("iteration_summaries", [])]
    assert "min_keep_fraction_guard" in reasons
    assert result["final_keep_fraction"] >= 0.95
    # Guard preserves previous mask; exclusions should not be applied in final output.
    assert int(np.sum(result["excluded_mask"])) == 0


def test_robust_global_event_reject_falls_back_to_global_linear_when_robust_fit_fails(monkeypatch):
    n = 800
    fs = 20.0
    t = np.arange(n, dtype=float) / fs
    uv = 2.5 + 0.3 * np.sin(2.0 * np.pi * 0.12 * t)
    sig = 1.9 * uv + 0.4

    cfg = Config(dynamic_fit_mode="robust_global_event_reject")
    chunk = _make_chunk(uv, sig, fs)
    chunk.uv_filt = chunk.uv_raw.copy()
    chunk.sig_filt = chunk.sig_raw.copy()

    def _forced_fail(*_args, **_kwargs):
        return None, "forced_failure"

    monkeypatch.setattr(regression_module, "_fit_robust_linear", _forced_fail)
    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")

    np.testing.assert_allclose(delta_f[:, 0], chunk.sig_raw[:, 0] - uv_fit[:, 0], rtol=0.0, atol=1e-12)
    warns = chunk.metadata.get("qc_warnings", [])
    assert any("ROBUST_GLOBAL_EVENT_REJECT_FALLBACK" in str(w) for w in warns)
    roi_meta = chunk.metadata.get("dynamic_fit_event_reject", {}).get("Region0", {})
    assert roi_meta.get("fallback_to_global_linear") is True
    assert roi_meta.get("robust_fit_backend_used") == "global_linear_fallback"


def test_robust_global_event_reject_constant_iso_degenerate_fallback_is_safe():
    n = 300
    fs = 20.0
    t = np.arange(n, dtype=float) / fs
    uv = np.ones(n, dtype=float) * 4.2
    sig = 0.8 + 0.3 * np.sin(2.0 * np.pi * 0.2 * t)

    cfg = Config(dynamic_fit_mode="robust_global_event_reject")
    chunk = _make_chunk(uv, sig, fs)
    chunk.uv_filt = chunk.uv_raw.copy()
    chunk.sig_filt = chunk.sig_raw.copy()

    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")
    assert np.isnan(uv_fit).all()
    assert np.isnan(delta_f).all()

    warns = chunk.metadata.get("qc_warnings", [])
    assert any("ROBUST_GLOBAL_EVENT_REJECT_FALLBACK" in str(w) for w in warns)
    assert any("ROBUST_GLOBAL_EVENT_REJECT_FALLBACK_DEGENERATE[DD2]" in str(w) for w in warns)
    roi_meta = chunk.metadata.get("dynamic_fit_event_reject", {}).get("Region0", {})
    assert roi_meta.get("fallback_to_global_linear") is True
    assert roi_meta.get("fallback_failed") is True


def test_adaptive_event_gated_regression_synthetic_event_and_drift_behavior():
    n = 3600
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    rng = np.random.default_rng(7)

    uv = 4.2 + 0.35 * np.sin(2.0 * np.pi * 0.015 * t) + 0.07 * np.sin(2.0 * np.pi * 0.07 * t)
    slope_t = 1.45 + 0.20 * np.sin(2.0 * np.pi * 0.004 * t)
    background = slope_t * uv + 0.7
    event = 2.8 * np.exp(-0.5 * ((t - 45.0) / 2.5) ** 2)
    sig = background + event + 0.03 * rng.standard_normal(n)

    cfg_roll = Config(
        dynamic_fit_mode="rolling_filtered_to_raw",
        window_sec=20.0,
        min_samples_per_window=60,
        lowpass_hz=2.0,
        filter_order=2,
    )
    cfg_global = Config(
        dynamic_fit_mode="global_linear_regression",
        lowpass_hz=2.0,
        filter_order=2,
    )
    cfg_adapt = Config(
        dynamic_fit_mode="adaptive_event_gated_regression",
        lowpass_hz=2.0,
        filter_order=2,
        adaptive_event_gate_residual_z_thresh=3.0,
        adaptive_event_gate_local_var_window_sec=8.0,
        adaptive_event_gate_local_var_ratio_thresh=4.0,
        adaptive_event_gate_smooth_window_sec=60.0,
        adaptive_event_gate_min_trust_fraction=0.50,
    )

    c_roll = _make_chunk(uv, sig, fs)
    c_global = _make_chunk(uv, sig, fs)
    c_adapt = _make_chunk(uv, sig, fs)
    _prepare_filtered(c_roll, cfg_roll)
    _prepare_filtered(c_global, cfg_global)
    _prepare_filtered(c_adapt, cfg_adapt)

    uv_roll, _ = fit_chunk_dynamic(c_roll, cfg_roll, mode="phasic")
    uv_global, _ = fit_chunk_dynamic(c_global, cfg_global, mode="phasic")
    uv_adapt, df_adapt = fit_chunk_dynamic(c_adapt, cfg_adapt, mode="phasic")

    np.testing.assert_allclose(df_adapt[:, 0], c_adapt.sig_raw[:, 0] - uv_adapt[:, 0], rtol=0.0, atol=1e-12)
    assert c_adapt.metadata["dynamic_fit_mode_resolved"] == "adaptive_event_gated_regression"
    assert c_adapt.metadata["dynamic_fit_engine"] == "adaptive_event_gated_regression_v1"

    event_mask = np.abs(t - 45.0) <= 3.0
    quiet_mask = ~event_mask
    event_mae_roll = float(np.nanmean(np.abs(uv_roll[event_mask, 0] - background[event_mask])))
    event_mae_adapt = float(np.nanmean(np.abs(uv_adapt[event_mask, 0] - background[event_mask])))
    quiet_mae_global = float(np.nanmean(np.abs(uv_global[quiet_mask, 0] - background[quiet_mask])))
    quiet_mae_adapt = float(np.nanmean(np.abs(uv_adapt[quiet_mask, 0] - background[quiet_mask])))
    assert event_mae_adapt < event_mae_roll
    assert quiet_mae_adapt < quiet_mae_global

    roi_meta = c_adapt.metadata.get("dynamic_fit_adaptive_event_gated", {}).get("Region0", {})
    gated = np.asarray(roi_meta.get("gated_mask", []), dtype=bool)
    assert gated.shape == (n,)
    assert float(np.mean(gated[event_mask])) > 0.2
    assert float(roi_meta.get("trust_fraction", 0.0)) > 0.5
    assert roi_meta.get("fallback_mode") == "none"


def test_adaptive_event_gated_regression_freezes_coefficients_through_gated_event_span():
    n = 3200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 3.8 + 0.3 * np.sin(2.0 * np.pi * 0.02 * t)
    background = 1.55 * uv + 0.6
    event = 3.0 * np.exp(-0.5 * ((t - 36.0) / 2.0) ** 2)
    sig = background + event

    cfg = Config(
        dynamic_fit_mode="adaptive_event_gated_regression",
        lowpass_hz=2.0,
        filter_order=2,
        adaptive_event_gate_residual_z_thresh=2.5,
        adaptive_event_gate_local_var_window_sec=8.0,
        adaptive_event_gate_local_var_ratio_thresh=3.5,
        adaptive_event_gate_smooth_window_sec=50.0,
        adaptive_event_gate_min_trust_fraction=0.5,
    )
    chunk = _make_chunk(uv, sig, fs)
    _prepare_filtered(chunk, cfg)
    uv_fit, _ = fit_chunk_dynamic(chunk, cfg, mode="phasic")
    assert uv_fit is not None

    roi_meta = chunk.metadata.get("dynamic_fit_adaptive_event_gated", {}).get("Region0", {})
    gated = np.asarray(roi_meta.get("gated_mask", []), dtype=bool)
    slope = np.asarray(roi_meta.get("coef_slope", []), dtype=float)
    assert gated.shape == slope.shape == (n,)
    assert roi_meta.get("freeze_interp_method") == "linear_hold"

    span = np.abs(t - 36.0) <= 1.6
    span_gated = span & gated
    assert int(np.sum(span_gated)) >= 20
    slope_span = slope[span_gated]
    assert np.nanmax(slope_span) - np.nanmin(slope_span) < 1e-6


def test_adaptive_event_gated_regression_sparse_trust_falls_back_safely():
    n = 2500
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    uv = 4.0 + 0.2 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 1.6 * uv + 0.5 + 2.4 * np.exp(-0.5 * ((t - 30.0) / 3.0) ** 2)

    cfg = Config(
        dynamic_fit_mode="adaptive_event_gated_regression",
        adaptive_event_gate_residual_z_thresh=0.2,
        adaptive_event_gate_local_var_window_sec=8.0,
        adaptive_event_gate_local_var_ratio_thresh=1.2,
        adaptive_event_gate_smooth_window_sec=60.0,
        adaptive_event_gate_min_trust_fraction=0.95,
    )
    chunk = _make_chunk(uv, sig, fs)
    _prepare_filtered(chunk, cfg)

    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")
    assert uv_fit is not None
    assert delta_f is not None
    np.testing.assert_allclose(delta_f[:, 0], chunk.sig_raw[:, 0] - uv_fit[:, 0], rtol=0.0, atol=1e-12)

    warns = chunk.metadata.get("qc_warnings", [])
    assert any("ADAPTIVE_EVENT_GATED_REGRESSION_FALLBACK" in str(w) for w in warns)
    roi_meta = chunk.metadata.get("dynamic_fit_adaptive_event_gated", {}).get("Region0", {})
    assert roi_meta.get("fallback_mode") in {
        "robust_global_event_reject",
        "global_linear_regression",
        "global_linear_regression_failed",
    }


def test_fit_adaptive_event_gated_regression_direct_helper_handles_zero_mad_without_crash():
    n = 1000
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.3 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 1.7 * iso + 0.9

    result = fit_adaptive_event_gated_regression(
        signal_raw=sig,
        iso_raw=iso,
        signal_fit_input=sig,
        iso_fit_input=iso,
        sample_rate_hz=fs,
        residual_z_thresh=3.5,
        local_var_window_sec=10.0,
        local_var_ratio_thresh=None,
        smooth_window_sec=60.0,
        min_trust_fraction=0.5,
        freeze_interp_method="linear_hold",
        use_intercept=True,
    )
    fit = np.asarray(result["iso_fit_signal_units"], dtype=float)
    np.testing.assert_allclose(fit, sig, rtol=0.0, atol=1e-8)
    assert result["n_trusted"] == result["n_finite"]
    assert result["gated_fraction"] == pytest.approx(0.0)


def test_fit_adaptive_event_gated_regression_default_matches_explicit_positive_polarity():
    n = 1400
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.0 + 0.2 * np.sin(2.0 * np.pi * 0.02 * t)
    sig = 1.7 * iso + 0.8 + 2.0 * np.exp(-0.5 * ((t - 15.0) / 2.0) ** 2)

    kwargs = dict(
        signal_raw=sig,
        iso_raw=iso,
        signal_fit_input=sig,
        iso_fit_input=iso,
        sample_rate_hz=fs,
        residual_z_thresh=3.0,
        local_var_window_sec=None,
        local_var_ratio_thresh=None,
        smooth_window_sec=60.0,
        min_trust_fraction=0.5,
        freeze_interp_method="linear_hold",
        use_intercept=True,
    )
    result_default = fit_adaptive_event_gated_regression(**kwargs)
    result_positive = fit_adaptive_event_gated_regression(
        **kwargs,
        signal_excursion_polarity="positive",
    )

    np.testing.assert_allclose(
        np.asarray(result_default["iso_fit_signal_units"], dtype=float),
        np.asarray(result_positive["iso_fit_signal_units"], dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        np.asarray(result_default["gated_mask"], dtype=bool),
        np.asarray(result_positive["gated_mask"], dtype=bool),
    )
    assert result_default["signal_excursion_polarity_applied"] == "positive"
    assert result_positive["signal_excursion_polarity_applied"] == "positive"


def test_fit_adaptive_event_gated_regression_negative_and_both_gate_negative_excursions():
    n = 3200
    fs = 40.0
    t = np.arange(n, dtype=float) / fs
    iso = 4.1 + 0.25 * np.sin(2.0 * np.pi * 0.02 * t)
    background = 1.6 * iso + 0.6
    neg_event = -2.9 * np.exp(-0.5 * ((t - 36.0) / 2.0) ** 2)
    sig = background + neg_event

    kwargs = dict(
        signal_raw=sig,
        iso_raw=iso,
        signal_fit_input=sig,
        iso_fit_input=iso,
        sample_rate_hz=fs,
        residual_z_thresh=3.0,
        local_var_window_sec=None,
        local_var_ratio_thresh=None,
        smooth_window_sec=50.0,
        min_trust_fraction=0.5,
        freeze_interp_method="linear_hold",
        use_intercept=True,
    )
    result_positive = fit_adaptive_event_gated_regression(**kwargs, signal_excursion_polarity="positive")
    result_negative = fit_adaptive_event_gated_regression(**kwargs, signal_excursion_polarity="negative")
    result_both = fit_adaptive_event_gated_regression(**kwargs, signal_excursion_polarity="both")

    event_mask = np.abs(t - 36.0) <= 1.8
    gated_pos = float(np.mean(np.asarray(result_positive["gated_mask"], dtype=bool)[event_mask]))
    gated_neg = float(np.mean(np.asarray(result_negative["gated_mask"], dtype=bool)[event_mask]))
    gated_both = float(np.mean(np.asarray(result_both["gated_mask"], dtype=bool)[event_mask]))
    assert gated_pos < 0.05
    assert gated_neg > 0.2
    assert gated_both > 0.2

    assert int(result_negative["n_gated_residual_lower_tail"]) > 0
    assert int(result_both["n_gated_residual_lower_tail"]) > 0
    assert result_negative["signal_excursion_polarity_applied"] == "negative"
    assert result_both["signal_excursion_polarity_applied"] == "both"


def test_config_rejects_invalid_dynamic_fit_mode(tmp_path):
    cfg_path = tmp_path / "invalid_dynamic_fit_mode.yaml"
    cfg_path.write_text("dynamic_fit_mode: not_a_mode\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid dynamic_fit_mode"):
        Config.from_yaml(str(cfg_path))


def test_config_accepts_new_rolling_modes_and_baseline_toggle(tmp_path):
    cfg_path = tmp_path / "new_dynamic_modes.yaml"
    cfg_path.write_text(
        "dynamic_fit_mode: rolling_filtered_to_filtered\nbaseline_subtract_before_fit: true\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(str(cfg_path))
    assert cfg.dynamic_fit_mode == "rolling_filtered_to_filtered"
    assert cfg.baseline_subtract_before_fit is True


def test_config_accepts_single_exponential_bleach_correction_mode(tmp_path):
    cfg_path = tmp_path / "bleach_mode.yaml"
    cfg_path.write_text("bleach_correction_mode: single_exponential\n", encoding="utf-8")
    cfg = Config.from_yaml(str(cfg_path))
    assert cfg.bleach_correction_mode == "single_exponential"


def test_config_accepts_double_exponential_bleach_correction_mode(tmp_path):
    cfg_path = tmp_path / "bleach_mode_double.yaml"
    cfg_path.write_text("bleach_correction_mode: double_exponential\n", encoding="utf-8")
    cfg = Config.from_yaml(str(cfg_path))
    assert cfg.bleach_correction_mode == "double_exponential"


def test_config_rejects_invalid_bleach_correction_mode(tmp_path):
    cfg_path = tmp_path / "invalid_bleach_mode.yaml"
    cfg_path.write_text("bleach_correction_mode: unsupported_mode\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid bleach_correction_mode"):
        Config.from_yaml(str(cfg_path))


def test_config_accepts_robust_global_event_reject_mode_and_params(tmp_path):
    cfg_path = tmp_path / "robust_mode.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "dynamic_fit_mode: robust_global_event_reject",
                "robust_event_reject_max_iters: 5",
                "robust_event_reject_residual_z_thresh: 3.2",
                "robust_event_reject_local_var_window_sec: 9.0",
                "robust_event_reject_local_var_ratio_thresh: 4.5",
                "robust_event_reject_min_keep_fraction: 0.6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(str(cfg_path))
    assert cfg.dynamic_fit_mode == "robust_global_event_reject"
    assert cfg.robust_event_reject_max_iters == 5
    assert cfg.robust_event_reject_residual_z_thresh == pytest.approx(3.2)
    assert cfg.robust_event_reject_local_var_window_sec == pytest.approx(9.0)
    assert cfg.robust_event_reject_local_var_ratio_thresh == pytest.approx(4.5)
    assert cfg.robust_event_reject_min_keep_fraction == pytest.approx(0.6)


def test_config_accepts_adaptive_event_gated_mode_and_params(tmp_path):
    cfg_path = tmp_path / "adaptive_mode.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "dynamic_fit_mode: adaptive_event_gated_regression",
                "adaptive_event_gate_residual_z_thresh: 3.2",
                "adaptive_event_gate_local_var_window_sec: 9.0",
                "adaptive_event_gate_local_var_ratio_thresh: 4.2",
                "adaptive_event_gate_smooth_window_sec: 75.0",
                "adaptive_event_gate_min_trust_fraction: 0.6",
                "adaptive_event_gate_freeze_interp_method: linear_hold",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = Config.from_yaml(str(cfg_path))
    assert cfg.dynamic_fit_mode == "adaptive_event_gated_regression"
    assert cfg.adaptive_event_gate_residual_z_thresh == pytest.approx(3.2)
    assert cfg.adaptive_event_gate_local_var_window_sec == pytest.approx(9.0)
    assert cfg.adaptive_event_gate_local_var_ratio_thresh == pytest.approx(4.2)
    assert cfg.adaptive_event_gate_smooth_window_sec == pytest.approx(75.0)
    assert cfg.adaptive_event_gate_min_trust_fraction == pytest.approx(0.6)
    assert cfg.adaptive_event_gate_freeze_interp_method == "linear_hold"
