import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core import preprocessing
from photometry_pipeline.core.normalization import compute_dff
from photometry_pipeline.core.regression import fit_chunk_dynamic
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
