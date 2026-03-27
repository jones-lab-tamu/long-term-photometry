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


def test_dynamic_fit_mode_default_matches_explicit_rolling():
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
        dynamic_fit_mode="rolling_local_regression",
    )

    c_default = _make_chunk(uv, sig, fs)
    c_explicit = _make_chunk(uv, sig, fs)
    _prepare_filtered(c_default, cfg_default)
    _prepare_filtered(c_explicit, cfg_explicit)

    uv_fit_default, delta_default = fit_chunk_dynamic(c_default, cfg_default, mode="phasic")
    uv_fit_explicit, delta_explicit = fit_chunk_dynamic(c_explicit, cfg_explicit, mode="phasic")

    np.testing.assert_allclose(uv_fit_default, uv_fit_explicit, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(delta_default, delta_explicit, rtol=0.0, atol=1e-12)
    assert c_default.metadata["dynamic_fit_mode_resolved"] == "rolling_local_regression"
    assert c_default.metadata["dynamic_fit_engine"] == "rolling_local_ols_v1"


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
