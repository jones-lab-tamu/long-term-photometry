import copy

import numpy as np
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.core import preprocessing
from photometry_pipeline.core.normalization import compute_dff
from photometry_pipeline.core.regression import fit_chunk_dynamic
from photometry_pipeline.core.types import Chunk, SessionStats


def _make_synth_chunk(fmt: str, channel_name: str, seed: int = 123) -> Chunk:
    rng = np.random.default_rng(seed)
    fs = 40.0
    n = 4800  # 120 s
    t = np.arange(n, dtype=float) / fs

    uv = (
        5.0
        + 0.6 * np.sin(2.0 * np.pi * 0.20 * t)
        + 0.2 * np.sin(2.0 * np.pi * 1.60 * t + 0.3)
        + 0.05 * rng.standard_normal(n)
    )
    a_t = 1.15 + 0.25 * np.sin(2.0 * np.pi * t / 80.0)
    b_t = 52.0 + 2.5 * np.cos(2.0 * np.pi * t / 110.0)
    bio = 0.30 * np.sin(2.0 * np.pi * 0.14 * t) + 0.12 * np.sin(2.0 * np.pi * 0.85 * t)
    sig = a_t * uv + b_t + bio + 0.05 * rng.standard_normal(n)

    return Chunk(
        chunk_id=0,
        source_file=f"{fmt}_synth.csv",
        format=fmt,
        time_sec=t,
        uv_raw=uv[:, None],
        sig_raw=sig[:, None],
        fs_hz=fs,
        channel_names=[channel_name],
        metadata={},
    )


def _clone_chunk(chunk: Chunk) -> Chunk:
    out = Chunk(
        chunk_id=chunk.chunk_id,
        source_file=chunk.source_file,
        format=chunk.format,
        time_sec=chunk.time_sec.copy(),
        uv_raw=chunk.uv_raw.copy(),
        sig_raw=chunk.sig_raw.copy(),
        fs_hz=float(chunk.fs_hz),
        channel_names=list(chunk.channel_names),
        metadata=dict(chunk.metadata or {}),
    )
    if chunk.uv_filt is not None:
        out.uv_filt = chunk.uv_filt.copy()
    if chunk.sig_filt is not None:
        out.sig_filt = chunk.sig_filt.copy()
    return out


def _prepare_filtered(chunk: Chunk, cfg: Config) -> None:
    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, cfg)
    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, cfg)


@pytest.mark.parametrize(
    "fmt,channel_name",
    [
        ("rwd", "CH5"),
        ("npm", "Region0"),
    ],
)
def test_dynamic_fit_produces_uvfit_deltaf_and_dff_for_rwd_and_npm(fmt, channel_name):
    chunk = _make_synth_chunk(fmt=fmt, channel_name=channel_name)
    cfg = Config(window_sec=45.0, min_samples_per_window=20, lowpass_hz=4.0, filter_order=2)
    _prepare_filtered(chunk, cfg)

    uv_fit, delta_f = fit_chunk_dynamic(chunk, cfg, mode="phasic")
    assert uv_fit is not None
    assert delta_f is not None
    assert uv_fit.shape == chunk.sig_raw.shape == chunk.uv_raw.shape
    assert delta_f.shape == chunk.sig_raw.shape
    assert np.isfinite(uv_fit).mean() > 0.99
    assert np.isfinite(delta_f).mean() > 0.99
    np.testing.assert_allclose(delta_f[:, 0], chunk.sig_raw[:, 0] - uv_fit[:, 0], rtol=0.0, atol=1e-12)

    chunk.uv_fit = uv_fit
    chunk.delta_f = delta_f
    f0 = float(np.nanpercentile(chunk.uv_raw[:, 0], 10.0))
    stats = SessionStats(f0_values={channel_name: f0})
    dff = compute_dff(chunk, stats, cfg)
    assert dff is not None
    assert dff.shape == chunk.sig_raw.shape
    assert np.isfinite(dff).mean() > 0.99

    info = chunk.metadata.get("dynamic_fit_engine_info", {})
    assert chunk.metadata.get("dynamic_fit_engine") == "rolling_local_ols_v1"
    assert set(info.get("legacy_knobs_not_used_in_engine", [])) == {
        "step_sec",
        "min_valid_windows",
        "r_low",
        "r_high",
        "g_min",
    }


def test_legacy_knobs_are_inert_for_active_rolling_fit():
    cfg = Config(window_sec=45.0, min_samples_per_window=16, lowpass_hz=3.5, filter_order=2)
    base = _make_synth_chunk(fmt="npm", channel_name="Region0", seed=777)
    _prepare_filtered(base, cfg)

    uv_base, df_base = fit_chunk_dynamic(_clone_chunk(base), cfg, mode="phasic")

    legacy_shifted = copy.deepcopy(cfg)
    legacy_shifted.step_sec = 1.0
    legacy_shifted.min_valid_windows = 999
    legacy_shifted.r_low = 0.97
    legacy_shifted.r_high = 0.98
    legacy_shifted.g_min = 123.0
    uv_legacy, df_legacy = fit_chunk_dynamic(_clone_chunk(base), legacy_shifted, mode="phasic")

    np.testing.assert_allclose(uv_base, uv_legacy, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(df_base, df_legacy, rtol=0.0, atol=1e-12)


def test_active_controls_window_lowpass_and_min_samples_change_fit():
    raw = _make_synth_chunk(fmt="npm", channel_name="Region0", seed=321)

    cfg_lp_low = Config(window_sec=45.0, min_samples_per_window=16, lowpass_hz=0.6, filter_order=2)
    c1 = _clone_chunk(raw)
    _prepare_filtered(c1, cfg_lp_low)
    uv_lp_low, _ = fit_chunk_dynamic(c1, cfg_lp_low, mode="phasic")

    cfg_lp_high = Config(window_sec=45.0, min_samples_per_window=16, lowpass_hz=8.0, filter_order=2)
    c2 = _clone_chunk(raw)
    _prepare_filtered(c2, cfg_lp_high)
    uv_lp_high, _ = fit_chunk_dynamic(c2, cfg_lp_high, mode="phasic")

    assert float(np.nanmean(np.abs(uv_lp_low - uv_lp_high))) > 1e-3

    c3 = _clone_chunk(raw)
    _prepare_filtered(c3, cfg_lp_high)
    cfg_w_small = copy.deepcopy(cfg_lp_high)
    cfg_w_small.window_sec = 15.0
    cfg_w_large = copy.deepcopy(cfg_lp_high)
    cfg_w_large.window_sec = 90.0
    uv_w_small, _ = fit_chunk_dynamic(_clone_chunk(c3), cfg_w_small, mode="phasic")
    uv_w_large, _ = fit_chunk_dynamic(_clone_chunk(c3), cfg_w_large, mode="phasic")
    assert float(np.nanmean(np.abs(uv_w_small - uv_w_large))) > 1e-3

    c4 = _clone_chunk(raw)
    _prepare_filtered(c4, cfg_lp_high)
    # Force sparse valid support so min_samples meaningfully changes fit behavior.
    c4.uv_filt[::5, 0] = np.nan
    c4.sig_filt[::5, 0] = np.nan
    cfg_ms_low = copy.deepcopy(cfg_lp_high)
    cfg_ms_low.window_sec = 30.0
    cfg_ms_low.min_samples_per_window = 5
    cfg_ms_high = copy.deepcopy(cfg_ms_low)
    cfg_ms_high.min_samples_per_window = 1150  # clipped to window_samples internally
    uv_ms_low, _ = fit_chunk_dynamic(_clone_chunk(c4), cfg_ms_low, mode="phasic")
    uv_ms_high, _ = fit_chunk_dynamic(_clone_chunk(c4), cfg_ms_high, mode="phasic")
    assert float(np.nanmean(np.abs(uv_ms_low - uv_ms_high))) > 1e-3
