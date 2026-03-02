
import numpy as np
import pytest
import photometry_pipeline.core.baseline as baseline
import photometry_pipeline.core.normalization as normalization
import photometry_pipeline.core.types as types
from photometry_pipeline.config import Config

def test_baseline_gain_invariance():
    """
    Verifies dff computation is invariant to global gain scaling
    when using consistent F0 definitions.
    """
    np.random.seed(42)
    fs = 40.0
    T = 2000
    t = np.arange(T) / fs
    
    # Construct synthetic signals
    # UV: positive baseline + noise
    uv_true_signal = 100.0 + np.random.normal(0, 1.0, size=T)
    
    # Bio: small sine wave
    bio_signal = 2.0 * np.sin(2.0 * np.pi * 0.5 * t)
    
    # Sig: a*uv + b + bio
    a_true = 0.5
    b_true = 50.0
    sig_true_signal = a_true * uv_true_signal + b_true + bio_signal
    
    # Base Config
    config = Config(
        baseline_method='uv_raw_percentile_session',
        baseline_percentile=10.0,
        f0_min_value=1e-9
    )
    
    # -------------------------------------------------------------------------
    # Case 1: uv_raw_percentile_session
    # -------------------------------------------------------------------------
    chunk1 = types.Chunk(
        chunk_id=0,
        source_file="virtual",
        format='npm',
        time_sec=t,
        uv_raw=uv_true_signal[:, None],
        sig_raw=sig_true_signal[:, None],
        fs_hz=fs,
        channel_names=["Region0"],
        delta_f=bio_signal[:, None],
        uv_filt=uv_true_signal[:, None],
        sig_filt=sig_true_signal[:, None]
    )
    
    # Method A: F0 is percentile of UV RAW
    f0_val = np.percentile(chunk1.uv_raw[:, 0], config.baseline_percentile)
    stats1 = types.SessionStats(
        f0_values={"Region0": f0_val},
        method_used='uv_raw_percentile_session'
    )
    
    # Note: compute_dff uses chunk.delta_f (if present) as the numerator.
    # dFF = 100 * delta_f / F0
    # Gain invariance test checks: (g*delta_f) / (g*F0) == delta_f / F0
    dff1 = normalization.compute_dff(chunk1, stats1, config)
    
    # Scale valid signals by gain g=3.0
    g = 3.0
    uv_g = uv_true_signal * g
    sig_g = sig_true_signal * g
    bio_g = bio_signal * g
    
    chunk2 = types.Chunk(
        chunk_id=0,
        source_file="virtual",
        format='npm',
        time_sec=t,
        uv_raw=uv_g[:, None],
        sig_raw=sig_g[:, None],
        fs_hz=fs,
        channel_names=["Region0"],
        delta_f=bio_g[:, None],
        uv_filt=uv_g[:, None],
        sig_filt=sig_g[:, None]
    )
    
    # Recompute F0 for scaled signal
    f0_val_g = np.percentile(chunk2.uv_raw[:, 0], config.baseline_percentile)
    stats2 = types.SessionStats(
        f0_values={"Region0": f0_val_g},
        method_used='uv_raw_percentile_session'
    )
    
    dff2 = normalization.compute_dff(chunk2, stats2, config)
    
    # Assert Invariance
    # (g*delta_f) / (g*F0) = delta_f/F0
    # Use standard assert to avoid np.testing output hang on Windows
    assert np.allclose(dff1, dff2, atol=1e-10), "Method A (raw) failed gain invariance"
    
    # -------------------------------------------------------------------------
    # Case 2: uv_globalfit_percentile_session
    # -------------------------------------------------------------------------
    config.baseline_method = 'uv_globalfit_percentile_session'
    
    # Fit 1 (Original)
    accum1 = baseline.GlobalFitAccumulator()
    accum1.add("Region0", chunk1.uv_raw[:, 0], chunk1.sig_raw[:, 0])
    res1 = accum1.solve()
    a1 = res1["Region0"]['a']
    b1 = res1["Region0"]['b']
    
    uv_est1 = a1 * chunk1.uv_raw[:, 0] + b1
    f0_fit1 = np.percentile(uv_est1, config.baseline_percentile)
    
    stats1_fit = types.SessionStats(
        f0_values={"Region0": f0_fit1},
        global_fit_params={"Region0": {'a': a1, 'b': b1}},
        method_used='uv_globalfit_percentile_session'
    )
    
    dff1_fit = normalization.compute_dff(chunk1, stats1_fit, config)
    
    # Fit 2 (Scaled)
    accum2 = baseline.GlobalFitAccumulator()
    accum2.add("Region0", chunk2.uv_raw[:, 0], chunk2.sig_raw[:, 0])
    res2 = accum2.solve()
    a2 = res2["Region0"]['a']
    b2 = res2["Region0"]['b']
    
    uv_est2 = a2 * chunk2.uv_raw[:, 0] + b2
    f0_fit2 = np.percentile(uv_est2, config.baseline_percentile)
    
    stats2_fit = types.SessionStats(
        f0_values={"Region0": f0_fit2},
        global_fit_params={"Region0": {'a': a2, 'b': b2}},
        method_used='uv_globalfit_percentile_session'
    )
    
    dff2_fit = normalization.compute_dff(chunk2, stats2_fit, config)
    
    assert np.allclose(dff1_fit, dff2_fit, atol=1e-10), "Method B (global fit) failed gain invariance"
