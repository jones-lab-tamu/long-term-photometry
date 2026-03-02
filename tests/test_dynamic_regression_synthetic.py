
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
from photometry_pipeline.core import regression, types
from photometry_pipeline.config import Config

def test_dynamic_regression_synthetic_recovery():
    """
    Verifies that dynamic regression recovers time-varying a(t) and b(t)
    and preserves bio signal with high fidelity.
    """
    np.random.seed(42)
    fs = 40.0
    duration_sec = 600.0
    T = int(fs * duration_sec)
    t = np.arange(T) / fs
    
    # 1. UV Signal
    # "filtered-noise UV driver EXACTLY"
    wn = np.random.normal(0, 1.0, size=T)
    uv = gaussian_filter1d(wn, sigma=10)
    uv = uv / np.std(uv) * 10.0
    uv = uv.astype(float)
    
    # 2. Time-varying parameters
    # a(t)
    a_t = 2.0 + 0.2 * np.sin(2 * np.pi * t / (T/fs/2))
    
    # b(t)
    b_t = 50.0 + 5.0 * np.cos(2 * np.pi * t / (T/fs))
    
    # 3. Bio signal
    bio_t = 1.0 * np.sin(2 * np.pi * t / 15.0)
    
    # 4. Composite Signal
    noise = np.random.normal(0, 0.1, size=T)
    sig = a_t * uv + b_t + bio_t + noise
    
    # 5. Build Chunk
    chunk = types.Chunk(
        chunk_id=0,
        source_file="virtual",
        format='npm',
        time_sec=t,
        uv_raw=uv[:, None],
        sig_raw=sig[:, None],
        fs_hz=fs,
        channel_names=["Region0"],
        uv_filt=uv[:, None].copy(),
        sig_filt=sig[:, None].copy()
    )
    
    # 6. Config
    config = Config(
        window_sec=60.0,
        step_sec=10.0,
        r_low=0.0,
        r_high=1.0,
        g_min=1.0,
        min_samples_per_window=0,
        min_valid_windows=5
    )
    
    # 7. Run Regression
    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, config, mode="phasic")
    
    # 8. Assertions
    # 1) RMSE of artifact reconstruction
    true_artifact = a_t * uv + b_t
    # uv_fit is (T, 1)
    rmse = np.sqrt(np.mean((uv_fit[:, 0] - true_artifact)**2))
    
    # 2) Correlation with bio
    # delta_f is (T, 1)
    corr = np.corrcoef(delta_f[:, 0], bio_t)[0, 1]
    
    # 3) Relative Error of Standard Deviation
    std_est = np.std(delta_f[:, 0])
    std_true = np.std(bio_t)
    rel_err = abs(std_est - std_true) / std_true
    
    assert rmse < 0.5, f"RMSE {rmse:.4f} too high (limit 0.5)"
    assert corr > 0.90, f"Correlation {corr:.4f} too low (limit 0.90)"
    assert rel_err < 0.10, f"Rel Std Error {rel_err:.4f} too high (limit 0.10)"
