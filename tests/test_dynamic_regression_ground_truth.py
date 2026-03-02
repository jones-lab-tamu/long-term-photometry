import pytest
import numpy as np
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.config import Config
from photometry_pipeline.core import regression

def test_dynamic_regression_recovers_ground_truth_strict():
    """
    Synthesizes data with known time-varying coefficients a(t), b(t) and bio(t).
    Verifies that fit_chunk_dynamic recovers the drift and preserves bio signal.
    """
    # 1. Setup
    fs = 20.0
    duration = 300.0 # 5 min
    N = int(duration * fs)
    t = np.arange(N) / fs
    
    # Config
    cfg = Config()
    cfg.target_fs_hz = fs
    cfg.window_sec = 60.0
    cfg.step_sec = 10.0
    cfg.min_samples_per_window = 0
    # Disable attenuation gating to verify mathematical correctness of OLS
    cfg.r_high = -1.0 
    cfg.r_low = -2.0
    cfg.lowpass_hz = 4.0 # Ensure bio signal passes
    cfg.filter_order = 2
    
    # 2. Synthesis
    np.random.seed(12345) # Fixed seed
    
    # Drift components (slow)
    # a(t): 0.5 to 1.5 sine wave (very slow)
    a_true = 1.0 + 0.5 * np.sin(2 * np.pi * t / 200.0) 
    # b(t): linear drift
    b_true = 50.0 + 0.1 * t
    
    # UV signal (artifact source)
    # "Breathing" high frequency component to allow OLS locking
    uv_drift = np.cumsum(np.random.randn(N)) * 0.1 + 100.0
    uv_fast = 5.0 * np.sin(2 * np.pi * 0.5 * t) # 0.5 Hz "breathing"
    uv_raw = uv_drift + uv_fast
    
    # Bio signal 
    # Must be distinct frequency or sparse.
    # Widen to ensure it survives lowpass filter (tau=20 samples = 1.0s)
    bio_true = np.zeros(N)
    # A few distinct peaks
    for i in range(200, N-200, 400):
        # 1.0s time constant exponential
        bio_true[i:i+60] += 20.0 * np.exp(-np.arange(60)/20.0)
        
    # Noise
    noise = np.random.randn(N) * 0.05
    
    # Sig construction
    # sig = a(t)*uv + b(t) + bio + noise
    sig_raw = a_true * uv_raw + b_true + bio_true + noise
    
    # Chunk object
    chunk = Chunk(
        chunk_id=0, source_file="synth", format="synth",
        time_sec=t, 
        uv_raw=uv_raw.reshape(-1, 1),
        sig_raw=sig_raw.reshape(-1, 1),
        fs_hz=fs,
        channel_names=["Region0"]
    )
    # Assume prefiltered input (regression uses these)
    chunk.uv_filt = uv_raw.reshape(-1, 1)
    chunk.sig_filt = sig_raw.reshape(-1, 1)
    
    # 3. Run Dynamic Regression
    uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, cfg, mode='phasic')
    
    # 4. Assertions
    
    # A. Check uv_fit approximation (should match a*uv + b)
    expected_fit = a_true.reshape(-1, 1) * uv_raw.reshape(-1, 1) + b_true.reshape(-1, 1)
    
    # Restrict comparison to center to verify OLS operation
    # fit_chunk_dynamic extrapolates constant values outside the window centers range.
    # We must exclude this region to test the actual regression performance.
    # Window=60s. Margin should be at least window/2 + step? Let's use 60s to be safe.
    margin_sec = cfg.window_sec 
    margin_idx = int(margin_sec * fs)
    valid_mask = np.zeros(N, dtype=bool)
    if 2 * margin_idx < N:
        valid_mask[margin_idx : N - margin_idx] = True
    
    valid_mask = valid_mask & np.isfinite(uv_fit.flatten())
    
    fit_error = uv_fit.flatten()[valid_mask] - expected_fit.flatten()[valid_mask]
    
    rmse_fit = np.sqrt(np.mean(fit_error**2))
    mae_fit = np.median(np.abs(fit_error))
    
    # Thresholds: With bio signal present, fit will have some bias/lag.
    # RMSE < 7.0 covers plausible dynamic tracking lag.
    # We also verify that uv_fit correlates strongly with expected artifact, proving we modeled it.
    assert rmse_fit < 7.0, f"Fit RMSE {rmse_fit:.3f} too high (MAE {mae_fit:.3f})"
    
    fit_corr = np.corrcoef(uv_fit.flatten()[valid_mask], expected_fit.flatten()[valid_mask])[0, 1]
    assert fit_corr > 0.99, f"Fit correlation {fit_corr:.4f} too low - failed to model artifact structure"
    
    # B. Check delta_f preservation
    recovered = delta_f.flatten()[valid_mask]
    true_bio_seg = bio_true.flatten()[valid_mask]
    
    # Correlation should be high (bio signal is preserved)
    # Note: OLS subtracts mean, so we correlate zero-mean signal with positive pulse.
    # 0.5 is typical.
    corr = np.corrcoef(recovered, true_bio_seg)[0, 1]
    assert corr > 0.4, f"Bio correlation {corr:.3f} low"
    
    # Amplitude recovery on peaks
    # Check max of bio vs max of recovered at peak locations
    peak_locs = np.where(true_bio_seg > 1.0)[0]
    if len(peak_locs) > 0:
        rec_peaks = recovered[peak_locs]
        true_peaks = true_bio_seg[peak_locs]
        # We want to ensure we haven't subtracted it out
        # It's okay if it's slightly noisy, but it should be positive and substantial
        mean_recovery = np.mean(rec_peaks)
        mean_true = np.mean(true_peaks)
        recovery_ratio = mean_recovery / mean_true
        assert recovery_ratio > 0.6, f"Bio signal attenuated too much: ratio {recovery_ratio:.2f}"
