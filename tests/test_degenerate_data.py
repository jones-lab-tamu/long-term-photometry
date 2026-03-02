import warnings
import numpy as np
import pytest
from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.core.regression import fit_chunk_dynamic
from photometry_pipeline.core.feature_extraction import extract_features

def _create_chunk(dff_val=None, u_val=None, s_val=None, n=100):
    t = np.arange(n) / 10.0
    uv = np.ones((n, 1)) if u_val is None else u_val
    sig = np.ones((n, 1)) if s_val is None else s_val
    
    chunk = Chunk(chunk_id=0, source_file="dummy", format="rwd", time_sec=t, uv_raw=uv, sig_raw=sig, fs_hz=10.0, channel_names=["Region0"], metadata={})
    chunk.uv_filt = uv.copy()
    chunk.sig_filt = sig.copy()
    if dff_val is not None:
        chunk.dff = dff_val
    return chunk

def test_dd1_only():
    """DD1: Exactly 1 finite sample in window."""
    config = Config()
    config.target_fs_hz = 10.0
    config.chunk_duration_sec = 10.0
    config.window_sec = 2.0
    config.step_sec = 1.0
    config.min_samples_per_window = 1 
    
    uv = np.ones((100, 1)) * np.nan
    sig = np.ones((100, 1)) * np.nan
    # One valid sample at index 10 (falls into first window center=10)
    uv[10, 0] = 1.0
    sig[10, 0] = 1.0
    
    chunk = _create_chunk(u_val=uv, s_val=sig)
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", RuntimeWarning)
        fit_chunk_dynamic(chunk, config, mode='phasic')
        
    assert 'qc_warnings' in chunk.metadata
    warns = chunk.metadata['qc_warnings']
    assert any("DEGENERATE[DD1]" in msg for msg in warns), "DD1 missing"
    assert not any("DEGENERATE[DD2]" in msg for msg in warns), "DD2 should not occur"

def test_dd2_denom_guard_global_fallback():
    """DD2 Denominator Guard: Force global fallback, make var_u ~0, ensure division is bypassed and DD2 emitted."""
    config = Config()
    config.target_fs_hz = 10.0
    config.chunk_duration_sec = 10.0
    # Make window > chunk so it triggers the fallback block
    config.window_sec = 20.0 
    
    # Needs len >= 2 to pass the DD1 guard in global fallback
    uv = np.ones((100, 1)) * 1.0 # Constant UV -> var_u = 0
    sig = np.arange(100).reshape(-1, 1).astype(float) # s_w has variation
    
    chunk = _create_chunk(u_val=uv, s_val=sig)
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", RuntimeWarning)
        # Should not raise any warnings for 0 division
        fit_chunk_dynamic(chunk, config, mode='phasic')
        
    assert 'qc_warnings' in chunk.metadata
    warns = chunk.metadata['qc_warnings']
    # Must emit DD2 since var_u = 0
    assert any("DEGENERATE[DD2]" in msg and "fallback" in msg for msg in warns), "DD2 missing in global fallback"
    # No DD1 should occur
    assert not any("DEGENERATE[DD1]" in msg for msg in warns), "DD1 should not occur"

def test_dd3_all_nan():
    """DD3: All-NaN/empty clean slice enforces return row without executing stats."""
    config = Config()
    config.peak_threshold_method = 'percentile'
    
    dff = np.ones((100, 1)) * np.nan
    chunk = _create_chunk(dff_val=dff)
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", RuntimeWarning)
        df = extract_features(chunk, config)
        
    assert 'qc_warnings' in chunk.metadata
    warns = chunk.metadata['qc_warnings']
    assert any("DEGENERATE[DD3]" in msg for msg in warns)
    # Ensure it returns the null row
    assert np.isnan(df['mean'].iloc[0])

def test_dd5_single_element_segment():
    """DD5: Valid segment of length 1 should skip find_peaks."""
    config = Config()
    config.peak_threshold_method = 'mean_std'
    
    dff = np.ones((100, 1)) * np.nan
    dff[0:5, 0] = 1.0  # Segment 1: length 5 
    dff[20:21, 0] = 2.0 # Segment 2: length 1 -> DD5
    
    chunk = _create_chunk(dff_val=dff)
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", RuntimeWarning)
        df = extract_features(chunk, config)
        
    assert 'qc_warnings' in chunk.metadata
    warns = chunk.metadata['qc_warnings']
    assert any("DEGENERATE[DD5]" in msg for msg in warns)
    # Length 1 segment skips peak find
    assert df['peak_count'].iloc[0] == 0 

def test_dd4_zero_robust_variance():
    """DD4: Flatline trace leads to zero robust variance."""
    config = Config()
    config.peak_threshold_method = 'median_mad'
    config.peak_threshold_k = 2.0
    
    dff = np.ones((100, 1)) * np.nan
    dff[0:10, 0] = 5.0 # Flatline segment of length 10
    
    chunk = _create_chunk(dff_val=dff)
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", RuntimeWarning)
        extract_features(chunk, config)
        
    assert 'qc_warnings' in chunk.metadata
    warns = chunk.metadata['qc_warnings']
    assert any("DEGENERATE[DD4]" in msg for msg in warns)
