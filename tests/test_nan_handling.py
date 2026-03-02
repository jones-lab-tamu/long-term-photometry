import pytest
import numpy as np
from photometry_pipeline.core.preprocessing import get_lowpass_sos, filter_finite_blocks
from photometry_pipeline.config import Config

def test_nan_safe_filtering_segmentation():
    """
    Verifies that filter_finite_blocks filters valid segments independently
    and preserves NaN gaps without bridging or contamination.
    """
    fs = 20.0
    t = np.arange(100) / fs
    sig = np.sin(2 * np.pi * 1.0 * t)
    
    # Insert NaN gap indices 40 to 60
    sig_gapped = sig.copy()
    sig_gapped[40:60] = np.nan
    
    sos = get_lowpass_sos(fs, 2.0, 2)
    # Order 2 -> padlen = 3 * (4+1) = 15
    # order = 2 
    
    filtered, meta = filter_finite_blocks(sig_gapped, sos)
    
    # 1. Gaps are preserved
    assert np.all(np.isnan(filtered[40:60]))
    
    # 2. Valid segments are filtered
    assert np.all(np.isfinite(filtered[0:40]))
    assert np.all(np.isfinite(filtered[60:100]))
    
    # 4. Metadata
    assert meta['n_gaps'] >= 1
    assert meta['rois_affected'] == 1

def test_short_segments_dropped_strict_padlen():
    """Verifies that segments shorter than padlen are skipped."""
    fs = 20.0
    sos = get_lowpass_sos(fs, 2.0, 2)
    # Order 2 -> n_sections=1 -> padlen = 3 * (2*1 + 1) = 9.
    
    # Length 8 < 9
    sig = np.ones(8)
    filtered, meta = filter_finite_blocks(sig, sos)
    
    assert np.all(np.isnan(filtered))
    assert meta['samples_skipped'] == 8

def test_inf_handling():
    """Verifies Inf treated as gap."""
    fs = 20.0
    sig = np.zeros(50)
    sig[25] = np.inf
    
    sos = get_lowpass_sos(fs, 2.0, 2)
    filtered, meta = filter_finite_blocks(sig, sos)
    
    assert np.isnan(filtered[25])
    assert meta['n_gaps'] >= 1
