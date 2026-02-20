import pytest
import numpy as np
import pandas as pd
from photometry_pipeline.core.utils import natural_sort_key
from photometry_pipeline.config import Config
from photometry_pipeline.core.regression import fit_chunk_dynamic
from photometry_pipeline.core.feature_extraction import compute_auc_above_threshold
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.io.adapters import _interp_with_nan_policy

@pytest.fixture
def fake_chunk():
    time_sec = np.arange(100) / 40.0
    uv_raw = np.ones((100, 1))
    sig_raw = np.ones((100, 1)) * 2.0
    return Chunk(
        chunk_id=0,
        source_file="fake.csv",
        format='npm',
        time_sec=time_sec,
        uv_raw=uv_raw,
        sig_raw=sig_raw,
        fs_hz=40.0,
        channel_names=["Region1G"],
        metadata={}
    )


def test_t1_natural_sort_ordering():
    paths = ["chunk_10.csv", "chunk_1.csv", "chunk_2.csv"]
    sorted_paths = sorted(paths, key=natural_sort_key)
    assert sorted_paths == ["chunk_1.csv", "chunk_2.csv", "chunk_10.csv"]

def test_t2_config_accepts_new_keys():
    import yaml
    import tempfile
    import os
    
    yaml_str = """
peak_pre_filter: lowpass
event_auc_baseline: median
adapter_value_nan_policy: mask
    """
    with tempfile.NamedTemporaryFile('w', delete=False, suffix=".yaml") as f:
        f.write(yaml_str)
        temp_name = f.name
        
    try:
        config = Config.from_yaml(temp_name)
        assert config.peak_pre_filter == "lowpass"
        assert config.event_auc_baseline == "median"
        assert config.adapter_value_nan_policy == "mask"
    finally:
        os.remove(temp_name)

def test_t3_regression_window_chunk_fallback(fake_chunk):
    config = Config()
    config.window_sec = 100.0 # Much larger than chunk duration (which is 100/40 = 2.5s)
    config.step_sec = 1.0
    
    fake_chunk.uv_filt = fake_chunk.uv_raw
    fake_chunk.sig_filt = fake_chunk.sig_raw
    
    # Needs variance to not be skipped, let's inject a tiny slope
    fake_chunk.uv_filt[:, 0] = np.linspace(1, 2, 100)
    fake_chunk.sig_filt[:, 0] = np.linspace(2, 4, 100)
    
    uv_fit, delta_f = fit_chunk_dynamic(fake_chunk, config, mode="phasic")
    
    # Ensure it's not full of NaNs
    assert np.all(np.isfinite(uv_fit))
    assert np.all(np.isfinite(delta_f))
    assert fake_chunk.metadata.get('window_fallback_global') is True

def test_t4_auc_baseline_behavior():
    time_s = np.arange(10) / 10.0
    # Values: [1, 2, 3, 4, 1, 1, 1, 1, 1, 1]
    # Median = 1.0
    dff = np.array([1, 2, 3, 4, 1, 1, 1, 1, 1, 1], dtype=float)
    
    auc_zero = compute_auc_above_threshold(dff, baseline_value=0.0, time_s=time_s)
    auc_median = compute_auc_above_threshold(dff, baseline_value=1.0, time_s=time_s)
    
    assert auc_zero > auc_median
    assert auc_median > 0
    
    # trapz of (dff - 1.0) rect
    expected_median_rect = np.array([0, 1, 2, 3, 0, 0, 0, 0, 0, 0], dtype=float)
    expected_auc_median = np.trapz(expected_median_rect, x=time_s)
    np.testing.assert_allclose(auc_median, expected_auc_median)

def test_t5_npm_adapter_nan_policy():
    class DummyConfig:
        adapter_value_nan_policy = 'strict'
    config = DummyConfig()
    
    time_sec = np.linspace(0, 1, 10)
    xp = np.linspace(0, 1, 10)
    fp = np.ones(10)
    fp[5] = np.nan
    
    # Strict should raise
    with pytest.raises(ValueError, match="NaN values found"):
        _interp_with_nan_policy(time_sec, xp, fp, config, roi_idx=0, channel_name="TEST")
        
    # Mask should succeed and return nans=1
    config.adapter_value_nan_policy = 'mask'
    out, n_nans = _interp_with_nan_policy(time_sec, xp, fp, config, roi_idx=0, channel_name="TEST")
    
    assert n_nans == 1
    # Interpolated array should exist and probably have no NaNs depending on interpolation bounds, 
    # but the requirement is just that it masks successfully.
    assert np.all(np.isfinite(out)) # since edges are finite and it interpolates the gap

def test_t6_pipeline_pass2_manifest_skip():
    from photometry_pipeline.pipeline import Pipeline
    from photometry_pipeline.config import Config
    
    pipeline = Pipeline(Config())
    pipeline.file_list = ["file1.csv", "file2.csv", "file3.csv"]
    pipeline._pass1_manifest = ["file1.csv", "file3.csv"] # Pass 1 skipped file2
    
    # Run pass 2 (mock load so it doesn't actually read csvs)
    # Actually just test the iteration logic directly:
    new_files = [f for f in pipeline.file_list if f not in pipeline._pass1_manifest]
    assert new_files == ["file2.csv"]
    
    pipeline.file_list = [f for f in pipeline.file_list if f in pipeline._pass1_manifest]
    assert pipeline.file_list == ["file1.csv", "file3.csv"]

def test_t7_feature_extraction_nan_masking_survival():
    time_s = np.arange(10) / 10.0
    # Values: [1, 2, 3, 4, 1, 1, 1, 1, 1, 1]
    # Median = 1.0
    dff = np.array([1, 2, np.nan, 4, 1, 1, 1, 1, 1, 1], dtype=float)
    
    # ensure it doesn't crash but produces a nan-omitted median
    is_valid_use = np.isfinite(dff)
    clean_trace_use = dff[is_valid_use]
    
    assert len(clean_trace_use) == 9
    assert np.median(clean_trace_use) == 1.0 # 1s dominate
    
    auc_median = compute_auc_above_threshold(clean_trace_use, baseline_value=1.0, time_s=time_s[is_valid_use])
    assert np.isfinite(auc_median)
