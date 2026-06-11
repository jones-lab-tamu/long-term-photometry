import os
import json
import h5py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.core.types import Chunk

@pytest.fixture
def mock_chunk():
    chunk = MagicMock(spec=Chunk)
    chunk.channel_names = ["ROI1", "ROI2"]
    chunk.time_sec = np.array([0.0, 0.1, 0.2])
    
    # 3 samples, 2 ROIs
    chunk.sig_raw = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])
    chunk.uv_raw = np.array([[0.5, 0.6], [0.55, 0.65], [0.6, 0.7]])
    
    # Optional fields
    chunk.delta_f = np.array([[0.1, 0.2], [0.11, 0.22], [0.12, 0.24]])
    chunk.dff = np.array([[10.0, 20.0], [11.0, 22.0], [12.0, 24.0]])
    return chunk


def test_hdf5_cache_tonic_schema(tmp_path, mock_chunk):
    cache_path = os.path.join(tmp_path, "tonic_trace_cache.h5")
    
    with Hdf5TraceCacheWriter(cache_path, "tonic", config=None) as writer:
        writer.add_chunk(mock_chunk, chunk_id=0, source_file="sess1/fluorescence.csv")
        writer.add_chunk(mock_chunk, chunk_id=1, source_file="sess2/fluorescence.csv")

    assert os.path.exists(cache_path)
    
    with h5py.File(cache_path, 'r') as f:
        # Check meta schema
        assert 'meta' in f
        assert f['meta/schema_version'][()][0] == 1
        assert f['meta'].attrs['mode'] == 'tonic'
        assert f['meta/n_chunks'][()][0] == 2
        
        rois = f['meta/rois'][:]
        assert list(rois) == [b"ROI1", b"ROI2"]
        
        chunk_ids = f['meta/chunk_ids'][:]
        assert list(chunk_ids) == [0, 1]
        
        sources = f['meta/source_files'][:]
        assert list(sources) == [b"sess1/fluorescence.csv", b"sess2/fluorescence.csv"]
        
        # Check ROI/Chunk hierarchy
        for r in ["ROI1", "ROI2"]:
            for c_id in [0, 1]:
                grp_path = f"roi/{r}/chunk_{c_id}"
                assert grp_path in f
                grp = f[grp_path]
                
                # Check datasets
                assert 'time_sec' in grp
                assert 'sig_raw' in grp
                assert 'uv_raw' in grp
                assert 'deltaF' in grp
                assert 'dff' not in grp  # Tonic mode does not export dff to cache directly here yet

                # Data shapes
                assert grp['time_sec'].shape == (3,)
                assert grp['sig_raw'].shape == (3,)


def test_hdf5_cache_phasic_schema(tmp_path, mock_chunk):
    cache_path = os.path.join(tmp_path, "phasic_trace_cache.h5")
    
    with Hdf5TraceCacheWriter(cache_path, "phasic", config=None) as writer:
        writer.add_chunk(mock_chunk, chunk_id=10, source_file="sess_A/fluorescence.csv")

    assert os.path.exists(cache_path)
    
    with h5py.File(cache_path, 'r') as f:
        assert f['meta'].attrs['mode'] == 'phasic'
        
        for r in ["ROI1", "ROI2"]:
            grp_path = f"roi/{r}/chunk_10"
            assert grp_path in f
            grp = f[grp_path]
            
            assert 'time_sec' in grp
            assert 'sig_raw' in grp
            assert 'uv_raw' in grp
            assert 'dff' in grp
            assert 'deltaF' not in grp  # Phasic mode
            
def test_hdf5_cache_abort_on_exception(tmp_path, mock_chunk):
    cache_path = os.path.join(tmp_path, "aborted_cache.h5")
    tmp_cache_path = cache_path + ".tmp"
    
    try:
        with Hdf5TraceCacheWriter(cache_path, "phasic", config=None) as writer:
            writer.add_chunk(mock_chunk, 0, "test")
            # tmp file should exist while open
            assert os.path.exists(tmp_cache_path)
            raise ValueError("Pipeline crashed")
    except ValueError:
        pass
        
    # tmp file should be cleaned up, and final file should never exist
    assert not os.path.exists(tmp_cache_path)
    assert not os.path.exists(cache_path)


def test_pipeline_integration_cache_production(tmp_path):
    import sys
    import subprocess
    input_dir = tmp_path / "input_RWD"
    input_dir.mkdir()
    
    # Needs a config file
    config_path = tmp_path / "config.yaml"
    import shutil
    shutil.copy2(os.path.join(os.path.dirname(__file__), "qc_universal_config.yaml"), config_path)
    with open(config_path, "a", encoding="utf-8") as f:
        f.write("\nbaseline_reference_smoothing_window_sec: 120.0\n")
    
    # 1. Generate minimal (10 min) synthetic data
    gen_cmd = [
        sys.executable, "tools/synth_photometry_dataset.py",
        "--out", str(input_dir),
        "--format", "rwd",
        "--config", str(config_path),
        "--total-days", "0.1",
        "--recordings-per-hour", "2",
        "--recording-duration-min", "10.0",
        "--n-rois", "1",
        "--seed", "42"
    ]
    subprocess.check_call(gen_cmd)
    
    # 2. Run analysis
    out_dir = tmp_path / "pipeline_out"
    run_cmd = [
        sys.executable, "tools/run_full_pipeline_deliverables.py",
        "--input", str(input_dir),
        "--out", str(out_dir),
        "--config", str(config_path),
        "--format", "rwd",
        "--mode", "both",
        "--sessions-per-hour", "2"
    ]
    subprocess.check_call(run_cmd)
    
    # 3. Validation
    # Tonic
    tonic_cache = out_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5"
    assert tonic_cache.exists()
    with h5py.File(tonic_cache, 'r') as f:
        assert f['meta/schema_version'][()][0] == 1
        assert 'meta/rois' in f
        assert 'meta/chunk_ids' in f
        assert 'meta/source_files' in f
        assert 'meta/n_chunks' in f
        
        rois = f['meta/rois'][:]
        assert len(rois) > 0
        roi = rois[0].decode('utf-8')
        
        chunk = f['meta/chunk_ids'][0]
        
        grp = f[f'roi/{roi}/chunk_{chunk}']
        assert 'time_sec' in grp
        assert 'deltaF' in grp
    
    # Phasic
    phasic_cache = out_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
    assert phasic_cache.exists()
    with h5py.File(phasic_cache, 'r') as f:
        assert f['meta/schema_version'][()][0] == 1
        assert 'meta/rois' in f
        assert 'meta/chunk_ids' in f
        assert 'meta/source_files' in f
        assert 'meta/n_chunks' in f
        
        rois = f['meta/rois'][:]
        roi = rois[0].decode('utf-8')
        chunk = f['meta/chunk_ids'][0]
        
        grp = f[f'roi/{roi}/chunk_{chunk}']
        assert 'time_sec' in grp
        assert 'dff' in grp
        found_dynamic_fit_qc_attrs = False
        for roi_raw in rois:
            roi_name = roi_raw.decode('utf-8') if isinstance(roi_raw, bytes) else str(roi_raw)
            for chunk_id in f['meta/chunk_ids'][:]:
                candidate = f[f'roi/{roi_name}/chunk_{int(chunk_id)}']
                if bool(candidate.attrs.get("dynamic_fit_qc_available", False)):
                    qc_attr_names = [
                        str(name)
                        for name in candidate.attrs.keys()
                        if str(name).startswith("dynamic_fit_qc_")
                    ]
                    if (
                        "dynamic_fit_qc_flags" in candidate.attrs
                        or any(name != "dynamic_fit_qc_available" for name in qc_attr_names)
                    ):
                        found_dynamic_fit_qc_attrs = True
                        break
            if found_dynamic_fit_qc_attrs:
                break
        assert found_dynamic_fit_qc_attrs

    qc_csv = out_dir / "_analysis" / "phasic_out" / "qc" / "dynamic_fit_qc_by_chunk.csv"
    qc_json = out_dir / "_analysis" / "phasic_out" / "qc" / "dynamic_fit_qc_by_chunk.json"
    assert qc_csv.exists()
    assert qc_json.exists()
    qc_json_text = qc_json.read_text(encoding="utf-8")
    assert "NaN" not in qc_json_text
    assert "Infinity" not in qc_json_text
    qc_df = pd.read_csv(qc_csv)
    assert len(qc_df) >= 1
    for col in [
        "roi",
        "chunk_id",
        "source_file",
        "dynamic_fit_mode",
        "slope_constraint",
        "fitted_ref_to_signal_range_ratio",
        "fitted_ref_response_scale_fraction",
        "dynamic_fit_needs_inspection",
        "dynamic_fit_qc_flags",
        "dynamic_fit_qc_hard_flags",
        "dynamic_fit_qc_soft_flags",
        "dynamic_fit_qc_severity",
        "dynamic_fit_has_hard_flags",
        "dynamic_fit_has_soft_flags",
    ]:
        assert col in qc_df.columns

    candidate_csv = out_dir / "_analysis" / "phasic_out" / "qc" / "baseline_reference_candidate_by_chunk.csv"
    candidate_json = out_dir / "_analysis" / "phasic_out" / "qc" / "baseline_reference_candidate_by_chunk.json"
    assert candidate_csv.exists()
    assert candidate_json.exists()
    candidate_json_text = candidate_json.read_text(encoding="utf-8")
    assert "NaN" not in candidate_json_text
    assert "Infinity" not in candidate_json_text
    candidate_json_payload = json.loads(candidate_json_text)
    assert candidate_json_payload
    assert isinstance(candidate_json_payload[0]["reference_comparison_flags"], list)
    candidate_df = pd.read_csv(candidate_csv)
    assert len(candidate_df) >= 1
    for col in [
        "roi",
        "chunk_id",
        "source_file",
        "recording_mode",
        "dynamic_fit_mode",
        "slope_constraint",
        "baseline_ref_candidate_available",
        "baseline_ref_method",
        "baseline_ref_lowpass_cutoff_hz",
        "baseline_ref_smoothing_window_sec",
        "baseline_ref_requested_smoothing_window_sec",
        "baseline_ref_actual_smoothing_window_sec",
        "baseline_ref_default_smoothing_window_sec",
        "baseline_ref_min_smoothing_window_sec",
        "baseline_ref_chunk_duration_sec",
        "baseline_ref_smoothing_window_fraction_of_chunk",
        "baseline_ref_max_window_fraction_of_chunk",
        "baseline_ref_large_window_fraction_warning",
        "baseline_ref_smoothing_window_adjusted",
        "baseline_ref_smoothing_window_warning",
        "baseline_ref_to_signal_range_ratio",
        "baseline_ref_response_scale_fraction",
        "dynamic_minus_baseline_ref_rms",
        "dynamic_fit_qc_severity",
        "dynamic_fit_qc_flags",
        "reference_comparison_class",
        "dynamic_reference_viability",
        "baseline_reference_viability",
        "reference_comparison_review_level",
        "reference_comparison_notes",
        "reference_comparison_flags",
        "dynamic_has_negative_or_mixed_coupling",
        "dynamic_has_response_scale_rich",
        "dynamic_has_low_or_flat_reference",
        "baseline_is_available",
        "baseline_has_low_or_flat_reference",
        "baseline_has_response_scale_rich",
        "baseline_window_large_fraction_of_chunk",
        "baseline_window_adjusted",
    ]:
        assert col in candidate_df.columns
    assert set(candidate_df["baseline_ref_requested_smoothing_window_sec"].astype(float)) == {120.0}
    assert set(candidate_df["baseline_ref_actual_smoothing_window_sec"].astype(float)) == {120.0}

    qc_summary_path = out_dir / "_analysis" / "phasic_out" / "qc" / "qc_summary.json"
    with open(qc_summary_path, "r", encoding="utf-8") as f:
        qc_summary = json.load(f)
    assert "baseline_reference_candidate_qc_summary" in qc_summary
    candidate_summary = qc_summary["baseline_reference_candidate_qc_summary"]
    assert candidate_summary["roi_chunk_candidate_count"] >= 1
    assert candidate_summary["roi_chunk_candidate_available_count"] >= 1
    assert "reference_candidate_comparison_summary" in qc_summary
    comparison_summary = qc_summary["reference_candidate_comparison_summary"]
    assert comparison_summary["roi_chunk_comparison_count"] >= 1
    assert isinstance(comparison_summary["reference_comparison_class_counts"], dict)
    assert isinstance(comparison_summary["dynamic_reference_viability_counts"], dict)
    assert isinstance(comparison_summary["baseline_reference_viability_counts"], dict)
    assert isinstance(comparison_summary["reference_comparison_review_level_counts"], dict)
    assert isinstance(comparison_summary["reference_comparison_flag_counts"], dict)
