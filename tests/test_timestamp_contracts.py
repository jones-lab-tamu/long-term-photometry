import pytest
import numpy as np
from photometry_pipeline.core.types import Chunk

def test_chunk_validate_strict_uniformity():
    """Confirms Chunk.validate() raises ValueError for non-uniform timestamps."""
    fs = 10.0
    t = np.arange(100) / fs
    
    # create a valid chunk first
    chunk = Chunk(
        chunk_id=1,
        source_file="dummy",
        format="npm",
        time_sec=t,
        uv_raw=np.zeros((100, 1)),
        sig_raw=np.zeros((100, 1)),
        fs_hz=fs
    )
    
    # Should pass
    chunk.validate()
    
    # Introduce a jitter
    t_bad = t.copy()
    t_bad[50] += 0.05 # 50% of a sample jitter
    chunk.time_sec = t_bad
    
    with pytest.raises(ValueError, match=r"Uniform grid violation: \d+ intervals \(\d+\.\d+%\) outside tolerance"):
        chunk.validate()

def test_chunk_validate_dropped_frame():
    """Confirms Chunk.validate() raises ValueError for a dropped frame (gap)."""
    fs = 10.0
    t = np.arange(100) / fs
    t_gap = np.delete(t, 50) # Remove one sample
    
    chunk = Chunk(
        chunk_id=1,
        source_file="dummy",
        format="npm",
        time_sec=t_gap,
        uv_raw=np.zeros((99, 1)),
        sig_raw=np.zeros((99, 1)),
        fs_hz=fs
    )

    with pytest.raises(ValueError, match="Uniform grid violation"):
        chunk.validate()

def test_chunk_validate_shape_mismatch():
    """Confirms Chunk.validate() raises ValueError for shape mismatch."""
    fs = 10.0
    t = np.arange(100) / fs
    
    chunk = Chunk(
        chunk_id=1,
        source_file="dummy",
        format="npm",
        time_sec=t,
        uv_raw=np.zeros((99, 1)), # Mismatch
        sig_raw=np.zeros((100, 1)),
        fs_hz=fs
    )
    
    with pytest.raises(ValueError, match="UV Raw shape mismatch"):
        chunk.validate()
