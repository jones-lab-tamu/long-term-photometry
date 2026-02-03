import numpy as np
from .types import Chunk, SessionStats

from ..config import Config

def compute_dff(chunk: Chunk, stats: SessionStats, config: Config) -> np.ndarray:
    """
    Computes dF/F0 using session-level baselines.
    dff = 100 * deltaF / F0
    """
    if chunk.delta_f is None:
        return None
        
    dff = np.zeros_like(chunk.delta_f) * np.nan
    n_rois = chunk.delta_f.shape[1]
    
    for i in range(n_rois):
        channel = chunk.channel_names[i]
        if channel in stats.f0_values:
            f0 = stats.f0_values[channel]
            if f0 > config.f0_min_value: 
                 # delta_f can be NaN if regression failed
                 dff[:, i] = 100.0 * chunk.delta_f[:, i] / f0
            else:
                 # F0 too low?
                 pass 
        else:
             # No baseline?
             pass
             
    return dff
