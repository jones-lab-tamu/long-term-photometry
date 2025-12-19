import numpy as np
from scipy.signal import sosfiltfilt, butter
from ..config import Config

def get_lowpass_sos(fs_hz: float, cutoff_hz: float, order: int):
    nyquist = 0.5 * fs_hz
    if cutoff_hz >= nyquist:
        # If cutoff is too high, return identity or warn?
        # For now, clamp or error. 1Hz vs 40Hz is fine.
        return None
    return butter(order, cutoff_hz / nyquist, btype='low', output='sos')

def lowpass_filter(data: np.ndarray, fs_hz: float, config: Config) -> np.ndarray:
    """
    Applies zero-phase lowpass filter to columns of data.
    data: (T, N)
    """
    sos = get_lowpass_sos(fs_hz, config.lowpass_hz, config.filter_order)
    if sos is None:
        return data
        
    # sosfiltfilt applies along axis 0 by default
    # Pad for edge effects? sosfiltfilt handles edges reasonably well with padtype='odd' output default
    filtered = sosfiltfilt(sos, data, axis=0)
    return filtered
