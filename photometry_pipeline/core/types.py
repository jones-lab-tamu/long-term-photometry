from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class SessionStats:
    """
    Contains session-level baseline quantities computed in Pass 1.
    Values should be interpreted based on 'method_used'.
    """
    # Method A: raw percentiles per ROI
    f0_values: Dict[str, float] = field(default_factory=dict)
    
    # Method B: Global fit parameters per ROI
    # Stored as {roi: {'a': float, 'b': float}} 
    global_fit_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    method_used: str = ""

@dataclass
class SessionTimeMetadata:
    """Minimal time semantics for CT/ZT alignment."""
    session_id: str = ""
    session_start_iso: str = "" # ISO-8601
    chunk_index: int = 0
    zt0_iso: str = "" # ISO-8601 of lights-on
    zt_offset_hours: float = float('nan')
    notes: str = ""

@dataclass
class Chunk:
    """
    Internal representation of a single photometry acquisition chunk.
    Enforces uniform time grid starting at 0.
    """
    chunk_id: int
    source_file: str
    format: str # 'rwd' | 'npm'
    
    # Time axis: Uniform grid, starting at 0.0
    time_sec: np.ndarray
    
    # Data arrays: Shape (T, N_regions)
    # MUST be aligned to time_sec
    uv_raw: np.ndarray
    sig_raw: np.ndarray
    
    # Filtered arrays (Populated in Pass 2 Preprocessing)
    uv_filt: Optional[np.ndarray] = None
    sig_filt: Optional[np.ndarray] = None
    
    # Fit arrays (Populated in Pass 2 Regression)
    # applied to RAW data: uv_fit = a(t)*uv_raw + b(t)
    uv_fit: Optional[np.ndarray] = None
    delta_f: Optional[np.ndarray] = None # sig_raw - uv_fit
    dff: Optional[np.ndarray] = None # 100 * delta_f / F0
    
    fs_hz: float = 0.0
    channel_names: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def validate(self, tolerance_frac: Optional[float] = None):
        """Strict validation of the chunk contract."""
        T = len(self.time_sec)
        
        # Grid uniformity check
        dt = np.diff(self.time_sec)
        if len(dt) > 0:
            expected_dt = 1.0 / self.fs_hz
            
            # Policy: use provided fraction or default strict 1%
            tol_frac = tolerance_frac if tolerance_frac is not None else 0.01
            tol_val = expected_dt * tol_frac
            
            # Check for grid violations
            bad_indices = np.where(np.abs(dt - expected_dt) > tol_val)[0]
            if len(bad_indices) > 0:
                # Stats for the error message
                min_dt = np.min(dt)
                max_dt = np.max(dt)
                first_bad = bad_indices[0]
                bad_val = dt[first_bad]
                fraction_bad = len(bad_indices) / len(dt)
                
                raise ValueError(
                    f"Uniform grid violation: {len(bad_indices)} intervals ({fraction_bad:.2%}) outside tolerance.\n"
                    f"Expected dt={expected_dt:.5f}s (+/-{tol_val:.5f}s, frac={tol_frac:.2f}).\n"
                    f"Range: [{min_dt:.5f}, {max_dt:.5f}].\n"
                    f"First violation at index {first_bad}: dt={bad_val:.5f}s."
                )

        if self.uv_raw.shape[0] != T:
            raise ValueError(f"UV Raw shape mismatch: {self.uv_raw.shape} vs Time {T}")
        if self.sig_raw.shape[0] != T:
            raise ValueError(f"Sig Raw shape mismatch: {self.sig_raw.shape} vs Time {T}")
        
        if self.uv_filt is not None and self.uv_filt.shape[0] != T:
             raise ValueError("UV Filt shape mismatch")
