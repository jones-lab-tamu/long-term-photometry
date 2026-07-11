from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

# Resolved (post-alias) dynamic-fit modes Pipeline can dispatch a ROI to.
# "rolling_local_regression" is a Config-level input alias for
# "rolling_filtered_to_raw" (see regression._DYNAMIC_FIT_MODE_ALIASES) and is
# resolved to it before reaching a PerRoiCorrectionSpec, so it is not listed
# here as a distinct dispatch target.
RESOLVED_DYNAMIC_FIT_MODES = (
    "global_linear_regression",
    "robust_global_event_reject",
    "adaptive_event_gated_regression",
    "rolling_filtered_to_raw",
    "rolling_filtered_to_filtered",
)
CORRECTION_STRATEGY_FAMILIES = ("dynamic_fit", "signal_only_f0")

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
    format: str # 'rwd' | 'npm' | 'custom_tabular'
    
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


@dataclass(frozen=True)
class PerRoiCorrectionSpec:
    """One ROI's authoritative, scientist-selected production correction strategy.

    This is the sole per-ROI correction authority Pipeline dispatches from.
    The global Config.dynamic_fit_mode field is never consulted directly by
    dispatch; for legacy/non-Guided runs a uniform map is synthesized from it
    exactly once (see regression.build_uniform_per_roi_correction_map), so the
    two never compete as independent authorities.

    parameter_identity/evidence_identity are opaque identity strings (not yet
    consumed by dispatch) reserved for the later provenance-integration stage.
    """
    roi_id: str
    strategy_family: str  # one of CORRECTION_STRATEGY_FAMILIES
    selected_strategy: str
    dynamic_fit_mode: Optional[str] = None
    parameter_identity: str = ""
    evidence_identity: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.roi_id, str) or not self.roi_id:
            raise ValueError("PerRoiCorrectionSpec requires a non-empty roi_id")
        if self.strategy_family not in CORRECTION_STRATEGY_FAMILIES:
            raise ValueError(f"Unsupported strategy_family: {self.strategy_family!r}")
        if self.strategy_family == "dynamic_fit":
            if self.dynamic_fit_mode not in RESOLVED_DYNAMIC_FIT_MODES:
                raise ValueError(f"Unsupported dynamic_fit_mode: {self.dynamic_fit_mode!r}")
            if self.selected_strategy != self.dynamic_fit_mode:
                raise ValueError(
                    f"selected_strategy ({self.selected_strategy!r}) must equal "
                    f"dynamic_fit_mode ({self.dynamic_fit_mode!r}) when "
                    "strategy_family='dynamic_fit'"
                )
        elif self.strategy_family == "signal_only_f0":
            if self.selected_strategy != "signal_only_f0":
                raise ValueError(
                    f"Unsupported signal_only_f0 selected_strategy: {self.selected_strategy!r}"
                )
            if self.dynamic_fit_mode is not None:
                raise ValueError(
                    "signal_only_f0 entries must have dynamic_fit_mode=None"
                )
