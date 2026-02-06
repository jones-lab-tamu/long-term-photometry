
import argparse
import os
import sys
import numpy as np
import pandas as pd
import datetime

# Ensure we can import config
sys.path.append(os.getcwd())
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from photometry_pipeline.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic photometry dataset (RWD/NPM).")
    
    # Required
    parser.add_argument('--out', required=True, help="Output root directory")
    parser.add_argument('--format', required=True, choices=['rwd', 'npm'], help="Output format")
    parser.add_argument('--config', required=True, help="Path to config YAML")
    
    # Scheduling
    parser.add_argument('--total-days', type=float, default=3.0)
    parser.add_argument('--recording-duration-min', type=float, default=10.0)
    parser.add_argument('--recordings-per-hour', type=int, default=2)
    parser.add_argument('--fs-hz', type=float, default=None, help="Sampling rate. If None, uses config.target_fs_hz")
    parser.add_argument('--n-rois', type=int, default=2)
    parser.add_argument('--start-iso', type=str, default="2025-01-01T00:00:00")
    parser.add_argument('--seed', type=int, default=42)
    
    # Baseline / Scaling
    # Deprecated fallback
    parser.add_argument('--raw-baseline', type=float, default=None, help="Deprecated: use --raw-baseline-sig/uv")
    
    parser.add_argument('--raw-baseline-sig', type=float, default=None, help="Base Signal level (AU). Default 250")
    parser.add_argument('--raw-baseline-uv', type=float, default=None, help="Base UV level (AU). Default 235")
    
    parser.add_argument('--raw-scale', type=float, default=1.0, help="Global scaling factor")
    
    # ROI-Specific Baselines (Additive Offsets in AU)
    parser.add_argument('--roi-baseline-offset-range', nargs=2, type=float, default=[5.0, 20.0], help="Range for per-ROI baseline offset (Signal)")
    parser.add_argument('--roi-uv-baseline-offset-range', nargs=2, type=float, default=[3.0, 15.0], help="Range for per-ROI baseline offset (UV)")
    
    # Tonic
    parser.add_argument('--tonic-amplitude', type=float, default=1.0)
    parser.add_argument('--tonic-amplitude-mode', choices=['high', 'low', 'custom'], default='high')
    parser.add_argument('--tonic-phase-ct', type=float, default=6.0)
    parser.add_argument('--tonic-phase-jitter-hr', type=float, default=1.0)
    parser.add_argument('--debug-tonic-plot', type=str, default=None, help="Path to save debug plot of tonic envelope (optional)")
    
    # Phasic
    parser.add_argument('--phasic-mode', choices=['low_phasic', 'high_phasic', 'very_high_phasic', 'phase_locked_to_tonic'], default='high_phasic')
    parser.add_argument('--phasic-base-rate-hz', type=float, default=0.0005) # Sparse default
    parser.add_argument('--phasic-peak-rate-hz', type=float, default=0.01)  # Sparse default
    parser.add_argument('--phasic-vonmises-kappa', type=float, default=4.0)
    parser.add_argument('--phasic-sine-depth', type=float, default=0.7)
    parser.add_argument('--phasic-refractory-sec', type=float, default=0.75)
    parser.add_argument('--phasic-min-events-per-chunk', type=int, default=0, help="For demos: force minimum events")
    
    # Day/Night Phasic Rate Logic
    parser.add_argument('--phasic-ct-mode', choices=['absolute', 'phase_aligned'], default='absolute', help="CT definition for day/night switch")
    parser.add_argument('--phasic-day-start-ct', type=float, default=0.0)
    parser.add_argument('--phasic-day-end-ct', type=float, default=12.0)
    parser.add_argument('--phasic-day-high', action='store_true', default=True, help="Peak rate during Day, Base rate during Night")
    parser.add_argument('--no-phasic-day-high', action='store_false', dest='phasic_day_high')
    
    # Phasic Waveform Params
    parser.add_argument('--phasic-decay-sec', type=float, default=1.2)
    parser.add_argument('--phasic-rise-sec', type=float, default=0.12)
    parser.add_argument('--phasic-amp-logmean', type=float, default=2.5) # Strong default
    parser.add_argument('--phasic-amp-logsd', type=float, default=0.35)
    parser.add_argument('--phasic-amp-scale', type=float, default=1.0)
    parser.add_argument('--phasic-dur-mult', type=float, default=6.0)

    # Artifacts
    parser.add_argument('--artifact-scale', type=float, default=1.0, help="Scaling factor applied only to motion and drift artifacts")
    
    # Motion
    parser.add_argument('--artifact-enable-motion', action='store_true', default=None)
    parser.add_argument('--no-artifact-enable-motion', action='store_true', default=False)
    
    # Drift
    parser.add_argument('--artifact-enable-drift', action='store_true', default=None)
    parser.add_argument('--no-artifact-enable-drift', action='store_true', default=False)
    
    parser.add_argument('--artifact-coupling', type=float, default=1.0)
    parser.add_argument('--artifact-motion-rate-per-day', type=float, default=20.0)
    parser.add_argument('--artifact-motion-min-per-day', type=float, default=0.0)
    parser.add_argument('--artifact-motion-amp-range', nargs=2, type=float, default=[8.0, 30.0], help="Amplitude range (AU)")
    # parser.add_argument('--artifact-motion-tau-sec-range', ...) # Removed as requested
    parser.add_argument('--artifact-motion-decay-sec', type=float, default=2.50, help="Motion decay (tau)")
    parser.add_argument('--artifact-motion-rise-sec', type=float, default=0.30, help="Motion rise time")
    parser.add_argument('--artifact-motion-dur-mult', type=float, default=6.0)
    
    # Polarity
    parser.add_argument('--artifact-motion-neg-prob', type=float, default=0.85, help="Prob motion is negative")
    parser.add_argument('--artifact-motion-same-sign', action='store_true', default=False, help="If set, force ALL motion artifacts negative. Default: False (Probabilistic).")
    parser.add_argument('--no-artifact-motion-same-sign', action='store_false', dest='artifact_motion_same_sign')
    
    parser.add_argument('--artifact-motion-refractory-sec', type=float, default=2.0)
    
    parser.add_argument('--artifact-drift-amp', type=float, default=20.0, help="Amplitude (AU) for slow drift/bleaching")
    parser.add_argument('--artifact-bleach-tau-days', type=float, default=6.0)
    parser.add_argument('--uv-drift-scale', type=float, default=1.0)
    
    # Correlated Baseline Wobble
    parser.add_argument('--shared-wobble-enable', action='store_true', default=False)
    parser.add_argument('--shared-wobble-amp', type=float, default=2.0)
    parser.add_argument('--shared-wobble-tau-sec', type=float, default=60.0)
    parser.add_argument('--shared-wobble-iso-scale', type=float, default=1.0)
    parser.add_argument('--shared-wobble-iso-lag-sec', type=float, default=0.0)
    
    # Shared Nuisance (Time-Varying Coupling)
    parser.add_argument('--shared-wobble-gain-enable', action='store_true', default=False)
    parser.add_argument('--shared-wobble-gain-mean', type=float, default=1.0)
    parser.add_argument('--shared-wobble-gain-sd', type=float, default=0.15)
    parser.add_argument('--shared-wobble-gain-tau-sec', type=float, default=120.0)
    
    parser.add_argument('--shared-wobble-offset-enable', action='store_true', default=False)
    parser.add_argument('--shared-wobble-offset-amp', type=float, default=1.0)
    parser.add_argument('--shared-wobble-offset-tau-sec', type=float, default=180.0)
    
    # Noise
    parser.add_argument('--noise-sig-std', type=float, default=0.8)
    parser.add_argument('--noise-uv-std', type=float, default=0.4)
    parser.add_argument('--noise-shared-std', type=float, default=0.4)
    parser.add_argument('--noise-shared-uv-scale', type=float, default=1.0)
    
    # Day Variation
    parser.add_argument('--day-variation-enable', action='store_true', default=False)
    parser.add_argument('--day-variation-au', type=float, default=0.0, help="Std dev of random daily offset in AU (no hidden multipliers)")
    
    # Strict Iso-True Model (V4 Refactor)
    parser.add_argument('--iso-slope-true', type=float, default=1.0)
    parser.add_argument('--iso-intercept-true', type=float, default=0.0)
    parser.add_argument('--debug-verify-iso-structure', action='store_true', help="Verify correlation and slope of generated signals")
    parser.add_argument('--debug-dump-phase-lock', action='store_true', help="Dump phase-lock debug CSV for Region0")
    
    # Validation Refinement (Soft Coupling)
    parser.add_argument('--phasic-phase-lock-alpha', type=float, default=0.7, help="Coupling strength [0,1]")
    parser.add_argument('--phasic-min-rate-frac', type=float, default=0.15, help="Minimum rate fraction of base_rate")
    parser.add_argument('--phasic-ct-gating-mode', choices=['none', 'hard', 'soft'], default='soft', help="Day/Night gating mode")
    parser.add_argument('--phasic-ct-soft-floor', type=float, default=0.25, help="Floor for soft CT gating (night rate multiplier)")

    # Event Count Control (Chunk-Level)
    parser.add_argument('--phasic-events-per-10min-mean', type=float, default=10.0, help="Target mean events per 10min chunk (Day)")
    parser.add_argument('--phasic-events-per-10min-sd', type=float, default=3.0, help="SD of target means (variability)")
    parser.add_argument('--phasic-events-per-10min-min', type=int, default=0, help="Min events per chunk")
    parser.add_argument('--phasic-events-per-10min-max', type=int, default=25, help="Max events per chunk")

    args = parser.parse_args()
    
    # Validate
    if args.artifact_motion_neg_prob < 0.0 or args.artifact_motion_neg_prob > 1.0:
        raise ValueError("--artifact-motion-neg-prob must be between 0 and 1")

    # Enforce Best-Biology Constraints
    if args.shared_wobble_gain_enable or args.shared_wobble_offset_enable:
         if abs(args.shared_wobble_iso_lag_sec) > 1e-6:
             print(f"WARNING: shared-wobble-iso-lag-sec {args.shared_wobble_iso_lag_sec} ignored/zeroed due to gain/offset dynamics.")
             args.shared_wobble_iso_lag_sec = 0.0

    # Resolve Artifact Flags
    if args.artifact_enable_motion is None:
        args.artifact_enable_motion = True
    if args.no_artifact_enable_motion:
        args.artifact_enable_motion = False
        
    if args.artifact_enable_drift is None:
        args.artifact_enable_drift = True
    if args.no_artifact_enable_drift:
        args.artifact_enable_drift = False

    # Resolve Baselines
    if args.raw_baseline is not None and args.raw_baseline_sig is None and args.raw_baseline_uv is None:
        args.raw_baseline_sig = args.raw_baseline
        args.raw_baseline_uv = args.raw_baseline
    else:
        if args.raw_baseline_sig is None: args.raw_baseline_sig = 250.0
        if args.raw_baseline_uv is None: args.raw_baseline_uv = 235.0

    return args

    return args

def smoothstep_cos(x):
    """
    Cosine smoothstep: y = 0.5 - 0.5*cos(pi*x) for x in [0,1]
    Clamped to [0,1].
    """
    x_clip = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x_clip)

# Envelope Definition (Single Source of Truth)
TONIC_LOW_FRAC = 0.25
TONIC_RISE_FRAC = 0.25
TONIC_HIGH_FRAC = 0.25
TONIC_FALL_FRAC = 0.25

def get_peak_center_phi01():
    """Derive peak center from envelope fractions."""
    # Low(Start) -> Rise -> High(Start) -> High(End) -> Fall
    # Peak Center is midpoint of High Plateau
    start_high = TONIC_LOW_FRAC + TONIC_RISE_FRAC
    end_high = start_high + TONIC_HIGH_FRAC
    return (start_high + end_high) / 2.0

def tonic_envelope_from_phase_01(phi_01):
    """
    Generates a periodic plateau+transition envelope in [0, 1] given phase [0, 1).
    Shape: Low -> Rise -> High -> Fall.
    Uses strictly locked module-level constants.
    """
    # Use Locked Constants
    low_plateau_frac = TONIC_LOW_FRAC
    rise_frac = TONIC_RISE_FRAC
    high_plateau_frac = TONIC_HIGH_FRAC
    fall_frac = TONIC_FALL_FRAC

    # Validate fractions
    total_frac = low_plateau_frac + rise_frac + high_plateau_frac + fall_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total_frac}")
    
    # Boundaries
    a = low_plateau_frac
    b = a + rise_frac
    c = b + high_plateau_frac
    
    y = np.zeros_like(phi_01)
    
    # Low mask implied (initial 0.0)
    
    # Rise mask
    mask_rise = (phi_01 >= a) & (phi_01 < b)
    if np.any(mask_rise):
        y[mask_rise] = smoothstep_cos((phi_01[mask_rise] - a) / rise_frac)
    
    # High mask
    mask_high = (phi_01 >= b) & (phi_01 < c)
    if np.any(mask_high):
        y[mask_high] = 1.0
    
    # Fall mask
    mask_fall = (phi_01 >= c)
    if np.any(mask_fall):
        y[mask_fall] = 1.0 - smoothstep_cos((phi_01[mask_fall] - c) / fall_frac)
    
    return np.clip(y, 0.0, 1.0)

def compute_ct_hours(t_global_hours, phase_hours, mode="absolute"):
    """
    Compute Circadian Time (CT) in hours [0, 24).
    mode='absolute': t_global_hours % 24 (assuming t=0 is CT0/Lights On)
    mode='phase_aligned': (t_global_hours - phase_hours) % 24
    """
    if mode == 'phase_aligned':
        return (t_global_hours - phase_hours) % 24.0
    else:
        return t_global_hours % 24.0

def generate_events(t_global_hours, phi_rad, phase_hr, args, rng):
    """
    Generate sparse phasic events with refractory period and waveform logic.
    phi_rad: Canonical tonic phase in radians [-pi, pi], where 0 is Tonic Peak (Upstate).
             Computed external to this function to ensure consistency.
    phase_hr: shift for CT day/night logic.
    """
    n = len(t_global_hours)
    signal_add = np.zeros(n)
    
    # Chunk-Level Poisson Sampling
    
    # Restore CT calculation
    ct = compute_ct_hours(t_global_hours, phase_hr, mode=args.phasic_ct_mode)
    is_day = (ct >= args.phasic_day_start_ct) & (ct < args.phasic_day_end_ct)

    # 1. Determine Target N for this chunk
    base_mu = args.phasic_events_per_10min_mean
    if args.phasic_events_per_10min_sd > 0:
        mu_chunk = rng.normal(base_mu, args.phasic_events_per_10min_sd)
    else:
        mu_chunk = base_mu
    mu_chunk = max(0.0, mu_chunk)
    
    # Scale mu by CT (Chunk Center approx)
    ct_center = ct[n//2]
    is_day_center = (ct_center >= args.phasic_day_start_ct) & (ct_center < args.phasic_day_end_ct)
    
    ct_mult = 1.0
    if args.phasic_ct_gating_mode == 'hard':
        ct_mult = 1.0 if is_day_center else 0.0
    elif args.phasic_ct_gating_mode == 'soft':
        ct_mult = 1.0 if is_day_center else args.phasic_ct_soft_floor
        
    mu_chunk *= ct_mult
    
    # 2. Draw N
    n_events_target = rng.poisson(mu_chunk)
    n_events_target = int(np.clip(n_events_target, args.phasic_events_per_10min_min, args.phasic_events_per_10min_max))
    
    # 3. Compute Phase/Time Weights
    # Calculate Gating Factor based on CT (per sample)
    if args.phasic_ct_gating_mode == 'none':
        gating_factor = np.ones(n)
    elif args.phasic_ct_gating_mode == 'hard':
        gating_factor = is_day.astype(float)
    else: # soft
        gating_factor = is_day.astype(float) + (~is_day).astype(float) * args.phasic_ct_soft_floor
    
    weights = np.ones(n)
    if args.phasic_mode == 'phase_locked_to_tonic':
        kappa = args.phasic_vonmises_kappa
        m = np.exp(kappa * np.cos(phi_rad))
        m_mean = np.mean(m) # over this chunk
        m_norm = m / m_mean
        
        alpha = args.phasic_phase_lock_alpha
        # w = (1-a) + a*m_norm
        # Ensure floor
        w_raw = (1.0 - alpha) + alpha * m_norm
        weights = np.maximum(w_raw, args.phasic_min_rate_frac)
        
    elif args.phasic_day_high:
         # Implicitly handled by gating_factor usually, but if not phase locked, 
         # we might want higher density in day.
         # For simplicity, if not phase locked, uniform weights (Poisson process) 
         # modulated by gating factor is enough.
         pass
         
    # Combine Phase Weights with CT Gating
    final_weights = weights * gating_factor
    
    # Zero out weights near edges to prevent wrap-around partial events if desired, 
    # but let's just allow it for now or rely on padding.
    
    w_sum = np.sum(final_weights)
    valid_indices = []
    dropped_minsep = 0
    
    if n_events_target > 0 and w_sum > 0:
        p = final_weights / w_sum
        
        # Sample with replacement then prune? Or without replacement?
        # Without replacement is safer for strict collision avoidance, 
        # but with small N relative to n_samples, it doesn't matter much.
        # numpy.random.choice is slow with large p. 
        # But n_samples ~ 24000. It's fine.
        candidates = rng.choice(n, size=n_events_target, replace=False, p=p)
        candidates.sort()
        
        # Enforce Min Separation
        refractory_samples = int(args.phasic_refractory_sec * args.fs_hz)
        last_idx = -refractory_samples - 1
        
        for idx in candidates:
            if idx - last_idx >= refractory_samples:
                valid_indices.append(idx)
                last_idx = idx
            else:
                dropped_minsep += 1
                
    # Generate Waveforms
    tau = args.phasic_decay_sec
    rise = args.phasic_rise_sec
    dur_s = args.phasic_dur_mult * tau
    dur_samples = int(dur_s * args.fs_hz)
    if dur_samples < 5: dur_samples = 5
    
    t_local_wave = np.arange(dur_samples) / args.fs_hz
    wave_template = (1.0 - np.exp(-t_local_wave / rise)) * np.exp(-t_local_wave / tau)
    if np.max(wave_template) > 0:
        wave_template /= np.max(wave_template)
        
    for idx in valid_indices:
        amp = rng.lognormal(mean=args.phasic_amp_logmean, sigma=args.phasic_amp_logsd)
        amp *= args.phasic_amp_scale
        
        this_dur = dur_samples
        if idx + this_dur > n:
            this_dur = n - idx
            
        signal_add[idx : idx + this_dur] += wave_template[:this_dur] * amp
        
    # Return rate vector mainly for debug/visualization (continuous prob profile)
    # Scale it so mean matches base_rate approx? Or just normalize to max 1?
    # Let's return the unnormalized weights as 'rate' proxy.
    rate_proxy = final_weights 
    
    stats = {
        'mu_target': mu_chunk,
        'n_drawn': n_events_target,
        'n_accepted': len(valid_indices),
        'n_dropped': dropped_minsep,
        'ct_center': ct_center,
        'is_day_center': is_day_center,
        'w_sum': w_sum,
        'tonic_w_frac': np.sum(final_weights[np.abs(phi_rad) <= np.pi/2]) / (w_sum + 1e-9)
    }

    return signal_add, valid_indices, rate_proxy, is_day, stats

def generate_motion_artifacts(n_samples, args, rng):
    """
    Generate shared artifacts: Impulse + Recovery.
    Uses Poisson count per chunk. Polarity biased negative.
    """
    total_sec = n_samples / args.fs_hz
    lam = args.artifact_motion_rate_per_day * (total_sec / 86400.0)
    
    if args.artifact_motion_min_per_day > 0:
        min_lam = args.artifact_motion_min_per_day * (total_sec / 86400.0)
        base_count = rng.poisson(lam)
        req = int(np.ceil(min_lam))
        n_events = max(base_count, req) if min_lam > 0.001 else base_count
    else:
        n_events = rng.poisson(lam)
    
    if n_events == 0:
        return np.zeros(n_samples)
        
    cand_indices = rng.choice(n_samples, size=n_events, replace=True)
    cand_indices.sort()
    
    # Refractory logic
    refractory_samples = int(args.artifact_motion_refractory_sec * args.fs_hz)
    indices = []
    last_idx = -refractory_samples - 1
    for idx in cand_indices:
        if idx - last_idx >= refractory_samples:
            indices.append(idx)
            last_idx = idx
    
    if not indices:
        return np.zeros(n_samples)
    
    artifact_sig = np.zeros(n_samples)
    
    for idx in indices:
        amp = rng.uniform(args.artifact_motion_amp_range[0], args.artifact_motion_amp_range[1])
        
        if args.artifact_motion_same_sign:
            # Deterministic mode: Always negative (downward)
            sign = -1.0
        else:
            # Probabilistic mode
            if rng.uniform() < args.artifact_motion_neg_prob:
                sign = -1.0
            else:
                sign = 1.0
                
                sign = 1.0
                
        amp *= sign
        
        # Kinetics
        tau = args.artifact_motion_decay_sec
        # Legacy/Range support removed in favor of explicit decay-sec, 
        # but if we wanted to support 'tau-sec-range' we would grab it here.
        # User requested explicit precedence or simplified model.
        # We adhere to explicit arg.
        
        rise = args.artifact_motion_rise_sec
        
        dur_samples = int(args.artifact_motion_dur_mult * tau * args.fs_hz)
        if dur_samples < 5: dur_samples = 5
        
        if idx + dur_samples > n_samples:
            dur_samples = n_samples - idx
            
        t_local = np.arange(dur_samples) / args.fs_hz
        
        # Normalized Alpha Function
        # wave = (1 - exp(-t/rise)) * exp(-t/tau)
        # Normalize to peak=1.0, then scale by amp.
        template = (1.0 - np.exp(-t_local / rise)) * np.exp(-t_local / tau)
        if np.max(np.abs(template)) > 0:
            template /= np.max(np.abs(template))
        
        artifact_sig[idx : idx+dur_samples] += template * amp
        
    return artifact_sig

def generate_ar1_series(n, args, rng, mean, sd, tau_sec, clamp_min=None):
    """
    Generate AR(1) process with defined stationary stats.
    x[t] = alpha * (x[t-1] - mean) + eps + mean
    """
    if tau_sec <= 0: return np.full(n, mean)
    
    dt = 1.0 / args.fs_hz
    alpha = np.exp(-dt / tau_sec)
    
    # Variance of AR1: Var = sigma_eps^2 / (1 - alpha^2)
    # We want Var = sd^2
    # sigma_eps = sd * sqrt(1 - alpha^2)
    scale = sd * np.sqrt(1 - alpha*alpha)
    
    eps = rng.normal(0, 1, n)
    w = np.zeros(n)
    val = 0.0 # start at mean 0 relative
    
    for i in range(n):
        val = alpha * val + scale * eps[i]
        w[i] = val
        
    res = w + mean
    if clamp_min is not None:
        res = np.maximum(res, clamp_min)
        
    return res

def generate_ar1_wobble(n, args, rng):
    if not args.shared_wobble_enable or args.shared_wobble_amp <= 0:
        return np.zeros(n)
        
    return generate_ar1_series(n, args, rng, 0.0, args.shared_wobble_amp, args.shared_wobble_tau_sec)

def main():
    args = parse_args()
    
    try:
        cfg = Config.from_yaml(args.config)
    except (OSError, ValueError, TypeError) as e:
        raise RuntimeError(f"Error loading config: {e}")

    # 1. Config Consistency
    if args.fs_hz is None:
        args.fs_hz = float(cfg.target_fs_hz)

    expected_chunk_dur = float(cfg.chunk_duration_sec)
    actual_dur = args.recording_duration_min * 60.0
    if abs(actual_dur - expected_chunk_dur) > (1.0 / args.fs_hz):
        raise ValueError(f"recording duration mismatch: args {actual_dur}s vs config {expected_chunk_dur}s")

    rng = np.random.default_rng(args.seed)
    
    os.makedirs(args.out, exist_ok=True)
    
    chunk_interval_sec = 3600.0 / args.recordings_per_hour
    start_dt = datetime.datetime.fromisoformat(args.start_iso)
    
    total_hours = args.total_days * 24.0
    total_intervals = int(np.floor(total_hours * args.recordings_per_hour))
    
    n_samples_chunk = int(expected_chunk_dur * args.fs_hz)
    
    # ROI Parameters
    roi_phases = []
    roi_sig_offsets = []
    roi_uv_offsets = []
    
    for i in range(args.n_rois):
        jitter = rng.uniform(-args.tonic_phase_jitter_hr, args.tonic_phase_jitter_hr)
        p = args.tonic_phase_ct + jitter
        roi_phases.append(p % 24.0) 
        
        off_sig = rng.uniform(args.roi_baseline_offset_range[0], args.roi_baseline_offset_range[1])
        if rng.random() > 0.5: off_sig *= -1
        roi_sig_offsets.append(off_sig)

        off_uv = rng.uniform(args.roi_uv_baseline_offset_range[0], args.roi_uv_baseline_offset_range[1])
        if rng.random() > 0.5: off_uv *= -1
        roi_uv_offsets.append(off_uv)
        
    print(f"Generating {total_intervals} chunks over {args.total_days} days...")
    
    # Verification accumulators
    verify_stats = {
        'n_up': [0]*args.n_rois,
        'n_down': [0]*args.n_rois,
        'n_up_day': [0]*args.n_rois,
        'n_down_day': [0]*args.n_rois,
        'n_day': [0]*args.n_rois,
        'n_night': [0]*args.n_rois,
        'total_events': [0]*args.n_rois,
        'accepted_counts_per_chunk': [[] for _ in range(args.n_rois)]
    }
    
    # Pre-calc phase map if needed
    for k in range(total_intervals):
        chunk_start_dt = start_dt + datetime.timedelta(seconds=k * chunk_interval_sec)
        
        chunk_time_offset_sec = k * chunk_interval_sec
        t_local = np.arange(n_samples_chunk) / args.fs_hz
        t_global_sec = chunk_time_offset_sec + t_local
        t_global_hours = t_global_sec / 3600.0
        
        daily_var = 0.0
        if args.day_variation_enable:
             daily_var = rng.normal(0, args.day_variation_au)
             
        data = {}
        if args.format == 'rwd':
             data[cfg.rwd_time_col] = t_local
        
        uv_data_all = []
        sig_data_all = []
        
        # Shared Artifacts
        if args.artifact_enable_motion:
            motion_base = generate_motion_artifacts(n_samples_chunk, args, rng)
        else:
            motion_base = np.zeros(n_samples_chunk)
        
        motion_base *= args.artifact_scale
            
        # Accumulating Bleach Drift
        t_days = t_global_sec / 86400.0
        bleach_frac = 1.0 - np.exp(-t_days / args.artifact_bleach_tau_days)
        drift_au = args.artifact_drift_amp * bleach_frac
        
        if not args.artifact_enable_drift:
            drift_au = np.zeros_like(drift_au)
            
        drift_au *= args.artifact_scale
        
        # Refined Noise Model
        noise_shared = rng.normal(0, args.noise_shared_std, n_samples_chunk)
        
        # Shared Wobble (Base is usually considered 'UV-like' in shape, applied to both)
        # Using AR1 helper
        wobble_base = generate_ar1_wobble(n_samples_chunk, args, rng)
        


        # Gain and Offset Dynamics
        # Model: 
        #   UV = iso_scale * wobble_base
        #   Sig = gain(t) * wobble_base + offset(t)
        #   gain(t) ~ AR1(mean, sd)
        #   offset(t) ~ AR1(0, amp)
        #   Synchronous: w_base is same for both. gain/offset modulate signal relative to UV.
        
        if args.shared_wobble_gain_enable:
             gain_vec = generate_ar1_series(n_samples_chunk, args, rng, 
                                            args.shared_wobble_gain_mean, 
                                            args.shared_wobble_gain_sd, 
                                            args.shared_wobble_gain_tau_sec, 
                                            clamp_min=0.1)
        else:
             gain_vec = np.ones(n_samples_chunk)
             
        if args.shared_wobble_offset_enable:
             offset_vec = generate_ar1_series(n_samples_chunk, args, rng, 
                                              0.0, 
                                              args.shared_wobble_offset_amp, 
                                              args.shared_wobble_offset_tau_sec)
        else:
             offset_vec = np.zeros(n_samples_chunk)

        # Apply Wobble (No Lag)
        # If gain/offset enabled, we strictly ignore iso-lag-sec as requested ("NO phase lag").
        # Even if iso-lag-sec is non-zero in args, we assume best-biology mode overrides it 
        # or the user knows not to set it. We use the same base.
        
        wobble_uv = wobble_base * args.shared_wobble_iso_scale
        wobble_sig = (wobble_base * gain_vec) + offset_vec
        
        for i in range(args.n_rois):
            phase = roi_phases[i]
            
            # Tonic (Plateau+Transition)
            # Replaces: tonic_norm = np.cos(2 * np.pi * (t_global_hours - phase) / 24.0)
            # tonic_val = tonic_norm * (20.0 * args.tonic_amplitude)
            
            period_sec = 24.0 * 3600.0
            # Phase is in hours (peak-aligned). 
            # Derive peak location from envelope structure (Single Source of Truth)
            peak_center_phi01 = get_peak_center_phi01()
            peak_center_hr = peak_center_phi01 * 24.0
            
            # We want 'phase' (args.tonic_phase_ct + jitter) to correspond to the Peak Center.
            # phase*3600 + offset = peak_center_phi01 * (24*3600)
            phase_offset = (peak_center_hr - phase) * 3600.0
            
            # Compute canonical phases
            # 1. phi_01 [0, 1): Linear phase of the cycle. 0 = Start of Low Plateau.
            #    phi =((t + offset) % P) / P
            phi_01 = ((t_global_sec + phase_offset) % period_sec) / period_sec
            
            # 2. phi_rad [-pi, pi]: Centered at Peak (High Plateau Center).
            #    Default High Plateau Center is at phi_01 = 0.625.
            #    We map 0.625 -> 0.
            #    phi_rad = (phi_01 - 0.625) * 2pi, wrapped.
            peak_center_phi01 = 0.625
            phi_rad_unwrapped = (phi_01 - peak_center_phi01) * 2.0 * np.pi
            # Wrap to [-pi, pi]
            phi_rad = (phi_rad_unwrapped + np.pi) % (2.0 * np.pi) - np.pi
            
            # Tonic Envelope (uses phi_01)
            tonic_env = tonic_envelope_from_phase_01(phi_01)
            
            # Application: Preserve 20.0 scale factor, but now envelope is [0, 1].
            # This makes the tonic component strictly positive (0 to 20*Amp)
            tonic_val = tonic_env * (20.0 * args.tonic_amplitude)
            
            # Phasic (uses phi_rad for locking)
            # generate_events now returns is_day too
            # Phasic (uses phi_rad for locking)
            phasic_val, event_indices, rate_vec, is_day_vec, chunk_stats = generate_events(t_global_hours, phi_rad, phase, args, rng)
            
            # Accumulate Verification Stats
            # Accumulate Verification Stats
            # Store (count, is_day_chunk) pair
            chunk_is_day = chunk_stats['is_day_center'] # approximate
            verify_stats['accepted_counts_per_chunk'][i].append( (chunk_stats['n_accepted'], chunk_is_day) )
            
            if args.phasic_mode == 'phase_locked_to_tonic' and len(event_indices) > 0:
                # Look up stats at event times
                phis_e = phi_rad[event_indices]
                is_day_e = is_day_vec[event_indices]
                
                # Upstate: |phi| <= pi/2
                is_up = np.abs(phis_e) <= (np.pi / 2.0)
                is_down = ~is_up
                
                # Metric 1: Unconditional Up/Down
                n_u = np.sum(is_up)
                n_d = np.sum(is_down)
                verify_stats['n_up'][i] += int(n_u)
                verify_stats['n_down'][i] += int(n_d)
                
                # Metric 2: Up/Down within CT Day
                is_up_day = is_up & is_day_e
                is_down_day = is_down & is_day_e
                verify_stats['n_up_day'][i] += int(np.sum(is_up_day))
                verify_stats['n_down_day'][i] += int(np.sum(is_down_day))
                
                # Metric 3: Day/Night
                n_day = np.sum(is_day_e)
                n_night = len(event_indices) - n_day
                verify_stats['n_day'][i] += int(n_day)
                verify_stats['n_night'][i] += int(n_night)
                
                verify_stats['total_events'][i] += len(event_indices)
                
            # Dump Debug CSV for Region0 (if enabled)
            if args.debug_dump_phase_lock and i == 0:
                # Dump chunks to separate log? Or detailed sample log? 
                # Let's keep sample log but add chunk mu to a separate summary file?
                # Actually, appending sample log is fine.
                dump_path = os.path.join(args.out, f"debug_phase_lock_region{i}.csv")
                write_header = not os.path.exists(dump_path)
                
                # Calculate CT here for dump since is_day_vec is bool
                ct_vec = compute_ct_hours(t_global_hours, phase, mode=args.phasic_ct_mode)
                
                df_dbg = pd.DataFrame({
                    'chunk_idx': k,
                    't_global_hr': t_global_hours,
                    'phi_rad': phi_rad,
                    'phi_01': phi_01,
                    'tonic_val': tonic_val,
                    'weights': rate_vec, # renamed
                    'is_upstate': np.abs(phi_rad) <= (np.pi / 2.0),
                    'ct_hours': ct_vec,
                    'is_day': is_day_vec
                })
                # Sample 1/10th or full? Full for debug
                df_dbg.to_csv(dump_path, mode='a', header=write_header, index=False)
                
                # Dump Chunk Summary
                dump_path_summary = os.path.join(args.out, f"debug_phase_lock_summary_region{i}.csv")
                write_header_sum = not os.path.exists(dump_path_summary)
                df_sum = pd.DataFrame([{
                    'chunk_idx': k,
                    'mu_target': chunk_stats['mu_target'],
                    'n_drawn': chunk_stats['n_drawn'],
                    'n_accepted': chunk_stats['n_accepted'],
                    'n_dropped': chunk_stats['n_dropped'],
                    'ct_center': chunk_stats['ct_center'],
                    'tonic_w_frac': chunk_stats['tonic_w_frac']
                }])
                df_sum.to_csv(dump_path_summary, mode='a', header=write_header_sum, index=False)

            
            # --- V4 Strict Signal Construction ---
            # 1. Generate UV First (The "Owner" of Drift/Shared Nuisance)
            # UV = Base + Offset + Scale * (Motion*Coupling - Drift*DriftScale + Wobble + NoiseShared + NoiseUV + Daily)
            comp_uv = (motion_base * args.artifact_coupling) - (drift_au * args.uv_drift_scale) + wobble_uv + noise_shared * args.noise_shared_uv_scale + daily_var
            
            # Add independent UV noise (pre-scaling? Old code added it at `comp_uv` level with raw_scale multiplier implicitly? 
            # Old: comp_uv = ... + noise_uv ...; uv = base + scale * comp_uv.
            # noise_uv was rng.normal(0, args.noise_uv_std).
            # So noise amplitude in final UV is scale * noise_uv_std.
            noise_uv = rng.normal(0, args.noise_uv_std, n_samples_chunk)
            
            uv_raw_no_noise = (args.raw_baseline_uv + roi_uv_offsets[i]) + args.raw_scale * comp_uv
            uv = uv_raw_no_noise + args.raw_scale * noise_uv
            
            # 2. Define "True Iso Fit" from UV
            # iso_true = slope * uv + intercept
            iso_true = args.iso_slope_true * uv + args.iso_intercept_true
            
            # 3. Generate Neural Component (Tonic + Phasic)
            # Neural in AU. Use raw_scale to match the scale of the rest of the signal?
            # Old code: sig = base + scale * (tonic + phasic + ...).
            # So neural component magnitude is scale * (tonic + phasic).
            neural = args.raw_scale * (tonic_val + phasic_val)
            
            # 4. Construct Sig Raw
            # sig_raw = iso_true + neural + sig_noise
            # sig_noise should be independent. 
            # In old code, sig had 'noise_sig' which was 'shared + independent'.
            # Here, 'shared' is already in 'iso_true' (via uv).
            # We add independent sig noise.
            noise_sig_independent = rng.normal(0, args.noise_sig_std, n_samples_chunk)
            sig = iso_true + neural + args.raw_scale * noise_sig_independent
            
            # Guardrails / Debug
            if args.debug_verify_iso_structure:
                # Correlation
                if np.std(uv) > 1e-9 and np.std(sig) > 1e-9:
                    c = np.corrcoef(uv, sig)[0,1]
                    # Polyfit
                    pf = np.polyfit(uv, sig, 1)
                    slope_hat = pf[0]
                    int_hat = pf[1]
                    
                    print(f"ISO_STRUCT ROI=Region{i} corr={c:.4f} slope_hat={slope_hat:.4f} intercept_hat={int_hat:.4f}")
                    
                    if args.artifact_enable_drift and args.uv_drift_scale == 1.0:
                         # Only enforce correlation if signal variance (drift) exceeds noise floor
                         # Noise std is ~0.4 - 0.8. If std(uv) is low, it's noise-dominated.
                         if np.std(uv) > 5.0 and c < 0.3:
                             raise RuntimeError(f"Region{i}: Correlation {c:.4f} too low (expected >= 0.3 with drift) for std(uv)={np.std(uv):.3f}")
                    
                    if abs(args.iso_slope_true - 1.0) < 0.01:
                        # Only enforce slope if signal variance is significant (leverage)
                        if np.std(uv) > 5.0:
                             if not (0.5 <= slope_hat <= 1.5):
                                 raise RuntimeError(f"Region{i}: Slope {slope_hat:.4f} out of bounds [0.5, 1.5] (std_uv={np.std(uv):.3f})")
                else:
                    print(f"ISO_STRUCT ROI=Region{i} SKIP (low variance)")

            # Clamp
            sig = np.maximum(sig, 1.0)
            uv = np.maximum(uv, 1.0)
            
            uv_data_all.append(uv)
            sig_data_all.append(sig)
            if args.format == 'rwd':
                r_name = f"Region{i}"
                if cfg.npm_region_prefix: pass 
                data[f"{r_name}{cfg.sig_suffix}"] = sig
                data[f"{r_name}{cfg.uv_suffix}"] = uv

        if args.format == 'rwd':
            dirname = chunk_start_dt.strftime("%Y_%m_%d-%H_%M_%S") 
            chunk_dir = os.path.join(args.out, dirname)
            os.makedirs(chunk_dir, exist_ok=True)
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(chunk_dir, "fluorescence.csv"), index=False)
            
        elif args.format == 'npm':
            fname = f"photometryData{chunk_start_dt.strftime('%Y-%m-%dT%H_%M_%S')}.csv"
            fpath = os.path.join(args.out, fname)
            
            half_dt = 0.5 / args.fs_hz
            n_total = 2 * n_samples_chunk
            
            frames = np.arange(n_total)
            leds = np.zeros(n_total, dtype=int)
            leds[0::2] = 1 # UV
            leds[1::2] = 2 # Sig
            
            # Timestamps
            t_base = t_local
            t_interleaved = np.zeros(n_total)
            t_interleaved[0::2] = t_base
            t_interleaved[1::2] = t_base + half_dt
            
            cols = {}
            cols[cfg.npm_frame_col] = frames
            cols[cfg.npm_led_col] = leds
            
            t_col_sys = cfg.npm_system_ts_col
            t_col_comp = cfg.npm_computer_ts_col
            
            if cfg.npm_time_axis == 'system_timestamp':
                cols[t_col_sys] = t_interleaved
                cols[t_col_comp] = t_interleaved 
            else:
                cols[t_col_comp] = t_interleaved
                cols[t_col_sys] = t_interleaved
            
            for i in range(args.n_rois):
                col_name = f"{cfg.npm_region_prefix}{i}{cfg.npm_region_suffix}"
                vals = np.zeros(n_total)
                vals[0::2] = uv_data_all[i]
                vals[1::2] = sig_data_all[i]
                cols[col_name] = vals
                
            df_temp = pd.DataFrame(cols)
            df_temp.to_csv(fpath, index=False)
            
    # Verify Phase Locking
    # Verify Phase Locking
    if args.phasic_mode == 'phase_locked_to_tonic':
        print("\n=== Phase-Locked Event Verification (Mixture Model) ===")
        print(f"Params: alpha={args.phasic_phase_lock_alpha}, min_rate_frac={args.phasic_min_rate_frac}, ct_mode={args.phasic_ct_gating_mode}")
        
        print("\n--- Metric 1: Tonic Up/Down (Unconditional) ---")
        print("Region | Up | Down | Ratio | Verdict")
        fail_tonic = False
        for i in range(args.n_rois):
            n_up = verify_stats['n_up'][i]
            n_down = verify_stats['n_down'][i]
            ratio = n_up / max(n_down, 1)
            # Requirement: 1.5 <= ratio <= 6.0
            verdict = "PASS" if (1.5 <= ratio <= 6.0) else "FAIL (Range 1.5-6.0)"
            if not(1.5 <= ratio <= 6.0): fail_tonic = True
            print(f"Region{i} | {n_up:6d} | {n_down:6d} | {ratio:5.2f} | {verdict}")

        print("\n--- Metric 2: Tonic Up/Down (Day Only) ---")
        print("Region | Up(Day) | Down(Day) | Ratio | Verdict")
        for i in range(args.n_rois):
            n_up = verify_stats['n_up_day'][i]
            n_down = verify_stats['n_down_day'][i]
            ratio = n_up / max(n_down, 1)
            print(f"Region{i} | {n_up:8d} | {n_down:10d} | {ratio:5.2f} | N/A")

        print("\n--- Metric 3: CT Day/Night ---")
        print("Region | Day | Night | Ratio | Verdict")
        for i in range(args.n_rois):
             n_d = verify_stats['n_day'][i]
             n_n = verify_stats['n_night'][i]
             ratio = n_d / max(n_n, 1)
             print(f"Region{i} | {n_d:6d} | {n_n:6d} | {ratio:5.2f} | N/A")
             
        print("\n--- Rate Sanity Check ---")
        print("Region | Actual/Hr | Expected/Hr | Ratio")
        expected_hourly = args.phasic_base_rate_hz * 3600.0
        for i in range(args.n_rois):
            actual = verify_stats['total_events'][i] / total_hours
            ratio = actual / expected_hourly
            print(f"Region{i} | {actual:9.1f} | {expected_hourly:11.1f} | {ratio:5.2f}")

        print("\n--- Event Count Stats (Per 10min Chunk) ---")
        print("Region | Day Mean | Night Mean | Global Mean | Day Target | Night Target")
        fail_count = False
        
        for i in range(args.n_rois):
            chunk_data = verify_stats['accepted_counts_per_chunk'][i] # list of (count, is_day)
            if not chunk_data:
                print(f"Region{i} | N/A")
                continue
                
            counts_day = [c for c, is_day in chunk_data if is_day]
            counts_night = [c for c, is_day in chunk_data if not is_day]
            
            mu_day = np.mean(counts_day) if counts_day else 0.0
            mu_night = np.mean(counts_night) if counts_night else 0.0
            mu_global = np.mean([c for c, _ in chunk_data])
            
            # Acceptance Logic
            # Day Target: [8, 12]
            # Night Target: [1, 4]
            # Strict Minimum Chunks requirement
            MIN_CHUNKS = 12
            
            if len(counts_day) < MIN_CHUNKS:
                print(f"Region{i} | Day Chunks={len(counts_day)} | INSUFFICIENT DATA (<{MIN_CHUNKS})")
                fail_count = True
                continue
            
            if len(counts_night) < MIN_CHUNKS:
                print(f"Region{i} | Night Chunks={len(counts_night)} | INSUFFICIENT DATA (<{MIN_CHUNKS})")
                fail_count = True
                continue
                
            day_ok = (8.0 <= mu_day <= 12.0)
            night_ok = (1.0 <= mu_night <= 4.0)
            
            
            status_str = "PASS"
            if not day_ok: status_str = "FAIL (Day Mean)"
            if not night_ok: status_str = "FAIL (Night Mean)"
            if not day_ok and not night_ok: status_str = "FAIL (Both)"
            
            if not day_ok or not night_ok:
                fail_count = True
            
            # Format: R0 | Day=10.1 | Night=2.3 | Global=6.2 | PASS
            print(f"Region{i} | Day={mu_day:5.1f} | Night={mu_night:5.1f} | Global={mu_global:5.1f} | {status_str}")

        print("\n--- Phase Locking Stats (Target: Ratio 1.5-6.0, MinDown >= 1) ---")
        print("Region | Up  | Down | Ratio | Verdict")
        fail_tonic_chk = False
        for i in range(args.n_rois):
            n_up = verify_stats['n_up'][i]
            n_down = verify_stats['n_down'][i]
            if n_down > 0:
                ratio = n_up / n_down
            else:
                ratio = 999.0
            
            pass_ratio = (1.5 <= ratio <= 6.0)
            pass_nonzero = (n_down >= 1)
            
            verdict = "PASS"
            if not pass_ratio: verdict = "FAIL (Ratio)"
            if not pass_nonzero: verdict = "FAIL (Zero Down events)"
            if not pass_ratio and not pass_nonzero: verdict = "FAIL (Both)"
            
            if verdict != "PASS":
                fail_tonic_chk = True
            
            print(f"Region{i} | {n_up:3d} | {n_down:4d} | {ratio:5.2f} | {verdict}")

        print("-" * 60)
        if fail_tonic_chk:
            print("FAILURE: Phase Locking criteria check.")
        if fail_count:
             print("FAILURE: Event Density criteria check.")
             
        if not fail_tonic_chk and not fail_count:
            print(">>> VERIFIED: All Metrics within bounds. Generator Ready.")
        else:
            print(">>> FAILED: Verification failed.")
            # raise RuntimeError("Synthesizer failed validation.") # Soft fail for CLI exploration, but visually obvious.
            
    # Debug Plot / Log
    if args.debug_tonic_plot:
        # Save simple plot
        pass # Only if requested, kept simplified logic out of strict verification block logic above which serves the user requirement.

    # Optional Debug Plot
    if args.debug_tonic_plot:
        try:
            import matplotlib.pyplot as plt
            # Generate 72h sample
            plt.figure(figsize=(10, 4))
            # Test curve
            t_test = np.linspace(0, 24*3600, 1000)
            p_test = (t_test % (24*3600)) / (24*3600)
            y_d = tonic_envelope_from_phase_01(p_test)
            
            plt.plot(t_test/3600.0, y_d, label='Tonic Envelope (Phase=0)')
            plt.xlabel('Time (Hours)')
            plt.ylabel('Envelope [0,1]')
            plt.title('Debug Tonic Waveform')
            plt.grid(True)
            plt.legend()
            plt.savefig(args.debug_tonic_plot)
            plt.close()
            print(f"Saved debug tonic plot to {args.debug_tonic_plot}")
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            print(f"Failed to generate debug plot: {e}")

if __name__ == '__main__':
    main()
