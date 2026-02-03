
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
    parser.add_argument('--phasic-mode', choices=['low_phasic', 'high_phasic', 'very_high_phasic'], default='high_phasic')
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

def tonic_envelope_plateau_transition(t_sec, period_sec, phase_offset_sec=0.0,
                                      low_plateau_frac=0.25, rise_frac=0.25, 
                                      high_plateau_frac=0.25, fall_frac=0.25):
    """
    Generates a periodic plateau+transition envelope in [0, 1].
    Shape: Low -> Rise -> High -> Fall.
    """
    # Validate fractions
    total_frac = low_plateau_frac + rise_frac + high_plateau_frac + fall_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total_frac}")
    
    if rise_frac <= 0 or fall_frac <= 0:
        raise ValueError("rise_frac and fall_frac must be > 0")

    # Compute Phase [0, 1)
    phi = ((t_sec + phase_offset_sec) % period_sec) / period_sec
    
    # Boundaries
    a = low_plateau_frac
    b = a + rise_frac
    c = b + high_plateau_frac
    
    # Piecewise
    # 1. Low Plateau: [0, a) -> 0.0
    # 2. Rise: [a, b) -> smoothstep((phi-a)/rise)
    # 3. High Plateau: [b, c) -> 1.0
    # 4. Fall: [c, 1.0) -> 1.0 - smoothstep((phi-c)/fall)
    
    y = np.zeros_like(phi)
    
    # Low mask implied (initial 0.0)
    
    # Rise mask
    mask_rise = (phi >= a) & (phi < b)
    if np.any(mask_rise):
        y[mask_rise] = smoothstep_cos((phi[mask_rise] - a) / rise_frac)
    
    # High mask
    mask_high = (phi >= b) & (phi < c)
    if np.any(mask_high):
        y[mask_high] = 1.0
    
    # Fall mask
    mask_fall = (phi >= c)
    if np.any(mask_fall):
        y[mask_fall] = 1.0 - smoothstep_cos((phi[mask_fall] - c) / fall_frac)
    
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

def generate_events(t_global_hours, phase, args, rng):
    """
    Generate sparse phasic events with refractory period and waveform logic.
    """
    n = len(t_global_hours)
    signal_add = np.zeros(n)
    
    # Circadian Rate Modulation (Hard Switch Day/Night)
    # Day is [start_ct, end_ct)
    phase_ct = (phase + (t_global_hours % 24.0)) % 24.0
    # Actually, t_global_hours is absolute. CT = (t_hours - phase) % 24 ??? 
    # Let's align with the Phase Definition: phase is the peak offset.
    # But for simple Day/Night, let's use CT = (t_global_hours) % 24.0 relative to "lights on" if phase is 0?
    # User Spec: "Define is_day = (ct >= day_start_ct) and (ct < day_end_ct)"
    # We should interpret 'phase' as 'CT0 offset' or similar, but the user code used it as peak offset.
    # To keep it robust/simple for testing:
    # We will assume t_global_hours IS CT (or ZT) for the sake of this binary switch, 
    # OR we treat 'phase' as the shift. 
    # Let's stick to the previous 'tonic' logic: `(t - phase)`.
    # So CT = (t_global_hours - phase) % 24.0 
    
    ct = compute_ct_hours(t_global_hours, phase, mode=args.phasic_ct_mode)
    is_day = (ct >= args.phasic_day_start_ct) & (ct < args.phasic_day_end_ct)
    
    rate = np.zeros(n)
    
    if args.phasic_day_high:
        rate[is_day] = args.phasic_peak_rate_hz
        rate[~is_day] = args.phasic_base_rate_hz
    else:
        rate[is_day] = args.phasic_base_rate_hz
        rate[~is_day] = args.phasic_peak_rate_hz
        
    # Poisson process
    dt_step = 1.0 / args.fs_hz
    p = rate * dt_step
    
    r_uni = rng.uniform(size=n)
    candidate_indices = np.where(r_uni < p)[0]
    
    # Demo visibility guardrail
    if args.phasic_min_events_per_chunk > 0:
        if len(candidate_indices) < args.phasic_min_events_per_chunk:
            needed = args.phasic_min_events_per_chunk - len(candidate_indices)
            extras = rng.choice(n, size=needed, replace=False)
            candidate_indices = np.concatenate([candidate_indices, extras])
            candidate_indices.sort()
            
    if len(candidate_indices) == 0:
        return signal_add
    
    # Apply Refractory Period
    refractory_samples = int(args.phasic_refractory_sec * args.fs_hz)
    valid_indices = []
    last_idx = -refractory_samples - 1
    
    for idx in candidate_indices:
        if idx - last_idx >= refractory_samples:
            valid_indices.append(idx)
            last_idx = idx
            
    if not valid_indices:
        return signal_add
        
    if not valid_indices:
        return signal_add
        
    tau = args.phasic_decay_sec
    rise = args.phasic_rise_sec
    
    dur_s = args.phasic_dur_mult * tau
    dur_samples = int(dur_s * args.fs_hz)
    if dur_samples < 5: dur_samples = 5
    
    t_local = np.arange(dur_samples) / args.fs_hz
    # Pre-compute waveform shape (alpha function)
    # A * (1 - exp(-t/rise)) * exp(-t/tau)
    # Peak is roughly at t ~ rise * ln(tau/rise) ? 
    # Just compute it.
    wave_template = (1.0 - np.exp(-t_local / rise)) * np.exp(-t_local / tau)
    # Normalize template to max 1.0 so 'amp' is the peak
    if np.max(wave_template) > 0:
        wave_template /= np.max(wave_template)
        
    for idx in valid_indices:
        amp = rng.lognormal(mean=args.phasic_amp_logmean, sigma=args.phasic_amp_logsd)
        
        # Explicit tuning scale
        amp *= args.phasic_amp_scale
        
        # Clip to end of buffer
        this_dur = dur_samples
        if idx + this_dur > n:
            this_dur = n - idx
            
        signal_add[idx : idx + this_dur] += wave_template[:this_dur] * amp
        
    return signal_add

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
            # Phase is in hours (peak-aligned in old code). 
            # We treat phase as a time shift for the envelope.
            # Using -phase*3600 shifts the pattern origin.
            phase_offset = -phase * 3600.0
            
            tonic_env = tonic_envelope_plateau_transition(t_global_sec, period_sec, phase_offset_sec=phase_offset)
            
            # Application: Preserve 20.0 scale factor, but now envelope is [0, 1].
            # This makes the tonic component strictly positive (0 to 20*Amp)
            tonic_val = tonic_env * (20.0 * args.tonic_amplitude)
            
            # Phasic
            phasic_val = generate_events(t_global_hours, phase, args, rng)
            
            # Noise (Explicit Components)
            noise_sig = noise_shared + rng.normal(0, args.noise_sig_std, n_samples_chunk)
            noise_uv = (noise_shared * args.noise_shared_uv_scale) + rng.normal(0, args.noise_uv_std, n_samples_chunk)
            
            # Composition
            # Signal: Base + ROI + Scale * (Tonic + Phasic - Drift + Motion + Wobble + Noise + Daily)
            # wobble_sig already includes gain/offset dynamics
            comp_sig = tonic_val + phasic_val - drift_au + motion_base + wobble_sig + noise_sig + daily_var
            sig = (args.raw_baseline_sig + roi_sig_offsets[i]) + args.raw_scale * comp_sig
            
            # UV: Base + ROI + Scale * (Motion*Coupling - Drift*Scale + Wobble + Noise + Daily)
            # wobble_uv is iso-scaled
            comp_uv = (motion_base * args.artifact_coupling) - (drift_au * args.uv_drift_scale) + wobble_uv + noise_uv + daily_var
            uv = (args.raw_baseline_uv + roi_uv_offsets[i]) + args.raw_scale * comp_uv
            
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

    # Optional Debug Plot
    if args.debug_tonic_plot:
        try:
            import matplotlib.pyplot as plt
            # Generate 72h sample
            t_d = np.linspace(0, 72*3600, 1000)
            y_d = tonic_envelope_plateau_transition(t_d, 24*3600, phase_offset_sec=0.0)
            
            plt.figure(figsize=(10, 4))
            plt.plot(t_d/3600.0, y_d, label='Tonic Envelope (Phase=0)')
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
