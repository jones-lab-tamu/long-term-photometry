"""
Generate synthetic multi-day photometry data (RWD format) with biophysical fidelity.

This script produces:
1.  A root output directory containing subdirectories for each chunk (RWD format).
2.  Within each chunk: 'fluorescence.csv' containing Time(s), UV, and Signal channel data.
3.  Audit artifacts including:
    - 'params.json': All generation parameters.
    - 'qc_metrics.json': Per-chunk QC stats including ZT and circadian phase.
    - 'sanity_check.png': Diagnostic plot verifying signal components and motion injection.
    - 'sanity_zoom_truth.npz': Raw arrays (TRUTH vectors) for a zoomed validation window.
    - 'tonic_compressed.png': Long-timescale circadian structure visualization.
    - 'phasic_stacked_24h.png': Hourly transient density visualization.

MODEL COMPONENTS:
- Tonic Baseline: Circadian oscillation for Signal (peak=Day), Flat for UV.
- Neural Transients: Frac. fluorescence N(t) modulated by circadian phase (Rate higher in Day).
- Bleaching: Exponential decay (optional).
- Artifacts:
    - Shared Slow Drift (band-limited noise).
    - Motion Artifacts (Sparse, mixed additive/multiplicative, probabilistic polarity).
- Noise: Independent Gaussian noise per channel.

TRUTH vs DATA:
The script saves internal 'TRUTH' vectors (e.g., motion_driver, effective baseline) to
sanity_zoom_truth.npz to allow strict auditing of the noisy 'DATA' outputs.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import shutil
from scipy.signal import butter, filtfilt

# ============================================================
# SECTION: ARGUMENT PARSING
# ============================================================
def parse_args():
    """Parse command line arguments for the synthetic data generator."""
    parser = argparse.ArgumentParser(description="Generate synthetic multi-day photometry data (RWD format).")
    
    # Filesystem Args
    parser.add_argument('--out_dir', required=True, help="Root output directory path.")  # Output location
    parser.add_argument('--overwrite', action='store_true', help="Overwrite output directory if exists")  # Safety flag
    
    # Schedule & Chunking Args
    parser.add_argument('--days', type=float, default=3.0, help="Duration in days")          # Total duration
    parser.add_argument('--fs_hz', type=int, choices=[20, 40], default=40, help="Sampling rate (Hz)")  # Sample rate
    parser.add_argument('--start_datetime', type=str, default="2025_01_01-00_00_00", help="Start timestamp YYYY_MM_DD-HH_MM_SS")  # T=0 reference
    
    # Dimensions
    parser.add_argument('--n_rois', type=int, default=4, help="Number of ROIs (regions) to generate")  # Channel count
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")       # RNG Seed
    
    # ZT Mapping Args
    parser.add_argument('--zt_start_hour', type=float, default=0.0, help="ZT at dataset start t=0 (hours)") # ZT Offset
    parser.add_argument('--zt_peak_hour', type=float, default=6.0, help="ZT hour where circadian driver peaks (0-12)") # Circadian Phase
    
    # Signal Model - Circadian Baseline
    parser.add_argument('--circadian_baseline_amp_sig', type=float, default=10.0, help="Amplitude of circadian baseline Oscillation (Signal channel)") # Tonic amplitude
    
    # Signal Model - Transients & SNR
    parser.add_argument('--transient_base_rate_hz', type=float, default=0.2, help="Mean rate of calcium transients (Hz)") # Base event rate
    parser.add_argument('--transient_rate_mod_depth', type=float, default=0.8, help="Circadian modulation depth for rate (0-1)") # Rate modulation
    parser.add_argument('--target_transient_peak_snr', type=float, default=8.0, help="Target median Peak/Noise ratio (Signal channel)") # Neural SNR
    
    # Signal Model - Noise
    parser.add_argument('--noise_sig_sd', type=float, default=1.0, help="Sigma of Gaussian noise on Signal channel") # Noise level Sig
    parser.add_argument('--noise_uv_sd', type=float, default=0.5, help="Sigma of Gaussian noise on UV channel")     # Noise level UV
    
    # Signal Model - Artifacts
    parser.add_argument('--motion_events_per_chunk_mean', type=float, default=4.0, help="Mean motion artifacts per chunk (Poisson mean)") # Event density
    parser.add_argument('--target_motion_peak_snr', type=float, default=6.0, help="Target Motion Dip SNR (magnitude relative to noise)")  # Motion SNR
    parser.add_argument('--motion_k_mult', type=float, default=0.02, help="Motion gain for multiplicative term")                           # Multiplicative scaling
    parser.add_argument('--slow_drift_amp', type=float, default=1.5, help="Shared slow drift amplitude")                                   # Drift magnitude
    parser.add_argument('--motion_negative_prob', type=float, default=0.70, help="Probability of negative motion dip")                     # Polarity bias

    # Plotting
    parser.add_argument('--phasic_zt_day_index', type=int, default=1, help="Which ZT-day to extract for stack plot validation") # Stack plot target

    # Misc
    parser.add_argument('--include_metadata_line', action='store_true', help="Add JSON metadata header line to CSVs") # Metadata header
    parser.add_argument('--bleach', action='store_true', help="Enable bleaching trend")                               # Bleaching flag
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 <= args.motion_negative_prob <= 1.0):
        parser.error("motion_negative_prob must be between 0.0 and 1.0")
        
    return args

# ============================================================
# SECTION: KERNEL GENERATORS
# ============================================================
def generate_variable_transient(fs_hz, tau_rise, tau_decay):
    """
    Generate a normalized biophysical calcium transient kernel.
    
    Kernel Form: (1 - exp(-t/tau_rise)) * exp(-t/tau_decay)
    Normalized such that peak amplitude is 1.0.
    """
    length_sec = 6.0 * tau_decay
    t = np.arange(0, length_sec, 1.0/fs_hz)
    k = (1 - np.exp(-t/tau_rise)) * np.exp(-t/tau_decay)
    k_max = np.max(k)
    if k_max > 0:
        k = k / k_max 
    return k

def generate_motion_kernel(fs_hz, duration_sec=2.0):
    """
    Generate a raw motion dip kernel (triangular/asymmetric).
    
    Parameters:
        fs_hz: Sampling rate.
        duration_sec: Total duration of the event.
    Returns:
        t: Time vector.
        k: Kernel vector (raw shape, sign/polarity applied later).
    """
    # Returns a negative shape by default, but we will handle sign in loop
    t = np.arange(0, duration_sec, 1.0/fs_hz)
    center = duration_sec / 3.0 
    k = np.zeros_like(t)
    # Fall phase
    mask_fall = t <= center
    k[mask_fall] = -1.0 * (t[mask_fall] / center)
    # Recovery phase
    mask_rec = t > center
    k[mask_rec] = -1.0 * (1.0 - (t[mask_rec]-center)/(duration_sec-center))
    return t, k

def generate_slow_drift(n_samples, fs_hz, amp_scale=1.0):
    """
    Generate shared slow drift artifact using band-limited noise.
    
    Band: 0.01 - 0.05 Hz noise (very low frequency).
    Padding is used to avoid filter edge artifacts.
    Scaled to range [0, 1] * amp_scale.
    """
    nyq = 0.5 * fs_hz
    low = 0.01 / nyq
    high = 0.05 / nyq
    b, a = butter(2, [low, high], btype='band')
    noise = np.random.normal(0, 1.0, size=n_samples + 2000) 
    drift = filtfilt(b, a, noise)
    # Remove padding to get valid central segment
    drift = drift[1000:-1000]
    if len(drift) != n_samples:
        drift = drift[:n_samples]
    # Normalize range and scale
    d_range = np.max(drift) - np.min(drift)
    if d_range > 0:
        drift = drift / d_range * amp_scale
    return drift

def get_zt_vals(t_abs_sec, zt_start_hour):
    """
    Calculate ZT (Zeitgeber Time) from absolute experiment seconds.
    
    Returns:
        zt: Current ZT hour (0-24).
        zt_day_idx: Integer count of ZT days elapsed.
    """
    hours_abs = t_abs_sec / 3600.0
    zt = (zt_start_hour + hours_abs) % 24.0
    zt_day_idx = int(np.floor((zt_start_hour + hours_abs) / 24.0))
    return zt, zt_day_idx

# ============================================================
# SECTION: MAIN EXECUTION
# ============================================================
def main():
    args = parse_args()
    
    # Validate critical argument
    assert 0.0 <= args.motion_negative_prob <= 1.0, "motion_negative_prob must be in [0,1]"
    
    # 1. Setup RNG
    np.random.seed(args.seed)  # Global seed
    rng = np.random.default_rng(args.seed)  # Generator instance (used for newer numpy methods)
    
    # Handle Output Directory
    if os.path.exists(args.out_dir):
        if args.overwrite:
            shutil.rmtree(args.out_dir)  # Clean start
        else:
            print(f"Error: Output directory {args.out_dir} exists. Use --overwrite.")
            sys.exit(1)
    
    os.makedirs(args.out_dir)
    
    # Parse Start Time
    try:
        start_dt = datetime.strptime(args.start_datetime, "%Y_%m_%d-%H_%M_%S")
    except ValueError:
        print("Error: Invalid start_datetime format.")
        sys.exit(1)

    # Schedule Constants (10 min ON, 20 min OFF implied)
    CHUNK_DURATION_SEC = 600      # 10 minutes recording ON
    CHUNK_PERIOD_SEC = 1800       # 30 minutes cycle total (Start-to-Start)
    TOTAL_CHUNKS = int(args.days * 24 * (3600 / CHUNK_PERIOD_SEC))  # Total active chunks
    SAMPLES_PER_CHUNK = int(CHUNK_DURATION_SEC * args.fs_hz)        # N samples per file
    
    # Storage for Sanity Check & Statistics
    sanity_data = {
        'timestamps': [],
        'baseline_sig': [],
        'baseline_uv': [], 
        'transient_counts': [],
        'zt_hour': []
    }
    
    qc_metrics = [] # List to store per-chunk QC metadata
    
    # Zoom Window Selection Logic for Audit
    # Target: A chunk near the ZT peak (Subjective Day) on the second full day
    target_zoom_sec = (1.0 * 24 + args.zt_peak_hour - args.zt_start_hour) * 3600.0
    if target_zoom_sec < 0: target_zoom_sec += 24*3600
    target_day_abs_sec = 24 * 3600
    target_zoom_sec += target_day_abs_sec
    
    target_zoom_chunk_idx_approx = int(round(target_zoom_sec / CHUNK_PERIOD_SEC))
    best_zoom_score = -9999.0
    zoomed_chunk_info = None

    print(f"Generating {TOTAL_CHUNKS} chunks over {args.days} days...")
    print(f"ZT Start: {args.zt_start_hour}, Peak: {args.zt_peak_hour}")
    print(f"Motion Polarity: p_neg={args.motion_negative_prob}")
    
    # Organizing chunks for plotting later
    chunks_by_zt = {} 
    
    # ============================================================
    # SECTION: GENERATION LOOP
    # ============================================================
    for chunk_idx in range(TOTAL_CHUNKS):
        # Time Calculations
        t_chunk_start_abs = chunk_idx * CHUNK_PERIOD_SEC
        t_chunk_mid_abs = t_chunk_start_abs + CHUNK_DURATION_SEC/2.0
        
        # Real-time folder name
        current_dt = start_dt + timedelta(seconds=t_chunk_start_abs)
        folder_name = current_dt.strftime("%Y_%m_%d-%H_%M_%S")
        
        chunk_dir = os.path.join(args.out_dir, folder_name)
        os.makedirs(chunk_dir)
        
        # within-chunk time vector 0..600s
        time_sec = np.arange(SAMPLES_PER_CHUNK) / args.fs_hz
        
        # ZT Calculations for this chunk
        zt_start_h, zt_day_idx = get_zt_vals(t_chunk_start_abs, args.zt_start_hour)
        zt_mid_h, _ = get_zt_vals(t_chunk_mid_abs, args.zt_start_hour)
        zt_hour_bin = int(np.floor(zt_start_h))
        
        # Circadian Driver Calculation
        # Cosine peak at zt_peak_hour
        circ_phase = 2.0 * np.pi * (zt_mid_h - args.zt_peak_hour) / 24.0
        circ_val = np.cos(circ_phase)  # Range [-1, 1], Peak at +1 (Day)
        circ01 = 0.5 * (1.0 + circ_val) # Range [0, 1] for modulation
        
        chunk_qc = {
            'chunk': folder_name,
            'zt_chunk_start_hours': float(zt_start_h),
            'zt_chunk_mid_hours': float(zt_mid_h),
            'zt_hour_bin': int(zt_hour_bin),
            'zt_day_index': int(zt_day_idx),
            'circ_val': float(circ_val),
            'motion_negative_prob': args.motion_negative_prob
        }
        
        chunk_transient_count = 0
        df_to_save = {}

        # ============================================================
        # SECTION: ROI LOOP
        # ============================================================
        for roi_i in range(args.n_rois):
            # Per-ROI Baseline Randomization
            roi_base_sig = 100.0 + rng.uniform(-5, 5) # Tonic Signal Base
            roi_base_uv = 50.0 + rng.uniform(-2, 2)   # Tonic UV Base
            
            # --- Artifact Generation (Shared per ROI) ---
            # Shared Slow Drift Artifact (TRUTH)
            drift_trace = generate_slow_drift(SAMPLES_PER_CHUNK, args.fs_hz, args.slow_drift_amp)
            
            # Motion Artifact Generation (TRUTH)
            # Sample number of events
            n_motion = rng.poisson(args.motion_events_per_chunk_mean)
            motion_driver = np.zeros(SAMPLES_PER_CHUNK) # The raw motion kernel summation (TRUTH)
            motion_peak_indices = []
            
            for _ in range(n_motion):
                onset_idx = rng.integers(0, SAMPLES_PER_CHUNK - int(args.fs_hz*3))
                mt, mk = generate_motion_kernel(args.fs_hz, rng.uniform(0.5, 2.0))
                
                # Normalize magnitude to 1.0
                scale = max(abs(np.min(mk)), abs(np.max(mk)))
                if scale > 0: mk /= scale
                
                # Apply Polarity (Probabilistic)
                is_neg = rng.uniform() < args.motion_negative_prob
                sign = -1.0 if is_neg else 1.0
                mk = np.abs(mk) * sign
                
                # Add to driver (summing overlap)
                end_idx = min(onset_idx + len(mk), SAMPLES_PER_CHUNK)
                k_trim = mk[:end_idx-onset_idx]
                motion_driver[onset_idx:end_idx] += k_trim
                
                # Track peak index (using abs max for polarity agnostic tracking)
                peak_offset = np.argmax(np.abs(k_trim))
                motion_peak_indices.append(onset_idx + peak_offset)

            # --- Neural Transient Generation (Signal only) ---
            # Rate Modulation (Deterministic Day/Night)
            r_mean = args.transient_base_rate_hz
            depth = args.transient_rate_mod_depth
            r_min = r_mean * (1.0 - depth)
            r_max = r_mean * (1.0 + depth)
            mod_rate = r_min + (r_max - r_min) * circ01 # Higher rate during Day
            mod_rate = max(mod_rate, 0.0)
            
            # Bleach trend (Linear approximation or exponential)
            if args.bleach:
               bleach = -10.0 * (1.0 - np.exp(-t_chunk_mid_abs / (86400*5)))
            else:
               bleach = 0.0
            
            # F0 Baseline (Tonic)
            # Signal: Modulated by Circadian rhythm + Bleach
            sig_baseline_val = roi_base_sig + args.circadian_baseline_amp_sig * circ_val + bleach
            f0_sig = np.full(SAMPLES_PER_CHUNK, sig_baseline_val) # TRUTH Baseline
            
            # Neural Trace Construction (TRUTH fractional N(t))
            neural_trace = np.zeros(SAMPLES_PER_CHUNK)
            n_events = rng.poisson(mod_rate * CHUNK_DURATION_SEC)
            calcium_onsets = []
            
            # Calcium Amplitude Target
            target_peak_height = args.target_transient_peak_snr * args.noise_sig_sd
            base_amp_n = target_peak_height / max(1.0, sig_baseline_val) # Convert to dF/F
            
            if n_events > 0:
                event_times = np.sort(rng.uniform(0, CHUNK_DURATION_SEC, size=n_events))
                valid_times = []
                last_t = -999.0
                # Refractory period filter
                for t in event_times:
                    if t - last_t >= 0.8:
                        valid_times.append(t)
                        last_t = t
                
                # Inject events
                for t_onset in valid_times:
                    k = generate_variable_transient(args.fs_hz, 0.1, rng.uniform(0.8, 1.2))
                    
                    # Amp mod: Higher amplitude during day (up to 30% boost)
                    amp_mod = (1.0 + 0.3 * circ01)
                    med_amp = base_amp_n * amp_mod
                    amp_val = rng.lognormal(mean=np.log(med_amp), sigma=0.4)
                    
                    start_idx = int(t_onset * args.fs_hz)
                    end_idx = min(start_idx + len(k), SAMPLES_PER_CHUNK)
                    if start_idx < SAMPLES_PER_CHUNK:
                        k_trim = k[:end_idx-start_idx]
                        neural_trace[start_idx:end_idx] += (k_trim * amp_val)
                        calcium_onsets.append(start_idx)
            
            # Save stats for ROI 0 only
            if roi_i == 0: chunk_transient_count = len(calcium_onsets)

            # --- Applied Motion Calculations ---
            # target_motion_peak_snr sets Magnitude (Amplitude) relative to noise
            target_motion_amp = args.target_motion_peak_snr * args.noise_sig_sd
            
            # UV Baseline (Flat + Bleach only, NO Circadian oscillation)
            f0_uv_scalar = roi_base_uv + bleach
            f0_uv = np.full(SAMPLES_PER_CHUNK, f0_uv_scalar) # TRUTH UV Baseline
            
            # UV Motion Terms
            # Coefficients (ensure non-negative scaling factors for addition)
            k_add_uv = target_motion_amp - (f0_uv_scalar * args.motion_k_mult)
            if k_add_uv < 0: k_add_uv = 0
            
            # UV Total Injected Motion (TRUTH) = Drift + Multiplicative + Additive
            mot_uv_app = drift_trace + (args.motion_k_mult * f0_uv * motion_driver) + (k_add_uv * motion_driver)
            
            # Signal Motion Terms
            k_add_sig = target_motion_amp - (sig_baseline_val * args.motion_k_mult)
            if k_add_sig < 0: k_add_sig = 0
            
            # Signal Total Injected Motion (TRUTH) = Drift + Multiplicative + Additive
            mot_sig_app = drift_trace + (args.motion_k_mult * f0_sig * motion_driver) + (k_add_sig * motion_driver)
            
            # --- Final Synthesis (DATA) ---
            # UV Final = F0 + AppliedMotion + GaussNoise
            uv_final = f0_uv + mot_uv_app + rng.normal(0, args.noise_uv_sd, size=SAMPLES_PER_CHUNK)
            
            # Signal Final = F0 * (1 + N(t)) * (1 + k*M(t)) + AdditiveM(t) + Drift + Noise
            # Note: mot_sig_app only captured additive/drift parts fully, multiplicative is entangled with neural.
            motion_mult_factor_sig = (1.0 + args.motion_k_mult * motion_driver)
            # The core fluorescence term with neural signal and multiplicative motion
            sig_fluor = f0_sig * (1.0 + neural_trace) * motion_mult_factor_sig + (k_add_sig * motion_driver) + drift_trace
            sig_final = sig_fluor + rng.normal(0, args.noise_sig_sd, size=SAMPLES_PER_CHUNK)
            
            df_to_save[f"Region{roi_i}-410"] = uv_final
            df_to_save[f"Region{roi_i}-470"] = sig_final
            
            # ============================================================
            # SECTION: ZOOM AUDIT LOGIC (ROI 0)
            # ============================================================
            if roi_i == 0:
                # Store for plotting later
                if zt_day_idx not in chunks_by_zt: chunks_by_zt[zt_day_idx] = {}
                if zt_hour_bin not in chunks_by_zt[zt_day_idx]: chunks_by_zt[zt_day_idx][zt_hour_bin] = []
                chunks_by_zt[zt_day_idx][zt_hour_bin].append({
                    'uv': uv_final, 'sig': sig_final, 'chunk_idx': chunk_idx
                })
                
                # Score this chunk for being the 'Sanity Zoom' exemplar
                for p_idx in motion_peak_indices:
                     # Find distance to nearest calcium event (to ensure clean window)
                     dists = [abs(p_idx - c) for c in calcium_onsets]
                     min_d = min(dists) if dists else 99999
                     
                     # Must be at least 2 seconds away from any transient
                     if min_d > 2.0 * args.fs_hz:
                         dist_target = abs(chunk_idx - target_zoom_chunk_idx_approx)
                         score = 1000 - dist_target - (0.1 * p_idx)
                         
                         if score > best_zoom_score:
                             best_zoom_score = score
                             t_c = p_idx / args.fs_hz
                             t_s = max(0, t_c - 30)
                             t_e = min(CHUNK_DURATION_SEC, t_c + 30)
                             if t_e - t_s < 60: t_s = max(0, t_e - 60)
                             
                             # Assertion: Mixed Polarity Safe
                             # Calculate the motion term at the peak
                             term_uv = (args.motion_k_mult * f0_uv[p_idx] * motion_driver[p_idx]) + (k_add_uv * motion_driver[p_idx])
                             
                             # Check consistency with driver sign
                             s = float(np.sign(motion_driver[p_idx]))
                             assert s != 0.0, "Motion driver sign is zero at peak index"
                             assert float(np.sign(term_uv)) == s, f"Injected motion term sign mismatch: driver {s}, term {term_uv}"
                             assert abs(term_uv) > (0.1 * args.target_motion_peak_snr * args.noise_sig_sd), f"Injected motion term too small: {term_uv}"

                             t_abs_zoom = t_chunk_start_abs + time_sec
                             zt_hours_vec = (args.zt_start_hour + t_abs_zoom/3600.0) % 24.0
                             
                             # Save TRUTH vectors for strict auditing
                             np.savez(os.path.join(args.out_dir, "sanity_zoom_truth.npz"),
                                      motion_driver=motion_driver,          # TRUTH: Raw kernel sum
                                      motion_uv_applied=mot_uv_app,         # TRUTH: Total motion in UV
                                      motion_sig_applied=mot_sig_app,       # TRUTH: Total motion in Signal
                                      uv_raw=uv_final, sig_raw=sig_final,   # DATA: Noisy output
                                      f0_uv=f0_uv, f0_sig=f0_sig,           # TRUTH: Underlying baselines
                                      time=time_sec, t_range=[t_s, t_e],    # Time context
                                      zt_hours_vec=zt_hours_vec,
                                      chunk_name=folder_name, chunk_idx=chunk_idx,
                                      zt_chunk_start_hours=zt_start_h, zt_chunk_mid_hours=zt_mid_h,
                                      zt_hour_bin=zt_hour_bin, zt_day_index=zt_day_idx)
                             zoomed_chunk_info = True

        # ============================================================
        # SECTION: FILE WRITING
        # ============================================================
        df = pd.DataFrame(df_to_save)
        df.insert(0, 'Time(s)', time_sec)
        out_f = os.path.join(chunk_dir, "fluorescence.csv")
        
        # Write metadata header if requested
        if args.include_metadata_line:
             with open(out_f, 'w') as f:
                 f.write(json.dumps({"description": "synthetic", "seed": args.seed}) + "\n")
             df.to_csv(out_f, mode='a', index=False)
        else:
             df.to_csv(out_f, index=False)
        
        # Accumulate stats
        sanity_data['timestamps'].append(current_dt)
        sanity_data['baseline_sig'].append(np.median(df["Region0-470"])) # Median Baseline (ROI 0)
        sanity_data['baseline_uv'].append(np.median(df["Region0-410"]))  # Median Baseline (ROI 0)
        sanity_data['transient_counts'].append(chunk_transient_count)    # Event Counts (ROI 0)
        sanity_data['zt_hour'].append(zt_mid_h)
        qc_metrics.append(chunk_qc)

    # ============================================================
    # SECTION: VALIDATION ARTIFACTS
    # ============================================================
    print("Generating validation artifacts...")
    
    # Save parameters
    with open(os.path.join(args.out_dir, "params.json"), 'w') as f: json.dump(vars(args), f, indent=2)
    # Save QC stats
    with open(os.path.join(args.out_dir, "qc_metrics.json"), 'w') as f: json.dump(qc_metrics, f, indent=2)
    
    # OUTPUT 1: Tonic Compressed Plot
    # Shows long-timescale circadian structure by concatenating 24h of data
    print("Plotting Tonic Compressed...")
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    zt_days = sorted(chunks_by_zt.keys())
    all_uv = []
    all_sig = []
    max_day = max(zt_days) if zt_days else 0
    current_x = 0
    x_ticks = []
    x_tick_labels = []
    
    # Concatenate grouped by ZT day and hour
    for d in range(max_day + 1):
        for h in range(24):
            if d in chunks_by_zt and h in chunks_by_zt[d]:
                chunks = sorted(chunks_by_zt[d][h], key=lambda x: x['chunk_idx'])
                for c in chunks:
                    all_uv.append(c['uv'])
                    all_sig.append(c['sig'])
            # Add tick info
            if d in chunks_by_zt and h in chunks_by_zt[d]:
                n_pts = len(chunks_by_zt[d][h]) * SAMPLES_PER_CHUNK
                x_ticks.append(current_x + n_pts/2)
                x_tick_labels.append(f"{h}")
                current_x += n_pts
                ax1.axvline(x=current_x, color='gray', linestyle=':', alpha=0.5)

    if all_uv:
        cat_uv = np.concatenate(all_uv)
        cat_sig = np.concatenate(all_sig)
        step = 10 # Downsample for display speed
        ax1.plot(np.arange(0, len(cat_uv), step), cat_uv[::step], color='#9400D3', alpha=0.6, label='UV')
        ax1.plot(np.arange(0, len(cat_sig), step), cat_sig[::step], color='#32CD32', alpha=0.6, label='Sig')
        
    ax1.set_xticks(x_ticks[::2]) 
    ax1.set_xticklabels(x_tick_labels[::2])
    ax1.set_xlabel("ZT Hour")
    ax1.set_title("Tonic Compressed (24h ZT cycle)")
    ax1.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "tonic_compressed.png"))
    plt.close(fig1)

    # OUTPUT 2: Phasic Stacked 24h Plot
    # Shows hourly transient density for a single day
    print("Plotting Phasic Stacked...")
    fig2, axes2 = plt.subplots(12, 2, figsize=(12, 24), sharex=True, sharey=True)
    target_day = args.phasic_zt_day_index
    day_data = chunks_by_zt.get(target_day, {})
    
    # Grid: 12 rows, 2 cols (Day vs Night)
    for h in range(24):
        col = 0 if h < 12 else 1  # Left=Day(0-11), Right=Night(12-23)
        row = h if h < 12 else h - 12
        ax = axes2[row, col]
        if h in day_data and len(day_data[h]) > 0:
            c_data = sorted(day_data[h], key=lambda x: x['chunk_idx'])[0]
            t_vec = np.arange(SAMPLES_PER_CHUNK) / args.fs_hz
            ax.plot(t_vec, c_data['uv'] - np.median(c_data['uv']) - 50, color='#9400D3', lw=0.5) 
            ax.plot(t_vec, c_data['sig'] - np.median(c_data['sig']) + 50, color='#32CD32', lw=0.5)
            ax.text(0.02, 0.9, f"ZT {h}", transform=ax.transAxes, fontsize=8)
        else:
            ax.text(0.5, 0.5, "Missing", transform=ax.transAxes, ha='center')
        if row == 11: ax.set_xlabel("Time (s)")
        if col == 0: ax.set_ylabel("dF (Stacked)")
    plt.suptitle(f"Phasic Stacked 24h (ZT-Day {target_day})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "phasic_stacked_24h.png"))
    plt.close(fig2)

    # OUTPUT 3: Summary QC
    # ZT0-11 (Day) vs ZT12-23 (Night) comparisions
    day_chunks = [m for m in qc_metrics if m['zt_day_index'] == 1]
    if not day_chunks: day_chunks = qc_metrics 
    zt0_11_base = [sanity_data['baseline_sig'][i] for i, m in enumerate(qc_metrics) if 0 <= m['zt_hour_bin'] <= 11]
    zt12_23_base = [sanity_data['baseline_sig'][i] for i, m in enumerate(qc_metrics) if 12 <= m['zt_hour_bin'] <= 23]
    zt0_11_rate = [sanity_data['transient_counts'][i] for i, m in enumerate(qc_metrics) if 0 <= m['zt_hour_bin'] <= 11]
    zt12_23_rate = [sanity_data['transient_counts'][i] for i, m in enumerate(qc_metrics) if 12 <= m['zt_hour_bin'] <= 23]
    
    # Check PASS/FAIL criteria (Day > Night for Baseline and Rate)
    base_pass = np.mean(zt0_11_base) > np.mean(zt12_23_base) if (zt0_11_base and zt12_23_base) else False
    rate_pass = np.mean(zt0_11_rate) > np.mean(zt12_23_rate) if (zt0_11_rate and zt12_23_rate) else False
    
    qc_summary = {
        'mean_baseline_zt0_11': float(np.mean(zt0_11_base)) if zt0_11_base else 0.0,
        'mean_baseline_zt12_23': float(np.mean(zt12_23_base)) if zt12_23_base else 0.0,
        'baseline_check_pass': bool(base_pass),
        'mean_rate_zt0_11': float(np.mean(zt0_11_rate)) if zt0_11_rate else 0.0,
        'mean_rate_zt12_23': float(np.mean(zt12_23_rate)) if zt12_23_rate else 0.0,
        'rate_check_pass': bool(rate_pass),
        'zt_configured_peak': args.zt_peak_hour
    }
    with open(os.path.join(args.out_dir, "summary_qc.json"), 'w') as f:
        json.dump(qc_summary, f, indent=2)

    # OUTPUT 4: Sanity Check Audit Plot
    # Shows detailed overlay of TRUTH vs DATA for the zoom window
    fig3, axes3 = plt.subplots(4, 1, figsize=(12, 16))
    truth_path = os.path.join(args.out_dir, "sanity_zoom_truth.npz")
    if os.path.exists(truth_path):
        zdata = np.load(truth_path)
        t_s, t_e = zdata['t_range']
        mask = (zdata['time'] >= t_s) & (zdata['time'] <= t_e)
        t_z = zdata['time'][mask]
        uv_r = zdata['uv_raw'][mask]
        sig_r = zdata['sig_raw'][mask]
        uv_inj = zdata['f0_uv'][mask] + zdata['motion_uv_applied'][mask] # TRUTH (Baseline + Motion)
        sig_inj = zdata['f0_sig'][mask] + zdata['motion_sig_applied'][mask] # TRUTH (Baseline + Motion)
        
        # Panel 1: UV Audit
        axes3[0].plot(t_z, uv_r, color='#9400D3', alpha=0.7, label='UV Data')
        axes3[0].plot(t_z, uv_inj, color='red', linestyle='--', label='UV Truth (F0+Motion)')
        axes3[0].set_title(f"Audit UV Raw ({zdata['chunk_name']})")
        axes3[0].legend()
        
        # Panel 2: Signal Audit
        axes3[1].plot(t_z, sig_r, color='#32CD32', alpha=0.7, label='Sig Data')
        axes3[1].plot(t_z, sig_inj, color='red', linestyle='--', label='Sig Truth (F0+Motion)')
        axes3[1].set_title("Audit Sig Raw")
        axes3[1].legend()
        
        # Panel 3: Detrended comparison (Data - Median)
        axes3[2].plot(t_z, uv_r - np.median(uv_r), color='#9400D3', label='UV Detrend')
        axes3[2].plot(t_z, sig_r - np.median(sig_r), color='#32CD32', label='Sig Detrend')
        axes3[2].set_title("Detrended")
        axes3[2].legend()
        
    # Panel 4: Circadian Baseline over time
    axes3[3].plot(sanity_data['timestamps'], sanity_data['baseline_sig'], color='#32CD32')
    axes3[3].set_title("Circadian Baseline")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "sanity_check.png"))
    plt.close(fig3)
    
    # 4. Self-Checks
    print("SUCCESS: Dataset generated with ZT mapping.")

if __name__ == "__main__":
    main()
