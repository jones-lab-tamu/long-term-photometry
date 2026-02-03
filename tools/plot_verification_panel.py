import argparse
import sys
import os
import subprocess
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
import hashlib
from scipy.stats import linregress, pearsonr
from scipy.signal import convolve
from pathlib import Path

# Safe Import from Pipeline
PIPELINE_AVAILABLE = False
PIPELINE_ERROR = None

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    from photometry_pipeline.core.regression import fit_chunk_dynamic
    from photometry_pipeline.core.types import Chunk
    from photometry_pipeline.config import Config
    PIPELINE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    PIPELINE_AVAILABLE = False
    PIPELINE_ERROR = str(e)

# Command to reproduce this specific verification panel:
# python tools/plot_verification_panel.py --out outputs/verification_panel.png --seed 42

def generate_data(temp_root, tonic_mode, seed):
    """
    Generate a 3-day continuous dataset with specified tonic amplitude.
    """
    out_dir = os.path.join(temp_root, f"data_{tonic_mode}")
    
    # Tonic Params
    if tonic_mode == 'high':
        amp = 1.0
    else:
        amp = 0.2
        
    config_path = os.path.join(temp_root, f"config_{tonic_mode}.yaml")
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write("""
chunk_duration_sec: 3600
target_fs_hz: 20
baseline_method: uv_raw_percentile_session
baseline_percentile: 10
rwd_time_col: TimeStamp
uv_suffix: "-410"
sig_suffix: "-470"
peak_threshold_method: mean_std
window_sec: 20.0
step_sec: 5.0
""")

    # Generator Command
    cmd = [
        sys.executable, "tools/synth_photometry_dataset.py",
        "--out", out_dir,
        "--format", "rwd",
        "--config", config_path,
        "--total-days", "3.0",
        "--recording-duration-min", "60.0",
        "--recordings-per-hour", "1",
        "--fs-hz", "20",
        "--tonic-amplitude", str(amp),
        "--seed", str(seed),
        # Fixed parameters for consistency
        "--phasic-mode", "high_phasic",
        "--phasic-base-rate-hz", "0.05",
        "--phasic-peak-rate-hz", "0.10",
        "--phasic-ct-mode", "absolute",
        "--artifact-enable-motion",
        "--artifact-motion-rate-per-day", "600.0",
        "--artifact-motion-neg-prob", "0.85",
        "--artifact-motion-amp-range", "8.0", "16.0",
        "--artifact-motion-rise-sec", "0.30",
        "--artifact-motion-decay-sec", "2.50",
        "--shared-wobble-enable",
        "--shared-wobble-amp", "2.0",
        "--shared-wobble-tau-sec", "60.0",
        "--shared-wobble-gain-enable",
        "--shared-wobble-gain-tau-sec", "120.0",
        "--shared-wobble-gain-sd", "0.30",
        "--shared-wobble-offset-enable",
        "--shared-wobble-offset-amp", "2.0",
        "--shared-wobble-offset-tau-sec", "180.0"
    ]
    
    print(f"Generating {tonic_mode} data (Amp={amp}, Seed={seed})")
    subprocess.run(cmd, check=True)
    
    # Load and Concatenate
    files = glob.glob(os.path.join(out_dir, "*", "fluorescence.csv"))
    files.sort()
    
    if not files:
        raise RuntimeError("No files generated!")
        
    dfs = []
    current_offset = 0.0
    for f in files:
        d = pd.read_csv(f)
        d['GlobalTime'] = d['TimeStamp'] + current_offset
        dfs.append(d)
        current_offset += 3600.0
        
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

def plot_panel(ax, df, title, x_label, x_lims=None, decimate=1):
    """
    Plot Sig and UV on the given axis.
    """
    if x_lims:
        # 1. Apply Time Mask to FULL dataframe to avoid indexing errors
        t0, t1 = x_lims
        mask = (df['GlobalTime'] >= t0) & (df['GlobalTime'] <= t1)
        d = df[mask]
        
        # 2. Decimate the slice if requested
        if decimate > 1:
            d = d.iloc[::decimate]
            
        t = d['GlobalTime'].values
        s = d['Region0-470'].values
        u = d['Region0-410'].values
        
        # Relative time in minutes
        t_plot = (t - t0) / 60.0
        
        ax.plot(t_plot, s, color='tab:blue', label='Signal (470)', linewidth=1.0, alpha=0.9)
        ax.plot(t_plot, u, color='tab:purple', label='Iso (410)', linewidth=1.0, alpha=0.9)
        ax.set_xlabel("Time (min)")
        ax.set_xlim(0, (t1 - t0)/60.0)
        
        # Robust Auto-Scale for Zoom
        combined = np.concatenate([s, u])
        if len(combined) > 10:
            y_lo = np.percentile(combined, 1.0)
            y_hi = np.percentile(combined, 99.0)
            yr = y_hi - y_lo
            if yr == 0: yr = 1.0
            ax.set_ylim(y_lo - 0.1*yr, y_hi + 0.3*yr) # Extra top headroom for spikes
            
        ax.legend(loc='upper right', fontsize='x-small')

    else:
        # Full View (Hours)
        if decimate > 1:
            d = df.iloc[::decimate]
        else:
            d = df
            
        t = d['GlobalTime'].values
        s = d['Region0-470'].values
        u = d['Region0-410'].values
        
        t_hours = t / 3600.0
        ax.plot(t_hours, s, color='tab:blue', label='Signal (470)', linewidth=0.5, alpha=0.8)
        ax.plot(t_hours, u, color='tab:purple', label='Iso (410)', linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Time (Hours)")
        ax.set_xlim(0, 72)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

def plot_residuals(ax, t_sec, s, u, window_sec=60.0, label_prefix="", return_series=False):
    """
    Plot Global LS Residual vs Dynamic Fit Residual on 'Clean' Data.
    Includes input hashing and visual diagnostics.
    """
    # [A] Input Normalization and Shape Safety
    t_sec = np.asarray(t_sec, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    
    T = len(t_sec)

    # 1. Hashing and Console Diags
    # distinct inputs for hashing: t, s, u
    t_f8 = np.ascontiguousarray(t_sec, dtype=np.float64)
    s_f8 = np.ascontiguousarray(s, dtype=np.float64)
    u_f8 = np.ascontiguousarray(u, dtype=np.float64)
    
    h = hashlib.sha256()
    h.update(t_f8.tobytes())
    h.update(s_f8.tobytes())
    h.update(u_f8.tobytes())
    digest = h.hexdigest()[:8]
    
    s_mean, s_std = (np.mean(s), np.std(s)) if T > 0 else (np.nan, np.nan)
    u_mean, u_std = (np.mean(u), np.std(u)) if T > 0 else (np.nan, np.nan)
    t0_val = t_sec[0] if T > 0 else 0
    t1_val = t_sec[-1] if T > 0 else 0
    
    print(f"RESID_INPUT {label_prefix} T={T} t0={t0_val:.1f} t1={t1_val:.1f} "
          f"s_mean={s_mean:.4f} s_std={s_std:.4f} u_mean={u_mean:.4f} u_std={u_std:.4f} hash={digest}")

    if len(s) != T or len(u) != T:
        msg = f"Shape mismatch: t={T}, s={len(s)}, u={len(u)}"
        ax.text(0.5, 0.5, msg, ha='center', transform=ax.transAxes)
        return None

    # [B] Timebase Safety
    if T < 3:
        ax.text(0.5, 0.5, "Insufficient time samples (N<3)", ha='center', transform=ax.transAxes)
        return None

    dt = np.nanmedian(np.diff(t_sec))
    if np.isnan(dt) or dt <= 0:
        ax.text(0.5, 0.5, "Invalid timebase (dt<=0 or NaN)", ha='center', transform=ax.transAxes)
        return None

    fs = 1.0 / dt
    if fs < 0.5 or fs > 5000:
        ax.text(0.5, 0.5, f"Invalid fs={fs:.1f}Hz (Bounds: 0.5-5000)", ha='center', transform=ax.transAxes)
        return None

    # [C] MASKING (Sigma-Based Motion & Percentile-Based Spikes)
    
    # 1. Motion: Sigma-Based (MAD) on dU
    du = np.diff(u, prepend=u[0])
    # Compute Sigma on finite data
    du_finite = du[np.isfinite(du)]
    
    if len(du_finite) > 10:
        med_du = np.median(du_finite)
        # MAD estimation of sigma: 1.4826 * median(|x - median(x)|)
        mad_du = np.median(np.abs(du_finite - med_du))
        sigma_du = 1.4826 * mad_du
        # Fallback to std if MAD is too small (e.g. constant signal)
        if sigma_du < 1e-12:
            sigma_du = np.std(du_finite)
    elif len(du_finite) >= 3:
        # Fallback for small N
        med_du = np.median(du_finite)
        sigma_du = np.std(du_finite)
    else:
        # Fallback for insufficient data
        med_du = 0.0
        sigma_du = 1e-12
        
    # Enforce floor
    if sigma_du < 1e-12:
        sigma_du = 1e-12
        
    # Motion Hits: > 3.0 sigma OR non-finite
    du_dist = np.abs(du - med_du)
    motion_hits = (du_dist > 3.0 * sigma_du) | (~np.isfinite(du))
    
    # Dilate radius 2 (kernel 5)
    motion_mask = convolve(motion_hits.astype(float), np.ones(5), mode='same') > 0
    
    # 2. Spikes: Percentile on Non-Motion dS
    ds = np.diff(s, prepend=s[0])
    ds_abs = np.abs(ds)
    
    # Non-motion finite samples for threshold
    ds_nm_indices = np.where((~motion_mask) & np.isfinite(ds))[0]
    
    if len(ds_nm_indices) > 50:
        ds_ref = ds_abs[ds_nm_indices]
        spike_thr = np.percentile(ds_ref, 99.0)
    else:
        # Fallback
        finite_ds = ds_abs[np.isfinite(ds_abs)]
        spike_thr = np.max(finite_ds) if len(finite_ds) > 0 else 100.0
        
    spike_mask = (ds_abs > spike_thr) | (~np.isfinite(ds_abs))
    clean_mask = ~(motion_mask | spike_mask)
    
    # 3. Finite Check on Data (Force NaN samples out)
    clean_mask &= (np.isfinite(u) & np.isfinite(s) & np.isfinite(t_sec))
    
    # 4. Fallback Logic if too aggressive
    if np.mean(clean_mask) < 0.5 and len(ds_nm_indices) > 50:
        # Recompute strict threshold (99.5%)
        ds_ref = ds_abs[ds_nm_indices]
        spike_thr = np.percentile(ds_ref, 99.5)
        
        spike_mask = (ds_abs > spike_thr) | (~np.isfinite(ds_abs))
        clean_mask = ~(motion_mask | spike_mask)
        clean_mask &= (np.isfinite(u) & np.isfinite(s) & np.isfinite(t_sec))

    s_clean = s[clean_mask]
    u_clean = u[clean_mask]
    
    if len(s_clean) < 50:
        ax.text(0.5, 0.5, f"Insufficient Clean Data (N={len(s_clean)})", ha='center', transform=ax.transAxes)
        return None

    # [D] GLOBAL LS (Masked)
    res = linregress(u_clean, s_clean)
    slope_g, intercept_g = res.slope, res.intercept
    res_global = s - (slope_g * u + intercept_g)
    
    # [E] DYNAMIC FIT (Using Pipeline Code)
    
    # Define local implementation function for fallback
    def local_rolling_ols(u_vec, s_vec, mask_vec, fs_val, win_sec):
        win_pts = int(win_sec * fs_val)
        win_pts = max(5, min(win_pts, len(s_vec)))
        # Minimum stride of 1
        stride_pts = max(1, win_pts // 10)
        
        n_pts = len(s_vec)
        # Centers for windows
        centers = np.arange(win_pts // 2, n_pts - win_pts // 2, stride_pts)
        
        stats_t = []
        stats_a = []
        stats_b = []
        
        # Fallback threshold: require 80% of window to be valid
        min_win_valid = max(20, int(win_pts * 0.8))
        
        for c_idx in centers:
            l_idx = max(0, c_idx - win_pts//2)
            r_idx = min(n_pts, c_idx + win_pts//2)
            
            mask_win = mask_vec[l_idx:r_idx]
            if np.sum(mask_win) < min_win_valid: continue
            
            ul = u_vec[l_idx:r_idx][mask_win]
            sl = s_vec[l_idx:r_idx][mask_win]
            
            # Avoid singular matrix
            if np.var(ul) < 1e-9: continue
            
            try:
                rr = linregress(ul, sl)
                stats_t.append(c_idx)
                stats_a.append(rr.slope)
                stats_b.append(rr.intercept)
            except ValueError: pass
            
        # Fallback to global if fails
        if len(stats_t) > 2:
            t_ind = np.arange(n_pts)
            a_vec = np.interp(t_ind, stats_t, stats_a)
            b_vec = np.interp(t_ind, stats_t, stats_b)
        else:
            # Use global fit parameters from outer scope
            a_vec = np.full(n_pts, slope_g)
            b_vec = np.full(n_pts, intercept_g)
            
        return s_vec - (a_vec * u_vec + b_vec)
    
    res_dynamic = None
    
    if not PIPELINE_AVAILABLE:
        msg = f"Pipeline Fail: {PIPELINE_ERROR}"
        ax.text(0.5, 0.5, msg[:200].replace('\n', ' '), ha='center', va='center', transform=ax.transAxes, color='red', fontsize=8, wrap=True)
        res_dynamic = local_rolling_ols(u, s, clean_mask, fs, window_sec)
    else:
        # Use Pipeline
        try:
            # Prepare inputs with NaNs for dirty points
            # regression.py fit_chunk_dynamic performs explicit check:
            # m = np.isfinite(u_win) & np.isfinite(s_win)
            u_filt_input = u.copy()
            s_filt_input = s.copy()
            u_filt_input[~clean_mask] = np.nan
            s_filt_input[~clean_mask] = np.nan
            
            # Reshape to (T, 1) and Validate
            uv_raw_2d = u.reshape(T, 1)
            sig_raw_2d = s.reshape(T, 1)
            
            # Construct Chunk with exact required fields
            chunk = Chunk(
                chunk_id=0,
                source_file="verification_panel",
                format="rwd",
                time_sec=t_sec,
                uv_raw=uv_raw_2d,
                sig_raw=sig_raw_2d,
                fs_hz=fs,
                channel_names=['Region0']
            )
            # Assign optional filtered arrays (mutable)
            chunk.uv_filt = u_filt_input.reshape(T, 1)
            chunk.sig_filt = s_filt_input.reshape(T, 1)
            
            # Config
            config = Config()
            config.window_sec = window_sec
            config.step_sec = 10.0
            config.min_samples_per_window = 0
            
            # Execution
            uv_fit_res, delta_f_res = fit_chunk_dynamic(chunk, config)
            
            # Post-check validation
            if uv_fit_res is None or delta_f_res is None:
                raise RuntimeError("Pipeline returned None")
            if delta_f_res.ndim != 2 or delta_f_res.shape[1] != 1:
                raise RuntimeError(f"Invalid result shape: {delta_f_res.shape}")
            if delta_f_res.shape[0] != T:
                 raise RuntimeError(f"Result length mismatch: {delta_f_res.shape[0]} != {T}")
                
            # Dynamic residual = signal - fit
            # Corresponds to delta_f_res[:, 0]
            res_dynamic = delta_f_res[:, 0]
            
            # Check for total failure (all NaNs)
            if np.all(np.isnan(res_dynamic)):
                 ax.text(0.05, 0.9, "Dynamic Fit All NaNs", transform=ax.transAxes, color='red', fontsize=8)
                 res_dynamic = res_global
                 
        except (RuntimeError, ValueError, IndexError, TypeError) as e:
            msg = f"Pipeline Err: {str(e)}"
            ax.text(0.5, 0.5, msg[:200].replace('\n', ' '), ha='center', va='center', transform=ax.transAxes, color='red', fontsize=8, wrap=True)
            res_dynamic = local_rolling_ols(u, s, clean_mask, fs, window_sec)

    # [G] PLOTTING (Smoothed)
    # Rolling mean 30s
    smo_pts = max(1, int(round(30.0 * fs)))
    kernel = np.ones(smo_pts)/smo_pts
    
    # Fill NaNs in res_dynamic before smoothing
    if np.any(np.isnan(res_dynamic)):
        res_dynamic_clean_mean = np.nanmean(res_dynamic)
        if np.isnan(res_dynamic_clean_mean): res_dynamic_clean_mean = 0.0
        res_dynamic_filled = res_dynamic.copy()
        res_dynamic_filled[np.isnan(res_dynamic_filled)] = res_dynamic_clean_mean
        res_dynamic_sm = convolve(res_dynamic_filled, kernel, mode='same')
    else:
        res_dynamic_sm = convolve(res_dynamic, kernel, mode='same')
        
    res_global_sm = convolve(res_global, kernel, mode='same')
    
    # Valid View: exclude edge artifacts from smoothing for autoscaling
    trim = smo_pts 
    if T > 2*trim + 10:
        valid_view = slice(trim, T-trim)
    else:
        valid_view = slice(0, None)
    
    # Time axis in minutes, shifted to 0
    t_plot = (t_sec - t_sec[0]) / 60.0 
    
    ax.plot(t_plot[valid_view], res_global_sm[valid_view], color='gray', label='Global LS residual (smoothed)', alpha=0.8, linewidth=1.5)
    ax.plot(t_plot[valid_view], res_dynamic_sm[valid_view], color='tab:red', label='Dynamic fit residual (smoothed)', alpha=0.9, linewidth=1.5)
    
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    
    ax.set_title("Residual Comparison (Global vs Dynamic)")
    ax.set_xlabel("Time (min)")
    ax.legend(loc='upper right', fontsize='x-small')
    
    # On-plot Diagnostic Overlay (Audit)
    info_text = (f"N={len(t_sec)} fs={fs:.1f}Hz\n"
                 f"S: {s_mean:.2f} +/- {s_std:.3f}\n"
                 f"U: {u_mean:.2f} +/- {u_std:.3f}\n"
                 f"Hash: {digest}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=6, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Auto-scale on SMOOTHED data
    combined_sm = np.concatenate([res_global_sm[valid_view], res_dynamic_sm[valid_view]])
    combined_sm = combined_sm[np.isfinite(combined_sm)]
    
    if len(combined_sm) > 10:
        y_lo = np.percentile(combined_sm, 1.0)
        y_hi = np.percentile(combined_sm, 99.0)
        yr = y_hi - y_lo
        if yr == 0: yr = 1.0
        ax.set_ylim(y_lo - 0.2*yr, y_hi + 0.5*yr)
        
    if return_series:
        return {
            "t_min": t_plot[valid_view],
            "global_sm": res_global_sm[valid_view],
            "dynamic_sm": res_dynamic_sm[valid_view],
            "input_hash_short": digest
        }
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help="Output PNG path")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--window-min', type=float, default=10.0, help="Zoom window size in minutes")
    parser.add_argument('--day-center-hr', type=float, default=30.0, help="Center hour for day zoom")
    parser.add_argument('--night-center-hr', type=float, default=42.0, help="Center hour for night zoom")
    parser.add_argument('--overview-decimate', type=int, default=100, help="Decimation factor for overview")
    parser.add_argument('--same-seed', action='store_true', help="Use same seed for low tonic (default: False)")
    parser.add_argument('--low-seed-offset', type=int, default=1, help="Offset for low tonic seed if not same-seed")
    args = parser.parse_args()
    
    # Determine Low Tonic Seed
    if args.same_seed:
        low_seed = args.seed
        print(f"Seed Strategy: SAME (High={args.seed}, Low={low_seed})")
    else:
        low_seed = args.seed + args.low_seed_offset
        print(f"Seed Strategy: DIVERSE (High={args.seed}, Low={low_seed})")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Generate Data (High and Low)
        print("Generating High Tonic Dataset")
        df_high = generate_data(temp_dir, 'high', args.seed)
        
        print("Generating Low Tonic Dataset")
        df_low = generate_data(temp_dir, 'low', low_seed) 
        
        # 2. Setup 2x5 Figure
        fig = plt.figure(figsize=(24, 10), constrained_layout=True)
        # 2 Rows: High Tonic, Low Tonic
        # 5 Cols: Overview, Day Zoom, Day Res, Night Zoom, Night Res
        gs = fig.add_gridspec(2, 5)
        
        # 3. Define Time Windows (Shared)
        window_hr = args.window_min / 60.0
        
        day_lims = ((args.day_center_hr - window_hr/2)*3600, (args.day_center_hr + window_hr/2)*3600)
        night_lims = ((args.night_center_hr - window_hr/2)*3600, (args.night_center_hr + window_hr/2)*3600)
        
        # Define Workflows
        # Row 0: High Tonic
        # Row 1: Low Tonic
        
        datasets = [
            (0, df_high, "High Tonic"),
            (1, df_low, "Low Tonic")
        ]
        
        # Storage for correlation check: [high_day, low_day] and [high_night, low_night]
        day_resid_data = []
        night_resid_data = []
        
        for row_idx, df, label in datasets:
            # Col 0: Overview
            ax_over = fig.add_subplot(gs[row_idx, 0])
            plot_panel(ax_over, df, f"{label}: 3-Day Overview", "Time (hrs)", decimate=args.overview_decimate)
            
            # Col 1: Day Zoom
            ax_day = fig.add_subplot(gs[row_idx, 1])
            plot_panel(ax_day, df, f"{label}: DAY ({args.window_min}m) Signal+Iso", "Time (min)", x_lims=day_lims)
            
            # Col 2: Day Residuals
            ax_day_res = fig.add_subplot(gs[row_idx, 2])
            # Extract Day Slice for Residuals
            d_day = df[(df['GlobalTime'] >= day_lims[0]) & (df['GlobalTime'] <= day_lims[1])]
            if len(d_day) > 100:
                 t_d = d_day['GlobalTime'].values
                 s_d = d_day['Region0-470'].values
                 u_d = d_day['Region0-410'].values
                 res_out = plot_residuals(ax_day_res, t_d, s_d, u_d, label_prefix=f"{label} DAY", return_series=True)
                 if res_out: day_resid_data.append(res_out)
            else:
                 ax_day_res.text(0.5, 0.5, "Not enough data", ha='center', transform=ax_day_res.transAxes)
                 ax_day_res.set_title("Day Residuals")

            # Col 3: Night Zoom
            ax_night = fig.add_subplot(gs[row_idx, 3])
            plot_panel(ax_night, df, f"{label}: NIGHT ({args.window_min}m) Signal+Iso", "Time (min)", x_lims=night_lims)
            
            # Col 4: Night Residuals
            ax_night_res = fig.add_subplot(gs[row_idx, 4])
            # Extract Night Slice for Residuals
            d_night = df[(df['GlobalTime'] >= night_lims[0]) & (df['GlobalTime'] <= night_lims[1])]
            if len(d_night) > 100:
                 t_n = d_night['GlobalTime'].values
                 s_n = d_night['Region0-470'].values
                 u_n = d_night['Region0-410'].values
                 res_out = plot_residuals(ax_night_res, t_n, s_n, u_n, label_prefix=f"{label} NIGHT", return_series=True)
                 if res_out: night_resid_data.append(res_out)
            else:
                 ax_night_res.text(0.5, 0.5, "Not enough data", ha='center', transform=ax_night_res.transAxes)
                 ax_night_res.set_title("Night Residuals")

        plt.savefig(args.out, dpi=150)
        print(f"Saved verification mosaic to {args.out}")
        
        # Correlation Checks (if we have pairs)
        if len(day_resid_data) == 2:
            r1 = day_resid_data[0]
            r2 = day_resid_data[1]
            # Ensure lengths match for correlation (they should if time windows are identical)
            min_len = min(len(r1['global_sm']), len(r2['global_sm']))
            g_corr, _ = pearsonr(r1['global_sm'][:min_len], r2['global_sm'][:min_len])
            d_corr, _ = pearsonr(r1['dynamic_sm'][:min_len], r2['dynamic_sm'][:min_len])
            print(f"RESID_CORR day global_r={g_corr:.4f} dynamic_r={d_corr:.4f}")
            
        if len(night_resid_data) == 2:
            r1 = night_resid_data[0]
            r2 = night_resid_data[1]
            min_len = min(len(r1['global_sm']), len(r2['global_sm']))
            g_corr, _ = pearsonr(r1['global_sm'][:min_len], r2['global_sm'][:min_len])
            d_corr, _ = pearsonr(r1['dynamic_sm'][:min_len], r2['dynamic_sm'][:min_len])
            print(f"RESID_CORR night global_r={g_corr:.4f} dynamic_r={d_corr:.4f}")

if __name__ == "__main__":
    main()
