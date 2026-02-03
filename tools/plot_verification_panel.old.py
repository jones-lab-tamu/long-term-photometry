
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
        
    # We use a config file generator or just pass a dummy one if the script allows?
    # The script requires --config. We need to create one.
    config_path = os.path.join(temp_root, "config_temp.yaml")
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
    # We use subprocess to call tools/synth_photometry_dataset.py
    # "Continuous" means recording-duration-min = 60, recordings-per-hour = 1
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
        "--phasic-mode", "high_phasic", # Constant across panels
        "--phasic-base-rate-hz", "0.05", # Updated stable rate
        "--phasic-peak-rate-hz", "0.10", # Ensure Day (Peak) > Night (Base)
        "--phasic-ct-mode", "absolute", # Match plot logic
        "--artifact-enable-motion", # Enabled
        "--artifact-motion-rate-per-day", "600.0",
        "--artifact-motion-neg-prob", "0.85",
        "--artifact-motion-amp-range", "8.0", "16.0", # Lower amplitude
        "--artifact-motion-rise-sec", "0.30", # Slow rise
        "--artifact-motion-decay-sec", "2.50", # Slow decay
        "--shared-wobble-enable",
        "--shared-wobble-amp", "2.0",
        "--shared-wobble-tau-sec", "60.0",
        "--shared-wobble-gain-enable", # Time-varying coupling
        "--shared-wobble-gain-tau-sec", "120.0",
        "--shared-wobble-gain-sd", "0.30",
        "--shared-wobble-offset-enable",
        "--shared-wobble-offset-amp", "2.0",
        "--shared-wobble-offset-tau-sec", "180.0"
    ]
    
    print(f"Generating {tonic_mode} data (Amp={amp})...")
    subprocess.run(cmd, check=True)
    
    # Load and Concatenate
    files = glob.glob(os.path.join(out_dir, "*", "fluorescence.csv"))
    files.sort()
    
    if not files:
        raise RuntimeError("No files generated!")
        
    dfs = []
    # We need to reconstruct absolute time because individual files restart t_local?
    # Actually the generator puts t_local in the CSV.
    # But we know the structure: 1 hour files.
    # File 0: 0-3600s, File 1: 3600-7200s, etc.
    
    current_offset = 0.0
    for f in files:
        d = pd.read_csv(f)
        # RWD format uses TimeStamp column
        d['GlobalTime'] = d['TimeStamp'] + current_offset
        dfs.append(d)
        current_offset += 3600.0
        
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

def plot_panel(ax, df, title, x_label, x_lims=None, decimate=1):
    """
    Plot Sig and UV on the given axis.
    """
    # Decimate for speed if needed
    if decimate > 1:
        d = df.iloc[::decimate]
    else:
        d = df
        
    t = d['GlobalTime'].values
    s = d['Region0-470'].values
    u = d['Region0-410'].values
    
    # Convert time for display
    # If 3-day view, use Hours.
    # If 10-min view, use Minutes relative to start of window?
    
    if x_lims:
        # Windowed View
        t0, t1 = x_lims
        mask = (t >= t0) & (t <= t1)
        # Apply mask
        d_slice = df[mask]
        t_slice = d_slice['GlobalTime'].values
        s_slice = d_slice['Region0-470'].values
        u_slice = d_slice['Region0-410'].values
        
        # Relative time in minutes
        t_plot = (t_slice - t0) / 60.0
        
        ax.plot(t_plot, s_slice, color='tab:blue', label='Signal (470)', linewidth=1.0, alpha=0.9)
        ax.plot(t_plot, u_slice, color='tab:purple', label='Iso (410)', linewidth=1.0, alpha=0.9)
        ax.set_xlabel("Time (min)")
        ax.set_xlim(0, (t1 - t0)/60.0)
        
        # Robust Auto-Scale for Zoom
        combined = np.concatenate([s_slice, u_slice])
        if len(combined) > 10:
            y_lo = np.percentile(combined, 1.0)
            y_hi = np.percentile(combined, 99.0)
            yr = y_hi - y_lo
            if yr == 0: yr = 1.0
            ax.set_ylim(y_lo - 0.1*yr, y_hi + 0.3*yr) # Extra top headroom for spikes

    else:
        # Full View (Hours)
        t_hours = t / 3600.0
        ax.plot(t_hours, s, color='tab:blue', label='Signal (470)', linewidth=0.5, alpha=0.8)
        ax.plot(t_hours, u, color='tab:purple', label='Iso (410)', linewidth=0.5, alpha=0.6)
        ax.set_xlabel("Time (Hours)")
        ax.set_xlim(0, 72)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

def plot_residuals(ax, t_sec, s, u, window_sec=60.0):
    """
    Plot Global LS Residual vs Dynamic Fit Residual on 'Clean' Data
    1. Infer fs (assert t_sec is seconds)
    2. Build Mask: ~(Motion | Spikes)
    3. Global LS on Masked: Fit s ~ a*u + b
    4. Rolling Fit on Masked: Fit s ~ a(t)*u + b(t)
    5. Plot smoothed residuals to show low-freq drift.
    """
    from scipy.stats import linregress
    
    dt = None
    if len(t_sec) < 3:
        ax.text(0.5, 0.5, "Insufficient time samples for residual plot", ha='center', transform=ax.transAxes)
        return

    dt = np.nanmedian(np.diff(t_sec))
    if np.isnan(dt) or dt <= 0:
        ax.text(0.5, 0.5, "Invalid timebase (dt<=0)", ha='center', transform=ax.transAxes)
        return

    fs = 1.0 / dt
    
    if fs < 0.5 or fs > 5000:
        raise ValueError(f"plot_residuals expects t in seconds. Inferred fs={fs:.2f} Hz (dt={dt:.5f}). Out of bounds (0.5-5000).")
        
    # 2. MASKING
    # A. Motion (du based)
    du = np.diff(u, prepend=u[0])
    motion_gate = 3.0
    motion_hits = np.abs(du) > motion_gate
    # Dilate radius 2 (kernel 5)
    motion_mask = np.convolve(motion_hits.astype(float), np.ones(5), mode='same') > 0
    
    # B. Spikes (ds based)
    ds = np.diff(s, prepend=s[0])
    
    # For threshold, exclude motion regions so large motion derivative doesn't skew percentile
    ds_clean_candidates = ds[~motion_mask]
    if len(ds_clean_candidates) < 100:
        ds_clean_candidates = ds
        
    if len(ds_clean_candidates) > 0:
        spike_thr = np.percentile(np.abs(ds_clean_candidates), 99.0)
    else:
        spike_thr = 100.0 
        
    spike_mask = np.abs(ds) > spike_thr
    
    clean_mask = ~(motion_mask | spike_mask)
    
    # Fallback if too aggressive
    # Fallback if too aggressive
    if np.mean(clean_mask) < 0.5:
         # Relax to 99.5th percentile of NON-motion data
         if len(ds_clean_candidates) > 0:
             spike_thr_relaxed = np.percentile(np.abs(ds_clean_candidates), 99.5)
         else:
             spike_thr_relaxed = 100.0
             
         spike_mask = np.abs(ds) > spike_thr_relaxed
         clean_mask = ~(motion_mask | spike_mask)

    s_clean = s[clean_mask]
    u_clean = u[clean_mask]
    
    if len(s_clean) < 50:
        ax.text(0.5, 0.5, "Insufficient Clean Data", ha='center')
        return

    # 3. GLOBAL LS (Masked)
    res = linregress(u_clean, s_clean)
    slope_g, intercept_g = res.slope, res.intercept
    res_global = s - (slope_g * u + intercept_g)
    
    # 4. DYNAMIC (Rolling Masked)
    win_pts = int(window_sec * fs)
    win_pts = max(20, min(win_pts, len(s)))
    stride_pts = max(1, win_pts // 10)
    
    n = len(s)
    centers = np.arange(0, n, stride_pts)
    params_g = []
    params_o = []
    valid_centers = []
    
    for c in centers:
        l = max(0, c - win_pts//2)
        r = min(n, c + win_pts//2)
        
        # Check sufficient clean data in window
        mask_win = clean_mask[l:r]
        if np.sum(mask_win) < max(20, 0.1 * (r-l)):
            continue
            
        sl = s[l:r][mask_win]
        ul = u[l:r][mask_win]
        
        # Need variance in u to fit
        if np.std(ul) < 1e-6:
            continue
            
        try:
            rr = linregress(ul, sl)
            params_g.append(rr.slope)
            params_o.append(rr.intercept)
            valid_centers.append(c)
        except:
            pass
            
    if len(valid_centers) > 1:
        g_vec = np.interp(np.arange(n), valid_centers, params_g)
        o_vec = np.interp(np.arange(n), valid_centers, params_o)
    else:
        # Fallback to global
        g_vec = np.full(n, slope_g)
        o_vec = np.full(n, intercept_g)
        
    res_dynamic = s - (g_vec * u + o_vec)
    
    # 5. PLOTTING (Smoothed)
    # Rolling mean 30s
    smo_pts = int(30.0 * fs)
    if smo_pts < 5: smo_pts = 5
    kernel = np.ones(smo_pts)/smo_pts
    
    # Mode='same' can have boundary artifacts, but acceptable for vis
    res_global_sm = np.convolve(res_global, kernel, mode='same')
    res_dynamic_sm = np.convolve(res_dynamic, kernel, mode='same')
    
    # Trim boundary artifacts
    valid_view = slice(smo_pts//2, -smo_pts//2)
    if (valid_view.stop - valid_view.start) < 10:
        valid_view = slice(0, None)
    
    # Assume t_sec starts at some large value, shift to 0 for plot
    t_plot = (t_sec - t_sec[0]) / 60.0 # to minutes
    
    ax.plot(t_plot[valid_view], res_global_sm[valid_view], color='gray', label='Global LS (Smoothed)', alpha=0.8, linewidth=2.0)
    ax.plot(t_plot[valid_view], res_dynamic_sm[valid_view], color='tab:red', label='Dynamic Fit (Smoothed)', alpha=0.9, linewidth=2.0)
    
    ax.axhline(0, color='k', linestyle=':', alpha=0.3)
    
    ax.set_title("Residual Comparison (Global vs Dynamic)")
    ax.set_xlabel("Time (min)")
    ax.legend(loc='upper right', fontsize='x-small')
    
    # Auto-scale
    combined_sm = np.concatenate([res_global_sm[valid_view], res_dynamic_sm[valid_view]])
    if len(combined_sm) > 10:
        y_lo = np.percentile(combined_sm, 1.0)
        y_hi = np.percentile(combined_sm, 99.0)
        yr = y_hi - y_lo
        if yr == 0: yr = 1.0
        ax.set_ylim(y_lo - 0.2*yr, y_hi + 0.2*yr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help="Output PNG path")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Generate Data
        print("Generating High Tonic Dataset...")
        df_high = generate_data(temp_dir, 'high', args.seed)
        
        print("Generating Low Tonic Dataset...")
        df_low = generate_data(temp_dir, 'low', args.seed) # Same seed
        
        # 2. Setup Figure
        # We want Residuals to be prominent.
        # Layout: 2x2.
        # Top Left: High Tonic Day Slice
        # Top Right: Residuals for Day Slice
        # Bottom Left: High Tonic Night Slice
        # Bottom Right: Low Tonic Overview
        
        # Or keep 2x3? The syntax error was due to 'pass' then 'else' outside indentation block or mixed.
        # Let's use 2x2 for clarity.
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)
        
        # 3. Define Time Windows
        # Use Absolute CT Logic (t_global % 24)
        # Day Slice: Day 2 (24h+) -> CT 6 (30h).
        # Night Slice: Day 2 (24h+) -> CT 18 (42h).
        
        day_center_hr = 30.0
        night_center_hr = 42.0
        window_min = 20.0 # Wide enough to see wobble
        window_hr = window_min / 60.0
        
        day_lims = ((day_center_hr - window_hr/2)*3600, (day_center_hr + window_hr/2)*3600)
        night_lims = ((night_center_hr - window_hr/2)*3600, (night_center_hr + window_hr/2)*3600)
        
        # 4. Plot Rows
        
        # Top Left: Day Slice
        ax_day = fig.add_subplot(gs[0, 0])
        # Filter for day slice
        d_day = df_high[(df_high['GlobalTime'] >= day_lims[0]) & (df_high['GlobalTime'] <= day_lims[1])]
        plot_panel(ax_day, df_high, "High Tonic: Day (Phasic Peak)", "Time (min)", x_lims=day_lims)
        
        # Top Right: Residuals
        ax_res = fig.add_subplot(gs[0, 1])
        if len(d_day) > 100:
             t_d = d_day['GlobalTime'].values
             s_d = d_day['Region0-470'].values
             u_d = d_day['Region0-410'].values
             plot_residuals(ax_res, t_d, s_d, u_d)
        else:
             ax_res.text(0.5, 0.5, "Not enough data", ha='center')
             
        # Bottom Left: Night Slice
        ax_night = fig.add_subplot(gs[1, 0])
        plot_panel(ax_night, df_high, "High Tonic: Night (Phasic Trough)", "Time (min)", x_lims=night_lims)
        
        # Bottom Right: Overview
        ax_over = fig.add_subplot(gs[1, 1])
        plot_panel(ax_over, df_low, "Low Tonic Overview (3 Days)", "Time (hrs)", decimate=100)
        
        plt.savefig(args.out, dpi=150)
        print(f"Saved verification panel to {args.out}")
        


if __name__ == "__main__":
    main()
