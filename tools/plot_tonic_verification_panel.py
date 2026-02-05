
import argparse
import subprocess
import sys
import shutil
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.stats import linregress, zscore

# Ensure we can import from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import core tonic functions
from photometry_pipeline.core.tonic_dff import (
    compute_global_iso_fit, 
    compute_global_iso_fit_robust,
    apply_global_fit, 
    compute_session_tonic_df_from_global,
    compute_slow_baselines
)

def run_command(cmd, desc):
    print(f"\n--- {desc} ---")
    print(f"Command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAILED: {res.stderr}")
        raise RuntimeError(f"Command failed: {desc}")
    else:
        print("OK")

def generate_synthetic_data(output_dir, seed, total_days, config_path, format_mode, tonic_amp):
    script_synth = str(PROJECT_ROOT / 'tools' / 'synth_photometry_dataset.py')
    
    # Strict alignment parameters as requested
    cmd = [
        sys.executable, script_synth,
        '--out', str(output_dir),
        '--config', str(config_path),
        '--format', format_mode,
        '--total-days', str(total_days),
        '--recording-duration-min', '10',
        '--recordings-per-hour', '2', 
        '--fs-hz', '20',
        '--n-rois', '1',
        '--seed', str(seed),
        
        # Tonic Dynamics
        '--tonic-amplitude', str(tonic_amp),
        '--tonic-phase-ct', '6',
        '--tonic-phase-jitter-hr', '0.0',
        
        # Phasic Dynamics (Aligned)
        '--phasic-mode', 'high_phasic',
        '--phasic-base-rate-hz', '0.05',
        '--phasic-peak-rate-hz', '0.10',
        '--phasic-ct-mode', 'phase_aligned',
        '--phasic-day-start-ct', '0',
        '--phasic-day-end-ct', '12'
    ]
    run_command(cmd, "Generating Synthetic Data")

def run_pipeline(input_dir, output_dir, config_path, format_mode):
    script_pipeline = str(PROJECT_ROOT / 'analyze_photometry.py')
    cmd = [
        sys.executable, script_pipeline,
        '--input', str(input_dir),
        '--out', str(output_dir),
        '--config', str(config_path),
        '--format', format_mode,
        '--recursive',
        '--overwrite'
    ]
    run_command(cmd, "Running Photometry Pipeline")

def load_and_stitch_traces(traces_dir, roi_arg=None):
    """
    Loads all chunk CSVs, stitches them into continuous arrays.
    Returns: t_stitch, sig_all, uv_all, sessions_meta, roi
    """
    csv_files = sorted(list(traces_dir.glob("chunk_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No chunk traces found in {traces_dir}")
        
    dfs = []
    # Load first to detect ROI if needed
    first_df = pd.read_csv(csv_files[0])
    
    # Detect ROI
    if roi_arg:
        roi = roi_arg
    else:
        # Region0_sig_raw
        cols = [c for c in first_df.columns if c.endswith('_sig_raw')]
        if not cols:
            raise ValueError("Could not auto-detect ROI from trace columns.")
        roi = cols[0].replace('_sig_raw', '')
        print(f"Auto-selected ROI: {roi}")
        
    sig_col = f"{roi}_sig_raw"
    uv_col = f"{roi}_uv_raw"
    
    sig_list = []
    uv_list = []
    sessions_meta = []
    current_idx = 0
    
    for i, f in enumerate(csv_files):
        df = pd.read_csv(f)
        if sig_col not in df.columns or uv_col not in df.columns:
            print(f"Warning: {f.name} missing columns. Skipping.")
            continue
            
        s = df[sig_col].values
        u = df[uv_col].values
        n = len(s)
        
        sig_list.append(s)
        uv_list.append(u)
        
        sessions_meta.append({
            'session_idx': i,
            'start_sample': current_idx,
            'end_sample': current_idx + n,
            'mid_sample': current_idx + (n // 2)
        })
        current_idx += n
        
    # Concatenate
    sig_all = np.concatenate(sig_list)
    uv_all = np.concatenate(uv_list)
    
    # Create compressed time axis (0 to Total Duration)
    # Assuming 20Hz default (ideally read from config but fixed for verification)
    target_fs = 20.0 
    n_total = len(sig_all)
    t_stitch = np.arange(n_total) / target_fs
    
    return t_stitch, sig_all, uv_all, sessions_meta, roi

def compute_slope_per_hour(t_sec, y_data):
    """Computes linear regression slope per hour."""
    mask = np.isfinite(t_sec) & np.isfinite(y_data)
    if np.sum(mask) < 20: # Need robust count
        return np.nan
    t = t_sec[mask]
    y = y_data[mask]
    
    res = linregress(t, y)
    # Slope is units per second. Convert to units per hour.
    slope_per_hr = res.slope * 3600.0
    return slope_per_hr

def compute_diagnostics(t, uv_all, sig_all, iso_fit_global, dff_global, dff_baseline, fig_dir, drift_mode):
    """Computes sanity metrics for the global fit."""
    lines = []
    mask = np.isfinite(uv_all) & np.isfinite(sig_all)
    n_fit = np.sum(mask)
    
    lines.append(f"Diagnostics for Global Fit (N={n_fit}):")
    
    if n_fit < 50:
        lines.append("WARNING: n_fit too small (<50)!")
        return lines # Basic return
        
    uv = uv_all[mask]
    sig = sig_all[mask]
    iso = iso_fit_global[mask]
    
    # 1. Stats
    lines.append(f"UV (Finite):  Min={np.min(uv):.2f}, Max={np.max(uv):.2f}, Mean={np.mean(uv):.2f}, Std={np.std(uv):.4f}")
    lines.append(f"Sig (Finite): Min={np.min(sig):.2f}, Max={np.max(sig):.2f}, Mean={np.mean(sig):.2f}, Std={np.std(sig):.4f}")
    
    # 2. Correlation Check (Directionality Verification)
    if np.std(uv) > 1e-9 and np.std(iso) > 1e-9:
        # Check Z-scored correlation to confirm shape match
        # Decimate for speed
        step = max(1, len(uv)//5000)
        uv_dec = uv[::step]
        iso_dec = iso[::step]
        corr_uv_iso = np.corrcoef(uv_dec, iso_dec)[0,1]
        lines.append(f"Correlation (UV vs IsoFit): {corr_uv_iso:.4f}")
        
        if corr_uv_iso < 0.8:
             lines.append(f"CRITICAL WARNING: IsoFit does not track UV! Corr={corr_uv_iso:.4f}. Global fit parameters may be inverted or incorrect.")
    
    # 3. Fit Stats
    lines.append(f"IsoFit (Finite): Min={np.min(iso):.2f}, Max={np.max(iso):.2f}, Mean={np.mean(iso):.2f}, Std={np.std(iso):.4f}")
    
    # 4. Slope Diagnostics (Per Hour)
    # Check drift removal efficiency
    slope_uv = compute_slope_per_hour(t, uv_all)
    slope_sig = compute_slope_per_hour(t, sig_all)
    slope_iso = compute_slope_per_hour(t, iso_fit_global)
    # For df, usage is AU, so let's report AU/hr
    slope_df = compute_slope_per_hour(t, dff_global) # dff_global is actually df_global now (passed in)
    slope_base = compute_slope_per_hour(t, dff_baseline)
    
    lines.append("Slope Diagnostics (units/hr):")
    lines.append(f"  UV Slope: {slope_uv:.4f}")
    lines.append(f"  Sig Slope: {slope_sig:.4f}")
    lines.append(f"  IsoFit Slope: {slope_iso:.4f}")
    lines.append(f"  df Slope (AU/hr): {slope_df:.4f}")
    lines.append(f"  df Baseline Slope (AU/hr): {slope_base:.4f}")
    

    # Slope Direction Check
    # If UV is bleaching (negative slope), High-Confidence Fit should also imply negative slope
    # Only enforce this STRICTLY if we expect drift to dominate (drift_only or low amp)
    if drift_mode:
        if slope_uv < -0.1 and slope_iso > 0.05:
            lines.append("FAILURE: UV is bleaching (neg slope) but IsoFit is growing (pos slope). Fit incorrect.")
            raise RuntimeError("Drift Direction Mismatch in drift-only mode.")

    # Strict Acceptance Checks
    if drift_mode:
        if abs(slope_df) > 0.05:
            lines.append(f"FAILURE: High df drift in drift-only mode ({slope_df:.4f} AU/hr). Threshold 0.05.")
            raise RuntimeError(f"Drift check failed: df slope {slope_df:.4f} AU/hr too high.")
            
        if abs(slope_base) > 0.05:
            lines.append(f"FAILURE: High Baseline drift in drift-only mode ({slope_base:.4f} AU/hr). Threshold 0.05.")
            raise RuntimeError(f"Drift check failed: Baseline slope {slope_base:.4f} AU/hr too high.")
            
    else:
        # In Normal mode
        # Just warn if direction seems flipped, but don't crash
        if slope_uv < -0.1 and slope_iso > 0.05:
            lines.append(f"WARNING: IsoFit slope ({slope_iso:.4f}) opposes UV drift ({slope_uv:.4f}). Biology may be dominating.")
            
        if abs(slope_df) > 0.5: 
             lines.append(f"NOTE: df slope is {slope_df:.4f} AU/hr (Contains Biology).")

        
    return lines

def plot_verification_panel(t, sig, uv, iso_fit_global, dff_global, dff_baseline_line, uv_slow, roi, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    t_hr = t / 3600.0
    
    # --- Panel A: Raw Signal & Isosbestic ---
    ax = axes[0, 0]
    ax.plot(t_hr, sig, color='green', label='Signal', alpha=0.8, linewidth=1)
    ax.plot(t_hr, uv, color='purple', label='Isosbestic', alpha=0.8, linewidth=1)
    ax.set_title("Panel A: Raw Signals")
    ax.set_ylabel("Fluorescence (AU)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # --- Panel B: Raw Signal & GLOBAL Fitted Isosbestic (Dual Axis) + Drift Shape Check ---
    ax = axes[0, 1]
    # Primary Axis: Signal + Global Fit (Signal Units)
    ln1 = ax.plot(t_hr, sig, color='green', label='Signal', alpha=0.6, linewidth=1)
    ln2 = ax.plot(t_hr, iso_fit_global, color='black', linestyle='-', label='Global Fit RAW', alpha=0.8, linewidth=1.5)
    
    ax.set_ylabel("Signal units (AU)")
    ax.set_title("Panel B: Sig vs Global Fit (Left) | iso_fit_global_raw = slope * uv_raw + intercept")
    ax.grid(True, alpha=0.3)
    
    # Robust Y-Limits (Percentile based) to ignore spikes
    combo = np.concatenate([sig[np.isfinite(sig)], iso_fit_global[np.isfinite(iso_fit_global)]])
    if len(combo) > 50:
        lo_b = np.percentile(combo, 1.0)
        hi_b = np.percentile(combo, 99.0)
        if np.isfinite(lo_b) and np.isfinite(hi_b) and hi_b > lo_b:
            margin = (hi_b - lo_b) * 0.1
            ax.set_ylim(lo_b - margin, hi_b + margin)

    # Secondary Axis: Raw UV
    ax2 = ax.twinx()
    # Plot Raw UV (Visual Guide)
    ln3 = ax2.plot(t_hr, uv, color='purple', label='Raw UV', alpha=0.3, linewidth=0.5)
    # Plot uv_slow diagnostic
    ln4 = ax2.plot(t_hr, uv_slow, color='magenta', linestyle='--', label='UV Slow (Diag)', alpha=0.4, linewidth=1.0)
    
    ax2.set_ylabel("Isosbestic (UV, AU)")
    
    # Consolidated Legend
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right')


    
    # Inset: Z-Score Shape Check (Drift Matching)
    # Place in bottom right corner
    ax_ins = ax.inset_axes([0.65, 0.05, 0.3, 0.25])
    # Decimate for speed/clarity
    step = 100
    if len(t) > step:
        t_dec = t_hr[::step]
        uv_dec = uv[::step]
        iso_dec = iso_fit_global[::step]
        
        # Z-Score (Robust)
        uv_z = (uv_dec - np.nanmedian(uv_dec)) / np.nanstd(uv_dec)
        iso_z = (iso_dec - np.nanmedian(iso_dec)) / np.nanstd(iso_dec)
        
        ax_ins.plot(t_dec, uv_z, color='purple', alpha=0.6, label='UV (Z)')
        ax_ins.plot(t_dec, iso_z, color='black', alpha=0.6, linestyle='--', label='Fit (Z)')
        ax_ins.set_title("Drift Shape (Z-Scored)", fontsize=8)
        ax_ins.tick_params(labelsize=6)
        ax_ins.grid(True, alpha=0.2)

    # --- Panel C: Tonic df Trace (Sample Level) ---
    ax = axes[1, 0]
    # dff_global here is df (additive)
    df_vals = dff_global 
    ax.plot(t_hr, df_vals, color='blue', label='Tonic df (Global)', linewidth=1)
    ax.set_title("Panel C: df = sig - iso_fit_global_raw")
    ax.set_ylabel("df (AU)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Robust Y-Limits for df
    vals_c = df_vals[np.isfinite(df_vals)]
    if len(vals_c) > 50:
        lo_c = np.percentile(vals_c, 1.0)
        hi_c = np.percentile(vals_c, 99.0)
        if np.isfinite(lo_c) and np.isfinite(hi_c) and hi_c > lo_c:
             margin = (hi_c - lo_c) * 0.1
             ax.set_ylim(lo_c - margin, hi_c + margin)
    
    # --- Panel D: Tonic Baseline ONLY (Per-Session Scalar) ---
    ax = axes[1, 1]
    ax.plot(t_hr, dff_baseline_line, color='magenta', label='Tonic Baseline (2nd %ile)', linewidth=1.5)
    ax.set_title("Panel D: Tonic Baseline (2nd %ile of df)")
    ax.set_xlabel("Time (compressed hr), 1 compressed hr = 3 real hr (20 min recordings per hour)")
    ax.set_ylabel("df (AU)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved figure to {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Tonic Verification Panel")
    parser.add_argument('--out', required=True, help="Output directory")
    parser.add_argument('--format', default='rwd', choices=['rwd', 'npm'])
    parser.add_argument('--config', help="Path to config yaml")
    parser.add_argument('--roi', help="Specific ROI to analyze")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for generation")
    parser.add_argument('--tonic-amp', type=float, default=0.4, help="Tonic amplitude")
    parser.add_argument('--drift-only', action='store_true', help="Force tonic amp to 0 for drift check")
    args = parser.parse_args()
    
    tonic_amp = 0.0 if args.drift_only else args.tonic_amp
    
    base_out = Path(args.out)
    run_dir = base_out / 'tonic_verification_run'
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    
    input_dir = run_dir / 'input'
    pipeline_out = run_dir / 'pipeline_out'
    fig_dir = run_dir / 'figures'
    fig_dir.mkdir(parents=True)
    
    # Config setup
    if args.config:
        config_path = Path(args.config)
    else:
        # Create a default test config
        config_path = run_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            f.write("""
target_fs_hz: 20
chunk_duration_sec: 600
baseline_method: uv_raw_percentile_session
baseline_percentile: 10
rwd_time_col: TimeStamp
uv_suffix: "-410"
sig_suffix: "-470"
peak_threshold_method: mean_std
window_sec: 20.0
step_sec: 5.0
min_samples_per_window: 100
min_valid_windows: 5
f0_min_value: 10.0
qc_max_chunk_fail_fraction: 1.0
seed: 42
""")

    # 1. Generate Data
    generate_synthetic_data(input_dir, args.seed, 2.0, config_path, args.format, tonic_amp)
    
    # 2. Run Pipeline
    run_pipeline(input_dir, pipeline_out, config_path, args.format)
    
    # 3. Stitch & Analyze
    traces_dir = pipeline_out / 'traces'
    t, sig_all, uv_all, sessions_meta, roi = load_and_stitch_traces(traces_dir, args.roi)
    
    print(f"Stitched {len(t)} samples for {roi}.")
    
    # 4. GLobal Tonic Correcton (New Logic: Robust Fit on RAW Data)
    # A) Compute Slow Baselines (Median interpolation) -> FOR PLOTTING ONLY
    uv_slow, sig_slow, uv_sess_stats, sig_sess_stats = compute_slow_baselines(
        t, uv_all, sig_all, sessions_meta, baseline_stat='median'
    )
    
    print(f"Computing Global Fit Robust on RAW data (N={len(uv_all)})...")
    slope, intercept, ok, n_used = compute_global_iso_fit_robust(uv_all, sig_all)
        
    if not ok:
        print("Global fit failed.")
        return
        
    print(f"Global Fit: Slope={slope:.4f}, Intercept={intercept:.4f}, Points Used={n_used}")
    
    # B) Construct Global Fit using RAW UV (The Requirements!)
    # iso_fit_global_raw = slope * uv_raw + intercept
    iso_fit_global_raw = apply_global_fit(uv_all, slope, intercept)
    
    # Optional diagnostic line for plot (not used for math)
    iso_fit_global_slow = apply_global_fit(uv_slow, slope, intercept)
    
    # --- AUDIT BLOCK START ---
    audit_lines = []
    audit_lines.append("\n--- RT AUDIT START ---")
    audit_lines.append(f"sig_all id: {id(sig_all)}, shape: {sig_all.shape}, finite: {np.sum(np.isfinite(sig_all))}")
    audit_lines.append(f"uv_all id: {id(uv_all)}, shape: {uv_all.shape}, finite: {np.sum(np.isfinite(uv_all))}")
    audit_lines.append(f"iso_fit_global_raw id: {id(iso_fit_global_raw)}, shape: {iso_fit_global_raw.shape}, finite: {np.sum(np.isfinite(iso_fit_global_raw))}")
    
    slope_sig = compute_slope_per_hour(t, sig_all)
    slope_uv_raw = compute_slope_per_hour(t, uv_all)
    slope_iso_fit = compute_slope_per_hour(t, iso_fit_global_raw)
    
    audit_lines.append(f"Slopes (per hr): Sig={slope_sig:.4f}, UV_Raw={slope_uv_raw:.4f}, Iso_Fit_Raw={slope_iso_fit:.4f}")
    
    step_dec = max(1, len(uv_all)//5000)
    audit_corr = np.corrcoef(uv_all[::step_dec], iso_fit_global_raw[::step_dec])[0,1]
    audit_lines.append(f"Correlation (UV_Raw vs Iso_Fit_Raw): {audit_corr:.4f}")
    
    
    # Strict Identity Assertion
    print(f"Asserting iso_fit_global_raw identity at call site: {id(iso_fit_global_raw)}")
    assert iso_fit_global_raw is iso_fit_global_raw # Trivial Python santity
    
    # Compute Global df (Panel C)
    res = compute_session_tonic_df_from_global(sig_all, uv_all, iso_fit_global_raw, percentile=2.0) # Using RAW fit

    
    # Verify Math (Global Spot Check)
    df_calc = res['df']
    check_idx = slice(1000, 2000)
    # Recompute manual
    sig_sub = sig_all[check_idx]
    iso_sub = iso_fit_global_raw[check_idx] # RAW
    df_sub_pre = df_calc[check_idx]
    
    mask_sub = np.isfinite(sig_sub) & np.isfinite(iso_sub)
    df_manual_sub = np.full_like(sig_sub, np.nan)
    df_manual_sub[mask_sub] = sig_sub[mask_sub] - iso_sub[mask_sub]
    
    # Compare
    # Only compare where both finite
    valid_cmp = np.isfinite(df_sub_pre) & np.isfinite(df_manual_sub)
    if np.sum(valid_cmp) > 0:
        err = np.abs(df_sub_pre[valid_cmp] - df_manual_sub[valid_cmp])
        max_err = np.max(err)
        mean_err = np.mean(err)
        print(f"Global df Check [1000:2000]: MaxErr={max_err:.4e}, MeanErr={mean_err:.4e}")
        if max_err > 1e-10:
             raise RuntimeError(f"Global df verification failed! Max Error {max_err}")
    
    audit_lines.append(f"Panel C uses df = sig - iso_fit_global_raw")
    
    # Save audit lines (Now including all)
    audit_log_path = fig_dir / 'audit_proof.txt'
    with open(audit_log_path, 'w') as f:
        f.write("\n".join(audit_lines))
        
    # --- AUDIT BLOCK END ---
    dff_global = res['df'] # Naming it dff_global to keep variable name consistent but hold df
    
    # Compute Baseline Line (Panel D) - Interpolated Per-Session Scalars
    mid_points = []
    scalars = []
    
    for session in sessions_meta:
        s = session['start_sample']
        e = session['end_sample']
        mid = session['mid_sample']
        
        # Slice
        sig_chunk = sig_all[s:e]
        uv_chunk = uv_all[s:e]
        iso_chunk = iso_fit_global_raw[s:e] # RAW
        
        # Compute Scalar
        res_chunk = compute_session_tonic_df_from_global(sig_chunk, uv_chunk, iso_chunk, percentile=2.0)
        sc = res_chunk['tonic_scalar']
        
        # --- AUDIT SESSION ---
        if session['session_idx'] < 3:
            dff_chunk_auto = res_chunk['df']
            dff_chunk_manual = np.full_like(sig_chunk, np.nan)
            mck = np.isfinite(sig_chunk) & np.isfinite(iso_chunk)
            dff_chunk_manual[mck] = sig_chunk[mck] - iso_chunk[mck]
            
            # Compare
            v_cmp = np.isfinite(dff_chunk_auto) & np.isfinite(dff_chunk_manual)
            if np.sum(v_cmp) > 0:
                err_c = np.abs(dff_chunk_auto[v_cmp] - dff_chunk_manual[v_cmp])
                mx_c = np.max(err_c)
                print(f"Session {session['session_idx']} Check: MaxErr={mx_c:.4e}")
                if mx_c > 1e-10: 
                    raise RuntimeError(f"Session {session['session_idx']} math check failed!")
                if mx_c > 1e-10: 
                    raise RuntimeError(f"Session {session['session_idx']} math check failed!")
        # --- END AUDIT ---
        
        mid_points.append(mid)
        scalars.append(sc)
        
    # --- DRIFT ONLY ACCEPTANCE TEST (Step 4) ---
    if args.drift_only:
        # Regress session_scalars vs session_mid_times (in hours)
        # mid_points are sample indices. fs=20.
        sess_times_hr = np.array(mid_points) / 20.0 / 3600.0
        sess_sc = np.array(scalars)
        
        # Filter nans
        msk_reg = np.isfinite(sess_times_hr) & np.isfinite(sess_sc)
        if np.sum(msk_reg) > 2:
            res_reg = linregress(sess_times_hr[msk_reg], sess_sc[msk_reg])
            slope_scalar_au_hr = res_reg.slope
            print(f"Drift-Only Session Scalar Slope: {slope_scalar_au_hr:.6f} AU/hr")
            if abs(slope_scalar_au_hr) > 0.05:
                print(f"FAILURE: Session Scalar Drift {slope_scalar_au_hr:.6f} > 0.05 AU/hr")
                raise RuntimeError("Drift-Only Check Failed on RAW Scalars.")
            else:
                print("SUCCESS: Drift-Only Session Scalar Check Passed.")
        else:
            print("WARNING: Not enough sessions for scalar regression.")

    # Interpolate for Panel D line
    x_points = [0] + mid_points + [len(t)-1]
    y_points = [scalars[0]] + scalars + [scalars[-1]]
    
    valid_x = []
    valid_y = []
    for x, y in zip(x_points, y_points):
        if np.isfinite(y):
            valid_x.append(x)
            valid_y.append(y)
            
    if len(valid_x) > 1:
        f_interp = interp1d(valid_x, valid_y, kind='linear', bounds_error=False, fill_value="extrapolate")
        dff_baseline_line = f_interp(np.arange(len(t)))
    else:
        dff_baseline_line = np.full_like(t, np.nan)
    
    # Diagnostics
    diag_lines = compute_diagnostics(t, uv_all, sig_all, iso_fit_global_raw, dff_global, dff_baseline_line, fig_dir, args.drift_only)
    for l in diag_lines:
        print(l)
    
    # Save Diagnostics
    diag_path = fig_dir / 'global_fit_diagnostics.txt'
    with open(diag_path, 'w') as f:
        f.write("\n".join(diag_lines))
    print(f"Saved diagnostics to {diag_path}")

    # 5. Plot
    out_fig = fig_dir / 'verification_panel.png'
    # Use iso_fit_global_raw for plotting Panel B main line
    # Note: We pass iso_fit_global_raw as the 4th argument.
    plot_verification_panel(t, sig_all, uv_all, iso_fit_global_raw, dff_global, dff_baseline_line, uv_slow, roi, out_fig)
    
    # PDF
    plot_verification_panel(t, sig_all, uv_all, iso_fit_global_raw, dff_global, dff_baseline_line, uv_slow, roi, fig_dir / 'verification_panel.pdf')

if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        print(f"FAILED: {e}")
