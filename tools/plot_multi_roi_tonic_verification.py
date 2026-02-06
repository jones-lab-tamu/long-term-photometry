
import argparse
import os
import sys
import glob
import pandas as pd
import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure processed repo root is in path
sys.path.append(os.getcwd())

from photometry_pipeline.core.tonic_dff import (
    compute_global_iso_fit_robust,
    apply_global_fit,
    compute_session_tonic_df_from_global
)

MIN_RATE_RATIO = 1.5

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-ROI Tonic Verification Plotter")
    parser.add_argument('--pipeline-out', required=True, help="Path to pipeline output folder (containing traces/)")
    parser.add_argument('--out', required=True, help="Output PNG path")
    parser.add_argument('--rois', default="Region0,Region1,Region2,Region3", help="Comma-separated ROI names")
    
    # Phasic Verification Args
    parser.add_argument('--ct-day-start', type=float, default=0.0)
    parser.add_argument('--ct-day-end', type=float, default=12.0)
    parser.add_argument('--event-hp-cutoff-hz', type=float, default=0.01)
    parser.add_argument('--event-threshold-k', type=float, default=3.0)
    parser.add_argument('--event-min-sep-sec', type=float, default=2.0)
    
    # Tonic Phase Args (Phase-Anchored)
    parser.add_argument('--circ-period-hr', type=float, default=24.0)
    parser.add_argument('--circ-bandwidth-hr', type=float, default=6.0)
    parser.add_argument('--phase-peak-halfwidth-hr', type=float, default=3.0)
    parser.add_argument('--phase-trough-halfwidth-hr', type=float, default=3.0)
    parser.add_argument('--phase-pass-ratio', type=float, default=1.5)
    
    parser.add_argument('--print-day-night-events', action='store_true', help="Print quantitative verification table (CT)")
    parser.add_argument('--print-phase-events', action='store_true', help="Print detailed phase event stats")
    
    return parser.parse_args()

def wrap_to_pi(angle):
    """Map angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def load_all_traces(traces_dir, roi):
    """
    Loads and concatenates time series for a specific ROI across all chunks.
    Assumes traces/chunk_XXXX.csv naming.
    """
    search_path = os.path.join(traces_dir, "*.csv")
    files = sorted(glob.glob(search_path))
    
    if not files:
        raise ValueError(f"No trace files found in {traces_dir}")
        
    t_list = []
    uv_list = []
    sig_list = []
    uv_fit_pipe_list = []
    chunk_id_list = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            # Check ROI cols exist
            uv_col = f"{roi}_uv_raw"
            sig_col = f"{roi}_sig_raw"
            uv_fit_col = f"{roi}_uv_fit" # Pipeline fitted
            
            if uv_col not in df.columns:
                print(f"SKIP {f}: Missing {uv_col}")
                continue
                
            t_vals = df['time_sec'].values
            uv_vals = df[uv_col].values
            sig_vals = df[sig_col].values
            
            if uv_fit_col not in df.columns:
                uv_fit_vals = np.full(len(df), np.nan)
            else:
                uv_fit_vals = df[uv_fit_col].values
            
            # Heuristic: Check file name for chunk ID to offset time
            fname = os.path.basename(f)
            try:
                chunk_idx = int(fname.split('_')[1].split('.')[0])
            except:
                chunk_idx = 0
                
            # Offset time
            offset_sec = chunk_idx * 30.0 * 60.0
            if t_vals[0] < 100.0:
                 t_vals = t_vals + offset_sec
            
            # Create Chunk IDs (Same length)
            chunk_ids_vals = np.full(len(t_vals), chunk_idx, dtype=np.int64)
            
            # Append with a NaN gap
            t_list.append(t_vals)
            t_list.append([np.nan])
            
            uv_list.append(uv_vals)
            uv_list.append([np.nan])
            
            sig_list.append(sig_vals)
            sig_list.append([np.nan])
            
            uv_fit_pipe_list.append(uv_fit_vals)
            uv_fit_pipe_list.append([np.nan])
            
            chunk_id_list.append(chunk_ids_vals)
            chunk_id_list.append([-1]) # Gap is -1
            
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not t_list:
        raise ValueError(f"No valid data loaded for ROI {roi}")
        
    t_all = np.concatenate(t_list).astype(float)
    uv_all = np.concatenate(uv_list).astype(float)
    sig_all = np.concatenate(sig_list).astype(float)
    uv_fit_pipe_all = np.concatenate(uv_fit_pipe_list).astype(float)
    chunk_id_all = np.concatenate(chunk_id_list).astype(np.int64)
    
    # INVARIANT CHECK
    n = len(t_all)
    if not (len(uv_all)==n and len(sig_all)==n and len(uv_fit_pipe_all)==n and len(chunk_id_all)==n):
        raise RuntimeError(
          f"Trace length mismatch: t={len(t_all)} uv={len(uv_all)} sig={len(sig_all)} "
          f"uv_fit={len(uv_fit_pipe_all)} chunk_id={len(chunk_id_all)}"
      )
    
    return t_all, uv_all, sig_all, uv_fit_pipe_all, chunk_id_all

def compute_my_tonic(t, uv, sig, uv_fit_pipe, roi_name):
    """
    Computes global robust fit and derive tonic df.
    Performs assertions to ensure correct sourcing.
    """
    # Filter NaNs for fitting
    mask = np.isfinite(uv) & np.isfinite(sig)
    
    if np.sum(mask) < 1000:
        return np.full_like(t, np.nan), np.full_like(t, np.nan), (0,0,False)
        
    uv_clean = uv[mask]
    sig_clean = sig[mask]
    
    # 1. Compute Correlation
    corr = np.corrcoef(uv_clean, sig_clean)[0, 1]
    
    # 2. Compute Global Fit
    slope, intercept, ok, n_used = compute_global_iso_fit_robust(uv_clean, sig_clean)
    
    print(f"[{roi_name}] Global Fit Stats:")
    print(f"  > slope={slope:.4f}, intercept={intercept:.4f}")
    print(f"  > corr(uv,sig)={corr:.4f}")
    print(f"  > n_used={n_used}")
    print(f"  > Tonic iso-fit source: global robust fit across full recording, not pipeline uv_fit")
    
    if not ok:
        print("Robust fit failed.")
        return np.full_like(t, np.nan), np.full_like(t, np.nan), (0,0,False)
        
    # 3. Apply to full trace
    iso_fit_global = apply_global_fit(uv, slope, intercept)
    
    # 4. Strictly Assert Independence
    if iso_fit_global is uv_fit_pipe:
        raise RuntimeError(f"CRITICAL ERROR: iso_fit_global IS uv_fit_pipe (Aliasing detected!)")
    
    # 5. Compute Tonic df
    res = compute_session_tonic_df_from_global(sig, uv, iso_fit_global)
    df_global = res['df']
    
    return df_global, iso_fit_global, (slope, intercept, ok)

def count_phasic_events(dates_t, df_trace, args):
    """
    Detects phasic events on tonic-corrected df trace.
    Returns peak_times (secs) and stats dictionary.
    """
    mask = np.isfinite(dates_t) & np.isfinite(df_trace)
    t0 = dates_t[mask]
    x0 = df_trace[mask]
    
    if len(t0) < 100:
        return np.array([]), {'n_day':0, 'n_night':0, 'rate_day':0, 'rate_night':0, 'ratio':0}

    # Estimate fs
    dt_med = np.median(np.diff(t0))
    if dt_med <= 0:
        fs = 20.0 # Fallback
    else:
        fs = 1.0 / dt_med
        
    # High-pass filter
    sos = scipy.signal.butter(2, args.event_hp_cutoff_hz, btype='highpass', fs=fs, output='sos')
    xhp = scipy.signal.sosfiltfilt(sos, x0)
    
    # Robust Threshold
    med = np.median(xhp)
    mad = np.median(np.abs(xhp - med))
    sigma_robust = 1.4826 * mad
    thresh_val = med + args.event_threshold_k * sigma_robust
    
    # Find Peaks
    dist_samples = int(args.event_min_sep_sec * fs)
    peaks, _ = scipy.signal.find_peaks(xhp, height=thresh_val, distance=dist_samples)
    
    peak_times = t0[peaks]
    
    # Day/Night Classification (Absolute CT)
    # CT = (t_sec / 3600.0) % 24.0
    ct_vals = (peak_times / 3600.0) % 24.0
    is_day = (ct_vals >= args.ct_day_start) & (ct_vals < args.ct_day_end)
    
    n_day = np.sum(is_day)
    n_night = np.sum(~is_day)
    
    # Rates
    day_len_hr = args.ct_day_end - args.ct_day_start
    total_day_hrs = 2.0 * day_len_hr # 2 days
    total_night_hrs = 48.0 - total_day_hrs
    
    rate_day = n_day / total_day_hrs if total_day_hrs > 0 else 0
    rate_night = n_night / total_night_hrs if total_night_hrs > 0 else 0
    
    ratio = (rate_day + 1e-9) / (rate_night + 1e-9)
    
    stats = {
        'n_day': n_day,
        'n_night': n_night,
        'rate_day': rate_day,
        'rate_night': rate_night,
        'ratio': ratio
    }
    
    return peak_times, stats

def compute_tonic_phase_metric(dates_t, df_trace, peak_times, chunk_ids, args):
    """
    Computes phase-anchored circadian enrichment stats using STABLE chunk-level SOS filtering.
    Returns: (t_chunk_centers, x_bp_chunk), stats_dict
    """
    empty_res = ((np.array([]), np.array([])), 
                 {'n_peak':0, 'n_trough':0, 'rate_peak':0, 'rate_trough':0, 'ratio':0,
                  'occ_peak':0, 'occ_trough':0, 'peak_hours':0, 'trough_hours':0})

    # STEP 1: Build Uniform Chunk Grid
    # Use MASKED arrays only
    mask_valid = (chunk_ids >= 0) & np.isfinite(df_trace) & np.isfinite(dates_t)
    chunk_ids_m = chunk_ids[mask_valid]
    dates_t_m = dates_t[mask_valid]
    df_trace_m = df_trace[mask_valid]
    
    unique_chunks = np.unique(chunk_ids_m)
    if len(unique_chunks) < 5:
        return empty_res
        
    t_chunk_list = []
    df_chunk_list = []
    
    for k in unique_chunks:
        mask_k = (chunk_ids_m == k)
        if np.sum(mask_k) < 10:
            continue
            
        t_center = np.nanmedian(dates_t_m[mask_k])
        val = np.nanmedian(df_trace_m[mask_k])
        
        t_chunk_list.append(t_center)
        df_chunk_list.append(val)
        
    if not t_chunk_list:
        return empty_res
        
    t_chunks = np.array(t_chunk_list)
    df_chunks = np.array(df_chunk_list)
    
    # Sort and Check Uniformity
    sort_idx = np.argsort(t_chunks)
    t_chunks = t_chunks[sort_idx]
    df_chunks = df_chunks[sort_idx]
    
    dt_chunks = np.median(np.diff(t_chunks))
    if dt_chunks <= 0: return empty_res
    
    # Verify cadence (30 min = 1800s)
    if abs(dt_chunks - 1800.0) > 60.0:
        print(f"Warning: Chunk interval {dt_chunks:.1f}s deviate from expected 1800s. Proceeding.")
        
    fs_chunk = 1.0 / dt_chunks
    
    # STEP 2: Compute Circadian Phase (SOS Filtering)
    p_lo_hr = args.circ_period_hr - args.circ_bandwidth_hr
    p_hi_hr = args.circ_period_hr + args.circ_bandwidth_hr
    
    if p_lo_hr <= 0:
        print(f"Warning: Low period bound {p_lo_hr} <= 0. Skipping phase metric.")
        return empty_res
        
    f_hi = 1.0 / (p_lo_hr * 3600.0) 
    f_lo = 1.0 / (p_hi_hr * 3600.0) 
    
    if not (0 < f_lo < f_hi < fs_chunk/2):
        print(f"Warning: Freq bounds [{f_lo}, {f_hi}] invalid for fs_chunk={fs_chunk}. Skipping.")
        return empty_res

    # Use SOS (Stable)
    sos = scipy.signal.butter(2, [f_lo/(fs_chunk/2), f_hi/(fs_chunk/2)], btype='bandpass', output='sos')
    # Remove median (DC) before filtering to reduce transients
    x_bp_chunk = scipy.signal.sosfiltfilt(sos, df_chunks - np.nanmedian(df_chunks))
    
    # Hilbert & Alignment
    z = scipy.signal.hilbert(x_bp_chunk)
    phi = np.angle(z)
    
    # Align Peak to 0
    imax = np.argmax(x_bp_chunk)
    shift = phi[imax]
    phi0 = wrap_to_pi(phi - shift)
    
    # STEP 3: Classify Events by Nearest Chunk
    # Map peak_times to nearest chunk index
    # peak_times is in seconds, same base as t_chunks
    
    idxs = np.searchsorted(t_chunks, peak_times)
    # Clip and Refine (Nearest Neighbor)
    idxs = np.clip(idxs, 0, len(t_chunks)-1)
    
    # searchsorted returns i where t_chunks[i-1] <= val < t_chunks[i]
    # We want strictly nearest
    for i in range(len(idxs)):
        idx = idxs[i]
        val = peak_times[i]
        
        # Check prev vs curr
        if idx > 0:
            curr_diff = abs(val - t_chunks[idx])
            prev_diff = abs(val - t_chunks[idx-1])
            if prev_diff < curr_diff:
                idxs[i] = idx - 1
                
    event_phases = phi0[idxs]
    
    # Define halfwidths
    peak_hw_rad = (args.phase_peak_halfwidth_hr / args.circ_period_hr) * 2 * np.pi
    trough_hw_rad = (args.phase_trough_halfwidth_hr / args.circ_period_hr) * 2 * np.pi
    
    # Classify
    d_peak = np.abs(wrap_to_pi(event_phases - 0.0))
    is_peak = d_peak <= peak_hw_rad
    
    d_trough = np.abs(wrap_to_pi(event_phases - np.pi))
    is_trough = d_trough <= trough_hw_rad
    
    n_peak = np.sum(is_peak)
    n_trough = np.sum(is_trough)
    
    # Occupancy Correction (on chunk phases)
    occ_peak_frac = np.mean(np.abs(wrap_to_pi(phi0 - 0.0)) <= peak_hw_rad)
    occ_trough_frac = np.mean(np.abs(wrap_to_pi(phi0 - np.pi)) <= trough_hw_rad)
    
    total_hours = (t_chunks[-1] - t_chunks[0]) / 3600.0
    peak_hours = occ_peak_frac * total_hours
    trough_hours = occ_trough_frac * total_hours
    
    rate_peak = n_peak / peak_hours if peak_hours > 0 else 0
    rate_trough = n_trough / trough_hours if trough_hours > 0 else 0
    ratio = (rate_peak + 1e-9) / (rate_trough + 1e-9)
    
    stats = {
        'n_peak': n_peak,
        'n_trough': n_trough,
        'rate_peak': rate_peak,
        'rate_trough': rate_trough,
        'ratio': ratio,
        'occ_peak': occ_peak_frac,
        'occ_trough': occ_trough_frac,
        'peak_hours': peak_hours,
        'trough_hours': trough_hours
    }
    
    return (t_chunks, x_bp_chunk), stats

def main():
    args = parse_args()
    traces_dir = os.path.join(args.pipeline_out, "traces")
    
    if not os.path.exists(traces_dir):
        print(f"Error: Traces dir not found: {traces_dir}")
        sys.exit(1)
        
    rois = [r.strip() for r in args.rois.split(',')]
    n_rois = len(rois)
    
    # Setup Figure: N ROIs x 3 Rows
    fig, axes = plt.subplots(n_rois, 3, figsize=(18, 4 * n_rois), squeeze=False)
    
    ct_stats_table = []
    tonic_stats_table = []
    
    for i, roi in enumerate(rois):
        print(f"\nProcessing {roi}...")
        try:
            try:
                t, uv, sig, uv_fit_pipe, chunk_ids = load_all_traces(traces_dir, roi)
                print(f"LEN OK roi={roi} n={len(t)}")
            except ValueError:
                print(f"Error loading {roi}...")
                continue

            # Compute Verification Tonic (My Calc)
            df_my, iso_fit_my, (slope, intercept, ok) = compute_my_tonic(t, uv, sig, uv_fit_pipe, roi)
            
            t_hr = t / 3600.0
            
            # Phasic Detection (Verification)
            peak_times, p_stats = count_phasic_events(t, df_my, args)
            ct_stats_table.append({'ROI': roi, **p_stats})
            
            # Tonic Phase Metric (Chunk-Level SOS)
            (t_chunks, x_bp_chunks), t_stats = compute_tonic_phase_metric(t, df_my, peak_times, chunk_ids, args)
            tonic_stats_table.append({'ROI': roi, **t_stats})
            
            # 1. Raw Sig & UV
            ax = axes[i, 0]
            ax.plot(t_hr, sig, color='green', label='Sig Raw', alpha=0.7, linewidth=0.5)
            # Plot UV on twin axis? Or same? Usually same if comparable units, but separate is clearer if scales differ.
            ax2 = ax.twinx()
            ax2.plot(t_hr, uv, color='purple', label='UV Raw', alpha=0.5, linewidth=0.5)
            
            ax.set_title(f"{roi}: Raw Signals")
            ax.set_xlabel("Time (Hours)")
            ax.set_ylabel("Sig (AU)", color='green')
            ax2.set_ylabel("UV (AU)", color='purple')
            
            # 2. Iso Fits (Pipeline vs Global Verification)
            ax = axes[i, 1]
            ax.plot(t_hr, sig, color='green', alpha=0.2, label='Sig') 
            ax.plot(t_hr, uv_fit_pipe, color='orange', label='Pipeline uv_fit (dynamic)', alpha=0.8, linewidth=1)
            ax.plot(t_hr, iso_fit_my, color='blue', linestyle='--', label=f'Global robust fit (tonic)', alpha=0.8, linewidth=1)
            
            ax.set_title(f"{roi}: Iso Fits")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # 3. Tonic dF (Calculated from Global)
            ax = axes[i, 2]
            
            # A) Background CT Shading
            # Shade [0+ct_day_start, 0+ct_day_end] and [24+ct_day_start, 24+ct_day_end]
            # Assumes 48h run approx.
            # Use alpha=0.12 (gray)
            for day_offset in [0, 24]:
                s_hr = day_offset + args.ct_day_start
                e_hr = day_offset + args.ct_day_end
                ax.axvspan(s_hr, e_hr, color='gray', alpha=0.12, label='CT Day' if day_offset==0 else None)
            
            t_finite_mask = np.isfinite(t) & np.isfinite(df_my)
            # Plot df
            ax.plot(t_hr, df_my, color='black', label='Tonic dF (Additive)', linewidth=0.8)
            
            # Safe Y-Limits from 1st-99th percentile of dF
            if np.sum(t_finite_mask) > 100:
                y_lo = np.nanpercentile(df_my[t_finite_mask], 1)
                y_hi = np.nanpercentile(df_my[t_finite_mask], 99)
                margin = (y_hi - y_lo) * 0.2
                ax.set_ylim(y_lo - margin, y_hi + margin)
            
            # B) Circadian Bandpass Component (Scaled Overlay)
            # Interpolate chunk-level BP to valid time points
            if len(t_chunks) > 5 and np.sum(t_finite_mask) > 100:
                x_bp_interp = np.interp(t[t_finite_mask], t_chunks, x_bp_chunks)
                
                # Scale Logic
                df_rng = np.nanpercentile(df_my[t_finite_mask], 95) - np.nanpercentile(df_my[t_finite_mask], 5)
                bp_rng = np.nanpercentile(x_bp_interp, 95) - np.nanpercentile(x_bp_interp, 5) + 1e-9
                scale = 0.25 * df_rng / bp_rng
                
                x_bp_disp = x_bp_interp * scale
                
                ax.plot(t[t_finite_mask]/3600.0, x_bp_disp, color='dodgerblue', alpha=0.6, linewidth=1.5, label='Circadian BP (Scaled)')

            
            # Overlay Peaks
            if len(peak_times) > 0:
                 peak_hr = peak_times / 3600.0
                 peak_idxs = np.searchsorted(t[np.isfinite(t)], peak_times)
                 valid_mask = np.isfinite(t) & np.isfinite(df_my)
                 if np.sum(valid_mask) > 10:
                     y_interp = np.interp(peak_times, t[valid_mask], df_my[valid_mask])
                     ax.scatter(peak_hr, y_interp, color='red', s=10, zorder=5, label='Detected Peaks')

            ax.set_title(f"{roi}: Tonic dF (Additive) (CT day shaded)")
            ax.set_ylabel("dF (AU)")
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Failed to process {roi}: {e}")
            import traceback
            traceback.print_exc()
            axes[i, 0].text(0.5, 0.5, f"Error: {e}", ha='center')
            
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved figure to {args.out}")
    
    if args.print_day_night_events:
        # Table 1: CT Day vs Night
        print("\nPHASIC DAY/NIGHT ENRICHMENT (verification-only; based on high-passed tonic dF peaks)")
        headers = ["ROI", "n_day", "n_night", "rate_day(/h)", "rate_night(/h)", "rate_ratio"]
        row_fmt = "{:<10} {:<8} {:<8} {:<14.2f} {:<14.2f} {:<10.2f}"
        print(" ".join(headers)) 
        
        all_passed_ct = True
        for row in ct_stats_table:
            print(row_fmt.format(
                row['ROI'], 
                int(row['n_day']), 
                int(row['n_night']), 
                row['rate_day'], 
                row['rate_night'], 
                row['ratio']
            ))
            if row['ratio'] < MIN_RATE_RATIO:
                all_passed_ct = False
                
        if all_passed_ct:
            print("\nVERDICT (CT): PASS")
        else:
            print("\nVERDICT (CT): FAIL")
            
        # Table 2: Tonic Phase Enrichment
        print("\nTONIC PHASE ENRICHMENT (phase-anchored; band-pass + Hilbert on tonic dF)")
        headers_t = ["ROI", "n_peak", "n_trough", "rate_peak(/h)", "rate_trough(/h)", "ratio_peak/trough"]
        print(" ".join(headers_t))
        
        all_passed_tonic = True
        for row in tonic_stats_table:
             print(row_fmt.format(
                row['ROI'], 
                int(row['n_peak']), 
                int(row['n_trough']), 
                row['rate_peak'], 
                row['rate_trough'], 
                row['ratio']
            ))
             
             if args.print_phase_events:
                 print(f"  > [DEBUG] {row['ROI']}: occ_peak={row['occ_peak']:.2f}, occ_trough={row['occ_trough']:.2f}, "
                       f"peak_hrs={row['peak_hours']:.2f}, trough_hrs={row['trough_hours']:.2f}")

             if row['ratio'] < args.phase_pass_ratio:
                 all_passed_tonic = False
                 
        if all_passed_tonic:
            print("\nVERDICT (TONIC): PASS_PHASE")
        else:
            print("\nVERDICT (TONIC): FAIL_PHASE")

if __name__ == '__main__':
    main()
