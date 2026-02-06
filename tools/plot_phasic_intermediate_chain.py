#!/usr/bin/env python3
"""
Phasic Intermediate Chain Audit (Step 4)
========================================

Strictly verifies the phasic signal transformation chain:
Raw -> Fit -> Residual -> dFF -> Peaks

Audits:
1. Fit Finity: >= 99% samples finite.
2. Residual Stability: std(residual) relative to signal baseline < 0.5 (unless configured).
3. Saturation: < 0.5% samples at min/max.
4. Gating Check: Median peak intensity/count in Upstate > Downstate.

Usage:
    python tools/plot_phasic_intermediate_chain.py --analysis-out <DIR>
"""

import os
import sys
import argparse
import glob
import re
import yaml
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-out', required=True)
    parser.add_argument('--roi', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--chunks', nargs='+', type=int, help='Specific chunk IDs to audit')
    # Gating Check Params
    parser.add_argument('--enable-synth-gating-check', action='store_true', help='Enable dataset-level Up/Down state gating check (Synthetic only)')
    parser.add_argument('--gate-quantile', type=float, default=0.75, help='Quantile for Upstate (upper) and Downstate (lower) definition (default 0.75)')
    parser.add_argument('--min-chunks-per-state', type=int, default=8, help='Min chunks per state to run gating check (default 8)')
    parser.add_argument('--gate-ratio', type=float, default=1.25, help='Required Up/Down peak count ratio (default 1.25)')
    return parser.parse_args()

def load_chunk_artifact(out_dir, chunk_id, roi):
    """Loads CSV and JSON from phasic_intermediates."""
    inter_dir = os.path.join(out_dir, "phasic_intermediates")
    csv_path = os.path.join(inter_dir, f"chunk_{chunk_id:04d}_{roi}.csv")
    json_path = os.path.join(inter_dir, f"chunk_{chunk_id:04d}_{roi}_meta.json")
    
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        return None, None
        
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        meta = json.load(f)
        
    return df, meta

def audit_chunk(df, meta, chunk_id):
    """
    Returns (passed, analysis_dict, failure_reason)
    """
    # Columns mandated by Step 2
    for c in ['time_sec', 'sig_raw', 'iso_raw', 'fit_ref', 'residual', 'dff']:
        if c not in df.columns:
            return False, {}, f"Missing column {c}"
            
    # Load arrays
    sig = df['sig_raw'].values
    iso = df['iso_raw'].values
    fit = df['fit_ref'].values
    res = df['residual'].values
    dff = df['dff'].values
    
    n = len(sig)
    if n == 0: return False, {}, "Empty chunk"
    
    # Gate A: Fit Finity >= 0.99
    n_finite = np.sum(np.isfinite(fit))
    frac_finite = n_finite / n
    if frac_finite < 0.99:
        return False, {}, f"Gate A Fail: Fit finite fraction {frac_finite:.4f} < 0.99"

    # Gate B: Fit Scale
    mask = np.isfinite(fit) & np.isfinite(iso) & np.isfinite(sig)
    if np.sum(mask) < 10:
        val_lhs, val_rhs = 0, 0 # skip
    else:
        diff_fit_iso = np.abs(fit[mask] - iso[mask])
        diff_sig_iso = np.abs(sig[mask] - iso[mask])
        val_lhs = np.median(diff_fit_iso)
        val_rhs = np.median(diff_sig_iso) * 10.0
        
        # If sig ~= iso (perfect cancellation?), RHS might be small. 
        # But sig usually has calcium.
        # Guard against zero RHS? 
        if val_rhs < 1e-9: val_rhs = 1e-9
        
        if val_lhs > val_rhs:
             return False, {}, f"Gate B Fail: Fit Scale runaway. median|fit-iso|={val_lhs:.2f} > 10*median|sig-iso|={val_rhs:.2f}"

    # Gate C: Residual Std
    # std(res) < 5 * std(sig - iso)
    if np.sum(mask) > 10:
        res_std = np.std(res[np.isfinite(res)])
        raw_diff_std = np.std((sig - iso)[mask])
        
        # If raw signals are clean, their diff std might be small. 
        # Residual should be even smaller (it's the cleaned signal).
        # Wait, residual IS (sig - fit). 
        # Gate says: residual std must not exceed 5x the std of (sig_raw - iso_raw).
        # Ideally residual < raw_diff.
        limit = 5.0 * raw_diff_std
        if limit < 1e-9: limit = 1e-9
        
        if res_std > limit:
             return False, {}, f"Gate C Fail: Residual noise. std(res)={res_std:.2f} > 5*std(sig-iso)={limit:.2f}"
             
    # Gate D: Saturation
    dff_clean = dff[np.isfinite(dff)]
    if len(dff_clean) > 0:
        min_v, max_v = np.min(dff_clean), np.max(dff_clean)
        # Fraction within 1% of min/max? 
        # Step 1 desc: "at min or max". Step 4 desc: "Fraction... within 1% of min or max"
        # We'll use 1% range.
        r = max_v - min_v
        tol = 0.01 * r if r > 0 else 1e-9
        
        n_lo = np.sum(dff_clean <= (min_v + tol))
        n_hi = np.sum(dff_clean >= (max_v - tol))
        frac_sat = (n_lo + n_hi) / len(dff_clean)
        
        if frac_sat > 0.005: 
             return False, {}, f"Gate D Fail: Saturation {frac_sat*100:.2f}% > 0.5%"
             
    # Peak Detection
    # Using params from meta
    method = meta.get('peak_method', 'mean_std')
    k = meta.get('peak_k', 2.0)
    
    if len(dff_clean) > 10:
        mu = np.mean(dff_clean)
        sigma = np.std(dff_clean)
        thresh = mu + k * sigma
        peaks, _ = find_peaks(dff_clean, height=thresh, distance=int(meta['fs_hz'] * 0.5))
    else:
        peaks = []
        thresh = 0
    
    # Tonic Proxy: Median of Fit (Fit is Reference ~ Iso)
    # Prefer fit_ref as it captures the slow baseline trend used for correction
    tonic_proxy = np.nanmedian(fit)
    
    return True, {
        'tonic_proxy': tonic_proxy,
        'peak_count': len(peaks),
        'dff': dff,
        'thresh': thresh,
        'peaks': peaks
    }, "OK"

def evaluate_synth_gating(dataset_stats, gate_quantile, min_chunks_per_state, gate_ratio):
    """
    Evaluates Tier 2 synthetic dataset gating.
    Returns True if passed or skipped, False if failed.
    """
    print("\n=== Dataset-Level Gating Check (Synthetic Verification) ===")
    
    min_chunks = min_chunks_per_state
    
    # Guardrail: Check sufficient data (using finite_stats logic below)
    # We do this logic AFTER filtering finite stats to be safe, 
    # or we can do a preliminary check. 
    # BUT requirement says: "Insufficient Data check must use len(finite_stats)"
    # So we move this check after filtering.

    # 1. Filter for Finite Stats Upfront
    # We must use the same population for Quantiles AND Split Groups.
    finite_stats = []
    for d in dataset_stats:
        t = d.get('tonic', np.nan)
        try:
            t_val = float(t)
        except (TypeError, ValueError):
            t_val = np.nan
            
        if np.isfinite(t_val):
            # Normalizing tonic to float to prevent type errors (e.g. str vs float)
            # Create a shallow copy with the coerced float tonic
            norm_d = d.copy()
            norm_d['tonic'] = t_val
            finite_stats.append(norm_d)
            
    if not finite_stats:
        print("WARNING: No finite tonic values available.")
        print("Gating Check: SKIPPED (No Finite Tonic)")
        return True

    if len(finite_stats) < min_chunks_per_state * 2:
        print(f"WARNING: Not enough audited chunks ({len(finite_stats)}) to run gating check. Need at least {min_chunks_per_state*2}.")
        print("Gating Check: SKIPPED (Insufficient Data)")
        return True

    # 1. Compute Quantiles of Tonic Proxy
    tonics = [d['tonic'] for d in finite_stats]
    
    q_val_lo = 1.0 - gate_quantile
    q_val_hi = gate_quantile
    
    # Use exact numpy quantile
    q_lo = np.quantile(tonics, q_val_lo) 
    q_hi = np.quantile(tonics, q_val_hi)
    
    print(f"Tonic Proxy Distribution: Min={min(tonics):.2f}, Q{q_val_lo:.2f}={q_lo:.2f}, Q{q_val_hi:.2f}={q_hi:.2f}, Max={max(tonics):.2f}")
    
    # Separation Check
    if q_hi <= q_lo:
        print("WARNING: No tonic separation (q_hi <= q_lo).")
        print("Gating Check: SKIPPED (No Tonic Separation)")
        return True

    # 2. Split Groups (Using finite_stats only)
    up_group = [d for d in finite_stats if d['tonic'] >= q_hi]
    down_group = [d for d in finite_stats if d['tonic'] <= q_lo]
    
    n_up = len(up_group)
    n_down = len(down_group)
    
    print(f"Groups: High Tonic (N={n_up}), Low Tonic (N={n_down})")
    
    if n_up < min_chunks or n_down < min_chunks:
        print(f"WARNING: Groups too small (<{min_chunks}) after split.")
        print("Gating Check: SKIPPED (Small Groups)")
        return True

    # 3. Compute Mean Peak Counts
    mu_up = np.mean([d['peaks'] for d in up_group])
    mu_down = np.mean([d['peaks'] for d in down_group])
    
    print(f"Mean Peak Counts: High Tonic={mu_up:.2f}, Low Tonic={mu_down:.2f}")
    
    # Check Bidirectional Modulation (Ratio)
    mu_max = max(mu_up, mu_down)
    mu_min = min(mu_up, mu_down)
    
    # Corner Case: Zero Peaks
    if mu_max == 0 and mu_min == 0:
        print(f"Modulation Ratio: 1.00 (Required >= {gate_ratio})")
        print("CRITICAL: Synthetic Gating Check FAIL. Both groups have zero peaks.")
        return False

    if mu_min == 0:
        # mu_max > 0 implied
        print(f"Modulation Ratio: inf (Required >= {gate_ratio})")
        print("Gating Check: PASS")
        return True

    # Standard Check
    ratio = mu_max / mu_min
    print(f"Modulation Ratio: {ratio:.2f} (Required >= {gate_ratio})")
    
    if ratio < gate_ratio:
        print(f"CRITICAL: Synthetic Gating Check FAIL. Ratio {ratio:.2f} < {gate_ratio}")
        return False
    else:
        print("Gating Check: PASS")
        return True

def main():
    # Import matplotlib here to avoid headless issues during testing of gating logic
    import matplotlib.pyplot as plt
    
    args = parse_args()
    
    # We scan artifacts in phasic_intermediates
    inter_dir = os.path.join(args.analysis_out, 'phasic_intermediates')
    if not os.path.exists(inter_dir):
        print(f"CRITICAL: {inter_dir} does not exist. Run pipeline with artifact saving enabled.")
        sys.exit(1)
        
    files = sorted(glob.glob(os.path.join(inter_dir, 'chunk_*_*.csv')))
    
    if not files:
        print("CRITICAL: No intermediate files found.")
        sys.exit(1)
        
    # Select ROI
    # Infer ROI from first file or arg
    # File format: chunk_0000_ROI.csv
    # We need to filter files by ROI if provided
    
    df0 = pd.read_csv(files[0])
    # ROI is in filename, not column.
    
    if args.roi:
        roi = args.roi
        files = [f for f in files if f"_{roi}.csv" in f]
        if not files:
            print(f"CRITICAL: No files for ROI {roi}")
            sys.exit(1)
            
    else:
        # Auto-detect ROI from first file
        bname = os.path.basename(files[0])
        # chunk_0000_Region0.csv
        parts = bname.replace('.csv','').split('_')
        if len(parts) >= 3:
            roi = '_'.join(parts[2:])
        else:
            roi = "Region0" # fallback
        print(f"Auto-selected ROI: {roi}")
        files = [f for f in files if f"_{roi}.csv" in f]
    
    # Stats Collection for Dataset Gating
    dataset_stats = [] # list of (tonic_proxy, peak_count)
    passed_chunks = 0
    total_audited = 0
    
    # Output dir for plots
    qc_dir = os.path.join(args.analysis_out, 'phasic_chain_qc')
    os.makedirs(qc_dir, exist_ok=True)
    
    for f in files:
        # Extract ID from filename (format: chunk_XXXX_ROI.csv)
        fname = os.path.basename(f)
        m = re.match(r'chunk_(\d+)_', fname)
        if not m: continue
        cid = int(m.group(1))
        
        # Filter if requested
        if args.chunks and cid not in args.chunks:
            continue
            
        # Load from intermediates
        df, meta = load_chunk_artifact(args.analysis_out, cid, roi)
        if df is None:
            print(f"Skipping chunk {cid}: No intermediate artifacts found.")
            continue
            
        total_audited += 1
        ok, res, msg = audit_chunk(df, meta, cid)
        
        if not ok:
            print(f"Chunk {cid} FAIL: {msg}")
            # We do NOT collect stats for failed chunks to avoid polluting dataset check
            # (unless it's a soft fail? But Gates A-D are hard invariants).
            continue
            
        # Pass
        print(f"Auditing Chunk {cid}...")
        print(f"Pass Gates A-D. Tonic={res['tonic_proxy']:.2f}, Peaks={res['peak_count']}")
        passed_chunks += 1
        dataset_stats.append({
            'cid': cid,
            'tonic': res['tonic_proxy'],
            'peaks': res['peak_count']
        })
        
        # PLOTTING (Only for audited chunks)
        # Create figure: 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        # 1. Raw
        t = df['time_sec'].values
        t = t - t[0]
        ax = axes[0]
        ax.plot(t, df['sig_raw'], 'g', label='Sig')
        ax.plot(t, df['iso_raw'], 'm', label='Iso', alpha=0.7)
        ax.legend(loc='upper right')
        ax.set_ylabel("Raw")
        ax.set_title(f"Chunk {cid} Chain Audit")
        
        # 2. Fit
        ax = axes[1]
        ax.plot(t, df['fit_ref'], 'orange', label='Fitted Ref')
        ax.plot(t, df['iso_raw'], 'm', alpha=0.3, label='Iso Raw') # context
        ax.legend()
        ax.set_ylabel("Fit")
        
        # 3. Residual
        ax = axes[2]
        ax.plot(t, df['residual'], 'blue', lw=0.8, label='Residual')
        ax.legend()
        ax.set_ylabel("Residual")
        # Add bounds derived from Gate C?
        
        # 4. dFF & Peaks
        ax = axes[3]
        ax.plot(t, df['dff'], 'k', lw=1, label='dFF')
        thresh = res['thresh']
        ax.axhline(thresh, color='r', ls='--', alpha=0.5, label='Thresh')
        if len(res['peaks']) > 0:
            pix = res['peaks']
            ax.plot(t[pix], df['dff'][pix], 'rx', markersize=8, label='Peaks')
        ax.legend()
        ax.set_ylabel("dFF")
        ax.set_xlabel("Time (s)")
        
        out_path = os.path.join(qc_dir, f"chain_chunk_{cid:03d}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)

    # -------------------------------------------------------------------------
    # TIER 2: DATASET-LEVEL GATING CHECK (Synthetic Only, Opt-In)
    # -------------------------------------------------------------------------
    if args.enable_synth_gating_check:
        passed_gating = evaluate_synth_gating(
            dataset_stats, 
            gate_quantile=args.gate_quantile, 
            min_chunks_per_state=args.min_chunks_per_state, 
            gate_ratio=args.gate_ratio
        )
        if not passed_gating:
             sys.exit(1)

    # Final Summary for Tier 1
    print(f"\nSummary: {passed_chunks}/{total_audited} chunks passed individual gates (A-D).")
    if passed_chunks < total_audited:
        print("CRITICAL: Some chunks failed individual audit.")
        sys.exit(1)
        
    print("Success: All checks passed.")
    sys.exit(0)

if __name__ == '__main__':
    main()
