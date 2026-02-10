#!/usr/bin/env python3
"""
Run Full Pipeline & Package Deliverables
========================================

Orchestrates the full photometry pipeline and packages outputs for delivery.
Enforces deterministic session duration and strict output naming.

Usage:
    python tools/run_full_pipeline_deliverables.py --input <IN> --out <OUT> --config <CFG> --format <FMT>
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
import glob
import re
import pandas as pd
import numpy as np
import time
from datetime import datetime

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    return cmd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--format', required=True, choices=['rwd', 'npm', 'auto'])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--sessions-per-hour', type=int, help="Force sessions per hour (integer)")
    parser.add_argument('--session-duration-s', type=float, help="Recording duration in seconds (data length per chunk). If provided, validated against traces.")
    parser.add_argument('--smooth-window-s', type=float, default=1.0)
    parser.add_argument('--validate-only', action='store_true',
                        help="Validate inputs and print the command that would run, then exit.")
    return parser.parse_args()

def validate_inputs(args):
    """Cheap preflight checks. Raises RuntimeError on any problem."""
    # Input path
    if not os.path.isdir(args.input):
        raise RuntimeError(f"Input directory does not exist or is not a directory: {args.input}")

    # Config path
    if not os.path.isfile(args.config):
        raise RuntimeError(f"Config file does not exist or is not a file: {args.config}")

    # Format (already constrained by argparse choices, but belt-and-suspenders)
    if args.format not in ('rwd', 'npm', 'auto'):
        raise RuntimeError(f"Invalid format: {args.format}")

    # sessions_per_hour
    if args.sessions_per_hour is not None:
        if args.sessions_per_hour < 1:
            raise RuntimeError(f"--sessions-per-hour must be >= 1, got {args.sessions_per_hour}")

    # session_duration_s
    if args.session_duration_s is not None:
        if args.session_duration_s <= 0:
            raise RuntimeError(f"--session-duration-s must be > 0, got {args.session_duration_s}")

    # Impossible schedule (only when both are provided)
    if args.sessions_per_hour is not None and args.session_duration_s is not None:
        stride_s = 3600.0 / args.sessions_per_hour
        if args.session_duration_s > stride_s + 1e-6:
            raise RuntimeError(
                f"Impossible schedule: Duration {args.session_duration_s:.2f}s > "
                f"Stride {stride_s:.2f}s (SPH={args.sessions_per_hour}).")

def main():
    args = parse_args()

    # 0. Validate-only preflight
    if args.validate_only:
        try:
            validate_inputs(args)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

        # Build the argv that a real run would use
        argv = [sys.executable, "tools/run_full_pipeline_deliverables.py",
                "--input", args.input,
                "--out", args.out,
                "--config", args.config,
                "--format", args.format]
        if args.overwrite:
            argv.append("--overwrite")
        if args.sessions_per_hour is not None:
            argv.extend(["--sessions-per-hour", str(args.sessions_per_hour)])
        if args.session_duration_s is not None:
            argv.extend(["--session-duration-s", str(args.session_duration_s)])
        argv.extend(["--smooth-window-s", str(args.smooth_window_s)])

        print("VALIDATE-ONLY: OK")
        print(f"VALIDATE-ONLY: argv={json.dumps(argv)}")
        sys.exit(0)

    # 1. Validation & Setup
    if os.path.exists(args.out):
        if not args.overwrite:
            print(f"Error: Output directory {args.out} exists. Use --overwrite.")
            sys.exit(1)
            
        # Robust rmtree
        max_retries = 5
        for i in range(max_retries):
            try:
                shutil.rmtree(args.out)
                break
            except OSError as e:
                if i == max_retries - 1:
                    print(f"Error: Failed to delete {args.out} after retries: {e}")
                    sys.exit(1)
                time.sleep(0.5 * (i + 1))
    
    os.makedirs(args.out)
    analysis_dir = os.path.join(args.out, '_analysis')
    tonic_out = os.path.join(analysis_dir, 'tonic_out')
    phasic_out = os.path.join(analysis_dir, 'phasic_out')
    
    manifest = {
        'tool': 'run_full_pipeline_deliverables',
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'commands': [],
        'regions': [],
        'deliverables': {} 
    }
    
    try:
        # 2. Run Analysis
        # Tonic
        cmd_tonic = [sys.executable, 'analyze_photometry.py', 
                     '--input', args.input, 
                     '--out', tonic_out, 
                     '--config', args.config, 
                     '--mode', 'tonic', 
                     '--format', args.format, 
                     '--recursive', '--overwrite']
        manifest['commands'].append(run_cmd(cmd_tonic))
        
        # Phasic
        cmd_phasic = [sys.executable, 'analyze_photometry.py', 
                      '--input', args.input, 
                      '--out', phasic_out, 
                      '--config', args.config, 
                      '--mode', 'phasic', 
                      '--format', args.format, 
                      '--recursive', '--overwrite']
        manifest['commands'].append(run_cmd(cmd_phasic))
        
        # 3. Determine Global Sessions Per Hour (Deterministic)
        trace_files = sorted(glob.glob(os.path.join(phasic_out, 'traces', 'chunk_*.csv')))
        if not trace_files:
            raise RuntimeError("No phases traces found.")
            
        # Inspect first trace for duration
        df0 = pd.read_csv(trace_files[0])
        if 'time_sec' not in df0.columns or len(df0) < 2:
             raise RuntimeError(f"Trace {trace_files[0]} missing time_sec or too short.")
             
        time_sec = df0['time_sec'].values
        trace_duration_s = time_sec[-1] - time_sec[0]
        
        if not np.isfinite(trace_duration_s) or trace_duration_s <= 0:
            raise RuntimeError(f"Invalid trace_duration_s: {trace_duration_s}")

        # Stride Inference
        stride_s = None
        
        # Method: Explicit Argument (Strict)
        # We do NOT infer from duration for duty-cycled data.
        if args.sessions_per_hour:
             sessions_per_hour = args.sessions_per_hour
             stride_s = 3600.0 / sessions_per_hour
        else:
             # If we had reliable absolute timestamps, we could infer stride here.
             # Since we don't assume them yet, and we strictly cannot use duration,
             # we must fail.
             raise RuntimeError("Cannot infer session stride (duty-cycled acquisition) without --sessions-per-hour. Please provide it explicitly.")

        # Validation
        computed_sph = int(round(3600.0 / stride_s))
        if computed_sph < 1 or computed_sph > 12: # Reasonable limits
             raise RuntimeError(f"Invalid sessions per hour: {computed_sph}")
             
        if abs(stride_s * computed_sph - 3600.0) > 2.0:
             raise RuntimeError(f"Stride {stride_s:.2f}s not compatible with integer sessions/hr {computed_sph}")
             
        # Definition:
        # session_duration_s := within-session trace length
        # session_stride_s := time between starts (cadence)
        
        if args.session_duration_s is not None:
             if args.session_duration_s <= 0:
                  raise RuntimeError(f"Provided session duration must be > 0, got {args.session_duration_s}")
                  
             # Validate against trace
             tol = max(2.0, 0.005 * args.session_duration_s)
             diff = abs(trace_duration_s - args.session_duration_s)
             if diff > tol:
                  raise RuntimeError(f"Session Duration Mismatch! Provided: {args.session_duration_s:.2f}s, Trace: {trace_duration_s:.2f}s (Diff: {diff:.2f}s, Tol: {tol:.2f}s). File: {trace_files[0]}")
             
             session_duration_s = args.session_duration_s
        else:
             session_duration_s = trace_duration_s
             
        # Impossible Schedule Check
        # Duration cannot exceed Stride
        if session_duration_s > (stride_s + 1e-6):
             raise RuntimeError(f"Impossible schedule: Duration {session_duration_s:.2f}s > Stride {stride_s:.2f}s (SPH={sessions_per_hour}).")
             
        sessions_per_hour = computed_sph
        manifest['sessions_per_hour'] = sessions_per_hour
        manifest['session_duration_s'] = session_duration_s
        manifest['session_stride_s'] = stride_s
        print(f"Deterministic Sessions Per Hour: {sessions_per_hour} (Stride={stride_s:.1f}s, Dur={session_duration_s:.1f}s)")

        # 4. Diagnostics & Per-Region Processing
        feats_csv = os.path.join(phasic_out, 'features', 'features.csv')
        df_feat = pd.read_csv(feats_csv)
        regions = sorted(df_feat['roi'].unique())
        manifest['regions'] = regions
        
        for roi in regions:
            print(f"Processing ROI: {roi}")
            reg_dir = os.path.join(args.out, roi)
            os.makedirs(reg_dir, exist_ok=True)
            files_written = []
            
            roi_feat = df_feat[df_feat['roi'] == roi].copy()
            roi_feat = roi_feat.sort_values('chunk_id')
            
            # Authoritative Mapping
            # (chunk_id) -> (day, hour, session)
            # We assume session indexing matches chunk list order here for consistency with fallbacks
            # unless we have robust timestamps.
            # Start strict: Index-based Fallback as default (matches user request "use ordered chunks")
            # If timestamps present, we'd use them, but strictly for determining day 0.
            
            # Diagnostic Selection (Day 0, H 12, S 0)
            # D0/H12 starts at index 12 * sessions_per_hour
            diag_idx = 12 * sessions_per_hour
            candidates = roi_feat['chunk_id'].values
            
            if len(candidates) > diag_idx:
                cid_diag = candidates[diag_idx]
            else:
                cid_diag = candidates[0]
                
            manifest['deliverables'][roi] = {'diagnostic_chunk_id': int(cid_diag)}
            
            # A. Phasic Correction Impact (3-Panel)
            cmd_impact = [sys.executable, 'tools/plot_phasic_correction_impact.py',
                          '--analysis-out', phasic_out,
                          '--roi', roi,
                          '--chunk-id', str(cid_diag),
                          '--out', os.path.join(reg_dir, "phasic_correction_impact.png")]
            manifest['commands'].append(run_cmd(cmd_impact))
            files_written.append("phasic_correction_impact.png")
            
            # Correction Data CSV
            c_csv = os.path.join(phasic_out, 'phasic_intermediates', f"chunk_{cid_diag:04d}_{roi}.csv")
            if not os.path.exists(c_csv): # Try unpadded
                 c_csv = os.path.join(phasic_out, 'phasic_intermediates', f"chunk_{cid_diag}_{roi}.csv")
            
            if os.path.exists(c_csv):
                df_c = pd.read_csv(c_csv)
                rename_map = {'time_sec': 't_s', 'fit_ref': 'iso_fit_dynamic', 'dff': 'dff_dynamic'}
                df_c = df_c.rename(columns=rename_map)
                keep = ['t_s', 'sig_raw', 'iso_raw', 'iso_fit_dynamic', 'dff_dynamic', 'region', 'chunk_id']
                df_c['region'] = roi
                df_c['chunk_id'] = cid_diag
                df_c[keep].to_csv(os.path.join(reg_dir, "phasic_correction_impact_session.csv"), index=False)
                files_written.append("phasic_correction_impact_session.csv")

            # B. Tonic Overview
            # Run Plotter
            cmd_tonic_roi = [sys.executable, 'tools/plot_tonic_48h.py', 
                             '--analysis-out', tonic_out, '--roi', roi]
            run_cmd(cmd_tonic_roi)
            
            src_tonic = os.path.join(tonic_out, 'tonic_qc', f"tonic_48h_overview_{roi}.png")
            if os.path.exists(src_tonic):
                shutil.copy2(src_tonic, os.path.join(reg_dir, "tonic_overview.png"))
                files_written.append("tonic_overview.png")
                
            # Tonic CSV
            tonic_files = sorted(glob.glob(os.path.join(tonic_out, 'traces', 'chunk_*.csv')))
            t_rows = []
            
            # Read fs output from meta if possible
            fs_tonic = 20.0
            
            for i, tf in enumerate(tonic_files):
                tdf = pd.read_csv(tf)
                col_d = f"{roi}_deltaF"
                if col_d in tdf.columns:
                    # Construct absolute time using STRIDE, not duration
                    # Gaps are real.
                    n_pts = len(tdf)
                    t_abs = (i * stride_s) + tdf['time_sec'].values
                    df_sub = pd.DataFrame({
                        'time_hours': t_abs / 3600.0,
                        'tonic_df': tdf[col_d].values
                    })
                    t_rows.append(df_sub)
                    
            if t_rows:
                full_tonic = pd.concat(t_rows, ignore_index=True)
                full_tonic.to_csv(os.path.join(reg_dir, "tonic_df_timeseries.csv"), index=False)
                files_written.append("tonic_df_timeseries.csv")

            # C. Phasic Time Series (Plots & CSV)
            # Use Modified Summary Plotter
            # Outputs to temp dir for ROI, then move
            ts_dir = os.path.join(phasic_out, f'viz_{roi}')
            cmd_ts = [sys.executable, 'tools/plot_phasic_time_series_summary.py',
                      '--analysis-out', phasic_out,
                      '--roi', roi,
                      '--sessions-per-hour', str(sessions_per_hour),
                      '--session-duration-s', str(session_duration_s),
                      '--out-dir', ts_dir,
                      '--export-csv']
            manifest['commands'].append(run_cmd(cmd_ts))
            
            # Copy Results
            pairs = [
                ("fig_phasic_peak_rate_timeseries.png", "phasic_peak_rate_timeseries.png"),
                ("fig_phasic_auc_timeseries.png", "phasic_auc_timeseries.png"),
                ("phasic_peak_rate_timeseries.csv", "phasic_peak_rate_timeseries.csv"),
                ("phasic_auc_timeseries.csv", "phasic_auc_timeseries.csv")
            ]
            for src_name, dst_name in pairs:
                s = os.path.join(ts_dir, src_name)
                if os.path.exists(s):
                    shutil.copy2(s, os.path.join(reg_dir, dst_name))
                    files_written.append(dst_name)

            # D. Per-Day Plots (Sig/Iso, dFF, Stacked)
            
            # 1. dFF Grid
            # Use unique output dir to prevent overwriting
            qc_dir = os.path.join(phasic_out, f'qc_dff_{roi}')
            cmd_qc = [sys.executable, 'tools/plot_phasic_qc_grid.py',
                      '--analysis-out', phasic_out,
                      '--roi', roi,
                      '--mode', 'dff',
                      '--sessions-per-hour', str(sessions_per_hour),
                      '--output-dir', qc_dir]
            run_cmd(cmd_qc)
            
            # 2. Sig/Iso Grid
            sess_dir = os.path.join(phasic_out, f'session_qc_{roi}')
            # Note: plot_session_grid might not accept output-dir? 
            # We checked: it uses <analysis-out>/session_qc hardcoded? 
            # Need to double check. If hardcoded, we have race condition/overwrite.
            # The prompt says: "Ensure plot_session_grid ... produces day-wise grids ... for the ROI".
            # It saves to 'session_qc/day_{d}_raw_iso_{roi}.png'. The ROI is in filename. Safe.
            # But we must ensure it uses correct sessions_per_hour.
            cmd_sess = [sys.executable, 'tools/plot_session_grid.py',
                        '--analysis-out', phasic_out,
                        '--roi', roi,
                        '--sessions-per-hour', str(sessions_per_hour),
                        '--session-duration-s', str(session_duration_s)]
            run_cmd(cmd_sess)
            
            # 3. Stacked
            cmd_stack = [sys.executable, 'tools/plot_phasic_stacked_day_smoothed.py',
                         '--analysis-out', phasic_out,
                         '--roi', roi,
                         '--out-dir', reg_dir,
                         '--sessions-per-hour', str(sessions_per_hour),
                         '--smooth-window-s', str(args.smooth_window_s)]
            manifest['commands'].append(run_cmd(cmd_stack))
            
            # Collect Per-Day Files
            days_generated = set()
            days_dff = set()
            days_sig_iso = set()
            
            # Copy dFF
            # Use strict ROI-specific directory
            qc_dir_roi = os.path.join(phasic_out, f'qc_dff_{roi}')
            for f in glob.glob(os.path.join(qc_dir_roi, "day_*.png")):
                # day_000.png
                m = re.match(r'day_(\d+)\.png', os.path.basename(f))
                if m:
                    day_idx = m.group(1)
                    dst = f"phasic_dFF_day_{day_idx}.png"
                    shutil.copy2(f, os.path.join(reg_dir, dst))
                    files_written.append(dst)
                    days_dff.add(day_idx)
            
            # Copy Sig/Iso
            sess_out_base = os.path.join(phasic_out, 'session_qc')
            for f in glob.glob(os.path.join(sess_out_base, f"day_*_raw_iso_{roi}.png")):
                 m = re.match(r'day_(\d+)_raw_iso_', os.path.basename(f))
                 if m:
                     day_idx = m.group(1)
                     dst = f"phasic_sig_iso_day_{day_idx}.png"
                     shutil.copy2(f, os.path.join(reg_dir, dst))
                     files_written.append(dst)
                     days_sig_iso.add(day_idx)
            
            # Stacked are already in reg_dir, just verify
            days_stacked = set()
            for f in glob.glob(os.path.join(reg_dir, "phasic_stacked_day_*.png")):
                files_written.append(os.path.basename(f))
                m = re.match(r'phasic_stacked_day_(\d+)\.png', os.path.basename(f))
                if m:
                    days_stacked.add(m.group(1))

            # Sort sets for consistency check
            s_dff = sorted(list(days_dff))
            s_sig = sorted(list(days_sig_iso))
            s_stk = sorted(list(days_stacked))
            
            manifest['deliverables'][roi]['days_dff'] = s_dff
            manifest['deliverables'][roi]['days_sig_iso'] = s_sig
            manifest['deliverables'][roi]['days_stacked'] = s_stk
            
            if not (s_dff == s_sig == s_stk):
                 raise RuntimeError(f"Inconsistent day sets for ROI {roi}: DFF={s_dff}, SigIso={s_sig}, Stacked={s_stk}")
                 
            manifest['deliverables'][roi]['files'] = sorted(list(set(files_written)))
            manifest['deliverables'][roi]['days_generated'] = s_dff

        # 5. Write Manifest
        with open(os.path.join(args.out, 'MANIFEST.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print("Deliverables Package Complete.")
        
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
