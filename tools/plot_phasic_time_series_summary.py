
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Plot phasic event frequency and AUC over time.")
    parser.add_argument('--analysis-out', required=True, help="Path to analysis output directory (containing features/features.csv)")
    parser.add_argument('--roi', help="ROI to plot. Defaults to first ROI alphabetically.")
    parser.add_argument('--sessions-per-hour', type=int, default=2, help="Expected sessions per hour for fallback timing.")
    parser.add_argument('--session-duration-s', type=float, default=None, help="Explicit session duration (window seconds).")
    parser.add_argument('--out-dir', help="Output directory. Defaults to <analysis-out>/viz")
    parser.add_argument('--dpi', type=int, default=150, help="DPI for output figures")
    parser.add_argument('--export-csv', action='store_true', help="Export CSVs of plotting data")
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        
        # Validation: sessions-per-hour must be >= 1
        if args.sessions_per_hour < 1:
            raise RuntimeError("--sessions-per-hour must be >= 1")
            
        # 1. Setup paths
        features_path = os.path.join(args.analysis_out, 'features', 'features.csv')
        if not os.path.exists(features_path):
            raise RuntimeError(f"features.csv not found at {features_path}")
            
        out_dir = args.out_dir if args.out_dir else os.path.join(args.analysis_out, 'viz')
        os.makedirs(out_dir, exist_ok=True)
        
        # 2. Load Data
        try:
            df = pd.read_csv(features_path)
        except Exception as e:
            raise RuntimeError(f"Error reading features.csv: {e}")
            
        # 3. Validate Columns
        required_cols = ['chunk_id', 'roi', 'peak_count', 'auc']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns in features.csv: {missing}")
            
        # Enforce numeric chunk_id
        df['chunk_id'] = pd.to_numeric(df['chunk_id'], errors='coerce')
        if df['chunk_id'].isna().any():
            raise RuntimeError("chunk_id contains non-numeric values after coercion")
            
        # Optional source_file
        if 'source_file' not in df.columns:
            df['source_file'] = np.nan
            
        # 4. Select ROI
        if args.roi:
            selected_roi = args.roi
        else:
            unique_rois = sorted(df['roi'].dropna().unique())
            if not unique_rois:
                raise RuntimeError("No ROIs found in features.csv")
            selected_roi = unique_rois[0]
            
        print(f"Selected ROI: {selected_roi}")
        
        # 5. Filter Data
        roi_df = df[df['roi'] == selected_roi].copy()
        if roi_df.empty:
            raise RuntimeError(f"No data found for ROI: {selected_roi}")
            
        # 6. Determine File Order and Time Axis
        # Priority 1: Datetime parsing
        def parse_dt_str(val):
            # Attempt to parse basename if it matches strictly %Y_%m_%d-%H_%M_%S
            if pd.isna(val): return None
            s_val = str(val)
            try:
                base = os.path.basename(s_val)
                root, _ = os.path.splitext(base)
                return datetime.strptime(root, "%Y_%m_%d-%H_%M_%S")
            except:
                return None

        # Two-stage parsing:
        # 1. Try strict format on basename (fast, matches pipeline output)
        dt_series_1 = roi_df['source_file'].apply(parse_dt_str)
        
        # 2. Convert to pandas Timestamp directly (handles other formats)
        dt_series_1 = pd.to_datetime(dt_series_1, errors='coerce')
        dt_series_2 = pd.to_datetime(roi_df['source_file'], errors='coerce')
        dt_final = dt_series_1.fillna(dt_series_2)
        roi_df['dt'] = dt_final
        
        valid_count = roi_df['dt'].notna().sum()
        total_count = len(roi_df)
        parse_rate = valid_count / total_count if total_count > 0 else 0
        
        use_datetime = (parse_rate >= 0.90)
        
        # Verify consistency if session_duration_s is provided
        if args.session_duration_s:
            # We need to check against actual trace duration if possible.
            # Traces are in <analysis_out>/traces/chunk_*.csv
            # We can pick the first chunk from features.csv to verify.
            first_chunk_id = roi_df['chunk_id'].iloc[0]
            
            # 1. Resolve Trace Path (Robust search)
            trace_path = None
            try:
                cid_val = int(first_chunk_id)
                candidates = [
                    os.path.join(args.analysis_out, 'traces', f"chunk_{cid_val:04d}.csv"),
                    os.path.join(args.analysis_out, 'traces', f"chunk_{cid_val:03d}.csv"),
                    os.path.join(args.analysis_out, 'traces', f"chunk_{cid_val}.csv")
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        trace_path = cand
                        break
            except:
                pass # If chunk_id not interpretable as int
            
            # 2. Check Existence
            if not trace_path:
                print(f"WARNING: Could not find trace file for chunk {first_chunk_id} to verify duration. Skipping check.")
            else:
                # 3. Read & Verify
                try:
                    tdf = pd.read_csv(trace_path)
                    if 'time_sec' not in tdf.columns:
                         print(f"WARNING: 'time_sec' column missing in {trace_path}. Skipping duration check.")
                    else:
                         t = tdf['time_sec'].values
                         if len(t) < 2:
                              print(f"WARNING: Insufficient points in {trace_path}. Skipping check.")
                         else:
                              trace_dur = t[-1] - t[0]
                              diff = abs(trace_dur - args.session_duration_s)
                              if diff > 2.0:
                                   raise RuntimeError(f"Session Duration Mismatch! Provided: {args.session_duration_s:.2f}s, Trace: {trace_dur:.2f}s (Diff: {diff:.2f}s). Chunk: {first_chunk_id}, File: {trace_path}")
                except RuntimeError:
                    raise # Propagate the fatal error
                except Exception as e:
                    print(f"WARNING: Failed to read/parse {trace_path}: {e}. Skipping check.")
        if args.session_duration_s:
            session_duration_s = args.session_duration_s
        else:
            # Fallback (Legacy behavior, but now technically incorrect for duty cycle)
            # We used to assume duration = 3600/sph. 
            # If not provided, we warn.
            session_duration_s = 3600.0 / float(args.sessions_per_hour)
            print(f"Warning: --session-duration-s not provided. inferred {session_duration_s}s from rate.")
        
        if use_datetime:
            roi_df = roi_df.sort_values('dt')
            dt0 = roi_df['dt'].min()
            if pd.isna(dt0):
                 raise RuntimeError("Datetime mode selected but no valid datetimes found.")
            roi_df['elapsed_hours'] = (roi_df['dt'] - dt0).dt.total_seconds() / 3600.0
            
            # Compute Grid coords from dt
            elapsed_s = (roi_df['dt'] - dt0.replace(hour=0, minute=0, second=0, microsecond=0)).dt.total_seconds()
            roi_df['day'] = (elapsed_s // 86400).astype(int)
            roi_df['hour'] = ((elapsed_s % 86400) // 3600).astype(int)
            # Use rank within hour for session_in_hour
            roi_df['session_in_hour'] = roi_df.groupby(['day', 'hour']).cumcount()
            
            time_mode = "Datetime-derived"
        else:
            roi_df = roi_df.sort_values('chunk_id')
            # Elapsed hours still fundamentally relies on STRIDE (rate), not duration.
            # 1 chunk = 1 interval = 1/rate hours.
            roi_df['elapsed_hours'] = roi_df['chunk_id'] / float(args.sessions_per_hour)
            
            roi_df['session_idx'] = np.arange(len(roi_df))
            roi_df['day'] = (roi_df['session_idx'] // (24 * args.sessions_per_hour)).astype(int)
            roi_df['hour'] = ((roi_df['session_idx'] // args.sessions_per_hour) % 24).astype(int)
            roi_df['session_in_hour'] = (roi_df['session_idx'] % args.sessions_per_hour).astype(int)
            
            time_mode = f"Chunk ID fallback (rate={args.sessions_per_hour}/hr)"

        print(f"Time axis mode: {time_mode}")
        print(f"Number of sessions: {len(roi_df)}")

        # 7. Compute Metrics
        # Rate = Count / (Duration_Min)
        roi_df['peak_rate_per_min'] = roi_df['peak_count'] / (session_duration_s / 60.0)
        
        # 8. Plot 1: Event Frequency
        x = roi_df['elapsed_hours'].astype(float)
        y_rate = roi_df['peak_rate_per_min'].astype(float)
        
        if y_rate.isna().all():
             raise RuntimeError("All Peak Rate values are NaN.")

        fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=args.dpi)
        ax1.plot(x, y_rate, marker='o', linestyle='-', label=f'Peak Rate (ROI: {selected_roi})')
        ax1.set_xlabel("Elapsed time (hours)")
        ax1.set_ylabel("Peaks per minute")
        ax1.set_title(f"Phasic event frequency over time (ROI {selected_roi})")
        ax1.grid(True, alpha=0.3)
        
        out_path1 = os.path.join(out_dir, "fig_phasic_peak_rate_timeseries.png")
        fig1.tight_layout()
        fig1.savefig(out_path1)
        plt.close(fig1) 
        print(f"Saved: {out_path1}")

        # 9. Plot 2: AUC
        y_auc = roi_df['auc'].astype(float)
        
        if y_auc.isna().all():
            raise RuntimeError("All AUC values are NaN.")

        fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=args.dpi)
        ax2.plot(x, y_auc, marker='o', linestyle='-', label=f'AUC (ROI: {selected_roi})')
        ax2.set_xlabel("Elapsed time (hours)")
        ax2.set_ylabel("AUC above threshold (dFF·s)")
        ax2.set_title(f"Phasic AUC over time (ROI {selected_roi})")
        ax2.grid(True, alpha=0.3)
        
        out_path2 = os.path.join(out_dir, "fig_phasic_auc_timeseries.png")
        fig2.tight_layout()
        fig2.savefig(out_path2)
        plt.close(fig2)
        print(f"Saved: {out_path2}")
        
        # 10. Export CSVs
        if args.export_csv:
            # Prepare DataFrame
            # Schema: time_hours, day, hour, session_in_hour, peak_rate_per_min, n_peaks, window_seconds, auc_above_threshold_dff_s, threshold_used
            
            # Common
            csv_df = pd.DataFrame()
            csv_df['time_hours'] = x
            csv_df['day'] = roi_df['day']
            csv_df['hour'] = roi_df['hour']
            csv_df['session_in_hour'] = roi_df['session_in_hour']
            csv_df['window_seconds'] = session_duration_s
            
            # Peak Rate CSV
            df_peak = csv_df.copy()
            df_peak['peak_rate_per_min'] = y_rate
            # Enforce integer type for peak_count
            df_peak['peak_count'] = roi_df['peak_count'].fillna(0).astype(int)
            
            p_path = os.path.join(out_dir, "phasic_peak_rate_timeseries.csv")
            df_peak.to_csv(p_path, index=False)
            print(f"Saved: {p_path}")
            
            # AUC CSV
            df_auc = csv_df.copy()
            df_auc['auc_above_threshold_dff_s'] = y_auc
            # Check for both threshold and threshold_used
            if 'threshold' in roi_df.columns:
                df_auc['threshold_used'] = roi_df['threshold']
            elif 'threshold_used' in roi_df.columns:
                df_auc['threshold_used'] = roi_df['threshold_used']
            
            a_path = os.path.join(out_dir, "phasic_auc_timeseries.csv")
            df_auc.to_csv(a_path, index=False)
            print(f"Saved: {a_path}")
        
    except RuntimeError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Unexpected Error: {e}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()
