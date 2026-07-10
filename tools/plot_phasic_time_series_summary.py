
import argparse
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Self-contained repo root bootstrap
from pathlib import Path
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from photometry_pipeline.io.hdf5_cache_reader import (
        open_phasic_cache, resolve_cache_roi, load_cache_chunk_fields,
        list_cache_chunk_ids, list_cache_source_files
    )
    from photometry_pipeline.viz.phasic_data_prep import compute_day_layout
    from photometry_pipeline.viz.phasic_data_prep import (
        build_authoritative_plot_sessions,
        build_feature_map,
    )
except ImportError:
    print("ERROR: Could not import photometry_pipeline. Ensure script is in tools/ and repo root is accessible.")
    sys.exit(1)


def _timeline_anchor_label(anchor_mode: str, fixed_daily_anchor_clock: str | None) -> str:
    mode = str(anchor_mode or "civil").strip().lower()
    if mode == "elapsed":
        return "elapsed-from-first-session"
    if mode == "fixed_daily_anchor":
        clock = str(fixed_daily_anchor_clock or "").strip() or "unset"
        if clock.count(":") == 1:
            clock = f"{clock}:00"
        return f"fixed-daily-anchor@{clock}"
    return "civil-clock"


def _summary_x_axis_label(anchor_mode: str, fixed_daily_anchor_clock: str | None) -> str:
    mode = str(anchor_mode or "civil").strip().lower()
    if mode == "elapsed":
        return "Elapsed time (hours from first session)"
    if mode == "fixed_daily_anchor":
        clock = str(fixed_daily_anchor_clock or "").strip() or "unset"
        if clock.count(":") == 1:
            clock = f"{clock}:00"
        return f"Anchored time (hours from daily anchor {clock})"
    return "Civil-clock time (hours from day-0 midnight)"


def _annotate_missing_sessions(ax, timeline, *, duration_hours: float, y_top: float):
    """Mark approved missing/excluded session slots without creating data."""
    for chunk in timeline.chunks:
        if chunk.status == "valid":
            continue
        if timeline.timeline_anchor_mode == "elapsed":
            x_hours = float(chunk.elapsed_from_start_sec) / 3600.0
        else:
            x_hours = (
                float(chunk.day_idx) * 24.0
                + float(chunk.hour_idx)
                + float(chunk.within_hour_offset_sec) / 3600.0
            )
        width = max(float(duration_hours), 1.0 / max(1, timeline.sessions_per_hour))
        label = (
            "Final incomplete session excluded"
            if chunk.status == "authorized_final_exclusion"
            else "Missing/corrupted session"
        )
        ax.axvspan(
            x_hours - 0.5 * width,
            x_hours + 0.5 * width,
            color="#e9a06d",
            alpha=0.28,
            zorder=0,
        )
        ax.text(
            x_hours,
            y_top,
            f"Session {int(chunk.session_index or chunk.chunk_id) + 1}\n{label}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color="#963014",
            clip_on=True,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Plot phasic event frequency and AUC over time.")
    parser.add_argument('--analysis-out', required=True, help="Path to analysis output directory (containing features/features.csv)")
    parser.add_argument('--roi', help="ROI to plot. Defaults to first ROI alphabetically.")
    parser.add_argument('--sessions-per-hour', type=int, default=2, help="Expected sessions per hour for fallback timing.")
    parser.add_argument(
        '--timeline-anchor-mode',
        choices=['civil', 'elapsed', 'fixed_daily_anchor'],
        default='civil',
        help="Global timeline anchor for day/hour/session placement."
    )
    parser.add_argument(
        '--fixed-daily-anchor-clock',
        default=None,
        help="Anchor clock for fixed_daily_anchor mode (HH:MM or HH:MM:SS)."
    )
    parser.add_argument('--session-duration-s', type=float, default=None, help="Explicit session duration (window seconds).")
    parser.add_argument('--out-rate-png', help="Output path for peak rate timeseries PNG")
    parser.add_argument('--out-auc-png', help="Output path for AUC timeseries PNG")
    parser.add_argument('--out-rate-csv', help="Output path for peak rate timeseries CSV")
    parser.add_argument('--out-auc-csv', help="Output path for AUC timeseries CSV")
    parser.add_argument('--out-dir', help="Fallback output directory based on previous design.")
    parser.add_argument('--dpi', type=int, default=150, help="DPI for output figures")
    parser.add_argument('--export-csv', action='store_true', help="Export CSVs of plotting data (if --out-*-csv not provided)")
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
        if not all([args.out_rate_png, args.out_auc_png, args.out_rate_csv, args.out_auc_csv]):
            os.makedirs(out_dir, exist_ok=True)

        
        # 2. Load Data
        try:
            df = pd.read_csv(features_path)
        except Exception as e:
            raise RuntimeError(f"Error reading features.csv: {e}")
            
        # 3. Validate Columns
        required_cols = ['roi', 'peak_count', 'auc']
        current_cache = None
        authoritative_sessions = None
        cache_path = os.path.join(args.analysis_out, 'phasic_trace_cache.h5')
        if os.path.exists(cache_path):
            current_cache = open_phasic_cache(cache_path)
            try:
                cache_cids = list_cache_chunk_ids(current_cache)
                cache_sfs = list_cache_source_files(current_cache)
                authoritative_sessions = build_authoritative_plot_sessions(
                    args.analysis_out, cache_cids, cache_sfs
                )
            finally:
                current_cache.close()
        required_cols.append(
            'session_index' if authoritative_sessions is not None else 'chunk_id'
        )
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing required columns in features.csv: {missing}")
            
        if authoritative_sessions is not None:
            df['session_index'] = pd.to_numeric(df['session_index'], errors='coerce')
            if df['session_index'].isna().any():
                raise RuntimeError("session_index contains non-numeric values after coercion")
            df['session_index'] = df['session_index'].astype(int)
        else:
            # Enforce numeric chunk_id for legacy outputs.
            df['chunk_id'] = pd.to_numeric(df['chunk_id'], errors='coerce')
            if df['chunk_id'].isna().any():
                raise RuntimeError("chunk_id contains non-numeric values after coercion")
            df['chunk_id'] = df['chunk_id'].astype(int)
            
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

        if authoritative_sessions is not None:
            existing_indices = {
                int(value) for value in roi_df['session_index'].dropna().tolist()
            }
            synthetic_rows = []
            for session in authoritative_sessions:
                idx = int(session["session_index"])
                if idx in existing_indices:
                    continue
                if session.get("status") == "valid":
                    raise RuntimeError(
                        f"features.csv omitted valid session index {idx}"
                    )
                synthetic_rows.append(
                    {
                        "chunk_id": np.nan,
                        "session_index": idx,
                        "source_file": session.get("source_file", ""),
                        "roi": selected_roi,
                        "peak_count": np.nan,
                        "auc": np.nan,
                        "status": session.get("status", "missing_corrupted"),
                        "expected_start_time": session.get("expected_start_time_text", ""),
                        "expected_duration_sec": session.get("expected_duration_sec"),
                        "missing_reason": session.get("missing_reason", ""),
                    }
                )
            if synthetic_rows:
                roi_df = pd.concat([roi_df, pd.DataFrame(synthetic_rows)], ignore_index=True)
            
        # 6. Determine canonical session timeline. Current runs enumerate every
        # expected session, while legacy outputs retain cache/chunk behavior.
        if authoritative_sessions is not None:
            chunk_entries = [
                (int(item["cache_chunk_id"]), str(item["cache_source_file"]))
                for item in authoritative_sessions
                if item.get("cache_chunk_id") is not None
            ]
            session_entries = authoritative_sessions
            feature_map = build_feature_map(
                features_path, roi=selected_roi, key_column="session_index"
            )
        else:
            roi_df['chunk_id'] = roi_df['chunk_id'].astype(int)
            chunk_rows = (
                roi_df[['chunk_id', 'source_file']]
                .drop_duplicates(subset=['chunk_id'], keep='first')
                .sort_values('chunk_id')
            )
            chunk_entries = []
            for row in chunk_rows.itertuples(index=False):
                source_val = row.source_file
                if isinstance(source_val, str) and source_val.strip():
                    source = source_val
                else:
                    source = f"chunk_{int(row.chunk_id)}.csv"
                chunk_entries.append((int(row.chunk_id), source))
            session_entries = None
            feature_map = None

        timeline = compute_day_layout(
            chunk_entries=chunk_entries,
            feature_map=feature_map,
            roi=selected_roi,
            sessions_per_hour=args.sessions_per_hour,
            timeline_anchor_mode=args.timeline_anchor_mode,
            fixed_daily_anchor_clock=args.fixed_daily_anchor_clock,
            session_index_entries=session_entries,
        )
        timeline_key = 'session_index' if authoritative_sessions is not None else 'chunk_id'
        timeline_by_key = {
            int(getattr(c, timeline_key)): c for c in timeline.chunks
        }
        anchor_label = _timeline_anchor_label(
            timeline.timeline_anchor_mode,
            timeline.fixed_daily_anchor_clock,
        )
        x_axis_label = _summary_x_axis_label(
            timeline.timeline_anchor_mode,
            timeline.fixed_daily_anchor_clock,
        )
        missing_ids = sorted(set(roi_df[timeline_key]) - set(timeline_by_key.keys()))
        if missing_ids:
            raise RuntimeError(
                "Canonical timeline mapping missing chunk IDs: "
                f"{missing_ids[:10]}"
            )

        use_datetime = sum(1 for c in timeline.chunks if c.datetime_inferred is not None) / max(1, len(timeline.chunks)) >= 0.90
        
        # Verify consistency if session_duration_s is provided
        if args.session_duration_s is not None:
            # Phasic (Migrated to HDF5 Cache)
            cache_path = os.path.join(args.analysis_out, 'phasic_trace_cache.h5')
            if not os.path.exists(cache_path):
                raise RuntimeError(f"Consistency check failed: phasic cache not found at {cache_path}")
            
            cache = open_phasic_cache(cache_path)
            try:
                roi = resolve_cache_roi(cache, args.roi)
                cids = list_cache_chunk_ids(cache)
                if not cids:
                    raise RuntimeError(f"Consistency check failed: No chunks found in phasic cache at {cache_path}")
                
                # Use the first available chunk for duration verification
                cid0 = cids[0]
                t_arr, = load_cache_chunk_fields(cache, roi, cid0, ['time_sec'])
                
                if len(t_arr) < 2:
                     raise RuntimeError(f"Consistency check failed: Insufficient points in cache: {roi}/chunk_{cid0}.")
                
                trace_dur = t_arr[-1] - t_arr[0]
                diff = abs(trace_dur - args.session_duration_s)
                tol = max(2.0, 0.005 * args.session_duration_s)
                if diff > tol:
                     raise RuntimeError(f"Session Duration Mismatch! Provided: {args.session_duration_s:.2f}s, Trace (Cache): {trace_dur:.2f}s (Diff: {diff:.2f}s, Tol: {tol:.2f}s). Trace: {roi}/chunk_{cid0}")
            finally:
                cache.close()

        if args.session_duration_s is not None:
            session_duration_s = args.session_duration_s
        else:
            # Fallback (Legacy behavior, but now technically incorrect for duty cycle)
            # We used to assume duration = 3600/sph. 
            # If not provided, we warn.
            session_duration_s = 3600.0 / float(args.sessions_per_hour)
            print(f"Warning: --session-duration-s not provided. inferred {session_duration_s}s from rate.")

        roi_df['elapsed_hours'] = roi_df[timeline_key].map(
            lambda key: float(timeline_by_key[int(key)].elapsed_from_start_sec) / 3600.0
        )
        if timeline.timeline_anchor_mode == "elapsed":
            roi_df['time_hours'] = roi_df['elapsed_hours']
        else:
            roi_df['time_hours'] = roi_df[timeline_key].map(
                lambda cid: (
                    float(timeline_by_key[int(cid)].day_idx) * 24.0
                    + float(timeline_by_key[int(cid)].hour_idx)
                    + float(timeline_by_key[int(cid)].within_hour_offset_sec) / 3600.0
                )
            )
        roi_df['day'] = roi_df[timeline_key].map(lambda cid: int(timeline_by_key[int(cid)].day_idx))
        roi_df['hour'] = roi_df[timeline_key].map(lambda cid: int(timeline_by_key[int(cid)].hour_idx))
        roi_df['session_in_hour'] = roi_df[timeline_key].map(
            lambda cid: int(timeline_by_key[int(cid)].hour_rank)
        )
        if authoritative_sessions is not None:
            session_meta = {
                int(item["session_index"]): item for item in authoritative_sessions
            }
            roi_df['expected_start_time'] = roi_df['session_index'].map(
                lambda idx: session_meta[int(idx)].get("expected_start_time_text", "")
            )
            roi_df['expected_duration_sec'] = roi_df['session_index'].map(
                lambda idx: session_meta[int(idx)].get("expected_duration_sec")
            )
            roi_df['missing_reason'] = roi_df['session_index'].map(
                lambda idx: session_meta[int(idx)].get("missing_reason", "")
            )
        roi_df = roi_df.sort_values(['time_hours', timeline_key])

        if use_datetime:
            time_mode = "Datetime-derived session timeline"
        else:
            time_mode = f"Chunk ID fallback timeline (rate={args.sessions_per_hour}/hr)"

        print(f"Time axis mode: {time_mode}")
        print(f"Timeline anchor: {anchor_label}")
        print(f"Summary x-axis: {x_axis_label}")
        print(f"Number of sessions: {len(roi_df)}")

        # 7. Compute Metrics
        # Rate = Count / (Duration_Min)
        roi_df['peak_rate_per_min'] = roi_df['peak_count'] / (session_duration_s / 60.0)
        
        # 8. Plot 1: Event Frequency
        x = roi_df['time_hours'].astype(float)
        y_rate = roi_df['peak_rate_per_min'].astype(float)
        
        if y_rate.isna().all():
             raise RuntimeError("All Peak Rate values are NaN.")

        fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=args.dpi)
        ax1.plot(x, y_rate, marker='o', linestyle='-', label=f'Peak Rate (ROI: {selected_roi})')
        _annotate_missing_sessions(
            ax1,
            timeline,
            duration_hours=float(session_duration_s) / 3600.0,
            y_top=ax1.get_ylim()[1],
        )
        ax1.set_xlabel(x_axis_label)
        ax1.set_ylabel("Peaks per minute")
        ax1.set_title(
            f"Phasic event frequency over time (ROI {selected_roi}) [{anchor_label}]"
        )
        ax1.grid(True, alpha=0.3)
        
        out_path1 = args.out_rate_png if args.out_rate_png else os.path.join(out_dir, "fig_phasic_peak_rate_timeseries.png")
        if os.path.dirname(out_path1): os.makedirs(os.path.dirname(out_path1), exist_ok=True)
        fig1.tight_layout()
        fig1.savefig(out_path1)
        plt.close(fig1) 
        print(f"Saved: {out_path1}")

        # 9. Plot 2: AUC
        y_auc = roi_df['auc'].astype(float)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=args.dpi)
        
        if y_auc.isna().all():
             sys.stderr.write("WARNING: All AUC values are NaN, writing placeholder plot.\n")
             ax2.text(0.5, 0.5, "AUC unavailable (all values NaN)", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax2.transAxes)
        else:
             ax2.plot(x, y_auc, marker='o', linestyle='-', label=f'AUC (ROI: {selected_roi})')

        _annotate_missing_sessions(
            ax2,
            timeline,
            duration_hours=float(session_duration_s) / 3600.0,
            y_top=ax2.get_ylim()[1],
        )

        ax2.set_xlabel(x_axis_label)
        ax2.set_ylabel("AUC above threshold (dFF·s)")
        ax2.set_title(
            f"Phasic AUC over time (ROI {selected_roi}) [{anchor_label}]"
        )
        ax2.grid(True, alpha=0.3)
        
        out_path2 = args.out_auc_png if args.out_auc_png else os.path.join(out_dir, "fig_phasic_auc_timeseries.png")
        if os.path.dirname(out_path2): os.makedirs(os.path.dirname(out_path2), exist_ok=True)
        fig2.tight_layout()
        fig2.savefig(out_path2)
        plt.close(fig2)
        print(f"Saved: {out_path2}")
        
        # 10. Export CSVs
        if args.export_csv or args.out_rate_csv or args.out_auc_csv:
            # Prepare DataFrame
            # Schema: time_hours, day, hour, session_in_hour, peak_rate_per_min, n_peaks, window_seconds, auc_above_threshold_dff_s, threshold_used
            
            # Common
            csv_df = pd.DataFrame()
            csv_df['roi'] = roi_df['roi'].astype(str)
            csv_df['source_file'] = roi_df['source_file']
            csv_df['session_index'] = roi_df['session_index'] if authoritative_sessions is not None else roi_df['chunk_id']
            csv_df['status'] = roi_df['status'] if 'status' in roi_df.columns else 'valid'
            csv_df['expected_start_time'] = roi_df.get('expected_start_time', pd.Series(index=roi_df.index, dtype=object))
            csv_df['expected_duration_sec'] = roi_df.get('expected_duration_sec', session_duration_s)
            csv_df['missing_reason'] = roi_df.get('missing_reason', '')
            csv_df['time_hours'] = x
            csv_df['day'] = roi_df['day']
            csv_df['hour'] = roi_df['hour']
            csv_df['session_in_hour'] = roi_df['session_in_hour']
            csv_df['window_seconds'] = session_duration_s
            csv_df['time_axis_semantics'] = x_axis_label
            csv_df['timeline_anchor_mode'] = timeline.timeline_anchor_mode
            csv_df['timeline_anchor_label'] = anchor_label
            csv_df['fixed_daily_anchor_clock'] = timeline.fixed_daily_anchor_clock
            
            # Peak Rate CSV
            df_peak = csv_df.copy()
            df_peak['peak_rate_per_min'] = y_rate
            # Missing sessions remain NaN.  A valid zero-event session is the
            # only case that carries a real zero count.
            df_peak['peak_count'] = pd.to_numeric(roi_df['peak_count'], errors='coerce')
            df_peak['n_peaks'] = df_peak['peak_count']
            
            p_path = args.out_rate_csv if args.out_rate_csv else os.path.join(out_dir, "phasic_peak_rate_timeseries.csv")
            if os.path.dirname(p_path): os.makedirs(os.path.dirname(p_path), exist_ok=True)
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
            
            a_path = args.out_auc_csv if args.out_auc_csv else os.path.join(out_dir, "phasic_auc_timeseries.csv")
            if os.path.dirname(a_path): os.makedirs(os.path.dirname(a_path), exist_ok=True)
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
