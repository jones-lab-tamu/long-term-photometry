
import argparse
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Ensure we can import config
sys.path.append(os.getcwd())
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from photometry_pipeline.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Plot raw stitched photometry data.")
    
    parser.add_argument('--input', required=True, help="Input directory")
    parser.add_argument('--format', required=True, choices=['rwd', 'npm'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--roi', type=int, default=0, help="ROI index to plot")
    parser.add_argument('--axis', choices=['recorded', 'experimental'], default='recorded')
    parser.add_argument('--stretch-visualization', action='store_true', help="Stretch chunks to fill gaps (visual only)")
    parser.add_argument('--auto-ylims-robust', action='store_true', help="Set Y-limits using 1-99 percentiles")
    parser.add_argument('--decimate', type=int, default=1, help="Decimate factor for plotting")
    
    return parser.parse_args()

def load_rwd(input_dir, config):
    subdirs = sorted(glob.glob(os.path.join(input_dir, "*")))
    chunks = []
    
    for d in subdirs:
        if not os.path.isdir(d): continue
        base = os.path.basename(d)
        try:
            dt = datetime.datetime.strptime(base, "%Y_%m_%d-%H_%M_%S")
        except ValueError:
            continue
        fpath = os.path.join(d, "fluorescence.csv")
        if not os.path.exists(fpath): continue
        df = pd.read_csv(fpath)
        chunks.append((dt, df))
        
    chunks.sort(key=lambda x: x[0])
    return chunks

def load_npm(input_dir, config):
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    chunks = []
    
    for f in files:
        base = os.path.basename(f)
        try:
            ts_part = base.replace('photometryData', '').replace('.csv', '')
            dt = datetime.datetime.strptime(ts_part, "%Y-%m-%dT%H_%M_%S")
        except ValueError:
             continue
        raw_df = pd.read_csv(f)
        led_col = config.npm_led_col
        if led_col not in raw_df.columns: continue
        
        uv_rows = raw_df[raw_df[led_col] == 1]
        sig_rows = raw_df[raw_df[led_col] == 2]
        
        time_col = config.npm_system_ts_col if config.npm_time_axis == 'system_timestamp' else config.npm_computer_ts_col
        if time_col not in raw_df.columns: continue
        
        t_uv = uv_rows[time_col].values
        if len(t_uv) == 0: continue
        t_min = t_uv[0]
        t_clean = t_uv - t_min
        
        n = min(len(uv_rows), len(sig_rows))
        
        data = {}
        data[config.rwd_time_col] = t_clean[:n]
        found_roi = False
        
        for c in raw_df.columns:
            if c.startswith(config.npm_region_prefix) and c.endswith(config.npm_region_suffix):
                try:
                    mid = c[len(config.npm_region_prefix) : -len(config.npm_region_suffix)]
                    ridx = int(mid)
                except ValueError:
                    continue
                found_roi = True
                u = uv_rows[c].values[:n]
                s = sig_rows[c].values[:n]
                name_base = f"Region{ridx}"
                data[f"{name_base}{config.sig_suffix}"] = s
                data[f"{name_base}{config.uv_suffix}"] = u
        
        if found_roi:
            df = pd.DataFrame(data)
            chunks.append((dt, df))
        
    chunks.sort(key=lambda x: x[0])
    return chunks

def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    os.makedirs(args.out, exist_ok=True)
    
    if args.format == 'rwd':
        chunks = load_rwd(args.input, cfg)
    else:
        chunks = load_npm(args.input, cfg)
        
    if not chunks:
        print("No data found.")
        sys.exit(0)
        
    all_t = []
    all_sig = []
    all_uv = []
    
    sig_col = f"Region{args.roi}{cfg.sig_suffix}"
    uv_col = f"Region{args.roi}{cfg.uv_suffix}"
    
    if sig_col not in chunks[0][1].columns:
        print(f"ROI {args.roi} not found (looked for {sig_col})")
        sys.exit(1)
        
    start_time_global = chunks[0][0]
    last_t_end = 0.0
    
    for i, (dt, df) in enumerate(chunks):
        if args.axis == 'recorded':
            t_chunk = df[cfg.rwd_time_col].values
            t_chunk = t_chunk - t_chunk[0] 
            if i > 0:
                t_global = last_t_end + t_chunk
            else:
                t_global = t_chunk
            
            if len(t_chunk) > 1:
                dt_step = np.median(np.diff(t_chunk))
            else:
                dt_step = 1.0 / cfg.target_fs_hz
            last_t_end = t_global[-1] + dt_step
            
        else:
            delta = (dt - start_time_global).total_seconds()
            t_local = df[cfg.rwd_time_col].values
            t_global = delta + t_local
            
            if args.stretch_visualization:
                if i < len(chunks)-1:
                    next_dt = chunks[i+1][0]
                    gap = (next_dt - dt).total_seconds()
                    if len(t_local) > 1:
                        t_global = np.linspace(delta, delta + gap, len(t_local), endpoint=False)
            
            if i > 0:
                all_t.append([np.nan])
                all_sig.append([np.nan])
                all_uv.append([np.nan])
                
        all_t.append(t_global)
        all_sig.append(df[sig_col].values)
        all_uv.append(df[uv_col].values)
        
    full_t = np.concatenate(all_t)
    full_sig = np.concatenate(all_sig)
    full_uv = np.concatenate(all_uv)
    
    if args.decimate > 1:
        full_t = full_t[::args.decimate]
        full_sig = full_sig[::args.decimate]
        full_uv = full_uv[::args.decimate]
    
    plt.figure(figsize=(12, 6))
    t_hr = full_t / 3600.0
    
    plt.plot(t_hr, full_sig, label='Signal', color='tab:blue', linewidth=0.8, alpha=0.9)
    # Visual tweak: UV often thinner? 
    plt.plot(t_hr, full_uv, label='Isosbestic', color='tab:purple', linewidth=0.8, alpha=0.8)
    
    if args.auto_ylims_robust:
        valid_vals = np.concatenate([full_sig[~np.isnan(full_sig)], full_uv[~np.isnan(full_uv)]])
        if len(valid_vals) > 0:
            p1 = np.percentile(valid_vals, 1)
            p99 = np.percentile(valid_vals, 99)
            margin = (p99 - p1) * 0.1
            plt.ylim(p1 - margin, p99 + margin)
    
    plt.xlabel(f"Time ({args.axis}, hours)")
    plt.ylabel("Raw Fluorescence")
    plt.title(f"Raw Stitched: ROI {args.roi} ({args.axis})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_name = f"raw_stitched_{args.format}_{args.axis}_roi{args.roi}{'_stretched' if args.stretch_visualization else ''}.png"
    out_path = os.path.join(args.out, out_name)
    plt.savefig(out_path)
    plt.close()
    
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    main()
