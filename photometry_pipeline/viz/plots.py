import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import List, Optional
from ..core.types import Chunk

import logging

def set_style():
    # Strict style: No external dependencies, strict Matplotlibv3 + defaults
    plt.style.use('default') 
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (12, 6)

def plot_single_session_raw(chunk: Chunk, roi: str, output_dir: str):
    """
    PLOT SET A: Single-Session Raw Traces
    """
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    
    t = chunk.time_sec
    if roi not in chunk.channel_names:
        plt.close(fig)
        return
        
    r_idx = chunk.channel_names.index(roi)
    
    # UV Raw
    ax[0].plot(t, chunk.uv_raw[:, r_idx], color='purple', label='UV Raw (Isosbestic)')
    ax[0].set_ylabel('Signal (a.u.)')
    ax[0].legend(loc='upper right')
    ax[0].set_title(f"Raw Inputs - {roi} (Single Chunk)")
    
    # Sig Raw
    ax[1].plot(t, chunk.sig_raw[:, r_idx], color='green', label='Sig Raw (Calcium)')
    ax[1].set_ylabel('Signal (a.u.)')
    ax[1].set_xlabel('Time (s)')
    ax[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_A_raw_traces_{roi}.png"))
    plt.close(fig)

def plot_correction_impact(chunk: Chunk, roi: str, interval: slice, output_dir: str):
    """
    PLOT SET D: Correction Impact Panel
    """
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
    
    t = chunk.time_sec[interval]
    if roi not in chunk.channel_names:
        plt.close(fig)
        return
        
    r_idx = chunk.channel_names.index(roi)
    
    uv = chunk.uv_raw[interval, r_idx]
    sig = chunk.sig_raw[interval, r_idx]
    
    if chunk.uv_fit is None or chunk.delta_f is None:
        plt.close(fig)
        raise ValueError(f"Missing regression output for {roi}")
        
    uv_fit = chunk.uv_fit[interval, r_idx]
    delta_f = chunk.delta_f[interval, r_idx]
    
    # 1. Raw Signals
    ax[0].plot(t, sig, color='green', label='Sig Raw')
    ax[0].plot(t, uv, color='purple', alpha=0.6, label='UV Raw')
    ax[0].set_ylabel('Raw (a.u.)')
    ax[0].legend(loc='upper right')
    ax[0].set_title(f"Correction Impact - {roi}")
    
    # 2. Fit Comparison
    ax[1].plot(t, sig, color='green', alpha=0.5, label='Sig Raw')
    ax[1].plot(t, uv_fit, color='black', linestyle='--', label='UV Fit (Artifact Est)')
    ax[1].set_ylabel('Model Fit')
    ax[1].legend(loc='upper right')
    
    # 3. Corrected
    ax[2].plot(t, delta_f, color='blue', label='Delta F (Corrected)')
    ax[2].set_ylabel('Delta F')
    ax[2].set_xlabel('Time (s)')
    ax[2].axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_D_correction_impact_{roi}.png"))
    plt.close(fig)

def plot_continuous_multiday(traces_dir: str, roi: str, output_dir: str, file_list: List[str]):
    """
    PLOT SET B: Continuous Multi-Day Overlays
    Concatenates chunks from traces_dir in order.
    Strictly requires 'time_sec' column.
    """
    sig_all = []
    uv_all = []
    t_all = []
    t_cursor = 0.0
    
    for fname in sorted(file_list):
        path = os.path.join(traces_dir, fname)
        try:
            df = pd.read_csv(path)
            uv_key = f"{roi}_uv_raw"
            sig_key = f"{roi}_sig_raw"
            
            if uv_key not in df.columns or sig_key not in df.columns:
                continue
            
            # STRICT Time Check
            if 'time_sec' not in df.columns:
                logging.warning(f"Skipping plot B for {fname}: Missing 'time_sec'")
                continue
                
            t_local = df['time_sec'].values
            uv = df[uv_key].values
            sig = df[sig_key].values
            
            duration = t_local[-1] - t_local[0]
            t_out = t_local - t_local[0] + t_cursor
            
            uv_all.append(uv)
            sig_all.append(sig)
            t_all.append(t_out)
            
            # Update cursor for next chunk
            dt = t_local[1] - t_local[0] if len(t_local) > 1 else 1.0
            t_cursor = t_out[-1] + dt
            
        except Exception as e:
            logging.warning(f"Error reading {fname} for Plot B: {e}")
            continue
            
    if not sig_all:
        return

    uv_concat = np.concatenate(uv_all)
    sig_concat = np.concatenate(sig_all)
    t_concat = np.concatenate(t_all)
    
    # Downsample for plotting if HUGE
    if len(t_concat) > 1000000:
        idx = np.arange(0, len(t_concat), 10) 
        t_concat = t_concat[idx]
        uv_concat = uv_concat[idx]
        sig_concat = sig_concat[idx]
        
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 6))
    
    ax[0].plot(t_concat / 3600.0, uv_concat, color='purple', lw=0.5)
    ax[0].set_ylabel('UV Raw')
    ax[0].set_title(f"Continuous Session - {roi}")
    
    ax[1].plot(t_concat / 3600.0, sig_concat, color='green', lw=0.5)
    ax[1].set_ylabel('Sig Raw')
    ax[1].set_xlabel('Time (hours)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_B_continuous_{roi}.png"), dpi=300)
    plt.close(fig)

def plot_stacked_session(traces_dir: str, roi: str, output_dir: str, file_list: List[str]):
    """
    PLOT SET C: Stacked Session-Aligned (Delta F)
    Memory safe (subsampling for range).
    Strict 'time_sec'.
    """
    traces = []
    
    for fname in sorted(file_list):
        path = os.path.join(traces_dir, fname)
        try:
            df = pd.read_csv(path)
            # Prefer deltaF
            key_df = f"{roi}_deltaF"
            key_dff = f"{roi}_dff"
            
            if key_df in df.columns:
                data = df[key_df].values
            elif key_dff in df.columns:
                data = df[key_dff].values 
            else:
                continue
            
            # STRICT Time Check
            if 'time_sec' not in df.columns:
                logging.warning(f"Skipping plot C for {fname}: Missing 'time_sec'")
                continue
                
            t = df['time_sec'].values
            traces.append((t, data))
        except:
            continue
            
    if not traces:
        return
        
    n_chunks = len(traces)
    if n_chunks == 0: return

    fig, ax = plt.subplots(figsize=(12, 0.5 * n_chunks + 2))
    if n_chunks > 50:
         fig.set_size_inches(12, 20) 
    
    # Determine step - Memory Safe
    # Subsample each trace 
    subsamples = []
    for _, y in traces:
        if len(y) > 0:
            # Take up to 100 points evenly spaced
            idx = np.linspace(0, len(y)-1, min(len(y), 100)).astype(int)
            subsamples.append(y[idx])
            
    if subsamples:
        all_vals = np.concatenate(subsamples)
        p99 = np.nanpercentile(all_vals, 99)
        p1 = np.nanpercentile(all_vals, 1)
        yrange = p99 - p1
    else:
        yrange = 0
        
    if np.isnan(yrange) or yrange == 0: yrange = 1.0
    step = yrange * 0.5
    
    for i, (t, y) in enumerate(traces):
        # Align time to start at 0 per chunk
        if len(t) == 0: continue
        t_plot = t - t[0]
        ax.plot(t_plot, y + i * step, color='black', lw=0.5)
        
    ax.set_yticks([])
    ax.set_xlabel('Time (s) within Chunk')
    ax.set_title(f"Stacked Chunks - {roi}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_C_stacked_{roi}.png"))
    plt.close(fig)
