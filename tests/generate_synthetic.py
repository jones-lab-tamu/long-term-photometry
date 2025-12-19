import numpy as np
import pandas as pd
import os
import argparse

def generate_rwd_chunk(path, t_start, duration, fs=40, roi="Region1G"):
    # Time (seconds)
    n = int(duration * fs)
    t = np.arange(n) / fs + t_start
    
    # UV: sin wave + noise
    uv = 100 + 10 * np.sin(0.1 * t) + np.random.normal(0, 1, n)
    
    # Signal: 2 * UV + Calcium events + noise
    # Calcium: fast rise, slow decay
    calcium = np.zeros(n)
    # Add a few events
    events = [int(0.2*n), int(0.5*n), int(0.8*n)]
    tau = 2.0 * fs
    for start in events:
        if start < n:
            length = n - start
            t_loc = np.arange(length)
            decay = np.exp(-t_loc / tau)
            calcium[start:] += 50 * decay
            
    sig = 2.0 * uv + calcium + np.random.normal(0, 1, n)
    
    # RWD Header
    with open(path, 'w') as f:
        f.write("Header Info\n")
        f.write("More Header\n")
        f.write("Time(s),Region1G-410,Region1G-470\n")
        for i in range(n):
            f.write(f"{t[i]:.4f},{uv[i]:.4f},{sig[i]:.4f}\n")

def generate_npm_chunk(path, t_start, duration, fs=40, roi="Region1G"):
    # NPM uses SystemTimestamp (seconds usually), FrameCounter, LedState
    # Interleaved 1 (UV) and 2 (Sig)
    
    # We need to simulate the interleaving. 
    # Total frames = duration * fs * 2
    n_total = int(duration * fs * 2)
    
    t = np.arange(n_total) / (fs * 2) + t_start
    
    led_state = np.zeros(n_total, dtype=int)
    # 1, 2, 1, 2...
    led_state[0::2] = 1
    led_state[1::2] = 2
    
    # UV signal (sampled at t where led=1)
    uv_vals = 100 + 10 * np.sin(0.1 * t) + np.random.normal(0, 1, n_total)
    
    # Sig signal
    calcium = np.zeros(n_total)
    events = [int(0.2*n_total), int(0.5*n_total), int(0.8*n_total)]
    tau = 2.0 * fs * 2 
    for start in events:
        if start < n_total:
            length = n_total - start
            t_loc = np.arange(length)
            decay = np.exp(-t_loc / tau)
            calcium[start:] += 50 * decay
            
    sig_vals = 2.0 * uv_vals + calcium + np.random.normal(0, 1, n_total)
    
    # Region column (only valid values when led is on?)
    # NPM usually has values for all, but we only care about corresponding ones
    region_vals = np.zeros(n_total)
    region_vals[led_state==1] = uv_vals[led_state==1]
    region_vals[led_state==2] = sig_vals[led_state==2]
    
    df = pd.DataFrame({
        'FrameCounter': np.arange(n_total),
        'SystemTimestamp': t,
        'ComputerTimestamp': t, # same for test
        'LedState': led_state,
        'Region0G': region_vals # Config default Region0 or Region1?
        # Default config in code is Region0G.
    })
    
    df.to_csv(path, index=False)

def main():
    os.makedirs('tests/data/rwd', exist_ok=True)
    os.makedirs('tests/data/npm', exist_ok=True)
    
    # Generate 5 minutes RWD
    generate_rwd_chunk('tests/data/rwd/chunk1.csv', t_start=0, duration=300)
    # chunk 2 starts at 600 (gap)
    generate_rwd_chunk('tests/data/rwd/chunk2.csv', t_start=600, duration=300)
    
    # Generate NPM
    generate_npm_chunk('tests/data/npm/chunk1.csv', t_start=1000, duration=300, roi="Region0G")

if __name__ == "__main__":
    main()
