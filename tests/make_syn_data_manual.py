
import os
import pandas as pd
import numpy as np
import shutil

def create_syn_data():
    base_dir = r"c:/Users/Jeff/Documents/Photometry_Code/tests/manual_test_data"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    session_dir = os.path.join(base_dir, "session_01")
    os.makedirs(session_dir)
    
    # Create 24 chunks (1 hour each)
    n_chunks = 24
    chunk_dur = 3600
    fs = 10
    
    print(f"Generating {n_chunks} chunks of {chunk_dur}s ({n_chunks} hours)...")
    
    for i in range(n_chunks):
        chunk_dir = os.path.join(session_dir, f"chunk_{i:02d}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Time generation (with overlap for strict coverage)
        t_start = i * chunk_dur
        t_end = (i + 1) * chunk_dur + 2.0 
        n_samples = int((t_end - t_start) * fs)
        
        time = np.linspace(t_start, t_end, n_samples, endpoint=False)
        
        # Tonic Trend: 24h Sine Wave
        # Period = 24 * 3600
        period = 24 * 3600
        phase = time * (2 * np.pi / period)
        tonic_baseline = 200 + 50 * np.sin(phase - np.pi/2) # Low at start/end, high in middle
        
        # UV: Baseline + noise
        uv = tonic_baseline * 0.5 + np.random.normal(0, 0.5, n_samples)
        
        # Signal: correlated with UV + Phasic Events
        # Phasic: Random Sparse Spikes
        phasic_events = np.zeros(n_samples)
        # Add ~10 events per chunk
        n_events = 10
        event_locs = np.random.randint(0, n_samples, n_events)
        phasic_events[event_locs] += np.random.uniform(20, 50, n_events)
        # Smooth events slightly (exponential decay simulation usually needed but simple gaussian bump is ok)
        # Let's just make them single point spikes for test or small bumps
        
        sig = 2.0 * uv + phasic_events + np.random.normal(0, 1.0, n_samples)
        
        df = pd.DataFrame({
            'TimeStamp': time,
            'Region0-470': sig,
            'Region0-410': uv
        })
        
        df.to_csv(os.path.join(chunk_dir, "fluorescence.csv"), index=False)
        
    print(f"Created 24h synthetic data in {session_dir}")

if __name__ == "__main__":
    create_syn_data()
