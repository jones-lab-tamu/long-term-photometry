
import argparse
import json
import numpy as np
import pandas as pd
import os
import scipy.signal
import scipy.stats

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Photometry Data Generator V1")
    parser.add_argument("--format", type=str, required=True, choices=["rwd", "npm"], help="Output format")
    parser.add_argument("--out", type=str, required=True, help="Output folder")
    parser.add_argument("--n_chunks", type=int, required=True, help="Number of chunks")
    parser.add_argument("--chunk_duration_sec", type=float, required=True, help="Duration of each chunk in seconds")
    parser.add_argument("--fs_hz", type=float, required=True, help="Sampling rate in Hz")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    
    # Stress knobs
    parser.add_argument("--drift_scale", type=float, default=1.0)
    parser.add_argument("--event_rate_scale", type=float, default=1.0)
    parser.add_argument("--artifact_deflection_rate_scale", type=float, default=1.0)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    
    return parser.parse_args()

class SyntheticGenerator:
    def __init__(self, args):
        self.args = args
        np.random.seed(args.seed)
        
        self.n_rois = 8
        
        # Generation is done at 2x fs to support NPM interleaving (UV@t, SIG@t+dt/2)
        # and RWD subsampling (UV@t, SIG@t)
        self.fs_gen = 2.0 * args.fs_hz
        
        # Round n_samples to nearest int
        self.n_samples_gen = int(np.round(args.chunk_duration_sec * self.fs_gen))
        
        # FIX E (Minor): Ensure even number of samples so NPM interleaving is clean
        if self.n_samples_gen % 2 != 0:
            self.n_samples_gen += 1
            
        self.dt_gen = 1.0 / self.fs_gen
        self.time_vec_gen = np.arange(self.n_samples_gen) * self.dt_gen
        
    def generate_kernel(self, tau_rise, tau_decay):
        # Difference of exponentials
        length = int(5 * tau_decay * self.fs_gen)
        t_k = np.arange(length) * self.dt_gen
        k = np.exp(-t_k / tau_decay) - np.exp(-t_k / tau_rise)
        max_val = np.max(k)
        if max_val > 0:
            k /= max_val
        return k

    def generate_calcium(self):
        # Event rate (Hz)
        base_rate = 0.03
        rate = base_rate * self.args.event_rate_scale
        
        # FIX A: Use Poisson with float mean, do not truncate to int
        # Expected number of events
        # Create n_events from Poisson distribution directly
        n_events = np.random.poisson(rate * self.args.chunk_duration_sec)
        
        signal = np.zeros(self.n_samples_gen)
        
        if n_events == 0:
            return signal, 0
            
        # Robust Placement Loop
        event_times = []
        max_attempts = max(1000, n_events * 50)
        attempts = 0
        
        while len(event_times) < n_events and attempts < max_attempts:
            t = np.random.uniform(0, self.args.chunk_duration_sec)
            # Enforce min interval 0.5s
            if all(abs(t - et) > 0.5 for et in event_times):
                event_times.append(t)
            attempts += 1
            
        if len(event_times) < n_events:
            print(f"Warning: Could only place {len(event_times)}/{n_events} events due to interval constraints.")
            
        event_times.sort()
        
        for t_start in event_times:
            # Amplitudes: Lognormal
            # Parameters valid to be visible: mu=2, sigma=0.5
            amp = np.random.lognormal(mean=np.log(20), sigma=0.4)
            
            # Kernel params
            tau_rise = np.random.uniform(0.1, 0.3)
            tau_decay = np.random.uniform(2.0, 6.0)
            
            kernel = self.generate_kernel(tau_rise, tau_decay)
            
            # Add to signal
            start_idx = int(t_start * self.fs_gen)
            end_idx = min(start_idx + len(kernel), self.n_samples_gen)
            if start_idx < self.n_samples_gen:
                len_add = end_idx - start_idx
                signal[start_idx:end_idx] += amp * kernel[:len_add]
                
        return signal, len(event_times)

    def generate_artifact_raw(self):
        # 1. Drift
        rw = np.cumsum(np.random.randn(self.n_samples_gen))
        # Lowpass at 0.05Hz
        # fs is self.fs_gen
        sos = scipy.signal.butter(2, 0.05, 'low', fs=self.fs_gen, output='sos')
        drift = scipy.signal.sosfiltfilt(sos, rw)
        
        if np.std(drift) > 0:
            drift /= np.std(drift)
            
        # FIX B: Implement drift_scale
        drift *= self.args.drift_scale
            
        # 2. Deflections
        expected_deflections = 1.0 * self.args.artifact_deflection_rate_scale
        n_defl = np.random.poisson(expected_deflections)
        
        deflection_sig = np.zeros(self.n_samples_gen)
        
        for _ in range(n_defl):
            center = np.random.uniform(0, self.args.chunk_duration_sec)
            dur = np.random.uniform(2, 30)
            sigma = dur / 4.0
            
            # Gaussian bump
            bump = np.exp(-0.5 * ((self.time_vec_gen - center) / sigma)**2)
            
            sign = np.random.choice([-1, 1])
            deflection_sig += sign * bump
            
        # Combine
        # Deflections are hardcoded to 3.0 scale relative to normalized drift (before drift_scale)
        # Re-eval: drift is now scaled. Deflections are not scaled by a separate knob, 
        # but their rate is. Const 3.0 is acceptable per implicit spec/feedback.
        artifact = drift + deflection_sig * 3.0
        return artifact

    def run(self):
        os.makedirs(self.args.out, exist_ok=True)
        
        summary_stats = {
            "fs_hz": self.args.fs_hz,
            "chunk_duration_sec": self.args.chunk_duration_sec,
            "n_chunks": self.args.n_chunks,
            "format": self.args.format,
            "chunks": []
        }
        
        for chunk_idx in range(self.args.n_chunks):
            print(f"Generating Chunk {chunk_idx}...")
            
            # 1. Generate Base Artifact High Res
            artifact_raw = self.generate_artifact_raw()
            
            # 2. Pre-generate ROI components to determine scaling
            roi_components = []
            fractions = []
            
            for i in range(self.n_rois):
                b_uv = np.random.uniform(200, 600)
                b_sig = np.random.uniform(150, 500)
                
                a_shared = np.random.uniform(0.8, 1.2)
                s_shared = np.random.uniform(0.6, 1.4)
                
                # Noise Generation
                sd_uv = np.random.uniform(0.003, 0.01) * b_uv * self.args.noise_scale
                sd_sig = np.random.uniform(0.003, 0.015) * b_sig * self.args.noise_scale
                
                # UV Noise Clipping (Hard Lock)
                noise_uv_raw = np.random.normal(0, sd_uv, self.n_samples_gen)
                limit = 3 * sd_uv
                noise_uv = np.clip(noise_uv_raw, -limit, limit)
                
                noise_sig = np.random.normal(0, sd_sig, self.n_samples_gen)
                
                # Calcium
                calc_sig, n_ev = self.generate_calcium()
                
                # Sig No Art
                sig_no_art = b_sig + noise_sig + calc_sig
                
                # Compute Fraction with Raw Artifact (scale=1.0)
                # Fraction = Var(s_shared * art) / Var(sig_no_art + s_shared * art)
                art_component = s_shared * artifact_raw
                var_art = np.var(art_component)
                var_total = np.var(sig_no_art + art_component)
                
                frac = 0.0
                if var_total > 0:
                    frac = var_art / var_total
                
                fractions.append(frac)
                
                roi_components.append({
                    "b_uv": b_uv, "noise_uv": noise_uv, "a_shared": a_shared,
                    "b_sig": b_sig, "noise_sig": noise_sig, "s_shared": s_shared,
                    "sig_no_art": sig_no_art, "n_ev": n_ev
                })
                
            # 3. Artifact Variance Scaling (Locked Logic)
            # Compute median fraction m
            m = np.median(fractions)
            if m == 0: m = 1e-9
            
            # Scale artifact by sqrt(0.15 / m)
            scale = np.sqrt(0.15 / m)
            artifact_final = artifact_raw * scale
            
            # 4. Construct Final Signals
            # RWD: Subsample at [0::2] -> Matches fs_hz
            # NPM: Interleave UV[0::2], SIG[1::2] -> Matches fs_hz per channel, 2*fs_hz total
            
            chunk_data = {
                "time_rwd": self.time_vec_gen[0::2], # timestamps for RWD
                "rois": []
            }
            
            chunk_stats = {}
            
            for i, comp in enumerate(roi_components):
                # Full Res Signals
                uv_full = comp["b_uv"] + comp["a_shared"] * artifact_final + comp["noise_uv"]
                sig_full = comp["sig_no_art"] + comp["s_shared"] * artifact_final
                
                # Format specific extraction
                if self.args.format == 'rwd':
                    uv_out = uv_full[0::2]
                    sig_out = sig_full[0::2]
                    
                    uv_check = uv_out
                    sig_check = sig_out
                    
                    # For stats, art_final is sampled correctly at 0::2
                    art_final_check = artifact_final[0::2]
                    
                else: # npm
                    # Interleaved for export later
                    # We store full res, write_chunk handles slicing
                    uv_out = uv_full 
                    sig_out = sig_full
                    
                    # FIX D: Correct indices for NPM Stats
                    # Use UV at 0::2 and SIG at 1::2
                    uv_check = uv_full[0::2]
                    sig_check = sig_full[1::2]
                    
                    # Artifact for fraction must match SIG indices (1::2)
                    art_final_check = artifact_final[1::2]

                corr, _ = scipy.stats.pearsonr(uv_check, sig_check)
                
                # Re-calc final fraction
                # var(s_shared * art_final) / var(sig_final)
                art_comp_final = comp["s_shared"] * art_final_check
                frac_final = np.var(art_comp_final) / np.var(sig_check)
                
                chunk_stats[f"ROI_{i}"] = {
                    "events": comp["n_ev"],
                    "artifact_fraction": frac_final,
                    "corr_uv_sig": corr
                }
                
            summary_stats["chunks"].append(chunk_stats)
            self.write_chunk(chunk_idx, chunk_data)
            
        with open(os.path.join(self.args.out, "summary.json"), "w") as f:
            json.dump(summary_stats, f, indent=2)
            
        print("Generation Complete.")

    def write_chunk(self, chunk_id, data):
        filename = f"chunk_{chunk_id:04d}.csv"
        path = os.path.join(self.args.out, filename)
        
        if self.args.format == "rwd":
            df = pd.DataFrame()
            df["TimeStamp"] = data["time_rwd"] * 1000.0
            df["Events"] = 0
            for i, roi in enumerate(data["rois"]):
                df[f"CH{i+1}-410"] = roi["uv"]
                df[f"CH{i+1}-470"] = roi["sig"]
            df.to_csv(path, index=False)
            
        elif self.args.format == "npm":
            # Data in rois contains full resolution (2*fs) arrays 
            # We want UV at t_rows[0::2] -> indices 0, 2, 4...
            # We want SIG at t_rows[1::2] -> indices 1, 3, 5...
            
            n_rows = self.n_samples_gen # This is always even now due to FIX E
                
            region_data = np.zeros((n_rows, self.n_rois))
            
            for i, roi in enumerate(data["rois"]):
                uv_full = roi["uv"]
                sig_full = roi["sig"]
                # UV on even rows (t=0, 2dt...) -> indices 0, 2...
                region_data[0::2, i] = uv_full[0:n_rows:2]
                # SIG on odd rows (t=dt, 3dt...) -> indices 1, 3...
                region_data[1::2, i] = sig_full[1:n_rows:2]
                
            df = pd.DataFrame()
            df["FrameCounter"] = np.arange(n_rows)
            # SystemTimestamp should match generation time exactly
            df["SystemTimestamp"] = self.time_vec_gen[:n_rows]
            df["LedState"] = np.tile([1, 2], n_rows // 2)
            
            for i in range(self.n_rois):
                df[f"Region{i}G"] = region_data[:, i]
                
            df.to_csv(path, index=False)

if __name__ == "__main__":
    args = parse_args()
    gen = SyntheticGenerator(args)
    gen.run()
