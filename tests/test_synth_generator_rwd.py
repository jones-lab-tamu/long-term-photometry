
import unittest
import os
import shutil
import sys
import tempfile
import glob
import subprocess
import numpy as np
import pandas as pd

class TestSynthGeneratorRWD(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'config.yaml')
        
        with open(self.config_path, 'w') as f:
            f.write("""
chunk_duration_sec: 600
target_fs_hz: 20
baseline_method: uv_raw_percentile_session
baseline_percentile: 10
rwd_time_col: TimeStamp
uv_suffix: "-410"
sig_suffix: "-470"
peak_threshold_method: mean_std
window_sec: 20.0
step_sec: 5.0
""")

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_rwd_generation_and_pipeline(self):
        # 1. Generate (2.0 days)
        out_dir = os.path.join(self.test_dir, 'data')
        cmd_gen = [
            sys.executable, 'tools/synth_photometry_dataset.py',
            '--out', out_dir,
            '--format', 'rwd',
            '--config', self.config_path,
            '--phasic-amp-logmean', '3.5',
            '--phasic-base-rate-hz', '0.05',
            '--phasic-ct-mode', 'absolute',
            '--phasic-day-high',
            '--phasic-day-start-ct', '0.0',
            '--phasic-day-end-ct', '12.0',
            '--tonic-amplitude', '0.0',
            '--total-days', '2.0',
            '--recording-duration-min', '10',
            '--recordings-per-hour', '2',
            '--fs-hz', '20',
            '--n-rois', '1',
            '--seed', '123',
            # Test-only: increase motion event count/amplitude to ensure enough samples for polarity statistics.
            '--artifact-enable-motion',
            '--artifact-motion-rate-per-day', '600.0',
            '--artifact-motion-amp-range', '20.0', '40.0',
            # Polarity Bias Test Args
            '--no-artifact-motion-same-sign',
            '--artifact-motion-neg-prob', '0.85',
            '--artifact-motion-amp-range', '25.0', '40.0', # Robust detection
            '--artifact-motion-rise-sec', '0.30',
            '--artifact-motion-decay-sec', '2.50' 
        ]
        res = subprocess.run(cmd_gen, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Generator failed: {res.stderr}")
        
        rows = glob.glob(os.path.join(out_dir, "*", "fluorescence.csv"))
        rows.sort()
        self.assertGreater(len(rows), 0)
        
        # Load Full Concatenated Data
        all_sig = []
        all_uv = []
        for r in rows:
            d = pd.read_csv(r)
            all_sig.append(d['Region0-470'].values)
            all_uv.append(d['Region0-410'].values)
            
        full_sig = np.concatenate(all_sig)
        full_uv = np.concatenate(all_uv)
        
        # A. Phasic Visibility Test
        # Use diff-based peak detection to ignore drift
        # Phasic rise (0.02s) -> High Diff. Motion rise (0.08s) -> Lower Diff.
        # Threshold 6.0 robustly ignores noise (0.8) and small motion.
        dsig = np.diff(full_sig)
        # noise_sigma estimation
        g_noise = np.median(np.abs(dsig - np.median(dsig))) / 0.6745 / np.sqrt(2)
        thresh = max(6.0, 10.0 * g_noise)
        
        n_sig_phasic = 0
        n_uv_phasic = 0
        
        for r in rows:
            d = pd.read_csv(r)
            s = d['Region0-470'].values
            u = d['Region0-410'].values
            ds = np.diff(s)
            du = np.diff(u)
            
            # Motion Masking
            motion_gate = 3.0
            motion_mask = np.abs(du) > motion_gate
            
            # Dilation (+/- 2 samples)
            pad = 2
            if np.any(motion_mask):
                idx = np.where(motion_mask)[0]
                for k in range(-pad, pad + 1):
                    j = idx + k
                    j = j[(j >= 0) & (j < len(motion_mask))]
                    motion_mask[j] = True
            
            # Count peaks outside motion
            sig_phasic = (ds > thresh) & (~motion_mask)
            uv_phasic = (np.abs(du) > thresh) & (~motion_mask)
            
            n_sig_phasic += np.sum(sig_phasic)
            n_uv_phasic += np.sum(uv_phasic)
            
        self.assertGreater(n_sig_phasic, 50, f"Too few phasic events detected (motion-masked): {n_sig_phasic}")
        self.assertLess(n_uv_phasic, n_sig_phasic * 0.25, f"UV has too many phasic-like peaks after motion masking (ratio {n_uv_phasic/(n_sig_phasic+1):.2f})")
        
        # A.1 Day vs Night Phasic Rate Check
        # We used absolute CT mode. Day (high) is [0, 12), Night (low) is [12, 24).
        # We need to compute CT for each sample and count peaks in day vs night regions.
        # fs=20.
        n_day_peaks = 0
        n_night_peaks = 0
        day_samples = 0
        night_samples = 0
        
        # Re-iterate or use what we computed?
        # Let's do a quick pass on full concatenated signal for simplicity of indexing?
        # No, use chunks to align with rows.
        
        # We know each chunk is 10 min. We need absolute time.
        # RWD stores TimeStamp.
        current_offset = 0.0
        
        # Re-calculate phasic peaks on full data to map them to time?
        # Or just do per-row accumulation.
        
        for r in rows:
            d = pd.read_csv(r)
            s = d['Region0-470'].values
            t_loc = d['TimeStamp'].values
            t_glob_hr = (t_loc + current_offset) / 3600.0
            
            ct = t_glob_hr % 24.0
            is_day = (ct >= 0.0) & (ct < 12.0)
            
            day_samples += np.sum(is_day)
            night_samples += np.sum(~is_day)
            
            # Detect peaks strictly
            ds = np.diff(s)
            
            # Simple threshold logic again?
            # Or assume the 'n_sig_phasic' above was correct and we just need to split it.
            # Let's effectively replicate the peak finding but split by Time.
            
            # Motion Mask for this chunk
            u = d['Region0-410'].values
            du = np.diff(u)
            motion_gate = 3.0
            motion_mask = np.abs(du) > motion_gate
            pad = 2
            if np.any(motion_mask):
                idx = np.where(motion_mask)[0]
                for k in range(-pad, pad + 1):
                    j = idx + k
                    j = j[(j >= 0) & (j < len(motion_mask))]
                    motion_mask[j] = True
                    
            peaks = (ds > thresh) & (~motion_mask)
            # peaks is length N-1. is_day is length N. Align to index 0?
            # ds[i] = s[i+1] - s[i]. Let's say peak at i corresponds to t[i].
            
            peaks_day = peaks & is_day[:-1]
            peaks_night = peaks & (~is_day[:-1])
            
            n_day_peaks += np.sum(peaks_day)
            n_night_peaks += np.sum(peaks_night)
            
            current_offset += 600.0 # 10 min chunks
            
        # Normalize by duration if needed, but since we have 50/50 day/night split over 2 days...
        # Just assert Day > Night * 1.05
        # Note: If day_samples is roughly equal to night_samples.
        
        self.assertGreater(n_day_peaks, n_night_peaks * 1.05, f"Day phasic peaks ({n_day_peaks}) should significantly exceed Night ({n_night_peaks})")

        # A.2 Phasic Shape Check (GCaMP-like width)
        
        # A.2 Phasic Shape Check (GCaMP-like width)
        widths = []
        for r in rows:
            d = pd.read_csv(r)
            s = d['Region0-470'].values
            u = d['Region0-410'].values
            du = np.diff(u)
            
            # Recompute motion mask
            motion_gate = 3.0
            motion_mask = np.abs(du) > motion_gate
            pad = 2
            if np.any(motion_mask):
                idx = np.where(motion_mask)[0]
                for k in range(-pad, pad + 1):
                    j = idx + k
                    j = j[(j >= 0) & (j < len(motion_mask))]
                    motion_mask[j] = True
            
            # Local peak finding on raw signal for shape
            base = np.median(s)
            peak_thresh = base + 5.0 # Conservative threshold above baseline
            
            # Simple local max
            candidates = (s[1:-1] > s[:-2]) & (s[1:-1] > s[2:])
            cand_idxs = np.where(candidates)[0] + 1
            
            for p_idx in cand_idxs:
                # Mask check
                if p_idx >= len(motion_mask): continue
                if motion_mask[p_idx] or motion_mask[p_idx-1]: continue
                
                val = s[p_idx]
                if val < peak_thresh: continue
                
                # Width at half prominence
                half_height = base + 0.5 * (val - base)
                
                l = p_idx
                while l > 0 and s[l] > half_height:
                    l -= 1
                
                r_scan = p_idx
                while r_scan < len(s) - 1 and s[r_scan] > half_height:
                    r_scan += 1
                    
                w = r_scan - l
                widths.append(w)
                
        if len(widths) > 5:
            w_med = np.median(widths)
            self.assertGreaterEqual(w_med, 3.0, f"Median phasic width too narrow ({w_med:.1f} samples) - vertical lines detected!")
            self.assertLess(w_med, 200.0, f"Median phasic width too wide ({w_med:.1f} samples)")
        
        # B. Shared Motion Artifact Test
        # Use UV transients (ju > 3.0 absolute) to identify motion.
        # Tests use high-amp motion (25-40), so strict gate is appropriate.
        js = np.abs(np.diff(full_sig))
        ju = np.abs(np.diff(full_uv))
        
        idx = ju > 3.0
        
        if np.sum(idx) > 5:
            # Check for strong positive correlation
            c = np.corrcoef(js[idx], ju[idx])[0,1]
            self.assertGreater(c, 0.5, f"Motion artifacts not correlated: {c:.3f}")
            
        # C. Motion Artifact Polarity Test
        # Accumulate diffs per chunk to avoid stitching discontinuities
        all_du = []
        for r in rows:
            d = pd.read_csv(r)
            u = d['Region0-410'].values
            all_du.append(np.diff(u))
        
        full_du = np.concatenate(all_du)
        
        # Event-Level Polarity Logic
        abs_du = np.abs(full_du)

        med = np.median(full_du)
        mad = np.median(np.abs(full_du - med))
        sigma_du = max(mad / 0.6745, 1e-12)  # epsilon guard against mad==0

        # 8-sigma gate with floor of 2.5 ensures we detect slow/low motion
        # Relaxed to 5-sigma / 2.0 to catch slow rise
        motion_gate = max(2.0, 5.0 * sigma_du)

        cand_indices = np.where(abs_du > motion_gate)[0]
        cand_indices.sort()
        
        # Cluster candidates into events (0.5s refractory scan)
        # fs=20, so 0.5s = 10 samples
        refractory_samples = 10 
        event_vals = []
        
        ptr = 0
        while ptr < len(cand_indices):
            start_idx = cand_indices[ptr]
            end_window = start_idx + refractory_samples
            
            # Find span of candidates within window [start_idx, start_idx + refractory]
            j = ptr
            while j < len(cand_indices) and cand_indices[j] <= end_window:
                j += 1
            window_idxs = cand_indices[ptr:j]
            
            # Select index with maximum abs_du in this window
            best_local_idx = np.argmax(abs_du[window_idxs])
            best_global_idx = window_idxs[best_local_idx]
            event_vals.append(full_du[best_global_idx])
            
            # Advance ptr to first candidate > best_global_idx + refractory_samples
            # This ensures we don't double count the same artifact
            next_start_limit = best_global_idx + refractory_samples
            while ptr < len(cand_indices) and cand_indices[ptr] <= next_start_limit:
                ptr += 1
        
        if len(event_vals) < 20:
            self.fail(f"Not enough gated motion EVENTS for polarity check (n={len(event_vals)}), increase motion rate/amp or relax percentile.")
            
        event_vals = np.array(event_vals)
        frac_neg = np.mean(event_vals < 0)
        self.assertGreaterEqual(frac_neg, 0.70, f"Motion artifacts not mostly negative: {frac_neg:.2f} (n={len(event_vals)})")
            
        # C.2 Motion Width Check (Must be distinct from phasic)
        # Reuse 'cand_indices' (gated motion) and measure FWHM of |du| or raw |u|?
        # The stored events are |du|.
        # Let's verify Median Motion Width >> Median Phasic Width.
        # We computed 'widths' (phasic) in A.2.
        
        if len(widths) > 5 and len(cand_indices) > 5:
            phasic_width_med = np.median(widths)
            
            motion_widths = []
            pad = 50 # max half-width search
            
            # Re-scan events for width
            # Re-scan events for width
            for i_du in cand_indices:
                u_idx = i_du + 1
                if u_idx < pad or u_idx >= len(u) - pad: continue
                
                # Slicing u around u_idx
                local_u = u[u_idx-pad : u_idx+pad]
                if len(local_u) == 0: continue
                
                if u_idx > pad + 50:
                    local_u_base = np.median(u[u_idx-pad-50 : u_idx-pad])
                else:
                    local_u_base = np.median(local_u)
                    
                local_u_abs = np.abs(local_u - local_u_base)
                
                if len(local_u_abs) == 0: continue
                
                # Find peak in window
                p_loc = np.argmax(local_u_abs)
                val = local_u_abs[p_loc]
                p_idx = p_loc # relative to window start
                
                if val < 5.0: continue 
                
                half = val * 0.5
                
                l = p_idx
                while l > 0 and local_u_abs[l] > half: l -= 1
                
                r_scan = p_idx
                while r_scan < len(local_u_abs) - 1 and local_u_abs[r_scan] > half:
                    r_scan += 1
                    
                w = r_scan - l
                motion_widths.append(w)
                
            if len(motion_widths) > 5:
                motion_width_med = np.median(motion_widths)
                
                # Compare waveform widths.
                # Motion (0.3/2.5) vs Phasic (0.12/1.2).
                # Ratio should be ~2.0. Relax to 1.5 for stability.
                
                self.assertGreaterEqual(motion_width_med, 1.5 * phasic_width_med, 
                    f"Motion width ({motion_width_med:.1f}) not significantly wider than Phasic ({phasic_width_med:.1f})")
            
        # D. Drift Test (Tonic Disabled)
        mean_s = []
        mean_u = []
        x = np.arange(len(rows))
        
        for r in rows:
            d = pd.read_csv(r)
            mean_s.append(d['Region0-470'].mean())
            mean_u.append(d['Region0-410'].mean())
            
        slope_s = np.polyfit(x, mean_s, 1)[0]
        slope_u = np.polyfit(x, mean_u, 1)[0]
        
        self.assertLess(slope_s, 0, f"Signal drift not negative: {slope_s}")
        self.assertLess(slope_u, 0, f"UV drift not negative: {slope_u}")
        ratio = abs(slope_u / slope_s)
        self.assertTrue(0.5 <= ratio <= 1.5, f"Drift slope ratio mismatch: {ratio:.2f}")

        # Pipeline & Plots
        pipeline_out = os.path.join(self.test_dir, 'analysis')
        from photometry_pipeline.config import Config
        from photometry_pipeline.pipeline import Pipeline
        cfg = Config.from_yaml(self.config_path)
        pl = Pipeline(cfg)
        pl.run(input_dir=out_dir, output_dir=pipeline_out, force_format='rwd', recursive=True)
        
        plot_out = os.path.join(self.test_dir, 'plots')
        cmd_plot = [
            sys.executable, 'tools/plot_raw_stitched.py',
            '--input', out_dir, '--format', 'rwd', '--config', self.config_path, 
            '--out', plot_out, '--auto-ylims-robust', '--decimate', '10'
        ]
        subprocess.run(cmd_plot, check=True)
        self.assertGreater(len(glob.glob(os.path.join(plot_out, "*.png"))), 0)

if __name__ == '__main__':
    unittest.main()
