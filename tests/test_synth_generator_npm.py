
import unittest
import os
import shutil
import sys
import tempfile
import glob
import subprocess
import pandas as pd
import numpy as np

class TestSynthGeneratorNPM(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'config.yaml')
        
        with open(self.config_path, 'w') as f:
            f.write("""
chunk_duration_sec: 600
target_fs_hz: 20
baseline_method: uv_raw_percentile_session
baseline_percentile: 10
uv_suffix: "G" 
sig_suffix: "R" 
npm_frame_col: FrameCounter
npm_system_ts_col: SystemTimestamp
npm_computer_ts_col: ComputerTimestamp
npm_led_col: LedState
npm_region_prefix: Region
npm_region_suffix: G
npm_time_axis: system_timestamp
rwd_time_col: TimeStamp 
peak_threshold_method: mean_std
window_sec: 20.0
step_sec: 5.0
""")

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_npm_generation_and_pipeline(self):
        # 1. Generate (2.0 days)
        out_dir = os.path.join(self.test_dir, 'data')
        cmd_gen = [
            sys.executable, 'tools/synth_photometry_dataset.py',
            '--out', out_dir,
            '--format', 'npm',
            '--config', self.config_path,
            '--total-days', '2.0',
            '--phasic-amp-logmean', '2.5',
            '--phasic-amp-logmean', '2.5',
            '--phasic-base-rate-hz', '0.012',
            '--phasic-ct-mode', 'absolute',
            '--phasic-day-high',
            '--tonic-amplitude', '0.0',
            # Test-only: increase motion event count/amplitude to ensure enough samples for polarity statistics.
            '--artifact-enable-motion',
            '--artifact-motion-rate-per-day', '600.0',
            '--artifact-motion-amp-range', '25.0', '40.0',
            '--artifact-motion-rise-sec', '0.30',
            '--artifact-motion-decay-sec', '2.50',
            '--no-artifact-motion-same-sign',
            '--artifact-motion-neg-prob', '0.85',
            '--recordings-per-hour', '2',
            '--fs-hz', '20',
            '--n-rois', '1',
            '--seed', '456'
        ]
        res = subprocess.run(cmd_gen, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Generator failed: {res.stderr}")
        
        files = glob.glob(os.path.join(out_dir, "*.csv"))
        files.sort()
        self.assertGreater(len(files), 0)
        
        # Load Full Concatenated Data
        all_sig = []
        all_uv = []
        
        for f in files:
            df = pd.read_csv(f)
            s = df[df['LedState'] == 2]['Region0G'].values
            u = df[df['LedState'] == 1]['Region0G'].values
            # Trim to match
            n = min(len(s), len(u))
            all_sig.append(s[:n])
            all_uv.append(u[:n])
            
        full_sig = np.concatenate(all_sig)
        full_uv = np.concatenate(all_uv)
        
        # A. Phasic Visibility Test
        dsig = np.diff(full_sig)
        noise_sigma = np.median(np.abs(dsig - np.median(dsig))) / 0.6745 / np.sqrt(2)
        thresh = max(6.0, 10.0 * noise_sigma)
        
        n_sig_phasic = 0
        n_uv_phasic = 0
        
        for i in range(len(files)):
            s = all_sig[i]
            u = all_uv[i]
            # Use diff to ignore drift and attenuate slow motion
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
        n_day_peaks = 0
        n_night_peaks = 0
        
        # NPM logic: timestamps are in dataframe?
        # Yes, SystemTimestamp.
        
        for i in range(len(files)):
            # Need to get timestamps from file
            # We already loaded s and u, but discarded df. Reread minimally or just use index?
            # Timestamps are strictly t_local + offset? Or interleaved? 
            # The test generates continuous data.
            # NPM files are 10 min (600s).
            # Start t_glob based on index i * 600.0.
            
            s = all_sig[i]
            t_glob_start = i * 600.0
            t_loc = np.arange(len(s)) / 20.0 # fs=20
            t_glob_hr = (t_glob_start + t_loc) / 3600.0
            
            ct = t_glob_hr % 24.0
            is_day = (ct >= 0.0) & (ct < 12.0)
            
            # Motion Mask (Recomputed locally)
            u = all_uv[i]
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

            ds = np.diff(s)
            peaks = (ds > thresh) & (~motion_mask)
            
            peaks_day = peaks & is_day[:-1]
            peaks_night = peaks & (~is_day[:-1])
            
            n_day_peaks += np.sum(peaks_day)
            n_night_peaks += np.sum(peaks_night)

        self.assertGreater(n_day_peaks, n_night_peaks * 1.05, f"Day phasic peaks ({n_day_peaks}) should significantly exceed Night ({n_night_peaks})")

        # A.2 Phasic Shape Check (GCaMP-like width)

        # A.2 Phasic Shape Check (GCaMP-like width)
        widths = []
        for i in range(len(files)):
            s = all_sig[i]
            u = all_uv[i]
            # Use diff to ignore drift and attenuate slow motion
            ds = np.diff(s)
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
        js = np.abs(np.diff(full_sig))
        ju = np.abs(np.diff(full_uv))
        
        # B. Shared Motion Artifact Test
        # Use UV transients (ju > 3.0 absolute) to identify motion.
        idx = ju > 3.0
        
        if np.sum(idx) > 5:
            # Check for strong positive correlation
            c = np.corrcoef(js[idx], ju[idx])[0,1]
            self.assertGreater(c, 0.5, f"Motion artifacts not correlated: {c:.3f}")

        # C. Motion Polarity Check
        all_du = []
        for i in range(len(files)):
            u = all_uv[i]
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

        # C.2 Motion Width Check
        if len(widths) > 5 and len(cand_indices) > 5:
            phasic_width_med = np.median(widths)
            
            motion_widths = []
            pad = 50
            for i_du in cand_indices:
                u_idx = i_du + 1
                if u_idx < pad or u_idx >= len(u) - pad: continue
                
                local_u = u[u_idx-pad : u_idx+pad]
                if len(local_u) == 0: continue
                
                # Fallback baseline
                if u_idx > pad+50:
                    local_u_base = np.median(u[u_idx-pad-50 : u_idx-pad])
                else: 
                    local_u_base = np.median(local_u)
                    
                local_u_abs = np.abs(local_u - local_u_base)
                if len(local_u_abs) == 0: continue
                
                p_loc = np.argmax(local_u_abs)
                val = local_u_abs[p_loc]
                p_idx = p_loc
                
                if val < 5.0: continue
                
                half = val * 0.5
                l = p_idx
                while l > 0 and local_u_abs[l] > half: l -= 1
                r = p_idx
                while r < len(local_u_abs) - 1 and local_u_abs[r] > half: r += 1
                
                motion_widths.append(r - l)
                
            if len(motion_widths) > 5:
                motion_width_med = np.median(motion_widths)
                self.assertGreaterEqual(motion_width_med, 1.5 * phasic_width_med, 
                    f"Motion width ({motion_width_med:.1f}) not significantly wider than Phasic ({phasic_width_med:.1f})")

        # C. Shared Drift Test
        mean_s = [np.mean(c) for c in all_sig]
        mean_u = [np.mean(c) for c in all_uv]
        
        if True: # Drift enabled by default
            self.assertGreater(mean_s[0], mean_s[-1], "Signal drift not accumulating")

        x = np.arange(len(mean_s))
        
        slope_s = np.polyfit(x, mean_s, 1)[0]
        slope_u = np.polyfit(x, mean_u, 1)[0]
        
        self.assertLess(slope_s, 0, f"Signal drift not negative: {slope_s}")
        self.assertLess(slope_u, 0, f"UV drift not negative: {slope_u}")
        ratio = abs(slope_u / slope_s)
        self.assertTrue(0.5 <= ratio <= 1.5, f"Drift slope ratio mismatch: {ratio:.2f}")

        # Pipeline
        pipeline_out = os.path.join(self.test_dir, 'analysis')
        from photometry_pipeline.config import Config
        from photometry_pipeline.pipeline import Pipeline
        cfg = Config.from_yaml(self.config_path)
        pl = Pipeline(cfg)
        pl.run(input_dir=out_dir, output_dir=pipeline_out, force_format='npm', recursive=False)
        
        # Plot
        plot_out = os.path.join(self.test_dir, 'plots')
        cmd_plot = [
            sys.executable, 'tools/plot_raw_stitched.py',
            '--input', out_dir, '--format', 'npm', '--config', self.config_path,
            '--out', plot_out, '--auto-ylims-robust', '--decimate', '10'
        ]
        subprocess.run(cmd_plot, check=True)
        self.assertGreater(len(glob.glob(os.path.join(plot_out, '*.png'))), 0)

    def test_npm_gain_drift(self):
        """Verify that gain drift produces changing slope"""
        from scipy.stats import linregress
        out_dir = os.path.join(self.test_dir, "npm_gain")
        os.makedirs(out_dir, exist_ok=True)
        
        cmd_gen = [
            sys.executable, 'tools/synth_photometry_dataset.py',
            '--out', out_dir,
            '--format', 'npm',
            '--config', self.config_path,
            '--total-days', '0.05', 
            '--recording-duration-min', '10',
            '--phasic-mode', 'low_phasic', 
            '--shared-wobble-enable',
            '--shared-wobble-gain-enable',
            '--shared-wobble-gain-tau-sec', '60.0', 
            '--shared-wobble-gain-sd', '0.3',
            '--no-artifact-enable-motion',
            '--fs-hz', '20',
            '--recordings-per-hour', '1',
            '--n-rois', '1'
        ]
        subprocess.run(cmd_gen, check=True)
        
        # Load one file (pick 470/410 cols)
        fname = [f for f in os.listdir(out_dir) if f.endswith('.csv')][0]
        d = pd.read_csv(os.path.join(out_dir, fname))
        
        # De-interleave
        leds = d['LedState'].values
        # 1=410, 2=470
        mask_410 = (leds & 1) > 0
        mask_470 = (leds & 2) > 0
        
        # Assuming Region0G
        r0 = d['Region0G'].values
        
        u = r0[mask_410]
        s = r0[mask_470]
        
        # Determine min length (rare dropping)
        n = min(len(u), len(s))
        if n < 100: self.fail("Not enough data")
        u = u[:n]
        s = s[:n]
        
        # Masking logic
        du = np.diff(u)
        du = np.concatenate(([0], du))
        motion_mask = np.abs(du) > 3.0
        motion_mask = np.convolve(motion_mask.astype(float), np.ones(5), mode='same') > 0
        
        thr = np.percentile(s, 95)
        spike_mask = s > thr
        
        mask = ~(motion_mask | spike_mask)
        
        u_clean = u[mask]
        s_clean = s[mask]
        
        n_clean = len(u_clean)
        if n_clean < 20: self.fail("Too much masking")
        
        mid = n_clean // 2
        res1 = linregress(u_clean[:mid], s_clean[:mid])
        res2 = linregress(u_clean[mid:], s_clean[mid:])
        
        slope_diff = abs(res1.slope - res2.slope)
        self.assertGreater(slope_diff, 0.05, f"Slopes did not diverge enough (d={slope_diff:.3f}) with gain drift")

if __name__ == '__main__':
    unittest.main()
