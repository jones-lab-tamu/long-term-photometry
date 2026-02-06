
import os
import glob
import json
import yaml
import logging
import pandas as pd
import numpy as np
from typing import List, Optional

from .config import Config
from .core.types import Chunk, SessionStats
from .io.adapters import load_chunk, sniff_format
from .core import preprocessing, regression, normalization, feature_extraction, baseline
from .core.reporting import generate_run_report, append_run_report_warnings
from .viz import plots

# Helper for robust config access (Dict vs Attribute)
def _get_cfg_value(cfg, key, default):
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default

class Pipeline:
    def __init__(self, config: Config, mode: str = 'phasic'):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.stats = SessionStats()
        self.stats.tonic_fit_params = {} # ROI -> {slope, intercept} (Ad-hoc extension)
        self.qc_summary = {
            'failed_chunks': [],
            'chunk_fail_fraction': 0.0,
            'roi_failures': {}
        }
        self.roi_map = {}

    def discover_files(self, input_path: str, recursive: bool = False, file_glob: str = "*.csv", force_format: str = 'auto'):
        if force_format == 'rwd':
            # RWD Discovery: Treat input_path as root containing timestamped subdirectories
            from .io.adapters import discover_rwd_chunks
            self.file_list = discover_rwd_chunks(input_path)
        elif os.path.isfile(input_path):
            self.file_list = [input_path]
        else:
            if recursive:
                search_pattern = os.path.join(input_path, "**", file_glob)
                self.file_list = glob.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(input_path, file_glob)
                self.file_list = glob.glob(search_pattern)
        
        self.file_list.sort()
        if not self.file_list:
            raise ValueError(f"No files found in {input_path}")
            
        print(f"Found {len(self.file_list)} files.")

    def _get_format(self, path: str, force_format: str) -> str:
        if force_format != 'auto':
            return force_format
        
        fmt = sniff_format(path, self.config)
        if fmt is None:
            raise ValueError(f"Could not automatically detect format for {path}. Use --format to specify.")
        return fmt

    def run_pass_1(self, force_format: str = 'auto'):
        """
        Baseline Computation.
        """
        print("Starting Pass 1: Baseline Computation...")
        
        method = self.config.baseline_method
        reservoir = baseline.DeterministicReservoir(seed=self.config.seed)
        
        if method == 'uv_raw_percentile_session':
            print("Pass 1 (Reservoir)...")
            for i, fpath in enumerate(self.file_list):
                try:
                    fmt = self._get_format(fpath, force_format)
                    chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                    
                    if not self.roi_map and chunk.metadata.get('roi_map'):
                        self.roi_map = chunk.metadata['roi_map']
                    
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        uv_data = chunk.uv_raw[:, ch_idx]
                        reservoir.add(ch_name, uv_data)
                        
                except Exception as e:
                    logging.warning(f"Pass 1: Skipping {fpath} due to error: {e}")
                    continue
            
            self.stats.method_used = method
            for ch in reservoir.buffer.keys():
                f0 = reservoir.get_percentile(ch, self.config.baseline_percentile)
                self.stats.f0_values[ch] = f0
                
        elif method == 'uv_globalfit_percentile_session':
            accumulator = baseline.GlobalFitAccumulator()
            
            print("Pass 1a (Stats)...")
            for i, fpath in enumerate(self.file_list):
                try:
                    fmt = self._get_format(fpath, force_format)
                    chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                    
                    if not self.roi_map and chunk.metadata.get('roi_map'):
                        self.roi_map = chunk.metadata['roi_map']
                    
                    # Compute filtered explicitly for fit accumulation
                    chunk.uv_filt = preprocessing.lowpass_filter(chunk.uv_raw, chunk.fs_hz, self.config)
                    chunk.sig_filt = preprocessing.lowpass_filter(chunk.sig_raw, chunk.fs_hz, self.config)
                    
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        accumulator.add(ch_name, chunk.uv_filt[:, ch_idx], chunk.sig_filt[:, ch_idx])
                        
                except Exception as e:
                    logging.warning(f"Pass 1a: Skipping {fpath}: {e}")
                    continue
            
            self.stats.global_fit_params = accumulator.solve()
            
            print("Pass 1b (Reservoir)...")
            for i, fpath in enumerate(self.file_list):
                try:
                    fmt = self._get_format(fpath, force_format)
                    chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                    
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        params = self.stats.global_fit_params.get(ch_name)
                        if params:
                            uv_val = chunk.uv_raw[:, ch_idx]
                            uv_est = params['a'] * uv_val + params['b']
                            reservoir.add(ch_name, uv_est)
                            
                except Exception as e:
                    continue
            
            self.stats.method_used = method
            for ch in reservoir.buffer.keys():
                f0 = reservoir.get_percentile(ch, self.config.baseline_percentile)
                self.stats.f0_values[ch] = f0

        print(f"Pass 1 Complete. F0: {self.stats.f0_values}")
        
        # Robustness: Check for Missing/Invalid Baselines
        from .core.reporting import append_run_report_warnings
        # We need output_dir. Check if run_pass_1 has access. No.
        # So we must move this check to run() or pass output_dir to run_pass_1.
        # Constraint: "Do not change output filenames or directory structure... No changing regression math..."
        # But changing signature of run_pass_1 might be allowed as internal API improvement for robustness? 
        # Alternatively, do this in run().
        # "After Pass 1... Append warning to run_report.json".
        # I will do this in the `run()` method right after `run_pass_1` returns.
        # Wait, run_pass_1 computes baselines. So best place is `run()`.
        # TONIC MODE: PASS 1c (Global Robust Fit)
        if self.mode == 'tonic':
            print("Pass 1c (Tonic Global Fit accumulation)...")
            from .core.tonic_dff import compute_global_iso_fit_robust
            
            # Aggregate data per channel for robust fit
            # We must fit on FULL dataset to satisfy requirement: "fit exactly once... using full arrays"
            # Using basic list accumulator (memory constrained? RWD 48h ~ 3.5e6 samples -> ~28MB per float64 column. Fine.)
            acc_uv = {}
            acc_sig = {}
            
            for i, fpath in enumerate(self.file_list):
                 try:
                    fmt = self._get_format(fpath, force_format)
                    chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        if ch_name not in acc_uv:
                            acc_uv[ch_name] = []
                            acc_sig[ch_name] = []
                        acc_uv[ch_name].append(chunk.uv_raw[:, ch_idx])
                        acc_sig[ch_name].append(chunk.sig_raw[:, ch_idx])
                 except Exception:
                     continue
            
            # Solve
            for ch in acc_uv.keys():
                uv_full = np.concatenate(acc_uv[ch])
                sig_full = np.concatenate(acc_sig[ch])
                slope, intercept, ok, n_used = compute_global_iso_fit_robust(uv_full, sig_full)
                if ok:
                    self.stats.tonic_fit_params[ch] = {'slope': slope, 'intercept': intercept}
                    print(f"  Tonic Fit ({ch}): slope={slope:.4f}, int={intercept:.4f} (N={n_used})")
                else:
                    logging.warning(f"  Tonic Fit ({ch}) FAILED.")
            
        # End Pass 1

    # Helper for Unit Testing / Invariant Enforcement
    def _process_chunk_tonic(self, chunk: Chunk, i: int):
         # Explicit Global Fit Application
         from .core.tonic_dff import apply_global_fit, compute_session_tonic_df_from_global
         
         if not hasattr(self.stats, 'tonic_fit_params'):
             raise RuntimeError("Tonic mode active but tonic_fit_params missing!")
             
         chunk.uv_fit = np.full_like(chunk.uv_raw, np.nan)
         chunk.delta_f = np.full_like(chunk.sig_raw, np.nan)
         chunk.dff = np.full_like(chunk.sig_raw, np.nan) # Derived from delta_f/F0 below
         
         # Provenance: Print exactly once per run
         if not getattr(self.stats, '_provenance_printed_tonic', False):
             print(f"Tonic iso-fit source: global robust fit (entire recording). Dynamic uv_fit ignored.")
             self.stats._provenance_printed_tonic = True
         
         # Check for missing params (Invariant A: No silent NaNs)
         missing_rois = [r for r in chunk.channel_names if r not in self.stats.tonic_fit_params]
         if missing_rois:
             raise RuntimeError(f"Chunk {i}: Missing tonic fit params for {len(missing_rois)} ROIs: {missing_rois[:5]}. Cannot compute Tonic DF.")
             
         # Tonic ROI Loop
         for r_idx, roi in enumerate(chunk.channel_names):
             params = self.stats.tonic_fit_params.get(roi)
             if params:
                  # Apply Global Fit
                  iso_fit = apply_global_fit(chunk.uv_raw[:, r_idx], params['slope'], params['intercept'])
                  chunk.uv_fit[:, r_idx] = iso_fit
                  
                  # Compute Tonic DF (Additive)
                  res = compute_session_tonic_df_from_global(chunk.sig_raw[:, r_idx], chunk.uv_raw[:, r_idx], iso_fit)
                  if not res.get('success', False):
                      reason = res.get('reason', 'Unknown failure in compute_session_tonic_df_from_global')
                      raise RuntimeError(f"Chunk {i}, ROI {roi}: Tonic DF compute failed. Reason: {reason}")
                  
                  # NaN Fraction Check (Tolerance)
                  valid_mask = res.get('valid_mask')
                  if valid_mask is None:
                      raise RuntimeError(f"Chunk {i}, ROI {roi}: Missing valid_mask from tonic DF result.")

                  n_total = len(valid_mask)
                  n_valid = int(np.sum(valid_mask))
                  frac_invalid = 1.0 - (n_valid / float(n_total)) if n_total > 0 else 1.0
                       
                  allowed_raw = _get_cfg_value(self.config, 'tonic_allowed_nan_frac', 0.0)
                  try:
                      allowed = float(allowed_raw)
                  except (ValueError, TypeError):
                      raise RuntimeError(f"Invalid tonic_allowed_nan_frac={allowed_raw!r} (type {type(allowed_raw).__name__}), must be a float")

                  if frac_invalid > allowed:
                      raise RuntimeError(f"Chunk {i}, ROI {roi}: Tonic NaN fraction ({frac_invalid:.4f}) exceeds allowed ({allowed}).")

                  chunk.delta_f[:, r_idx] = res['df']
         
         
         # Invariant Post-Check: Ensure no NaNs in explicitly computed ROIs
         # RELAXED for Nan-Tolerance Logic: normalization.compute_dff will handle excessive NaN checking.
         # for r_idx, roi in enumerate(chunk.channel_names):
         #     if np.any(np.isnan(chunk.delta_f[:, r_idx])):
         #         raise RuntimeError(f"Chunk {i}, ROI {roi}: Tonic delta_f contains NaNs after computation. Strict invariant violated.")

         # Compute dFF (using normalization.compute_dff which uses chunk.delta_f / F0)
         chunk.dff = normalization.compute_dff(chunk, self.stats, self.config)

    def run_pass_2(self, output_dir: str, force_format: str = 'auto'):
        print("Starting Pass 2: Analysis...")
        
        traces_dir = os.path.join(output_dir, 'traces')
        # Robustness: strict directory creation
        os.makedirs(os.path.join(output_dir, 'qc'), exist_ok=True)
        os.makedirs(traces_dir, exist_ok=True)
        
        all_features = []
        failed_count = 0
        first_success_chunk = None
        
        print("Pass 2 (Analysis)...")
        for i, fpath in enumerate(self.file_list):
            try:
                fmt = self._get_format(fpath, force_format)
                chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                
                # Capture ROI map if missing (e.g. if Pass 1 failed or skipped)
                if not self.roi_map and chunk.metadata.get("roi_map"):
                    self.roi_map = chunk.metadata["roi_map"]
                
                chunk.uv_filt = preprocessing.lowpass_filter(chunk.uv_raw, chunk.fs_hz, self.config)
                chunk.sig_filt = preprocessing.lowpass_filter(chunk.sig_raw, chunk.fs_hz, self.config)
                
                if self.mode == 'tonic':
                     self._process_chunk_tonic(chunk, i)
                     
                else:
                     # PHASIC MODE (Dynamic)
                     uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, self.config, mode=self.mode)
                     chunk.uv_fit = uv_fit
                     chunk.delta_f = delta_f
                
                     chunk.dff = normalization.compute_dff(chunk, self.stats, self.config)
                
                feats_df = feature_extraction.extract_features(chunk, self.config)
                all_features.append(feats_df)
                
                trace_data = {'time_sec': chunk.time_sec}
                for r_idx, roi in enumerate(chunk.channel_names):
                    trace_data[f'{roi}_uv_raw'] = chunk.uv_raw[:, r_idx]
                    trace_data[f'{roi}_sig_raw'] = chunk.sig_raw[:, r_idx]
                    
                    if chunk.uv_fit is not None:
                        trace_data[f'{roi}_uv_fit'] = chunk.uv_fit[:, r_idx]
                    if chunk.delta_f is not None:
                        trace_data[f'{roi}_deltaF'] = chunk.delta_f[:, r_idx]
                    if chunk.dff is not None:
                        trace_data[f'{roi}_dff'] = chunk.dff[:, r_idx]
                        
                trace_df = pd.DataFrame(trace_data)
                trace_path = os.path.join(traces_dir, f"chunk_{i:04d}.csv")
                trace_df.to_csv(trace_path, index=False)
                
                if self.mode == 'phasic':
                    # Strict Verification Artifacts (Step 2 of Protocol)
                    inter_dir = os.path.join(self.output_dir, 'phasic_intermediates')
                    os.makedirs(inter_dir, exist_ok=True)
                    
                    for r_idx, roi in enumerate(chunk.channel_names):
                        # 1. CSV
                        meta_df = pd.DataFrame({
                            'time_sec': chunk.time_sec,
                            'sig_raw': chunk.sig_raw[:, r_idx],
                            'iso_raw': chunk.uv_raw[:, r_idx],
                            'fit_ref': chunk.uv_fit[:, r_idx] if chunk.uv_fit is not None else np.nan,
                            'residual': chunk.delta_f[:, r_idx] if chunk.delta_f is not None else np.nan,
                            'dff': chunk.dff[:, r_idx] if chunk.dff is not None else np.nan
                        })
                        # Filename: chunk_XXX_{roi}.csv to handle multi-roi
                        # User requested chunk_XXX.csv but implied per-ROI columns or structure. 
                        # To be safe for multi-ROI, we must distinguish.
                        # We will use chunk_0000_Region0.csv pattern.
                        csv_path = os.path.join(inter_dir, f"chunk_{i:04d}_{roi}.csv")
                        meta_df.to_csv(csv_path, index=False)
                        
                        # 2. Meta JSON
                        meta = {
                            'fs_hz': float(chunk.fs_hz),
                            'fit_method': 'dynamic_windowed' if self.config.window_sec > 0 else 'global', # Simplified
                            'window_sec': self.config.window_sec,
                            'peak_method': self.config.peak_threshold_method,
                            'peak_k': self.config.peak_threshold_k
                        }
                        json_path = os.path.join(inter_dir, f"chunk_{i:04d}_{roi}_meta.json")
                        with open(json_path, 'w') as f:
                            json.dump(meta, f, indent=2)

                # VIZ: Plot Set A & D (First SUCCESSFUL Chunk Only)
                if first_success_chunk is None:
                    first_success_chunk = chunk
                    viz_dir = os.path.join(output_dir, 'viz')
                    os.makedirs(viz_dir, exist_ok=True)
                    plots.set_style()
                    for roi in chunk.channel_names:
                        try:
                            # A: Single Session Raw
                            plots.plot_single_session_raw(chunk, roi, viz_dir)
                            # D: Correction Impact (Use full chunk as interval)
                            plots.plot_correction_impact(chunk, roi, slice(None), viz_dir)
                        except Exception as e:
                            logging.warning(f"Viz failure (chunk {i}) {roi}: {e}")
                
            except Exception as e:
                logging.error(f"Failed chunk {i} ({fpath}): {e}")
                self.qc_summary['failed_chunks'].append({'file': fpath, 'error': str(e)})
                failed_count += 1
                
        if all_features:
            full_feats = pd.concat(all_features, ignore_index=True)
            feats_dir = os.path.join(output_dir, 'features')
            os.makedirs(feats_dir, exist_ok=True)
            
            full_feats.to_csv(os.path.join(feats_dir, 'features.csv'), index=False)

        total_chunks = len(self.file_list)
        if total_chunks > 0:
            self.qc_summary['chunk_fail_fraction'] = failed_count / total_chunks
            
        # Robustness: Add baseline invalid counts if tracked
        if 'invalid_baseline_rois' in self.qc_summary:
            bad_rois = self.qc_summary['invalid_baseline_rois']
            # D3: Ensure explicit counts always present if key exists
            self.qc_summary['baseline_invalid_roi_count'] = len(bad_rois)
            total_affected = len(bad_rois) * total_chunks
            self.qc_summary['baseline_invalid_roi_chunk_pairs'] = total_affected
            if bad_rois:
                logging.warning(f"Baseline invalid for {len(bad_rois)} ROIs across {total_chunks} chunks ({total_affected} pairs).")
            
        with open(os.path.join(output_dir, 'qc', 'qc_summary.json'), 'w') as f:
            json.dump(self.qc_summary, f, indent=2)
            
        run_meta = {
            'target_fs_hz': self.config.target_fs_hz,
            'seed': self.config.seed,
            'allow_partial_final_chunk': self.config.allow_partial_final_chunk,
            'roi_map': self.roi_map,
            'baseline_method': self.stats.method_used,
            'f0_values': self.stats.f0_values,
            'global_fit_params': self.stats.global_fit_params,
            # Validation Metadata for Paper Alignment
            'f0_source': self.stats.method_used,
            'phasic_uv_fit_method': 'dynamic', # Strict requirement for this pipeline version
            'f0_is_from_uv_fit': False,        # Constraint: explicit separation
            'regression_window_sec': self.config.window_sec,
            'regression_step_sec': self.config.step_sec,
            'regression_window_sec': self.config.window_sec,
            'regression_step_sec': self.config.step_sec,
            'regression_mode': self.mode,
            # D1: Write invalid baseline ROIs
            'invalid_baseline_rois': self.qc_summary.get('invalid_baseline_rois', [])
        }
        with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
            json.dump(run_meta, f, indent=2)
            
        if self.qc_summary['chunk_fail_fraction'] > self.config.qc_max_chunk_fail_fraction:
            logging.error(f"High failure rate: {self.qc_summary['chunk_fail_fraction']:.2%}")

        # -----------------------------
        # VIZ: Canonical Visualization
        # -----------------------------
        print("Generating visualizations...")
        plots.set_style()
        viz_dir = os.path.join(output_dir, 'viz')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Get ROIs from stats or first chunk
        rois = list(self.stats.f0_values.keys()) if self.stats.f0_values else []
        if not rois and self.roi_map: rois = list(self.roi_map.keys())

        # Trace files list
        trace_files = sorted([f for f in os.listdir(traces_dir) if f.endswith('.csv')])
        
        for roi in rois:
            try:
                # Plot B: Continuous
                plots.plot_continuous_multiday(traces_dir, roi, viz_dir, trace_files)
                # Plot C: Stacked
                plots.plot_stacked_session(traces_dir, roi, viz_dir, trace_files)
            except Exception as e:
                logging.warning(f"Viz failure for {roi}: {e}")
                
    def run(self, input_dir: str, output_dir: str, force_format: str = 'auto', recursive: bool = False, glob_pattern: str = "*.csv"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'qc'), exist_ok=True)
        self.discover_files(input_dir, recursive, glob_pattern, force_format=force_format)
        
        # 1. Run Report (Pre-Analysis)
        generate_run_report(self.config, output_dir)
        
        self.run_pass_1(force_format)
        
        baseline_warnings = []
        invalid_rois = []
        
        # Robustness: Always track these keys
        self.qc_summary['invalid_baseline_rois'] = []
        self.qc_summary['baseline_invalid_roi_count'] = 0
        
        # D2: ROI Union
        keys_map = list(self.roi_map.keys()) if self.roi_map else []
        keys_stats = list(self.stats.f0_values.keys())
        all_known_rois = sorted(list(set(keys_map) | set(keys_stats)))
        
        for roi in all_known_rois:
            f0 = self.stats.f0_values.get(roi, float('nan'))
            if np.isnan(f0) or np.isinf(f0) or f0 <= self.config.f0_min_value:
                invalid_rois.append(roi)
                baseline_warnings.append(f"Invalid F0 for ROI '{roi}': {f0}. (Min allowed: {self.config.f0_min_value})")
                
        if baseline_warnings:
             append_run_report_warnings(output_dir, baseline_warnings)
             self.qc_summary['invalid_baseline_rois'] = invalid_rois
             self.qc_summary['baseline_invalid_roi_count'] = len(invalid_rois)

        self.run_pass_2(output_dir, force_format)
        
        print("Pipeline Done.")
