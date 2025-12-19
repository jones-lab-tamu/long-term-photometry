import os
import glob
import json
import yaml
import logging
import pandas as pd
import numpy as np
from typing import List, Optional
# from tqdm import tqdm


from .config import Config
from .core.types import Chunk, SessionStats
from .io.adapters import load_chunk, sniff_format
from .core import preprocessing, regression, normalization, feature_extraction, baseline

class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.file_list = []
        self.stats = SessionStats()
        self.qc_summary = {
            'failed_chunks': [],
            'chunk_fail_fraction': 0.0,
            'roi_failures': {}
        }

    def discover_files(self, input_path: str, recursive: bool = False, file_glob: str = "*.csv"):
        if os.path.isfile(input_path):
            self.file_list = [input_path]
        else:
            if recursive:
                search_pattern = os.path.join(input_path, "**", file_glob)
                self.file_list = glob.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(input_path, file_glob)
                self.file_list = glob.glob(search_pattern)
        
        # Sort for deterministic order
        self.file_list.sort()
        if not self.file_list:
            raise ValueError(f"No files found in {input_path}")
            
        print(f"Found {len(self.file_list)} files.")

    def _get_format(self, path: str, force_format: str) -> str:
        if force_format != 'auto':
            return force_format
        
        fmt = sniff_format(path, self.config)
        if fmt is None:
            # Hard fail as requested
            raise ValueError(f"Could not automatically detect format for {path}. Use --format to specify.")
        return fmt

    def run_pass_1(self, force_format: str = 'auto'):
        """
        Baseline Computation.
        Supports Method A (Percentile) and Method B (Global Fit + Percentile).
        """
        print("Starting Pass 1: Baseline Computation...")
        
        method = self.config.baseline_method
        reservoir = baseline.ReservoirSampler()
        
        if method == 'uv_raw_percentile_session':
            # Single Pass
            print("Pass 1 (Reservoir)...")
            for i, fpath in enumerate(self.file_list):
                try:
                    fmt = self._get_format(fpath, force_format)
                    chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                    
                    # Accumulate UV Raw
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        uv_data = chunk.uv_raw[:, ch_idx]
                        reservoir.add(ch_name, uv_data)
                        
                except Exception as e:
                    logging.warning(f"Pass 1: Skipping {fpath} due to error: {e}")
                    continue
            
            # Compute F0
            self.stats.method_used = method
            for ch in reservoir.buffer.keys():
                f0 = reservoir.get_percentile(ch, self.config.baseline_percentile)
                self.stats.f0_values[ch] = f0
                
        elif method == 'uv_globalfit_percentile_session':
            # Pass 1a: Accumulate Stats
            accumulator = baseline.GlobalFitAccumulator()
            
            print("Pass 1a (Stats)...")
            for i, fpath in enumerate(self.file_list):
                try:
                    fmt = self._get_format(fpath, force_format)
                    chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                    
                    # Only need filtered? No, usually global fit is on Raw or Filtered?
                    # Spec says: "Pass 1a: accumulate ... uv, sig ... compute a_global".
                    # Usually regression is on filtered. But application is on raw.
                    # The spec for dynamic regression says "Use FILTERED arrays for fitting".
                    # For global fit, let's assume FILTERED too for consistency in parameter estimation,
                    # but maybe raw is fine if robust.
                    # Spec section 10: "Method B... Pass 1a: accumulate Sigma uv...". Doesn't specify filtered.
                    # However, Section 9 says "Dynamic... Use FILTERED... for fitting".
                    # Let's use PREPROCESSED (Filtered) strings for the fit to be clean.
                    # So we must filter in Pass 1 too!
                    
                    chunk.uv_filt = preprocessing.lowpass_filter(chunk.uv_raw, chunk.fs_hz, self.config)
                    chunk.sig_filt = preprocessing.lowpass_filter(chunk.sig_raw, chunk.fs_hz, self.config)
                    
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        accumulator.add(ch_name, chunk.uv_filt[:, ch_idx], chunk.sig_filt[:, ch_idx])
                        
                except Exception as e:
                    logging.warning(f"Pass 1a: Skipping {fpath}: {e}")
                    continue
            
            # Solve Global Fit
            self.stats.global_fit_params = accumulator.solve()
            
            # Pass 1b: Reservoir on Fitted
            print("Pass 1b (Reservoir)...")
            for i, fpath in enumerate(self.file_list):
                try:
                    fmt = self._get_format(fpath, force_format)
                    chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                    
                    # Do we need filtered here? 
                    # "feed into ReservoirSampler... calculate uv_est = a*uv + b"
                    # uv_est should use RAW uv if we are baseline-correcting RAW signal? 
                    # Usually F0 is computed on the "artifact-corrected" trace? 
                    # Wait. F0 uses "independent calcium-insensitive reference (isosbestic-based)".
                    # Spec: "F0 uses an independent calcium-insensitive reference... never DeltaF".
                    # "Method B... stream uv_global_fit = a_global*uv_raw + b_global".
                    # So it uses RAW.
                    
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

    def run_pass_2(self, output_dir: str, force_format: str = 'auto'):
        print("Starting Pass 2: Analysis...")
        
        traces_dir = os.path.join(output_dir, 'traces')
        os.makedirs(traces_dir, exist_ok=True)
        
        all_features = []
        
        
        failed_count = 0
        
        print("Pass 2 (Analysis)...")
        for i, fpath in enumerate(self.file_list):
            try:
                fmt = self._get_format(fpath, force_format)
                chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                
                # Filter
                chunk.uv_filt = preprocessing.lowpass_filter(chunk.uv_raw, chunk.fs_hz, self.config)
                chunk.sig_filt = preprocessing.lowpass_filter(chunk.sig_raw, chunk.fs_hz, self.config)
                
                # Regress
                # Returns uv_fit, delta_f. Mutates chunk inputs? 
                # Chunk class has fields.
                uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, self.config)
                chunk.uv_fit = uv_fit
                chunk.delta_f = delta_f
                
                # Norm
                chunk.dff = normalization.compute_dff(chunk, self.stats)
                
                # Features
                feats_df = feature_extraction.extract_features(chunk, self.config)
                all_features.append(feats_df)
                
                # Save Trace
                # chunk_id, time, uv_raw, sig_raw, uv_fit, deltaF, dff
                # Per ROI? Or wide format? 
                # Spec: "per-chunk files with: time_sec, uv_raw, sig_raw, uv_fit, deltaF, dff"
                # If multiple ROIs, we need distinct columns.
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
                
            except Exception as e:
                logging.error(f"Failed chunk {i} ({fpath}): {e}")
                self.qc_summary['failed_chunks'].append({'file': fpath, 'error': str(e)})
                failed_count += 1
                
        # Aggregate Features
        if all_features:
            full_feats = pd.concat(all_features, ignore_index=True)
            # Save
            feats_dir = os.path.join(output_dir, 'features')
            os.makedirs(feats_dir, exist_ok=True)
            
            full_feats.to_csv(os.path.join(feats_dir, 'features.csv'), index=False)
            try:
                full_feats.to_parquet(os.path.join(feats_dir, 'features.parquet'))
            except ImportError:
                logging.warning("pyarrow/fastparquet not installed, skipping parquet output")

        # Final QC
        total_chunks = len(self.file_list)
        if total_chunks > 0:
            self.qc_summary['chunk_fail_fraction'] = failed_count / total_chunks
            
        with open(os.path.join(output_dir, 'qc', 'qc_summary.json'), 'w') as f:
            json.dump(self.qc_summary, f, indent=2)
            
        # Metadata
        run_meta = {
            'baseline_method': self.stats.method_used,
            'f0_values': self.stats.f0_values,
            'global_fit_params': self.stats.global_fit_params
        }
        with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
            json.dump(run_meta, f, indent=2)
            
        # Save Config Used
        with open(os.path.join(output_dir, 'config_used.yaml'), 'w') as f:
            yaml.dump(self.config.__dict__, f)
            
        if self.qc_summary['chunk_fail_fraction'] > self.config.qc_max_chunk_fail_fraction:
            logging.error(f"High failure rate: {self.qc_summary['chunk_fail_fraction']:.2%}")
            # Exit non-zero handled by CLI?
            
    def run(self, input_dir: str, output_dir: str, force_format: str = 'auto', recursive: bool = False, glob_pattern: str = "*.csv"):
        os.makedirs(os.path.join(output_dir, 'qc'), exist_ok=True)
        
        self.discover_files(input_dir, recursive, glob_pattern)
        
        self.run_pass_1(force_format)
        self.run_pass_2(output_dir, force_format)
        
        print("Pipeline Done.")
