
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
from .core.reporting import generate_run_report
from .viz import plots

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
        self.roi_map = {}

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

    def run_pass_2(self, output_dir: str, force_format: str = 'auto'):
        print("Starting Pass 2: Analysis...")
        
        traces_dir = os.path.join(output_dir, 'traces')
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
                
                uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, self.config)
                chunk.uv_fit = uv_fit
                chunk.delta_f = delta_f
                
                chunk.dff = normalization.compute_dff(chunk, self.stats)
                
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
            try:
                full_feats.to_parquet(os.path.join(feats_dir, 'features.parquet'))
            except ImportError:
                logging.warning("pyarrow/fastparquet not installed, skipping parquet output")

        total_chunks = len(self.file_list)
        if total_chunks > 0:
            self.qc_summary['chunk_fail_fraction'] = failed_count / total_chunks
            
        with open(os.path.join(output_dir, 'qc', 'qc_summary.json'), 'w') as f:
            json.dump(self.qc_summary, f, indent=2)
            
        run_meta = {
            'target_fs_hz': self.config.target_fs_hz,
            'seed': self.config.seed,
            'allow_partial_final_chunk': self.config.allow_partial_final_chunk,
            'roi_map': self.roi_map,
            'baseline_method': self.stats.method_used,
            'f0_values': self.stats.f0_values,
            'global_fit_params': self.stats.global_fit_params
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
        os.makedirs(os.path.join(output_dir, 'qc'), exist_ok=True)
        self.discover_files(input_dir, recursive, glob_pattern)
        
        # 1. Run Report (Pre-Analysis)
        generate_run_report(self.config, output_dir)
        
        self.run_pass_1(force_format)
        self.run_pass_2(output_dir, force_format)
        print("Pipeline Done.")
