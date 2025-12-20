
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
        pass

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
        os.makedirs(os.path.join(output_dir, 'qc'), exist_ok=True)
        self.discover_files(input_dir, recursive, glob_pattern, force_format=force_format)
        
        # 1. Run Report (Pre-Analysis)
        generate_run_report(self.config, output_dir)
        
        self.run_pass_1(force_format)
        
        # Robustness: Baseline Check & Report Update
        from .core.reporting import append_run_report_warnings
        baseline_warnings = []
        invalid_rois = []
        
        # Robustness: Baseline Check & Report Update
        from .core.reporting import append_run_report_warnings
        baseline_warnings = []
        invalid_rois = []
        
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
             # Approx count if we don't know exact chunk count yet (Pipeline.run continues after this block? 
             # No, run_pass_2 is called AFTER this block. So valid total_chunks is not known here?)
             # Wait, `run_pass_2` sets `total_chunks`.
             # The user asked for "Whenever invalid baselines exist... ensure qc_summary.json contains...".
             # `qc_summary` is written inside `run_pass_2`. 
             # So I should populate `qc_summary` here with the ROIs, and let `run_pass_2` compute the chunk counts?
             # Or just pass the list to `run_pass_2`.
             # I am updating `self.qc_summary` here. `run_pass_2` uses `self.qc_summary`.
             # So I should update `run_pass_2` logic to compute the counts from the list.
             pass

        self.run_pass_2(output_dir, force_format)
        
        # D1: Add to run_metadata (Requires pass 2 completion to grab final metadata dict? No, Pipeline.run structure runs pass 2 then prints done)
        # run_metadata is written INSIDE run_pass_2.
        # So I need to pass `invalid_rois` to run_pass_2 or store efficiently.
        # I stored it in self.qc_summary['invalid_baseline_rois'].
        # I will update run_pass_2 to read from there for metadata.
        pass
        print("Pipeline Done.")
