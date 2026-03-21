
import os
import glob
import json
import yaml
import logging
import pathlib
import time
import pandas as pd
import numpy as np
from typing import List, Optional

from .config import Config
from .core.types import Chunk, SessionStats
from .io.adapters import load_chunk, sniff_format
from .core import preprocessing, regression, normalization, feature_extraction, baseline
from .core.utils import natural_sort_key
from .core.reporting import generate_run_report, append_run_report_warnings
# from .viz import plots # Moved to run() to avoid side effects

def _sanitize_metadata(obj):
    """
    Recursively convert metadata to JSON-safe primitives.
    Handles numpy types explicitly so they become Python scalars/lists.
    Unknown types are converted to repr(obj).
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize_metadata(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_metadata(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)

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
        self._pass1_manifest = []
        self._selected_rois = None
        self.roi_selection = None
        self.traces_only = False
        self._phasic_started_at = None
        self._phasic_phase_buckets = {}
        self._phasic_detail_buckets = {}
        self._phasic_metrics = {}

    def _is_phasic_timing_enabled(self) -> bool:
        return self.mode == 'phasic'

    def _add_phasic_phase_bucket(self, bucket: str, elapsed_sec: float):
        if not self._is_phasic_timing_enabled():
            return
        self._phasic_phase_buckets[bucket] = self._phasic_phase_buckets.get(bucket, 0.0) + float(elapsed_sec)

    def _add_phasic_detail_bucket(self, bucket: str, elapsed_sec: float):
        if not self._is_phasic_timing_enabled():
            return
        self._phasic_detail_buckets[bucket] = self._phasic_detail_buckets.get(bucket, 0.0) + float(elapsed_sec)

    def _set_phasic_metric(self, name: str, value):
        if not self._is_phasic_timing_enabled():
            return
        self._phasic_metrics[name] = value

    def _add_phasic_metric(self, name: str, delta):
        if not self._is_phasic_timing_enabled():
            return
        self._phasic_metrics[name] = self._phasic_metrics.get(name, 0) + delta

    def _emit_phasic_timing_details(self, total_elapsed_sec: float):
        if not self._is_phasic_timing_enabled():
            return

        for bucket in sorted(self._phasic_phase_buckets.keys()):
            elapsed = self._phasic_phase_buckets[bucket]
            print(f"TIMING DETAIL phase=phasic_analysis bucket={bucket} elapsed_sec={elapsed:.3f}", flush=True)

        for bucket in sorted(self._phasic_detail_buckets.keys()):
            elapsed = self._phasic_detail_buckets[bucket]
            print(f"TIMING DETAIL phase=phasic_analysis bucket={bucket} elapsed_sec={elapsed:.3f}", flush=True)

        phase_explicit_sum = float(sum(self._phasic_phase_buckets.values()))
        phase_remainder = max(0.0, float(total_elapsed_sec) - phase_explicit_sum)
        print(f"TIMING DETAIL phase=phasic_analysis bucket=phase.remainder elapsed_sec={phase_remainder:.3f}", flush=True)
        print(f"TIMING DETAIL phase=phasic_analysis bucket=phase.total elapsed_sec={float(total_elapsed_sec):.3f}", flush=True)

        for name in sorted(self._phasic_metrics.keys()):
            value = self._phasic_metrics[name]
            print(f"TIMING METRIC phase=phasic_analysis name={name} value={value}", flush=True)

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
                if force_format == 'auto' and not self.file_list:
                    # Auto-mode parity: allow RWD roots with chunk subdirectories.
                    from .io.adapters import discover_rwd_chunks
                    try:
                        self.file_list = discover_rwd_chunks(input_path)
                    except Exception:
                        self.file_list = []
        
        self.file_list.sort(key=natural_sort_key)
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
        pass1_started = time.perf_counter()
        pass1_chunk_load_sec = 0.0
        pass1_filter_sec = 0.0
        pass1_accumulate_sec = 0.0
        pass1_solve_sec = 0.0
        pass1_f0_compute_sec = 0.0
        
        method = self.config.baseline_method
        reservoir = baseline.DeterministicReservoir(seed=self.config.seed)
        
        if method == 'uv_raw_percentile_session':
            print("Pass 1 (Reservoir)...")
            for i, fpath in enumerate(self.file_list):
                try:
                    t_load = time.perf_counter()
                    fmt = self._get_format(fpath, force_format)
                    chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                    if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                    pass1_chunk_load_sec += (time.perf_counter() - t_load)
                    
                    if not self.roi_map and chunk.metadata.get('roi_map'):
                        self.roi_map = chunk.metadata['roi_map']
                    
                    t_acc = time.perf_counter()
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        uv_data = chunk.uv_raw[:, ch_idx]
                        reservoir.add(ch_name, uv_data)
                    pass1_accumulate_sec += (time.perf_counter() - t_acc)
                        
                    if fpath not in self._pass1_manifest:
                        self._pass1_manifest.append(fpath)
                        
                except Exception as e:
                    logging.warning(f"Pass 1: Skipping {fpath} due to error: {e}")
                    if not any(x['file'] == fpath for x in self.qc_summary['failed_chunks']):
                        self.qc_summary['failed_chunks'].append({'file': fpath, 'error': str(e)})
                    continue
            
            self.stats.method_used = method
            t_f0 = time.perf_counter()
            for ch in reservoir.buffer.keys():
                f0 = reservoir.get_percentile(ch, self.config.baseline_percentile)
                self.stats.f0_values[ch] = f0
            pass1_f0_compute_sec += (time.perf_counter() - t_f0)
                
        elif method == 'uv_globalfit_percentile_session':
            accumulator = baseline.GlobalFitAccumulator()
            
            print("Pass 1a (Stats)...")
            for i, fpath in enumerate(self.file_list):
                try:
                    t_load = time.perf_counter()
                    fmt = self._get_format(fpath, force_format)
                    chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                    if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                    pass1_chunk_load_sec += (time.perf_counter() - t_load)
                    
                    if not self.roi_map and chunk.metadata.get('roi_map'):
                        self.roi_map = chunk.metadata['roi_map']
                    
                    # Compute filtered explicitly for fit accumulation
                    t_filter = time.perf_counter()
                    chunk.uv_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, self.config)
                    chunk.sig_filt, _ = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, self.config)
                    pass1_filter_sec += (time.perf_counter() - t_filter)
                    
                    t_acc = time.perf_counter()
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        accumulator.add(ch_name, chunk.uv_filt[:, ch_idx], chunk.sig_filt[:, ch_idx])
                    pass1_accumulate_sec += (time.perf_counter() - t_acc)
                        
                except Exception as e:
                    logging.warning(f"Pass 1a: Skipping {fpath}: {e}")
                    if not any(x['file'] == fpath for x in self.qc_summary['failed_chunks']):
                        self.qc_summary['failed_chunks'].append({'file': fpath, 'error': str(e)})
                    continue
            
            t_solve = time.perf_counter()
            self.stats.global_fit_params = accumulator.solve()
            pass1_solve_sec += (time.perf_counter() - t_solve)
            
            print("Pass 1b (Reservoir)...")
            for i, fpath in enumerate(self.file_list):
                try:
                    t_load = time.perf_counter()
                    fmt = self._get_format(fpath, force_format)
                    chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                    if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                    pass1_chunk_load_sec += (time.perf_counter() - t_load)
                    
                    t_acc = time.perf_counter()
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        params = self.stats.global_fit_params.get(ch_name)
                        if params:
                            uv_val = chunk.uv_raw[:, ch_idx]
                            uv_est = params['a'] * uv_val + params['b']
                            reservoir.add(ch_name, uv_est)
                    pass1_accumulate_sec += (time.perf_counter() - t_acc)
                            
                    if fpath not in self._pass1_manifest:
                        self._pass1_manifest.append(fpath)
                except Exception as e:
                    continue
            
            self.stats.method_used = method
            t_f0 = time.perf_counter()
            for ch in reservoir.buffer.keys():
                f0 = reservoir.get_percentile(ch, self.config.baseline_percentile)
                self.stats.f0_values[ch] = f0
            pass1_f0_compute_sec += (time.perf_counter() - t_f0)

        print(f"Pass 1 Complete. F0: {self.stats.f0_values}")
        if self._is_phasic_timing_enabled():
            pass1_total_sec = time.perf_counter() - pass1_started
            self._add_phasic_detail_bucket("pass1.total", pass1_total_sec)
            self._add_phasic_detail_bucket("pass1.chunk_load", pass1_chunk_load_sec)
            self._add_phasic_detail_bucket("pass1.filter", pass1_filter_sec)
            self._add_phasic_detail_bucket("pass1.accumulate", pass1_accumulate_sec)
            self._add_phasic_detail_bucket("pass1.solve", pass1_solve_sec)
            self._add_phasic_detail_bucket("pass1.f0_compute", pass1_f0_compute_sec)
            pass1_explicit = (
                pass1_chunk_load_sec
                + pass1_filter_sec
                + pass1_accumulate_sec
                + pass1_solve_sec
                + pass1_f0_compute_sec
            )
            self._add_phasic_detail_bucket("pass1.remainder", max(0.0, pass1_total_sec - pass1_explicit))
            self._set_phasic_metric("pass1.files_seen", len(self.file_list))
            self._set_phasic_metric("pass1.manifest_size", len(self._pass1_manifest))
            self._set_phasic_metric("pass1.baseline_method", method)
        
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
                    if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                    for ch_idx, ch_name in enumerate(chunk.channel_names):
                        if ch_name not in acc_uv:
                            acc_uv[ch_name] = []
                            acc_sig[ch_name] = []
                        acc_uv[ch_name].append(chunk.uv_raw[:, ch_idx])
                        acc_sig[ch_name].append(chunk.sig_raw[:, ch_idx])
                        
                    if fpath not in self._pass1_manifest:
                        self._pass1_manifest.append(fpath)
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

    def _apply_standard_analysis(self, chunk, chunk_id):
        """
        Shared source of truth for standard analysis steps (preprocessing -> regression -> normalization).
        Returns the processed chunk.
        """
        t_filter = time.perf_counter()
        uv_filt, uv_meta = preprocessing.lowpass_filter_with_meta(chunk.uv_raw, chunk.fs_hz, self.config)
        sig_filt, sig_meta = preprocessing.lowpass_filter_with_meta(chunk.sig_raw, chunk.fs_hz, self.config)
        if self._is_phasic_timing_enabled():
            self._add_phasic_detail_bucket("pass2.filter_lowpass", time.perf_counter() - t_filter)
        
        chunk.uv_filt = uv_filt
        chunk.sig_filt = sig_filt
        
        # Warning aggregation (NaN Safety Policy 2)
        for m in [uv_meta, sig_meta]:
             if m.get('rois_affected', 0) > 0:
                 msg = f"Chunk {chunk_id} Block-wise filtering active: {m['rois_affected']} ROIs, {m['samples_skipped']} samples skipped."
                 if 'scan_warnings' not in self.qc_summary: self.qc_summary['scan_warnings'] = []
                 self.qc_summary['scan_warnings'].append(msg)
        
        if self.mode == 'tonic':
             self._process_chunk_tonic(chunk, chunk_id)
        else:
             # PHASIC MODE (Dynamic)
             t_reg = time.perf_counter()
             uv_fit, delta_f = regression.fit_chunk_dynamic(chunk, self.config, mode=self.mode)
             if self._is_phasic_timing_enabled():
                 self._add_phasic_detail_bucket("pass2.dynamic_regression", time.perf_counter() - t_reg)
                 dyn_timing = None
                 if hasattr(chunk, 'metadata') and chunk.metadata:
                     dyn_timing = chunk.metadata.get('dynamic_regression_timing')
                 if isinstance(dyn_timing, dict):
                     for sub_bucket, sub_elapsed in dyn_timing.get('buckets', {}).items():
                         self._add_phasic_detail_bucket(
                             f"pass2.dynamic_regression.{sub_bucket}",
                             float(sub_elapsed)
                         )
                     for metric_name, metric_value in dyn_timing.get('metrics', {}).items():
                         self._add_phasic_metric(
                             f"pass2.dynamic_regression.{metric_name}",
                             metric_value
                         )
             chunk.uv_fit = uv_fit
             chunk.delta_f = delta_f
             t_dff = time.perf_counter()
             chunk.dff = normalization.compute_dff(chunk, self.stats, self.config)
             if self._is_phasic_timing_enabled():
                 self._add_phasic_detail_bucket("pass2.dff_compute", time.perf_counter() - t_dff)
        
        return chunk

    def _resolve_representative_session(self, force_format: str, emitter=None):
        """
        Resolves the representative session only after file_list is finalized.
        Implements legacy fallback and emits audit event.
        """
        n_sessions_resolved = len(self.file_list)
        user_idx = self.config.representative_session_index
        rep_idx_effective = None
        rep_session_id = None
        user_provided = (user_idx is not None)

        if user_provided:
            if not isinstance(user_idx, int) or not (0 <= user_idx < n_sessions_resolved):
                raise ValueError(f"representative_session_index out of range: idx={user_idx}, n_sessions={n_sessions_resolved}")
            rep_idx_effective = user_idx
            rep_session_id = self._session_entry_to_id(self.file_list[user_idx])
        else:
            # Legacy Default: find first loadable
            for i, fpath in enumerate(self.file_list):
                try:
                    fmt = self._get_format(fpath, force_format)
                    load_chunk(fpath, fmt, self.config, chunk_id=i) # Validation load
                    rep_idx_effective = i
                    rep_session_id = self._session_entry_to_id(fpath)
                    break
                except Exception as e:
                    logging.warning(f"Resolution: Skipping session {fpath} for representative selection: {e}")
        
        self.representative_session_index = rep_idx_effective
        self.representative_session_id = rep_session_id
        self.n_sessions_resolved = n_sessions_resolved
        self.representative_user_provided = user_provided

        self.representative_session_info = {
            "representative_session_index": self.representative_session_index,
            "representative_session_id": self.representative_session_id,
            "n_sessions_resolved": self.n_sessions_resolved,
            "resolved_session_ids_preview": [self._session_entry_to_id(f) for f in self.file_list[:5]],
            "user_provided": self.representative_user_provided
        }

        if emitter:
            emitter.emit("inputs", "representative_session", "Representative session resolved",
                         payload=self.representative_session_info)


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
                       
                  allowed_raw = getattr(self.config, 'tonic_allowed_nan_frac', 0.0)
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
        # Lazy import for VIZ
        from .viz import plots
        pass2_started = time.perf_counter()
        pass2_manifest_check_sec = 0.0
        pass2_chunk_read_sec = 0.0
        pass2_feature_extract_sec = 0.0
        pass2_cache_write_sec = 0.0
        pass2_qc_scan_sec = 0.0
        pass2_features_csv_write_sec = 0.0
        pass2_qc_summary_write_sec = 0.0
        pass2_run_metadata_write_sec = 0.0
        pass2_rep_validation_sec = 0.0
        pass2_chunks_processed = 0
        pass2_sample_rows_processed = 0
        pass2_roi_samples_processed = 0
        pass2_features_rows = 0
        pass2_peak_count_total = 0
        
        # Robustness: strict directory creation
        os.makedirs(os.path.join(output_dir, 'qc'), exist_ok=True)
        
        all_features = []
        rep_chunk_for_plotting = None
        rep_idx = self.representative_session_index  # set by _resolve_representative_session
        
        # Freeze manifest to ensure it cannot be mutated after Pass 1
        frozen_manifest = tuple(self._pass1_manifest)
        
        # Check for new files not in pass 1 manifest
        t_manifest = time.perf_counter()
        new_files = [f for f in self.file_list if f not in frozen_manifest]
        for f in new_files:
            if not any(x['file'] == f for x in self.qc_summary['failed_chunks']):
                self.qc_summary['failed_chunks'].append({'file': f, 'error': 'Ignored (Not in Pass 1 manifest)'})
        pass2_manifest_check_sec += (time.perf_counter() - t_manifest)
                
        if new_files:
             logging.warning(f"Pass 2: Found {len(new_files)} new or skipped files not in Pass 1 manifest. First few: {new_files[:3]}. They will be ignored.")
        
        print("Pass 2 (Analysis)...")
        # Ensure we only iterate over files successfully processed in Pass 1
        for i, fpath in enumerate(frozen_manifest):
            try:
                t_read = time.perf_counter()
                fmt = self._get_format(fpath, force_format)
                chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                if self._selected_rois is not None: chunk = self._apply_roi_filter(chunk)
                pass2_chunk_read_sec += (time.perf_counter() - t_read)
                pass2_chunks_processed += 1
                if hasattr(chunk, 'sig_raw') and chunk.sig_raw is not None:
                    pass2_sample_rows_processed += int(chunk.sig_raw.shape[0])
                    pass2_roi_samples_processed += int(chunk.sig_raw.shape[0] * chunk.sig_raw.shape[1])
                
                # Capture ROI map if missing (e.g. if Pass 1 failed or skipped)
                if not self.roi_map and chunk.metadata.get("roi_map"):
                    self.roi_map = chunk.metadata["roi_map"]

                
                # SHARED PROCESSING (single source of truth for filtering, regression, dff)
                chunk = self._apply_standard_analysis(chunk, i)
                
                # Retain for representative plotting
                if rep_idx is not None and i == rep_idx:
                    rep_chunk_for_plotting = chunk

                
                # traces-only: skip feature extraction and all feature-derived outputs.
                # NOTE: This pipeline does NOT perform event detection as a separate stage.
                # "events" in this codebase refers to NDJSON lifecycle logging (engine:start,
                # engine:context, etc.), not signal event detection.  The only analysis step
                # gated here is feature_extraction.extract_features(), which computes
                # per-chunk statistics (peak count, AUC, mean, etc.).
                if not self.traces_only:
                    t_feats = time.perf_counter()
                    feats_df = feature_extraction.extract_features(chunk, self.config)
                    pass2_feature_extract_sec += (time.perf_counter() - t_feats)
                    all_features.append(feats_df)
                    pass2_features_rows += int(len(feats_df))
                    if 'peak_count' in feats_df.columns:
                        peak_sum = pd.to_numeric(feats_df['peak_count'], errors='coerce').fillna(0).sum()
                        pass2_peak_count_total += int(peak_sum)
                
                
                if hasattr(self, '_cache_writer'):
                    t_cache = time.perf_counter()
                    self._cache_writer.add_chunk(chunk, i, fpath)
                    pass2_cache_write_sec += (time.perf_counter() - t_cache)
                
                t_scan = time.perf_counter()
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    qc_warnings = chunk.metadata.get('qc_warnings', [])
                    if any("DEGENERATE" in w for w in qc_warnings):
                        if not any(x['file'] == fpath for x in self.qc_summary['failed_chunks']):
                            self.qc_summary['failed_chunks'].append({'file': fpath, 'error': 'QC: Degenerate data detected'})
                pass2_qc_scan_sec += (time.perf_counter() - t_scan)


                # Legacy VIZ was here, now moved out of loop for strict resolution/loading
                
            except Exception as e:
                # Requirement B4: Fail fast if a file in the manifest cannot be loaded in Pass 2
                raise RuntimeError(f"Pass 2: Cannot reliably read manifest file {fpath} successfully processed in Pass 1. Error: {e}")
                
        if all_features and self.mode != 'tonic':
            t_feats_write = time.perf_counter()
            full_feats = pd.concat(all_features, ignore_index=True)
            feats_dir = os.path.join(output_dir, 'features')
            os.makedirs(feats_dir, exist_ok=True)
            
            full_feats.to_csv(os.path.join(feats_dir, 'features.csv'), index=False)
            pass2_features_csv_write_sec += (time.perf_counter() - t_feats_write)

        total_chunks = len(self.file_list)
        if total_chunks > 0:
            self.qc_summary['chunk_fail_fraction'] = len(self.qc_summary.get('failed_chunks', [])) / total_chunks
            
        # Robustness: Add baseline invalid counts if tracked
        if 'invalid_baseline_rois' in self.qc_summary:
            bad_rois = self.qc_summary['invalid_baseline_rois']
            # D3: Ensure explicit counts always present if key exists
            self.qc_summary['baseline_invalid_roi_count'] = len(bad_rois)
            total_affected = len(bad_rois) * total_chunks
            self.qc_summary['baseline_invalid_roi_chunk_pairs'] = total_affected
            if bad_rois:
                logging.warning(f"Baseline invalid for {len(bad_rois)} ROIs across {total_chunks} chunks ({total_affected} pairs).")
            
        if self.mode != 'tonic':
            t_qc_write = time.perf_counter()
            with open(os.path.join(output_dir, 'qc', 'qc_summary.json'), 'w') as f:
                json.dump(_sanitize_metadata(self.qc_summary), f, indent=2)
            pass2_qc_summary_write_sec += (time.perf_counter() - t_qc_write)
            
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
            'regression_mode': 'dynamic' if self.mode == 'phasic' else self.mode,
            # D1: Write invalid baseline ROIs
            'invalid_baseline_rois': self.qc_summary.get('invalid_baseline_rois', [])
        }
        t_meta_write = time.perf_counter()
        with open(os.path.join(output_dir, 'run_metadata.json'), 'w') as f:
            json.dump(_sanitize_metadata(run_meta), f, indent=2)
        pass2_run_metadata_write_sec += (time.perf_counter() - t_meta_write)
            
        if self.qc_summary['chunk_fail_fraction'] > self.config.qc_max_chunk_fail_fraction:
            logging.error(f"High failure rate: {self.qc_summary['chunk_fail_fraction']:.2%}")

        # -----------------------------
        # Representative Session Validation
        # -----------------------------
        rep_fpath = self.file_list[rep_idx] if (rep_idx is not None and 0 <= rep_idx < len(self.file_list)) else None
        
        t_rep = time.perf_counter()
        if rep_chunk_for_plotting is None:
            if self.representative_user_provided:
                raise RuntimeError(
                    f"FAILED to process requested representative session "
                    f"(index={rep_idx}, file={rep_fpath}, stage=analysis/pass-2). "
                    f"Session was not successfully processed during Pass 2."
                )
        pass2_rep_validation_sec += (time.perf_counter() - t_rep)

        if self._is_phasic_timing_enabled():
            pass2_total_sec = time.perf_counter() - pass2_started
            self._add_phasic_detail_bucket("pass2.total", pass2_total_sec)
            self._add_phasic_detail_bucket("pass2.manifest_check", pass2_manifest_check_sec)
            self._add_phasic_detail_bucket("pass2.chunk_read", pass2_chunk_read_sec)
            self._add_phasic_detail_bucket("pass2.feature_extraction", pass2_feature_extract_sec)
            self._add_phasic_detail_bucket("pass2.cache_write", pass2_cache_write_sec)
            self._add_phasic_detail_bucket("pass2.qc_warning_scan", pass2_qc_scan_sec)
            self._add_phasic_detail_bucket("pass2.features_csv_write", pass2_features_csv_write_sec)
            self._add_phasic_detail_bucket("pass2.qc_summary_write", pass2_qc_summary_write_sec)
            self._add_phasic_detail_bucket("pass2.run_metadata_write", pass2_run_metadata_write_sec)
            self._add_phasic_detail_bucket("pass2.rep_validation", pass2_rep_validation_sec)
            pass2_explicit = (
                pass2_manifest_check_sec
                + pass2_chunk_read_sec
                + pass2_feature_extract_sec
                + pass2_cache_write_sec
                + pass2_qc_scan_sec
                + pass2_features_csv_write_sec
                + pass2_qc_summary_write_sec
                + pass2_run_metadata_write_sec
                + pass2_rep_validation_sec
                + self._phasic_detail_buckets.get("pass2.filter_lowpass", 0.0)
                + self._phasic_detail_buckets.get("pass2.dynamic_regression", 0.0)
                + self._phasic_detail_buckets.get("pass2.dff_compute", 0.0)
            )
            self._add_phasic_detail_bucket("pass2.remainder", max(0.0, pass2_total_sec - pass2_explicit))
            self._set_phasic_metric("pass2.chunks_processed", pass2_chunks_processed)
            self._set_phasic_metric("pass2.samples_processed_rows", pass2_sample_rows_processed)
            self._set_phasic_metric("pass2.samples_processed_roi_values", pass2_roi_samples_processed)
            self._set_phasic_metric("pass2.features_rows", pass2_features_rows)
            self._set_phasic_metric("pass2.peaks_detected_total", pass2_peak_count_total)
                
    def _apply_roi_filter(self, chunk):
        """Filter chunk data to only include channels in self._selected_rois."""
        selected = set(self._selected_rois)
        keep_idx = [i for i, name in enumerate(chunk.channel_names) if name in selected]
        chunk.channel_names = [chunk.channel_names[i] for i in keep_idx]
        chunk.uv_raw = chunk.uv_raw[:, keep_idx]
        chunk.sig_raw = chunk.sig_raw[:, keep_idx]
        if chunk.metadata and "roi_map" in chunk.metadata:
            chunk.metadata["roi_map"] = {k: v for k, v in chunk.metadata["roi_map"].items() if k in selected}
        return chunk

    def _session_entry_to_id(self, entry: str) -> str:
        """Returns a stable session ID for a file path or RWD folder."""
        p = pathlib.Path(entry)
        if p.name == "fluorescence.csv":
            return p.parent.name
        return p.stem

    def run(self, input_dir: str, output_dir: str, force_format: str = 'auto', recursive: bool = False, glob_pattern: str = "*.csv", include_rois: List[str] = None, exclude_rois: List[str] = None, traces_only: bool = False, emitter=None, sessions_per_hour: int = None):
        # Lazy import to avoid GUI side effects at module level
        from .viz import plots
        run_started = time.perf_counter()
        if self._is_phasic_timing_enabled():
            self._phasic_started_at = run_started
        
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'qc'), exist_ok=True)
        t_discovery = time.perf_counter()
        self.discover_files(input_dir, recursive, glob_pattern, force_format=force_format)
        self._add_phasic_phase_bucket("phase.input_discovery", time.perf_counter() - t_discovery)
        self._set_phasic_metric("files_discovered", len(self.file_list))
        
        # --- Preview Mode: limit to first N sessions ---
        n_total_discovered = len(self.file_list)
        preview_first_n = self.config.preview_first_n
        t_preview = time.perf_counter()
        if preview_first_n is not None:
            limit_n = min(preview_first_n, n_total_discovered)
            self.file_list = self.file_list[:limit_n]
            self.run_type = "preview"
            self.preview_info = {
                "selector": "first_n",
                "first_n": preview_first_n,
                "n_total_discovered": n_total_discovered,
                "n_sessions_resolved": len(self.file_list)
            }
            logging.info(f"Preview mode: processing {len(self.file_list)} of {n_total_discovered} discovered sessions (first_n={preview_first_n}).")
        else:
            self.run_type = "full"
            self.preview_info = None
        self._add_phasic_phase_bucket("phase.preview_selection", time.perf_counter() - t_preview)
        self._set_phasic_metric("files_after_preview", len(self.file_list))
        
        # Emit inputs:preview audit event if emitter provided
        if emitter and self.preview_info is not None:
            emitter.emit("inputs", "preview", "Preview selection resolved",
                         payload=self.preview_info)
        
        # --- ROI Discovery & Resolution ---
        roi_read_sec = 0.0
        channels_seen = []
        for i, fpath in enumerate(self.file_list):
            try:
                t_roi_read = time.perf_counter()
                fmt = self._get_format(fpath, force_format)
                chunk = load_chunk(fpath, fmt, self.config, chunk_id=i)
                roi_read_sec += (time.perf_counter() - t_roi_read)
                channels_seen.append(chunk.channel_names)
            except Exception as e:
                logging.warning(f"ROI Discovery: Failed to read {fpath}: {e}")
        self._add_phasic_phase_bucket("phase.roi_discovery_read_chunks", roi_read_sec)
        
        if not channels_seen:
            raise RuntimeError("No valid data files found for ROI discovery.")
            
        # Intersection over all valid chunks, preserving discovered order from first chunk
        t_roi_resolve = time.perf_counter()
        channel_sets = [set(cx) for cx in channels_seen]
        discovered_rois = [r for r in channels_seen[0] if all(r in cs for cs in channel_sets)]
                
        selected_rois = list(discovered_rois)
        
        if include_rois is not None:
             missing = [r for r in include_rois if r not in discovered_rois]
             if missing:
                 raise ValueError(f"Validation Error: Included ROIs not found in discovered ROIs: {missing}")
             # Preserve discovered order, filter by include_rois
             selected_rois = [r for r in discovered_rois if r in include_rois]
             
        if exclude_rois is not None:
             missing = [r for r in exclude_rois if r not in discovered_rois]
             if missing:
                 logging.warning(f"Excluded ROIs not found in discovered ROIs (ignoring): {missing}")
             selected_rois = [r for r in selected_rois if r not in exclude_rois]
             
        self.roi_selection = {
            "discovered_rois": discovered_rois,
            "include_rois": include_rois,
            "exclude_rois": exclude_rois,
            "selected_rois": selected_rois
        }
        self._add_phasic_phase_bucket("phase.roi_selection_resolution", time.perf_counter() - t_roi_resolve)
        self._set_phasic_metric("rois_discovered", len(discovered_rois))
        self._set_phasic_metric("rois_selected", len(selected_rois))
        
        self._selected_rois = selected_rois
        self.traces_only = traces_only

        # --- Representative Session Resolution ---
        t_rep = time.perf_counter()
        self._resolve_representative_session(force_format, emitter=emitter)
        self._add_phasic_phase_bucket("phase.representative_resolution", time.perf_counter() - t_rep)

        # 1. Run Report (Pre-Analysis)

        t_report = time.perf_counter()
        generate_run_report(
            self.config, output_dir, 
            roi_selection=self.roi_selection, 
            traces_only=traces_only,
            representative_info=self.representative_session_info,
            preview_info=self.preview_info,
            sessions_per_hour=sessions_per_hour,
            sessions_per_hour_source=None
        )
        self._add_phasic_phase_bucket("phase.run_report_write", time.perf_counter() - t_report)
        
        t_pass1 = time.perf_counter()
        self.run_pass_1(force_format)
        self._add_phasic_phase_bucket("phase.pass1_total", time.perf_counter() - t_pass1)
        
        baseline_warnings = []
        invalid_rois = []
        
        # Robustness: Always track these keys
        self.qc_summary['invalid_baseline_rois'] = []
        self.qc_summary['baseline_invalid_roi_count'] = 0
        
        # D2: ROI Union
        keys_map = list(self.roi_map.keys()) if self.roi_map else []
        keys_stats = list(self.stats.f0_values.keys())
        all_known_rois = sorted(list(set(keys_map) | set(keys_stats)))
        
        t_baseline_check = time.perf_counter()
        for roi in all_known_rois:
            f0 = self.stats.f0_values.get(roi, float('nan'))
            if np.isnan(f0) or np.isinf(f0) or f0 <= self.config.f0_min_value:
                invalid_rois.append(roi)
                baseline_warnings.append(f"Invalid F0 for ROI '{roi}': {f0}. (Min allowed: {self.config.f0_min_value})")
                
        if baseline_warnings:
             append_run_report_warnings(output_dir, baseline_warnings)
             self.qc_summary['invalid_baseline_rois'] = invalid_rois
             self.qc_summary['baseline_invalid_roi_count'] = len(invalid_rois)
        self._add_phasic_phase_bucket("phase.baseline_validation", time.perf_counter() - t_baseline_check)
        self._set_phasic_metric("baseline_invalid_roi_count", len(invalid_rois))

        from .io.hdf5_cache import Hdf5TraceCacheWriter
        t_cache_init = time.perf_counter()
        cache_path = os.path.join(output_dir, f"{self.mode}_trace_cache.h5")
        self._cache_writer = Hdf5TraceCacheWriter(cache_path, self.mode, self.config)
        self._add_phasic_phase_bucket("phase.cache_writer_init", time.perf_counter() - t_cache_init)
        
        try:
            t_pass2 = time.perf_counter()
            self.run_pass_2(output_dir, force_format)
            self._add_phasic_phase_bucket("phase.pass2_total", time.perf_counter() - t_pass2)
            t_finalize = time.perf_counter()
            self._cache_writer.finalize()
            self._add_phasic_phase_bucket("phase.cache_finalize", time.perf_counter() - t_finalize)
        except Exception:
            self._cache_writer.abort()
            raise

        if self._is_phasic_timing_enabled():
            dyn_total = self._phasic_detail_buckets.get("pass2.dynamic_regression", 0.0)
            dyn_sub_sum = sum(
                value
                for key, value in self._phasic_detail_buckets.items()
                if key.startswith("pass2.dynamic_regression.")
                and not key.endswith(".remainder")
                and "." not in key[len("pass2.dynamic_regression."):]
            )
            self._add_phasic_detail_bucket(
                "pass2.dynamic_regression.remainder",
                max(0.0, dyn_total - dyn_sub_sum)
            )

            wp_total = self._phasic_detail_buckets.get("pass2.dynamic_regression.window_pearson_gating", 0.0)
            wp_sub_sum = sum(
                value
                for key, value in self._phasic_detail_buckets.items()
                if key.startswith("pass2.dynamic_regression.window_pearson_gating.")
                and not key.endswith(".remainder")
            )
            self._add_phasic_detail_bucket(
                "pass2.dynamic_regression.window_pearson_gating.remainder",
                max(0.0, wp_total - wp_sub_sum)
            )

            self._set_phasic_metric("traces_only", int(bool(self.traces_only)))
            self._emit_phasic_timing_details(time.perf_counter() - run_started)
        
        print("Pipeline Done.")
