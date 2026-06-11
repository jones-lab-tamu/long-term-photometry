"""
HDF5 Cache Writer
=================

Phase 1 helper for producing `.h5` output caches from the analysis pipeline
without rereading chunks from disk. Writes to a temporary file, then renames
to the final path upon success. Implements Safe Write Semantics.
"""
import os
import h5py
import numpy as np

class Hdf5TraceCacheWriter:
    """
    Writes trace data iteratively into an HDF5 schema defined by Phase 0 contract.
    """
    def __init__(self, output_path: str, mode: str, config):
        """
        Args:
            output_path: Final desired path, e.g., '.../tonic_out/tonic_trace_cache.h5'
            mode: 'tonic' or 'phasic'
            config: Pipeline Config object
        """
        self.output_path = output_path
        self.tmp_path = output_path + '.tmp'
        self.mode = mode
        self.config = config
        
        self.f = h5py.File(self.tmp_path, 'w')
        
        # Meta schema setup
        self.meta = self.f.create_group('meta')
        self.meta.attrs['mode'] = self.mode
        
        # Track for finalization
        self._rois = set()
        self._chunk_ids = []
        self._source_files = []
        
        self.roi_group = self.f.create_group('roi')
        self._is_aborted = False
        self._dataset_create_kwargs = {
            'dtype': np.float64,
            'compression': 'gzip',
        }
        if self.mode == 'phasic':
            # Keep gzip for compatibility while reducing compression CPU on the
            # measured phasic cache-write hotspot.
            self._dataset_create_kwargs['compression_opts'] = 1

    def add_chunk(self, chunk, chunk_id: int, source_file: str):
        """
        Extract chunk arrays into the HDF5 structure.
        """
        if self._is_aborted:
            return

        self._chunk_ids.append(chunk_id)
        self._source_files.append(source_file)
        
        chunk_group_name = f"chunk_{chunk_id}"
        
        for r_idx, roi in enumerate(chunk.channel_names):
            self._rois.add(roi)
            
            # Ensure ROI group exists
            if roi not in self.roi_group:
                self.roi_group.create_group(roi)
                
            grp = self.roi_group[roi].create_group(chunk_group_name)

            grp.attrs['fs_hz'] = float(chunk.fs_hz)
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                if "acquisition_mode" in chunk.metadata:
                    grp.attrs["acquisition_mode"] = str(chunk.metadata.get("acquisition_mode"))
                for key in (
                    "window_index",
                    "window_start_sec",
                    "window_end_sec",
                    "window_duration_sec",
                    "original_file_duration_sec",
                    "continuous_window_sec",
                    "continuous_step_sec",
                ):
                    if key in chunk.metadata:
                        try:
                            grp.attrs[key] = float(chunk.metadata.get(key))
                        except Exception:
                            pass
                if "is_partial_final_window" in chunk.metadata:
                    grp.attrs["is_partial_final_window"] = bool(
                        chunk.metadata.get("is_partial_final_window")
                    )
            grp.attrs["source_file"] = str(source_file)

            if self.mode == 'phasic' and self.config:
                grp.attrs['peak_threshold_method'] = str(self.config.peak_threshold_method)
                grp.attrs['peak_threshold_k'] = float(self.config.peak_threshold_k)
                grp.attrs['signal_excursion_polarity'] = str(
                    getattr(self.config, 'signal_excursion_polarity', 'positive')
                )
                grp.attrs['bleach_correction_mode'] = str(
                    getattr(self.config, 'bleach_correction_mode', 'none')
                )
                grp.attrs['bleach_correction_target'] = "signal_and_isosbestic_independent"
                if (
                    hasattr(chunk, 'metadata')
                    and isinstance(chunk.metadata, dict)
                ):
                    roi_bleach = (
                        chunk.metadata.get('bleach_correction', {})
                        if isinstance(chunk.metadata.get('bleach_correction', {}), dict)
                        else {}
                    )
                    roi_bleach_meta = roi_bleach.get(str(roi), {}) if isinstance(roi_bleach, dict) else {}
                    grp.attrs['bleach_correction_applied'] = bool(
                        chunk.metadata.get('bleach_correction_applied', False)
                    )
                    if isinstance(roi_bleach_meta, dict):
                        sig_meta = roi_bleach_meta.get('signal', {}) if isinstance(roi_bleach_meta.get('signal', {}), dict) else {}
                        uv_meta = roi_bleach_meta.get('isosbestic', {}) if isinstance(roi_bleach_meta.get('isosbestic', {}), dict) else {}
                        def _set_float_attr_if_finite(attr_name: str, payload: dict, key: str) -> None:
                            try:
                                value = float(payload.get(key, np.nan))
                            except Exception:
                                return
                            if np.isfinite(value):
                                grp.attrs[attr_name] = value
                        def _write_bleach_fit_attrs(prefix: str, payload: dict) -> None:
                            grp.attrs[f'{prefix}_fit_succeeded'] = bool(payload.get('fit_succeeded', False))
                            fit_model = str(payload.get('fit_model', '')).strip().lower()
                            if fit_model:
                                grp.attrs[f'{prefix}_fit_model'] = fit_model
                            _set_float_attr_if_finite(f'{prefix}_offset', payload, 'offset')
                            _set_float_attr_if_finite(f'{prefix}_fit_rmse', payload, 'fit_rmse')
                            _set_float_attr_if_finite(f'{prefix}_tau_sec', payload, 'tau_sec')
                            _set_float_attr_if_finite(f'{prefix}_amplitude', payload, 'amplitude')
                            _set_float_attr_if_finite(f'{prefix}_tau_fast_sec', payload, 'tau_fast_sec')
                            _set_float_attr_if_finite(f'{prefix}_amplitude_fast', payload, 'amplitude_fast')
                            _set_float_attr_if_finite(f'{prefix}_tau_slow_sec', payload, 'tau_slow_sec')
                            _set_float_attr_if_finite(f'{prefix}_amplitude_slow', payload, 'amplitude_slow')
                            fail_reason = str(payload.get('fit_failure_reason', '')).strip()
                            if fail_reason:
                                grp.attrs[f'{prefix}_fit_failure_reason'] = fail_reason

                        _write_bleach_fit_attrs('bleach_signal', sig_meta)
                        _write_bleach_fit_attrs('bleach_iso', uv_meta)

                    fit_mode = str(chunk.metadata.get('dynamic_fit_mode_resolved', '')).strip()
                    if fit_mode:
                        grp.attrs['dynamic_fit_mode_resolved'] = fit_mode
                    engine = str(chunk.metadata.get('dynamic_fit_engine', '')).strip()
                    if engine:
                        grp.attrs['dynamic_fit_engine'] = engine

                    def _roi_fit_meta() -> dict:
                        if fit_mode == 'global_linear_regression':
                            return chunk.metadata.get('dynamic_fit_global_linear', {}).get(str(roi), {})
                        if fit_mode == 'robust_global_event_reject':
                            return chunk.metadata.get('dynamic_fit_event_reject', {}).get(str(roi), {})
                        if fit_mode == 'adaptive_event_gated_regression':
                            return chunk.metadata.get('dynamic_fit_adaptive_event_gated', {}).get(str(roi), {})
                        return chunk.metadata.get('dynamic_fit_rolling_local', {}).get(str(roi), {})

                    roi_fit_meta = _roi_fit_meta()
                    slope_summary = (
                        roi_fit_meta.get('slope_summary', {})
                        if isinstance(roi_fit_meta, dict)
                        else {}
                    )
                    if isinstance(slope_summary, dict) and slope_summary:
                        grp.attrs['dynamic_fit_slope_summary_available'] = True
                        warning_level = str(slope_summary.get('warning_level', 'none')).strip() or 'none'
                        grp.attrs['dynamic_fit_slope_warning_level'] = warning_level
                        for key in (
                            'slope_min',
                            'slope_max',
                            'slope_median',
                            'slope_mean',
                            'slope_negative_fraction',
                            'slope_nonfinite_fraction',
                            'n_slope_samples',
                            'n_negative_slope_samples',
                            'n_nonfinite_slope_samples',
                            'n_negative_slope_spans',
                            'longest_negative_slope_span_samples',
                            'longest_negative_slope_span_sec',
                        ):
                            value = slope_summary.get(key, None)
                            if value is None:
                                continue
                            try:
                                numeric = float(value)
                            except Exception:
                                continue
                            if np.isfinite(numeric):
                                grp.attrs[f'dynamic_fit_slope_{key}'] = numeric
                    constraint_summary = (
                        roi_fit_meta.get('slope_constraint_summary', {})
                        if isinstance(roi_fit_meta, dict)
                        else {}
                    )
                    if isinstance(constraint_summary, dict) and constraint_summary:
                        mode_value = str(
                            constraint_summary.get('slope_constraint_mode', 'unconstrained')
                        ).strip() or 'unconstrained'
                        grp.attrs['dynamic_fit_slope_constraint_mode'] = mode_value
                        grp.attrs['dynamic_fit_slope_constraint_applied'] = bool(
                            constraint_summary.get('slope_constraint_applied', False)
                        )
                        for attr_name, key in (
                            ('dynamic_fit_slope_min_allowed', 'slope_min_allowed'),
                            ('dynamic_fit_slope_clamped_fraction', 'slope_clamped_fraction'),
                            ('dynamic_fit_slope_n_clamped_slope_samples', 'n_clamped_slope_samples'),
                            ('dynamic_fit_slope_n_clamped_slope_spans', 'n_clamped_slope_spans'),
                            (
                                'dynamic_fit_slope_longest_clamped_slope_span_samples',
                                'longest_clamped_slope_span_samples',
                            ),
                            (
                                'dynamic_fit_slope_longest_clamped_slope_span_sec',
                                'longest_clamped_slope_span_sec',
                            ),
                            (
                                'dynamic_fit_slope_n_negative_slope_support_windows',
                                'n_negative_slope_support_windows',
                            ),
                            (
                                'dynamic_fit_slope_n_negative_slope_support_samples',
                                'n_negative_slope_support_samples',
                            ),
                            (
                                'dynamic_fit_slope_negative_slope_support_fraction',
                                'negative_slope_support_fraction',
                            ),
                            (
                                'dynamic_fit_slope_n_valid_nonnegative_support_windows',
                                'n_valid_nonnegative_support_windows',
                            ),
                            (
                                'dynamic_fit_slope_n_valid_nonnegative_support_samples',
                                'n_valid_nonnegative_support_samples',
                            ),
                            (
                                'dynamic_fit_slope_valid_nonnegative_support_fraction',
                                'valid_nonnegative_support_fraction',
                            ),
                            (
                                'dynamic_fit_slope_longest_negative_slope_span_sec',
                                'longest_negative_slope_span_sec',
                            ),
                        ):
                            value = constraint_summary.get(key, None)
                            if value is None:
                                continue
                            try:
                                numeric = float(value)
                            except Exception:
                                continue
                            if np.isfinite(numeric):
                                grp.attrs[attr_name] = numeric

                        grp.attrs['dynamic_fit_slope_nonnegative_support_insufficient'] = bool(
                            constraint_summary.get('nonnegative_support_insufficient', False)
                        )
                        grp.attrs['dynamic_fit_slope_fallback_used'] = bool(
                            constraint_summary.get('fallback_used', False)
                        )
                        fallback_reason = constraint_summary.get('fallback_reason')
                        if fallback_reason is not None:
                            grp.attrs['dynamic_fit_slope_fallback_reason'] = str(fallback_reason)

                        grp.attrs['dynamic_fit_slope_intercept_recomputed'] = bool(
                            constraint_summary.get('intercept_recomputed', False)
                        )
                        grp.attrs['dynamic_fit_slope_global_negative_slope_constrained'] = bool(
                            constraint_summary.get('global_negative_slope_constrained', False)
                        )

                        def _write_nested_slope_attrs(prefix: str, payload: dict) -> None:
                            if not isinstance(payload, dict):
                                return
                            for attr_suffix, key in (
                                ('slope_min', 'slope_min'),
                                ('slope_max', 'slope_max'),
                                ('slope_negative_fraction', 'slope_negative_fraction'),
                            ):
                                value = payload.get(key, None)
                                if value is None:
                                    continue
                                try:
                                    numeric = float(value)
                                except Exception:
                                    continue
                                if np.isfinite(numeric):
                                    grp.attrs[f'{prefix}_{attr_suffix}'] = numeric

                        _write_nested_slope_attrs(
                            'dynamic_fit_slope_unconstrained',
                            constraint_summary.get('unconstrained_slope_summary', {}),
                        )
                        _write_nested_slope_attrs(
                            'dynamic_fit_slope_constrained',
                            constraint_summary.get('constrained_slope_summary', {}),
                        )

                    validity_by_roi = chunk.metadata.get('dynamic_fit_validity_qc', {})
                    validity_meta = (
                        validity_by_roi.get(str(roi), {})
                        if isinstance(validity_by_roi, dict)
                        else {}
                    )
                    if isinstance(validity_meta, dict) and validity_meta:
                        grp.attrs['dynamic_fit_qc_available'] = True
                        flags = validity_meta.get('dynamic_fit_qc_flags', [])
                        if isinstance(flags, (list, tuple)):
                            grp.attrs['dynamic_fit_qc_flags'] = ';'.join(str(x) for x in flags)
                        elif flags is not None:
                            grp.attrs['dynamic_fit_qc_flags'] = str(flags)
                        for key in (
                            'fitted_ref_to_signal_range_ratio',
                            'fitted_ref_to_iso_range_ratio',
                            'signal_iso_corr',
                            'signal_fitted_ref_corr',
                            'iso_fitted_ref_corr',
                            'slope_fraction_negative',
                            'unconstrained_slope_fraction_negative',
                            'final_slope_fraction_negative',
                            'fitted_ref_total_variance',
                            'fitted_ref_baseline_scale_fraction',
                            'fitted_ref_response_scale_fraction',
                        ):
                            value = validity_meta.get(key, None)
                            if value is None:
                                continue
                            try:
                                numeric = float(value)
                            except Exception:
                                continue
                            if np.isfinite(numeric):
                                grp.attrs[f'dynamic_fit_qc_{key}'] = numeric
                        for key in (
                            'dynamic_fit_reference_flat_or_uninformative',
                            'dynamic_fit_reference_low_range',
                            'dynamic_fit_negative_or_mixed_coupling',
                            'dynamic_fit_response_scale_rich',
                            'dynamic_fit_needs_inspection',
                        ):
                            if key in validity_meta:
                                grp.attrs[f'dynamic_fit_qc_{key}'] = bool(validity_meta.get(key))
            
            # Required Time axis
            grp.create_dataset('time_sec', data=chunk.time_sec, **self._dataset_create_kwargs)
            
            # Common core traces
            grp.create_dataset('sig_raw', data=chunk.sig_raw[:, r_idx], **self._dataset_create_kwargs)

            grp.create_dataset('uv_raw', data=chunk.uv_raw[:, r_idx], **self._dataset_create_kwargs)
            
            # Modes
            if self.mode == 'tonic' and chunk.delta_f is not None:
                grp.create_dataset('deltaF', data=chunk.delta_f[:, r_idx], **self._dataset_create_kwargs)
            elif self.mode == 'phasic':
                if chunk.dff is not None:
                    grp.create_dataset('dff', data=chunk.dff[:, r_idx], **self._dataset_create_kwargs)

                if chunk.uv_fit is not None:
                    grp.create_dataset('fit_ref', data=chunk.uv_fit[:, r_idx], **self._dataset_create_kwargs)

                if chunk.delta_f is not None:
                    grp.create_dataset('delta_f', data=chunk.delta_f[:, r_idx], **self._dataset_create_kwargs)

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finalize()
        else:
            self.abort()

    def finalize(self):
        """
        Finish writing metadata, close the file, and atomic rename.
        """
        if self._is_aborted:
            return
            
        try:
            # Sort ROIs for determinism
            sorted_rois = sorted(list(self._rois))
            
            # Use specific type for variable-length strings
            dt_str = h5py.string_dtype(encoding='utf-8')
            
            
            self.meta.create_dataset('rois', data=np.array(sorted_rois, dtype=object), dtype=dt_str)
            self.meta.create_dataset('chunk_ids', data=np.array(self._chunk_ids, dtype=int))
            self.meta.create_dataset('source_files', data=np.array(self._source_files, dtype=object), dtype=dt_str)
            self.meta.create_dataset('schema_version', data=np.array([1], dtype=int))
            self.meta.create_dataset('n_chunks', data=np.array([len(self._chunk_ids)], dtype=int))
            
            self.f.close()
            
            # Atomic replace
            os.replace(self.tmp_path, self.output_path)
            print(f"HDF5 Cache Producer: Saved {self.output_path}")
        except Exception as e:
            self.abort() # Close and delete
            raise RuntimeError(f"Failed to finalize HDF5 cache: {e}") from e

    def abort(self):
        """
        Close the file and remove the temporary file.
        """
        if self._is_aborted:
            return
        self._is_aborted = True
        try:
            self.f.close()
        except Exception:
            pass
        if os.path.exists(self.tmp_path):
            os.remove(self.tmp_path)
