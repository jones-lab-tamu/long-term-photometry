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
