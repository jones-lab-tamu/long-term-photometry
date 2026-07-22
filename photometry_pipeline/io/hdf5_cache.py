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

from ..core.regression import classify_per_roi_dynamic_fit_mode_contract
from ..core.types import CORRECTION_STRATEGY_FAMILIES, RESOLVED_DYNAMIC_FIT_MODES

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

    def _validate_native_signal_only_evidence(
        self, chunk, chunk_id: int, source_file: str
    ) -> None:
        """Fail closed for native Signal-Only F0 cache evidence.

        The key is deliberately optional for legacy chunks. Once a producer
        writes the native consumed-strategy contract, however, a selected
        Signal-Only ROI must carry its complete aligned production evidence;
        the writer must never turn missing evidence into a silently incomplete
        cache.
        """
        if not hasattr(chunk, 'metadata'):
            return
        metadata = chunk.metadata
        if not isinstance(metadata, dict) or 'correction_strategy_consumed_by_roi' not in metadata:
            return

        consumed_by_roi = metadata['correction_strategy_consumed_by_roi']
        if not isinstance(consumed_by_roi, dict):
            raise ValueError(
                "Native correction_strategy_consumed_by_roi must be a dict "
                f"(chunk_id={chunk_id}, source_file={source_file!r})"
            )
        channel_names = [str(roi) for roi in getattr(chunk, 'channel_names', [])]
        if len(channel_names) != len(set(channel_names)):
            raise ValueError(
                "Native correction strategy contains duplicate ROI identities "
                f"(chunk_id={chunk_id}, source_file={source_file!r})"
            )
        sig_raw = np.asarray(getattr(chunk, 'sig_raw', np.empty((0, 0))))
        time_sec = np.asarray(getattr(chunk, 'time_sec', np.empty(0)))
        if sig_raw.ndim != 2 or sig_raw.shape[1] != len(channel_names):
            raise ValueError(
                "Native Signal-Only cache evidence has inconsistent signal/ROI shape "
                f"(sig_raw={sig_raw.shape}, rois={len(channel_names)}, "
                f"chunk_id={chunk_id}, source_file={source_file!r})"
            )
        if time_sec.ndim != 1 or time_sec.shape[0] != sig_raw.shape[0]:
            raise ValueError(
                "Native Signal-Only cache evidence has inconsistent time length "
                f"(time={time_sec.shape}, samples={sig_raw.shape[0]}, "
                f"chunk_id={chunk_id}, source_file={source_file!r})"
            )
        if not np.all(np.isfinite(time_sec)):
            raise ValueError(
                "Native Signal-Only cache evidence contains non-finite time values "
                f"(chunk_id={chunk_id}, source_file={source_file!r})"
            )
        baseline_by_roi = metadata.get('signal_only_f0_production_baseline')
        production_qc_by_roi = metadata.get('signal_only_f0_production_qc')
        expected_count = int(sig_raw.shape[0])
        coverage_fraction = float(
            getattr(self.config, 'signal_only_f0_min_coverage_fraction', 0.80)
        ) if self.config is not None else 0.80
        if not np.isfinite(coverage_fraction) or not 0.0 < coverage_fraction <= 1.0:
            raise ValueError(
                "Invalid native Signal-Only coverage policy "
                f"{coverage_fraction!r} (chunk_id={chunk_id}, source_file={source_file!r})"
            )
        min_required = max(10, int(np.ceil(coverage_fraction * expected_count)))

        expected_roi_keys = set(channel_names)
        actual_roi_keys = set(consumed_by_roi.keys())
        if actual_roi_keys != expected_roi_keys:
            missing = sorted(expected_roi_keys - actual_roi_keys)
            extra = sorted(actual_roi_keys - expected_roi_keys)
            raise ValueError(
                "Native correction_strategy_consumed_by_roi must cover exactly "
                f"chunk.channel_names; missing={missing}, extra={extra} "
                f"(chunk_id={chunk_id}, source_file={source_file!r})"
            )

        signal_only_rois = set()
        for roi in channel_names:
            consumed = consumed_by_roi[roi]
            if not isinstance(consumed, dict):
                raise ValueError(
                    f"Native correction strategy entry for ROI {roi!r} is malformed "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            if consumed.get('roi_id') != roi:
                raise ValueError(
                    f"Native correction strategy ROI identity mismatch for {roi!r} "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            if consumed.get('execution_status') != 'consumed':
                raise ValueError(
                    f"Native correction strategy for ROI {roi!r} was not consumed "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            family = consumed.get('strategy_family')
            if not isinstance(family, str) or family not in CORRECTION_STRATEGY_FAMILIES:
                raise ValueError(
                    f"Native correction strategy for ROI {roi!r} has unknown strategy_family "
                    f"{family!r} (chunk_id={chunk_id}, source_file={source_file!r})"
                )
            selected_strategy = consumed.get('selected_strategy')
            dynamic_fit_mode = consumed.get('dynamic_fit_mode')
            if family == 'dynamic_fit':
                if not isinstance(dynamic_fit_mode, str) or dynamic_fit_mode not in RESOLVED_DYNAMIC_FIT_MODES:
                    raise ValueError(
                        f"Native dynamic_fit strategy for ROI {roi!r} has unsupported dynamic_fit_mode "
                        f"{dynamic_fit_mode!r} (chunk_id={chunk_id}, source_file={source_file!r})"
                    )
                if selected_strategy != dynamic_fit_mode:
                    raise ValueError(
                        f"Native dynamic_fit strategy for ROI {roi!r} has selected_strategy="
                        f"{selected_strategy!r}, expected {dynamic_fit_mode!r} "
                        f"(chunk_id={chunk_id}, source_file={source_file!r})"
                    )
                continue

            signal_only_rois.add(roi)
            if selected_strategy != 'signal_only_f0':
                raise ValueError(
                    f"Native Signal-Only strategy for ROI {roi!r} has an invalid selected_strategy "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            if dynamic_fit_mode is not None:
                raise ValueError(
                    f"Native Signal-Only strategy for ROI {roi!r} must not carry dynamic_fit_mode "
                    f"{dynamic_fit_mode!r} (chunk_id={chunk_id}, source_file={source_file!r})"
                )

        for evidence_name, evidence_by_roi in (
            ('signal_only_f0_production_baseline', baseline_by_roi),
            ('signal_only_f0_production_qc', production_qc_by_roi),
        ):
            if evidence_by_roi is None:
                continue
            if not isinstance(evidence_by_roi, dict):
                raise ValueError(
                    f"Native {evidence_name} must be a dict when present "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            orphan_rois = sorted(set(evidence_by_roi.keys()) - signal_only_rois)
            if orphan_rois:
                raise ValueError(
                    f"Native {evidence_name} contains evidence for non-Signal-Only ROI(s) "
                    f"{orphan_rois} (chunk_id={chunk_id}, source_file={source_file!r})"
                )

        for roi in sorted(signal_only_rois):
            dff = getattr(chunk, 'dff', None)
            if dff is None:
                raise ValueError(
                    f"Native correction contract requires canonical dF/F for ROI {roi!r} "
                    f"before cache writing (chunk_id={chunk_id}, source_file={source_file!r})"
                )
            dff_arr = np.asarray(dff)
            if dff_arr.ndim != 2 or dff_arr.shape != sig_raw.shape:
                raise ValueError(
                    "Native correction contract has malformed canonical dF/F shape "
                    f"{dff_arr.shape}; expected {sig_raw.shape} "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            if not isinstance(production_qc_by_roi, dict):
                raise ValueError(
                    f"Native Signal-Only production QC is missing for ROI {roi!r} "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            production_qc = production_qc_by_roi.get(roi)
            if not isinstance(production_qc, dict):
                raise ValueError(
                    f"Native Signal-Only production QC entry is missing or malformed for ROI {roi!r} "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            production_available = production_qc.get(
                'signal_only_f0_production_available',
                production_qc.get('production_available', False),
            )
            if not isinstance(production_available, (bool, np.bool_)) or not bool(
                production_available
            ):
                raise ValueError(
                    f"Native Signal-Only production QC is not available for ROI {roi!r} "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            if not isinstance(baseline_by_roi, dict) or roi not in baseline_by_roi:
                raise ValueError(
                    f"Native Signal-Only production F0 baseline is missing for ROI {roi!r} "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            baseline = np.asarray(baseline_by_roi[roi])
            if baseline.ndim != 1 or baseline.shape[0] != expected_count:
                raise ValueError(
                    f"Native Signal-Only production F0 baseline for ROI {roi!r} has shape "
                    f"{baseline.shape}; expected ({expected_count},) "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            baseline_finite_count = int(np.sum(np.isfinite(baseline)))
            if baseline_finite_count < min_required:
                raise ValueError(
                    f"Native Signal-Only production F0 baseline for ROI {roi!r} has insufficient "
                    f"finite coverage ({baseline_finite_count}/{expected_count}; {min_required} required) "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )
            roi_index = channel_names.index(roi)
            dff_finite_count = int(np.sum(np.isfinite(dff_arr[:, roi_index])))
            if dff_finite_count < min_required:
                raise ValueError(
                    f"Native Signal-Only canonical dF/F for ROI {roi!r} has insufficient finite coverage "
                    f"({dff_finite_count}/{expected_count}; {min_required} required) "
                    f"(chunk_id={chunk_id}, source_file={source_file!r})"
                )

    def add_chunk(self, chunk, chunk_id: int, source_file: str):
        """
        Extract chunk arrays into the HDF5 structure.
        """
        if self._is_aborted:
            return

        self._validate_native_signal_only_evidence(chunk, chunk_id, source_file)

        # Fail closed before writing anything for this chunk: a present but
        # non-dict dynamic_fit_mode_resolved_by_roi is malformed current
        # metadata (fit_chunk_dynamic always writes a dict there), not
        # legacy/pre-grouping data, and must never fall back to the flat
        # dynamic_fit_mode_resolved value -- that would silently mislabel
        # every ROI in this chunk with a single borrowed mode.
        if (
            hasattr(chunk, 'metadata')
            and isinstance(chunk.metadata, dict)
            and classify_per_roi_dynamic_fit_mode_contract(chunk.metadata) == 'malformed'
        ):
            raise ValueError(
                "chunk.metadata['dynamic_fit_mode_resolved_by_roi'] is present but not a "
                f"dict (got {type(chunk.metadata['dynamic_fit_mode_resolved_by_roi'])!r}); "
                "refusing to write dynamic-fit cache attributes that could mislabel any ROI "
                f"(chunk_id={chunk_id}, source_file={source_file!r})"
            )

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
                if chunk.metadata.get("output_time_basis"):
                    grp.attrs["output_time_basis"] = str(
                        chunk.metadata["output_time_basis"]
                    )
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
                    "guided_npm_chronological_position",
                    "guided_npm_actual_elapsed_sec",
                    "guided_npm_nominal_expected_elapsed_sec",
                    "guided_npm_within_session_start_sec",
                    "guided_npm_within_session_end_sec",
                    "guided_npm_recording_time_start_sec",
                    "guided_npm_recording_time_end_sec",
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
                for key in (
                    "guided_npm_authoritative_source_start_time",
                    "guided_npm_cross_session_time_authority",
                    "guided_npm_within_session_output_time_basis",
                ):
                    if chunk.metadata.get(key):
                        grp.attrs[key] = str(chunk.metadata[key])
            grp.attrs["source_file"] = str(source_file)

            # B1: per-chunk, per-ROI consumed parser/channel evidence -- the
            # actual columns/rows this specific session resolved during
            # real execution, not a first-chunk-only snapshot and not a
            # value copied from the requested/configured contract. Written
            # every chunk, mirroring the existing correction-evidence attr
            # pattern above. Absent (non-RWD, or a format that doesn't
            # populate these chunk.metadata keys) is simply omitted, never
            # fabricated.
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                if "rwd_time_col_resolved" in chunk.metadata and chunk.metadata.get(
                    "rwd_time_col_resolved"
                ):
                    grp.attrs["resolved_time_column"] = str(
                        chunk.metadata["rwd_time_col_resolved"]
                    )
                if chunk.metadata.get("rwd_header_row_resolved") is not None:
                    try:
                        grp.attrs["resolved_header_row"] = int(
                            chunk.metadata["rwd_header_row_resolved"]
                        )
                    except (TypeError, ValueError):
                        pass
                if chunk.metadata.get("rwd_timestamp_unit"):
                    grp.attrs["resolved_timestamp_unit"] = str(
                        chunk.metadata["rwd_timestamp_unit"]
                    )
                rwd_fps = chunk.metadata.get("rwd_metadata_fps")
                if rwd_fps is not None:
                    try:
                        if np.isfinite(float(rwd_fps)):
                            grp.attrs["resolved_source_metadata_fps"] = float(rwd_fps)
                    except (TypeError, ValueError):
                        pass
                roi_map = chunk.metadata.get("roi_map")
                if isinstance(roi_map, dict):
                    roi_channels = roi_map.get(str(roi))
                    if isinstance(roi_channels, dict):
                        raw_sig = roi_channels.get("raw_sig")
                        raw_uv = roi_channels.get("raw_uv")
                        if raw_sig:
                            grp.attrs["resolved_signal_source"] = str(raw_sig)
                        if raw_uv:
                            grp.attrs["resolved_reference_source"] = str(raw_uv)

            if self.config and (
                self.mode == 'phasic'
                or (
                    hasattr(chunk, 'metadata')
                    and isinstance(chunk.metadata, dict)
                    and 'correction_strategy_consumed_by_roi' in chunk.metadata
                )
            ):
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
                    consumed_by_roi = chunk.metadata.get(
                        'correction_strategy_consumed_by_roi', {}
                    )
                    consumed_meta = (
                        consumed_by_roi.get(str(roi), {})
                        if isinstance(consumed_by_roi, dict)
                        else {}
                    )
                    if isinstance(consumed_meta, dict) and consumed_meta:
                        for attr_name, key in (
                            ('correction_strategy_family', 'strategy_family'),
                            ('correction_selected_strategy', 'selected_strategy'),
                            ('correction_dynamic_fit_mode', 'dynamic_fit_mode'),
                            ('correction_parameter_identity', 'parameter_identity'),
                            ('correction_evidence_identity', 'evidence_identity'),
                            ('correction_execution_status', 'execution_status'),
                            ('correction_production_baseline_dataset', 'production_baseline_dataset'),
                            ('correction_production_baseline_source', 'production_baseline_source'),
                            # Narrow addition for CR1-D1: the accepted
                            # continuous-RWD segment-correction kernel (C4b)
                            # carries a fallback chain (requested strategy may
                            # fall back to a simpler one) and a per-ROI QC
                            # blob that the legacy per-file correction path has
                            # no equivalent for. Both are optional (only
                            # present when consumed_meta supplies them), so
                            # existing intermittent producers are unaffected.
                            ('correction_applied_strategy', 'applied_strategy'),
                            ('correction_fallback_path', 'fallback_path'),
                            ('correction_qc_json', 'qc_json'),
                        ):
                            value = consumed_meta.get(key)
                            if value is not None and str(value):
                                grp.attrs[attr_name] = str(value)
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

                    # Per-ROI, not chunk-wide: a mixed-strategy chunk resolves a
                    # different dynamic_fit_mode per ROI (grouped dispatch).
                    # Three-state contract (see classify_per_roi_dynamic_fit_
                    # mode_contract): "authoritative" (key present, dict, even
                    # empty) -- a ROI absent from it did not undergo dynamic
                    # fitting and gets no mode/engine attribution at all, never
                    # the chunk-wide flat value; "absent" (key missing --
                    # legacy/pre-grouping) -- falls back to the flat value for
                    # every ROI. "malformed" already raised above, before this
                    # loop started.
                    contract_state = classify_per_roi_dynamic_fit_mode_contract(chunk.metadata)
                    if contract_state == 'authoritative':
                        mode_by_roi = chunk.metadata['dynamic_fit_mode_resolved_by_roi']
                        fit_mode = str(mode_by_roi.get(str(roi), '')).strip()
                    else:
                        fit_mode = str(chunk.metadata.get('dynamic_fit_mode_resolved', '')).strip()
                    has_per_roi_contract = contract_state == 'authoritative'
                    if fit_mode:
                        grp.attrs['dynamic_fit_mode_resolved'] = fit_mode
                        engine_by_mode = chunk.metadata.get('dynamic_fit_engine_by_mode', {})
                        engine = ''
                        if isinstance(engine_by_mode, dict) and fit_mode in engine_by_mode:
                            engine = str(engine_by_mode[fit_mode].get('engine', '') or '').strip()
                        if not engine and not has_per_roi_contract:
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
                        for flag_key in (
                            'dynamic_fit_qc_flags',
                            'dynamic_fit_qc_hard_flags',
                            'dynamic_fit_qc_soft_flags',
                        ):
                            flags = validity_meta.get(flag_key, [])
                            if isinstance(flags, (list, tuple)):
                                grp.attrs[flag_key] = ';'.join(str(x) for x in flags)
                            elif flags is not None:
                                grp.attrs[flag_key] = str(flags)
                        if validity_meta.get('dynamic_fit_qc_severity') is not None:
                            grp.attrs['dynamic_fit_qc_severity'] = str(
                                validity_meta.get('dynamic_fit_qc_severity')
                            )
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
                            'dynamic_fit_has_hard_flags',
                            'dynamic_fit_has_soft_flags',
                        ):
                            if key in validity_meta:
                                grp.attrs[f'dynamic_fit_qc_{key}'] = bool(validity_meta.get(key))

                    baseline_by_roi = chunk.metadata.get('baseline_reference_candidate_qc', {})
                    baseline_meta = (
                        baseline_by_roi.get(str(roi), {})
                        if isinstance(baseline_by_roi, dict)
                        else {}
                    )
                    baseline_trace_by_roi = chunk.metadata.get(
                        'baseline_reference_candidate_trace', {}
                    )
                    baseline_trace = (
                        baseline_trace_by_roi.get(str(roi))
                        if isinstance(baseline_trace_by_roi, dict)
                        else None
                    )
                    if isinstance(baseline_meta, dict) and baseline_meta:
                        grp.attrs['baseline_ref_candidate_available'] = bool(
                            baseline_meta.get('baseline_ref_candidate_available', False)
                        )
                        for attr_name in (
                            'baseline_ref_method',
                            'baseline_ref_smoothing_window_warning',
                            'baseline_ref_fit_stage',
                            'baseline_ref_status',
                        ):
                            value = baseline_meta.get(attr_name)
                            if value is not None:
                                grp.attrs[attr_name] = str(value)
                        for attr_name in (
                            'baseline_ref_actual_smoothing_window_sec',
                            'baseline_ref_requested_smoothing_window_sec',
                        ):
                            value = baseline_meta.get(attr_name)
                            if value is None:
                                continue
                            try:
                                numeric = float(value)
                            except Exception:
                                continue
                            if np.isfinite(numeric):
                                grp.attrs[attr_name] = numeric
                    if baseline_trace is not None:
                        trace_arr = np.asarray(baseline_trace, dtype=np.float64).reshape(-1)
                        if (
                            trace_arr.shape == np.asarray(chunk.sig_raw[:, r_idx]).reshape(-1).shape
                            and np.any(np.isfinite(trace_arr))
                        ):
                            grp.create_dataset(
                                'baseline_ref_candidate',
                                data=trace_arr,
                                **self._dataset_create_kwargs,
                            )
                    f0_by_roi = chunk.metadata.get('signal_only_f0_candidate_qc', {})
                    f0_meta = (
                        f0_by_roi.get(str(roi), {})
                        if isinstance(f0_by_roi, dict)
                        else {}
                    )
                    f0_trace_by_roi = chunk.metadata.get(
                        'signal_only_f0_candidate_trace', {}
                    )
                    f0_trace = (
                        f0_trace_by_roi.get(str(roi))
                        if isinstance(f0_trace_by_roi, dict)
                        else None
                    )
                    if isinstance(f0_meta, dict) and f0_meta:
                        grp.attrs['signal_only_f0_candidate_available'] = bool(
                            f0_meta.get('signal_only_f0_candidate_available', False)
                        )
                        for attr_name in (
                            'signal_only_f0_status',
                            'signal_only_f0_warning',
                            'signal_only_f0_method',
                            'signal_only_f0_candidate_viability',
                            'signal_only_f0_candidate_confidence',
                            'signal_only_f0_high_state_context_mode',
                            'signal_only_f0_anchor_status',
                            'signal_only_f0_edge_extrapolation_mode',
                        ):
                            value = f0_meta.get(attr_name)
                            if value is not None:
                                grp.attrs[attr_name] = str(value)
                        if 'signal_only_f0_state_aware_used' in f0_meta:
                            grp.attrs['signal_only_f0_state_aware_used'] = bool(
                                f0_meta.get('signal_only_f0_state_aware_used')
                            )
                        if 'signal_only_f0_high_state_context_applied' in f0_meta:
                            grp.attrs['signal_only_f0_high_state_context_applied'] = bool(
                                f0_meta.get('signal_only_f0_high_state_context_applied')
                            )
                        for attr_name in (
                            'signal_only_f0_high_state_context_cap',
                            'signal_only_f0_anchor_count',
                            'signal_only_f0_low_support_fraction',
                            'signal_only_f0_anchor_support_fraction',
                            'signal_only_f0_direct_support_fraction',
                            'signal_only_f0_interpolated_fraction',
                            'signal_only_f0_extrapolated_fraction',
                            'signal_only_f0_edge_extrapolation_fraction',
                            'signal_only_f0_max_anchor_gap_fraction_observed',
                            'signal_only_f0_max_anchor_gap_sec_observed',
                        ):
                            value = f0_meta.get(attr_name)
                            if value is not None:
                                try:
                                    numeric = float(value)
                                except Exception:
                                    numeric = np.nan
                                if np.isfinite(numeric):
                                    grp.attrs[attr_name] = numeric
                        flags = f0_meta.get('signal_only_f0_flags', [])
                        if isinstance(flags, (list, tuple)):
                            grp.attrs['signal_only_f0_flags'] = ';'.join(str(x) for x in flags)
                        elif flags is not None:
                            grp.attrs['signal_only_f0_flags'] = str(flags)
                    if f0_trace is not None:
                        trace_arr = np.asarray(f0_trace, dtype=np.float64).reshape(-1)
                        if (
                            trace_arr.shape == np.asarray(chunk.sig_raw[:, r_idx]).reshape(-1).shape
                            and np.any(np.isfinite(trace_arr))
                        ):
                            grp.create_dataset(
                                'signal_only_f0_candidate',
                                data=trace_arr,
                                **self._dataset_create_kwargs,
                            )
                    production_qc_by_roi = chunk.metadata.get(
                        'signal_only_f0_production_qc', {}
                    )
                    production_qc = (
                        production_qc_by_roi.get(str(roi), {})
                        if isinstance(production_qc_by_roi, dict)
                        else {}
                    )
                    production_baseline_by_roi = chunk.metadata.get(
                        'signal_only_f0_production_baseline', {}
                    )
                    production_baseline = (
                        production_baseline_by_roi.get(str(roi))
                        if isinstance(production_baseline_by_roi, dict)
                        else None
                    )
                    if isinstance(production_qc, dict) and production_qc:
                        grp.attrs['signal_only_f0_production_available'] = bool(
                            production_qc.get('signal_only_f0_production_available', False)
                        )
                        grp.attrs['signal_only_f0_production_baseline_source'] = str(
                            production_qc.get(
                                'signal_only_f0_production_baseline_source',
                                'signal_only_f0_candidate_uncapped',
                            )
                        )
                        grp.attrs['signal_only_f0_production_formula'] = str(
                            production_qc.get(
                                'signal_only_f0_production_formula',
                                '100 * (signal - f0) / f0',
                            )
                        )
                        for attr_name, key in (
                            (
                                'signal_only_f0_production_baseline_p05',
                                'signal_only_f0_production_baseline_p05',
                            ),
                            (
                                'signal_only_f0_production_baseline_p50',
                                'signal_only_f0_production_baseline_p50',
                            ),
                            (
                                'signal_only_f0_production_baseline_p95',
                                'signal_only_f0_production_baseline_p95',
                            ),
                            (
                                'signal_only_f0_production_valid_sample_count',
                                'signal_only_f0_production_valid_sample_count',
                            ),
                            (
                                'signal_only_f0_production_expected_sample_count',
                                'signal_only_f0_production_expected_sample_count',
                            ),
                            (
                                'signal_only_f0_production_baseline_finite_count',
                                'signal_only_f0_production_baseline_finite_count',
                            ),
                            (
                                'signal_only_f0_production_dff_finite_count',
                                'signal_only_f0_production_dff_finite_count',
                            ),
                            (
                                'signal_only_f0_production_min_required_samples',
                                'signal_only_f0_production_min_required_samples',
                            ),
                            (
                                'signal_only_f0_production_valid_fraction',
                                'signal_only_f0_production_valid_fraction',
                            ),
                        ):
                            value = production_qc.get(key)
                            if value is None:
                                continue
                            try:
                                numeric = float(value)
                            except Exception:
                                continue
                            if np.isfinite(numeric):
                                grp.attrs[attr_name] = numeric
                    if production_baseline is not None:
                        trace_arr = np.asarray(production_baseline, dtype=np.float64).reshape(-1)
                        if (
                            trace_arr.shape == np.asarray(chunk.sig_raw[:, r_idx]).reshape(-1).shape
                            and np.any(np.isfinite(trace_arr))
                        ):
                            grp.create_dataset(
                                'signal_only_f0_baseline',
                                data=trace_arr,
                                **self._dataset_create_kwargs,
                            )
            
            # Required Time axis
            grp.create_dataset('time_sec', data=chunk.time_sec, **self._dataset_create_kwargs)
            
            # Common core traces
            grp.create_dataset('sig_raw', data=chunk.sig_raw[:, r_idx], **self._dataset_create_kwargs)

            grp.create_dataset('uv_raw', data=chunk.uv_raw[:, r_idx], **self._dataset_create_kwargs)
            
            # Modes
            if self.mode == 'tonic' and chunk.delta_f is not None:
                grp.create_dataset('deltaF', data=chunk.delta_f[:, r_idx], **self._dataset_create_kwargs)
                if (
                    hasattr(chunk, 'metadata')
                    and isinstance(chunk.metadata, dict)
                    and 'correction_strategy_consumed_by_roi' in chunk.metadata
                ):
                    if chunk.dff is not None:
                        grp.create_dataset('dff', data=chunk.dff[:, r_idx], **self._dataset_create_kwargs)
                    if chunk.uv_fit is not None:
                        grp.create_dataset('fit_ref', data=chunk.uv_fit[:, r_idx], **self._dataset_create_kwargs)
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
