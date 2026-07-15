"""Exact frozen-input adapter for the Guided intermittent NPM worker.

Only the discovery and interpretation boundary is specialized.  Exact bytes
are passed into the existing NPM resampling implementation, and resulting
chunks continue through the existing Pipeline correction, feature, output, and
provenance paths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

from photometry_pipeline.config import Config
from photometry_pipeline.guided_npm_production_execution_request import (
    GuidedNpmProductionSessionRuntimeProjection,
    verify_guided_npm_production_execution_request,
)
from photometry_pipeline.guided_npm_worker_prelaunch_claim import stored_paths_equal
from photometry_pipeline.guided_npm_worker_request import (
    GuidedNpmWorkerRequest,
    verify_guided_npm_worker_request,
)
from photometry_pipeline.guided_normalized_recording import (
    compute_npm_parser_contract_digest,
)
from photometry_pipeline.guided_production_mapping import (
    guided_production_strategy_map_to_correction_specs,
)
from photometry_pipeline.io.adapters import load_npm_authorized_bytes
from photometry_pipeline.io.npm_contract import NpmParserContract


_CORRECTION_CONFIG_NAMES = {
    "slope_constraint": "dynamic_fit_slope_constraint",
    "min_slope": "dynamic_fit_min_slope",
}
_IRRELEVANT_CONFIG_FIELDS = frozenset(
    {
        "rwd_excluded_source_files",
        "rwd_contract_validation",
        "authorized_missing_sessions",
        "rwd_time_col",
        "uv_suffix",
        "sig_suffix",
        "custom_tabular_time_col",
        "custom_tabular_uv_suffix",
        "custom_tabular_sig_suffix",
        "continuous_window_sec",
        "continuous_step_sec",
        "allow_partial_final_window",
        "preview_first_n",
    }
)


@dataclass(frozen=True)
class GuidedNpmAuthorizedConfigFieldAudit:
    field_name: str
    classification: str
    authority: str


@dataclass(frozen=True)
class GuidedNpmAuthorizedInput:
    worker_request_identity: str
    execution_request_identity: str
    source_root_path: str
    run_directory_path: str
    source_path_style: str
    ordered_sessions: tuple[GuidedNpmProductionSessionRuntimeProjection, ...]
    ordered_session_paths: tuple[str, ...]
    ordered_session_identities: tuple[str, ...]
    chronological_positions: tuple[int, ...]
    authoritative_source_start_times: tuple[str, ...]
    actual_elapsed_sec_by_chunk: tuple[float, ...]
    nominal_expected_elapsed_sec_by_chunk: tuple[float, ...]
    physical_roi_ids: tuple[str, ...]
    canonical_roi_ids: tuple[str, ...]
    selected_canonical_roi_ids: tuple[str, ...]
    physical_to_canonical_roi_map: tuple[tuple[str, str], ...]
    parser_contract: NpmParserContract
    parser_contract_identity: str
    parser_contract_content_json: str
    reference_led_value: int | float | str
    signal_led_value: int | float | str
    target_fs_hz: float
    configured_session_duration_sec: float
    sessions_per_hour: int
    chronology_policy: str
    gap_policy: str
    overlap_policy: str
    output_time_basis: str


@dataclass(frozen=True)
class GuidedNpmAuthorizedRuntime:
    authorized_input: GuidedNpmAuthorizedInput
    config: Config
    mode: str
    per_roi_correction: dict[str, Any]
    per_roi_feature_config: dict[str, Config]
    per_roi_feature_provenance: dict[str, dict[str, Any]]
    config_field_audit: tuple[GuidedNpmAuthorizedConfigFieldAudit, ...]
    correction_authority_identity: str
    feature_authority_identity: str
    canonical_guided_npm_authorized_runtime_identity: str


@dataclass(frozen=True)
class GuidedNpmAuthorizedChunkLoadResult:
    chunk: Any
    consumed_source_record: Any


def verify_guided_npm_authorized_input(
    authorized: GuidedNpmAuthorizedInput,
) -> None:
    """Require one exact frozen session/path/chunk chronology projection."""
    if type(authorized) is not GuidedNpmAuthorizedInput:
        raise TypeError("authorized_npm_input_invalid")
    sessions = authorized.ordered_sessions
    count = len(sessions)
    positions = tuple(item.chronological_position for item in sessions)
    paths = tuple(item.source_path for item in sessions)
    starts = tuple(item.authoritative_source_start_time for item in sessions)
    actual = tuple(item.actual_elapsed_sec for item in sessions)
    nominal = tuple(item.nominal_expected_elapsed_sec for item in sessions)
    if (
        count == 0
        or authorized.ordered_session_paths != paths
        or authorized.chronological_positions != positions
        or positions != tuple(range(count))
        or authorized.authoritative_source_start_times != starts
        or authorized.actual_elapsed_sec_by_chunk != actual
        or authorized.nominal_expected_elapsed_sec_by_chunk != nominal
        or any(
            len(values) != count
            for values in (
                authorized.ordered_session_paths,
                authorized.chronological_positions,
                authorized.authoritative_source_start_times,
                authorized.actual_elapsed_sec_by_chunk,
                authorized.nominal_expected_elapsed_sec_by_chunk,
            )
        )
        or len(set(paths)) != count
        or len(set(positions)) != count
        or any(not isinstance(value, str) or not value for value in starts)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in actual + nominal
        )
        or any(
            float(later) <= float(earlier)
            for earlier, later in zip(actual, actual[1:])
        )
        or any(
            float(later) <= float(earlier)
            for earlier, later in zip(nominal, nominal[1:])
        )
    ):
        raise ValueError("authorized_npm_chronology_invalid")


def _typed_values(values) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        if item.field_name in result:
            raise ValueError("authorized_npm_duplicate_typed_field")
        result[item.field_name] = item.value
    return result


def _production_correction_specs(entries):
    """Convert the B2-C6A signal-only token to Pipeline's exact backend token."""
    backend_entries = tuple(
        replace(
            item,
            strategy_family="signal_only_f0",
            selected_strategy="signal_only_f0",
        )
        if (
            item.strategy_family == "signal_only"
            and item.selected_strategy == "signal_only"
            and item.dynamic_fit_mode is None
        )
        else item
        for item in entries
    )
    return guided_production_strategy_map_to_correction_specs(backend_entries)


def _parser_contract(worker: GuidedNpmWorkerRequest) -> NpmParserContract:
    execution = worker.execution_request
    parser = execution.parser_runtime_projection
    timing = execution.timing_runtime_projection
    try:
        content = json.loads(parser.parser_policy_content_json)
        sampling = content["sampling"]
        contract = NpmParserContract(
            npm_time_axis=parser.time_axis_mode,
            npm_system_ts_col=sampling["system_timestamp_column"],
            npm_computer_ts_col=sampling["computer_timestamp_column"],
            npm_led_col=parser.led_state_column,
            npm_region_prefix=parser.roi_prefix,
            npm_region_suffix=parser.roi_suffix,
            target_fs_hz=timing.target_fs_hz,
            session_duration_sec=timing.configured_session_duration_sec,
            allow_partial_final_chunk=sampling["allow_partial_final_chunk"],
            adapter_value_nan_policy=parser.roi_value_nan_policy,
            timestamp_cv_max=sampling["timestamp_cv_max"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("authorized_npm_parser_contract_invalid") from exc
    canonical = json.dumps(
        contract.content(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    if (
        canonical != parser.parser_policy_content_json
        or contract.digest != parser.parser_policy_identity
        or compute_npm_parser_contract_digest(content) != parser.parser_policy_identity
        or contract.timestamp_column_candidates != parser.ordered_timestamp_candidates
        or contract.support_policy != parser.support_policy
    ):
        raise ValueError("authorized_npm_parser_contract_mismatch")
    return contract


def _config_and_audit(worker: GuidedNpmWorkerRequest, contract: NpmParserContract):
    execution = worker.execution_request
    timing = execution.timing_runtime_projection
    correction = execution.correction_runtime_projection
    feature = execution.feature_runtime_projection
    defaults = Config()
    values = asdict(defaults)
    frozen: dict[str, Any] = {
        "acquisition_mode": "intermittent",
        "target_fs_hz": timing.target_fs_hz,
        "chunk_duration_sec": timing.configured_session_duration_sec,
        "allow_partial_final_chunk": contract.allow_partial_final_chunk,
        "adapter_value_nan_policy": contract.adapter_value_nan_policy,
        "npm_time_axis": contract.npm_time_axis,
        "npm_system_ts_col": contract.npm_system_ts_col,
        "npm_computer_ts_col": contract.npm_computer_ts_col,
        "npm_led_col": contract.npm_led_col,
        "npm_region_prefix": contract.npm_region_prefix,
        "npm_region_suffix": contract.npm_region_suffix,
        "timestamp_cv_max": contract.timestamp_cv_max,
    }
    for name, value in _typed_values(correction.correction_parameter_values).items():
        frozen[_CORRECTION_CONFIG_NAMES.get(name, name)] = value
    frozen.update(_typed_values(feature.effective_values))
    unknown = set(frozen) - set(values)
    if unknown:
        raise ValueError("authorized_npm_config_field_unmapped")
    values.update(frozen)
    # Construct every Config field explicitly.  Non-frozen numerical defaults
    # are fixed production invariants bound by the exact application build.
    config = Config(**values)
    audit = tuple(
        GuidedNpmAuthorizedConfigFieldAudit(
            item.name,
            (
                "frozen_worker_authority"
                if item.name in frozen
                else "irrelevant_to_guided_npm"
                if item.name in _IRRELEVANT_CONFIG_FIELDS
                else "fixed_build_invariant"
            ),
            (
                "worker_request_projection"
                if item.name in frozen
                else "not_consumed_by_authorized_path"
                if item.name in _IRRELEVANT_CONFIG_FIELDS
                else "Config default bound by application_build_identity"
            ),
        )
        for item in fields(Config)
    )
    return config, audit


def build_guided_npm_authorized_runtime(
    worker: GuidedNpmWorkerRequest,
) -> GuidedNpmAuthorizedRuntime:
    """Purely bind every child runtime input to the verified worker request."""
    verify_guided_npm_worker_request(worker)
    execution = worker.execution_request
    verify_guided_npm_production_execution_request(execution)
    source = execution.source_runtime_projection
    parser = execution.parser_runtime_projection
    roi = execution.roi_runtime_projection
    timing = execution.timing_runtime_projection
    correction = execution.correction_runtime_projection
    feature = execution.feature_runtime_projection
    sessions = source.ordered_sessions
    paths = tuple(item.source_path for item in sessions)
    positions = tuple(item.chronological_position for item in sessions)
    starts = tuple(item.authoritative_source_start_time for item in sessions)
    actual_elapsed = tuple(item.actual_elapsed_sec for item in sessions)
    nominal_elapsed = tuple(item.nominal_expected_elapsed_sec for item in sessions)
    identities = tuple(item.canonical_session_runtime_identity for item in sessions)
    if (
        paths != source.ordered_source_paths
        or tuple(item.canonical_relative_path for item in sessions)
        != source.ordered_source_relative_paths
        or positions != tuple(range(len(sessions)))
        or len(paths) != len(set(paths))
        or len(positions) != len(set(positions))
        or len(identities) != len(set(identities))
        or any(not path.lower().endswith(".csv") for path in paths)
    ):
        raise ValueError("authorized_npm_session_authority_invalid")
    mapping = tuple(
        (item.physical_source_column, item.canonical_roi_id)
        for item in roi.physical_to_canonical_roi_mapping
    )
    if (
        tuple(item[0] for item in mapping) != roi.complete_physical_source_columns
        or tuple(item[1] for item in mapping) != roi.complete_canonical_roi_ids
        or tuple(item.roi_id for item in correction.per_roi_correction_strategy_map)
        != roi.selected_canonical_roi_ids
        or tuple(item.roi_id for item in feature.per_roi_feature_event_map)
        != roi.selected_canonical_roi_ids
    ):
        raise ValueError("authorized_npm_roi_scope_invalid")
    contract = _parser_contract(worker)
    config, audit = _config_and_audit(worker, contract)
    per_roi_correction = _production_correction_specs(
        correction.per_roi_correction_strategy_map
    )
    per_roi_feature_config: dict[str, Config] = {}
    per_roi_feature_provenance: dict[str, dict[str, Any]] = {}
    for item in feature.per_roi_feature_event_map:
        effective = _typed_values(item.effective_config_fields)
        overrides = _typed_values(item.override_config_fields)
        if set(effective) - {field.name for field in fields(Config)}:
            raise ValueError("authorized_npm_feature_field_unmapped")
        per_roi_feature_config[item.roi_id] = replace(config, **effective)
        per_roi_feature_provenance[item.roi_id] = {
            "source": item.source,
            "feature_event_profile_id": item.feature_event_profile_id,
            "override_config_fields": overrides,
            "effective_config_fields": effective,
        }
    selected = roi.selected_canonical_roi_ids
    if (
        tuple(per_roi_correction) != selected
        or tuple(per_roi_feature_config) != selected
        or tuple(per_roi_feature_provenance) != selected
    ):
        raise ValueError("authorized_npm_per_roi_authority_invalid")
    authorized = GuidedNpmAuthorizedInput(
        worker.canonical_worker_request_identity,
        execution.canonical_execution_request_identity,
        source.source_root_canonical,
        execution.output_runtime_projection.run_directory_path,
        source.source_path_style,
        sessions,
        paths,
        identities,
        positions,
        starts,
        actual_elapsed,
        nominal_elapsed,
        roi.complete_physical_source_columns,
        roi.complete_canonical_roi_ids,
        selected,
        mapping,
        contract,
        parser.parser_policy_identity,
        parser.parser_policy_content_json,
        parser.reference_led_value,
        parser.signal_led_value,
        timing.target_fs_hz,
        timing.configured_session_duration_sec,
        timing.sessions_per_hour,
        timing.chronology_policy,
        timing.gap_policy,
        timing.overlap_policy,
        timing.output_time_basis,
    )
    verify_guided_npm_authorized_input(authorized)
    provisional = GuidedNpmAuthorizedRuntime(
        authorized,
        config,
        execution.execution_mode,
        per_roi_correction,
        per_roi_feature_config,
        per_roi_feature_provenance,
        audit,
        correction.canonical_correction_runtime_projection_identity,
        feature.canonical_feature_runtime_projection_identity,
        "0" * 64,
    )
    from photometry_pipeline.guided_npm_worker_acknowledgement import (
        compute_guided_npm_authorized_runtime_identity,
    )

    return replace(
        provisional,
        canonical_guided_npm_authorized_runtime_identity=(
            compute_guided_npm_authorized_runtime_identity(provisional, worker)
        ),
    )


def verify_guided_npm_authorized_runtime(
    runtime: GuidedNpmAuthorizedRuntime,
    worker: GuidedNpmWorkerRequest,
) -> None:
    """Purely require exact equality with the runtime derived from authority."""
    if type(runtime) is not GuidedNpmAuthorizedRuntime:
        raise TypeError("authorized_npm_runtime_invalid")
    expected = build_guided_npm_authorized_runtime(worker)
    if runtime != expected:
        raise ValueError("authorized_npm_runtime_authority_mismatch")


def _stable_authorized_bytes(
    session: GuidedNpmProductionSessionRuntimeProjection,
) -> tuple[bytes, os.stat_result]:
    path = Path(session.source_path)
    try:
        before = path.stat(follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError("authorized_npm_source_not_regular")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            content = handle.read()
            after_open = os.fstat(handle.fileno())
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError("authorized_npm_source_unavailable") from exc
    facts = lambda value: (
        value.st_size,
        value.st_mtime_ns,
        value.st_dev,
        value.st_ino,
        value.st_mode,
    )
    if not (facts(before) == facts(opened) == facts(after_open) == facts(after)):
        raise ValueError("authorized_npm_source_changed_during_read")
    if (
        len(content) != session.source_size_bytes
        or hashlib.sha256(content).hexdigest() != session.source_sha256
    ):
        raise ValueError("authorized_npm_source_identity_mismatch")
    return content, after


def load_guided_npm_authorized_chunk_with_record(
    authorized: GuidedNpmAuthorizedInput,
    path: str,
    config: Config,
    chunk_id: int,
):
    """Open and consume one exact authorized session with no rediscovery."""
    verify_guided_npm_authorized_input(authorized)
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id < 0:
        raise ValueError("authorized_npm_chunk_id_invalid")
    if chunk_id >= len(authorized.ordered_sessions):
        raise ValueError("authorized_npm_chunk_id_invalid")
    session = authorized.ordered_sessions[chunk_id]
    if not stored_paths_equal(path, session.source_path, authorized.source_path_style):
        raise ValueError("authorized_npm_source_path_mismatch")
    content, consumed_file_facts = _stable_authorized_bytes(session)
    chunk = load_npm_authorized_bytes(
        session.source_path,
        content,
        config,
        chunk_id,
        contract=authorized.parser_contract,
        resolved_timestamp_column=session.resolved_timestamp_column,
        reference_led_value=authorized.reference_led_value,
        signal_led_value=authorized.signal_led_value,
        physical_to_canonical_roi_mapping=authorized.physical_to_canonical_roi_map,
        authorized_timing_geometry={
            "overlap_origin_absolute": session.overlap_origin_absolute,
            "inner_start_rel_overlap": session.support_start_offset_sec,
            "inner_end_rel_overlap": session.support_end_offset_sec,
            "resolved_support_start_absolute": session.support_start_absolute,
            "resolved_support_end_absolute": session.support_end_absolute,
            "observed_duration_sec": session.observed_support_duration_sec,
        },
    )
    metadata = chunk.metadata
    metadata.update(
        {
            "guided_npm_chronological_position": session.chronological_position,
            "guided_npm_actual_elapsed_sec": session.actual_elapsed_sec,
            "guided_npm_nominal_expected_elapsed_sec": session.nominal_expected_elapsed_sec,
            "guided_npm_authoritative_source_start_time": session.authoritative_source_start_time,
            "guided_npm_cross_session_time_authority": "frozen_worker_projection",
            "guided_npm_within_session_output_time_basis": session.output_time_basis,
        }
    )
    observed_timing = (
        metadata["npm_overlap_origin_absolute"],
        metadata["npm_resolved_support_start_offset_sec"],
        metadata["npm_resolved_support_end_offset_sec"],
        metadata["npm_resolved_support_start_absolute"],
        metadata["npm_resolved_support_end_absolute"],
        metadata["npm_observed_duration_sec"],
        metadata["npm_output_time_basis"],
        metadata["npm_support_policy"],
    )
    expected_timing = (
        session.overlap_origin_absolute,
        session.support_start_offset_sec,
        session.support_end_offset_sec,
        session.support_start_absolute,
        session.support_end_absolute,
        session.observed_support_duration_sec,
        session.output_time_basis,
        session.support_policy,
    )
    if (
        tuple(chunk.channel_names) != authorized.canonical_roi_ids
        or metadata["npm_resolved_timestamp_column"]
        != session.resolved_timestamp_column
        or metadata["npm_timestamp_unit"] != session.timestamp_unit
        or observed_timing != expected_timing
        or chunk.fs_hz != authorized.target_fs_hz
        or metadata["guided_npm_chronological_position"] != chunk_id
        or metadata["guided_npm_actual_elapsed_sec"]
        != authorized.actual_elapsed_sec_by_chunk[chunk_id]
        or metadata["guided_npm_nominal_expected_elapsed_sec"]
        != authorized.nominal_expected_elapsed_sec_by_chunk[chunk_id]
        or metadata["guided_npm_authoritative_source_start_time"]
        != authorized.authoritative_source_start_times[chunk_id]
    ):
        raise ValueError("authorized_npm_loaded_session_mismatch")
    from photometry_pipeline.guided_npm_worker_acknowledgement import (
        GuidedNpmConsumedSourceRecord,
        compute_guided_npm_consumed_source_record_identity,
    )

    record = GuidedNpmConsumedSourceRecord(
        session.chronological_position,
        session.source_path,
        session.canonical_relative_path,
        len(content),
        hashlib.sha256(content).hexdigest(),
        consumed_file_facts.st_mtime_ns,
        consumed_file_facts.st_dev,
        consumed_file_facts.st_ino,
        consumed_file_facts.st_mode,
        session.canonical_session_runtime_identity,
        metadata["npm_resolved_timestamp_column"],
        metadata["npm_timestamp_unit"],
        session.resolved_led_column,
        authorized.reference_led_value,
        authorized.signal_led_value,
        metadata["npm_support_policy"],
        metadata["npm_output_time_basis"],
        session.physical_roi_inventory,
        tuple(metadata["npm_observed_physical_roi_ids"]),
        tuple(chunk.channel_names),
        authorized.physical_to_canonical_roi_map,
        session.actual_elapsed_sec,
        float(session.actual_elapsed_sec + chunk.time_sec[0]),
        float(session.actual_elapsed_sec + chunk.time_sec[-1]),
        "0" * 64,
    )
    record = replace(
        record,
        canonical_consumed_source_record_identity=(
            compute_guided_npm_consumed_source_record_identity(record)
        ),
    )
    return GuidedNpmAuthorizedChunkLoadResult(chunk, record)


def load_guided_npm_authorized_chunk(
    authorized: GuidedNpmAuthorizedInput,
    path: str,
    config: Config,
    chunk_id: int,
):
    """Compatibility wrapper returning only the reconciled authorized chunk."""
    return load_guided_npm_authorized_chunk_with_record(
        authorized, path, config, chunk_id
    ).chunk
