from __future__ import annotations

import builtins
import copy
from dataclasses import FrozenInstanceError, replace
import inspect
import math
import ntpath
import os
from pathlib import Path
import subprocess

import pytest

import analyze_photometry
import photometry_pipeline.guided_npm_production_execution_request as request_module
from photometry_pipeline.guided_npm_production_execution_request import (
    GUIDED_NPM_PRODUCTION_COMPONENT_IDENTIFIERS,
    GUIDED_NPM_PRODUCTION_DEFERRED_ACTIONS,
    GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_CONTRACT_VERSION,
    GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_SCHEMA_NAME,
    GuidedNpmProductionExecutionRequest,
    GuidedNpmProductionExecutionRequestFailure,
    build_guided_npm_production_execution_request,
    compute_guided_npm_production_correction_runtime_projection_identity,
    compute_guided_npm_production_adapter_runtime_projection_identity,
    compute_guided_npm_production_execution_request_identity,
    compute_guided_npm_production_feature_runtime_projection_identity,
    compute_guided_npm_production_parser_runtime_projection_identity,
    compute_guided_npm_production_roi_runtime_projection_identity,
    compute_guided_npm_production_session_runtime_identity,
    compute_guided_npm_production_session_sequence_identity,
    compute_guided_npm_production_source_content_identity,
    compute_guided_npm_production_source_membership_identity,
    compute_guided_npm_production_source_runtime_projection_identity,
    compute_guided_npm_production_timing_runtime_projection_identity,
    deserialize_guided_npm_production_execution_request,
    serialize_guided_npm_production_execution_request,
    verify_guided_npm_production_execution_request,
)
from photometry_pipeline.guided_npm_startup_claim import (
    GuidedNpmStartupClaimReceipt,
    claim_guided_npm_startup_artifact,
    claim_guided_npm_startup_artifact_path,
    compute_guided_npm_startup_claim_receipt_identity,
)
from photometry_pipeline.guided_npm_startup_payload import (
    GuidedNpmStartupRoiMappingEntry,
    compute_guided_npm_startup_correction_projection_identity,
    compute_guided_npm_startup_execution_projection_identity,
    compute_guided_npm_startup_feature_projection_identity,
    compute_guided_npm_startup_payload_identity,
    compute_guided_npm_startup_roi_projection_identity,
    compute_guided_npm_startup_session_identity,
    compute_guided_npm_startup_session_sequence_identity,
    compute_guided_npm_startup_source_projection_identity,
)
from photometry_pipeline.guided_npm_startup_persistence import (
    GuidedNpmStartupPersistenceReceipt,
    persist_guided_npm_startup_payload,
)
from tests.test_guided_npm_startup_payload import _payload


def _claimed(tmp_path: Path, payload=None, *, direct=False):
    payload = payload or _payload(tmp_path)
    persisted = persist_guided_npm_startup_payload(payload)
    assert isinstance(persisted, GuidedNpmStartupPersistenceReceipt)
    if direct:
        claim = claim_guided_npm_startup_artifact_path(
            persisted.startup_artifact_path,
            current_application_build_identity=payload.application_build_identity,
        )
    else:
        claim = claim_guided_npm_startup_artifact(
            persisted,
            current_application_build_identity=payload.application_build_identity,
        )
    assert isinstance(claim, GuidedNpmStartupClaimReceipt)
    return payload, claim


def _request(tmp_path: Path, payload=None, *, direct=False):
    payload, claim = _claimed(tmp_path, payload, direct=direct)
    result = build_guided_npm_production_execution_request(claim, payload)
    assert isinstance(result, GuidedNpmProductionExecutionRequest)
    return payload, claim, result


def _issue(result, category):
    assert isinstance(result, GuidedNpmProductionExecutionRequestFailure)
    assert result.blocking_issues[0].category == category


def _mode_payload(payload, mode):
    feature = replace(
        payload.feature_projection,
        execution_mode=mode,
        inactive_for_execution=(mode == "tonic"),
        canonical_feature_projection_identity="0" * 64,
    )
    feature = replace(
        feature,
        canonical_feature_projection_identity=compute_guided_npm_startup_feature_projection_identity(feature),
    )
    execution = replace(
        payload.execution_projection,
        execution_mode=mode,
        feature_projection_identity=feature.canonical_feature_projection_identity,
        canonical_execution_projection_identity="0" * 64,
    )
    execution = replace(
        execution,
        canonical_execution_projection_identity=compute_guided_npm_startup_execution_projection_identity(execution),
    )
    changed = replace(
        payload,
        execution_mode=mode,
        feature_projection=feature,
        execution_projection=execution,
        canonical_startup_payload_identity="0" * 64,
    )
    return replace(
        changed,
        canonical_startup_payload_identity=compute_guided_npm_startup_payload_identity(changed),
    )


def _two_roi_payload(payload, *, first_strategy="dynamic_fit", second_strategy="signal_only"):
    mapping = payload.roi_projection.physical_to_canonical_roi_mapping
    second_mapping = GuidedNpmStartupRoiMappingEntry("Region4G", "Region4")
    sessions = []
    for original in payload.source_projection.ordered_sessions:
        session = replace(
            original,
            physical_roi_inventory=original.physical_roi_inventory + ("Region4G",),
            physical_to_canonical_roi_mapping=original.physical_to_canonical_roi_mapping + (second_mapping,),
            canonical_startup_session_identity="0" * 64,
        )
        sessions.append(replace(session, canonical_startup_session_identity=compute_guided_npm_startup_session_identity(session)))
    sessions = tuple(sessions)
    source = replace(
        payload.source_projection,
        ordered_sessions=sessions,
        ordered_session_sequence_identity=compute_guided_npm_startup_session_sequence_identity(sessions),
        canonical_source_projection_identity="0" * 64,
    )
    source = replace(source, canonical_source_projection_identity=compute_guided_npm_startup_source_projection_identity(source))
    roi = replace(
        payload.roi_projection,
        complete_canonical_roi_ids=("Region0", "Region4"),
        selected_canonical_roi_ids=("Region0", "Region4"),
        excluded_canonical_roi_ids=(),
        complete_physical_source_columns=(mapping[0].physical_source_column, "Region4G"),
        physical_to_canonical_roi_mapping=(mapping[0], second_mapping),
        selected_physical_source_columns=(mapping[0].physical_source_column, "Region4G"),
        selected_physical_to_canonical_roi_mapping=(mapping[0], second_mapping),
        canonical_roi_projection_identity="0" * 64,
    )
    roi = replace(roi, canonical_roi_projection_identity=compute_guided_npm_startup_roi_projection_identity(roi))
    base_strategy = payload.correction_projection.per_roi_correction_strategy_map[0]
    def strategy(roi_id, family):
        return replace(
            base_strategy,
            roi_id=roi_id,
            strategy_family=family,
            dynamic_fit_mode=(base_strategy.dynamic_fit_mode if family == "dynamic_fit" else None),
            selected_strategy=(base_strategy.selected_strategy if family == "dynamic_fit" else "signal_only"),
            evidence_source_type=(base_strategy.evidence_source_type if family == "dynamic_fit" else "explicit_signal_only_selection"),
        )
    correction = replace(
        payload.correction_projection,
        selected_canonical_roi_ids=("Region0", "Region4"),
        per_roi_correction_strategy_map=(strategy("Region0", first_strategy), strategy("Region4", second_strategy)),
        canonical_correction_projection_identity="0" * 64,
    )
    correction = replace(correction, canonical_correction_projection_identity=compute_guided_npm_startup_correction_projection_identity(correction))
    base_feature = payload.feature_projection.per_roi_feature_event_map[0]
    feature = replace(
        payload.feature_projection,
        selected_canonical_roi_ids=("Region0", "Region4"),
        per_roi_feature_event_map=(base_feature, replace(base_feature, roi_id="Region4", source="override")),
        canonical_feature_projection_identity="0" * 64,
    )
    feature = replace(feature, canonical_feature_projection_identity=compute_guided_npm_startup_feature_projection_identity(feature))
    execution = replace(
        payload.execution_projection,
        source_projection_identity=source.canonical_source_projection_identity,
        session_sequence_identity=source.ordered_session_sequence_identity,
        roi_projection_identity=roi.canonical_roi_projection_identity,
        correction_projection_identity=correction.canonical_correction_projection_identity,
        feature_projection_identity=feature.canonical_feature_projection_identity,
        canonical_execution_projection_identity="0" * 64,
    )
    execution = replace(execution, canonical_execution_projection_identity=compute_guided_npm_startup_execution_projection_identity(execution))
    changed = replace(
        payload,
        source_projection=source,
        roi_projection=roi,
        correction_projection=correction,
        feature_projection=feature,
        execution_projection=execution,
        canonical_startup_payload_identity="0" * 64,
    )
    return replace(changed, canonical_startup_payload_identity=compute_guided_npm_startup_payload_identity(changed))


def _reidentify_request(request, **changes):
    changed = replace(request, **changes, canonical_execution_request_identity="0" * 64)
    return replace(
        changed,
        canonical_execution_request_identity=compute_guided_npm_production_execution_request_identity(changed),
    )


def _reidentify_claim(claim, **changes):
    changed = replace(
        claim,
        **changes,
        canonical_claim_receipt_identity="0" * 64,
    )
    return replace(
        changed,
        canonical_claim_receipt_identity=compute_guided_npm_startup_claim_receipt_identity(changed),
    )


def _reidentify_adapter(request, **changes):
    adapter = replace(
        request.adapter_runtime_projection,
        **changes,
        canonical_adapter_runtime_projection_identity="0" * 64,
    )
    return replace(
        adapter,
        canonical_adapter_runtime_projection_identity=compute_guided_npm_production_adapter_runtime_projection_identity(adapter),
    )


def _request_with_projection(request, field, projection, adapter_reference_field=None):
    adapter = request.adapter_runtime_projection
    if adapter_reference_field is not None:
        adapter = _reidentify_adapter(
            request,
            **{adapter_reference_field: getattr(projection, next(name for name in projection.__dict__ if name.startswith("canonical_") and name.endswith("_identity")))},
        )
    return _reidentify_request(request, **{field: projection, "adapter_runtime_projection": adapter})


def _request_with_source(request, source):
    source = replace(
        source,
        runtime_source_membership_identity=compute_guided_npm_production_source_membership_identity(source.ordered_sessions),
        runtime_source_content_identity=compute_guided_npm_production_source_content_identity(source.ordered_sessions),
        runtime_session_sequence_identity=compute_guided_npm_production_session_sequence_identity(source.ordered_sessions),
        canonical_source_runtime_projection_identity="0" * 64,
    )
    source = replace(source, canonical_source_runtime_projection_identity=compute_guided_npm_production_source_runtime_projection_identity(source))
    adapter = _reidentify_adapter(request, source_runtime_projection_identity=source.canonical_source_runtime_projection_identity)
    return _reidentify_request(request, source_runtime_projection=source, adapter_runtime_projection=adapter)


def test_constructs_complete_immutable_nonlaunched_request(tmp_path):
    payload, claim, request = _request(tmp_path)
    assert request.request_schema_name == GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_SCHEMA_NAME
    assert request.request_contract_version == GUIDED_NPM_PRODUCTION_EXECUTION_REQUEST_CONTRACT_VERSION
    assert request.source_claim_receipt_identity == claim.canonical_claim_receipt_identity
    assert request.source_startup_payload_identity == payload.canonical_startup_payload_identity
    assert request.claim_status == "claimed_for_npm_startup"
    assert request.request_status == "constructed_for_production_adapter"
    assert request.launch_status == "not_launched"
    assert request.execution_status == "not_started"
    assert request.completion_status == "not_available"
    assert request.runnable is False
    verify_guided_npm_production_execution_request(request)
    with pytest.raises(FrozenInstanceError):
        request.runnable = True


def test_direct_artifact_claim_is_accepted_without_persistence_receipt_input(tmp_path):
    payload, claim, request = _request(tmp_path, direct=True)
    assert claim.source_persistence_receipt_identity is None
    assert request.source_startup_payload_identity == payload.canonical_startup_payload_identity


def test_source_and_session_projection_preserves_exact_authority_order(tmp_path):
    payload, _, request = _request(tmp_path)
    startup = payload.source_projection.ordered_sessions
    runtime = request.source_runtime_projection
    assert runtime.ordered_source_paths == tuple(x.authorized_absolute_source_reference for x in startup)
    assert runtime.ordered_source_relative_paths == tuple(x.canonical_relative_path for x in startup)
    assert runtime.ordered_source_digests == tuple(x.sha256_content_digest for x in startup)
    assert tuple(x.chronological_position for x in runtime.ordered_sessions) == tuple(range(len(startup)))
    assert tuple(x.resolved_timestamp_column for x in runtime.ordered_sessions) == tuple(x.resolved_timestamp_column for x in startup)
    assert tuple(x.support_start_offset_sec for x in runtime.ordered_sessions) == tuple(x.resolved_support_start_offset_sec for x in startup)


def test_all_runtime_projections_preserve_payload_authority(tmp_path):
    payload, _, request = _request(tmp_path)
    assert request.parser_runtime_projection.parser_policy_content_json == payload.recording_policy.parser_policy_content_json
    assert request.parser_runtime_projection.ordered_timestamp_candidates == payload.recording_policy.ordered_timestamp_candidates
    assert request.roi_runtime_projection.selected_physical_to_canonical_roi_mapping == payload.roi_projection.selected_physical_to_canonical_roi_mapping
    assert request.correction_runtime_projection.per_roi_correction_strategy_map == payload.correction_projection.per_roi_correction_strategy_map
    assert request.feature_runtime_projection.per_roi_feature_event_map == payload.feature_projection.per_roi_feature_event_map
    assert request.timing_runtime_projection.output_time_basis == payload.recording_policy.output_time_basis
    assert request.output_runtime_projection.startup_artifact_path == request.startup_artifact_path
    assert request.output_runtime_projection.run_directory_path == os.path.dirname(request.startup_artifact_path)


@pytest.mark.parametrize(
    ("first_strategy", "second_strategy"),
    (("signal_only", "signal_only"), ("dynamic_fit", "signal_only")),
)
def test_signal_only_and_mixed_per_roi_runtime_maps_are_preserved(
    tmp_path, first_strategy, second_strategy
):
    payload = _two_roi_payload(
        _payload(tmp_path),
        first_strategy=first_strategy,
        second_strategy=second_strategy,
    )
    _, _, request = _request(tmp_path, payload)
    assert request.roi_runtime_projection.selected_canonical_roi_ids == ("Region0", "Region4")
    assert tuple(x.strategy_family for x in request.correction_runtime_projection.per_roi_correction_strategy_map) == (first_strategy, second_strategy)
    assert tuple(x.roi_id for x in request.feature_runtime_projection.per_roi_feature_event_map) == ("Region0", "Region4")
    assert request.feature_runtime_projection.per_roi_feature_event_map[1].source == "override"


def test_production_component_mapping_is_exact_and_deferred(tmp_path):
    _, _, request = _request(tmp_path)
    adapter = request.adapter_runtime_projection
    assert adapter.production_component_identifiers == GUIDED_NPM_PRODUCTION_COMPONENT_IDENTIFIERS
    assert adapter.deferred_runtime_actions == GUIDED_NPM_PRODUCTION_DEFERRED_ACTIONS
    assert "photometry_pipeline.io.adapters._load_npm" in adapter.production_component_identifiers
    assert "photometry_pipeline.core.feature_extraction.extract_features" in adapter.production_component_identifiers


def test_tonic_projection_preserves_feature_metadata_but_marks_it_inactive(tmp_path):
    payload = _mode_payload(_payload(tmp_path), "tonic")
    _, _, request = _request(tmp_path, payload)
    assert request.execution_mode == "tonic"
    assert request.feature_runtime_projection.inactive_for_execution is True
    assert request.feature_runtime_projection.per_roi_feature_event_map == payload.feature_projection.per_roi_feature_event_map


def test_combined_mode_refuses_until_atomic_production_mapping_exists(tmp_path):
    payload = _mode_payload(_payload(tmp_path), "both")
    payload, claim = _claimed(tmp_path, payload)
    _issue(
        build_guided_npm_production_execution_request(claim, payload),
        "production_component_mapping_unavailable",
    )


def test_claim_from_payload_a_cannot_bind_payload_b(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    payload_a, claim = _claimed(tmp_path / "a")
    payload_b = _payload(tmp_path / "b")
    _issue(
        build_guided_npm_production_execution_request(claim, payload_b),
        "claim_payload_identity_mismatch",
    )


@pytest.mark.parametrize(
    ("field", "value", "category"),
    (
        ("guided_plan_identity", "1" * 64, "claim_plan_identity_mismatch"),
        ("validation_revision", 999, "claim_validation_revision_mismatch"),
    ),
)
def test_claim_payload_binding_mismatch_refuses(tmp_path, field, value, category):
    payload, claim = _claimed(tmp_path)
    changed = _reidentify_claim(claim, **{field: value})
    _issue(build_guided_npm_production_execution_request(changed, payload), category)


def test_deterministic_serialization_round_trip_and_identity(tmp_path):
    payload, claim, first = _request(tmp_path)
    second = build_guided_npm_production_execution_request(claim, payload)
    assert second == first
    serialized = serialize_guided_npm_production_execution_request(first)
    reordered = dict(reversed(tuple(serialized.items())))
    assert deserialize_guided_npm_production_execution_request(reordered) == first
    assert serialize_guided_npm_production_execution_request(first) == serialized


@pytest.mark.parametrize(
    "mutation",
    ("schema", "missing", "nonfinite", "nested_identity", "top_identity", "launch", "execution", "runnable"),
)
def test_serialization_tampering_refuses(tmp_path, mutation):
    _, _, request = _request(tmp_path)
    serialized = copy.deepcopy(serialize_guided_npm_production_execution_request(request))
    if mutation == "schema":
        serialized["request_schema_version"] = "v999"
    elif mutation == "missing":
        del serialized["guided_plan_identity"]
    elif mutation == "nonfinite":
        serialized["timing_runtime_projection"]["target_fs_hz"] = math.nan
    elif mutation == "nested_identity":
        serialized["source_runtime_projection"]["canonical_source_runtime_projection_identity"] = "1" * 64
    elif mutation == "top_identity":
        serialized["canonical_execution_request_identity"] = "1" * 64
    elif mutation == "launch":
        serialized["launch_status"] = "started"
    elif mutation == "execution":
        serialized["execution_status"] = "running"
    else:
        serialized["runnable"] = True
    with pytest.raises(ValueError, match="execution_request_serialization_invalid"):
        deserialize_guided_npm_production_execution_request(serialized)


def test_source_order_tampering_refuses_after_outer_reidentification(tmp_path):
    _, _, request = _request(tmp_path)
    source = request.source_runtime_projection
    changed_source = replace(
        source,
        ordered_source_paths=tuple(reversed(source.ordered_source_paths)),
        canonical_source_runtime_projection_identity="0" * 64,
    )
    changed_source = replace(
        changed_source,
        canonical_source_runtime_projection_identity=compute_guided_npm_production_source_runtime_projection_identity(changed_source),
    )
    changed = _reidentify_request(request, source_runtime_projection=changed_source)
    with pytest.raises(ValueError, match="source_order_mismatch"):
        verify_guided_npm_production_execution_request(changed)


def test_correction_and_feature_coverage_tampering_refuses(tmp_path):
    _, _, request = _request(tmp_path)
    correction = request.correction_runtime_projection
    changed_correction = replace(
        correction,
        per_roi_correction_strategy_map=correction.per_roi_correction_strategy_map[:-1],
        canonical_correction_runtime_projection_identity="0" * 64,
    )
    changed_correction = replace(changed_correction, canonical_correction_runtime_projection_identity=compute_guided_npm_production_correction_runtime_projection_identity(changed_correction))
    with pytest.raises(ValueError, match="correction_roi_coverage_mismatch"):
        verify_guided_npm_production_execution_request(_reidentify_request(request, correction_runtime_projection=changed_correction))
    feature = request.feature_runtime_projection
    changed_feature = replace(
        feature,
        per_roi_feature_event_map=feature.per_roi_feature_event_map[:-1],
        canonical_feature_runtime_projection_identity="0" * 64,
    )
    changed_feature = replace(changed_feature, canonical_feature_runtime_projection_identity=compute_guided_npm_production_feature_runtime_projection_identity(changed_feature))
    with pytest.raises(ValueError, match="feature_roi_coverage_mismatch"):
        verify_guided_npm_production_execution_request(_reidentify_request(request, feature_runtime_projection=changed_feature))


def test_builder_serializer_and_verifier_are_pure_and_do_not_launch(tmp_path, monkeypatch):
    payload, claim = _claimed(tmp_path)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("I/O, source reinterpretation, or execution attempted")
    for name in ("open",):
        monkeypatch.setattr(builtins, name, forbidden)
    for name in ("read_bytes", "write_bytes", "exists", "is_file", "is_dir", "stat", "resolve"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    result = build_guided_npm_production_execution_request(claim, payload)
    assert isinstance(result, GuidedNpmProductionExecutionRequest)
    serialized = serialize_guided_npm_production_execution_request(result)
    assert deserialize_guided_npm_production_execution_request(serialized) == result
    verify_guided_npm_production_execution_request(result)


def test_module_is_npm_specific_and_wrapper_remains_claim_only():
    source = inspect.getsource(request_module).lower()
    assert "guided_candidate_manifest" not in source
    assert "subprocess" not in source
    assert "multiprocessing" not in source
    assert "build_guided_npm_startup_payload(" not in source
    wrapper_source = inspect.getsource(analyze_photometry)
    assert "guided_npm_production_execution_request" not in wrapper_source
    assert "claimed_execution_not_implemented" in wrapper_source


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reference_led_value", 999),
        ("signal_led_value", 999),
        ("roi_prefix", "Other"),
        ("roi_suffix", "X"),
        ("timestamp_finite_policy", "other"),
        ("roi_value_nan_policy", "mask"),
        ("timestamp_unit", "milliseconds"),
        ("ordered_timestamp_candidates", ("OtherTimestamp",)),
        ("led_state_column", "OtherLed"),
    ),
)
def test_outer_reidentified_parser_field_conflict_refuses(tmp_path, field, value):
    _, _, request = _request(tmp_path)
    parser = replace(
        request.parser_runtime_projection,
        **{field: value},
        canonical_parser_runtime_projection_identity="0" * 64,
    )
    parser = replace(parser, canonical_parser_runtime_projection_identity=compute_guided_npm_production_parser_runtime_projection_identity(parser))
    adapter = _reidentify_adapter(request, parser_runtime_projection_identity=parser.canonical_parser_runtime_projection_identity)
    changed = _reidentify_request(request, parser_runtime_projection=parser, adapter_runtime_projection=adapter)
    with pytest.raises(ValueError, match="parser_runtime_projection_invalid"):
        verify_guided_npm_production_execution_request(changed)


@pytest.mark.parametrize(
    "identity_field",
    ("runtime_source_membership_identity", "runtime_source_content_identity", "runtime_session_sequence_identity"),
)
def test_arbitrary_runtime_source_identity_refuses_after_outer_reidentification(tmp_path, identity_field):
    _, _, request = _request(tmp_path)
    source = replace(
        request.source_runtime_projection,
        **{identity_field: "1" * 64},
        canonical_source_runtime_projection_identity="0" * 64,
    )
    source = replace(source, canonical_source_runtime_projection_identity=compute_guided_npm_production_source_runtime_projection_identity(source))
    adapter = _reidentify_adapter(request, source_runtime_projection_identity=source.canonical_source_runtime_projection_identity)
    changed = _reidentify_request(request, source_runtime_projection=source, adapter_runtime_projection=adapter)
    with pytest.raises(ValueError, match=identity_field.replace("runtime_", "runtime_")[:-9]):
        verify_guided_npm_production_execution_request(changed)


def test_upstream_source_identities_and_session_reference_are_provenance_only(tmp_path):
    _, _, request = _request(tmp_path)
    source = replace(
        request.source_runtime_projection,
        source_startup_projection_identity="1" * 64,
        source_startup_snapshot_identity="2" * 64,
    )
    session = replace(
        source.ordered_sessions[0],
        source_startup_session_identity_reference="3" * 64,
        canonical_session_runtime_identity="0" * 64,
    )
    session = replace(session, canonical_session_runtime_identity=compute_guided_npm_production_session_runtime_identity(session))
    source = replace(source, ordered_sessions=(session,) + source.ordered_sessions[1:])
    changed = _request_with_source(request, source)
    verify_guided_npm_production_execution_request(changed)


def test_session_roi_inventory_and_mapping_must_match_global_authority(tmp_path):
    _, _, request = _request(tmp_path)
    for changes in (
        {"physical_roi_inventory": ("Other",)},
        {"physical_to_canonical_roi_mapping": (replace(request.source_runtime_projection.ordered_sessions[0].physical_to_canonical_roi_mapping[0], canonical_roi_id="Other"),)},
    ):
        session = replace(
            request.source_runtime_projection.ordered_sessions[0],
            **changes,
            canonical_session_runtime_identity="0" * 64,
        )
        session = replace(session, canonical_session_runtime_identity=compute_guided_npm_production_session_runtime_identity(session))
        source = replace(request.source_runtime_projection, ordered_sessions=(session,) + request.source_runtime_projection.ordered_sessions[1:])
        with pytest.raises(ValueError, match="session_global_roi_mapping_mismatch"):
            verify_guided_npm_production_execution_request(_request_with_source(request, source))


@pytest.mark.parametrize("mutation", ("overlap", "gap", "duplicate_roi", "duplicate_physical", "selected_mapping", "selected_physical"))
def test_outer_reidentified_roi_partition_and_mapping_attacks_refuse(tmp_path, mutation):
    payload = _two_roi_payload(_payload(tmp_path))
    _, _, request = _request(tmp_path, payload)
    roi = request.roi_runtime_projection
    changes = {}
    if mutation == "overlap": changes["excluded_canonical_roi_ids"] = (roi.selected_canonical_roi_ids[-1],)
    elif mutation == "gap": changes["excluded_canonical_roi_ids"] = (); changes["selected_canonical_roi_ids"] = roi.selected_canonical_roi_ids[:-1]
    elif mutation == "duplicate_roi": changes["complete_canonical_roi_ids"] = (roi.complete_canonical_roi_ids[0],) * 2
    elif mutation == "duplicate_physical": changes["complete_physical_source_columns"] = (roi.complete_physical_source_columns[0],) * 2
    elif mutation == "selected_mapping": changes["selected_physical_to_canonical_roi_mapping"] = tuple(reversed(roi.selected_physical_to_canonical_roi_mapping))
    else: changes["selected_physical_source_columns"] = tuple(reversed(roi.selected_physical_source_columns))
    changed_roi = replace(roi, **changes, canonical_roi_runtime_projection_identity="0" * 64)
    changed_roi = replace(changed_roi, canonical_roi_runtime_projection_identity=compute_guided_npm_production_roi_runtime_projection_identity(changed_roi))
    adapter = _reidentify_adapter(request, roi_runtime_projection_identity=changed_roi.canonical_roi_runtime_projection_identity)
    changed = _reidentify_request(request, roi_runtime_projection=changed_roi, adapter_runtime_projection=adapter)
    with pytest.raises(ValueError, match="roi|selected"):
        verify_guided_npm_production_execution_request(changed)


@pytest.mark.parametrize("kind", ("correction", "feature"))
def test_reordered_same_set_runtime_maps_refuse(tmp_path, kind):
    payload = _two_roi_payload(_payload(tmp_path))
    _, _, request = _request(tmp_path, payload)
    projection = getattr(request, f"{kind}_runtime_projection")
    map_field = "per_roi_correction_strategy_map" if kind == "correction" else "per_roi_feature_event_map"
    identity_field = f"canonical_{kind}_runtime_projection_identity"
    compute = compute_guided_npm_production_correction_runtime_projection_identity if kind == "correction" else compute_guided_npm_production_feature_runtime_projection_identity
    changed_projection = replace(projection, **{map_field: tuple(reversed(getattr(projection, map_field))), identity_field: "0" * 64})
    changed_projection = replace(changed_projection, **{identity_field: compute(changed_projection)})
    adapter = _reidentify_adapter(request, **{f"{kind}_runtime_projection_identity": getattr(changed_projection, identity_field)})
    changed = _reidentify_request(request, **{f"{kind}_runtime_projection": changed_projection, "adapter_runtime_projection": adapter})
    with pytest.raises(ValueError, match=f"{kind}_roi_coverage_mismatch"):
        verify_guided_npm_production_execution_request(changed)


@pytest.mark.parametrize("mutation", ("outside", "relative_traversal", "absolute_relative", "zero_size", "bad_digest"))
def test_outer_reidentified_source_path_and_primitive_attacks_refuse(tmp_path, mutation):
    _, _, request = _request(tmp_path)
    source = request.source_runtime_projection
    session = source.ordered_sessions[0]
    changes = {}
    if mutation == "outside": changes["source_path"] = ntpath.join(ntpath.dirname(source.source_root_canonical), "outside.csv")
    elif mutation == "relative_traversal": changes["canonical_relative_path"] = "..\\outside.csv"
    elif mutation == "absolute_relative": changes["canonical_relative_path"] = session.source_path
    elif mutation == "zero_size": changes["source_size_bytes"] = 0
    else: changes["source_sha256"] = "not-a-digest"
    if mutation in {"zero_size", "bad_digest"}:
        clone = object.__new__(type(session))
        for name, original in session.__dict__.items(): object.__setattr__(clone, name, changes.get(name, original))
        session = clone
    else:
        session = replace(session, **changes, canonical_session_runtime_identity="0" * 64)
        session = replace(session, canonical_session_runtime_identity=compute_guided_npm_production_session_runtime_identity(session))
    source = replace(source, ordered_sessions=(session,) + source.ordered_sessions[1:])
    with pytest.raises(ValueError, match="source|session"):
        verify_guided_npm_production_execution_request(_request_with_source(request, source))


def test_timing_vector_short_and_adapter_source_binding_refuse(tmp_path):
    _, _, request = _request(tmp_path)
    timing = replace(
        request.timing_runtime_projection,
        ordered_actual_elapsed_sec=request.timing_runtime_projection.ordered_actual_elapsed_sec[:-1],
        canonical_timing_runtime_projection_identity="0" * 64,
    )
    timing = replace(timing, canonical_timing_runtime_projection_identity=compute_guided_npm_production_timing_runtime_projection_identity(timing))
    adapter = _reidentify_adapter(request, timing_runtime_projection_identity=timing.canonical_timing_runtime_projection_identity)
    with pytest.raises(ValueError, match="timing_source_order_mismatch"):
        verify_guided_npm_production_execution_request(_reidentify_request(request, timing_runtime_projection=timing, adapter_runtime_projection=adapter))
    adapter = _reidentify_adapter(request, source_runtime_projection_identity="1" * 64)
    with pytest.raises(ValueError, match="adapter_runtime_projection_invalid"):
        verify_guided_npm_production_execution_request(_reidentify_request(request, adapter_runtime_projection=adapter))


@pytest.mark.parametrize("mutation", ("root_relative", "doubled_separator", "windows_under_posix", "posix_under_windows", "inspected_style"))
def test_pure_source_path_model_refuses_incompatible_or_noncanonical_paths(tmp_path, mutation):
    _, _, request = _request(tmp_path)
    source = request.source_runtime_projection
    if mutation == "root_relative":
        source = replace(source, source_root_canonical="relative/root")
    elif mutation == "inspected_style":
        source = replace(source, source_root_inspected="/posix/root")
    elif mutation in {"windows_under_posix", "posix_under_windows"}:
        if mutation == "windows_under_posix":
            source = replace(source, source_path_style="posix_absolute")
        else:
            sessions = []
            for index, original in enumerate(source.ordered_sessions):
                session = replace(
                    original,
                    source_path=f"/source/session{index}.csv",
                    canonical_relative_path=f"session{index}.csv",
                    canonical_session_runtime_identity="0" * 64,
                )
                sessions.append(replace(session, canonical_session_runtime_identity=compute_guided_npm_production_session_runtime_identity(session)))
            source = replace(
                source,
                source_root_canonical="/source",
                source_root_inspected="/source",
                source_path_style="windows_drive",
                ordered_sessions=tuple(sessions),
                ordered_source_paths=tuple(x.source_path for x in sessions),
                ordered_source_relative_paths=tuple(x.canonical_relative_path for x in sessions),
            )
    else:
        session = source.ordered_sessions[0]
        session = replace(
            session,
            source_path=session.source_path.replace("\\", "\\\\", 1),
            canonical_session_runtime_identity="0" * 64,
        )
        session = replace(session, canonical_session_runtime_identity=compute_guided_npm_production_session_runtime_identity(session))
        source = replace(source, ordered_sessions=(session,) + source.ordered_sessions[1:])
    with pytest.raises(ValueError, match="source"):
        verify_guided_npm_production_execution_request(_request_with_source(request, source))


@pytest.mark.parametrize(("kind", "mutation"), (("correction", "duplicate"), ("correction", "missing"), ("correction", "extra"), ("feature", "duplicate"), ("feature", "missing"), ("feature", "extra")))
def test_exact_ordered_coverage_refuses_duplicates_missing_and_extra(tmp_path, kind, mutation):
    payload = _two_roi_payload(_payload(tmp_path))
    _, _, request = _request(tmp_path, payload)
    projection = getattr(request, f"{kind}_runtime_projection")
    map_field = "per_roi_correction_strategy_map" if kind == "correction" else "per_roi_feature_event_map"
    values = getattr(projection, map_field)
    if mutation == "duplicate": changed_values = (values[0], values[0])
    elif mutation == "missing": changed_values = values[:-1]
    else: changed_values = values + (replace(values[0], roi_id="ExcludedRoi"),)
    identity_field = f"canonical_{kind}_runtime_projection_identity"
    compute = compute_guided_npm_production_correction_runtime_projection_identity if kind == "correction" else compute_guided_npm_production_feature_runtime_projection_identity
    changed_projection = replace(projection, **{map_field: changed_values, identity_field: "0" * 64})
    changed_projection = replace(changed_projection, **{identity_field: compute(changed_projection)})
    adapter = _reidentify_adapter(request, **{f"{kind}_runtime_projection_identity": getattr(changed_projection, identity_field)})
    changed = _reidentify_request(request, **{f"{kind}_runtime_projection": changed_projection, "adapter_runtime_projection": adapter})
    with pytest.raises(ValueError, match=f"{kind}_roi_coverage_mismatch"):
        verify_guided_npm_production_execution_request(changed)
