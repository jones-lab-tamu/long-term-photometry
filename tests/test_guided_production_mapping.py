from __future__ import annotations

import builtins
import ast
from dataclasses import FrozenInstanceError, fields, replace
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_backend_validation_request as contracts
import photometry_pipeline.guided_production_mapping as mapping
from tests.test_guided_backend_validator import (
    _local_preview_request,
    _request as _valid_request,
)


_A = "a" * 64
_B = "b" * 64


def _build(**changes):
    values = {
        "distribution_name": "photometry-pipeline",
        "distribution_version": "1.0.0",
        "source_revision_kind": "git",
        "source_revision": "abc123",
        "source_tree_state": "clean",
    }
    values.update(changes)
    return mapping.build_application_build_identity(**values)


def _contract():
    return mapping.build_guided_production_mapping_contract()


def _unchecked(instance, **changes):
    result = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            result,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return result


def _map(request=None, *, build=None, contract=None, identity=None):
    request = request or _valid_request()
    identity = identity or contracts.compute_guided_backend_validation_request_identity(
        request
    )
    return mapping.map_guided_validation_request_to_execution_intent(
        request,
        canonical_request_identity=identity,
        application_build_identity=build or _build(),
        mapping_contract=contract or _contract(),
    )


def _category(result):
    return result.blocking_issues[0].category


def test_build_identity_clean_dirty_and_packaged_forms():
    clean = _build()
    dirty = _build(
        source_tree_state="dirty_content_bound",
        source_tree_digest=_A,
    )
    packaged = _build(
        source_revision_kind="packaged_artifact",
        source_revision="release-1",
        source_tree_state="unavailable",
        build_artifact_digest=_B,
    )
    assert len({item.canonical_identity for item in (clean, dirty, packaged)}) == 3
    assert all(len(item.canonical_identity) == 64 for item in (clean, dirty, packaged))


@pytest.mark.parametrize(
    "changes",
    [
        {"source_revision": "different"},
        {"distribution_version": "2.0.0"},
        {"source_tree_state": "dirty_content_bound", "source_tree_digest": _A},
        {"source_tree_state": "dirty_content_bound", "source_tree_digest": _B},
    ],
)
def test_build_identity_is_sensitive(changes):
    assert _build(**changes).canonical_identity != _build().canonical_identity


def test_build_identity_rejects_unbound_or_malformed_source():
    with pytest.raises(ValueError):
        _build(
            source_revision_kind="unavailable",
            source_revision="unavailable",
            source_tree_state="unavailable",
        )
    with pytest.raises(ValueError):
        _build(
            source_tree_state="dirty_content_bound",
            source_tree_digest="bad",
        )


def test_valid_request_maps_to_frozen_identity_bearing_intent():
    result = _map()
    assert isinstance(result, mapping.GuidedProductionMappingSuccess)
    assert result.status == "mapped"
    assert result.intent.source_request_identity == result.source_request_identity
    assert result.intent.canonical_intent_identity == result.canonical_intent_identity
    assert result.intent.input_source.source_format == "rwd"
    assert result.intent.execution_profile.execution_mode == "phasic"
    assert result.intent.execution_profile.run_type == "full"
    assert result.intent.execution_profile.traces_only is False
    assert result.intent.output_policy.overwrite is False
    assert result.intent.input_source.candidate_files[0].canonical_relative_path
    with pytest.raises(FrozenInstanceError):
        result.intent.mapping_contract_version = "changed"  # type: ignore[misc]


def test_local_preview_request_maps_without_diagnostic_cache_fields():
    """4J16k9 regression: the supported local-preview path (no diagnostic
    cache) must map to a production execution intent identically to the
    cache-backed path, proving production mapping does not require
    cache_root_path/artifact_record_path or read diagnostic-cache facts."""
    result = _map(request=_local_preview_request())
    assert isinstance(result, mapping.GuidedProductionMappingSuccess)
    assert result.status == "mapped"
    assert result.intent.execution_profile.execution_mode == "phasic"
    assert result.intent.execution_profile.run_type == "full"


def test_intent_identity_recomputes_and_is_deterministic():
    first = _map()
    second = _map()
    assert first.canonical_intent_identity == second.canonical_intent_identity
    assert (
        mapping.compute_guided_production_execution_intent_identity(first.intent)
        == first.canonical_intent_identity
    )


def test_identity_changes_with_build_mapping_and_runner_contract():
    base = _map()
    changed_build = _map(build=_build(source_revision="different"))
    changed_mapping = _map(
        contract=mapping.build_guided_production_mapping_contract(
            mapping_contract_version="guided_production_mapping.v2"
        )
    )
    changed_runner = _map(
        contract=mapping.build_guided_production_mapping_contract(
            runner_contract_version="guided_runner_candidate_roi_bound.v2"
        )
    )
    assert len(
        {
            base.canonical_intent_identity,
            changed_build.canonical_intent_identity,
            changed_mapping.canonical_intent_identity,
            changed_runner.canonical_intent_identity,
        }
    ) == 4


@pytest.mark.parametrize(
    "section,change",
    [
        (
            "source",
            lambda r: _unchecked(
                r,
                source=_unchecked(
                    r.source,
                    candidate_files=(
                        _unchecked(
                            r.source.candidate_files[0],
                            sha256_content_digest="d" * 64,
                        ),
                    ),
                ),
            ),
        ),
        (
            "roi",
            lambda r: _unchecked(
                r,
                roi_scope=_unchecked(
                    r.roi_scope,
                    discovered_roi_ids=("ROI1", "ROI0"),
                    included_roi_ids=("ROI0",),
                    excluded_roi_ids=("ROI1",),
                ),
            ),
        ),
        (
            "correction_parameter",
            lambda r: _unchecked(
                r,
                correction=_unchecked(
                    r.correction,
                    dynamic_fit_parameter_values=(
                        _unchecked(
                            r.correction.dynamic_fit_parameter_values[0],
                            value="robust_global_event_reject",
                        ),
                    ),
                ),
            ),
        ),
        (
            "feature",
            lambda r: _unchecked(
                r,
                feature_event=_unchecked(
                    r.feature_event,
                    effective_values=(
                        _unchecked(
                            r.feature_event.effective_values[0],
                            value="delta_f",
                        ),
                    ),
                ),
            ),
        ),
        (
            "output",
            lambda r: _unchecked(
                r,
                output=_unchecked(r.output, output_base_canonical=r"c:\other"),
            ),
        ),
    ],
)
def test_intent_identity_changes_with_semantic_sections(section, change):
    request = _valid_request()
    changed = change(request)
    base = _map(request)
    result = _map(
        changed,
        identity=contracts.compute_guided_backend_validation_request_identity(changed),
    )
    assert isinstance(result, mapping.GuidedProductionMappingSuccess), section
    assert result.canonical_intent_identity != base.canonical_intent_identity


@pytest.mark.parametrize(
    "input_value,identity,expected",
    [
        (object(), _A, "request_missing_or_invalid"),
        (_valid_request(), "bad", "request_identity_invalid"),
        (_valid_request(), "A" * 64, "request_identity_invalid"),
        (_valid_request(), "f" * 64, "request_identity_mismatch"),
    ],
)
def test_request_and_identity_refusals(input_value, identity, expected):
    result = mapping.map_guided_validation_request_to_execution_intent(
        input_value,  # type: ignore[arg-type]
        canonical_request_identity=identity,
        application_build_identity=_build(),
        mapping_contract=_contract(),
    )
    assert _category(result) == expected
    assert result.partial_intent is None
    assert result.canonical_intent_identity is None


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda r: _unchecked(r, request_schema_version="v2"), "unsupported_request_schema"),
        (lambda r: _unchecked(r, validation_scope="wrong"), "unsupported_validation_scope"),
        (lambda r: _unchecked(r, subset_rule_version="wrong"), "unsupported_subset_rule"),
        (
            lambda r: _unchecked(
                r, source=_unchecked(r.source, source_format="npm")
            ),
            "unsupported_source_format",
        ),
        (
            lambda r: _unchecked(
                r,
                acquisition_dataset=_unchecked(
                    r.acquisition_dataset, acquisition_mode="continuous"
                ),
            ),
            "unsupported_acquisition_mode",
        ),
        (
            lambda r: _unchecked(
                r,
                acquisition_dataset=_unchecked(
                    r.acquisition_dataset, allow_partial_final_window=True
                ),
            ),
                "incomplete_final_policy_not_supported",
        ),
        (
            lambda r: _unchecked(
                r,
                acquisition_dataset=_unchecked(
                    r.acquisition_dataset,
                    exclude_incomplete_final_rwd_chunk=True,
                ),
            ),
                None,
        ),
        (
            lambda r: _unchecked(
                r,
                correction=_unchecked(
                    r.correction, global_dynamic_fit_mode="signal_only_f0"
                ),
            ),
            "unsupported_correction_strategy",
        ),
        (
            lambda r: _unchecked(
                r,
                correction=_unchecked(
                    r.correction,
                    confirmed_marks=(
                        _unchecked(
                            r.correction.confirmed_marks[0],
                            selected_dynamic_fit_mode="robust_global_event_reject",
                        ),
                    ),
                ),
            ),
            "mixed_strategy_not_supported",
        ),
        (
            lambda r: _unchecked(
                r, output=_unchecked(r.output, overwrite=True)
            ),
            "output_policy_not_supported",
        ),
        (
            lambda r: _unchecked(
                r, output=_unchecked(r.output, precreate=True)
            ),
            "output_policy_not_supported",
        ),
        (
            lambda r: _unchecked(
                r, output=_unchecked(r.output, blocker_categories=("unsafe",))
            ),
            "output_policy_not_supported",
        ),
        (
            lambda r: _unchecked(
                r,
                local_contract=_unchecked(
                    r.local_contract,
                    deferred_capabilities=r.local_contract.deferred_capabilities
                    + ("unknown_future",),
                ),
            ),
            "deferred_capability_blocks_mapping",
        ),
    ],
)
def test_semantic_refusals(mutate, expected):
    request = mutate(_valid_request())
    result = _map(
        request,
        identity=contracts.compute_guided_backend_validation_request_identity(
            request
        ),
    )
    if expected is None:
        assert isinstance(result, mapping.GuidedProductionMappingSuccess)
    else:
        assert _category(result) == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        (
            "runner_contract_version",
            "runner_contract_unavailable",
        ),
        (
            "candidate_manifest_execution_contract_version",
            "candidate_snapshot_execution_contract_unavailable",
        ),
        (
            "roi_execution_contract_version",
            "roi_execution_contract_unavailable",
        ),
        (
            "config_mapping_contract_version",
            "mapping_contract_unavailable",
        ),
    ],
)
def test_missing_execution_contract_refuses(field, expected):
    contract = _unchecked(_contract(), **{field: "unknown"})
    assert _category(_map(contract=contract)) == expected


def test_unusable_build_identity_refuses():
    build = _unchecked(_build(), canonical_identity=_B)
    assert _category(_map(build=build)) == "app_build_identity_unusable"


def test_unknown_typed_config_field_refuses():
    request = _valid_request()
    request = _unchecked(
        request,
        feature_event=_unchecked(
            request.feature_event,
            effective_values=(
                _unchecked(
                    request.feature_event.effective_values[0],
                    field_name="unknown_config_field",
                ),
            ),
        ),
    )
    result = _map(
        request,
        identity=contracts.compute_guided_backend_validation_request_identity(
            request
        ),
    )
    assert _category(result) == "production_config_field_unmapped"


def test_correction_typed_fields_accept_real_dynamic_fit_contract_field_names():
    """4J16k10: GuidedNewAnalysisDynamicFitParameterContract emits
    'slope_constraint' and 'min_slope' (not the stale 'dynamic_fit_
    slope_constraint'/'dynamic_fit_min_slope' names). Production mapping
    must accept the real field names a Guided GUI plan actually produces."""
    request = _valid_request()
    request = _unchecked(
        request,
        correction=_unchecked(
            request.correction,
            dynamic_fit_parameter_values=(
                contracts.GuidedBackendTypedFieldValue(
                    "dynamic_fit_mode", "str", "global_linear_regression"
                ),
                contracts.GuidedBackendTypedFieldValue(
                    "slope_constraint", "str", "unconstrained"
                ),
                contracts.GuidedBackendTypedFieldValue(
                    "min_slope", "float", 0.0
                ),
            ),
        ),
    )
    result = _map(
        request,
        identity=contracts.compute_guided_backend_validation_request_identity(
            request
        ),
    )
    assert isinstance(result, mapping.GuidedProductionMappingSuccess)
    mapped_names = {
        item.field_name
        for item in result.intent.correction.dynamic_fit_parameter_values
    }
    assert {"slope_constraint", "min_slope"} <= mapped_names


def test_acquisition_typed_fields_accept_real_gui_dataset_contract_fields():
    """4J16k10: the real GUI cache-free dataset-contract snapshot
    (gui/main_window.py _guided_new_analysis_dataset_contract_candidate)
    always emits acquisition_mode, allow_partial_final_window,
    exclude_incomplete_final_rwd_chunk, input_format, resolved_input_format,
    continuous_window_sec, and continuous_step_sec as typed semantic
    values, in addition to rwd_time_col/uv_suffix/sig_suffix. These are
    duplicates of fields already validated elsewhere in the request, not
    new production capability; production mapping must accept them rather
    than refuse with production_config_field_unmapped."""
    request = _valid_request()
    real_gui_semantic_values = (
        contracts.GuidedBackendTypedFieldValue(
            "rwd_time_col", "str", "Time(s)"
        ),
        contracts.GuidedBackendTypedFieldValue("uv_suffix", "str", "-410"),
        contracts.GuidedBackendTypedFieldValue("sig_suffix", "str", "-470"),
        contracts.GuidedBackendTypedFieldValue(
            "acquisition_mode", "str", "intermittent"
        ),
        contracts.GuidedBackendTypedFieldValue(
            "allow_partial_final_window", "bool", False
        ),
        contracts.GuidedBackendTypedFieldValue(
            "exclude_incomplete_final_rwd_chunk", "bool", False
        ),
        contracts.GuidedBackendTypedFieldValue("input_format", "str", "rwd"),
        contracts.GuidedBackendTypedFieldValue(
            "resolved_input_format", "str", "rwd"
        ),
        contracts.GuidedBackendTypedFieldValue(
            "continuous_window_sec", "NoneType", None
        ),
        contracts.GuidedBackendTypedFieldValue(
            "continuous_step_sec", "NoneType", None
        ),
    )
    request = _unchecked(
        request,
        acquisition_dataset=_unchecked(
            request.acquisition_dataset,
            semantic_values=real_gui_semantic_values,
        ),
    )
    result = _map(
        request,
        identity=contracts.compute_guided_backend_validation_request_identity(
            request
        ),
    )
    assert isinstance(result, mapping.GuidedProductionMappingSuccess)
    mapped_names = {
        item.field_name for item in result.intent.acquisition.semantic_values
    }
    assert {item.field_name for item in real_gui_semantic_values} <= mapped_names


def test_intent_identity_canonicalizes_per_roi_production_strategy_map():
    """4J16k10: a real Guided GUI plan populates
    correction.per_roi_production_strategy_map with
    GuidedBackendPerRoiProductionStrategy entries. The identity
    canonicalizer (_canonical_value / _INTENT_IDENTITY_MODEL_FIELDS) must
    know how to encode GuidedProductionPerRoiStrategy, or intent
    construction fails with ValueError('Unsupported production intent
    value type.') for every real plan, not just synthetic fixtures that
    happen to leave this field empty."""
    request = _valid_request()
    per_roi_entry = contracts.GuidedBackendPerRoiProductionStrategy(
        roi_id="ROI0",
        strategy_family="dynamic_fit",
        dynamic_fit_mode="global_linear_regression",
        selected_strategy="global_linear_regression",
        evidence_source_type="local_correction_preview",
        evidence_reference_json="{}",
        explicit_user_mark=True,
        current_or_stale="current",
    )
    request = _unchecked(
        request,
        correction=_unchecked(
            request.correction,
            production_strategy_map_version="guided_production_strategy_map.v1",
            per_roi_production_strategy_map=(per_roi_entry,),
        ),
    )
    result = _map(
        request,
        identity=contracts.compute_guided_backend_validation_request_identity(
            request
        ),
    )
    assert isinstance(result, mapping.GuidedProductionMappingSuccess)
    assert len(result.intent.correction.per_roi_production_strategy_map) == 1
    mapped_entry = result.intent.correction.per_roi_production_strategy_map[0]
    assert isinstance(mapped_entry, mapping.GuidedProductionPerRoiStrategy)
    assert mapped_entry.roi_id == "ROI0"
    # The identity must be computable (no "Unsupported production intent
    # value type." ValueError) and must recompute consistently.
    recomputed = mapping.compute_guided_production_execution_intent_identity(
        result.intent
    )
    assert recomputed == result.canonical_intent_identity


def test_mapping_contract_blocking_deferred_capability_refuses():
    request = _valid_request()
    request = _unchecked(
        request,
        local_contract=_unchecked(
            request.local_contract,
            deferred_capabilities=request.local_contract.deferred_capabilities
            + ("future_production_blocker",),
        ),
    )
    contract = _unchecked(
        _contract(),
        blocking_deferred_capabilities=(
            "future_production_blocker",
        ),
    )
    result = _map(
        request,
        contract=contract,
        identity=contracts.compute_guided_backend_validation_request_identity(
            request
        ),
    )
    assert _category(result) == "deferred_capability_blocks_mapping"


def test_explicitly_allowed_nonblocking_capabilities_remain_in_intent():
    result = _map()
    assert isinstance(result, mapping.GuidedProductionMappingSuccess)
    assert "backend_validation" in result.intent.deferred_capabilities
    assert "full_source_manifest_identity" in result.intent.deferred_capabilities
    assert "strict_roi_inventory_identity" in result.intent.deferred_capabilities


def test_build_identity_is_resolved_and_removed_from_intent():
    result = _map()
    assert isinstance(result, mapping.GuidedProductionMappingSuccess)
    assert "app_build_identity" not in result.intent.deferred_capabilities
    assert result.intent.application_build_identity == _build()


def test_run_authorization_is_explicitly_stage_deferred():
    contract = _contract()
    assert contract.stage_deferred_capabilities == ("run_authorization",)
    assert "run_authorization" not in contract.blocking_deferred_capabilities
    result = _map(contract=contract)
    assert isinstance(result, mapping.GuidedProductionMappingSuccess)
    assert "run_authorization" in result.intent.deferred_capabilities


def test_request_field_classification_is_complete():
    expected_types = {
        contracts.GuidedBackendValidationRequest,
        contracts.GuidedBackendSourceRequest,
        contracts.GuidedBackendSourceCandidateFile,
        contracts.GuidedBackendAcquisitionDatasetRequest,
        contracts.GuidedBackendRwdParserRequest,
        contracts.GuidedBackendRoiScopeRequest,
        contracts.GuidedBackendCorrectionRequest,
        contracts.GuidedBackendConfirmedStrategyMark,
        contracts.GuidedBackendDiagnosticEvidenceRequest,
        contracts.GuidedBackendEvidenceReference,
        contracts.GuidedBackendFeatureEventRequest,
        contracts.GuidedBackendOutputRequest,
        contracts.GuidedBackendOutputRelationship,
        contracts.GuidedBackendLocalContractState,
        contracts.GuidedBackendTypedFieldValue,
    }
    assert set(mapping.REQUEST_FIELD_CLASSIFICATIONS) == expected_types
    for model, classifications in mapping.REQUEST_FIELD_CLASSIFICATIONS.items():
        assert set(classifications) == {item.name for item in fields(model)}
        assert set(classifications.values()) <= {
            "mapped_to_intent",
            "gate_only",
            "provenance_only",
            "externally_resolved",
            "deferred_allowed",
            "deferred_blocking",
        }


def test_intent_identity_field_coverage_is_explicit():
    coverage = mapping._INTENT_IDENTITY_MODEL_FIELDS
    for model, mapped in coverage.items():
        expected = tuple(
            item.name
            for item in fields(model)
            if not (
                model is mapping.GuidedProductionExecutionIntent
                and item.name == "canonical_intent_identity"
            )
        )
        assert mapped == expected


def test_intent_has_no_execution_artifact_fields():
    prohibited = {
        "run_id",
        "run_dir",
        "output_dir_allocated",
        "config_path",
        "argv",
        "command",
        "timestamp",
        "process",
    }
    for model in mapping._INTENT_IDENTITY_MODEL_FIELDS:
        if model is mapping.ApplicationBuildIdentity:
            continue
        names = {item.name for item in fields(model)}
        assert not any(marker in name for marker in prohibited for name in names)


def test_mapper_and_build_helper_perform_no_filesystem_io(monkeypatch):
    request = _valid_request()
    identity = contracts.compute_guided_backend_validation_request_identity(request)

    def fail(*_args, **_kwargs):
        raise AssertionError("filesystem I/O is prohibited")

    monkeypatch.setattr(builtins, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(Path, "read_bytes", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(Path, "touch", fail)
    monkeypatch.setattr(os, "stat", fail)
    monkeypatch.setattr(os, "scandir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    build = _build()
    result = mapping.map_guided_validation_request_to_execution_intent(
        request,
        canonical_request_identity=identity,
        application_build_identity=build,
        mapping_contract=_contract(),
    )
    assert isinstance(result, mapping.GuidedProductionMappingSuccess)


def test_module_has_no_gui_runner_or_mapping_side_effect_imports():
    source = Path(mapping.__file__).read_text(encoding="utf-8")
    imports = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    prohibited = (
        "gui.",
        "run_full_pipeline_deliverables",
        "materialize_guided_backend",
        "compile_guided_backend",
        "guided_backend_validation_workflow",
    )
    assert not any(
        imported == "gui"
        or any(item in imported for item in prohibited)
        for imported in imports
    )


# Per-ROI feature/event map wiring (4J16k32c)


def _typed_contract(name, value):
    return contracts.GuidedBackendTypedFieldValue(
        field_name=name, value_type=type(value).__name__, value=value
    )


def test_map_carries_per_roi_feature_event_map_and_backend_shapes_are_complete():
    override_entry = contracts.GuidedBackendPerRoiFeatureEvent(
        roi_id="ROI0",
        source="override",
        feature_event_profile_id="custom-roi0",
        override_config_fields=(
            _typed_contract("peak_threshold_method", "percentile"),
        ),
        effective_config_fields=(
            _typed_contract("event_signal", "dff"),
            _typed_contract("signal_excursion_polarity", "positive"),
            _typed_contract("peak_threshold_method", "percentile"),
            _typed_contract("peak_threshold_k", 2.5),
            _typed_contract("peak_threshold_percentile", 95.0),
            _typed_contract("peak_threshold_abs", 0.0),
            _typed_contract("peak_min_distance_sec", 1.0),
            _typed_contract("peak_min_prominence_k", 2.0),
            _typed_contract("peak_min_width_sec", 0.3),
            _typed_contract("peak_pre_filter", "none"),
            _typed_contract("event_auc_baseline", "zero"),
        ),
        explicit_user_mark=True,
        current_or_stale="current",
    )
    default_entry = contracts.GuidedBackendPerRoiFeatureEvent(
        roi_id="ROI1",
        source="default",
        feature_event_profile_id="profile-001",
        override_config_fields=(),
        effective_config_fields=(_typed_contract("event_signal", "dff"),),
        explicit_user_mark=True,
        current_or_stale="current",
    )
    base_request = _valid_request()
    request = replace(
        base_request,
        feature_event=replace(
            base_request.feature_event,
            per_roi_feature_event_map_version="per_roi_feature_event_map.v1",
            per_roi_feature_event_map=(override_entry, default_entry),
        ),
    )

    result = _map(request=request)

    assert isinstance(result, mapping.GuidedProductionMappingSuccess)
    intent = result.intent
    by_roi = {
        entry.roi_id: entry
        for entry in intent.feature_event.per_roi_feature_event_map
    }
    assert set(by_roi) == {"ROI0", "ROI1"}
    assert by_roi["ROI0"].source == "override"
    assert by_roi["ROI1"].source == "default"
    assert (
        intent.feature_event.per_roi_feature_event_map_version
        == "per_roi_feature_event_map.v1"
    )

    shapes = mapping.build_per_roi_feature_event_backend_shapes(intent)

    assert set(shapes["per_roi_override_config_fields"]) == {"ROI0"}
    assert shapes["per_roi_override_config_fields"]["ROI0"] == {
        "peak_threshold_method": "percentile"
    }

    # Applied-dF/F must receive complete effective fields, never the sparse
    # override alone (guided_applied_dff_orchestration.py fails closed on
    # anything less).
    from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS

    effective_for_overrides = shapes[
        "per_roi_effective_feature_config_fields_for_overrides"
    ]
    assert set(effective_for_overrides) == {"ROI0"}
    assert set(effective_for_overrides["ROI0"]) == FEATURE_EVENT_CONFIG_FIELDS
    assert effective_for_overrides["ROI0"]["peak_threshold_method"] == "percentile"

    provenance = shapes["per_roi_feature_provenance"]
    assert set(provenance) == {"ROI0", "ROI1"}
    assert provenance["ROI0"]["source"] == "override"
    assert provenance["ROI0"]["feature_event_profile_id"] == "custom-roi0"
    assert provenance["ROI1"]["source"] == "default"


def test_backend_shapes_empty_for_global_only_request_with_no_per_roi_map():
    request = _valid_request()
    assert request.feature_event.per_roi_feature_event_map == ()

    result = _map(request=request)
    assert isinstance(result, mapping.GuidedProductionMappingSuccess)

    shapes = mapping.build_per_roi_feature_event_backend_shapes(result.intent)
    assert shapes == {
        "per_roi_override_config_fields": {},
        "per_roi_effective_feature_config_fields_for_overrides": {},
        "per_roi_feature_provenance": {},
    }


def test_backend_shapes_rejects_non_intent_argument():
    with pytest.raises(TypeError):
        mapping.build_per_roi_feature_event_backend_shapes(object())
