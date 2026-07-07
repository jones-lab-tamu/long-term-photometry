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
            "incomplete_final_policy_not_supported",
        ),
        (
            lambda r: _unchecked(
                r,
                correction=_unchecked(
                    r.correction, global_dynamic_fit_mode="signal_only_f0"
                ),
            ),
            "signal_only_not_supported",
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
