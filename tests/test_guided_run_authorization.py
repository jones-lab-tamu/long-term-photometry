from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_backend_validation_request as contracts
import photometry_pipeline.guided_backend_validator as validator
import photometry_pipeline.guided_execution_preflight as preflight
import photometry_pipeline.guided_production_mapping as mapping
import photometry_pipeline.guided_run_authorization as authorization
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    GuidedBackendValidationWorkflowIssue,
    GuidedBackendValidationWorkflowOutcome,
    build_guided_backend_validation_parser_contract,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from tests.test_guided_backend_validator import (
    _contract as _validator_contract,
    _request as _valid_request,
)


def _unchecked(instance, **changes):
    result = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            result,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return result


def _accepted_outcome(request=None):
    request = request or _valid_request()
    identity = contracts.compute_guided_backend_validation_request_identity(request)
    compiled = contracts.GuidedBackendValidationCompileSuccess(request, identity)
    validated = validator.validate_guided_backend_validation_request(
        request,
        canonical_request_identity=identity,
        validator_contract=_validator_contract(),
    )
    assert validated.accepted
    return GuidedBackendValidationWorkflowOutcome(
        status="validator_accepted",
        accepted_for_backend_validation=True,
        run_authorization=False,
        request_identity=identity,
        validation_result=validated,
        compile_result=compiled,
        materialization_result=object(),  # Presence is the workflow contract.
        blocking_issues=(),
        user_summary="Accepted.",
    )


def _build():
    return mapping.build_application_build_identity(
        distribution_name="photometry-pipeline",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="abc123",
        source_tree_state="clean",
    )


def _request(outcome=None, *, revision=3, build=None, contract=None):
    return authorization.build_guided_run_authorization_request(
        stored_validation_outcome=outcome or _accepted_outcome(),
        stored_validation_outcome_revision=revision,
        current_gui_revision=revision,
        current_validation_context=GuidedBackendValidationGuiContext(
            draft=GuidedNewAnalysisDraftPlan(),
            parser_contract=build_guided_backend_validation_parser_contract(),
            additional_protected_roots=(),
            validator_contract=_validator_contract(),
            revision=revision,
        ),
        application_build_identity=build or _build(),
        production_mapping_contract=(
            contract or mapping.build_guided_production_mapping_contract()
        ),
    )


def _accepted_candidate(candidate_request):
    provisional = preflight.GuidedCandidateManifestExecutionPreflightResult(
        status="accepted",
        accepted=True,
        contract_version=candidate_request.contract_version,
        runner_contract_version=candidate_request.runner_contract_version,
        expected_candidate_set_digest=(
            candidate_request.expected_candidate_set_digest
        ),
        expected_candidate_content_digest=(
            candidate_request.expected_candidate_content_digest
        ),
        actual_candidate_set_digest=candidate_request.expected_candidate_set_digest,
        actual_candidate_content_digest=(
            candidate_request.expected_candidate_content_digest
        ),
        actual_candidates=candidate_request.expected_candidates,
        blocking_issues=(),
        canonical_preflight_identity="0" * 64,
    )
    return replace(
        provisional,
        canonical_preflight_identity=(
            preflight.compute_guided_candidate_preflight_identity(provisional)
        ),
    )


def _accepted_roi(roi_request):
    provisional = preflight.GuidedRoiExecutionPreflightResult(
        status="accepted",
        accepted=True,
        contract_version=roi_request.contract_version,
        runner_contract_version=roi_request.runner_contract_version,
        accepted_candidate_preflight_identity=(
            roi_request.accepted_candidate_preflight_identity
        ),
        source_candidate_content_digest=roi_request.source_candidate_content_digest,
        parser_contract_digest=roi_request.parser_contract_digest,
        expected_strict_roi_inventory_digest=(
            roi_request.expected_strict_roi_inventory_digest
        ),
        actual_strict_roi_inventory_digest=(
            roi_request.expected_strict_roi_inventory_digest
        ),
        actual_discovered_roi_ids=roi_request.expected_discovered_roi_ids,
        actual_included_roi_ids=roi_request.expected_included_roi_ids,
        actual_excluded_roi_ids=roi_request.expected_excluded_roi_ids,
        blocking_issues=(),
        canonical_preflight_identity="0" * 64,
    )
    return replace(
        provisional,
        canonical_preflight_identity=(
            preflight.compute_guided_roi_preflight_identity(provisional)
        ),
    )


@pytest.fixture
def composed(monkeypatch):
    request = _request()
    monkeypatch.setattr(
        authorization.validation_workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: request.stored_validation_outcome,
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_candidate_manifest_execution_preflight",
        lambda candidate_request, **_kwargs: _accepted_candidate(candidate_request),
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_roi_execution_preflight",
        lambda roi_request, **_kwargs: _accepted_roi(roi_request),
    )
    return request


def _authorize(request):
    return authorization.authorize_guided_run(request)


def _category(result):
    return result.blocking_issues[0].category


def test_happy_path_authorizes_with_complete_frozen_proof(composed):
    result = _authorize(composed)
    assert result.status == "authorized"
    assert result.authorized and result.run_authorization
    assert result.production_intent is not None
    assert result.candidate_preflight_result.accepted
    assert result.roi_preflight_result.accepted
    assert authorization.compute_guided_run_authorization_identity(result) == (
        result.canonical_authorization_identity
    )
    assert all(
        (
            result.no_files_written,
            result.no_directories_created,
            result.no_artifacts_created,
            result.no_output_allocated,
            result.no_run_id_allocated,
            result.no_config_or_argv_generated,
            result.no_runner_invoked,
        )
    )
    with pytest.raises(FrozenInstanceError):
        result.status = "refused"  # type: ignore[misc]


@pytest.mark.parametrize(
    "change,expected",
    [
        (lambda r: _unchecked(r, stored_validation_outcome=object()), "stored_validation_missing"),
        (
            lambda r: _unchecked(
                r,
                stored_validation_outcome=_unchecked(
                    r.stored_validation_outcome,
                    status="validator_refused",
                    accepted_for_backend_validation=False,
                ),
            ),
            "stored_validation_not_accepted",
        ),
        (
            lambda r: _unchecked(
                r,
                stored_validation_outcome=_unchecked(
                    r.stored_validation_outcome, stale=True
                ),
            ),
            "stored_validation_stale",
        ),
        (
            lambda r: _unchecked(
                r,
                stored_validation_outcome=_unchecked(
                    r.stored_validation_outcome, run_authorization=True
                ),
            ),
            "stored_validation_not_accepted",
        ),
        (lambda r: _unchecked(r, current_gui_revision=4), "guided_revision_mismatch"),
        (
            lambda r: _unchecked(
                r,
                stored_validation_outcome=_unchecked(
                    r.stored_validation_outcome, compile_result=None
                ),
            ),
            "stored_validation_not_accepted",
        ),
        (
            lambda r: _unchecked(
                r,
                stored_validation_outcome=_unchecked(
                    r.stored_validation_outcome, validation_result=None
                ),
            ),
            "stored_validation_not_accepted",
        ),
    ],
)
def test_stored_validation_refusals(composed, change, expected):
    assert _category(_authorize(change(composed))) == expected


@pytest.mark.parametrize(
    "field,expected",
    [
        ("request_identity", "stored_request_identity_invalid"),
        ("compile", "stored_request_identity_inconsistent"),
        ("validation", "stored_request_identity_inconsistent"),
        ("request", "stored_request_identity_inconsistent"),
    ],
)
def test_stored_identity_refusals(composed, field, expected):
    outcome = composed.stored_validation_outcome
    if field == "request_identity":
        outcome = _unchecked(outcome, request_identity="bad")
    elif field == "compile":
        outcome = _unchecked(
            outcome,
            compile_result=_unchecked(
                outcome.compile_result, canonical_request_identity="d" * 64
            ),
        )
    elif field == "validation":
        outcome = _unchecked(
            outcome,
            validation_result=_unchecked(
                outcome.validation_result, request_identity="d" * 64
            ),
        )
    else:
        changed = _unchecked(
            outcome.compile_result.request,
            validator_capability_version="changed.v1",
        )
        outcome = _unchecked(
            outcome,
            compile_result=_unchecked(outcome.compile_result, request=changed),
        )
    assert _category(_authorize(_unchecked(composed, stored_validation_outcome=outcome))) == expected


def test_fresh_refusal_and_identity_continuity(composed, monkeypatch):
    refused = _unchecked(
        composed.stored_validation_outcome,
        status="validator_refused",
        accepted_for_backend_validation=False,
    )
    monkeypatch.setattr(
        authorization.validation_workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: refused,
    )
    assert _category(_authorize(composed)) == "fresh_validation_refused"

    changed_request = _valid_request()
    fresh = _accepted_outcome(
        _unchecked(
            changed_request,
            output=_unchecked(
                changed_request.output,
                output_base_canonical=r"c:\different-output",
            ),
        )
    )
    monkeypatch.setattr(
        authorization.validation_workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: fresh,
    )
    assert _category(_authorize(composed)) == "fresh_request_identity_mismatch"


def test_fresh_internal_identity_mismatch_refuses(composed, monkeypatch):
    fresh = _unchecked(
        composed.stored_validation_outcome,
        validation_result=_unchecked(
            composed.stored_validation_outcome.validation_result,
            request_identity="d" * 64,
        ),
    )
    monkeypatch.setattr(
        authorization.validation_workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: fresh,
    )
    assert _category(_authorize(composed)) == "fresh_request_identity_inconsistent"


def test_build_and_mapping_refusals(composed, monkeypatch):
    missing = _unchecked(composed, application_build_identity=object())
    assert _category(_authorize(missing)) == "application_build_identity_missing"
    unusable = _unchecked(
        composed.application_build_identity, canonical_identity="d" * 64
    )
    assert _category(
        _authorize(_unchecked(composed, application_build_identity=unusable))
    ) == "application_build_identity_unusable"

    failure = mapping.GuidedProductionMappingFailure(
        (
            mapping.GuidedProductionMappingIssue(
                "mapping_internal_error", "mapping", "Refused.", "test"
            ),
        )
    )
    monkeypatch.setattr(
        authorization.production_mapping,
        "map_guided_validation_request_to_execution_intent",
        lambda *_args, **_kwargs: failure,
    )
    result = _authorize(composed)
    assert _category(result) == "production_mapping_refused"
    assert result.blocking_issues[0].detail_code == "mapping_internal_error"


def test_mapping_identity_inconsistency_refuses(composed, monkeypatch):
    original = (
        authorization.production_mapping.map_guided_validation_request_to_execution_intent
    )

    def mismatched(*args, **kwargs):
        result = original(*args, **kwargs)
        return _unchecked(result, source_request_identity="d" * 64)

    monkeypatch.setattr(
        authorization.production_mapping,
        "map_guided_validation_request_to_execution_intent",
        mismatched,
    )
    assert _category(_authorize(composed)) == "production_intent_identity_inconsistent"


@pytest.mark.parametrize(
    "contract_field",
    (
        "mapping_contract_version",
        "runner_contract_version",
        "candidate_manifest_execution_contract_version",
        "roi_execution_contract_version",
    ),
)
def test_self_consistent_intent_must_bind_to_supplied_production_contract(
    composed,
    monkeypatch,
    contract_field,
):
    original = (
        authorization.production_mapping.map_guided_validation_request_to_execution_intent
    )

    def mismatched(*args, **kwargs):
        result = original(*args, **kwargs)
        intent = result.intent
        if contract_field == "mapping_contract_version":
            intent = replace(intent, mapping_contract_version="mapping.changed.v1")
        elif contract_field == "runner_contract_version":
            intent = replace(intent, runner_contract_version="runner.changed.v1")
        elif contract_field == "candidate_manifest_execution_contract_version":
            intent = replace(
                intent,
                input_source=replace(
                    intent.input_source,
                    candidate_manifest_execution_contract_version=(
                        "candidate.changed.v1"
                    ),
                ),
            )
        else:
            intent = replace(
                intent,
                roi_scope=replace(
                    intent.roi_scope,
                    roi_execution_contract_version="roi.changed.v1",
                ),
            )
        intent = replace(intent, canonical_intent_identity="0" * 64)
        intent = replace(
            intent,
            canonical_intent_identity=(
                mapping.compute_guided_production_execution_intent_identity(intent)
            ),
        )
        return mapping.GuidedProductionMappingSuccess(
            intent=intent,
            canonical_intent_identity=intent.canonical_intent_identity,
            source_request_identity=intent.source_request_identity,
        )

    monkeypatch.setattr(
        authorization.production_mapping,
        "map_guided_validation_request_to_execution_intent",
        mismatched,
    )
    result = _authorize(composed)
    assert _category(result) == "production_intent_identity_inconsistent"
    assert result.blocking_issues[0].section == "production_mapping"
    assert (
        result.blocking_issues[0].detail_code
        == "production_contract_binding_mismatch"
    )


def test_candidate_refusal_identity_and_binding_cases(composed, monkeypatch):
    candidate_request = preflight.derive_candidate_manifest_preflight_request_from_intent(
        _authorize(composed).production_intent
    )
    accepted = _accepted_candidate(candidate_request)
    refusal = _unchecked(
        accepted,
        status="refused",
        accepted=False,
        blocking_issues=(
            preflight.GuidedCandidateManifestExecutionPreflightIssue(
                "candidate_file_digest_mismatch", "source", "Changed.", "changed"
            ),
        ),
        canonical_preflight_identity=None,
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_candidate_manifest_execution_preflight",
        lambda *_args, **_kwargs: refusal,
    )
    assert _category(_authorize(composed)) == "candidate_preflight_refused"

    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_candidate_manifest_execution_preflight",
        lambda *_args, **_kwargs: _unchecked(
            accepted, canonical_preflight_identity="d" * 64
        ),
    )
    assert _category(_authorize(composed)) == "candidate_preflight_identity_inconsistent"

    changed = _unchecked(accepted, actual_candidate_content_digest="d" * 64)
    changed = replace(
        changed,
        canonical_preflight_identity=preflight.compute_guided_candidate_preflight_identity(
            changed
        ),
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_candidate_manifest_execution_preflight",
        lambda *_args, **_kwargs: changed,
    )
    assert _category(_authorize(composed)) == "candidate_intent_binding_mismatch"


def test_roi_refusal_identity_candidate_and_intent_cases(composed, monkeypatch):
    baseline = _authorize(composed)
    roi = baseline.roi_preflight_result
    refusal = _unchecked(
        roi,
        status="refused",
        accepted=False,
        blocking_issues=(
            preflight.GuidedRoiExecutionPreflightIssue(
                "roi_tuple_mismatch", "roi", "Changed.", "changed"
            ),
        ),
        canonical_preflight_identity=None,
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_roi_execution_preflight",
        lambda *_args, **_kwargs: refusal,
    )
    assert _category(_authorize(composed)) == "roi_preflight_refused"

    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_roi_execution_preflight",
        lambda *_args, **_kwargs: _unchecked(
            roi, canonical_preflight_identity="d" * 64
        ),
    )
    assert _category(_authorize(composed)) == "roi_preflight_identity_inconsistent"

    changed_candidate = _unchecked(
        roi, accepted_candidate_preflight_identity="d" * 64
    )
    changed_candidate = replace(
        changed_candidate,
        canonical_preflight_identity=preflight.compute_guided_roi_preflight_identity(
            changed_candidate
        ),
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_roi_execution_preflight",
        lambda *_args, **_kwargs: changed_candidate,
    )
    assert _category(_authorize(composed)) == "roi_candidate_binding_mismatch"

    changed_intent = _unchecked(roi, parser_contract_digest="d" * 64)
    changed_intent = replace(
        changed_intent,
        canonical_preflight_identity=preflight.compute_guided_roi_preflight_identity(
            changed_intent
        ),
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_roi_execution_preflight",
        lambda *_args, **_kwargs: changed_intent,
    )
    assert _category(_authorize(composed)) == "roi_intent_binding_mismatch"


def test_authorization_identity_is_deterministic_sensitive_and_revision_free(
    composed,
):
    result = _authorize(composed)
    assert _authorize(composed).canonical_authorization_identity == (
        result.canonical_authorization_identity
    )
    changed = replace(
        result,
        authorized_gui_revision=99,
        canonical_authorization_identity="0" * 64,
    )
    assert authorization.compute_guided_run_authorization_identity(changed) == (
        result.canonical_authorization_identity
    )
    for field in (
        "fresh_request_identity",
        "production_intent_identity",
        "application_build_identity",
        "candidate_preflight_identity",
        "roi_preflight_identity",
    ):
        mutated = replace(
            result,
            **{field: "d" * 64, "canonical_authorization_identity": "0" * 64},
        )
        assert authorization.compute_guided_run_authorization_identity(mutated) != (
            result.canonical_authorization_identity
        )
    changed_intent = replace(
        result.production_intent,
        mapping_contract_version="mapping.changed.v1",
    )
    mutated = replace(
        result,
        production_intent=changed_intent,
        canonical_authorization_identity="0" * 64,
    )
    assert authorization.compute_guided_run_authorization_identity(mutated) != (
        result.canonical_authorization_identity
    )


def test_identity_field_coverage_is_explicit():
    assert authorization.GUIDED_RUN_AUTHORIZATION_IDENTITY_FIELDS == (
        "authorization_contract_version",
        "validation_request_identity",
        "production_execution_intent_identity",
        "application_build_identity",
        "production_mapping_contract_version",
        "runner_contract_version",
        "candidate_preflight_contract_version",
        "candidate_preflight_identity",
        "roi_preflight_contract_version",
        "roi_preflight_identity",
    )


@pytest.mark.parametrize("cancel_call", range(1, 7))
def test_cancellation_at_each_boundary(composed, cancel_call):
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls == cancel_call

    result = authorization.authorize_guided_run(
        composed, cancellation_check=cancel
    )
    assert result.status == "cancelled"
    assert _category(result) == "authorization_cancelled"
    assert result.canonical_authorization_identity is None
    assert result.production_intent is None
    assert result.candidate_preflight_result is None
    assert result.roi_preflight_result is None
    assert len(result.blocking_issues) == 1


def test_request_and_result_models_exclude_execution_fields(composed):
    prohibited = {
        "run_id",
        "run_dir",
        "config_path",
        "argv",
        "command",
        "artifact_path",
        "output_directory",
        "widget",
    }
    for model in (
        authorization.GuidedRunAuthorizationRequest,
        authorization.GuidedRunAuthorizationResult,
    ):
        names = {item.name for item in fields(model)}
        assert not names & prohibited


def test_authorization_calls_no_write_or_allocation_api(composed, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("write/allocation API is prohibited")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(Path, "touch", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    assert _authorize(composed).authorized


def test_module_import_boundary():
    source = Path(authorization.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    prohibited = (
        "gui",
        "gui.run_spec",
        "photometry_pipeline.config",
        "photometry_pipeline.pipeline",
        "tools.run_full_pipeline_deliverables",
    )
    assert not any(
        name == marker or name.startswith(f"{marker}.")
        for name in imported
        for marker in prohibited
    )
