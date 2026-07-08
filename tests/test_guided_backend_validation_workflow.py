from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import photometry_pipeline.guided_backend_validation_workflow as workflow
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
)


class _Materialized:
    def __init__(self):
        self.facts = object()


class _MaterializationFailure:
    def __init__(self):
        self.blocking_issues = (
            SimpleNamespace(
                category="missing_source",
                section="source",
                message="Source is missing.",
                detail_code="source_missing",
            ),
        )


class _Compiled:
    def __init__(self):
        self.request = object()
        self.canonical_request_identity = "a" * 64


class _CompileFailure:
    def __init__(self):
        self.blocking_issues = (
            SimpleNamespace(
                category="missing_source_snapshot",
                section="source",
                message="Source facts are missing.",
                detail_code="source_facts_missing",
            ),
        )


def _accepted_validation():
    return SimpleNamespace(
        accepted=True,
        blocking_issues=(),
        request_identity="a" * 64,
        run_authorization=False,
    )


def _refused_validation():
    return SimpleNamespace(
        accepted=False,
        blocking_issues=(
            SimpleNamespace(
                category="local_contract_not_clean",
                section="local_contract",
                message="Local contract is not clean.",
                detail_code="local_contract_invalid",
            ),
        ),
        request_identity="a" * 64,
        run_authorization=False,
    )


@pytest.fixture
def successful_stages(monkeypatch):
    monkeypatch.setattr(
        workflow,
        "GuidedBackendValidationMaterializationSuccess",
        _Materialized,
    )
    monkeypatch.setattr(
        workflow,
        "GuidedBackendValidationMaterializationFailure",
        _MaterializationFailure,
    )
    monkeypatch.setattr(
        workflow,
        "GuidedBackendValidationCompileSuccess",
        _Compiled,
    )
    monkeypatch.setattr(
        workflow,
        "GuidedBackendValidationCompileFailure",
        _CompileFailure,
    )
    monkeypatch.setattr(
        workflow,
        "materialize_guided_backend_validation_facts",
        lambda *_args, **_kwargs: _Materialized(),
    )
    monkeypatch.setattr(
        workflow,
        "compile_guided_backend_validation_request",
        lambda *_args, **_kwargs: _Compiled(),
    )
    monkeypatch.setattr(
        workflow,
        "validate_guided_backend_validation_request",
        lambda *_args, **_kwargs: _accepted_validation(),
    )


def _call(**kwargs):
    return workflow.validate_current_guided_draft_for_backend(
        GuidedNewAnalysisDraftPlan(),
        parser_contract=(
            workflow.build_guided_backend_validation_parser_contract()
        ),
        validator_contract=workflow.build_guided_backend_validator_contract(),
        **kwargs,
    )


def test_application_owned_contract_factories_are_explicit_and_resolved():
    parser = workflow.build_guided_backend_validation_parser_contract()
    validator = workflow.build_guided_backend_validator_contract()
    # 4J16k18: extended to also recognize "TimeStamp" as a real-world RWD
    # time-column name, alongside the original "Time(s)".
    assert parser.time_column_candidates == ("Time(s)", "TimeStamp")
    assert parser.uv_suffix_candidates == ("-410",)
    assert parser.signal_suffix_candidates == ("-470",)
    assert parser.unresolved_inputs == ()
    assert validator.validator_capability_version
    assert validator.validator_capability_version.lower() != "unknown"


def test_accepted_workflow_preserves_identity_and_never_authorizes_run(
    successful_stages,
):
    outcome = _call()
    assert outcome.status == "validator_accepted"
    assert outcome.accepted_for_backend_validation is True
    assert outcome.request_identity == "a" * 64
    assert outcome.validation_result.accepted is True
    assert outcome.compile_result is not None
    assert outcome.materialization_result is not None
    assert outcome.blocking_issues == ()
    assert outcome.run_authorization is False
    assert outcome.no_files_written is True
    assert outcome.no_directories_created is True
    assert outcome.no_artifacts_created is True
    assert outcome.no_run_id_allocated is True
    assert outcome.no_runner_invoked is True


def test_outcome_models_are_frozen_and_exclude_execution_fields(
    successful_stages,
):
    outcome = _call()
    with pytest.raises(FrozenInstanceError):
        outcome.status = "cancelled"  # type: ignore[misc]
    prohibited = {
        "config_path",
        "argv",
        "command_text",
        "run_dir",
        "run_id",
        "artifact_path",
        "timestamp",
    }
    assert not prohibited & {item.name for item in fields(outcome)}


def test_materialization_failure_is_normalized(monkeypatch, successful_stages):
    monkeypatch.setattr(
        workflow,
        "materialize_guided_backend_validation_facts",
        lambda *_args, **_kwargs: _MaterializationFailure(),
    )
    outcome = _call()
    assert outcome.status == "materialization_failed"
    assert outcome.request_identity is None
    assert outcome.compile_result is None
    assert outcome.validation_result is None
    issue = outcome.blocking_issues[0]
    assert (
        issue.stage,
        issue.category,
        issue.section,
        issue.detail_code,
    ) == ("materialization", "missing_source", "source", "source_missing")


def test_compile_failure_is_normalized(monkeypatch, successful_stages):
    monkeypatch.setattr(
        workflow,
        "compile_guided_backend_validation_request",
        lambda *_args, **_kwargs: _CompileFailure(),
    )
    outcome = _call()
    assert outcome.status == "compile_failed"
    assert outcome.materialization_result is not None
    assert outcome.request_identity is None
    assert outcome.validation_result is None
    assert outcome.blocking_issues[0].stage == "compile"


def test_validator_refusal_is_normalized(monkeypatch, successful_stages):
    monkeypatch.setattr(
        workflow,
        "validate_guided_backend_validation_request",
        lambda *_args, **_kwargs: _refused_validation(),
    )
    outcome = _call()
    assert outcome.status == "validator_refused"
    assert outcome.request_identity == "a" * 64
    assert outcome.validation_result.accepted is False
    assert outcome.blocking_issues[0].stage == "validator"
    assert outcome.run_authorization is False


@pytest.mark.parametrize("stage", ["materialization", "compile", "validator"])
def test_internal_exceptions_map_to_safe_internal_error(
    monkeypatch,
    successful_stages,
    stage: str,
):
    target = {
        "materialization": "materialize_guided_backend_validation_facts",
        "compile": "compile_guided_backend_validation_request",
        "validator": "validate_guided_backend_validation_request",
    }[stage]

    def fail(*_args, **_kwargs):
        raise RuntimeError("sensitive repr must not escape")

    monkeypatch.setattr(workflow, target, fail)
    outcome = _call()
    assert outcome.status == "internal_error"
    assert outcome.blocking_issues[0].category == "workflow_internal_error"
    assert "sensitive" not in outcome.user_summary
    assert outcome.request_identity is None


@pytest.mark.parametrize("cancel_on_call", [1, 2, 3])
def test_cancellation_between_each_stage(
    successful_stages,
    cancel_on_call: int,
):
    calls = 0

    def cancellation_check():
        nonlocal calls
        calls += 1
        return calls == cancel_on_call

    outcome = _call(cancellation_check=cancellation_check)
    assert outcome.status == "cancelled"
    assert outcome.accepted_for_backend_validation is False
    assert outcome.request_identity is None
    assert outcome.blocking_issues == ()
    assert outcome.run_authorization is False


def test_outcome_invariants_reject_run_authorization(successful_stages):
    accepted = _call()
    with pytest.raises(ValueError):
        workflow.GuidedBackendValidationWorkflowOutcome(
            **{
                **accepted.__dict__,
                "run_authorization": True,
            }
        )


def test_workflow_forwards_protected_roots_and_contracts(
    monkeypatch,
    successful_stages,
):
    captured = {}

    def materialize(draft, **kwargs):
        captured.update(kwargs)
        return _Materialized()

    monkeypatch.setattr(
        workflow, "materialize_guided_backend_validation_facts", materialize
    )
    roots = (("completed_run", r"c:\completed"),)
    _call(additional_protected_roots=roots)
    assert captured["additional_protected_roots"] == roots
    assert captured["parser_contract"].time_column_candidates == (
        "Time(s)",
        "TimeStamp",
    )


def test_workflow_calls_no_write_or_run_api(monkeypatch, successful_stages):
    def fail(*_args, **_kwargs):
        raise AssertionError("write/run API is prohibited")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(Path, "touch", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    assert _call().status == "validator_accepted"
