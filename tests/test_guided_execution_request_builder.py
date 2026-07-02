from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import photometry_pipeline.guided_execution_request_builder as builder
from photometry_pipeline.guided_startup_transaction import (
    GuidedStartupTransactionRequest,
)
from tests.test_gui_guided_backend_validation_context import _failure_outcome
from tests.test_guided_execution_payloads import _request_auth
from tests.test_guided_startup_transaction import startup_request


@pytest.fixture
def validation_state():
    request = _request_auth()
    return request.current_validation_context, request.stored_validation_outcome


def _patch_success(monkeypatch, request):
    monkeypatch.setattr(
        builder,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(
            build_identity=request.application_build_identity
        ),
    )
    monkeypatch.setattr(
        builder,
        "build_guided_production_mapping_contract",
        lambda: object(),
    )
    monkeypatch.setattr(
        builder,
        "build_guided_run_authorization_request",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        builder,
        "authorize_guided_run",
        lambda _request: request.authorization_result,
    )
    monkeypatch.setattr(
        builder,
        "build_guided_execution_startup_mapping_contract",
        lambda: request.startup_mapping_contract,
    )
    monkeypatch.setattr(
        builder,
        "derive_guided_execution_payloads",
        lambda _authorization, **_kwargs: request.payload_result,
    )


def _build(monkeypatch, request, validation_state, **changes):
    _patch_success(monkeypatch, request)
    context, outcome = validation_state
    values = {
        "validation_context": context,
        "validation_outcome": outcome,
        "current_gui_revision": context.revision,
        "project_root": Path(__file__).resolve().parent.parent,
    }
    values.update(changes)
    return builder.build_guided_startup_request_from_validation(**values)


def test_accepted_current_validation_builds_real_bound_request(
    startup_request, validation_state, monkeypatch
):
    result = _build(monkeypatch, startup_request, validation_state)
    request = result.startup_transaction_request
    assert result.status == "built"
    assert result.ok and result.request_ready
    assert isinstance(result.authorization_result, type(startup_request.authorization_result))
    assert isinstance(result.payload_result, type(startup_request.payload_result))
    assert isinstance(request, GuidedStartupTransactionRequest)
    assert request.authorization_result is result.authorization_result
    assert request.payload_result is result.payload_result
    assert request.current_guided_revision == result.current_gui_revision
    assert request.explicit_user_run_transition is True
    assert result.blocking_issues == ()
    assert result.no_files_written and result.no_execution_invoked


def test_request_has_direct_unique_child_and_trusted_wrapper(
    startup_request, validation_state, monkeypatch
):
    result = _build(monkeypatch, startup_request, validation_state)
    request = result.startup_transaction_request
    assert request.planned_run_id
    assert Path(request.planned_allocated_run_dir).parent == Path(
        request.output_base_canonical
    )
    assert Path(request.planned_allocated_run_dir).name == request.planned_run_id
    assert not Path(request.planned_allocated_run_dir).exists()
    assert request.wrapper_entrypoint.trusted_entrypoint is True
    assert request.wrapper_entrypoint.supports_guided_preallocated_run_dir is True
    assert request.wrapper_entrypoint.supports_guided_candidate_manifest is True
    assert request.one_shot_token_current is True
    assert request.one_shot_token_unused is True
    assert request.filesystem_policy.planned_child_directly_under_base is True


def test_deterministic_factories_control_time_token_and_run_id(
    startup_request, validation_state, monkeypatch
):
    now = datetime(2026, 7, 2, 12, 30, tzinfo=timezone.utc)
    result = _build(
        monkeypatch,
        startup_request,
        validation_state,
        current_time_utc=now,
        token_factory=lambda: "fixed-one-shot-token",
        run_id_factory=lambda value: (
            "guided_run_fixed" if value == now else "wrong"
        ),
    )
    request = result.startup_transaction_request
    assert request.planned_run_id == "guided_run_fixed"
    assert request.one_shot_consumption_token == "fixed-one-shot-token"
    assert request.current_time_utc_iso == now.isoformat()


def test_context_revision_mismatch_refuses(
    startup_request, validation_state, monkeypatch
):
    context, _outcome = validation_state
    result = _build(
        monkeypatch,
        startup_request,
        validation_state,
        current_gui_revision=context.revision + 1,
    )
    assert result.status == "validation_not_current"
    assert not result.ok


@pytest.mark.parametrize(
    ("outcome", "status"),
    (
        (_failure_outcome("validator_refused"), "validation_not_accepted"),
        (
            replace(
                _request_auth().stored_validation_outcome,
                stale=True,
            ),
            "validation_not_current",
        ),
    ),
)
def test_refused_or_stale_validation_refuses(
    startup_request,
    validation_state,
    monkeypatch,
    outcome,
    status,
):
    result = _build(
        monkeypatch,
        startup_request,
        validation_state,
        validation_outcome=outcome,
    )
    assert result.status == status
    assert result.startup_transaction_request is None


def test_missing_build_identity_refuses(
    startup_request, validation_state, monkeypatch
):
    _patch_success(monkeypatch, startup_request)
    monkeypatch.setattr(
        builder,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(build_identity=None),
    )
    context, outcome = validation_state
    result = builder.build_guided_startup_request_from_validation(
        validation_context=context,
        validation_outcome=outcome,
        current_gui_revision=context.revision,
    )
    assert result.status == "build_identity_unavailable"


def test_authorization_refusal_is_classified(
    startup_request, validation_state, monkeypatch
):
    _patch_success(monkeypatch, startup_request)
    monkeypatch.setattr(builder, "authorize_guided_run", lambda _request: None)
    context, outcome = validation_state
    result = builder.build_guided_startup_request_from_validation(
        validation_context=context,
        validation_outcome=outcome,
        current_gui_revision=context.revision,
    )
    assert result.status == "authorization_failed"


def test_payload_failure_is_classified(
    startup_request, validation_state, monkeypatch
):
    _patch_success(monkeypatch, startup_request)
    monkeypatch.setattr(
        builder,
        "derive_guided_execution_payloads",
        lambda *_args, **_kwargs: None,
    )
    context, outcome = validation_state
    result = builder.build_guided_startup_request_from_validation(
        validation_context=context,
        validation_outcome=outcome,
        current_gui_revision=context.revision,
    )
    assert result.status == "payload_derivation_failed"


def test_builder_writes_no_files_or_directories(
    startup_request, validation_state, monkeypatch
):
    _patch_success(monkeypatch, startup_request)

    def fail(*_args, **_kwargs):
        raise AssertionError("write attempted")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    context, outcome = validation_state
    result = builder.build_guided_startup_request_from_validation(
        validation_context=context,
        validation_outcome=outcome,
        current_gui_revision=context.revision,
    )
    assert result.ok


def test_builder_import_boundary_and_no_execution_calls():
    source = Path(builder.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    prohibited = (
        "gui",
        "subprocess",
        "photometry_pipeline.pipeline",
        "photometry_pipeline.guided_startup_orchestration",
        "photometry_pipeline.guided_startup_allocation",
        "photometry_pipeline.guided_startup_materialization",
        "tools.run_full_pipeline_deliverables",
        "analyze_photometry",
    )
    assert not any(
        name == marker or name.startswith(f"{marker}.")
        for name in imported
        for marker in prohibited
    )
    prohibited_calls = (
        "execute_guided_backend_run",
        "run_guided_startup_to_wrapper",
        "allocate_guided_startup_directory",
        "materialize_guided_startup_artifacts",
        "subprocess.run",
        "Pipeline(",
    )
    assert not any(name in source for name in prohibited_calls)
