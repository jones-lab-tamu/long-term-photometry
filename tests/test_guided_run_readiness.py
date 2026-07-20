from __future__ import annotations

import ast
from dataclasses import fields
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import photometry_pipeline.guided_backend_execution as backend_execution
import photometry_pipeline.guided_run_readiness as readiness
import photometry_pipeline.guided_startup_orchestration as orchestration
from photometry_pipeline.guided_npm_startup_bridge import GuidedStartupAuthority
from tests.test_guided_execution_payloads import _accepted_outcome
from tests.test_guided_startup_transaction import startup_request


def _unchecked(instance, **changes):
    result = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            result,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return result


@pytest.fixture
def ready_state(startup_request):
    return {
        "validation_outcome": _accepted_outcome(),
        "validation_revision": startup_request.current_guided_revision,
        "current_gui_revision": startup_request.current_guided_revision,
        "startup_authority": startup_request.startup_authority,
        "payload_result": startup_request.payload_result,
        "backend_execution_available": True,
        "startup_orchestration_available": True,
    }


def _evaluate(state, **changes):
    return readiness.evaluate_guided_run_readiness(**{**state, **changes})


def test_no_validation_requires_validation(ready_state):
    result = _evaluate(ready_state, validation_outcome=None)
    assert result.status == "no_validation"
    assert result.ready is False
    assert result.user_visible_state == "needs_validation"


def test_validation_not_accepted_refuses(ready_state):
    outcome = _unchecked(
        ready_state["validation_outcome"],
        status="validator_refused",
        accepted_for_backend_validation=False,
    )
    result = _evaluate(ready_state, validation_outcome=outcome)
    assert result.status == "validation_not_accepted"
    assert not result.ready


def test_validation_revision_mismatch_is_stale(ready_state):
    result = _evaluate(
        ready_state,
        validation_revision=ready_state["current_gui_revision"] - 1,
    )
    assert result.status == "validation_stale"
    assert result.user_visible_state == "needs_revalidation"


def test_validation_outcome_marked_stale_requires_revalidation(ready_state):
    outcome = _unchecked(ready_state["validation_outcome"], stale=True)
    result = _evaluate(ready_state, validation_outcome=outcome)
    assert result.status == "validation_stale"
    assert result.user_visible_state == "needs_revalidation"


def test_missing_authorization_refuses(ready_state):
    result = _evaluate(ready_state, startup_authority=None)
    assert result.status == "authorization_missing"
    assert not result.ready
    assert result.user_visible_state == "cannot_run"
    assert result.user_summary == (
        "Guided validation succeeded, but Guided Run execution is unavailable "
        "in this build."
    )
    assert "Validate the setup first" not in result.user_summary


def test_unauthorized_authorization_refuses(ready_state):
    authorization = _unchecked(
        ready_state["startup_authority"].rwd,
        status="refused",
        authorized=False,
        run_authorization=False,
    )
    result = _evaluate(
        ready_state, startup_authority=GuidedStartupAuthority(rwd=authorization)
    )
    assert result.status == "authorization_not_accepted"


def test_authorization_revision_mismatch_is_stale(ready_state):
    authorization = _unchecked(
        ready_state["startup_authority"].rwd,
        authorized_gui_revision=ready_state["current_gui_revision"] - 1,
    )
    result = _evaluate(
        ready_state, startup_authority=GuidedStartupAuthority(rwd=authorization)
    )
    assert result.status == "authorization_stale"


def test_missing_payload_refuses(ready_state):
    result = _evaluate(ready_state, payload_result=None)
    assert result.status == "payload_missing"


@pytest.mark.parametrize(
    "changes",
    (
        {"ok": False},
        {"status": "refused"},
        {"runnable": True},
        {"runner_request": object()},
    ),
)
def test_invalid_payload_states_are_not_ready(ready_state, changes):
    payload = _unchecked(ready_state["payload_result"], **changes)
    result = _evaluate(ready_state, payload_result=payload)
    assert result.status == "payload_not_ready"
    assert result.ready is False


def test_unexpected_payload_limiter_is_not_ready(ready_state):
    issue = _unchecked(
        ready_state["payload_result"].limiting_issues[0],
        category="different_limiter",
    )
    payload = _unchecked(ready_state["payload_result"], limiting_issues=(issue,))
    result = _evaluate(ready_state, payload_result=payload)
    assert result.status == "payload_not_ready"


@pytest.mark.parametrize(
    "changes",
    (
        {"backend_execution_available": False},
        {"startup_orchestration_available": False},
    ),
)
def test_backend_or_orchestration_unavailable(ready_state, changes):
    result = _evaluate(ready_state, **changes)
    assert result.status == "backend_unavailable"
    assert result.user_visible_state == "cannot_run"


def test_fully_accepted_state_is_ready_but_hidden(ready_state):
    result = _evaluate(ready_state)
    assert result.status == "ready_hidden"
    assert result.ready is True
    assert result.user_visible_state == "ready_for_future_run_hidden"
    assert result.blocking_issues == ()
    assert result.authorization_identity
    assert result.payload_status == "payloads_derived_nonrunnable"
    assert result.execution_would_be_backend_only is True
    assert result.visible_run_control_enabled is False
    assert result.visible_run_control_present is False
    assert result.execution_invoked is False
    assert result.files_written is False
    assert result.exposes_manifest_path_to_user is False
    assert result.exposes_internal_cli_to_user is False


def _npm_ready_state(intent, npm_payload):
    """Plain, non-fixture-dependent readiness kwargs for an NPM authority.
    Deliberately independent of the `ready_state`/`startup_request` RWD
    fixtures: combining a real RWD authorization build with a fresh real
    NPM validation in the same test triggers a pre-existing, unrelated
    test-isolation hazard (a stale/mismatched materialization result),
    reproduced and confirmed unrelated to this repair -- not something to
    paper over by depending on it here."""
    return {
        "validation_outcome": _accepted_outcome(),
        "backend_execution_available": True,
        "startup_orchestration_available": True,
    }


def test_accepted_npm_validation_is_ready_when_authority_and_payload_are_accepted(
    tmp_path,
):
    """Repair regression: a validated, accepted Guided NPM setup is no
    longer refused as "not available yet" -- readiness truthfully reports
    ready_hidden using NPM's own accepted authority, matching RWD's
    existing behavior."""
    from photometry_pipeline.guided_execution_payloads import (
        build_guided_execution_startup_mapping_contract,
    )
    from photometry_pipeline.guided_npm_startup_bridge import (
        compile_npm_generic_execution_payloads,
    )
    from tests.test_guided_npm_startup_bridge import _npm_pair

    intent, authority = _npm_pair(tmp_path)
    assert intent.execution_mode == "both"
    contract = build_guided_execution_startup_mapping_contract()
    npm_payload = compile_npm_generic_execution_payloads(
        intent, authority, startup_mapping_contract=contract
    )
    assert npm_payload.ok is True

    result = _evaluate(
        _npm_ready_state(intent, npm_payload),
        validation_revision=intent.validation_revision,
        current_gui_revision=intent.validation_revision,
        startup_authority=GuidedStartupAuthority(
            npm_intent=intent, npm_authority=authority
        ),
        payload_result=npm_payload,
    )

    assert result.status == "ready_hidden"
    assert result.ready is True
    assert result.user_visible_state == "ready_for_future_run_hidden"
    assert result.authorization_identity == authority.canonical_authority_identity


def test_npm_execution_mode_other_than_both_is_not_ready(tmp_path):
    """Guided Mode exposes no phasic-versus-tonic choice: a genuinely
    accepted NPM authority whose intent carries phasic/tonic (a valid
    backend state, just not a Guided one) must not read as ready."""
    from photometry_pipeline.guided_execution_payloads import (
        build_guided_execution_startup_mapping_contract,
    )
    from photometry_pipeline.guided_npm_startup_bridge import (
        compile_npm_generic_execution_payloads,
    )
    from tests.test_guided_npm_startup_bridge import _npm_pair

    intent, authority = _npm_pair(tmp_path, execution_mode="phasic")
    contract = build_guided_execution_startup_mapping_contract()
    npm_payload = compile_npm_generic_execution_payloads(
        intent, authority, startup_mapping_contract=contract
    )
    assert npm_payload.ok is False

    result = _evaluate(
        _npm_ready_state(intent, npm_payload),
        validation_revision=intent.validation_revision,
        current_gui_revision=intent.validation_revision,
        startup_authority=GuidedStartupAuthority(
            npm_intent=intent, npm_authority=authority
        ),
        payload_result=npm_payload,
    )

    assert result.status in ("payload_missing", "payload_not_ready")
    assert result.ready is False


def test_user_summaries_exclude_internal_terms():
    prohibited = (
        "manifest",
        "preallocated",
        "command_invoked",
        "wrapper claim",
        "startup transaction",
        "hash",
        "--guided",
        "config_effective.yaml",
        "runner_request",
        "startup_transaction_unavailable",
    )
    for summary in readiness._SUMMARIES.values():
        lowered = summary.lower()
        assert not any(term in lowered for term in prohibited)


def test_evaluation_calls_no_execution_path(ready_state, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("execution path must not be called")

    monkeypatch.setattr(backend_execution, "execute_guided_backend_run", fail)
    monkeypatch.setattr(orchestration, "run_guided_startup_to_wrapper", fail)
    assert _evaluate(ready_state).ready


def test_evaluation_writes_no_files_or_directories(ready_state, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("filesystem write must not occur")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    assert _evaluate(ready_state).ready


def test_readiness_import_boundary_and_no_visible_control():
    source = Path(readiness.__file__).read_text(encoding="utf-8")
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
        "tools.run_full_pipeline_deliverables",
        "photometry_pipeline.guided_backend_execution",
        "photometry_pipeline.guided_startup_orchestration",
        "photometry_pipeline.guided_startup_allocation",
        "photometry_pipeline.guided_startup_materialization",
    )
    assert not any(
        name == marker or name.startswith(f"{marker}.")
        for name in imported
        for marker in prohibited
    )
    assert "execute_guided_backend_run" not in source
    assert "run_guided_startup_to_wrapper" not in source
