from __future__ import annotations

import ast
import builtins
from dataclasses import fields, replace
import json
import os
from pathlib import Path

import pytest
import yaml

import photometry_pipeline.guided_execution_payloads as payloads
import photometry_pipeline.guided_run_authorization as authorization
import photometry_pipeline.guided_startup_transaction as startup
from photometry_pipeline.config import Config
from photometry_pipeline.guided_manifest_verification import (
    load_guided_candidate_manifest,
)
from tests.test_guided_execution_payloads import (
    _accepted_candidate,
    _accepted_roi,
    _request_auth,
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


@pytest.fixture
def startup_request(monkeypatch):
    auth_request = _request_auth()
    monkeypatch.setattr(
        authorization.validation_workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: auth_request.stored_validation_outcome,
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
    auth = authorization.authorize_guided_run(auth_request)
    contract = payloads.build_guided_execution_startup_mapping_contract()
    derived = payloads.derive_guided_execution_payloads(
        auth, startup_mapping_contract=contract
    )
    return startup.GuidedStartupTransactionRequest(
        authorization_result=auth,
        payload_result=derived,
        startup_mapping_contract=contract,
        application_build_identity=auth.production_intent.application_build_identity,
        current_guided_revision=auth.authorized_gui_revision,
        explicit_user_run_transition=True,
        output_base_canonical=auth.production_intent.output_policy.output_base_canonical,
        source_root_canonical=auth.production_intent.input_source.source_root_canonical,
        planned_run_id="guided_run_20260101T000000Z_abcdef",
        planned_allocated_run_dir=(
            r"c:\output\guided_run_20260101T000000Z_abcdef"
        ),
        wrapper_entrypoint=startup.GuidedWrapperEntrypointIdentity(
            entrypoint_kind="script_path",
            entrypoint_value="tools/run_full_pipeline_deliverables.py",
            trusted_application_root=r"c:\application",
            wrapper_identity_digest="e" * 64,
            supported_contract_version="run_full_pipeline_deliverables.v1",
            supports_guided_preallocated_run_dir=True,
            supports_guided_candidate_manifest=True,
            trusted_entrypoint=True,
        ),
        one_shot_consumption_token="one-shot-token-0001",
        one_shot_token_current=True,
        one_shot_token_unused=True,
        current_time_utc_iso="2026-01-01T00:00:00Z",
        filesystem_policy=startup.GuidedStartupFilesystemPolicy(
            output_base_exists_or_creatable=True,
            output_base_is_directory_or_creatable=True,
            output_base_overlaps_source=False,
            output_base_is_completed_run_root=False,
            output_base_is_guided_diagnostic_cache_root=False,
            output_base_is_protected_ineligible_root=False,
            planned_child_directly_under_base=True,
            planned_child_already_exists=False,
            overwrite_requested=False,
            protected_root_context_complete=True,
        ),
    )


def test_valid_request_produces_pure_non_effectful_plan(startup_request):
    result = startup.plan_guided_startup_transaction(startup_request)
    assert result.status == "planned_non_effectful"
    assert result.ok is True
    assert result.ready_for_effectful_startup is True
    assert result.blocking_issues == ()
    assert result.identities is not None
    assert all(
        (
            result.no_files_written,
            result.no_directories_created,
            result.no_runner_invoked,
            result.no_gui_mutation,
        )
    )


def test_planning_preserves_nonrunnable_payload_semantics(startup_request):
    payload = startup_request.payload_result
    assert payload.status == payloads.GUIDED_EXECUTION_PAYLOAD_STATUS_NONRUNNABLE
    assert payload.runnable is False
    assert payload.runner_request is None
    assert [item.category for item in payload.limiting_issues] == [
        "startup_transaction_unavailable"
    ]
    assert startup.plan_guided_startup_transaction(startup_request).ok


@pytest.mark.parametrize(
    "change,category",
    (
        (
            lambda request: replace(
                request, explicit_user_run_transition=False
            ),
            "explicit_run_transition_required",
        ),
        (
            lambda request: replace(
                request, current_guided_revision=request.current_guided_revision + 1
            ),
            "guided_revision_stale",
        ),
        (
            lambda request: replace(request, one_shot_token_unused=False),
            "one_shot_token_unavailable",
        ),
        (
            lambda request: replace(
                request,
                payload_result=_unchecked(
                    request.payload_result, status="runnable_payloads_derived"
                ),
            ),
            "payload_status_unsupported",
        ),
        (
            lambda request: replace(
                request,
                payload_result=_unchecked(
                    request.payload_result, runner_request=object()
                ),
            ),
            "payload_runner_request_prohibited",
        ),
        (
            lambda request: replace(
                request,
                payload_result=_unchecked(
                    request.payload_result,
                    limiting_issues=(
                        payloads.GuidedExecutionPayloadIssue(
                            "different_limiter", "payload", "Different."
                        ),
                    ),
                ),
            ),
            "payload_limiter_unsupported",
        ),
    ),
)
def test_authority_and_payload_refusals(startup_request, change, category):
    result = startup.plan_guided_startup_transaction(change(startup_request))
    assert result.status == "refused"
    assert result.ready_for_effectful_startup is False
    assert result.blocking_issues[0].category == category
    assert result.startup_status_bytes is None
    assert result.planned_command_argv == ()


@pytest.mark.parametrize(
    "field,category",
    (
        ("output_base_overlaps_source", "output_source_overlap"),
        ("planned_child_already_exists", "planned_child_exists"),
        ("overwrite_requested", "overwrite_prohibited"),
        ("output_base_is_completed_run_root", "completed_run_root_prohibited"),
        (
            "output_base_is_guided_diagnostic_cache_root",
            "diagnostic_cache_root_prohibited",
        ),
    ),
)
def test_output_policy_refusals(startup_request, field, category):
    policy = replace(startup_request.filesystem_policy, **{field: True})
    result = startup.plan_guided_startup_transaction(
        replace(startup_request, filesystem_policy=policy)
    )
    assert result.blocking_issues[0].category == category
    assert result.no_files_written is True
    assert result.no_directories_created is True


def test_refuses_wrapper_without_preallocated_support(startup_request):
    wrapper = replace(
        startup_request.wrapper_entrypoint,
        supports_guided_preallocated_run_dir=False,
    )
    result = startup.plan_guided_startup_transaction(
        replace(startup_request, wrapper_entrypoint=wrapper)
    )
    assert result.blocking_issues[0].category == "wrapper_contract_unsupported"


def test_manifest_bytes_are_deterministic_and_loader_compatible(
    startup_request, tmp_path
):
    payload = startup_request.payload_result.candidate_manifest_payload
    first = startup.serialize_guided_candidate_manifest_payload_to_bytes(payload)
    second = startup.serialize_guided_candidate_manifest_payload_to_bytes(payload)
    assert first == second
    path = tmp_path / "manifest.json"
    path.write_bytes(first.content_bytes)
    loaded = load_guided_candidate_manifest(path)
    assert loaded.accepted
    assert loaded.manifest.canonical_candidate_manifest_payload_identity == (
        first.semantic_identity
    )


def test_config_bytes_are_deterministic_and_round_trip(startup_request, tmp_path):
    payload = startup_request.payload_result.config_payload
    first = startup.serialize_guided_config_payload_to_yaml_bytes(payload)
    second = startup.serialize_guided_config_payload_to_yaml_bytes(payload)
    assert first == second
    path = tmp_path / "config.yaml"
    path.write_bytes(first.content_bytes)
    loaded = Config.from_yaml(str(path))
    expected = {item.name: item.value for item in payload.values}
    serialized = yaml.safe_load(first.content_bytes)
    assert set(serialized) == set(expected)
    for name, value in expected.items():
        normalized = list(value) if isinstance(value, tuple) else value
        assert getattr(loaded, name) == normalized


def test_status_and_provenance_never_claim_completion(startup_request):
    result = startup.plan_guided_startup_transaction(startup_request)
    status = json.loads(result.startup_status_bytes)
    provenance = json.loads(result.startup_provenance_bytes)
    assert status["status"] == "allocated_preparation_pending"
    assert status["completed_run_claim"] is False
    assert status["runner_started"] is False
    assert provenance["state"] == "prepared_runner_not_started"
    assert provenance["completed_run_claim"] is False
    assert provenance["runner_started"] is False


def test_command_plan_is_future_internal_phasic_full_handoff(startup_request):
    result = startup.plan_guided_startup_transaction(startup_request)
    command = result.command_plan
    assert command.executable_now is False
    assert command.requires_future_wrapper_preallocated_mode is True
    assert "--guided-candidate-manifest" in command.argv
    assert "--guided-preallocated-run-dir" in command.argv
    assert command.argv[command.argv.index("--mode") + 1] == "phasic"
    assert command.argv[command.argv.index("--run-type") + 1] == "full"
    assert "tonic" not in command.argv
    assert "both" not in command.argv


def test_guided_preallocated_run_dir_is_a_bare_boolean_flag(startup_request):
    """4J16k19: --guided-preallocated-run-dir is action='store_true' in
    tools/run_full_pipeline_deliverables.py's argparse definition -- it
    takes no value. The actual run directory is passed separately via
    --out. This locks that contract explicitly: the flag is the last argv
    element with nothing following it, which is correct, not malformed."""
    result = startup.plan_guided_startup_transaction(startup_request)
    argv = result.command_plan.argv
    assert argv[-1] == "--guided-preallocated-run-dir"
    assert argv.count("--guided-preallocated-run-dir") == 1
    out_index = argv.index("--out")
    assert argv[out_index + 1] == startup_request.planned_allocated_run_dir


def test_command_plan_sessions_per_hour_sourced_from_production_intent(
    startup_request,
):
    """4J16k12: the wrapper's own --sessions-per-hour must be threaded from
    the already-validated Guided production intent
    (intent.acquisition.sessions_per_hour), not omitted, hardcoded, or
    inferred from fixture-specific state. Without this, the real backend
    wrapper crashes post-analysis with "Cannot infer session stride
    (duty-cycled acquisition) without sessions_per_hour" for every
    intermittent Guided run, since neither the wrapper's own CLI default
    nor its config-YAML fallback (Config has no sessions_per_hour field)
    can supply it."""
    intent_sessions_per_hour = (
        startup_request.authorization_result.production_intent.acquisition.sessions_per_hour
    )
    assert intent_sessions_per_hour == 6

    command = startup.build_guided_startup_command_plan(startup_request)
    assert "--sessions-per-hour" in command.argv
    assert command.argv[command.argv.index("--sessions-per-hour") + 1] == "6"

    # Prove the value tracks the intent rather than being hardcoded: a
    # request whose intent carries a different sessions_per_hour must
    # produce a correspondingly different argv value.
    changed_auth = replace(
        startup_request.authorization_result,
        production_intent=replace(
            startup_request.authorization_result.production_intent,
            acquisition=replace(
                startup_request.authorization_result.production_intent.acquisition,
                sessions_per_hour=20,
            ),
        ),
    )
    changed_request = replace(
        startup_request, authorization_result=changed_auth
    )
    changed_command = startup.build_guided_startup_command_plan(changed_request)
    assert (
        changed_command.argv[changed_command.argv.index("--sessions-per-hour") + 1]
        == "20"
    )


def test_command_identity_changes_with_manifest_path(startup_request):
    first = startup.build_guided_startup_command_plan(startup_request)
    second = startup.build_guided_startup_command_plan(
        startup_request, manifest_path=r"c:\different\manifest.json"
    )
    assert first.canonical_command_identity != second.canonical_command_identity


def test_startup_identity_changes_with_run_id_or_config_hash():
    base = {
        "run_id": "run-a",
        "serialized_config_sha256": "a" * 64,
    }
    identity = startup.compute_guided_startup_transaction_identity(base)
    assert identity != startup.compute_guided_startup_transaction_identity(
        {**base, "run_id": "run-b"}
    )
    assert identity != startup.compute_guided_startup_transaction_identity(
        {**base, "serialized_config_sha256": "b" * 64}
    )


def test_planning_calls_no_write_allocation_or_launch_api(
    startup_request, monkeypatch
):
    def fail(*_args, **_kwargs):
        raise AssertionError("effectful API is prohibited")

    monkeypatch.setattr(builtins, "open", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    result = startup.plan_guided_startup_transaction(startup_request)
    assert result.ok


def test_module_import_boundary():
    source = Path(startup.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    prohibited = (
        "gui",
        "subprocess",
        "photometry_pipeline.pipeline",
        "tools.run_full_pipeline_deliverables",
        "analyze_photometry",
    )
    assert not any(
        name == marker or name.startswith(f"{marker}.")
        for name in imported
        for marker in prohibited
    )
