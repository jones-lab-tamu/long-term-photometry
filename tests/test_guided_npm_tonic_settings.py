"""NPM-specific focused tests for the Guided tonic-settings repair.

Guided NPM shares the same validation/materialization/compilation pipeline as
RWD (GuidedNewAnalysisDraftPlan -> validate_current_guided_draft_for_backend
-> map_guided_npm_validation_outcome_to_execution_intent ->
build_guided_npm_execution_authority -> compile_npm_generic_execution_payloads),
so these tests exercise the real, production-shaped NPM path end to end using
a small synthetic NPM fixture -- never the large real NPM dataset.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from photometry_pipeline.guided_backend_validation_workflow import (
    validate_current_guided_draft_for_backend,
)
from photometry_pipeline.guided_execution_payloads import (
    build_guided_execution_startup_mapping_contract,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisExecutionIntent,
    GuidedNewAnalysisTonicSettingsContract,
)
from photometry_pipeline.guided_npm_execution_authority import (
    build_guided_npm_execution_authority,
)
from photometry_pipeline.guided_npm_startup_bridge import (
    GuidedStartupAuthority,
    compile_npm_generic_execution_payloads,
)
from photometry_pipeline.run_completion_contract import (
    tonic_settings_completion_error,
)

from tests.test_guided_backend_validation_materialization import (
    _valid_npm_stage2c_draft,
)
from tests.test_guided_npm_production_mapping import _map
from tests.test_guided_npm_startup_bridge import _npm_validation_fixture_kwargs
from tests.test_guided_tonic_settings import _write_guided_marker, _write_yaml


def _accepted_npm_with_tonic(
    tmp_path: Path,
    *,
    tonic_output_mode: str,
    tonic_timeline_mode: str,
):
    """Same real materialization path as _accepted_npm_with_mode, but with
    an explicit shared tonic-settings selection on the draft plan."""
    draft = _valid_npm_stage2c_draft(tmp_path)
    draft.execution_intent = GuidedNewAnalysisExecutionIntent(execution_mode="both")
    draft.tonic_settings_contract = GuidedNewAnalysisTonicSettingsContract(
        tonic_output_mode=tonic_output_mode,
        tonic_timeline_mode=tonic_timeline_mode,
    )
    kwargs = _npm_validation_fixture_kwargs()
    outcome = validate_current_guided_draft_for_backend(
        draft, validation_revision=4, **kwargs
    )
    return outcome


def _npm_pair_with_tonic(tmp_path: Path, *, tonic_output_mode, tonic_timeline_mode):
    outcome = _accepted_npm_with_tonic(
        tmp_path,
        tonic_output_mode=tonic_output_mode,
        tonic_timeline_mode=tonic_timeline_mode,
    )
    assert outcome.status == "validator_accepted", outcome.blocking_issues
    mapped = _map(outcome)
    intent = mapped.intent
    authority = build_guided_npm_execution_authority(intent)
    return intent, authority


def _compile(intent, authority):
    contract = build_guided_execution_startup_mapping_contract()
    return compile_npm_generic_execution_payloads(
        intent, authority, startup_mapping_contract=contract
    )


# ---------------------------------------------------------------------------
# 1/2/3. Generated-config evidence: default, non-default output mode, and
# gap-free timeline (permitted -- no missing sessions in this fixture).
# ---------------------------------------------------------------------------


def test_npm_default_tonic_settings_reach_generated_config(tmp_path):
    intent, authority = _npm_pair_with_tonic(
        tmp_path,
        tonic_output_mode="preserve_raw_session_shape",
        tonic_timeline_mode="real_elapsed_time",
    )
    result = _compile(intent, authority)
    assert result.ok is True
    emitted = {item.name: item.value for item in result.config_payload.values}
    assert emitted["tonic_output_mode"] == "preserve_raw_session_shape"
    assert emitted["tonic_timeline_mode"] == "real_elapsed_time"
    assert emitted["lowpass_hz"] == 1.0


def test_npm_non_default_output_mode_reaches_generated_config(tmp_path):
    intent, authority = _npm_pair_with_tonic(
        tmp_path,
        tonic_output_mode="flatten_session_bleach_preserve_session_baseline",
        tonic_timeline_mode="real_elapsed_time",
    )
    result = _compile(intent, authority)
    assert result.ok is True
    emitted = {item.name: item.value for item in result.config_payload.values}
    assert emitted["tonic_output_mode"] == (
        "flatten_session_bleach_preserve_session_baseline"
    )


def test_npm_gap_free_timeline_reaches_generated_config_when_permitted(tmp_path):
    intent, authority = _npm_pair_with_tonic(
        tmp_path,
        tonic_output_mode="preserve_raw_session_shape",
        tonic_timeline_mode="gap_free_elapsed_time",
    )
    result = _compile(intent, authority)
    assert result.ok is True
    emitted = {item.name: item.value for item in result.config_payload.values}
    assert emitted["tonic_timeline_mode"] == "gap_free_elapsed_time"


def test_npm_unsupported_tonic_output_mode_is_refused_before_run(tmp_path):
    with pytest.raises(ValueError):
        _accepted_npm_with_tonic(
            tmp_path,
            tonic_output_mode="not_a_real_mode",
            tonic_timeline_mode="real_elapsed_time",
        )


# ---------------------------------------------------------------------------
# 4. NPM startup gate verifies exact parity with the accepted plan.
#
# These drive the real production gate (guided_startup_transaction.
# plan_guided_startup_transaction, which calls _gate_issue) with a complete,
# otherwise-valid GuidedStartupTransactionRequest -- not an independent
# reimplementation of the parity comparison. Tampering replaces one config
# value AND recomputes every identity that must still agree with it (config
# payload identity, provenance seed identity), so the ONLY thing that can
# reject the request is the gate's own
# config.get("tonic_*") == npm_correction_values.get("tonic_*") comparison --
# not a generic identity-mismatch failure that would fire regardless of
# whether that comparison exists.
# ---------------------------------------------------------------------------


def _npm_startup_request(intent, authority, payload_result):
    from photometry_pipeline import guided_startup_transaction as startup
    from photometry_pipeline.guided_execution_payloads import (
        build_guided_execution_startup_mapping_contract,
    )

    return startup.GuidedStartupTransactionRequest(
        startup_authority=GuidedStartupAuthority(
            npm_intent=intent, npm_authority=authority
        ),
        payload_result=payload_result,
        startup_mapping_contract=build_guided_execution_startup_mapping_contract(),
        application_build_identity=intent.application_build_identity,
        current_guided_revision=intent.validation_revision,
        explicit_user_run_transition=True,
        output_base_canonical=intent.output_policy.output_base_canonical,
        source_root_canonical=intent.source_root_canonical,
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


def _tampered_payload_result(result, *, field_name, tampered_value):
    """Return a copy of `result` with exactly one config field value changed
    and every dependent identity (config payload, provenance seed)
    recomputed to stay internally self-consistent -- so only the gate's own
    tonic-parity comparison, not a generic identity mismatch, can reject it.
    """
    from dataclasses import replace as dc_replace
    from photometry_pipeline.guided_execution_payloads import (
        GuidedConfigFieldValue,
        compute_guided_execution_config_payload_identity,
        compute_guided_startup_provenance_seed_identity,
        replace_config_payload_identity,
        replace_provenance_seed_identity,
    )

    tampered_values = tuple(
        GuidedConfigFieldValue(field_name, tampered_value)
        if item.name == field_name
        else item
        for item in result.config_payload.values
    )
    provisional_config = dc_replace(result.config_payload, values=tampered_values)
    new_config_identity = compute_guided_execution_config_payload_identity(
        provisional_config
    )
    tampered_config_payload = replace_config_payload_identity(
        provisional_config, new_config_identity
    )

    provisional_seed = dc_replace(
        result.provenance_seed, config_payload_identity=new_config_identity
    )
    new_seed_identity = compute_guided_startup_provenance_seed_identity(
        provisional_seed
    )
    tampered_seed = replace_provenance_seed_identity(
        provisional_seed, new_seed_identity
    )

    return dc_replace(
        result,
        config_payload=tampered_config_payload,
        config_payload_identity=new_config_identity,
        provenance_seed=tampered_seed,
        provenance_seed_identity=new_seed_identity,
    )


def test_npm_startup_gate_accepts_matching_tonic_config(tmp_path):
    from photometry_pipeline import guided_startup_transaction as startup

    intent, authority = _npm_pair_with_tonic(
        tmp_path,
        tonic_output_mode="flatten_session_bleach_preserve_session_baseline",
        tonic_timeline_mode="gap_free_elapsed_time",
    )
    result = _compile(intent, authority)
    request = _npm_startup_request(intent, authority, result)

    plan = startup.plan_guided_startup_transaction(request)

    assert plan.status == "planned_non_effectful"
    assert plan.ok is True
    assert plan.ready_for_effectful_startup is True
    assert plan.blocking_issues == ()
    assert plan.identities is not None


def test_npm_startup_gate_rejects_tonic_output_mode_mismatch(tmp_path):
    from photometry_pipeline import guided_startup_transaction as startup

    intent, authority = _npm_pair_with_tonic(
        tmp_path,
        tonic_output_mode="preserve_raw_session_shape",
        tonic_timeline_mode="real_elapsed_time",
    )
    result = _compile(intent, authority)
    tampered_result = _tampered_payload_result(
        result,
        field_name="tonic_output_mode",
        tampered_value="flatten_session_bleach_preserve_session_baseline",
    )
    request = _npm_startup_request(intent, authority, tampered_result)

    plan = startup.plan_guided_startup_transaction(request)

    assert plan.status == "refused"
    assert plan.ok is False
    assert plan.ready_for_effectful_startup is False
    assert len(plan.blocking_issues) == 1
    assert plan.blocking_issues[0].category == "first_subset_contract_unsupported"
    # No run allocation or execution proceeded.
    assert plan.identities is None
    assert plan.command_plan is None
    assert plan.planned_command_argv == ()
    assert plan.startup_status_bytes is None
    assert plan.candidate_manifest_bytes is None
    assert plan.config_effective_bytes is None
    assert plan.no_files_written is True
    assert plan.no_directories_created is True
    assert plan.no_runner_invoked is True


def test_npm_startup_gate_rejects_tonic_timeline_mode_mismatch(tmp_path):
    from photometry_pipeline import guided_startup_transaction as startup

    intent, authority = _npm_pair_with_tonic(
        tmp_path,
        tonic_output_mode="preserve_raw_session_shape",
        tonic_timeline_mode="real_elapsed_time",
    )
    result = _compile(intent, authority)
    tampered_result = _tampered_payload_result(
        result,
        field_name="tonic_timeline_mode",
        tampered_value="gap_free_elapsed_time",
    )
    request = _npm_startup_request(intent, authority, tampered_result)

    plan = startup.plan_guided_startup_transaction(request)

    assert plan.status == "refused"
    assert plan.ok is False
    assert plan.ready_for_effectful_startup is False
    assert len(plan.blocking_issues) == 1
    assert plan.blocking_issues[0].category == "first_subset_contract_unsupported"
    assert plan.identities is None
    assert plan.command_plan is None
    assert plan.no_runner_invoked is True


# ---------------------------------------------------------------------------
# 6/7. Requested-versus-consumed verification (format-agnostic function,
# exercised here with NPM-shaped evidence).
# ---------------------------------------------------------------------------


def test_npm_requested_versus_consumed_match_is_not_fatal(tmp_path):
    run_dir = str(tmp_path)
    _write_guided_marker(run_dir)
    _write_yaml(
        f"{run_dir}/config_effective.yaml",
        {
            "tonic_output_mode": "flatten_session_bleach_preserve_session_baseline",
            "tonic_timeline_mode": "gap_free_elapsed_time",
        },
    )
    _write_yaml(
        f"{run_dir}/_analysis/tonic_out/config_used.yaml",
        {
            "tonic_output_mode": "flatten_session_bleach_preserve_session_baseline",
            "tonic_timeline_mode": "gap_free_elapsed_time",
        },
    )
    error = tonic_settings_completion_error(
        run_dir, {"tonic_analysis": True, "phasic_analysis": False}
    )
    assert error == ""


def test_npm_requested_versus_consumed_mismatch_is_fatal(tmp_path):
    run_dir = str(tmp_path)
    _write_guided_marker(run_dir)
    _write_yaml(
        f"{run_dir}/config_effective.yaml",
        {
            "tonic_output_mode": "flatten_session_bleach_preserve_session_baseline",
            "tonic_timeline_mode": "gap_free_elapsed_time",
        },
    )
    _write_yaml(
        f"{run_dir}/_analysis/tonic_out/config_used.yaml",
        {
            "tonic_output_mode": "preserve_raw_session_shape",
            "tonic_timeline_mode": "gap_free_elapsed_time",
        },
    )
    error = tonic_settings_completion_error(
        run_dir, {"tonic_analysis": True, "phasic_analysis": False}
    )
    assert error != ""


# ---------------------------------------------------------------------------
# 8. No NPM-specific numerical behavior: the tonic core functions take no
# format/adapter parameter at all, so no NPM-specific branch can exist.
# ---------------------------------------------------------------------------


def test_tonic_numerical_functions_take_no_format_parameter():
    import inspect
    from photometry_pipeline.core.tonic_output import (
        apply_tonic_output_mode_to_session,
    )
    from photometry_pipeline.core.tonic_timeline import (
        build_tonic_chunk_time_axis,
    )

    for func in (apply_tonic_output_mode_to_session, build_tonic_chunk_time_axis):
        params = set(inspect.signature(func).parameters)
        assert "format" not in params
        assert "adapter_format" not in params
        assert "source_format" not in params
        assert "is_npm" not in params
