from __future__ import annotations

from pathlib import Path

import pytest

from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    validate_current_guided_draft_for_backend,
)
from photometry_pipeline.guided_backend_validator import (
    GuidedBackendValidatorContract,
)
from photometry_pipeline.guided_npm_run_launch_builder import (
    GuidedNpmRunLaunchBuildResult,
    build_guided_npm_worker_prelaunch_claim_from_validation,
)
from photometry_pipeline.guided_npm_worker_prelaunch_claim import (
    GuidedNpmWorkerPrelaunchClaim,
)
from photometry_pipeline.io.npm_contract import NpmParserContract

from tests.test_guided_backend_validation_materialization import (
    _valid_npm_stage2c_draft,
)
from tests.test_guided_npm_production_mapping import _unsafe_replace


_VALIDATOR_CONTRACT = GuidedBackendValidatorContract(
    validation_scope="guided_rwd_intermittent_phasic_full_validate",
    validation_contract_version="guided_backend_validation_contract.v1",
    validator_capability_version="test_validator_capability.v1",
    supported_subset_rule_version="global_dynamic_fit_only.v1",
)

_PARSER_CONTRACT = NpmParserContract(
    npm_time_axis="system_timestamp",
    npm_system_ts_col="SystemTimestamp",
    npm_computer_ts_col="ComputerTimestamp",
    npm_led_col="LedState",
    npm_region_prefix="Region",
    npm_region_suffix="G",
    target_fs_hz=2.0,
    session_duration_sec=2.0,
    allow_partial_final_chunk=False,
    adapter_value_nan_policy="strict",
)


def _accepted_context_and_outcome(tmp_path: Path, *, revision: int = 4):
    draft = _valid_npm_stage2c_draft(tmp_path)
    outcome = validate_current_guided_draft_for_backend(
        draft,
        parser_contract=_PARSER_CONTRACT,
        validator_contract=_VALIDATOR_CONTRACT,
        validation_revision=revision,
    )
    assert outcome.status == "validator_accepted"
    assert outcome.stale is False
    context = GuidedBackendValidationGuiContext(
        draft=draft,
        parser_contract=_PARSER_CONTRACT,
        additional_protected_roots=(),
        validator_contract=_VALIDATOR_CONTRACT,
        revision=revision,
    )
    return context, outcome


def test_builder_produces_valid_prelaunch_claim(tmp_path):
    context, outcome = _accepted_context_and_outcome(tmp_path)
    result = build_guided_npm_worker_prelaunch_claim_from_validation(
        validation_context=context,
        validation_outcome=outcome,
        current_gui_revision=4,
        project_root=Path(__file__).resolve().parent.parent,
    )
    assert isinstance(result, GuidedNpmRunLaunchBuildResult)
    assert result.ok is True
    assert result.status == "built"
    assert isinstance(result.prelaunch_claim, GuidedNpmWorkerPrelaunchClaim)
    assert result.blocking_issues == ()
    assert result.application_build_identity is not None
    assert (
        result.application_build_identity
        == result.prelaunch_claim.application_build_identity
    )


def test_builder_refuses_stale_validation(tmp_path):
    context, outcome = _accepted_context_and_outcome(tmp_path)
    result = build_guided_npm_worker_prelaunch_claim_from_validation(
        validation_context=context,
        validation_outcome=outcome,
        current_gui_revision=5,
        project_root=Path(__file__).resolve().parent.parent,
    )
    assert result.ok is False
    assert result.status == "validation_not_current"
    assert result.prelaunch_claim is None


def test_builder_refuses_non_npm_draft(tmp_path):
    from tests.test_guided_backend_validation_materialization import (
        _valid_stage2c_draft,
        _valid_parser_contract,
    )

    draft = _valid_stage2c_draft(tmp_path)
    rwd_parser_contract = _valid_parser_contract()
    outcome = validate_current_guided_draft_for_backend(
        draft,
        parser_contract=rwd_parser_contract,
        validator_contract=_VALIDATOR_CONTRACT,
        validation_revision=4,
    )
    assert outcome.status == "validator_accepted"
    context = GuidedBackendValidationGuiContext(
        draft=draft,
        parser_contract=rwd_parser_contract,
        additional_protected_roots=(),
        validator_contract=_VALIDATOR_CONTRACT,
        revision=4,
    )
    result = build_guided_npm_worker_prelaunch_claim_from_validation(
        validation_context=context,
        validation_outcome=outcome,
        current_gui_revision=4,
        project_root=Path(__file__).resolve().parent.parent,
    )
    assert result.ok is False
    assert result.status == "unsupported_format"


def test_builder_refuses_unaccepted_validation(tmp_path):
    draft = _valid_npm_stage2c_draft(tmp_path)
    outcome = validate_current_guided_draft_for_backend(
        draft,
        parser_contract=_PARSER_CONTRACT,
        validator_contract=_VALIDATOR_CONTRACT,
        validation_revision=4,
    )
    refused = _unsafe_replace(
        outcome,
        status="validator_refused",
        accepted_for_backend_validation=False,
    )
    context = GuidedBackendValidationGuiContext(
        draft=draft,
        parser_contract=_PARSER_CONTRACT,
        additional_protected_roots=(),
        validator_contract=_VALIDATOR_CONTRACT,
        revision=4,
    )
    result = build_guided_npm_worker_prelaunch_claim_from_validation(
        validation_context=context,
        validation_outcome=refused,
        current_gui_revision=4,
        project_root=Path(__file__).resolve().parent.parent,
    )
    assert result.ok is False
    assert result.status == "validation_not_accepted"
