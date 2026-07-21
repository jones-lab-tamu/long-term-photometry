"""Focused tests for the minimal Guided tonic-settings release.

Covers: plan defaults/contract, GUI controls, validation invalidation,
Review Plan display, plan identity, execution-payload threading (including
gap-free/missing-session rejection), requested-vs-consumed completion
verification, Results display, and default numerical-equivalence proof.
"""
from __future__ import annotations

from dataclasses import fields, replace
import os
import json

import pytest
import yaml

import photometry_pipeline.guided_backend_validation_request as validation_request
import photometry_pipeline.guided_backend_validator as validator
import photometry_pipeline.guided_execution_payloads as payloads
import photometry_pipeline.guided_run_authorization as authorization
from photometry_pipeline.config import Config
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    GuidedBackendValidationWorkflowOutcome,
    build_guided_backend_validation_parser_contract,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GUIDED_SUPPORTED_TONIC_OUTPUT_MODES,
    GUIDED_SUPPORTED_TONIC_TIMELINE_MODES,
    GuidedNewAnalysisDraftPlan,
    GuidedNewAnalysisTonicSettingsContract,
)
from photometry_pipeline.guided_plan_identity import (
    compute_guided_new_analysis_draft_plan_identity,
)
from photometry_pipeline.run_completion_contract import (
    tonic_settings_completion_error,
)
from photometry_pipeline.completed_run_review import format_tonic_settings_summary

from tests.test_guided_execution_payloads import (
    _accepted_candidate,
    _accepted_roi,
    _build_app_identity,
    _request_auth,
)
from tests.test_guided_backend_validator import (
    _contract as _validator_contract,
    _normalized_recording_identity_for,
    _request as _valid_request,
    _typed,
)


# ---------------------------------------------------------------------------
# 1. Guided defaults
# ---------------------------------------------------------------------------


def test_guided_tonic_defaults_are_preserve_and_real_elapsed():
    contract = GuidedNewAnalysisTonicSettingsContract()
    assert contract.tonic_output_mode == "preserve_raw_session_shape"
    assert contract.tonic_timeline_mode == "real_elapsed_time"
    plan = GuidedNewAnalysisDraftPlan()
    assert plan.tonic_settings_contract.tonic_output_mode == "preserve_raw_session_shape"
    assert plan.tonic_settings_contract.tonic_timeline_mode == "real_elapsed_time"


# ---------------------------------------------------------------------------
# 3. Enum mapping: Guided-supported values are a subset of Config's allowed values
# ---------------------------------------------------------------------------


def test_guided_supported_tonic_values_match_config_allowed_values():
    config = Config()
    output_field = next(
        f for f in fields(config) if f.name == "tonic_output_mode"
    )
    timeline_field = next(
        f for f in fields(config) if f.name == "tonic_timeline_mode"
    )
    # Both Guided-supported output modes are valid Config values.
    for value in GUIDED_SUPPORTED_TONIC_OUTPUT_MODES:
        object.__setattr__(config, "tonic_output_mode", value)
    # Guided deliberately excludes compressed_recording_time.
    assert "compressed_recording_time" not in GUIDED_SUPPORTED_TONIC_TIMELINE_MODES
    for value in GUIDED_SUPPORTED_TONIC_TIMELINE_MODES:
        object.__setattr__(config, "tonic_timeline_mode", value)
    del output_field, timeline_field  # presence-only check above


def test_gui_tonic_choice_values_are_exactly_guided_supported_sets():
    from gui.main_window import (
        GUIDED_TONIC_OUTPUT_MODE_CHOICES,
        GUIDED_TONIC_TIMELINE_MODE_CHOICES,
    )

    assert {value for _, value in GUIDED_TONIC_OUTPUT_MODE_CHOICES} == (
        GUIDED_SUPPORTED_TONIC_OUTPUT_MODES
    )
    assert {value for _, value in GUIDED_TONIC_TIMELINE_MODE_CHOICES} == (
        GUIDED_SUPPORTED_TONIC_TIMELINE_MODES
    )


# ---------------------------------------------------------------------------
# 6. Plan identity changes when either tonic value changes
# ---------------------------------------------------------------------------


def test_plan_identity_changes_with_tonic_output_mode():
    base = GuidedNewAnalysisDraftPlan()
    changed = replace(
        base,
        tonic_settings_contract=GuidedNewAnalysisTonicSettingsContract(
            tonic_output_mode="flatten_session_bleach_preserve_session_baseline"
        ),
    )
    assert compute_guided_new_analysis_draft_plan_identity(
        base
    ) != compute_guided_new_analysis_draft_plan_identity(changed)


def test_plan_identity_changes_with_tonic_timeline_mode():
    base = GuidedNewAnalysisDraftPlan()
    changed = replace(
        base,
        tonic_settings_contract=GuidedNewAnalysisTonicSettingsContract(
            tonic_timeline_mode="gap_free_elapsed_time"
        ),
    )
    assert compute_guided_new_analysis_draft_plan_identity(
        base
    ) != compute_guided_new_analysis_draft_plan_identity(changed)


def test_plan_identity_stable_for_unchanged_tonic_settings():
    base = GuidedNewAnalysisDraftPlan()
    same = replace(base, tonic_settings_contract=GuidedNewAnalysisTonicSettingsContract())
    assert compute_guided_new_analysis_draft_plan_identity(
        base
    ) == compute_guided_new_analysis_draft_plan_identity(same)


# ---------------------------------------------------------------------------
# Contract validation: allowed values only
# ---------------------------------------------------------------------------


def test_tonic_settings_contract_rejects_unsupported_output_mode():
    with pytest.raises(ValueError):
        GuidedNewAnalysisTonicSettingsContract(tonic_output_mode="not_a_real_mode")


def test_tonic_settings_contract_rejects_compressed_recording_time():
    with pytest.raises(ValueError):
        GuidedNewAnalysisTonicSettingsContract(
            tonic_timeline_mode="compressed_recording_time"
        )


# ---------------------------------------------------------------------------
# 23. No per-ROI tonic setting exists in the plan
# ---------------------------------------------------------------------------


def test_tonic_settings_contract_has_no_per_roi_structure():
    field_names = {f.name for f in fields(GuidedNewAnalysisTonicSettingsContract)}
    assert field_names == {
        "schema_version",
        "tonic_output_mode",
        "tonic_timeline_mode",
        "provenance",
    }


# ---------------------------------------------------------------------------
# 7/9/16/17/18: end-to-end execution-payload threading + gap-free rejection
# ---------------------------------------------------------------------------


def _accepted_outcome_with(
    *, tonic_output_mode: str, tonic_timeline_mode: str, approved_missing=()
):
    request = _valid_request()
    new_dataset = replace(
        request.acquisition_dataset,
        semantic_values=request.acquisition_dataset.semantic_values
        + (_typed("target_fs_hz", 40.0),),
    )
    new_source = (
        replace(request.source, approved_missing_candidates=approved_missing)
        if approved_missing
        else request.source
    )
    new_correction = replace(
        request.correction,
        global_tonic_output_mode=tonic_output_mode,
        global_tonic_timeline_mode=tonic_timeline_mode,
    )
    request = replace(
        request,
        acquisition_dataset=new_dataset,
        source=new_source,
        correction=new_correction,
        normalized_recording_description_identity=(
            _normalized_recording_identity_for(
                new_source, new_dataset, request.roi_scope, request.parser
            )
        ),
    )
    identity = validation_request.compute_guided_backend_validation_request_identity(
        request
    )
    compiled = validation_request.GuidedBackendValidationCompileSuccess(
        request, identity
    )
    validated = validator.validate_guided_backend_validation_request(
        request,
        canonical_request_identity=identity,
        validator_contract=_validator_contract(),
    )
    if not validated.accepted:
        return validated, None
    return validated, GuidedBackendValidationWorkflowOutcome(
        status="validator_accepted",
        accepted_for_backend_validation=True,
        run_authorization=False,
        request_identity=identity,
        validation_result=validated,
        compile_result=compiled,
        materialization_result=object(),
        blocking_issues=(),
        user_summary="Accepted.",
    )


def test_gap_free_is_rejected_by_backend_validator_when_missing_sessions_exist():
    base_request = _valid_request()
    candidate = base_request.source.candidate_files[0]
    validated, _outcome = _accepted_outcome_with(
        tonic_output_mode="preserve_raw_session_shape",
        tonic_timeline_mode="gap_free_elapsed_time",
        approved_missing=(candidate,),
    )
    assert validated.accepted is False
    assert validated.blocking_issues[0].category == (
        "tonic_timeline_gap_free_blocked_by_missing_sessions"
    )


def test_gap_free_accepted_when_no_missing_sessions():
    validated, _outcome = _accepted_outcome_with(
        tonic_output_mode="preserve_raw_session_shape",
        tonic_timeline_mode="gap_free_elapsed_time",
    )
    assert validated.accepted is True


def _authorize(monkeypatch, outcome):
    req = _request_auth(outcome=outcome)
    monkeypatch.setattr(
        authorization.validation_workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: req.stored_validation_outcome,
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
    res = authorization.authorize_guided_run(req)
    assert res.status == "authorized", res
    return res


def test_non_default_tonic_settings_are_emitted_into_generated_config(monkeypatch):
    _validated, outcome = _accepted_outcome_with(
        tonic_output_mode="flatten_session_bleach_preserve_session_baseline",
        tonic_timeline_mode="gap_free_elapsed_time",
    )
    auth_result = _authorize(monkeypatch, outcome)
    contract = payloads.build_guided_execution_startup_mapping_contract()
    result = payloads.derive_guided_execution_payloads(
        auth_result, startup_mapping_contract=contract
    )
    assert result.config_payload is not None
    emitted = {item.name: item.value for item in result.config_payload.values}
    assert emitted["tonic_output_mode"] == (
        "flatten_session_bleach_preserve_session_baseline"
    )
    assert emitted["tonic_timeline_mode"] == "gap_free_elapsed_time"
    # lowpass_hz stays fixed regardless of the tonic selection.
    assert emitted["lowpass_hz"] == 1.0


def test_default_tonic_settings_are_emitted_into_generated_config(monkeypatch):
    _validated, outcome = _accepted_outcome_with(
        tonic_output_mode="preserve_raw_session_shape",
        tonic_timeline_mode="real_elapsed_time",
    )
    auth_result = _authorize(monkeypatch, outcome)
    contract = payloads.build_guided_execution_startup_mapping_contract()
    result = payloads.derive_guided_execution_payloads(
        auth_result, startup_mapping_contract=contract
    )
    emitted = {item.name: item.value for item in result.config_payload.values}
    assert emitted["tonic_output_mode"] == "preserve_raw_session_shape"
    assert emitted["tonic_timeline_mode"] == "real_elapsed_time"


def test_lowpass_hz_disposition_remains_fixed():
    assert (
        payloads.GUIDED_CONFIG_FIELD_DISPOSITIONS["lowpass_hz"]
        == payloads.CONFIG_DISPOSITION_FIXED
    )
    assert payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES["lowpass_hz"] == 1.0


def test_tonic_settings_disposition_is_intent():
    assert (
        payloads.GUIDED_CONFIG_FIELD_DISPOSITIONS["tonic_output_mode"]
        == payloads.CONFIG_DISPOSITION_INTENT
    )
    assert (
        payloads.GUIDED_CONFIG_FIELD_DISPOSITIONS["tonic_timeline_mode"]
        == payloads.CONFIG_DISPOSITION_INTENT
    )
    assert "tonic_output_mode" not in payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES
    assert "tonic_timeline_mode" not in payloads.GUIDED_CONFIG_DEFAULT_OVERRIDES


# ---------------------------------------------------------------------------
# 8. No new wrapper CLI arguments
# ---------------------------------------------------------------------------


def test_wrapper_argparse_has_no_top_level_tonic_flags():
    """Real, bounded inspection of the wrapper's own argparse.ArgumentParser
    -- not a source-text search. Fails if `--tonic-output-mode` or
    `--tonic-timeline-mode` become top-level wrapper flags; passes today
    because those flags exist only on the separate plot_tonic_48h.py
    subprocess argv this wrapper builds, which never touches this parser
    object at all.
    """
    import argparse as argparse_module
    import tools.run_full_pipeline_deliverables as rfp

    captured: dict = {}

    class _ParserBuilt(Exception):
        pass

    def _capture_and_stop(self, *_args, **_kwargs):
        captured["parser"] = self
        raise _ParserBuilt()

    original = argparse_module.ArgumentParser.parse_args
    argparse_module.ArgumentParser.parse_args = _capture_and_stop
    try:
        with pytest.raises(_ParserBuilt):
            rfp.parse_args()
    finally:
        argparse_module.ArgumentParser.parse_args = original

    assert "parser" in captured, (
        "Could not locate the wrapper's argument-parser boundary "
        "(tools.run_full_pipeline_deliverables.parse_args no longer calls "
        "ArgumentParser.parse_args as expected); cannot verify the absence "
        "of top-level tonic flags."
    )
    option_strings = set(captured["parser"]._option_string_actions.keys())
    assert "--tonic-output-mode" not in option_strings
    assert "--tonic-timeline-mode" not in option_strings


def test_tonic_settings_arrive_via_generated_config_not_cli(tmp_path):
    """Direct production-path proof: the wrapper's own parsed CLI namespace
    carries no tonic_output_mode/tonic_timeline_mode attribute at all, while
    Config.from_yaml -- the wrapper's actual consumption path -- resolves
    both from the generated config file."""
    import sys
    import tools.run_full_pipeline_deliverables as rfp
    from photometry_pipeline.config import Config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tonic_output_mode": "flatten_session_bleach_preserve_session_baseline",
                "tonic_timeline_mode": "gap_free_elapsed_time",
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "run_full_pipeline_deliverables.py",
        "--input",
        str(tmp_path / "input"),
        "--out",
        str(tmp_path / "out"),
        "--config",
        str(config_path),
        "--format",
        "rwd",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        args = rfp.parse_args()
    finally:
        sys.argv = old_argv
    assert not hasattr(args, "tonic_output_mode")
    assert not hasattr(args, "tonic_timeline_mode")

    config = Config.from_yaml(str(config_path))
    assert config.tonic_output_mode == (
        "flatten_session_bleach_preserve_session_baseline"
    )
    assert config.tonic_timeline_mode == "gap_free_elapsed_time"


# ---------------------------------------------------------------------------
# 9. Config.from_yaml consumes both fields (natural wrapper path)
# ---------------------------------------------------------------------------


def test_config_from_yaml_round_trips_non_default_tonic_settings(tmp_path):
    from photometry_pipeline.config import Config

    payload = {
        "tonic_output_mode": "flatten_session_bleach_preserve_session_baseline",
        "tonic_timeline_mode": "gap_free_elapsed_time",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = Config.from_yaml(str(path))
    assert config.tonic_output_mode == (
        "flatten_session_bleach_preserve_session_baseline"
    )
    assert config.tonic_timeline_mode == "gap_free_elapsed_time"


# ---------------------------------------------------------------------------
# 16/17/18: requested-versus-consumed completion verification
# ---------------------------------------------------------------------------


def _write_yaml(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle)


def _write_guided_marker(run_dir):
    # Minimal definitive Guided marker so classify_guided_current_native_state
    # treats this run directory as a current Guided run.
    from photometry_pipeline.guided_startup_transaction import (
        GUIDED_STARTUP_STATUS_FILENAME,
    )

    with open(
        os.path.join(run_dir, GUIDED_STARTUP_STATUS_FILENAME), "w", encoding="utf-8"
    ) as handle:
        json.dump({"status": "materialized"}, handle)


def test_tonic_settings_match_is_not_fatal(tmp_path):
    run_dir = str(tmp_path)
    _write_guided_marker(run_dir)
    _write_yaml(
        os.path.join(run_dir, "config_effective.yaml"),
        {"tonic_output_mode": "preserve_raw_session_shape", "tonic_timeline_mode": "real_elapsed_time"},
    )
    _write_yaml(
        os.path.join(run_dir, "_analysis", "tonic_out", "config_used.yaml"),
        {"tonic_output_mode": "preserve_raw_session_shape", "tonic_timeline_mode": "real_elapsed_time"},
    )
    error = tonic_settings_completion_error(
        run_dir, {"tonic_analysis": True, "phasic_analysis": False}
    )
    assert error == ""


def test_tonic_output_mode_mismatch_is_fatal(tmp_path):
    run_dir = str(tmp_path)
    _write_guided_marker(run_dir)
    _write_yaml(
        os.path.join(run_dir, "config_effective.yaml"),
        {"tonic_output_mode": "flatten_session_bleach_preserve_session_baseline", "tonic_timeline_mode": "real_elapsed_time"},
    )
    _write_yaml(
        os.path.join(run_dir, "_analysis", "tonic_out", "config_used.yaml"),
        {"tonic_output_mode": "preserve_raw_session_shape", "tonic_timeline_mode": "real_elapsed_time"},
    )
    error = tonic_settings_completion_error(
        run_dir, {"tonic_analysis": True, "phasic_analysis": False}
    )
    assert error != ""
    assert "tonic_output_mode" in error


def test_tonic_timeline_mode_mismatch_is_fatal(tmp_path):
    run_dir = str(tmp_path)
    _write_guided_marker(run_dir)
    _write_yaml(
        os.path.join(run_dir, "config_effective.yaml"),
        {"tonic_output_mode": "preserve_raw_session_shape", "tonic_timeline_mode": "gap_free_elapsed_time"},
    )
    _write_yaml(
        os.path.join(run_dir, "_analysis", "tonic_out", "config_used.yaml"),
        {"tonic_output_mode": "preserve_raw_session_shape", "tonic_timeline_mode": "real_elapsed_time"},
    )
    error = tonic_settings_completion_error(
        run_dir, {"tonic_analysis": True, "phasic_analysis": False}
    )
    assert error != ""
    assert "tonic_timeline_mode" in error


def test_missing_consumed_evidence_is_fatal(tmp_path):
    run_dir = str(tmp_path)
    _write_guided_marker(run_dir)
    _write_yaml(
        os.path.join(run_dir, "config_effective.yaml"),
        {"tonic_output_mode": "preserve_raw_session_shape", "tonic_timeline_mode": "real_elapsed_time"},
    )
    # No _analysis/tonic_out/config_used.yaml written at all.
    error = tonic_settings_completion_error(
        run_dir, {"tonic_analysis": True, "phasic_analysis": False}
    )
    assert error != ""


def test_tonic_settings_check_is_noop_when_tonic_not_enabled(tmp_path):
    run_dir = str(tmp_path)
    error = tonic_settings_completion_error(run_dir, {"tonic_analysis": False, "phasic_analysis": True})
    assert error == ""


def test_tonic_settings_check_is_noop_for_non_guided_run(tmp_path):
    run_dir = str(tmp_path)
    # No Guided marker files at all -- an ordinary Full Control run.
    error = tonic_settings_completion_error(run_dir, {"tonic_analysis": True, "phasic_analysis": True})
    assert error == ""


# ---------------------------------------------------------------------------
# 19/20: Results display
# ---------------------------------------------------------------------------


def test_format_tonic_settings_summary_default():
    text = format_tonic_settings_summary(
        {"tonic_output_mode": "preserve_raw_session_shape", "tonic_timeline_mode": "real_elapsed_time"}
    )
    assert text == "Tonic timeline: Real elapsed time\nSession shape: Preserved"


def test_format_tonic_settings_summary_non_default():
    text = format_tonic_settings_summary(
        {
            "tonic_output_mode": "flatten_session_bleach_preserve_session_baseline",
            "tonic_timeline_mode": "gap_free_elapsed_time",
        }
    )
    assert text == (
        "Tonic timeline: Gap-free elapsed time\n"
        "Session shape: Within-session bleaching trend removed"
    )


def test_format_tonic_settings_summary_empty_when_no_evidence():
    assert format_tonic_settings_summary({}) == ""
