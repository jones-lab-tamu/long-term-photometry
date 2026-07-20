from __future__ import annotations

import ast
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_startup_allocation as allocation
import photometry_pipeline.guided_startup_materialization as materialization
import photometry_pipeline.guided_startup_transaction as startup
from gui.run_report_parser import classify_completed_run_candidate
from photometry_pipeline.config import Config
from photometry_pipeline.guided_manifest_verification import (
    load_guided_candidate_manifest,
)
from tests.test_guided_startup_allocation import allocation_case


@pytest.fixture
def allocated_case(allocation_case):
    request, plan = allocation_case
    allocated = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert allocated.ok
    return request, plan, allocated


def _materialize(case):
    request, plan, allocated = case
    return materialization.materialize_guided_startup_artifacts(
        request=request,
        pure_plan=plan,
        allocation_result=allocated,
    )


def test_materializes_exact_planned_startup_artifacts(allocated_case):
    request, plan, _allocated = allocated_case
    result = _materialize(allocated_case)
    run_dir = Path(result.allocated_run_dir)
    expected = {
        startup.GUIDED_STARTUP_STATUS_FILENAME,
        startup.GUIDED_CANDIDATE_MANIFEST_FILENAME,
        startup.GUIDED_CONFIG_EFFECTIVE_FILENAME,
        startup.GUIDED_STARTUP_PROVENANCE_FILENAME,
        startup.GUIDED_COMMAND_RECORD_FILENAME,
        startup.GUIDED_NORMALIZED_RECORDING_DESCRIPTION_FILENAME,
    }
    assert result.status == "startup_artifacts_materialized"
    assert result.ok and result.materialized
    assert {item.name for item in run_dir.iterdir()} == expected
    assert (run_dir / startup.GUIDED_CANDIDATE_MANIFEST_FILENAME).read_bytes() == (
        plan.candidate_manifest_bytes
    )
    assert (run_dir / startup.GUIDED_CONFIG_EFFECTIVE_FILENAME).read_bytes() == (
        plan.config_effective_bytes
    )
    assert (run_dir / startup.GUIDED_STARTUP_PROVENANCE_FILENAME).read_bytes() == (
        plan.startup_provenance_bytes
    )
    assert (run_dir / startup.GUIDED_COMMAND_RECORD_FILENAME).read_bytes() == (
        plan.command_record_bytes
    )
    assert (run_dir / startup.GUIDED_STARTUP_STATUS_FILENAME).read_bytes() == (
        plan.startup_status_bytes
    )
    assert result.startup_status_updated is False
    assert result.startup_transaction_identity == (
        plan.identities.startup_transaction_identity
    )
    assert all(
        (
            result.no_runner_invoked,
            result.no_wrapper_invoked,
            result.no_gui_mutation,
            result.no_completed_run_claim,
            result.no_production_status_written,
            result.no_manifest_json_production_written,
        )
    )
    assert request.payload_result.runner_request is None


def test_manifest_round_trip_identity_matches_plan(allocated_case):
    _request, plan, _allocated = allocated_case
    result = _materialize(allocated_case)
    loaded = load_guided_candidate_manifest(result.manifest_path)
    assert loaded.accepted
    assert loaded.manifest.canonical_candidate_manifest_payload_identity == (
        plan.identities.candidate_manifest_payload_identity
    )


def test_config_round_trip_matches_payload(allocated_case):
    request, _plan, _allocated = allocated_case
    result = _materialize(allocated_case)
    loaded = Config.from_yaml(result.config_path)
    for item in request.payload_result.config_payload.values:
        expected = list(item.value) if isinstance(item.value, tuple) else item.value
        assert getattr(loaded, item.name) == expected


def test_provenance_and_command_remain_nonexecuting(allocated_case):
    _request, plan, _allocated = allocated_case
    result = _materialize(allocated_case)
    provenance = json.loads(Path(result.provenance_path).read_bytes())
    command = Path(result.command_record_path).read_text(encoding="utf-8")
    assert provenance["state"] == "prepared_runner_not_started"
    assert provenance["runner_started"] is False
    assert provenance["completed_run_claim"] is False
    assert provenance["startup_transaction_identity"] == (
        plan.identities.startup_transaction_identity
    )
    assert "--guided-candidate-manifest\n" in command
    assert "--guided-preallocated-run-dir\n" in command
    assert "--mode\nphasic\n" in command
    assert "--run-type\nfull\n" in command
    assert "\ntonic\n" not in command
    assert "\nboth\n" not in command
    assert plan.command_plan.executable_now is False


def test_refuses_unaccepted_allocation_result(allocated_case):
    request, plan, allocated = allocated_case
    refused = replace(allocated, status="allocation_failed", ok=False)
    result = materialization.materialize_guided_startup_artifacts(
        request=request, pure_plan=plan, allocation_result=refused
    )
    assert result.status == "refused_before_materialization"
    assert result.blocking_issues[0].category == "allocation_not_accepted"
    assert tuple(Path(allocated.allocated_run_dir).iterdir()) == (
        Path(allocated.startup_status_path),
    )


def test_refuses_missing_startup_status(allocated_case):
    request, plan, allocated = allocated_case
    Path(allocated.startup_status_path).unlink()
    result = materialization.materialize_guided_startup_artifacts(
        request=request, pure_plan=plan, allocation_result=allocated
    )
    assert result.blocking_issues[0].category == "startup_status_missing_or_invalid"
    assert tuple(Path(allocated.allocated_run_dir).iterdir()) == ()


def test_refuses_changed_startup_status_bytes(allocated_case):
    request, plan, allocated = allocated_case
    Path(allocated.startup_status_path).write_bytes(b"{}\n")
    result = materialization.materialize_guided_startup_artifacts(
        request=request, pure_plan=plan, allocation_result=allocated
    )
    assert result.blocking_issues[0].category == "startup_status_mismatch"


def test_refuses_unexpected_extra_file(allocated_case):
    request, plan, allocated = allocated_case
    run_dir = Path(allocated.allocated_run_dir)
    (run_dir / "unexpected.txt").write_text("unexpected")
    result = materialization.materialize_guided_startup_artifacts(
        request=request, pure_plan=plan, allocation_result=allocated
    )
    assert result.blocking_issues[0].category == "allocated_directory_not_pristine"
    assert not (run_dir / startup.GUIDED_CANDIDATE_MANIFEST_FILENAME).exists()


@pytest.mark.parametrize(
    "filename",
    (
        startup.GUIDED_CANDIDATE_MANIFEST_FILENAME,
        startup.GUIDED_CONFIG_EFFECTIVE_FILENAME,
        startup.GUIDED_STARTUP_PROVENANCE_FILENAME,
        startup.GUIDED_COMMAND_RECORD_FILENAME,
    ),
)
def test_refuses_preexisting_target_artifact(allocated_case, filename):
    request, plan, allocated = allocated_case
    run_dir = Path(allocated.allocated_run_dir)
    (run_dir / filename).write_bytes(b"existing")
    result = materialization.materialize_guided_startup_artifacts(
        request=request, pure_plan=plan, allocation_result=allocated
    )
    assert result.blocking_issues[0].category == "startup_artifact_already_exists"


@pytest.mark.parametrize("filename", ("status.json", "MANIFEST.json", "run_report.json"))
def test_refuses_production_shaped_file(allocated_case, filename):
    request, plan, allocated = allocated_case
    run_dir = Path(allocated.allocated_run_dir)
    (run_dir / filename).write_text("{}")
    result = materialization.materialize_guided_startup_artifacts(
        request=request, pure_plan=plan, allocation_result=allocated
    )
    assert result.blocking_issues[0].category == "production_artifact_prohibited"
    assert result.files_written == ()


def test_manifest_write_failure_writes_no_later_files(allocated_case, monkeypatch):
    original_open = Path.open

    def fail_manifest(self, mode="r", *args, **kwargs):
        if (
            self.name == startup.GUIDED_CANDIDATE_MANIFEST_FILENAME
            and mode == "xb"
        ):
            raise OSError("simulated manifest write failure")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_manifest)
    result = _materialize(allocated_case)
    run_dir = Path(result.allocated_run_dir)
    assert result.status == "materialization_failed_partial"
    assert result.files_written == ()
    assert {item.name for item in run_dir.iterdir()} == {
        startup.GUIDED_STARTUP_STATUS_FILENAME
    }
    assert result.no_runner_invoked and result.no_wrapper_invoked


def test_config_write_failure_retains_manifest_only(allocated_case, monkeypatch):
    original_open = Path.open

    def fail_config(self, mode="r", *args, **kwargs):
        if self.name == startup.GUIDED_CONFIG_EFFECTIVE_FILENAME and mode == "xb":
            raise OSError("simulated config write failure")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_config)
    result = _materialize(allocated_case)
    run_dir = Path(result.allocated_run_dir)
    assert result.status == "materialization_failed_partial"
    assert result.files_written == (startup.GUIDED_CANDIDATE_MANIFEST_FILENAME,)
    assert {item.name for item in run_dir.iterdir()} == {
        startup.GUIDED_STARTUP_STATUS_FILENAME,
        startup.GUIDED_CANDIDATE_MANIFEST_FILENAME,
    }
    assert not (run_dir / startup.GUIDED_COMMAND_RECORD_FILENAME).exists()
    assert not (run_dir / startup.GUIDED_STARTUP_PROVENANCE_FILENAME).exists()


def test_prepared_directory_is_completed_run_ineligible(allocated_case):
    result = _materialize(allocated_case)
    accepted, _reason = classify_completed_run_candidate(result.allocated_run_dir)
    assert accepted is False


def test_partial_directory_is_completed_run_ineligible(
    allocated_case, monkeypatch
):
    original_open = Path.open

    def fail_config(self, mode="r", *args, **kwargs):
        if self.name == startup.GUIDED_CONFIG_EFFECTIVE_FILENAME and mode == "xb":
            raise OSError("simulated config write failure")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_config)
    result = _materialize(allocated_case)
    accepted, _reason = classify_completed_run_candidate(result.allocated_run_dir)
    assert result.status == "materialization_failed_partial"
    assert accepted is False


def test_no_runner_wrapper_or_gui_api_is_called(allocated_case, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("execution API must not be called")

    monkeypatch.setattr(os, "system", fail)
    result = _materialize(allocated_case)
    assert result.ok
    assert result.no_runner_invoked
    assert result.no_wrapper_invoked
    assert result.no_gui_mutation


def test_materialization_module_import_boundary():
    source = Path(materialization.__file__).read_text(encoding="utf-8")
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


def test_materialize_never_writes_posthoc_strategy_map_even_when_flag_enabled(allocated_case, monkeypatch):
    # The obsolete Guided post-hoc applied-dF/F route has been retired from
    # current-Guided production. This positive_legacy-shaped (empty
    # per-ROI map) request used to trigger guided_correction_strategy_map.json
    # materialization when applied_dff_orchestration_enabled was True; that
    # field is now inert deprecated input and must not influence
    # materialization output at all.
    request, plan, allocated = allocated_case

    object.__setattr__(
        request.startup_authority.rwd.production_intent.correction,
        "applied_dff_orchestration_enabled",
        True
    )
    monkeypatch.setattr(
        materialization,
        "_validate_preconditions",
        lambda r, p, a: (Path(a.allocated_run_dir), None)
    )

    result = materialization.materialize_guided_startup_artifacts(
        request=request,
        pure_plan=plan,
        allocation_result=allocated,
    )
    assert result.ok, result.blocking_issues
    strategy_map_path = Path(result.allocated_run_dir) / "guided_correction_strategy_map.json"
    assert not strategy_map_path.exists()


@pytest.mark.parametrize(
    "selected",
    (
        ("robust_global_event_reject", "global_linear_regression"),
        ("signal_only_f0", "signal_only_f0"),
        ("robust_global_event_reject", "signal_only_f0"),
    ),
)
def test_current_native_materialization_never_writes_posthoc_artifact(
    allocated_case, monkeypatch, selected
):
    from photometry_pipeline.guided_production_mapping import (
        GuidedProductionPerRoiStrategy,
    )

    request, plan, allocated = allocated_case
    correction = request.startup_authority.rwd.production_intent.correction
    included = tuple(f"ROI{index + 1}" for index in range(len(selected)))
    object.__setattr__(
        request.startup_authority.rwd.production_intent.roi_scope,
        "included_roi_ids", included,
    )
    entries = tuple(
        GuidedProductionPerRoiStrategy(
            roi_id=roi_id,
            strategy_family=(
                "signal_only_f0" if strategy == "signal_only_f0" else "dynamic_fit"
            ),
            dynamic_fit_mode=None if strategy == "signal_only_f0" else strategy,
            selected_strategy=strategy,
            evidence_source_type="test",
            evidence_reference_json="{}",
            explicit_user_mark=True,
            current_or_stale="current",
        )
        for roi_id, strategy in zip(included, selected)
    )
    object.__setattr__(correction, "production_strategy_map_version", "per_roi_correction_strategy_map.v1")
    object.__setattr__(correction, "per_roi_production_strategy_map", entries)
    object.__setattr__(correction, "applied_dff_orchestration_enabled", True)
    monkeypatch.setattr(
        materialization, "_validate_preconditions",
        lambda r, p, a: (Path(a.allocated_run_dir), None),
    )

    result = materialization.materialize_guided_startup_artifacts(
        request=request, pure_plan=plan, allocation_result=allocated
    )
    run_dir = Path(result.allocated_run_dir)
    assert result.ok, result.blocking_issues
    assert (run_dir / "guided_per_roi_correction.json").is_file()
    assert not (run_dir / "guided_correction_strategy_map.json").exists()


@pytest.mark.parametrize(
    "scenario_label,execution_mode,selected",
    (
        ("phasic_all_signal_only_f0", "phasic", ("signal_only_f0", "signal_only_f0")),
        ("tonic_all_signal_only_f0", "tonic", ("signal_only_f0", "signal_only_f0")),
        ("combined_all_signal_only_f0", "both", ("signal_only_f0", "signal_only_f0")),
        (
            "combined_mixed_strategies",
            "both",
            ("robust_global_event_reject", "signal_only_f0"),
        ),
        (
            "fit_only_native_per_roi_execution",
            "phasic",
            ("robust_global_event_reject", "global_linear_regression"),
        ),
    ),
)
def test_native_materialization_never_calls_old_orchestration(
    allocated_case, monkeypatch, scenario_label, execution_mode, selected
):
    """Interception test (task: retire obsolete Guided post-hoc applied-dF/F
    route): for every current-native strategy matrix, materialization must
    never invoke the retired guided_applied_dff_orchestration entry point,
    regardless of applied_dff_orchestration_enabled or execution_mode."""
    from photometry_pipeline.guided_production_mapping import (
        GuidedProductionPerRoiStrategy,
    )
    import photometry_pipeline.guided_applied_dff_orchestration as old_orchestration

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError(
            f"old Guided orchestration entry point must not be called for "
            f"{scenario_label}"
        )

    monkeypatch.setattr(
        old_orchestration,
        "run_guided_applied_dff_orchestration_if_enabled",
        _fail_if_called,
    )

    request, plan, allocated = allocated_case
    correction = request.startup_authority.rwd.production_intent.correction
    included = tuple(f"ROI{index + 1}" for index in range(len(selected)))
    object.__setattr__(
        request.startup_authority.rwd.production_intent.roi_scope,
        "included_roi_ids", included,
    )
    object.__setattr__(
        request.startup_authority.rwd.production_intent.execution_profile,
        "execution_mode", execution_mode,
    )
    entries = tuple(
        GuidedProductionPerRoiStrategy(
            roi_id=roi_id,
            strategy_family=(
                "signal_only_f0" if strategy == "signal_only_f0" else "dynamic_fit"
            ),
            dynamic_fit_mode=None if strategy == "signal_only_f0" else strategy,
            selected_strategy=strategy,
            evidence_source_type="test",
            evidence_reference_json="{}",
            explicit_user_mark=True,
            current_or_stale="current",
        )
        for roi_id, strategy in zip(included, selected)
    )
    object.__setattr__(correction, "production_strategy_map_version", "per_roi_correction_strategy_map.v1")
    object.__setattr__(correction, "per_roi_production_strategy_map", entries)
    # Explicitly True to prove the deprecated flag cannot trigger the old
    # route even when set.
    object.__setattr__(correction, "applied_dff_orchestration_enabled", True)
    monkeypatch.setattr(
        materialization, "_validate_preconditions",
        lambda r, p, a: (Path(a.allocated_run_dir), None),
    )

    result = materialization.materialize_guided_startup_artifacts(
        request=request, pure_plan=plan, allocation_result=allocated
    )
    run_dir = Path(result.allocated_run_dir)
    assert result.ok, result.blocking_issues
    assert (run_dir / "guided_per_roi_correction.json").is_file()
    assert not (run_dir / "guided_correction_strategy_map.json").exists()
    assert not (run_dir / "applied_trace_cache.h5").exists()


def test_guided_execution_chain_never_imports_old_orchestration_module():
    """Static import-boundary proof that the current-native Guided
    wrapper/execution chain has no dependency on the retired post-hoc
    applied-dF/F orchestration module.

    tools/run_full_pipeline_deliverables.py is the real production wrapper
    (see guided_execution_request_builder.py's wrapper_path, which points
    at it and records supported_contract_version="run_full_pipeline_deliverables.v1");
    analyze_photometry.py is a separate, lower-level analysis entry point.
    Both, plus the startup-to-wrapper orchestration modules, must be free
    of this dependency."""
    import photometry_pipeline.guided_backend_execution as backend_execution
    import photometry_pipeline.guided_startup_orchestration as orchestration
    import analyze_photometry as analysis_entrypoint
    import tools.run_full_pipeline_deliverables as production_wrapper

    prohibited = "photometry_pipeline.guided_applied_dff_orchestration"
    for module in (
        backend_execution,
        orchestration,
        analysis_entrypoint,
        production_wrapper,
    ):
        source = Path(module.__file__).read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        assert not any(
            name == prohibited or name.startswith(f"{prohibited}.")
            for name in imported
        ), f"{module.__name__} must not import {prohibited}"


def test_materialize_skips_strategy_map_when_orchestration_disabled(allocated_case, monkeypatch):
    request, plan, allocated = allocated_case

    object.__setattr__(
        request.startup_authority.rwd.production_intent.correction,
        "applied_dff_orchestration_enabled",
        False
    )
    monkeypatch.setattr(
        materialization,
        "_validate_preconditions",
        lambda r, p, a: (Path(a.allocated_run_dir), None)
    )

    result = materialization.materialize_guided_startup_artifacts(
        request=request,
        pure_plan=plan,
        allocation_result=allocated,
    )
    assert result.ok
    strategy_map_path = Path(result.allocated_run_dir) / "guided_correction_strategy_map.json"
    assert not strategy_map_path.exists()
