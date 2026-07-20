from __future__ import annotations

import ast
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_backend_validation_request as validation_request
import photometry_pipeline.guided_backend_validator as validator
import photometry_pipeline.guided_execution_payloads as payloads
import photometry_pipeline.guided_production_mapping as mapping
import photometry_pipeline.guided_run_authorization as authorization
import photometry_pipeline.guided_startup_allocation as allocation
import photometry_pipeline.guided_startup_transaction as startup
from photometry_pipeline.guided_npm_startup_bridge import GuidedStartupAuthority
from gui.run_report_parser import classify_completed_run_candidate
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    GuidedBackendValidationWorkflowOutcome,
    build_guided_backend_validation_parser_contract,
)
from photometry_pipeline.guided_diagnostic_cache import (
    DIAGNOSTIC_CACHE_ARTIFACT_FILENAME,
    DIAGNOSTIC_CACHE_PROVENANCE_FILENAME,
    DIAGNOSTIC_CACHE_PURPOSE,
    DIAGNOSTIC_CACHE_SCHEMA_VERSION,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from tests.test_guided_backend_validator import (
    _contract as _validator_contract,
    _request as _valid_request,
    _typed,
)
from tests.test_guided_execution_payloads import (
    _accepted_candidate,
    _accepted_roi,
    _build_app_identity,
)


def _accepted_outcome(source_root: Path, output_base: Path):
    from tests.test_guided_backend_validator import (
        _normalized_recording_identity_for,
    )

    request = _valid_request()
    new_source = replace(
        request.source,
        source_root_canonical=os.path.abspath(source_root),
    )
    acquisition_dataset = replace(
        request.acquisition_dataset,
        semantic_values=request.acquisition_dataset.semantic_values
        + (_typed("target_fs_hz", 40.0),),
    )
    request = replace(
        request,
        source=new_source,
        output=replace(
            request.output,
            output_base_canonical=os.path.abspath(output_base),
        ),
        acquisition_dataset=acquisition_dataset,
        # source_root_canonical is part of the normalized recording
        # description's identity (recording_source_identity); it must be
        # recomputed after moving the source root to this test's tmp_path.
        normalized_recording_description_identity=_normalized_recording_identity_for(
            new_source,
            acquisition_dataset,
            request.roi_scope,
            request.parser,
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
    assert validated.accepted
    return GuidedBackendValidationWorkflowOutcome(
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


def _request_for_paths(
    monkeypatch,
    *,
    source_root: Path,
    output_base: Path,
    run_id: str = "guided_run_20260101T000000Z_abcdef",
    planned_dir: Path | None = None,
):
    outcome = _accepted_outcome(source_root, output_base)
    auth_request = authorization.build_guided_run_authorization_request(
        stored_validation_outcome=outcome,
        stored_validation_outcome_revision=3,
        current_gui_revision=3,
        current_validation_context=GuidedBackendValidationGuiContext(
            draft=GuidedNewAnalysisDraftPlan(),
            parser_contract=build_guided_backend_validation_parser_contract(),
            additional_protected_roots=(),
            validator_contract=_validator_contract(),
            revision=3,
        ),
        application_build_identity=_build_app_identity(),
        production_mapping_contract=mapping.build_guided_production_mapping_contract(),
    )
    monkeypatch.setattr(
        authorization.validation_workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: outcome,
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_candidate_manifest_execution_preflight",
        lambda request, **_kwargs: _accepted_candidate(request),
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_roi_execution_preflight",
        lambda request, **_kwargs: _accepted_roi(request),
    )
    auth = authorization.authorize_guided_run(auth_request)
    assert auth.authorized
    contract = payloads.build_guided_execution_startup_mapping_contract()
    derived = payloads.derive_guided_execution_payloads(
        auth, startup_mapping_contract=contract
    )
    planned_dir = planned_dir or output_base / run_id
    request = startup.GuidedStartupTransactionRequest(
        startup_authority=GuidedStartupAuthority(rwd=auth),
        payload_result=derived,
        startup_mapping_contract=contract,
        application_build_identity=auth.production_intent.application_build_identity,
        current_guided_revision=auth.authorized_gui_revision,
        explicit_user_run_transition=True,
        output_base_canonical=os.path.abspath(output_base),
        source_root_canonical=os.path.abspath(source_root),
        planned_run_id=run_id,
        planned_allocated_run_dir=os.path.abspath(planned_dir),
        wrapper_entrypoint=startup.GuidedWrapperEntrypointIdentity(
            entrypoint_kind="script_path",
            entrypoint_value="tools/run_full_pipeline_deliverables.py",
            trusted_application_root=os.path.abspath(Path.cwd()),
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
    return request, startup.plan_guided_startup_transaction(request)


@pytest.fixture
def allocation_case(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    return _request_for_paths(
        monkeypatch, source_root=source, output_base=output
    )


def test_allocates_and_writes_exactly_first_startup_status(allocation_case):
    request, plan = allocation_case
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    run_dir = Path(result.allocated_run_dir)
    status_path = run_dir / startup.GUIDED_STARTUP_STATUS_FILENAME
    assert result.status == "allocated_startup_status_written"
    assert result.ok and result.allocated and result.startup_status_written
    assert status_path.read_bytes() == plan.startup_status_bytes
    assert tuple(item.name for item in run_dir.iterdir()) == (
        startup.GUIDED_STARTUP_STATUS_FILENAME,
    )
    assert result.startup_status_sha256 == plan.identities.startup_status_bytes_sha256
    assert all(
        (
            result.no_runner_invoked,
            result.no_manifest_written,
            result.no_config_written,
            result.no_provenance_written,
            result.no_command_written,
            result.no_gui_mutation,
        )
    )
    assert result.completed_run_claim is False


def test_startup_only_directory_is_completed_run_ineligible(allocation_case):
    request, plan = allocation_case
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    accepted, _reason = classify_completed_run_candidate(result.allocated_run_dir)
    assert accepted is False


def test_refused_pure_plan_writes_nothing(allocation_case):
    request, _plan = allocation_case
    refused_request = replace(request, explicit_user_run_transition=False)
    refused_plan = startup.plan_guided_startup_transaction(refused_request)
    result = allocation.allocate_guided_startup_directory(
        request=refused_request, pure_plan=refused_plan
    )
    assert result.status == "refused_before_allocation"
    assert not Path(refused_request.planned_allocated_run_dir).exists()


def test_creates_missing_but_creatable_output_base(tmp_path, monkeypatch):
    """4J16k11: output_base is intentionally never created before Guided
    Run is pressed (the standard "new analysis" case), so it will not
    exist yet here. Since its immediate parent (tmp_path) already exists,
    allocation must create it -- once every safety check has passed --
    rather than refuse, matching the single-level "creatable" contract
    already enforced when the output destination was selected in the GUI."""
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "not-yet-created-output"
    request, plan = _request_for_paths(
        monkeypatch, source_root=source, output_base=output
    )
    assert not output.exists()
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.status == "allocated_startup_status_written"
    assert result.ok and result.allocated and result.startup_status_written
    assert output.is_dir()
    run_dir = Path(result.allocated_run_dir)
    assert run_dir.is_dir()
    assert run_dir.parent == output


def test_refuses_output_base_whose_parent_is_also_missing(tmp_path, monkeypatch):
    """A not-yet-existing output_base is only creatable one level deep,
    matching the single-level contract enforced at output-destination
    selection time; allocation must not silently create multiple missing
    directory levels."""
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "missing-parent" / "missing-output"
    request, plan = _request_for_paths(
        monkeypatch, source_root=source, output_base=output
    )
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.blocking_issues[0].category == "output_base_missing"
    assert not output.exists()
    assert not (tmp_path / "missing-parent").exists()


def test_refuses_output_base_that_is_not_directory(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output-file"
    output.write_text("not a directory")
    request, plan = _request_for_paths(
        monkeypatch, source_root=source, output_base=output
    )
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.blocking_issues[0].category == "output_base_not_directory"


def test_refuses_existing_planned_child(allocation_case):
    request, plan = allocation_case
    Path(request.planned_allocated_run_dir).mkdir()
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.blocking_issues[0].category == "planned_child_exists"


def test_refuses_source_output_overlap(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    request, plan = _request_for_paths(
        monkeypatch, source_root=source, output_base=source
    )
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.blocking_issues[0].category == "source_output_overlap"


def test_refuses_child_not_directly_under_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    request, plan = _request_for_paths(
        monkeypatch,
        source_root=source,
        output_base=output,
        planned_dir=output / "nested" / "guided_run_20260101T000000Z_abcdef",
    )
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.blocking_issues[0].category == "planned_child_not_direct"


def test_refuses_run_id_path_traversal(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    request, plan = _request_for_paths(
        monkeypatch,
        source_root=source,
        output_base=output,
        run_id="../escape",
        planned_dir=tmp_path / "escape",
    )
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.status == "refused_before_allocation"
    assert not (tmp_path / "escape").exists()


def test_refuses_completed_run_output_base(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "completed"
    source.mkdir()
    output.mkdir()
    (output / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"})
    )
    request, plan = _request_for_paths(
        monkeypatch, source_root=source, output_base=output
    )
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.blocking_issues[0].category == "completed_run_root_prohibited"


def test_refuses_output_base_inside_completed_run_root(tmp_path, monkeypatch):
    source = tmp_path / "source"
    completed = tmp_path / "completed"
    output = completed / "nested-output"
    source.mkdir()
    output.mkdir(parents=True)
    (completed / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"})
    )
    request, plan = _request_for_paths(
        monkeypatch, source_root=source, output_base=output
    )
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.blocking_issues[0].category == "completed_run_root_prohibited"
    assert not Path(request.planned_allocated_run_dir).exists()


def _write_diagnostic_cache_metadata(root: Path):
    artifact = {
        "artifact_contract_version": DIAGNOSTIC_CACHE_SCHEMA_VERSION,
        "cache_id": "cache-1",
        "purpose": DIAGNOSTIC_CACHE_PURPOSE,
        "production_analysis": False,
        "cache_root_path": str(root),
        "source_setup_signature": "source",
        "build_request_signature": "request",
        "diagnostic_scope_signature": "scope",
        "session_chunk_inventory_summary": {
            "preliminary_cache": True,
            "production_analysis": False,
        },
    }
    provenance = {
        "schema_version": DIAGNOSTIC_CACHE_SCHEMA_VERSION,
        "purpose": DIAGNOSTIC_CACHE_PURPOSE,
        "preliminary_cache": True,
        "production_analysis": False,
        "build_request": {},
        "artifact": artifact,
    }
    (root / DIAGNOSTIC_CACHE_ARTIFACT_FILENAME).write_text(json.dumps(artifact))
    (root / DIAGNOSTIC_CACHE_PROVENANCE_FILENAME).write_text(
        json.dumps(provenance)
    )


def test_refuses_guided_diagnostic_cache_output_base(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "cache"
    source.mkdir()
    output.mkdir()
    _write_diagnostic_cache_metadata(output)
    request, plan = _request_for_paths(
        monkeypatch, source_root=source, output_base=output
    )
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.blocking_issues[0].category == "diagnostic_cache_root_prohibited"


def test_refuses_output_base_inside_guided_diagnostic_cache_root(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    output = cache / "nested-output"
    source.mkdir()
    output.mkdir(parents=True)
    _write_diagnostic_cache_metadata(cache)
    request, plan = _request_for_paths(
        monkeypatch, source_root=source, output_base=output
    )
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.blocking_issues[0].category == "diagnostic_cache_root_prohibited"
    assert not Path(request.planned_allocated_run_dir).exists()


def test_second_allocation_does_not_overwrite_status(allocation_case):
    request, plan = allocation_case
    first = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    original = Path(first.startup_status_path).read_bytes()
    second = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert second.status == "refused_before_allocation"
    assert Path(first.startup_status_path).read_bytes() == original


def test_status_write_failure_retains_allocated_directory(
    allocation_case, monkeypatch
):
    request, plan = allocation_case
    original_open = Path.open

    def fail_status(self, mode="r", *args, **kwargs):
        if self.name == startup.GUIDED_STARTUP_STATUS_FILENAME and mode == "xb":
            raise OSError("simulated status failure")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_status)
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    run_dir = Path(request.planned_allocated_run_dir)
    assert result.status == "allocated_status_write_failed"
    assert result.allocated is True
    assert result.startup_status_written is False
    assert run_dir.is_dir()
    assert tuple(run_dir.iterdir()) == ()


def test_no_runner_or_gui_api_is_used(allocation_case, monkeypatch):
    request, plan = allocation_case

    def fail(*_args, **_kwargs):
        raise AssertionError("runner or GUI API must not be called")

    monkeypatch.setattr(os, "system", fail)
    result = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert result.ok
    assert result.no_runner_invoked and result.no_gui_mutation


def test_allocation_module_import_boundary():
    source = Path(allocation.__file__).read_text(encoding="utf-8")
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


# ---------------------------------------------------------------------------
# classify_output_base_reuse_eligibility: shared structural classifier used
# by the GUI output-policy staleness check to distinguish "this app's own
# prior allocation" from a genuinely suspicious pre-existing directory.
# ---------------------------------------------------------------------------


def _write_recognized_run_dir(parent: Path, run_id: str) -> Path:
    child = parent / run_id
    child.mkdir()
    status = {
        "schema_name": "guided_startup_status",
        "schema_version": "v1",
        "run_id": run_id,
        "allocated_run_dir": str(child.resolve()),
    }
    (child / startup.GUIDED_STARTUP_STATUS_FILENAME).write_text(
        json.dumps(status), encoding="utf-8"
    )
    return child


def test_reuse_eligibility_nonexistent_output_base_is_not_reusable(tmp_path):
    missing = tmp_path / "not-yet-created"
    reusable, reason = allocation.classify_output_base_reuse_eligibility(missing)
    assert reusable is False
    assert reason == "output_base_not_directory"


def test_reuse_eligibility_existing_empty_output_base_matches_allocator_contract(
    tmp_path,
):
    """An appeared-but-empty directory carries no proof of prior Guided
    ownership, so the classifier keeps it non-reusable -- matching
    test_new_analysis_output_policy_marks_stale_when_target_appears, which
    requires an empty appeared directory to remain stale/blocked. This is
    deliberately stricter than allocate_guided_startup_directory itself,
    which already tolerates an empty pre-existing output base unconditionally
    (see test_allocates_and_writes_exactly_first_startup_status); the
    classifier only relaxes the GUI's advisory staleness check for bases it
    can prove this app previously allocated into."""
    output_base = tmp_path / "output"
    output_base.mkdir()
    reusable, reason = allocation.classify_output_base_reuse_eligibility(output_base)
    assert reusable is False
    assert reason == "output_base_empty"


def test_reuse_eligibility_one_valid_failed_run_is_reusable(tmp_path):
    output_base = tmp_path / "output"
    output_base.mkdir()
    _write_recognized_run_dir(output_base, "guided_run_20260101T000000Z_aaaaaa")
    reusable, reason = allocation.classify_output_base_reuse_eligibility(output_base)
    assert reusable is True
    assert reason is None


def test_reuse_eligibility_failed_and_successful_runs_are_reusable(tmp_path):
    output_base = tmp_path / "output"
    output_base.mkdir()
    _write_recognized_run_dir(output_base, "guided_run_20260101T000000Z_aaaaaa")
    succeeded = _write_recognized_run_dir(
        output_base, "guided_run_20260101T010000Z_bbbbbb"
    )
    # A completed run carries real deliverables alongside its allocation-time
    # ownership evidence; extra files inside an owned run dir must not
    # disqualify it.
    (succeeded / "run_report.json").write_text(
        json.dumps({"status": "success", "phase": "final"}), encoding="utf-8"
    )
    reusable, reason = allocation.classify_output_base_reuse_eligibility(output_base)
    assert reusable is True
    assert reason is None


def test_reuse_eligibility_loose_file_blocks_reuse(tmp_path):
    output_base = tmp_path / "output"
    output_base.mkdir()
    _write_recognized_run_dir(output_base, "guided_run_20260101T000000Z_aaaaaa")
    (output_base / "notes.txt").write_text("unexpected", encoding="utf-8")
    reusable, reason = allocation.classify_output_base_reuse_eligibility(output_base)
    assert reusable is False
    assert reason == "output_base_contains_unrecognized_entry"


def test_reuse_eligibility_foreign_directory_blocks_reuse(tmp_path):
    output_base = tmp_path / "output"
    output_base.mkdir()
    _write_recognized_run_dir(output_base, "guided_run_20260101T000000Z_aaaaaa")
    (output_base / "unrelated_project").mkdir()
    reusable, reason = allocation.classify_output_base_reuse_eligibility(output_base)
    assert reusable is False
    assert reason == "output_base_contains_unrecognized_entry"


def test_reuse_eligibility_name_matching_directory_without_evidence_blocks_reuse(
    tmp_path,
):
    output_base = tmp_path / "output"
    output_base.mkdir()
    # Looks like a run id, but was never allocated by this app: no
    # guided_startup_status.json at all. Ownership must not be inferred
    # from the name alone.
    (output_base / "guided_run_20260101T020000Z_cccccc").mkdir()
    reusable, reason = allocation.classify_output_base_reuse_eligibility(output_base)
    assert reusable is False
    assert reason == "output_base_contains_unrecognized_entry"


def test_reuse_eligibility_malformed_run_directory_blocks_reuse(tmp_path):
    output_base = tmp_path / "output"
    output_base.mkdir()
    child = output_base / "guided_run_20260101T030000Z_dddddd"
    child.mkdir()
    # Present but wrong: run_id does not match the directory's own name.
    (child / startup.GUIDED_STARTUP_STATUS_FILENAME).write_text(
        json.dumps(
            {
                "schema_name": "guided_startup_status",
                "schema_version": "v1",
                "run_id": "some_other_run_id",
                "allocated_run_dir": str(child.resolve()),
            }
        ),
        encoding="utf-8",
    )
    reusable, reason = allocation.classify_output_base_reuse_eligibility(output_base)
    assert reusable is False
    assert reason == "output_base_contains_unrecognized_entry"


def test_reuse_eligibility_symlink_child_blocks_reuse(tmp_path):
    output_base = tmp_path / "output"
    output_base.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    real = _write_recognized_run_dir(elsewhere, "guided_run_20260101T040000Z_eeeeee")
    try:
        link = output_base / "guided_run_20260101T040000Z_eeeeee"
        os.symlink(str(real), str(link), target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError) as exc:
        pytest.skip(f"Symlinks are not fully supported on this platform: {exc}")
    reusable, reason = allocation.classify_output_base_reuse_eligibility(output_base)
    assert reusable is False
    assert reason == "output_base_contains_unrecognized_entry"
