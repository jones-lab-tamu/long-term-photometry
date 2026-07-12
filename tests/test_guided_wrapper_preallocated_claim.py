from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import photometry_pipeline.guided_startup_allocation as allocation
import photometry_pipeline.guided_startup_claim as claim
import photometry_pipeline.guided_startup_materialization as materialization
import photometry_pipeline.guided_startup_transaction as startup
import tools.run_full_pipeline_deliverables as wrapper
from gui.run_report_parser import classify_completed_run_candidate
from tests.test_guided_startup_allocation import allocation_case


@pytest.fixture
def prepared_case(allocation_case):
    request, plan = allocation_case
    allocated = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    prepared = materialization.materialize_guided_startup_artifacts(
        request=request,
        pure_plan=plan,
        allocation_result=allocated,
    )
    assert prepared.ok
    return request, plan, allocated, prepared


def _args(request, prepared, **changes):
    values = dict(
        input=request.source_root_canonical,
        out=prepared.allocated_run_dir,
        out_base=None,
        config=prepared.config_path,
        format="rwd",
        mode="phasic",
        run_type="full",
        overwrite=False,
        include_rois=None,
        exclude_rois=None,
        traces_only=False,
        acquisition_mode=None,
        preview_first_n=None,
        validate_only=False,
        discover=False,
        guided_candidate_manifest=prepared.manifest_path,
        guided_preallocated_run_dir=True,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _validate(prepared_case, **changes):
    request, _plan, _allocated, prepared = prepared_case
    return wrapper.validate_guided_preallocated_mode_args(
        _args(request, prepared, **changes)
    )


def test_valid_preallocated_directory_validates_and_claims_once(prepared_case):
    validation = _validate(prepared_case)
    assert validation.accepted
    result = claim.claim_guided_preallocated_startup(
        validation, claimed_utc="2026-01-01T00:00:00Z", process_id=123
    )
    assert result.status == "claimed"
    assert result.claimed is True
    payload = json.loads(Path(result.claim_path).read_bytes())
    assert payload["schema_name"] == claim.GUIDED_STARTUP_WRAPPER_CLAIM_SCHEMA_NAME
    assert payload["completed_run_claim"] is False
    assert payload["startup_transaction_identity"] == (
        validation.startup_transaction_identity
    )
    assert not (Path(validation.run_dir) / "status.json").exists()

    replay = claim.claim_guided_preallocated_startup(
        validation, claimed_utc="2026-01-01T00:00:01Z", process_id=124
    )
    assert replay.claimed is False
    assert replay.blocking_issues[0].category == "startup_already_claimed"


def test_current_native_artifact_mutation_refuses_before_wrapper_claim(prepared_case):
    request, _plan, _allocated, prepared = prepared_case
    run_dir = Path(prepared.allocated_run_dir)
    native_path = run_dir / startup.GUIDED_PER_ROI_CORRECTION_FILENAME
    native_bytes = b'{"authorized":"native-correction-test"}\n'
    native_path.write_bytes(native_bytes)
    provenance_path = run_dir / startup.GUIDED_STARTUP_PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["startup_contract_version"] = (
        startup.GUIDED_STARTUP_TRANSACTION_CONTRACT_VERSION
    )
    provenance["serialized_native_correction_sha256"] = hashlib.sha256(
        native_bytes
    ).hexdigest()
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    accepted = claim.validate_guided_preallocated_startup(
        input_dir=request.source_root_canonical,
        output_dir=prepared.allocated_run_dir,
        config_path=prepared.config_path,
        manifest_path=prepared.manifest_path,
    )
    assert accepted.accepted

    native_path.write_bytes(native_bytes + b" ")
    refused = claim.validate_guided_preallocated_startup(
        input_dir=request.source_root_canonical,
        output_dir=prepared.allocated_run_dir,
        config_path=prepared.config_path,
        manifest_path=prepared.manifest_path,
    )
    assert not refused.accepted
    assert refused.blocking_issues[0].category == "startup_artifact_hash_mismatch"


@pytest.mark.parametrize(
    "changes",
    (
        {"guided_candidate_manifest": None},
        {"out": None},
        {"out_base": "different"},
        {"overwrite": True},
        {"mode": "tonic"},
        {"mode": "both"},
        {"preview_first_n": 1},
        {"discover": True},
        {"validate_only": True},
        {"traces_only": True},
        {"acquisition_mode": "continuous"},
        {"format": "npm"},
        {"run_type": "tuning_prep"},
    ),
)
def test_internal_flag_conflicts_refuse(prepared_case, changes):
    with pytest.raises(RuntimeError, match="handoff refused"):
        _validate(prepared_case, **changes)


@pytest.mark.parametrize(
    "filename",
    (
        startup.GUIDED_STARTUP_STATUS_FILENAME,
        startup.GUIDED_CANDIDATE_MANIFEST_FILENAME,
        startup.GUIDED_CONFIG_EFFECTIVE_FILENAME,
        startup.GUIDED_STARTUP_PROVENANCE_FILENAME,
        startup.GUIDED_COMMAND_RECORD_FILENAME,
    ),
)
def test_missing_required_startup_file_refuses(prepared_case, filename):
    request, _plan, _allocated, prepared = prepared_case
    (Path(prepared.allocated_run_dir) / filename).unlink()
    validation = wrapper.validate_guided_preallocated_mode_args(
        _args(request, prepared)
    )
    assert validation.accepted is False
    assert not (
        Path(prepared.allocated_run_dir)
        / claim.GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME
    ).exists()


@pytest.mark.parametrize(
    "name",
    ("status.json", "MANIFEST.json", "run_report.json", "qc", "cache"),
)
def test_existing_production_artifact_refuses(prepared_case, name):
    request, _plan, _allocated, prepared = prepared_case
    path = Path(prepared.allocated_run_dir) / name
    if "." in name:
        path.write_text("{}")
    else:
        path.mkdir()
    validation = wrapper.validate_guided_preallocated_mode_args(
        _args(request, prepared)
    )
    assert validation.accepted is False


def test_manifest_and_config_must_be_inside_preallocated_directory(
    prepared_case, tmp_path
):
    request, _plan, _allocated, prepared = prepared_case
    outside_manifest = tmp_path / "guided_candidate_manifest.json"
    outside_manifest.write_bytes(Path(prepared.manifest_path).read_bytes())
    outside_config = tmp_path / "config_effective.yaml"
    outside_config.write_bytes(Path(prepared.config_path).read_bytes())
    assert not wrapper.validate_guided_preallocated_mode_args(
        _args(request, prepared, guided_candidate_manifest=str(outside_manifest))
    ).accepted
    assert not wrapper.validate_guided_preallocated_mode_args(
        _args(request, prepared, config=str(outside_config))
    ).accepted


@pytest.mark.parametrize(
    "filename",
    (
        startup.GUIDED_CANDIDATE_MANIFEST_FILENAME,
        startup.GUIDED_CONFIG_EFFECTIVE_FILENAME,
        startup.GUIDED_COMMAND_RECORD_FILENAME,
    ),
)
def test_provenance_hash_mismatch_refuses(prepared_case, filename):
    request, _plan, _allocated, prepared = prepared_case
    path = Path(prepared.allocated_run_dir) / filename
    path.write_bytes(path.read_bytes() + b"\n")
    validation = wrapper.validate_guided_preallocated_mode_args(
        _args(request, prepared)
    )
    assert validation.accepted is False


def test_command_without_preallocated_flag_refuses(prepared_case):
    request, _plan, _allocated, prepared = prepared_case
    run_dir = Path(prepared.allocated_run_dir)
    command_path = run_dir / startup.GUIDED_COMMAND_RECORD_FILENAME
    content = command_path.read_text().replace(
        "--guided-preallocated-run-dir\n", ""
    )
    command_path.write_text(content)
    provenance_path = run_dir / startup.GUIDED_STARTUP_PROVENANCE_FILENAME
    provenance = json.loads(provenance_path.read_bytes())
    provenance["command_record_sha256"] = hashlib.sha256(
        command_path.read_bytes()
    ).hexdigest()
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n"
    )
    validation = wrapper.validate_guided_preallocated_mode_args(
        _args(request, prepared)
    )
    assert validation.accepted is False


def test_claimed_prepared_directory_remains_completed_run_ineligible(prepared_case):
    validation = _validate(prepared_case)
    claimed = claim.claim_guided_preallocated_startup(
        validation, claimed_utc="2026-01-01T00:00:00Z"
    )
    accepted, _reason = classify_completed_run_candidate(validation.run_dir)
    assert claimed.claimed
    assert accepted is False


def test_wrapper_orders_live_verification_before_claim_and_status_write():
    source = Path(wrapper.__file__).read_text(encoding="utf-8")
    main_source = source[source.index("def main():") :]
    verify_index = main_source.index("verify_guided_manifest_before_output(args)")
    claim_index = main_source.index("claim_guided_preallocated_startup(")
    initial_write_index = main_source.index('# Initial write (phase="running")')
    status_index = main_source.index(
        "_write_status_json(status_path, status_data)", initial_write_index
    )
    assert verify_index < claim_index < status_index


def test_discovery_failure_before_status_does_not_consume_claim(
    prepared_case, monkeypatch
):
    request, _plan, _allocated, prepared = prepared_case
    args = _args(request, prepared, discover=True)
    accepted_validation = claim.validate_guided_preallocated_startup(
        input_dir=args.input,
        output_dir=args.out,
        config_path=args.config,
        manifest_path=args.guided_candidate_manifest,
    )
    assert accepted_validation.accepted
    live_verified = []
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)
    monkeypatch.setattr(
        wrapper,
        "validate_guided_preallocated_mode_args",
        lambda _args: accepted_validation,
    )
    monkeypatch.setattr(
        wrapper,
        "verify_guided_manifest_before_output",
        lambda _args: live_verified.append(True),
    )

    def claim_must_not_run(*_args, **_kwargs):
        raise AssertionError("claim must not occur before discovery preflight")

    monkeypatch.setattr(
        wrapper, "claim_guided_preallocated_startup", claim_must_not_run
    )
    import photometry_pipeline.discovery as discovery

    monkeypatch.setattr(
        discovery,
        "discover_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated discovery failure")
        ),
    )
    with pytest.raises(SystemExit):
        wrapper.main()
    run_dir = Path(prepared.allocated_run_dir)
    assert live_verified == [True]
    assert not (run_dir / claim.GUIDED_STARTUP_WRAPPER_CLAIM_FILENAME).exists()
    assert not (run_dir / "status.json").exists()
    assert {item.name for item in run_dir.iterdir()} == {
        startup.GUIDED_STARTUP_STATUS_FILENAME,
        startup.GUIDED_CANDIDATE_MANIFEST_FILENAME,
        startup.GUIDED_CONFIG_EFFECTIVE_FILENAME,
        startup.GUIDED_STARTUP_PROVENANCE_FILENAME,
        startup.GUIDED_COMMAND_RECORD_FILENAME,
    }
    assert claim.validate_guided_preallocated_startup(
        input_dir=args.input,
        output_dir=args.out,
        config_path=args.config,
        manifest_path=args.guided_candidate_manifest,
    ).accepted


def test_status_transaction_identity_must_match_provenance_when_present(
    prepared_case,
):
    request, _plan, _allocated, prepared = prepared_case
    status_path = (
        Path(prepared.allocated_run_dir) / startup.GUIDED_STARTUP_STATUS_FILENAME
    )
    status = json.loads(status_path.read_bytes())
    status["startup_transaction_identity"] = "d" * 64
    status_path.write_text(
        json.dumps(status, sort_keys=True, separators=(",", ":")) + "\n"
    )
    validation = claim.validate_guided_preallocated_startup(
        input_dir=request.source_root_canonical,
        output_dir=prepared.allocated_run_dir,
        config_path=prepared.config_path,
        manifest_path=prepared.manifest_path,
    )
    assert not validation.accepted
    assert validation.blocking_issues[0].category == "startup_provenance_invalid"


def test_preallocated_setup_bypasses_root_creation_and_cleanup():
    source = Path(wrapper.__file__).read_text(encoding="utf-8")
    assert "if args.guided_preallocated_run_dir:" in source
    assert "if not args.guided_preallocated_run_dir:\n        os.makedirs(" in source
    assert "Guided startup-preallocated " in source


def test_no_preallocated_mode_is_unchanged():
    args = SimpleNamespace(guided_preallocated_run_dir=False)
    assert wrapper.validate_guided_preallocated_mode_args(args) is None


def test_claim_module_import_boundary():
    source = Path(claim.__file__).read_text(encoding="utf-8")
    prohibited = (
        "gui",
        "subprocess",
        "photometry_pipeline.pipeline",
        "tools.run_full_pipeline_deliverables",
        "analyze_photometry",
    )
    tree = __import__("ast").parse(source)
    imported = set()
    for node in __import__("ast").walk(tree):
        if isinstance(node, __import__("ast").Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, __import__("ast").ImportFrom):
            imported.add(node.module or "")
    assert not any(
        name == marker or name.startswith(f"{marker}.")
        for name in imported
        for marker in prohibited
    )
