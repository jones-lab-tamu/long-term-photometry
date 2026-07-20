"""Independent cross-contract proof for NPM Guided manifest verification.

test_guided_manifest_runner_integration_npm.py proves build_guided_manifest_
current_facts(..., source_format="npm") agrees with itself: it uses the
function's own output to hand-build the manifest it then re-verifies. That
is circular -- it cannot detect a reconstruction that is wrong in a way that
is still internally self-consistent.

This module instead builds a real NPM Guided manifest through the actual
independent production path -- validation -> map_guided_npm_validation_
outcome_to_execution_intent -> build_guided_npm_execution_authority ->
compile_npm_generic_execution_payloads -> the same
serialize_guided_candidate_manifest_payload_to_bytes real startup
materialization uses -- and proves build_guided_manifest_current_facts's
live NPM reconstruction agrees with that independently produced authority,
not with itself.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import photometry_pipeline.guided_manifest_current_facts as current_facts_module
import photometry_pipeline.guided_manifest_verification as verification
import tools.run_full_pipeline_deliverables as wrapper
from photometry_pipeline.config import Config
from photometry_pipeline.guided_execution_payloads import (
    build_guided_execution_startup_mapping_contract,
)
from photometry_pipeline.guided_manifest_current_facts import (
    build_guided_manifest_current_facts,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisExecutionIntent,
    compute_guided_local_preview_source_setup_signature,
)
from photometry_pipeline.guided_npm_execution_authority import (
    GuidedNpmRoiAuthority,
    build_guided_npm_execution_authority,
)
from photometry_pipeline.guided_npm_startup_bridge import (
    compile_npm_generic_execution_payloads,
)
from photometry_pipeline.guided_startup_transaction import (
    serialize_guided_candidate_manifest_payload_to_bytes,
)
from photometry_pipeline.guided_backend_validation_workflow import (
    validate_current_guided_draft_for_backend,
)
from photometry_pipeline.io.npm_contract import NpmParserContract, inspect_npm_csv
from photometry_pipeline.io.npm_source_snapshot import (
    build_npm_source_candidate_snapshot,
)
from photometry_pipeline.pipeline import Pipeline

from tests.test_guided_backend_validation_materialization import (
    _valid_npm_stage2c_draft,
)
from tests.test_guided_npm_production_mapping import _map
from tests.test_guided_npm_startup_bridge import _npm_validation_fixture_kwargs


_TWO_ROI_CSV = (
    "Timestamp,LedState,Region2G,Region10G\n"
    "{t0:.1f},1,10.0,20.0\n"
    "{t0half:.1f},2,100.0,200.0\n"
    "{t1:.1f},1,11.0,21.0\n"
    "{t1half:.1f},2,101.0,201.0\n"
    "{t2:.1f},1,12.0,22.0\n"
    "{t2half:.1f},2,102.0,202.0\n"
)


def _two_roi_csv(t0: float) -> str:
    return _TWO_ROI_CSV.format(
        t0=t0, t0half=t0 + 0.5, t1=t0 + 1.0, t1half=t0 + 1.5, t2=t0 + 2.0, t2half=t0 + 2.5
    )


def _npm_config_from_parser(parser: NpmParserContract) -> Config:
    return Config(
        allow_partial_final_chunk=parser.allow_partial_final_chunk,
        target_fs_hz=parser.target_fs_hz,
        chunk_duration_sec=parser.session_duration_sec,
        adapter_value_nan_policy=parser.adapter_value_nan_policy,
        npm_time_axis=parser.npm_time_axis,
        npm_system_ts_col=parser.npm_system_ts_col,
        npm_computer_ts_col=parser.npm_computer_ts_col,
        npm_led_col=parser.npm_led_col,
        npm_region_prefix=parser.npm_region_prefix,
        npm_region_suffix=parser.npm_region_suffix,
        timestamp_cv_max=parser.timestamp_cv_max,
    )


def _real_npm_startup_manifest(tmp_path: Path):
    """Build a real, independently-produced NPM Guided candidate manifest
    through the actual production path -- two NPM files, two ROIs -- with
    no involvement of build_guided_manifest_current_facts anywhere in its
    construction."""
    draft = _valid_npm_stage2c_draft(tmp_path)
    draft.execution_intent = GuidedNewAnalysisExecutionIntent(execution_mode="both")

    source_root = Path(draft.input_source_path)
    first_file = next(source_root.glob("*.csv"))
    first_file.write_text(_two_roi_csv(100.0), encoding="utf-8")
    second_file = source_root / "photometryData2026-06-30T13_00_00.csv"
    second_file.write_text(_two_roi_csv(2000.0), encoding="utf-8")

    draft.discovered_roi_ids = ["Region0", "Region1"]
    draft.included_roi_ids = ["Region0", "Region1"]
    draft.excluded_roi_ids = []
    draft.dataset_contract_snapshot = replace(
        draft.dataset_contract_snapshot,
        source_identity=replace(
            draft.dataset_contract_snapshot.source_identity,
            discovered_roi_ids=("Region0", "Region1"),
            included_roi_ids=("Region0", "Region1"),
        ),
    )
    local_preview_signature = compute_guided_local_preview_source_setup_signature(draft)
    first_choice = replace(
        draft.per_roi_correction_strategy_choices[0],
        source_setup_signature=local_preview_signature,
    )
    draft.per_roi_correction_strategy_choices = [
        first_choice,
        replace(first_choice, roi_id="Region1"),
    ]

    kwargs = _npm_validation_fixture_kwargs()
    outcome = validate_current_guided_draft_for_backend(
        draft, validation_revision=4, **kwargs
    )
    assert outcome.status == "validator_accepted", outcome.blocking_issues
    assert outcome.compile_result is not None

    mapped = _map(outcome)
    intent = mapped.intent
    authority = build_guided_npm_execution_authority(intent)

    contract = build_guided_execution_startup_mapping_contract()
    result = compile_npm_generic_execution_payloads(
        intent, authority, startup_mapping_contract=contract
    )
    assert result.ok is True, result.blocking_issues
    manifest_payload = result.candidate_manifest_payload
    assert manifest_payload is not None
    assert len(manifest_payload.discovered_roi_ids) == 2
    assert len(manifest_payload.candidate_files) == 2

    config = _npm_config_from_parser(kwargs["parser_contract"])
    return source_root, config, manifest_payload, intent, authority, first_file


# ---------------------------------------------------------------------------
# 1. Independent cross-contract identity equality
# ---------------------------------------------------------------------------


def test_current_facts_matches_independently_produced_startup_manifest(tmp_path):
    source_root, config, manifest_payload, _intent, _authority, _first_file = (
        _real_npm_startup_manifest(tmp_path)
    )

    facts = build_guided_manifest_current_facts(
        source_root=source_root,
        config=config,
        manifest_included_roi_ids=manifest_payload.included_roi_ids,
        source_format="npm",
    )
    inventory = facts.current_roi_inventory

    assert inventory.parser_contract_digest == manifest_payload.parser_contract_digest
    assert inventory.discovered_roi_ids == manifest_payload.discovered_roi_ids
    assert inventory.included_roi_ids == manifest_payload.included_roi_ids
    assert inventory.excluded_roi_ids == manifest_payload.excluded_roi_ids
    assert (
        inventory.strict_roi_inventory_digest
        == manifest_payload.strict_roi_inventory_digest
    )

    # Candidate set/content digest: build_guided_manifest_current_facts does
    # not expose these on GuidedManifestCurrentFacts directly (mirroring
    # RWD's own current-facts contract), so compare against an independent
    # fresh snapshot of the same live source root using the same builder
    # build_guided_manifest_current_facts itself calls internally.
    live_snapshot = build_npm_source_candidate_snapshot(str(source_root))
    assert (
        live_snapshot.source_candidate_set_digest
        == manifest_payload.source_candidate_set_digest
    )
    assert (
        live_snapshot.source_candidate_content_digest
        == manifest_payload.source_candidate_content_digest
    )

    # Ordered candidate path/size/content-digest entries.
    current_paths = tuple(item.canonical_relative_path for item in facts.current_candidates)
    manifest_paths = tuple(item.canonical_relative_path for item in manifest_payload.candidate_files)
    assert current_paths == manifest_paths
    live_by_path = {item.canonical_relative_path: item for item in live_snapshot.candidates}
    for entry in manifest_payload.candidate_files:
        live = live_by_path[entry.canonical_relative_path]
        assert live.size_bytes == entry.size_bytes
        assert live.sha256_content_digest == entry.sha256_content_digest


# ---------------------------------------------------------------------------
# 2. Physical-to-canonical mapping orientation
# ---------------------------------------------------------------------------


def test_physical_to_canonical_mapping_orientation_and_reconstruction(
    tmp_path, monkeypatch
):
    source_root, config, manifest_payload, _intent, _authority, first_file = (
        _real_npm_startup_manifest(tmp_path)
    )

    # Ground truth, independent of build_guided_manifest_current_facts:
    # inspect_npm_csv's own returned contract.
    parser_contract = NpmParserContract.from_config(config)
    ground_truth = inspect_npm_csv(str(first_file), parser_contract)
    # inspect_npm_csv's own documented orientation (io/npm_contract.py):
    # (canonical_roi_id, physical_column) pairs, canonical id built as
    # f"{region_prefix}{position}" enumerated over the sorted physical
    # columns -- not (physical_column, canonical_roi_id).
    assert ground_truth.physical_to_canonical_roi_mapping == (
        ("Region0", "Region2G"),
        ("Region1", "Region10G"),
    )
    assert ground_truth.roi_ids == ("Region0", "Region1")

    # Spy on the exact constructor build_guided_manifest_current_facts's
    # NPM branch calls, to assert on the *reconstructed* mapping directly --
    # not only on the final strict_roi_inventory_digest hash equality
    # already proven in the sibling cross-contract-identity test.
    captured = {}
    real_cls = current_facts_module.GuidedNpmRoiAuthority

    def spy(**kwargs):
        captured.update(kwargs)
        return real_cls(**kwargs)

    monkeypatch.setattr(current_facts_module, "GuidedNpmRoiAuthority", spy)

    facts = build_guided_manifest_current_facts(
        source_root=source_root,
        config=config,
        manifest_included_roi_ids=manifest_payload.included_roi_ids,
        source_format="npm",
    )
    assert facts.current_roi_inventory.discovered_roi_ids == ("Region0", "Region1")

    assert "physical_to_canonical_roi_mapping" in captured
    reconstructed_mapping = captured["physical_to_canonical_roi_mapping"]
    assert len(reconstructed_mapping) == 2
    for entry in reconstructed_mapping:
        assert isinstance(entry, current_facts_module.GuidedNpmRoiMappingEntry)
    by_canonical = {entry.canonical_roi_id: entry.physical_source_column for entry in reconstructed_mapping}
    assert by_canonical == {"Region0": "Region2G", "Region1": "Region10G"}
    for entry in reconstructed_mapping:
        assert entry.canonical_roi_id in {"Region0", "Region1"}
        assert entry.physical_source_column in {"Region2G", "Region10G"}

    assert captured["complete_physical_source_columns"] == ("Region2G", "Region10G")
    assert captured["complete_canonical_roi_ids"] == ("Region0", "Region1")


# ---------------------------------------------------------------------------
# 3. Wrapper boundary with the real, independently-produced manifest
# ---------------------------------------------------------------------------


def test_wrapper_accepts_real_startup_produced_npm_manifest(tmp_path):
    source_root, config, manifest_payload, _intent, _authority, _first_file = (
        _real_npm_startup_manifest(tmp_path)
    )

    artifact = serialize_guided_candidate_manifest_payload_to_bytes(manifest_payload)
    manifest_path = tmp_path / "guided_candidate_manifest.json"
    manifest_path.write_bytes(artifact.content_bytes)

    # load_guided_candidate_manifest must accept the real serialized bytes
    # unmodified -- proves the on-disk shape produced by the real startup
    # serializer round-trips through the wrapper's own loader.
    loaded = verification.load_guided_candidate_manifest(str(manifest_path))
    assert loaded.accepted is True, loaded.blocking_issues

    config_path = tmp_path / "config.yaml"
    config_dict = {
        "allow_partial_final_chunk": config.allow_partial_final_chunk,
        "target_fs_hz": config.target_fs_hz,
        "chunk_duration_sec": config.chunk_duration_sec,
        "adapter_value_nan_policy": config.adapter_value_nan_policy,
        "npm_time_axis": config.npm_time_axis,
        "npm_system_ts_col": config.npm_system_ts_col,
        "npm_computer_ts_col": config.npm_computer_ts_col,
        "npm_led_col": config.npm_led_col,
        "npm_region_prefix": config.npm_region_prefix,
        "npm_region_suffix": config.npm_region_suffix,
        "timestamp_cv_max": config.timestamp_cv_max,
    }
    config_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    args = SimpleNamespace(
        input=str(source_root),
        config=str(config_path),
        format="npm",
        mode="phasic",
        run_type="full",
        traces_only=False,
        discover=False,
        validate_only=False,
        overwrite=False,
        preview_first_n=None,
        include_rois=None,
        exclude_rois=None,
        acquisition_mode=None,
        guided_candidate_manifest=str(manifest_path),
    )
    facts, verified = wrapper.verify_guided_manifest_before_output(args)
    assert verified.accepted is True, verified.blocking_issues
    assert verified.blocking_issues == ()
    assert facts.current_roi_inventory.discovered_roi_ids == manifest_payload.discovered_roi_ids


# ---------------------------------------------------------------------------
# 4. Pipeline child re-verification with the real, independently-produced
#    manifest
# ---------------------------------------------------------------------------


def test_pipeline_reverification_accepts_real_startup_produced_npm_manifest(
    tmp_path, monkeypatch
):
    source_root, config, manifest_payload, _intent, _authority, _first_file = (
        _real_npm_startup_manifest(tmp_path)
    )

    artifact = serialize_guided_candidate_manifest_payload_to_bytes(manifest_payload)
    manifest_path = tmp_path / "guided_candidate_manifest.json"
    manifest_path.write_bytes(artifact.content_bytes)

    pipeline = Pipeline(config, mode="phasic")

    class StopAfterVerification(RuntimeError):
        pass

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normal discovery must be skipped")

    monkeypatch.setattr(pipeline, "discover_files", forbidden)
    monkeypatch.setattr(
        pipeline,
        "_resolve_representative_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StopAfterVerification()),
    )
    with pytest.raises(StopAfterVerification):
        pipeline.run(
            str(source_root),
            str(tmp_path / "out"),
            force_format="npm",
            guided_manifest_path=str(manifest_path),
        )
    # Reaching _resolve_representative_session (and not an earlier
    # RuntimeError("Guided manifest verification refused: ...")) proves
    # Pipeline.run's own re-verification -- using source_format=force_format
    # -- accepted the real, independently-produced NPM manifest.
    expected_paths = {
        str(Path(source_root) / Path(item.canonical_relative_path))
        for item in manifest_payload.candidate_files
    }
    import os as _os

    assert {_os.path.normcase(_os.path.abspath(p)) for p in pipeline.file_list} == {
        _os.path.normcase(_os.path.abspath(p)) for p in expected_paths
    }
