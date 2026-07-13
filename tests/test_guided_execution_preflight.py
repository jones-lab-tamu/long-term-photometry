from __future__ import annotations

import ast
from dataclasses import fields, replace
import os
from pathlib import Path

import pytest

import photometry_pipeline.guided_backend_validation_request as contracts
import photometry_pipeline.guided_execution_preflight as preflight
import photometry_pipeline.guided_production_mapping as mapping
from photometry_pipeline.guided_backend_validation_workflow import (
    GUIDED_BACKEND_RWD_TIME_COLUMN_CANDIDATES,
)
from photometry_pipeline.io.rwd_contract import (
    RwdHeaderParsingContract,
    compute_rwd_header_parsing_contract_digest,
)
from photometry_pipeline.io.rwd_source_snapshot import (
    build_rwd_source_candidate_snapshot,
)
from tests.test_guided_backend_validator import _request as _base_request


def _unchecked(instance, **changes):
    result = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            result,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return result


def _write_session(
    root: Path, name: str, rois=("ROI0", "ROI1"), time_col="Time(s)"
) -> Path:
    path = root / name / "fluorescence.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [time_col]
    row = ["0"]
    for roi in rois:
        columns.extend((f"{roi}-410", f"{roi}-470"))
        row.extend(("1", "2"))
    path.write_text(",".join(columns) + "\n" + ",".join(row) + "\n", encoding="utf-8")
    return path


def _parser(time_column_candidates=("Time(s)",)):
    return RwdHeaderParsingContract(
        time_column_candidates=time_column_candidates,
        uv_suffix_candidates=("-410",),
        signal_suffix_candidates=("-470",),
    )


def _intent(
    tmp_path: Path,
    *,
    rois=("ROI0", "ROI1"),
    time_col="Time(s)",
    time_column_candidates=("Time(s)",),
):
    root = tmp_path / "source"
    _write_session(root, "2025_01_01-00_00_00", rois=rois, time_col=time_col)
    _write_session(root, "2025_01_01-00_10_00", rois=rois, time_col=time_col)
    snapshot = build_rwd_source_candidate_snapshot(str(root))
    parser_contract = _parser(time_column_candidates=time_column_candidates)
    parser_digest = compute_rwd_header_parsing_contract_digest(parser_contract)
    request = _base_request()
    source = _unchecked(
        request.source,
        source_root_canonical=snapshot.source_root_canonical,
        source_root_path_style=snapshot.source_root_path_style,
        snapshot_schema_name=snapshot.snapshot_schema_name,
        snapshot_schema_version=snapshot.snapshot_schema_version,
        discovery_rule_version=snapshot.discovery_rule_version,
        path_canonicalization_version=snapshot.path_canonicalization_version,
        relative_path_rule_version=snapshot.relative_path_rule_version,
        ignored_files_policy=snapshot.ignored_files_policy,
        source_candidate_set_digest=snapshot.source_candidate_set_digest,
        source_candidate_content_digest=snapshot.source_candidate_content_digest,
        candidate_files=tuple(
            contracts.GuidedBackendSourceCandidateFile(
                item.canonical_relative_path,
                item.size_bytes,
                item.sha256_content_digest,
            )
            for item in snapshot.candidates
        ),
    )
    parser = _unchecked(
        request.parser,
        schema_name=parser_contract.schema_name,
        schema_version=parser_contract.schema_version,
        header_search_line_limit=parser_contract.header_search_line_limit,
        time_column_candidates=parser_contract.time_column_candidates,
        uv_suffix_candidates=parser_contract.uv_suffix_candidates,
        signal_suffix_candidates=parser_contract.signal_suffix_candidates,
        column_normalization_rule=parser_contract.column_normalization_rule,
        roi_name_rule=parser_contract.roi_name_rule,
        ambiguity_policy=parser_contract.ambiguity_policy,
        parser_contract_digest=parser_digest,
    )
    roi = _unchecked(
        request.roi_scope,
        inventory_source_content_digest=snapshot.source_candidate_content_digest,
    )
    acquisition_dataset = _unchecked(
        request.acquisition_dataset,
        rwd_time_col=time_col,
        semantic_values=(
            contracts.GuidedBackendTypedFieldValue(
                field_name="rwd_time_col",
                value_type="str",
                value=time_col,
            ),
        ),
    )
    request = _unchecked(
        request,
        source=source,
        parser=parser,
        roi_scope=roi,
        acquisition_dataset=acquisition_dataset,
    )
    identity = contracts.compute_guided_backend_validation_request_identity(request)
    build = mapping.build_application_build_identity(
        distribution_name="photometry-pipeline",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="abc123",
        source_tree_state="clean",
    )
    mapped = mapping.map_guided_validation_request_to_execution_intent(
        request,
        canonical_request_identity=identity,
        application_build_identity=build,
        mapping_contract=mapping.build_guided_production_mapping_contract(),
    )
    assert isinstance(mapped, mapping.GuidedProductionMappingSuccess)
    return mapped.intent, root


def _candidate(intent):
    request = preflight.derive_candidate_manifest_preflight_request_from_intent(
        intent
    )
    return request, preflight.run_candidate_manifest_execution_preflight(request)


def _roi(intent, candidate_result):
    request = preflight.derive_roi_execution_preflight_request_from_intent(
        intent,
        accepted_candidate_preflight_identity=(
            candidate_result.canonical_preflight_identity
        ),
    )
    return request, preflight.run_roi_execution_preflight(request)


def _category(result):
    return result.blocking_issues[0].category


def test_candidate_preflight_accepts_unchanged_manifest(tmp_path):
    intent, _ = _intent(tmp_path)
    request, result = _candidate(intent)
    assert result.accepted is True
    assert result.status == "accepted"
    assert result.actual_candidates == request.expected_candidates
    assert result.canonical_preflight_identity
    assert preflight.compute_guided_candidate_preflight_identity(result) == (
        result.canonical_preflight_identity
    )


@pytest.mark.parametrize("change,expected", [("missing", "candidate_file_missing"), ("extra", "candidate_file_extra")])
def test_candidate_preflight_refuses_missing_or_extra(tmp_path, change, expected):
    intent, root = _intent(tmp_path)
    request = preflight.derive_candidate_manifest_preflight_request_from_intent(intent)
    if change == "missing":
        (root / "2025_01_01-00_10_00" / "fluorescence.csv").unlink()
    else:
        _write_session(root, "2025_01_01-00_20_00")
    result = preflight.run_candidate_manifest_execution_preflight(request)
    assert _category(result) == expected
    assert result.canonical_preflight_identity is None


def test_candidate_preflight_refuses_changed_content_and_size(tmp_path):
    intent, root = _intent(tmp_path)
    request = preflight.derive_candidate_manifest_preflight_request_from_intent(intent)
    target = root / "2025_01_01-00_00_00" / "fluorescence.csv"
    original = target.read_text(encoding="utf-8")
    target.write_text(original + "1,2,3,4,5\n", encoding="utf-8")
    result = preflight.run_candidate_manifest_execution_preflight(request)
    assert _category(result) == "candidate_file_size_mismatch"


def test_candidate_preflight_refuses_same_size_content_change(tmp_path):
    intent, root = _intent(tmp_path)
    request = preflight.derive_candidate_manifest_preflight_request_from_intent(intent)
    target = root / "2025_01_01-00_00_00" / "fluorescence.csv"
    raw = target.read_bytes()
    target.write_bytes(raw[:-2] + (b"9\n"))
    result = preflight.run_candidate_manifest_execution_preflight(request)
    assert _category(result) == "candidate_file_digest_mismatch"


def test_candidate_order_and_duplicate_refuse(tmp_path):
    intent, _ = _intent(tmp_path)
    request = preflight.derive_candidate_manifest_preflight_request_from_intent(intent)
    reordered = replace(request, expected_candidates=tuple(reversed(request.expected_candidates)))
    assert _category(preflight.run_candidate_manifest_execution_preflight(reordered)) == "candidate_relative_path_mismatch"
    duplicate = replace(
        request,
        expected_candidates=(request.expected_candidates[0], request.expected_candidates[0]),
    )
    assert _category(preflight.run_candidate_manifest_execution_preflight(duplicate)) == "candidate_duplicate_path"


def test_candidate_contract_and_format_refuse(tmp_path):
    intent, _ = _intent(tmp_path)
    request = preflight.derive_candidate_manifest_preflight_request_from_intent(intent)
    assert _category(
        preflight.run_candidate_manifest_execution_preflight(
            replace(request, source_format="npm")
        )
    ) == "source_format_unsupported"
    assert _category(
        preflight.run_candidate_manifest_execution_preflight(
            replace(request, discovery_rule_version="wrong")
        )
    ) == "candidate_preflight_contract_unavailable"


def test_candidate_identity_is_deterministic(tmp_path):
    intent, _ = _intent(tmp_path)
    _, first = _candidate(intent)
    _, second = _candidate(intent)
    assert first.canonical_preflight_identity == second.canonical_preflight_identity
    changed = replace(
        first,
        actual_candidates=(
            replace(first.actual_candidates[0], sha256_content_digest="d" * 64),
        )
        + first.actual_candidates[1:],
        canonical_preflight_identity="0" * 64,
    )
    assert preflight.compute_guided_candidate_preflight_identity(changed) != (
        first.canonical_preflight_identity
    )


def test_roi_preflight_accepts_all_candidates(tmp_path):
    intent, _ = _intent(tmp_path)
    _, candidate = _candidate(intent)
    request, result = _roi(intent, candidate)
    assert result.accepted is True
    assert result.actual_discovered_roi_ids == ("ROI0", "ROI1")
    assert result.actual_included_roi_ids == ("ROI0",)
    assert result.actual_excluded_roi_ids == ("ROI1",)
    assert result.actual_strict_roi_inventory_digest == (
        request.expected_strict_roi_inventory_digest
    )
    assert preflight.compute_guided_roi_preflight_identity(result) == (
        result.canonical_preflight_identity
    )


def test_roi_preflight_accepts_timestamp_time_column(tmp_path):
    """4J16k18: a real RWD source whose fluorescence.csv header uses
    "TimeStamp" (not "Time(s)") as its time column must not be refused by
    ROI execution preflight with roi_preflight_refused / roi_discovery_failed.
    Uses the actual production candidate constants (post-fix) so this test
    would fail if the fix were reverted."""
    intent, _ = _intent(
        tmp_path,
        time_col="TimeStamp",
        time_column_candidates=GUIDED_BACKEND_RWD_TIME_COLUMN_CANDIDATES,
    )
    _, candidate = _candidate(intent)
    assert candidate.accepted is True
    request, result = _roi(intent, candidate)
    assert result.accepted is True
    assert result.blocking_issues == ()
    assert result.actual_discovered_roi_ids == ("ROI0", "ROI1")
    assert result.actual_included_roi_ids == ("ROI0",)


@pytest.mark.parametrize("change", ["same_size_content", "changed_size"])
def test_roi_preflight_refuses_candidate_change_after_candidate_acceptance(
    tmp_path,
    change,
):
    intent, root = _intent(tmp_path)
    _, candidate = _candidate(intent)
    assert candidate.accepted is True
    request = preflight.derive_roi_execution_preflight_request_from_intent(
        intent,
        accepted_candidate_preflight_identity=(
            candidate.canonical_preflight_identity
        ),
    )
    target = root / "2025_01_01-00_00_00" / "fluorescence.csv"
    raw = target.read_bytes()
    if change == "same_size_content":
        target.write_bytes(raw[:-2] + b"9\n")
    else:
        target.write_bytes(raw + b"1,2,3,4,5\n")

    result = preflight.run_roi_execution_preflight(request)

    assert result.accepted is False
    assert _category(result) == "roi_source_digest_mismatch"
    assert result.blocking_issues[0].detail_code == (
        "candidate_content_binding_mismatch"
    )
    assert result.canonical_preflight_identity is None


def test_roi_candidate_tuple_difference_refuses(tmp_path, monkeypatch):
    intent, root = _intent(tmp_path)
    _, candidate = _candidate(intent)
    accepted_snapshot = build_rwd_source_candidate_snapshot(str(root))
    _write_session(root, "2025_01_01-00_10_00", rois=("ROI0",))
    monkeypatch.setattr(
        preflight,
        "build_rwd_source_candidate_snapshot",
        lambda *_args, **_kwargs: accepted_snapshot,
    )
    request = preflight.derive_roi_execution_preflight_request_from_intent(
        intent,
        accepted_candidate_preflight_identity=candidate.canonical_preflight_identity,
    )
    result = preflight.run_roi_execution_preflight(request)
    assert _category(result) == "roi_tuple_mismatch"


def test_roi_parser_digest_and_columns_refuse(tmp_path):
    intent, _ = _intent(tmp_path)
    _, candidate = _candidate(intent)
    request = preflight.derive_roi_execution_preflight_request_from_intent(
        intent,
        accepted_candidate_preflight_identity=candidate.canonical_preflight_identity,
    )
    assert _category(
        preflight.run_roi_execution_preflight(
            replace(request, parser_contract_digest="d" * 64)
        )
    ) == "parser_digest_mismatch"
    assert _category(
        preflight.run_roi_execution_preflight(
            replace(request, expected_selected_time_column="Timestamp")
        )
    ) == "roi_parser_column_mismatch"
    assert _category(
        preflight.run_roi_execution_preflight(
            replace(request, expected_uv_suffix="-415")
        )
    ) == "roi_parser_column_mismatch"


def test_roi_ambiguous_header_refuses(tmp_path, monkeypatch):
    intent, root = _intent(tmp_path)
    _, candidate = _candidate(intent)
    accepted_snapshot = build_rwd_source_candidate_snapshot(str(root))
    target = root / "2025_01_01-00_00_00" / "fluorescence.csv"
    target.write_text(
        "Time(s),ROI0-410,ROI0-410,ROI0-470,ROI1-410,ROI1-470\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preflight,
        "build_rwd_source_candidate_snapshot",
        lambda *_args, **_kwargs: accepted_snapshot,
    )
    request = preflight.derive_roi_execution_preflight_request_from_intent(
        intent,
        accepted_candidate_preflight_identity=candidate.canonical_preflight_identity,
    )
    assert _category(preflight.run_roi_execution_preflight(request)) == "roi_ambiguous"


def test_roi_order_and_source_digest_refuse(tmp_path):
    intent, _ = _intent(tmp_path)
    _, candidate = _candidate(intent)
    request = preflight.derive_roi_execution_preflight_request_from_intent(
        intent,
        accepted_candidate_preflight_identity=candidate.canonical_preflight_identity,
    )
    order = replace(
        request,
        expected_discovered_roi_ids=("ROI1", "ROI0"),
        expected_included_roi_ids=("ROI0",),
        expected_excluded_roi_ids=("ROI1",),
    )
    assert _category(preflight.run_roi_execution_preflight(order)) == "roi_tuple_mismatch"
    source = replace(request, expected_inventory_source_content_digest="d" * 64)
    assert _category(preflight.run_roi_execution_preflight(source)) == "roi_source_digest_mismatch"


def test_roi_missing_included_and_extra_analyzed_refuse(tmp_path):
    intent, _ = _intent(tmp_path)
    _, candidate = _candidate(intent)
    request = preflight.derive_roi_execution_preflight_request_from_intent(
        intent,
        accepted_candidate_preflight_identity=candidate.canonical_preflight_identity,
    )
    missing = replace(
        request,
        expected_included_roi_ids=("MISSING",),
        expected_excluded_roi_ids=("ROI0", "ROI1"),
    )
    assert _category(
        preflight.run_roi_execution_preflight(missing)
    ) == "roi_missing_included"
    extra = replace(
        request,
        expected_discovered_roi_ids=("ROI0",),
        expected_included_roi_ids=("ROI0",),
        expected_excluded_roi_ids=(),
    )
    assert _category(
        preflight.run_roi_execution_preflight(extra)
    ) == "roi_extra_analyzed"


def test_strict_roi_digest_is_deterministic_and_order_sensitive():
    kwargs = dict(
        source_candidate_content_digest="a" * 64,
        parser_contract_digest="b" * 64,
        discovered_roi_ids=("ROI0", "ROI1"),
        included_roi_ids=("ROI0",),
        excluded_roi_ids=("ROI1",),
        selection_mode="include",
    )
    first = preflight.compute_guided_strict_roi_inventory_digest(**kwargs)
    assert first == preflight.compute_guided_strict_roi_inventory_digest(**kwargs)
    changed = dict(kwargs, discovered_roi_ids=("ROI1", "ROI0"))
    assert first != preflight.compute_guided_strict_roi_inventory_digest(**changed)


def test_roi_identity_changes_with_inventory(tmp_path):
    intent, _ = _intent(tmp_path)
    _, candidate = _candidate(intent)
    _, result = _roi(intent, candidate)
    changed = replace(result, actual_excluded_roi_ids=("OTHER",), canonical_preflight_identity="0" * 64)
    changed_identity = preflight.compute_guided_roi_preflight_identity(changed)
    assert changed_identity != result.canonical_preflight_identity


def test_candidate_derivation_uses_only_declared_intent_sections(tmp_path):
    intent, _ = _intent(tmp_path)
    original = preflight.derive_candidate_manifest_preflight_request_from_intent(intent)
    unrelated = replace(
        intent,
        output_policy=replace(intent.output_policy, output_base_canonical="/other"),
        application_build_identity=mapping.build_application_build_identity(
            distribution_name="other",
            distribution_version="2",
            source_revision_kind="git",
            source_revision="other",
            source_tree_state="clean",
        ),
        canonical_intent_identity=intent.canonical_intent_identity,
    )
    assert preflight.derive_candidate_manifest_preflight_request_from_intent(unrelated) == original
    changed = replace(
        intent,
        input_source=replace(
            intent.input_source, source_candidate_content_digest="d" * 64
        ),
        canonical_intent_identity=intent.canonical_intent_identity,
    )
    assert preflight.derive_candidate_manifest_preflight_request_from_intent(changed) != original


def test_roi_derivation_uses_only_declared_intent_sections(tmp_path):
    intent, _ = _intent(tmp_path)
    identity = "a" * 64
    original = preflight.derive_roi_execution_preflight_request_from_intent(
        intent, accepted_candidate_preflight_identity=identity
    )
    unrelated = replace(
        intent,
        feature_event=replace(intent.feature_event, profile_id="other"),
        output_policy=replace(intent.output_policy, output_base_canonical="/other"),
        canonical_intent_identity=intent.canonical_intent_identity,
    )
    assert preflight.derive_roi_execution_preflight_request_from_intent(
        unrelated, accepted_candidate_preflight_identity=identity
    ) == original
    changed = replace(
        intent,
        acquisition=replace(intent.acquisition, rwd_time_col="Timestamp"),
        canonical_intent_identity=intent.canonical_intent_identity,
    )
    assert preflight.derive_roi_execution_preflight_request_from_intent(
        changed, accepted_candidate_preflight_identity=identity
    ) != original


def test_preflight_requests_exclude_execution_and_unrelated_fields():
    prohibited = (
        "run_id",
        "run_dir",
        "config_path",
        "argv",
        "command",
        "timestamp",
        "artifact",
        "process",
        "output_policy",
        "correction",
        "feature_event",
        "diagnostic_evidence",
        "application_build_identity",
    )
    for model in (
        preflight.GuidedCandidateManifestExecutionPreflightRequest,
        preflight.GuidedRoiExecutionPreflightRequest,
    ):
        names = tuple(item.name for item in fields(model))
        assert not any(marker in name for marker in prohibited for name in names)


def test_identity_and_derivation_coverage_constants():
    assert set(preflight.CANDIDATE_IDENTITY_FIELDS) == {
        "contract_version",
        "runner_contract_version",
        "expected_candidate_set_digest",
        "expected_candidate_content_digest",
        "actual_candidate_set_digest",
        "actual_candidate_content_digest",
        "actual_candidates",
        "accepted",
    }
    assert "input_source" in preflight.CANDIDATE_INTENT_DERIVATION_FIELDS
    assert "output_policy" not in preflight.CANDIDATE_INTENT_DERIVATION_FIELDS
    assert "roi_scope" in preflight.ROI_INTENT_DERIVATION_FIELDS
    assert "correction" not in preflight.ROI_INTENT_DERIVATION_FIELDS


def test_preflights_do_not_call_write_allocation_or_runner_apis(tmp_path, monkeypatch):
    intent, _ = _intent(tmp_path)

    def fail(*_args, **_kwargs):
        raise AssertionError("write/allocation/runner API is prohibited")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(Path, "touch", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    _, candidate = _candidate(intent)
    _, roi = _roi(intent, candidate)
    assert candidate.accepted and roi.accepted
    for result in (candidate, roi):
        assert result.no_files_written
        assert result.no_directories_created
        assert result.no_artifacts_created
        assert result.no_run_id_allocated
        assert result.no_config_or_argv_generated
        assert result.no_runner_invoked


def test_module_import_boundary():
    source = Path(preflight.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    prohibited = (
        "gui",
        "tools.run_full_pipeline_deliverables",
        "photometry_pipeline.pipeline",
        "photometry_pipeline.guided_backend_validation_workflow",
        "photometry_pipeline.guided_backend_validation_materialization",
        "photometry_pipeline.guided_run_authorization",
    )
    assert not any(
        name == marker or name.startswith(f"{marker}.")
        for name in imported
        for marker in prohibited
    )
