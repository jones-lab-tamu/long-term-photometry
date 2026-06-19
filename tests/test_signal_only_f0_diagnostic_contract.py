import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from photometry_pipeline.signal_only_f0_diagnostics import contract
from photometry_pipeline.guided_diagnostic_cache import (
    DiagnosticCacheBuildRequest,
    artifact_record_from_request,
    write_artifact_record_json,
    write_build_request_json,
    write_json_file,
)


def _make_completed_run(tmp_path: Path, *, successful: bool = True) -> Path:
    run_dir = tmp_path / "run_001"
    phasic = run_dir / "_analysis" / "phasic_out"
    phasic.mkdir(parents=True)
    (phasic / "phasic_trace_cache.h5").write_bytes(b"cache")
    (phasic / "config_used.yaml").write_text("sample_rate_hz: 20\n", encoding="utf-8")
    status = "success" if successful else "failed"
    (run_dir / "run_report.json").write_text(
        json.dumps({"status": status, "phase": "final"}),
        encoding="utf-8",
    )
    return run_dir


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _make_diagnostic_cache(tmp_path: Path) -> Path:
    run_dir = _make_completed_run(tmp_path)
    cache_root = tmp_path / "diagnostic_cache"
    phasic_src = run_dir / "_analysis" / "phasic_out"
    phasic_dest = cache_root / "_analysis" / "phasic_out"
    phasic_dest.mkdir(parents=True)
    (phasic_dest / "phasic_trace_cache.h5").write_bytes((phasic_src / "phasic_trace_cache.h5").read_bytes())
    (phasic_dest / "config_used.yaml").write_text(
        (phasic_src / "config_used.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (cache_root / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    request = DiagnosticCacheBuildRequest(
        raw_input_path=str(tmp_path / "raw_input"),
        input_format="rwd",
        acquisition_mode="intermittent",
        included_roi_ids=("CH1",),
        output_base=str(tmp_path),
        requested_cache_path=str(cache_root),
        requested_at_utc="2026-06-19T12:00:00Z",
    )
    request_path = cache_root / "guided_diagnostic_cache_request.json"
    write_build_request_json(request_path, request)
    record = artifact_record_from_request(
        request,
        cache_id="cache_001",
        cache_root_path=str(cache_root),
        phasic_trace_cache_path=str(phasic_dest / "phasic_trace_cache.h5"),
        config_used_path=str(phasic_dest / "config_used.yaml"),
        status_marker_path=str(cache_root / "status.json"),
        request_json_path=str(request_path),
        roi_inventory=("CH1",),
    )
    write_artifact_record_json(cache_root / "guided_diagnostic_cache_artifact.json", record)
    write_json_file(
        cache_root / "guided_diagnostic_cache_provenance.json",
        {
            "schema_version": "guided_diagnostic_cache.v1",
            "purpose": "guided_diagnostic_cache",
            "production_analysis": False,
        },
    )
    return cache_root


def test_diagnostic_id_is_safe_and_deterministic_with_fixed_inputs():
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=timezone.utc)

    diagnostic_id = contract.make_signal_only_f0_diagnostic_id(now=now, token="abc123")

    assert diagnostic_id == "signal_only_f0_20260617T120000Z_abc123"
    assert diagnostic_id.startswith("signal_only_f0_")
    assert " " not in diagnostic_id
    assert "/" not in diagnostic_id
    assert "\\" not in diagnostic_id


def test_diagnostic_id_normal_calls_do_not_collide():
    ids = {contract.make_signal_only_f0_diagnostic_id() for _ in range(5)}

    assert len(ids) == 5


def test_completed_run_source_resolver_accepts_valid_minimal_run_and_is_read_only(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    before = _snapshot_tree(run_dir)

    result = contract.resolve_completed_run_signal_only_f0_source(run_dir)

    assert result.ok is True
    assert result.source_type == contract.SOURCE_TYPE_COMPLETED_RUN
    assert result.completed_run_dir == os.path.realpath(os.path.abspath(run_dir))
    assert result.phasic_out_dir.endswith(os.path.join("_analysis", "phasic_out"))
    assert result.phasic_trace_cache_path.endswith("phasic_trace_cache.h5")
    assert result.config_source_path.endswith("config_used.yaml")
    assert _snapshot_tree(run_dir) == before


def test_completed_run_source_resolver_reports_missing_directory(tmp_path):
    result = contract.resolve_completed_run_signal_only_f0_source(tmp_path / "missing")

    assert result.ok is False
    assert result.code == "completed_run_missing"
    assert "does not exist" in result.reason


def test_completed_run_source_resolver_rejects_unsuccessful_run(tmp_path):
    run_dir = _make_completed_run(tmp_path, successful=False)

    result = contract.resolve_completed_run_signal_only_f0_source(run_dir)

    assert result.ok is False
    assert result.code == "completed_run_not_successful"
    assert "successful" in result.reason or "success" in result.reason


def test_completed_run_source_resolver_reports_missing_cache(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    (run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").unlink()

    result = contract.resolve_completed_run_signal_only_f0_source(run_dir)

    assert result.ok is False
    assert result.code == "phasic_trace_cache_missing"
    assert "phasic trace cache" in result.reason


def test_completed_run_source_resolver_reports_missing_config_snapshot(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    (run_dir / "_analysis" / "phasic_out" / "config_used.yaml").unlink()

    result = contract.resolve_completed_run_signal_only_f0_source(run_dir)

    assert result.ok is False
    assert result.code == "config_snapshot_missing"
    assert "config snapshot" in result.reason


def test_direct_phasic_out_resolver_is_backend_only_and_read_only(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    before = _snapshot_tree(phasic)

    result = contract.resolve_phasic_out_signal_only_f0_source(phasic)

    assert result.ok is True
    assert result.source_type == contract.SOURCE_TYPE_PHASIC_OUT_BACKEND_ONLY
    assert result.completed_run_dir == ""
    assert _snapshot_tree(phasic) == before


def test_direct_phasic_out_resolver_reports_missing_cache_or_config(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    (phasic / "phasic_trace_cache.h5").unlink()
    missing_cache = contract.resolve_phasic_out_signal_only_f0_source(phasic)
    assert missing_cache.ok is False
    assert missing_cache.code == "phasic_trace_cache_missing"

    (phasic / "phasic_trace_cache.h5").write_bytes(b"cache")
    (phasic / "config_used.yaml").unlink()
    missing_config = contract.resolve_phasic_out_signal_only_f0_source(phasic)
    assert missing_config.ok is False
    assert missing_config.code == "config_snapshot_missing"


def test_diagnostic_cache_resolver_accepts_guided_cache_identity_and_is_read_only(tmp_path):
    cache_root = _make_diagnostic_cache(tmp_path)
    before = _snapshot_tree(cache_root)

    result = contract.resolve_diagnostic_cache_signal_only_f0_source(cache_root)

    assert result.ok is True
    assert result.source_type == contract.SOURCE_TYPE_DIAGNOSTIC_CACHE
    assert result.completed_run_dir == ""
    assert result.phasic_out_dir.endswith(os.path.join("_analysis", "phasic_out"))
    assert result.diagnostic_cache_metadata["cache_id"] == "cache_001"
    assert result.diagnostic_cache_metadata["source_type"] == "diagnostic_cache"
    assert result.diagnostic_cache_metadata["production_analysis"] is False
    assert _snapshot_tree(cache_root) == before


def test_default_namespace_builder_uses_accepted_stage4d_namespace(tmp_path):
    diagnostic_id = "signal_only_f0_20260617T120000Z_abc123"

    output = contract.build_default_signal_only_f0_diagnostic_output_dir(
        tmp_path / "run_001",
        diagnostic_id,
    )

    assert output == (
        tmp_path
        / "run_001"
        / "_guided_workflow"
        / "signal_only_f0_diagnostics"
        / diagnostic_id
    )
    assert not output.exists()


def test_output_validator_accepts_default_namespace_leaf(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    diagnostic_id = "signal_only_f0_20260617T120000Z_abc123"
    output = contract.build_default_signal_only_f0_diagnostic_output_dir(run_dir, diagnostic_id)

    result = contract.validate_signal_only_f0_diagnostic_output_dir(
        output,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        diagnostic_id=diagnostic_id,
    )

    assert result.ok is True
    assert not output.exists()


@pytest.mark.parametrize(
    ("relative_path", "expected_code"),
    [
        (".", "output_is_completed_run"),
        (os.path.join("_analysis", "phasic_out"), "output_is_phasic_out"),
        (os.path.join("_analysis", "phasic_out", "diagnostics"), "inside_phasic_out"),
        (os.path.join("_analysis", "phasic_out", "features"), "inside_legacy_features"),
        (
            os.path.join("_analysis", "phasic_out", "features", "diagnostics"),
            "inside_legacy_features",
        ),
        (os.path.join("_analysis", "phasic_out", "applied_dff"), "inside_applied_dff"),
        (
            os.path.join("_analysis", "phasic_out", "applied_dff", "diagnostics"),
            "inside_applied_dff",
        ),
    ],
)
def test_output_validator_rejects_protected_namespaces(tmp_path, relative_path, expected_code):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    output = run_dir / relative_path

    result = contract.validate_signal_only_f0_diagnostic_output_dir(
        output,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
    )

    assert result.ok is False
    assert result.code == expected_code


def test_output_validator_rejects_existing_file(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    output = tmp_path / "not_a_dir.txt"
    output.write_text("existing", encoding="utf-8")

    result = contract.validate_signal_only_f0_diagnostic_output_dir(
        output,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
        allow_existing=True,
    )

    assert result.ok is False
    assert result.code == "output_dir_is_file"
    assert output.read_text(encoding="utf-8") == "existing"


def test_output_validator_rejects_existing_directory_unless_safe_leaf(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    diagnostic_id = "signal_only_f0_20260617T120000Z_abc123"
    output = contract.build_default_signal_only_f0_diagnostic_output_dir(run_dir, diagnostic_id)
    output.mkdir(parents=True)
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    without_overwrite = contract.validate_signal_only_f0_diagnostic_output_dir(
        output,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        diagnostic_id=diagnostic_id,
    )
    with_overwrite = contract.validate_signal_only_f0_diagnostic_output_dir(
        output,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        diagnostic_id=diagnostic_id,
        allow_existing=True,
    )

    assert without_overwrite.ok is False
    assert without_overwrite.code == "output_dir_exists"
    assert with_overwrite.ok is True
    assert marker.read_text(encoding="utf-8") == "keep"


def test_output_validator_rejects_existing_non_leaf_even_with_allow_existing(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    existing_parent = run_dir / "_guided_workflow" / "signal_only_f0_diagnostics"
    existing_parent.mkdir(parents=True)
    marker = existing_parent / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    result = contract.validate_signal_only_f0_diagnostic_output_dir(
        existing_parent,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
        allow_existing=True,
    )

    assert result.ok is False
    assert result.code == "inside_completed_run_not_diagnostic_leaf"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_output_validator_rejects_arbitrary_existing_external_directory_with_allow_existing(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    external = tmp_path / "external_existing"
    external.mkdir()
    marker = external / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    result = contract.validate_signal_only_f0_diagnostic_output_dir(
        external,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
        allow_existing=True,
    )

    assert result.ok is False
    assert result.code == "outside_completed_run_diagnostic_leaf"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_output_validator_rejects_new_external_output_when_completed_run_is_provided(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    output = tmp_path / "new_external_output"
    before = _snapshot_tree(tmp_path)

    result = contract.validate_signal_only_f0_diagnostic_output_dir(
        output,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
    )

    assert result.ok is False
    assert result.code == "outside_completed_run_diagnostic_leaf"
    assert not output.exists()
    assert _snapshot_tree(tmp_path) == before


def test_output_validator_rejects_external_matching_suffix_even_with_allow_existing(tmp_path):
    run_dir = _make_completed_run(tmp_path / "source")
    phasic = run_dir / "_analysis" / "phasic_out"
    diagnostic_id = "signal_only_f0_20260617T120000Z_abc123"
    external = (
        tmp_path
        / "external"
        / "_guided_workflow"
        / "signal_only_f0_diagnostics"
        / diagnostic_id
    )
    external.mkdir(parents=True)
    marker = external / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    result = contract.validate_signal_only_f0_diagnostic_output_dir(
        external,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        diagnostic_id=diagnostic_id,
        allow_existing=True,
    )

    assert result.ok is False
    assert result.code == "outside_completed_run_diagnostic_leaf"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_output_validator_rejects_parent_that_contains_source_cache_and_legacy_features(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    features_csv = phasic / "features" / "features.csv"
    features_csv.parent.mkdir()
    features_csv.write_text("legacy", encoding="utf-8")

    result = contract.validate_signal_only_f0_diagnostic_output_dir(
        tmp_path,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
    )

    assert result.ok is False
    assert result.code in {"contains_phasic_trace_cache", "contains_legacy_features"}
    assert features_csv.read_text(encoding="utf-8") == "legacy"


def test_provenance_skeleton_contains_required_no_execution_flags_and_serializes(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"

    provenance = contract.build_signal_only_f0_diagnostic_provenance(
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
        created_at_utc="2026-06-17T12:00:00Z",
        source_type=contract.SOURCE_TYPE_COMPLETED_RUN,
        completed_run_dir=run_dir,
        phasic_out_dir=phasic,
        phasic_trace_cache_path=phasic / "phasic_trace_cache.h5",
        config_source_path=phasic / "config_used.yaml",
        selected_rois=("CH1",),
        selected_chunks=(1, 2),
        selected_window={"start_sec": 0, "end_sec": 10},
    )

    assert provenance["diagnostic_only"] is True
    assert provenance["signal_only_f0"] is True
    for key in [
        "manifest_written",
        "applied_dff_routed",
        "production_applied_dff_output_written",
        "feature_extraction_run",
        "pipeline_run_executed",
        "validation_run_executed",
    ]:
        assert provenance[key] is False
    assert provenance["strategy_recommendation"] is None
    assert provenance["selected_rois"] == ["CH1"]
    assert provenance["selected_chunks"] == [1, 2]
    assert "diagnostic evidence only" in provenance["warning"]
    json.dumps(provenance)


def test_summary_skeleton_accepts_allowed_statuses_and_serializes():
    for status in contract.SUPPORTED_DIAGNOSTIC_STATUSES:
        summary = contract.build_signal_only_f0_diagnostic_summary(
            diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
            status=status,
            roi_statuses={"CH1": {"status": "not_run"}},
            chunk_statuses={"1": {"status": "not_run"}},
            warnings=["not generated"],
            errors=[],
            generated_artifact_paths=[],
            diagnostic_metrics={},
        )
        assert summary["status"] == status
        assert summary["strategy_recommendation"] is None
        json.dumps(summary)


def test_summary_skeleton_rejects_unknown_status():
    with pytest.raises(ValueError, match="Unsupported Signal-Only F0 diagnostic status"):
        contract.build_signal_only_f0_diagnostic_summary(
            diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
            status="recommended",
        )


def test_backend_isolation_no_gui_or_applied_dff_imports_and_no_generation(tmp_path):
    source = inspect.getsource(contract)
    forbidden_snippets = [
        "import gui",
        "from gui",
        "write_applied_dff_cache",
        "run_applied_dff_features",
        "run_applied_dff_pipeline",
        "run_applied_dff_batch",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in source

    run_dir = _make_completed_run(tmp_path)
    diagnostic_id = "signal_only_f0_20260617T120000Z_abc123"
    output = contract.build_default_signal_only_f0_diagnostic_output_dir(run_dir, diagnostic_id)
    result = contract.validate_signal_only_f0_diagnostic_output_dir(
        output,
        completed_run_dir=run_dir,
        phasic_out_dir=run_dir / "_analysis" / "phasic_out",
        diagnostic_id=diagnostic_id,
    )

    assert result.ok is True
    assert not output.exists()
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "applied_dff").exists()
