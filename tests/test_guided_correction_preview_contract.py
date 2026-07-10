import json
from pathlib import Path

from photometry_pipeline.preview.correction_preview import (
    GUIDED_REFERENCE_PREVIEW_METHODS,
    build_preview_provenance,
    build_preview_summary,
    make_guided_preview_id,
    resolve_completed_run_preview_source,
    resolve_phasic_cache_preview_source,
    validate_guided_preview_output_dir,
    validate_preview_methods,
)


def _make_completed_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    (run_dir / "run_report.json").write_text(
        json.dumps({"status": "success", "configuration": {}, "analytical_contract": {}}),
        encoding="utf-8",
    )
    (phasic_out / "phasic_trace_cache.h5").write_bytes(b"cache")
    (phasic_out / "config_used.yaml").write_text("dynamic_fit_mode: robust_global_event_reject\n", encoding="utf-8")
    return run_dir


def test_preview_output_allows_completed_run_preview_namespace(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    preview_id = "preview_20260617T010203Z_abcd1234"
    preview_dir = run_dir / "_guided_workflow" / "previews" / preview_id

    result = validate_guided_preview_output_dir(
        preview_dir,
        completed_run_dir=run_dir,
        phasic_out=run_dir / "_analysis" / "phasic_out",
    )

    assert result.ok is True
    assert result.code == "ok"
    assert result.resolved_path.endswith(str(preview_dir))
    assert not preview_dir.exists()


def test_preview_output_rejects_completed_run_root(tmp_path):
    run_dir = _make_completed_run(tmp_path)

    result = validate_guided_preview_output_dir(run_dir, completed_run_dir=run_dir)

    assert result.ok is False
    assert result.code == "completed_run_root"


def test_preview_output_rejects_completed_run_child_outside_preview_namespace(tmp_path):
    run_dir = _make_completed_run(tmp_path)

    result = validate_guided_preview_output_dir(run_dir / "not_preview", completed_run_dir=run_dir)

    assert result.ok is False
    assert result.code == "inside_completed_run_outside_preview_namespace"


def test_preview_output_rejects_unsafe_completed_run_preview_id(tmp_path):
    run_dir = _make_completed_run(tmp_path)

    result = validate_guided_preview_output_dir(
        run_dir / "_guided_workflow" / "previews" / "bad" / "nested",
        completed_run_dir=run_dir,
    )

    assert result.ok is False
    assert result.code == "unsafe_preview_id"


def test_preview_output_rejects_phasic_out_and_children(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic_out = run_dir / "_analysis" / "phasic_out"

    root_result = validate_guided_preview_output_dir(tmp_path / "out", phasic_out=tmp_path / "out")
    child_result = validate_guided_preview_output_dir(phasic_out / "preview", phasic_out=phasic_out)

    assert root_result.ok is False
    assert root_result.code == "inside_phasic_out"
    assert child_result.ok is False
    assert child_result.code == "inside_phasic_out"


def test_preview_output_rejects_legacy_features_namespace(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic_out = run_dir / "_analysis" / "phasic_out"
    features = phasic_out / "features"

    root_result = validate_guided_preview_output_dir(features, phasic_out=phasic_out)
    child_result = validate_guided_preview_output_dir(features / "preview", phasic_out=phasic_out)

    assert root_result.ok is False
    assert root_result.code == "inside_legacy_features"
    assert child_result.ok is False
    assert child_result.code == "inside_legacy_features"


def test_preview_output_rejects_source_roots_and_path_traversal(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    traversal = source / ".." / "input" / "preview"

    root_result = validate_guided_preview_output_dir(source, source_roots=[source])
    child_result = validate_guided_preview_output_dir(traversal, source_roots=[source])

    assert root_result.ok is False
    assert root_result.code == "inside_source_root"
    assert child_result.ok is False
    assert child_result.code == "inside_source_root"


def test_preview_output_rejects_applied_dff_root(tmp_path):
    applied = tmp_path / "applied_dff"
    applied.mkdir()

    result = validate_guided_preview_output_dir(applied / "preview", applied_dff_roots=[applied])

    assert result.ok is False
    assert result.code == "inside_applied_dff_root"


def test_preview_output_rejects_existing_file(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("not a directory", encoding="utf-8")

    result = validate_guided_preview_output_dir(path)

    assert result.ok is False
    assert result.code == "preview_path_is_file"


def test_preview_output_allows_future_external_preview_root(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    preview = tmp_path / "preview_root" / "_guided_workflow" / "previews" / "p1"

    result = validate_guided_preview_output_dir(preview, source_roots=[source])

    assert result.ok is True
    assert result.code == "ok"


def test_make_guided_preview_id_is_safe_and_unique():
    first = make_guided_preview_id()
    second = make_guided_preview_id()

    assert first
    assert second
    assert first != second
    for value in (first, second):
        assert "/" not in value
        assert "\\" not in value
        assert ".." not in value


def test_validate_preview_methods_accepts_reference_methods_in_order():
    methods = [
        "global_linear_regression",
        "robust_global_event_reject",
        "adaptive_event_gated_regression",
    ]

    result = validate_preview_methods(methods)

    assert result.ok is True
    assert result.methods == tuple(methods)


def test_validate_preview_methods_rejects_disallowed_values():
    for method in ["signal_only_f0", "auto", "needs_review", "no_correction", "unknown"]:
        result = validate_preview_methods([method])
        assert result.ok is False
        assert result.code == "unsupported_preview_method"
        assert method in result.invalid_methods


def test_validate_preview_methods_rejects_duplicates():
    result = validate_preview_methods(["robust_global_event_reject", "robust_global_event_reject"])

    assert result.ok is False
    assert result.code == "duplicate_preview_method"
    assert result.invalid_methods == ("robust_global_event_reject",)


def test_preview_provenance_builder_records_preview_only_no_execution_contract(tmp_path):
    preview_id = "preview_20260617T010203Z_abcd1234"
    provenance = build_preview_provenance(
        preview_id=preview_id,
        source_type="completed_run",
        completed_run_dir=tmp_path / "run",
        phasic_out=tmp_path / "run" / "_analysis" / "phasic_out",
        phasic_trace_cache_path=tmp_path / "run" / "_analysis" / "phasic_out" / "phasic_trace_cache.h5",
        preview_output_dir=tmp_path / "run" / "_guided_workflow" / "previews" / preview_id,
        selected_roi="CH1",
        selected_chunk=0,
        correction_methods_compared=GUIDED_REFERENCE_PREVIEW_METHODS,
        backend_method_values={"dynamic_fit_mode": "robust_global_event_reject"},
        config_values={"target_fs_hz": 20.0},
        config_source_path=tmp_path / "run" / "_analysis" / "phasic_out" / "config_used.yaml",
        source_artifact_hashes={"phasic_trace_cache.h5": "abc"},
        created_at_utc="2026-06-17T00:00:00+00:00",
    )

    assert provenance["preview_only"] is True
    assert provenance["pipeline_run_executed"] is False
    assert provenance["manifest_written"] is False
    assert provenance["applied_dff_routed"] is False
    assert provenance["feature_extraction_run"] is False
    assert provenance["production_output"] is False
    assert provenance["strategy_recommendation"] is None
    assert provenance["correction_methods_compared"] == list(GUIDED_REFERENCE_PREVIEW_METHODS)


def test_preview_summary_builder_has_no_strategy_recommendation():
    summary = build_preview_summary(
        preview_id="preview_20260617T010203Z_abcd1234",
        status="not_run",
        method_statuses={"robust_global_event_reject": {"status": "not_run"}},
        warnings=["preview not generated"],
        stale=True,
    )

    assert summary["status"] == "not_run"
    assert "strategy_recommendation" not in summary
    assert summary["method_statuses"]["robust_global_event_reject"]["status"] == "not_run"
    assert summary["stale"] is True


def test_resolve_completed_run_preview_source_success(tmp_path):
    run_dir = _make_completed_run(tmp_path)

    result = resolve_completed_run_preview_source(run_dir)

    assert result.ok is True
    assert result.source_type == "completed_run"
    assert result.completed_run_dir == str(run_dir.resolve())
    assert result.phasic_trace_cache_path.endswith("phasic_trace_cache.h5")
    assert result.config_path.endswith("config_used.yaml")


def test_resolve_completed_run_preview_source_fails_without_success_metadata(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    (run_dir / "run_report.json").write_text(
        json.dumps({"status": "failed", "configuration": {}, "analytical_contract": {}}),
        encoding="utf-8",
    )

    result = resolve_completed_run_preview_source(run_dir)

    assert result.ok is False
    assert result.code == "completed_run_not_successful"


def test_resolve_completed_run_preview_source_fails_without_cache(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    (run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").unlink()

    result = resolve_completed_run_preview_source(run_dir)

    assert result.ok is False
    assert result.code == "phasic_cache_missing"


def test_resolve_completed_run_preview_source_fails_without_config(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    (run_dir / "_analysis" / "phasic_out" / "config_used.yaml").unlink()

    result = resolve_completed_run_preview_source(run_dir)

    assert result.ok is False
    assert result.code == "config_snapshot_missing"


def test_resolve_phasic_cache_preview_source_success(tmp_path):
    phasic_out = tmp_path / "phasic_out"
    phasic_out.mkdir()
    (phasic_out / "phasic_trace_cache.h5").write_bytes(b"cache")
    (phasic_out / "config_used.yaml").write_text("dynamic_fit_mode: global_linear_regression\n", encoding="utf-8")

    result = resolve_phasic_cache_preview_source(phasic_out)

    assert result.ok is True
    assert result.source_type == "phasic_cache"
    assert result.phasic_out == str(phasic_out.resolve())


def test_resolve_phasic_cache_preview_source_fails_without_config(tmp_path):
    phasic_out = tmp_path / "phasic_out"
    phasic_out.mkdir()
    (phasic_out / "phasic_trace_cache.h5").write_bytes(b"cache")

    result = resolve_phasic_cache_preview_source(phasic_out)

    assert result.ok is False
    assert result.code == "config_snapshot_missing"

