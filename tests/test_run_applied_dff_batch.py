import hashlib
import json
from pathlib import Path

import pytest

from tools.run_applied_dff_batch import AppliedDffBatchError, run_applied_dff_batch
from tools.run_applied_dff_pipeline import AppliedDffPipelineError


def _make_phasic_out(tmp_path: Path) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    (phasic_out / "phasic_trace_cache.h5").write_bytes(b"source cache")
    return phasic_out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(path: Path, rows: list[dict[str, str]], fieldnames=None) -> Path:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    lines = [",".join(fieldnames)]
    for row in rows:
        lines.append(",".join(row.get(key, "") for key in fieldnames))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fake_success_report(output_root, output_name, roi, strategy, *, source="dynamic_fit_dff"):
    base = Path(output_root) / output_name
    return {
        "summary_json": str(base / "pipeline" / "applied_dff_pipeline_summary.json"),
        "provenance_json": str(base / "pipeline" / "applied_dff_pipeline_provenance.json"),
        "summary": {
            "pipeline_passed": True,
            "roi": roi,
            "strategy": strategy,
            "applied_trace_source": source,
            "applied_trace_complete": True,
            "n_chunks_processed": 581,
            "n_features": 581,
            "semantic_status": "pass",
            "feature_output_granularity": "chunk_summary",
            "one_feature_row_per_chunk": True,
            "one_feature_row_per_chunk_matches_detector": True,
            "hdf5_modified_source_phasic_cache": False,
            "legacy_features_modified": False,
        },
    }


def test_valid_manifest_runs_two_explicit_rows(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {"roi": "CH8", "strategy": "signal_only_f0"},
            {"roi": "CH9", "strategy": "dynamic_fit"},
        ],
        fieldnames=["roi", "strategy"],
    )
    calls = []

    import tools.run_applied_dff_batch as batch

    def _fake_pipeline(*args, roi, strategy, output_root, output_name, **kwargs):
        calls.append({"roi": roi, "strategy": strategy, "output_root": output_root, "output_name": output_name})
        source = "signal_only_f0_dff" if strategy == "signal_only_f0" else "dynamic_fit_dff"
        return _fake_success_report(output_root, output_name, roi, strategy, source=source)

    monkeypatch.setattr(batch, "run_applied_dff_pipeline", _fake_pipeline)

    report = run_applied_dff_batch(
        phasic_out,
        manifest=manifest,
        output_root=tmp_path / "out",
        overwrite=True,
    )

    assert [(call["roi"], call["strategy"]) for call in calls] == [
        ("CH8", "signal_only_f0"),
        ("CH9", "dynamic_fit"),
    ]
    summary = report["summary"]
    assert summary["batch_passed"] is True
    assert summary["n_rows_completed"] == 2
    assert summary["n_rows_failed"] == 0
    forbidden = {"recommended_strategy", "chosen_strategy", "selected_strategy", "best_strategy"}
    assert forbidden.isdisjoint(summary)
    assert all(forbidden.isdisjoint(row) for row in summary["rows"])
    assert json.loads(Path(report["summary_json"]).read_text(encoding="utf-8"))["batch_passed"] is True


@pytest.mark.parametrize("strategy", ["auto", "no_correction"])
def test_rejects_auto_and_no_correction_before_running(tmp_path, monkeypatch, strategy):
    phasic_out = _make_phasic_out(tmp_path)
    manifest = _write_manifest(tmp_path / "manifest.csv", [{"roi": "CH1", "strategy": strategy}], ["roi", "strategy"])
    calls = []

    import tools.run_applied_dff_batch as batch

    monkeypatch.setattr(batch, "run_applied_dff_pipeline", lambda *args, **kwargs: calls.append(kwargs))

    with pytest.raises(AppliedDffBatchError, match="unsupported strategy"):
        run_applied_dff_batch(phasic_out, manifest=manifest, output_root=tmp_path / "out")

    assert calls == []


@pytest.mark.parametrize("fieldnames", [["roi"], ["strategy"]])
def test_rejects_missing_required_columns(tmp_path, fieldnames):
    phasic_out = _make_phasic_out(tmp_path)
    manifest = _write_manifest(tmp_path / "manifest.csv", [{"roi": "CH1", "strategy": "dynamic_fit"}], fieldnames)

    with pytest.raises(AppliedDffBatchError, match="missing required column"):
        run_applied_dff_batch(phasic_out, manifest=manifest, output_root=tmp_path / "out")


@pytest.mark.parametrize("output_name", ["../bad", "nested/name", "C:\\bad", ""])
def test_rejects_unsafe_output_name_before_running(tmp_path, monkeypatch, output_name):
    phasic_out = _make_phasic_out(tmp_path)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [{"roi": "CH1", "strategy": "dynamic_fit", "output_name": output_name}],
        ["roi", "strategy", "output_name"],
    )
    calls = []

    import tools.run_applied_dff_batch as batch

    monkeypatch.setattr(batch, "run_applied_dff_pipeline", lambda *args, **kwargs: calls.append(kwargs))

    with pytest.raises(AppliedDffBatchError, match="unsafe output_name"):
        run_applied_dff_batch(phasic_out, manifest=manifest, output_root=tmp_path / "out")

    assert calls == []


def test_duplicate_roi_strategy_rows_fail(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {"roi": "CH1", "strategy": "dynamic_fit", "output_name": "one"},
            {"roi": "CH1", "strategy": "dynamic_fit", "output_name": "two"},
        ],
        ["roi", "strategy", "output_name"],
    )

    with pytest.raises(AppliedDffBatchError, match="duplicate ROI/strategy"):
        run_applied_dff_batch(phasic_out, manifest=manifest, output_root=tmp_path / "out")


def test_stops_on_first_failure_by_default(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [{"roi": "CH1", "strategy": "dynamic_fit"}, {"roi": "CH2", "strategy": "signal_only_f0"}],
        ["roi", "strategy"],
    )
    calls = []

    import tools.run_applied_dff_batch as batch

    def _fake_pipeline(*args, roi, strategy, **kwargs):
        calls.append((roi, strategy))
        raise AppliedDffPipelineError("forced row failure", report={"pipeline_passed": False})

    monkeypatch.setattr(batch, "run_applied_dff_pipeline", _fake_pipeline)

    with pytest.raises(AppliedDffBatchError, match="forced row failure") as exc_info:
        run_applied_dff_batch(phasic_out, manifest=manifest, output_root=tmp_path / "out", overwrite=True)

    assert calls == [("CH1", "dynamic_fit")]
    assert exc_info.value.report["summary"]["batch_passed"] is False
    assert exc_info.value.report["summary"]["n_rows_skipped"] == 1


def test_continue_on_error_attempts_later_rows(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [{"roi": "CH1", "strategy": "dynamic_fit"}, {"roi": "CH2", "strategy": "signal_only_f0"}],
        ["roi", "strategy"],
    )
    calls = []

    import tools.run_applied_dff_batch as batch

    def _fake_pipeline(*args, roi, strategy, output_root, output_name, **kwargs):
        calls.append((roi, strategy))
        if len(calls) == 1:
            raise AppliedDffPipelineError("forced row failure", report={"pipeline_passed": False})
        return _fake_success_report(output_root, output_name, roi, strategy, source="signal_only_f0_dff")

    monkeypatch.setattr(batch, "run_applied_dff_pipeline", _fake_pipeline)

    with pytest.raises(AppliedDffBatchError) as exc_info:
        run_applied_dff_batch(
            phasic_out,
            manifest=manifest,
            output_root=tmp_path / "out",
            overwrite=True,
            continue_on_error=True,
        )

    assert calls == [("CH1", "dynamic_fit"), ("CH2", "signal_only_f0")]
    summary = exc_info.value.report["summary"]
    assert summary["batch_passed"] is False
    assert summary["n_rows_completed"] == 1
    assert summary["n_rows_failed"] == 1


def test_dry_run_validates_manifest_and_writes_nothing(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    manifest = _write_manifest(tmp_path / "manifest.csv", [{"roi": "CH1", "strategy": "dynamic_fit"}], ["roi", "strategy"])
    calls = []

    import tools.run_applied_dff_batch as batch

    monkeypatch.setattr(batch, "run_applied_dff_pipeline", lambda *args, **kwargs: calls.append(kwargs))

    report = run_applied_dff_batch(
        phasic_out,
        manifest=manifest,
        output_root=tmp_path / "out",
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert report["planned_rows"][0]["would_run"] is True
    assert calls == []
    assert not (tmp_path / "out").exists()


def test_read_only_source_guarantees(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    legacy = phasic_out / "features" / "features.csv"
    legacy.parent.mkdir()
    legacy.write_text("roi,chunk_id\nCH1,0\n", encoding="utf-8")
    source_before = _sha256(phasic_out / "phasic_trace_cache.h5")
    legacy_before = _sha256(legacy)
    manifest = _write_manifest(tmp_path / "manifest.csv", [{"roi": "CH1", "strategy": "dynamic_fit"}], ["roi", "strategy"])

    import tools.run_applied_dff_batch as batch

    def _fake_pipeline(*args, roi, strategy, output_root, output_name, **kwargs):
        return _fake_success_report(output_root, output_name, roi, strategy)

    monkeypatch.setattr(batch, "run_applied_dff_pipeline", _fake_pipeline)

    report = run_applied_dff_batch(phasic_out, manifest=manifest, output_root=tmp_path / "out", overwrite=True)

    assert _sha256(phasic_out / "phasic_trace_cache.h5") == source_before
    assert _sha256(legacy) == legacy_before
    assert report["summary"]["hdf5_modified_source_phasic_cache"] is False
    assert report["summary"]["legacy_features_modified"] is False
    provenance = json.loads(Path(report["provenance_json"]).read_text(encoding="utf-8"))
    assert provenance["hdf5_modified_source_phasic_cache"] is False
    assert provenance["legacy_features_modified"] is False


def test_unsafe_output_root_refuses_without_deleting_source_cache(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    source = phasic_out / "phasic_trace_cache.h5"
    before = _sha256(source)
    manifest = _write_manifest(tmp_path / "manifest.csv", [{"roi": "CH1", "strategy": "dynamic_fit"}], ["roi", "strategy"])

    with pytest.raises(AppliedDffBatchError, match="equals phasic_out"):
        run_applied_dff_batch(phasic_out, manifest=manifest, output_root=phasic_out, overwrite=True)

    assert source.exists()
    assert _sha256(source) == before


def test_unsafe_output_root_legacy_features_refuses_without_deleting_features(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    legacy = phasic_out / "features" / "features.csv"
    legacy.parent.mkdir()
    legacy.write_text("roi,chunk_id\nCH1,0\n", encoding="utf-8")
    before = _sha256(legacy)
    manifest = _write_manifest(tmp_path / "manifest.csv", [{"roi": "CH1", "strategy": "dynamic_fit"}], ["roi", "strategy"])

    with pytest.raises(AppliedDffBatchError, match="legacy features directory"):
        run_applied_dff_batch(phasic_out, manifest=manifest, output_root=phasic_out / "features", overwrite=True)

    assert legacy.exists()
    assert _sha256(legacy) == before


def test_output_root_containing_source_cache_refuses_without_deleting(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    source = phasic_out / "phasic_trace_cache.h5"
    before = _sha256(source)
    manifest = _write_manifest(tmp_path / "manifest.csv", [{"roi": "CH1", "strategy": "dynamic_fit"}], ["roi", "strategy"])

    with pytest.raises(AppliedDffBatchError, match="contains phasic_trace_cache"):
        run_applied_dff_batch(phasic_out, manifest=manifest, output_root=phasic_out.parent, overwrite=True)

    assert source.exists()
    assert _sha256(source) == before
