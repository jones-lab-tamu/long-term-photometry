import csv
import inspect
import json
import os
from pathlib import Path

import h5py
import numpy as np

from photometry_pipeline.signal_only_f0_diagnostics import (
    SIGNAL_ONLY_F0_DIAGNOSTIC_CHUNKS_FILENAME,
    SIGNAL_ONLY_F0_DIAGNOSTIC_PROVENANCE_FILENAME,
    SIGNAL_ONLY_F0_DIAGNOSTIC_SUMMARY_FILENAME,
    build_default_signal_only_f0_diagnostic_output_dir,
    run_signal_only_f0_diagnostic_review,
)
from photometry_pipeline.signal_only_f0_diagnostics import generate


def _make_completed_run(
    tmp_path: Path,
    *,
    roi: str = "CH1",
    n_chunks: int = 3,
    bad_chunk_ids: set[int] | None = None,
) -> Path:
    run_dir = tmp_path / "run_001"
    phasic = run_dir / "_analysis" / "phasic_out"
    phasic.mkdir(parents=True)
    (run_dir / "run_report.json").write_text(
        json.dumps({"status": "success", "phase": "final"}),
        encoding="utf-8",
    )
    (phasic / "config_used.yaml").write_text("sample_rate_hz: 20\n", encoding="utf-8")
    bad_chunk_ids = set(bad_chunk_ids or set())
    with h5py.File(phasic / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([roi.encode("utf-8")]))
        meta.create_dataset("chunk_ids", data=np.asarray(list(range(n_chunks)), dtype=int))
        meta.create_dataset(
            "source_files",
            data=np.asarray([f"chunk_{idx}.csv".encode("utf-8") for idx in range(n_chunks)]),
        )
        roi_group = h5.create_group(f"roi/{roi}")
        t = np.linspace(0.0, 2.0, 41)
        base_signal = 1.0 + 0.12 * np.sin(np.linspace(0, 4 * np.pi, t.size))
        base_signal += 0.03 * np.linspace(0, 1, t.size)
        for chunk_id in range(n_chunks):
            grp = roi_group.create_group(f"chunk_{chunk_id}")
            if chunk_id in bad_chunk_ids:
                grp.create_dataset("time_sec", data=np.asarray([0.0, 0.05, 0.10]))
                grp.create_dataset("sig_raw", data=np.asarray([1.0, 1.0, 1.0]))
            else:
                grp.create_dataset("time_sec", data=t)
                grp.create_dataset("sig_raw", data=base_signal + 0.01 * chunk_id)
    return run_dir


def _snapshot_files(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_successful_minimal_generation_writes_diagnostic_only_artifacts(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    diagnostic_id = "signal_only_f0_20260617T120000Z_abc123"

    result = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH1",
        chunk_ids=[0],
        diagnostic_id=diagnostic_id,
    )

    expected_output = build_default_signal_only_f0_diagnostic_output_dir(run_dir, diagnostic_id)
    assert result["ok"] is True
    assert result["status"] == "success"
    assert Path(result["output_dir"]) == expected_output.resolve()
    assert Path(result["provenance_path"]).name == SIGNAL_ONLY_F0_DIAGNOSTIC_PROVENANCE_FILENAME
    assert Path(result["summary_path"]).name == SIGNAL_ONLY_F0_DIAGNOSTIC_SUMMARY_FILENAME
    assert Path(result["chunk_csv_path"]).name == SIGNAL_ONLY_F0_DIAGNOSTIC_CHUNKS_FILENAME
    assert result["trace_csv_paths"] == []

    provenance = _read_json(result["provenance_path"])
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

    summary = _read_json(result["summary_path"])
    assert summary["status"] == "success"
    assert summary["strategy_recommendation"] is None
    assert summary["diagnostic_metrics"]["plots_written"] is False
    assert summary["diagnostic_metrics"]["trace_csvs_written"] is False

    rows = _read_csv(result["chunk_csv_path"])
    assert len(rows) == 1
    assert rows[0]["roi"] == "CH1"
    assert rows[0]["chunk_id"] == "0"
    assert rows[0]["status"] == "success"
    assert "recommend" not in ";".join(rows[0].values()).lower()
    assert "fallback" not in ";".join(rows[0].values()).lower()


def test_default_chunk_selection_is_bounded_to_first_chunk_by_default(tmp_path):
    run_dir = _make_completed_run(tmp_path, n_chunks=4)

    result = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH1",
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
    )

    rows = _read_csv(result["chunk_csv_path"])
    assert result["status"] == "success"
    assert [row["chunk_id"] for row in rows] == ["0"]
    summary = _read_json(result["summary_path"])
    assert summary["diagnostic_metrics"]["default_chunk_selection"] == "first_available_chunk"


def test_default_chunk_selection_respects_max_chunks(tmp_path):
    run_dir = _make_completed_run(tmp_path, n_chunks=4)

    result = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH1",
        max_chunks=2,
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
    )

    rows = _read_csv(result["chunk_csv_path"])
    assert [row["chunk_id"] for row in rows] == ["0", "1"]


def test_selected_chunks_process_only_requested_and_missing_chunk_is_partial(tmp_path):
    run_dir = _make_completed_run(tmp_path, n_chunks=2)

    result = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH1",
        chunk_ids=[1, 99],
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
    )

    rows = _read_csv(result["chunk_csv_path"])
    assert result["ok"] is False
    assert result["status"] == "partial"
    assert [row["chunk_id"] for row in rows] == ["1", "99"]
    assert rows[0]["status"] == "success"
    assert rows[1]["status"] == "failed"
    assert "not found" in rows[1]["error"]


def test_missing_roi_fails_before_output_creation(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    diagnostic_id = "signal_only_f0_20260617T120000Z_abc123"
    output = build_default_signal_only_f0_diagnostic_output_dir(run_dir, diagnostic_id)

    result = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH9",
        diagnostic_id=diagnostic_id,
    )

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert "not found" in result["errors"][0]
    assert not output.exists()


def test_output_safety_rejects_external_output_and_writes_nothing(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    external = tmp_path / "external"

    result = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH1",
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
        output_dir=external,
    )

    assert result["ok"] is False
    assert "exact resolved default diagnostic leaf" in result["errors"][0]
    assert not external.exists()


def test_output_safety_rejects_source_side_namespaces_and_writes_nothing(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    diagnostic_id = "signal_only_f0_20260617T120000Z_abc123"
    for unsafe in [
        phasic / "qc" / "diagnostic",
        phasic / "features" / "diagnostic",
        phasic / "applied_dff" / "diagnostic",
    ]:
        result = run_signal_only_f0_diagnostic_review(
            run_dir,
            roi="CH1",
            diagnostic_id=diagnostic_id,
            output_dir=unsafe,
        )
        assert result["ok"] is False
        assert not unsafe.exists()


def test_existing_output_dir_requires_allow_existing_and_exact_safe_leaf(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    diagnostic_id = "signal_only_f0_20260617T120000Z_abc123"
    output = build_default_signal_only_f0_diagnostic_output_dir(run_dir, diagnostic_id)
    output.mkdir(parents=True)
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    rejected = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH1",
        diagnostic_id=diagnostic_id,
    )
    accepted = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH1",
        diagnostic_id=diagnostic_id,
        allow_existing=True,
    )

    assert rejected["ok"] is False
    assert "already exists" in rejected["errors"][0]
    assert accepted["status"] == "success"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_source_files_are_read_only_and_no_source_side_outputs_are_created(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic = run_dir / "_analysis" / "phasic_out"
    before = _snapshot_files(phasic)

    result = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH1",
        chunk_ids=[0],
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
    )

    assert result["status"] == "success"
    assert _snapshot_files(phasic) == before
    assert not (phasic / "qc").exists()
    assert not (phasic / "features").exists()
    assert not (phasic / "applied_dff").exists()
    assert not (run_dir / "manifest.csv").exists()
    assert not (run_dir / "applied_dff_manifest.csv").exists()


def test_all_chunks_failed_after_output_creation_writes_failed_artifacts(tmp_path):
    run_dir = _make_completed_run(tmp_path, n_chunks=1, bad_chunk_ids={0})
    diagnostic_id = "signal_only_f0_20260617T120000Z_abc123"

    result = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH1",
        chunk_ids=[0],
        diagnostic_id=diagnostic_id,
    )

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert Path(result["provenance_path"]).exists()
    assert Path(result["summary_path"]).exists()
    assert Path(result["chunk_csv_path"]).exists()
    rows = _read_csv(result["chunk_csv_path"])
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"]


def test_some_chunks_failed_after_output_creation_writes_partial_artifacts(tmp_path):
    run_dir = _make_completed_run(tmp_path, n_chunks=2, bad_chunk_ids={1})

    result = run_signal_only_f0_diagnostic_review(
        run_dir,
        roi="CH1",
        chunk_ids=[0, 1],
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
    )

    assert result["ok"] is False
    assert result["status"] == "partial"
    summary = _read_json(result["summary_path"])
    assert summary["status"] == "partial"
    assert summary["roi_statuses"]["CH1"]["n_chunks_success"] == 1
    assert summary["roi_statuses"]["CH1"]["n_chunks_failed"] == 1


def test_generator_source_has_no_gui_or_production_applied_dff_imports():
    source = inspect.getsource(generate)
    forbidden_snippets = [
        "import gui",
        "from gui",
        "write_applied_dff_cache",
        "run_applied_dff_pipeline",
        "run_applied_dff_batch",
        "run_applied_dff_features",
        "recompute_signal_only_f0_candidates",
        "correction_preview",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in source


def test_source_resolution_failure_writes_nothing(tmp_path):
    missing = tmp_path / "missing_run"

    result = run_signal_only_f0_diagnostic_review(
        missing,
        roi="CH1",
        diagnostic_id="signal_only_f0_20260617T120000Z_abc123",
    )

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert not missing.exists()
