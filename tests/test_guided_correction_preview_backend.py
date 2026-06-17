import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from photometry_pipeline.preview.correction_preview import (
    GUIDED_REFERENCE_PREVIEW_METHODS,
    METHOD_DIAGNOSTICS_FILENAME_TEMPLATE,
    PREVIEW_PROVENANCE_FILENAME,
    PREVIEW_SUMMARY_FILENAME,
    run_guided_correction_preview_comparison,
)


PREVIEW_ID = "preview_20260617T010203Z_abcd1234"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_completed_run(tmp_path: Path, *, missing_field: str | None = None) -> Path:
    run_dir = tmp_path / "run"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    (run_dir / "run_report.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (phasic_out / "config_used.yaml").write_text(
        "\n".join(
            [
                "target_fs_hz: 20.0",
                "lowpass_hz: 1.0",
                "filter_order: 3",
                "dynamic_fit_mode: robust_global_event_reject",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    t = np.arange(400, dtype=float) / 20.0
    uv = 1.0 + 0.02 * np.sin(t * 0.7)
    sig = 1.2 * uv + 0.05 * np.exp(-0.5 * ((t - 8.0) / 0.5) ** 2) + 0.01 * np.sin(t * 1.5)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([b"CH1"]))
        meta.create_dataset("chunk_ids", data=np.asarray([0], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"mock.csv"]))
        grp = h5.create_group("roi/CH1/chunk_0")
        if missing_field != "time_sec":
            grp.create_dataset("time_sec", data=t)
        if missing_field != "sig_raw":
            grp.create_dataset("sig_raw", data=sig)
        if missing_field != "uv_raw":
            grp.create_dataset("uv_raw", data=uv)
    return run_dir


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_preview_backend_success_from_completed_run_writes_only_preview_artifacts(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic_out = run_dir / "_analysis" / "phasic_out"
    cache = phasic_out / "phasic_trace_cache.h5"
    config = phasic_out / "config_used.yaml"
    cache_before = _sha256(cache)
    config_before = _sha256(config)

    result = run_guided_correction_preview_comparison(
        run_dir,
        roi="CH1",
        preview_id=PREVIEW_ID,
        overwrite=True,
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    preview_dir = Path(result["preview_output_dir"])
    assert preview_dir.exists()
    assert (preview_dir / PREVIEW_PROVENANCE_FILENAME).exists()
    assert (preview_dir / PREVIEW_SUMMARY_FILENAME).exists()
    for method in GUIDED_REFERENCE_PREVIEW_METHODS:
        assert result["method_statuses"][method]["status"] == "success"
        assert (preview_dir / METHOD_DIAGNOSTICS_FILENAME_TEMPLATE.format(method=method)).exists()
        assert (preview_dir / f"method_{method}_trace.csv").exists()

    assert _sha256(cache) == cache_before
    assert _sha256(config) == config_before
    assert not (preview_dir / "MANIFEST.json").exists()
    assert not (preview_dir / "applied_dff").exists()
    assert not (preview_dir / "features").exists()
    assert not (phasic_out / "applied_dff").exists()
    assert not (phasic_out / "features").exists()


def test_preview_backend_success_from_phasic_cache_source_requires_external_output(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic_out = run_dir / "_analysis" / "phasic_out"
    preview_dir = tmp_path / "external_preview"

    result = run_guided_correction_preview_comparison(
        phasic_out,
        preview_dir,
        roi="CH1",
        methods=["global_linear_regression"],
        preview_id=PREVIEW_ID,
        source_type="phasic_cache",
        overwrite=True,
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert Path(result["preview_output_dir"]) == preview_dir.resolve()
    provenance = _load_json(preview_dir / PREVIEW_PROVENANCE_FILENAME)
    assert provenance["source_type"] == "phasic_cache"


def test_preview_backend_refuses_protected_output_namespaces_without_writing(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic_out = run_dir / "_analysis" / "phasic_out"
    unsafe_dirs = [
        run_dir,
        run_dir / "not_preview",
        phasic_out,
        phasic_out / "features" / "preview",
        phasic_out / "applied_dff" / "preview",
    ]

    for output_dir in unsafe_dirs:
        result = run_guided_correction_preview_comparison(
            run_dir,
            output_dir,
            roi="CH1",
            preview_id=PREVIEW_ID,
            overwrite=True,
        )
        assert result["ok"] is False
        assert not (Path(output_dir) / PREVIEW_PROVENANCE_FILENAME).exists()
        assert not (Path(output_dir) / PREVIEW_SUMMARY_FILENAME).exists()


def test_preview_backend_refuses_existing_output_without_overwrite(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    preview_dir = run_dir / "_guided_workflow" / "previews" / PREVIEW_ID
    preview_dir.mkdir(parents=True)
    sentinel = preview_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    result = run_guided_correction_preview_comparison(
        run_dir,
        preview_dir,
        roi="CH1",
        preview_id=PREVIEW_ID,
        overwrite=False,
    )

    assert result["ok"] is False
    assert sentinel.exists()


def test_preview_backend_overwrite_clears_only_valid_preview_dir(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    phasic_out = run_dir / "_analysis" / "phasic_out"
    preview_dir = run_dir / "_guided_workflow" / "previews" / PREVIEW_ID
    preview_dir.mkdir(parents=True)
    (preview_dir / "old.txt").write_text("old", encoding="utf-8")
    (phasic_out / "source_sentinel.txt").write_text("source", encoding="utf-8")

    result = run_guided_correction_preview_comparison(
        run_dir,
        preview_dir,
        roi="CH1",
        methods=["global_linear_regression"],
        preview_id=PREVIEW_ID,
        overwrite=True,
    )

    assert result["ok"] is True
    assert not (preview_dir / "old.txt").exists()
    assert (phasic_out / "source_sentinel.txt").exists()


def test_preview_backend_refuses_to_delete_existing_arbitrary_external_dir(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    preview_dir = tmp_path / "external_preview"
    preview_dir.mkdir()
    sentinel = preview_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    result = run_guided_correction_preview_comparison(
        run_dir,
        preview_dir,
        roi="CH1",
        methods=["global_linear_regression"],
        preview_id=PREVIEW_ID,
        overwrite=True,
    )

    assert result["ok"] is False
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (preview_dir / PREVIEW_PROVENANCE_FILENAME).exists()
    assert not (preview_dir / PREVIEW_SUMMARY_FILENAME).exists()


def test_preview_backend_can_overwrite_existing_external_preview_scoped_leaf(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    preview_dir = tmp_path / "preview_root" / "_guided_workflow" / "previews" / PREVIEW_ID
    preview_dir.mkdir(parents=True)
    old = preview_dir / "old.txt"
    old.write_text("old", encoding="utf-8")

    result = run_guided_correction_preview_comparison(
        run_dir,
        preview_dir,
        roi="CH1",
        methods=["global_linear_regression"],
        preview_id=PREVIEW_ID,
        overwrite=True,
    )

    assert result["ok"] is True
    assert not old.exists()
    assert (preview_dir / PREVIEW_PROVENANCE_FILENAME).exists()
    assert (preview_dir / PREVIEW_SUMMARY_FILENAME).exists()


def test_preview_backend_rejects_disallowed_methods_without_writing(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    preview_dir = run_dir / "_guided_workflow" / "previews" / PREVIEW_ID
    for method in ["signal_only_f0", "auto", "needs_review", "no_correction", "unknown"]:
        result = run_guided_correction_preview_comparison(
            run_dir,
            preview_dir,
            roi="CH1",
            methods=[method],
            preview_id=PREVIEW_ID,
            overwrite=True,
        )
        assert result["ok"] is False
        assert not preview_dir.exists()


def test_preview_backend_missing_cache_or_config_fails_without_writing(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    preview_dir = run_dir / "_guided_workflow" / "previews" / PREVIEW_ID
    (run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").unlink()

    result = run_guided_correction_preview_comparison(
        run_dir,
        preview_dir,
        roi="CH1",
        preview_id=PREVIEW_ID,
        overwrite=True,
    )

    assert result["ok"] is False
    assert not preview_dir.exists()


def test_preview_backend_missing_roi_chunk_window_or_required_array_fails_without_writing(tmp_path):
    run_dir = _make_completed_run(tmp_path)
    preview_dir = run_dir / "_guided_workflow" / "previews" / PREVIEW_ID

    missing_roi = run_guided_correction_preview_comparison(
        run_dir,
        preview_dir,
        roi="CH2",
        preview_id=PREVIEW_ID,
        overwrite=True,
    )
    assert missing_roi["ok"] is False
    assert not preview_dir.exists()

    missing_chunk = run_guided_correction_preview_comparison(
        run_dir,
        preview_dir,
        roi="CH1",
        chunk_index=99,
        preview_id=PREVIEW_ID,
        overwrite=True,
    )
    assert missing_chunk["ok"] is False
    assert not preview_dir.exists()

    bad_window = run_guided_correction_preview_comparison(
        run_dir,
        preview_dir,
        roi="CH1",
        window=(999.0, 1000.0),
        preview_id=PREVIEW_ID,
        overwrite=True,
    )
    assert bad_window["ok"] is False
    assert not preview_dir.exists()

    missing_field_run = _make_completed_run(tmp_path / "missing_field", missing_field="uv_raw")
    missing_field_preview = missing_field_run / "_guided_workflow" / "previews" / PREVIEW_ID
    missing_field = run_guided_correction_preview_comparison(
        missing_field_run,
        missing_field_preview,
        roi="CH1",
        preview_id=PREVIEW_ID,
        overwrite=True,
    )
    assert missing_field["ok"] is False
    assert "missing" in ";".join(missing_field["errors"]).lower()
    assert not missing_field_preview.exists()


def test_preview_backend_provenance_summary_and_diagnostics_have_no_recommendation(tmp_path):
    run_dir = _make_completed_run(tmp_path)

    result = run_guided_correction_preview_comparison(
        run_dir,
        roi="CH1",
        methods=["global_linear_regression"],
        preview_id=PREVIEW_ID,
        overwrite=True,
    )

    assert result["ok"] is True
    preview_dir = Path(result["preview_output_dir"])
    provenance = _load_json(preview_dir / PREVIEW_PROVENANCE_FILENAME)
    summary = _load_json(preview_dir / PREVIEW_SUMMARY_FILENAME)
    diagnostics = _load_json(
        preview_dir / METHOD_DIAGNOSTICS_FILENAME_TEMPLATE.format(method="global_linear_regression")
    )

    assert provenance["preview_only"] is True
    assert provenance["pipeline_run_executed"] is False
    assert provenance["manifest_written"] is False
    assert provenance["applied_dff_routed"] is False
    assert provenance["feature_extraction_run"] is False
    assert provenance["production_output"] is False
    assert provenance["strategy_recommendation"] is None
    assert summary["preview_only"] is True
    assert summary["strategy_recommendation"] is None
    assert diagnostics["preview_only"] is True
    assert diagnostics["strategy_recommendation"] is None
