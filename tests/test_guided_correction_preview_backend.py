import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import photometry_pipeline.preview.correction_preview as correction_preview_module

from photometry_pipeline.preview.correction_preview import (
    GUIDED_REFERENCE_PREVIEW_METHODS,
    METHOD_DIAGNOSTICS_FILENAME_TEMPLATE,
    PREVIEW_PROVENANCE_FILENAME,
    PREVIEW_SUMMARY_FILENAME,
    compute_guided_local_preview_dff_trace_in_memory,
    compute_guided_local_signal_only_f0_preview,
    run_guided_correction_preview_comparison,
    run_guided_local_correction_preview,
)
from photometry_pipeline.guided_diagnostic_cache import (
    DiagnosticCacheBuildRequest,
    artifact_record_from_request,
    write_artifact_record_json,
    write_build_request_json,
    write_json_file,
)
from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk


PREVIEW_ID = "preview_20260617T010203Z_abcd1234"


def test_local_dynamic_fit_preview_dff_uses_fractional_ratio_units():
    chunk = Chunk(
        chunk_id=0,
        source_file="memory",
        format="preview",
        time_sec=np.array([0.0, 1.0, 2.0]),
        uv_raw=np.full((3, 1), 10.0),
        sig_raw=np.array([[10.0], [10.5], [9.5]]),
        uv_fit=np.full((3, 1), 10.0),
        delta_f=np.array([[0.0], [0.5], [-0.5]]),
        fs_hz=1.0,
        channel_names=["CH1"],
    )

    evidence = (
        correction_preview_module._compute_local_dynamic_fit_dff_evidence(
            chunk,
            Config(
                baseline_method="uv_raw_percentile_session",
                baseline_percentile=10.0,
            ),
            method="global_linear_regression",
            roi="CH1",
        )
    )

    np.testing.assert_allclose(
        evidence["preview_dff"], [0.0, 0.05, -0.05]
    )
    assert evidence["local_preview_f0"] == 10.0
    assert evidence["dff_scale"] == "fractional_ratio"
    assert evidence["dff_formula"] == "delta_f / local_preview_f0"


def test_local_signal_only_f0_preview_computes_in_memory_and_preserves_negative():
    time_sec = np.arange(2400, dtype=float) / 20.0
    signal = (
        1.0
        + 0.08 * np.sin(time_sec * 0.2)
        + 0.03 * np.sin(time_sec * 1.3)
    )

    evidence = compute_guided_local_signal_only_f0_preview(
        signal, time_sec, roi_id="CH1"
    )

    assert evidence["status"] == "success"
    assert evidence["valid"] is True
    assert evidence["preview_only"] is True
    assert evidence["production_analysis"] is False
    assert evidence["strategy_family"] == "signal_only_f0"
    assert evidence["selected_strategy"] == "signal_only_f0"
    assert evidence["dynamic_fit_mode"] is None
    assert evidence["explicit_user_mark"] is False
    assert evidence["current_or_stale"] == "current"
    assert len(evidence["time_sec"]) == len(signal)
    assert len(evidence["preview_dff"]) == len(signal)
    assert evidence["metrics"]["negative_dff_count"] > 0
    assert np.min(evidence["preview_dff"]) < 0


def test_local_signal_only_f0_preview_flags_invalid_denominator():
    evidence = compute_guided_local_signal_only_f0_preview(
        np.zeros(100, dtype=float),
        np.arange(100, dtype=float) / 10.0,
        roi_id="CH1",
    )

    assert evidence["status"] == "invalid"
    assert evidence["valid"] is False
    assert evidence["preview_only"] is True
    assert evidence["preview_dff"].size == 0
    assert evidence["issues"]


def test_local_signal_only_f0_preview_exception_path_is_defensive(monkeypatch):
    class BadArray:
        def __array__(self, *_args, **_kwargs):
            raise RuntimeError("cannot coerce")

    monkeypatch.setattr(
        correction_preview_module,
        "compute_signal_only_f0_dff",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("candidate failed")
        ),
    )

    evidence = compute_guided_local_signal_only_f0_preview(
        BadArray(), BadArray(), roi_id="CH1"
    )

    assert evidence["status"] == "invalid"
    assert evidence["signal_raw"].size == 0
    assert evidence["time_sec"].size == 0
    assert evidence["issues"] == ["candidate failed"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_realistic_rwd_session(path: Path, *, offset: float) -> None:
    path.parent.mkdir(parents=True)
    rows = [
        '{"Light":{"Led410Enable":true;"Led470Enable":true};'
        '"Excitation":{"continuous_time":600};"Fps":40.0}',
        "TimeStamp,Events,CH1-410,CH1-470,CH2-410,CH2-470,",
    ]
    for index in range(12000):
        timestamp_ms = index * 50.0
        uv1 = offset + 100.0 + 0.01 * np.sin(index / 50.0)
        sig1 = offset + 125.0 + 0.012 * np.sin(index / 50.0)
        uv2 = offset + 90.0 + 0.008 * np.cos(index / 60.0)
        sig2 = offset + 112.0 + 0.01 * np.cos(index / 60.0)
        rows.append(
            f"{timestamp_ms:.3f},,{uv1:.6f},{sig1:.6f},"
            f"{uv2:.6f},{sig2:.6f},"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


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
    artifact_path = cache_root / "guided_diagnostic_cache_artifact.json"
    write_artifact_record_json(artifact_path, record)
    write_json_file(
        cache_root / "guided_diagnostic_cache_provenance.json",
        {
            "schema_version": "guided_diagnostic_cache.v1",
            "purpose": "guided_diagnostic_cache",
            "production_analysis": False,
        },
    )
    return cache_root


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


def test_preview_backend_success_from_diagnostic_cache_preserves_source_identity(tmp_path):
    cache_root = _make_diagnostic_cache(tmp_path)
    preview_dir = cache_root / "_guided_workflow" / "previews" / PREVIEW_ID

    result = run_guided_correction_preview_comparison(
        cache_root,
        preview_dir,
        roi="CH1",
        methods=["global_linear_regression"],
        preview_id=PREVIEW_ID,
        source_type="diagnostic_cache",
        overwrite=True,
    )

    assert result["ok"] is True
    provenance = _load_json(preview_dir / PREVIEW_PROVENANCE_FILENAME)
    assert provenance["source_type"] == "diagnostic_cache"
    assert provenance["completed_run_dir"] == ""
    assert provenance["diagnostic_cache"]["source_type"] == "diagnostic_cache"
    assert provenance["diagnostic_cache"]["cache_id"] == "cache_001"
    assert provenance["diagnostic_cache"]["cache_root_path"] == str(cache_root)
    assert provenance["diagnostic_cache"]["production_analysis"] is False
    assert provenance["diagnostic_cache"]["preliminary_cache"] is True
    assert provenance["pipeline_run_executed"] is False
    assert provenance["production_output"] is False


def test_preview_backend_refuses_production_diagnostic_cache(tmp_path):
    cache_root = _make_diagnostic_cache(tmp_path)
    artifact_path = cache_root / "guided_diagnostic_cache_artifact.json"
    artifact = _load_json(artifact_path)
    artifact["production_analysis"] = True
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    result = run_guided_correction_preview_comparison(
        cache_root,
        cache_root / "_guided_workflow" / "previews" / PREVIEW_ID,
        roi="CH1",
        methods=["global_linear_regression"],
        preview_id=PREVIEW_ID,
        source_type="diagnostic_cache",
        overwrite=True,
    )

    assert result["ok"] is False
    assert "must not be marked as production analysis" in "; ".join(result["errors"])


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


def test_local_preview_real_rwd_nonfirst_session_uses_selected_file_and_local_chunk_zero(
    tmp_path,
):
    first = tmp_path / "session-0" / "fluorescence.csv"
    third = tmp_path / "session-2" / "fluorescence.csv"
    _write_realistic_rwd_session(first, offset=0.0)
    _write_realistic_rwd_session(third, offset=20.0)
    config = tmp_path / "config.yaml"
    config.write_text(
        "target_fs_hz: 20.0\n"
        "chunk_duration_sec: 600.0\n"
        "rwd_time_col: TimeStamp\n"
        "uv_suffix: '-410'\n"
        "sig_suffix: '-470'\n"
        "lowpass_hz: 1.0\n"
        "filter_order: 3\n",
        encoding="utf-8",
    )

    result = run_guided_local_correction_preview(
        third,
        tmp_path / "local-preview",
        roi="CH2",
        chunk_index=2,
        adapter_chunk_index=0,
        segment_label="session-2",
        input_format="rwd",
        config_path=config,
        methods=["global_linear_regression"],
        preview_id="local_rwd_segment_2",
    )

    assert result["status"] == "success"
    assert result["source_file"] == str(third.resolve())
    assert result["source_file"] != str(first.resolve())
    assert result["adapter_local_chunk_id"] == 0
    provenance = _load_json(Path(result["preview_provenance_path"]))
    assert provenance["selected_segment_index"] == 2
    assert provenance["selected_segment_label"] == "session-2"
    assert provenance["adapter_local_chunk_id"] == 0
    assert provenance["source_file"] == str(third.resolve())
    assert provenance["selected_roi"] == "CH2"
    assert provenance["source_type"] == "local_raw_segment"
    assert provenance["preview_only"] is True
    assert provenance["production_analysis"] is False
    dynamic_evidence = result["method_statuses"][
        "global_linear_regression"
    ]["local_preview_dff_evidence"]
    assert dynamic_evidence["valid"] is True
    assert dynamic_evidence["roi_id"] == "CH2"
    assert dynamic_evidence["dynamic_fit_mode"] == (
        "global_linear_regression"
    )
    assert dynamic_evidence["trace_source"] == (
        "local_correction_preview_dff"
    )
    assert dynamic_evidence["dff_scale"] == "fractional_ratio"
    assert dynamic_evidence["preview_only"] is True
    assert dynamic_evidence["production_analysis"] is False
    assert len(dynamic_evidence["time_sec"]) == len(
        dynamic_evidence["preview_dff"]
    )
    diagnostics = _load_json(
        Path(
            result["method_statuses"]["global_linear_regression"][
                "diagnostics_json"
            ]
        )
    )
    assert "_retained_local_dff_evidence" not in diagnostics
    assert "preview_dff" not in diagnostics
    summary = _load_json(Path(result["preview_summary_path"]))
    assert "local_preview_dff_evidence" not in summary[
        "method_statuses"
    ]["global_linear_regression"]
    evidence = result["signal_only_f0_preview_evidence"]
    assert evidence["evidence_source_type"] == "local_preview"
    assert evidence["strategy_family"] == "signal_only_f0"
    assert evidence["preview_only"] is True
    assert evidence["production_analysis"] is False
    assert evidence["explicit_user_mark"] is False
    assert evidence["current_or_stale"] == "current"
    assert len(evidence["time_sec"]) == len(evidence["preview_dff"])
    assert provenance["signal_only_f0_preview_evidence"][
        "strategy_family"
    ] == "signal_only_f0"
    assert not (tmp_path / "applied_dff").exists()


def test_on_demand_local_preview_service_is_no_write_and_strategy_exact(
    tmp_path, monkeypatch
):
    source = tmp_path / "session-2" / "fluorescence.csv"
    _write_realistic_rwd_session(source, offset=20.0)
    config = tmp_path / "config.yaml"
    config.write_text(
        "target_fs_hz: 20.0\n"
        "chunk_duration_sec: 600.0\n"
        "rwd_time_col: TimeStamp\n"
        "uv_suffix: '-410'\n"
        "sig_suffix: '-470'\n"
        "lowpass_hz: 1.0\n"
        "filter_order: 3\n",
        encoding="utf-8",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("on-demand preview attempted an artifact write")

    monkeypatch.setattr(correction_preview_module, "_write_json", forbidden)
    monkeypatch.setattr(
        correction_preview_module, "_write_method_trace_csv", forbidden
    )
    monkeypatch.setattr(
        correction_preview_module.os, "makedirs", forbidden
    )

    result = compute_guided_local_preview_dff_trace_in_memory(
        source,
        roi="CH2",
        chunk_index=2,
        adapter_chunk_index=0,
        segment_label="session-2",
        input_format="rwd",
        config_path=config,
        strategy_family="dynamic_fit",
        strategy="global_linear_regression",
        dynamic_fit_mode="global_linear_regression",
    )

    assert result["valid"] is True
    assert result["strategy_family"] == "dynamic_fit"
    assert result["strategy"] == "global_linear_regression"
    assert result["dynamic_fit_mode"] == "global_linear_regression"
    assert result["discovered_session_index"] == 2
    assert result["adapter_chunk_index"] == 0
    assert result["dff_scale"] == "fractional_ratio"
    assert result["preview_only"] is True
    assert result["production_analysis"] is False
    assert len(result["time_sec"]) == len(result["preview_dff"])
    assert not (tmp_path / "local-preview").exists()


def test_on_demand_local_preview_service_rejects_strategy_mismatch(
    tmp_path,
):
    source = tmp_path / "session-0" / "fluorescence.csv"
    _write_realistic_rwd_session(source, offset=0.0)
    config = tmp_path / "config.yaml"
    config.write_text(
        "target_fs_hz: 20.0\nchunk_duration_sec: 600.0\n",
        encoding="utf-8",
    )

    result = compute_guided_local_preview_dff_trace_in_memory(
        source,
        roi="CH1",
        chunk_index=0,
        input_format="rwd",
        config_path=config,
        strategy_family="signal_only_f0",
        strategy="global_linear_regression",
    )

    assert result["valid"] is False
    assert "Unsupported correction strategy" in result["issues"][0]


def test_local_preview_can_omit_signal_only_f0_evidence(tmp_path):
    source = tmp_path / "session-0" / "fluorescence.csv"
    _write_realistic_rwd_session(source, offset=0.0)
    config = tmp_path / "config.yaml"
    config.write_text(
        "target_fs_hz: 20.0\n"
        "chunk_duration_sec: 600.0\n"
        "rwd_time_col: TimeStamp\n"
        "uv_suffix: '-410'\n"
        "sig_suffix: '-470'\n"
        "lowpass_hz: 1.0\n"
        "filter_order: 3\n",
        encoding="utf-8",
    )

    result = run_guided_local_correction_preview(
        source,
        tmp_path / "local-preview",
        roi="CH1",
        chunk_index=0,
        adapter_chunk_index=0,
        segment_label="session-0",
        input_format="rwd",
        config_path=config,
        methods=["global_linear_regression"],
        include_signal_only_f0_preview=False,
        preview_id="local_rwd_without_signal_f0",
    )

    assert result["status"] == "success"
    assert result["signal_only_f0_preview_requested"] is False
    assert "signal_only_f0_preview_evidence" not in result
    provenance = _load_json(Path(result["preview_provenance_path"]))
    assert provenance["signal_only_f0_preview_requested"] is False
    assert "signal_only_f0_preview_evidence" not in provenance
    assert not (tmp_path / "applied_dff").exists()


def test_local_preview_can_run_signal_only_f0_without_reference_methods(
    tmp_path,
):
    source = tmp_path / "session-0" / "fluorescence.csv"
    _write_realistic_rwd_session(source, offset=0.0)
    config = tmp_path / "config.yaml"
    config.write_text(
        "target_fs_hz: 20.0\n"
        "chunk_duration_sec: 600.0\n"
        "rwd_time_col: TimeStamp\n"
        "uv_suffix: '-410'\n"
        "sig_suffix: '-470'\n",
        encoding="utf-8",
    )

    result = run_guided_local_correction_preview(
        source,
        tmp_path / "signal-only-preview",
        roi="CH1",
        chunk_index=0,
        input_format="rwd",
        config_path=config,
        methods=[],
        include_signal_only_f0_preview=True,
        preview_id="local_rwd_signal_f0_only",
    )

    assert result["status"] == "success"
    assert result["ok"] is True
    assert result["method_statuses"] == {}
    assert result["signal_only_f0_preview_requested"] is True
    assert result["signal_only_f0_preview_evidence"]["valid"] is True
    assert not (tmp_path / "applied_dff").exists()


def test_local_preview_adapter_failure_returns_full_load_context(tmp_path):
    source = tmp_path / "session-2" / "fluorescence.csv"
    _write_realistic_rwd_session(source, offset=20.0)
    config = tmp_path / "config.yaml"
    config.write_text(
        "target_fs_hz: 50.0\n"
        "chunk_duration_sec: 600.0\n"
        "rwd_time_col: TimeStamp\n"
        "uv_suffix: '-410'\n"
        "sig_suffix: '-470'\n",
        encoding="utf-8",
    )

    result = run_guided_local_correction_preview(
        source,
        tmp_path / "failed-preview",
        roi="CH1",
        chunk_index=2,
        adapter_chunk_index=0,
        segment_label="session-2",
        input_format="rwd",
        config_path=config,
        methods=["global_linear_regression"],
        preview_id="failed_local_rwd_segment_2",
    )

    assert result["status"] == "failed"
    details = result["local_preview_diagnostics"]
    assert details["selected_segment_label"] == "session-2"
    assert details["selected_segment_index"] == 2
    assert details["source_path"] == str(source.resolve())
    assert details["adapter_local_chunk_id"] == 0
    assert details["input_format"] == "rwd"
    assert "End Coverage Failure" in details["adapter_error"]
