import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.run_applied_dff_pipeline import AppliedDffPipelineError, run_applied_dff_pipeline


def _make_phasic_out(tmp_path: Path) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    t = np.arange(0.0, 20.0, 0.1)
    dff = 0.02 * np.sin(t)
    for center in (5.0, 12.0):
        dff += np.exp(-0.5 * ((t - center) / 0.12) ** 2)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([b"CH1"]))
        meta.create_dataset("chunk_ids", data=np.asarray([0, 1], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"chunk0.csv", b"chunk1.csv"]))
        roi_group = h5.create_group("roi/CH1")
        for chunk_id in (0, 1):
            grp = roi_group.create_group(f"chunk_{chunk_id}")
            grp.create_dataset("time_sec", data=t)
            grp.create_dataset("sig_raw", data=1.0 + dff + chunk_id)
            grp.create_dataset("dff", data=dff + chunk_id)
    return phasic_out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _patch_core_f0(monkeypatch):
    import tools.write_applied_dff_cache as writer

    def _fake_candidate(signal, time=None, *, return_uncapped_candidate=False, **_kwargs):
        assert return_uncapped_candidate is True
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
        f0_arr = np.ones_like(signal_arr)
        return {
            "signal_only_f0_candidate": np.minimum(f0_arr, signal_arr),
            "signal_only_f0_candidate_uncapped": f0_arr,
            "signal_only_f0_candidate_viability": "viable",
            "signal_only_f0_candidate_confidence": "high",
            "signal_only_f0_flags": [],
        }

    monkeypatch.setattr(writer.signal_f0_core, "compute_signal_only_f0_candidate", _fake_candidate)


def test_dynamic_fit_end_to_end_pipeline_passes(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = run_applied_dff_pipeline(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        output_root=tmp_path / "out",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["pipeline_passed"] is True
    assert summary["stage_write_applied_cache_passed"] is True
    assert summary["stage_verify_applied_cache_passed"] is True
    assert summary["stage_run_applied_features_passed"] is True
    assert summary["stage_verify_feature_outputs_passed"] is True
    assert summary["n_chunks"] > 0
    base = tmp_path / "out" / "CH1_dynamic_fit"
    assert (base / "applied" / "applied_trace_cache.h5").exists()
    assert (base / "features" / "features.csv").exists()
    assert (base / "features" / "feature_output_semantic_verification.json").exists()
    feature_summary = json.loads((base / "features" / "feature_summary.json").read_text(encoding="utf-8"))
    assert summary["n_chunks"] == feature_summary["n_chunks"]
    assert json.loads(Path(report["summary_json"]).read_text(encoding="utf-8"))["pipeline_passed"] is True


@pytest.mark.parametrize("roi", ["../CH1", "CH/1", "CH\\1"])
def test_unsafe_roi_refuses_before_writing_or_deleting(tmp_path, roi):
    phasic_out = _make_phasic_out(tmp_path)
    output_root = tmp_path / "out"
    sentinel = output_root / "sentinel.txt"
    output_root.mkdir()
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(AppliedDffPipelineError, match="unsafe"):
        run_applied_dff_pipeline(
            phasic_out,
            roi=roi,
            strategy="dynamic_fit",
            output_root=output_root,
            overwrite=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not any(output_root.glob("*dynamic_fit"))


def test_overwrite_deletion_target_must_be_inside_output_root(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    output_root = tmp_path / "out"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "do_not_delete.txt").write_text("keep", encoding="utf-8")

    import tools.run_applied_dff_pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "_paths",
        lambda output_root, roi, strategy: {
            "base": outside,
            "applied": outside / "applied",
            "features": outside / "features",
            "pipeline": outside / "pipeline",
        },
    )

    with pytest.raises(AppliedDffPipelineError, match="outside output_root"):
        run_applied_dff_pipeline(
            phasic_out,
            roi="CH1",
            strategy="dynamic_fit",
            output_root=output_root,
            overwrite=True,
        )

    assert (outside / "do_not_delete.txt").exists()


def test_default_feature_config_hash_matches_feature_summary(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = run_applied_dff_pipeline(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        output_root=tmp_path / "out",
        overwrite=True,
    )

    feature_summary = json.loads((tmp_path / "out" / "CH1_dynamic_fit" / "features" / "feature_summary.json").read_text(encoding="utf-8"))
    assert report["summary"]["feature_config_hash"]
    assert report["summary"]["feature_config_hash"] == feature_summary["feature_config_hash"]


def test_signal_only_f0_end_to_end_pipeline_passes(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch)

    report = run_applied_dff_pipeline(
        phasic_out,
        roi="CH1",
        strategy="signal_only_f0",
        output_root=tmp_path / "out",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["pipeline_passed"] is True
    assert summary["applied_trace_source"] == "signal_only_f0_dff"
    assert summary["stage_verify_feature_outputs_passed"] is True


@pytest.mark.parametrize("strategy", ["auto", "no_correction"])
def test_unsupported_strategies_fail_clearly(tmp_path, strategy):
    phasic_out = _make_phasic_out(tmp_path)
    with pytest.raises(AppliedDffPipelineError, match="unsupported"):
        run_applied_dff_pipeline(phasic_out, roi="CH1", strategy=strategy, output_root=tmp_path / "out")


def test_dry_run_writes_nothing(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    report = run_applied_dff_pipeline(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        output_root=tmp_path / "out",
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert report["would_write_applied_cache"] is True
    assert not (tmp_path / "out").exists()


def test_failure_stops_downstream_stages(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    import tools.run_applied_dff_pipeline as pipeline

    def _fail_verify(*_args, **_kwargs):
        raise RuntimeError("forced verify failure")

    def _feature_should_not_run(*_args, **_kwargs):
        raise AssertionError("feature runner should not execute")

    monkeypatch.setattr(pipeline, "verify_applied_dff_cache", _fail_verify)
    monkeypatch.setattr(pipeline, "run_applied_dff_features", _feature_should_not_run)

    with pytest.raises(AppliedDffPipelineError, match="forced verify failure") as exc_info:
        run_applied_dff_pipeline(phasic_out, roi="CH1", strategy="dynamic_fit", output_root=tmp_path / "out", overwrite=True)

    assert exc_info.value.report["failed_stage"] == "verify_applied_dff_cache"
    assert not (tmp_path / "out" / "CH1_dynamic_fit" / "features" / "features.csv").exists()


def test_output_exists_without_overwrite_fails(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    run_applied_dff_pipeline(phasic_out, roi="CH1", strategy="dynamic_fit", output_root=tmp_path / "out", overwrite=True)

    with pytest.raises(AppliedDffPipelineError, match="refusing without --overwrite"):
        run_applied_dff_pipeline(phasic_out, roi="CH1", strategy="dynamic_fit", output_root=tmp_path / "out")


def test_overwrite_only_affects_pipeline_output_directory(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    legacy = phasic_out / "features" / "features.csv"
    legacy.parent.mkdir()
    legacy.write_text("roi,chunk_id\nCH1,0\n", encoding="utf-8")
    before = _sha256(legacy)

    run_applied_dff_pipeline(phasic_out, roi="CH1", strategy="dynamic_fit", output_root=tmp_path / "out", overwrite=True)
    run_applied_dff_pipeline(phasic_out, roi="CH1", strategy="dynamic_fit", output_root=tmp_path / "out", overwrite=True)

    assert _sha256(legacy) == before


def test_read_only_source_guarantees(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    source = phasic_out / "phasic_trace_cache.h5"
    before = _sha256(source)

    report = run_applied_dff_pipeline(phasic_out, roi="CH1", strategy="dynamic_fit", output_root=tmp_path / "out", overwrite=True)

    assert _sha256(source) == before
    assert report["summary"]["hdf5_modified_source_phasic_cache"] is False


def test_feature_config_passes_through(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    config = tmp_path / "feature_config.json"
    config.write_text(json.dumps({"peak_threshold_method": "absolute", "peak_threshold_abs": 0.5}), encoding="utf-8")

    report = run_applied_dff_pipeline(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        output_root=tmp_path / "out",
        feature_config=config,
        overwrite=True,
    )

    assert report["summary"]["feature_config_path"] == str(config.resolve())
    assert report["summary"]["feature_config_hash"]
    feature_summary = json.loads((tmp_path / "out" / "CH1_dynamic_fit" / "features" / "feature_summary.json").read_text(encoding="utf-8"))
    assert report["summary"]["feature_config_hash"] == feature_summary["feature_config_hash"]
    assert report["summary"]["feature_config_hash"] != _sha256(config)
    provenance = json.loads((tmp_path / "out" / "CH1_dynamic_fit" / "features" / "feature_provenance.json").read_text(encoding="utf-8"))
    assert provenance["feature_config_path"] == str(config.resolve())


def test_failed_feature_output_semantic_verification_causes_pipeline_failure(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    import tools.run_applied_dff_pipeline as pipeline

    def _fail_semantic(*_args, **_kwargs):
        raise RuntimeError("forced semantic failure")

    monkeypatch.setattr(pipeline, "verify_applied_dff_feature_outputs", _fail_semantic)

    with pytest.raises(AppliedDffPipelineError, match="forced semantic failure") as exc_info:
        run_applied_dff_pipeline(phasic_out, roi="CH1", strategy="dynamic_fit", output_root=tmp_path / "out", overwrite=True)

    assert exc_info.value.report["failed_stage"] == "verify_applied_dff_feature_outputs"
