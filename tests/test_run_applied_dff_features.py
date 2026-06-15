import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tools.run_applied_dff_features import AppliedDffFeatureRunError, run_applied_dff_features
from tools.write_applied_dff_cache import write_applied_dff_cache


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


def test_dynamic_fit_applied_features_run_from_verified_cache(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    source_before = _sha256(phasic_out / "phasic_trace_cache.h5")
    write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)

    report = run_applied_dff_features(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        applied_output_dir=phasic_out / "applied_dff",
        overwrite=True,
    )

    assert Path(report["features_csv"]).exists()
    assert Path(report["features_json"]).exists()
    assert Path(report["feature_summary_csv"]).exists()
    assert Path(report["feature_summary_json"]).exists()
    assert Path(report["feature_provenance_json"]).exists()
    summary = report["summary"]
    assert summary["requested_correction_strategy"] == "dynamic_fit"
    assert summary["applied_trace_source"] == "dynamic_fit_dff"
    assert summary["feature_detection_input"] == "applied_dff"
    assert summary["applied_cache_verification_passed"] is True
    assert summary["n_chunks_processed"] == 2
    assert _sha256(phasic_out / "phasic_trace_cache.h5") == source_before
    assert not (phasic_out / "features" / "features.csv").exists()


def test_signal_only_f0_applied_features_run_from_verified_cache(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch)
    write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="signal_only_f0", overwrite=True)

    report = run_applied_dff_features(
        phasic_out,
        roi="CH1",
        strategy="signal_only_f0",
        applied_output_dir=phasic_out / "applied_dff",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["requested_correction_strategy"] == "signal_only_f0"
    assert summary["applied_trace_source"] == "signal_only_f0_dff"
    features = pd.read_csv(report["features_csv"])
    assert set(features["feature_detection_input"]) == {"applied_dff"}
    assert set(features["applied_trace_source"]) == {"signal_only_f0_dff"}
    assert features["applied_trace_cache_sha256"].astype(str).str.len().min() > 0


def test_verifier_failure_blocks_feature_detection(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    writer_report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    with h5py.File(writer_report["applied_trace_cache_path"], "a") as h5:
        h5["roi/CH1/chunk_0/applied_dff"][0] += 1.0
    out = tmp_path / "features"

    with pytest.raises(AppliedDffFeatureRunError, match="applied cache verification failed"):
        run_applied_dff_features(
            phasic_out,
            roi="CH1",
            strategy="dynamic_fit",
            applied_output_dir=writer_report["output_dir"],
            output_dir=out,
            overwrite=True,
        )
    assert not (out / "features.csv").exists()


def test_incomplete_applied_cache_blocks_feature_detection(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        del h5["roi/CH1/chunk_1/dff"]
    writer_report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    out = tmp_path / "features"

    with pytest.raises(AppliedDffFeatureRunError, match="applied trace is incomplete"):
        run_applied_dff_features(
            phasic_out,
            roi="CH1",
            strategy="dynamic_fit",
            applied_output_dir=writer_report["output_dir"],
            output_dir=out,
            overwrite=True,
        )
    assert not (out / "features.csv").exists()


def test_wrong_strategy_blocks_feature_detection(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    writer_report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)

    with pytest.raises(AppliedDffFeatureRunError, match="applied cache verification failed"):
        run_applied_dff_features(
            phasic_out,
            roi="CH1",
            strategy="signal_only_f0",
            applied_output_dir=writer_report["output_dir"],
            output_dir=tmp_path / "features",
            overwrite=True,
        )


def test_output_exists_without_overwrite_fails(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    writer_report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    run_applied_dff_features(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer_report["output_dir"], output_dir=tmp_path / "features", overwrite=True)

    with pytest.raises(AppliedDffFeatureRunError, match="refusing to overwrite"):
        run_applied_dff_features(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer_report["output_dir"], output_dir=tmp_path / "features")


def test_dry_run_writes_no_outputs_and_modifies_nothing(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    before = _sha256(phasic_out / "phasic_trace_cache.h5")
    report = run_applied_dff_features(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        applied_output_dir=tmp_path / "missing",
        output_dir=tmp_path / "features",
        dry_run=True,
    )
    assert report["dry_run"] is True
    assert report["would_verify_applied_cache"] is True
    assert not (tmp_path / "features").exists()
    assert _sha256(phasic_out / "phasic_trace_cache.h5") == before


def test_legacy_features_are_not_modified(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    legacy = phasic_out / "features" / "features.csv"
    legacy.parent.mkdir()
    legacy.write_text("roi,chunk_id,peak_count\nCH1,0,999\n", encoding="utf-8")
    before = _sha256(legacy)
    writer_report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)

    report = run_applied_dff_features(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer_report["output_dir"], output_dir=tmp_path / "features", overwrite=True)

    assert _sha256(legacy) == before
    assert report["summary"]["legacy_features_modified"] is False


def test_applied_cache_is_not_modified(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    writer_report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    before = _sha256(Path(writer_report["applied_trace_cache_path"]))

    run_applied_dff_features(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer_report["output_dir"], output_dir=tmp_path / "features", overwrite=True)

    assert _sha256(Path(writer_report["applied_trace_cache_path"])) == before


def test_feature_rows_carry_required_provenance_columns(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    writer_report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)

    report = run_applied_dff_features(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer_report["output_dir"], output_dir=tmp_path / "features", overwrite=True)

    features = pd.read_csv(report["features_csv"])
    for column in (
        "roi",
        "chunk_id",
        "source_file",
        "requested_correction_strategy",
        "applied_correction_strategy",
        "applied_trace_source",
        "feature_detection_input",
        "applied_trace_cache_sha256",
        "source_phasic_cache_sha256",
        "upstream_warning_level",
        "upstream_review_required",
        "upstream_flags",
    ):
        assert column in features.columns


def test_skip_verification_is_recorded_as_skipped_not_passed(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    writer_report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)

    report = run_applied_dff_features(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        applied_output_dir=writer_report["output_dir"],
        output_dir=tmp_path / "features",
        overwrite=True,
        skip_verification=True,
    )

    summary = report["summary"]
    assert summary["applied_cache_verification_passed"] is False
    assert summary["applied_cache_verification_skipped"] is True
    provenance = json.loads(Path(report["feature_provenance_json"]).read_text(encoding="utf-8"))
    assert provenance["applied_cache_verification_passed"] is False
    assert provenance["applied_cache_verification_skipped"] is True
    assert provenance["applied_cache_verification_summary"] == {"skipped": True}


@pytest.mark.parametrize("strategy", ["no_correction", "auto"])
def test_no_correction_and_auto_refuse_clearly(tmp_path, strategy):
    phasic_out = _make_phasic_out(tmp_path)
    with pytest.raises(AppliedDffFeatureRunError, match="unsupported"):
        run_applied_dff_features(
            phasic_out,
            roi="CH1",
            strategy=strategy,
            applied_output_dir=tmp_path / "missing",
            output_dir=tmp_path / "features",
        )


def test_feature_config_provenance_is_recorded(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    writer_report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    first = run_applied_dff_features(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer_report["output_dir"], output_dir=tmp_path / "default", overwrite=True)
    assert first["summary"]["feature_config_hash"]
    assert first["summary"]["feature_config_path"] == ""

    config_path = tmp_path / "feature_config.json"
    config_path.write_text(json.dumps({"peak_threshold_method": "absolute", "peak_threshold_abs": 0.5}), encoding="utf-8")
    second = run_applied_dff_features(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        applied_output_dir=writer_report["output_dir"],
        output_dir=tmp_path / "custom",
        feature_config=config_path,
        overwrite=True,
    )
    assert second["summary"]["feature_config_path"] == str(config_path.resolve())
    provenance = json.loads(Path(second["feature_provenance_json"]).read_text(encoding="utf-8"))
    assert json.loads(provenance["feature_config_json"])["peak_threshold_abs"] == 0.5
