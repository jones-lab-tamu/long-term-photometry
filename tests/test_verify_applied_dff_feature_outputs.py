import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tools.run_applied_dff_features import run_applied_dff_features
from tools.verify_applied_dff_feature_outputs import (
    AppliedDffFeatureOutputVerificationError,
    verify_applied_dff_feature_outputs,
)
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


def _make_outputs(tmp_path: Path, *, strategy: str = "dynamic_fit", monkeypatch=None):
    phasic_out = _make_phasic_out(tmp_path)
    if strategy == "signal_only_f0":
        _patch_core_f0(monkeypatch)
    writer = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy=strategy, overwrite=True)
    runner = run_applied_dff_features(
        phasic_out,
        roi="CH1",
        strategy=strategy,
        applied_output_dir=writer["output_dir"],
        output_dir=tmp_path / "features",
        overwrite=True,
    )
    return phasic_out, writer, runner


def _sync_features_json(feature_dir: Path, df: pd.DataFrame) -> None:
    (feature_dir / "features.json").write_text(
        json.dumps({"features": df.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )


def test_dynamic_fit_applied_feature_outputs_verify_successfully(tmp_path):
    phasic_out, writer, runner = _make_outputs(tmp_path)

    report = verify_applied_dff_feature_outputs(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        applied_output_dir=writer["output_dir"],
        feature_output_dir=runner["output_dir"],
    )

    assert report["verification_passed"] is True
    assert report["semantic_status"] == "pass"
    assert report["n_detector_row_count_mismatches"] == 0
    assert report["n_detector_value_mismatches"] == 0


def test_signal_only_f0_applied_feature_outputs_verify_successfully(tmp_path, monkeypatch):
    phasic_out, writer, runner = _make_outputs(tmp_path, strategy="signal_only_f0", monkeypatch=monkeypatch)

    report = verify_applied_dff_feature_outputs(
        phasic_out,
        roi="CH1",
        strategy="signal_only_f0",
        applied_output_dir=writer["output_dir"],
        feature_output_dir=runner["output_dir"],
    )

    assert report["verification_passed"] is True
    assert report["applied_trace_source"] == "signal_only_f0_dff"
    features = pd.read_csv(runner["features_csv"])
    assert set(features["applied_trace_source"]) == {"signal_only_f0_dff"}


def test_feature_output_detector_mismatch_fails(tmp_path):
    phasic_out, writer, runner = _make_outputs(tmp_path)
    feature_dir = Path(runner["output_dir"])
    df = pd.read_csv(feature_dir / "features.csv")
    df.loc[0, "peak_count"] = int(df.loc[0, "peak_count"]) + 1
    df.to_csv(feature_dir / "features.csv", index=False)
    _sync_features_json(feature_dir, df)

    with pytest.raises(AppliedDffFeatureOutputVerificationError, match="detector value mismatch"):
        verify_applied_dff_feature_outputs(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer["output_dir"], feature_output_dir=feature_dir)


def test_feature_provenance_mismatch_fails(tmp_path):
    phasic_out, writer, runner = _make_outputs(tmp_path)
    feature_dir = Path(runner["output_dir"])
    df = pd.read_csv(feature_dir / "features.csv")
    df.loc[0, "applied_trace_source"] = "signal_only_f0_dff"
    df.to_csv(feature_dir / "features.csv", index=False)
    _sync_features_json(feature_dir, df)

    with pytest.raises(AppliedDffFeatureOutputVerificationError, match="feature provenance value mismatch"):
        verify_applied_dff_feature_outputs(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer["output_dir"], feature_output_dir=feature_dir)


def test_missing_required_provenance_column_fails(tmp_path):
    phasic_out, writer, runner = _make_outputs(tmp_path)
    feature_dir = Path(runner["output_dir"])
    df = pd.read_csv(feature_dir / "features.csv").drop(columns=["applied_trace_cache_sha256"])
    df.to_csv(feature_dir / "features.csv", index=False)
    _sync_features_json(feature_dir, df)

    with pytest.raises(AppliedDffFeatureOutputVerificationError, match="missing required provenance columns") as exc_info:
        verify_applied_dff_feature_outputs(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer["output_dir"], feature_output_dir=feature_dir)
    assert exc_info.value.report["n_missing_required_provenance_columns"] > 0


def test_one_row_per_chunk_behavior_is_classified_not_failed(tmp_path):
    phasic_out, writer, runner = _make_outputs(tmp_path)
    report = verify_applied_dff_feature_outputs(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer["output_dir"], feature_output_dir=runner["output_dir"])
    assert report["one_feature_row_per_chunk"] is True
    assert report["one_feature_row_per_chunk_matches_detector"] is True
    assert report["feature_output_granularity"] == "chunk_summary"
    assert report["verification_passed"] is True


def test_empty_output_behavior_is_classified_when_detector_also_empty(tmp_path, monkeypatch):
    phasic_out, writer, runner = _make_outputs(tmp_path)
    feature_dir = Path(runner["output_dir"])
    empty = pd.DataFrame(columns=pd.read_csv(feature_dir / "features.csv").columns)
    empty.to_csv(feature_dir / "features.csv", index=False)
    _sync_features_json(feature_dir, empty)
    summary = json.loads((feature_dir / "feature_summary.json").read_text(encoding="utf-8"))
    summary["n_features"] = 0
    (feature_dir / "feature_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    import tools.verify_applied_dff_feature_outputs as verifier

    monkeypatch.setattr(verifier, "extract_features", lambda chunk, cfg: pd.DataFrame())
    report = verifier.verify_applied_dff_feature_outputs(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        applied_output_dir=writer["output_dir"],
        feature_output_dir=feature_dir,
    )
    assert report["feature_output_granularity"] == "empty"
    assert report["n_detector_rows_expected"] == 0
    assert report["verification_passed"] is True


def test_applied_cache_verification_failure_blocks_semantic_verifier(tmp_path):
    phasic_out, writer, runner = _make_outputs(tmp_path)
    with h5py.File(writer["applied_trace_cache_path"], "a") as h5:
        h5["roi/CH1/chunk_0/applied_dff"][0] += 1.0

    with pytest.raises(AppliedDffFeatureOutputVerificationError, match="applied cache verification failed"):
        verify_applied_dff_feature_outputs(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer["output_dir"], feature_output_dir=runner["output_dir"])


def test_applied_cache_missing_dataset_fails_before_semantic_comparison(tmp_path):
    phasic_out, writer, runner = _make_outputs(tmp_path)
    with h5py.File(writer["applied_trace_cache_path"], "a") as h5:
        del h5["roi/CH1/chunk_0/applied_dff"]

    with pytest.raises(AppliedDffFeatureOutputVerificationError, match="applied cache verification failed") as exc_info:
        verify_applied_dff_feature_outputs(
            phasic_out,
            roi="CH1",
            strategy="dynamic_fit",
            applied_output_dir=writer["output_dir"],
            feature_output_dir=runner["output_dir"],
        )

    assert "missing required dataset" in str(exc_info.value)
    assert not isinstance(exc_info.value.__cause__, KeyError)


def test_read_only_verification(tmp_path):
    phasic_out, writer, runner = _make_outputs(tmp_path)
    legacy = phasic_out / "features" / "features.csv"
    legacy.parent.mkdir()
    legacy.write_text("roi,chunk_id\nCH1,0\n", encoding="utf-8")
    source_before = _sha256(phasic_out / "phasic_trace_cache.h5")
    applied_before = _sha256(Path(writer["applied_trace_cache_path"]))
    legacy_before = _sha256(legacy)
    features_before = _sha256(Path(runner["features_csv"]))

    report = verify_applied_dff_feature_outputs(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=writer["output_dir"], feature_output_dir=runner["output_dir"])

    assert report["hdf5_modified_source_phasic_cache"] is False
    assert report["applied_cache_modified"] is False
    assert report["legacy_features_modified"] is False
    assert report["feature_outputs_modified"] is False
    assert _sha256(phasic_out / "phasic_trace_cache.h5") == source_before
    assert _sha256(Path(writer["applied_trace_cache_path"])) == applied_before
    assert _sha256(legacy) == legacy_before
    assert _sha256(Path(runner["features_csv"])) == features_before


def test_wrong_strategy_fails(tmp_path):
    phasic_out, writer, runner = _make_outputs(tmp_path)
    with pytest.raises(AppliedDffFeatureOutputVerificationError, match="applied cache verification failed"):
        verify_applied_dff_feature_outputs(phasic_out, roi="CH1", strategy="signal_only_f0", applied_output_dir=writer["output_dir"], feature_output_dir=runner["output_dir"])
