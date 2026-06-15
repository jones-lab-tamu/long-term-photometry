import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tools.verify_applied_dff_cache import (
    AppliedDffCacheVerificationError,
    verify_applied_dff_cache,
)
from tools.write_applied_dff_cache import write_applied_dff_cache


def _make_phasic_out(tmp_path: Path) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
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
            grp.create_dataset("time_sec", data=np.arange(4, dtype=float))
            grp.create_dataset("sig_raw", data=np.asarray([0.8, 1.0, 1.2, 0.9], dtype=float) + chunk_id)
            grp.create_dataset("dff", data=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float) + chunk_id)
    return phasic_out


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _patch_core_f0(monkeypatch, *, f0=None):
    import tools.write_applied_dff_cache as writer

    def _fake_candidate(signal, time=None, *, return_uncapped_candidate=False, **_kwargs):
        assert return_uncapped_candidate is True
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
        f0_arr = np.ones_like(signal_arr) if f0 is None else np.asarray(f0(signal_arr), dtype=float).reshape(-1)
        return {
            "signal_only_f0_candidate": np.minimum(f0_arr, signal_arr),
            "signal_only_f0_candidate_uncapped": f0_arr,
            "signal_only_f0_candidate_viability": "viable",
            "signal_only_f0_candidate_confidence": "high",
            "signal_only_f0_flags": [],
        }

    monkeypatch.setattr(writer.signal_f0_core, "compute_signal_only_f0_candidate", _fake_candidate)


def _write_summary_files(output_dir: Path, summary: dict) -> None:
    summary_json = output_dir / "applied_correction_summary.json"
    summary_csv = output_dir / "applied_correction_summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    rows[0].update({key: str(value).lower() if isinstance(value, bool) else value for key, value in summary.items()})
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: rows[0].get(key, "") for key in fields})


def _load_summary(output_dir: Path) -> dict:
    return json.loads((output_dir / "applied_correction_summary.json").read_text(encoding="utf-8"))


def _refresh_applied_hash(output_dir: Path) -> None:
    summary = _load_summary(output_dir)
    summary["applied_trace_cache_sha256"] = _sha256(output_dir / "applied_trace_cache.h5")
    _write_summary_files(output_dir, summary)


def test_dynamic_fit_production_cache_verifies_successfully(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)

    report = verify_applied_dff_cache(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        applied_output_dir=phasic_out / "applied_dff",
    )

    assert report["verification_passed"] is True
    assert report["applied_trace_source"] == "dynamic_fit_dff"
    assert report["n_trace_formula_failures"] == 0
    assert report["n_missing_required_datasets"] == 0


def test_signal_only_f0_production_cache_verifies_successfully(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch, f0=lambda signal: np.ones_like(signal))
    write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="signal_only_f0", overwrite=True)

    report = verify_applied_dff_cache(
        phasic_out,
        roi="CH1",
        strategy="signal_only_f0",
        applied_output_dir=phasic_out / "applied_dff",
    )

    assert report["verification_passed"] is True
    assert report["applied_trace_source"] == "signal_only_f0_dff"
    assert report["n_trace_formula_failures"] == 0
    assert report["negative_dff_present"] is True
    summary = _load_summary(phasic_out / "applied_dff")
    assert summary["f0_source_for_signal_only_f0"] == "core_uncapped_signal_only_f0_candidate"
    assert summary["signal_only_f0_denominator_source"] == "signal_only_f0_candidate_uncapped"


def test_dynamic_fit_mismatch_fails(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    with h5py.File(report["applied_trace_cache_path"], "a") as h5:
        h5["roi/CH1/chunk_0/applied_dff"][0] += 1.0
    _refresh_applied_hash(Path(report["output_dir"]))

    with pytest.raises(AppliedDffCacheVerificationError, match="dynamic_fit trace mismatch"):
        verify_applied_dff_cache(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=report["output_dir"])


def test_signal_only_f0_formula_mismatch_fails(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch, f0=lambda signal: np.ones_like(signal))
    report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="signal_only_f0", overwrite=True)
    with h5py.File(report["applied_trace_cache_path"], "a") as h5:
        h5["roi/CH1/chunk_0/applied_dff"][0] += 1.0
    _refresh_applied_hash(Path(report["output_dir"]))

    with pytest.raises(AppliedDffCacheVerificationError, match="signal_only_f0 formula mismatch"):
        verify_applied_dff_cache(phasic_out, roi="CH1", strategy="signal_only_f0", applied_output_dir=report["output_dir"])


def test_applied_cache_hash_mismatch_fails(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    with h5py.File(report["applied_trace_cache_path"], "a") as h5:
        h5["roi/CH1/chunk_0/applied_dff"][0] += 1.0

    with pytest.raises(AppliedDffCacheVerificationError, match="applied_trace_cache_sha256 mismatch"):
        verify_applied_dff_cache(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=report["output_dir"])


def test_source_cache_hash_mismatch_fails(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        h5["roi/CH1/chunk_0/dff"][0] += 1.0

    with pytest.raises(AppliedDffCacheVerificationError, match="source_phasic_cache_sha256 mismatch"):
        verify_applied_dff_cache(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=report["output_dir"])


def test_wrong_strategy_in_summary_fails(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch, f0=lambda signal: np.ones_like(signal))
    report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="signal_only_f0", overwrite=True)
    summary = _load_summary(Path(report["output_dir"]))
    summary["requested_correction_strategy"] = "dynamic_fit"
    summary["applied_correction_strategy"] = "dynamic_fit"
    summary["applied_trace_source"] = "dynamic_fit_dff"
    _write_summary_files(Path(report["output_dir"]), summary)

    with pytest.raises(AppliedDffCacheVerificationError, match="requested strategy mismatch"):
        verify_applied_dff_cache(phasic_out, roi="CH1", strategy="signal_only_f0", applied_output_dir=report["output_dir"])


def test_missing_required_dataset_fails(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    with h5py.File(report["applied_trace_cache_path"], "a") as h5:
        del h5["roi/CH1/chunk_0/applied_dff"]
    _refresh_applied_hash(Path(report["output_dir"]))

    with pytest.raises(AppliedDffCacheVerificationError, match="missing required dataset"):
        verify_applied_dff_cache(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=report["output_dir"])


def test_missing_chunk_applied_trace_source_fails_without_raw_keyerror(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    with h5py.File(report["applied_trace_cache_path"], "a") as h5:
        del h5["roi/CH1/chunk_0/applied_trace_source"]
    _refresh_applied_hash(Path(report["output_dir"]))

    with pytest.raises(AppliedDffCacheVerificationError, match="missing required dataset") as exc_info:
        verify_applied_dff_cache(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=report["output_dir"])

    assert "missing required dataset" in str(exc_info.value)
    assert exc_info.value.report["n_missing_required_datasets"] > 0


def test_signal_only_f0_wrong_dataset_presence_fails(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch, f0=lambda signal: np.ones_like(signal))
    report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="signal_only_f0", overwrite=True)
    with h5py.File(report["applied_trace_cache_path"], "a") as h5:
        h5["roi/CH1/chunk_0"].create_dataset("dynamic_fit_dff", data=np.zeros(4))
    _refresh_applied_hash(Path(report["output_dir"]))

    with pytest.raises(AppliedDffCacheVerificationError, match="wrong strategy dataset"):
        verify_applied_dff_cache(phasic_out, roi="CH1", strategy="signal_only_f0", applied_output_dir=report["output_dir"])


def test_unavailable_chunk_contract_enforced(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        del h5["roi/CH1/chunk_1/dff"]
    report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)

    verified = verify_applied_dff_cache(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=report["output_dir"])
    assert verified["verification_passed"] is True
    assert verified["n_unavailable_chunks_checked"] == 1

    summary = _load_summary(Path(report["output_dir"]))
    summary["applied_trace_flags"] = ""
    summary["applied_trace_review_required"] = False
    _write_summary_files(Path(report["output_dir"]), summary)

    with pytest.raises(AppliedDffCacheVerificationError, match="partial output missing APPLIED_TRACE_PARTIAL"):
        verify_applied_dff_cache(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=report["output_dir"])


def test_nonfinite_applied_dff_without_flag_fails(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    report = write_applied_dff_cache(phasic_out, roi="CH1", requested_correction_strategy="dynamic_fit", overwrite=True)
    output_dir = Path(report["output_dir"])
    with h5py.File(report["applied_trace_cache_path"], "a") as h5:
        h5["roi/CH1/chunk_0/applied_dff"][0] = np.nan
        h5["roi/CH1/chunk_0/dynamic_fit_dff"][0] = np.nan
        h5["roi/CH1/chunk_0/flags"][...] = ""
    summary = _load_summary(output_dir)
    summary["applied_trace_flags"] = ""
    summary["applied_trace_cache_sha256"] = _sha256(output_dir / "applied_trace_cache.h5")
    _write_summary_files(output_dir, summary)
    chunks = pd.read_csv(output_dir / "applied_correction_chunks.csv")
    chunks["flags"] = ""
    chunks.to_csv(output_dir / "applied_correction_chunks.csv", index=False)
    chunks_payload = json.loads((output_dir / "applied_correction_chunks.json").read_text(encoding="utf-8"))
    for row in chunks_payload["chunks"]:
        row["flags"] = ""
    (output_dir / "applied_correction_chunks.json").write_text(json.dumps(chunks_payload), encoding="utf-8")

    with pytest.raises(AppliedDffCacheVerificationError, match="non-finite applied_dff without"):
        verify_applied_dff_cache(phasic_out, roi="CH1", strategy="dynamic_fit", applied_output_dir=output_dir)


def test_dry_run_writes_no_output_and_modifies_nothing(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    before = _sha256(phasic_out / "phasic_trace_cache.h5")
    summary_path = tmp_path / "summary.json"

    report = verify_applied_dff_cache(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        applied_output_dir=tmp_path / "missing",
        write_summary=summary_path,
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert not summary_path.exists()
    assert _sha256(phasic_out / "phasic_trace_cache.h5") == before
