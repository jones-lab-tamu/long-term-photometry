import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tools.write_applied_dff_cache import (
    AppliedDffCacheWriteError,
    write_applied_dff_cache,
)


def _make_phasic_out(
    tmp_path: Path,
    *,
    roi: str = "CH1",
    include_dff: bool = True,
    include_time: bool = True,
    include_signal: bool = True,
    nonfinite: bool = False,
    length_mismatch: bool = False,
) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([roi.encode("utf-8")]))
        meta.create_dataset("chunk_ids", data=np.asarray([0, 1], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"chunk0.csv", b"chunk1.csv"]))
        roi_group = h5.create_group(f"roi/{roi}")
        for chunk_id in (0, 1):
            grp = roi_group.create_group(f"chunk_{chunk_id}")
            if include_time:
                n_time = 4 if not (length_mismatch and chunk_id == 1) else 3
                grp.create_dataset("time_sec", data=np.arange(n_time, dtype=float))
            if include_signal:
                signal = np.asarray([0.8, 1.0, 1.2, 0.9], dtype=float) + chunk_id
                grp.create_dataset("sig_raw", data=signal)
            if include_dff:
                dff = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float) + chunk_id
                if nonfinite and chunk_id == 1:
                    dff[2] = np.nan
                grp.create_dataset("dff", data=dff)
    return phasic_out


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_summary_csv(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    return rows[0]


def _patch_core_f0(monkeypatch, *, f0=None, flags=None, confidence="high", viability="viable"):
    import tools.write_applied_dff_cache as writer

    def _fake_candidate(signal, time=None, *, return_uncapped_candidate=False, **_kwargs):
        assert return_uncapped_candidate is True
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
        if f0 is None:
            f0_arr = np.ones_like(signal_arr)
        elif callable(f0):
            f0_arr = np.asarray(f0(signal_arr), dtype=float).reshape(-1)
        else:
            f0_arr = np.asarray(f0, dtype=float).reshape(-1)
        capped = np.minimum(f0_arr, signal_arr) if f0_arr.shape == signal_arr.shape else f0_arr
        return {
            "signal_only_f0_candidate": capped,
            "signal_only_f0_candidate_uncapped": f0_arr,
            "signal_only_f0_candidate_viability": viability,
            "signal_only_f0_candidate_confidence": confidence,
            "signal_only_f0_flags": list(flags or []),
        }

    monkeypatch.setattr(writer.signal_f0_core, "compute_signal_only_f0_candidate", _fake_candidate)


def test_dynamic_fit_writes_applied_cache_from_synthetic_phasic_cache(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="dynamic_fit",
        overwrite=True,
    )

    output_dir = Path(report["output_dir"])
    cache_path = output_dir / "applied_trace_cache.h5"
    assert cache_path.exists()
    assert Path(report["summary_csv"]).exists()
    assert Path(report["summary_json"]).exists()
    assert Path(report["chunks_csv"]).exists()
    assert Path(report["chunks_json"]).exists()

    with h5py.File(phasic_out / "phasic_trace_cache.h5", "r") as src, h5py.File(cache_path, "r") as out:
        for chunk_id in (0, 1):
            src_grp = src[f"roi/CH1/chunk_{chunk_id}"]
            out_grp = out[f"roi/CH1/chunk_{chunk_id}"]
            np.testing.assert_array_equal(out_grp["time_sec"][()], src_grp["time_sec"][()])
            np.testing.assert_array_equal(out_grp["applied_dff"][()], src_grp["dff"][()])
            np.testing.assert_array_equal(out_grp["dynamic_fit_dff"][()], src_grp["dff"][()])
            assert "signal_raw_for_dff" not in out_grp
            assert "signal_only_f0_uncapped_for_dff" not in out_grp
            assert "signal_only_f0_dff" not in out_grp
            assert "denominator_trace" not in out_grp


def test_signal_only_f0_writes_applied_cache_from_synthetic_phasic_cache(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch, f0=lambda signal: np.ones_like(signal))

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="signal_only_f0",
        overwrite=True,
    )

    with h5py.File(phasic_out / "phasic_trace_cache.h5", "r") as src, h5py.File(report["applied_trace_cache_path"], "r") as out:
        for chunk_id in (0, 1):
            src_grp = src[f"roi/CH1/chunk_{chunk_id}"]
            out_grp = out[f"roi/CH1/chunk_{chunk_id}"]
            sig = src_grp["sig_raw"][()]
            f0 = out_grp["signal_only_f0_uncapped_for_dff"][()]
            expected = (sig - f0) / f0
            np.testing.assert_array_equal(out_grp["time_sec"][()], src_grp["time_sec"][()])
            np.testing.assert_array_equal(out_grp["signal_raw_for_dff"][()], sig)
            np.testing.assert_allclose(out_grp["applied_dff"][()], expected)
            np.testing.assert_allclose(out_grp["signal_only_f0_dff"][()], expected)
            assert "dynamic_fit_dff" not in out_grp
            assert "denominator_trace" not in out_grp


def test_signal_only_f0_summary_fields_are_correct(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    source = phasic_out / "phasic_trace_cache.h5"
    before = _sha256(source)
    _patch_core_f0(monkeypatch)

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="signal_only_f0",
        overwrite=True,
    )

    assert _sha256(source) == before
    summary = report["summary"]
    assert summary["requested_correction_strategy"] == "signal_only_f0"
    assert summary["applied_correction_strategy"] == "signal_only_f0"
    assert summary["applied_trace_source"] == "signal_only_f0_dff"
    assert summary["applied_trace_units"] == "dff"
    assert summary["applied_trace_available"] is True
    assert summary["applied_trace_complete"] is True
    assert summary["feature_detection_input"] is False
    assert summary["hdf5_modified_source_phasic_cache"] is False
    assert summary["applied_trace_cache_sha256"]
    assert summary["applied_trace_cache_sha256_location"] == "external_summary_after_cache_finalization"
    assert summary["f0_source_for_signal_only_f0"] == "core_uncapped_signal_only_f0_candidate"
    assert summary["signal_only_f0_denominator_source"] == "signal_only_f0_candidate_uncapped"
    assert summary["signal_only_f0_negative_dff_preserved"] is True


def test_signal_only_f0_preserves_negative_dff(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch, f0=lambda signal: np.ones_like(signal))

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="signal_only_f0",
        overwrite=True,
    )

    with h5py.File(report["applied_trace_cache_path"], "r") as h5:
        values = h5["roi/CH1/chunk_0/applied_dff"][()]
    assert float(np.nanmin(values)) < 0.0


def test_signal_only_f0_invalid_denominator_fails_clearly(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch, f0=lambda signal: np.zeros_like(signal))
    output_dir = tmp_path / "out"

    with pytest.raises(AppliedDffCacheWriteError, match="non-positive"):
        write_applied_dff_cache(
            phasic_out,
            roi="CH1",
            requested_correction_strategy="signal_only_f0",
            output_dir=output_dir,
            overwrite=True,
        )
    assert not (output_dir / "applied_trace_cache.h5").exists()


def test_signal_only_f0_missing_sig_raw_records_incomplete_output(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        del h5["roi/CH1/chunk_1/sig_raw"]
    _patch_core_f0(monkeypatch)

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="signal_only_f0",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["applied_trace_available"] is True
    assert summary["applied_trace_complete"] is False
    assert summary["n_chunks_available"] == 1
    assert summary["n_chunks_unavailable"] == 1
    assert "APPLIED_TRACE_PARTIAL" in summary["applied_trace_flags"]


def test_signal_only_f0_denominator_length_mismatch_fails(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch, f0=np.asarray([1.0, 1.0]))
    output_dir = tmp_path / "out"

    with pytest.raises(AppliedDffCacheWriteError, match="denominator has"):
        write_applied_dff_cache(
            phasic_out,
            roi="CH1",
            requested_correction_strategy="signal_only_f0",
            output_dir=output_dir,
            overwrite=True,
        )
    assert not (output_dir / "applied_trace_cache.h5").exists()


def test_signal_only_f0_warning_flags_propagate(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(
        monkeypatch,
        flags=["SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION"],
        confidence="low",
        viability="contextual",
    )

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="signal_only_f0",
        overwrite=True,
    )

    summary = report["summary"]
    assert "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION" in summary["applied_trace_flags"]
    assert summary["applied_trace_review_required"] is True
    assert summary["applied_trace_warning_level"] == "caution"
    chunks = pd.read_csv(report["chunks_csv"])
    assert set(chunks["warning_level"]) == {"caution"}
    assert chunks["review_required"].astype(str).str.lower().eq("true").all()


def test_signal_only_f0_unusable_viability_marks_chunk_unavailable_and_partial(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)

    import tools.write_applied_dff_cache as writer

    def _fake_candidate(signal, time=None, *, return_uncapped_candidate=False, **_kwargs):
        assert return_uncapped_candidate is True
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
        return {
            "signal_only_f0_candidate": np.ones_like(signal_arr),
            "signal_only_f0_candidate_uncapped": np.ones_like(signal_arr),
            "signal_only_f0_candidate_viability": "unusable" if signal_arr[0] > 1.5 else "viable",
            "signal_only_f0_candidate_confidence": "high",
            "signal_only_f0_flags": [],
        }

    monkeypatch.setattr(writer.signal_f0_core, "compute_signal_only_f0_candidate", _fake_candidate)

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="signal_only_f0",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["applied_trace_available"] is True
    assert summary["applied_trace_complete"] is False
    assert summary["n_chunks_available"] == 1
    assert summary["n_chunks_unavailable"] == 1
    assert summary["applied_trace_review_required"] is True
    assert summary["applied_trace_warning_level"] == "severe"
    assert "APPLIED_TRACE_PARTIAL" in summary["applied_trace_flags"]
    assert "SIGNAL_ONLY_F0_UNUSABLE" in summary["applied_trace_flags"]

    chunks = pd.read_csv(report["chunks_csv"])
    unavailable = chunks.loc[chunks["chunk_id"] == 1].iloc[0]
    assert str(unavailable["available"]).lower() == "false"
    assert unavailable["warning_level"] == "severe"
    assert str(unavailable["review_required"]).lower() == "true"
    assert "SIGNAL_ONLY_F0_UNUSABLE" in unavailable["flags"]

    with h5py.File(report["applied_trace_cache_path"], "r") as h5:
        assert "applied_dff" in h5["roi/CH1/chunk_0"]
        assert "applied_dff" not in h5["roi/CH1/chunk_1"]


def test_source_phasic_cache_is_not_modified(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    source = phasic_out / "phasic_trace_cache.h5"
    before = _sha256(source)

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="dynamic_fit",
        overwrite=True,
    )

    after = _sha256(source)
    assert after == before
    summary = report["summary"]
    assert summary["source_phasic_cache_sha256"] == before
    assert summary["hdf5_modified_source_phasic_cache"] is False


def test_summary_fields_are_correct_for_complete_dynamic_fit(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="dynamic_fit",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["requested_correction_strategy"] == "dynamic_fit"
    assert summary["correction_strategy_selection"] == "explicit"
    assert summary["applied_correction_strategy"] == "dynamic_fit"
    assert summary["applied_trace_source"] == "dynamic_fit_dff"
    assert summary["applied_trace_units"] == "dff"
    assert summary["applied_trace_available"] is True
    assert summary["applied_trace_complete"] is True
    assert summary["n_chunks_available"] == summary["n_chunks"]
    assert summary["n_chunks_unavailable"] == 0
    assert summary["applied_trace_warning_level"] == "none"
    assert summary["applied_trace_review_required"] is False
    assert summary["feature_detection_input"] is False
    assert summary["applied_trace_cache_sha256"]
    assert summary["applied_trace_cache_sha256_location"] == "external_summary_after_cache_finalization"

    summary_csv = _read_summary_csv(report["summary_csv"])
    assert summary_csv["applied_trace_complete"] == "true"
    assert summary_csv["feature_detection_input"] == "false"
    assert summary_csv["applied_trace_cache_sha256"]
    assert summary_csv["applied_trace_cache_sha256_location"] == "external_summary_after_cache_finalization"
    summary_json = json.loads(Path(report["summary_json"]).read_text(encoding="utf-8"))
    assert summary_json["applied_trace_cache_sha256"]
    assert summary_json["applied_trace_cache_sha256_location"] == "external_summary_after_cache_finalization"


def test_hdf5_metadata_fields_exist(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="dynamic_fit",
        overwrite=True,
    )

    with h5py.File(report["applied_trace_cache_path"], "r") as h5:
        for path in (
            "meta/schema_version",
            "meta/mode",
            "meta/source_phasic_cache_path",
            "meta/source_phasic_cache_sha256",
            "recording/CH1/summary",
            "recording/CH1/provenance_json",
        ):
            assert path in h5


def test_missing_roi_fails_clearly_and_writes_no_applied_cache(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    output_dir = tmp_path / "out"

    with pytest.raises(AppliedDffCacheWriteError, match="not found"):
        write_applied_dff_cache(
            phasic_out,
            roi="MISSING",
            requested_correction_strategy="dynamic_fit",
            output_dir=output_dir,
            overwrite=True,
        )

    assert not (output_dir / "applied_trace_cache.h5").exists()


@pytest.mark.parametrize(
    ("strategy", "message"),
    [
        ("no_correction", "no_correction production applied cache writing is not implemented yet"),
        ("auto", "auto strategy selection is not implemented"),
    ],
)
def test_unsupported_strategies_fail_and_write_no_applied_cache(tmp_path, strategy, message):
    phasic_out = _make_phasic_out(tmp_path)
    output_dir = tmp_path / f"out_{strategy}"

    with pytest.raises(AppliedDffCacheWriteError, match=message):
        write_applied_dff_cache(
            phasic_out,
            roi="CH1",
            requested_correction_strategy=strategy,
            output_dir=output_dir,
            overwrite=True,
        )

    assert not (output_dir / "applied_trace_cache.h5").exists()


def test_output_exists_without_overwrite_fails(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    output_dir = tmp_path / "out"
    write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="dynamic_fit",
        output_dir=output_dir,
        overwrite=True,
    )

    with pytest.raises(AppliedDffCacheWriteError, match="refusing to overwrite"):
        write_applied_dff_cache(
            phasic_out,
            roi="CH1",
            requested_correction_strategy="dynamic_fit",
            output_dir=output_dir,
            overwrite=False,
        )


def test_dry_run_writes_no_outputs(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    output_dir = tmp_path / "out"

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="dynamic_fit",
        output_dir=output_dir,
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert report["source_phasic_cache_exists"] is True
    assert report["n_chunks_planned"] == 2
    assert not output_dir.exists()


def test_missing_chunk_dff_records_incomplete_output(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        del h5["roi/CH1/chunk_1/dff"]

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="dynamic_fit",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["applied_trace_available"] is True
    assert summary["applied_trace_complete"] is False
    assert summary["n_chunks_available"] == 1
    assert summary["n_chunks_unavailable"] == 1
    assert "APPLIED_TRACE_PARTIAL" in summary["applied_trace_flags"]
    assert summary["applied_trace_review_required"] is True
    assert summary["applied_trace_warning_level"] in {"caution", "severe"}

    chunks = pd.read_csv(report["chunks_csv"])
    assert chunks.loc[chunks["chunk_id"] == 1, "available"].astype(str).str.lower().iloc[0] == "false"


def test_nonfinite_dff_values_are_flagged_and_trace_remains_complete(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, nonfinite=True)

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="dynamic_fit",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["applied_trace_complete"] is True
    assert "NONFINITE_APPLIED_DFF_VALUES" in summary["applied_trace_flags"]
    assert summary["applied_trace_review_required"] is True
    assert summary["applied_trace_warning_level"] == "caution"
    chunks = pd.read_csv(report["chunks_csv"])
    row = chunks[chunks["chunk_id"] == 1].iloc[0]
    assert "NONFINITE_APPLIED_DFF_VALUES" in row["flags"]
    assert str(row["review_required"]).lower() == "true"


def test_length_mismatch_fails_clearly_without_accepting_malformed_cache(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, length_mismatch=True)
    output_dir = tmp_path / "out"

    with pytest.raises(AppliedDffCacheWriteError, match="length mismatch"):
        write_applied_dff_cache(
            phasic_out,
            roi="CH1",
            requested_correction_strategy="dynamic_fit",
            output_dir=output_dir,
            overwrite=True,
        )

    assert not (output_dir / "applied_trace_cache.h5").exists()


def test_cli_strategy_alias_preserves_requested_correction_strategy(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = write_applied_dff_cache(
        phasic_out,
        roi="CH1",
        requested_correction_strategy="dynamic_fit",
        output_dir=tmp_path / "out",
        overwrite=True,
    )

    assert report["summary"]["requested_correction_strategy"] == "dynamic_fit"
