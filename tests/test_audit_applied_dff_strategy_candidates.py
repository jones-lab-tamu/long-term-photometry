import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tools.audit_applied_dff_strategy_candidates import (
    AppliedDffStrategyCandidateAuditError,
    audit_applied_dff_strategy_candidates,
)


def _make_phasic_out(tmp_path: Path, rois=("CH1",)) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([x.encode("utf-8") for x in rois]))
        meta.create_dataset("chunk_ids", data=np.asarray([0, 1], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"chunk0.csv", b"chunk1.csv"]))
        for roi in rois:
            roi_group = h5.create_group(f"roi/{roi}")
            for chunk_id in (0, 1):
                grp = roi_group.create_group(f"chunk_{chunk_id}")
                grp.create_dataset("time_sec", data=np.arange(4, dtype=float))
                grp.create_dataset("sig_raw", data=np.asarray([0.8, 1.0, 1.2, 0.9], dtype=float) + chunk_id)
                grp.create_dataset("dff", data=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float) + chunk_id)
    return phasic_out


def _make_phasic_out_with_chunks(tmp_path: Path, *, n_chunks: int, roi: str = "CH1") -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([roi.encode("utf-8")]))
        meta.create_dataset("chunk_ids", data=np.arange(n_chunks, dtype=int))
        meta.create_dataset("source_files", data=np.asarray([f"chunk{x}.csv".encode("utf-8") for x in range(n_chunks)]))
        roi_group = h5.create_group(f"roi/{roi}")
        for chunk_id in range(n_chunks):
            grp = roi_group.create_group(f"chunk_{chunk_id}")
            grp.create_dataset("time_sec", data=np.arange(4, dtype=float))
            grp.create_dataset("sig_raw", data=np.asarray([0.8, 1.0, 1.2, 0.9], dtype=float) + chunk_id)
            grp.create_dataset("dff", data=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float) + chunk_id)
    return phasic_out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _patch_core_f0(monkeypatch, *, flags=None, confidence="high", viability="viable"):
    import tools.audit_applied_dff_strategy_candidates as audit

    def _fake_candidate(signal, time=None, *, return_uncapped_candidate=False, **_kwargs):
        assert return_uncapped_candidate is True
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
        f0_arr = np.ones_like(signal_arr)
        return {
            "signal_only_f0_candidate": np.minimum(f0_arr, signal_arr),
            "signal_only_f0_candidate_uncapped": f0_arr,
            "signal_only_f0_candidate_viability": viability,
            "signal_only_f0_candidate_confidence": confidence,
            "signal_only_f0_flags": list(flags or []),
        }

    monkeypatch.setattr(audit.signal_f0_core, "compute_signal_only_f0_candidate", _fake_candidate)


def _patch_core_f0_by_chunk(monkeypatch, diagnostics_by_chunk):
    import tools.audit_applied_dff_strategy_candidates as audit

    def _fake_candidate(signal, time=None, *, return_uncapped_candidate=False, **_kwargs):
        assert return_uncapped_candidate is True
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
        chunk_id = int(round(float(signal_arr[0] - 0.8)))
        values = diagnostics_by_chunk.get(chunk_id, {})
        f0_arr = np.ones_like(signal_arr)
        return {
            "signal_only_f0_candidate": np.minimum(f0_arr, signal_arr),
            "signal_only_f0_candidate_uncapped": f0_arr,
            "signal_only_f0_candidate_viability": values.get("viability", "viable"),
            "signal_only_f0_candidate_confidence": values.get("confidence", "high"),
            "signal_only_f0_flags": list(values.get("flags", [])),
        }

    monkeypatch.setattr(audit.signal_f0_core, "compute_signal_only_f0_candidate", _fake_candidate)


def _row(report, strategy):
    rows = [row for row in report["rows"] if row["strategy_candidate"] == strategy]
    assert len(rows) == 1
    return rows[0]


def test_dynamic_fit_candidate_available(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    before = _sha256(phasic_out / "phasic_trace_cache.h5")
    _patch_core_f0(monkeypatch)

    report = audit_applied_dff_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    row = _row(report, "dynamic_fit")
    assert row["strategy_candidate_status"] in {"available", "available_with_cautions"}
    assert _sha256(phasic_out / "phasic_trace_cache.h5") == before


def test_dynamic_fit_blocked_when_dff_missing(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        del h5["roi/CH1/chunk_1/dff"]
    _patch_core_f0(monkeypatch)

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    row = _row(report, "dynamic_fit")
    assert row["strategy_candidate_status"] == "blocked"
    assert "missing dff" in row["blocking_issues"]


def test_signal_only_f0_viable(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch)

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    row = _row(report, "signal_only_f0")
    assert row["strategy_candidate_status"] in {"viable", "viable_with_cautions"}
    assert not (tmp_path / "audit" / "applied_trace_cache.h5").exists()


def test_signal_only_f0_blocked_when_raw_signal_missing(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        del h5["roi/CH1/chunk_1/sig_raw"]
    _patch_core_f0(monkeypatch)

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    row = _row(report, "signal_only_f0")
    assert row["strategy_candidate_status"] == "blocked"
    assert "missing signal/raw input" in row["blocking_issues"]


def test_multiple_roi_audit(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path, rois=("CH1", "CH2"))
    _patch_core_f0(monkeypatch)

    report = audit_applied_dff_strategy_candidates(phasic_out, output_dir=tmp_path / "audit", overwrite=True)

    assert len(report["rows"]) == 4
    assert {row["roi"] for row in report["rows"]} == {"CH1", "CH2"}


def test_dry_run_writes_nothing(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = audit_applied_dff_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert report["would_write_outputs"] is True
    assert not (tmp_path / "audit").exists()


def test_read_only_guarantees(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    legacy = phasic_out / "features" / "features.csv"
    legacy.parent.mkdir()
    legacy.write_text("roi,chunk_id\nCH1,0\n", encoding="utf-8")
    source_before = _sha256(phasic_out / "phasic_trace_cache.h5")
    legacy_before = _sha256(legacy)
    _patch_core_f0(monkeypatch)

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    assert _sha256(phasic_out / "phasic_trace_cache.h5") == source_before
    assert _sha256(legacy) == legacy_before
    assert report["summary"]["hdf5_modified_source_phasic_cache"] is False
    assert report["summary"]["legacy_features_modified"] is False


def test_no_recommendation_fields(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch)

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    forbidden = {"recommended_strategy", "chosen_strategy", "selected_strategy", "best_strategy"}
    rows = json.loads(Path(report["audit_json"]).read_text(encoding="utf-8"))["rows"]
    summary = json.loads(Path(report["summary_json"]).read_text(encoding="utf-8"))
    provenance = json.loads(Path(report["provenance_json"]).read_text(encoding="utf-8"))
    assert forbidden.isdisjoint(pd.read_csv(report["audit_csv"]).columns)
    assert forbidden.isdisjoint(summary)
    assert forbidden.isdisjoint(provenance)
    assert all(forbidden.isdisjoint(row) for row in rows)


def test_output_overwrite_behavior(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    sentinel = audit_dir / "old.txt"
    sentinel.write_text("old", encoding="utf-8")
    _patch_core_f0(monkeypatch)

    with pytest.raises(AppliedDffStrategyCandidateAuditError, match="refusing without --overwrite"):
        audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=audit_dir)

    audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=audit_dir, overwrite=True)
    assert not sentinel.exists()
    assert (audit_dir / "applied_dff_strategy_candidate_audit.csv").exists()
    assert (phasic_out / "phasic_trace_cache.h5").exists()


def test_output_dir_equal_to_phasic_out_refuses_without_deleting_source_cache(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    source = phasic_out / "phasic_trace_cache.h5"
    source_before = _sha256(source)
    _patch_core_f0(monkeypatch)

    with pytest.raises(AppliedDffStrategyCandidateAuditError, match="unsafe audit output_dir equals phasic_out"):
        audit_applied_dff_strategy_candidates(
            phasic_out,
            roi="CH1",
            output_dir=phasic_out,
            overwrite=True,
        )

    assert source.exists()
    assert _sha256(source) == source_before


def test_output_dir_equal_to_legacy_features_refuses_without_deleting_features(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    legacy = phasic_out / "features" / "features.csv"
    legacy.parent.mkdir()
    legacy.write_text("roi,chunk_id\nCH1,0\n", encoding="utf-8")
    legacy_before = _sha256(legacy)
    _patch_core_f0(monkeypatch)

    with pytest.raises(AppliedDffStrategyCandidateAuditError, match="legacy features directory"):
        audit_applied_dff_strategy_candidates(
            phasic_out,
            roi="CH1",
            output_dir=phasic_out / "features",
            overwrite=True,
        )

    assert legacy.exists()
    assert _sha256(legacy) == legacy_before


def test_output_dir_outside_phasic_out_still_works(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    output_dir = tmp_path / "outside_audit"
    _patch_core_f0(monkeypatch)

    report = audit_applied_dff_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=output_dir,
        overwrite=True,
    )

    assert Path(report["audit_csv"]).exists()
    assert Path(report["audit_csv"]).parent == output_dir


def test_candidate_warnings_and_cautions_are_counted(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(
        monkeypatch,
        flags=["SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION"],
        confidence="low",
        viability="contextual",
    )

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    row = _row(report, "signal_only_f0")
    assert row["review_required"] is True
    assert row["n_candidate_cautions"] > 0
    assert report["summary"]["n_candidates_with_cautions"] > 0


def test_compact_signal_only_f0_summaries_are_populated(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out_with_chunks(tmp_path, n_chunks=4)
    _patch_core_f0_by_chunk(
        monkeypatch,
        {
            0: {"viability": "viable", "confidence": "high", "flags": []},
            1: {"viability": "contextual", "confidence": "medium", "flags": ["SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP"]},
            2: {"viability": "hard_inspect", "confidence": "low", "flags": ["SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_FEW_ANCHORS"]},
            3: {"viability": "viable", "confidence": "low", "flags": ["SIGNAL_ONLY_F0_ABOVE_SIGNAL_EXCESSIVE"]},
        },
    )

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    row = _row(report, "signal_only_f0")
    assert row["viability_count_summary"] == "contextual=1; hard_inspect=1; viable=2"
    assert row["confidence_count_summary"] == "high=1; low=2; medium=1"
    assert "SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP=1" in row["top_flag_counts"]
    assert row["n_viable_chunks"] == 2
    assert row["n_contextual_chunks"] == 1
    assert row["n_hard_inspect_chunks"] == 1
    assert row["n_low_confidence_chunks"] == 2
    assert row["n_medium_confidence_chunks"] == 1
    assert row["n_high_confidence_chunks"] == 1
    assert row["n_chunks_with_large_anchor_gap"] == 1
    assert row["n_chunks_with_few_anchors"] == 1
    assert row["n_chunks_with_above_signal_excessive"] == 1
    assert row["example_problem_chunks"] == "1,2,3"
    csv_row = pd.read_csv(report["audit_csv"]).query("strategy_candidate == 'signal_only_f0'").iloc[0]
    assert csv_row["viability_count_summary"] == row["viability_count_summary"]
    json_rows = json.loads(Path(report["audit_json"]).read_text(encoding="utf-8"))["rows"]
    json_row = [item for item in json_rows if item["strategy_candidate"] == "signal_only_f0"][0]
    assert json_row["top_flag_counts"] == row["top_flag_counts"]


def test_top_flag_counts_are_limited_to_eight_entries(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out_with_chunks(tmp_path, n_chunks=1)
    _patch_core_f0_by_chunk(
        monkeypatch,
        {
            0: {
                "viability": "contextual",
                "confidence": "medium",
                "flags": [f"FLAG_{idx}" for idx in range(10)],
            }
        },
    )

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    entries = [x for x in _row(report, "signal_only_f0")["top_flag_counts"].split("; ") if x]
    assert len(entries) == 8


def test_few_anchor_chunk_count_is_unique_per_chunk(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out_with_chunks(tmp_path, n_chunks=1)
    _patch_core_f0_by_chunk(
        monkeypatch,
        {
            0: {
                "viability": "contextual",
                "confidence": "medium",
                "flags": [
                    "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_FEW_ANCHORS",
                    "SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS",
                ],
            }
        },
    )

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    row = _row(report, "signal_only_f0")
    assert row["flag_counts"]["SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_FEW_ANCHORS"] == 1
    assert row["flag_counts"]["SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS"] == 1
    assert row["n_chunks_with_few_anchors"] == 1


def test_example_problem_chunks_are_bounded(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out_with_chunks(tmp_path, n_chunks=25)
    _patch_core_f0_by_chunk(
        monkeypatch,
        {
            chunk_id: {
                "viability": "hard_inspect",
                "confidence": "low",
                "flags": ["SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP"],
            }
            for chunk_id in range(25)
        },
    )

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    chunks = [int(x) for x in _row(report, "signal_only_f0")["example_problem_chunks"].split(",") if x]
    assert len(chunks) == 20
    assert chunks == list(range(20))


def test_dynamic_fit_rows_have_blank_signal_only_f0_compact_summaries(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    _patch_core_f0(monkeypatch)

    report = audit_applied_dff_strategy_candidates(phasic_out, roi="CH1", output_dir=tmp_path / "audit", overwrite=True)

    row = _row(report, "dynamic_fit")
    assert row["viability_count_summary"] == ""
    assert row["confidence_count_summary"] == ""
    assert row["top_flag_counts"] == ""
    assert row["example_problem_chunks"] == ""
    assert row["n_viable_chunks"] == 0
    assert row["n_low_confidence_chunks"] == 0
    assert row["n_chunks_with_large_anchor_gap"] == 0
