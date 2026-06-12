import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tools.recompute_signal_only_f0_candidates import recompute_signal_only_f0_candidates


def _make_phasic_out(tmp_path: Path, *, include_signal: bool = True) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    qc = phasic_out / "qc"
    qc.mkdir(parents=True)
    cache_path = phasic_out / "phasic_trace_cache.h5"
    t = np.linspace(0.0, 99.0, 100)
    signal = 1.0 + 0.2 * np.sin(2.0 * np.pi * 0.02 * t)
    signal[40:50] += 0.6
    with h5py.File(cache_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.create_dataset("schema_version", data=np.array([1], dtype=np.int32))
        dt = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("rois", data=np.array(["Region0"], dtype=object), dtype=dt)
        meta.create_dataset("chunk_ids", data=np.array([0], dtype=np.int32))
        meta.create_dataset("source_files", data=np.array(["source.csv"], dtype=object), dtype=dt)
        meta.create_dataset("n_chunks", data=np.array([1], dtype=np.int32))
        grp = f.create_group("roi").create_group("Region0").create_group("chunk_0")
        grp.attrs["fs_hz"] = 1.0
        grp.attrs["source_file"] = "source.csv"
        grp.create_dataset("time_sec", data=t)
        if include_signal:
            grp.create_dataset("sig_raw", data=signal)
    return phasic_out


def _record() -> dict:
    return {
        "roi": "Region0",
        "chunk_id": 0,
        "source_file": "source.csv",
        "baseline_ref_candidate_available": True,
        "dynamic_fit_qc_flags": ["DYNAMIC_CONTEXT"],
        "reference_comparison_flags": ["REFERENCE_CONTEXT"],
        "signal_state_flags": ["SIGNAL_PARTIAL_HIGH_STATE_CANDIDATE"],
        "signal_state_candidate_class": "candidate_mixed_dynamic_high_state",
        "proposed_correction_mode_balanced": "dynamic_isosbestic",
        "proposal_flags_balanced": ["POLICY_CONTEXT"],
        "signal_only_f0_candidate_viability": "stale",
        "signal_only_f0_flags": ["STALE_F0_FLAG"],
    }


def test_recompute_writes_signal_only_f0_fields_and_preserves_policy(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    pd.DataFrame([_record()]).to_csv(qc / "baseline_reference_candidate_by_chunk.csv", index=False)
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps([_record()]), encoding="utf-8"
    )
    (qc / "qc_summary.json").write_text(
        json.dumps(
            {
                "correction_policy_proposal_summary": {"kept": True},
                "signal_state_diagnostics_summary": {"kept": True},
                "signal_only_f0_candidate_summary": {"stale": True},
            }
        ),
        encoding="utf-8",
    )

    report = recompute_signal_only_f0_candidates(phasic_out, backup=False)

    assert report["records_processed"] == 1
    df = pd.read_csv(qc / "baseline_reference_candidate_by_chunk.csv")
    assert "signal_only_f0_candidate_available" in df.columns
    assert "signal_only_f0_state_aware_used" in df.columns
    assert "signal_only_f0_anchor_status" in df.columns
    assert "signal_only_f0_anchor_count" in df.columns
    assert bool(df.loc[0, "signal_only_f0_candidate_available"])
    assert df.loc[0, "signal_only_f0_candidate_viability"] != "stale"
    assert df.loc[0, "proposed_correction_mode_balanced"] == "dynamic_isosbestic"
    assert df.loc[0, "proposal_flags_balanced"] == "POLICY_CONTEXT"
    assert "SIGNAL_ONLY_F0_PARTIAL_HIGH_STATE_PRESENT" in df.loc[0, "signal_only_f0_flags"]

    records = json.loads((qc / "baseline_reference_candidate_by_chunk.json").read_text(encoding="utf-8"))
    assert records[0]["signal_only_f0_candidate_available"] is True
    assert "signal_only_f0_state_aware_used" in records[0]
    assert "signal_only_f0_anchor_status" in records[0]
    assert "signal_only_f0_anchor_count" in records[0]
    assert isinstance(records[0]["signal_only_f0_flags"], list)
    assert records[0]["proposal_flags_balanced"] == ["POLICY_CONTEXT"]
    assert "signal_only_f0_candidate" not in records[0]

    summary = json.loads((qc / "qc_summary.json").read_text(encoding="utf-8"))
    assert summary["correction_policy_proposal_summary"] == {"kept": True}
    assert summary["signal_state_diagnostics_summary"] == {"kept": True}
    f0_summary = summary["signal_only_f0_candidate_summary"]
    assert f0_summary["roi_chunk_signal_only_f0_count"] == 1
    assert "signal_only_f0_state_aware_used_counts" in f0_summary
    assert "signal_only_f0_anchor_status_counts" in f0_summary
    assert "signal_only_f0_anchor_count" in f0_summary
    assert f0_summary["hdf5_trace_write_back"] == "deferred"


def test_recompute_uses_completed_run_config_when_available(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    (phasic_out / "config_used.yaml").write_text(
        "signal_only_f0_low_quantile: 0.25\n"
        "signal_only_f0_window_fraction: 0.15\n",
        encoding="utf-8",
    )
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps([_record()]), encoding="utf-8"
    )
    (qc / "qc_summary.json").write_text("{}\n", encoding="utf-8")

    report = recompute_signal_only_f0_candidates(phasic_out, backup=False)

    assert report["using_default_signal_only_f0_config"] is False
    records = json.loads((qc / "baseline_reference_candidate_by_chunk.json").read_text(encoding="utf-8"))
    assert records[0]["signal_only_f0_low_quantile"] == 0.25
    assert records[0]["signal_only_f0_window_fraction"] == 0.15


def test_dry_run_does_not_modify_or_backup(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    csv_path = qc / "baseline_reference_candidate_by_chunk.csv"
    csv_path.write_text("roi,chunk_id,source_file\nRegion0,0,source.csv\n", encoding="utf-8")
    (qc / "qc_summary.json").write_text("{}\n", encoding="utf-8")
    original = csv_path.read_text(encoding="utf-8")

    report = recompute_signal_only_f0_candidates(phasic_out, dry_run=True)

    assert report["dry_run"] is True
    assert report["csv_updated"] is False
    assert csv_path.read_text(encoding="utf-8") == original
    assert not list(qc.glob("*.bak_*"))


def test_backups_created_by_default(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    pd.DataFrame([_record()]).to_csv(qc / "baseline_reference_candidate_by_chunk.csv", index=False)
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps([_record()]), encoding="utf-8"
    )
    (qc / "qc_summary.json").write_text("{}\n", encoding="utf-8")

    report = recompute_signal_only_f0_candidates(phasic_out)

    assert len(report["backups_created"]) == 3
    assert list(qc.glob("baseline_reference_candidate_by_chunk.csv.bak_*"))
    assert list(qc.glob("baseline_reference_candidate_by_chunk.json.bak_*"))
    assert list(qc.glob("qc_summary.json.bak_*"))


def test_missing_raw_signal_or_time_fails_clearly(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, include_signal=False)
    qc = phasic_out / "qc"
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps([_record()]), encoding="utf-8"
    )
    (qc / "qc_summary.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="raw signal/time data are missing"):
        recompute_signal_only_f0_candidates(phasic_out, backup=False)
