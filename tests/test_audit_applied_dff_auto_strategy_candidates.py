import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tools.audit_applied_dff_auto_strategy_candidates import (
    AppliedDffAutoStrategyCandidateAuditError,
    audit_applied_dff_auto_strategy_candidates,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_phasic_out(
    tmp_path: Path,
    *,
    roi: str = "CH1",
    n_chunks: int = 4,
    mode: str = "clean_dynamic",
) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([roi.encode("utf-8")]))
        meta.create_dataset("chunk_ids", data=np.arange(n_chunks, dtype=int))
        meta.create_dataset(
            "source_files",
            data=np.asarray([f"chunk{x}.csv".encode("utf-8") for x in range(n_chunks)]),
        )
        roi_group = h5.create_group(f"roi/{roi}")
        for chunk_id in range(n_chunks):
            grp = roi_group.create_group(f"chunk_{chunk_id}")
            t = np.arange(4, dtype=float)
            signal = np.asarray([0.8, 1.0, 1.2, 0.9], dtype=float) + 0.01 * chunk_id
            uv = signal * 0.8 + 0.1
            fit_ref = uv.copy()
            dff = np.asarray([0.0, 0.1, 0.2, -0.05], dtype=float)
            grp.create_dataset("time_sec", data=t)
            grp.create_dataset("sig_raw", data=signal)
            grp.create_dataset("uv_raw", data=uv)
            grp.create_dataset("fit_ref", data=fit_ref)
            grp.create_dataset("dff", data=dff)
            grp.create_dataset("signal_only_f0_candidate", data=np.ones_like(signal))
            grp.create_dataset("baseline_ref_candidate", data=np.ones_like(signal))
            _set_attrs(grp, mode=mode, chunk_id=chunk_id, n_chunks=n_chunks)
    return phasic_out


def _set_attrs(grp, *, mode: str, chunk_id: int, n_chunks: int) -> None:
    clean_dynamic = {
        "dynamic_fit_qc_available": True,
        "dynamic_fit_qc_severity": "ok",
        "dynamic_fit_qc_flags": "",
        "dynamic_fit_qc_soft_flags": "",
        "dynamic_fit_qc_hard_flags": "",
        "dynamic_fit_qc_dynamic_fit_has_hard_flags": False,
        "dynamic_fit_qc_dynamic_fit_has_soft_flags": False,
        "dynamic_fit_qc_dynamic_fit_needs_inspection": False,
        "dynamic_fit_qc_dynamic_fit_negative_or_mixed_coupling": False,
        "dynamic_fit_qc_dynamic_fit_reference_flat_or_uninformative": False,
        "dynamic_fit_qc_dynamic_fit_reference_low_range": False,
        "dynamic_fit_qc_dynamic_fit_response_scale_rich": False,
        "dynamic_fit_qc_signal_iso_corr": 0.9,
        "dynamic_fit_qc_signal_fitted_ref_corr": 0.9,
        "dynamic_fit_qc_iso_fitted_ref_corr": 1.0,
        "dynamic_fit_qc_fitted_ref_to_signal_range_ratio": 0.8,
        "dynamic_fit_qc_fitted_ref_to_iso_range_ratio": 1.0,
        "dynamic_fit_qc_fitted_ref_total_variance": 1.0,
        "dynamic_fit_qc_fitted_ref_response_scale_fraction": 0.2,
        "dynamic_fit_qc_fitted_ref_baseline_scale_fraction": 0.6,
        "dynamic_fit_qc_slope_fraction_negative": 0.0,
        "dynamic_fit_slope_warning_level": "none",
        "dynamic_fit_slope_slope_min": 0.8,
        "dynamic_fit_slope_slope_max": 0.8,
        "dynamic_fit_slope_slope_median": 0.8,
        "dynamic_fit_slope_slope_negative_fraction": 0.0,
        "dynamic_fit_slope_clamped_fraction": 0.0,
        "dynamic_fit_slope_fallback_used": False,
        "dynamic_fit_slope_constraint_applied": False,
        "dynamic_fit_slope_nonnegative_support_insufficient": False,
    }
    clean_signal = {
        "signal_only_f0_candidate_available": True,
        "signal_only_f0_candidate_viability": "viable",
        "signal_only_f0_candidate_confidence": "high",
        "signal_only_f0_flags": "SIGNAL_ONLY_F0_AVAILABLE;SIGNAL_ONLY_F0_VIABLE",
        "signal_only_f0_anchor_count": 8.0,
        "signal_only_f0_anchor_status": "sufficient_anchors",
        "signal_only_f0_low_support_fraction": 0.5,
        "signal_only_f0_direct_support_fraction": 0.5,
        "signal_only_f0_extrapolated_fraction": 0.1,
        "signal_only_f0_edge_extrapolation_fraction": 0.1,
        "signal_only_f0_max_anchor_gap_fraction_observed": 0.05,
        "signal_only_f0_max_anchor_gap_sec_observed": 30.0,
        "signal_only_f0_high_state_context_applied": False,
        "signal_only_f0_state_aware_used": True,
        "signal_only_f0_status": "ok",
        "signal_only_f0_warning": "",
    }
    attrs = {**clean_dynamic, **clean_signal}
    if mode == "reference_failure":
        attrs.update(
            {
                "dynamic_fit_qc_severity": "context",
                "dynamic_fit_qc_flags": "NEGATIVE_OR_MIXED_REFERENCE_COUPLING",
                "dynamic_fit_qc_soft_flags": "NEGATIVE_OR_MIXED_REFERENCE_COUPLING",
                "dynamic_fit_qc_dynamic_fit_has_soft_flags": True,
                "dynamic_fit_qc_dynamic_fit_negative_or_mixed_coupling": True,
                "dynamic_fit_qc_signal_iso_corr": -0.1,
                "dynamic_fit_qc_signal_fitted_ref_corr": 0.2,
                "dynamic_fit_qc_slope_fraction_negative": 1.0,
                "dynamic_fit_slope_warning_level": "critical",
                "dynamic_fit_slope_slope_median": -0.1,
                "dynamic_fit_slope_slope_negative_fraction": 1.0,
            }
        )
    elif mode == "ambiguous":
        is_bad = chunk_id % 2 == 0
        attrs.update(
            {
                "dynamic_fit_qc_dynamic_fit_has_soft_flags": is_bad,
                "dynamic_fit_qc_dynamic_fit_negative_or_mixed_coupling": is_bad,
                "dynamic_fit_qc_flags": "NEGATIVE_OR_MIXED_REFERENCE_COUPLING" if is_bad else "",
                "dynamic_fit_qc_soft_flags": "NEGATIVE_OR_MIXED_REFERENCE_COUPLING" if is_bad else "",
                "dynamic_fit_qc_signal_iso_corr": 0.25,
                "dynamic_fit_qc_signal_fitted_ref_corr": 0.3,
                "dynamic_fit_slope_warning_level": "critical" if is_bad else "none",
                "dynamic_fit_slope_slope_negative_fraction": 1.0 if is_bad else 0.0,
                "signal_only_f0_candidate_confidence": "low",
                "signal_only_f0_candidate_viability": "contextual",
                "signal_only_f0_flags": "SIGNAL_ONLY_F0_CONTEXTUAL;SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION",
            }
        )
    elif mode == "qc_failure_signal_weak":
        attrs.update(
            {
                "dynamic_fit_qc_dynamic_fit_has_soft_flags": True,
                "dynamic_fit_qc_dynamic_fit_negative_or_mixed_coupling": True,
                "dynamic_fit_qc_flags": "NEGATIVE_OR_MIXED_REFERENCE_COUPLING",
                "dynamic_fit_qc_soft_flags": "NEGATIVE_OR_MIXED_REFERENCE_COUPLING",
                "dynamic_fit_slope_warning_level": "critical",
                "dynamic_fit_qc_signal_iso_corr": -0.2,
                "dynamic_fit_qc_signal_fitted_ref_corr": 0.1,
                "signal_only_f0_candidate_confidence": "low",
                "signal_only_f0_candidate_viability": "hard_inspect",
                "signal_only_f0_flags": (
                    "SIGNAL_ONLY_F0_HARD_INSPECT;"
                    "SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP;"
                    "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_FEW_ANCHORS"
                ),
            }
        )
    for key, value in attrs.items():
        grp.attrs[key] = value


def _read_row(report):
    assert len(report["rows"]) == 1
    return report["rows"][0]


def test_clean_dynamic_fit_supporting_case_decides_dynamic_fit(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="clean_dynamic")
    source = phasic_out / "phasic_trace_cache.h5"
    before = _sha256(source)

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    row = _read_row(report)
    assert row["auto_strategy_decision"] == "dynamic_fit"
    assert row["auto_strategy_decision_status"] == "decided"
    assert row["auto_strategy_review_required"] is False
    assert _sha256(source) == before
    assert report["summary"]["hdf5_modified_source_phasic_cache"] is False


def test_reference_failure_with_viable_signal_only_does_not_decide_dynamic_fit(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="reference_failure")

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    row = _read_row(report)
    assert row["auto_strategy_decision"] == "signal_only_f0"
    assert row["auto_strategy_confidence"] == "high"
    assert "CORRECTION_REFERENCE_FAILURE_EVIDENCE" in row["auto_strategy_flags"]
    assert "SIGNAL_ONLY_F0_RESCUE_EVIDENCE_CLEAN" in row["auto_strategy_flags"]


def test_reference_failure_with_caution_heavy_signal_only_decides_needs_review(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="qc_failure_signal_weak")

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    row = _read_row(report)
    assert row["auto_strategy_decision"] == "needs_review"
    assert row["auto_strategy_decision_status"] == "needs_review"
    assert "CORRECTION_REFERENCE_FAILURE_EVIDENCE" in row["auto_strategy_flags"]
    assert "SIGNAL_ONLY_F0_RESCUE_CANDIDATE" in row["auto_strategy_flags"]


def test_ambiguous_mixed_evidence_decides_needs_review(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="ambiguous")

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    row = _read_row(report)
    assert row["auto_strategy_decision"] == "needs_review"
    assert row["auto_strategy_decision_status"] == "needs_review"


def test_missing_required_evidence_records_blockers(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="clean_dynamic")
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        del h5["roi/CH1/chunk_0/fit_ref"]

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    row = _read_row(report)
    assert row["auto_strategy_decision"] == "needs_review"
    assert row["auto_strategy_decision_status"] == "blocked"
    assert "missing required datasets" in row["decision_blockers"]


def test_signal_only_computable_but_dynamic_fit_clean_stays_dynamic_fit(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="clean_dynamic")

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    row = _read_row(report)
    assert row["signal_only_f0_available_fraction"] == 1.0
    assert row["auto_strategy_decision"] == "dynamic_fit"


def test_dynamic_fit_source_available_but_qc_failure_not_dynamic_fit(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="qc_failure_signal_weak")

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    row = _read_row(report)
    assert row["n_chunks_with_required_dynamic_fit_data"] == row["n_chunks"]
    assert row["auto_strategy_decision"] != "dynamic_fit"


def test_read_only_immutability_with_legacy_features(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="clean_dynamic")
    legacy = phasic_out / "features" / "features.csv"
    legacy.parent.mkdir()
    legacy.write_text("roi,chunk_id\nCH1,0\n", encoding="utf-8")
    source_before = _sha256(phasic_out / "phasic_trace_cache.h5")
    legacy_before = _sha256(legacy)

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    assert _sha256(phasic_out / "phasic_trace_cache.h5") == source_before
    assert _sha256(legacy) == legacy_before
    assert report["summary"]["legacy_features_modified"] is False


@pytest.mark.parametrize(
    "relative_output",
    [
        ".",
        "auto_audit",
        "features",
        "features/auto_audit",
        "phasic_trace_cache.h5",
    ],
)
def test_unsafe_source_side_output_dirs_refuse_before_cleanup(tmp_path, relative_output):
    phasic_out = _make_phasic_out(tmp_path, mode="clean_dynamic")
    legacy = phasic_out / "features" / "features.csv"
    legacy.parent.mkdir(exist_ok=True)
    legacy.write_text("roi,chunk_id\nCH1,0\n", encoding="utf-8")
    output_dir = phasic_out if relative_output == "." else phasic_out / relative_output
    if relative_output in {"auto_audit", "features/auto_audit"}:
        output_dir.mkdir(parents=True)
        (output_dir / "sentinel.txt").write_text("do not delete", encoding="utf-8")
    source = phasic_out / "phasic_trace_cache.h5"
    source_before = _sha256(source)
    legacy_before = _sha256(legacy)

    with pytest.raises(
        AppliedDffAutoStrategyCandidateAuditError,
        match="separate from source phasic_out and legacy features",
    ):
        audit_applied_dff_auto_strategy_candidates(
            phasic_out,
            roi="CH1",
            output_dir=output_dir,
            overwrite=True,
        )

    assert source.exists()
    assert legacy.exists()
    assert _sha256(source) == source_before
    assert _sha256(legacy) == legacy_before
    if relative_output in {"auto_audit", "features/auto_audit"}:
        assert (output_dir / "sentinel.txt").read_text(encoding="utf-8") == "do not delete"
    else:
        assert not (phasic_out / "auto_strategy_candidate_audit.csv").exists()
        assert not (legacy.parent / "auto_strategy_candidate_audit.csv").exists()


def test_existing_file_path_output_dir_refuses_before_write(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="clean_dynamic")
    output_file = tmp_path / "not_a_directory.txt"
    output_file.write_text("do not replace", encoding="utf-8")

    with pytest.raises(AppliedDffAutoStrategyCandidateAuditError, match="selected path is a file"):
        audit_applied_dff_auto_strategy_candidates(
            phasic_out,
            roi="CH1",
            output_dir=output_file,
            overwrite=True,
        )

    assert output_file.read_text(encoding="utf-8") == "do not replace"


def test_valid_overwrite_only_clears_output_dir_not_phasic_out(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="clean_dynamic")
    legacy = phasic_out / "features" / "features.csv"
    legacy.parent.mkdir(exist_ok=True)
    legacy.write_text("roi,chunk_id\nCH1,0\n", encoding="utf-8")
    output_dir = tmp_path / "safe_audit"
    output_dir.mkdir()
    sentinel = output_dir / "old.txt"
    sentinel.write_text("old", encoding="utf-8")
    source_before = _sha256(phasic_out / "phasic_trace_cache.h5")
    legacy_before = _sha256(legacy)

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=output_dir,
        overwrite=True,
    )

    assert not sentinel.exists()
    assert Path(report["audit_csv"]).exists()
    assert _sha256(phasic_out / "phasic_trace_cache.h5") == source_before
    assert _sha256(legacy) == legacy_before


def test_no_production_side_effects_or_batch_call(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path, mode="clean_dynamic")
    import tools.audit_applied_dff_auto_strategy_candidates as audit

    assert not hasattr(audit, "run_applied_dff_batch")

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    output_dir = Path(report["summary"]["output_dir"])
    assert not (output_dir / "explicit_applied_dff_manifest.csv").exists()
    assert not (output_dir / "applied_trace_cache.h5").exists()
    assert not (output_dir / "features").exists()
    assert report["provenance"]["no_pipeline_execution"] is True
    assert report["provenance"]["no_feature_routing"] is True
    assert report["provenance"]["no_manifest_written"] is True


def test_repeated_runs_are_deterministic_except_paths_and_timestamps(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="reference_failure")

    first = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit1",
        overwrite=True,
    )
    second = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit2",
        overwrite=True,
    )

    first_row = dict(_read_row(first))
    second_row = dict(_read_row(second))
    for row in (first_row, second_row):
        row.pop("source_phasic_cache_path", None)
    assert first_row == second_row


def test_cli_smoke_writes_expected_outputs(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="clean_dynamic")
    output_dir = tmp_path / "audit"

    result = subprocess.run(
        [
            sys.executable,
            "tools/audit_applied_dff_auto_strategy_candidates.py",
            "--phasic-out",
            str(phasic_out),
            "--roi",
            "CH1",
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "audit_passed: True" in result.stdout
    assert (output_dir / "auto_strategy_candidate_audit.csv").exists()
    assert (output_dir / "auto_strategy_candidate_audit.json").exists()
    assert (output_dir / "auto_strategy_candidate_audit_summary.json").exists()
    assert (output_dir / "auto_strategy_candidate_audit_provenance.json").exists()
    summary = json.loads((output_dir / "auto_strategy_candidate_audit_summary.json").read_text())
    assert summary["decision_counts"]["dynamic_fit"] == 1


def test_csv_contains_required_columns(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, mode="clean_dynamic")

    report = audit_applied_dff_auto_strategy_candidates(
        phasic_out,
        roi="CH1",
        output_dir=tmp_path / "audit",
        overwrite=True,
    )

    df = pd.read_csv(report["audit_csv"])
    required = {
        "roi",
        "auto_strategy_decision",
        "auto_strategy_confidence",
        "dynamic_fit_negative_or_mixed_coupling_fraction",
        "signal_only_f0_available_fraction",
        "no_pipeline_execution",
        "no_feature_routing",
    }
    assert required.issubset(df.columns)
