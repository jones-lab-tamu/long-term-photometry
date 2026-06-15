import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tools.verify_applied_dff_preview_stack import (
    PreviewStackVerificationError,
    verify_applied_dff_preview_stack,
)


def _make_phasic_out(tmp_path: Path) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    qc = phasic_out / "qc"
    qc.mkdir(parents=True)
    t = np.arange(0.0, 60.0, 0.05)
    dff = 0.02 * np.sin(t)
    for center in (10.0, 30.0, 45.0):
        dff += 1.0 * np.exp(-0.5 * ((t - center) / 0.18) ** 2)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([b"CH1"]))
        meta.create_dataset("chunk_ids", data=np.asarray([0, 1], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"mock.csv"]))
        roi_group = h5.create_group("roi/CH1")
        for chunk_id in (0, 1):
            grp = roi_group.create_group(f"chunk_{chunk_id}")
            grp.create_dataset("time_sec", data=t)
            grp.create_dataset("sig_raw", data=1.0 + dff)
            grp.create_dataset("dff", data=dff)
    records = []
    for chunk_id in (0, 1):
        records.append(
            {
                "roi": "CH1",
                "chunk_id": chunk_id,
                "signal_only_f0_candidate_viability": "viable",
                "signal_only_f0_candidate_confidence": "high",
                "signal_only_f0_anchor_count": 5,
                "signal_only_f0_low_support_fraction": 0.5,
                "signal_only_f0_extrapolated_fraction": 0.0,
                "signal_only_f0_max_anchor_gap_fraction_observed": 0.1,
                "signal_only_f0_flags": [],
            }
        )
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return phasic_out


def test_signal_only_f0_stack_passes_with_synthetic_cache(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)

    import tools.export_signal_only_f0_dff_diagnostics as exporter

    def _fake_core_candidate(signal, time=None, *, return_uncapped_candidate=False, **_kwargs):
        assert return_uncapped_candidate is True
        return {
            "signal_only_f0_candidate": np.ones_like(np.asarray(signal, dtype=float)),
            "signal_only_f0_candidate_uncapped": np.ones_like(np.asarray(signal, dtype=float)),
            "signal_only_f0_candidate_viability": "viable",
            "signal_only_f0_candidate_confidence": "high",
            "signal_only_f0_flags": [],
        }

    monkeypatch.setattr(exporter, "compute_signal_only_f0_candidate", _fake_core_candidate)

    report = verify_applied_dff_preview_stack(
        phasic_out,
        roi="CH1",
        strategy="signal_only_f0",
        output_dir=tmp_path / "verify",
        overwrite=True,
    )

    assert report["verification_passed"] is True
    assert report["applied_correction_strategy"] == "signal_only_f0"
    assert report["applied_trace_source"] == "signal_only_f0_dff"
    assert report["feature_detection_input_trace"] == "applied_dff"
    assert report["feature_detection_input_source"] == "signal_only_f0_dff"
    assert report["peak_detector_mode"] == "peak_only_no_event_segmentation"
    assert report["n_event_rows_with_metrics_populated"] == 0
    assert report["hdf5_modified"] is False
    assert report["replaces_existing_feature_outputs"] is False


def test_dynamic_fit_stack_passes_with_synthetic_cache(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = verify_applied_dff_preview_stack(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        output_dir=tmp_path / "verify",
        overwrite=True,
    )

    assert report["verification_passed"] is True
    assert report["applied_correction_strategy"] == "dynamic_fit"
    assert report["applied_trace_source"] == "dynamic_fit_dff"
    assert report["feature_detection_input_source"] == "dynamic_fit_dff"
    assert report["feature_detection_input_strategy"] == "dynamic_fit"


def test_no_correction_refusal_is_expected_pass(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = verify_applied_dff_preview_stack(
        phasic_out,
        roi="CH1",
        strategy="no_correction",
        output_dir=tmp_path / "verify",
        overwrite=True,
    )

    assert report["verification_passed"] is True
    assert report["applied_correction_strategy"] == "no_correction"
    assert report["applied_trace_available"] is False
    assert report["feature_preview_ran"] is False
    assert report["feature_preview_refused_expectedly"] is True
    assert "no_correction has no corrected applied_dff" in report["feature_preview_refusal_message"]
    assert not (Path(report["peak_preview_dir"]) / "applied_dff_feature_events.csv").exists()


def _fake_applied_report(output_dir: Path, *, strategy: str, source: str, complete: bool = True) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "applied_correction_strategy": strategy,
        "correction_strategy_selection": "explicit",
        "applied_trace_source": source,
        "applied_trace_units": "dff",
        "applied_trace_available": True,
        "applied_trace_complete": complete,
        "applied_trace_warning_level": "none",
        "applied_trace_review_required": False,
        "n_chunks": 1,
        "n_chunks_available": 1 if complete else 0,
        "n_chunks_unavailable": 0 if complete else 1,
        "hdf5_modified": False,
        "feature_detection_input": False,
    }
    summary_csv = output_dir / "applied_correction_summary.csv"
    summary_json = output_dir / "applied_correction_summary.json"
    trace_csv = output_dir / f"CH1_{strategy}_applied_trace.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        {
            "roi": ["CH1"],
            "chunk_id": [0],
            "sample_index": [0],
            "time_sec": [0.0],
            "applied_dff": [0.0],
        }
    ).to_csv(trace_csv, index=False)
    return {
        "summary": summary,
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "trace_csv": str(trace_csv),
    }


def _fake_peak_report(output_dir: Path, *, strategy: str, source: str, hash_value: str = "abc", metric: str = "") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    events_csv = output_dir / "applied_dff_feature_events.csv"
    pd.DataFrame(
        [
            {
                "detection_input_trace": "applied_dff",
                "detection_input_strategy": strategy,
                "detection_input_source": source,
                "detection_preview": "true",
                "event_boundary_mode": "peak_only_no_event_segmentation",
                "event_start_sample": "",
                "event_end_sample": "",
                "event_auc": metric,
                "event_duration_sec": "",
                "event_metrics_available": "false",
                "peak_detection_config_hash": hash_value,
            }
        ]
    ).to_csv(events_csv, index=False)
    summary = {
        "feature_detection_input_trace": "applied_dff",
        "feature_detection_input_strategy": strategy,
        "feature_detection_input_source": source,
        "feature_detection_input_units": "dff",
        "feature_detection_preview": True,
        "hdf5_modified": False,
        "replaces_existing_feature_outputs": False,
        "peak_detector_source_function": "get_peak_indices_for_trace",
        "peak_detector_mode": "peak_only_no_event_segmentation",
        "peak_detection_config_source": "preview_default_Config_event_signal_dff",
        "peak_detection_config_path": "",
        "peak_detection_config_hash": hash_value,
        "n_events": 1,
    }
    return {"summary": summary, "events_csv": str(events_csv)}


def test_wrong_applied_strategy_fails(tmp_path, monkeypatch):
    import tools.verify_applied_dff_preview_stack as verifier

    def _fake_apply(*_args, output_dir, **_kwargs):
        return _fake_applied_report(Path(output_dir), strategy="dynamic_fit", source="dynamic_fit_dff")

    monkeypatch.setattr(verifier, "apply_recording_correction_strategy", _fake_apply)

    with pytest.raises(PreviewStackVerificationError, match="applied strategy does not match"):
        verify_applied_dff_preview_stack(
            tmp_path,
            roi="CH1",
            strategy="signal_only_f0",
            output_dir=tmp_path / "verify",
            overwrite=True,
        )


def test_incomplete_applied_trace_fails(tmp_path, monkeypatch):
    import tools.verify_applied_dff_preview_stack as verifier

    def _fake_apply(*_args, output_dir, **_kwargs):
        return _fake_applied_report(
            Path(output_dir),
            strategy="signal_only_f0",
            source="signal_only_f0_dff",
            complete=False,
        )

    monkeypatch.setattr(verifier, "apply_recording_correction_strategy", _fake_apply)

    with pytest.raises(PreviewStackVerificationError, match="applied trace incomplete"):
        verify_applied_dff_preview_stack(
            tmp_path,
            roi="CH1",
            strategy="signal_only_f0",
            output_dir=tmp_path / "verify",
            overwrite=True,
        )


def test_event_rows_with_metrics_populated_fail(tmp_path, monkeypatch):
    import tools.verify_applied_dff_preview_stack as verifier

    def _fake_apply(*_args, output_dir, **_kwargs):
        return _fake_applied_report(Path(output_dir), strategy="dynamic_fit", source="dynamic_fit_dff")

    def _fake_peak(*_args, output_dir, **_kwargs):
        return _fake_peak_report(Path(output_dir), strategy="dynamic_fit", source="dynamic_fit_dff", metric="123.0")

    monkeypatch.setattr(verifier, "apply_recording_correction_strategy", _fake_apply)
    monkeypatch.setattr(verifier, "run_applied_dff_feature_preview", _fake_peak)

    with pytest.raises(PreviewStackVerificationError, match="event rows have populated event metrics"):
        verify_applied_dff_preview_stack(
            tmp_path,
            roi="CH1",
            strategy="dynamic_fit",
            output_dir=tmp_path / "verify",
            overwrite=True,
        )


def test_event_hash_mismatch_fails(tmp_path, monkeypatch):
    import tools.verify_applied_dff_preview_stack as verifier

    def _fake_apply(*_args, output_dir, **_kwargs):
        return _fake_applied_report(Path(output_dir), strategy="dynamic_fit", source="dynamic_fit_dff")

    def _fake_peak(*_args, output_dir, **_kwargs):
        report = _fake_peak_report(Path(output_dir), strategy="dynamic_fit", source="dynamic_fit_dff", hash_value="summary_hash")
        events = pd.read_csv(report["events_csv"])
        events["peak_detection_config_hash"] = "event_hash"
        events.to_csv(report["events_csv"], index=False)
        return report

    monkeypatch.setattr(verifier, "apply_recording_correction_strategy", _fake_apply)
    monkeypatch.setattr(verifier, "run_applied_dff_feature_preview", _fake_peak)

    with pytest.raises(PreviewStackVerificationError, match="event rows have wrong detection provenance"):
        verify_applied_dff_preview_stack(
            tmp_path,
            roi="CH1",
            strategy="dynamic_fit",
            output_dir=tmp_path / "verify",
            overwrite=True,
        )


def test_dry_run_writes_no_outputs(tmp_path):
    report = verify_applied_dff_preview_stack(
        tmp_path / "missing_phasic",
        roi="CH1",
        strategy="dynamic_fit",
        output_dir=tmp_path / "verify",
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert report["would_run_applied_stage"] is True
    assert report["would_run_peak_preview"] is True
    assert not (tmp_path / "verify").exists()
