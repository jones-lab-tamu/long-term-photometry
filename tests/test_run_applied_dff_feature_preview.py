import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tools.run_applied_dff_feature_preview import (
    AppliedFeaturePreviewError,
    run_applied_dff_feature_preview,
)


def _write_applied_preview(
    base: Path,
    *,
    strategy: str = "signal_only_f0",
    source: str = "signal_only_f0_dff",
    units: str = "dff",
    available: bool = True,
    complete: bool = True,
    review_required: bool = False,
    warning_level: str = "none",
    flags: str = "",
    include_trace: bool = True,
    trace_in_summary: bool = True,
) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    trace_path = base / f"CH1_{strategy}_applied_trace.csv"
    summary = {
        "roi": "CH1",
        "recording_key": "mock_recording.csv",
        "requested_correction_strategy": strategy,
        "correction_strategy_selection": "explicit",
        "applied_correction_strategy": strategy,
        "applied_trace_source": source,
        "applied_trace_units": units,
        "applied_trace_available": available,
        "applied_trace_complete": complete,
        "applied_trace_review_required": review_required,
        "applied_trace_warning_level": warning_level,
        "applied_trace_flags": flags,
        "feature_detection_input": False,
        "hdf5_modified": False,
        "output_dir": str(base),
        "trace_csv": str(trace_path) if trace_in_summary else "",
    }
    pd.DataFrame([summary]).to_csv(base / "applied_correction_summary.csv", index=False)
    (base / "applied_correction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if include_trace:
        fs = 20.0
        t = np.arange(0.0, 60.0, 1.0 / fs)
        y = 0.02 * np.sin(t)
        for center in (10.0, 30.0, 45.0):
            y += 1.0 * np.exp(-0.5 * ((t - center) / 0.18) ** 2)
        rows = pd.DataFrame(
            {
                "roi": "CH1",
                "chunk_id": 0,
                "sample_index": np.arange(t.size),
                "time_sec": t,
                "applied_dff": y,
            }
        )
        rows.to_csv(trace_path, index=False)
    return base


def _replace_trace_with_positive_baseline_multi_peak(applied_dir: Path) -> None:
    trace_path = applied_dir / "CH1_signal_only_f0_applied_trace.csv"
    fs = 20.0
    t = np.arange(0.0, 80.0, 1.0 / fs)
    rng = np.random.default_rng(7)
    y = 0.2 + 0.015 * np.sin(0.9 * t) + 0.01 * rng.standard_normal(t.size)
    for center in (15.0, 35.0, 55.0):
        y += 1.1 * np.exp(-0.5 * ((t - center) / 0.28) ** 2)
    pd.DataFrame(
        {
            "roi": "CH1",
            "chunk_id": 0,
            "sample_index": np.arange(t.size),
            "time_sec": t,
            "applied_dff": y,
        }
    ).to_csv(trace_path, index=False)


def test_reads_explicit_applied_trace_and_writes_feature_preview(tmp_path):
    applied_dir = _write_applied_preview(tmp_path / "applied")

    report = run_applied_dff_feature_preview(applied_dir, overwrite=True)

    assert Path(report["summary_csv"]).exists()
    assert Path(report["summary_json"]).exists()
    assert Path(report["events_csv"]).exists()
    summary = report["summary"]
    assert summary["feature_detection_input_trace"] == "applied_dff"
    assert summary["feature_detection_preview"] is True
    assert summary["hdf5_modified"] is False
    assert summary["replaces_existing_feature_outputs"] is False
    events = pd.read_csv(report["events_csv"])
    assert len(events) > 0
    assert set(
        [
            "event_start_sample",
            "event_peak_sample",
            "event_end_sample",
            "detection_input_trace",
            "detection_preview",
            "event_boundary_mode",
            "event_metrics_available",
        ]
    ).issubset(events.columns)
    assert set(events["detection_input_trace"]) == {"applied_dff"}
    assert set(events["event_boundary_mode"]) == {"peak_only_no_event_segmentation"}
    assert set(events["event_metrics_available"].astype(str).str.lower()) == {"false"}
    assert events["event_start_sample"].isna().all()
    assert events["event_end_sample"].isna().all()
    assert events["event_auc"].isna().all()
    assert events["event_duration_sec"].isna().all()


def test_positive_baseline_multi_peak_trace_does_not_emit_broad_duplicate_windows(tmp_path):
    applied_dir = _write_applied_preview(tmp_path / "applied")
    _replace_trace_with_positive_baseline_multi_peak(applied_dir)

    report = run_applied_dff_feature_preview(applied_dir, overwrite=True)
    events = pd.read_csv(report["events_csv"])

    assert len(events) >= 2
    assert events["event_peak_sample"].nunique() == len(events)
    assert events["event_start_sample"].isna().all()
    assert events["event_end_sample"].isna().all()
    assert events["event_auc"].isna().all()
    assert events["event_duration_sec"].isna().all()
    assert set(events["event_boundary_mode"]) == {"peak_only_no_event_segmentation"}


def test_provenance_is_carried_through(tmp_path):
    applied_dir = _write_applied_preview(
        tmp_path / "applied",
        strategy="dynamic_fit",
        source="dynamic_fit_dff",
        complete=True,
        review_required=True,
        warning_level="caution",
        flags="DYNAMIC_FIT_REVIEW",
    )

    report = run_applied_dff_feature_preview(applied_dir, overwrite=True)
    summary = report["summary"]

    assert summary["requested_correction_strategy"] == "dynamic_fit"
    assert summary["applied_correction_strategy"] == "dynamic_fit"
    assert summary["applied_trace_source"] == "dynamic_fit_dff"
    assert summary["applied_trace_complete"] is True
    assert summary["applied_trace_warning_level"] == "caution"
    assert summary["feature_detection_input_warning_level"] == "caution"
    assert summary["feature_detection_input_warning_flags"] == "DYNAMIC_FIT_REVIEW"


def test_refuses_no_correction(tmp_path):
    applied_dir = _write_applied_preview(
        tmp_path / "applied",
        strategy="no_correction",
        source="none",
        units="none",
        available=False,
        complete=False,
        include_trace=False,
    )

    with pytest.raises(AppliedFeaturePreviewError, match="no_correction has no corrected applied_dff"):
        run_applied_dff_feature_preview(applied_dir, overwrite=True)

    assert not (applied_dir / "feature_event_preview" / "applied_dff_feature_events.csv").exists()


def test_refuses_incomplete_applied_trace_by_default(tmp_path):
    applied_dir = _write_applied_preview(
        tmp_path / "applied",
        complete=False,
        review_required=True,
        warning_level="caution",
        flags="APPLIED_TRACE_PARTIAL",
    )

    with pytest.raises(AppliedFeaturePreviewError, match="applied_trace_complete is false"):
        run_applied_dff_feature_preview(applied_dir, overwrite=True)


def test_allows_partial_only_with_explicit_flag(tmp_path):
    applied_dir = _write_applied_preview(
        tmp_path / "applied",
        complete=False,
        review_required=True,
        warning_level="none",
        flags="APPLIED_TRACE_PARTIAL",
    )

    report = run_applied_dff_feature_preview(
        applied_dir,
        allow_partial_applied_trace=True,
        overwrite=True,
    )

    summary = report["summary"]
    assert "FEATURE_PREVIEW_PARTIAL_APPLIED_TRACE_INPUT" in summary["feature_preview_warning_flags"]
    assert summary["feature_preview_review_required"] is True
    assert summary["feature_preview_warning_level"] == "caution"


def test_does_not_modify_hdf5(tmp_path):
    applied_dir = _write_applied_preview(tmp_path / "applied")
    h5_path = applied_dir / "dummy.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("x", data=np.asarray([1, 2, 3]))
    before = h5_path.read_bytes()

    run_applied_dff_feature_preview(applied_dir, overwrite=True)

    assert h5_path.read_bytes() == before


def test_does_not_replace_existing_feature_outputs(tmp_path):
    applied_dir = _write_applied_preview(tmp_path / "applied")
    existing = applied_dir / "features" / "features.csv"
    existing.parent.mkdir(parents=True)
    existing.write_text("roi,chunk_id,peak_count\nCH1,0,999\n", encoding="utf-8")
    before = existing.read_text(encoding="utf-8")

    report = run_applied_dff_feature_preview(applied_dir, overwrite=True)

    assert existing.read_text(encoding="utf-8") == before
    assert Path(report["events_csv"]).parent.name == "feature_event_preview"
    assert Path(report["events_csv"]) != existing


def test_dry_run_writes_no_outputs(tmp_path):
    applied_dir = _write_applied_preview(tmp_path / "applied")

    report = run_applied_dff_feature_preview(applied_dir, dry_run=True)

    assert report["dry_run"] is True
    assert report["summary"]["feature_detection_input_trace"] == "applied_dff"
    assert not Path(report["summary_csv"]).exists()
    assert not Path(report["events_csv"]).exists()
    assert not Path(report["preview_plot"]).exists()


def test_missing_trace_csv_fails_clearly(tmp_path):
    applied_dir = _write_applied_preview(tmp_path / "applied", include_trace=False)

    with pytest.raises(AppliedFeaturePreviewError, match="applied trace CSV not found"):
        run_applied_dff_feature_preview(applied_dir, overwrite=True)


def test_ambiguous_multiple_trace_csvs_fail_without_explicit_trace(tmp_path):
    applied_dir = _write_applied_preview(
        tmp_path / "applied",
        trace_in_summary=False,
    )
    (applied_dir / "CH1_dynamic_fit_applied_trace.csv").write_text(
        (applied_dir / "CH1_signal_only_f0_applied_trace.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(AppliedFeaturePreviewError, match="multiple .*applied_trace"):
        run_applied_dff_feature_preview(applied_dir, overwrite=True)
