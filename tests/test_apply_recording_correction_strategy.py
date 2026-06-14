import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tools.apply_recording_correction_strategy import apply_recording_correction_strategy


def _make_phasic_out(tmp_path: Path, *, include_dff: bool = True, include_signal: bool = True) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    qc = phasic_out / "qc"
    qc.mkdir(parents=True)
    cache_path = phasic_out / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([b"CH1"]))
        meta.create_dataset("chunk_ids", data=np.asarray([0, 1], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"mock_recording.csv"]))
        roi_group = h5.create_group("roi/CH1")
        for chunk_id in (0, 1):
            grp = roi_group.create_group(f"chunk_{chunk_id}")
            grp.create_dataset("time_sec", data=np.asarray([0.0, 1.0, 2.0], dtype=float))
            if include_signal:
                signal = np.asarray([0.9, 1.0, 1.1], dtype=float) + chunk_id
                grp.create_dataset("sig_raw", data=signal)
                grp.create_dataset("signal_only_f0_candidate", data=np.minimum(signal, 1.0 + chunk_id))
            if include_dff:
                grp.create_dataset("dff", data=np.asarray([-0.2, 0.0, 0.2], dtype=float) + chunk_id)
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


def test_explicit_signal_only_f0_creates_applied_dff_with_core_uncapped_and_negative_values(
    tmp_path, monkeypatch
):
    phasic_out = _make_phasic_out(tmp_path)

    import tools.export_signal_only_f0_dff_diagnostics as exporter

    def _fake_core_candidate(signal, time=None, *, return_uncapped_candidate=False, **_kwargs):
        assert return_uncapped_candidate is True
        signal = np.asarray(signal, dtype=float)
        return {
            "signal_only_f0_candidate": np.minimum(signal, np.ones_like(signal)),
            "signal_only_f0_candidate_uncapped": np.ones_like(signal),
            "signal_only_f0_candidate_viability": "viable",
            "signal_only_f0_candidate_confidence": "high",
            "signal_only_f0_flags": [],
        }

    monkeypatch.setattr(exporter, "compute_signal_only_f0_candidate", _fake_core_candidate)

    report = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="signal_only_f0",
        max_preview_chunks=1,
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["requested_correction_strategy"] == "signal_only_f0"
    assert summary["correction_strategy_selection"] == "explicit"
    assert summary["applied_correction_strategy"] == "signal_only_f0"
    assert summary["applied_trace_source"] == "signal_only_f0_dff"
    assert summary["applied_trace_available"] is True
    assert summary["applied_trace_complete"] is True
    assert "APPLIED_TRACE_PARTIAL" not in summary["applied_trace_flags"]
    assert summary["f0_source_for_signal_only_f0"] == "uncapped_core_state_aware_recompute"
    assert summary["feature_detection_input"] is False
    assert summary["hdf5_modified"] is False
    trace = pd.read_csv(report["trace_csv"])
    assert "applied_dff" in trace.columns
    assert "signal_only_f0_uncapped_for_dff" in trace.columns
    assert float(trace["applied_dff"].min()) < 0.0
    assert Path(report["preview_plots"][0]).exists()


def test_explicit_dynamic_fit_uses_existing_dff_and_does_not_fallback(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)

    import tools.export_signal_only_f0_dff_diagnostics as exporter

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("signal_only_f0 should not be computed for dynamic_fit")

    monkeypatch.setattr(exporter, "compute_signal_only_f0_candidate", _fail_if_called)

    report = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        max_preview_chunks=1,
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["applied_correction_strategy"] == "dynamic_fit"
    assert summary["applied_trace_source"] == "dynamic_fit_dff"
    assert summary["applied_trace_available"] is True
    assert summary["applied_trace_complete"] is True
    assert "APPLIED_TRACE_PARTIAL" not in summary["applied_trace_flags"]
    trace = pd.read_csv(report["trace_csv"])
    assert list(trace.columns) == [
        "roi",
        "chunk_id",
        "sample_index",
        "time_sec",
        "applied_dff",
        "dynamic_fit_dff",
    ]
    np.testing.assert_allclose(trace["applied_dff"].to_numpy(), trace["dynamic_fit_dff"].to_numpy())
    assert Path(report["preview_plots"][0]).exists()


def test_no_correction_does_not_create_applied_dff(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="no_correction",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["applied_correction_strategy"] == "no_correction"
    assert summary["applied_trace_available"] is False
    assert summary["applied_trace_complete"] is False
    assert summary["applied_trace_source"] == "none"
    assert summary["applied_trace_units"] == "none"
    assert report["trace_csv"] == ""
    assert not list(Path(report["output_dir"]).glob("*applied_trace.csv"))


def test_no_silent_fallback_for_unavailable_dynamic_fit(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path, include_dff=False)

    import tools.export_signal_only_f0_dff_diagnostics as exporter

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("signal_only_f0 should not be fallback for dynamic_fit")

    monkeypatch.setattr(exporter, "compute_signal_only_f0_candidate", _fail_if_called)

    report = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["applied_trace_available"] is False
    assert summary["applied_trace_source"] == "dynamic_fit_dff"
    assert "dynamic_fit_dff_unavailable" in summary["reason_if_unavailable"]
    assert report["trace_csv"] == ""


def test_partial_dynamic_fit_trace_is_available_but_not_complete(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        del h5["roi/CH1/chunk_1/dff"]

    report = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["n_chunks"] == 2
    assert summary["n_chunks_available"] == 1
    assert summary["n_chunks_unavailable"] == 1
    assert summary["applied_trace_available"] is True
    assert summary["applied_trace_complete"] is False
    assert "APPLIED_TRACE_PARTIAL" in summary["applied_trace_flags"]
    assert summary["applied_trace_review_required"] is True
    assert summary["applied_trace_warning_level"] in {"caution", "severe"}
    trace = pd.read_csv(report["trace_csv"])
    assert set(trace["chunk_id"].unique()) == {0}


def test_no_silent_fallback_for_unavailable_signal_only_f0(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, include_signal=False)

    report = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="signal_only_f0",
        overwrite=True,
    )

    summary = report["summary"]
    assert summary["applied_trace_available"] is False
    assert summary["applied_trace_source"] == "signal_only_f0_dff"
    assert "sig_raw" in summary["reason_if_unavailable"]
    assert not summary["reason_if_unavailable"].startswith("dynamic")
    assert report["trace_csv"] == ""


def test_one_strategy_per_roi_recording_summary_has_no_chunkwise_strategy_column(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        overwrite=True,
    )

    summary_df = pd.read_csv(report["summary_csv"])
    assert summary_df.shape[0] == 1
    assert list(summary_df["applied_correction_strategy"].unique()) == ["dynamic_fit"]
    assert str(summary_df.loc[0, "applied_trace_complete"]).lower() == "true"
    assert "chunk_strategy" not in summary_df.columns


def test_read_only_hdf5_bytes_unchanged_after_export(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    cache = phasic_out / "phasic_trace_cache.h5"
    before = cache.read_bytes()

    apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        overwrite=True,
    )

    assert cache.read_bytes() == before


def test_dry_run_reports_but_writes_no_trace_or_summary(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        max_preview_chunks=1,
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert report["summary"]["applied_trace_available"] is True
    assert report["trace_csv"]
    assert not Path(report["trace_csv"]).exists()
    assert not Path(report["summary_csv"]).exists()
    assert report["preview_plots"]
    assert not Path(report["preview_plots"][0]).exists()


def test_existing_output_without_overwrite_uses_timestamped_directory(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    first = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        overwrite=True,
    )
    second = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="dynamic_fit",
        overwrite=False,
    )

    assert first["output_dir"] != second["output_dir"]
    assert Path(second["summary_json"]).exists()


def test_preview_plot_creation_for_signal_only_f0(tmp_path, monkeypatch):
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

    report = apply_recording_correction_strategy(
        phasic_out,
        roi="CH1",
        strategy="signal_only_f0",
        chunks=[1],
        max_preview_chunks=1,
        overwrite=True,
    )

    assert len(report["preview_plots"]) == 1
    assert "chunk_1_signal_only_f0_applied_preview.png" in report["preview_plots"][0]
    assert Path(report["preview_plots"][0]).exists()
