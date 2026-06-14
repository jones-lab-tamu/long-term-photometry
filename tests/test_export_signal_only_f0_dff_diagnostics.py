import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tools.export_signal_only_f0_dff_diagnostics import (
    FLAG_F0_FLOOR_APPLIED,
    FLAG_NONFINITE,
    compute_signal_only_f0_dff_diagnostic,
    export_signal_only_f0_dff_diagnostics,
)


def _make_phasic_out(tmp_path: Path) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    qc = phasic_out / "qc"
    qc.mkdir(parents=True)
    cache_path = phasic_out / "phasic_trace_cache.h5"
    with h5py.File(cache_path, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["mode"] = "phasic"
        meta.attrs["schema_version"] = "1.0"
        meta.create_dataset("rois", data=np.asarray([b"CH1", b"CH2"]))
        meta.create_dataset("chunk_ids", data=np.asarray([0, 1], dtype=int))
        meta.create_dataset("source_files", data=np.asarray([b"mock.csv"]))
        for roi in ("CH1", "CH2"):
            roi_group = h5.create_group(f"roi/{roi}")
            for chunk_id in (0, 1):
                grp = roi_group.create_group(f"chunk_{chunk_id}")
                t = np.linspace(0.0, 9.0, 10)
                sig = np.linspace(2.0 + chunk_id, 3.0 + chunk_id, 10)
                f0 = np.linspace(1.0 + chunk_id, 1.5 + chunk_id, 10)
                grp.create_dataset("time_sec", data=t)
                grp.create_dataset("sig_raw", data=sig)
                grp.create_dataset("signal_only_f0_candidate", data=f0)
    records = []
    for roi in ("CH1", "CH2"):
        for chunk_id in (0, 1):
            records.append(
                {
                    "roi": roi,
                    "chunk_id": chunk_id,
                    "signal_only_f0_candidate_viability": "viable",
                    "signal_only_f0_candidate_confidence": "high" if chunk_id == 0 else "low",
                    "signal_only_f0_anchor_count": 5,
                    "signal_only_f0_low_support_fraction": 0.5,
                    "signal_only_f0_extrapolated_fraction": 0.0,
                    "signal_only_f0_max_anchor_gap_fraction_observed": 0.1,
                    "signal_only_f0_flags": [],
                    "signal_state_candidate_class": "ordinary",
                    "signal_state_warning": "",
                }
            )
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (qc / "recording_correction_strategy_proposals.csv").write_text(
        "roi,kept\nCH1,true\n", encoding="utf-8"
    )
    (qc / "recording_correction_strategy_proposals.json").write_text(
        json.dumps([{"roi": "CH1"}], indent=2), encoding="utf-8"
    )
    (qc / "qc_summary.json").write_text(json.dumps({"kept": True}), encoding="utf-8")
    return phasic_out


def test_formula_correctness():
    signal = np.asarray([0.9, 1.0, 1.1])
    f0 = np.asarray([1.0, 1.0, 1.0])

    result = compute_signal_only_f0_dff_diagnostic(signal, f0)

    assert result["available"] is True
    np.testing.assert_allclose(result["dff"], np.asarray([-0.1, 0.0, 0.1]))
    assert result["dff_min"] < 0.0


def test_f0_floor_does_not_clip_negative_dff_for_valid_positive_f0():
    signal = np.asarray([0.5, 0.75, 1.0])
    f0 = np.asarray([1.0, 1.0, 1.0])

    result = compute_signal_only_f0_dff_diagnostic(signal, f0)

    assert result["f0_floor_applied"] is False
    np.testing.assert_allclose(result["dff"], np.asarray([-0.5, -0.25, 0.0]))
    assert result["dff_min"] < 0.0


def test_f0_floor_guardrail_adds_flag():
    signal = np.asarray([1.0, 2.0, 3.0])
    f0 = np.asarray([1.0, 0.0, -1.0])

    result = compute_signal_only_f0_dff_diagnostic(signal, f0)

    assert result["f0_floor_applied"] is True
    assert FLAG_F0_FLOOR_APPLIED in result["diagnostic_flags"]
    assert result["diagnostic_warning_level"] == "severe"


def test_nonfinite_handling_marks_unavailable_or_severe():
    signal = np.asarray([np.nan, np.inf])
    f0 = np.asarray([1.0, 1.0])

    result = compute_signal_only_f0_dff_diagnostic(signal, f0)

    assert result["available"] is False
    assert result["diagnostic_warning_level"] == "severe"
    assert FLAG_NONFINITE in result["diagnostic_flags"]


def test_export_writes_diagnostic_only_metadata_and_png(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = export_signal_only_f0_dff_diagnostics(phasic_out, roi="CH1", chunks=[0], overwrite=True)

    summary_json = Path(report["summary_json"])
    summary_csv = Path(report["summary_csv"])
    assert summary_json.exists()
    assert summary_csv.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["diagnostic_only"] is True
    assert payload["applied_correction"] is False
    assert payload["feature_detection_input"] is False
    assert payload["modifies_hdf5"] is False
    assert len(payload["records"]) == 1
    assert Path(report["plots_written"][0]).exists()


def test_export_summary_and_plot_preserve_negative_dff(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        grp = h5["roi/CH1/chunk_0"]
        del grp["sig_raw"]
        del grp["signal_only_f0_candidate"]
        grp.create_dataset("sig_raw", data=np.asarray([0.9, 1.0, 1.1], dtype=float))
        grp.create_dataset("signal_only_f0_candidate", data=np.asarray([1.0, 1.0, 1.0], dtype=float))
        del grp["time_sec"]
        grp.create_dataset("time_sec", data=np.asarray([0.0, 1.0, 2.0], dtype=float))

    import tools.export_signal_only_f0_dff_diagnostics as exporter

    def _fake_core_candidate(signal, time=None, *, return_uncapped_candidate=False, **_kwargs):
        assert return_uncapped_candidate is True
        return {
            "signal_only_f0_candidate": np.asarray([0.9, 1.0, 1.0], dtype=float),
            "signal_only_f0_candidate_uncapped": np.asarray([1.0, 1.0, 1.0], dtype=float),
            "signal_only_f0_candidate_viability": "viable",
            "signal_only_f0_candidate_confidence": "high",
            "signal_only_f0_flags": [],
        }

    monkeypatch.setattr(exporter, "compute_signal_only_f0_candidate", _fake_core_candidate)

    report = export_signal_only_f0_dff_diagnostics(phasic_out, roi="CH1", chunks=[0], overwrite=True)
    df = pd.read_csv(report["summary_csv"])

    assert float(df.loc[0, "dff_min"]) < 0.0
    assert float(df.loc[0, "dff_p01"]) < 0.0
    assert Path(report["plots_written"][0]).exists()


def test_hdf5_capped_candidate_is_not_used_for_dff_denominator(tmp_path, monkeypatch):
    phasic_out = _make_phasic_out(tmp_path)
    with h5py.File(phasic_out / "phasic_trace_cache.h5", "a") as h5:
        grp = h5["roi/CH1/chunk_0"]
        signal = np.asarray([0.9, 1.0, 1.1], dtype=float)
        true_uncapped_f0 = np.asarray([1.0, 1.0, 1.0], dtype=float)
        capped_hdf5_candidate = np.minimum(signal, true_uncapped_f0)
        del grp["sig_raw"]
        del grp["signal_only_f0_candidate"]
        del grp["time_sec"]
        grp.create_dataset("sig_raw", data=signal)
        grp.create_dataset("signal_only_f0_candidate", data=capped_hdf5_candidate)
        grp.create_dataset("time_sec", data=np.asarray([0.0, 1.0, 2.0], dtype=float))

    import tools.export_signal_only_f0_dff_diagnostics as exporter

    def _fake_core_candidate(signal, time=None, *, return_uncapped_candidate=False, **_kwargs):
        assert return_uncapped_candidate is True
        return {
            "signal_only_f0_candidate": np.asarray([0.9, 1.0, 1.0], dtype=float),
            "signal_only_f0_candidate_uncapped": np.asarray([1.0, 1.0, 1.0], dtype=float),
            "signal_only_f0_candidate_viability": "viable",
            "signal_only_f0_candidate_confidence": "high",
            "signal_only_f0_flags": [],
        }

    monkeypatch.setattr(exporter, "compute_signal_only_f0_candidate", _fake_core_candidate)

    report = export_signal_only_f0_dff_diagnostics(phasic_out, roi="CH1", chunks=[0], overwrite=True)
    df = pd.read_csv(report["summary_csv"])
    row = df.iloc[0]

    assert float(row["dff_min"]) < 0.0
    assert row["f0_source_for_dff"] == "uncapped_core_state_aware_recompute"
    assert str(row["hdf5_signal_only_f0_candidate_available"]).lower() == "true"
    assert str(row["hdf5_candidate_used_for_dff"]).lower() == "false"


def test_read_only_behavior_preserves_existing_inputs(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    cache = phasic_out / "phasic_trace_cache.h5"
    inputs = [
        cache,
        qc / "baseline_reference_candidate_by_chunk.json",
        qc / "recording_correction_strategy_proposals.csv",
        qc / "recording_correction_strategy_proposals.json",
        qc / "qc_summary.json",
    ]
    before = {path: path.read_bytes() for path in inputs}

    export_signal_only_f0_dff_diagnostics(phasic_out, roi="CH1", chunks=[0], overwrite=True)

    for path in inputs:
        assert path.read_bytes() == before[path]


def test_roi_and_chunk_filters(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = export_signal_only_f0_dff_diagnostics(phasic_out, roi="CH2", chunks=[1], overwrite=True)
    df = pd.read_csv(report["summary_csv"])

    assert list(df["roi"]) == ["CH2"]
    assert list(df["chunk_id"]) == [1]


def test_dry_run_writes_no_output_files(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)

    report = export_signal_only_f0_dff_diagnostics(phasic_out, roi="CH1", chunks=[0], dry_run=True)

    assert report["dry_run"] is True
    assert not Path(report["summary_json"]).exists()
    assert not Path(report["summary_csv"]).exists()
    assert report["plots_written"]
    assert not Path(report["plots_written"][0]).exists()


def test_existing_output_without_overwrite_uses_safe_timestamped_directory(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    first = export_signal_only_f0_dff_diagnostics(phasic_out, roi="CH1", chunks=[0], overwrite=True)
    second = export_signal_only_f0_dff_diagnostics(phasic_out, roi="CH1", chunks=[0], overwrite=False)

    assert first["output_dir"] != second["output_dir"]
    assert Path(second["summary_json"]).exists()


def test_overwrite_reuses_output_directory(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    first = export_signal_only_f0_dff_diagnostics(phasic_out, roi="CH1", chunks=[0], overwrite=True)
    second = export_signal_only_f0_dff_diagnostics(phasic_out, roi="CH1", chunks=[0], overwrite=True)

    assert first["output_dir"] == second["output_dir"]
    assert Path(second["summary_json"]).exists()
