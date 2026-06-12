import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from tools.plot_signal_only_f0_candidates import plot_signal_only_f0_candidates


def _make_phasic_out(tmp_path: Path) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    qc = phasic_out / "qc"
    qc.mkdir(parents=True)
    t = np.linspace(0.0, 99.0, 100)
    signal = 1.0 + 0.2 * np.sin(2.0 * np.pi * 0.02 * t)
    signal[40:50] += 0.6
    cache_path = phasic_out / "phasic_trace_cache.h5"
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
        grp.create_dataset("sig_raw", data=signal)
    record = {
        "roi": "Region0",
        "chunk_id": 0,
        "source_file": "source.csv",
        "signal_state_flags": ["SIGNAL_PARTIAL_HIGH_STATE_CANDIDATE"],
        "signal_state_candidate_class": "candidate_mixed_dynamic_high_state",
        "proposed_correction_mode_balanced": "dynamic_isosbestic",
        "proposal_flags_balanced": ["POLICY_CONTEXT"],
    }
    pd.DataFrame([record]).to_csv(qc / "baseline_reference_candidate_by_chunk.csv", index=False)
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps([record], indent=2), encoding="utf-8"
    )
    return phasic_out


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_plot_signal_only_f0_candidates_is_read_only_except_png(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    cache_path = phasic_out / "phasic_trace_cache.h5"
    json_path = qc / "baseline_reference_candidate_by_chunk.json"
    csv_path = qc / "baseline_reference_candidate_by_chunk.csv"
    before_json = _read_bytes(json_path)
    before_csv = _read_bytes(csv_path)
    before_h5 = _read_bytes(cache_path)

    report = plot_signal_only_f0_candidates(
        phasic_out,
        roi="Region0",
        chunks=[0],
        dpi=80,
    )

    assert report["using_default_signal_only_f0_config"] is True
    assert len(report["plots_written"]) == 1
    png = Path(report["plots_written"][0])
    assert png.exists()
    assert png.name == "Region0_chunk_0_signal_only_f0_candidate.png"
    assert png.parent == qc / "signal_only_f0_candidate_plots"
    assert png.stat().st_size > 0
    assert _read_bytes(json_path) == before_json
    assert _read_bytes(csv_path) == before_csv
    assert _read_bytes(cache_path) == before_h5
