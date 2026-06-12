import json
from pathlib import Path

import pandas as pd

from tools.propose_recording_correction_strategy import (
    propose_recording_correction_strategy,
)


def _make_phasic_out(tmp_path: Path, records: list[dict]) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    qc = phasic_out / "qc"
    qc.mkdir(parents=True)
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (qc / "qc_summary.json").write_text(
        json.dumps({"existing": {"kept": True}}, indent=2), encoding="utf-8"
    )
    (phasic_out / "phasic_trace_cache.h5").write_bytes(b"not opened")
    return phasic_out


def _record(
    source: str,
    roi: str,
    chunk_id: int,
    mode: str,
    *,
    dynamic: str = "viable",
    f0_viability: str = "viable",
    f0_confidence: str = "medium",
    f0_flags=None,
    proposal_flags=None,
    warning: str = "none",
    review_required: bool = False,
) -> dict:
    return {
        "source_file": source,
        "roi": roi,
        "chunk_id": chunk_id,
        "dynamic_reference_viability": dynamic,
        "baseline_reference_viability": "contextual",
        "proposed_correction_mode_balanced": mode,
        "proposal_confidence_balanced": "medium",
        "warning_level_balanced": warning,
        "review_required_balanced": review_required,
        "review_queue_candidate_balanced": mode != "dynamic_isosbestic",
        "review_priority_balanced": "medium" if review_required else "none",
        "proposal_reason_balanced": "test",
        "proposal_flags_balanced": proposal_flags or [],
        "signal_only_f0_candidate_viability": f0_viability,
        "signal_only_f0_candidate_confidence": f0_confidence,
        "signal_only_f0_flags": f0_flags or [],
    }


def test_dynamic_dominant_recording_proposes_dynamic_fit(tmp_path):
    records = [
        _record("s1.csv", "R0", i, "dynamic_isosbestic", dynamic="viable")
        for i in range(9)
    ] + [
        _record("s1.csv", "R0", 9, "no_clean_reference_candidate", dynamic="contextual")
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = propose_recording_correction_strategy(phasic_out)

    row = report["rows"][0]
    assert row["applied_correction_strategy_proposed"] == "dynamic_fit"
    assert row["auto_selection_confidence"] in {"high", "medium"}
    assert "auto_selected_dynamic_fit" not in row.values()


def test_signal_only_dominant_recording_proposes_signal_only_f0(tmp_path):
    records = [
        _record(
            "s1.csv",
            "R0",
            i,
            "signal_only_f0_candidate",
            dynamic="contextual",
            f0_confidence="medium",
            proposal_flags=["DYNAMIC_CONTEXTUAL"],
        )
        for i in range(7)
    ] + [
        _record("s1.csv", "R0", i + 7, "no_clean_reference_candidate", dynamic="contextual")
        for i in range(3)
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = propose_recording_correction_strategy(phasic_out)

    row = report["rows"][0]
    assert row["applied_correction_strategy_proposed"] == "signal_only_f0"
    assert row["auto_selection_reason"] == "signal_only_f0_supported_reference_problem_widespread"


def test_no_defensible_strategy_requires_review(tmp_path):
    records = [
        _record(
            "s1.csv",
            "R0",
            i,
            "review_required",
            dynamic="hard_inspect",
            f0_viability="hard_inspect",
            f0_flags=["SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS"],
            review_required=True,
            warning="severe",
        )
        for i in range(5)
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = propose_recording_correction_strategy(phasic_out)

    row = report["rows"][0]
    assert row["applied_correction_strategy_proposed"] == "no_correction"
    assert row["auto_selection_review_required"] is True
    assert "RECORDING_NO_SINGLE_STRATEGY_DEFENSIBLE" in row["auto_selection_flags"]


def test_grouping_keeps_sources_and_rois_separate(tmp_path):
    records = [
        _record("s1.csv", "R0", 0, "dynamic_isosbestic"),
        _record("s2.csv", "R0", 0, "dynamic_isosbestic"),
        _record("s1.csv", "R1", 0, "dynamic_isosbestic"),
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = propose_recording_correction_strategy(phasic_out)

    keys = {(row["source_file"], row["roi"]) for row in report["rows"]}
    assert keys == {("s1.csv", "R0"), ("s2.csv", "R0"), ("s1.csv", "R1")}


def test_read_only_inputs_and_json_csv_consistency(tmp_path):
    records = [_record("s1.csv", "R0", i, "dynamic_isosbestic") for i in range(3)]
    phasic_out = _make_phasic_out(tmp_path, records)
    qc = phasic_out / "qc"
    chunk_json = qc / "baseline_reference_candidate_by_chunk.json"
    summary = qc / "qc_summary.json"
    h5 = phasic_out / "phasic_trace_cache.h5"
    before_json = chunk_json.read_bytes()
    before_summary = summary.read_bytes()
    before_h5 = h5.read_bytes()

    report = propose_recording_correction_strategy(phasic_out)

    assert chunk_json.read_bytes() == before_json
    assert summary.read_bytes() == before_summary
    assert h5.read_bytes() == before_h5
    csv_df = pd.read_csv(report["output_csv"])
    json_rows = json.loads(Path(report["output_json"]).read_text(encoding="utf-8"))
    assert len(csv_df) == len(json_rows) == 1
    assert csv_df.loc[0, "applied_correction_strategy_proposed"] == json_rows[0][
        "applied_correction_strategy_proposed"
    ]


def test_dry_run_does_not_write_outputs(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, [_record("s1.csv", "R0", 0, "dynamic_isosbestic")])

    report = propose_recording_correction_strategy(phasic_out, dry_run=True)

    assert report["dry_run"] is True
    assert not Path(report["output_csv"]).exists()
    assert not Path(report["output_json"]).exists()


def test_roi_and_source_filters(tmp_path):
    records = [
        _record("s1.csv", "R0", 0, "dynamic_isosbestic"),
        _record("s1.csv", "R1", 0, "dynamic_isosbestic"),
        _record("s2.csv", "R0", 0, "dynamic_isosbestic"),
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    roi_report = propose_recording_correction_strategy(phasic_out, roi="R1", dry_run=True)
    source_report = propose_recording_correction_strategy(
        phasic_out, source_file="s2.csv", dry_run=True
    )

    assert {(row["source_file"], row["roi"]) for row in roi_report["rows"]} == {("s1.csv", "R1")}
    assert {(row["source_file"], row["roi"]) for row in source_report["rows"]} == {("s2.csv", "R0")}
