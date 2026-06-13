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

    keys = {(row["recording_key"], row["roi"]) for row in report["rows"]}
    assert keys == {("s1.csv", "R0"), ("s2.csv", "R0"), ("s1.csv", "R1")}


def test_nested_timestamp_source_files_group_to_recording_key(tmp_path):
    root = tmp_path / "root"
    sources = [
        root / "2026_05_20-09_01_19" / "2026_05_20-09_01_19" / "fluorescence.csv",
        root / "2026_05_20-09_01_19" / "2026_05_20-09_31_20" / "fluorescence.csv",
        root / "2026_05_20-09_01_19" / "2026_05_20-10_01_20" / "fluorescence.csv",
    ]
    records = [
        _record(str(source), "CH1", idx, "dynamic_isosbestic")
        for idx, source in enumerate(sources)
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = propose_recording_correction_strategy(phasic_out)

    assert report["recordings_found"] == 1
    row = report["rows"][0]
    assert row["n_chunks"] == 3
    assert row["source_file_count"] == 3
    assert row["recording_key"].endswith(str(root / "2026_05_20-09_01_19"))
    assert row["grouping_mode"] == "auto"


def test_different_top_level_recordings_stay_separate(tmp_path):
    root = tmp_path / "root"
    records = [
        _record(str(root / "recordingA" / "chunk1" / "fluorescence.csv"), "CH1", 0, "dynamic_isosbestic"),
        _record(str(root / "recordingB" / "chunk1" / "fluorescence.csv"), "CH1", 1, "dynamic_isosbestic"),
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = propose_recording_correction_strategy(phasic_out, grouping_mode="grandparent")

    assert report["recordings_found"] == 2


def test_different_rois_same_recording_stay_separate(tmp_path):
    source = str(
        tmp_path
        / "root"
        / "2026_05_20-09_01_19"
        / "2026_05_20-09_31_20"
        / "fluorescence.csv"
    )
    records = [
        _record(source, "CH1", 0, "dynamic_isosbestic"),
        _record(source, "CH2", 0, "dynamic_isosbestic"),
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = propose_recording_correction_strategy(phasic_out)

    assert {(row["recording_key"], row["roi"]) for row in report["rows"]} == {
        (str(Path(source).parent.parent), "CH1"),
        (str(Path(source).parent.parent), "CH2"),
    }


def test_grouping_mode_source_file_preserves_one_row_per_full_source_file(tmp_path):
    root = tmp_path / "root"
    sources = [
        root / "2026_05_20-09_01_19" / "2026_05_20-09_01_19" / "fluorescence.csv",
        root / "2026_05_20-09_01_19" / "2026_05_20-09_31_20" / "fluorescence.csv",
        root / "2026_05_20-09_01_19" / "2026_05_20-10_01_20" / "fluorescence.csv",
    ]
    records = [
        _record(str(source), "CH1", idx, "dynamic_isosbestic")
        for idx, source in enumerate(sources)
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = propose_recording_correction_strategy(phasic_out, grouping_mode="source_file")

    assert report["recordings_found"] == 3
    assert {row["n_chunks"] for row in report["rows"]} == {1}


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

    assert {(row["recording_key"], row["roi"]) for row in roi_report["rows"]} == {("s1.csv", "R1")}
    assert {(row["recording_key"], row["roi"]) for row in source_report["rows"]} == {("s2.csv", "R0")}


def test_dynamic_fit_not_reviewed_due_only_to_signal_only_large_gap(tmp_path):
    records = [
        _record(
            "s1.csv",
            "R0",
            i,
            "dynamic_isosbestic",
            dynamic="viable",
            f0_flags=["SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP"],
        )
        for i in range(10)
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = propose_recording_correction_strategy(phasic_out)

    row = report["rows"][0]
    assert row["applied_correction_strategy_proposed"] == "dynamic_fit"
    assert row["auto_selection_review_required"] is False


def test_signal_only_strategy_reviews_signal_only_badness(tmp_path):
    records = [
        _record(
            "s1.csv",
            "R0",
            i,
            "signal_only_f0_candidate",
            dynamic="contextual",
            proposal_flags=["DYNAMIC_CONTEXTUAL"],
        )
        for i in range(8)
    ]
    records[0]["signal_only_f0_flags"] = ["SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS"]
    records[1]["signal_only_f0_flags"] = ["SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP"]
    records[2]["review_required_balanced"] = True
    phasic_out = _make_phasic_out(tmp_path, records)

    report = propose_recording_correction_strategy(phasic_out)

    row = report["rows"][0]
    assert row["applied_correction_strategy_proposed"] == "signal_only_f0"
    assert row["auto_selection_review_required"] is True
    assert 0 in row["review_chunk_ids"]
    assert 2 in row["review_chunk_ids"]
    assert 1 in row["caution_chunk_ids"]
