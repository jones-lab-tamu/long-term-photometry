import csv
import json
from pathlib import Path

import pandas as pd

from tools.export_recording_correction_strategy_report import (
    COMPACT_FIELDS,
    export_recording_correction_strategy_report,
)


def _proposal_row(roi: str, *, review_ids="", caution_ids="", strategy="dynamic_fit") -> dict:
    return {
        "recording_key": f"recording_{roi}",
        "roi": roi,
        "n_chunks": "10",
        "source_file_count": "2",
        "applied_correction_strategy_proposed": strategy,
        "auto_selection_confidence": "high" if strategy == "dynamic_fit" else "medium",
        "auto_selection_review_required": "false" if strategy == "dynamic_fit" else "true",
        "auto_selection_reason": "dynamic_fit_dominant_clean"
        if strategy == "dynamic_fit"
        else "signal_only_f0_best_available_dynamic_problem_widespread",
        "auto_selection_flags": "RECORDING_DYNAMIC_FIT_DOMINANT"
        if strategy == "dynamic_fit"
        else "RECORDING_SIGNAL_ONLY_F0_BEST_AVAILABLE;RECORDING_REVIEW_REQUIRED",
        "fraction_dynamic_isosbestic": "0.900000" if strategy == "dynamic_fit" else "0.100000",
        "fraction_dynamic_problem": "0.100000" if strategy == "dynamic_fit" else "0.900000",
        "fraction_dynamic_hard_inspect": "0.000000" if strategy == "dynamic_fit" else "0.300000",
        "fraction_signal_only_f0_candidate": "0.000000" if strategy == "dynamic_fit" else "0.400000",
        "fraction_signal_only_f0_usable": "1.000000" if strategy == "dynamic_fit" else "0.900000",
        "fraction_signal_only_f0_medium_or_high_confidence": "1.000000" if strategy == "dynamic_fit" else "0.600000",
        "fraction_signal_only_f0_bad": "0.000000" if strategy == "dynamic_fit" else "0.100000",
        "fraction_no_clean_reference_candidate": "0.000000" if strategy == "dynamic_fit" else "0.500000",
        "fraction_review_required": "0.000000" if strategy == "dynamic_fit" else "0.200000",
        "n_dynamic_hard_inspect": "0" if strategy == "dynamic_fit" else "3",
        "n_dynamic_contextual": "1" if strategy == "dynamic_fit" else "6",
        "n_signal_only_f0_low_confidence": "0" if strategy == "dynamic_fit" else "2",
        "n_signal_only_f0_high_extrapolation_or_large_gap": "0" if strategy == "dynamic_fit" else "1",
        "n_signal_only_f0_insufficient_anchors": "0" if strategy == "dynamic_fit" else "1",
        "n_signal_only_f0_insufficient_low_support": "0",
        "review_chunk_ids": review_ids,
        "caution_chunk_ids": caution_ids,
    }


def _make_phasic_out(tmp_path: Path, rows: list[dict]) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    qc = phasic_out / "qc"
    qc.mkdir(parents=True)
    csv_path = qc / "recording_correction_strategy_proposals.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (qc / "recording_correction_strategy_proposals.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return phasic_out


def test_basic_report_generation_writes_markdown_and_compact_csv(tmp_path):
    phasic_out = _make_phasic_out(
        tmp_path,
        [_proposal_row("CH1"), _proposal_row("CH2", strategy="signal_only_f0")],
    )

    report = export_recording_correction_strategy_report(phasic_out)

    assert Path(report["output_md"]).exists()
    assert Path(report["output_csv"]).exists()
    compact = pd.read_csv(report["output_csv"])
    assert len(compact) == 2
    assert set(compact["roi"]) == {"CH1", "CH2"}


def test_markdown_contains_required_human_context(tmp_path):
    phasic_out = _make_phasic_out(
        tmp_path,
        [_proposal_row("CH1"), _proposal_row("CH2", strategy="signal_only_f0")],
    )

    report = export_recording_correction_strategy_report(phasic_out)
    markdown = Path(report["output_md"]).read_text(encoding="utf-8")

    assert "# Recording-Level Correction Strategy Report" in markdown
    assert "proposed strategy is global for that ROI recording" in markdown
    assert "review regions under the proposed global strategy" in markdown
    assert "does not indicate chunkwise switching" in markdown
    assert "does not mean auto failed to choose" in markdown
    assert "does not necessarily mean the ROI is unusable" in markdown
    assert "Strategy proposals are configuration-specific" in markdown
    assert "regenerate the per-chunk QC, recording-level proposals, and this report" in markdown
    assert "Do not interpret per-chunk proposals as applied chunkwise correction modes" in markdown
    assert "## ROI CH1" in markdown
    assert "## ROI CH2" in markdown
    assert "## Interpretation guide" in markdown


def test_compact_csv_fields_and_name_mapping(tmp_path):
    phasic_out = _make_phasic_out(
        tmp_path,
        [_proposal_row("CH1", strategy="signal_only_f0")],
    )

    report = export_recording_correction_strategy_report(phasic_out)
    compact = pd.read_csv(report["output_csv"])

    assert list(compact.columns) == COMPACT_FIELDS
    row = compact.iloc[0]
    assert row["proposed_strategy"] == "signal_only_f0"
    assert str(row["review_required"]).lower() == "true"
    assert row["reason"] == "signal_only_f0_best_available_dynamic_problem_widespread"


def test_warning_preview_truncation(tmp_path):
    phasic_out = _make_phasic_out(
        tmp_path,
        [
            _proposal_row(
                "CH1",
                review_ids="1;2;3;4;5",
                caution_ids="10;11;12;13",
                strategy="signal_only_f0",
            )
        ],
    )

    report = export_recording_correction_strategy_report(phasic_out, max_warning_chunks=3)
    compact = pd.read_csv(report["output_csv"], dtype=str)
    row = compact.iloc[0]

    assert row["review_chunk_ids_preview"] == "1;2;3;..."
    assert row["caution_chunk_ids_preview"] == "10;11;12;..."
    assert row["review_chunk_count"] == "5"
    assert row["caution_chunk_count"] == "4"


def test_dry_run_prints_preview_and_writes_no_outputs(tmp_path, capsys):
    phasic_out = _make_phasic_out(tmp_path, [_proposal_row("CH1")])

    report = export_recording_correction_strategy_report(phasic_out, dry_run=True)
    captured = capsys.readouterr()

    assert "# Recording-Level Correction Strategy Report" in captured.out
    assert "# Recording-Level Correction Strategy Report" in report["markdown"]
    assert not Path(report["output_md"]).exists()
    assert not Path(report["output_csv"]).exists()


def test_roi_filter_writes_only_matching_roi(tmp_path):
    phasic_out = _make_phasic_out(
        tmp_path,
        [_proposal_row("CH1"), _proposal_row("CH2", strategy="signal_only_f0")],
    )

    report = export_recording_correction_strategy_report(phasic_out, roi="CH1")
    compact = pd.read_csv(report["output_csv"])

    assert list(compact["roi"]) == ["CH1"]
    assert "## ROI CH1" in Path(report["output_md"]).read_text(encoding="utf-8")
    assert "## ROI CH2" not in Path(report["output_md"]).read_text(encoding="utf-8")


def test_exporter_is_read_only_for_existing_qc_inputs(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, [_proposal_row("CH1")])
    qc = phasic_out / "qc"
    proposal_csv = qc / "recording_correction_strategy_proposals.csv"
    proposal_json = qc / "recording_correction_strategy_proposals.json"
    unrelated = qc / "qc_summary.json"
    unrelated.write_text(json.dumps({"kept": True}), encoding="utf-8")
    before_csv = proposal_csv.read_bytes()
    before_json = proposal_json.read_bytes()
    before_unrelated = unrelated.read_bytes()

    export_recording_correction_strategy_report(phasic_out)

    assert proposal_csv.read_bytes() == before_csv
    assert proposal_json.read_bytes() == before_json
    assert unrelated.read_bytes() == before_unrelated


def test_json_input_keeps_list_flags_readable(tmp_path):
    phasic_out = _make_phasic_out(tmp_path, [_proposal_row("CH1")])
    json_path = phasic_out / "qc" / "recording_correction_strategy_proposals.json"
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    rows[0]["auto_selection_flags"] = ["A", "B"]
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    report = export_recording_correction_strategy_report(phasic_out, input_json=json_path)
    compact = pd.read_csv(report["output_csv"])

    assert compact.loc[0, "key_flags"] == "A;B"
