import json
from pathlib import Path

import pandas as pd

from tools.recompute_correction_policy_proposals import recompute_policy_proposals


def _row(dynamic="viable", baseline="hard_inspect", relationship="positive_reference_relationship"):
    return {
        "roi": "Region0",
        "chunk_id": 0,
        "source_file": "source.csv",
        "dynamic_reference_viability": dynamic,
        "baseline_reference_viability": baseline,
        "baseline_fit_relationship_class": relationship,
        "reference_comparison_flags": "BASELINE_POSITIVE_REFERENCE_RELATIONSHIP",
        "unrelated_metric": 123,
        "proposed_correction_mode_balanced": "baseline_reference_candidate",
        "proposal_flags_balanced": "BASELINE_REFERENCE_CANDIDATE_ACCEPTED",
    }


def _make_phasic_out(tmp_path: Path) -> Path:
    phasic_out = tmp_path / "_analysis" / "phasic_out"
    (phasic_out / "qc").mkdir(parents=True)
    return phasic_out


def test_recompute_from_csv_updates_policy_fields_and_summary(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    pd.DataFrame(
        [
            _row(dynamic="viable", baseline="hard_inspect"),
            _row(dynamic="hard_inspect", baseline="viable"),
        ]
    ).to_csv(qc / "baseline_reference_candidate_by_chunk.csv", index=False)
    (qc / "qc_summary.json").write_text('{"existing_key": {"kept": true}}\n', encoding="utf-8")

    report = recompute_policy_proposals(phasic_out, backup=False)

    assert report["records_processed"] == 2
    df = pd.read_csv(qc / "baseline_reference_candidate_by_chunk.csv")
    assert "unrelated_metric" in df.columns
    assert "baseline_reference_candidate" not in set(df["proposed_correction_mode_balanced"])
    assert set(df["proposed_correction_mode_balanced"]) == {
        "dynamic_isosbestic",
        "no_clean_reference_candidate",
    }
    assert "BASELINE_REFERENCE_CANDIDATE_ACCEPTED" not in ";".join(
        df["proposal_flags_balanced"].astype(str)
    )

    summary = json.loads((qc / "qc_summary.json").read_text(encoding="utf-8"))
    assert summary["existing_key"] == {"kept": True}
    policy_summary = summary["correction_policy_proposal_summary"]
    assert (
        "baseline_reference_candidate"
        not in policy_summary["balanced"]["proposed_correction_mode_counts"]
    )


def test_recompute_from_json_preserves_json_flag_arrays(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    records = [
        {
            **_row(dynamic="contextual", baseline="contextual"),
            "reference_comparison_flags": [
                "BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP",
            ],
            "baseline_fit_relationship_class": "negative_reference_relationship",
        }
    ]
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    (qc / "qc_summary.json").write_text("{}\n", encoding="utf-8")

    recompute_policy_proposals(phasic_out, backup=False)

    out_records = json.loads(
        (qc / "baseline_reference_candidate_by_chunk.json").read_text(encoding="utf-8")
    )
    assert isinstance(out_records[0]["proposal_flags_balanced"], list)
    assert "INVERTED_REFERENCE_RELATIONSHIP" in out_records[0]["proposal_flags_balanced"]
    assert out_records[0]["proposed_correction_mode_balanced"] == "no_clean_reference_candidate"


def test_json_flag_arrays_are_semicolon_serialized_in_csv(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    records = [
        {
            **_row(dynamic="contextual", baseline="contextual"),
            "reference_comparison_flags": [
                "BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP",
                "EXTRA_CONTEXT_FLAG",
            ],
            "dynamic_fit_qc_hard_flags": ["HARD_A", "HARD_B"],
            "dynamic_fit_qc_soft_flags": ["SOFT_A"],
            "dynamic_fit_qc_flags": ["HARD_A", "SOFT_A"],
            "baseline_fit_relationship_class": "negative_reference_relationship",
        }
    ]
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    (qc / "qc_summary.json").write_text("{}\n", encoding="utf-8")

    recompute_policy_proposals(phasic_out, backup=False)

    csv_text = (qc / "baseline_reference_candidate_by_chunk.csv").read_text(
        encoding="utf-8"
    )
    assert "['BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP'" not in csv_text
    df = pd.read_csv(qc / "baseline_reference_candidate_by_chunk.csv")
    assert (
        df.loc[0, "reference_comparison_flags"]
        == "BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP;EXTRA_CONTEXT_FLAG"
    )
    assert df.loc[0, "dynamic_fit_qc_hard_flags"] == "HARD_A;HARD_B"
    assert df.loc[0, "dynamic_fit_qc_soft_flags"] == "SOFT_A"
    assert df.loc[0, "dynamic_fit_qc_flags"] == "HARD_A;SOFT_A"


def test_backups_created_by_default(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    pd.DataFrame([_row()]).to_csv(qc / "baseline_reference_candidate_by_chunk.csv", index=False)
    (qc / "baseline_reference_candidate_by_chunk.json").write_text(
        json.dumps([_row()]), encoding="utf-8"
    )
    (qc / "qc_summary.json").write_text("{}\n", encoding="utf-8")

    report = recompute_policy_proposals(phasic_out)

    assert len(report["backups_created"]) == 3
    assert list(qc.glob("baseline_reference_candidate_by_chunk.csv.bak_*"))
    assert list(qc.glob("baseline_reference_candidate_by_chunk.json.bak_*"))
    assert list(qc.glob("qc_summary.json.bak_*"))


def test_dry_run_does_not_write_or_backup(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    csv_path = qc / "baseline_reference_candidate_by_chunk.csv"
    pd.DataFrame([_row()]).to_csv(csv_path, index=False)
    original = csv_path.read_text(encoding="utf-8")
    (qc / "qc_summary.json").write_text("{}\n", encoding="utf-8")

    report = recompute_policy_proposals(phasic_out, dry_run=True)

    assert report["dry_run"] is True
    assert report["csv_updated"] is False
    assert csv_path.read_text(encoding="utf-8") == original
    assert not list(qc.glob("*.bak_*"))


def test_proposal_flags_serialization_csv_semicolon_json_arrays(tmp_path):
    phasic_out = _make_phasic_out(tmp_path)
    qc = phasic_out / "qc"
    pd.DataFrame([_row(dynamic="hard_inspect", baseline="viable")]).to_csv(
        qc / "baseline_reference_candidate_by_chunk.csv", index=False
    )
    (qc / "qc_summary.json").write_text("{}\n", encoding="utf-8")

    recompute_policy_proposals(phasic_out, backup=False)

    df = pd.read_csv(qc / "baseline_reference_candidate_by_chunk.csv")
    assert isinstance(df.loc[0, "proposal_flags_balanced"], str)
    assert ";" in df.loc[0, "proposal_flags_balanced"]
    records = json.loads(
        (qc / "baseline_reference_candidate_by_chunk.json").read_text(encoding="utf-8")
    )
    assert isinstance(records[0]["proposal_flags_balanced"], list)
