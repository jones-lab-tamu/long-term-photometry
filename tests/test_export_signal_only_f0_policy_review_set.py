import json
from pathlib import Path

import pandas as pd

from tools.export_signal_only_f0_policy_review_set import (
    export_signal_only_f0_policy_review_set,
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
    return phasic_out


def _record(
    roi: str,
    chunk_id: int,
    *,
    policy: str = "balanced",
    mode: str = "signal_only_f0_candidate",
    flags=None,
    proposal_flags=None,
    confidence: str = "medium",
    warning: str = "caution",
    dynamic: str = "contextual",
) -> dict:
    return {
        "roi": roi,
        "chunk_id": chunk_id,
        "source_file": "source.csv",
        "dynamic_reference_viability": dynamic,
        "baseline_reference_viability": "contextual",
        f"proposed_correction_mode_{policy}": mode,
        f"proposal_confidence_{policy}": confidence,
        f"warning_level_{policy}": warning,
        f"review_required_{policy}": False,
        f"review_queue_candidate_{policy}": True,
        f"review_priority_{policy}": "medium",
        f"proposal_reason_{policy}": "test_reason",
        f"proposal_flags_{policy}": proposal_flags or [],
        "dynamic_fit_qc_flags": flags or [],
        "reference_comparison_flags": [],
        "baseline_fit_relationship_class": "positive_reference_relationship",
        "signal_state_candidate_class": "ordinary",
        "signal_state_flags": [],
        "signal_only_f0_candidate_viability": "viable",
        "signal_only_f0_candidate_confidence": "medium",
        "signal_only_f0_anchor_status": "sufficient_anchors",
        "signal_only_f0_anchor_count": 7,
        "signal_only_f0_low_support_fraction": 0.2,
        "signal_only_f0_interpolated_fraction": 0.5,
        "signal_only_f0_extrapolated_fraction": 0.1,
        "signal_only_f0_max_anchor_gap_fraction_observed": 0.2,
        "signal_only_f0_flags": [],
    }


def test_basic_export_filters_signal_only_rows_and_serializes_lists(tmp_path):
    records = [
        _record(
            "Region0",
            1,
            flags=["NEGATIVE_OR_MIXED_REFERENCE_COUPLING"],
            proposal_flags=["DYNAMIC_CONTEXTUAL", "SIGNAL_ONLY_F0_POLICY_CANDIDATE"],
        ),
        _record("Region0", 2, mode="dynamic_isosbestic"),
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = export_signal_only_f0_policy_review_set(phasic_out, policy="balanced")

    df = pd.read_csv(report["output_csv"])
    assert len(df) == 1
    assert df.loc[0, "proposed_correction_mode_balanced"] == "signal_only_f0_candidate"
    assert "bucket" in df.columns
    assert "signal_only_f0_flags" in df.columns
    assert df.loc[0, "proposal_flags_balanced"] == (
        "DYNAMIC_CONTEXTUAL;SIGNAL_ONLY_F0_POLICY_CANDIDATE"
    )


def test_bucket_assignment_priority(tmp_path):
    records = [
        _record("A", 1, proposal_flags=["DYNAMIC_HARD_INSPECT"], dynamic="hard_inspect"),
        _record("A", 2, flags=["NEGATIVE_OR_MIXED_REFERENCE_COUPLING"]),
        _record("A", 3, flags=["FITTED_REFERENCE_LOW_RANGE"]),
        {
            **_record("A", 4),
            "reference_comparison_flags": ["BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP"],
        },
        {
            **_record("A", 5),
            "signal_only_f0_flags": ["SIGNAL_ONLY_F0_HIGH_STATE_PRESENT"],
        },
        {
            **_record("A", 6),
            "signal_only_f0_flags": ["SIGNAL_ONLY_F0_PARTIAL_HIGH_STATE_PRESENT"],
        },
        {
            **_record("A", 7),
            "signal_only_f0_flags": ["SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP"],
        },
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = export_signal_only_f0_policy_review_set(phasic_out, policy="balanced")

    df = pd.read_csv(report["output_csv"])
    buckets = set(df["bucket"])
    assert "dynamic_hard_inspect" in buckets
    assert "dynamic_contextual_negative_mixed" in buckets
    assert "dynamic_contextual_low_flat_reference" in buckets
    assert "dynamic_contextual_baseline_negative_or_inverted" in buckets
    assert "signal_high_or_edge_state" in buckets
    assert "signal_partial_high_state" in buckets
    assert "high_extrapolation_or_large_gap" in buckets


def test_per_bucket_limit(tmp_path):
    records = [
        _record("A", i, proposal_flags=["DYNAMIC_HARD_INSPECT"], dynamic="hard_inspect")
        for i in range(5)
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    report = export_signal_only_f0_policy_review_set(
        phasic_out,
        policy="balanced",
        per_bucket=2,
    )

    df = pd.read_csv(report["output_csv"])
    assert len(df) == 2
    assert df["bucket"].value_counts().max() <= 2


def test_policy_selection(tmp_path):
    records = [
        {
            **_record("A", 1, mode="dynamic_isosbestic"),
            "proposed_correction_mode_liberal": "signal_only_f0_candidate",
            "proposal_confidence_liberal": "medium",
            "warning_level_liberal": "contextual",
            "review_required_liberal": False,
            "review_queue_candidate_liberal": True,
            "review_priority_liberal": "low",
            "proposal_reason_liberal": "liberal_reason",
            "proposal_flags_liberal": ["SIGNAL_ONLY_F0_POLICY_CANDIDATE"],
        }
    ]
    phasic_out = _make_phasic_out(tmp_path, records)

    balanced = export_signal_only_f0_policy_review_set(phasic_out, policy="balanced")
    liberal = export_signal_only_f0_policy_review_set(phasic_out, policy="liberal")

    assert pd.read_csv(balanced["output_csv"]).empty
    assert len(pd.read_csv(liberal["output_csv"])) == 1


def test_plot_command_export_groups_sorted_chunks_and_is_read_only(tmp_path):
    records = [
        _record("RegionB", 3, proposal_flags=["DYNAMIC_HARD_INSPECT"], dynamic="hard_inspect"),
        _record("RegionB", 1, proposal_flags=["DYNAMIC_HARD_INSPECT"], dynamic="hard_inspect"),
        _record("RegionA", 2, proposal_flags=["DYNAMIC_HARD_INSPECT"], dynamic="hard_inspect"),
    ]
    phasic_out = _make_phasic_out(tmp_path, records)
    qc = phasic_out / "qc"
    json_path = qc / "baseline_reference_candidate_by_chunk.json"
    summary_path = qc / "qc_summary.json"
    before_json = json_path.read_bytes()
    before_summary = summary_path.read_bytes()

    report = export_signal_only_f0_policy_review_set(
        phasic_out,
        policy="balanced",
        include_plot_commands=True,
    )

    plot_path = Path(report["plot_command_path"])
    text = plot_path.read_text(encoding="utf-8")
    assert "tools/plot_signal_only_f0_candidates.py" in text
    assert "--roi RegionA --chunks 2" in text
    assert "--roi RegionB --chunks 1,3" in text
    assert json_path.read_bytes() == before_json
    assert summary_path.read_bytes() == before_summary
