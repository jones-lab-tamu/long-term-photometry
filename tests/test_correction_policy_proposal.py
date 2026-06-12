import json

import pandas as pd

from photometry_pipeline.core.correction_policy_proposal import (
    apply_correction_policy_proposals,
    propose_correction_policy,
    summarize_correction_policy_proposals,
)


def _record(dynamic="viable", baseline="viable", flags=None, relationship="positive_reference_relationship"):
    return {
        "dynamic_reference_viability": dynamic,
        "baseline_reference_viability": baseline,
        "reference_comparison_flags": flags or ["BASELINE_POSITIVE_REFERENCE_RELATIONSHIP"],
        "baseline_fit_relationship_class": relationship,
    }


def _with_signal_only_f0(
    record,
    *,
    viability="viable",
    confidence="high",
    anchor_status="sufficient_anchors",
    state_aware=True,
    flags=None,
    available=True,
):
    out = dict(record)
    out.update(
        {
            "signal_only_f0_candidate_available": available,
            "signal_only_f0_candidate_viability": viability,
            "signal_only_f0_candidate_confidence": confidence,
            "signal_only_f0_anchor_status": anchor_status,
            "signal_only_f0_state_aware_used": state_aware,
            "signal_only_f0_flags": flags or [
                "SIGNAL_ONLY_F0_STATE_AWARE_USED",
                "SIGNAL_ONLY_F0_LOW_SUPPORT_ANCHORED",
            ],
        }
    )
    return out


def test_balanced_dynamic_viable_accepts_dynamic_without_review():
    record = _with_signal_only_f0(_record(dynamic="viable", baseline="hard_inspect"))
    result = propose_correction_policy(
        comparison_record=record,
        policy="balanced",
    )

    assert result["proposed_correction_mode"] == "dynamic_isosbestic"
    assert result["proposal_confidence"] == "high"
    assert result["review_required"] is False
    assert result["review_queue_candidate"] is False
    assert result["review_priority"] == "none"
    assert result["warning_level"] == "none"


def test_balanced_dynamic_hard_signal_only_viable_high_proposes_signal_only_f0_candidate():
    record = _with_signal_only_f0(_record(dynamic="hard_inspect", baseline="unavailable"))

    result = propose_correction_policy(comparison_record=record, policy="balanced")

    assert result["proposed_correction_mode"] == "signal_only_f0_candidate"
    assert result["proposal_confidence"] in {"high", "medium"}
    assert result["review_queue_candidate"] is True
    assert result["review_required"] is False
    assert result["warning_level"] == "caution"
    assert "SIGNAL_ONLY_F0_POLICY_CANDIDATE" in result["proposal_flags"]
    assert "SIGNAL_ONLY_F0_STATE_AWARE_ANCHORED" in result["proposal_flags"]


def test_balanced_dynamic_hard_signal_only_contextual_high_state_is_caution_review():
    record = _with_signal_only_f0(
        _record(dynamic="hard_inspect", baseline="unavailable"),
        viability="contextual",
        confidence="medium",
        flags=[
            "SIGNAL_ONLY_F0_STATE_AWARE_USED",
            "SIGNAL_ONLY_F0_LOW_SUPPORT_ANCHORED",
            "SIGNAL_ONLY_F0_HIGH_STATE_PRESENT",
            "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION",
        ],
    )

    balanced = propose_correction_policy(comparison_record=record, policy="balanced")
    conservative = propose_correction_policy(comparison_record=record, policy="conservative")

    assert balanced["proposed_correction_mode"] == "signal_only_f0_candidate"
    assert balanced["proposal_confidence"] == "medium"
    assert balanced["warning_level"] == "caution"
    assert balanced["review_queue_candidate"] is True
    assert balanced["review_required"] is True
    assert "SIGNAL_ONLY_F0_POLICY_REQUIRES_REVIEW" in balanced["proposal_flags"]
    assert conservative["proposed_correction_mode"] == "review_required"


def test_signal_only_f0_hard_inspect_is_not_policy_candidate():
    record = _with_signal_only_f0(
        _record(dynamic="hard_inspect", baseline="unavailable"),
        viability="hard_inspect",
        confidence="low",
        flags=["SIGNAL_ONLY_F0_HARD_INSPECT"],
    )

    result = propose_correction_policy(comparison_record=record, policy="balanced")

    assert result["proposed_correction_mode"] != "signal_only_f0_candidate"
    assert result["proposed_correction_mode"] == "review_required"


def test_contextual_negative_dynamic_can_use_signal_only_f0_viable_medium():
    record = _with_signal_only_f0(
        _record(
            dynamic="contextual",
            baseline="contextual",
            flags=[
                "DYNAMIC_NEGATIVE_OR_MIXED_COUPLING",
                "BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP",
            ],
            relationship="negative_reference_relationship",
        ),
        confidence="medium",
    )

    balanced = propose_correction_policy(comparison_record=record, policy="balanced")
    liberal = propose_correction_policy(comparison_record=record, policy="liberal")

    assert balanced["proposed_correction_mode"] == "signal_only_f0_candidate"
    assert balanced["warning_level"] in {"contextual", "caution", "severe"}
    assert balanced["warning_level"] != "none"
    assert liberal["proposed_correction_mode"] == "signal_only_f0_candidate"
    assert "INVERTED_REFERENCE_RELATIONSHIP" in balanced["proposal_flags"]


def test_signal_only_f0_insufficient_anchors_is_never_policy_candidate():
    record = _with_signal_only_f0(
        _record(dynamic="hard_inspect", baseline="unavailable"),
        anchor_status="insufficient_anchors",
        flags=["SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS"],
    )

    for policy in ("conservative", "balanced", "liberal"):
        result = propose_correction_policy(comparison_record=record, policy=policy)
        assert result["proposed_correction_mode"] != "signal_only_f0_candidate"


def test_balanced_dynamic_hard_baseline_viable_is_no_clean_reference_audit_candidate():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="hard_inspect", baseline="viable"),
        policy="balanced",
    )

    assert result["proposed_correction_mode"] == "no_clean_reference_candidate"
    assert result["proposal_confidence"] == "low"
    assert result["review_required"] is False
    assert result["review_queue_candidate"] is True
    assert result["review_priority"] == "medium"
    assert result["warning_level"] == "caution"
    assert "BASELINE_REFERENCE_CANDIDATE_ACCEPTED" not in result["proposal_flags"]
    assert "NO_CLEAN_REFERENCE_CANDIDATE" in result["proposal_flags"]


def test_balanced_contextual_negative_baseline_is_audit_candidate_not_mandatory_review():
    result = propose_correction_policy(
        comparison_record=_record(
            dynamic="contextual",
            baseline="contextual",
            flags=[
                "DYNAMIC_NEGATIVE_OR_MIXED_COUPLING",
                "BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP",
            ],
            relationship="negative_reference_relationship",
        ),
        policy="balanced",
    )

    assert result["proposed_correction_mode"] == "no_clean_reference_candidate"
    assert result["proposal_confidence"] == "low"
    assert result["review_required"] is False
    assert result["review_queue_candidate"] is True
    assert result["review_priority"] == "low"
    assert result["warning_level"] == "contextual"
    assert "INVERTED_REFERENCE_RELATIONSHIP" in result["proposal_flags"]


def test_balanced_dynamic_contextual_baseline_viable_is_caution_audit_candidate():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="contextual", baseline="viable"),
        policy="balanced",
    )

    assert result["proposed_correction_mode"] == "no_clean_reference_candidate"
    assert result["proposal_confidence"] == "low"
    assert result["review_required"] is False
    assert result["review_queue_candidate"] is True
    assert result["review_priority"] == "medium"
    assert result["warning_level"] == "caution"


def test_balanced_dynamic_hard_baseline_contextual_is_high_priority_review():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="hard_inspect", baseline="contextual"),
        policy="balanced",
    )

    assert result["proposed_correction_mode"] == "review_required"
    assert result["review_required"] is True
    assert result["review_queue_candidate"] is True
    assert result["review_priority"] == "high"
    assert result["warning_level"] == "severe"


def test_balanced_dynamic_hard_baseline_hard_or_unavailable_is_severe_review():
    for baseline in ("hard_inspect", "unavailable"):
        result = propose_correction_policy(
            comparison_record=_record(dynamic="hard_inspect", baseline=baseline),
            policy="balanced",
        )

        assert result["proposed_correction_mode"] == "review_required"
        assert result["review_required"] is True
        assert result["review_queue_candidate"] is True
        assert result["review_priority"] == "high"
        assert result["warning_level"] == "severe"


def test_conservative_dynamic_contextual_requires_review():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="contextual", baseline="viable"),
        policy="conservative",
    )

    assert result["proposed_correction_mode"] == "review_required"
    assert result["review_required"] is True
    assert result["review_queue_candidate"] is True


def test_conservative_dynamic_hard_baseline_viable_is_reviewed_no_clean_reference():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="hard_inspect", baseline="viable"),
        policy="conservative",
    )

    assert result["proposed_correction_mode"] == "no_clean_reference_candidate"
    assert result["review_required"] is True
    assert result["review_queue_candidate"] is True
    assert result["review_priority"] == "high"
    assert "BASELINE_REFERENCE_CANDIDATE_ACCEPTED" not in result["proposal_flags"]


def test_liberal_contextual_candidates_propose_dynamic_for_screening():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="contextual", baseline="contextual"),
        policy="liberal",
    )

    assert result["proposed_correction_mode"] == "no_clean_reference_candidate"
    assert result["proposal_confidence"] == "low"
    assert result["review_required"] is False
    assert result["review_queue_candidate"] is True
    assert result["review_priority"] == "low"
    assert result["warning_level"] == "contextual"


def test_liberal_dynamic_hard_baseline_viable_is_no_clean_reference_for_screening():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="hard_inspect", baseline="viable"),
        policy="liberal",
    )

    assert result["proposed_correction_mode"] == "no_clean_reference_candidate"
    assert result["proposal_confidence"] == "low"
    assert result["review_required"] is False
    assert result["review_priority"] == "medium"
    assert "BASELINE_REFERENCE_CANDIDATE_ACCEPTED" not in result["proposal_flags"]


def test_negative_baseline_relationship_is_never_auto_proposed_as_baseline():
    record = _record(
        dynamic="contextual",
        baseline="contextual",
        flags=["BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP"],
        relationship="negative_reference_relationship",
    )

    for policy in ("conservative", "balanced", "liberal"):
        result = propose_correction_policy(comparison_record=record, policy=policy)
        assert result["proposed_correction_mode"] != "baseline_reference_candidate"


def test_inconsistent_viable_negative_baseline_relationship_is_not_proposed_as_baseline():
    record = _record(
        dynamic="hard_inspect",
        baseline="viable",
        flags=["BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP"],
        relationship="negative_reference_relationship",
    )

    for policy in ("conservative", "balanced", "liberal"):
        result = propose_correction_policy(comparison_record=record, policy=policy)
        assert result["proposed_correction_mode"] != "baseline_reference_candidate"
        assert result["review_queue_candidate"] is True


def test_missing_baseline_relationship_class_is_not_proposed_as_baseline():
    record = {
        "dynamic_reference_viability": "hard_inspect",
        "baseline_reference_viability": "viable",
        "reference_comparison_flags": [],
    }

    for policy in ("conservative", "balanced", "liberal"):
        result = propose_correction_policy(comparison_record=record, policy=policy)
        assert result["proposed_correction_mode"] != "baseline_reference_candidate"
        assert result["review_queue_candidate"] is True


def test_positive_viable_legacy_baseline_relationship_is_not_policy_fallback():
    record = _record(
        dynamic="hard_inspect",
        baseline="viable",
        flags=["BASELINE_POSITIVE_REFERENCE_RELATIONSHIP"],
        relationship="positive_reference_relationship",
    )

    balanced = propose_correction_policy(comparison_record=record, policy="balanced")
    assert balanced["proposed_correction_mode"] == "no_clean_reference_candidate"
    assert balanced["review_required"] is False
    assert balanced["review_queue_candidate"] is True
    assert balanced["warning_level"] == "caution"

    liberal = propose_correction_policy(comparison_record=record, policy="liberal")
    assert liberal["proposed_correction_mode"] == "no_clean_reference_candidate"
    assert liberal["review_required"] is False
    assert liberal["review_queue_candidate"] is True
    assert liberal["warning_level"] == "caution"

    conservative = propose_correction_policy(comparison_record=record, policy="conservative")
    assert conservative["proposed_correction_mode"] == "no_clean_reference_candidate"
    assert conservative["review_required"] is True
    assert conservative["review_queue_candidate"] is True
    assert conservative["warning_level"] == "caution"


def test_no_policy_emits_legacy_baseline_reference_candidate_mode():
    records = [
        _record(dynamic="viable", baseline="hard_inspect"),
        _record(dynamic="contextual", baseline="contextual"),
        _record(dynamic="contextual", baseline="viable"),
        _record(dynamic="hard_inspect", baseline="viable"),
        _record(dynamic="hard_inspect", baseline="contextual"),
        _record(dynamic="hard_inspect", baseline="hard_inspect"),
        {
            "dynamic_reference_viability": "hard_inspect",
            "baseline_reference_viability": "viable",
            "reference_comparison_flags": ["BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP"],
            "baseline_fit_relationship_class": "negative_reference_relationship",
        },
    ]

    for record in records:
        for policy in ("conservative", "balanced", "liberal"):
            result = propose_correction_policy(comparison_record=record, policy=policy)
            assert result["proposed_correction_mode"] != "baseline_reference_candidate"


def test_policy_summary_counts_signal_only_mode_and_excludes_legacy_baseline_mode():
    records = [
        apply_correction_policy_proposals(
            _with_signal_only_f0(_record(dynamic="hard_inspect", baseline="unavailable"))
        ),
        apply_correction_policy_proposals(_record(dynamic="viable", baseline="hard_inspect")),
    ]

    summary = summarize_correction_policy_proposals(records)

    assert summary["balanced"]["proposed_correction_mode_counts"][
        "signal_only_f0_candidate"
    ] == 1
    assert (
        "baseline_reference_candidate"
        not in summary["balanced"]["proposed_correction_mode_counts"]
    )
    assert (
        "SIGNAL_ONLY_F0_POLICY_CANDIDATE"
        in summary["balanced"]["proposal_flag_counts"]
    )


def test_policy_flags_csv_and_json_serialization_shapes():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="hard_inspect", baseline="viable"),
        policy="balanced",
    )
    assert isinstance(result["proposal_flags"], list)
    json_text = json.dumps([result], allow_nan=False)
    assert "[" in json_text

    row = dict(result)
    row["proposal_flags"] = ";".join(row["proposal_flags"])
    df = pd.DataFrame([row])
    assert isinstance(df.loc[0, "proposal_flags"], str)
    assert ";" in df.loc[0, "proposal_flags"]
