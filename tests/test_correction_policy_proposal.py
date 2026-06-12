import json

import pandas as pd

from photometry_pipeline.core.correction_policy_proposal import propose_correction_policy


def _record(dynamic="viable", baseline="viable", flags=None, relationship="positive_reference_relationship"):
    return {
        "dynamic_reference_viability": dynamic,
        "baseline_reference_viability": baseline,
        "reference_comparison_flags": flags or ["BASELINE_POSITIVE_REFERENCE_RELATIONSHIP"],
        "baseline_fit_relationship_class": relationship,
    }


def test_balanced_dynamic_viable_accepts_dynamic_without_review():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="viable", baseline="hard_inspect"),
        policy="balanced",
    )

    assert result["proposed_correction_mode"] == "dynamic"
    assert result["proposal_confidence"] == "high"
    assert result["review_required"] is False
    assert result["review_priority"] == "none"


def test_balanced_dynamic_hard_baseline_viable_proposes_reviewed_baseline():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="hard_inspect", baseline="viable"),
        policy="balanced",
    )

    assert result["proposed_correction_mode"] == "baseline_reference_candidate"
    assert result["proposal_confidence"] == "medium"
    assert result["review_required"] is True
    assert result["review_priority"] == "medium"


def test_balanced_contextual_negative_baseline_requires_review():
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

    assert result["proposed_correction_mode"] == "review_required"
    assert result["review_required"] is True
    assert result["review_priority"] == "medium"
    assert "INVERTED_REFERENCE_RELATIONSHIP" in result["proposal_flags"]


def test_balanced_dynamic_hard_baseline_contextual_is_high_priority_review():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="hard_inspect", baseline="contextual"),
        policy="balanced",
    )

    assert result["proposed_correction_mode"] == "review_required"
    assert result["review_required"] is True
    assert result["review_priority"] == "high"


def test_conservative_dynamic_contextual_requires_review():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="contextual", baseline="viable"),
        policy="conservative",
    )

    assert result["proposed_correction_mode"] == "review_required"
    assert result["review_required"] is True


def test_conservative_dynamic_hard_baseline_viable_is_high_priority_reviewed_baseline():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="hard_inspect", baseline="viable"),
        policy="conservative",
    )

    assert result["proposed_correction_mode"] == "baseline_reference_candidate"
    assert result["review_required"] is True
    assert result["review_priority"] == "high"


def test_liberal_contextual_candidates_propose_dynamic_for_screening():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="contextual", baseline="contextual"),
        policy="liberal",
    )

    assert result["proposed_correction_mode"] == "dynamic"
    assert result["proposal_confidence"] == "low"
    assert result["review_required"] is False
    assert result["review_priority"] == "low"


def test_liberal_dynamic_hard_baseline_viable_proposes_baseline_for_screening():
    result = propose_correction_policy(
        comparison_record=_record(dynamic="hard_inspect", baseline="viable"),
        policy="liberal",
    )

    assert result["proposed_correction_mode"] == "baseline_reference_candidate"
    assert result["proposal_confidence"] == "medium"
    assert result["review_required"] is False
    assert result["review_priority"] == "low"


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
        assert result["review_required"] is True


def test_missing_baseline_relationship_class_is_not_proposed_as_baseline():
    record = {
        "dynamic_reference_viability": "hard_inspect",
        "baseline_reference_viability": "viable",
        "reference_comparison_flags": [],
    }

    for policy in ("conservative", "balanced", "liberal"):
        result = propose_correction_policy(comparison_record=record, policy=policy)
        assert result["proposed_correction_mode"] != "baseline_reference_candidate"
        assert result["review_required"] is True


def test_positive_viable_baseline_relationship_can_still_be_proposed_as_baseline():
    record = _record(
        dynamic="hard_inspect",
        baseline="viable",
        flags=["BASELINE_POSITIVE_REFERENCE_RELATIONSHIP"],
        relationship="positive_reference_relationship",
    )

    balanced = propose_correction_policy(comparison_record=record, policy="balanced")
    assert balanced["proposed_correction_mode"] == "baseline_reference_candidate"
    assert balanced["review_required"] is True

    liberal = propose_correction_policy(comparison_record=record, policy="liberal")
    assert liberal["proposed_correction_mode"] == "baseline_reference_candidate"
    assert liberal["review_required"] is False

    conservative = propose_correction_policy(comparison_record=record, policy="conservative")
    assert conservative["proposed_correction_mode"] == "baseline_reference_candidate"
    assert conservative["review_required"] is True


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
