from photometry_pipeline.core.reference_candidate_comparison import classify_reference_candidates


def _dynamic(severity, hard_flags=None, soft_flags=None):
    hard_flags = hard_flags or []
    soft_flags = soft_flags or []
    return {
        "dynamic_fit_qc_severity": severity,
        "dynamic_fit_qc_hard_flags": hard_flags,
        "dynamic_fit_qc_soft_flags": soft_flags,
        "dynamic_fit_qc_flags": [*hard_flags, *soft_flags],
    }


def _baseline(**overrides):
    payload = {
        "baseline_ref_candidate_available": True,
        "baseline_ref_low_range": False,
        "baseline_ref_flat_or_uninformative": False,
        "baseline_ref_response_scale_rich": False,
        "baseline_ref_smoothing_window_adjusted": False,
        "baseline_ref_smoothing_window_fraction_of_chunk": 0.25,
        "baseline_ref_large_window_fraction_warning": 0.5,
    }
    payload.update(overrides)
    return payload


def test_dynamic_ok_baseline_hard_inspect():
    result = classify_reference_candidates(
        dynamic_qc=_dynamic("ok"),
        baseline_record=_baseline(baseline_ref_low_range=True),
    )

    assert result["dynamic_reference_viability"] == "viable"
    assert result["baseline_reference_viability"] == "hard_inspect"
    assert result["reference_comparison_class"] == "dynamic_viable_baseline_hard_inspect"
    assert "BASELINE_CANDIDATE_LOW_OR_FLAT" in result["reference_comparison_flags"]


def test_dynamic_context_negative_mixed_baseline_viable():
    result = classify_reference_candidates(
        dynamic_qc=_dynamic(
            "context",
            soft_flags=["NEGATIVE_OR_MIXED_REFERENCE_COUPLING"],
        ),
        baseline_record=_baseline(),
    )

    assert result["dynamic_reference_viability"] == "contextual"
    assert result["baseline_reference_viability"] == "viable"
    assert result["reference_comparison_class"] == "dynamic_context_baseline_viable"
    assert "DYNAMIC_NEGATIVE_OR_MIXED_COUPLING" in result["reference_comparison_flags"]


def test_dynamic_context_response_scale_rich_baseline_viable_is_not_failure_class():
    result = classify_reference_candidates(
        dynamic_qc=_dynamic(
            "context",
            soft_flags=["FITTED_REFERENCE_RESPONSE_SCALE_RICH"],
        ),
        baseline_record=_baseline(),
    )

    assert result["dynamic_reference_viability"] == "contextual"
    assert result["reference_comparison_class"] == "dynamic_context_baseline_viable"
    assert "hard_inspect" not in result["reference_comparison_class"]
    assert "DYNAMIC_RESPONSE_SCALE_RICH" in result["reference_comparison_flags"]


def test_dynamic_hard_inspect_baseline_viable():
    result = classify_reference_candidates(
        dynamic_qc=_dynamic(
            "inspect",
            hard_flags=["FITTED_REFERENCE_LOW_RANGE", "FITTED_REFERENCE_FLAT_OR_UNINFORMATIVE"],
        ),
        baseline_record=_baseline(),
    )

    assert result["dynamic_reference_viability"] == "hard_inspect"
    assert result["baseline_reference_viability"] == "viable"
    assert result["reference_comparison_class"] == "dynamic_hard_inspect_baseline_viable"
    assert "DYNAMIC_LOW_OR_FLAT_REFERENCE" in result["reference_comparison_flags"]


def test_both_hard_inspect_review_level_high():
    result = classify_reference_candidates(
        dynamic_qc=_dynamic("inspect", hard_flags=["FITTED_REFERENCE_LOW_RANGE"]),
        baseline_record=_baseline(baseline_ref_flat_or_uninformative=True),
    )

    assert result["reference_comparison_class"] == "dynamic_hard_inspect_baseline_hard_inspect"
    assert result["reference_comparison_review_level"] == "high"


def test_baseline_unavailable():
    result = classify_reference_candidates(
        dynamic_qc=_dynamic("ok"),
        baseline_record=_baseline(baseline_ref_candidate_available=False),
    )

    assert result["baseline_reference_viability"] == "unavailable"
    assert result["reference_comparison_class"] == "baseline_unavailable"
    assert "BASELINE_CANDIDATE_UNAVAILABLE" in result["reference_comparison_flags"]
