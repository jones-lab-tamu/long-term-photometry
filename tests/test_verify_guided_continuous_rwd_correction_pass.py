from __future__ import annotations

import numpy as np
import pytest

import tools.verify_guided_continuous_rwd_correction_pass as subject


_SYNTHETIC_ROI_STRATEGIES = {
    "ROI1": "global_linear_regression",
    "ROI2": "signal_only_f0",
}


def _synthetic_roi_cli_args() -> list[str]:
    return [
        "--roi",
        "ROI1=global_linear_regression",
        "--roi",
        "ROI2=signal_only_f0",
    ]


def _values(indices):
    time = indices / 10.0
    control1 = 2.0 + 0.15 * np.cos(0.17 * time)
    control2 = 3.0 + 0.12 * np.sin(0.11 * time + 0.4)
    signal1 = 5.0 + 1.6 * control1 + 0.08 * np.sin(0.7 * time)
    signal2 = 7.0 + 0.30 * np.cos(0.23 * time) + 0.04 * np.sin(1.3 * time)
    return (
        time.astype(np.float64),
        np.column_stack((control1, control2)).astype(np.float64),
        np.column_stack((signal1, signal2)).astype(np.float64),
    )


def _write_synthetic_source(folder, *, sample_count=6001):
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "Fluorescence.csv"
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(sample_count):
        time, control, signal = _values(np.array([index], dtype=float))
        lines.append(
            f"{time[0]:.1f},{control[0,0]:.12f},{signal[0,0]:.12f},"
            f"{control[0,1]:.12f},{signal[0,1]:.12f}\n"
        )
    source.write_text("".join(lines), encoding="utf-8", newline="")
    return source


@pytest.fixture(scope="module")
def synthetic_source(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_c4d_synth") / "recording"
    _write_synthetic_source(folder)
    return folder


# ---------------------------------------------------------------------------
# Source-folder argument/environment handling
# ---------------------------------------------------------------------------


def test_resolve_source_folder_uses_cli_argument(synthetic_source):
    args = subject.parse_args(["--source-folder", str(synthetic_source)])
    assert subject.resolve_source_folder(args) == str(synthetic_source)


def test_resolve_source_folder_falls_back_to_env_var(synthetic_source, monkeypatch):
    monkeypatch.setenv(subject.SOURCE_FOLDER_ENV_VAR, str(synthetic_source))
    args = subject.parse_args([])
    assert subject.resolve_source_folder(args) == str(synthetic_source)


def test_resolve_source_folder_refuses_when_neither_supplied(monkeypatch):
    monkeypatch.delenv(subject.SOURCE_FOLDER_ENV_VAR, raising=False)
    args = subject.parse_args([])
    with pytest.raises(subject.VerificationError):
        subject.resolve_source_folder(args)


def test_resolve_source_folder_refuses_missing_directory(tmp_path):
    missing = tmp_path / "does_not_exist"
    args = subject.parse_args(["--source-folder", str(missing)])
    with pytest.raises(subject.VerificationError):
        subject.resolve_source_folder(args)


def test_resolve_source_folder_refuses_directory_without_fluorescence_csv(tmp_path):
    empty = tmp_path / "empty_folder"
    empty.mkdir()
    args = subject.parse_args(["--source-folder", str(empty)])
    with pytest.raises(subject.VerificationError):
        subject.resolve_source_folder(args)


def test_main_returns_nonzero_when_source_missing(monkeypatch):
    monkeypatch.delenv(subject.SOURCE_FOLDER_ENV_VAR, raising=False)
    exit_code = subject.main(
        ["--source-folder", "C:/does/not/exist", *_synthetic_roi_cli_args()]
    )
    assert exit_code != 0


# ---------------------------------------------------------------------------
# Explicit per-ROI verification-strategy requirement (no automatic cycling)
# ---------------------------------------------------------------------------


def test_parse_roi_strategy_arguments_builds_a_plain_mapping():
    parsed = subject._parse_roi_strategy_arguments(
        ["ROI1=global_linear_regression", "ROI2=signal_only_f0"]
    )
    assert parsed == _SYNTHETIC_ROI_STRATEGIES


def test_parse_roi_strategy_arguments_refuses_conflicting_duplicates():
    with pytest.raises(subject.VerificationError):
        subject._parse_roi_strategy_arguments(
            ["ROI1=global_linear_regression", "ROI1=signal_only_f0"]
        )


def test_parse_roi_strategy_arguments_allows_identical_duplicates():
    parsed = subject._parse_roi_strategy_arguments(
        ["ROI1=global_linear_regression", "ROI1=global_linear_regression"]
    )
    assert parsed == {"ROI1": "global_linear_regression"}


def test_require_explicit_roi_strategies_refuses_when_none_supplied():
    with pytest.raises(subject.VerificationError):
        subject._require_explicit_roi_strategies(("ROI1", "ROI2"), None)
    with pytest.raises(subject.VerificationError):
        subject._require_explicit_roi_strategies(("ROI1", "ROI2"), {})


def test_require_explicit_roi_strategies_refuses_missing_roi():
    with pytest.raises(subject.VerificationError):
        subject._require_explicit_roi_strategies(
            ("ROI1", "ROI2"), {"ROI1": "global_linear_regression"}
        )


def test_require_explicit_roi_strategies_refuses_unknown_roi():
    with pytest.raises(subject.VerificationError):
        subject._require_explicit_roi_strategies(
            ("ROI1",), {"ROI1": "global_linear_regression", "ROI9": "signal_only_f0"}
        )


def test_require_explicit_roi_strategies_refuses_unsupported_strategy():
    with pytest.raises(subject.VerificationError):
        subject._require_explicit_roi_strategies(
            ("ROI1",), {"ROI1": "not_a_real_strategy"}
        )


def test_require_explicit_roi_strategies_accepts_complete_explicit_map():
    result = subject._require_explicit_roi_strategies(
        ("ROI1", "ROI2"), dict(_SYNTHETIC_ROI_STRATEGIES)
    )
    assert result == _SYNTHETIC_ROI_STRATEGIES


def test_main_refuses_when_no_roi_arguments_supplied(synthetic_source):
    exit_code = subject.main(["--source-folder", str(synthetic_source)])
    assert exit_code != 0


def test_main_refuses_when_an_included_roi_has_no_strategy(synthetic_source):
    exit_code = subject.main(
        [
            "--source-folder",
            str(synthetic_source),
            "--roi",
            "ROI1=global_linear_regression",
        ]
    )
    assert exit_code != 0


def test_main_refuses_unknown_roi_id(synthetic_source):
    exit_code = subject.main(
        [
            "--source-folder",
            str(synthetic_source),
            "--roi",
            "ROI1=global_linear_regression",
            "--roi",
            "ROI2=signal_only_f0",
            "--roi",
            "ROI9=global_linear_regression",
        ]
    )
    assert exit_code != 0


def test_main_refuses_conflicting_duplicate_roi_assignment(synthetic_source):
    exit_code = subject.main(
        [
            "--source-folder",
            str(synthetic_source),
            "--roi",
            "ROI1=global_linear_regression",
            "--roi",
            "ROI1=robust_global_event_reject",
            "--roi",
            "ROI2=signal_only_f0",
        ]
    )
    assert exit_code != 0


def test_build_shared_authorities_requires_roi_strategies_argument(synthetic_source):
    with pytest.raises(subject.VerificationError):
        subject.build_shared_authorities(
            str(synthetic_source), continuous_window_sec=20.0, roi_strategies=None
        )


# ---------------------------------------------------------------------------
# Successful synthetic verification: report shape, timing, coverage
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_report(synthetic_source):
    return subject.run_verification(
        str(synthetic_source),
        continuous_window_sec=20.0,
        production_block_size=100_000,
        comparison_block_size=733,
        roi_strategies=dict(_SYNTHETIC_ROI_STRATEGIES),
    )


def test_report_contains_required_top_level_fields(synthetic_report):
    for key in (
        "source",
        "verification_plan",
        "continuous_window_sec",
        "production_partition",
        "comparison_partition",
        "c2_partition_comparison",
        "total_runtime_seconds",
        "overall_success",
    ):
        assert key in synthetic_report

    source = synthetic_report["source"]
    for key in (
        "source_folder",
        "fluorescence_file",
        "source_file_size_bytes",
        "source_row_count",
        "recording_duration_seconds",
        "source_cadence_seconds",
        "roi_ids",
        "per_roi_strategies",
    ):
        assert key in source

    partition = synthetic_report["production_partition"]
    for key in (
        "block_size",
        "block_count",
        "target_sample_count",
        "segment_count",
        "final_tail_classification",
        "first_segment",
        "last_segment",
        "dynamic_f0_authority_identity",
        "corrected_segment_count",
        "ordered_segment_identity_digest",
        "completion_identity",
        "full_coverage_result",
        "late_verification_result",
        "pass1_runtime_seconds",
        "pass2_runtime_seconds",
        "memory_before_pass1",
        "memory_after_pass1",
        "memory_after_pass2",
    ):
        assert key in partition


def test_pass1_and_pass2_timing_are_distinct_measurements(synthetic_report):
    partition = synthetic_report["production_partition"]
    assert isinstance(partition["pass1_runtime_seconds"], float)
    assert isinstance(partition["pass2_runtime_seconds"], float)
    assert partition["pass1_runtime_seconds"] >= 0.0
    assert partition["pass2_runtime_seconds"] >= 0.0
    # They must be two independently taken measurements, not the same value
    # aliased into both fields (a real bug this guards against).
    assert partition["pass1_runtime_seconds"] != partition["pass2_runtime_seconds"]


def test_full_coverage_and_late_verification_are_true_on_success(synthetic_report):
    partition = synthetic_report["production_partition"]
    assert partition["full_coverage_result"] is True
    assert partition["late_verification_result"] is True
    assert synthetic_report["overall_success"] is True


@pytest.mark.parametrize(
    "duration,expected_classification,expected_sample_counts",
    [
        (600.0, "exact_full_final_segment", [6000]),
        (298.9, "viable_retained_short_tail", [2989, 2989, 22]),
        (299.9, "merged_nonviable_short_tail", [2999, 3001]),
    ],
)
def test_final_tail_classification_matches_accepted_c4a_plan(
    synthetic_source, duration, expected_classification, expected_sample_counts
):
    report = subject.run_verification(
        str(synthetic_source),
        continuous_window_sec=duration,
        production_block_size=100_000,
        comparison_block_size=733,
        roi_strategies=dict(_SYNTHETIC_ROI_STRATEGIES),
    )
    partition = report["production_partition"]
    assert partition["final_tail_classification"] == expected_classification
    assert partition["segment_count"] == len(expected_sample_counts)
    assert partition["last_segment"]["sample_count"] == expected_sample_counts[-1]
    assert partition["first_segment"]["sample_count"] == expected_sample_counts[0]


def test_c2_partition_comparison_reports_identical_for_valid_partitions(synthetic_report):
    comparison = synthetic_report["c2_partition_comparison"]
    assert comparison["identical"] is True
    assert all(comparison["checks"].values())
    assert comparison["production_block_size"] != comparison["comparison_block_size"]


def test_compare_partition_reports_detects_a_real_mismatch():
    base = {
        "block_size": 100_000,
        "block_count": 1,
        "target_sample_count": 100,
        "segment_count": 1,
        "corrected_segment_count": 1,
        "dynamic_f0_authority_identity": "a" * 64,
        "segment_result_identities": ["b" * 64],
        "ordered_segment_identity_digest": "c" * 64,
        "completion_identity": "d" * 64,
        "full_coverage_result": True,
    }
    other = dict(base, block_size=733, completion_identity="e" * 64)
    result = subject.compare_partition_reports(base, other)
    assert result["identical"] is False
    assert result["checks"]["completion_identity"] is False
    assert result["checks"]["target_sample_count"] is True


# ---------------------------------------------------------------------------
# Truthful verification-plan provenance (not a correction-preview claim)
# ---------------------------------------------------------------------------


def test_verification_plan_identifies_explicit_verification_input_for_direct_api_call(
    synthetic_report,
):
    """``synthetic_report`` comes from a direct ``run_verification(...)`` call
    (as this whole test module does, not a CLI invocation), so the report
    must not claim CLI origin for it -- only that it was explicit
    verification input.
    """
    plan = synthetic_report["verification_plan"]
    assert plan["strategy_source"] == "explicit_verification_input"
    assert plan["strategy_source"] != "explicit_cli"
    assert plan["per_roi_strategies"] == _SYNTHETIC_ROI_STRATEGIES
    assert plan["scientific_endorsement"] is False


def test_verification_plan_label_is_identical_for_cli_invocation(synthetic_source, tmp_path):
    """The same neutral label is used whether the strategies arrived via
    --roi CLI parsing or a direct function call -- the report never
    distinguishes CLI origin from importable-API origin.
    """
    report_path = tmp_path / "cli_report.json"
    exit_code = subject.main(
        [
            "--source-folder",
            str(synthetic_source),
            "--continuous-window-sec",
            "600.0",
            "--report-path",
            str(report_path),
            *_synthetic_roi_cli_args(),
        ]
    )
    assert exit_code == 0
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verification_plan"]["strategy_source"] == "explicit_verification_input"


def test_verification_plan_does_not_claim_preview_evidence(synthetic_source):
    """The choice provenance must never claim a local correction preview

    occurred, since this harness never runs one -- only that the caller
    explicitly supplied the strategy for CR1-C4d verification purposes.
    """
    _source_evidence, _verification_plan, _binding, draft, _grid = (
        subject.build_shared_authorities(
            str(synthetic_source),
            continuous_window_sec=20.0,
            roi_strategies=dict(_SYNTHETIC_ROI_STRATEGIES),
        )
    )
    for choice in draft.per_roi_correction_strategy_choices:
        assert choice.source_type == subject.VERIFICATION_EVIDENCE_SOURCE_TYPE
        assert choice.source_type != "local_correction_preview"
        assert (
            choice.evidence_reference.get("evidence_source_type")
            == subject.VERIFICATION_EVIDENCE_SOURCE_TYPE
        )
        assert choice.evidence_reference.get("evidence_source_type") != "local_correction_preview"
        assert "preview" not in str(choice.source_type).lower()


def test_verification_plan_scientific_endorsement_is_always_false(synthetic_report):
    # Explicit guard against ever quietly flipping this to True without a
    # real evidentiary basis -- the harness has no mechanism to establish one.
    assert synthetic_report["verification_plan"]["scientific_endorsement"] is False


# ---------------------------------------------------------------------------
# Failure paths: nonzero exit, no false success
# ---------------------------------------------------------------------------


def test_main_returns_nonzero_and_no_false_success_when_pass2_fails(
    synthetic_source, monkeypatch
):
    def broken_traversal(*args, **kwargs):
        raise RuntimeError("simulated Pass 2 failure")

    monkeypatch.setattr(
        subject, "iterate_guided_continuous_rwd_corrected_segments", broken_traversal
    )
    exit_code = subject.main(
        [
            "--source-folder",
            str(synthetic_source),
            "--continuous-window-sec",
            "20.0",
            *_synthetic_roi_cli_args(),
        ]
    )
    assert exit_code != 0


def test_run_one_partition_refuses_incomplete_traversal_state(synthetic_source, monkeypatch):
    """A traversal that never reaches "completed" must never report success.

    ``GuidedContinuousRwdCorrectionPassTraversal.__next__`` is a dunder method
    looked up on the type by ``for``/``next()``, so an instance-level
    monkeypatch of it has no effect; instead this forces a genuine
    mid-traversal failure in the real traversal by breaking the C4b
    correction call it delegates to on the second segment.
    """
    _source_evidence, _verification_plan, binding, draft, grid = (
        subject.build_shared_authorities(
            str(synthetic_source),
            continuous_window_sec=20.0,
            roi_strategies=dict(_SYNTHETIC_ROI_STRATEGIES),
        )
    )
    import photometry_pipeline.guided_continuous_rwd_correction_pass as pass_module

    real_correct = pass_module.correct_guided_continuous_rwd_segment
    calls = {"count": 0}

    def flaky_correct(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated mid-traversal failure")
        return real_correct(*args, **kwargs)

    monkeypatch.setattr(pass_module, "correct_guided_continuous_rwd_segment", flaky_correct)
    probe = subject._MemoryProbe()
    with pytest.raises(Exception):
        subject.run_one_partition(binding, draft, grid, block_size=733, memory_probe=probe)
    assert calls["count"] >= 2


def test_main_does_not_write_report_containing_source_path_unless_requested(
    synthetic_source, tmp_path
):
    report_path = tmp_path / "report.json"
    assert not report_path.exists()
    exit_code = subject.main(
        [
            "--source-folder",
            str(synthetic_source),
            "--continuous-window-sec",
            "600.0",
            *_synthetic_roi_cli_args(),
        ]
    )
    assert exit_code == 0
    assert not report_path.exists()
