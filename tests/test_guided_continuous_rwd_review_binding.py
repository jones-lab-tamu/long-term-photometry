from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import photometry_pipeline.guided_continuous_rwd_review_binding as subject
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    EVALUATION_INTERRUPTED,
    MATERIAL_LONG_INTERVAL_DETECTED,
    SHORT_AND_LONG_DISCONTINUITIES_DETECTED,
    SHORT_INTERVAL_ANOMALY_DETECTED,
    SOURCE_CHANGED_OR_MISMATCHED,
    ContinuousRwdDiscontinuityExample,
    evaluate_continuous_rwd_timestamp_continuity,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    GuidedContinuousRwdReviewBindingError,
    build_guided_continuous_rwd_review_binding,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedNewAnalysisTonicSettingsContract,
)
from photometry_pipeline.guided_plan_identity import (
    compute_guided_new_analysis_draft_plan_identity,
)
from photometry_pipeline.io.rwd_continuous_source import (
    inspect_continuous_rwd_acquisition_folder,
)


def _authorities(tmp_path: Path):
    folder = tmp_path / "recording"
    folder.mkdir()
    source = folder / "Fluorescence.csv"
    source.write_text(
        "synthetic preamble\n"
        "Time(s),CH1-410,CH1-470,CH2-410,CH2-470,"
        "CH10-410,CH10-470,CH3-410,CH3-470\n"
        + "".join(f"{second},1,2,3,4,5,6,7,8\n" for second in range(601)),
        encoding="utf-8",
    )
    inspection = inspect_continuous_rwd_acquisition_folder(folder)
    assert inspection.status == "completed"
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("CH2", "CH1")
    )
    evaluation = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=source
    )
    return source, recording, evaluation


def _draft(*, source_folder=r"C:\accepted\recording", **overrides):
    values = dict(
        input_source_path=str(source_folder),
        resolved_input_source_path=str(source_folder),
        input_format="rwd",
        acquisition_mode="continuous",
        continuous_window_sec=600.0,
        continuous_step_sec=600.0,
        discovered_roi_ids=["CH1", "CH10", "CH2", "CH3"],
        included_roi_ids=["CH1", "CH2"],
        excluded_roi_ids=["CH10", "CH3"],
        output_base_path=r"C:\output",
        global_correction_strategy="global_linear_regression",
        feature_event_profile_id="default",
        feature_event_values={"peak_threshold_k": 8.0},
    )
    values.update(overrides)
    return GuidedNewAnalysisDraftPlan(**values)


@pytest.fixture
def valid_case(tmp_path):
    source, recording, evaluation = _authorities(tmp_path)
    return _draft(source_folder=source.parent), source, recording, evaluation


def _build(case, **overrides):
    draft, source, recording, evaluation = case
    values = dict(
        recording=recording,
        continuity_evaluation=evaluation,
        current_source_path=source,
    )
    values.update(overrides)
    return build_guided_continuous_rwd_review_binding(draft, **values)


def test_valid_authorities_build_one_frozen_deterministic_binding(valid_case):
    draft, source, recording, evaluation = valid_case
    binding = _build(valid_case)
    repeated = _build(valid_case)

    assert binding == repeated
    assert binding.draft_plan_identity == compute_guided_new_analysis_draft_plan_identity(draft)
    assert binding.recording is recording
    assert binding.continuity_evaluation is evaluation
    assert binding.current_source_path == str(source)
    with pytest.raises(FrozenInstanceError):
        binding.current_source_path = "changed"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"acquisition_mode": "intermittent"}, "continuous RWD"),
        ({"input_format": "npm"}, "continuous RWD"),
    ],
)
def test_non_continuous_rwd_draft_is_refused(valid_case, changes, message):
    draft = replace(valid_case[0], **changes)
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match=message):
        build_guided_continuous_rwd_review_binding(
            draft,
            recording=valid_case[2],
            continuity_evaluation=valid_case[3],
            current_source_path=valid_case[1],
        )


def test_malformed_draft_is_refused_through_committed_identity_api(valid_case):
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="draft authority"):
        build_guided_continuous_rwd_review_binding(
            object(),
            recording=valid_case[2],
            continuity_evaluation=valid_case[3],
            current_source_path=valid_case[1],
        )


def test_malformed_draft_schema_is_refused(valid_case):
    malformed = replace(valid_case[0], schema_version="invalid")
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="draft authority"):
        build_guided_continuous_rwd_review_binding(
            malformed,
            recording=valid_case[2],
            continuity_evaluation=valid_case[3],
            current_source_path=valid_case[1],
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"global_correction_strategy": "robust_global_event_reject"},
        {
            "tonic_settings_contract": GuidedNewAnalysisTonicSettingsContract(
                tonic_output_mode="flatten_session_bleach_preserve_session_baseline"
            )
        },
        {"feature_event_values": {"peak_threshold_k": 9.0}},
        {"output_base_path": r"C:\different-output"},
    ],
)
def test_settings_change_only_changes_retained_draft_identity(valid_case, changes):
    original = _build(valid_case)
    changed_draft = replace(valid_case[0], **changes)
    changed = build_guided_continuous_rwd_review_binding(
        changed_draft,
        recording=valid_case[2],
        continuity_evaluation=valid_case[3],
        current_source_path=valid_case[1],
    )
    assert changed.draft_plan_identity != original.draft_plan_identity
    assert changed.recording is original.recording
    assert changed.continuity_evaluation is original.continuity_evaluation


@pytest.mark.parametrize(
    "outcome",
    [
        SHORT_INTERVAL_ANOMALY_DETECTED,
        MATERIAL_LONG_INTERVAL_DETECTED,
        SHORT_AND_LONG_DISCONTINUITIES_DETECTED,
        EVALUATION_INTERRUPTED,
        SOURCE_CHANGED_OR_MISMATCHED,
    ],
)
def test_every_nonpass_outcome_is_refused(valid_case, outcome):
    evaluation = replace(valid_case[3], outcome=outcome)
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="did not pass"):
        _build(valid_case, continuity_evaluation=evaluation)


def test_pass_with_failure_reason_is_refused(valid_case):
    evaluation = replace(valid_case[3], failure_reason="inconsistent")
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="failure reason"):
        _build(valid_case, continuity_evaluation=evaluation)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("recording_identity", "0" * 64, "recording identity"),
        ("source_content_identity", "0" * 64, "source-content identity"),
        ("parser_interpretation_identity", "0" * 64, "parser identity"),
        ("cadence_evidence_identity", "0" * 64, "cadence identity"),
        ("observed_source_sha256", "0" * 64, "source facts"),
        ("observed_source_size_bytes", 1, "source facts"),
        ("policy_name", "different-policy", "policy"),
        ("policy_version", "v999", "policy"),
    ],
)
def test_identity_source_and_policy_mismatches_are_refused(
    valid_case, field, value, message
):
    evaluation = replace(valid_case[3], **{field: value})
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match=message):
        _build(valid_case, continuity_evaluation=evaluation)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("valid_row_count_evaluated", 600, "valid-row"),
        ("positive_interval_count_evaluated", 599, "positive-interval"),
        ("normal_interval_count", 599, "normal-interval"),
        ("short_interval_anomaly_count", 1, "discontinuity evidence"),
        ("material_long_interval_count", 1, "discontinuity evidence"),
        (
            "short_examples",
            (ContinuousRwdDiscontinuityExample("short", 2, 1, 0.0, 0.5, 0.5, 1.0, 0.02, -0.5),),
            "discontinuity evidence",
        ),
        (
            "long_examples",
            (ContinuousRwdDiscontinuityExample("long", 2, 1, 0.0, 2.0, 2.0, 1.0, 0.02, 1.0),),
            "discontinuity evidence",
        ),
        ("maximum_short_residual_seconds", 0.1, "discontinuity evidence"),
        ("maximum_long_residual_seconds", 0.1, "discontinuity evidence"),
    ],
)
def test_inconsistent_pass_counts_and_evidence_are_refused(
    valid_case, field, value, message
):
    evaluation = replace(valid_case[3], **{field: value})
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match=message):
        _build(valid_case, continuity_evaluation=evaluation)


def test_rows_minus_one_invariant_is_independently_enforced(valid_case):
    recording = valid_case[2]
    source = replace(recording.source, valid_timestamp_count=602)
    malformed = replace(recording, source=source)
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="recording authority"):
        _build(valid_case, recording=malformed)


def test_invalid_recording_uses_committed_b1_validation(valid_case):
    malformed = replace(valid_case[2], schema_version="invalid")
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="recording authority"):
        _build(valid_case, recording=malformed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("included_roi_ids", ["CH1"], "included ROI"),
        ("excluded_roi_ids", ["CH10"], "excluded ROI"),
        (
            "discovered_roi_ids",
            ["CH1", "CH10", "CH2", "DRAFT_ONLY"],
            "discovered ROI inventory",
        ),
        (
            "discovered_roi_ids",
            ["CH1", "CH10", "CH2", "CH3", "DRAFT_ONLY"],
            "discovered ROI inventory",
        ),
        (
            "discovered_roi_ids",
            ["CH1", "CH10", "CH2"],
            "discovered ROI inventory",
        ),
    ],
)
def test_draft_roi_mismatch_is_refused(valid_case, field, value, message):
    draft = replace(valid_case[0], **{field: value})
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match=message):
        build_guided_continuous_rwd_review_binding(
            draft,
            recording=valid_case[2],
            continuity_evaluation=valid_case[3],
            current_source_path=valid_case[1],
        )


def test_discovered_roi_order_is_preserved_and_wrong_order_is_refused(valid_case):
    draft = replace(
        valid_case[0], discovered_roi_ids=list(reversed(valid_case[0].discovered_roi_ids))
    )
    with pytest.raises(
        GuidedContinuousRwdReviewBindingError, match="discovered ROI inventory"
    ):
        build_guided_continuous_rwd_review_binding(
            draft,
            recording=valid_case[2],
            continuity_evaluation=valid_case[3],
            current_source_path=valid_case[1],
        )


def test_included_and_excluded_order_uses_existing_plan_identity_normalization(
    valid_case,
):
    draft = replace(
        valid_case[0],
        included_roi_ids=list(reversed(valid_case[0].included_roi_ids)),
        excluded_roi_ids=list(reversed(valid_case[0].excluded_roi_ids)),
    )
    binding = build_guided_continuous_rwd_review_binding(
        draft,
        recording=valid_case[2],
        continuity_evaluation=valid_case[3],
        current_source_path=valid_case[1],
    )
    assert binding.draft_plan_identity == _build(valid_case).draft_plan_identity


@pytest.mark.parametrize(
    "path",
    [
        Path(r"Z:\not-present\Fluorescence.csv"),
        Path("relative/Fluorescence.csv"),
    ],
)
def test_nonexistent_string_and_path_are_accepted_without_inspection(valid_case, path):
    draft = replace(
        valid_case[0],
        input_source_path=str(path.parent),
        resolved_input_source_path=str(path.parent),
    )
    binding = build_guided_continuous_rwd_review_binding(
        draft,
        recording=valid_case[2],
        continuity_evaluation=valid_case[3],
        current_source_path=str(path) if path.is_absolute() else path,
    )
    assert binding.current_source_path == str(path)


@pytest.mark.parametrize("path", ["", "   ", None, 123])
def test_invalid_current_source_path_is_refused(valid_case, path):
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="source path"):
        _build(valid_case, current_source_path=path)


def test_path_only_change_preserves_all_scientific_authority(valid_case):
    first_path = Path("first/location/Fluorescence.csv")
    second_path = Path("moved/location/Fluorescence.csv")
    first_draft = replace(
        valid_case[0],
        input_source_path=str(first_path.parent),
        resolved_input_source_path=str(first_path.parent),
    )
    second_draft = replace(
        valid_case[0],
        input_source_path=str(second_path.parent),
        resolved_input_source_path=str(second_path.parent),
    )
    first = build_guided_continuous_rwd_review_binding(
        first_draft,
        recording=valid_case[2],
        continuity_evaluation=valid_case[3],
        current_source_path=first_path,
    )
    second = build_guided_continuous_rwd_review_binding(
        second_draft,
        recording=valid_case[2],
        continuity_evaluation=valid_case[3],
        current_source_path=second_path,
    )
    assert first.draft_plan_identity != second.draft_plan_identity
    assert first.recording is second.recording
    assert first.continuity_evaluation is second.continuity_evaluation
    assert first.current_source_path != second.current_source_path


def test_unrelated_current_source_path_is_refused(valid_case):
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="source folder"):
        _build(valid_case, current_source_path=Path("unrelated/Fluorescence.csv"))


@pytest.mark.parametrize(
    "filename", ["Unrelated.csv", "Fluorescence.csv.backup"]
)
def test_unrelated_filename_in_selected_folder_is_refused(valid_case, filename):
    current = valid_case[1].parent / filename
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="source filename"):
        _build(valid_case, current_source_path=current)


def test_malformed_b1_filename_provenance_is_refused(valid_case):
    recording = valid_case[2]
    malformed_source = replace(
        recording.source,
        fluorescence_path_canonical=recording.source.selected_folder_canonical,
    )
    malformed_recording = replace(recording, source=malformed_source)
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="B1 source filename"):
        _build(valid_case, recording=malformed_recording)


def test_resolved_source_folder_takes_precedence_over_input_folder(valid_case):
    draft = replace(
        valid_case[0],
        input_source_path="unrelated/input-folder",
        resolved_input_source_path=str(valid_case[1].parent),
    )
    binding = build_guided_continuous_rwd_review_binding(
        draft,
        recording=valid_case[2],
        continuity_evaluation=valid_case[3],
        current_source_path=valid_case[1],
    )
    assert binding.current_source_path == str(valid_case[1])


def test_input_source_folder_is_fallback_when_resolved_path_is_absent(valid_case):
    draft = replace(
        valid_case[0],
        input_source_path=str(valid_case[1].parent),
        resolved_input_source_path=None,
    )
    binding = build_guided_continuous_rwd_review_binding(
        draft,
        recording=valid_case[2],
        continuity_evaluation=valid_case[3],
        current_source_path=valid_case[1],
    )
    assert binding.current_source_path == str(valid_case[1])


def test_matching_input_does_not_override_unrelated_resolved_folder(valid_case):
    draft = replace(
        valid_case[0],
        input_source_path=str(valid_case[1].parent),
        resolved_input_source_path="unrelated/resolved-folder",
    )
    with pytest.raises(GuidedContinuousRwdReviewBindingError, match="source folder"):
        build_guided_continuous_rwd_review_binding(
            draft,
            recording=valid_case[2],
            continuity_evaluation=valid_case[3],
            current_source_path=valid_case[1],
        )


def test_module_has_only_the_bounded_in_memory_dependency_surface():
    tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_modules = {
        "gui",
        "photometry_pipeline.guided_backend_validation_materialization",
        "photometry_pipeline.guided_backend_validation_workflow",
        "photometry_pipeline.guided_production_mapping",
        "photometry_pipeline.guided_run_authorization",
        "photometry_pipeline.guided_startup_transaction",
        "photometry_pipeline.guided_backend_execution",
        "photometry_pipeline.run_completion_contract",
        "photometry_pipeline.guided_normalized_recording",
        "json",
        "hashlib",
    }
    assert imports.isdisjoint(forbidden_modules)
    public_names = set(vars(subject))
    assert not any("serializ" in name.lower() for name in public_names)
    assert "review_binding_identity" not in public_names
    assert not any("evidence" in name.lower() for name in public_names)
    fields = set(subject.GuidedContinuousRwdReviewBinding.__dataclass_fields__)
    assert fields == {
        "draft_plan_identity",
        "recording",
        "continuity_evaluation",
        "current_source_path",
    }
