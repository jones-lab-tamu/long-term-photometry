"""Terminal completion contract: a run is successful only as one coherent set (4J16k40)."""

import json
from pathlib import Path

import pytest

from gui.run_report_parser import (
    classify_completed_run_candidate,
    is_successful_completed_run_dir,
)
from photometry_pipeline.run_completion_contract import (
    COMPLETION_CONTRACT_VERSION,
    COMPLETION_KEY,
    FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY,
    FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY,
    FAMILY_PHASIC_DAY_PLOTS,
    FAMILY_PHASIC_TIMESERIES,
    PROFILE_CONTINUOUS,
    PROFILE_FULL_INTERMITTENT,
    PROFILE_TUNING_PREP,
    TERMINAL_CORRUPTED,
    TERMINAL_FAILED,
    TERMINAL_INTERRUPTED,
    TERMINAL_NOT_A_RUN,
    TERMINAL_SUCCESS_CURRENT,
    TERMINAL_SUCCESS_LEGACY,
    RunCompletionError,
    build_continuous_window_index,
    build_manifest_completion_block,
    classify_run_terminal_state,
    normalize_run_mode,
    required_artifacts_for_run_mode,
    required_deliverables_for_run_mode,
    run_mode_structural_error,
)
from tests.terminal_run_fixtures import (
    DEFAULT_RUN_ID,
    repin_status_to_manifest,
    write_current_run,
    write_legacy_run,
    write_region_deliverable,
)


def _full_run_mode(**overrides):
    """A coherent full-production run mode, before any single override."""
    kwargs = dict(
        run_profile="full",
        run_type="full",
        acquisition_mode="intermittent",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_FULL_INTERMITTENT,
        expected_rois=["Region0"],
    )
    kwargs.update(overrides)
    return normalize_run_mode(**kwargs)


def _state(run_dir) -> str:
    return classify_run_terminal_state(str(run_dir)).state


def _rewrite_json(path: Path, mutate) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# 1. Valid current successful run ---------------------------------------------


def test_valid_current_run_is_successful(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    classification = classify_run_terminal_state(str(run_dir))

    assert classification.state == TERMINAL_SUCCESS_CURRENT
    assert classification.run_id == DEFAULT_RUN_ID
    assert classification.contract_version == COMPLETION_CONTRACT_VERSION
    assert is_successful_completed_run_dir(str(run_dir))[0] is True
    assert classify_completed_run_candidate(str(run_dir))[0] is True


def test_valid_tonic_only_run_is_successful(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", phasic=False, tonic=True, features=False)
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT


def test_traces_only_run_needs_no_feature_outputs(tmp_path: Path):
    run_dir = write_current_run(
        tmp_path / "run", run_profile="tuning_prep", run_type="tuning_prep",
        traces_only=True, features=False,
    )
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT


# 2/3. run_report.json is mandatory for current runs ---------------------------


def test_missing_run_report_is_corrupted_not_legacy(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    (run_dir / "run_report.json").unlink()

    classification = classify_run_terminal_state(str(run_dir))
    assert classification.state == TERMINAL_CORRUPTED
    assert classification.state != TERMINAL_SUCCESS_LEGACY
    assert is_successful_completed_run_dir(str(run_dir))[0] is False


def test_malformed_run_report_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    (run_dir / "run_report.json").write_text("{not valid json", encoding="utf-8")

    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_unsupported_report_contract_version_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    _rewrite_json(
        run_dir / "run_report.json",
        lambda d: d["completion_contract"].__setitem__("contract_version", "run_completion.v99"),
    )
    assert _state(run_dir) == TERMINAL_CORRUPTED


# 4. Missing final manifest ----------------------------------------------------


def test_missing_manifest_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    (run_dir / "MANIFEST.json").unlink()

    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_non_final_manifest_is_interrupted(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", manifest_final=False)
    repin_status_to_manifest(run_dir)

    assert _state(run_dir) == TERMINAL_INTERRUPTED


# 5/6. Mandatory vs optional artifacts ----------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "_analysis/phasic_out/config_used.yaml",
        "_analysis/phasic_out/phasic_trace_cache.h5",
        "_analysis/phasic_out/features/feature_event_provenance.json",
    ],
)
def test_missing_mandatory_artifact_is_rejected(tmp_path: Path, rel_path: str):
    run_dir = write_current_run(tmp_path / "run")
    (run_dir / Path(rel_path)).unlink()

    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_absent_optional_artifact_is_accepted(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", optional_artifacts=["events.ndjson"])
    (run_dir / "events.ndjson").unlink()

    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT


def test_run_mode_decides_the_mandatory_set(tmp_path: Path):
    phasic_only = _full_run_mode(tonic_analysis=False)
    required = required_artifacts_for_run_mode(phasic_only)
    assert "_analysis/phasic_out/features/feature_event_provenance.json" in required
    assert not any("tonic_out" in path for path in required)


# 7. Identity mismatch ---------------------------------------------------------


def test_report_identity_mismatch_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", report_run_id="some_other_run")
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_manifest_identity_mismatch_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", manifest_run_id="some_other_run")
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_status_identity_mismatch_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", status_run_id="some_other_run")
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_run_mode_disagreement_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    _rewrite_json(run_dir / "status.json", lambda d: d.__setitem__("traces_only", True))
    assert _state(run_dir) == TERMINAL_CORRUPTED


# 8. Tampered artifacts --------------------------------------------------------


def test_tampered_digested_artifact_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    (run_dir / "_analysis" / "phasic_out" / "config_used.yaml").write_text(
        "target_fs_hz: 999.0\n", encoding="utf-8"
    )
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_tampered_manifest_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    # Rewrite the manifest without re-pinning the status digest.
    _rewrite_json(run_dir / "MANIFEST.json", lambda d: d.__setitem__("tool", "tampered"))
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_manifest_that_omits_a_mandatory_artifact_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")

    def drop_provenance(data):
        block = data[COMPLETION_KEY]
        block["artifacts"] = [
            entry
            for entry in block["artifacts"]
            if "feature_event_provenance" not in entry["relative_path"]
        ]

    _rewrite_json(run_dir / "MANIFEST.json", drop_provenance)
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED


# 9. Interrupted / failed states ----------------------------------------------


def test_in_progress_status_is_interrupted(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", status_state="running", status_phase="plots_total")
    assert _state(run_dir) == TERMINAL_INTERRUPTED


def test_failed_status_is_failed(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", status_state="error")
    assert _state(run_dir) == TERMINAL_FAILED
    assert is_successful_completed_run_dir(str(run_dir))[0] is False


def test_cancelled_status_is_interrupted(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", status_state="cancelled")
    assert _state(run_dir) == TERMINAL_INTERRUPTED


def test_report_present_but_no_status_is_interrupted(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", write_status=False)
    assert _state(run_dir) == TERMINAL_INTERRUPTED


def test_success_status_recording_errors_is_contradictory(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    _rewrite_json(run_dir / "status.json", lambda d: d.__setitem__("errors", ["boom"]))
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_empty_directory_is_not_a_run(tmp_path: Path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    assert _state(run_dir) == TERMINAL_NOT_A_RUN


def test_partial_staging_directory_is_not_successful(tmp_path: Path):
    """Analysis outputs exist but nothing terminal was ever written."""
    run_dir = tmp_path / "staging"
    (run_dir / "_analysis" / "phasic_out" / "features").mkdir(parents=True)
    (run_dir / "_analysis" / "phasic_out" / "features" / "features.csv").write_text("roi\n", encoding="utf-8")
    write_region_deliverable(run_dir)

    assert _state(run_dir) == TERMINAL_NOT_A_RUN
    assert is_successful_completed_run_dir(str(run_dir))[0] is False


# 11. Current versus legacy ----------------------------------------------------


def test_positively_identified_legacy_run_still_loads(tmp_path: Path):
    run_dir = write_legacy_run(tmp_path / "legacy")

    classification = classify_run_terminal_state(str(run_dir))
    assert classification.state == TERMINAL_SUCCESS_LEGACY
    assert classification.is_success is True
    assert "earlier version" in classification.reason
    assert "cannot be verified" in classification.reason
    assert classify_completed_run_candidate(str(run_dir))[0] is True


def test_legacy_run_with_report_only_beside_analysis_outputs_loads(tmp_path: Path):
    run_dir = tmp_path / "old_legacy"
    phasic_out = run_dir / "_analysis" / "phasic_out"
    phasic_out.mkdir(parents=True)
    (phasic_out / "run_report.json").write_text(
        json.dumps({"configuration": {}, "analytical_contract": {}}), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}), encoding="utf-8"
    )
    write_region_deliverable(run_dir)

    assert _state(run_dir) == TERMINAL_SUCCESS_LEGACY


def test_status_only_directory_is_corrupt_not_legacy(tmp_path: Path):
    run_dir = tmp_path / "status_only"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}), encoding="utf-8"
    )
    write_region_deliverable(run_dir)

    assert _state(run_dir) == TERMINAL_CORRUPTED
    assert is_successful_completed_run_dir(str(run_dir))[0] is False


def test_manifest_only_directory_is_corrupt_not_legacy(tmp_path: Path):
    run_dir = tmp_path / "manifest_only"
    run_dir.mkdir()
    (run_dir / "MANIFEST.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    write_region_deliverable(run_dir)

    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_current_run_stripped_of_completion_records_is_not_legacy(tmp_path: Path):
    """A damaged current run must never be silently downgraded to legacy."""
    run_dir = write_current_run(tmp_path / "run")

    # Remove every completion marker, leaving the historical report shape behind.
    for name, key in (("run_report.json", "completion_contract"), ("status.json", COMPLETION_KEY), ("MANIFEST.json", COMPLETION_KEY)):
        _rewrite_json(run_dir / name, lambda d, k=key: d.pop(k, None))

    # The phasic report still declares the current per-ROI settings contract,
    # which positively identifies this as a run from a current build.
    phasic_report = run_dir / "_analysis" / "phasic_out" / "run_report.json"
    _rewrite_json(
        phasic_report,
        lambda d: d.__setitem__("feature_event_provenance", {"contract_version": "feature_event_provenance.v3"}),
    )

    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_failed_legacy_run_is_failed(tmp_path: Path):
    run_dir = write_legacy_run(tmp_path / "legacy_failed", status="failed")
    assert _state(run_dir) == TERMINAL_FAILED


def test_legacy_run_without_success_record_is_interrupted(tmp_path: Path):
    run_dir = write_legacy_run(tmp_path / "legacy_no_status", status=None)
    assert _state(run_dir) == TERMINAL_INTERRUPTED


# Writer-side guard ------------------------------------------------------------


def test_final_manifest_cannot_be_built_for_an_incomplete_run(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    (run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").unlink()

    run_mode = _full_run_mode()
    with pytest.raises(RunCompletionError, match="Mandatory outputs"):
        build_manifest_completion_block(
            str(run_dir), run_id=DEFAULT_RUN_ID, run_mode=run_mode,
            finalized_utc="2026-07-09T00:00:00+00:00",
        )


# Feature extraction is an execution fact, not an output-presence inference -----


def test_feature_requirement_does_not_depend_on_the_files_it_validates():
    """The classic circular bug: missing features.csv must not make itself optional."""
    ran = _full_run_mode(feature_extraction_ran=True)
    skipped = _full_run_mode(feature_extraction_ran=False, traces_only=True)

    assert "_analysis/phasic_out/features/features.csv" in required_artifacts_for_run_mode(ran)
    assert (
        "_analysis/phasic_out/features/feature_event_provenance.json"
        in required_artifacts_for_run_mode(ran)
    )
    assert not any("features/" in path for path in required_artifacts_for_run_mode(skipped))


def test_feature_extraction_expected_and_both_files_present_succeeds(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", features=True)
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT


@pytest.mark.parametrize(
    "rel_path",
    [
        "_analysis/phasic_out/features/features.csv",
        "_analysis/phasic_out/features/feature_event_provenance.json",
    ],
)
def test_feature_extraction_expected_but_output_missing_fails(tmp_path: Path, rel_path: str):
    run_dir = write_current_run(tmp_path / "run", features=True)
    (run_dir / Path(*rel_path.split("/"))).unlink()

    assert _state(run_dir) == TERMINAL_CORRUPTED
    with pytest.raises(RunCompletionError, match="Mandatory outputs"):
        build_manifest_completion_block(
            str(run_dir), run_id=DEFAULT_RUN_ID, run_mode=_full_run_mode(),
            finalized_utc="2026-07-09T00:00:00+00:00",
        )


def test_explicit_traces_only_skip_does_not_require_feature_outputs(tmp_path: Path):
    run_dir = write_current_run(
        tmp_path / "run", run_profile="tuning_prep", run_type="tuning_prep",
        traces_only=True, features=False,
    )
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT


def test_feature_extraction_ran_without_phasic_is_structurally_impossible():
    assert run_mode_structural_error(_full_run_mode(phasic_analysis=False))
    assert run_mode_structural_error(_full_run_mode(traces_only=True))


def test_manifest_claiming_an_impossible_run_mode_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    _rewrite_json(
        run_dir / "MANIFEST.json",
        lambda d: d[COMPLETION_KEY]["run_mode"].__setitem__("traces_only", True),
    )
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED


# Mandatory deliverables follow the selected production profile ----------------


def test_full_profile_requires_per_roi_deliverables_for_every_expected_roi():
    run_mode = _full_run_mode(expected_rois=["Region0", "Region1"], tonic_analysis=True)
    required = required_deliverables_for_run_mode(run_mode)

    for roi in ("Region0", "Region1"):
        assert f"{roi}/summary/phasic_correction_impact.png" in required
        assert f"{roi}/summary/tonic_overview.png" in required
        assert f"{roi}/tables/tonic_df_timeseries.csv" in required
        assert f"{roi}/summary/phasic_peak_rate_timeseries.png" in required
        assert f"{roi}/tables/phasic_auc_timeseries.csv" in required
        assert f"{roi}/day_plots/phasic_sig_iso_day_000.png" in required


def test_tuning_prep_and_continuous_profiles_promise_no_per_roi_deliverables():
    assert required_deliverables_for_run_mode(
        _full_run_mode(deliverable_profile=PROFILE_TUNING_PREP)
    ) == []
    assert required_deliverables_for_run_mode(
        _full_run_mode(deliverable_profile=PROFILE_CONTINUOUS)
    ) == []


def test_full_multi_roi_run_with_all_deliverables_succeeds(tmp_path: Path):
    run_dir = write_current_run(
        tmp_path / "run", expected_rois=["Region0", "Region1"], tonic=True,
    )
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT


def test_missing_expected_roi_directory_fails(tmp_path: Path):
    import shutil

    run_dir = write_current_run(tmp_path / "run", expected_rois=["Region0", "Region1"])
    shutil.rmtree(run_dir / "Region1")

    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_one_arbitrary_summary_folder_cannot_satisfy_a_multi_roi_run(tmp_path: Path):
    import shutil

    run_dir = write_current_run(tmp_path / "run", expected_rois=["Region0", "Region1"])
    shutil.rmtree(run_dir / "Region1")
    write_region_deliverable(run_dir, "SomeOtherRegion", "summary")

    assert _state(run_dir) == TERMINAL_CORRUPTED
    assert classify_completed_run_candidate(str(run_dir))[0] is False


@pytest.mark.parametrize(
    "rel_path",
    [
        "Region0/summary/phasic_correction_impact.png",
        "Region0/tables/phasic_auc_timeseries.csv",
        "Region0/day_plots/phasic_sig_iso_day_000.png",
    ],
)
def test_missing_mandatory_per_roi_deliverable_fails(tmp_path: Path, rel_path: str):
    run_dir = write_current_run(tmp_path / "run")
    (run_dir / Path(*rel_path.split("/"))).unlink()

    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_absent_extra_day_plot_is_accepted(tmp_path: Path):
    """Only the day-000 members are promised; later days vary with recording length."""
    run_dir = write_current_run(tmp_path / "run")
    extra = run_dir / "Region0" / "day_plots" / "phasic_sig_iso_day_007.png"
    extra.write_text("extra", encoding="utf-8")
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT

    extra.unlink()
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT


def test_deliberately_skipped_family_is_recorded_and_not_required(tmp_path: Path):
    run_dir = write_current_run(
        tmp_path / "run",
        skipped_deliverable_families=[FAMILY_PHASIC_DAY_PLOTS, FAMILY_PHASIC_TIMESERIES],
    )
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT

    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    deliverables = manifest[COMPLETION_KEY]["deliverables"]
    assert deliverables["profile"] == PROFILE_FULL_INTERMITTENT
    assert deliverables["expected_rois"] == ["Region0"]
    assert set(deliverables["intentionally_skipped_families"]) == {
        FAMILY_PHASIC_DAY_PLOTS,
        FAMILY_PHASIC_TIMESERIES,
    }
    assert not any("day_plots" in path for path in deliverables["required"])
    assert any("phasic_correction_impact" in path for path in deliverables["required"])


def test_full_run_that_analyzed_no_rois_cannot_complete():
    assert run_mode_structural_error(_full_run_mode(expected_rois=[]))


# Continuous runs are bound to the window outputs they promise -----------------


def _continuous_run_mode(**overrides):
    kwargs = dict(
        run_profile="full",
        run_type="full",
        acquisition_mode="continuous",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=True,
        feature_extraction_ran=True,
        deliverable_profile=PROFILE_CONTINUOUS,
        expected_rois=["Region0"],
        continuous_outputs_ran=True,
    )
    kwargs.update(overrides)
    return normalize_run_mode(**kwargs)


def test_continuous_required_deliverables_follow_the_executed_analyses():
    both = _continuous_run_mode()
    assert required_deliverables_for_run_mode(both) == [
        "Region0/tables/continuous_phasic_window_summary.csv",
        "Region0/tables/continuous_tonic_window_summary.csv",
    ]

    tonic_only = _continuous_run_mode(phasic_analysis=False, feature_extraction_ran=False)
    assert required_deliverables_for_run_mode(tonic_only) == [
        "Region0/tables/continuous_tonic_window_summary.csv"
    ]

    # A traces-only continuous run never writes features.csv, so
    # generate_continuous_phasic_summary has nothing to summarize.
    traces_only = _continuous_run_mode(traces_only=True, feature_extraction_ran=False)
    assert required_deliverables_for_run_mode(traces_only) == [
        "Region0/tables/continuous_tonic_window_summary.csv"
    ]


def test_continuous_deliverables_cover_every_expected_roi():
    run_mode = _continuous_run_mode(expected_rois=["Region0", "Region1"], tonic_analysis=False)
    assert required_deliverables_for_run_mode(run_mode) == [
        "Region0/tables/continuous_phasic_window_summary.csv",
        "Region1/tables/continuous_phasic_window_summary.csv",
    ]


def test_valid_continuous_run_is_successful(tmp_path: Path):
    run_dir = write_current_run(
        tmp_path / "run", acquisition_mode="continuous", tonic=True,
    )
    classification = classify_run_terminal_state(str(run_dir))
    assert classification.state == TERMINAL_SUCCESS_CURRENT, classification.reason

    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    index = manifest[COMPLETION_KEY]["deliverables"]["continuous_window_index"]
    assert index["families"][FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY]["window_row_counts"] == {
        "Region0": 3
    }


@pytest.mark.parametrize(
    "rel_path",
    [
        "Region0/tables/continuous_phasic_window_summary.csv",
        "Region0/tables/continuous_tonic_window_summary.csv",
    ],
)
def test_missing_continuous_window_summary_is_rejected(tmp_path: Path, rel_path: str):
    run_dir = write_current_run(tmp_path / "run", acquisition_mode="continuous", tonic=True)
    (run_dir / Path(*rel_path.split("/"))).unlink()

    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_continuous_run_stripped_of_all_scientist_facing_outputs_is_rejected(tmp_path: Path):
    """Internal report/config/cache intact is not a completed continuous run."""
    import shutil

    run_dir = write_current_run(tmp_path / "run", acquisition_mode="continuous", tonic=True)
    shutil.rmtree(run_dir / "Region0")

    assert (run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").is_file()
    assert (run_dir / "run_report.json").is_file()
    assert _state(run_dir) == TERMINAL_CORRUPTED
    assert is_successful_completed_run_dir(str(run_dir))[0] is False


def test_missing_continuous_index_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", acquisition_mode="continuous")
    _rewrite_json(
        run_dir / "MANIFEST.json",
        lambda d: d[COMPLETION_KEY]["deliverables"].pop("continuous_window_index"),
    )
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_continuous_index_that_omits_an_expected_roi_is_rejected(tmp_path: Path):
    run_dir = write_current_run(
        tmp_path / "run", expected_rois=["Region0", "Region1"], acquisition_mode="continuous",
    )

    def drop_roi(data):
        family = data[COMPLETION_KEY]["deliverables"]["continuous_window_index"]["families"][
            FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY
        ]
        family["relative_paths"].pop("Region1")
        family["window_row_counts"].pop("Region1")

    _rewrite_json(run_dir / "MANIFEST.json", drop_roi)
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_continuous_index_recording_zero_windows_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", acquisition_mode="continuous")

    def zero_windows(data):
        data[COMPLETION_KEY]["deliverables"]["continuous_window_index"]["families"][
            FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY
        ]["window_row_counts"]["Region0"] = 0

    _rewrite_json(run_dir / "MANIFEST.json", zero_windows)
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_malformed_continuous_index_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", acquisition_mode="continuous")
    _rewrite_json(
        run_dir / "MANIFEST.json",
        lambda d: d[COMPLETION_KEY]["deliverables"].__setitem__(
            "continuous_window_index", {"families": "not-a-mapping"}
        ),
    )
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_skipped_continuous_family_is_recorded_and_not_required(tmp_path: Path):
    run_dir = write_current_run(
        tmp_path / "run",
        acquisition_mode="continuous",
        tonic=True,
        skipped_deliverable_families=[FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY],
    )
    assert not (run_dir / "Region0" / "tables" / "continuous_tonic_window_summary.csv").exists()
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT

    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    deliverables = manifest[COMPLETION_KEY]["deliverables"]
    assert deliverables["intentionally_skipped_families"] == [
        FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY
    ]
    assert FAMILY_CONTINUOUS_TONIC_WINDOW_SUMMARY not in deliverables[
        "continuous_window_index"
    ]["families"]


def test_absent_optional_continuous_plot_is_accepted(tmp_path: Path):
    """Continuous plots skip when a column has no finite values, so they are optional."""
    run_dir = write_current_run(tmp_path / "run", acquisition_mode="continuous")
    plot = run_dir / "Region0" / "summary" / "phasic_peak_rate_timeseries.png"
    plot.parent.mkdir(parents=True, exist_ok=True)
    plot.write_text("plot", encoding="utf-8")
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT

    plot.unlink()
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT


def test_continuous_outputs_ran_outside_continuous_mode_is_impossible():
    assert run_mode_structural_error(
        _continuous_run_mode(deliverable_profile=PROFILE_FULL_INTERMITTENT)
    )


def test_continuous_run_promising_windows_with_no_rois_is_impossible():
    assert run_mode_structural_error(_continuous_run_mode(expected_rois=[]))


def test_continuous_final_manifest_cannot_be_built_without_its_window_tables(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", acquisition_mode="continuous")
    (run_dir / "Region0" / "tables" / "continuous_phasic_window_summary.csv").unlink()

    run_mode = _continuous_run_mode(tonic_analysis=False)
    with pytest.raises(RunCompletionError, match="Mandatory outputs"):
        build_manifest_completion_block(
            str(run_dir), run_id=DEFAULT_RUN_ID, run_mode=run_mode,
            finalized_utc="2026-07-09T00:00:00+00:00",
            continuous_index=build_continuous_window_index(
                str(run_dir), run_mode=run_mode,
                row_counts_by_family={FAMILY_CONTINUOUS_PHASIC_WINDOW_SUMMARY: {"Region0": 3}},
            ),
        )


def test_continuous_final_manifest_cannot_be_built_without_an_index(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", acquisition_mode="continuous")
    with pytest.raises(RunCompletionError, match="Continuous window outputs are incomplete"):
        build_manifest_completion_block(
            str(run_dir), run_id=DEFAULT_RUN_ID, run_mode=_continuous_run_mode(tonic_analysis=False),
            finalized_utc="2026-07-09T00:00:00+00:00",
        )


# Input-processing completeness binding (4J16k41 / C8) -------------------------

from photometry_pipeline.input_processing_completeness import (  # noqa: E402
    INPUT_COMPLETENESS_FILENAME,
)


def test_intermittent_run_requires_and_accepts_completeness_record(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", tonic=True)
    assert (run_dir / "_analysis" / "phasic_out" / INPUT_COMPLETENESS_FILENAME).is_file()
    assert (run_dir / "_analysis" / "tonic_out" / INPUT_COMPLETENESS_FILENAME).is_file()

    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    required = {a["relative_path"] for a in manifest[COMPLETION_KEY]["artifacts"] if a["required"]}
    assert f"_analysis/phasic_out/{INPUT_COMPLETENESS_FILENAME}" in required
    assert f"_analysis/tonic_out/{INPUT_COMPLETENESS_FILENAME}" in required
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT


def test_missing_completeness_record_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    (run_dir / "_analysis" / "phasic_out" / INPUT_COMPLETENESS_FILENAME).unlink()
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_missing_tonic_completeness_record_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", tonic=True)
    (run_dir / "_analysis" / "tonic_out" / INPUT_COMPLETENESS_FILENAME).unlink()
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_malformed_completeness_record_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    (run_dir / "_analysis" / "phasic_out" / INPUT_COMPLETENESS_FILENAME).write_text(
        "{not valid", encoding="utf-8"
    )
    # Rewrite so the manifest digest still matches the tampered file, proving the
    # reader validates content, not just the digest pin.
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_completeness_accounting_mismatch_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    rec_path = run_dir / "_analysis" / "phasic_out" / INPUT_COMPLETENESS_FILENAME
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    rec["processed"] = rec["processed"][:-1]  # processed < admitted
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_completeness_duplicate_processed_record_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    rec_path = run_dir / "_analysis" / "phasic_out" / INPUT_COMPLETENESS_FILENAME
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    rec["processed"].append(dict(rec["processed"][0]))
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED


def test_continuous_run_does_not_require_a_completeness_record(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", acquisition_mode="continuous")
    assert not (run_dir / "_analysis" / "phasic_out" / INPUT_COMPLETENESS_FILENAME).exists()
    assert _state(run_dir) == TERMINAL_SUCCESS_CURRENT


def test_legacy_run_without_completeness_record_still_loads_as_legacy(tmp_path: Path):
    # A positively-identified historical run predates the record and must remain
    # loadable as legacy, without claiming current input-completeness verification.
    run_dir = write_legacy_run(tmp_path / "legacy")
    classification = classify_run_terminal_state(str(run_dir))
    assert classification.state == TERMINAL_SUCCESS_LEGACY
    assert "cannot be verified" in classification.reason


# Completed-with-missing-sessions is a distinct terminal outcome (4J16k41c) -----

from photometry_pipeline.run_completion_contract import TERMINAL_SUCCESS_WITH_MISSING  # noqa: E402


def test_run_with_approved_missing_session_is_completed_with_missing(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", missing_session_indices=(1,))
    classification = classify_run_terminal_state(str(run_dir))

    assert classification.state == TERMINAL_SUCCESS_WITH_MISSING
    assert classification.is_success is True
    assert classification.missing_session_count == 1
    assert "missing session" in classification.reason


def test_clean_run_is_not_completed_with_missing(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run")
    assert classify_run_terminal_state(str(run_dir)).state == TERMINAL_SUCCESS_CURRENT


def test_missing_session_without_timestamp_is_rejected(tmp_path: Path):
    run_dir = write_current_run(tmp_path / "run", missing_session_indices=(1,))
    rec_path = run_dir / "_analysis" / "phasic_out" / INPUT_COMPLETENESS_FILENAME
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    for entry in rec["expected"]:
        if entry.get("disposition") == "authorized_missing_corrupted":
            entry["expected_start_time"] = ""
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    repin_status_to_manifest(run_dir)
    assert _state(run_dir) == TERMINAL_CORRUPTED
