from __future__ import annotations

import dataclasses
import json
import os

import pandas as pd
import pytest

from photometry_pipeline import completed_continuous_rwd_review as subject
from photometry_pipeline import guided_continuous_rwd_correction_pass as c4c_module
from photometry_pipeline import guided_continuous_rwd_phasic_detection as detection_module
from photometry_pipeline.core import feature_extraction as feature_extraction_module
from photometry_pipeline.guided_continuous_rwd_combined_run import (
    execute_guided_continuous_rwd_combined_run,
)
from photometry_pipeline.guided_continuous_rwd_correction_run import (
    execute_guided_continuous_rwd_correction_run,
)
from photometry_pipeline.guided_continuous_rwd_phasic_run import (
    execute_guided_continuous_rwd_phasic_run,
)
from photometry_pipeline.guided_continuous_rwd_tonic_run import (
    execute_guided_continuous_rwd_tonic_run,
)
from photometry_pipeline.io.hdf5_cache_reader import list_cache_rois
from photometry_pipeline.run_completion_contract import (
    TERMINAL_SUCCESS_CURRENT,
    classify_run_terminal_state,
)

from tests.test_guided_continuous_rwd_correction_pass_persistence import (
    _build_case,
    _pass_inputs,
)


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_e1") / "recording"
    return _build_case(folder, continuous_window_sec=90.0)


@pytest.fixture(scope="module")
def real_config(accepted_case):
    from photometry_pipeline.guided_continuous_rwd_segment_correction import (
        _resolve_segment_correction_settings,
    )

    _binding, _grid, _draft, contract, _source = accepted_case
    config, _identity = _resolve_segment_correction_settings(contract)
    return dataclasses.replace(
        config,
        peak_threshold_method="percentile",
        peak_threshold_percentile=50.0,
        peak_min_distance_sec=1.0,
    )


def _kwargs(inputs, real_config, output_base):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    return dict(
        review_binding=binding,
        target_grid=grid,
        block_plan=block_plan,
        segment_plan=segment_plan,
        dynamic_f0_authority=f0,
        accepted_draft=draft,
        startup_mapping_contract=contract,
        output_base=str(output_base),
        config=real_config,
    )


def _positional_call(func, inputs, real_config, output_base):
    kwargs = _kwargs(inputs, real_config, output_base)
    return func(
        kwargs.pop("review_binding"),
        kwargs.pop("target_grid"),
        kwargs.pop("block_plan"),
        kwargs.pop("segment_plan"),
        kwargs.pop("dynamic_f0_authority"),
        **kwargs,
    )


@pytest.fixture(scope="module")
def correction_only_run(accepted_case, real_config, tmp_path_factory):
    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e1_correction")
    return _positional_call(
        execute_guided_continuous_rwd_correction_run, inputs, real_config, output_base
    )


@pytest.fixture(scope="module")
def tonic_only_run(accepted_case, real_config, tmp_path_factory):
    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e1_tonic")
    return _positional_call(
        execute_guided_continuous_rwd_tonic_run, inputs, real_config, output_base
    )


@pytest.fixture(scope="module")
def phasic_only_run(accepted_case, real_config, tmp_path_factory):
    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e1_phasic")
    return _positional_call(
        execute_guided_continuous_rwd_phasic_run, inputs, real_config, output_base
    )


@pytest.fixture(scope="module")
def combined_run(accepted_case, real_config, tmp_path_factory):
    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e1_combined")
    return _positional_call(
        execute_guided_continuous_rwd_combined_run, inputs, real_config, output_base
    )


@pytest.fixture(scope="module")
def included_roi_ids(accepted_case):
    binding = accepted_case[0]
    return tuple(binding.recording.roi.included_roi_ids)


# ---------------------------------------------------------------------------
# Section 22: completed-run recognition, all four run modes
# ---------------------------------------------------------------------------


def test_correction_only_is_recognized_as_current_continuous(
    correction_only_run, included_roi_ids
):
    assert correction_only_run.terminal_state == TERMINAL_SUCCESS_CURRENT
    overview = subject.load_continuous_run_overview(correction_only_run.run_dir)
    assert overview.acquisition_mode == "continuous"
    assert overview.included_roi_ids == included_roi_ids
    assert overview.tonic_analysis is False
    assert overview.phasic_analysis is False
    assert overview.correction_completed is True


def test_tonic_only_is_recognized_as_current_continuous(tonic_only_run, included_roi_ids):
    assert tonic_only_run.terminal_state == TERMINAL_SUCCESS_CURRENT
    overview = subject.load_continuous_run_overview(tonic_only_run.run_dir)
    assert overview.acquisition_mode == "continuous"
    assert overview.included_roi_ids == included_roi_ids
    assert overview.tonic_analysis is True
    assert overview.phasic_analysis is False


def test_phasic_only_is_recognized_as_current_continuous(phasic_only_run, included_roi_ids):
    assert phasic_only_run.terminal_state == TERMINAL_SUCCESS_CURRENT
    overview = subject.load_continuous_run_overview(phasic_only_run.run_dir)
    assert overview.acquisition_mode == "continuous"
    assert overview.included_roi_ids == included_roi_ids
    assert overview.tonic_analysis is False
    assert overview.phasic_analysis is True
    assert overview.feature_extraction_ran is True


def test_combined_is_recognized_as_current_continuous(combined_run, included_roi_ids):
    assert combined_run.terminal_state == TERMINAL_SUCCESS_CURRENT
    overview = subject.load_continuous_run_overview(combined_run.run_dir)
    assert overview.acquisition_mode == "continuous"
    assert overview.included_roi_ids == included_roi_ids
    assert overview.tonic_analysis is True
    assert overview.phasic_analysis is True


def test_overview_never_represents_storage_windows_as_sessions(combined_run):
    """The model has no session concept at all -- one recording, not N
    sessions/windows-as-sessions."""
    overview = subject.load_continuous_run_overview(combined_run.run_dir)
    field_names = {f.name for f in dataclasses.fields(overview)}
    assert not any("session" in name.lower() for name in field_names)
    # The window COUNT is a plain int describing storage/reporting windows,
    # never a per-window "session" record list.
    assert isinstance(overview.corrected_segment_count, int)


# ---------------------------------------------------------------------------
# Section 23: tonic-only reopening
# ---------------------------------------------------------------------------


def test_tonic_only_summaries_and_trace_open_for_every_roi(tonic_only_run, included_roi_ids):
    overview = subject.load_continuous_run_overview(tonic_only_run.run_dir)
    for roi_id in included_roi_ids:
        trace = subject.load_continuous_roi_trace(
            tonic_only_run.run_dir, family="tonic", roi_id=roi_id
        )
        assert trace.time_sec.shape[0] == overview.target_sample_count
        assert trace.primary_trace.shape[0] == overview.target_sample_count

        summary = subject.load_continuous_window_summary(
            tonic_only_run.run_dir, family="tonic", roi_id=roi_id
        )
        assert len(summary) == overview.tonic_window_row_counts[roi_id]
        # Displayed values must match the persisted CSV exactly (no
        # re-derivation).
        raw = pd.read_csv(
            os.path.join(overview.run_dir, overview.tonic_summary_relative_paths[roi_id])
        )
        pd.testing.assert_frame_equal(summary, raw)


def test_tonic_only_final_short_window_present(tonic_only_run, included_roi_ids):
    overview = subject.load_continuous_run_overview(tonic_only_run.run_dir)
    assert overview.final_window is not None
    assert overview.final_window.duration_sec > 0
    roi_id = included_roi_ids[0]
    summary = subject.load_continuous_window_summary(
        tonic_only_run.run_dir, family="tonic", roi_id=roi_id
    ).sort_values("window_index")
    assert summary.iloc[-1]["window_duration_sec"] < summary.iloc[0]["window_duration_sec"]
    assert summary.iloc[-1]["window_duration_sec"] == pytest.approx(overview.final_window.duration_sec)


def test_tonic_only_has_no_phasic_model_entries(tonic_only_run):
    overview = subject.load_continuous_run_overview(tonic_only_run.run_dir)
    assert overview.phasic_cache_relative_path is None
    assert overview.features_relative_path is None
    assert overview.events_relative_path is None
    assert overview.phasic_window_row_counts == {}
    assert overview.phasic_event_counts_by_roi == {}
    assert overview.phasic_event_total == 0
    with pytest.raises(subject.CompletedContinuousRwdReviewError):
        subject.load_continuous_phasic_events(tonic_only_run.run_dir)


# ---------------------------------------------------------------------------
# Section 24: phasic-only reopening
# ---------------------------------------------------------------------------


def test_phasic_only_cache_features_and_events_open(phasic_only_run, included_roi_ids):
    overview = subject.load_continuous_run_overview(phasic_only_run.run_dir)
    assert overview.phasic_event_total > 0  # tuned threshold must actually find events

    for roi_id in included_roi_ids:
        trace = subject.load_continuous_roi_trace(
            phasic_only_run.run_dir, family="phasic", roi_id=roi_id
        )
        assert trace.time_sec.shape[0] == overview.target_sample_count

    features = pd.read_csv(
        os.path.join(overview.run_dir, overview.features_relative_path)
    )
    assert len(features) > 0

    events = subject.load_continuous_phasic_events(phasic_only_run.run_dir)
    assert len(events) == overview.phasic_event_total
    raw_events = pd.read_csv(os.path.join(overview.run_dir, overview.events_relative_path))
    pd.testing.assert_frame_equal(events.reset_index(drop=True), raw_events.reset_index(drop=True))


def test_phasic_only_event_totals_and_summary_conservation(phasic_only_run, included_roi_ids):
    overview = subject.load_continuous_run_overview(phasic_only_run.run_dir)
    for roi_id in included_roi_ids:
        roi_events = subject.load_continuous_phasic_events(phasic_only_run.run_dir, roi_id=roi_id)
        assert len(roi_events) == overview.phasic_event_counts_by_roi[roi_id]

        raw_events = pd.read_csv(
            os.path.join(overview.run_dir, overview.events_relative_path)
        )
        raw_roi_events = raw_events[raw_events["roi"] == roi_id].reset_index(drop=True)
        pd.testing.assert_series_equal(
            roi_events["global_time_sec"].reset_index(drop=True),
            raw_roi_events["global_time_sec"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            roi_events["polarity"].reset_index(drop=True).astype(raw_roi_events["polarity"].dtype),
            raw_roi_events["polarity"].reset_index(drop=True),
        )

        summary = subject.load_continuous_window_summary(
            phasic_only_run.run_dir, family="phasic", roi_id=roi_id
        )
        assert int(summary["event_count"].sum()) == overview.phasic_event_counts_by_roi[roi_id]


def test_phasic_only_reopening_never_invokes_the_event_detector(phasic_only_run, monkeypatch):
    """Guards the actual canonical detector symbol -- this module never
    imports it, proving nothing in the reopening path regenerates events."""
    calls = {"count": 0}
    real_fn = feature_extraction_module.get_peak_indices_for_trace

    def guarded(*args, **kwargs):
        calls["count"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(feature_extraction_module, "get_peak_indices_for_trace", guarded)

    overview = subject.load_continuous_run_overview(phasic_only_run.run_dir)
    subject.load_continuous_phasic_events(phasic_only_run.run_dir)
    for roi_id in overview.included_roi_ids:
        subject.load_continuous_roi_trace(phasic_only_run.run_dir, family="phasic", roi_id=roi_id)
        subject.load_continuous_window_summary(
            phasic_only_run.run_dir, family="phasic", roi_id=roi_id
        )

    assert calls["count"] == 0


def test_phasic_only_has_no_tonic_model_entries(phasic_only_run):
    overview = subject.load_continuous_run_overview(phasic_only_run.run_dir)
    assert overview.tonic_cache_relative_path is None
    assert overview.tonic_window_row_counts == {}
    assert overview.tonic_summary_relative_paths == {}
    with pytest.raises(subject.CompletedContinuousRwdReviewError):
        subject.load_continuous_roi_trace(
            phasic_only_run.run_dir, family="tonic", roi_id=overview.included_roi_ids[0]
        )


# ---------------------------------------------------------------------------
# Section 25: combined reopening
# ---------------------------------------------------------------------------


def test_combined_both_families_available_with_matching_coverage(combined_run, included_roi_ids):
    overview = subject.load_continuous_run_overview(combined_run.run_dir)
    assert overview.tonic_analysis and overview.phasic_analysis

    tonic_cache = os.path.join(overview.run_dir, overview.tonic_cache_relative_path)
    phasic_cache = os.path.join(overview.run_dir, overview.phasic_cache_relative_path)
    from photometry_pipeline.io.hdf5_cache_reader import open_phasic_cache, open_tonic_cache

    with open_tonic_cache(tonic_cache) as tc, open_phasic_cache(phasic_cache) as pc:
        assert list_cache_rois(tc) == list_cache_rois(pc) == list(included_roi_ids)

    for roi_id in included_roi_ids:
        assert overview.tonic_window_row_counts[roi_id] == overview.phasic_window_row_counts[roi_id]

    events = subject.load_continuous_phasic_events(combined_run.run_dir)
    assert len(events) == overview.phasic_event_total
    assert overview.phasic_event_total == sum(overview.phasic_event_counts_by_roi.values())


def test_combined_final_short_window_in_both_summary_families(combined_run, included_roi_ids):
    roi_id = included_roi_ids[0]
    tonic_summary = subject.load_continuous_window_summary(
        combined_run.run_dir, family="tonic", roi_id=roi_id
    ).sort_values("window_index")
    phasic_summary = subject.load_continuous_window_summary(
        combined_run.run_dir, family="phasic", roi_id=roi_id
    ).sort_values("window_index")
    assert tonic_summary.iloc[-1]["window_duration_sec"] < tonic_summary.iloc[0]["window_duration_sec"]
    assert phasic_summary.iloc[-1]["window_duration_sec"] < phasic_summary.iloc[0]["window_duration_sec"]
    assert tonic_summary.iloc[-1]["window_index"] == phasic_summary.iloc[-1]["window_index"]


def test_combined_reopening_never_reruns_correction_or_detection(combined_run, monkeypatch):
    def flaky_traversal(*args, **kwargs):
        raise AssertionError("reopening must never rerun correction")

    def flaky_detection(*args, **kwargs):
        raise AssertionError("reopening must never rerun D3b-A detection")

    monkeypatch.setattr(
        c4c_module, "iterate_guided_continuous_rwd_corrected_segments", flaky_traversal
    )
    monkeypatch.setattr(
        detection_module, "detect_guided_continuous_rwd_phasic_features", flaky_detection
    )

    overview = subject.load_continuous_run_overview(combined_run.run_dir)
    assert overview.terminal_state == TERMINAL_SUCCESS_CURRENT
    subject.load_continuous_phasic_events(combined_run.run_dir)
    for roi_id in overview.included_roi_ids:
        subject.load_continuous_roi_trace(combined_run.run_dir, family="tonic", roi_id=roi_id)
        subject.load_continuous_roi_trace(combined_run.run_dir, family="phasic", roi_id=roi_id)


# ---------------------------------------------------------------------------
# Section 26: correction-only reopening
# ---------------------------------------------------------------------------


def test_correction_only_truthful_and_no_fabricated_tabs(correction_only_run, included_roi_ids):
    overview = subject.load_continuous_run_overview(correction_only_run.run_dir)
    assert overview.terminal_state == TERMINAL_SUCCESS_CURRENT
    assert overview.included_roi_ids == included_roi_ids
    assert overview.correction_completed is True
    assert overview.tonic_analysis is False
    assert overview.phasic_analysis is False
    assert overview.tonic_summary_relative_paths == {}
    assert overview.phasic_summary_relative_paths == {}
    assert overview.final_window is not None  # from the corrected cache itself
    # No downstream _analysis directories exist at all for this run type.
    assert not os.path.isdir(os.path.join(overview.run_dir, "_analysis", "tonic_out"))
    assert not os.path.isdir(os.path.join(overview.run_dir, "_analysis", "phasic_out"))


# ---------------------------------------------------------------------------
# Section 27: malformed completed artifacts
# ---------------------------------------------------------------------------


def test_missing_tonic_cache_refuses_with_clear_error(tonic_only_run, tmp_path_factory, monkeypatch):
    import shutil

    broken = tmp_path_factory.mktemp("cr1_e1_broken_tonic")
    shutil.copytree(tonic_only_run.run_dir, str(broken), dirs_exist_ok=True)
    os.remove(os.path.join(str(broken), "_analysis", "tonic_out", "tonic_trace_cache.h5"))

    # The completion classifier itself is the first authority to catch a
    # missing mandatory artifact (see run_completion_contract.py); this
    # loader's own "tonic trace file is missing" message only fires for
    # cases the classifier's own required-artifact check does not cover.
    with pytest.raises(subject.CompletedContinuousRwdReviewError, match="tonic_trace_cache"):
        subject.load_continuous_run_overview(str(broken))


def test_missing_event_csv_refuses_without_redetecting(
    phasic_only_run, tmp_path_factory, monkeypatch
):
    import shutil

    broken = tmp_path_factory.mktemp("cr1_e1_broken_events")
    shutil.copytree(phasic_only_run.run_dir, str(broken), dirs_exist_ok=True)
    os.remove(
        os.path.join(
            str(broken), "_analysis", "phasic_out", "features", "continuous_phasic_events.csv"
        )
    )

    calls = {"count": 0}
    real_fn = feature_extraction_module.get_peak_indices_for_trace

    def guarded(*args, **kwargs):
        calls["count"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(feature_extraction_module, "get_peak_indices_for_trace", guarded)

    with pytest.raises(subject.CompletedContinuousRwdReviewError, match="saved event results"):
        subject.load_continuous_run_overview(str(broken))
    assert calls["count"] == 0


def test_invalid_event_row_is_rejected(phasic_only_run, tmp_path_factory):
    import shutil

    broken = tmp_path_factory.mktemp("cr1_e1_broken_event_row")
    shutil.copytree(phasic_only_run.run_dir, str(broken), dirs_exist_ok=True)
    events_path = os.path.join(
        str(broken), "_analysis", "phasic_out", "features", "continuous_phasic_events.csv"
    )
    df = pd.read_csv(events_path)
    df.loc[0, "polarity"] = 0  # invalid: only +1/-1 are accepted
    df.to_csv(events_path, index=False)

    with pytest.raises(subject.CompletedContinuousRwdReviewError, match="polarity"):
        subject.load_continuous_run_overview(str(broken))


def test_invalid_event_time_outside_support_is_rejected(phasic_only_run, tmp_path_factory):
    import shutil

    broken = tmp_path_factory.mktemp("cr1_e1_broken_event_order")
    shutil.copytree(phasic_only_run.run_dir, str(broken), dirs_exist_ok=True)
    events_path = os.path.join(
        str(broken), "_analysis", "phasic_out", "features", "continuous_phasic_events.csv"
    )
    df = pd.read_csv(events_path)
    # Push one event's time far beyond the recording's own true support --
    # must be caught without needing ground truth from a rerun detection.
    df.loc[0, "global_time_sec"] = float(df["global_time_sec"].max()) + 1_000_000.0
    df.to_csv(events_path, index=False)

    with pytest.raises(subject.CompletedContinuousRwdReviewError, match="time support"):
        subject.load_continuous_run_overview(str(broken))


def test_invalid_event_chronological_order_is_rejected(phasic_only_run, tmp_path_factory):
    import shutil

    broken = tmp_path_factory.mktemp("cr1_e1_broken_event_chronology")
    shutil.copytree(phasic_only_run.run_dir, str(broken), dirs_exist_ok=True)
    events_path = os.path.join(
        str(broken), "_analysis", "phasic_out", "features", "continuous_phasic_events.csv"
    )
    df = pd.read_csv(events_path)
    roi_with_multiple_events = df["roi"].value_counts()
    roi_with_multiple_events = roi_with_multiple_events[roi_with_multiple_events >= 2].index
    assert len(roi_with_multiple_events) >= 1, "fixture must produce >=2 events for some ROI"
    roi_id = roi_with_multiple_events[0]
    roi_rows = df.index[df["roi"] == roi_id].tolist()
    # Swap the first two rows of that ROI's block to break chronological
    # order while every time value stays within valid support.
    a, b = roi_rows[0], roi_rows[1]
    df.loc[a, "global_time_sec"], df.loc[b, "global_time_sec"] = (
        df.loc[b, "global_time_sec"],
        df.loc[a, "global_time_sec"],
    )
    df.to_csv(events_path, index=False)

    with pytest.raises(subject.CompletedContinuousRwdReviewError, match="chronological order"):
        subject.load_continuous_run_overview(str(broken))


def test_stale_or_invalid_provenance_refuses(phasic_only_run, tmp_path_factory):
    import shutil

    broken = tmp_path_factory.mktemp("cr1_e1_broken_provenance")
    shutil.copytree(phasic_only_run.run_dir, str(broken), dirs_exist_ok=True)
    provenance_path = os.path.join(
        str(broken), "_analysis", "phasic_out", "features", "feature_event_provenance.json"
    )
    os.remove(provenance_path)

    with pytest.raises(subject.CompletedContinuousRwdReviewError):
        subject.load_continuous_run_overview(str(broken))


def test_failed_run_is_never_opened_as_completed(tmp_path):
    run_dir = tmp_path / "not_a_real_run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "bogus",
                "phase": "final",
                "status": "error",
                "errors": ["simulated failure"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success
    with pytest.raises(subject.CompletedContinuousRwdReviewError):
        subject.load_continuous_run_overview(str(run_dir))
