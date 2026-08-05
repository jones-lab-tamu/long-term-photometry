"""End-to-end recovery of the continuous demo's known tonic component."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gui.synthetic_demo_generator import (
    GUIDED_CONTINUOUS_DEMO_FILE_NAME,
    GUIDED_CONTINUOUS_DEMO_TONIC_TRUTH_FILENAME,
    generate_guided_continuous_demo,
)
from photometry_pipeline import guided_continuous_rwd_block_plan as block_subject
from photometry_pipeline import guided_continuous_rwd_correction_segments as c4a
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    evaluate_continuous_rwd_timestamp_continuity,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    build_guided_continuous_rwd_review_binding,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    build_guided_continuous_rwd_target_grid,
)
from photometry_pipeline.guided_continuous_rwd_tonic_run import (
    TONIC_METHOD_GLOBAL_ISOSBESTIC,
    TONIC_UNITS_FRACTIONAL,
    execute_guided_continuous_rwd_tonic_run,
)
from photometry_pipeline.guided_execution_payloads import (
    build_guided_execution_startup_mapping_contract,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedNewAnalysisExecutionIntent,
    GuidedPlanCorrectionChoice,
)
from photometry_pipeline.io.csv_continuous_source import (
    ContinuousCsvRoiSelection,
    inspect_continuous_csv_recording,
)
from photometry_pipeline.io.rwd_continuous_projection_reader import (
    iter_project_guided_continuous_rwd_blocks,
)


def _choices(strategies: dict[str, str]) -> list[GuidedPlanCorrectionChoice]:
    return [
        GuidedPlanCorrectionChoice(
            roi_id=roi_id,
            selected_strategy=strategy,
            source_type="local_correction_preview",
            current_or_stale="current",
            explicit_user_mark=True,
            evidence_reference={"strategy": strategy, "roi": roi_id},
        )
        for roi_id, strategy in strategies.items()
    ]


def _build_inputs(folder: Path):
    source = folder / GUIDED_CONTINUOUS_DEMO_FILE_NAME
    inspection = inspect_continuous_csv_recording(
        source,
        time_column="ElapsedSeconds",
        time_unit="seconds",
        roi_selections=[
            ContinuousCsvRoiSelection("ROI1", "ROI1_Signal", "ROI1_Reference"),
            ContinuousCsvRoiSelection("ROI2", "ROI2_Signal", "ROI2_Reference"),
        ],
    )
    assert inspection.status == "completed", inspection.outcome_category
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("ROI1", "ROI2")
    )
    continuity = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=source
    )
    assert continuity.outcome == "continuity_passed"
    strategies = {"ROI1": "global_linear_regression", "ROI2": "signal_only_f0"}
    draft = GuidedNewAnalysisDraftPlan(
        input_source_path=str(folder),
        resolved_input_source_path=str(folder),
        input_format="custom_tabular",
        acquisition_mode="continuous",
        continuous_window_sec=600.0,
        continuous_step_sec=600.0,
        discovered_roi_ids=["ROI1", "ROI2"],
        included_roi_ids=["ROI1", "ROI2"],
        excluded_roi_ids=[],
        output_base_path=str(folder / "output"),
        global_correction_strategy=next(iter(strategies.values())),
        per_roi_correction_strategy_choices=_choices(strategies),
        feature_event_profile_id="default",
        feature_event_values={},
        execution_intent=GuidedNewAnalysisExecutionIntent(
            recording_start_clock="00:00:00",
            recording_start_clock_source="user_entered",
        ),
    )
    binding = build_guided_continuous_rwd_review_binding(
        draft,
        recording=recording,
        continuity_evaluation=continuity,
        current_source_path=source,
    )
    grid = build_guided_continuous_rwd_target_grid(recording, continuity)
    contract = build_guided_execution_startup_mapping_contract()
    block_plan = block_subject.build_guided_continuous_rwd_block_plan(grid)
    segment_plan = c4a.build_guided_continuous_rwd_correction_segment_plan(
        binding, grid, accepted_draft=draft, startup_mapping_contract=contract
    )
    dynamic_f0 = c4a.prepare_guided_continuous_rwd_dynamic_f0_authority(
        binding,
        grid,
        block_plan,
        segment_plan,
        iter_project_guided_continuous_rwd_blocks(binding, grid, block_plan),
        accepted_draft=draft,
        startup_mapping_contract=contract,
    )
    return binding, grid, draft, contract, block_plan, segment_plan, dynamic_f0


@pytest.fixture(scope="module")
def full_continuous_demo(tmp_path_factory):
    result = generate_guided_continuous_demo(tmp_path_factory.mktemp("full_demo"))
    assert result.success, result.message
    truth = json.loads(
        (
            result.input_dir / GUIDED_CONTINUOUS_DEMO_TONIC_TRUTH_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert truth["duration_hours"] == pytest.approx(48.0)
    return result.input_dir, truth, _build_inputs(result.input_dir)


def _daily_harmonic_fit(
    hours: np.ndarray, values: np.ndarray
) -> tuple[float, float, float, float]:
    """Estimate period/phase only for validating the generated truth recovery."""
    periods = np.linspace(20.0, 28.0, 161)
    best = None
    for period in periods:
        angle = 2.0 * np.pi * hours / period
        design = np.column_stack((np.ones(hours.size), np.cos(angle), np.sin(angle)))
        coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
            design, values, rcond=None
        )
        residual = float(np.mean((design @ coefficients - values) ** 2))
        if best is None or residual < best[0]:
            best = residual, period, coefficients
    _residual, period, coefficients = best
    cosine, sine = float(coefficients[1]), float(coefficients[2])
    phase = (np.arctan2(sine, cosine) * period / (2.0 * np.pi)) % period
    amplitude = float(np.hypot(cosine, sine))
    return float(coefficients[0]), amplitude, float(phase), float(period)


def test_full_continuous_demo_native_tonic_recovers_truth(
    full_continuous_demo, tmp_path
):
    _folder, truth, inputs = full_continuous_demo
    binding, grid, draft, contract, block_plan, segment_plan, dynamic_f0 = inputs

    from photometry_pipeline.guided_continuous_rwd_segment_correction import (
        _resolve_segment_correction_settings,
    )

    config, _identity = _resolve_segment_correction_settings(contract)
    result = execute_guided_continuous_rwd_tonic_run(
        binding,
        grid,
        block_plan,
        segment_plan,
        dynamic_f0,
        accepted_draft=draft,
        startup_mapping_contract=contract,
        output_base=str(tmp_path),
        config=config,
    )

    assert result.terminal_state == "success_current"
    for roi in ("ROI1", "ROI2"):
        frame = pd.read_csv(
            Path(result.run_dir)
            / roi
            / "tables"
            / "continuous_tonic_window_summary.csv"
        ).sort_values("window_index")
        assert set(frame["tonic_method"]) == {TONIC_METHOD_GLOBAL_ISOSBESTIC}
        assert set(frame["units"]) == {TONIC_UNITS_FRACTIONAL}
        assert set(frame["tonic_fallback"]) == {False}

        hours = frame["elapsed_hour_mid"].to_numpy(dtype=float)
        recovered = frame["tonic_value"].to_numpy(dtype=float)
        roi_truth = next(item for item in truth["rois"] if item["roi_id"] == roi)
        expected = roi_truth["tonic_offset_au"] + roi_truth[
            "tonic_amplitude_au"
        ] * np.cos(
            2.0
            * np.pi
            * (hours - roi_truth["tonic_peak_phase_hours"])
            / roi_truth["tonic_period_hours"]
        )
        assert np.all(np.isfinite(recovered))
        assert recovered.min() < 0.0 < recovered.max()
        assert float(np.corrcoef(recovered, expected)[0, 1]) > 0.8

        _offset, _amplitude, phase, period = _daily_harmonic_fit(hours, recovered)
        assert period == pytest.approx(24.0, abs=1.0)
        assert phase == pytest.approx(7.0, abs=1.0)
        fit_hours = np.arange(0.0, 48.0, 0.01)
        fitted = _offset + _amplitude * np.cos(
            2.0 * np.pi * (fit_hours - phase) / period
        )
        first_day = np.flatnonzero((fit_hours >= 0.0) & (fit_hours < 24.0))
        second_day = np.flatnonzero((fit_hours >= 24.0) & (fit_hours < 48.0))
        first_peak = fit_hours[first_day[np.argmax(fitted[first_day])]]
        first_trough = fit_hours[first_day[np.argmin(fitted[first_day])]]
        second_peak = fit_hours[second_day[np.argmax(fitted[second_day])]]
        second_trough = fit_hours[second_day[np.argmin(fitted[second_day])]]
        assert first_peak == pytest.approx(7.0, abs=1.5)
        assert first_trough == pytest.approx(19.0, abs=1.5)
        assert second_peak == pytest.approx(31.0, abs=1.5)
        assert second_trough == pytest.approx(43.0, abs=1.5)
