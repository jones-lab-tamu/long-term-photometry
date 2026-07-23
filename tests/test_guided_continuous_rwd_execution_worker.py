"""CR1-E2: focused tests for the internal continuous-RWD execution bridge.

Reuses the exact same accepted-authority fixtures as the CR1-D2/D3a/D3b/D4
backend tests (`tests/test_guided_continuous_rwd_correction_pass_persistence.
py`'s `_build_case`/`_pass_inputs`) -- the same objects the accepted
`execute_guided_continuous_rwd_*_run` entry points already require -- rather
than hand-building shallow mocks for every authority.

Continuous Guided Run remains hidden/disabled: nothing here enables the Run
button or wires a real GUI trigger. These tests exercise
`gui.main_window._execute_guided_continuous_rwd` /
`_GuidedRunExecutionWorker` directly, exactly as the worker's own `run()`
would call them off the GUI thread.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

import gui.main_window as main_window_module
from gui.main_window import (
    _GuidedContinuousRwdExecutionRequest,
    _GuidedRunExecutionWorker,
    _execute_guided_continuous_rwd,
    _guided_continuous_rwd_analysis_selection,
    _select_guided_continuous_rwd_backend,
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
from photometry_pipeline.guided_execution_payloads import (
    build_guided_execution_startup_mapping_contract,
)
from photometry_pipeline.guided_new_analysis_plan import (
    GuidedNewAnalysisDraftPlan,
    GuidedNewAnalysisExecutionIntent,
)
from photometry_pipeline.io.rwd_continuous_projection_reader import (
    iter_project_guided_continuous_rwd_blocks,
)
from photometry_pipeline.run_completion_contract import (
    TERMINAL_SUCCESS_CURRENT,
    classify_run_terminal_state,
)

from tests.test_guided_continuous_rwd_correction_pass_persistence import (
    _build_case,
    _choices,
    _pass_inputs,
)


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_e2") / "recording"
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


def _build_case_for_mode(
    folder,
    *,
    execution_mode: str,
    feature_confirmed: bool = False,
    input_format: str = "rwd",
    acquisition_mode: str = "continuous",
    continuous_window_sec: float = 90.0,
):
    """Build one full accepted-authority chain (review binding, target grid,
    block plan, segment plan, dynamic-F0 authority) from a draft that
    already carries the requested analysis selection from the moment it is
    constructed.

    `accepted_guided_plan_identity` (baked into the review binding and
    re-validated by the correction traversal) is computed from the draft's
    own fields, including `execution_intent` -- swapping `execution_intent`
    on a copy *after* the binding/segment plan were already built against a
    different draft breaks that identity and is rejected by the backend's
    own authority-consistency check. Every distinct execution_mode/
    feature_confirmed combination therefore needs its own freshly-built
    chain, exactly mirroring
    ``tests.test_guided_continuous_rwd_correction_pass_persistence.
    _build_case``.
    """
    import numpy as np

    from photometry_pipeline.io.rwd_continuous_source import (
        inspect_continuous_rwd_acquisition_folder,
    )
    from tests.test_guided_continuous_rwd_correction_pass_persistence import _values

    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "Fluorescence.csv"
    lines = ["Time(s),ROI1-410,ROI1-470,ROI2-410,ROI2-470\n"]
    for index in range(6001):
        time, control, signal = _values(np.array([index], dtype=float))
        lines.append(
            f"{time[0]:.1f},{control[0,0]:.12f},{signal[0,0]:.12f},"
            f"{control[0,1]:.12f},{signal[0,1]:.12f}\n"
        )
    source.write_text("".join(lines), encoding="utf-8", newline="")
    inspection = inspect_continuous_rwd_acquisition_folder(folder)
    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=("ROI1", "ROI2")
    )
    continuity = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=source
    )
    strategies = {"ROI1": "global_linear_regression", "ROI2": "signal_only_f0"}
    draft_kwargs = dict(
        input_source_path=str(folder),
        resolved_input_source_path=str(folder),
        input_format=input_format,
        acquisition_mode=acquisition_mode,
        continuous_window_sec=continuous_window_sec,
        continuous_step_sec=continuous_window_sec,
        discovered_roi_ids=["ROI1", "ROI2"],
        included_roi_ids=["ROI1", "ROI2"],
        excluded_roi_ids=[],
        output_base_path=str(folder / "output"),
        global_correction_strategy=next(iter(strategies.values())),
        per_roi_correction_strategy_choices=_choices(strategies),
        feature_event_profile_id="default",
        feature_event_values={},
        execution_intent=GuidedNewAnalysisExecutionIntent(execution_mode=execution_mode),
    )
    if feature_confirmed:
        draft_kwargs["feature_event_profile_status"] = "default_initialized"
    draft = GuidedNewAnalysisDraftPlan(**draft_kwargs)

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
    f0 = c4a.prepare_guided_continuous_rwd_dynamic_f0_authority(
        binding,
        grid,
        block_plan,
        segment_plan,
        iter_project_guided_continuous_rwd_blocks(binding, grid, block_plan),
        accepted_draft=draft,
        startup_mapping_contract=contract,
    )
    return binding, grid, draft, contract, block_plan, segment_plan, f0, str(source)


def _mode_draft(draft, *, execution_mode: str, feature_confirmed: bool = False):
    """A draft carrying one requested analysis selection, matching what the
    accepted Guided plan already carries via execution_intent -- never a
    hand-invented field.

    Only safe for tests that never reach a real backend call (validation
    refuses first): swapping execution_intent on a copy invalidates the
    accepted_guided_plan_identity baked into an already-built review binding
    / segment plan (see `_build_case_for_mode`)."""
    updated = dataclasses.replace(
        draft,
        execution_intent=GuidedNewAnalysisExecutionIntent(execution_mode=execution_mode),
    )
    if feature_confirmed:
        updated = dataclasses.replace(
            updated,
            feature_event_profile_status="default_initialized",
        )
    return updated


def _execution_request(
    inputs,
    config,
    output_base,
    *,
    execution_mode: str,
    feature_confirmed: bool = False,
    cancellation_requested=None,
    draft_override=None,
):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    plan = draft_override if draft_override is not None else _mode_draft(
        draft, execution_mode=execution_mode, feature_confirmed=feature_confirmed
    )
    return _GuidedContinuousRwdExecutionRequest(
        review_binding=binding,
        target_grid=grid,
        block_plan=block_plan,
        segment_plan=segment_plan,
        dynamic_f0_authority=f0,
        accepted_draft=plan,
        startup_mapping_contract=contract,
        output_base=str(output_base),
        config=config,
        cancellation_requested=cancellation_requested,
    )


# ---------------------------------------------------------------------------
# Four-mode routing
# ---------------------------------------------------------------------------


def _guard(monkeypatch, module_path: str, attr: str, calls: dict, key: str):
    import importlib

    module = importlib.import_module(module_path)
    real_fn = getattr(module, attr)

    def guarded(*args, **kwargs):
        calls[key] = calls.get(key, 0) + 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(module, attr, guarded)


def _guard_all_backends(monkeypatch):
    calls: dict[str, int] = {}
    _guard(
        monkeypatch,
        "photometry_pipeline.guided_continuous_rwd_correction_run",
        "execute_guided_continuous_rwd_correction_run",
        calls,
        "correction",
    )
    _guard(
        monkeypatch,
        "photometry_pipeline.guided_continuous_rwd_tonic_run",
        "execute_guided_continuous_rwd_tonic_run",
        calls,
        "tonic",
    )
    _guard(
        monkeypatch,
        "photometry_pipeline.guided_continuous_rwd_phasic_run",
        "execute_guided_continuous_rwd_phasic_run",
        calls,
        "phasic",
    )
    _guard(
        monkeypatch,
        "photometry_pipeline.guided_continuous_rwd_combined_run",
        "execute_guided_continuous_rwd_combined_run",
        calls,
        "combined",
    )
    return calls


def test_correction_only_calls_only_correction_backend():
    # No accepted signal requests any analysis at all: neither tonic nor
    # phasic. There is no dedicated "correction only" execution_mode value,
    # so this drives the selector directly (see module docstring / CR1-E2
    # report) rather than through a built plan.
    tonic_analysis, phasic_analysis = False, False
    backend = _select_guided_continuous_rwd_backend(tonic_analysis, phasic_analysis)
    assert backend.__name__ == "execute_guided_continuous_rwd_correction_run"


def test_tonic_only_calls_only_tonic_backend(real_config, tmp_path_factory, monkeypatch):
    calls = _guard_all_backends(monkeypatch)
    output_base = tmp_path_factory.mktemp("cr1_e2_tonic_route")
    inputs = _build_case_for_mode(
        tmp_path_factory.mktemp("cr1_e2_tonic_case") / "recording",
        execution_mode="tonic",
    )
    request = _execution_request(
        inputs, real_config, output_base, execution_mode="tonic"
    )
    _execute_guided_continuous_rwd(request)
    assert calls.get("tonic") == 1
    assert "correction" not in calls
    assert "phasic" not in calls
    assert "combined" not in calls


def test_phasic_only_calls_only_phasic_backend(
    real_config, tmp_path_factory, monkeypatch
):
    calls = _guard_all_backends(monkeypatch)
    output_base = tmp_path_factory.mktemp("cr1_e2_phasic_route")
    inputs = _build_case_for_mode(
        tmp_path_factory.mktemp("cr1_e2_phasic_case") / "recording",
        execution_mode="phasic",
        feature_confirmed=True,
    )
    request = _execution_request(
        inputs,
        real_config,
        output_base,
        execution_mode="phasic",
        feature_confirmed=True,
    )
    _execute_guided_continuous_rwd(request)
    assert calls.get("phasic") == 1
    assert "correction" not in calls
    assert "tonic" not in calls
    assert "combined" not in calls


def test_combined_calls_only_combined_backend(
    real_config, tmp_path_factory, monkeypatch
):
    calls = _guard_all_backends(monkeypatch)
    output_base = tmp_path_factory.mktemp("cr1_e2_combined_route")
    inputs = _build_case_for_mode(
        tmp_path_factory.mktemp("cr1_e2_combined_case") / "recording",
        execution_mode="both",
        feature_confirmed=True,
    )
    request = _execution_request(
        inputs,
        real_config,
        output_base,
        execution_mode="both",
        feature_confirmed=True,
    )
    _execute_guided_continuous_rwd(request)
    assert calls.get("combined") == 1
    assert "correction" not in calls
    assert "tonic" not in calls
    assert "phasic" not in calls


# ---------------------------------------------------------------------------
# Unchanged authority identity forwarding
# ---------------------------------------------------------------------------


def test_backend_receives_every_accepted_authority_unchanged_by_identity(
    accepted_case, real_config, tmp_path_factory, monkeypatch
):
    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e2_identity")
    request = _execution_request(
        inputs,
        real_config,
        output_base,
        execution_mode="both",
        feature_confirmed=True,
    )

    captured = {}

    def fake_combined(
        review_binding,
        target_grid,
        block_plan,
        segment_plan,
        dynamic_f0_authority,
        *,
        accepted_draft,
        startup_mapping_contract,
        output_base,
        config,
        cancellation_requested=None,
    ):
        captured["review_binding"] = review_binding
        captured["target_grid"] = target_grid
        captured["block_plan"] = block_plan
        captured["segment_plan"] = segment_plan
        captured["dynamic_f0_authority"] = dynamic_f0_authority
        captured["accepted_draft"] = accepted_draft
        captured["startup_mapping_contract"] = startup_mapping_contract
        captured["output_base"] = output_base
        captured["config"] = config
        captured["cancellation_requested"] = cancellation_requested
        return "sentinel-result"

    monkeypatch.setattr(
        "photometry_pipeline.guided_continuous_rwd_combined_run.execute_guided_continuous_rwd_combined_run",
        fake_combined,
    )

    cancel_callable = lambda: False
    request = dataclasses.replace(request, cancellation_requested=cancel_callable)
    result = _execute_guided_continuous_rwd(request)

    assert result == "sentinel-result"
    assert captured["review_binding"] is request.review_binding
    assert captured["target_grid"] is request.target_grid
    assert captured["block_plan"] is request.block_plan
    assert captured["segment_plan"] is request.segment_plan
    assert captured["dynamic_f0_authority"] is request.dynamic_f0_authority
    assert captured["accepted_draft"] is request.accepted_draft
    assert captured["startup_mapping_contract"] is request.startup_mapping_contract
    assert captured["output_base"] == request.output_base
    assert captured["config"] is request.config
    assert captured["cancellation_requested"] is cancel_callable


# ---------------------------------------------------------------------------
# Success through the actual worker
# ---------------------------------------------------------------------------


def test_worker_real_combined_run_succeeds_and_reopens_through_continuous_results(
    real_config, tmp_path_factory
):
    inputs = _build_case_for_mode(
        tmp_path_factory.mktemp("cr1_e2_worker_success_case") / "recording",
        execution_mode="both",
        feature_confirmed=True,
    )
    output_base = tmp_path_factory.mktemp("cr1_e2_worker_success")
    request = _execution_request(
        inputs,
        real_config,
        output_base,
        execution_mode="both",
        feature_confirmed=True,
    )

    worker = _GuidedRunExecutionWorker(None, None, continuous_execution=request)
    outcomes = {"succeeded": None, "failed": None}
    worker.succeeded.connect(lambda result: outcomes.__setitem__("succeeded", result))
    worker.failed.connect(lambda message: outcomes.__setitem__("failed", message))

    worker.run()

    assert outcomes["failed"] is None
    result = outcomes["succeeded"]
    assert result is not None
    assert os.path.isdir(result.run_dir)

    classification = classify_run_terminal_state(result.run_dir)
    assert classification.state == TERMINAL_SUCCESS_CURRENT

    from photometry_pipeline.completed_continuous_rwd_review import (
        load_continuous_run_overview,
    )

    overview = load_continuous_run_overview(result.run_dir)
    assert overview.tonic_analysis is True
    assert overview.phasic_analysis is True


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_correction_stage_cancellation_is_not_success(real_config, tmp_path_factory):
    inputs = _build_case_for_mode(
        tmp_path_factory.mktemp("cr1_e2_cancel_correction_case") / "recording",
        execution_mode="tonic",
    )
    output_base = tmp_path_factory.mktemp("cr1_e2_cancel_correction")
    request = _execution_request(
        inputs,
        real_config,
        output_base,
        execution_mode="tonic",
        cancellation_requested=lambda: True,
    )

    worker = _GuidedRunExecutionWorker(None, None, continuous_execution=request)
    outcomes = {"succeeded": None, "failed": None}
    worker.succeeded.connect(lambda result: outcomes.__setitem__("succeeded", result))
    worker.failed.connect(lambda message: outcomes.__setitem__("failed", message))

    worker.run()

    assert outcomes["succeeded"] is None
    assert outcomes["failed"] is not None
    assert outcomes["failed"].startswith("cancelled:")


def test_phasic_between_roi_cancellation_is_not_success(real_config, tmp_path_factory):
    from photometry_pipeline.guided_continuous_rwd_phasic_detection import (
        GuidedContinuousRwdPhasicDetectionError,
    )

    inputs = _build_case_for_mode(
        tmp_path_factory.mktemp("cr1_e2_cancel_phasic_case") / "recording",
        execution_mode="phasic",
        feature_confirmed=True,
    )
    output_base = tmp_path_factory.mktemp("cr1_e2_cancel_phasic")

    calls = {"count": 0}

    def flaky_detection(*args, **kwargs):
        calls["count"] += 1
        raise GuidedContinuousRwdPhasicDetectionError(
            "simulated between-ROI cancellation", category="phasic_detection_interrupted"
        )

    import photometry_pipeline.guided_continuous_rwd_phasic_run as phasic_run_module

    original = phasic_run_module.detect_guided_continuous_rwd_phasic_features
    phasic_run_module.detect_guided_continuous_rwd_phasic_features = flaky_detection
    try:
        request = _execution_request(
            inputs,
            real_config,
            output_base,
            execution_mode="phasic",
            feature_confirmed=True,
        )
        worker = _GuidedRunExecutionWorker(None, None, continuous_execution=request)
        outcomes = {"succeeded": None, "failed": None}
        worker.succeeded.connect(lambda result: outcomes.__setitem__("succeeded", result))
        worker.failed.connect(lambda message: outcomes.__setitem__("failed", message))

        worker.run()
    finally:
        phasic_run_module.detect_guided_continuous_rwd_phasic_features = original

    assert calls["count"] == 1
    assert outcomes["succeeded"] is None
    assert outcomes["failed"] is not None
    assert outcomes["failed"].startswith("cancelled:")


# ---------------------------------------------------------------------------
# Genuine failure (not cancellation)
# ---------------------------------------------------------------------------


def test_genuine_backend_failure_is_reported_as_error_not_cancelled(
    accepted_case, real_config, tmp_path_factory, monkeypatch
):
    def broken_correction(*args, **kwargs):
        raise RuntimeError("simulated genuine backend failure")

    monkeypatch.setattr(
        "photometry_pipeline.guided_continuous_rwd_tonic_run.execute_guided_continuous_rwd_tonic_run",
        broken_correction,
    )

    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e2_genuine_failure")
    request = _execution_request(
        inputs, real_config, output_base, execution_mode="tonic"
    )

    worker = _GuidedRunExecutionWorker(None, None, continuous_execution=request)
    outcomes = {"succeeded": None, "failed": None}
    worker.succeeded.connect(lambda result: outcomes.__setitem__("succeeded", result))
    worker.failed.connect(lambda message: outcomes.__setitem__("failed", message))

    worker.run()

    assert outcomes["succeeded"] is None
    assert outcomes["failed"] is not None
    assert not outcomes["failed"].startswith("cancelled:")
    assert "simulated genuine backend failure" in outcomes["failed"]


# ---------------------------------------------------------------------------
# Inconsistent accepted plan -- refuse before invoking any backend
# ---------------------------------------------------------------------------


def test_phasic_selected_without_confirmed_feature_settings_refuses(
    accepted_case, real_config, tmp_path_factory, monkeypatch
):
    calls = _guard_all_backends(monkeypatch)
    inputs = _pass_inputs(accepted_case)
    output_base = tmp_path_factory.mktemp("cr1_e2_missing_features")
    request = _execution_request(
        inputs,
        real_config,
        output_base,
        execution_mode="phasic",
        feature_confirmed=False,
    )
    with pytest.raises(ValueError, match="feature-detection settings"):
        _execute_guided_continuous_rwd(request)
    assert calls == {}


def test_roi_authority_mismatch_refuses(
    accepted_case, real_config, tmp_path_factory, monkeypatch
):
    calls = _guard_all_backends(monkeypatch)
    inputs = _pass_inputs(accepted_case)
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    mismatched_draft = _mode_draft(draft, execution_mode="tonic")
    mismatched_draft = dataclasses.replace(
        mismatched_draft, included_roi_ids=["ROI1"]
    )
    output_base = tmp_path_factory.mktemp("cr1_e2_roi_mismatch")
    request = _execution_request(
        inputs,
        real_config,
        output_base,
        execution_mode="tonic",
        draft_override=mismatched_draft,
    )
    with pytest.raises(ValueError, match="included regions do not match"):
        _execute_guided_continuous_rwd(request)
    assert calls == {}


def test_unrecognized_execution_mode_refuses(
    accepted_case, real_config, tmp_path_factory, monkeypatch
):
    calls = _guard_all_backends(monkeypatch)
    inputs = _pass_inputs(accepted_case)
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    invalid_draft = _mode_draft(draft, execution_mode="not_a_real_mode")
    output_base = tmp_path_factory.mktemp("cr1_e2_invalid_mode")
    request = _execution_request(
        inputs,
        real_config,
        output_base,
        execution_mode="tonic",
        draft_override=invalid_draft,
    )
    with pytest.raises(ValueError, match="not a recognized continuous-RWD analysis selection"):
        _execute_guided_continuous_rwd(request)
    assert calls == {}


def test_non_continuous_draft_refuses(
    accepted_case, real_config, tmp_path_factory, monkeypatch
):
    calls = _guard_all_backends(monkeypatch)
    inputs = _pass_inputs(accepted_case)
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    intermittent_draft = dataclasses.replace(
        _mode_draft(draft, execution_mode="tonic"), acquisition_mode="intermittent"
    )
    output_base = tmp_path_factory.mktemp("cr1_e2_not_continuous")
    request = _execution_request(
        inputs,
        real_config,
        output_base,
        execution_mode="tonic",
        draft_override=intermittent_draft,
    )
    with pytest.raises(ValueError, match="not a continuous-RWD analysis"):
        _execute_guided_continuous_rwd(request)
    assert calls == {}


# ---------------------------------------------------------------------------
# Legacy chunked continuous-output routing must not be captured
# ---------------------------------------------------------------------------


def test_legacy_custom_tabular_continuous_draft_is_refused_not_routed(
    accepted_case, real_config, tmp_path_factory, monkeypatch
):
    """Guards CR1-E2 handoff section 9: acquisition_mode == "continuous" is
    shared with the older chunked custom_tabular continuous-output workflow
    -- input_format == "rwd" is required in addition, never acquisition_mode
    alone."""
    calls = _guard_all_backends(monkeypatch)
    inputs = _pass_inputs(accepted_case)
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    legacy_draft = dataclasses.replace(
        _mode_draft(draft, execution_mode="tonic"), input_format="custom_tabular"
    )
    output_base = tmp_path_factory.mktemp("cr1_e2_legacy_custom_tabular")
    request = _execution_request(
        inputs,
        real_config,
        output_base,
        execution_mode="tonic",
        draft_override=legacy_draft,
    )
    with pytest.raises(ValueError, match="not a continuous-RWD"):
        _execute_guided_continuous_rwd(request)
    assert calls == {}


def test_analysis_selection_helper_maps_execution_intent_directly(accepted_case):
    binding, grid, draft, contract, _source = accepted_case
    both = _mode_draft(draft, execution_mode="both")
    tonic = _mode_draft(draft, execution_mode="tonic")
    phasic = _mode_draft(draft, execution_mode="phasic")
    assert _guided_continuous_rwd_analysis_selection(both) == (True, True)
    assert _guided_continuous_rwd_analysis_selection(tonic) == (True, False)
    assert _guided_continuous_rwd_analysis_selection(phasic) == (False, True)
    with pytest.raises(ValueError):
        _guided_continuous_rwd_analysis_selection(
            _mode_draft(draft, execution_mode="bogus")
        )


# ---------------------------------------------------------------------------
# Guided continuous Run remains disabled -- no ordinary call site wires
# continuous_execution
# ---------------------------------------------------------------------------


def test_no_existing_call_site_passes_continuous_execution():
    """Proves the execution branch exists and is directly callable/testable
    without changing today's enablement state: no real call site in
    main_window.py outside this worker's own definition passes
    `continuous_execution=` to `_GuidedRunExecutionWorker`."""
    source_path = main_window_module.__file__
    with open(source_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    call_sites = [
        line
        for line in source.splitlines()
        if "_GuidedRunExecutionWorker(" in line and "class " not in line
    ]
    assert call_sites, "expected at least one _GuidedRunExecutionWorker(...) call site"
    for line in call_sites:
        assert "continuous_execution" not in line


def test_guided_run_execution_worker_default_continuous_execution_is_none():
    worker = _GuidedRunExecutionWorker(request=None, runner=None)
    assert worker._continuous_execution is None
