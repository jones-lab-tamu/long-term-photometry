"""Guided NPM Run affordance: button enablement, readiness text, and click
safety for the shared Guided execution path (Phase 4C).

Guided NPM Run no longer has its own bespoke dispatch, threading, launch,
or reconciliation wiring reachable from the GUI -- `_guided_run_btn.clicked`
connects to the exact same `_on_guided_run_clicked_backend_guarded` handler
RWD uses, which routes NPM through the shared startup-request builder
(`guided_execution_request_builder.build_guided_npm_startup_request_from_validation`)
and the shared execution worker (`_GuidedRunExecutionWorker`,
`_start_guided_run_execution_worker`) -- the identical machinery this file's
RWD sibling (`test_guided_gui_run_affordance.py`) already covers. This file
mirrors that sibling's scope and pattern for NPM: fast, isolated
enablement/readiness/click-safety tests using a real (non-GUI-driven)
accepted NPM startup request, not the full wizard walkthrough.

Coverage split (avoiding duplication with other files that already prove
these, per the same shared machinery):
  - "One Run click invokes the shared startup path, hands off to the
    shared orchestration path, execution_mode == 'both'" -- proven via a
    real, full Guided-wizard walkthrough in
    tests/test_guided_npm_gui_natural_path.py.
  - "Stale/changed source refuses before allocation" (the live NPM
    prelaunch freshness recheck) -- proven in
    tests/test_guided_npm_gui_natural_path.py::
    test_natural_path_npm_run_refuses_after_post_setup_source_rename.
  - "No hidden Full Control mode control affects the result" -- proven at
    the shared, format-neutral draft-construction level in
    tests/test_gui_guided_new_analysis_plan.py::
    test_guided_draft_execution_mode_independent_of_full_control_mode_combo
    and at the authorization level in
    tests/test_guided_gui_run_completed_boundary.py::
    test_real_gui_hidden_full_control_mode_combo_does_not_affect_guided_authorization.
  - Lower-level bespoke-path coverage (build/launch/reconcile, worker
    threading, terminal-receipt mapping) remains fully exercised in its own
    dedicated suites (test_guided_npm_worker_launch.py,
    test_guided_npm_worker_reconciliation.py,
    test_guided_npm_run_launch_builder.py) -- that machinery is retired
    from the GUI dispatch path (Section 12) but not deleted, and stays
    covered independently of whether the GUI reaches it.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

import photometry_pipeline.guided_backend_execution as backend_execution
import photometry_pipeline.guided_execution_request_builder as request_builder_module
import photometry_pipeline.guided_npm_run_launch_builder as npm_builder_module
import photometry_pipeline.guided_npm_worker_launch as npm_launch_module
import photometry_pipeline.guided_production_mapping as production_mapping
import photometry_pipeline.guided_startup_allocation as allocation
import photometry_pipeline.guided_startup_materialization as materialization
import photometry_pipeline.guided_startup_orchestration as orchestration
import photometry_pipeline.pipeline as pipeline
import tools.run_full_pipeline_deliverables as wrapper
import gui.main_window as main_window_module
from gui.main_window import MainWindow, _GuidedNpmRunWorker
from gui.run_report_parser import classify_completed_run_candidate
from tests.test_npm_user_language import _accepted_npm_outcome


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance._guided_backend_execution_active = False
    thread = getattr(instance, "_guided_npm_run_worker_thread", None)
    if thread is not None and thread.isRunning():
        thread.quit()
        thread.wait(2000)
    instance.close()
    instance.deleteLater()


def _visible_text(window):
    return " ".join(
        (
            window._guided_run_btn.text(),
            window._guided_run_btn.toolTip(),
            window._guided_run_readiness_label.text(),
        )
    )


def _accepted_npm_startup_request(tmp_path, monkeypatch):
    """Build a real, accepted `GuidedStartupTransactionRequest` for NPM
    through the actual production validation/mapping/authority/payload/
    finalizer chain (guided_backend_validation_workflow.
    validate_current_guided_draft_for_backend ->
    guided_execution_request_builder.
    build_guided_npm_startup_request_from_validation), the same real
    building block test_guided_npm_production_mapping.py's `_accepted_npm`
    uses -- not a hand-built SimpleNamespace shortcut. Only the deep
    application-build-identity resolution is mocked, mirroring the RWD
    sibling file's own `startup_request` fixture."""
    from tests.test_guided_backend_validation_materialization import (
        _valid_npm_stage2c_draft,
    )
    from photometry_pipeline.guided_backend_validation_workflow import (
        GuidedBackendValidationGuiContext,
        validate_current_guided_draft_for_backend,
    )
    from photometry_pipeline.guided_backend_validator import (
        GuidedBackendValidatorContract,
    )
    from photometry_pipeline.io.npm_contract import NpmParserContract
    from photometry_pipeline.guided_new_analysis_plan import (
        GuidedNewAnalysisExecutionIntent,
    )

    draft = _valid_npm_stage2c_draft(tmp_path)
    # Guided always authorizes execution_mode == "both" (Phase 4C); the raw
    # draft-plan builder's own dataclass default is "phasic" (unrelated to
    # any GUI widget), so it must be overridden explicitly here the same
    # way gui/main_window.py's real draft-plan producer does.
    draft.execution_intent = GuidedNewAnalysisExecutionIntent(
        execution_mode="both"
    )
    validator_contract = GuidedBackendValidatorContract(
        validation_scope="guided_rwd_intermittent_phasic_full_validate",
        validation_contract_version="guided_backend_validation_contract.v1",
        validator_capability_version="test_validator_capability.v1",
        supported_subset_rule_version="global_dynamic_fit_only.v1",
    )
    parser_contract = NpmParserContract(
        npm_time_axis="system_timestamp",
        npm_system_ts_col="SystemTimestamp",
        npm_computer_ts_col="ComputerTimestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
        target_fs_hz=2.0,
        session_duration_sec=2.0,
        allow_partial_final_chunk=False,
        adapter_value_nan_policy="strict",
    )
    revision = 4
    outcome = validate_current_guided_draft_for_backend(
        draft,
        parser_contract=parser_contract,
        validator_contract=validator_contract,
        validation_revision=revision,
    )
    assert outcome.status == "validator_accepted", outcome.blocking_issues

    build_identity = production_mapping.build_application_build_identity(
        distribution_name="photometry-pipeline",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="abc123",
        source_tree_state="clean",
    )
    monkeypatch.setattr(
        request_builder_module,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(build_identity=build_identity),
    )
    context = GuidedBackendValidationGuiContext(
        draft=draft,
        parser_contract=parser_contract,
        additional_protected_roots=(),
        validator_contract=validator_contract,
        revision=revision,
    )
    result = request_builder_module.build_guided_npm_startup_request_from_validation(
        validation_context=context,
        validation_outcome=outcome,
        current_gui_revision=revision,
    )
    assert result.ok is True, result.blocking_issues
    return result.startup_transaction_request


@pytest.fixture
def npm_startup_request(tmp_path, monkeypatch):
    return _accepted_npm_startup_request(tmp_path, monkeypatch)


def _set_ready(window, request):
    from photometry_pipeline.guided_plan_identity import (
        compute_guided_new_analysis_draft_plan_identity,
    )

    window._guided_format_combo.setCurrentText("npm")
    window._guided_backend_validation_revision = request.current_guided_revision
    window._guided_backend_validation_outcome = _accepted_npm_outcome()
    window._guided_backend_validation_outcome_revision = (
        request.current_guided_revision
    )
    window._guided_startup_authority = request.startup_authority
    window._guided_execution_payload_result = request.payload_result
    # See tests/test_guided_gui_run_execution_wiring.py::_set_ready -- the
    # authoritative identity check requires the freshly-rebuilt draft's
    # canonical identity to match what was "validated".
    window._guided_validated_plan_identity = (
        compute_guided_new_analysis_draft_plan_identity(
            window._build_guided_new_analysis_draft_plan()
        )
    )
    window._refresh_guided_run_readiness_display()


# ---------------------------------------------------------------------------
# Enablement: Run stays disabled for no/stale/refused validation, and for
# an accepted validation missing its authority or payload; Run enables only
# once every accepted-state condition is genuinely satisfied.
# ---------------------------------------------------------------------------


def test_no_validation_message_asks_for_validation(window):
    window._guided_format_combo.setCurrentText("npm")
    window._guided_backend_validation_outcome = None
    window._guided_backend_validation_outcome_revision = None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "no_validation"
    assert "Validate the Guided setup" in window._guided_run_readiness_label.text()
    assert window._guided_run_btn.isEnabled() is False


def test_stale_validation_message_asks_for_revalidation(window):
    window._guided_format_combo.setCurrentText("npm")
    window._guided_backend_validation_revision = 4
    window._guided_backend_validation_outcome = replace(
        _accepted_npm_outcome(), stale=True
    )
    window._guided_backend_validation_outcome_revision = 4
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "validation_stale"
    assert "Validate again" in window._guided_run_readiness_label.text()
    assert window._guided_run_btn.isEnabled() is False


def test_validation_failure_does_not_enable_run(window):
    from tests.test_guided_npm_production_mapping import _unsafe_replace

    window._guided_format_combo.setCurrentText("npm")
    refused = _unsafe_replace(
        _accepted_npm_outcome(),
        status="validator_refused",
        accepted_for_backend_validation=False,
    )
    window._guided_backend_validation_outcome = refused
    window._guided_backend_validation_outcome_revision = (
        window._guided_backend_validation_revision
    )
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "validation_not_accepted"
    assert window._guided_run_btn.isEnabled() is False


def test_missing_authorization_keeps_button_disabled(window, npm_startup_request):
    _set_ready(window, npm_startup_request)
    window._guided_startup_authority = None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "authorization_missing"
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_run_readiness_label.text() == (
        "Guided validation succeeded, but Guided Run execution is unavailable "
        "in this build."
    )


def test_missing_payload_keeps_button_disabled(window, npm_startup_request):
    _set_ready(window, npm_startup_request)
    window._guided_execution_payload_result = None
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "payload_missing"
    assert window._guided_run_btn.isEnabled() is False


def test_payload_not_ready_keeps_button_disabled(window, npm_startup_request):
    _set_ready(window, npm_startup_request)
    window._guided_execution_payload_result = replace(
        npm_startup_request.payload_result, ok=False
    )
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_readiness.status == "payload_not_ready"
    assert window._guided_run_btn.isEnabled() is False


def test_ready_hidden_enables_guarded_button(window, npm_startup_request):
    """Run enabled for a supported, accepted NPM plan -- governed solely
    by the shared, format-neutral evaluate_guided_run_readiness, the same
    predicate that already drives the RWD button."""
    _set_ready(window, npm_startup_request)
    assert window._guided_run_readiness.status == "ready_hidden"
    assert window._guided_run_readiness.ready is True
    assert window._guided_startup_authority.is_npm is True
    assert window._guided_startup_authority.execution_mode == "both"
    assert window._guided_run_btn.isEnabled() is True
    assert window._guided_run_btn.toolTip() == "Guided Run is ready to start."
    assert window._guided_run_readiness_label.text() == (
        "Guided Run is ready to start."
    )


def test_active_run_disables_run(window, npm_startup_request):
    _set_ready(window, npm_startup_request)
    window._guided_backend_execution_active = True
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False


def test_completed_run_keeps_run_disabled(window, npm_startup_request):
    _set_ready(window, npm_startup_request)
    window._guided_backend_execution_result = object()
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False


# ---------------------------------------------------------------------------
# Click safety: an enabled button whose retained request/authority is not
# genuinely current must refuse before any allocation, materialization, or
# wrapper invocation -- and duplicate clicks while startup is active must
# not invoke the shared path a second time.
# ---------------------------------------------------------------------------


def test_enabled_click_without_retained_request_calls_no_execution_path(
    window, npm_startup_request, monkeypatch
):
    def fail(*_args, **_kwargs):
        raise AssertionError("execution must not be called")

    monkeypatch.setattr(backend_execution, "execute_guided_backend_run", fail)
    monkeypatch.setattr(orchestration, "run_guided_startup_to_wrapper", fail)
    monkeypatch.setattr(allocation, "allocate_guided_startup_directory", fail)
    monkeypatch.setattr(
        materialization, "materialize_guided_startup_artifacts", fail
    )
    monkeypatch.setattr(wrapper, "main", fail)
    monkeypatch.setattr(wrapper.subprocess, "run", fail)
    monkeypatch.setattr(pipeline, "Pipeline", fail)
    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        fail,
    )
    monkeypatch.setattr(npm_launch_module, "launch_guided_npm_worker_runtime", fail)
    monkeypatch.setattr(main_window_module.MainWindow, "_on_guided_npm_run_clicked", fail)
    _set_ready(window, npm_startup_request)
    window._guided_run_btn.click()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_run_readiness_label.text() == (
        "Guided Run could not start because the validated setup is no longer "
        "current."
    )


def test_enabled_click_writes_nothing_and_creates_no_completed_run(
    window, npm_startup_request, tmp_path, monkeypatch
):
    _set_ready(window, npm_startup_request)
    before = tuple(tmp_path.iterdir())

    def fail(*_args, **_kwargs):
        raise AssertionError("write must not occur")

    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(Path, "write_bytes", fail)
    monkeypatch.setattr(Path, "mkdir", fail)
    monkeypatch.setattr(os, "mkdir", fail)
    monkeypatch.setattr(os, "makedirs", fail)
    window._guided_run_btn.click()
    assert tuple(tmp_path.iterdir()) == before
    assert classify_completed_run_candidate(str(tmp_path))[0] is False


def test_double_click_never_reaches_bespoke_worker(window, npm_startup_request):
    """Neither an enabled nor a refused click ever falls back to the
    retired bespoke NPM worker chain -- proven by asserting it is never
    called, across two rapid clicks."""
    _set_ready(window, npm_startup_request)
    window._guided_run_btn.click()
    window._guided_run_btn.click()
    window._on_guided_run_clicked_backend_guarded()
    assert window._guided_npm_run_worker is None
    assert window._guided_npm_launch_runtime is None


# ---------------------------------------------------------------------------
# Scientist-facing text: no internal jargon leaks into the visible button/
# readiness text, for either format-neutral shared refusals or NPM-specific
# readiness summaries.
# ---------------------------------------------------------------------------


def test_visible_text_excludes_internal_terms(window, npm_startup_request):
    _set_ready(window, npm_startup_request)
    window._guided_run_btn.click()
    text = _visible_text(window).lower()
    prohibited = (
        "manifest",
        "preallocated",
        "command_invoked",
        "wrapper claim",
        "startup transaction",
        "hash",
        "--guided",
        "config_effective.yaml",
        "runner_request",
        "startup_transaction_unavailable",
        "guided_candidate_manifest",
        "guided_startup",
        "wrapper_claim",
        "backend adapter",
        "orchestration",
        "subprocess",
        "worker",
        "receipt",
        "authority",
        "reconciliation",
        "backend",
        "json",
        "runtime",
    )
    assert not any(term in text for term in prohibited)


def test_npm_readiness_text_is_scientist_facing():
    from photometry_pipeline.guided_run_readiness import _SUMMARIES

    forbidden = (
        "artifact",
        "receipt",
        "authority",
        "reconciliation",
        "worker",
        "subprocess",
        "exit code",
        "backend",
        "manifest",
        "json",
        "runtime",
        "process identity",
        "startup payload",
        "execution request",
    )
    for summary in _SUMMARIES.values():
        lowered = summary.lower()
        for term in forbidden:
            assert term not in lowered, f"{term!r} leaked into: {summary!r}"


# ---------------------------------------------------------------------------
# Full Control is unaffected by Guided NPM readiness/click.
# ---------------------------------------------------------------------------


def test_full_control_run_control_is_unchanged_by_refresh_and_click(
    window, npm_startup_request
):
    full_control_run = window._run_btn
    before = (
        full_control_run.text(),
        full_control_run.isEnabled(),
        full_control_run.toolTip(),
    )
    _set_ready(window, npm_startup_request)
    window._guided_run_btn.click()
    after = (
        full_control_run.text(),
        full_control_run.isEnabled(),
        full_control_run.toolTip(),
    )
    assert after == before


# ---------------------------------------------------------------------------
# Bespoke worker isolation: the retired worker class remains independently
# well-formed (still directly unit-testable, still has no MainWindow
# reference), even though the GUI no longer constructs or drives it.
# ---------------------------------------------------------------------------


def test_bespoke_npm_worker_still_isolated_and_gui_free():
    import inspect

    run_source = inspect.getsource(_GuidedNpmRunWorker.run)
    prohibited = (
        "MainWindow",
        "self._window",
        "self.window",
        "_guided_run_btn",
        "_guided_run_readiness_label",
        "_guided_backend_execution_active",
        "_capture_guided_backend_validation_context",
    )
    for term in prohibited:
        assert term not in run_source, f"{term!r} leaked into worker.run source"
