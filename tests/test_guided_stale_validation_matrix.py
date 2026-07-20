"""Stale-validation invalidation matrix: authoritative canonical-plan-identity
architecture (see photometry_pipeline/guided_plan_identity.py).

These tests drive a real MainWindow through a real Setup-check (Validate)
click, confirm Run becomes enabled, then apply one analysis-defining
mutation and prove:

- the Run button disables;
- direct invocation of the guarded Run click handler performs no
  execution (no worker construction, no allocation);
- the retained validation/authorization/startup-request state can no
  longer be reused.

Several of the mutation paths exercised here (output destination text
edit, correction-method card selection, per-ROI feature override removal)
were confirmed during the architecture audit to NOT bump the old manual
revision counter (`_invalidate_guided_backend_validation`) -- the
authoritative canonical-identity comparison introduced by this task closes
those gaps structurally, without requiring every callback to remember to
invalidate.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from gui.main_window import (
    GUIDED_SIGNAL_ONLY_F0_CARD,
    GUIDED_WORKFLOW_STEPS,
    MainWindow,
)
from tests.test_gui_guided_new_analysis_plan import (
    _confirm_detected_dataset_settings_via_review_plan_button,
)
from tests.test_guided_gui_run_completed_boundary import (
    _configure_real_analysis_duration_new_analysis_draft,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _validate_and_authorize(window, monkeypatch):
    """Real Validate click, returning the authorized canonical identity."""
    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping

    build_identity = production_mapping.build_application_build_identity(
        distribution_name="photometry-pipeline",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="abc123",
        source_tree_state="clean",
    )
    monkeypatch.setattr(
        request_builder,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(build_identity=build_identity),
    )
    window._guided_backend_validate_btn.click()
    outcome = window._guided_backend_validation_outcome
    assert outcome.status == "validator_accepted", outcome.blocking_issues
    assert window._guided_run_btn.isEnabled() is True
    return window._guided_startup_authority.canonical_authorization_identity


def _configure_and_validate(
    window, tmp_path, monkeypatch, *, strategy_by_roi=None, analysis_mode="phasic"
):
    strategy_by_roi = strategy_by_roi or {
        "CH1": "Robust Global Event-Reject Fit",
        "CH2": "Global Linear Regression",
    }
    _configure_real_analysis_duration_new_analysis_draft(
        window,
        tmp_path,
        monkeypatch,
        strategy_by_roi=strategy_by_roi,
        analysis_mode=analysis_mode,
        rois=tuple(strategy_by_roi),
    )
    _confirm_detected_dataset_settings_via_review_plan_button(window, monkeypatch)
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    window._guided_review_go_to_run_btn.click()
    identity = _validate_and_authorize(window, monkeypatch)
    return identity


def _assert_run_disabled_and_guard_refuses(window, monkeypatch):
    """Central assertion block for every stale-validation matrix case.

    Every mutation path exercised in this file is now wired to the
    central invalidation transition, so the button and retained state
    must already be cleared IMMEDIATELY after the mutation -- before
    _on_guided_run_clicked_backend_guarded is ever invoked. This proves
    "Run disables immediately" (policy 1), not merely "Run refuses when
    eventually clicked". The guard is invoked afterward only to prove the
    click path itself performs no execution -- it does not normalize an
    initially-enabled button by clicking first.
    """
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_validated_plan_identity is None
    assert window._guided_startup_authority is None
    assert window._guided_execution_payload_result is None
    assert window._guided_startup_transaction_request is None
    label_text = window._guided_run_readiness_label.text()
    assert (
        "Validate again" in label_text
        or "Setup check" in label_text
        or "no longer current" in label_text
    ), label_text

    starts = []
    monkeypatch.setattr(
        window,
        "_start_guided_run_execution_worker",
        lambda request: starts.append(request),
    )
    monkeypatch.setattr(window, "_start_guided_run_live_status", lambda *_: None)
    window._on_guided_run_clicked_backend_guarded()
    assert starts == []
    assert window._guided_backend_execution_active is False
    assert window._guided_run_btn.isEnabled() is False


@pytest.fixture
def window(qapp, tmp_path):
    instance = MainWindow(
        settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    )
    yield instance
    instance.close()


def test_output_destination_change_invalidates(window, tmp_path, monkeypatch):
    """Confirmed audit gap: the Setup-step output folder textbox was never
    wired to the old manual revision-invalidation callback."""
    _configure_and_validate(window, tmp_path, monkeypatch)
    new_output = tmp_path / "a_different_output_dir"
    new_output.mkdir()
    window._guided_output_dir_edit.setText(str(new_output))
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)


def test_roi_exclusion_change_invalidates(window, tmp_path, monkeypatch):
    _configure_and_validate(window, tmp_path, monkeypatch)
    for index in range(window._guided_roi_list.count()):
        item = window._guided_roi_list.item(index)
        if item.text() == "CH2" or "CH2" in item.text():
            from PySide6.QtCore import Qt

            item.setCheckState(Qt.Unchecked)
            break
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)


def test_correction_strategy_switch_to_signal_only_invalidates(
    window, tmp_path, monkeypatch
):
    """Re-confirms CH1 with a different strategy through the same real
    per-ROI local-preview-row flow the initial setup used (the only
    strategy-confirmation control a real user has on this path), proving a
    genuine per-ROI correction-strategy change invalidates."""
    _configure_and_validate(window, tmp_path, monkeypatch)
    roi = "CH1"
    roi_idx = window._guided_preview_roi_combo.findData(roi)
    assert roi_idx >= 0
    window._guided_preview_roi_combo.setCurrentIndex(roi_idx)
    strategy_index = window._guided_confirm_strategy_combo.findText("Signal-Only F0")
    if strategy_index < 0:
        strategy_index = window._guided_confirm_strategy_combo.findData(
            "Signal-Only F0"
        )
    assert strategy_index >= 0
    window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
    window._guided_preview_generate_btn.click()
    result = window._guided_preview_last_result
    assert result["status"] in {"success", "partial"}, result
    row = window._guided_local_preview_confirmation_rows[roi]
    row_combo = row["strategy_combo"]
    row_strategy_index = row_combo.findData("signal_only_f0")
    assert row_strategy_index >= 0
    row_combo.setCurrentIndex(row_strategy_index)
    assert row["action_button"].isEnabled()
    row["action_button"].click()
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)


def test_missing_session_approval_change_invalidates(window, tmp_path, monkeypatch):
    from photometry_pipeline.guided_new_analysis_plan import (
        GuidedApprovedMissingSession,
    )

    _configure_and_validate(window, tmp_path, monkeypatch)
    approval = GuidedApprovedMissingSession(
        canonical_relative_path="session-2/fluorescence.csv",
        size_bytes=1024,
        sha256_content_digest="b" * 64,
        session_index=2,
        expected_start_time="2026-01-01T00:00:00Z",
        expected_duration_sec=600.0,
    )
    window._add_guided_missing_session_approval(approval)
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)


def test_full_control_mode_combo_change_does_not_invalidate(
    window, tmp_path, monkeypatch
):
    """Guided Mode exposes no phasic-versus-tonic choice: Full Control's
    `_mode_combo` is never shown by the Guided workflow and must not
    invalidate an accepted Guided authorization. This replaces a retired
    test that asserted the opposite (now-incorrect) behavior from before
    the Guided `execution_mode` root-cause repair -- see the equivalent
    regression in test_guided_gui_run_completed_boundary.py."""
    identity = _configure_and_validate(window, tmp_path, monkeypatch)
    request = window._guided_startup_transaction_request
    window._mode_combo.setCurrentText("both")
    assert window._guided_run_btn.isEnabled() is True
    assert window._guided_startup_transaction_request is request
    assert (
        window._guided_startup_authority.canonical_authorization_identity
        == identity
    )


def test_session_schedule_change_invalidates(window, tmp_path, monkeypatch):
    _configure_and_validate(window, tmp_path, monkeypatch)
    window._guided_sessions_per_hour_edit.setText("4")
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)


def test_correction_method_card_selection_invalidates(window, tmp_path, monkeypatch):
    """Confirmed audit gap (this task): selecting a different Guided
    correction-method card -- the legacy global-strategy intent, still
    consulted absent a unanimous per-ROI override -- was never wired to
    invalidation. Fixed directly inside _select_guided_reference_correction_card."""
    _configure_and_validate(window, tmp_path, monkeypatch)
    window._select_guided_reference_correction_card("Adaptive Event-Gated Fit")
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)


def test_signal_only_f0_correction_intent_selection_invalidates(
    window, tmp_path, monkeypatch
):
    _configure_and_validate(window, tmp_path, monkeypatch)
    window._select_guided_signal_only_f0_intent()
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)


def test_dynamic_fit_mode_combo_direct_change_invalidates(window, tmp_path, monkeypatch):
    """Confirmed audit gap (this task): changing the shared Dynamic Fit
    Mode combo directly (not via a Guided card click -- e.g. from the
    Full Control tab) reaches Guided's correction intent only through
    _sync_guided_correction_from_full, which was never wired to
    invalidation."""
    _configure_and_validate(window, tmp_path, monkeypatch)
    target = "adaptive_event_gated_regression"
    idx = window._dynamic_fit_mode_combo.findData(target)
    assert idx >= 0
    if window._dynamic_fit_mode_combo.currentIndex() == idx:
        target = "global_linear_regression"
        idx = window._dynamic_fit_mode_combo.findData(target)
    window._dynamic_fit_mode_combo.setCurrentIndex(idx)
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)


def test_full_control_roi_select_none_invalidates(window, tmp_path, monkeypatch):
    """Confirmed audit gap (this task): a bulk Select All/Select None
    action on the Full Control ROI checklist reaches Guided's mirrored
    ROI list only through _sync_guided_discovery_from_full's signal-
    blocked repopulation, which was never wired to invalidation."""
    _configure_and_validate(window, tmp_path, monkeypatch)
    window._on_roi_select_none()
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)


def test_preview_evidence_regeneration_without_reconfirm_invalidates(
    window, tmp_path, monkeypatch
):
    """Confirmed audit gap (this task): generating a NEW local preview for
    an already-confirmed ROI stales that ROI's confirmed choice
    (current_or_stale flips to "stale") via
    _mark_guided_local_preview_choices_stale, which was never wired to
    invalidation -- even without the user re-confirming a new strategy."""
    _configure_and_validate(window, tmp_path, monkeypatch)
    roi = "CH1"
    roi_idx = window._guided_preview_roi_combo.findData(roi)
    assert roi_idx >= 0
    window._guided_preview_roi_combo.setCurrentIndex(roi_idx)
    assert window._guided_preview_generate_btn.isEnabled()
    window._guided_preview_generate_btn.click()
    result = window._guided_preview_last_result
    assert result["status"] in {"success", "partial"}, result
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)


# ---------------------------------------------------------------------------
# Non-revival for the newly-wired paths: A -> B -> A must not revive.
# ---------------------------------------------------------------------------


def test_correction_card_selection_restoration_does_not_revive(
    window, tmp_path, monkeypatch
):
    _configure_and_validate(window, tmp_path, monkeypatch)
    window._select_guided_reference_correction_card("Adaptive Event-Gated Fit")
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_validated_plan_identity is None

    # Restore whatever the unanimous/legacy intent was before -- since the
    # two configured ROIs use different strategies (non-unanimous), the
    # original intent card title is recoverable from the combo's original
    # mode; simplest robust restoration is re-selecting the very first
    # reference card used in setup.
    window._select_guided_reference_correction_card(
        "Robust Global Event-Reject Fit"
    )
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_validated_plan_identity is None
    assert window._guided_startup_authority is None
    starts = []
    monkeypatch.setattr(
        window,
        "_start_guided_run_execution_worker",
        lambda request: starts.append(request),
    )
    window._on_guided_run_clicked_backend_guarded()
    assert starts == []


def test_dynamic_fit_mode_combo_restoration_does_not_revive(
    window, tmp_path, monkeypatch
):
    _configure_and_validate(window, tmp_path, monkeypatch)
    original_index = window._dynamic_fit_mode_combo.currentIndex()
    other_idx = window._dynamic_fit_mode_combo.findData(
        "adaptive_event_gated_regression"
    )
    if other_idx == original_index:
        other_idx = window._dynamic_fit_mode_combo.findData(
            "global_linear_regression"
        )
    assert other_idx >= 0
    window._dynamic_fit_mode_combo.setCurrentIndex(other_idx)
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_validated_plan_identity is None

    window._dynamic_fit_mode_combo.setCurrentIndex(original_index)
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_validated_plan_identity is None
    assert window._guided_startup_authority is None
    starts = []
    monkeypatch.setattr(
        window,
        "_start_guided_run_execution_worker",
        lambda request: starts.append(request),
    )
    window._on_guided_run_clicked_backend_guarded()
    assert starts == []


def test_roi_selection_restoration_via_full_control_does_not_revive(
    window, tmp_path, monkeypatch
):
    _configure_and_validate(window, tmp_path, monkeypatch)
    window._on_roi_select_none()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_validated_plan_identity is None

    window._on_roi_select_all()
    window._refresh_guided_run_readiness_display()
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_validated_plan_identity is None
    assert window._guided_startup_authority is None
    starts = []
    monkeypatch.setattr(
        window,
        "_start_guided_run_execution_worker",
        lambda request: starts.append(request),
    )
    window._on_guided_run_clicked_backend_guarded()
    assert starts == []


def test_preview_evidence_replacement_restoration_does_not_revive(
    window, tmp_path, monkeypatch
):
    """Regenerating a preview for CH1 (staling its confirmed choice), then
    regenerating an A-equivalent preview and re-confirming the exact same
    visible strategy, must still require a fresh Setup check -- the
    original validated identity/authorization must never be silently
    reused just because the re-confirmed strategy name looks the same."""
    _configure_and_validate(window, tmp_path, monkeypatch)
    roi = "CH1"
    roi_idx = window._guided_preview_roi_combo.findData(roi)
    assert roi_idx >= 0
    window._guided_preview_roi_combo.setCurrentIndex(roi_idx)
    window._guided_preview_generate_btn.click()
    result = window._guided_preview_last_result
    assert result["status"] in {"success", "partial"}, result
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_validated_plan_identity is None

    # Re-confirm the same (A-equivalent) strategy for CH1 through the real
    # per-ROI row flow, matching what was originally confirmed.
    row = window._guided_local_preview_confirmation_rows[roi]
    row_combo = row["strategy_combo"]
    row_strategy_index = row_combo.findData("robust_global_event_reject")
    assert row_strategy_index >= 0
    row_combo.setCurrentIndex(row_strategy_index)
    assert row["action_button"].isEnabled()
    row["action_button"].click()

    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_validated_plan_identity is None
    assert window._guided_startup_authority is None
    starts = []
    monkeypatch.setattr(
        window,
        "_start_guided_run_execution_worker",
        lambda request: starts.append(request),
    )
    window._on_guided_run_clicked_backend_guarded()
    assert starts == []


# ---------------------------------------------------------------------------
# Non-invalidation: harmless UI-only actions must not disturb a validated,
# ready plan. Assertions check the authoritative identity match and Run
# readiness itself, not merely "a callback was not called".
# ---------------------------------------------------------------------------


def test_harmless_ui_actions_preserve_readiness(window, tmp_path, monkeypatch):
    _configure_and_validate(window, tmp_path, monkeypatch)
    assert window._guided_run_btn.isEnabled() is True
    validated_identity = window._guided_validated_plan_identity
    assert validated_identity

    # Expand/collapse an informational panel.
    toggle = getattr(window, "_guided_review_advanced_toggle", None)
    if toggle is not None:
        toggle.click()
        toggle.click()

    # Resize the window.
    original_size = window.size()
    window.resize(original_size.width() + 40, original_size.height() + 40)
    window.resize(original_size)

    # Switch workflow steps back and forth (read-only navigation).
    run_index = list(GUIDED_WORKFLOW_STEPS).index("Run")
    draft_index = list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    window._guided_workflow_stepper.setCurrentRow(draft_index)
    window._guided_workflow_stepper.setCurrentRow(run_index)

    # Repeated, identical draft (re)construction for display only.
    plan_first = window._build_guided_new_analysis_draft_plan()
    plan_second = window._build_guided_new_analysis_draft_plan()
    from photometry_pipeline.guided_plan_identity import (
        compute_guided_new_analysis_draft_plan_identity,
    )

    assert compute_guided_new_analysis_draft_plan_identity(
        plan_first
    ) == compute_guided_new_analysis_draft_plan_identity(plan_second)

    # Readiness display refresh, repeated.
    window._refresh_guided_run_readiness_display()
    window._refresh_guided_run_readiness_display()

    assert window._guided_validated_plan_identity == validated_identity
    assert window._guided_current_plan_identity_is_validated() is True
    assert window._guided_run_btn.isEnabled() is True
    assert window._guided_run_readiness.status == "ready_hidden"
    assert window._guided_run_readiness.ready is True


def test_repopulation_with_identical_values_does_not_invalidate(
    window, tmp_path, monkeypatch
):
    """The newly-wired paths in this task must distinguish harmless
    re-population/no-op reselection (identical values) from a genuine
    change -- re-syncing or re-selecting the same value must not
    invalidate a current, validated plan."""
    _configure_and_validate(window, tmp_path, monkeypatch)
    assert window._guided_run_btn.isEnabled() is True
    validated_identity = window._guided_validated_plan_identity

    # Re-select whatever correction card is already selected (a genuine
    # no-op reselection, not a change).
    current_intent = window._guided_correction_intent
    if current_intent and current_intent != GUIDED_SIGNAL_ONLY_F0_CARD:
        window._select_guided_reference_correction_card(current_intent)
    # Re-run discovery sync with the identical underlying ROI state
    # (a harmless repopulation, not a genuine selection change).
    window._sync_guided_discovery_from_full()
    # Re-set the dynamic fit mode combo to its own current value.
    window._dynamic_fit_mode_combo.setCurrentIndex(
        window._dynamic_fit_mode_combo.currentIndex()
    )

    window._refresh_guided_run_readiness_display()
    assert window._guided_validated_plan_identity == validated_identity
    assert window._guided_current_plan_identity_is_validated() is True
    assert window._guided_run_btn.isEnabled() is True


# ---------------------------------------------------------------------------
# Restoration policy (chosen: the safer, non-reviving policy -- see
# photometry_pipeline/guided_plan_identity.py's module docstring and
# gui/main_window.py's _clear_guided_execution_authorization_state).
# ---------------------------------------------------------------------------


def test_restoring_an_old_visible_value_does_not_revive_validation(
    window, tmp_path, monkeypatch
):
    """Validate plan A (CH1+CH2 included), exclude CH2 to produce plan B
    (Run disables), then restore CH2 -- reproducing plan A's visible
    field values exactly. Run must remain disabled: retained validation
    and authorization are cleared on any defining mutation and are never
    revived merely because a field now looks the same again."""
    from PySide6.QtCore import Qt

    _configure_and_validate(window, tmp_path, monkeypatch)
    assert window._guided_run_btn.isEnabled() is True

    def _ch2_item():
        for index in range(window._guided_roi_list.count()):
            item = window._guided_roi_list.item(index)
            if "CH2" in item.text():
                return item
        raise AssertionError("CH2 row not found")

    _ch2_item().setCheckState(Qt.Unchecked)
    _assert_run_disabled_and_guard_refuses(window, monkeypatch)
    assert window._guided_validated_plan_identity is None
    assert window._guided_startup_authority is None
    assert window._guided_startup_transaction_request is None

    # Restore CH2 -- the draft plan's included_roi_ids now exactly matches
    # what was originally validated.
    _ch2_item().setCheckState(Qt.Checked)
    window._refresh_guided_run_readiness_display()

    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_validated_plan_identity is None
    assert window._guided_startup_authority is None
    assert window._guided_startup_transaction_request is None
    starts = []
    monkeypatch.setattr(
        window,
        "_start_guided_run_execution_worker",
        lambda request: starts.append(request),
    )
    window._on_guided_run_clicked_backend_guarded()
    assert starts == []


# ---------------------------------------------------------------------------
# Missing-session integration. The full approve -> revalidate -> succeed ->
# run cycle for a genuine missing session is covered end to end by
# tests/test_guided_missing_session_authorization.py; this test adds the
# complementary fail-closed case and confirms the approval is bound into
# the canonical identity (not just a revision-counter bump).
# ---------------------------------------------------------------------------


def test_missing_session_approval_not_matching_real_content_fails_closed(
    window, tmp_path, monkeypatch
):
    """An approval that does not correspond to a real discovered/missing
    session must refuse revalidation rather than silently authorizing."""
    from photometry_pipeline.guided_new_analysis_plan import (
        GuidedApprovedMissingSession,
    )

    _configure_and_validate(window, tmp_path, monkeypatch)
    approval = GuidedApprovedMissingSession(
        canonical_relative_path="session-2/fluorescence.csv",
        size_bytes=1024,
        sha256_content_digest="c" * 64,
        session_index=2,
        expected_start_time="2026-01-01T00:00:00Z",
        expected_duration_sec=600.0,
    )
    window._add_guided_missing_session_approval(approval)
    assert window._guided_run_btn.isEnabled() is False

    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping

    build_identity = production_mapping.build_application_build_identity(
        distribution_name="photometry-pipeline",
        distribution_version="1.0.0",
        source_revision_kind="git",
        source_revision="abc123",
        source_tree_state="clean",
    )
    monkeypatch.setattr(
        request_builder,
        "resolve_application_build_identity",
        lambda **_kwargs: SimpleNamespace(build_identity=build_identity),
    )
    window._guided_backend_validate_btn.click()
    outcome = window._guided_backend_validation_outcome
    assert outcome.status != "validator_accepted"
    assert window._guided_run_btn.isEnabled() is False
    assert window._guided_startup_authority is None


# ---------------------------------------------------------------------------
# Signal-Only F0 regression: the temporary production capability gate and
# the obsolete post-hoc applied-dF/F route must not have returned.
# ---------------------------------------------------------------------------


def test_no_signal_only_capability_gate_or_applied_dff_route_regression():
    import gui.main_window as main_window_module

    source = Path(main_window_module.__file__).read_text(encoding="utf-8")
    for prohibited in (
        "GuidedExecutionCapabilities",
        "allow_signal_only_f0_execution",
        "guided_execution_capabilities",
        "run_guided_applied_dff_orchestration_if_enabled",
        "This correction approach is not available to run yet.",
    ):
        assert prohibited not in source, prohibited
    import importlib

    assert importlib.util.find_spec(
        "photometry_pipeline.guided_execution_capabilities"
    ) is None
