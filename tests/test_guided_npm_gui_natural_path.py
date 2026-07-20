"""Guided NPM natural-path regression: drives the real Guided New Analysis
wizard for a supported NPM intermittent recording all the way through
Review Plan, real Check My Setup, and Run -- without bypassing Review Plan
by constructing any request directly, and without ever touching Full
Control's `_mode_combo` to select execution behavior.

Guided Mode exposes no phasic-versus-tonic choice (Phase 4C): a Guided NPM
draft always authorizes and runs `execution_mode == "both"`, routed through
the exact same shared startup-request builder, allocation, materialization,
readiness, and wrapper-launch machinery RWD already uses
(guided_execution_request_builder.build_guided_npm_startup_request_from_validation
-> guided_startup_transaction.py -> guided_startup_orchestration.py ->
tools/run_full_pipeline_deliverables.py). The older, bespoke, NPM-only
worker chain (guided_npm_run_launch_builder.py /
guided_npm_worker_launch.py / _on_guided_npm_run_clicked) is proven NOT
invoked by any test in this module -- it remains present but unreachable
from the Guided GUI.

The underlying NPM execution architecture (production mapping, execution
authority, startup payload/claim/persistence) is pre-existing and heavily
tested elsewhere (guided_npm_* test modules); this test does not
re-implement or re-prove that machinery, only that the real Guided wizard
now reaches the shared path with a natural, untouched "both" outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

import gui.main_window as main_window_module
import photometry_pipeline.guided_execution_request_builder as request_builder_module
import photometry_pipeline.guided_npm_run_launch_builder as npm_builder_module
import photometry_pipeline.guided_npm_worker_launch as npm_launch_module
import photometry_pipeline.preview.correction_preview as correction_preview_module
from gui.main_window import GUIDED_WORKFLOW_STEPS, MainWindow
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.guided_new_analysis_plan import (
    evaluate_guided_new_analysis_execution_subset_readiness,
)
from photometry_pipeline.guided_npm_startup_bridge import GuidedStartupAuthority

NPM_ROIS = ("Region0", "Region1", "Region2")


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


def _pump_until(qapp, condition, *, timeout_s: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met before timeout")
        qapp.processEvents()


def _write_real_npm_session(path: Path, *, n_seconds: float, rate_hz: float) -> None:
    """Write one structurally real, LedState-interleaved NPM CSV session
    with multiple ROI columns, at roughly `rate_hz` per channel."""
    step = 1.0 / (2.0 * rate_hz)
    rows = ["Timestamp,LedState," + ",".join(f"{roi}G" for roi in NPM_ROIS)]
    t = 0.0
    led = 1
    n_samples = int(n_seconds * 2.0 * rate_hz)
    for i in range(n_samples):
        values = ",".join(
            f"{10.0 + 0.01 * i + roi_index:.4f}"
            for roi_index in range(len(NPM_ROIS))
        )
        rows.append(f"{t:.4f},{led},{values}")
        t += step
        led = 2 if led == 1 else 1
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _configure_npm_new_analysis_setup(window, tmp_path, monkeypatch):
    """Drive Select data -> Recording structure with a real, minimal NPM
    dataset, faked discovery (matching the established RWD test pattern),
    and confirmed dataset-contract settings sourced from the app's own
    baseline NPM defaults (the real production candidate path -- proven
    by test_npm_dataset_contract_candidate_does_not_call_legacy_inference
    to never call legacy RWD-style inference for NPM). Full Control's
    `_mode_combo` is deliberately left untouched -- the Guided workflow
    never shows it and always authorizes execution_mode == "both"."""
    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()

    input_dir = tmp_path / "npm_raw_input"
    input_dir.mkdir()
    session_files = []
    for index in range(2):
        session_path = input_dir / f"photometryData2026-01-0{index + 1}T12_00_00.csv"
        _write_real_npm_session(session_path, n_seconds=3.0, rate_hz=45.0)
        session_files.append(session_path)

    window._guided_input_dir_edit.setText(str(input_dir))
    window._guided_format_combo.setCurrentText("npm")

    discovery = {
        "resolved_format": "npm",
        "n_total_discovered": len(session_files),
        "n_preview": len(session_files),
        "sessions": [
            {
                "session_id": source_file.stem,
                "path": str(source_file),
                "included_in_preview": True,
            }
            for source_file in session_files
        ],
        "rois": [{"roi_id": roi} for roi in NPM_ROIS],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)

    acq_idx = window._guided_acquisition_mode_combo.findData("intermittent")
    assert acq_idx >= 0
    window._guided_acquisition_mode_combo.setCurrentIndex(acq_idx)
    window._guided_sessions_per_hour_edit.setText("1")
    window._guided_session_duration_edit.setText("3")

    return input_dir, session_files


def _fake_npm_load_chunk(path, input_format, _config, chunk_id, **_kwargs):
    import numpy as np

    time_sec = np.arange(120, dtype=float) / 40.0
    uv = 1.0 + 0.03 * np.sin(time_sec * 0.2)
    sig = 1.2 * uv + 0.05 * np.sin(time_sec * 0.9)
    return Chunk(
        chunk_id=chunk_id,
        source_file=path,
        format=input_format,
        time_sec=time_sec,
        uv_raw=np.column_stack([uv] * len(NPM_ROIS)),
        sig_raw=np.column_stack([sig * (1.0 + 0.01 * i) for i in range(len(NPM_ROIS))]),
        fs_hz=40.0,
        channel_names=list(NPM_ROIS),
        metadata={},
    )


def _confirm_npm_correction_strategies(window, qapp, monkeypatch, *, included_rois):
    monkeypatch.setattr(
        correction_preview_module, "load_chunk", _fake_npm_load_chunk
    )
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    for roi in included_rois:
        roi_idx = window._guided_preview_roi_combo.findData(roi)
        assert roi_idx >= 0
        window._guided_preview_roi_combo.setCurrentIndex(roi_idx)
        assert window._guided_preview_generate_btn.isEnabled()
        window._guided_preview_generate_btn.click()
        _pump_until(
            qapp, lambda: window._guided_correction_preview_thread is None
        )
        result = window._guided_preview_last_result
        assert result["status"] in {"success", "partial"}, result

        window._guided_confirm_roi_combo.setCurrentIndex(
            window._guided_confirm_roi_combo.findData(roi)
        )
        window._guided_confirm_chunk_combo.setCurrentIndex(0)
        strategy_index = window._guided_confirm_strategy_combo.findText(
            "Global Linear Regression"
        )
        assert strategy_index >= 0
        window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
        window._guided_confirm_ack_cb.setChecked(True)
        assert window._guided_confirm_mark_btn.isEnabled()
        window._guided_confirm_mark_btn.click()


def _drive_npm_to_check_my_setup(
    window,
    tmp_path,
    monkeypatch,
    qapp,
    *,
    apply_feature_defaults: bool,
):
    """Drive the real Guided New Analysis wizard for NPM from Select data
    through a real Check My Setup click, returning (outcome, output_dir,
    included_rois, excluded_roi). Reused by every natural-path test in
    this module -- the only real variable between them is whether Feature
    Detection defaults are explicitly applied, both supported, real
    scientist-facing states (see _feature_event_profile_current_for_first_subset:
    a loaded default profile is current whether left as
    "default_initialized" or explicitly applied). No mode is ever
    selected: the accepted authorization always naturally carries
    execution_mode == "both"."""
    _configure_npm_new_analysis_setup(window, tmp_path, monkeypatch)

    output_dir = tmp_path / "npm_output"
    output_dir.mkdir()
    window._guided_output_dir_edit.setText(str(output_dir))

    included_rois = NPM_ROIS[:2]  # exclude Region2
    excluded_roi = NPM_ROIS[2]
    for index in range(window._guided_roi_list.count()):
        item = window._guided_roi_list.item(index)
        if item.text() == excluded_roi:
            item.setCheckState(main_window_module.Qt.Unchecked)

    _confirm_npm_correction_strategies(
        window, qapp, monkeypatch, included_rois=included_rois
    )

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Feature detection")
    )
    if apply_feature_defaults:
        window._guided_feature_event_apply_btn.click()
    # else: leave the auto-loaded valid Default profile as
    # "default_initialized" -- never explicitly applied. The real Guided
    # wizard already presents this state as complete (Continue is
    # enabled) without requiring the Apply click.

    # Confirm the detected NPM dataset settings.
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.execution_intent.execution_mode == "both"
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    assert any(
        issue.category in ("missing_npm_channel_mapping", "missing_npm_dataset_contract")
        for issue in subset.blocking_issues
    )
    status_text = window._guided_review_plan_status_label.text()
    assert "does not yet support this configuration" not in status_text
    assert "have not been confirmed yet" in status_text
    assert window._guided_review_dataset_contract_action_btn.isHidden() is False

    window._guided_dataset_contract_apply_btn.click()
    plan = window._build_guided_new_analysis_draft_plan()
    assert plan.dataset_contract_snapshot.current_applied is True
    assert plan.execution_intent.execution_mode == "both"
    subset = evaluate_guided_new_analysis_execution_subset_readiness(plan)
    assert subset.first_subset_executable is True

    # Trigger the same refresh the real navigation path uses.
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Feature detection")
    )
    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    status_text = window._guided_review_plan_status_label.text()
    assert "does not yet support this configuration" not in status_text
    assert "have not been confirmed yet" not in status_text
    assert "This plan is ready" in status_text

    window._guided_review_go_to_run_btn.click()

    import photometry_pipeline.guided_production_mapping as production_mapping
    from types import SimpleNamespace

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
    window._guided_backend_validate_btn.click()

    outcome = window._guided_backend_validation_outcome
    assert outcome.status == "validator_accepted", outcome.blocking_issues
    assert outcome.accepted_for_backend_validation is True

    assert (
        "not available yet" not in window._guided_backend_validation_status_label.text()
    )
    assert window._guided_run_btn.isEnabled() is True
    assert window._guided_run_readiness_label.text() == "Guided Run is ready to start."

    authority = window._guided_startup_authority
    assert isinstance(authority, GuidedStartupAuthority)
    assert authority.is_npm is True
    assert authority.execution_mode == "both"

    return outcome, output_dir, included_rois, excluded_roi


def _press_run_and_capture_shared_dispatch(window, monkeypatch, qapp):
    """Click Run and capture the single shared-path dispatch call
    (`_start_guided_run_execution_worker`, the exact same format-neutral
    entry point RWD uses) without letting the worker thread actually
    start a subprocess. Also proves the bespoke NPM-only worker chain
    (build_guided_npm_worker_prelaunch_claim_from_validation,
    launch_guided_npm_worker_runtime, _on_guided_npm_run_clicked) is never
    invoked -- no fallback to the pre-4C bespoke path.

    Returns the single captured GuidedStartupTransactionRequest."""

    def _fail_if_called(name):
        def _fail(*_args, **_kwargs):
            raise AssertionError(f"bespoke NPM worker path must not be called: {name}")
        return _fail

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fail_if_called("build_guided_npm_worker_prelaunch_claim_from_validation"),
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        _fail_if_called("launch_guided_npm_worker_runtime"),
    )
    monkeypatch.setattr(
        main_window_module.MainWindow,
        "_on_guided_npm_run_clicked",
        _fail_if_called("_on_guided_npm_run_clicked"),
    )

    dispatched = []
    monkeypatch.setattr(
        window,
        "_start_guided_run_execution_worker",
        lambda request: dispatched.append(request),
    )
    monkeypatch.setattr(window, "_start_guided_run_live_status", lambda *_: None)

    window._guided_run_btn.click()

    assert len(dispatched) == 1
    return dispatched[0]


def test_natural_path_npm_reaches_check_my_setup_and_dispatches_shared_run(
    window, tmp_path, monkeypatch, qapp
):
    """Explicit-Apply natural path: a genuine Guided NPM setup authorizes
    execution_mode == "both" with no mode selection anywhere, and one Run
    press reaches exactly one shared-path dispatch -- not the bespoke
    worker chain."""
    outcome, output_dir, included_rois, excluded_roi = (
        _drive_npm_to_check_my_setup(
            window,
            tmp_path,
            monkeypatch,
            qapp,
            apply_feature_defaults=True,
        )
    )

    request = _press_run_and_capture_shared_dispatch(window, monkeypatch, qapp)

    assert request.startup_authority.is_npm is True
    assert request.startup_authority.execution_mode == "both"
    assert request.startup_authority.included_roi_ids == tuple(included_rois)
    assert Path(request.planned_allocated_run_dir).parent == output_dir.resolve()
    assert window._guided_backend_execution_active is True

    # No numerical execution and no allocation: the worker itself was
    # replaced above, so the wrapper subprocess is never spawned and the
    # planned run directory is never created.
    assert not Path(request.planned_allocated_run_dir).exists()


def test_natural_path_npm_loaded_defaults_without_apply_dispatches_shared_run(
    window, tmp_path, monkeypatch, qapp
):
    """Repair regression: reproduces Jeff's real defect exactly.

    A loaded Default Feature Detection profile left as
    "default_initialized" (never explicitly applied -- the GUI already
    presents this as complete, and Check My Setup already accepts it,
    exactly as the real diagnostic evidence showed) must still reach a
    successful, unmocked production mapping / execution authority / shared
    startup-request build -- not the false per_roi_feature_entry_incomplete
    / feature_authority_incomplete refusals both existing before this
    repair, and still with a natural execution_mode == "both"."""
    outcome, output_dir, included_rois, excluded_roi = (
        _drive_npm_to_check_my_setup(
            window,
            tmp_path,
            monkeypatch,
            qapp,
            apply_feature_defaults=False,
        )
    )

    request = _press_run_and_capture_shared_dispatch(window, monkeypatch, qapp)
    assert request.startup_authority.execution_mode == "both"

    # Every included ROI has a complete feature/event mapping; the
    # excluded ROI is absent from the request the wrapper would consume.
    validated_request = outcome.compile_result.request
    mapped_roi_ids = {
        entry.roi_id for entry in validated_request.feature_event.per_roi_feature_event_map
    }
    assert mapped_roi_ids == set(included_rois)
    assert excluded_roi not in mapped_roi_ids
    for entry in validated_request.feature_event.per_roi_feature_event_map:
        # Truthful provenance: the scientist never pressed Apply, so this
        # default-sourced entry stays explicit_user_mark=False. It is
        # still accepted because the enclosing profile is current and
        # default_initialized (see feature_entry_provenance_valid).
        assert entry.source == "default"
        assert entry.explicit_user_mark is False
        assert entry.current_or_stale == "current"
        assert entry.effective_config_fields


def test_natural_path_npm_run_refuses_after_post_setup_source_rename(
    window, tmp_path, monkeypatch, qapp
):
    """Section 10: the live prelaunch freshness recheck re-validates the
    NPM source against the accepted authorization before allocation. A
    source rename after Check My Setup (no GUI field touched) must refuse
    Run and must not reach the shared dispatch -- proving the recheck
    actually re-runs validate_current_guided_draft_for_backend against the
    live NPM source rather than trusting stale retained state."""
    _, output_dir, _included_rois, _excluded_roi = _drive_npm_to_check_my_setup(
        window,
        tmp_path,
        monkeypatch,
        qapp,
        apply_feature_defaults=True,
    )
    assert window._guided_run_btn.isEnabled() is True

    input_dir = tmp_path / "npm_raw_input"
    original = input_dir / "photometryData2026-01-02T12_00_00.csv"
    moved = input_dir / "photometryData2026-01-03T12_00_00.csv"
    original.rename(moved)

    assert window._guided_run_btn.isEnabled()  # no GUI field changed

    dispatched = []
    monkeypatch.setattr(
        window,
        "_start_guided_run_execution_worker",
        lambda request: dispatched.append(request),
    )
    window._on_guided_run_clicked_backend_guarded()

    assert dispatched == []
    assert not window._guided_backend_execution_active
    assert window._guided_startup_authority is None
    assert window._guided_startup_transaction_request is None
    assert "recording sessions changed" in (
        window._guided_run_readiness_label.text().lower()
    )


def _npm_shared_completion_runner(monkeypatch):
    """Drive the REAL wrapper (tools/run_full_pipeline_deliverables.py)
    in-process, exactly as tests/test_guided_gui_run_completed_boundary.py::
    _completion_runner does for RWD -- real preallocated-mode validation,
    real manifest/normalized-recording gate calls (stubbed only at their
    own boundary, matching the existing RWD pattern), real orchestration
    of both components for execution_mode == "both". Only the innermost
    numerical subprocess boundary (the analyze_photometry.py call for each
    component) is replaced with a deterministic fixture writer
    (terminal_run_fixtures.seed_wrapper_analysis_outputs /
    seed_wrapper_deliverables) that produces the same shared completed-run
    artifact shape a real analysis leaves behind -- for both the phasic
    and tonic branches -- without running real numerical analysis."""
    import tools.run_full_pipeline_deliverables as wrapper
    from tests.terminal_run_fixtures import (
        seed_wrapper_analysis_outputs,
        seed_wrapper_deliverables,
        write_phasic_feature_outputs,
    )

    calls = {"analysis_stub": 0}
    original_preallocated_validation = wrapper.validate_guided_preallocated_mode_args
    original_config_loader = wrapper.Config.from_yaml

    def _extend_trace_cache_to_two_sessions(cache_path: Path) -> None:
        """The generic seed_wrapper_analysis_outputs fixture writes a
        single-chunk cache; this test's real NPM setup discovers two real
        sessions (_configure_npm_new_analysis_setup writes two session
        files), so the wrapper's own authoritative-session-plot step
        expects a second cache contribution to exist. Add it here rather
        than in the shared fixture, which other single-session tests
        still rely on unchanged."""
        import h5py
        import numpy as np

        with h5py.File(cache_path, "a") as cache:
            del cache["meta/chunk_ids"]
            cache["meta"].create_dataset("chunk_ids", data=np.array([0, 1], dtype=np.int64))
            source = cache["roi/Region0/chunk_0"]
            dest = cache.create_group("roi/Region0/chunk_1")
            for name, dataset in source.items():
                dest.create_dataset(name, data=dataset[()])

    def _write_matching_cache_source_files(cache_path: Path, run_dir: Path) -> None:
        """The generic seed_wrapper_analysis_outputs cache stamps one
        placeholder meta/source_files entry; the shared completion
        contract's per-cache-chunk source verification
        (run_completion_contract._correction_completion_error_for_analysis)
        requires one entry per real chunk, matching the same authorized
        session source identities already written into the completeness
        record below."""
        import json as _json

        import h5py
        import numpy as np

        description = _json.loads(
            (run_dir / "guided_normalized_recording_description.json").read_text(
                encoding="utf-8"
            )
        )
        sources_by_position = {
            session["chronological_position"]: session["canonical_source_reference"]
            for session in description["sessions"]
        }
        with h5py.File(cache_path, "a") as cache:
            chunk_ids = [int(c) for c in cache["meta/chunk_ids"][()]]
            sources = [sources_by_position[position].encode("utf-8") for position in chunk_ids]
            del cache["meta/source_files"]
            cache["meta"].create_dataset("source_files", data=np.array(sources))
            for roi_name in cache["roi"]:
                for chunk_id in chunk_ids:
                    group = cache[f"roi/{roi_name}/chunk_{chunk_id}"]
                    group.attrs["source_file"] = sources_by_position[chunk_id]

    def _extend_trace_cache_to_two_rois(cache_path: Path) -> None:
        """The generic seed_wrapper_analysis_outputs fixture writes a
        single-ROI ("Region0") cache; this test authorizes two ROIs
        (Region0, Region1 -- see seed_wrapper_deliverables below). Mirror
        Region0's chunks into a second ROI so the cache's processed ROI
        set matches the authorized included ROI set."""
        import h5py
        import numpy as np

        with h5py.File(cache_path, "a") as cache:
            del cache["meta/rois"]
            cache["meta"].create_dataset(
                "rois", data=np.array([b"Region0", b"Region1"])
            )
            for chunk_name in list(cache["roi/Region0"]):
                source = cache[f"roi/Region0/{chunk_name}"]
                dest = cache.create_group(f"roi/Region1/{chunk_name}")
                for name, dataset in source.items():
                    dest.create_dataset(name, data=dataset[()])

    def _write_matching_npm_completeness_record(run_dir: Path) -> None:
        """seed_wrapper_analysis_outputs writes a generic placeholder
        completeness record (source "/frozen/chunk_0000"), built for
        RWD-shaped wrapper tests that never carry Guided normalized
        recording provenance. This test's run genuinely authorizes an NPM
        recording (guided_normalized_recording_description.json is
        written by real materialization before this stub ever runs), so
        the shared completion comparator holds the analysis-consumed
        evidence to that real authorization. Rebuild the completeness
        record from the same authorized session facts already on disk,
        rather than from an unrelated placeholder."""
        import json as _json

        from photometry_pipeline.input_processing_completeness import (
            INPUT_COMPLETENESS_CONTRACT_VERSION,
            INPUT_COMPLETENESS_FILENAME,
        )

        description_path = run_dir / "guided_normalized_recording_description.json"
        description = _json.loads(description_path.read_text(encoding="utf-8"))
        expected = []
        processed = []
        for session in description["sessions"]:
            entry = {
                "source": session["canonical_source_reference"],
                "size_bytes": session["size_bytes"],
                "sha256": session["content_digest"],
                "index": session["chronological_position"],
                "disposition": session["disposition"],
                "expected_start_time": session["authoritative_source_start_time"],
                "expected_duration_sec": session["expected_duration_sec"],
            }
            expected.append(entry)
            if session["disposition"] == "process":
                processed.append(
                    {
                        "index": entry["index"],
                        "source": entry["source"],
                        "cache_chunk_id": entry["index"],
                    }
                )
        record = {
            "contract_version": INPUT_COMPLETENESS_CONTRACT_VERSION,
            "acquisition_mode": description["acquisition_mode"],
            "input_format": "npm",
            "expected": expected,
            "processed": processed,
            "missing": [],
        }
        for analysis_out in ("phasic_out", "tonic_out"):
            (run_dir / "_analysis" / analysis_out / INPUT_COMPLETENESS_FILENAME).write_text(
                _json.dumps(record), encoding="utf-8"
            )
        return description

    def _stamp_authorized_chunk_evidence(
        cache_path: Path, target_fs_hz: float, dynamic_fit_mode: str
    ) -> None:
        """A real Guided-authorized NPM execution always stamps fs_hz, the
        combined output_time_basis, and per-chunk correction-execution
        evidence onto every chunk (see
        Pipeline._bind_authorized_chunk_chronology and
        Pipeline._build_requested_correction_provenance); the generic
        seed_wrapper_analysis_outputs stub cache does not. Stamp the same
        facts this stub's completeness/correction-provenance records now
        authorize, matching run_completion_contract's per-chunk
        dynamic-fit verification exactly."""
        import h5py

        from photometry_pipeline.io.npm_contract import (
            NPM_GUIDED_COMBINED_OUTPUT_TIME_BASIS,
        )

        with h5py.File(cache_path, "a") as cache:
            for roi_name in cache["roi"]:
                for chunk_name in cache[f"roi/{roi_name}"]:
                    attrs = cache[f"roi/{roi_name}/{chunk_name}"].attrs
                    attrs["fs_hz"] = target_fs_hz
                    attrs["output_time_basis"] = NPM_GUIDED_COMBINED_OUTPUT_TIME_BASIS
                    attrs["correction_execution_status"] = "consumed"
                    attrs["correction_strategy_family"] = "dynamic_fit"
                    attrs["correction_selected_strategy"] = dynamic_fit_mode
                    attrs["correction_dynamic_fit_mode"] = dynamic_fit_mode
                    attrs["dynamic_fit_mode_resolved"] = dynamic_fit_mode
                    attrs["dynamic_fit_engine"] = "rolling_local_regression"

    def _write_matching_correction_provenance(
        run_dir: Path, description: dict, analysis_kind: str, dynamic_fit_mode: str
    ) -> None:
        """The real numerical subprocess this test's stub replaces always
        writes correction-provenance evidence into both run_metadata.json
        and run_report.json (see Pipeline._build_requested_correction_
        provenance); the shared completed-run Review loader
        (completed_run_review.load_completed_review_overview) requires it
        before a real or stubbed run can be opened for review. Build it
        with the same production helper Pipeline itself uses
        (regression.build_uniform_per_roi_correction_map), from the real
        accepted request's included ROI set -- never canned values."""
        import json as _json

        from photometry_pipeline.core import regression
        from photometry_pipeline.run_completion_contract import (
            CORRECTION_PROVENANCE_SCHEMA_VERSION,
        )

        included_roi_ids = [
            item["roi_id"] for item in description["roi_channels"] if item["included"]
        ]
        strategy_map = regression.build_uniform_per_roi_correction_map(
            included_roi_ids, dynamic_fit_mode
        )
        requested_by_roi = [
            {
                "roi_id": strategy_map[roi].roi_id,
                "strategy_family": strategy_map[roi].strategy_family,
                "selected_strategy": strategy_map[roi].selected_strategy,
                "dynamic_fit_mode": strategy_map[roi].dynamic_fit_mode,
                "parameter_identity": strategy_map[roi].parameter_identity,
                "evidence_identity": strategy_map[roi].evidence_identity,
            }
            for roi in included_roi_ids
        ]
        provenance = {
            "schema_version": CORRECTION_PROVENANCE_SCHEMA_VERSION,
            "source": "legacy_uniform_translation",
            "analysis_mode": analysis_kind,
            "included_roi_ids": included_roi_ids,
            "requested_by_roi": requested_by_roi,
            "finite_coverage_fraction": 0.8,
        }
        analysis_out = run_dir / "_analysis" / f"{analysis_kind}_out"
        metadata_path = analysis_out / "run_metadata.json"
        metadata = (
            _json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        metadata["correction_provenance"] = provenance
        metadata_path.write_text(_json.dumps(metadata), encoding="utf-8")
        report_path = analysis_out / "run_report.json"
        report = _json.loads(report_path.read_text(encoding="utf-8"))
        report.setdefault("derived_settings", {})
        report["derived_settings"]["correction_provenance"] = provenance
        report_path.write_text(_json.dumps(report), encoding="utf-8")

    def runner(command):
        monkeypatch.setattr(wrapper.sys, "argv", [command[1], *command[2:]])

        def validate_preallocated(args):
            return original_preallocated_validation(args)

        def verify_live(_args):
            return object(), object()

        def verify_normalized_recording_live(_args, _facts, _verified):
            return None

        def validate_inputs(_args):
            return None

        def load_config(path):
            return original_config_loader(path)

        def run_cmd(command_argv, roi_label=None):
            if Path(command_argv[1]).name == "analyze_photometry.py":
                calls["analysis_stub"] += 1
                output_dir = Path(command_argv[command_argv.index("--out") + 1])
                run_dir = output_dir.parents[1]
                seed_wrapper_analysis_outputs(run_dir)
                # seed_wrapper_analysis_outputs writes features.csv for a
                # single default ROI; the wrapper freezes its own
                # authoritative expected-ROI set from this file (see
                # run_full_pipeline_deliverables.py's `regions =
                # sorted(df_feat['roi'].unique())`), so it must reflect
                # every ROI this run actually authorizes.
                write_phasic_feature_outputs(
                    run_dir / "_analysis" / "phasic_out", rois=list(NPM_ROIS[:2])
                )
                seed_wrapper_deliverables(run_dir, list(NPM_ROIS[:2]), tonic=True)
                _extend_trace_cache_to_two_sessions(
                    run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
                )
                _extend_trace_cache_to_two_sessions(
                    run_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5"
                )
                _extend_trace_cache_to_two_rois(
                    run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5"
                )
                _extend_trace_cache_to_two_rois(
                    run_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5"
                )
                _write_matching_cache_source_files(
                    run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5",
                    run_dir,
                )
                _write_matching_cache_source_files(
                    run_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5",
                    run_dir,
                )
                description = _write_matching_npm_completeness_record(run_dir)
                target_fs_hz = float(description["sampling"]["target_fs_hz"])
                config_path = command_argv[command_argv.index("--config") + 1]
                dynamic_fit_mode = original_config_loader(
                    config_path
                ).dynamic_fit_mode
                _stamp_authorized_chunk_evidence(
                    run_dir / "_analysis" / "phasic_out" / "phasic_trace_cache.h5",
                    target_fs_hz,
                    dynamic_fit_mode,
                )
                _stamp_authorized_chunk_evidence(
                    run_dir / "_analysis" / "tonic_out" / "tonic_trace_cache.h5",
                    target_fs_hz,
                    dynamic_fit_mode,
                )
                _write_matching_correction_provenance(
                    run_dir, description, "phasic", dynamic_fit_mode
                )
                _write_matching_correction_provenance(
                    run_dir, description, "tonic", dynamic_fit_mode
                )
            return {
                "cmd": command_argv,
                "started_utc": "2026-07-02T00:00:00Z",
                "finished_utc": "2026-07-02T00:00:00Z",
                "elapsed_sec": 0.0,
                "returncode": 0,
                "roi_label": roi_label,
            }

        monkeypatch.setattr(
            wrapper, "validate_guided_preallocated_mode_args", validate_preallocated
        )
        monkeypatch.setattr(wrapper, "verify_guided_manifest_before_output", verify_live)
        monkeypatch.setattr(
            wrapper,
            "verify_guided_normalized_recording_description_before_output",
            verify_normalized_recording_live,
        )
        monkeypatch.setattr(wrapper, "validate_inputs", validate_inputs)
        monkeypatch.setattr(wrapper.Config, "from_yaml", staticmethod(load_config))
        monkeypatch.setattr(wrapper, "run_cmd", run_cmd)
        monkeypatch.setattr(wrapper, "_GUIDED_TEST_STOP_AFTER_INITIAL_STATUS", None)
        import photometry_pipeline.guided_startup_orchestration as orchestration

        try:
            wrapper.main()
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            return orchestration.GuidedWrapperProcessResult(
                returncode=code,
                stdout="",
                stderr="wrapper exited",
                command=command,
                started=True,
                completed=True,
            )
        return orchestration.GuidedWrapperProcessResult(
            returncode=0,
            stdout="wrapper completed",
            stderr="",
            command=command,
            started=True,
            completed=True,
        )

    return runner, calls


def test_natural_path_npm_reaches_shared_completion_and_results_handoff(
    window, tmp_path, monkeypatch, qapp
):
    """Final Phase 4C acceptance test: a genuine accepted Guided NPM
    startup request traverses the FULL mature shared path beyond GUI
    dispatch -- real allocation, real materialization, the real wrapper's
    own orchestration of both components (execution_mode == "both"), real
    shared completion classification, and the ordinary in-app Guided
    Review/Results handoff -- not just transaction dispatch. Only the
    innermost numerical subprocess boundary is replaced with a
    deterministic fixture; no bespoke NPM worker/claim/receipt/launch/
    reconciliation function is ever called, and no NPM-specific
    open-folder-only handoff is used."""
    import gui.main_window as main_window_module
    import photometry_pipeline.guided_npm_run_launch_builder as npm_builder_module
    import photometry_pipeline.guided_npm_worker_launch as npm_launch_module

    outcome, output_dir, included_rois, excluded_roi = _drive_npm_to_check_my_setup(
        window,
        tmp_path,
        monkeypatch,
        qapp,
        apply_feature_defaults=True,
    )
    request = window._guided_startup_transaction_request
    assert request.startup_authority.is_npm is True
    assert request.startup_authority.execution_mode == "both"

    from photometry_pipeline.guided_startup_transaction import (
        plan_guided_startup_transaction,
    )

    plan = plan_guided_startup_transaction(request)
    assert plan.ok is True, plan.blocking_issues
    argv = plan.command_plan.argv
    assert argv[argv.index("--format") + 1] == "npm"
    assert argv[argv.index("--mode") + 1] == "both"

    def _fail_if_called(name):
        def _fail(*_args, **_kwargs):
            raise AssertionError(f"bespoke NPM worker path must not be called: {name}")
        return _fail

    monkeypatch.setattr(
        npm_builder_module,
        "build_guided_npm_worker_prelaunch_claim_from_validation",
        _fail_if_called("build_guided_npm_worker_prelaunch_claim_from_validation"),
    )
    monkeypatch.setattr(
        npm_launch_module,
        "launch_guided_npm_worker_runtime",
        _fail_if_called("launch_guided_npm_worker_runtime"),
    )
    monkeypatch.setattr(
        main_window_module.MainWindow,
        "_on_guided_npm_run_clicked",
        _fail_if_called("_on_guided_npm_run_clicked"),
    )

    runner, calls = _npm_shared_completion_runner(monkeypatch)
    window._guided_backend_execution_runner = runner

    window._guided_run_btn.click()
    assert window._guided_backend_execution_active is True
    _pump_until(
        qapp,
        lambda: window._guided_run_execution_thread is None,
        timeout_s=60.0,
    )
    assert window._guided_backend_execution_active is False

    result = window._guided_backend_execution_result
    assert result.status == "wrapper_completed_needs_review_loading", (
        tuple(issue.message for issue in result.blocking_issues), result.diagnostics
    )
    assert calls["analysis_stub"] == 2  # phasic component + tonic component
    run_dir = Path(result.completed_run_candidate_path)
    from gui.run_report_parser import classify_completed_run_candidate

    assert classify_completed_run_candidate(str(run_dir))[0] is True

    # The ordinary in-app run-review/Results loading path -- not the
    # NPM-specific open-folder-only handoff -- is what surfaces next.
    assert window._guided_load_completed_run_for_review_btn.isHidden() is False
    assert not hasattr(window, "_guided_npm_open_output_btn") or (
        window._guided_npm_open_output_btn.isHidden() is True
    )
    # The persisted completed-output format must be truthful, not the
    # prior hardcoded "rwd" -- _refresh_guided_run_readiness_display's
    # stale-handoff cleanup below only recognizes a completed NPM state
    # via this exact flag.
    assert window._guided_completed_output_format == "npm"

    # The stubbed numerical subprocess must leave behind the same
    # correction-provenance evidence a real run would, matching across
    # run_metadata.json and run_report.json exactly as the shared Review
    # loader requires (completed_run_review._validate_requested_provenance).
    for branch in ("phasic", "tonic"):
        branch_dir = run_dir / "_analysis" / f"{branch}_out"
        metadata_provenance = json.loads(
            (branch_dir / "run_metadata.json").read_text(encoding="utf-8")
        )["correction_provenance"]
        report_provenance = json.loads(
            (branch_dir / "run_report.json").read_text(encoding="utf-8")
        )["derived_settings"]["correction_provenance"]
        assert metadata_provenance == report_provenance
        assert metadata_provenance["analysis_mode"] == branch

    window._guided_load_completed_run_for_review_btn.click()
    _pump_until(
        qapp,
        lambda: window._guided_completed_review_load_thread is None,
        timeout_s=30.0,
    )

    assert window._guided_workflow_mode == "open_results"
    assert window._current_run_dir == str(run_dir)
    # The ordinary "Load completed run for review" click loads the real
    # compact completed-run overview (completed_run_review.
    # load_completed_review_overview) -- deliberately never opening an
    # HDF5 cache or materializing a full-resolution trace at this step
    # (see RunReportViewer.load_report's `_completed_review_overview`
    # branch). phasic_review_model is the separate, heavier full-trace
    # model populated only by an explicit later per-ROI load, so it is
    # correctly still None here; the real proof of a successful ordinary
    # Results handoff is the accepted overview itself.
    viewer = window._guided_report_viewer
    assert viewer.phasic_review_model is None
    overview = viewer._completed_review_overview
    assert set(overview["analysis_branches"]) == {"phasic", "tonic"}
    assert overview["format"] == "npm"

    # A stale completed NPM output handoff must never survive into a
    # different/new setup being prepared -- the same readiness refresh
    # ordinary setup changes trigger must recognize and clear it, now
    # that _guided_completed_output_format truthfully records "npm"
    # rather than the prior hardcoded "rwd".
    window._refresh_guided_run_readiness_display()
    assert window._guided_npm_completed_output_dir is None
    assert window._guided_completed_output_format is None
    assert not hasattr(window, "_guided_npm_open_output_btn") or (
        window._guided_npm_open_output_btn.isHidden() is True
        and window._guided_npm_open_output_btn.isEnabled() is False
    )
