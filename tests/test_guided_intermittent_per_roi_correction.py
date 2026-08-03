from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import (
    GUIDED_REFERENCE_CORRECTION_CARD_TO_MODE,
    GUIDED_WORKFLOW_STEPS,
    MainWindow,
)
from photometry_pipeline.run_completion_contract import (
    SUCCESS_STATES,
    classify_run_terminal_state,
)
from tests.test_guided_gui_run_execution_wiring import _pump_until


pytestmark = pytest.mark.extended


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow()
    yield instance
    instance._guided_backend_execution_active = False
    thread = getattr(instance, "_guided_run_execution_thread", None)
    if thread is not None and thread.isRunning():
        thread.quit()
        thread.wait(2000)
    instance.close()
    instance.deleteLater()


def _configure_intermittent_csv_draft(
    window,
    tmp_path: Path,
    *,
    strategy_by_roi: dict[str, str],
) -> None:
    """Drive the ordinary mapped-CSV Guided workflow to an authorized draft."""
    from gui.synthetic_demo_generator import generate_guided_csv_demo

    window._guided_workflow_stepper.setCurrentRow(0)
    window._guided_start_setup_btn.click()

    generated = generate_guided_csv_demo(
        tmp_path / "generated_source",
        _session_count=2,
        _rows_per_session=12000,
    )
    assert generated.success, generated.message
    source_root = generated.input_dir
    preview_output = tmp_path / "preview_output"
    preview_output.mkdir()
    output_parent = tmp_path / "output_parent"
    output_parent.mkdir()
    source_files = sorted(source_root.glob("session_*.csv"))

    window._guided_format_combo.setCurrentText("custom_tabular")
    window._guided_input_dir_edit.setText(str(source_root))
    window._guided_output_dir_edit.setText(str(preview_output))
    window._guided_acquisition_mode_combo.setCurrentIndex(
        window._guided_acquisition_mode_combo.findData("intermittent")
    )
    window._guided_timeline_mode_combo.setCurrentIndex(
        window._guided_timeline_mode_combo.findData("fixed_daily_anchor")
    )
    window._guided_fixed_daily_anchor_clock_edit.setText("07:00")
    window._guided_recording_start_clock_edit.setText("12:00:00")
    window._refresh_guided_csv_source_interpretation()
    window._guided_csv_time_column_combo.setCurrentText("ElapsedSeconds")
    window._guided_csv_time_units_combo.setCurrentText("seconds")

    mapping_row = window._guided_csv_mapping_rows[0]
    mapping_row["name"].setText("ROI1")
    mapping_row["signal"].setCurrentText("ROI1_Signal")
    mapping_row["reference"].setCurrentText("ROI1_Reference")
    window._add_guided_csv_mapping_row()
    mapping_row = window._guided_csv_mapping_rows[1]
    mapping_row["name"].setText("ROI2")
    mapping_row["signal"].setCurrentText("ROI2_Signal")
    mapping_row["reference"].setCurrentText("ROI2_Reference")
    window._guided_csv_order_confirm_cb.setChecked(True)
    discovery = {
        "resolved_format": "custom_tabular",
        "n_total_discovered": 2,
        "n_preview": 2,
        "sessions": [
            {
                "index": index,
                "session_id": source.stem,
                "path": str(source),
                "included_in_preview": True,
            }
            for index, source in enumerate(source_files)
        ],
        "rois": [{"roi_id": "ROI1"}, {"roi_id": "ROI2"}],
    }
    window._discovery_cache = discovery
    window._populate_discovery_ui(discovery)
    window._guided_sessions_per_hour_edit.setText("2")
    window._guided_session_duration_edit.setText("600")

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Correction approach")
    )
    for roi_id, strategy_text in strategy_by_roi.items():
        window._guided_preview_roi_combo.setCurrentIndex(
            window._guided_preview_roi_combo.findData(roi_id)
        )
        strategy_index = window._guided_confirm_strategy_combo.findText(
            strategy_text
        )
        assert strategy_index >= 0, strategy_text
        window._guided_confirm_strategy_combo.setCurrentIndex(strategy_index)
        window._guided_preview_generate_btn.click()
        result = window._guided_preview_last_result
        assert result["status"] in {"success", "partial"}, result

        row = window._guided_local_preview_confirmation_rows[roi_id]
        strategy_value = (
            "signal_only_f0"
            if strategy_text == "Signal-Only F0"
            else GUIDED_REFERENCE_CORRECTION_CARD_TO_MODE[strategy_text]
        )
        row_strategy_index = row["strategy_combo"].findData(strategy_value)
        assert row_strategy_index >= 0, (roi_id, strategy_text, strategy_value)
        row["strategy_combo"].setCurrentIndex(row_strategy_index)
        if strategy_text == "Signal-Only F0":
            candidate = window._guided_local_preview_locked_evidence_for_roi(
                roi_id, "signal_only_f0"
            )
        else:
            candidate = result
        assert row["action_button"].isEnabled(), {
            key: candidate.get(key)
            for key in (
                "valid",
                "selectable",
                "locked",
                "current_or_stale",
                "strategy_family",
                "selected_strategy",
                "dynamic_fit_mode",
                "issues",
                "warnings",
            )
        }
        row["action_button"].click()

    # The final visible confirmation rows are the authority used by the
    # backend materializer. Reconfirm any row rebuilt after the second ROI.
    window._rebuild_guided_local_preview_confirmation_rows()
    for roi_id, strategy_text in strategy_by_roi.items():
        row = window._guided_local_preview_confirmation_rows[roi_id]
        if row["action_button"].text() == "Confirmed":
            continue
        strategy_value = (
            "signal_only_f0"
            if strategy_text == "Signal-Only F0"
            else GUIDED_REFERENCE_CORRECTION_CARD_TO_MODE[strategy_text]
        )
        row_strategy_index = row["strategy_combo"].findData(strategy_value)
        assert row_strategy_index >= 0, (roi_id, strategy_text, strategy_value)
        row["strategy_combo"].setCurrentIndex(row_strategy_index)
        assert row["action_button"].isEnabled()
        row["action_button"].click()

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Feature detection")
    )
    window._guided_feature_event_apply_btn.click()
    for roi_id in strategy_by_roi:
        window._guided_feature_preview_roi_combo.setCurrentText(roi_id)
        window._guided_feature_preview_generate_btn.click()
        preview = window._guided_feature_preview_last_result
        assert preview is not None
        assert (
            len(preview.positive_peak_indices)
            + len(preview.negative_peak_indices)
        ) > 0

    window._guided_workflow_stepper.setCurrentRow(
        list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
    )
    window._guided_review_dataset_contract_action_btn.click()
    output_target = output_parent / "guided_csv_output"
    window._guided_output_path_edit.setText(str(output_target))
    window._guided_output_apply_btn.click()
    assert window._guided_new_analysis_output_policy_status == "applied", (
        window._guided_output_status_label.text()
    )
    window._guided_review_go_to_run_btn.click()


def _assert_persisted_correction_output(
    run_dir: Path, strategy_by_roi: dict[str, str]
) -> None:
    assert classify_run_terminal_state(str(run_dir)).state in SUCCESS_STATES
    expected_selected = {
        roi_id: (
            "signal_only_f0"
            if label == "Signal-Only F0"
            else GUIDED_REFERENCE_CORRECTION_CARD_TO_MODE[label]
        )
        for roi_id, label in strategy_by_roi.items()
    }
    expected_family = {
        roi_id: (
            "signal_only_f0" if selected == "signal_only_f0" else "dynamic_fit"
        )
        for roi_id, selected in expected_selected.items()
    }

    for branch in ("phasic", "tonic"):
        branch_dir = run_dir / "_analysis" / f"{branch}_out"
        metadata = json.loads(
            (branch_dir / "run_metadata.json").read_text(encoding="utf-8")
        )
        requested = {
            record["roi_id"]: record
            for record in metadata["correction_provenance"]["requested_by_roi"]
        }
        assert {
            roi_id: record["selected_strategy"]
            for roi_id, record in requested.items()
        } == expected_selected

        with h5py.File(branch_dir / f"{branch}_trace_cache.h5", "r") as cache:
            chunk_ids = [int(value) for value in cache["meta/chunk_ids"][()]]
            assert chunk_ids == [0, 1]
            for roi_id, selected in expected_selected.items():
                for chunk_id in chunk_ids:
                    group = cache[f"roi/{roi_id}/chunk_{chunk_id}"]
                    assert group.attrs["correction_strategy_family"] == (
                        expected_family[roi_id]
                    )
                    assert group.attrs["correction_selected_strategy"] == selected
                    assert group.attrs["correction_applied_strategy"] == selected
                    if selected == "signal_only_f0":
                        assert group.attrs["correction_applied_source"] == (
                            "signal_only_f0_baseline"
                        )
                        assert "signal_only_f0_baseline" in group
                        if "fit_ref" in group:
                            assert not np.isfinite(group["fit_ref"][()]).any()
                    else:
                        assert group.attrs["correction_applied_source"] == (
                            "fitted_reference"
                        )
                        assert group.attrs["correction_dynamic_fit_mode"] == selected
                        assert "fit_ref" in group
                        assert group["fit_ref"][()].size > 0


@pytest.mark.parametrize(
    "strategy_by_roi",
    [
        {"ROI1": "Signal-Only F0", "ROI2": "Signal-Only F0"},
        {
            "ROI1": "Signal-Only F0",
            "ROI2": "Global Linear Regression",
        },
    ],
    ids=["all_signal_only", "mixed_signal_only_and_reference"],
)
def test_intermittent_mapped_csv_applies_confirmed_per_roi_strategies(
    window, tmp_path, monkeypatch, qapp, strategy_by_roi
):
    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping
    import gui.main_window as main_window_module

    _configure_intermittent_csv_draft(
        window, tmp_path, strategy_by_roi=strategy_by_roi
    )
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
    assert window._guided_backend_validation_outcome.status == "validator_accepted", (
        window._guided_backend_validation_outcome.blocking_issues,
        window._log_view.toPlainText(),
    )
    assert window._guided_run_btn.isEnabled()
    request = window._current_guided_startup_transaction_request()
    assert request is not None
    assert {
        entry.roi_id: entry.selected_strategy
        for entry in request.startup_authority.per_roi_correction_strategy_map
    } == {
        roi_id: (
            "signal_only_f0"
            if label == "Signal-Only F0"
            else GUIDED_REFERENCE_CORRECTION_CARD_TO_MODE[label]
        )
        for roi_id, label in strategy_by_roi.items()
    }

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        staticmethod(lambda *_args, **_kwargs: None),
    )
    window.show()
    window._guided_run_btn.click()
    _pump_until(
        qapp,
        lambda: window._guided_run_execution_thread is None,
        timeout_s=240.0,
    )
    result = window._guided_backend_execution_result
    assert result.status == "wrapper_completed_needs_review_loading", (
        tuple(issue.message for issue in result.blocking_issues),
        result.diagnostics,
    )
    _assert_persisted_correction_output(Path(result.run_directory), strategy_by_roi)


def test_intermittent_rwd_shared_handoff_applies_signal_only_per_roi(
    tmp_path, monkeypatch, qapp
):
    """The shared intermittent startup handoff also applies native RWD output."""
    from PySide6.QtCore import QSettings

    import photometry_pipeline.guided_execution_request_builder as request_builder
    import photometry_pipeline.guided_production_mapping as production_mapping
    import gui.main_window as main_window_module
    from tests.test_guided_gui_run_completed_boundary import (
        _configure_real_analysis_duration_new_analysis_draft,
    )
    from tests.test_gui_guided_new_analysis_plan import (
        _confirm_detected_dataset_settings_via_review_plan_button,
    )

    strategy_by_roi = {"CH1": "Signal-Only F0"}
    window = MainWindow(
        settings=QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    )
    try:
        _configure_real_analysis_duration_new_analysis_draft(
            window,
            tmp_path,
            monkeypatch,
            strategy_by_roi=strategy_by_roi,
            rois=("CH1",),
        )
        _confirm_detected_dataset_settings_via_review_plan_button(window, monkeypatch)
        window._guided_workflow_stepper.setCurrentRow(
            list(GUIDED_WORKFLOW_STEPS).index("Draft plan")
        )
        window._guided_review_go_to_run_btn.click()

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
        assert window._guided_backend_validation_outcome.status == (
            "validator_accepted"
        )
        assert window._guided_run_btn.isEnabled()

        monkeypatch.setattr(
            main_window_module.QMessageBox,
            "information",
            staticmethod(lambda *_args, **_kwargs: None),
        )
        window.show()
        window._guided_run_btn.click()
        _pump_until(
            qapp,
            lambda: window._guided_run_execution_thread is None,
            timeout_s=240.0,
        )
        result = window._guided_backend_execution_result
        assert result.status == "wrapper_completed_needs_review_loading", (
            tuple(issue.message for issue in result.blocking_issues),
            result.diagnostics,
        )
        run_dir = Path(result.run_directory)
        assert classify_run_terminal_state(str(run_dir)).state in SUCCESS_STATES

        for branch in ("phasic", "tonic"):
            branch_dir = run_dir / "_analysis" / f"{branch}_out"
            metadata = json.loads(
                (branch_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            requested = metadata["correction_provenance"]["requested_by_roi"]
            assert len(requested) == 1
            assert requested[0]["roi_id"] == "CH1"
            assert requested[0]["strategy_family"] == "signal_only_f0"
            assert requested[0]["selected_strategy"] == "signal_only_f0"
            assert requested[0]["dynamic_fit_mode"] is None
            with h5py.File(branch_dir / f"{branch}_trace_cache.h5", "r") as cache:
                for chunk_id in (0, 1):
                    group = cache[f"roi/CH1/chunk_{chunk_id}"]
                    assert group.attrs["correction_selected_strategy"] == (
                        "signal_only_f0"
                    )
                    assert group.attrs["correction_applied_strategy"] == (
                        "signal_only_f0"
                    )
                    assert group.attrs["correction_applied_source"] == (
                        "signal_only_f0_baseline"
                    )
                    assert "signal_only_f0_baseline" in group
                    if "fit_ref" in group:
                        assert not np.isfinite(group["fit_ref"][()]).any()
    finally:
        window._guided_backend_execution_active = False
        window.close()
