import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from photometry_pipeline.config import Config
from photometry_pipeline.core.reporting import generate_run_report
from photometry_pipeline.guided_manifest_current_facts import (
    build_guided_manifest_current_facts,
)
from photometry_pipeline.guided_new_analysis_plan import (
    evaluate_new_analysis_plan_readiness,
)
from photometry_pipeline.guided_normalized_recording import (
    build_custom_tabular_normalized_recording_description,
    compute_custom_tabular_parser_contract_digest,
    deserialize_normalized_recording_description,
    serialize_normalized_recording_description,
)
from photometry_pipeline.guided_normalized_recording_consumption import (
    build_custom_tabular_consumed_normalized_recording_evidence,
    compare_requested_and_consumed_normalized_recording,
)
from photometry_pipeline.io.adapters import load_chunk
from photometry_pipeline.io.custom_tabular_source_snapshot import (
    build_custom_tabular_source_candidate_snapshot,
)
from photometry_pipeline.pipeline import Pipeline


pytestmark = pytest.mark.usefixtures("no_real_modals")


def _csv(headers=("Elapsed", "Green", "Iso"), *, milliseconds=False) -> str:
    times = (5000, 5100, 5200, 5300, 5400) if milliseconds else (5, 5.1, 5.2, 5.3, 5.4)
    rows = [",".join(headers)]
    values = {
        "Elapsed": times,
        "Green": (2, 2.1, 2.2, 2.3, 2.4),
        "Iso": (1, 1.1, 1.2, 1.3, 1.4),
        "Extra": (9, 9, 9, 9, 9),
    }
    for index in range(5):
        rows.append(",".join(str(values[name][index]) for name in headers))
    return "\n".join(rows) + "\n"


def _mapping():
    return [
        {
            "roi_id": "Fiber A",
            "signal_column": "Green",
            "reference_column": "Iso",
        }
    ]


def _config(*, milliseconds=False) -> Config:
    return Config(
        target_fs_hz=10.0,
        chunk_duration_sec=0.5,
        custom_tabular_time_col="Elapsed",
        custom_tabular_time_unit=(
            "milliseconds" if milliseconds else "seconds"
        ),
        custom_tabular_roi_mapping_json=json.dumps(_mapping()),
    )


def test_exact_mapping_scales_milliseconds_and_uses_exact_names(tmp_path: Path):
    path = tmp_path / "session_1.csv"
    path.write_text(_csv(milliseconds=True), encoding="utf-8")
    chunk = load_chunk(str(path), "custom_tabular", _config(milliseconds=True), 0)

    assert chunk.channel_names == ["Fiber A"]
    assert chunk.time_sec[0] == pytest.approx(0.0)
    assert chunk.time_sec[-1] == pytest.approx(0.4)
    assert chunk.sig_raw[:, 0] == pytest.approx([2, 2.1, 2.2, 2.3, 2.4])
    assert chunk.uv_raw[:, 0] == pytest.approx([1, 1.1, 1.2, 1.3, 1.4])
    assert chunk.metadata["resolved_time_column"] == "Elapsed"
    assert chunk.metadata["resolved_timestamp_unit"] == "milliseconds"


def test_exact_mapping_allows_different_column_order_and_extra_columns(tmp_path: Path):
    first = tmp_path / "session_1.csv"
    second = tmp_path / "session_2.csv"
    first.write_text(_csv(("Elapsed", "Green", "Iso")), encoding="utf-8")
    second.write_text(
        _csv(("Extra", "Iso", "Elapsed", "Green")), encoding="utf-8"
    )

    one = load_chunk(str(first), "custom_tabular", _config(), 0)
    two = load_chunk(str(second), "custom_tabular", _config(), 1)
    assert one.channel_names == two.channel_names == ["Fiber A"]
    assert one.sig_raw[:, 0] == pytest.approx(two.sig_raw[:, 0])


def test_duplicate_raw_headers_are_rejected_before_pandas_renaming(tmp_path: Path):
    path = tmp_path / "session_1.csv"
    path.write_text("Elapsed,Green,Green\n0,1,2\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate CSV header.*Green"):
        load_chunk(str(path), "custom_tabular", _config(), 0)


def test_custom_tabular_snapshot_uses_top_level_natural_filename_order(tmp_path: Path):
    for name in (
        "Session_11.CSV",
        "session_10.csv",
        "session_2.csv",
        "session_1.csv",
    ):
        (tmp_path / name).write_text(_csv(), encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "session_0.csv").write_text(_csv(), encoding="utf-8")

    snapshot = build_custom_tabular_source_candidate_snapshot(str(tmp_path))
    assert [item.canonical_relative_path for item in snapshot.candidates] == [
        "session_1.csv",
        "session_2.csv",
        "session_10.csv",
        "Session_11.CSV",
    ]


def test_normalized_csv_authority_has_no_fabricated_acquisition_timestamps(
    tmp_path: Path,
):
    path = tmp_path / "session_1.csv"
    path.write_text(_csv(), encoding="utf-8")
    snapshot = build_custom_tabular_source_candidate_snapshot(str(tmp_path))
    interpretation = {
        "time_column": "Elapsed",
        "time_unit": "seconds",
        "time_scale_to_seconds": 1.0,
        "header_rule": "ordinary_first_row",
        "delimiter": "comma",
        "roi_mappings": _mapping(),
        "chronology_authority": "confirmed_filename_order",
    }
    description = build_custom_tabular_normalized_recording_description(
        source_root_canonical=snapshot.source_root_canonical,
        candidate_snapshot=snapshot,
        session_metadata={"session_1.csv": {}},
        session_duration_sec=0.5,
        sessions_per_hour=4,
        timeline_anchor_mode="elapsed",
        acquisition_mode="intermittent",
        discovered_roi_ids=("Fiber A",),
        included_roi_ids=("Fiber A",),
        interpretation=interpretation,
        target_fs_hz=10.0,
    )
    payload = serialize_normalized_recording_description(description)
    restored = deserialize_normalized_recording_description(payload)

    assert restored.adapter_format == "custom_tabular"
    assert restored.sessions[0].authoritative_source_start_time is None
    assert restored.sessions[0].source_timing_evidence == "confirmed_filename_order"
    assert restored.sampling.parser_contract_identity == (
        compute_custom_tabular_parser_contract_digest(interpretation)
    )
    assert restored.roi_channels[0].signal_channel_identity == "Green"
    assert "rwd_time_col" not in restored.adapter_evidence


def test_manifest_current_facts_rechecks_csv_mapping_and_source(tmp_path: Path):
    (tmp_path / "session_1.csv").write_text(_csv(), encoding="utf-8")
    facts = build_guided_manifest_current_facts(
        source_root=tmp_path,
        config=_config(),
        manifest_included_roi_ids=("Fiber A",),
        source_format="custom_tabular",
    )
    assert facts.current_roi_inventory.discovered_roi_ids == ("Fiber A",)
    assert facts.current_candidates[0].canonical_relative_path == "session_1.csv"


def test_run_report_records_exact_csv_mapping_without_suffix_placeholders(
    tmp_path: Path,
):
    config = _config()
    generate_run_report(config, str(tmp_path))
    report = json.loads((tmp_path / "run_report.json").read_text(encoding="utf-8"))
    contract = report["run_context"]["custom_tabular_contract"]

    assert contract["time_col"] == "Elapsed"
    assert contract["time_unit"] == "seconds"
    assert contract["roi_mappings"] == _mapping()
    assert "uv_suffix" not in contract
    assert "sig_suffix" not in contract


def test_original_csv_reaches_pipeline_cache_and_completion_reconciliation(
    tmp_path: Path,
):
    source_root = tmp_path / "csv_source"
    source_root.mkdir()
    for session_name, sample_count in (
        ("Session_1.csv", 201),
        ("Session_2.csv", 200),
    ):
        time_values = 5.0 + np.arange(sample_count, dtype=float) * 0.049999
        (source_root / session_name).write_text(
            "Elapsed,Green,Iso\n"
            + "\n".join(
                f"{time_value:.6f},{2.0 + 0.01 * index:.6f},"
                f"{1.0 + 0.005 * index:.6f}"
                for index, time_value in enumerate(time_values)
            )
            + "\n",
            encoding="utf-8",
        )
    config = Config(
        target_fs_hz=50.0,
        chunk_duration_sec=10.0,
        custom_tabular_time_col="Elapsed",
        custom_tabular_time_unit="seconds",
        custom_tabular_roi_mapping_json=json.dumps(_mapping()),
    )
    run_dir = tmp_path / "run"
    analysis_out = run_dir / "_analysis" / "phasic_out"
    Pipeline(config, mode="phasic").run(
        str(source_root),
        str(analysis_out),
        force_format="custom_tabular",
        recursive=False,
        sessions_per_hour=1,
    )

    snapshot = build_custom_tabular_source_candidate_snapshot(str(source_root))
    interpretation = {
        "time_column": "Elapsed",
        "time_unit": "seconds",
        "time_scale_to_seconds": 1.0,
        "header_rule": "ordinary_first_row",
        "delimiter": "comma",
        "roi_mappings": _mapping(),
        "chronology_authority": "confirmed_filename_order",
    }
    requested = build_custom_tabular_normalized_recording_description(
        source_root_canonical=snapshot.source_root_canonical,
        candidate_snapshot=snapshot,
        session_metadata={"Session_1.csv": {}, "Session_2.csv": {}},
        session_duration_sec=10.0,
        sessions_per_hour=1,
        timeline_anchor_mode="elapsed",
        acquisition_mode="intermittent",
        discovered_roi_ids=("Fiber A",),
        included_roi_ids=("Fiber A",),
        interpretation=interpretation,
        target_fs_hz=50.0,
    )
    consumed = build_custom_tabular_consumed_normalized_recording_evidence(
        run_dir=str(run_dir),
        analysis_kind="phasic",
        requested=requested,
    )

    assert consumed.parser_policy_satisfied is True
    assert (
        compare_requested_and_consumed_normalized_recording(
            requested, consumed
        )
        == ""
    )
    assert (analysis_out / "phasic_trace_cache.h5").exists()
    assert (analysis_out / "features" / "features.csv").exists()
    assert not list(run_dir.rglob("fluorescence.csv"))


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_guided_csv_controls_show_label_order_and_require_confirmation(
    qapp, tmp_path: Path
):
    first = tmp_path / "session_10.csv"
    second = tmp_path / "session_2.csv"
    first.write_text(_csv(), encoding="utf-8")
    second.write_text(_csv(), encoding="utf-8")
    window = MainWindow()
    try:
        window.show()
        index = window._guided_format_combo.findText("custom_tabular")
        window._guided_format_combo.setCurrentIndex(index)
        window._guided_input_dir_edit.setText(str(tmp_path))
        qapp.processEvents()

        assert window._guided_format_combo.itemText(index) == (
            "CSV files (one file per session)"
        )
        assert window._guided_format_combo.currentText() == "custom_tabular"
        assert not window._guided_csv_interpretation_group.isHidden()
        assert [
            window._guided_csv_session_order_list.item(i).text()
            for i in range(window._guided_csv_session_order_list.count())
        ] == ["session_2.csv", "session_10.csv"]
        with pytest.raises(ValueError, match="Confirm.*filename order"):
            window._guided_csv_interpretation()

        window._guided_csv_time_column_combo.setCurrentText("Elapsed")
        row = window._guided_csv_mapping_rows[0]
        row["name"].setText("Fiber A")
        row["signal"].setCurrentText("Green")
        row["reference"].setCurrentText("Iso")
        window._guided_csv_order_confirm_cb.setChecked(True)
        interpretation = window._guided_csv_interpretation()
        assert interpretation["chronology_authority"] == "confirmed_filename_order"
        assert interpretation["ordered_source_files"] == [
            "session_2.csv",
            "session_10.csv",
        ]
        window._discovery_cache = {"resolved_format": "custom_tabular"}
        acquisition_index = (
            window._guided_acquisition_mode_combo.findData("intermittent")
        )
        window._guided_acquisition_mode_combo.setCurrentIndex(
            acquisition_index
        )
        window._guided_sessions_per_hour_edit.setText("1")
        window._guided_session_duration_edit.setText("0.5")
        candidate = (
            window._guided_new_analysis_dataset_contract_candidate()
        )
        assert candidate.status == "inferred"
        assert candidate.contract_values["target_fs_hz"] == 10.0
        window._guided_new_analysis_dataset_contract_snapshot = replace(
            candidate,
            status="applied",
            explicitly_applied=True,
        )
        plan = window._build_guided_new_analysis_draft_plan()
        summary = window._guided_new_analysis_draft_plan_summary_text(
            plan, evaluate_new_analysis_plan_readiness(plan)
        )
        assert "Source type: CSV files" in summary
        assert "CSV session order: confirmed filename order" in summary
        assert "Sampling rate: 10 Hz" in summary
        assert "Automatically determined from the recording." in summary
        assert "custom_tabular" not in summary
        assert "RWD" not in summary
        assert "Doric" not in summary

        other = tmp_path / "other"
        other.mkdir()
        (other / "plain.csv").write_text(_csv(), encoding="utf-8")
        window._guided_input_dir_edit.setText(str(other))
        assert not window._guided_csv_order_confirm_cb.isChecked()
    finally:
        window.close()
        window.deleteLater()
