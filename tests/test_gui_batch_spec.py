import csv
import json
from pathlib import Path

import yaml

from gui.batch_spec import (
    BATCH_MANIFEST_CSV_FIELDS,
    BatchDatasetRow,
    DiscoveredDataset,
    compute_batch_summary_counts,
    discover_batch_datasets,
    make_batch_run_spec,
    plan_batch_outputs,
    sanitize_dataset_slug,
    write_batch_config_used_yaml,
    write_batch_manifest_csv,
    write_batch_manifest_json,
    write_batch_run_spec_json,
)


def test_dataset_discovery_immediate_subfolders_ignores_root_files_deterministically(tmp_path: Path):
    (tmp_path / "z_dataset").mkdir()
    (tmp_path / "A dataset").mkdir()
    (tmp_path / "m_dataset").mkdir()
    (tmp_path / "notes.csv").write_text("not,a,dataset\n", encoding="utf-8")
    (tmp_path / "README.txt").write_text("ignored root file\n", encoding="utf-8")

    result = discover_batch_datasets(str(tmp_path))

    assert [d.dataset_name for d in result.datasets] == [
        "A dataset",
        "m_dataset",
        "z_dataset",
    ]
    assert len(result.ignored_root_files) == 2
    assert all(Path(p).is_absolute() for p in result.ignored_root_files)
    assert result.to_dict()["ignored_root_files_count"] == 2


def test_dataset_discovery_does_not_recurse(tmp_path: Path):
    animal = tmp_path / "Animal01"
    nested = animal / "nested_recording_folder"
    nested.mkdir(parents=True)
    (nested / "data.csv").write_text("time,sig\n0,1\n", encoding="utf-8")

    result = discover_batch_datasets(str(tmp_path))

    assert [d.dataset_name for d in result.datasets] == ["Animal01"]
    assert result.datasets[0].input_path == str(animal.resolve())


def test_slug_output_planning_handles_spaces_punctuation_duplicates_and_unique_paths(tmp_path: Path):
    datasets = [
        DiscoveredDataset("Animal A!", tmp_path / "Animal A!"),
        DiscoveredDataset("Animal@A", tmp_path / "Animal@A"),
        DiscoveredDataset("...###", tmp_path / "punct"),
    ]
    out_root = tmp_path / "batch_out"

    rows = plan_batch_outputs(datasets, str(out_root), overwrite=False)

    assert sanitize_dataset_slug("Animal A!") == "Animal_A"
    assert sanitize_dataset_slug("...###") == "dataset"
    assert [row.dataset_id for row in rows] == ["dataset_001", "dataset_002", "dataset_003"]
    assert [Path(row.output_path).name for row in rows] == [
        "Animal_A_001",
        "Animal_A_002",
        "dataset_003",
    ]
    assert len({row.output_path for row in rows}) == 3
    assert all(Path(row.output_path).parent == out_root.resolve() / "runs" for row in rows)
    assert all(row.status == "pending" for row in rows)


def test_output_collision_skips_when_overwrite_false_and_preserves_when_true(tmp_path: Path):
    dataset_dir = tmp_path / "Animal A"
    dataset_dir.mkdir()
    out_root = tmp_path / "out"
    existing = out_root / "runs" / "Animal_A_001"
    existing.mkdir(parents=True)
    (existing / "sentinel.txt").write_text("do not delete\n", encoding="utf-8")
    discovered = [DiscoveredDataset("Animal A", dataset_dir)]

    skipped_rows = plan_batch_outputs(discovered, str(out_root), overwrite=False)
    overwrite_rows = plan_batch_outputs(discovered, str(out_root), overwrite=True)

    assert skipped_rows[0].status == "skipped"
    assert "already exists" in skipped_rows[0].message
    assert overwrite_rows[0].status == "pending"
    assert overwrite_rows[0].message == ""
    assert (existing / "sentinel.txt").read_text(encoding="utf-8") == "do not delete\n"


def test_batch_summary_counts_and_unknown_status_rejection(tmp_path: Path):
    rows = [
        BatchDatasetRow("dataset_001", "a", tmp_path / "a", tmp_path / "out_a"),
        BatchDatasetRow("dataset_002", "b", tmp_path / "b", tmp_path / "out_b", status="success"),
        BatchDatasetRow("dataset_003", "c", tmp_path / "c", tmp_path / "out_c", status="failed"),
        BatchDatasetRow("dataset_004", "d", tmp_path / "d", tmp_path / "out_d", status="skipped"),
        BatchDatasetRow("dataset_005", "e", tmp_path / "e", tmp_path / "out_e", status="cancelled"),
    ]

    counts = compute_batch_summary_counts(rows)

    assert counts["total"] == 5
    assert counts["pending"] == 1
    assert counts["success"] == 1
    assert counts["failed"] == 1
    assert counts["skipped"] == 1
    assert counts["cancelled"] == 1


def test_unknown_dataset_status_raises(tmp_path: Path):
    try:
        BatchDatasetRow(
            "dataset_001",
            "bad",
            tmp_path / "bad",
            tmp_path / "out_bad",
            status="mystery",
        )
    except ValueError as exc:
        assert "Unknown batch dataset status" in str(exc)
    else:
        raise AssertionError("unknown batch status should raise")


def test_manifest_csv_json_and_config_writing_round_trip(tmp_path: Path):
    rows = [
        BatchDatasetRow(
            "dataset_001",
            "Animal A",
            tmp_path / "Animal A",
            tmp_path / "out" / "runs" / "Animal_A_001",
            status="success",
            message="completed",
            elapsed_sec=1.25,
            status_json_path=tmp_path / "out" / "runs" / "Animal_A_001" / "status.json",
            run_report_path=tmp_path / "out" / "runs" / "Animal_A_001" / "run_report.json",
        ),
        BatchDatasetRow(
            "dataset_002",
            "Animal B",
            tmp_path / "Animal B",
            tmp_path / "out" / "runs" / "Animal_B_002",
            status="failed",
            message="validation failed",
        ),
    ]
    shared = {
        "mode": "both",
        "format": "custom_tabular",
        "acquisition_mode": "continuous",
        "continuous_window_sec": 600.0,
        "continuous_step_sec": 600.0,
        "config_source_path": "config.yaml",
    }
    spec = make_batch_run_spec(
        batch_id="batch_test",
        created_at="2026-01-01T00:00:00+00:00",
        batch_input_root=tmp_path / "input",
        batch_output_root=tmp_path / "out",
        shared_settings=shared,
        datasets=rows,
        software={"git_commit": "abc123"},
    )

    csv_path = write_batch_manifest_csv(spec, tmp_path / "out" / "batch_manifest.csv")
    json_path = write_batch_manifest_json(spec, tmp_path / "out" / "batch_manifest.json")
    run_spec_path = write_batch_run_spec_json(spec, tmp_path / "out" / "batch_run_spec.json")
    yaml_path = write_batch_config_used_yaml(spec, tmp_path / "out" / "batch_config_used.yaml")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0].keys() == set(BATCH_MANIFEST_CSV_FIELDS)
    assert csv_rows[0]["batch_id"] == "batch_test"
    assert csv_rows[0]["dataset_name"] == "Animal A"
    assert csv_rows[0]["status"] == "success"
    assert csv_rows[0]["format"] == "custom_tabular"
    assert csv_rows[1]["message"] == "validation failed"

    with open(json_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["batch_id"] == "batch_test"
    assert manifest["shared_settings"] == shared
    assert manifest["summary"]["total"] == 2
    assert manifest["summary"]["success"] == 1
    assert manifest["summary"]["failed"] == 1
    assert len(manifest["datasets"]) == 2
    assert manifest["software"]["git_commit"] == "abc123"

    with open(run_spec_path, "r", encoding="utf-8") as f:
        frozen_spec = json.load(f)
    assert frozen_spec["batch_id"] == "batch_test"
    assert frozen_spec["datasets"][0]["run_dir"] == rows[0].output_path

    with open(yaml_path, "r", encoding="utf-8") as f:
        config_used = yaml.safe_load(f)
    assert config_used == shared


def test_batch_run_spec_serialization_is_json_safe_and_freezes_shared_settings(tmp_path: Path):
    shared = {
        "format": "rwd",
        "config_source_path": tmp_path / "cfg.yaml",
        "nested": {"values": [tmp_path / "a", "b"]},
    }
    base = {"input_dir": tmp_path / "input", "format": "rwd"}
    row = BatchDatasetRow(
        "dataset_001",
        "Animal A",
        tmp_path / "Animal A",
        tmp_path / "out" / "runs" / "Animal_A_001",
    )

    spec = make_batch_run_spec(
        batch_id="batch_freeze",
        created_at="2026-01-01T00:00:00+00:00",
        batch_input_root=tmp_path / "input",
        batch_output_root=tmp_path / "out",
        base_run_spec=base,
        shared_settings=shared,
        datasets=[row],
    )
    shared["format"] = "npm"
    shared["nested"]["values"].append("mutated")
    base["format"] = "custom_tabular"
    row.status = "success"

    payload = spec.to_dict()

    assert payload["shared_settings"]["format"] == "rwd"
    assert payload["base_run_spec"]["format"] == "rwd"
    assert payload["datasets"][0]["status"] == "pending"
    json.dumps(payload)
    assert "Path" not in json.dumps(payload)
