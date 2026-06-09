import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from gui.batch_runner import (
    BatchCommandResult,
    BatchCancelToken,
    BatchRunner,
    default_subprocess_command_runner,
    derive_dataset_run_spec,
)
from gui.batch_spec import BatchDatasetRow, make_batch_run_spec
from gui.run_spec import RunSpec


def _write_valid_config(path: Path) -> Path:
    cfg = {
        "target_fs_hz": 10.0,
        "chunk_duration_sec": 1.0,
        "lowpass_hz": 1.0,
        "filter_order": 3,
        "window_sec": 1.0,
        "step_sec": 1.0,
        "min_valid_windows": 1,
        "baseline_method": "uv_raw_percentile_session",
        "baseline_percentile": 10.0,
        "f0_min_value": 1e-9,
        "peak_threshold_method": "mean_std",
        "peak_threshold_k": 2.5,
        "peak_threshold_percentile": 95.0,
        "peak_threshold_abs": 0.0,
        "peak_min_distance_sec": 1.0,
        "peak_min_prominence_k": 2.0,
        "peak_min_width_sec": 0.3,
        "dynamic_fit_mode": "rolling_local_regression",
        "allow_partial_final_chunk": False,
        "rwd_time_col": "Time(s)",
        "uv_suffix": "-410",
        "sig_suffix": "-470",
        "custom_tabular_time_col": "time_sec",
        "custom_tabular_uv_suffix": "_iso",
        "custom_tabular_sig_suffix": "_sig",
    }
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return path


def _base_run_spec_dict(tmp_path: Path, *, config_path: Path) -> dict:
    spec = RunSpec(
        input_dir=str(tmp_path / "placeholder_input"),
        run_dir=str(tmp_path / "placeholder_output"),
        format="custom_tabular",
        config_source_path=str(config_path),
        acquisition_mode="continuous",
        continuous_window_sec=600.0,
        continuous_step_sec=600.0,
        allow_partial_final_window=True,
        mode="phasic",
        include_roi_ids=["Region0"],
        sig_iso_render_mode="full",
        dff_render_mode="full",
        stacked_render_mode="full",
        run_profile="full",
        config_overrides={"peak_threshold_k": 3.0},
        user_set_fields=["format", "acquisition_mode"],
    )
    return spec.to_dict()


def _rows(tmp_path: Path, *names: str) -> list[BatchDatasetRow]:
    rows = []
    for idx, name in enumerate(names, start=1):
        input_dir = tmp_path / "inputs" / name
        input_dir.mkdir(parents=True, exist_ok=True)
        rows.append(
            BatchDatasetRow(
                dataset_id=f"dataset_{idx:03d}",
                dataset_name=name,
                input_path=input_dir,
                output_path=tmp_path / "batch_out" / "runs" / f"{name}_{idx:03d}",
            )
        )
    return rows


def _batch_spec(tmp_path: Path, rows: list[BatchDatasetRow], **kwargs):
    config_path = _write_valid_config(tmp_path / "base_config.yaml")
    return make_batch_run_spec(
        batch_id="batch_runner_test",
        created_at="2026-01-01T00:00:00+00:00",
        batch_input_root=tmp_path / "inputs",
        batch_output_root=tmp_path / "batch_out",
        base_run_spec=_base_run_spec_dict(tmp_path, config_path=config_path),
        shared_settings={
            "format": "custom_tabular",
            "mode": "phasic",
            "acquisition_mode": "continuous",
            "continuous_window_sec": 600.0,
            "continuous_step_sec": 600.0,
            "config_source_path": str(config_path),
        },
        datasets=rows,
        **kwargs,
    )


def _write_success_status(run_dir: str) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(run_dir, "status.json").write_text(
        json.dumps({"schema_version": 1, "phase": "final", "status": "success"}),
        encoding="utf-8",
    )
    Path(run_dir, "run_report.json").write_text(
        json.dumps({"run_context": {"run_type": "full"}}),
        encoding="utf-8",
    )


def _write_failure_status(run_dir: str, message: str = "intentional failure") -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(run_dir, "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "final",
                "status": "error",
                "errors": [message],
            }
        ),
        encoding="utf-8",
    )


def _write_custom_tabular_csv(path: Path, *, n_samples: int = 30, fs_hz: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["time_sec,Region0_iso,Region0_sig"]
    for idx in range(n_samples):
        t = idx / fs_hz
        iso = 1.0 + 0.01 * idx
        sig = 2.0 + 0.02 * idx
        lines.append(f"{t:.6f},{iso:.6f},{sig:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_derive_dataset_run_spec_preserves_shared_settings_and_does_not_mutate_base(tmp_path: Path):
    rows = _rows(tmp_path, "AnimalA", "AnimalB")
    batch = _batch_spec(tmp_path, rows)
    before = json.dumps(batch.base_run_spec, sort_keys=True)

    spec_a = derive_dataset_run_spec(batch, rows[0], validate_only=True, overwrite=True)
    spec_b = derive_dataset_run_spec(batch, rows[1], validate_only=False, overwrite=False)

    assert spec_a.input_dir == rows[0].input_path
    assert spec_a.run_dir == rows[0].output_path
    assert spec_b.input_dir == rows[1].input_path
    assert spec_b.run_dir == rows[1].output_path
    assert spec_a.format == "custom_tabular"
    assert spec_a.acquisition_mode == "continuous"
    assert spec_a.continuous_window_sec == 600.0
    assert spec_a.allow_partial_final_window is True
    assert spec_a.mode == "phasic"
    assert spec_a.include_roi_ids == ["Region0"]
    assert spec_a.validate_only is True
    assert spec_a.overwrite is True
    assert spec_b.validate_only is False
    assert spec_b.overwrite is False
    assert json.dumps(batch.base_run_spec, sort_keys=True) == before


def test_wrapper_argv_reuse_includes_dataset_paths_and_shared_flags(tmp_path: Path):
    rows = _rows(tmp_path, "AnimalA")
    batch = _batch_spec(tmp_path, rows)
    spec = derive_dataset_run_spec(batch, rows[0], validate_only=True, overwrite=True)
    rows[0].output_path = spec.run_dir
    Path(spec.run_dir).mkdir(parents=True, exist_ok=True)
    spec.generate_derived_config(spec.run_dir)

    argv = spec.build_runner_argv()

    assert argv[1].endswith(os.path.join("tools", "run_full_pipeline_deliverables.py"))
    assert argv[argv.index("--input") + 1] == rows[0].input_path
    assert argv[argv.index("--out") + 1] == rows[0].output_path
    assert argv[argv.index("--format") + 1] == "custom_tabular"
    assert argv[argv.index("--mode") + 1] == "phasic"
    assert argv[argv.index("--acquisition-mode") + 1] == "continuous"
    assert argv[argv.index("--continuous-window-sec") + 1] == "600.0"
    assert argv[argv.index("--continuous-step-sec") + 1] == "600.0"
    assert argv[argv.index("--include-rois") + 1] == "Region0"
    assert "--validate-only" in argv
    assert "--overwrite" in argv


def test_sequential_success_with_fake_runner_writes_manifests_and_paths(tmp_path: Path):
    rows = _rows(tmp_path, "AnimalA", "AnimalB")
    batch = _batch_spec(tmp_path, rows)
    calls = []

    def fake_runner(argv, row, _run_spec, _cancel_requested):
        calls.append(row.dataset_id)
        assert argv[argv.index("--input") + 1] == row.input_path
        assert argv[argv.index("--out") + 1] == row.output_path
        _write_success_status(row.output_path)
        return BatchCommandResult(exit_code=0)

    runner = BatchRunner(batch, command_runner=fake_runner)
    result = runner.run(validate_only=True)

    assert calls == ["dataset_001", "dataset_002"]
    assert [row.status for row in result.datasets] == ["success", "success"]
    assert all(row.status_json_path and Path(row.status_json_path).exists() for row in result.datasets)
    assert all(row.run_report_path and Path(row.run_report_path).exists() for row in result.datasets)
    assert Path(runner.batch_manifest_json_path).exists()
    assert Path(runner.batch_manifest_csv_path).exists()
    assert Path(runner.batch_run_spec_json_path).exists()
    assert Path(runner.batch_config_used_yaml_path).exists()

    manifest = json.loads(Path(runner.batch_manifest_json_path).read_text(encoding="utf-8"))
    assert manifest["summary"]["success"] == 2
    assert manifest["summary"]["total"] == 2
    with open(runner.batch_manifest_csv_path, "r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert [row["status"] for row in csv_rows] == ["success", "success"]


def test_failure_continues_when_stop_on_failure_false(tmp_path: Path):
    rows = _rows(tmp_path, "AnimalA", "AnimalB")
    batch = _batch_spec(tmp_path, rows, stop_on_failure=False)
    calls = []

    def fake_runner(_argv, row, _run_spec, _cancel_requested):
        calls.append(row.dataset_id)
        if row.dataset_id == "dataset_001":
            _write_failure_status(row.output_path, "first failed")
            return BatchCommandResult(exit_code=1, stderr="first stderr")
        _write_success_status(row.output_path)
        return BatchCommandResult(exit_code=0)

    result = BatchRunner(batch, command_runner=fake_runner).run(validate_only=False)

    assert calls == ["dataset_001", "dataset_002"]
    assert [row.status for row in result.datasets] == ["failed", "success"]
    assert "first failed" in result.datasets[0].message


def test_stop_on_failure_skips_remaining_rows(tmp_path: Path):
    rows = _rows(tmp_path, "AnimalA", "AnimalB")
    batch = _batch_spec(tmp_path, rows, stop_on_failure=True)
    calls = []

    def fake_runner(_argv, row, _run_spec, _cancel_requested):
        calls.append(row.dataset_id)
        _write_failure_status(row.output_path, "stop here")
        return BatchCommandResult(exit_code=1)

    result = BatchRunner(batch, command_runner=fake_runner).run(validate_only=False)

    assert calls == ["dataset_001"]
    assert result.datasets[0].status == "failed"
    assert result.datasets[1].status == "skipped"
    assert "stop_on_failure" in result.datasets[1].message


def test_skipped_rows_are_not_executed(tmp_path: Path):
    rows = _rows(tmp_path, "AnimalA", "AnimalB")
    rows[0].status = "skipped"
    rows[0].message = "Output path already exists and overwrite is disabled."
    batch = _batch_spec(tmp_path, rows)
    calls = []

    def fake_runner(_argv, row, _run_spec, _cancel_requested):
        calls.append(row.dataset_id)
        _write_success_status(row.output_path)
        return BatchCommandResult(exit_code=0)

    result = BatchRunner(batch, command_runner=fake_runner).run(validate_only=True)

    assert calls == ["dataset_002"]
    assert [row.status for row in result.datasets] == ["skipped", "success"]
    assert "already exists" in result.datasets[0].message


def test_manifest_updates_incrementally_before_second_row_finishes(tmp_path: Path):
    rows = _rows(tmp_path, "AnimalA", "AnimalB")
    batch = _batch_spec(tmp_path, rows)
    runner = None
    calls = []

    def fake_runner(_argv, row, _run_spec, _cancel_requested):
        calls.append(row.dataset_id)
        if row.dataset_id == "dataset_002":
            manifest = json.loads(
                Path(runner.batch_manifest_json_path).read_text(encoding="utf-8")
            )
            assert manifest["datasets"][0]["status"] == "success"
            assert manifest["datasets"][1]["status"] == "running"
        _write_success_status(row.output_path)
        return BatchCommandResult(exit_code=0)

    runner = BatchRunner(batch, command_runner=fake_runner)
    runner.run(validate_only=False)

    assert calls == ["dataset_001", "dataset_002"]


def test_cancellation_marks_current_and_remaining_rows(tmp_path: Path):
    rows = _rows(tmp_path, "AnimalA", "AnimalB")
    batch = _batch_spec(tmp_path, rows)
    runner = None

    def fake_runner(_argv, row, _run_spec, _cancel_requested):
        runner.request_cancel()
        assert _cancel_requested() is True
        return BatchCommandResult(exit_code=130, cancelled=True, message="cancelled in fake")

    runner = BatchRunner(batch, command_runner=fake_runner)
    result = runner.run(validate_only=False)

    assert result.datasets[0].status == "cancelled"
    assert result.datasets[0].message == "cancelled in fake"
    assert result.datasets[1].status == "cancelled"
    assert "before dataset execution" in result.datasets[1].message
    assert Path(result.datasets[0].output_path, "CANCEL.REQUESTED").exists()


def test_external_cancel_token_reaches_active_command_runner(tmp_path: Path):
    rows = _rows(tmp_path, "AnimalA", "AnimalB")
    token = BatchCancelToken()
    command_started = threading.Event()
    cancel_seen = threading.Event()
    release_command = threading.Event()

    def fake_runner(_argv, _row, _run_spec, cancel_requested):
        command_started.set()
        if cancel_requested():
            cancel_seen.set()
        while not cancel_seen.is_set():
            if cancel_requested():
                cancel_seen.set()
                break
            release_command.wait(0.01)
        release_command.wait(1.0)
        return BatchCommandResult(
            exit_code=130,
            cancelled=True,
            message="cancelled by external token",
        )

    runner = BatchRunner(
        _batch_spec(tmp_path, rows),
        command_runner=fake_runner,
        cancel_requested=token.is_cancel_requested,
    )
    result_holder = {}
    thread = threading.Thread(
        target=lambda: result_holder.setdefault("result", runner.run(validate_only=True)),
        daemon=True,
    )

    thread.start()
    assert command_started.wait(2.0)
    token.request_cancel()
    assert cancel_seen.wait(2.0)
    release_command.set()
    thread.join(2.0)

    assert not thread.is_alive()
    result = result_holder["result"]
    assert result.datasets[0].status == "cancelled"
    assert result.datasets[0].message == "cancelled by external token"
    assert result.datasets[1].status == "cancelled"
    assert Path(result.datasets[0].output_path, "CANCEL.REQUESTED").exists()


def test_callbacks_receive_row_batch_and_finished_updates(tmp_path: Path):
    rows = _rows(tmp_path, "AnimalA")
    batch = _batch_spec(tmp_path, rows)
    row_updates = []
    batch_updates = []
    finished = []

    def fake_runner(_argv, row, _run_spec, _cancel_requested):
        _write_success_status(row.output_path)
        return BatchCommandResult(exit_code=0)

    BatchRunner(
        batch,
        command_runner=fake_runner,
        on_row_update=lambda row: row_updates.append(row.status),
        on_batch_update=lambda spec: batch_updates.append(spec.to_dict()["summary"]["total"]),
        on_finished=lambda spec: finished.append(spec.finished_at),
    ).run(validate_only=True)

    assert "validating" in row_updates
    assert "success" in row_updates
    assert batch_updates
    assert finished and finished[0] is not None


def test_default_subprocess_command_runner_writes_stdout_and_stderr_on_success(tmp_path: Path):
    row = BatchDatasetRow(
        "dataset_001",
        "AnimalA",
        tmp_path / "input",
        tmp_path / "out" / "AnimalA_001",
    )
    argv = [
        sys.executable,
        "-c",
        "import sys; print('hello stdout'); print('hello stderr', file=sys.stderr)",
    ]

    result = default_subprocess_command_runner(
        argv,
        row,
        RunSpec(),
        cancel_requested=lambda: False,
    )

    assert result.exit_code == 0
    assert "hello stdout" in result.stdout
    assert "hello stderr" in result.stderr
    assert (Path(row.output_path) / "stdout.txt").read_text(encoding="utf-8").strip() == "hello stdout"
    assert (Path(row.output_path) / "stderr.txt").read_text(encoding="utf-8").strip() == "hello stderr"


def test_default_subprocess_command_runner_writes_stdout_and_stderr_on_failure(tmp_path: Path):
    row = BatchDatasetRow(
        "dataset_001",
        "AnimalA",
        tmp_path / "input",
        tmp_path / "out" / "AnimalA_001",
    )
    argv = [
        sys.executable,
        "-c",
        "import sys; print('before failure'); print('fatal stderr', file=sys.stderr); sys.exit(7)",
    ]

    result = default_subprocess_command_runner(
        argv,
        row,
        RunSpec(),
        cancel_requested=lambda: False,
    )

    assert result.exit_code == 7
    assert "before failure" in (Path(row.output_path) / "stdout.txt").read_text(encoding="utf-8")
    assert "fatal stderr" in (Path(row.output_path) / "stderr.txt").read_text(encoding="utf-8")


def test_default_subprocess_command_runner_writes_logs_on_cancellation(tmp_path: Path):
    row = BatchDatasetRow(
        "dataset_001",
        "AnimalA",
        tmp_path / "input",
        tmp_path / "out" / "AnimalA_001",
    )
    argv = [
        sys.executable,
        "-c",
        (
            "import sys, time; "
            "print('started stdout', flush=True); "
            "print('started stderr', file=sys.stderr, flush=True); "
            "time.sleep(10)"
        ),
    ]
    start = time.monotonic()

    def cancel_requested() -> bool:
        return time.monotonic() - start > 0.3

    result = default_subprocess_command_runner(
        argv,
        row,
        RunSpec(),
        cancel_requested=cancel_requested,
    )

    assert result.cancelled is True
    assert (Path(row.output_path) / "CANCEL.REQUESTED").exists()
    assert "started stdout" in (Path(row.output_path) / "stdout.txt").read_text(encoding="utf-8")
    assert "started stderr" in (Path(row.output_path) / "stderr.txt").read_text(encoding="utf-8")


def test_batch_runner_real_wrapper_validate_only_custom_tabular_smoke(tmp_path: Path):
    input_root = tmp_path / "batch_inputs"
    dataset_a = input_root / "AnimalA"
    dataset_b = input_root / "AnimalB"
    _write_custom_tabular_csv(dataset_a / "session_000.csv")
    _write_custom_tabular_csv(dataset_b / "session_000.csv")
    config_path = _write_valid_config(tmp_path / "batch_validate_config.yaml")
    rows = [
        BatchDatasetRow(
            "dataset_001",
            "AnimalA",
            dataset_a,
            tmp_path / "batch_out" / "runs" / "AnimalA_001",
        ),
        BatchDatasetRow(
            "dataset_002",
            "AnimalB",
            dataset_b,
            tmp_path / "batch_out" / "runs" / "AnimalB_002",
        ),
    ]
    base = RunSpec(
        input_dir=str(input_root / "placeholder"),
        run_dir=str(tmp_path / "placeholder_out"),
        format="custom_tabular",
        config_source_path=str(config_path),
        validate_only=True,
        mode="phasic",
        overwrite=True,
    ).to_dict()
    batch = make_batch_run_spec(
        batch_id="batch_wrapper_smoke",
        created_at="2026-01-01T00:00:00+00:00",
        batch_input_root=input_root,
        batch_output_root=tmp_path / "batch_out",
        base_run_spec=base,
        shared_settings={
            "format": "custom_tabular",
            "mode": "phasic",
            "config_source_path": str(config_path),
        },
        datasets=rows,
        overwrite=True,
    )

    runner = BatchRunner(batch)
    result = runner.run(validate_only=True)

    assert [row.status for row in result.datasets] == ["success", "success"]
    assert Path(runner.batch_manifest_csv_path).exists()
    assert Path(runner.batch_manifest_json_path).exists()
    assert Path(runner.batch_run_spec_json_path).exists()
    assert Path(runner.batch_config_used_yaml_path).exists()
    for row in result.datasets:
        run_dir = Path(row.output_path)
        assert (run_dir / "status.json").exists()
        assert (run_dir / "stdout.txt").exists()
        assert (run_dir / "stderr.txt").exists()
        assert row.status_json_path == str(run_dir / "status.json")
        assert "VALIDATE-ONLY: OK" in (run_dir / "stdout.txt").read_text(encoding="utf-8")

    manifest = json.loads(Path(runner.batch_manifest_json_path).read_text(encoding="utf-8"))
    assert manifest["summary"]["success"] == 2
