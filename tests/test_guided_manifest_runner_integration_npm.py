"""NPM counterpart to test_guided_manifest_runner_integration.py.

Proves the real (unstubbed) tools/run_full_pipeline_deliverables.py Guided
manifest verification boundary -- build_guided_manifest_current_facts and
verify_guided_candidate_manifest_consumption -- now dispatches correctly for
NPM instead of unconditionally running RWD discovery
(no_rwd_fluorescence_files), while continuing to refuse the same classes of
drift RWD already refuses.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import photometry_pipeline.guided_manifest_current_facts as current_facts_module
import photometry_pipeline.guided_manifest_verification as verification
import tools.run_full_pipeline_deliverables as wrapper
from photometry_pipeline.config import Config
from photometry_pipeline.guided_manifest_current_facts import (
    build_guided_manifest_current_facts,
)
from photometry_pipeline.io.npm_source_snapshot import (
    build_npm_source_candidate_snapshot,
)
from photometry_pipeline.io.rwd_source_snapshot import (
    build_rwd_source_candidate_snapshot,
)


_NPM_CONFIG_YAML = (
    "allow_partial_final_chunk: true\n"
    "target_fs_hz: 1.0\n"
    "chunk_duration_sec: 2.0\n"
    "npm_time_axis: system_timestamp\n"
    "npm_system_ts_col: SystemTimestamp\n"
    "npm_computer_ts_col: ComputerTimestamp\n"
    "npm_led_col: LedState\n"
    "npm_region_prefix: Region\n"
    "npm_region_suffix: G\n"
)


def _npm_config(**overrides) -> Config:
    cfg = Config(
        allow_partial_final_chunk=True,
        target_fs_hz=1.0,
        chunk_duration_sec=2.0,
        npm_time_axis="system_timestamp",
        npm_system_ts_col="SystemTimestamp",
        npm_computer_ts_col="ComputerTimestamp",
        npm_led_col="LedState",
        npm_region_prefix="Region",
        npm_region_suffix="G",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _write_npm_csv(path: Path, t0: float = 0.0, region0=10.0, region1=11.0) -> None:
    header = ["SystemTimestamp", "ComputerTimestamp", "LedState", "Region0G", "Region1G"]
    rows = [
        [t0 + 0.0, t0 + 0.0, 1, region0, region1],
        [t0 + 0.0, t0 + 0.0, 2, region0 + 90.0, region1 + 99.0],
        [t0 + 1.0, t0 + 1.0, 1, region0 + 10.0, region1 + 10.0],
        [t0 + 1.0, t0 + 1.0, 2, region0 + 190.0, region1 + 199.0],
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _npm_manifest(tmp_path, included=("Region0",)):
    root = tmp_path / "source"
    root.mkdir()
    _write_npm_csv(root / "photometryData2025-03-05T15_37_44.csv", t0=0.0)
    _write_npm_csv(root / "photometryData2025-03-05T15_38_01.csv", t0=100.0)
    config = _npm_config()
    facts = build_guided_manifest_current_facts(
        source_root=root,
        config=config,
        manifest_included_roi_ids=included,
        source_format="npm",
    )
    snapshot = build_npm_source_candidate_snapshot(str(root))
    manifest = verification.GuidedCandidateManifestForRunner(
        manifest_schema_name=verification.GUIDED_MANIFEST_SCHEMA_NAME,
        manifest_schema_version=verification.GUIDED_MANIFEST_SCHEMA_VERSION,
        candidate_consumption_contract_version=(
            verification.GUIDED_CANDIDATE_CONSUMPTION_CONTRACT_VERSION
        ),
        source_root_canonical=snapshot.source_root_canonical,
        source_candidate_set_digest=snapshot.source_candidate_set_digest,
        source_candidate_content_digest=snapshot.source_candidate_content_digest,
        candidate_files=tuple(
            verification.GuidedManifestCandidateFile(
                item.canonical_relative_path,
                item.size_bytes,
                item.sha256_content_digest,
            )
            for item in snapshot.candidates
        ),
        parser_contract_digest=facts.current_roi_inventory.parser_contract_digest,
        discovered_roi_ids=facts.current_roi_inventory.discovered_roi_ids,
        included_roi_ids=facts.current_roi_inventory.included_roi_ids,
        excluded_roi_ids=facts.current_roi_inventory.excluded_roi_ids,
        strict_roi_inventory_digest=(
            facts.current_roi_inventory.strict_roi_inventory_digest
        ),
        candidate_preflight_identity="a" * 64,
        roi_preflight_identity="b" * 64,
        canonical_candidate_manifest_payload_identity="0" * 64,
    )
    manifest = verification.GuidedCandidateManifestForRunner(
        **{
            **asdict(manifest),
            "candidate_files": manifest.candidate_files,
            "canonical_candidate_manifest_payload_identity": (
                verification.compute_guided_candidate_manifest_for_runner_identity(
                    manifest
                )
            ),
        }
    )
    path = tmp_path / "guided_manifest.json"
    payload = asdict(manifest)
    payload["candidate_files"] = [asdict(item) for item in manifest.candidate_files]
    path.write_text(json.dumps(payload), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_NPM_CONFIG_YAML, encoding="utf-8")
    return root, path, config_path, manifest


def _npm_args(root, manifest_path, config_path, **changes):
    values = dict(
        input=str(root),
        config=str(config_path),
        format="npm",
        mode="phasic",
        run_type="full",
        traces_only=False,
        discover=False,
        validate_only=False,
        overwrite=False,
        preview_first_n=None,
        include_rois=None,
        exclude_rois=None,
        acquisition_mode=None,
        guided_candidate_manifest=str(manifest_path),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_wrapper_verifies_valid_npm_manifest(tmp_path):
    """The real failure this session reproduced: an NPM Guided run's
    manifest verification previously always raised RuntimeError wrapping
    no_rwd_fluorescence_files, because build_guided_manifest_current_facts
    unconditionally ran RWD discovery. It must now accept a genuine NPM
    manifest against real NPM CSV files."""
    root, path, config_path, _ = _npm_manifest(tmp_path)
    facts, result = wrapper.verify_guided_manifest_before_output(
        _npm_args(root, path, config_path)
    )
    assert result.accepted is True
    assert result.blocking_issues == ()
    assert facts.current_roi_inventory.discovered_roi_ids == ("Region0", "Region1")


def test_npm_manifest_refuses_content_changed_same_filename_and_size(tmp_path):
    root, path, config_path, _ = _npm_manifest(tmp_path)
    target = sorted(root.glob("*.csv"))[0]
    original = target.read_text(encoding="utf-8")
    # Same length replacement -- isolates digest drift from size drift.
    changed = original.replace("10.0", "99.0", 1)
    assert len(changed) == len(original)
    target.write_text(changed, encoding="utf-8")
    with pytest.raises(RuntimeError, match="verification refused"):
        wrapper.verify_guided_manifest_before_output(
            _npm_args(root, path, config_path)
        )


def test_npm_manifest_refuses_size_mismatch(tmp_path):
    root, path, config_path, _ = _npm_manifest(tmp_path)
    target = sorted(root.glob("*.csv"))[0]
    # A trailing blank line changes size/digest without adding a data row
    # (pandas.read_csv skips blank lines), isolating the size/digest check
    # from ROI/cadence content changes.
    with open(target, "a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="verification refused"):
        wrapper.verify_guided_manifest_before_output(
            _npm_args(root, path, config_path)
        )


def test_npm_manifest_refuses_removed_candidate(tmp_path):
    root, path, config_path, _ = _npm_manifest(tmp_path)
    target = sorted(root.glob("*.csv"))[0]
    target.unlink()
    with pytest.raises(RuntimeError, match="verification refused"):
        wrapper.verify_guided_manifest_before_output(
            _npm_args(root, path, config_path)
        )


def test_npm_manifest_refuses_extra_candidate(tmp_path):
    root, path, config_path, _ = _npm_manifest(tmp_path)
    _write_npm_csv(root / "photometryData2025-03-05T15_39_50.csv", t0=200.0)
    with pytest.raises(RuntimeError, match="verification refused"):
        wrapper.verify_guided_manifest_before_output(
            _npm_args(root, path, config_path)
        )


def test_npm_current_facts_does_not_call_rwd_snapshot_builder(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    _write_npm_csv(root / "photometryData2025-03-05T15_37_44.csv")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("RWD snapshot builder must not be called for NPM")

    monkeypatch.setattr(
        current_facts_module, "build_rwd_source_candidate_snapshot", forbidden
    )
    facts = build_guided_manifest_current_facts(
        source_root=root,
        config=_npm_config(),
        manifest_included_roi_ids=("Region0",),
        source_format="npm",
    )
    assert facts.current_candidates


def test_rwd_current_facts_does_not_call_npm_snapshot_builder(tmp_path, monkeypatch):
    root = tmp_path / "source"
    session = root / "2025_01_01-00_00_00"
    session.mkdir(parents=True)
    (session / "fluorescence.csv").write_text(
        "Time(s),ROI0-410,ROI0-470\n0,1,2\n", encoding="utf-8"
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("NPM snapshot builder must not be called for RWD")

    monkeypatch.setattr(
        current_facts_module, "build_npm_source_candidate_snapshot", forbidden
    )
    facts = build_guided_manifest_current_facts(
        source_root=root,
        config=Config(),
        manifest_included_roi_ids=("ROI0",),
        source_format="rwd",
    )
    assert facts.current_candidates


class _StopAfterAnalysisCommand(RuntimeError):
    pass


def test_real_wrapper_boundary_npm_gets_past_manifest_verification(
    tmp_path, monkeypatch
):
    """Drives the real wrapper.main() entry point (real argv parsing, real
    unstubbed verify_guided_manifest_before_output/
    build_guided_manifest_current_facts against real NPM CSV files) and
    stops only at the point the real analysis subprocess would be spawned
    -- the next intended execution boundary after manifest verification.
    Before this session's fix this reproducibly failed inside manifest
    verification itself with "no_rwd_fluorescence_files"."""
    root, path, config_path, _ = _npm_manifest(tmp_path)
    out_dir = tmp_path / "out"

    captured = {}

    def capture_and_stop(cmd, roi_label=None):
        captured["cmd_phasic"] = cmd
        raise _StopAfterAnalysisCommand()

    monkeypatch.setattr(wrapper, "run_cmd", capture_and_stop)
    # Mirrors the established pattern in
    # test_guided_startup_orchestration_real_wrapper_boundary.py: this
    # artifact belongs to a separate, already-materialized startup step and
    # its absence here is orthogonal to what this test proves. It is only
    # reached (and only needs stubbing) *after* the real, unstubbed
    # verify_guided_manifest_before_output already accepted the manifest.
    monkeypatch.setattr(
        wrapper,
        "verify_guided_normalized_recording_description_before_output",
        lambda _args, _facts, _verified: None,
    )
    monkeypatch.setattr(
        wrapper.sys,
        "argv",
        [
            "run_full_pipeline_deliverables.py",
            "--input", str(root),
            "--out", str(out_dir),
            "--config", str(config_path),
            "--format", "npm",
            "--mode", "phasic",
            "--guided-candidate-manifest", str(path),
            "--sessions-per-hour", "6",
        ],
    )
    try:
        wrapper.main()
    except (_StopAfterAnalysisCommand, SystemExit):
        pass
    assert "cmd_phasic" in captured, (
        "real wrapper never got past manifest verification to run_cmd(cmd_phasic)"
    )
    cmd_phasic = captured["cmd_phasic"]
    assert "--guided-candidate-manifest" in cmd_phasic
    assert "--overwrite" not in cmd_phasic
