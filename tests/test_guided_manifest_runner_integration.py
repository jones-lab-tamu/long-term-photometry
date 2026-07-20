from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import photometry_pipeline.guided_manifest_verification as verification
import tools.run_full_pipeline_deliverables as wrapper
from photometry_pipeline.config import Config
from photometry_pipeline.guided_manifest_current_facts import (
    build_guided_manifest_current_facts,
)
from photometry_pipeline.io.rwd_source_snapshot import (
    build_rwd_source_candidate_snapshot,
)
from photometry_pipeline.pipeline import Pipeline


def _write_session(root: Path, name: str, rois=("ROI0", "ROI1"), time_col="Time(s)"):
    path = root / name / "fluorescence.csv"
    path.parent.mkdir(parents=True)
    columns = [time_col]
    row = ["0"]
    for roi in rois:
        columns.extend((f"{roi}-410", f"{roi}-470"))
        row.extend(("1", "2"))
    path.write_text(",".join(columns) + "\n" + ",".join(row) + "\n")


def _manifest(tmp_path, time_col="Time(s)"):
    root = tmp_path / "source"
    _write_session(root, "2025_01_01-00_00_00", time_col=time_col)
    _write_session(root, "2025_01_01-00_10_00", time_col=time_col)
    config = Config(rwd_time_col=time_col)
    facts = build_guided_manifest_current_facts(
        source_root=root,
        config=config,
        manifest_included_roi_ids=("ROI0",),
        source_format="rwd",
    )
    snapshot = build_rwd_source_candidate_snapshot(str(root))
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
    config_path.write_text("{}\n", encoding="utf-8")
    return root, path, config_path, manifest


def _args(root, manifest_path, config_path, **changes):
    values = dict(
        input=str(root),
        config=str(config_path),
        format="rwd",
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


def test_wrapper_verifies_valid_manifest_and_refuses_mismatches(tmp_path):
    root, path, config_path, _ = _manifest(tmp_path)
    _facts, result = wrapper.verify_guided_manifest_before_output(
        _args(root, path, config_path)
    )
    assert result.accepted

    target = root / "2025_01_01-00_00_00" / "fluorescence.csv"
    target.write_text(target.read_text() + "1,2,3,4,5\n")
    with pytest.raises(RuntimeError, match="verification refused"):
        wrapper.verify_guided_manifest_before_output(
            _args(root, path, config_path)
        )


def test_wrapper_verifies_valid_manifest_for_timestamp_time_column(tmp_path):
    """4J16k19: a real Guided source whose fluorescence.csv header uses
    "TimeStamp" must not be refused by the wrapper's live manifest
    re-verification with guided_manifest_parser_contract_mismatch. Before
    the fix, build_guided_manifest_current_facts reconstructed a
    single-value parser contract from config.rwd_time_col, which could
    never digest-match the multi-candidate contract
    (GUIDED_BACKEND_RWD_TIME_COLUMN_CANDIDATES) already embedded in the
    manifest at Guided Run press time."""
    root, path, config_path, _ = _manifest(tmp_path, time_col="TimeStamp")
    _facts, result = wrapper.verify_guided_manifest_before_output(
        _args(root, path, config_path)
    )
    assert result.accepted is True
    assert result.blocking_issues == ()


def test_wrapper_refuses_malformed_and_roi_mismatched_manifest(tmp_path):
    root, path, config_path, manifest = _manifest(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="verification refused"):
        wrapper.verify_guided_manifest_before_output(
            _args(root, path, config_path)
        )

    changed = verification.GuidedCandidateManifestForRunner(
        **{
            **asdict(manifest),
            "candidate_files": manifest.candidate_files,
            "strict_roi_inventory_digest": "d" * 64,
            "canonical_candidate_manifest_payload_identity": "0" * 64,
        }
    )
    changed = verification.GuidedCandidateManifestForRunner(
        **{
            **asdict(changed),
            "candidate_files": changed.candidate_files,
            "canonical_candidate_manifest_payload_identity": (
                verification.compute_guided_candidate_manifest_for_runner_identity(
                    changed
                )
            ),
        }
    )
    payload = asdict(changed)
    payload["candidate_files"] = [asdict(item) for item in changed.candidate_files]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="verification refused"):
        wrapper.verify_guided_manifest_before_output(
            _args(root, path, config_path)
        )


@pytest.mark.parametrize("mode", ("tonic", "both"))
def test_wrapper_accepts_native_tonic_and_combined_manifest_modes(tmp_path, mode):
    root, path, config_path, _ = _manifest(tmp_path)
    _facts, verified = wrapper.verify_guided_manifest_before_output(
        _args(root, path, config_path, mode=mode)
    )
    assert verified.accepted is True


def test_wrapper_missing_manifest_refuses_before_run_dir_resolution(
    tmp_path, monkeypatch
):
    root = tmp_path / "source"
    root.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    args = _args(root, tmp_path / "missing.json", config)
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("run directory must not be resolved")

    monkeypatch.setattr(wrapper, "resolve_run_dir", forbidden)
    with pytest.raises(SystemExit):
        wrapper.main()


def test_config_preview_refuses_before_run_dir_resolution(
    tmp_path, monkeypatch, capsys
):
    root, path, config_path, _ = _manifest(tmp_path)
    config_path.write_text("preview_first_n: 1\n", encoding="utf-8")
    args = _args(root, path, config_path, preview_first_n=None)
    monkeypatch.setattr(wrapper, "parse_args", lambda: args)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("run directory must not be resolved")

    monkeypatch.setattr(wrapper, "resolve_run_dir", forbidden)
    with pytest.raises(SystemExit):
        wrapper.main()
    assert "guided_manifest_cli_conflict" in capsys.readouterr().err


def test_manifest_argument_reaches_both_native_analysis_branches():
    args = SimpleNamespace(guided_candidate_manifest="internal.json")
    tonic = ["python", "analyze_photometry.py", "--mode", "tonic"]
    phasic = ["python", "analyze_photometry.py", "--mode", "phasic"]
    wrapper._append_guided_manifest_to_analysis_command(
        tonic, args, mode="tonic"
    )
    wrapper._append_guided_manifest_to_analysis_command(
        phasic, args, mode="phasic"
    )
    assert tonic[-2:] == ["--guided-candidate-manifest", "internal.json"]
    assert phasic[-2:] == ["--guided-candidate-manifest", "internal.json"]


def test_no_manifest_wrapper_path_is_unchanged():
    assert wrapper.verify_guided_manifest_before_output(
        SimpleNamespace(guided_candidate_manifest=None)
    ) is None


def test_pipeline_binds_manifest_files_and_rois_and_skips_discovery(
    tmp_path, monkeypatch
):
    root, path, _, manifest = _manifest(tmp_path)
    pipeline = Pipeline(Config(), mode="phasic")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normal discovery must be skipped")

    monkeypatch.setattr(pipeline, "discover_files", forbidden)

    class StopAfterBinding(RuntimeError):
        pass

    monkeypatch.setattr(
        pipeline,
        "_resolve_representative_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StopAfterBinding()),
    )
    with pytest.raises(StopAfterBinding):
        pipeline.run(
            str(root),
            str(tmp_path / "out"),
            force_format="rwd",
            guided_manifest_path=str(path),
        )
    assert pipeline.file_list == [
        os.path.normcase(
            os.path.abspath(str(Path(root) / Path(item.canonical_relative_path)))
        )
        for item in manifest.candidate_files
    ]
    assert pipeline._selected_rois == ["ROI0"]


def test_passes_consume_verified_candidate_list(tmp_path, monkeypatch):
    root, path, _, manifest = _manifest(tmp_path)
    pipeline = Pipeline(Config(), mode="phasic")
    expected = [
        os.path.normcase(
            os.path.abspath(str(Path(root) / item.canonical_relative_path))
        )
        for item in manifest.candidate_files
    ]
    observed = {}

    def representative(*_args, **_kwargs):
        pipeline.representative_session_index = 0
        pipeline.representative_session_id = "2025_01_01-00_00_00"
        pipeline.n_sessions_resolved = len(pipeline.file_list)
        pipeline.representative_user_provided = False
        pipeline.representative_session_info = {}

    monkeypatch.setattr(pipeline, "_resolve_representative_session", representative)
    monkeypatch.setattr(
        "photometry_pipeline.pipeline.generate_run_report",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline,
        "run_pass_1",
        lambda *_args, **_kwargs: observed.setdefault(
            "pass1", list(pipeline.file_list)
        ),
    )

    class StopAfterPass2(RuntimeError):
        pass

    def pass2(*_args, **_kwargs):
        observed["pass2"] = list(pipeline.file_list)
        raise StopAfterPass2()

    monkeypatch.setattr(pipeline, "run_pass_2", pass2)

    class FakeWriter:
        def __init__(self, *_args, **_kwargs):
            pass

        def abort(self):
            pass

    monkeypatch.setattr(
        "photometry_pipeline.io.hdf5_cache.Hdf5TraceCacheWriter",
        FakeWriter,
    )
    with pytest.raises(StopAfterPass2):
        pipeline.run(
            str(root),
            str(tmp_path / "out"),
            force_format="rwd",
            guided_manifest_path=str(path),
        )
    assert observed == {"pass1": expected, "pass2": expected}


def test_no_manifest_pipeline_still_uses_normal_discovery(tmp_path, monkeypatch):
    pipeline = Pipeline(Config(), mode="phasic")
    called = []

    class Stop(RuntimeError):
        pass

    def discover(*_args, **_kwargs):
        called.append(True)
        raise Stop()

    monkeypatch.setattr(pipeline, "discover_files", discover)
    with pytest.raises(Stop):
        pipeline.run(str(tmp_path), str(tmp_path / "out"))
    assert called == [True]
