"""Terminal normalized-recording-provenance verification and Guided
current-native/legacy classification, against real native cache output.

Modeled on tests/test_run_completion_correction_provenance.py's fixture
pattern: a real Pipeline execution produces the analysis output, then a
minimal real terminal set (and, here, a real guided_normalized_recording_description.json
plus the other Guided marker files) is assembled around it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from photometry_pipeline.config import Config
from photometry_pipeline.completed_run_review import (
    CompletedRunReviewError,
    load_completed_phasic_review,
)
from photometry_pipeline.pipeline import Pipeline
from photometry_pipeline.guided_normalized_recording import (
    build_rwd_normalized_recording_description,
    serialize_normalized_recording_description,
)
from photometry_pipeline.run_completion_contract import (
    COMPLETION_KEY,
    PROFILE_TUNING_PREP,
    GUIDED_CURRENT_NATIVE_STATE_CORRUPTED,
    GUIDED_CURRENT_NATIVE_STATE_CURRENT_NATIVE,
    GUIDED_CURRENT_NATIVE_STATE_LEGACY,
    GUIDED_CURRENT_NATIVE_STATE_NOT_GUIDED,
    build_manifest_completion_block,
    build_report_completion_block,
    build_status_completion_block,
    classify_guided_current_native_state,
    classify_run_terminal_state,
    normalize_run_mode,
    normalized_recording_completion_error,
    sha256_file,
)


def _write_source(path: Path) -> None:
    n = 200
    values = {"TimeStamp": np.arange(n) / 20.0}
    values["Region0-410"] = np.random.rand(n) + 1.0
    values["Region0-470"] = np.random.rand(n) + 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(values).to_csv(path, index=False)


def _root_for_case(tmp_path: Path, analysis: Path, name: str) -> Path:
    root = tmp_path / name
    shutil.copytree(analysis, root / "_analysis" / "phasic_out")
    return root


def _write_terminal_set(root: Path, mode: dict, *, run_id: str = "native-run") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_report.json").write_text(
        json.dumps({"completion_contract": build_report_completion_block(run_id=run_id)}, indent=2),
        encoding="utf-8",
    )
    manifest = {
        COMPLETION_KEY: build_manifest_completion_block(
            str(root),
            run_id=run_id,
            run_mode=mode,
            finalized_utc="2026-07-11T00:00:00+00:00",
        )
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status = {
        "run_id": run_id,
        "run_profile": mode["run_profile"],
        "run_type": mode["run_type"],
        "acquisition_mode": mode["acquisition_mode"],
        "traces_only": mode["traces_only"],
        "phase": "final",
        "status": "success",
        "errors": [],
        COMPLETION_KEY: build_status_completion_block(
            run_id=run_id,
            manifest_sha256=sha256_file(str(root / "MANIFEST.json")),
        ),
    }
    (root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def _write_guided_markers(root: Path, requested) -> None:
    """Write the Guided marker files a real Setup-check/startup would have
    produced. Only guided_normalized_recording_description.json's content
    matters for this module's verification; the others only need to exist
    (their own content contracts are exercised by the startup test suites)."""
    (root / "guided_candidate_manifest.json").write_text("{}", encoding="utf-8")
    (root / "guided_startup_provenance.json").write_text("{}", encoding="utf-8")
    (root / "guided_startup_status.json").write_text("{}", encoding="utf-8")
    (root / "guided_normalized_recording_description.json").write_text(
        json.dumps(serialize_normalized_recording_description(requested)),
        encoding="utf-8",
    )


def test_valid_guided_normalized_provenance_reconciles(tmp_path):
    analysis, mode, requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_valid")
    _write_terminal_set(root, mode)
    _write_guided_markers(root, requested)

    error = normalized_recording_completion_error(str(root), mode)
    assert error == ""
    assert classify_run_terminal_state(str(root)).is_success


def test_classifier_current_native_for_valid_guided_run(tmp_path):
    analysis, mode, requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_current_native")
    _write_terminal_set(root, mode)
    _write_guided_markers(root, requested)
    assert (
        classify_guided_current_native_state(str(root), mode)
        == GUIDED_CURRENT_NATIVE_STATE_CURRENT_NATIVE
    )


def test_classifier_not_guided_for_generic_artifacts_only(tmp_path):
    analysis, mode, _requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_not_guided")
    _write_terminal_set(root, mode)
    # Only generic, non-definitive artifacts: config_effective.yaml /
    # command_invoked.txt are the kind of file a Full Control run could
    # also carry -- neither is a definitive Guided marker.
    (root / "config_effective.yaml").write_text("target_fs_hz: 20.0\n", encoding="utf-8")
    (root / "command_invoked.txt").write_text("python analyze.py\n", encoding="utf-8")
    assert (
        classify_guided_current_native_state(str(root), mode)
        == GUIDED_CURRENT_NATIVE_STATE_NOT_GUIDED
    )


def test_classifier_corrupted_for_partial_definitive_marker_set(tmp_path):
    analysis, mode, requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_partial")
    _write_terminal_set(root, mode)
    # A definitive marker is present (candidate manifest), but the mandatory
    # normalized recording description is missing.
    (root / "guided_candidate_manifest.json").write_text("{}", encoding="utf-8")
    assert (
        classify_guided_current_native_state(str(root), mode)
        == GUIDED_CURRENT_NATIVE_STATE_CORRUPTED
    )


def test_normalized_recording_completion_error_missing_description_refuses(tmp_path):
    analysis, mode, _requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_missing_description")
    _write_terminal_set(root, mode)
    (root / "guided_candidate_manifest.json").write_text("{}", encoding="utf-8")
    error = normalized_recording_completion_error(str(root), mode)
    assert "no normalized recording description" in error
    assert classify_run_terminal_state(str(root)).state == "corrupted"
    with pytest.raises(CompletedRunReviewError):
        load_completed_phasic_review(str(root))


def test_normalized_recording_completion_error_tampered_c8_digest_refuses(tmp_path):
    analysis, mode, requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_tampered_digest")
    _write_terminal_set(root, mode)
    _write_guided_markers(root, requested)

    ledger_path = root / "_analysis" / "phasic_out" / "input_processing_completeness.json"
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["expected"][0]["sha256"] = "0" * 64
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")

    error = normalized_recording_completion_error(str(root), mode)
    assert "digest" in error
    assert classify_run_terminal_state(str(root)).state == "corrupted"


def test_normalized_recording_completion_error_no_guided_markers_is_unaffected(tmp_path):
    """A plain Full Control run (no Guided markers at all) must never be
    penalized by normalized-recording verification."""
    analysis, mode, _requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_full_control")
    _write_terminal_set(root, mode)
    assert normalized_recording_completion_error(str(root), mode) == ""
    assert classify_run_terminal_state(str(root)).is_success


def test_completed_review_exposes_normalized_recording_for_current_native(tmp_path):
    analysis, mode, requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_review")
    _write_terminal_set(root, mode)
    _write_guided_markers(root, requested)
    model = load_completed_phasic_review(str(root))
    assert model.normalized_recording is not None
    assert model.normalized_recording.recording_source_identity == requested.recording_source_identity


def test_completed_review_reconstructs_after_source_deletion(tmp_path):
    """Reconstructed Review (reopening the app on a completed run dir) must
    load identically whether or not the original source recording still
    exists -- this loader only ever reads the run directory itself."""
    analysis, mode, requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_review_source_gone")
    _write_terminal_set(root, mode)
    _write_guided_markers(root, requested)

    immediate = load_completed_phasic_review(str(root))

    source_root = Path(requested.recording_source_identity)
    assert source_root.is_dir()
    shutil.rmtree(source_root)

    reconstructed = load_completed_phasic_review(str(root))
    assert reconstructed.normalized_recording == immediate.normalized_recording
    assert reconstructed.rois == immediate.rois
    assert reconstructed.current_native == immediate.current_native


def test_classifier_mixed_when_definitive_marker_coexists_with_legacy_report(tmp_path):
    analysis, mode, requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_mixed")
    legacy_report = {"analytical_contract": {}, "configuration": {}}
    # A positively-identified legacy report shape at the root, AND at the
    # nested analysis location (the real Pipeline-produced nested report
    # declares current-build markers like feature_event_provenance, which
    # would otherwise make _find_legacy_report refuse to call this legacy
    # at all -- overwritten here so the root's legacy shape is genuinely
    # positively identified) ...
    (root / "run_report.json").write_text(json.dumps(legacy_report), encoding="utf-8")
    (root / "_analysis" / "phasic_out" / "run_report.json").write_text(
        json.dumps(legacy_report), encoding="utf-8"
    )
    # ... coexisting with a definitive Guided marker.
    (root / "guided_candidate_manifest.json").write_text("{}", encoding="utf-8")
    assert (
        classify_guided_current_native_state(str(root), mode)
        == "mixed"
    )
    assert classify_run_terminal_state(str(root)).state == "corrupted"
    with pytest.raises(CompletedRunReviewError):
        load_completed_phasic_review(str(root))


def test_completed_review_refuses_when_normalized_recording_disagrees(tmp_path):
    analysis, mode, requested = _fixture_values(tmp_path)
    root = _root_for_case(tmp_path, analysis, "case_review_disagree")
    _write_terminal_set(root, mode)
    _write_guided_markers(root, requested)

    ledger_path = root / "_analysis" / "phasic_out" / "input_processing_completeness.json"
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["expected"][0]["sha256"] = "0" * 64
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CompletedRunReviewError):
        load_completed_phasic_review(str(root))


def _fixture_values(tmp_path):
    """Build the native_run fixture body directly (pytest fixtures cannot
    be called directly from a non-test function, so this factors the
    shared construction out for reuse across the tests above)."""
    source_root = tmp_path / f"input_{id(tmp_path)}"
    source = source_root / "2025_01_01-00_00_00" / "fluorescence.csv"
    _write_source(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    size = source.stat().st_size

    cfg = Config(
        target_fs_hz=20.0,
        chunk_duration_sec=10.0,
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
    )
    analysis = tmp_path / f"analysis_{id(source_root)}"
    Pipeline(cfg, mode="phasic").run(
        str(source_root), str(analysis), force_format="rwd", recursive=True,
        sessions_per_hour=1,
    )
    mode = normalize_run_mode(
        run_profile="tuning_prep",
        run_type="full",
        acquisition_mode="intermittent",
        traces_only=False,
        phasic_analysis=True,
        tonic_analysis=False,
        feature_extraction_ran=False,
        deliverable_profile=PROFILE_TUNING_PREP,
        expected_rois=["Region0"],
        chunked_input_processing=True,
        shared_input_manifest=False,
    )

    candidate = SimpleNamespace(
        canonical_relative_path="2025_01_01-00_00_00/fluorescence.csv",
        size_bytes=size,
        sha256_content_digest=digest,
    )
    snapshot = SimpleNamespace(
        candidates=(candidate,),
        source_candidate_set_digest="s" * 64,
        source_candidate_content_digest="c" * 64,
    )
    requested = build_rwd_normalized_recording_description(
        source_root_canonical=str(source_root),
        candidate_snapshot=snapshot,
        session_duration_sec=10.0,
        sessions_per_hour=1,
        timeline_anchor_mode="none",
        acquisition_mode="intermittent",
        discovered_roi_ids=("Region0",),
        included_roi_ids=("Region0",),
        rwd_time_col="TimeStamp",
        uv_suffix="-410",
        sig_suffix="-470",
        parser_contract_digest="d" * 64,
    )
    return analysis, mode, requested
