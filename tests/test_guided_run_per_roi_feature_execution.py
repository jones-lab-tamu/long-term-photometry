"""4J16k34: connect Guided per-ROI feature-detection settings to Run execution.

Covers the wiring from a resolved per-ROI feature/event map (32c) through:
  guided_startup_materialization -> guided_startup_orchestration (subprocess
  argv) -> analyze_photometry.py's Pipeline construction, and
  tools/run_full_pipeline_deliverables.py's applied-dF/F orchestration call.

Before this wiring, guided_production_mapping.build_per_roi_feature_event_backend_shapes
produced ready-to-use data that no caller in this codebase ever consumed for
execution (see that function's docstring). These tests prove the missing
consumption now exists end to end.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import photometry_pipeline.guided_backend_validation_request as validation_request
import photometry_pipeline.guided_backend_validator as validator
import photometry_pipeline.guided_execution_payloads as payloads
import photometry_pipeline.guided_manifest_verification as verification
import photometry_pipeline.guided_production_mapping as mapping
import photometry_pipeline.guided_run_authorization as authorization
import photometry_pipeline.guided_startup_allocation as allocation
import photometry_pipeline.guided_startup_materialization as materialization
import photometry_pipeline.guided_startup_orchestration as orchestration
import photometry_pipeline.guided_startup_transaction as startup
from photometry_pipeline.config import Config
from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS
from photometry_pipeline.guided_backend_validation_workflow import (
    GuidedBackendValidationGuiContext,
    GuidedBackendValidationWorkflowOutcome,
    build_guided_backend_validation_parser_contract,
)
from photometry_pipeline.guided_manifest_current_facts import (
    build_guided_manifest_current_facts,
)
from photometry_pipeline.guided_new_analysis_plan import GuidedNewAnalysisDraftPlan
from photometry_pipeline.io.rwd_source_snapshot import (
    build_rwd_source_candidate_snapshot,
)
from tests.test_guided_backend_validator import (
    _contract as _validator_contract,
    _request as _valid_request,
    _typed,
)
from tests.test_guided_execution_payloads import (
    _accepted_candidate,
    _accepted_roi,
    _build_app_identity,
)


def _typed_contract(name, value):
    return validation_request.GuidedBackendTypedFieldValue(
        field_name=name, value_type=type(value).__name__, value=value
    )


def _per_roi_feature_event_map():
    """CH1 ("ROI0") Custom with a raised peak_threshold_k; CH2 ("ROI1") Default."""
    override_entry = validation_request.GuidedBackendPerRoiFeatureEvent(
        roi_id="ROI0",
        source="override",
        feature_event_profile_id="custom-roi0",
        override_config_fields=(_typed_contract("peak_threshold_k", 10.0),),
        effective_config_fields=tuple(
            _typed_contract(name, value)
            for name, value in {
                "event_signal": "dff",
                "signal_excursion_polarity": "positive",
                "peak_threshold_method": "mean_std",
                "peak_threshold_k": 10.0,
                "peak_threshold_percentile": 95.0,
                "peak_threshold_abs": 0.0,
                "peak_min_distance_sec": 1.0,
                "peak_min_prominence_k": 2.0,
                "peak_min_width_sec": 0.3,
                "peak_pre_filter": "none",
                "event_auc_baseline": "zero",
            }.items()
        ),
        explicit_user_mark=True,
        current_or_stale="current",
    )
    default_entry = validation_request.GuidedBackendPerRoiFeatureEvent(
        roi_id="ROI1",
        source="default",
        feature_event_profile_id="profile-001",
        override_config_fields=(),
        effective_config_fields=(_typed_contract("event_signal", "dff"),),
        explicit_user_mark=True,
        current_or_stale="current",
    )
    return override_entry, default_entry


def _default_only_feature_event_map():
    """A realistic 32c-resolved map where every included ROI landed on the
    default profile: non-empty per_roi_feature_event_map, but no ROI is
    Customized (source="override"). This is the actual default-only/
    global-only shape the resolver produces -- not the same as an empty
    map -- and must not trigger writing the per-ROI feature artifact."""
    roi0_default = validation_request.GuidedBackendPerRoiFeatureEvent(
        roi_id="ROI0",
        source="default",
        feature_event_profile_id="profile-001",
        override_config_fields=(),
        effective_config_fields=(_typed_contract("event_signal", "dff"),),
        explicit_user_mark=True,
        current_or_stale="current",
    )
    roi1_default = validation_request.GuidedBackendPerRoiFeatureEvent(
        roi_id="ROI1",
        source="default",
        feature_event_profile_id="profile-001",
        override_config_fields=(),
        effective_config_fields=(_typed_contract("event_signal", "dff"),),
        explicit_user_mark=True,
        current_or_stale="current",
    )
    return roi0_default, roi1_default


def _accepted_outcome_with_per_roi_feature_map(
    source_root: Path, output_base: Path, feature_event_map
):
    """Same recipe as tests.test_guided_startup_allocation._accepted_outcome,
    plus both ROIs included and a resolved per-ROI feature/event map on the
    request before validation."""
    request = _valid_request()
    roi1_mark = replace(request.correction.confirmed_marks[0], roi_id="ROI1")
    roi1_evidence = replace(
        request.diagnostic_evidence.evidence_references[0], roi_id="ROI1"
    )
    request = replace(
        request,
        source=replace(
            request.source,
            source_root_canonical=os.path.abspath(source_root),
        ),
        output=replace(
            request.output,
            output_base_canonical=os.path.abspath(output_base),
        ),
        acquisition_dataset=replace(
            request.acquisition_dataset,
            semantic_values=request.acquisition_dataset.semantic_values
            + (_typed("target_fs_hz", 40.0),),
        ),
        roi_scope=replace(
            request.roi_scope,
            discovered_roi_ids=("ROI0", "ROI1"),
            included_roi_ids=("ROI0", "ROI1"),
            excluded_roi_ids=(),
        ),
        correction=replace(
            request.correction,
            confirmed_marks=request.correction.confirmed_marks + (roi1_mark,),
        ),
        diagnostic_evidence=replace(
            request.diagnostic_evidence,
            evidence_references=request.diagnostic_evidence.evidence_references
            + (roi1_evidence,),
        ),
        feature_event=replace(
            request.feature_event,
            per_roi_feature_event_map_version="per_roi_feature_event_map.v1",
            per_roi_feature_event_map=tuple(feature_event_map),
        ),
    )
    identity = validation_request.compute_guided_backend_validation_request_identity(
        request
    )
    compiled = validation_request.GuidedBackendValidationCompileSuccess(
        request, identity
    )
    validated = validator.validate_guided_backend_validation_request(
        request,
        canonical_request_identity=identity,
        validator_contract=_validator_contract(),
    )
    assert validated.accepted
    return GuidedBackendValidationWorkflowOutcome(
        status="validator_accepted",
        accepted_for_backend_validation=True,
        run_authorization=False,
        request_identity=identity,
        validation_result=validated,
        compile_result=compiled,
        materialization_result=object(),
        blocking_issues=(),
        user_summary="Accepted.",
    )


def _request_for_paths_with_per_roi_feature_map(
    monkeypatch,
    *,
    source_root: Path,
    output_base: Path,
    feature_event_map,
    run_id: str = "guided_run_20260101T000000Z_abcdef",
):
    outcome = _accepted_outcome_with_per_roi_feature_map(
        source_root, output_base, feature_event_map
    )
    auth_request = authorization.build_guided_run_authorization_request(
        stored_validation_outcome=outcome,
        stored_validation_outcome_revision=3,
        current_gui_revision=3,
        current_validation_context=GuidedBackendValidationGuiContext(
            draft=GuidedNewAnalysisDraftPlan(),
            parser_contract=build_guided_backend_validation_parser_contract(),
            additional_protected_roots=(),
            validator_contract=_validator_contract(),
            revision=3,
        ),
        application_build_identity=_build_app_identity(),
        production_mapping_contract=mapping.build_guided_production_mapping_contract(),
    )
    monkeypatch.setattr(
        authorization.validation_workflow,
        "validate_current_guided_draft_for_backend",
        lambda *_args, **_kwargs: outcome,
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_candidate_manifest_execution_preflight",
        lambda request, **_kwargs: _accepted_candidate(request),
    )
    monkeypatch.setattr(
        authorization.execution_preflight,
        "run_roi_execution_preflight",
        lambda request, **_kwargs: _accepted_roi(request),
    )
    auth = authorization.authorize_guided_run(auth_request)
    assert auth.authorized
    contract = payloads.build_guided_execution_startup_mapping_contract()
    derived = payloads.derive_guided_execution_payloads(
        auth, startup_mapping_contract=contract
    )
    planned_dir = output_base / run_id
    request = startup.GuidedStartupTransactionRequest(
        authorization_result=auth,
        payload_result=derived,
        startup_mapping_contract=contract,
        application_build_identity=auth.production_intent.application_build_identity,
        current_guided_revision=auth.authorized_gui_revision,
        explicit_user_run_transition=True,
        output_base_canonical=os.path.abspath(output_base),
        source_root_canonical=os.path.abspath(source_root),
        planned_run_id=run_id,
        planned_allocated_run_dir=os.path.abspath(planned_dir),
        wrapper_entrypoint=startup.GuidedWrapperEntrypointIdentity(
            entrypoint_kind="script_path",
            entrypoint_value="tools/run_full_pipeline_deliverables.py",
            trusted_application_root=os.path.abspath(Path.cwd()),
            wrapper_identity_digest="e" * 64,
            supported_contract_version="run_full_pipeline_deliverables.v1",
            supports_guided_preallocated_run_dir=True,
            supports_guided_candidate_manifest=True,
            trusted_entrypoint=True,
        ),
        one_shot_consumption_token="one-shot-token-0001",
        one_shot_token_current=True,
        one_shot_token_unused=True,
        current_time_utc_iso="2026-01-01T00:00:00Z",
        filesystem_policy=startup.GuidedStartupFilesystemPolicy(
            output_base_exists_or_creatable=True,
            output_base_is_directory_or_creatable=True,
            output_base_overlaps_source=False,
            output_base_is_completed_run_root=False,
            output_base_is_guided_diagnostic_cache_root=False,
            output_base_is_protected_ineligible_root=False,
            planned_child_directly_under_base=True,
            planned_child_already_exists=False,
            overwrite_requested=False,
            protected_root_context_complete=True,
        ),
    )
    return request, startup.plan_guided_startup_transaction(request)


@pytest.fixture
def per_roi_case(tmp_path, monkeypatch):
    """Mixed map: ROI0 Custom (override), ROI1 Default."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    return _request_for_paths_with_per_roi_feature_map(
        monkeypatch,
        source_root=source,
        output_base=output,
        feature_event_map=_per_roi_feature_event_map(),
    )


@pytest.fixture
def default_only_case(tmp_path, monkeypatch):
    """Realistic 32c default-only map: both ROIs resolved to the default
    profile, no ROI Customized. Distinct from an empty map."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    return _request_for_paths_with_per_roi_feature_map(
        monkeypatch,
        source_root=source,
        output_base=output,
        feature_event_map=_default_only_feature_event_map(),
    )


def _materialize(request, plan):
    allocated = allocation.allocate_guided_startup_directory(
        request=request, pure_plan=plan
    )
    assert allocated.ok
    result = materialization.materialize_guided_startup_artifacts(
        request=request, pure_plan=plan, allocation_result=allocated
    )
    return allocated, result


# ---------------------------------------------------------------------------
# A. Audit/contract: the artifact is materialized and carries the resolved
#    per-ROI shapes through to the run directory the wrapper subprocess reads.
# ---------------------------------------------------------------------------


def test_materialization_writes_per_roi_feature_config_artifact(per_roi_case):
    request, plan = per_roi_case
    _allocated, result = _materialize(request, plan)

    assert result.status == "startup_artifacts_materialized"
    assert result.ok and result.materialized
    assert startup.GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME in result.files_written

    run_dir = Path(result.allocated_run_dir)
    path = run_dir / startup.GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert set(payload["per_roi_override_config_fields"]) == {"ROI0"}
    assert payload["per_roi_override_config_fields"]["ROI0"] == {
        "peak_threshold_k": 10.0
    }

    effective_for_overrides = payload[
        "per_roi_effective_feature_config_fields_for_overrides"
    ]
    assert set(effective_for_overrides) == {"ROI0"}
    assert set(effective_for_overrides["ROI0"]) == FEATURE_EVENT_CONFIG_FIELDS
    assert effective_for_overrides["ROI0"]["peak_threshold_k"] == 10.0

    provenance = payload["per_roi_feature_provenance"]
    assert set(provenance) == {"ROI0", "ROI1"}
    assert provenance["ROI0"]["source"] == "override"
    assert provenance["ROI1"]["source"] == "default"


def test_orchestration_accepts_per_roi_artifact_and_invokes_wrapper_once(per_roi_case):
    request, _plan = per_roi_case
    calls = []

    def runner(command):
        calls.append(command)
        return orchestration.GuidedWrapperProcessResult(
            returncode=0,
            stdout="",
            stderr="",
            command=command,
            started=True,
            completed=True,
        )

    result = orchestration.run_guided_startup_to_wrapper(
        request=request, subprocess_runner=runner
    )
    assert result.status == "wrapper_completed"
    assert result.ok
    assert calls == [result.wrapper_command]
    run_dir = Path(result.allocated_run_dir)
    assert (run_dir / startup.GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME).exists()


# ---------------------------------------------------------------------------
# B. Standard Pipeline execution: analyze_photometry.py's Pipeline
#    construction site receives per-ROI settings for the Custom ROI only.
# ---------------------------------------------------------------------------


def _write_per_roi_feature_config_artifact(run_dir: Path):
    payload = {
        "schema_name": "guided_per_roi_feature_config",
        "schema_version": "v1",
        "per_roi_override_config_fields": {"CH1": {"peak_threshold_k": 10.0}},
        "per_roi_effective_feature_config_fields_for_overrides": {
            "CH1": {
                "event_signal": "dff",
                "signal_excursion_polarity": "positive",
                "peak_threshold_method": "mean_std",
                "peak_threshold_k": 10.0,
                "peak_threshold_percentile": 95.0,
                "peak_threshold_abs": 0.0,
                "peak_min_distance_sec": 1.0,
                "peak_min_prominence_k": 2.0,
                "peak_min_width_sec": 0.3,
                "peak_pre_filter": "none",
                "event_auc_baseline": "zero",
            }
        },
        "per_roi_feature_provenance": {
            "CH1": {
                "source": "override",
                "feature_event_profile_id": "custom-ch1",
                "override_config_fields": {"peak_threshold_k": 10.0},
                "effective_config_fields": {"peak_threshold_k": 10.0},
            },
            "CH2": {
                "source": "default",
                "feature_event_profile_id": "profile-001",
                "override_config_fields": {},
                "effective_config_fields": {"event_signal": "dff"},
            },
        },
    }
    manifest_path = run_dir / "guided_candidate_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    (run_dir / startup.GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return manifest_path


def test_load_guided_per_roi_feature_settings_builds_ch1_only_config(tmp_path):
    from analyze_photometry import load_guided_per_roi_feature_settings

    manifest_path = _write_per_roi_feature_config_artifact(tmp_path)
    base_config = Config()

    per_roi_feature_config, per_roi_feature_provenance = (
        load_guided_per_roi_feature_settings(str(manifest_path), base_config)
    )

    assert set(per_roi_feature_config) == {"CH1"}
    assert per_roi_feature_config["CH1"].peak_threshold_k == 10.0
    # Every other field on CH1's Config is untouched (still the base config).
    assert (
        per_roi_feature_config["CH1"].peak_threshold_method
        == base_config.peak_threshold_method
    )
    assert set(per_roi_feature_provenance) == {"CH1", "CH2"}
    assert per_roi_feature_provenance["CH1"]["source"] == "override"
    assert per_roi_feature_provenance["CH2"]["source"] == "default"


def test_load_guided_per_roi_feature_settings_absent_returns_none(tmp_path):
    from analyze_photometry import load_guided_per_roi_feature_settings

    base_config = Config()
    # No guided_candidate_manifest at all (plain non-Guided invocation).
    assert load_guided_per_roi_feature_settings(None, base_config) == (None, None)

    # guided_candidate_manifest set, but no sibling per-ROI artifact exists
    # (global-only Guided run).
    manifest_path = tmp_path / "guided_candidate_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    assert load_guided_per_roi_feature_settings(str(manifest_path), base_config) == (
        None,
        None,
    )


def _analyze_photometry_argv(tmp_path, manifest_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    return [
        "analyze_photometry.py",
        "--input", str(tmp_path / "in"),
        "--out", str(tmp_path / "out"),
        "--config", str(config_path),
        "--format", "rwd",
        "--mode", "phasic",
        "--guided-candidate-manifest", str(manifest_path),
    ]


def test_analyze_photometry_wires_per_roi_feature_config_into_pipeline(tmp_path):
    manifest_path = _write_per_roi_feature_config_artifact(tmp_path)
    argv = _analyze_photometry_argv(tmp_path, manifest_path)

    import analyze_photometry

    with patch("sys.argv", argv), patch(
        "analyze_photometry.Pipeline"
    ) as mock_pipeline_cls:
        mock_pipeline_cls.return_value.roi_selection = None
        try:
            analyze_photometry.main()
        except SystemExit:
            pass

    args, kwargs = mock_pipeline_cls.call_args
    assert set(kwargs["per_roi_feature_config"]) == {"CH1"}
    assert kwargs["per_roi_feature_config"]["CH1"].peak_threshold_k == 10.0
    assert "CH2" not in kwargs["per_roi_feature_config"]
    assert set(kwargs["per_roi_feature_provenance"]) == {"CH1", "CH2"}


# ---------------------------------------------------------------------------
# C. applied-dF/F orchestration: tools/run_full_pipeline_deliverables.py
#    passes complete effective fields, never the sparse override.
# ---------------------------------------------------------------------------


def test_load_guided_per_roi_feature_event_overrides_uses_complete_fields(tmp_path):
    from tools.run_full_pipeline_deliverables import (
        _load_guided_per_roi_feature_event_overrides,
    )
    from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS

    _write_per_roi_feature_config_artifact(tmp_path)

    overrides = _load_guided_per_roi_feature_event_overrides(str(tmp_path))

    assert set(overrides) == {"CH1"}
    assert set(overrides["CH1"]) == FEATURE_EVENT_CONFIG_FIELDS
    assert overrides["CH1"]["peak_threshold_k"] == 10.0


def test_load_guided_per_roi_feature_event_overrides_absent_returns_none(tmp_path):
    from tools.run_full_pipeline_deliverables import (
        _load_guided_per_roi_feature_event_overrides,
    )

    assert _load_guided_per_roi_feature_event_overrides(str(tmp_path)) is None


def test_applied_dff_call_site_wires_per_roi_feature_event_overrides():
    """Audit/contract: the applied-dF/F orchestration call site in
    tools/run_full_pipeline_deliverables.py must pass per_roi_feature_event_overrides
    built from the materialized artifact, not omit it (the pre-4J16k34 gap)."""
    import inspect
    import tools.run_full_pipeline_deliverables as wrapper

    source = inspect.getsource(wrapper)
    call_start = source.index("ran = run_guided_applied_dff_orchestration_if_enabled(")
    call_text = source[call_start : call_start + 400]
    assert "per_roi_feature_event_overrides" in call_text
    assert "_load_guided_per_roi_feature_event_overrides(run_dir)" in call_text


# ---------------------------------------------------------------------------
# D. Global-only regression: no Custom ROI means unaffected behavior, both
#    for an empty per-ROI map (older/no-map request shape) and for the
#    realistic 32c default-only map (non-empty map, every entry source=
#    "default", nothing Customized).
# ---------------------------------------------------------------------------


def test_empty_map_materialization_writes_no_per_roi_artifact(tmp_path, monkeypatch):
    from tests.test_guided_startup_allocation import _request_for_paths

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    request, plan = _request_for_paths(
        monkeypatch, source_root=source, output_base=output
    )
    assert request.authorization_result.production_intent.feature_event.per_roi_feature_event_map == ()

    _allocated, result = _materialize(request, plan)

    assert result.status == "startup_artifacts_materialized"
    assert startup.GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME not in result.files_written


def test_default_only_map_materialization_writes_no_per_roi_artifact(default_only_case):
    """The realistic 32c case: a non-empty per_roi_feature_event_map where
    every included ROI resolved to the default profile. Must not write the
    artifact -- writing here would be the bug this follow-up fixes."""
    request, plan = default_only_case
    feature_event = request.authorization_result.production_intent.feature_event
    assert feature_event.per_roi_feature_event_map != ()
    assert all(
        entry.source == "default" for entry in feature_event.per_roi_feature_event_map
    )

    _allocated, result = _materialize(request, plan)

    assert result.status == "startup_artifacts_materialized"
    assert startup.GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME not in result.files_written


def test_default_only_map_analyze_photometry_does_not_pass_provenance(
    default_only_case,
):
    """End-to-end through the real materialized run directory: since no
    artifact was written for a default-only map, analyze_photometry.py's
    loader (and therefore Pipeline) must not receive per_roi_feature_config
    or per_roi_feature_provenance -- global-only execution, unchanged."""
    from analyze_photometry import load_guided_per_roi_feature_settings

    request, plan = default_only_case
    _allocated, result = _materialize(request, plan)
    assert result.status == "startup_artifacts_materialized"

    manifest_path = Path(result.allocated_run_dir) / startup.GUIDED_CANDIDATE_MANIFEST_FILENAME
    assert manifest_path.exists()

    per_roi_feature_config, per_roi_feature_provenance = (
        load_guided_per_roi_feature_settings(str(manifest_path), Config())
    )
    assert per_roi_feature_config is None
    assert per_roi_feature_provenance is None


def test_analyze_photometry_global_only_pipeline_kwargs_are_none(tmp_path):
    """No guided_candidate_manifest at all: Pipeline gets None/None, matching
    pre-4J16k34 behavior exactly (no behavior change for non-Guided runs)."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    argv = [
        "analyze_photometry.py",
        "--input", str(tmp_path / "in"),
        "--out", str(tmp_path / "out"),
        "--config", str(config_path),
    ]

    import analyze_photometry

    with patch("sys.argv", argv), patch(
        "analyze_photometry.Pipeline"
    ) as mock_pipeline_cls:
        mock_pipeline_cls.return_value.roi_selection = None
        try:
            analyze_photometry.main()
        except SystemExit:
            pass

    _args, kwargs = mock_pipeline_cls.call_args
    assert kwargs["per_roi_feature_config"] is None
    assert kwargs["per_roi_feature_provenance"] is None


# ---------------------------------------------------------------------------
# E. End-to-end behavioral test: real analyze_photometry.main() run over a
#    synthetic two-ROI RWD session where ROI0 and ROI1 carry byte-identical
#    underlying traces, so a difference in detected peak counts can only come
#    from the per-ROI feature-detection settings, not from different data.
# ---------------------------------------------------------------------------


def _write_synthetic_rwd_session(root: Path, name: str = "session_a"):
    path = root / name / "fluorescence.csv"
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(42)
    fs = 20.0
    n = 400
    t = np.arange(n) / fs
    uv = 100 + 5 * np.sin(0.05 * t) + rng.normal(0, 0.5, n)
    calcium = np.zeros(n)
    for start in (60, 160, 260, 320):
        length = n - start
        t_local = np.arange(length)
        calcium[start:] += 30 * np.exp(-t_local / 40.0)
    sig = 2.0 * uv + calcium + rng.normal(0, 0.5, n)
    df = pd.DataFrame(
        {
            "Time(s)": t,
            "ROI0-410": uv,
            "ROI0-470": sig,
            "ROI1-410": uv,
            "ROI1-470": sig,
        }
    )
    df.to_csv(path, index=False)


def _build_two_roi_guided_manifest(tmp_path):
    root = tmp_path / "source"
    _write_synthetic_rwd_session(root)
    config = Config()
    facts = build_guided_manifest_current_facts(
        source_root=root,
        config=config,
        manifest_included_roi_ids=("ROI0", "ROI1"),
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
    manifest_path = tmp_path / "guided_manifest.json"
    payload = asdict(manifest)
    payload["candidate_files"] = [asdict(item) for item in manifest.candidate_files]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return root, manifest_path


def test_end_to_end_custom_roi_feature_counts_diverge_from_default(tmp_path):
    root, manifest_path = _build_two_roi_guided_manifest(tmp_path)

    per_roi_payload = {
        "schema_name": "guided_per_roi_feature_config",
        "schema_version": "v1",
        "per_roi_override_config_fields": {
            "ROI0": {
                "peak_threshold_method": "percentile",
                "peak_threshold_percentile": 1.0,
                "peak_min_prominence_k": 0.0,
                "peak_min_width_sec": 0.0,
            }
        },
        "per_roi_effective_feature_config_fields_for_overrides": {
            "ROI0": {
                "event_signal": "dff",
                "signal_excursion_polarity": "positive",
                "peak_threshold_method": "percentile",
                "peak_threshold_k": 2.5,
                "peak_threshold_percentile": 1.0,
                "peak_threshold_abs": 0.0,
                "peak_min_distance_sec": 1.0,
                "peak_min_prominence_k": 0.0,
                "peak_min_width_sec": 0.0,
                "peak_pre_filter": "none",
                "event_auc_baseline": "zero",
            }
        },
        "per_roi_feature_provenance": {
            "ROI0": {
                "source": "override",
                "feature_event_profile_id": "custom-roi0",
                "override_config_fields": {"peak_threshold_method": "percentile"},
                "effective_config_fields": {"peak_threshold_method": "percentile"},
            },
            "ROI1": {
                "source": "default",
                "feature_event_profile_id": "profile-001",
                "override_config_fields": {},
                "effective_config_fields": {"peak_threshold_method": "mean_std"},
            },
        },
    }
    (manifest_path.parent / startup.GUIDED_PER_ROI_FEATURE_CONFIG_FILENAME).write_text(
        json.dumps(per_roi_payload), encoding="utf-8"
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "chunk_duration_sec: 20.0\ntarget_fs_hz: 20.0\nallow_partial_final_chunk: true\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "analyze_photometry.py"),
        "--input", str(root),
        "--out", str(out_dir),
        "--config", str(config_path),
        "--format", "rwd",
        "--mode", "phasic",
        "--guided-candidate-manifest", str(manifest_path),
    ]

    # Run as a real subprocess, exactly as Guided execution does in
    # production: analyze_photometry.py imports and runs the real Pipeline
    # (photometry_pipeline.core.feature_extraction included), which would
    # otherwise pollute sys.modules for the rest of this test session and
    # break unrelated purity/non-execution-guarantee tests elsewhere.
    completed = subprocess.run(
        cmd, cwd=str(repo_root), capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    feats = pd.read_csv(
        os.path.join(out_dir, "features", "features.csv")
    ).set_index("roi")

    assert feats.loc["ROI0", "peak_count"] > feats.loc["ROI1", "peak_count"]

    provenance_path = os.path.join(
        out_dir, "features", "feature_event_provenance.json"
    )
    assert os.path.exists(provenance_path)
    with open(provenance_path, "r", encoding="utf-8") as handle:
        provenance_payload = json.load(handle)
    by_roi = {entry["roi"]: entry for entry in provenance_payload["rois"]}
    assert by_roi["ROI0"]["source"] == "override"
    assert by_roi["ROI1"]["source"] == "default"
