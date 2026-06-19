import pytest

from photometry_pipeline.guided_diagnostic_cache import (
    DIAGNOSTIC_CACHE_PURPOSE,
    DiagnosticCacheArtifactRecord,
    DiagnosticCacheBuildRequest,
    DiagnosticCacheContractError,
    artifact_record_from_request,
    compare_request_to_artifact,
    read_artifact_record_json,
    read_build_request_json,
    validate_diagnostic_cache_artifact,
    write_artifact_record_json,
    write_build_request_json,
)


def _request(tmp_path, **overrides):
    values = {
        "raw_input_path": str(tmp_path / "raw_input"),
        "input_format": "npm",
        "acquisition_mode": "intermittent",
        "sessions_per_hour": 6,
        "session_duration_sec": 300.0,
        "included_roi_ids": ("CH1", "CH2"),
        "excluded_roi_ids": ("CH3",),
        "baseline_config_source_path": str(tmp_path / "baseline.yaml"),
        "baseline_config_source_kind": "custom_config",
        "config_identity": "sha256:abc",
        "diagnostic_scope": "full_selected_input",
        "preview_first_n": None,
        "output_base": str(tmp_path / "output"),
        "requested_cache_path": str(tmp_path / "output" / "_guided_diagnostic_cache" / "cache_001"),
        "requested_at_utc": "2026-06-19T12:00:00Z",
    }
    values.update(overrides)
    return DiagnosticCacheBuildRequest(**values)


def _artifact_paths(tmp_path):
    cache_root = tmp_path / "output" / "_guided_diagnostic_cache" / "cache_001"
    phasic = cache_root / "_analysis" / "phasic_out"
    phasic.mkdir(parents=True)
    status = cache_root / "status.json"
    run_report = cache_root / "run_report.json"
    cache = phasic / "phasic_trace_cache.h5"
    config = phasic / "config_used.yaml"
    status.write_text("{}", encoding="utf-8")
    run_report.write_text("{}", encoding="utf-8")
    cache.write_bytes(b"cache")
    config.write_text("event_signal: dff\n", encoding="utf-8")
    return cache_root, status, run_report, cache, config


def _record(tmp_path, request=None, **overrides):
    request = request or _request(tmp_path)
    cache_root, status, run_report, cache, config = _artifact_paths(tmp_path)
    values = {
        "cache_id": "cache_001",
        "cache_root_path": str(cache_root),
        "status_marker_path": str(status),
        "run_report_path": str(run_report),
        "phasic_trace_cache_path": str(cache),
        "config_used_path": str(config),
        "roi_inventory": ("CH1", "CH2"),
        "created_at_utc": "2026-06-19T12:05:00Z",
    }
    values.update(overrides)
    return artifact_record_from_request(request, **values)


def test_build_request_constructs_representative_intermittent_npm(tmp_path):
    request = _request(tmp_path)

    assert request.input_format == "npm"
    assert request.acquisition_mode == "intermittent"
    assert request.sessions_per_hour == 6
    assert request.session_duration_sec == pytest.approx(300.0)
    assert request.source_setup_signature
    assert request.diagnostic_scope_signature
    assert request.request_signature


def test_build_request_constructs_representative_continuous_custom_tabular(tmp_path):
    request = _request(
        tmp_path,
        input_format="custom_tabular",
        acquisition_mode="continuous",
        sessions_per_hour=None,
        session_duration_sec=None,
        continuous_window_sec=600.0,
        continuous_step_sec=600.0,
        allow_partial_final_window=True,
        diagnostic_scope="first_n",
        preview_first_n=5,
    )

    assert request.input_format == "custom_tabular"
    assert request.acquisition_mode == "continuous"
    assert request.allow_partial_final_window is True
    assert request.preview_first_n == 5


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"raw_input_path": ""}, "raw_input_path is required"),
        ({"output_base": ""}, "output_base is required"),
        ({"requested_cache_path": ""}, "requested_cache_path is required"),
        ({"included_roi_ids": ()}, "included_roi_ids must contain at least one ROI"),
        ({"input_format": "csv"}, "unsupported input_format"),
        ({"acquisition_mode": "burst"}, "unsupported acquisition_mode"),
        ({"diagnostic_scope": "first_n", "preview_first_n": None}, "preview_first_n must be"),
    ],
)
def test_build_request_rejects_invalid_required_fields(tmp_path, overrides, expected):
    with pytest.raises(DiagnosticCacheContractError, match=expected):
        _request(tmp_path, **overrides)


def test_build_request_json_round_trip(tmp_path):
    request = _request(tmp_path, included_roi_ids=("CH2", "CH1"))
    restored = DiagnosticCacheBuildRequest.from_json_dict(request.to_json_dict())

    assert restored.to_json_dict() == request.to_json_dict()
    assert restored.request_signature == request.request_signature


def test_build_request_file_round_trip(tmp_path):
    request = _request(tmp_path)
    path = tmp_path / "request.json"

    write_build_request_json(path, request)
    restored = read_build_request_json(path)

    assert restored.to_json_dict() == request.to_json_dict()


def test_signature_same_semantic_roi_set_ignores_order(tmp_path):
    a = _request(tmp_path, included_roi_ids=("CH1", "CH2"), excluded_roi_ids=("CH3", "CH4"))
    b = _request(tmp_path, included_roi_ids=("CH2", "CH1"), excluded_roi_ids=("CH4", "CH3"))

    assert a.source_setup_signature == b.source_setup_signature
    assert a.request_signature == b.request_signature


@pytest.mark.parametrize(
    "overrides, changed_signature",
    [
        ({"raw_input_path": "different"}, "source_setup_signature"),
        ({"input_format": "custom_tabular"}, "source_setup_signature"),
        ({"acquisition_mode": "continuous"}, "source_setup_signature"),
        ({"sessions_per_hour": 4}, "source_setup_signature"),
        ({"included_roi_ids": ("CH1",)}, "source_setup_signature"),
        ({"baseline_config_source_path": "other.yaml"}, "source_setup_signature"),
        ({"config_identity": "sha256:def"}, "source_setup_signature"),
        ({"diagnostic_scope": "first_n", "preview_first_n": 3}, "diagnostic_scope_signature"),
    ],
)
def test_signatures_change_for_relevant_setup_and_scope_fields(tmp_path, overrides, changed_signature):
    base = _request(tmp_path)
    changed = _request(tmp_path, **overrides)

    assert getattr(base, changed_signature) != getattr(changed, changed_signature)
    assert base.request_signature != changed.request_signature


def test_artifact_record_identity_and_json_round_trip(tmp_path):
    request = _request(tmp_path)
    record = _record(tmp_path, request)

    assert record.purpose == DIAGNOSTIC_CACHE_PURPOSE
    assert record.production_analysis is False

    restored = DiagnosticCacheArtifactRecord.from_json_dict(record.to_json_dict())
    assert restored.to_json_dict() == record.to_json_dict()


def test_artifact_record_file_round_trip(tmp_path):
    record = _record(tmp_path)
    path = tmp_path / "artifact.json"

    write_artifact_record_json(path, record)
    restored = read_artifact_record_json(path)

    assert restored.to_json_dict() == record.to_json_dict()


def test_artifact_validation_success_when_required_files_exist(tmp_path):
    record = _record(tmp_path)

    status = validate_diagnostic_cache_artifact(record)

    assert status.ok
    assert status.code == "ok"
    assert status.missing_artifacts == ()


def test_artifact_validation_fails_when_phasic_cache_missing(tmp_path):
    record = _record(tmp_path)
    (tmp_path / "output" / "_guided_diagnostic_cache" / "cache_001" / "_analysis" / "phasic_out" / "phasic_trace_cache.h5").unlink()

    status = validate_diagnostic_cache_artifact(record)

    assert not status.ok
    assert status.code == "missing_artifacts"
    assert "phasic_trace_cache_path" in status.missing_artifacts


def test_artifact_validation_fails_when_config_used_missing(tmp_path):
    record = _record(tmp_path)
    (tmp_path / "output" / "_guided_diagnostic_cache" / "cache_001" / "_analysis" / "phasic_out" / "config_used.yaml").unlink()

    status = validate_diagnostic_cache_artifact(record)

    assert not status.ok
    assert status.code == "missing_artifacts"
    assert "config_used_path" in status.missing_artifacts


def test_artifact_validation_fails_if_purpose_is_not_diagnostic_cache(tmp_path):
    record = _record(tmp_path, purpose="completed_run")

    status = validate_diagnostic_cache_artifact(record)

    assert not status.ok
    assert status.code == "invalid_purpose"


def test_artifact_validation_fails_if_marked_production_analysis(tmp_path):
    record = _record(tmp_path, production_analysis=True)

    status = validate_diagnostic_cache_artifact(record)

    assert not status.ok
    assert status.code == "production_analysis_not_allowed"


def test_matching_request_and_artifact_are_current(tmp_path):
    request = _request(tmp_path)
    record = _record(tmp_path, request)

    status = compare_request_to_artifact(request, record)

    assert status.ok
    assert status.code == "current"
    assert not status.stale
    assert status.stale_reasons == ()


def test_changed_request_returns_structured_stale_reasons(tmp_path):
    original = _request(tmp_path)
    record = _record(tmp_path, original)
    changed = _request(
        tmp_path,
        input_format="custom_tabular",
        included_roi_ids=("CH1",),
        diagnostic_scope="first_n",
        preview_first_n=2,
    )

    status = compare_request_to_artifact(changed, record)

    assert not status.ok
    assert status.code == "stale"
    assert status.stale
    assert "input format changed" in status.stale_reasons
    assert "ROI inclusion/exclusion changed" in status.stale_reasons
    assert "diagnostic scope changed" in status.stale_reasons
    assert "build request changed" in status.stale_reasons
