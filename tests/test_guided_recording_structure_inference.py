from photometry_pipeline.guided_recording_structure_inference import (
    infer_guided_recording_structure,
)


def _discovery(session_ids, resolved_format="RWD", input_dir="/data/first"):
    return {
        "resolved_format": resolved_format,
        "input_dir": input_dir,
        "sessions": [
            {
                "session_id": session_id,
                "path": f"{input_dir}/{session_id}/fluorescence.csv",
            }
            for session_id in session_ids
        ],
    }


def test_rwd_recording_structure_inference_uses_duration_and_cadence():
    discovery = _discovery(
        [
            "2025_01_01-00_00_00",
            "2025_01_01-00_30_00",
            "2025_01_01-01_00_00",
        ]
    )

    result = infer_guided_recording_structure(
        discovery,
        "/data/first",
        "rwd",
        rwd_chunk_contracts=[
            {"chunk_duration_sec": 600.0},
            {"chunk_duration_sec": 600.1},
            {"chunk_duration_sec": 599.9},
        ],
    )

    assert result.supported is True
    assert result.status == "inferred"
    assert result.sessions_per_hour == 2
    assert result.session_duration_sec == 600.0
    assert result.evidence["duration_source"] == "rwd_chunk_contracts"
    assert result.evidence["cadence_source"] == "ordered_rwd_session_ids"


def test_rwd_recording_structure_inference_does_not_invent_single_chunk_cadence():
    result = infer_guided_recording_structure(
        _discovery(["2025_01_01-00_00_00"]),
        "/data/first",
        "rwd",
        rwd_chunk_contracts=[{"chunk_duration_sec": 600.0}],
    )

    assert result.supported is True
    assert result.status == "ambiguous"
    assert result.sessions_per_hour is None
    assert result.session_duration_sec == 600.0


def test_rwd_inferred_duration_normalizes_only_near_whole_seconds():
    discovery = _discovery(
        ["2025_01_01-00_00_00", "2025_01_01-00_30_00"]
    )
    normalized = infer_guided_recording_structure(
        discovery,
        "/data/first",
        "rwd",
        rwd_chunk_contracts=[
            {"chunk_duration_sec": 599.988},
            {"chunk_duration_sec": 599.988},
        ],
    )
    fractional = infer_guided_recording_structure(
        discovery,
        "/data/first",
        "rwd",
        rwd_chunk_contracts=[
            {"chunk_duration_sec": 599.4},
            {"chunk_duration_sec": 599.4},
        ],
    )

    assert normalized.session_duration_sec == 600.0
    assert "~600 s/session" in normalized.message
    assert normalized.evidence["raw_session_duration_sec"] == 599.988
    assert normalized.evidence["display_session_duration_sec"] == 600.0
    assert normalized.evidence["duration_display_normalized"] is True
    assert fractional.session_duration_sec == 599.4
    assert "~599.4 s/session" in fractional.message
    assert fractional.evidence["duration_display_normalized"] is False


def test_recording_structure_inference_rejects_unsupported_formats():
    for input_format in ("custom_tabular", "npm", "unknown"):
        result = infer_guided_recording_structure(
            _discovery([], resolved_format=input_format),
            "/data/unsupported",
            input_format,
        )

        assert result.supported is False
        assert result.status == "unsupported"
        assert result.sessions_per_hour is None
        assert result.session_duration_sec is None
