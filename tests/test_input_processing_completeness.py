"""Input-processing completeness: one terminal disposition per admitted chunk (4J16k41 / C8)."""

import json
from pathlib import Path

import pytest

from photometry_pipeline.input_processing_completeness import (
    DISPOSITION_AUTHORIZED_EXCLUSION,
    DISPOSITION_PROCESS,
    INPUT_COMPLETENESS_CONTRACT_VERSION,
    POLICY_INCOMPLETE_FINAL_RWD_CHUNK,
    InputProcessingAccountant,
    InputProcessingError,
    source_drift_reason,
    source_identity,
    validate_input_completeness,
)


def _write_source(path: Path, text: str = "data") -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _admitted(tmp_path: Path, n: int, *, exclude_final: bool = False):
    sources = [_write_source(tmp_path / f"chunk_{i}.csv", f"chunk-{i}") for i in range(n)]
    excluded = sources[-1] if exclude_final else None
    accountant = InputProcessingAccountant.from_admitted_manifest(
        acquisition_mode="intermittent",
        input_format="rwd",
        ordered_sources=sources,
        excluded_source=excluded,
        exclusion_policy=POLICY_INCOMPLETE_FINAL_RWD_CHUNK if excluded else "",
    )
    return accountant, sources


# Accountant: fail-closed accounting ------------------------------------------


def test_all_admitted_chunks_processed_finalizes(tmp_path: Path):
    accountant, sources = _admitted(tmp_path, 3)
    for i, src in enumerate(sources):
        accountant.before_load(src, phase="pass2")
        accountant.mark_processed(src, cache_chunk_id=i)

    payload = accountant.finalize()
    assert validate_input_completeness(payload) == ""
    assert len(payload["processed"]) == 3
    assert [e["disposition"] for e in payload["expected"]] == [DISPOSITION_PROCESS] * 3


def test_unprocessed_admitted_chunk_fails_finalize(tmp_path: Path):
    accountant, sources = _admitted(tmp_path, 3)
    accountant.mark_processed(sources[0], cache_chunk_id=0)
    accountant.mark_processed(sources[2], cache_chunk_id=2)  # middle omitted

    with pytest.raises(InputProcessingError) as excinfo:
        accountant.finalize()
    assert excinfo.value.category == "unprocessed_admitted_chunk"


def test_authorized_final_exclusion_is_not_required_to_process(tmp_path: Path):
    accountant, sources = _admitted(tmp_path, 3, exclude_final=True)
    accountant.mark_processed(sources[0], cache_chunk_id=0)
    accountant.mark_processed(sources[1], cache_chunk_id=1)

    payload = accountant.finalize()
    assert validate_input_completeness(payload) == ""
    dispositions = [e["disposition"] for e in payload["expected"]]
    assert dispositions == [DISPOSITION_PROCESS, DISPOSITION_PROCESS, DISPOSITION_AUTHORIZED_EXCLUSION]
    assert payload["expected"][-1]["policy"] == POLICY_INCOMPLETE_FINAL_RWD_CHUNK


def test_processing_an_excluded_chunk_is_rejected(tmp_path: Path):
    accountant, sources = _admitted(tmp_path, 2, exclude_final=True)
    with pytest.raises(InputProcessingError) as excinfo:
        accountant.mark_processed(sources[-1], cache_chunk_id=1)
    assert excinfo.value.category == "excluded_chunk_processed"


# Source drift -----------------------------------------------------------------


def test_source_drift_detects_missing_resized_and_swapped(tmp_path: Path):
    src = tmp_path / "chunk.csv"
    identity = source_identity(_write_source(src, "original-content"))
    assert source_drift_reason(identity) == ""

    src.write_text("original-conten!", encoding="utf-8")  # same length, different bytes
    assert source_drift_reason(identity, size_only=True) == ""
    assert "contents changed" in source_drift_reason(identity)

    src.write_text("longer content now", encoding="utf-8")
    assert "changed size" in source_drift_reason(identity, size_only=True)

    src.unlink()
    assert "missing" in source_drift_reason(identity)


def test_before_load_fails_on_source_drift(tmp_path: Path):
    accountant, sources = _admitted(tmp_path, 2)
    Path(sources[1]).write_text("resized-differently-now", encoding="utf-8")
    with pytest.raises(InputProcessingError) as excinfo:
        accountant.before_load(sources[1], phase="load")
    assert excinfo.value.category == "source_drift"


def test_before_load_fails_on_unexpected_source(tmp_path: Path):
    accountant, _sources = _admitted(tmp_path, 2)
    stray = _write_source(tmp_path / "stray.csv")
    with pytest.raises(InputProcessingError) as excinfo:
        accountant.before_load(stray, phase="load")
    assert excinfo.value.category == "unexpected_source"


def test_building_over_a_missing_source_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        InputProcessingAccountant.from_admitted_manifest(
            acquisition_mode="intermittent",
            input_format="rwd",
            ordered_sources=[str(tmp_path / "does_not_exist.csv")],
        )


# Record validation ------------------------------------------------------------


def _valid_payload(n=3, *, exclude_final=False):
    expected, processed = [], []
    for i in range(n):
        entry = {"source": f"/s/{i}", "size_bytes": i + 1, "sha256": f"d{i}", "index": i,
                 "disposition": DISPOSITION_PROCESS}
        if exclude_final and i == n - 1:
            entry["disposition"] = DISPOSITION_AUTHORIZED_EXCLUSION
            entry["policy"] = POLICY_INCOMPLETE_FINAL_RWD_CHUNK
        else:
            processed.append({"index": i, "source": entry["source"], "cache_chunk_id": i})
        expected.append(entry)
    return {
        "contract_version": INPUT_COMPLETENESS_CONTRACT_VERSION,
        "acquisition_mode": "intermittent",
        "input_format": "rwd",
        "expected": expected,
        "processed": processed,
    }


def test_valid_record_passes():
    assert validate_input_completeness(_valid_payload()) == ""
    assert validate_input_completeness(_valid_payload(exclude_final=True)) == ""


def test_unsupported_version_rejected():
    payload = _valid_payload()
    payload["contract_version"] = "input_processing_completeness.v99"
    assert "unsupported version" in validate_input_completeness(payload)


def test_processed_fewer_than_admitted_rejected():
    payload = _valid_payload()
    payload["processed"] = payload["processed"][:-1]  # drop one processed record
    assert "never processed" in validate_input_completeness(payload)


def test_duplicate_processed_record_rejected():
    payload = _valid_payload()
    payload["processed"].append(dict(payload["processed"][0]))
    assert "more than once" in validate_input_completeness(payload)


def test_processed_index_not_admitted_rejected():
    payload = _valid_payload()
    payload["processed"].append({"index": 99, "source": "/s/99", "cache_chunk_id": 99})
    assert "not an admitted" in validate_input_completeness(payload)


def test_non_final_exclusion_rejected():
    payload = _valid_payload()
    payload["expected"][0]["disposition"] = DISPOSITION_AUTHORIZED_EXCLUSION
    payload["expected"][0]["policy"] = POLICY_INCOMPLETE_FINAL_RWD_CHUNK
    payload["processed"] = [p for p in payload["processed"] if p["index"] != 0]
    assert "not the final chronological chunk" in validate_input_completeness(payload)


def test_two_exclusions_rejected():
    payload = _valid_payload(n=3)
    for idx in (1, 2):
        payload["expected"][idx]["disposition"] = DISPOSITION_AUTHORIZED_EXCLUSION
        payload["expected"][idx]["policy"] = POLICY_INCOMPLETE_FINAL_RWD_CHUNK
    payload["processed"] = [p for p in payload["processed"] if p["index"] == 0]
    assert "more than one authorized exclusion" in validate_input_completeness(payload)


def test_exclusion_without_policy_rejected():
    payload = _valid_payload(exclude_final=True)
    payload["expected"][-1].pop("policy")
    assert "without a policy" in validate_input_completeness(payload)


def test_non_contiguous_indices_rejected():
    payload = _valid_payload(n=2)
    payload["expected"][1]["index"] = 5
    payload["processed"][1]["index"] = 5
    assert "contiguous" in validate_input_completeness(payload)


# Scientist-facing message -----------------------------------------------------


def test_scientist_message_is_plain_and_names_the_segment():
    err = InputProcessingError(
        chunk_index=1, source="/data/2024_01_01-01_00_00/fluorescence.csv",
        phase="pass2", category="processing_exception", reason="ValueError: boom",
    )
    msg = err.scientist_message()
    assert "recording segment 2" in msg
    assert "could not be processed" in msg
    assert "Traceback" not in msg
    assert "ValueError" not in msg
    # The developer-facing form still carries the identity and category.
    assert "processing_exception" in str(err)
