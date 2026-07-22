from __future__ import annotations

from dataclasses import replace

import pytest

from photometry_pipeline import guided_continuous_rwd_correction_pass as c4c
from photometry_pipeline import guided_continuous_rwd_correction_run as subject
from photometry_pipeline.io.hdf5_cache import Hdf5TraceCacheWriter
from photometry_pipeline.io.hdf5_cache_reader import (
    list_cache_chunk_ids,
    list_cache_rois,
    open_phasic_cache,
)
from photometry_pipeline.run_completion_contract import (
    TERMINAL_SUCCESS_CURRENT,
    classify_run_terminal_state,
)

# Reuse the D1 module's synthetic-recording builders rather than duplicating
# them: they already produce a coherent, accepted set of continuous-RWD
# authorities (review binding, target grid, block plan, segment plan,
# dynamic-F0 authority) through the real accepted construction path.
from tests.test_guided_continuous_rwd_correction_pass_persistence import (
    _build_case,
    _pass_inputs,
)


@pytest.fixture(scope="module")
def accepted_case(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cr1_d2") / "recording"
    return _build_case(folder)


@pytest.fixture(scope="module")
def real_config(accepted_case):
    from photometry_pipeline.guided_continuous_rwd_segment_correction import (
        _resolve_segment_correction_settings,
    )

    _binding, _grid, _draft, contract, _source = accepted_case
    config, _identity = _resolve_segment_correction_settings(contract)
    return config


def _run(inputs, real_config, output_base, **kwargs):
    binding, grid, draft, contract, block_plan, segment_plan, f0, _source = inputs
    return subject.execute_guided_continuous_rwd_correction_run(
        binding,
        grid,
        block_plan,
        segment_plan,
        f0,
        accepted_draft=draft,
        startup_mapping_contract=contract,
        output_base=str(output_base),
        config=real_config,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Successful lifecycle
# ---------------------------------------------------------------------------


def test_successful_lifecycle_publishes_current_run(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    binding, grid = inputs[0], inputs[1]
    result = _run(inputs, real_config, tmp_path)

    assert result.terminal_state == TERMINAL_SUCCESS_CURRENT
    assert result.run_dir.startswith(str(tmp_path))
    import os

    assert os.path.isfile(os.path.join(result.run_dir, "status.json"))
    assert os.path.isfile(os.path.join(result.run_dir, "MANIFEST.json"))
    assert os.path.isfile(os.path.join(result.run_dir, "run_report.json"))
    assert os.path.isfile(result.corrected_cache_path)

    classification = classify_run_terminal_state(result.run_dir)
    assert classification.is_success
    assert classification.state == TERMINAL_SUCCESS_CURRENT

    cache = open_phasic_cache(result.corrected_cache_path)
    try:
        assert list_cache_rois(cache) == list(binding.recording.roi.included_roi_ids)
        chunk_ids = list_cache_chunk_ids(cache)
        assert len(chunk_ids) == result.completion.corrected_segment_count
        assert cache["meta"].attrs["continuous_completion_identity"] == (
            result.completion.completion_identity
        )
        assert int(cache["meta"].attrs["continuous_target_sample_count"]) == (
            grid.target_sample_count
        )
    finally:
        cache.close()


def test_second_call_allocates_a_distinct_run_directory(accepted_case, real_config, tmp_path):
    inputs = _pass_inputs(accepted_case)
    first = _run(inputs, real_config, tmp_path)
    second = _run(_pass_inputs(accepted_case), real_config, tmp_path)
    assert first.run_dir != second.run_dir
    assert first.run_id != second.run_id


def test_one_segment_continuous_run_is_a_single_chunk_not_a_session_list(
    real_config, tmp_path, tmp_path_factory
):
    folder = tmp_path_factory.mktemp("cr1_d2_single") / "recording"
    case = _build_case(folder, continuous_window_sec=600.0)
    inputs = _pass_inputs(case)
    result = _run(inputs, real_config, tmp_path)

    assert result.terminal_state == TERMINAL_SUCCESS_CURRENT
    assert result.completion.corrected_segment_count == 1
    cache = open_phasic_cache(result.corrected_cache_path)
    try:
        assert list_cache_chunk_ids(cache) == [0]
    finally:
        cache.close()

    classification = classify_run_terminal_state(result.run_dir)
    assert classification.is_success
    # A single storage chunk must not be represented as an intermittent
    # multi-session run: no chunked-input-processing ledger is required or
    # written for this run mode.
    assert classification.run_mode.get("chunked_input_processing") is False


# ---------------------------------------------------------------------------
# Mid-correction failure
# ---------------------------------------------------------------------------


def test_mid_correction_failure_leaves_no_successful_run(
    accepted_case, real_config, tmp_path, monkeypatch
):
    inputs = _pass_inputs(accepted_case)
    real_correct = c4c.correct_guided_continuous_rwd_segment
    calls = {"count": 0}

    def flaky_correct(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated mid-correction failure")
        return real_correct(*args, **kwargs)

    monkeypatch.setattr(c4c, "correct_guided_continuous_rwd_segment", flaky_correct)

    with pytest.raises(Exception):
        _run(inputs, real_config, tmp_path)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success

    import json

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "error"
    assert status["phase"] == "final"
    assert not (run_dir / "MANIFEST.json").exists()


def test_cancellation_leaves_no_successful_run(accepted_case, real_config, tmp_path, monkeypatch):
    written = {"count": 0}
    real_add_chunk = Hdf5TraceCacheWriter.add_chunk

    def counting_add_chunk(self, chunk, chunk_id, source_file):
        result = real_add_chunk(self, chunk, chunk_id, source_file)
        written["count"] += 1
        return result

    monkeypatch.setattr(Hdf5TraceCacheWriter, "add_chunk", counting_add_chunk)

    def cancel_after_first_segment():
        return written["count"] >= 1

    inputs = _pass_inputs(accepted_case)
    with pytest.raises(Exception):
        _run(inputs, real_config, tmp_path, cancellation_requested=cancel_after_first_segment)

    assert written["count"] >= 1
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    import json

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "cancelled"
    assert status["phase"] == "final"
    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success


# ---------------------------------------------------------------------------
# Failure after corrected-cache finalization
# ---------------------------------------------------------------------------


def test_failure_after_cache_finalization_leaves_no_successful_run(
    accepted_case, real_config, tmp_path, monkeypatch
):
    inputs = _pass_inputs(accepted_case)

    def flaky_manifest_block(*args, **kwargs):
        raise RuntimeError("simulated manifest-build failure after cache finalize")

    monkeypatch.setattr(subject, "build_manifest_completion_block", flaky_manifest_block)

    with pytest.raises(Exception):
        _run(inputs, real_config, tmp_path)

    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    import os

    # The cache finalized successfully before the simulated failure and is
    # not required to be deleted -- only the run must not appear completed.
    assert os.path.isfile(os.path.join(str(run_dir), subject.CORRECTED_CACHE_RELATIVE_PATH))
    assert os.path.isfile(os.path.join(str(run_dir), "run_report.json"))
    assert not os.path.exists(os.path.join(str(run_dir), "MANIFEST.json"))

    classification = classify_run_terminal_state(str(run_dir))
    assert not classification.is_success

    import json

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "error"
    assert status["phase"] == "final"


# ---------------------------------------------------------------------------
# Cache coherence failure (defense in depth against a mismatched artifact)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def persisted_case(accepted_case, real_config, tmp_path_factory):
    from photometry_pipeline import guided_continuous_rwd_correction_pass_persistence as d1

    from tests.test_guided_continuous_rwd_correction_pass_persistence import _traversal

    inputs = _pass_inputs(accepted_case)
    binding, grid = inputs[0], inputs[1]
    traversal = _traversal(inputs)
    output_path = str(tmp_path_factory.mktemp("cr1_d2_coherence") / "cache.h5")
    completion = d1.persist_guided_continuous_rwd_correction_pass(
        traversal,
        review_binding=binding,
        target_grid=grid,
        output_path=output_path,
        config=real_config,
    )
    return binding, grid, completion, output_path


def test_validate_persisted_cache_accepts_matching_authorities(persisted_case):
    binding, grid, completion, output_path = persisted_case
    subject._validate_persisted_cache(
        output_path, review_binding=binding, target_grid=grid, completion=completion
    )


def test_validate_persisted_cache_rejects_completion_identity_mismatch(persisted_case):
    binding, grid, completion, output_path = persisted_case
    tampered = replace(completion, completion_identity="0" * 64)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionRunError):
        subject._validate_persisted_cache(
            output_path, review_binding=binding, target_grid=grid, completion=tampered
        )


def test_validate_persisted_cache_rejects_target_grid_identity_mismatch(persisted_case):
    binding, grid, completion, output_path = persisted_case
    tampered = replace(grid, target_grid_identity="0" * 64)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionRunError):
        subject._validate_persisted_cache(
            output_path, review_binding=binding, target_grid=tampered, completion=completion
        )


def test_validate_persisted_cache_rejects_target_sample_count_mismatch(persisted_case):
    binding, grid, completion, output_path = persisted_case
    tampered = replace(grid, target_sample_count=grid.target_sample_count + 1)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionRunError):
        subject._validate_persisted_cache(
            output_path, review_binding=binding, target_grid=tampered, completion=completion
        )


def test_validate_persisted_cache_rejects_segment_count_mismatch(persisted_case):
    binding, grid, completion, output_path = persisted_case
    tampered = replace(
        completion, corrected_segment_count=completion.corrected_segment_count + 1
    )
    with pytest.raises(subject.GuidedContinuousRwdCorrectionRunError):
        subject._validate_persisted_cache(
            output_path, review_binding=binding, target_grid=grid, completion=tampered
        )


def test_validate_persisted_cache_rejects_roi_order_mismatch(persisted_case):
    binding, grid, completion, output_path = persisted_case
    reversed_roi = replace(
        binding.recording.roi,
        included_roi_ids=tuple(reversed(binding.recording.roi.included_roi_ids)),
    )
    tampered_recording = replace(binding.recording, roi=reversed_roi)
    tampered_binding = replace(binding, recording=tampered_recording)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionRunError):
        subject._validate_persisted_cache(
            output_path, review_binding=tampered_binding, target_grid=grid, completion=completion
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_output_base_must_be_absolute(accepted_case, real_config):
    inputs = _pass_inputs(accepted_case)
    with pytest.raises(subject.GuidedContinuousRwdCorrectionRunError):
        _run(inputs, real_config, "relative/output/base")
