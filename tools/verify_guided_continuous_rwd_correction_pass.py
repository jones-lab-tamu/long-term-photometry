"""CR1-C4d: real-source verification of the accepted two-pass continuous
correction path.

This is a verification script, not a production entry point. It calls only
already-accepted public backend APIs directly (C3b bounded projection, C4a
correction-segment planning and recording-global dynamic-F0 preparation, C4c
corrected-segment traversal and completion authority). It performs no
persistence, output integration, GUI/worker wiring, or downstream
tonic/phasic analysis, and it does not enable Guided Run. Continuous Guided
remains hidden and non-runnable; nothing here changes that.

Usage::

    python tools/verify_guided_continuous_rwd_correction_pass.py \\
        --source-folder "<path to a folder containing Fluorescence.csv>"

or, with the source folder supplied via environment variable::

    set PHOTOMETRY_REAL_RWD_ROOT=<path>
    python tools/verify_guided_continuous_rwd_correction_pass.py

The script never hard-codes a private dataset path. The optional
``--report-path`` writes a JSON report to a location of the caller's
choosing; by default nothing is written to disk, and the console summary
never prints more than the source folder the caller supplied.

Verification-plan provenance
-----------------------------
Every included ROI requires an explicit correction-strategy choice; the
harness never infers, cycles, or otherwise chooses a strategy on the
caller's behalf. On the CLI this is supplied via repeatable
``--roi ROI_ID=STRATEGY`` arguments; ``run_verification``/
``build_shared_authorities`` accept the exact same requirement directly
through their ``roi_strategies`` argument for importable (non-CLI) callers,
such as this module's own test suite. Both are equally "explicit
verification input" -- the report never claims CLI origin for a choice that
was actually supplied through a direct function call. These choices are
recorded truthfully as explicit CR1-C4d verification input
(``VERIFICATION_EVIDENCE_SOURCE_TYPE``), never labeled as having come from a
real local correction preview, and the report's ``verification_plan`` field
(``strategy_source": "explicit_verification_input"``) states plainly that no
scientific endorsement is implied -- this harness verifies that accepted
correction implementations execute correctly on real channel values, not
that a given strategy is the best biological analysis choice for a given
ROI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

# Self-contained repo root bootstrap, matching the existing convention used by
# tools/run_full_pipeline_deliverables.py and tools/verification/*.py, so this
# script works when invoked directly (python tools/verify_....py) without
# requiring the caller to set PYTHONPATH first.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from photometry_pipeline import guided_continuous_rwd_block_plan as block_plan_module
from photometry_pipeline.guided_continuous_rwd_block_plan import (
    build_guided_continuous_rwd_block_plan,
)
from photometry_pipeline.guided_continuous_rwd_correction_pass import (
    iterate_guided_continuous_rwd_corrected_segments,
)
from photometry_pipeline.guided_continuous_rwd_correction_segments import (
    build_guided_continuous_rwd_correction_segment_plan,
    prepare_guided_continuous_rwd_dynamic_f0_authority,
)
from photometry_pipeline.guided_continuous_rwd_discontinuity_evaluation import (
    CONTINUITY_PASSED,
    evaluate_continuous_rwd_timestamp_continuity,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    build_guided_continuous_rwd_recording_description,
)
from photometry_pipeline.guided_continuous_rwd_review_binding import (
    build_guided_continuous_rwd_review_binding,
)
from photometry_pipeline.guided_continuous_rwd_target_grid import (
    build_guided_continuous_rwd_target_grid,
)
from photometry_pipeline.guided_execution_payloads import (
    build_guided_execution_startup_mapping_contract,
)
from photometry_pipeline.guided_new_analysis_plan import (
    FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES,
    GuidedNewAnalysisDraftPlan,
    GuidedPlanCorrectionChoice,
)
from photometry_pipeline.io.rwd_continuous_projection_reader import (
    iter_project_guided_continuous_rwd_blocks,
)
from photometry_pipeline.io.rwd_continuous_source import (
    inspect_continuous_rwd_acquisition_folder,
)


SOURCE_FOLDER_ENV_VAR = "PHOTOMETRY_REAL_RWD_ROOT"
DEFAULT_CONTINUOUS_WINDOW_SEC = 600.0
DEFAULT_PRODUCTION_BLOCK_SIZE = 100_000
DEFAULT_COMPARISON_BLOCK_SIZE = 20_000
# The accepted correction-strategy vocabulary this harness will pass through
# unmodified. C4d does not choose a strategy for any ROI -- it only accepts
# whatever the caller explicitly supplies and validates it against this set.
SUPPORTED_STRATEGIES = frozenset(FIRST_SUBSET_DYNAMIC_FIT_STRATEGIES) | {"signal_only_f0"}
# Truthfully identifies these choices as explicit CR1-C4d verification input,
# never as evidence from a real local correction preview (none occurred).
# The accepted production mapping (guided_new_analysis_plan.
# build_guided_per_roi_production_strategy_map /
# guided_production_mapping.guided_production_per_roi_strategy_to_correction_spec)
# does not restrict this field's value -- it only requires
# explicit_user_mark=True and current_or_stale="current" -- so no adapter or
# schema change is needed to represent this truthfully (see the module
# docstring's "Verification-plan provenance" note).
VERIFICATION_EVIDENCE_SOURCE_TYPE = "cr1_c4d_explicit_verification_choice"


class VerificationError(RuntimeError):
    """A narrow, reported real-source verification refusal or failure."""


class _MemoryProbe:
    """Python-allocation peak memory via tracemalloc, plus opportunistic RSS.

    tracemalloc (stdlib) is the guaranteed, dependency-free measurement: it
    reports peak *Python-tracked* allocation, not total process memory.
    ``psutil``, when already importable in the environment, additionally
    reports actual process resident-set-size (RSS) as a second, clearly
    labeled data point. Neither value is ever presented as the other.
    """

    def __init__(self) -> None:
        self._process = None
        try:
            import psutil  # optional, opportunistic only -- never required

            self._process = psutil.Process()
        except Exception:
            self._process = None
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def reset_phase(self) -> None:
        tracemalloc.reset_peak()

    def snapshot(self) -> dict[str, Any]:
        _, traced_peak_bytes = tracemalloc.get_traced_memory()
        rss_bytes = None
        if self._process is not None:
            try:
                rss_bytes = int(self._process.memory_info().rss)
            except Exception:
                rss_bytes = None
        return {
            "tracemalloc_peak_python_allocation_bytes": int(traced_peak_bytes),
            "process_rss_bytes": rss_bytes,
        }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-folder",
        default=None,
        help=(
            "Folder containing the real continuous RWD Fluorescence.csv. "
            f"Falls back to the {SOURCE_FOLDER_ENV_VAR} environment variable "
            "if omitted."
        ),
    )
    parser.add_argument(
        "--continuous-window-sec",
        type=float,
        default=DEFAULT_CONTINUOUS_WINDOW_SEC,
        help="Accepted continuous correction-segment duration in seconds.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_PRODUCTION_BLOCK_SIZE,
        help="C2 owned-sample block size for the production partition run.",
    )
    parser.add_argument(
        "--comparison-block-size",
        type=int,
        default=DEFAULT_COMPARISON_BLOCK_SIZE,
        help="C2 owned-sample block size for the second, comparison partition run.",
    )
    parser.add_argument(
        "--roi",
        action="append",
        default=None,
        metavar="ROI_ID=STRATEGY",
        help=(
            "Explicit per-ROI correction strategy choice, repeatable and "
            "required for every included ROI (e.g. "
            "--roi CH1=robust_global_event_reject). C4d does not infer or "
            "cycle strategies by channel order; every discovered ROI must "
            "have exactly one supplied strategy."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path to write the JSON verification report.",
    )
    return parser.parse_args(argv)


def resolve_source_folder(args: argparse.Namespace) -> str:
    candidate = args.source_folder or os.environ.get(SOURCE_FOLDER_ENV_VAR)
    if not candidate or not str(candidate).strip():
        raise VerificationError(
            "A real continuous RWD source folder is required: pass "
            f"--source-folder or set {SOURCE_FOLDER_ENV_VAR}."
        )
    path = Path(candidate)
    if not path.is_dir():
        raise VerificationError(f"Source folder does not exist or is not a directory: {path}")
    if not (path / "Fluorescence.csv").is_file():
        raise VerificationError(f"Source folder does not contain Fluorescence.csv: {path}")
    return str(path)


def _parse_roi_strategy_arguments(raw: list[str] | None) -> dict[str, str]:
    """Parse repeatable ``--roi ROI_ID=STRATEGY`` arguments into a mapping.

    Refuses (rather than silently keeping the last one) when the same ROI is
    supplied more than once with conflicting strategies.
    """
    strategies: dict[str, str] = {}
    for item in raw or ():
        if "=" not in item:
            raise VerificationError(f"Invalid --roi argument (expected ROI_ID=STRATEGY): {item!r}")
        roi_id, strategy = item.split("=", 1)
        roi_id = roi_id.strip()
        strategy = strategy.strip()
        if not roi_id or not strategy:
            raise VerificationError(f"Invalid --roi argument (expected ROI_ID=STRATEGY): {item!r}")
        if roi_id in strategies and strategies[roi_id] != strategy:
            raise VerificationError(
                f"Conflicting --roi strategies supplied for ROI {roi_id!r}: "
                f"{strategies[roi_id]!r} vs {strategy!r}."
            )
        strategies[roi_id] = strategy
    return strategies


def _require_explicit_roi_strategies(
    available_roi_ids: tuple[str, ...],
    roi_strategies: dict[str, str] | None,
) -> dict[str, str]:
    """Fail closed unless every included ROI has one explicit, supported strategy.

    C4d never infers a correction strategy from ROI/channel order -- every
    included ROI must be named explicitly by the caller.
    """
    roi_strategies = dict(roi_strategies or {})
    if not roi_strategies:
        raise VerificationError(
            "No --roi strategy choices were supplied. C4d requires an "
            "explicit verification strategy for every included ROI "
            "(e.g. --roi CH1=global_linear_regression) and does not infer "
            "scientific choices from channel order."
        )
    available = set(available_roi_ids)
    unknown = sorted(set(roi_strategies) - available)
    if unknown:
        raise VerificationError(
            f"--roi named unknown ROI IDs not present in the source: {unknown}. "
            f"Discovered ROI IDs are: {sorted(available)}."
        )
    missing = sorted(available - set(roi_strategies))
    if missing:
        raise VerificationError(
            "Missing an explicit --roi verification strategy for included "
            f"ROI(s): {missing}. C4d requires an explicit verification "
            "strategy for every included ROI and does not infer scientific "
            "choices from channel order."
        )
    unsupported = {
        roi_id: strategy
        for roi_id, strategy in roi_strategies.items()
        if strategy not in SUPPORTED_STRATEGIES
    }
    if unsupported:
        raise VerificationError(
            f"Unsupported --roi strategy choice(s): {unsupported}. Supported "
            f"strategies are: {sorted(SUPPORTED_STRATEGIES)}."
        )
    return roi_strategies


def _build_draft(
    *,
    source_folder: str,
    discovered_roi_ids: tuple[str, ...],
    included_roi_ids: tuple[str, ...],
    excluded_roi_ids: tuple[str, ...],
    roi_strategies: dict[str, str],
    continuous_window_sec: float,
) -> GuidedNewAnalysisDraftPlan:
    choices = [
        GuidedPlanCorrectionChoice(
            roi_id=roi_id,
            selected_strategy=roi_strategies[roi_id],
            # Truthful provenance: this choice was supplied as explicit CR1-C4d
            # verification input (whether via --roi on the CLI or a direct
            # run_verification/build_shared_authorities call), never derived
            # from a real local correction preview. See
            # VERIFICATION_EVIDENCE_SOURCE_TYPE.
            source_type=VERIFICATION_EVIDENCE_SOURCE_TYPE,
            current_or_stale="current",
            explicit_user_mark=True,
            evidence_reference={
                "evidence_source_type": VERIFICATION_EVIDENCE_SOURCE_TYPE,
                "strategy": roi_strategies[roi_id],
                "roi": roi_id,
                "verification_context": "cr1_c4d_real_source_verification",
            },
        )
        for roi_id in included_roi_ids
    ]
    return GuidedNewAnalysisDraftPlan(
        input_source_path=source_folder,
        resolved_input_source_path=source_folder,
        input_format="rwd",
        acquisition_mode="continuous",
        continuous_window_sec=continuous_window_sec,
        continuous_step_sec=continuous_window_sec,
        discovered_roi_ids=list(discovered_roi_ids),
        included_roi_ids=list(included_roi_ids),
        excluded_roi_ids=list(excluded_roi_ids),
        output_base_path=str(Path(source_folder) / "cr1_c4d_verification_output"),
        global_correction_strategy=next(iter(roi_strategies.values())),
        per_roi_correction_strategy_choices=choices,
        feature_event_profile_id="default",
        feature_event_values={},
    )


def _final_tail_classification(segment_plan: Any) -> str:
    last = segment_plan.descriptors[-1]
    if last.absorbed_short_tail:
        return "merged_nonviable_short_tail"
    if last.sample_count == segment_plan.nominal_segment_sample_count:
        return "exact_full_final_segment"
    return "viable_retained_short_tail"


def _descriptor_summary(descriptor: Any) -> dict[str, Any]:
    return {
        "segment_index": descriptor.segment_index,
        "start_target_index": descriptor.start_target_index,
        "stop_target_index": descriptor.stop_target_index,
        "sample_count": descriptor.sample_count,
        "is_final": descriptor.is_final,
        "absorbed_short_tail": descriptor.absorbed_short_tail,
    }


def build_shared_authorities(
    source_folder: str,
    *,
    continuous_window_sec: float,
    roi_strategies: dict[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any]:
    """Steps 1-5 of the required execution path: inspect through C1 grid.

    Returns ``(source_evidence, verification_plan, binding, draft, grid)``.
    ``roi_strategies`` must name an explicit, supported strategy for every
    ROI the recording includes; see ``_require_explicit_roi_strategies``.
    """
    inspection = inspect_continuous_rwd_acquisition_folder(source_folder)
    if inspection.status != "completed":
        raise VerificationError(
            f"Source inspection did not complete: status={inspection.status!r}, "
            f"summary={inspection.scientist_summary!r}"
        )

    available_roi_ids = tuple(
        sorted(pair.roi_id for pair in inspection.channels.roi_pairs)
    )
    if not available_roi_ids:
        raise VerificationError("Source inspection discovered no ROI channels.")

    recording = build_guided_continuous_rwd_recording_description(
        inspection, included_roi_ids=available_roi_ids
    )
    continuity = evaluate_continuous_rwd_timestamp_continuity(
        recording, source_path=Path(source_folder) / "Fluorescence.csv"
    )
    if continuity.outcome != CONTINUITY_PASSED:
        raise VerificationError(
            f"Continuity evaluation did not pass: outcome={continuity.outcome!r}, "
            f"reason={continuity.failure_reason!r}"
        )

    roi_strategies = _require_explicit_roi_strategies(
        tuple(item.roi_id for item in recording.roi.available_roi_channels),
        roi_strategies,
    )

    draft = _build_draft(
        source_folder=source_folder,
        discovered_roi_ids=tuple(item.roi_id for item in recording.roi.available_roi_channels),
        included_roi_ids=recording.roi.included_roi_ids,
        excluded_roi_ids=recording.roi.excluded_roi_ids,
        roi_strategies=roi_strategies,
        continuous_window_sec=continuous_window_sec,
    )
    binding = build_guided_continuous_rwd_review_binding(
        draft,
        recording=recording,
        continuity_evaluation=continuity,
        current_source_path=str(Path(source_folder) / "Fluorescence.csv"),
    )
    grid = build_guided_continuous_rwd_target_grid(recording, continuity)

    source_evidence = {
        "source_folder": source_folder,
        "fluorescence_file": recording.source.fluorescence_path_canonical,
        "source_file_size_bytes": recording.source.file_size_bytes,
        "source_row_count": recording.source.valid_timestamp_count,
        "recording_duration_seconds": recording.time.measured_support_end_seconds,
        "source_cadence_seconds": recording.cadence.nominal_cadence_seconds,
        "roi_ids": list(recording.roi.included_roi_ids),
        "per_roi_strategies": dict(roi_strategies),
    }
    # Explicit, truthful provenance: these strategies were supplied directly
    # to build_shared_authorities/run_verification's roi_strategies argument
    # -- whether that argument was itself populated by parsing --roi CLI
    # arguments or by a direct importable call (as the test suite does) is
    # not distinguished here, since both are equally "explicit verification
    # input" and neither implies a local correction preview occurred. They
    # are not preview-derived, not scientifically validated or optimized,
    # and not a recommendation -- only proof that the accepted implementation
    # executes on real values.
    verification_plan = {
        "strategy_source": "explicit_verification_input",
        "per_roi_strategies": dict(roi_strategies),
        "scientific_endorsement": False,
    }
    return source_evidence, verification_plan, binding, draft, grid


def run_one_partition(
    binding: Any,
    draft: Any,
    grid: Any,
    *,
    block_size: int,
    memory_probe: _MemoryProbe,
) -> dict[str, Any]:
    """Steps 6-13 of the required execution path for one C2 partition."""
    contract = build_guided_execution_startup_mapping_contract()
    original_block_size = block_plan_module.MAXIMUM_OWNED_SAMPLES_PER_BLOCK
    try:
        block_plan_module.MAXIMUM_OWNED_SAMPLES_PER_BLOCK = block_size
        block_plan = build_guided_continuous_rwd_block_plan(grid)
        segment_plan = build_guided_continuous_rwd_correction_segment_plan(
            binding, grid, accepted_draft=draft, startup_mapping_contract=contract
        )

        memory_before_pass1 = memory_probe.snapshot()
        memory_probe.reset_phase()
        pass1_start = time.perf_counter()
        dynamic_f0_authority = prepare_guided_continuous_rwd_dynamic_f0_authority(
            binding,
            grid,
            block_plan,
            segment_plan,
            iter_project_guided_continuous_rwd_blocks(binding, grid, block_plan),
            accepted_draft=draft,
            startup_mapping_contract=contract,
        )
        pass1_runtime = time.perf_counter() - pass1_start
        memory_after_pass1 = memory_probe.snapshot()

        memory_probe.reset_phase()
        pass2_start = time.perf_counter()
        traversal = iterate_guided_continuous_rwd_corrected_segments(
            binding,
            grid,
            block_plan,
            segment_plan,
            dynamic_f0_authority,
            accepted_draft=draft,
            startup_mapping_contract=contract,
        )

        segment_result_identities: list[str] = []
        first_segment_summary: dict[str, Any] | None = None
        last_segment_summary: dict[str, Any] | None = None
        expected_start = 0
        for corrected in traversal:
            # Only small, bounded per-segment evidence is retained -- never
            # the full corrected arrays -- so this loop stays bounded in
            # memory regardless of recording length or segment count.
            if corrected.start_target_index != expected_start:
                raise VerificationError(
                    "Corrected-segment coverage gap detected: expected start "
                    f"{expected_start}, got {corrected.start_target_index}."
                )
            for array in (
                corrected.target_elapsed_seconds,
                corrected.correction_reference_values,
                corrected.delta_f_values,
                corrected.dff_values,
            ):
                if not _all_finite(array):
                    raise VerificationError(
                        f"Corrected segment {corrected.segment_index} published a "
                        "non-finite value in a published array."
                    )
            summary = {
                "segment_index": corrected.segment_index,
                "start_target_index": corrected.start_target_index,
                "stop_target_index": corrected.stop_target_index,
                "included_roi_ids": list(corrected.included_roi_ids),
                "result_identity": corrected.result_identity,
            }
            if first_segment_summary is None:
                first_segment_summary = summary
            last_segment_summary = summary
            segment_result_identities.append(corrected.result_identity)
            expected_start = corrected.stop_target_index

        if traversal.state != "completed":
            raise VerificationError(
                f"Pass 2 traversal did not reach a completed state (state={traversal.state!r})."
            )
        completion = traversal.completion
        pass2_runtime = time.perf_counter() - pass2_start
        memory_after_pass2 = memory_probe.snapshot()

        full_coverage_result = (
            completion.corrected_segment_count == segment_plan.segment_count
            and completion.target_sample_count == grid.target_sample_count
            and expected_start == grid.target_sample_count
        )
        largest_segment_sample_count = max(d.sample_count for d in segment_plan.descriptors)

        return {
            "block_size": block_size,
            "block_count": block_plan.block_count,
            "target_sample_count": grid.target_sample_count,
            "segment_count": segment_plan.segment_count,
            "nominal_segment_sample_count": segment_plan.nominal_segment_sample_count,
            "largest_segment_sample_count": largest_segment_sample_count,
            "roi_count": len(binding.recording.roi.included_roi_ids),
            "final_tail_classification": _final_tail_classification(segment_plan),
            "first_segment": _descriptor_summary(segment_plan.descriptors[0]),
            "last_segment": _descriptor_summary(segment_plan.descriptors[-1]),
            "first_corrected_segment": first_segment_summary,
            "last_corrected_segment": last_segment_summary,
            "dynamic_f0_authority_identity": dynamic_f0_authority.authority_identity,
            "corrected_segment_count": completion.corrected_segment_count,
            "segment_result_identities": segment_result_identities,
            "ordered_segment_identity_digest": completion.ordered_segment_identity_digest,
            "completion_identity": completion.completion_identity,
            "completion_state": completion.completion_state,
            "full_coverage_result": full_coverage_result,
            "late_verification_result": traversal.state == "completed",
            "pass1_runtime_seconds": pass1_runtime,
            "pass2_runtime_seconds": pass2_runtime,
            "memory_before_pass1": memory_before_pass1,
            "memory_after_pass1": memory_after_pass1,
            "memory_after_pass2": memory_after_pass2,
            "overall_success": True,
        }
    finally:
        block_plan_module.MAXIMUM_OWNED_SAMPLES_PER_BLOCK = original_block_size


def _all_finite(array: Any) -> bool:
    import numpy as np

    return bool(np.all(np.isfinite(np.asarray(array))))


def compare_partition_reports(report_a: dict[str, Any], report_b: dict[str, Any]) -> dict[str, Any]:
    """Step 15 of the required execution path: compare two C2 partitions."""
    checks = {
        "target_sample_count": report_a["target_sample_count"] == report_b["target_sample_count"],
        "segment_count": report_a["segment_count"] == report_b["segment_count"],
        "corrected_segment_count": (
            report_a["corrected_segment_count"] == report_b["corrected_segment_count"]
        ),
        "dynamic_f0_authority_identity": (
            report_a["dynamic_f0_authority_identity"] == report_b["dynamic_f0_authority_identity"]
        ),
        "segment_result_identities": (
            report_a["segment_result_identities"] == report_b["segment_result_identities"]
        ),
        "ordered_segment_identity_digest": (
            report_a["ordered_segment_identity_digest"] == report_b["ordered_segment_identity_digest"]
        ),
        "completion_identity": report_a["completion_identity"] == report_b["completion_identity"],
        "full_coverage_result": bool(
            report_a["full_coverage_result"] and report_b["full_coverage_result"]
        ),
    }
    return {
        "production_block_size": report_a["block_size"],
        "comparison_block_size": report_b["block_size"],
        "production_block_count": report_a["block_count"],
        "comparison_block_count": report_b["block_count"],
        "checks": checks,
        "identical": all(checks.values()),
    }


def run_verification(
    source_folder: str,
    *,
    continuous_window_sec: float = DEFAULT_CONTINUOUS_WINDOW_SEC,
    production_block_size: int = DEFAULT_PRODUCTION_BLOCK_SIZE,
    comparison_block_size: int = DEFAULT_COMPARISON_BLOCK_SIZE,
    roi_strategies: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the full CR1-C4d verification.

    ``roi_strategies`` must supply an explicit correction strategy for every
    ROI included in the recording; see ``_require_explicit_roi_strategies``.
    There is no default assignment -- omitting a required ROI, or omitting
    ``roi_strategies`` entirely, fails closed.
    """
    probe = _MemoryProbe()
    total_start = time.perf_counter()
    source_evidence, verification_plan, binding, draft, grid = build_shared_authorities(
        source_folder,
        continuous_window_sec=continuous_window_sec,
        roi_strategies=roi_strategies,
    )
    production_report = run_one_partition(
        binding, draft, grid, block_size=production_block_size, memory_probe=probe
    )
    comparison_report = run_one_partition(
        binding, draft, grid, block_size=comparison_block_size, memory_probe=probe
    )
    total_runtime = time.perf_counter() - total_start
    comparison = compare_partition_reports(production_report, comparison_report)

    return {
        "source": source_evidence,
        "verification_plan": verification_plan,
        "continuous_window_sec": continuous_window_sec,
        "production_partition": production_report,
        "comparison_partition": comparison_report,
        "c2_partition_comparison": comparison,
        "total_runtime_seconds": total_runtime,
        "overall_success": (
            production_report["overall_success"]
            and comparison_report["overall_success"]
            and comparison["identical"]
        ),
    }


def _print_summary(report: dict[str, Any]) -> None:
    source = report["source"]
    prod = report["production_partition"]
    comp = report["comparison_partition"]
    cmp_result = report["c2_partition_comparison"]
    print("=== CR1-C4d real-source verification ===")
    print(f"source_folder: {source['source_folder']}")
    print(f"source_file_size_bytes: {source['source_file_size_bytes']}")
    print(f"source_row_count: {source['source_row_count']}")
    print(f"recording_duration_seconds: {source['recording_duration_seconds']}")
    print(f"source_cadence_seconds: {source['source_cadence_seconds']}")
    print(f"roi_ids: {source['roi_ids']}")
    plan = report["verification_plan"]
    print(
        f"verification_plan: strategy_source={plan['strategy_source']!r} "
        f"scientific_endorsement={plan['scientific_endorsement']} "
        f"per_roi_strategies={plan['per_roi_strategies']}"
    )
    print(f"target_sample_count: {prod['target_sample_count']}")
    print(
        f"production partition: block_size={prod['block_size']} "
        f"block_count={prod['block_count']} segment_count={prod['segment_count']} "
        f"final_tail={prod['final_tail_classification']}"
    )
    print(
        f"  pass1_runtime_seconds={prod['pass1_runtime_seconds']:.3f} "
        f"pass2_runtime_seconds={prod['pass2_runtime_seconds']:.3f}"
    )
    print(
        f"comparison partition: block_size={comp['block_size']} "
        f"block_count={comp['block_count']} segment_count={comp['segment_count']}"
    )
    print(
        f"  pass1_runtime_seconds={comp['pass1_runtime_seconds']:.3f} "
        f"pass2_runtime_seconds={comp['pass2_runtime_seconds']:.3f}"
    )
    print(f"total_runtime_seconds: {report['total_runtime_seconds']:.3f}")
    print(f"c2_partition_comparison.identical: {cmp_result['identical']}")
    print(f"completion_identity (production): {prod['completion_identity']}")
    print(f"overall_success: {report['overall_success']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_folder = resolve_source_folder(args)
        cli_roi_strategies = _parse_roi_strategy_arguments(args.roi)
        if not cli_roi_strategies:
            raise VerificationError(
                "No --roi strategy choices were supplied. C4d requires an "
                "explicit verification strategy for every included ROI "
                "(e.g. --roi CH1=global_linear_regression) and does not "
                "infer scientific choices from channel order."
            )
    except VerificationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    try:
        report = run_verification(
            source_folder,
            continuous_window_sec=args.continuous_window_sec,
            production_block_size=args.block_size,
            comparison_block_size=args.comparison_block_size,
            roi_strategies=cli_roi_strategies,
        )
    except VerificationError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"VERIFICATION FAILED (unexpected {type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    _print_summary(report)
    if args.report_path:
        Path(args.report_path).write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Report written to {args.report_path}")

    return 0 if report["overall_success"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
