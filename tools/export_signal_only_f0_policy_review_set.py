#!/usr/bin/env python3
"""Export stratified review rows for signal-only F0 policy proposals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photometry_pipeline.core.correction_policy_proposal import (  # noqa: E402
    SUPPORTED_CORRECTION_POLICIES,
    normalize_policy_flags,
)


MODE_SIGNAL_ONLY_F0 = "signal_only_f0_candidate"

BUCKETS = (
    "dynamic_hard_inspect",
    "dynamic_contextual_negative_mixed",
    "dynamic_contextual_low_flat_reference",
    "dynamic_contextual_baseline_negative_or_inverted",
    "signal_high_or_edge_state",
    "signal_partial_high_state",
    "high_extrapolation_or_large_gap",
    "other_signal_only_f0_candidate",
)

OUTPUT_COLUMNS = [
    "bucket",
    "policy",
    "roi",
    "chunk_id",
    "proposed_correction_mode_{policy}",
    "proposal_confidence_{policy}",
    "warning_level_{policy}",
    "review_required_{policy}",
    "review_queue_candidate_{policy}",
    "review_priority_{policy}",
    "proposal_reason_{policy}",
    "proposal_flags_{policy}",
    "dynamic_reference_candidate_viability",
    "baseline_reference_candidate_viability",
    "dynamic_fit_qc_flags",
    "reference_comparison_flags",
    "baseline_reference_relationship_class",
    "signal_state_candidate_class",
    "signal_state_flags",
    "signal_only_f0_candidate_viability",
    "signal_only_f0_candidate_confidence",
    "signal_only_f0_anchor_status",
    "signal_only_f0_anchor_count",
    "signal_only_f0_low_support_fraction",
    "signal_only_f0_interpolated_fraction",
    "signal_only_f0_extrapolated_fraction",
    "signal_only_f0_max_anchor_gap_fraction_observed",
    "signal_only_f0_flags",
]


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(x) for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return [dict(x) for x in data["records"] if isinstance(x, dict)]
    return []


def _flag_set(record: dict[str, Any], key: str) -> set[str]:
    return set(normalize_policy_flags(record.get(key)))


def _text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(x).strip() for x in value if str(x).strip())
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _float(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(record.get(key, default))
    except Exception:
        return float(default)
    return value if value == value else float(default)


def _chunk_sort_key(record: dict[str, Any]) -> tuple[str, int, str]:
    roi = _text(record, "roi")
    try:
        chunk_id = int(float(record.get("chunk_id", 0)))
    except Exception:
        chunk_id = 0
    return roi, chunk_id, _text(record, "source_file")


def _assign_bucket(record: dict[str, Any], policy: str) -> str:
    proposal_flags = _flag_set(record, f"proposal_flags_{policy}")
    dynamic_flags = _flag_set(record, "dynamic_fit_qc_flags")
    comparison_flags = _flag_set(record, "reference_comparison_flags")
    f0_flags = _flag_set(record, "signal_only_f0_flags")
    signal_flags = _flag_set(record, "signal_state_flags")
    all_dynamic_reference_flags = dynamic_flags | comparison_flags

    if "DYNAMIC_HARD_INSPECT" in proposal_flags or _text(
        record, "dynamic_reference_viability"
    ) == "hard_inspect":
        return "dynamic_hard_inspect"
    if all_dynamic_reference_flags & {
        "NEGATIVE_OR_MIXED_REFERENCE_COUPLING",
        "DYNAMIC_NEGATIVE_OR_MIXED_COUPLING",
    }:
        return "dynamic_contextual_negative_mixed"
    if all_dynamic_reference_flags & {
        "FITTED_REFERENCE_LOW_RANGE",
        "FITTED_REFERENCE_FLAT_OR_UNINFORMATIVE",
        "DYNAMIC_LOW_OR_FLAT_REFERENCE",
    }:
        return "dynamic_contextual_low_flat_reference"
    if (proposal_flags | comparison_flags) & {
        "BASELINE_NEGATIVE_REFERENCE_RELATIONSHIP",
        "INVERTED_REFERENCE_RELATIONSHIP",
    }:
        return "dynamic_contextual_baseline_negative_or_inverted"
    if (f0_flags | signal_flags) & {
        "SIGNAL_ONLY_F0_HIGH_STATE_PRESENT",
        "SIGNAL_ONLY_F0_EDGE_HIGH_STATE_PRESENT",
        "SIGNAL_HIGH_STATE_CANDIDATE",
        "SIGNAL_EDGE_HIGH_STATE_CANDIDATE",
    }:
        return "signal_high_or_edge_state"
    if (f0_flags | signal_flags) & {
        "SIGNAL_ONLY_F0_PARTIAL_HIGH_STATE_PRESENT",
        "SIGNAL_PARTIAL_HIGH_STATE_CANDIDATE",
    }:
        return "signal_partial_high_state"
    if (
        f0_flags
        & {
            "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION",
            "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_LARGE_GAP",
            "SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP",
        }
    ) or _float(record, "signal_only_f0_extrapolated_fraction") >= 0.50:
        return "high_extrapolation_or_large_gap"
    return "other_signal_only_f0_candidate"


def _serialize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(x).strip() for x in value if str(x).strip())
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _output_row(record: dict[str, Any], *, bucket: str, policy: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for template in OUTPUT_COLUMNS:
        key = template.format(policy=policy)
        if template == "bucket":
            value = bucket
        elif template == "policy":
            value = policy
        elif template == "dynamic_reference_candidate_viability":
            value = record.get("dynamic_reference_candidate_viability", record.get("dynamic_reference_viability"))
        elif template == "baseline_reference_candidate_viability":
            value = record.get("baseline_reference_candidate_viability", record.get("baseline_reference_viability"))
        elif template == "baseline_reference_relationship_class":
            value = record.get("baseline_reference_relationship_class", record.get("baseline_fit_relationship_class"))
        else:
            value = record.get(key)
        row[key] = _serialize_value(value)
    return row


def _sort_records(records: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    confidence_rank = {"high": 0, "medium": 1, "low": 2, "none": 3, "": 4}
    warning_rank = {"severe": 0, "caution": 1, "contextual": 2, "none": 3, "": 4}

    def key(record: dict[str, Any]) -> tuple[int, int, str, int, str]:
        roi, chunk_id, source = _chunk_sort_key(record)
        return (
            confidence_rank.get(_text(record, f"proposal_confidence_{policy}"), 9),
            warning_rank.get(_text(record, f"warning_level_{policy}"), 9),
            roi,
            chunk_id,
            source,
        )

    return sorted(records, key=key)


def _write_plot_commands(
    *,
    phasic_path: Path,
    rows: list[dict[str, str]],
    path: Path,
) -> None:
    by_roi: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        roi = row.get("roi", "").strip()
        if not roi:
            continue
        try:
            chunk_id = int(float(row.get("chunk_id", "")))
        except Exception:
            continue
        by_roi[roi].add(chunk_id)
    lines = []
    for roi in sorted(by_roi):
        chunks = ",".join(str(x) for x in sorted(by_roi[roi]))
        lines.append(
            'python tools/plot_signal_only_f0_candidates.py '
            f'--phasic-out "{phasic_path}" --roi {roi} --chunks {chunks}'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_signal_only_f0_policy_review_set(
    phasic_out: str | os.PathLike[str],
    *,
    policy: str = "balanced",
    per_bucket: int = 8,
    output: str | os.PathLike[str] | None = None,
    include_plot_commands: bool = False,
    roi: str | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    policy_norm = str(policy or "").strip().lower()
    if policy_norm not in SUPPORTED_CORRECTION_POLICIES:
        raise ValueError(
            f"Unsupported policy: {policy}. Allowed: {', '.join(SUPPORTED_CORRECTION_POLICIES)}"
        )
    phasic_path = Path(phasic_out).resolve()
    qc_dir = phasic_path / "qc"
    json_path = qc_dir / "baseline_reference_candidate_by_chunk.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing QC JSON: {json_path}")
    records = _load_json_records(json_path)
    if roi is not None:
        roi_filter = str(roi)
        records = [rec for rec in records if str(rec.get("roi", "")) == roi_filter]

    mode_key = f"proposed_correction_mode_{policy_norm}"
    candidates = [
        rec for rec in records if str(rec.get(mode_key, "")).strip() == MODE_SIGNAL_ONLY_F0
    ]
    bucket_counts = Counter(_assign_bucket(rec, policy_norm) for rec in candidates)
    selected: list[tuple[str, dict[str, Any]]] = []
    for bucket in BUCKETS:
        bucket_records = [rec for rec in candidates if _assign_bucket(rec, policy_norm) == bucket]
        for rec in _sort_records(bucket_records, policy_norm)[: max(0, int(per_bucket))]:
            selected.append((bucket, rec))
    if max_rows is not None:
        selected = selected[: max(0, int(max_rows))]

    rows = [
        _output_row(record, bucket=bucket, policy=policy_norm)
        for bucket, record in selected
    ]
    out_path = (
        Path(output).resolve()
        if output is not None
        else qc_dir / f"signal_only_f0_policy_review_set_{policy_norm}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [col.format(policy=policy_norm) for col in OUTPUT_COLUMNS]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_path = None
    if include_plot_commands:
        plot_path = qc_dir / f"signal_only_f0_policy_review_plot_commands_{policy_norm}.txt"
        _write_plot_commands(phasic_path=phasic_path, rows=rows, path=plot_path)

    report = {
        "phasic_out": str(phasic_path),
        "policy": policy_norm,
        "records_scanned": int(len(records)),
        "signal_only_f0_candidate_records_found": int(len(candidates)),
        "output_csv": str(out_path),
        "plot_command_path": str(plot_path) if plot_path is not None else None,
        "bucket_counts_before_sampling": {k: int(bucket_counts.get(k, 0)) for k in BUCKETS},
        "rows_written": int(len(rows)),
    }
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"phasic_out: {report['phasic_out']}")
    print(f"policy: {report['policy']}")
    print(f"records_scanned: {report['records_scanned']}")
    print(
        "signal_only_f0_candidate_records_found: "
        f"{report['signal_only_f0_candidate_records_found']}"
    )
    print(f"output_csv: {report['output_csv']}")
    if report.get("plot_command_path"):
        print(f"plot_command_path: {report['plot_command_path']}")
    print("bucket_counts_before_sampling:")
    for bucket, count in report["bucket_counts_before_sampling"].items():
        print(f"  {bucket}: {count}")
    print(f"rows_written: {report['rows_written']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a stratified review CSV for signal_only_f0_candidate proposals."
    )
    parser.add_argument("--phasic-out", required=True, help="Path to _analysis/phasic_out")
    parser.add_argument(
        "--policy",
        default="balanced",
        choices=SUPPORTED_CORRECTION_POLICIES,
        help="Policy suffix to inspect.",
    )
    parser.add_argument("--per-bucket", type=int, default=8, help="Rows to export per bucket")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    parser.add_argument(
        "--include-plot-commands",
        action="store_true",
        help="Also write grouped plot commands by ROI.",
    )
    parser.add_argument("--roi", default=None, help="Optional ROI filter")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional global row cap")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = export_signal_only_f0_policy_review_set(
            args.phasic_out,
            policy=args.policy,
            per_bucket=args.per_bucket,
            output=args.output,
            include_plot_commands=bool(args.include_plot_commands),
            roi=args.roi,
            max_rows=args.max_rows,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
