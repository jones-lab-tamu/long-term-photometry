#!/usr/bin/env python3
"""Propose one recording-level correction strategy per source-file/ROI pair."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photometry_pipeline.core.correction_policy_proposal import (  # noqa: E402
    SUPPORTED_CORRECTION_POLICIES,
    normalize_policy_flags,
)


DYNAMIC_STRONG_FRACTION = 0.80
DYNAMIC_MIN_FRACTION = 0.60
SIGNAL_ONLY_STRONG_FRACTION = 0.60
SIGNAL_ONLY_MIN_FRACTION = 0.40
REVIEW_REQUIRED_MAX_FRACTION_FOR_AUTO = 0.10
HARD_INSPECT_MAX_FRACTION_FOR_DYNAMIC = 0.10
SIGNAL_ONLY_BAD_MAX_FRACTION = 0.20
HIGH_RISK_WIDESPREAD_FRACTION = 0.25

MODE_DYNAMIC = "dynamic_isosbestic"
MODE_SIGNAL_ONLY = "signal_only_f0_candidate"
MODE_NO_CLEAN = "no_clean_reference_candidate"
MODE_REVIEW = "review_required"

OUTPUT_FIELDS = [
    "source_file",
    "roi",
    "policy",
    "n_chunks",
    "requested_correction_strategy",
    "applied_correction_strategy_proposed",
    "correction_strategy_selection",
    "auto_selection_confidence",
    "auto_selection_reason",
    "auto_selection_review_required",
    "auto_selection_flags",
    "n_dynamic_isosbestic",
    "n_signal_only_f0_candidate",
    "n_no_clean_reference_candidate",
    "n_review_required",
    "fraction_dynamic_isosbestic",
    "fraction_signal_only_f0_candidate",
    "fraction_no_clean_reference_candidate",
    "fraction_review_required",
    "n_dynamic_hard_inspect",
    "n_dynamic_contextual",
    "n_dynamic_viable",
    "n_signal_only_f0_viable",
    "n_signal_only_f0_contextual",
    "n_signal_only_f0_hard_inspect",
    "n_signal_only_f0_high_confidence",
    "n_signal_only_f0_medium_confidence",
    "n_signal_only_f0_low_confidence",
    "n_high_state_or_edge_state",
    "n_partial_high_state",
    "n_signal_only_f0_insufficient_anchors",
    "n_signal_only_f0_insufficient_low_support",
    "n_signal_only_f0_high_extrapolation_or_large_gap",
    "review_chunk_ids",
    "caution_chunk_ids",
    "signal_only_f0_candidate_chunk_ids",
    "dynamic_problem_chunk_ids",
]


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(x) for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return [dict(x) for x in data["records"] if isinstance(x, dict)]
    return []


def _text(record: dict[str, Any], key: str, default: str = "") -> str:
    value = record.get(key, default)
    if value is None:
        return str(default)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def _chunk_id(record: dict[str, Any]) -> int:
    try:
        return int(float(record.get("chunk_id", 0)))
    except Exception:
        return 0


def _flags(record: dict[str, Any], *keys: str) -> set[str]:
    out: set[str] = set()
    for key in keys:
        out.update(normalize_policy_flags(record.get(key)))
    return out


def _group_key(record: dict[str, Any]) -> tuple[str, str]:
    source = (
        _text(record, "source_file")
        or _text(record, "recording_id")
        or _text(record, "session_id")
        or "unknown_source"
    )
    roi = _text(record, "roi") or "unknown_roi"
    return source, roi


def _fraction(count: int, total: int) -> float:
    return float(count) / float(total) if total else 0.0


def _ids(records: list[dict[str, Any]]) -> list[int]:
    return sorted({_chunk_id(rec) for rec in records})


def _join_ids(ids: list[int]) -> str:
    return ";".join(str(x) for x in sorted(set(ids)))


def _is_dynamic_problem(record: dict[str, Any], policy: str) -> bool:
    dynamic = _text(record, "dynamic_reference_viability").lower()
    prop_flags = _flags(record, f"proposal_flags_{policy}")
    return dynamic in {"hard_inspect", "contextual"} or bool(
        prop_flags & {"DYNAMIC_HARD_INSPECT", "DYNAMIC_CONTEXTUAL"}
    )


def _is_dynamic_hard(record: dict[str, Any], policy: str) -> bool:
    return _text(record, "dynamic_reference_viability").lower() == "hard_inspect" or (
        "DYNAMIC_HARD_INSPECT" in _flags(record, f"proposal_flags_{policy}")
    )


def _is_dynamic_contextual(record: dict[str, Any], policy: str) -> bool:
    return _text(record, "dynamic_reference_viability").lower() == "contextual" or (
        "DYNAMIC_CONTEXTUAL" in _flags(record, f"proposal_flags_{policy}")
    )


def _signal_only_bad(record: dict[str, Any]) -> bool:
    viability = _text(record, "signal_only_f0_candidate_viability").lower()
    flags = _flags(record, "signal_only_f0_flags")
    return viability in {"hard_inspect", "unavailable"} or bool(
        flags
        & {
            "SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS",
            "SIGNAL_ONLY_F0_INSUFFICIENT_LOW_SUPPORT",
            "SIGNAL_ONLY_F0_ABOVE_SIGNAL_EXCESSIVE",
        }
    )


def _signal_only_high_or_medium(record: dict[str, Any]) -> bool:
    return _text(record, "signal_only_f0_candidate_confidence").lower() in {"high", "medium"}


def _high_state_or_edge(record: dict[str, Any]) -> bool:
    return bool(
        _flags(record, "signal_only_f0_flags", "signal_state_flags")
        & {
            "SIGNAL_ONLY_F0_HIGH_STATE_PRESENT",
            "SIGNAL_ONLY_F0_EDGE_HIGH_STATE_PRESENT",
            "SIGNAL_HIGH_STATE_CANDIDATE",
            "SIGNAL_EDGE_HIGH_STATE_CANDIDATE",
        }
    )


def _partial_high(record: dict[str, Any]) -> bool:
    return bool(
        _flags(record, "signal_only_f0_flags", "signal_state_flags")
        & {
            "SIGNAL_ONLY_F0_PARTIAL_HIGH_STATE_PRESENT",
            "SIGNAL_PARTIAL_HIGH_STATE_CANDIDATE",
        }
    )


def _large_gap_or_high_extrapolation(record: dict[str, Any]) -> bool:
    flags = _flags(record, "signal_only_f0_flags")
    try:
        extrap = float(record.get("signal_only_f0_extrapolated_fraction", 0.0))
    except Exception:
        extrap = 0.0
    return extrap >= 0.50 or bool(
        flags
        & {
            "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_EXTRAPOLATION",
            "SIGNAL_ONLY_F0_CONFIDENCE_CAPPED_LARGE_GAP",
            "SIGNAL_ONLY_F0_LARGE_ANCHOR_GAP",
        }
    )


def _review_required(record: dict[str, Any], policy: str) -> bool:
    return _boolish(record.get(f"review_required_{policy}"))


def _warning_is_caution(record: dict[str, Any], policy: str) -> bool:
    return _text(record, f"warning_level_{policy}").lower() in {"contextual", "caution", "severe"}


def _summarize_group(source: str, roi: str, records: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    n = len(records)
    mode_key = f"proposed_correction_mode_{policy}"
    modes = Counter(_text(rec, mode_key) for rec in records)
    n_dynamic = modes.get(MODE_DYNAMIC, 0)
    n_signal = modes.get(MODE_SIGNAL_ONLY, 0)
    n_no_clean = modes.get(MODE_NO_CLEAN, 0)
    n_review = modes.get(MODE_REVIEW, 0)

    dynamic_hard = [rec for rec in records if _is_dynamic_hard(rec, policy)]
    dynamic_contextual = [rec for rec in records if _is_dynamic_contextual(rec, policy)]
    dynamic_problem = [rec for rec in records if _is_dynamic_problem(rec, policy)]
    signal_bad = [rec for rec in records if _signal_only_bad(rec)]
    signal_candidates = [rec for rec in records if _text(rec, mode_key) == MODE_SIGNAL_ONLY]
    signal_supported = [rec for rec in signal_candidates if _signal_only_high_or_medium(rec)]
    review_records = [rec for rec in records if _review_required(rec, policy)]
    caution_records = [rec for rec in records if _warning_is_caution(rec, policy)]
    high_state = [rec for rec in records if _high_state_or_edge(rec)]
    partial_high = [rec for rec in records if _partial_high(rec)]
    gap_or_extrap = [rec for rec in records if _large_gap_or_high_extrapolation(rec)]

    dynamic_fraction = _fraction(n_dynamic, n)
    signal_fraction = _fraction(n_signal, n)
    review_fraction = _fraction(n_review, n)
    dynamic_hard_fraction = _fraction(len(dynamic_hard), n)
    dynamic_problem_fraction = _fraction(len(dynamic_problem), n)
    signal_bad_fraction = _fraction(len(signal_bad), n)
    signal_supported_fraction = _fraction(len(signal_supported), max(1, n_signal))
    high_risk_fraction = _fraction(len(review_records) + len(signal_bad) + len(gap_or_extrap), n)

    flags: list[str] = []
    if (
        dynamic_fraction >= DYNAMIC_STRONG_FRACTION
        and review_fraction <= REVIEW_REQUIRED_MAX_FRACTION_FOR_AUTO
        and dynamic_hard_fraction <= HARD_INSPECT_MAX_FRACTION_FOR_DYNAMIC
    ):
        strategy = "dynamic_fit"
        confidence = "high"
        reason = "dynamic_fit_dominant_clean"
        flags.append("RECORDING_DYNAMIC_FIT_DOMINANT")
    elif (
        dynamic_fraction >= DYNAMIC_MIN_FRACTION
        and signal_fraction < SIGNAL_ONLY_STRONG_FRACTION
        and review_fraction <= REVIEW_REQUIRED_MAX_FRACTION_FOR_AUTO
    ):
        strategy = "dynamic_fit"
        confidence = "medium"
        reason = "dynamic_fit_majority_with_warnings"
        flags.append("RECORDING_DYNAMIC_FIT_MAJORITY_WITH_WARNINGS")
    elif (
        signal_fraction >= SIGNAL_ONLY_STRONG_FRACTION
        and dynamic_fraction < DYNAMIC_MIN_FRACTION
        and signal_bad_fraction <= SIGNAL_ONLY_BAD_MAX_FRACTION
        and signal_supported_fraction >= 0.50
    ):
        strategy = "signal_only_f0"
        confidence = "high" if signal_supported_fraction >= 0.80 else "medium"
        reason = "signal_only_f0_supported_reference_problem_widespread"
        flags.append("RECORDING_SIGNAL_ONLY_F0_CANDIDATE_DOMINANT")
    elif (
        signal_fraction >= SIGNAL_ONLY_MIN_FRACTION
        and dynamic_problem_fraction >= DYNAMIC_MIN_FRACTION
        and signal_bad_fraction <= SIGNAL_ONLY_BAD_MAX_FRACTION
        and review_fraction <= REVIEW_REQUIRED_MAX_FRACTION_FOR_AUTO
    ):
        strategy = "signal_only_f0"
        confidence = "medium"
        reason = "signal_only_f0_candidate_supported_dynamic_problem"
        flags.append("RECORDING_SIGNAL_ONLY_F0_SUPPORTED_REFERENCE_PROBLEM")
    else:
        strategy = "no_correction"
        confidence = "low"
        reason = "no_single_strategy_defensible"
        flags.append("RECORDING_NO_SINGLE_STRATEGY_DEFENSIBLE")

    if dynamic_problem_fraction >= DYNAMIC_MIN_FRACTION:
        flags.append("RECORDING_HIGH_DYNAMIC_PROBLEM_FRACTION")
    if signal_bad_fraction > SIGNAL_ONLY_BAD_MAX_FRACTION:
        flags.append("RECORDING_HIGH_SIGNAL_ONLY_BAD_FRACTION")
    if high_state:
        flags.append("RECORDING_HAS_HIGH_STATE_OR_EDGE_STATE_CHUNKS")
    if gap_or_extrap:
        flags.append("RECORDING_HAS_LARGE_GAP_OR_HIGH_EXTRAPOLATION_CHUNKS")

    auto_review = bool(
        review_fraction > REVIEW_REQUIRED_MAX_FRACTION_FOR_AUTO
        or confidence == "low"
        or strategy == "no_correction"
        or signal_bad_fraction > SIGNAL_ONLY_BAD_MAX_FRACTION
        or high_risk_fraction > HIGH_RISK_WIDESPREAD_FRACTION
    )
    if auto_review:
        flags.append("RECORDING_REVIEW_REQUIRED")

    return {
        "source_file": source,
        "roi": roi,
        "policy": policy,
        "n_chunks": n,
        "requested_correction_strategy": "auto",
        "applied_correction_strategy_proposed": strategy,
        "correction_strategy_selection": "auto",
        "auto_selection_confidence": confidence,
        "auto_selection_reason": reason,
        "auto_selection_review_required": auto_review,
        "auto_selection_flags": list(dict.fromkeys(flags)),
        "n_dynamic_isosbestic": n_dynamic,
        "n_signal_only_f0_candidate": n_signal,
        "n_no_clean_reference_candidate": n_no_clean,
        "n_review_required": n_review,
        "fraction_dynamic_isosbestic": dynamic_fraction,
        "fraction_signal_only_f0_candidate": signal_fraction,
        "fraction_no_clean_reference_candidate": _fraction(n_no_clean, n),
        "fraction_review_required": review_fraction,
        "n_dynamic_hard_inspect": len(dynamic_hard),
        "n_dynamic_contextual": len(dynamic_contextual),
        "n_dynamic_viable": sum(1 for rec in records if _text(rec, "dynamic_reference_viability") == "viable"),
        "n_signal_only_f0_viable": sum(1 for rec in records if _text(rec, "signal_only_f0_candidate_viability") == "viable"),
        "n_signal_only_f0_contextual": sum(1 for rec in records if _text(rec, "signal_only_f0_candidate_viability") == "contextual"),
        "n_signal_only_f0_hard_inspect": sum(1 for rec in records if _text(rec, "signal_only_f0_candidate_viability") == "hard_inspect"),
        "n_signal_only_f0_high_confidence": sum(1 for rec in records if _text(rec, "signal_only_f0_candidate_confidence") == "high"),
        "n_signal_only_f0_medium_confidence": sum(1 for rec in records if _text(rec, "signal_only_f0_candidate_confidence") == "medium"),
        "n_signal_only_f0_low_confidence": sum(1 for rec in records if _text(rec, "signal_only_f0_candidate_confidence") == "low"),
        "n_high_state_or_edge_state": len(high_state),
        "n_partial_high_state": len(partial_high),
        "n_signal_only_f0_insufficient_anchors": sum(1 for rec in records if "SIGNAL_ONLY_F0_INSUFFICIENT_ANCHORS" in _flags(rec, "signal_only_f0_flags")),
        "n_signal_only_f0_insufficient_low_support": sum(1 for rec in records if "SIGNAL_ONLY_F0_INSUFFICIENT_LOW_SUPPORT" in _flags(rec, "signal_only_f0_flags")),
        "n_signal_only_f0_high_extrapolation_or_large_gap": len(gap_or_extrap),
        "review_chunk_ids": _ids(review_records if strategy != "dynamic_fit" else review_records + dynamic_hard),
        "caution_chunk_ids": _ids(caution_records if strategy != "signal_only_f0" else caution_records + gap_or_extrap + high_state),
        "signal_only_f0_candidate_chunk_ids": _ids(signal_candidates),
        "dynamic_problem_chunk_ids": _ids(dynamic_problem),
    }


def _serialize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(x) for x in sorted(value) if str(x))
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{key: _serialize_value(row.get(key)) for key in OUTPUT_FIELDS} for row in rows]


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{stamp}")
    shutil.copy2(path, backup)
    return backup


def propose_recording_correction_strategy(
    phasic_out: str | os.PathLike[str],
    *,
    policy: str = "balanced",
    output_csv: str | os.PathLike[str] | None = None,
    output_json: str | os.PathLike[str] | None = None,
    roi: str | None = None,
    source_file: str | None = None,
    dry_run: bool = False,
    backup: bool = False,
) -> dict[str, Any]:
    policy_norm = str(policy or "").strip().lower()
    if policy_norm not in SUPPORTED_CORRECTION_POLICIES:
        raise ValueError(f"Unsupported policy: {policy}")
    phasic_path = Path(phasic_out).resolve()
    qc_dir = phasic_path / "qc"
    records_path = qc_dir / "baseline_reference_candidate_by_chunk.json"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing QC JSON: {records_path}")

    records = _load_json_records(records_path)
    if roi is not None:
        records = [rec for rec in records if _text(rec, "roi") == str(roi)]
    if source_file is not None:
        records = [rec for rec in records if (_group_key(rec)[0] == str(source_file))]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[_group_key(rec)].append(rec)

    rows = [
        _summarize_group(source, group_roi, sorted(group_records, key=_chunk_id), policy_norm)
        for (source, group_roi), group_records in sorted(groups.items())
    ]
    csv_path = (
        Path(output_csv).resolve()
        if output_csv is not None
        else qc_dir / "recording_correction_strategy_proposals.csv"
    )
    json_path = (
        Path(output_json).resolve()
        if output_json is not None
        else qc_dir / "recording_correction_strategy_proposals.json"
    )

    backups = []
    if not dry_run:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if backup:
            for path in (csv_path, json_path):
                b = _backup(path)
                if b is not None:
                    backups.append(str(b))
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(_csv_rows(rows))
        json_path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    strategy_counts = Counter(row["applied_correction_strategy_proposed"] for row in rows)
    confidence_counts = Counter(row["auto_selection_confidence"] for row in rows)
    review_counts = Counter(str(bool(row["auto_selection_review_required"])).lower() for row in rows)
    return {
        "phasic_out": str(phasic_path),
        "policy": policy_norm,
        "records_scanned": len(records),
        "recordings_found": len(rows),
        "output_csv": str(csv_path),
        "output_json": str(json_path),
        "dry_run": bool(dry_run),
        "backups_created": backups,
        "rows": rows,
        "applied_correction_strategy_proposed_counts": dict(sorted(strategy_counts.items())),
        "auto_selection_confidence_counts": dict(sorted(confidence_counts.items())),
        "auto_selection_review_required_counts": dict(sorted(review_counts.items())),
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"phasic_out: {report['phasic_out']}")
    print(f"policy: {report['policy']}")
    print(f"records_scanned: {report['records_scanned']}")
    print(f"recordings_found: {report['recordings_found']}")
    print(f"output_csv: {report['output_csv']}")
    print(f"output_json: {report['output_json']}")
    print(f"dry_run: {str(report['dry_run']).lower()}")
    print(
        "applied_correction_strategy_proposed counts: "
        f"{report['applied_correction_strategy_proposed_counts']}"
    )
    print(f"auto_selection_confidence counts: {report['auto_selection_confidence_counts']}")
    print(
        "auto_selection_review_required counts: "
        f"{report['auto_selection_review_required_counts']}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Propose one recording-level correction strategy per source-file/ROI."
    )
    parser.add_argument("--phasic-out", required=True, help="Path to _analysis/phasic_out")
    parser.add_argument("--policy", default="balanced", choices=SUPPORTED_CORRECTION_POLICIES)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--roi", default=None)
    parser.add_argument("--source-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true", help="Back up existing output files before writing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = propose_recording_correction_strategy(
            args.phasic_out,
            policy=args.policy,
            output_csv=args.output_csv,
            output_json=args.output_json,
            roi=args.roi,
            source_file=args.source_file,
            dry_run=bool(args.dry_run),
            backup=bool(args.backup),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
