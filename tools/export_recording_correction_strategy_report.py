#!/usr/bin/env python3
"""Export compact human-readable recording-level correction strategy reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


COMPACT_FIELDS = [
    "recording_key",
    "roi",
    "n_chunks",
    "source_file_count",
    "proposed_strategy",
    "confidence",
    "review_required",
    "reason",
    "key_flags",
    "fraction_dynamic_isosbestic",
    "fraction_dynamic_problem",
    "fraction_dynamic_hard_inspect",
    "fraction_signal_only_f0_candidate",
    "fraction_signal_only_f0_usable",
    "fraction_signal_only_f0_medium_or_high_confidence",
    "fraction_signal_only_f0_bad",
    "fraction_no_clean_reference_candidate",
    "fraction_review_required",
    "n_dynamic_hard_inspect",
    "n_dynamic_contextual",
    "n_signal_only_f0_low_confidence",
    "n_signal_only_f0_high_extrapolation_or_large_gap",
    "n_signal_only_f0_insufficient_anchors",
    "n_signal_only_f0_insufficient_low_support",
    "review_chunk_count",
    "caution_chunk_count",
    "review_chunk_ids_preview",
    "caution_chunk_ids_preview",
]

FRACTION_FIELDS = [
    "fraction_dynamic_isosbestic",
    "fraction_dynamic_problem",
    "fraction_dynamic_hard_inspect",
    "fraction_signal_only_f0_candidate",
    "fraction_signal_only_f0_usable",
    "fraction_signal_only_f0_medium_or_high_confidence",
    "fraction_signal_only_f0_bad",
    "fraction_no_clean_reference_candidate",
    "fraction_review_required",
]

COUNT_FIELDS = [
    "n_chunks",
    "source_file_count",
    "n_dynamic_hard_inspect",
    "n_dynamic_contextual",
    "n_signal_only_f0_low_confidence",
    "n_signal_only_f0_high_extrapolation_or_large_gap",
    "n_signal_only_f0_insufficient_anchors",
    "n_signal_only_f0_insufficient_low_support",
]


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def _bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    raw = str(value or "").strip().lower()
    return "true" if raw in {"true", "1", "yes", "y"} else "false"


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _int_value(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _split_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    if ";" in raw:
        return [part.strip() for part in raw.split(";") if part.strip()]
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [raw]


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(x) for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return [dict(x) for x in data["records"] if isinstance(x, dict)]
    return []


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _resolve_input_path(
    phasic_out: Path,
    input_csv: str | os.PathLike[str] | None,
    input_json: str | os.PathLike[str] | None,
) -> tuple[Path, str]:
    if input_csv and input_json:
        raise ValueError("Use only one of --input-csv or --input-json")
    if input_json:
        return Path(input_json).resolve(), "json"
    if input_csv:
        return Path(input_csv).resolve(), "csv"
    return (phasic_out / "qc" / "recording_correction_strategy_proposals.csv"), "csv"


def _load_records(path: Path, kind: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing recording strategy proposal input: {path}")
    if kind == "json" or path.suffix.lower() == ".json":
        return _load_json_records(path)
    return _load_csv_records(path)


def _preview_ids(value: Any, max_items: int) -> tuple[str, int]:
    ids = _split_list(value)
    count = len(ids)
    if count == 0:
        return "", 0
    limit = max(0, int(max_items))
    shown = ids[:limit] if limit else []
    preview = ";".join(shown)
    if count > limit:
        preview = f"{preview};..." if preview else "..."
    return preview, count


def _compact_row(record: dict[str, Any], max_warning_chunks: int) -> dict[str, str]:
    review_preview, review_count = _preview_ids(record.get("review_chunk_ids"), max_warning_chunks)
    caution_preview, caution_count = _preview_ids(record.get("caution_chunk_ids"), max_warning_chunks)
    out: dict[str, str] = {
        "recording_key": _text(record.get("recording_key")),
        "roi": _text(record.get("roi")),
        "proposed_strategy": _text(record.get("applied_correction_strategy_proposed")),
        "confidence": _text(record.get("auto_selection_confidence")),
        "review_required": _bool_text(record.get("auto_selection_review_required")),
        "reason": _text(record.get("auto_selection_reason")),
        "key_flags": _text(record.get("auto_selection_flags")),
        "review_chunk_count": str(review_count),
        "caution_chunk_count": str(caution_count),
        "review_chunk_ids_preview": review_preview,
        "caution_chunk_ids_preview": caution_preview,
    }
    for key in COUNT_FIELDS:
        out[key] = str(_int_value(record.get(key)))
    for key in FRACTION_FIELDS:
        out[key] = f"{_float_value(record.get(key)):.6f}"
    return {key: out.get(key, "") for key in COMPACT_FIELDS}


def _format_fraction(value: Any) -> str:
    return f"{_float_value(value):.3f}"


def _none_or_text(value: Any) -> str:
    text = _text(value)
    return text if text else "none"


def _warning_preview_text(value: Any, count: Any) -> str:
    text = _text(value)
    n = _int_value(count)
    if not text:
        return "none"
    if text.endswith(";...") or text == "...":
        return f"{text} (first shown; {n} total)"
    return text


def _markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| ROI | n chunks | proposed strategy | confidence | review? | reason | dynamic clean | dynamic problem | signal-only usable | signal-only bad |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _none_or_text(row.get("roi")),
                    _none_or_text(row.get("n_chunks")),
                    _none_or_text(row.get("proposed_strategy")),
                    _none_or_text(row.get("confidence")),
                    _none_or_text(row.get("review_required")),
                    _none_or_text(row.get("reason")),
                    _format_fraction(row.get("fraction_dynamic_isosbestic")),
                    _format_fraction(row.get("fraction_dynamic_problem")),
                    _format_fraction(row.get("fraction_signal_only_f0_usable")),
                    _format_fraction(row.get("fraction_signal_only_f0_bad")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _build_markdown(
    *,
    phasic_out: Path,
    input_path: Path,
    generated_utc: str,
    rows: list[dict[str, str]],
    strategy_counts: Counter[str],
    confidence_counts: Counter[str],
    review_counts: Counter[str],
) -> str:
    lines: list[str] = [
        "# Recording-Level Correction Strategy Report",
        "",
        "## Metadata",
        "",
        f"- phasic_out: `{phasic_out}`",
        f"- input file: `{input_path}`",
        f"- generated timestamp: `{generated_utc}`",
        f"- n_recordings: {len(rows)}",
        f"- strategy counts: {dict(sorted(strategy_counts.items()))}",
        f"- confidence counts: {dict(sorted(confidence_counts.items()))}",
        f"- review_required counts: {dict(sorted(review_counts.items()))}",
        "",
        "This report summarizes recording-level correction strategy proposals. Each row represents one ROI recording. The proposed strategy is global for that ROI recording. Chunk-level QC warnings identify regions requiring caution or review. The report does not indicate chunkwise switching of correction modes.",
        "",
        "`review_required = true` does not mean the selector refused to choose a strategy. It means the proposed global strategy carries warnings and should be inspected.",
        "",
        "## Summary",
        "",
        _markdown_table(rows),
        "",
    ]
    for row in rows:
        roi = _none_or_text(row.get("roi"))
        lines.extend(
            [
                f"## ROI {roi}",
                "",
                f"- Recording key: `{_none_or_text(row.get('recording_key'))}`",
                f"- Proposed global strategy: `{_none_or_text(row.get('proposed_strategy'))}`",
                f"- Confidence: `{_none_or_text(row.get('confidence'))}`",
                f"- Review required: `{_none_or_text(row.get('review_required'))}`",
                f"- Reason: `{_none_or_text(row.get('reason'))}`",
                f"- Key flags: `{_none_or_text(row.get('key_flags'))}`",
                "",
                "Metrics:",
                "",
                f"- dynamic_isosbestic fraction: {_format_fraction(row.get('fraction_dynamic_isosbestic'))}",
                f"- dynamic_problem fraction: {_format_fraction(row.get('fraction_dynamic_problem'))}",
                f"- dynamic_hard_inspect fraction: {_format_fraction(row.get('fraction_dynamic_hard_inspect'))}",
                f"- signal_only_f0_candidate fraction: {_format_fraction(row.get('fraction_signal_only_f0_candidate'))}",
                f"- signal_only_f0_usable fraction: {_format_fraction(row.get('fraction_signal_only_f0_usable'))}",
                f"- signal_only_f0_medium_or_high_confidence fraction: {_format_fraction(row.get('fraction_signal_only_f0_medium_or_high_confidence'))}",
                f"- signal_only_f0_bad fraction: {_format_fraction(row.get('fraction_signal_only_f0_bad'))}",
                f"- no_clean_reference_candidate fraction: {_format_fraction(row.get('fraction_no_clean_reference_candidate'))}",
                f"- review_required fraction: {_format_fraction(row.get('fraction_review_required'))}",
                "",
                "Warning preview:",
                "",
                f"- Review chunk IDs: {_warning_preview_text(row.get('review_chunk_ids_preview'), row.get('review_chunk_count'))}",
                f"- Caution chunk IDs: {_warning_preview_text(row.get('caution_chunk_ids_preview'), row.get('caution_chunk_count'))}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation guide",
            "",
            "- `dynamic_fit`, high confidence, review false: Dynamic isosbestic correction appears globally clean.",
            "- `signal_only_f0`, medium/low confidence, review true: Dynamic/reference correction appears broadly problematic, while signal-only F0 is broadly usable. Signal-only F0 is proposed as the best global strategy with warnings.",
            "- `no_correction`, low confidence, review true: No global correction strategy was selected confidently. Inspect before proceeding.",
            "",
        ]
    )
    return "\n".join(lines)


def export_recording_correction_strategy_report(
    phasic_out: str | os.PathLike[str],
    *,
    input_csv: str | os.PathLike[str] | None = None,
    input_json: str | os.PathLike[str] | None = None,
    output_md: str | os.PathLike[str] | None = None,
    output_csv: str | os.PathLike[str] | None = None,
    roi: str | None = None,
    dry_run: bool = False,
    max_warning_chunks: int = 20,
) -> dict[str, Any]:
    phasic_path = Path(phasic_out).resolve()
    input_path, input_kind = _resolve_input_path(phasic_path, input_csv, input_json)
    records = _load_records(input_path, input_kind)
    if roi is not None:
        records = [rec for rec in records if _text(rec.get("roi")) == str(roi)]

    compact_rows = [_compact_row(rec, max_warning_chunks) for rec in records]
    md_path = (
        Path(output_md).resolve()
        if output_md is not None
        else phasic_path / "qc" / "recording_correction_strategy_report.md"
    )
    compact_csv_path = (
        Path(output_csv).resolve()
        if output_csv is not None
        else phasic_path / "qc" / "recording_correction_strategy_compact.csv"
    )
    generated_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    strategy_counts = Counter(row["proposed_strategy"] for row in compact_rows)
    confidence_counts = Counter(row["confidence"] for row in compact_rows)
    review_counts = Counter(row["review_required"] for row in compact_rows)
    markdown = _build_markdown(
        phasic_out=phasic_path,
        input_path=input_path,
        generated_utc=generated_utc,
        rows=compact_rows,
        strategy_counts=strategy_counts,
        confidence_counts=confidence_counts,
        review_counts=review_counts,
    )

    if dry_run:
        print(markdown)
    else:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        compact_csv_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
        with compact_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COMPACT_FIELDS)
            writer.writeheader()
            writer.writerows(compact_rows)

    return {
        "phasic_out": str(phasic_path),
        "input_path": str(input_path),
        "output_md": str(md_path),
        "output_csv": str(compact_csv_path),
        "dry_run": bool(dry_run),
        "recordings_read": len(compact_rows),
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "review_required_counts": dict(sorted(review_counts.items())),
        "rows": compact_rows,
        "markdown": markdown,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"phasic_out: {report['phasic_out']}")
    print(f"input path: {report['input_path']}")
    print(f"output_md: {report['output_md']}")
    print(f"output_csv: {report['output_csv']}")
    print(f"dry_run: {str(report['dry_run']).lower()}")
    print(f"recordings_read: {report['recordings_read']}")
    print(f"strategy counts: {report['strategy_counts']}")
    print(f"confidence counts: {report['confidence_counts']}")
    print(f"review_required counts: {report['review_required_counts']}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export compact recording-level correction strategy Markdown and CSV reports."
    )
    parser.add_argument("--phasic-out", required=True, help="Path to _analysis/phasic_out")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--roi", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-warning-chunks", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = export_recording_correction_strategy_report(
            args.phasic_out,
            input_csv=args.input_csv,
            input_json=args.input_json,
            output_md=args.output_md,
            output_csv=args.output_csv,
            roi=args.roi,
            dry_run=bool(args.dry_run),
            max_warning_chunks=args.max_warning_chunks,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
