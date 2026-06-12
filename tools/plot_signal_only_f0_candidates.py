#!/usr/bin/env python3
"""Plot read-only diagnostic signal-only F0 candidate traces for selected chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photometry_pipeline.core.signal_only_f0_candidate import (  # noqa: E402
    compute_signal_only_f0_candidate,
)
from photometry_pipeline.io.hdf5_cache_reader import (  # noqa: E402
    load_cache_chunk_fields,
    open_phasic_cache,
)
from tools.recompute_signal_only_f0_candidates import (  # noqa: E402
    _load_csv_records,
    _load_json_records,
    _load_signal_only_f0_config,
    _normalize_loaded_record,
)


def _parse_chunk_ids(chunks: str) -> list[int]:
    out: list[int] = []
    seen = set()
    for part in str(chunks or "").split(","):
        part = part.strip()
        if not part:
            continue
        cid = int(part)
        if cid not in seen:
            out.append(cid)
            seen.add(cid)
    if not out:
        raise ValueError("At least one chunk ID must be supplied with --chunks.")
    return out


def _find_phasic_cache(phasic_path: Path) -> Path:
    direct = phasic_path / "phasic_trace_cache.h5"
    if direct.exists():
        return direct
    matches = sorted(phasic_path.rglob("phasic_trace_cache.h5"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing phasic_trace_cache.h5 under {phasic_path}")


def _load_qc_records(phasic_path: Path) -> list[dict[str, Any]]:
    qc_dir = phasic_path / "qc"
    json_path = qc_dir / "baseline_reference_candidate_by_chunk.json"
    csv_path = qc_dir / "baseline_reference_candidate_by_chunk.csv"
    if json_path.exists():
        records = _load_json_records(json_path)
    elif csv_path.exists():
        records, _columns = _load_csv_records(csv_path)
    else:
        raise FileNotFoundError(
            "Missing baseline_reference_candidate_by_chunk.json or "
            f"baseline_reference_candidate_by_chunk.csv under {qc_dir}"
        )
    return [_normalize_loaded_record(rec) for rec in records]


def _records_by_key(records: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for rec in records:
        roi = str(rec.get("roi", "")).strip()
        if not roi:
            continue
        try:
            chunk_id = int(rec.get("chunk_id"))
        except Exception:
            continue
        out[(roi, chunk_id)] = rec
    return out


def _safe_roi_name(roi: str) -> str:
    return str(roi).replace(os.sep, "_").replace("/", "_").replace("\\", "_")


def _metadata_text(roi: str, chunk_id: int, record: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    flags = diagnostics.get("signal_only_f0_flags", [])
    if isinstance(flags, (list, tuple)):
        flags_text = ";".join(str(x) for x in flags)
    else:
        flags_text = str(flags or "")
    parts = [
        f"ROI: {roi}",
        f"chunk_id: {int(chunk_id)}",
        f"signal_state_candidate_class: {record.get('signal_state_candidate_class', '')}",
        f"signal_only_f0_candidate_viability: {diagnostics.get('signal_only_f0_candidate_viability', '')}",
        f"signal_only_f0_candidate_confidence: {diagnostics.get('signal_only_f0_candidate_confidence', '')}",
        f"signal_only_f0_flags: {flags_text}",
    ]
    return "\n".join(textwrap.wrap(" | ".join(parts), width=115))


def _plot_one(
    *,
    phasic_path: Path,
    roi: str,
    chunk_id: int,
    time_sec: np.ndarray,
    sig_raw: np.ndarray,
    record: dict[str, Any],
    config: dict[str, Any],
    dpi: int,
) -> Path:
    diagnostics = compute_signal_only_f0_candidate(
        signal=np.asarray(sig_raw, dtype=float),
        time=np.asarray(time_sec, dtype=float),
        signal_state=record,
        config=config,
    )
    candidate = np.asarray(diagnostics.get("signal_only_f0_candidate"), dtype=float)
    if candidate.shape != np.asarray(sig_raw).reshape(-1).shape:
        raise RuntimeError(f"Signal-only F0 candidate shape mismatch for {roi}/chunk_{chunk_id}")

    out_dir = phasic_path / "qc" / "signal_only_f0_candidate_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_roi_name(roi)}_chunk_{int(chunk_id)}_signal_only_f0_candidate.png"

    t = np.asarray(time_sec, dtype=float).reshape(-1)
    sig = np.asarray(sig_raw, dtype=float).reshape(-1)
    if t.size and np.isfinite(t[0]):
        t_plot = t - float(t[0])
    else:
        t_plot = np.arange(sig.size, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t_plot, sig, color="forestgreen", linewidth=0.9, label="sig_raw")
    ax.plot(
        t_plot,
        candidate,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="signal_only_f0_candidate",
    )
    ax.set_xlabel("Time within chunk (s)")
    ax.set_ylabel("Signal / F0 candidate")
    ax.set_title(_metadata_text(roi, chunk_id, record, diagnostics), fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    return out_path


def plot_signal_only_f0_candidates(
    phasic_out: str | os.PathLike[str],
    *,
    roi: str,
    chunks: list[int],
    dpi: int = 150,
) -> dict[str, Any]:
    phasic_path = Path(phasic_out).resolve()
    cache_path = _find_phasic_cache(phasic_path)
    records = _records_by_key(_load_qc_records(phasic_path))
    config, using_defaults, config_source = _load_signal_only_f0_config(phasic_path)

    output_paths: list[str] = []
    missing_records: list[str] = []
    with open_phasic_cache(str(cache_path)) as cache:
        for chunk_id in chunks:
            key = (str(roi), int(chunk_id))
            record = records.get(key)
            if record is None:
                missing_records.append(f"{roi}/chunk_{int(chunk_id)}")
                continue
            time_sec, sig_raw = load_cache_chunk_fields(
                cache, str(roi), int(chunk_id), ["time_sec", "sig_raw"]
            )
            out_path = _plot_one(
                phasic_path=phasic_path,
                roi=str(roi),
                chunk_id=int(chunk_id),
                time_sec=time_sec,
                sig_raw=sig_raw,
                record=record,
                config=config,
                dpi=int(dpi),
            )
            output_paths.append(str(out_path))
    if missing_records:
        raise RuntimeError(
            "Missing QC records for requested ROI/chunks: " + ", ".join(missing_records)
        )
    return {
        "phasic_out": str(phasic_path),
        "cache_path": str(cache_path),
        "roi": str(roi),
        "chunks": [int(x) for x in chunks],
        "plots_written": output_paths,
        "using_default_signal_only_f0_config": bool(using_defaults),
        "signal_only_f0_config_source": config_source,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot read-only diagnostic signal-only F0 candidates for selected chunks."
    )
    parser.add_argument("--phasic-out", required=True, help="Path to _analysis/phasic_out")
    parser.add_argument("--roi", required=True, help="ROI name, for example CH3")
    parser.add_argument("--chunks", required=True, help="Comma-separated chunk IDs, for example 28,29")
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = plot_signal_only_f0_candidates(
            args.phasic_out,
            roi=args.roi,
            chunks=_parse_chunk_ids(args.chunks),
            dpi=int(args.dpi),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"phasic_out: {report['phasic_out']}")
    print(f"cache_path: {report['cache_path']}")
    print(
        "using_default_signal_only_f0_config: "
        f"{str(report['using_default_signal_only_f0_config']).lower()}"
    )
    if report.get("signal_only_f0_config_source"):
        print(f"signal_only_f0_config_source: {report['signal_only_f0_config_source']}")
    for path in report["plots_written"]:
        print(f"wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
