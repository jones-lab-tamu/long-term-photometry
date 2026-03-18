#!/usr/bin/env python3
"""
Cache-driven downstream retune CLI.

Runs bounded post-run retuning from an already completed run's phasic cache.
Only downstream-retunable overrides are accepted.
"""

import argparse
import json
import os
import sys

# Repo root bootstrap
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from photometry_pipeline.tuning.cache_downstream_retune import (
    parse_key_value_overrides,
    run_cache_downstream_retune,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retune downstream phasic feature settings from cache for one ROI, "
            "without rerunning full production deliverables."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Completed run directory.")
    parser.add_argument("--roi", required=True, help="Selected ROI (e.g., Region0).")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Override in KEY=VALUE format. Repeat as needed.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional retune output base directory. Default: <run_dir>/tuning_retune",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help=(
            "Optional chunk/session id for trace-level inspection diagnostics. "
            "Default: first available chunk in cache."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        overrides = parse_key_value_overrides(args.set_overrides)
        result = run_cache_downstream_retune(
            run_dir=os.path.abspath(args.run_dir),
            roi=args.roi,
            overrides=overrides,
            chunk_id=args.chunk_id,
            out_dir=args.out_dir,
        )
        print("CACHE-RETUNE: OK")
        print(f"CACHE-RETUNE: retune_dir={result['retune_dir']}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"CACHE-RETUNE: ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
