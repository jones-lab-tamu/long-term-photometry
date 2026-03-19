#!/usr/bin/env python3
"""CLI wrapper for cache-driven correction-sensitive retune backend."""

from __future__ import annotations

import argparse
import json
import sys

from photometry_pipeline.tuning.cache_correction_retune import (
    parse_key_value_overrides,
    run_cache_correction_retune,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run correction-sensitive cache retune for one ROI across all chunks."
    )
    parser.add_argument("--run-dir", required=True, help="Completed run directory")
    parser.add_argument("--roi", required=True, help="ROI name")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override key-value pair in KEY=VALUE form. Repeatable.",
    )
    parser.add_argument("--chunk-id", type=int, default=None, help="Optional inspection chunk id")
    parser.add_argument("--out-dir", default=None, help="Optional output base directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        overrides = parse_key_value_overrides(args.overrides)
        result = run_cache_correction_retune(
            run_dir=args.run_dir,
            roi=args.roi,
            overrides=overrides,
            chunk_id=args.chunk_id,
            out_dir=args.out_dir,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
