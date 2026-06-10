"""Generate the curated long-duration synthetic demo dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.synthetic_demo_generator import run_long_duration_demo  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "example_data" / "synthetic_long_duration_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the curated long-duration synthetic photometry demo dataset."
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output dataset folder. Default: example_data/synthetic_long_duration_demo/",
    )
    parser.add_argument(
        "--total-days",
        type=float,
        default=2.0,
        help="Test-only duration override. Normal demo default is 2 days.",
    )
    parser.add_argument(
        "--recording-duration-min",
        type=float,
        default=10.0,
        help="Test-only session duration override. Normal demo default is 10 minutes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the selected output folder if it already exists and is non-empty.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    result = run_long_duration_demo(
        out,
        overwrite=bool(args.overwrite),
        total_days=float(args.total_days),
        recording_duration_min=float(args.recording_duration_min),
    )
    print(result.message)
    print(f"input folder: {result.input_dir}")
    print(f"config path: {result.config_path}")
    print("recommended GUI settings:")
    print(f"  Input Directory: {result.input_dir}")
    print(f"  Config: {result.config_path}")
    print(f"  Format: {result.format}")
    print(f"  Sessions per hour: {result.sessions_per_hour}")
    print(f"  Mode: {result.mode}")
    if result.stdout_path is not None:
        print(f"stdout log: {result.stdout_path}")
    if result.stderr_path is not None:
        print(f"stderr log: {result.stderr_path}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
