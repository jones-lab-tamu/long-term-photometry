"""Developer-only: Guided demo variants with a chosen length and tonic level.

This is a private testing/figure-development utility. It is NOT part of Guided
Mode, is not exposed in the application, and is not a general biological
simulator. It only calls the existing shipped demo generators in
``gui.synthetic_demo_generator`` with developer overrides; it defines no signal
model of its own.

Modes, matching the two shipped Guided demos:

* ``intermittent``  session-based Guided CSV demo. ``--days`` is realized at
                    the shipped cadence of 2 sessions per hour (48 sessions per
                    day) and rounds to the nearest whole session.
* ``continuous``    one uninterrupted Guided recording at the shipped 8 Hz.
                    ``--days`` is realized as an exact sample count.

In both modes the circadian/Slow Signal rhythm is a continuous function of
elapsed time from a fixed origin, so it runs unbroken across day boundaries and
is never reset per day.

What ``--tonic`` varies, and nothing else:

* ``high``  the shipped tonic amplitudes (ROI1 4.0 AU, ROI2 4.5 AU in both
            demos), numerically identical to the shipped demo at the same
            duration.
* ``low``   the same rhythm scaled to 0.25 of that amplitude (ROI1 1.0 AU,
            ROI2 1.125 AU). Period (24 h), phase (07:00 peak), offset, cadence,
            phasic events, reference channels, noise, ROIs, seed, and file
            structure are unchanged; the fixed seed means every non-tonic
            sample is bit-identical between ``high`` and ``low``.

Examples:

    python examples/generate_guided_demo_variant.py \\
        --mode intermittent --days 12 --tonic high --output /tmp/int_high
    python examples/generate_guided_demo_variant.py \\
        --mode intermittent --days 12 --tonic low  --output /tmp/int_low
    python examples/generate_guided_demo_variant.py \\
        --mode continuous   --days 12 --tonic high --output /tmp/cont_high
    python examples/generate_guided_demo_variant.py \\
        --mode continuous   --days 12 --tonic low  --output /tmp/cont_low
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.synthetic_demo_generator import (  # noqa: E402
    GUIDED_CONTINUOUS_DEMO_FS_HZ,
    GUIDED_CONTINUOUS_DEMO_TONIC_AMPLITUDE_AU,
    GUIDED_DEMO_SESSIONS_PER_DAY,
    GUIDED_DEMO_TONIC_AMPLITUDE_AU,
    generate_guided_continuous_demo,
    generate_guided_csv_demo,
)

# HIGH reproduces each shipped demo's tonic exactly; LOW is a straight scaling
# of that same rhythm, low enough to look clearly different but still well
# within what Slow Signal analysis recovers. Both demos ship the same
# amplitudes, so one pair of scales covers both modes.
TONIC_SCALES = {"high": 1.0, "low": 0.25}
INTERMITTENT = "intermittent"
CONTINUOUS = "continuous"
SECONDS_PER_DAY = 86400.0


def _positive_days(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--days must be a number: {text!r}") from exc
    if not value > 0.0:
        raise argparse.ArgumentTypeError(f"--days must be positive: {text!r}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Developer-only: generate a Guided demo dataset of an arbitrary "
            "length with either the shipped (high) or a reduced (low) tonic "
            "rhythm. Not part of Guided Mode; not a biological simulator."
        ),
        epilog=(
            f"high tonic amplitude = {GUIDED_DEMO_TONIC_AMPLITUDE_AU[0]:g}/"
            f"{GUIDED_DEMO_TONIC_AMPLITUDE_AU[1]:g} AU (ROI1/ROI2) in both modes; "
            f"low = x{TONIC_SCALES['low']:g} of that. Only the amplitude changes."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=(INTERMITTENT, CONTINUOUS),
        required=True,
        help=(
            f"'{INTERMITTENT}' = session-based Guided CSV demo "
            f"({GUIDED_DEMO_SESSIONS_PER_DAY} sessions/day); "
            f"'{CONTINUOUS}' = one uninterrupted recording at "
            f"{GUIDED_CONTINUOUS_DEMO_FS_HZ} Hz."
        ),
    )
    parser.add_argument(
        "--days",
        type=_positive_days,
        required=True,
        help="Requested recording duration in days.",
    )
    parser.add_argument(
        "--tonic",
        choices=sorted(TONIC_SCALES),
        required=True,
        help="Slow Signal amplitude: 'high' (shipped) or 'low' (x0.25).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Parent folder to create the dataset in. The dataset folder itself "
            "is created inside it and must not already exist."
        ),
    )
    return parser.parse_args()


def _run_intermittent(days: float, tonic_scale: float, parent: Path):
    sessions = int(round(days * GUIDED_DEMO_SESSIONS_PER_DAY))
    if sessions < 1:
        raise SystemExit(
            f"--days {days:g} is shorter than one "
            f"{1 / GUIDED_DEMO_SESSIONS_PER_DAY:g}-day session."
        )
    actual_days = sessions / GUIDED_DEMO_SESSIONS_PER_DAY

    def _progress(done: int, total: int) -> None:
        if done == total or done % 48 == 0:
            print(f"  session {done}/{total}", flush=True)

    print(
        f"Generating {actual_days:g}-day intermittent Guided demo "
        f"({sessions} sessions)..."
    )
    result = generate_guided_csv_demo(
        parent,
        progress=_progress,
        _session_count=sessions,
        _tonic_scale=tonic_scale,
    )
    return result, actual_days, f"sessions: {sessions}"


def _run_continuous(days: float, tonic_scale: float, parent: Path):
    duration_sec = days * SECONDS_PER_DAY
    samples = int(round(duration_sec * GUIDED_CONTINUOUS_DEMO_FS_HZ))
    if samples < 2:
        raise SystemExit(f"--days {days:g} is too short to sample.")
    actual_days = samples / GUIDED_CONTINUOUS_DEMO_FS_HZ / SECONDS_PER_DAY

    def _progress(done: int, total: int) -> None:
        if done == total or done % 72 == 0:
            print(f"  block {done}/{total}", flush=True)

    print(
        f"Generating {actual_days:g}-day continuous Guided demo "
        f"({samples} samples at {GUIDED_CONTINUOUS_DEMO_FS_HZ} Hz)..."
    )
    result = generate_guided_continuous_demo(
        parent,
        progress=_progress,
        _duration_sec=duration_sec,
        _tonic_scale=tonic_scale,
    )
    return result, actual_days, f"samples: {samples}"


def main() -> int:
    args = parse_args()
    tonic_scale = TONIC_SCALES[args.tonic]
    shipped = (
        GUIDED_DEMO_TONIC_AMPLITUDE_AU
        if args.mode == INTERMITTENT
        else GUIDED_CONTINUOUS_DEMO_TONIC_AMPLITUDE_AU
    )
    amplitudes = tuple(value * tonic_scale for value in shipped)

    runner = _run_intermittent if args.mode == INTERMITTENT else _run_continuous
    result, actual_days, size_line = runner(
        float(args.days), tonic_scale, Path(args.output).expanduser()
    )

    print(result.message)
    if not result.success:
        return 1
    print(f"input folder: {result.input_dir}")
    print(f"format: {result.format}")
    print(f"mode: {args.mode}")
    print(f"generated duration: {actual_days:g} days ({actual_days * 24:g} hours)")
    print(size_line)
    print(f"tonic: {args.tonic} (x{tonic_scale:g})")
    print(f"tonic amplitude AU (ROI1/ROI2): {amplitudes[0]:g}/{amplitudes[1]:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
