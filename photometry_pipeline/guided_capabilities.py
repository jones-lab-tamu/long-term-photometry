"""Production capabilities exposed by the complete Guided workflow."""

from __future__ import annotations


# A mode belongs here only when Guided can validate, authorize, execute,
# complete, and open its result through the production path.
GUIDED_PRODUCTION_ACQUISITION_MODES: tuple[str, ...] = ("intermittent",)


def is_guided_production_acquisition_mode(mode: object) -> bool:
    """Return whether *mode* is supported by the complete Guided path."""
    return (
        isinstance(mode, str)
        and mode.strip().lower() in GUIDED_PRODUCTION_ACQUISITION_MODES
    )
