"""Production capabilities exposed by the complete Guided workflow."""

from __future__ import annotations


# A mode belongs here only when Guided can validate, authorize, execute,
# complete, and open its result through the production path.
#
# "continuous" runs the one complete continuous production path (recording
# check -> review binding -> authority preparation -> one accepted continuous
# backend -> completed-run classification -> continuous Results). It admits RWD
# acquisition folders, and one generic CSV file whose columns the scientist has
# mapped in the ordinary CSV controls; both produce the same normalized
# recording description, so nothing downstream differs. Guided Setup refuses
# continuous for any other input format, refuses a CSV folder that does not
# hold exactly one recording file, and the older chunked custom_tabular
# continuous output workflow is not this path.
GUIDED_PRODUCTION_ACQUISITION_MODES: tuple[str, ...] = (
    "intermittent",
    "continuous",
)


def is_guided_production_acquisition_mode(mode: object) -> bool:
    """Return whether *mode* is supported by the complete Guided path."""
    return (
        isinstance(mode, str)
        and mode.strip().lower() in GUIDED_PRODUCTION_ACQUISITION_MODES
    )
