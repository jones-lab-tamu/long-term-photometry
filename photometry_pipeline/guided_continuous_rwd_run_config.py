"""CR1-E3: the one run-level ``Config`` a live continuous-RWD Guided run uses.

The accepted continuous-RWD backends
(``execute_guided_continuous_rwd_{correction,tonic,phasic,combined}_run``)
take a ``Config`` and forward it to the analysis-level publication steps --
the tonic/phasic HDF5 writers, ``generate_run_report``, and, for phasic, the
D3b-A detection kernel, which reads its detector settings from that object
(``feature_fields_from_config(config)`` /
``get_peak_indices_for_trace(trace, fs_hz, config)``).

Correction is deliberately NOT configured from here: the C4b segment
correction path resolves its own settings from the accepted startup mapping
contract on every segment and never consults the caller's ``Config``. What
this module composes therefore governs analysis publication and, for phasic,
event detection.

That makes the feature/event half load-bearing: handing the backends a bare
correction-only ``Config`` would silently detect events with bare
``Config()`` defaults instead of the settings the scientist confirmed in
Guided's Feature Detection step. This module composes exactly two accepted
sources and nothing else:

1. the accepted base correction settings, via
   ``guided_continuous_rwd_segment_correction.
   resolve_guided_continuous_rwd_correction_settings`` (the same
   ``(Config, identity)`` pair C4b resolves for itself); and
2. the accepted effective feature/event values of the accepted draft, via
   ``guided_new_analysis_plan.
   build_guided_feature_event_effective_values_preview`` -- the same pure
   builder Guided's backend-validation materialization already uses for this
   exact purpose.

It fails closed. A draft whose saved Feature Detection settings are not
current, whose effective values are unresolved or invalid, or whose active
fields would fall back to a silent backend default produces an error rather
than a run configured with values the scientist never confirmed. That mirrors
``guided_backend_validation_materialization``'s existing gate rather than
introducing a second, weaker rule.

Note on ``signal_excursion_polarity``: it belongs to
``FEATURE_EVENT_CONFIG_FIELDS`` and is applied here from the confirmed
profile. C4b separately pins its own copy as a fixed policy constant for
correction; because correction does not read this ``Config``, the two never
conflict.

No widget state, no filesystem, no GUI import: this is a pure function of an
accepted draft and the accepted startup mapping contract.
"""

from __future__ import annotations

from photometry_pipeline.config import Config
from photometry_pipeline.feature_event_config import FEATURE_EVENT_CONFIG_FIELDS


class GuidedContinuousRwdRunConfigError(ValueError):
    """The accepted plan cannot produce a complete run configuration."""


def build_guided_continuous_rwd_run_config(
    accepted_draft: object,
    startup_mapping_contract: object,
) -> Config:
    """Return the one ``Config`` for this accepted continuous-RWD plan.

    Raises :class:`GuidedContinuousRwdRunConfigError` when the accepted
    draft's confirmed feature/event settings cannot be resolved completely.
    The message is scientist-facing: callers may show it directly.
    """
    from photometry_pipeline.guided_backend_validation_request import (
        is_saved_feature_event_profile_current,
    )
    from photometry_pipeline.guided_continuous_rwd_segment_correction import (
        resolve_guided_continuous_rwd_correction_settings,
    )
    from photometry_pipeline.guided_new_analysis_plan import (
        build_guided_feature_event_effective_values_preview,
    )

    try:
        config, _identity = resolve_guided_continuous_rwd_correction_settings(
            startup_mapping_contract
        )
    except Exception as exc:
        raise GuidedContinuousRwdRunConfigError(
            "The accepted analysis settings for this recording are not "
            "available."
        ) from exc

    if not is_saved_feature_event_profile_current(
        str(getattr(accepted_draft, "feature_event_profile_status", "")),
        bool(getattr(accepted_draft, "feature_event_explicitly_applied", False)),
    ):
        raise GuidedContinuousRwdRunConfigError(
            "The saved Feature Detection settings are not ready for this "
            "analysis."
        )

    try:
        preview = build_guided_feature_event_effective_values_preview(
            accepted_draft
        )
    except Exception as exc:
        raise GuidedContinuousRwdRunConfigError(
            "The Feature Detection settings for this analysis could not be "
            "read."
        ) from exc

    validation_errors = list(preview.get("validation_errors") or ())
    if validation_errors:
        raise GuidedContinuousRwdRunConfigError(
            f"The Feature Detection settings are not valid: {validation_errors[0]}"
        )

    fields: dict[str, object] = {}
    for item in preview.get("effective_values") or ():
        name = item.get("field_name")
        if name not in FEATURE_EVENT_CONFIG_FIELDS:
            continue
        value = item.get("effective_value")
        # An ACTIVE field must come from the confirmed profile. A dormant
        # field legitimately carries its backend default (e.g.
        # peak_threshold_abs under the percentile method) and is still
        # serialized, exactly as the accepted materialization gate treats it.
        if item.get("active_or_inactive") == "active" and (
            item.get("source") != "applied_guided_profile" or value is None
        ):
            raise GuidedContinuousRwdRunConfigError(
                "The Feature Detection setting "
                f"'{name}' has not been confirmed for this analysis."
            )
        fields[str(name)] = value

    missing = sorted(FEATURE_EVENT_CONFIG_FIELDS - set(fields))
    if missing:
        raise GuidedContinuousRwdRunConfigError(
            "The Feature Detection settings are incomplete: "
            f"{missing}"
        )

    for name, value in fields.items():
        setattr(config, name, value)
    return config
