"""Explicit application capabilities for temporary Guided eligibility gates."""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GuidedExecutionCapabilities:
    """Non-serialized UI capability policy.

    Production construction uses the default. Tests may inject the sole
    temporary capability explicitly; it is never copied into plans or runner
    artifacts.
    """

    allow_signal_only_f0_execution: bool = False


@dataclass(frozen=True)
class GuidedSignalOnlyExecutionEligibility:
    allowed: bool
    category: str
    user_message: str


def evaluate_guided_signal_only_execution_eligibility(
    selected_strategies: Iterable[str],
    capabilities: GuidedExecutionCapabilities,
) -> GuidedSignalOnlyExecutionEligibility:
    """Apply the temporary Signal-Only gate at any Run entry boundary."""
    if not isinstance(capabilities, GuidedExecutionCapabilities):
        raise TypeError("capabilities must be GuidedExecutionCapabilities")
    has_signal_only = any(
        str(strategy) == "signal_only_f0" for strategy in selected_strategies
    )
    if has_signal_only and not capabilities.allow_signal_only_f0_execution:
        return GuidedSignalOnlyExecutionEligibility(
            allowed=False,
            category="signal_only_f0_execution_not_available",
            user_message="This correction approach is not available to run yet.",
        )
    return GuidedSignalOnlyExecutionEligibility(
        allowed=True,
        category="eligible",
        user_message="",
    )


PRODUCTION_GUIDED_EXECUTION_CAPABILITIES = GuidedExecutionCapabilities()
