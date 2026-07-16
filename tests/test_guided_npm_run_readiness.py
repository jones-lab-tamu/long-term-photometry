from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import photometry_pipeline.guided_npm_run_readiness as npm_readiness
from tests.test_npm_user_language import _accepted_npm_outcome


def _unchecked(instance, **changes):
    result = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            result, item.name, changes.get(item.name, getattr(instance, item.name))
        )
    return result


def _evaluate(**changes):
    base = {
        "validation_outcome": _accepted_npm_outcome(),
        "validation_revision": 7,
        "current_gui_revision": 7,
        "execution_active": False,
        "execution_result_pending": False,
    }
    base.update(changes)
    return npm_readiness.evaluate_guided_npm_run_readiness(**base)


def test_no_validation_requires_validation():
    result = _evaluate(validation_outcome=None)
    assert result.status == "no_validation"
    assert result.ready is False


def test_validation_not_accepted_refuses():
    outcome = _unchecked(
        _accepted_npm_outcome(),
        status="validator_refused",
        accepted_for_backend_validation=False,
    )
    result = _evaluate(validation_outcome=outcome)
    assert result.status == "validation_not_accepted"
    assert result.ready is False


def test_stale_validation_revision_mismatch_refuses():
    result = _evaluate(validation_revision=6)
    assert result.status == "validation_stale"
    assert result.ready is False


def test_stale_validation_outcome_flag_refuses():
    outcome = _unchecked(_accepted_npm_outcome(), stale=True)
    result = _evaluate(validation_outcome=outcome)
    assert result.status == "validation_stale"
    assert result.ready is False


def test_active_run_refuses():
    result = _evaluate(execution_active=True)
    assert result.status == "run_active"
    assert result.ready is False


def test_pending_result_refuses():
    result = _evaluate(execution_result_pending=True)
    assert result.status == "result_pending"
    assert result.ready is False


def test_fully_accepted_state_is_ready():
    result = _evaluate()
    assert result.status == "ready"
    assert result.ready is True
    assert result.user_summary == (
        "This NPM recording setup was checked successfully and is ready "
        "to run."
    )


def test_status_precedence_prefers_validation_over_active_run():
    # A stale/unaccepted validation must be reported even if a run is
    # (impossibly) also marked active -- validation state is checked first.
    result = _evaluate(validation_outcome=None, execution_active=True)
    assert result.status == "no_validation"


def test_wrong_type_validation_outcome_requires_validation():
    result = _evaluate(validation_outcome=SimpleNamespace())
    assert result.status == "no_validation"


def test_user_summaries_exclude_internal_terms():
    prohibited = (
        "manifest",
        "artifact",
        "receipt",
        "authority",
        "reconciliation",
        "worker",
        "subprocess",
        "exit code",
        "backend",
        "runtime",
        "identity",
        "canonical",
    )
    for summary in npm_readiness._SUMMARIES.values():
        lowered = summary.lower()
        assert not any(term in lowered for term in prohibited), summary
