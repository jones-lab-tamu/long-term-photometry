from __future__ import annotations

import ast
import builtins
from dataclasses import fields, replace
import inspect
import math
from pathlib import Path

import pytest

import photometry_pipeline.guided_continuous_rwd_discontinuity_policy as subject
import photometry_pipeline.guided_continuous_rwd_recording as recording
from photometry_pipeline.guided_continuous_rwd_discontinuity_policy import (
    ABSOLUTE_FLOOR_SECONDS,
    EMPIRICAL_MULTIPLIER,
    INTERVAL_CATEGORIES,
    MATERIAL_LONG_INTERVAL,
    NORMAL_INTERVAL,
    POLICY_NAME,
    POLICY_VERSION,
    RELATIVE_TOLERANCE_FRACTION,
    SHORT_INTERVAL_ANOMALY,
    UPPER_CAP_FRACTION,
    ContinuousRwdDiscontinuityPolicyError,
    ContinuousRwdIntervalClassification,
    classify_continuous_rwd_interval,
    resolve_continuous_rwd_discontinuity_tolerance,
)
from photometry_pipeline.guided_continuous_rwd_recording import (
    ContinuousRwdCadenceAuthority,
    ContinuousRwdCadenceQuantile,
)


def _cadence(
    nominal: float = 0.1,
    *,
    q01: float | None = None,
    q50: float | None = None,
    q99: float | None = None,
) -> ContinuousRwdCadenceAuthority:
    q01 = nominal * 0.999 if q01 is None else q01
    q50 = nominal if q50 is None else q50
    q99 = nominal * 1.001 if q99 is None else q99
    draft = ContinuousRwdCadenceAuthority(
        cadence_evidence_policy_version="relative-to-measured-median.v1",
        positive_interval_count=1000,
        nominal_cadence_seconds=nominal,
        minimum_positive_dt_seconds=min(q01, q50, q99),
        maximum_positive_dt_seconds=max(q01, q50, q99),
        mean_positive_dt_seconds=nominal,
        standard_deviation_positive_dt_seconds=nominal * 0.0001,
        coefficient_of_variation=0.0001,
        quantile_method="deterministic_reservoir_linear.v1",
        quantile_sample_count=1000,
        quantiles=(
            ContinuousRwdCadenceQuantile(0.01, q01),
            ContinuousRwdCadenceQuantile(0.5, q50),
            ContinuousRwdCadenceQuantile(0.99, q99),
        ),
        unusually_long_interval_count=0,
        unusually_short_interval_count=0,
        largest_unusual_intervals=(),
        smallest_unusual_intervals=(),
        cadence_evidence_identity="",
    )
    identity = recording._digest(
        recording.CADENCE_EVIDENCE_IDENTITY_DOMAIN,
        recording._cadence_payload(draft),
    )
    result = replace(draft, cadence_evidence_identity=identity)
    recording._validate_cadence(result)
    return result


def _with_quantiles(cadence, quantiles):
    return replace(cadence, quantiles=tuple(quantiles))


def test_exact_policy_metadata_constants_and_categories():
    assert POLICY_NAME == "continuous-rwd-observed-cadence-discontinuity"
    assert POLICY_VERSION == "v1"
    assert ABSOLUTE_FLOOR_SECONDS == 0.001
    assert RELATIVE_TOLERANCE_FRACTION == 0.02
    assert EMPIRICAL_MULTIPLIER == 1.5
    assert UPPER_CAP_FRACTION == 0.20
    assert INTERVAL_CATEGORIES == (
        "normal_interval",
        "short_interval_anomaly",
        "material_long_interval",
    )


@pytest.mark.parametrize("nominal", [0.1, 0.05, 0.025])
def test_valid_10_20_and_40_hz_cadence_resolves_deterministically(nominal):
    cadence = _cadence(nominal)
    first = resolve_continuous_rwd_discontinuity_tolerance(cadence)
    second = resolve_continuous_rwd_discontinuity_tolerance(cadence)
    assert first == second
    assert first.policy_name == POLICY_NAME
    assert first.policy_version == POLICY_VERSION
    assert first.nominal_cadence_seconds == nominal
    assert first.q01_seconds == cadence.quantiles[0].dt_seconds
    assert first.q99_seconds == cadence.quantiles[2].dt_seconds
    assert first.absolute_floor_seconds == ABSOLUTE_FLOOR_SECONDS
    assert first.uncapped_tolerance_seconds == max(
        first.absolute_floor_seconds,
        first.relative_component_seconds,
        first.empirical_component_seconds,
    )
    assert first.final_tolerance_seconds == min(
        first.upper_cap_seconds,
        first.uncapped_tolerance_seconds,
    )


@pytest.mark.parametrize("missing", [0.01, 0.5, 0.99])
def test_missing_required_quantile_is_refused(missing):
    cadence = _cadence()
    malformed = _with_quantiles(
        cadence, (item for item in cadence.quantiles if item.probability != missing)
    )
    with pytest.raises(ContinuousRwdDiscontinuityPolicyError, match="exactly once"):
        resolve_continuous_rwd_discontinuity_tolerance(malformed)


def test_duplicate_required_quantile_is_refused():
    cadence = _cadence()
    malformed = _with_quantiles(
        cadence,
        cadence.quantiles[:2]
        + (ContinuousRwdCadenceQuantile(0.5, 0.1),)
        + cadence.quantiles[2:],
    )
    with pytest.raises(ContinuousRwdDiscontinuityPolicyError):
        resolve_continuous_rwd_discontinuity_tolerance(malformed)


def test_required_quantile_probabilities_must_already_be_strictly_ordered():
    cadence = _cadence()
    malformed = _with_quantiles(cadence, reversed(cadence.quantiles))
    with pytest.raises(ContinuousRwdDiscontinuityPolicyError, match="ascending"):
        resolve_continuous_rwd_discontinuity_tolerance(malformed)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"q50": 0.1001}, "agree"),
        ({"q01": 0.1001}, "bracket"),
        ({"q99": 0.0999}, "bracket"),
    ],
)
def test_required_quantiles_must_agree_with_and_bracket_nominal(changes, message):
    cadence = _cadence()
    values = {
        0.01: changes.get("q01", cadence.quantiles[0].dt_seconds),
        0.5: changes.get("q50", cadence.quantiles[1].dt_seconds),
        0.99: changes.get("q99", cadence.quantiles[2].dt_seconds),
    }
    malformed = _with_quantiles(
        cadence,
        (
            ContinuousRwdCadenceQuantile(probability, values[probability])
            for probability in (0.01, 0.5, 0.99)
        ),
    )
    with pytest.raises(ContinuousRwdDiscontinuityPolicyError, match=message):
        resolve_continuous_rwd_discontinuity_tolerance(malformed)


@pytest.mark.parametrize("nominal", [math.nan, math.inf, -math.inf])
def test_nonfinite_nominal_is_refused(nominal):
    cadence = replace(_cadence(), nominal_cadence_seconds=nominal)
    with pytest.raises(ContinuousRwdDiscontinuityPolicyError, match="finite"):
        resolve_continuous_rwd_discontinuity_tolerance(cadence)


@pytest.mark.parametrize("nominal", [0.0, -0.1])
def test_nonpositive_nominal_is_refused(nominal):
    cadence = replace(_cadence(), nominal_cadence_seconds=nominal)
    with pytest.raises(ContinuousRwdDiscontinuityPolicyError, match="greater than zero"):
        resolve_continuous_rwd_discontinuity_tolerance(cadence)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_required_quantile_is_refused(value):
    cadence = _cadence()
    malformed = _with_quantiles(
        cadence,
        (
            cadence.quantiles[0],
            ContinuousRwdCadenceQuantile(0.5, value),
            cadence.quantiles[2],
        ),
    )
    with pytest.raises(ContinuousRwdDiscontinuityPolicyError, match="finite"):
        resolve_continuous_rwd_discontinuity_tolerance(malformed)


@pytest.mark.parametrize("value", [0.0, -0.1])
def test_nonpositive_required_quantile_is_refused(value):
    cadence = _cadence()
    malformed = _with_quantiles(
        cadence,
        (
            ContinuousRwdCadenceQuantile(0.01, value),
            cadence.quantiles[1],
            cadence.quantiles[2],
        ),
    )
    with pytest.raises(ContinuousRwdDiscontinuityPolicyError, match="greater than zero"):
        resolve_continuous_rwd_discontinuity_tolerance(malformed)


def test_wrong_cadence_type_is_refused():
    with pytest.raises(ContinuousRwdDiscontinuityPolicyError, match="cadence must"):
        resolve_continuous_rwd_discontinuity_tolerance(object())


def test_absolute_floor_dominates_for_small_regular_period():
    resolved = resolve_continuous_rwd_discontinuity_tolerance(
        _cadence(0.025, q01=0.025, q99=0.025)
    )
    assert resolved.final_tolerance_seconds == 0.001


def test_relative_component_dominates_for_regular_10_hz_cadence():
    resolved = resolve_continuous_rwd_discontinuity_tolerance(
        _cadence(0.1, q01=0.0999, q99=0.1001)
    )
    assert resolved.final_tolerance_seconds == pytest.approx(0.002)
    assert resolved.final_tolerance_seconds == resolved.relative_component_seconds


def test_empirical_component_dominates_wider_observed_spread():
    resolved = resolve_continuous_rwd_discontinuity_tolerance(
        _cadence(0.1, q01=0.095, q99=0.105)
    )
    assert resolved.empirical_component_seconds == pytest.approx(0.0075)
    assert resolved.final_tolerance_seconds == resolved.empirical_component_seconds


def test_upper_cap_truncates_excessively_wide_empirical_component():
    resolved = resolve_continuous_rwd_discontinuity_tolerance(
        _cadence(0.1, q01=0.01, q99=0.19)
    )
    assert resolved.uncapped_tolerance_seconds > resolved.upper_cap_seconds
    assert resolved.final_tolerance_seconds == pytest.approx(0.02)


@pytest.mark.parametrize(
    "cadence",
    [
        _cadence(0.1, q01=0.1, q99=0.1),
        _cadence(0.05, q01=0.0495, q99=0.0505),
        _cadence(0.025, q01=0.02, q99=0.03),
    ],
)
def test_final_tolerance_is_positive_capped_and_deterministic(cadence):
    first = resolve_continuous_rwd_discontinuity_tolerance(cadence)
    second = resolve_continuous_rwd_discontinuity_tolerance(cadence)
    assert first == second
    assert math.isfinite(first.final_tolerance_seconds)
    assert 0.0 < first.final_tolerance_seconds <= 0.20 * first.nominal_cadence_seconds


def test_classification_result_has_exact_fields():
    assert tuple(item.name for item in fields(ContinuousRwdIntervalClassification)) == (
        "category",
        "dt_seconds",
        "nominal_cadence_seconds",
        "tolerance_seconds",
        "residual_seconds",
    )


def test_exact_nominal_and_modest_symmetric_jitter_are_normal():
    tolerance = resolve_continuous_rwd_discontinuity_tolerance(_cadence())
    for dt in (0.1, 0.099, 0.101):
        result = classify_continuous_rwd_interval(dt, tolerance=tolerance)
        assert result.category == NORMAL_INTERVAL
        assert result.dt_seconds == dt
        assert result.nominal_cadence_seconds == 0.1
        assert result.tolerance_seconds == tolerance.final_tolerance_seconds
        assert result.residual_seconds == abs(dt - 0.1)


def test_exact_tolerance_boundaries_are_normal():
    tolerance = resolve_continuous_rwd_discontinuity_tolerance(_cadence())
    lower = tolerance.nominal_cadence_seconds - tolerance.final_tolerance_seconds
    upper = tolerance.nominal_cadence_seconds + tolerance.final_tolerance_seconds
    assert classify_continuous_rwd_interval(lower, tolerance=tolerance).category == NORMAL_INTERVAL
    assert classify_continuous_rwd_interval(upper, tolerance=tolerance).category == NORMAL_INTERVAL


def test_immediately_outside_tolerance_boundaries_is_anomalous():
    tolerance = resolve_continuous_rwd_discontinuity_tolerance(_cadence())
    lower = tolerance.nominal_cadence_seconds - tolerance.final_tolerance_seconds
    upper = tolerance.nominal_cadence_seconds + tolerance.final_tolerance_seconds
    assert classify_continuous_rwd_interval(math.nextafter(lower, 0.0), tolerance=tolerance).category == SHORT_INTERVAL_ANOMALY
    assert classify_continuous_rwd_interval(math.nextafter(upper, math.inf), tolerance=tolerance).category == MATERIAL_LONG_INTERVAL


@pytest.mark.parametrize(
    ("factor", "expected"),
    [
        (0.25, SHORT_INTERVAL_ANOMALY),
        (0.5, SHORT_INTERVAL_ANOMALY),
        (2.0, MATERIAL_LONG_INTERVAL),
        (2.4, MATERIAL_LONG_INTERVAL),
    ],
)
def test_clear_short_and_long_intervals_use_only_three_categories(factor, expected):
    tolerance = resolve_continuous_rwd_discontinuity_tolerance(_cadence())
    result = classify_continuous_rwd_interval(0.1 * factor, tolerance=tolerance)
    assert result.category == expected
    assert result.category in INTERVAL_CATEGORIES


@pytest.mark.parametrize(
    "value",
    [0.0, -0.1, math.nan, math.inf, -math.inf, True, False, "0.1"],
)
def test_invalid_interval_is_refused(value):
    tolerance = resolve_continuous_rwd_discontinuity_tolerance(_cadence())
    with pytest.raises(ContinuousRwdDiscontinuityPolicyError):
        classify_continuous_rwd_interval(value, tolerance=tolerance)


def test_target_rate_and_metadata_arguments_are_not_accepted():
    cadence = _cadence()
    with pytest.raises(TypeError):
        resolve_continuous_rwd_discontinuity_tolerance(cadence, target_fs_hz=20.0)
    with pytest.raises(TypeError):
        resolve_continuous_rwd_discontinuity_tolerance(cadence, metadata_fps=20.0)
    tolerance = resolve_continuous_rwd_discontinuity_tolerance(cadence)
    with pytest.raises(TypeError):
        classify_continuous_rwd_interval(0.1, tolerance=tolerance, led_count=2)


def test_policy_performs_no_filesystem_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("filesystem access is forbidden")

    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "open", forbidden)
        guarded.setattr(Path, "open", forbidden)
        guarded.setattr(Path, "exists", forbidden)
        guarded.setattr(Path, "stat", forbidden)
        tolerance = resolve_continuous_rwd_discontinuity_tolerance(_cadence())
        assert classify_continuous_rwd_interval(0.1, tolerance=tolerance).category == NORMAL_INTERVAL


def test_module_imports_only_standard_library_and_cr1b1_cadence_authority():
    tree = ast.parse(inspect.getsource(subject))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imports == {
        "__future__",
        "dataclasses",
        "typing",
        "photometry_pipeline.guided_continuous_rwd_recording",
    }
    direct_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert direct_imports == {"math"}


def test_policy_exposes_no_path_serialization_or_identity_surface():
    assert set(item.name for item in fields(subject.ContinuousRwdDiscontinuityTolerance)).isdisjoint(
        {"source_path", "recording_identity", "policy_identity"}
    )
    assert set(item.name for item in fields(ContinuousRwdIntervalClassification)).isdisjoint(
        {"source_path", "row_index", "elapsed_time", "recording_identity"}
    )
    public_names = {name for name in subject.__dict__ if not name.startswith("_")}
    assert not any(name.startswith(("serialize", "deserialize", "compute_identity")) for name in public_names)
    source = inspect.getsource(subject)
    assert "inspect_continuous_rwd_acquisition_folder" not in source
    assert "import gui" not in source
    assert "import photometry_pipeline.pipeline" not in source
