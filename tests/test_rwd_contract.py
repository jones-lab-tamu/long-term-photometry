from __future__ import annotations

import ast
from pathlib import Path

import pytest

from photometry_pipeline.io import rwd_contract
from photometry_pipeline.io.rwd_contract import (
    AMBIGUITY_POLICY,
    COLUMN_NORMALIZATION_RULE,
    ROI_NAME_RULE,
    RwdHeaderInspectionError,
    RwdHeaderParsingContract,
    compute_rwd_header_parsing_contract_digest,
    inspect_rwd_header_contract,
)
from photometry_pipeline.guided_backend_validation_workflow import (
    GUIDED_BACKEND_RWD_SIGNAL_SUFFIX_CANDIDATES,
    GUIDED_BACKEND_RWD_TIME_COLUMN_CANDIDATES,
    GUIDED_BACKEND_RWD_UV_SUFFIX_CANDIDATES,
)


def _contract(**overrides: object) -> RwdHeaderParsingContract:
    values: dict[str, object] = {
        "header_search_line_limit": 6,
        "time_column_candidates": ("Time(s)", "TimeStamp"),
        "uv_suffix_candidates": ("-410", "-415"),
        "signal_suffix_candidates": ("-470",),
        "column_normalization_rule": COLUMN_NORMALIZATION_RULE,
        "roi_name_rule": ROI_NAME_RULE,
        "ambiguity_policy": AMBIGUITY_POLICY,
    }
    values.update(overrides)
    return RwdHeaderParsingContract(**values)


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_valid_rwd_header_is_acceptable_and_preserves_exact_columns(tmp_path: Path):
    path = _write(
        tmp_path / "fluorescence.csv",
        [
            "vendor metadata",
            "Time(s),ROI1-410,ROI1-470,ROI2-415,ROI2-470",
            "0,1,2,3,4",
        ],
    )
    result = inspect_rwd_header_contract(str(path), parsing_contract=_contract())

    assert result.acceptable_for_strict_identity is True
    assert result.header_row_index == 1
    assert result.selected_time_column == "Time(s)"
    assert result.roi_ids == ("ROI1", "ROI2")
    assert [
        (pair.roi_id, pair.raw_uv_column, pair.raw_signal_column)
        for pair in result.roi_channel_pairs
    ] == [
        ("ROI1", "ROI1-410", "ROI1-470"),
        ("ROI2", "ROI2-415", "ROI2-470"),
    ]
    assert result.blocking_findings == ()


def test_rwd_header_contract_accepts_timestamp_time_column(tmp_path: Path):
    """4J16k18: a real RWD fluorescence.csv can have a non-CSV preamble
    line followed by a header row whose time column is named "TimeStamp"
    rather than "Time(s)", with non-contiguous ROI numbering (CH3 absent
    here). Uses the actual Guided production candidate constants, not a
    test-local contract, so this test would fail if the fix were
    reverted."""
    path = _write(
        tmp_path / "fluorescence.csv",
        [
            "vendor metadata",
            "TimeStamp,Events,CH1-410,CH1-470,CH2-410,CH2-470,CH4-410,CH4-470",
            "0.000,,56.605,32.896,46.159,27.513,78.174,38.478",
        ],
    )
    contract = RwdHeaderParsingContract(
        time_column_candidates=GUIDED_BACKEND_RWD_TIME_COLUMN_CANDIDATES,
        uv_suffix_candidates=GUIDED_BACKEND_RWD_UV_SUFFIX_CANDIDATES,
        signal_suffix_candidates=GUIDED_BACKEND_RWD_SIGNAL_SUFFIX_CANDIDATES,
    )
    result = inspect_rwd_header_contract(str(path), parsing_contract=contract)

    assert result.header_row_index == 1
    assert result.selected_time_column == "TimeStamp"
    assert result.roi_ids == ("CH1", "CH2", "CH4")
    assert result.acceptable_for_strict_identity is True
    assert result.blocking_findings == ()


def test_rwd_header_contract_still_accepts_time_s_time_column(tmp_path: Path):
    """4J16k18 companion: the pre-existing "Time(s)" column name must keep
    working unchanged after adding "TimeStamp" as a second candidate."""
    path = _write(
        tmp_path / "fluorescence.csv",
        ["Time(s),CH1-410,CH1-470"],
    )
    contract = RwdHeaderParsingContract(
        time_column_candidates=GUIDED_BACKEND_RWD_TIME_COLUMN_CANDIDATES,
        uv_suffix_candidates=GUIDED_BACKEND_RWD_UV_SUFFIX_CANDIDATES,
        signal_suffix_candidates=GUIDED_BACKEND_RWD_SIGNAL_SUFFIX_CANDIDATES,
    )
    result = inspect_rwd_header_contract(str(path), parsing_contract=contract)

    assert result.selected_time_column == "Time(s)"
    assert result.roi_ids == ("CH1",)
    assert result.acceptable_for_strict_identity is True


def test_raw_duplicate_columns_block_before_pandas_mangling(tmp_path: Path):
    path = _write(
        tmp_path / "fluorescence.csv",
        ["Time(s),ROI1-410,ROI1-410,ROI1-470"],
    )
    result = inspect_rwd_header_contract(str(path), parsing_contract=_contract())
    assert result.acceptable_for_strict_identity is False
    assert result.duplicate_raw_columns
    assert {finding.category for finding in result.blocking_findings} >= {
        "duplicate_raw_column",
        "duplicate_roi_base",
    }


def test_multiple_time_candidates_return_blocking_inspection(tmp_path: Path):
    path = _write(
        tmp_path / "fluorescence.csv",
        ["Time(s),TimeStamp,ROI1-410,ROI1-470"],
    )
    result = inspect_rwd_header_contract(str(path), parsing_contract=_contract())
    assert result.selected_time_column is None
    assert result.ambiguous_time_columns
    assert result.acceptable_for_strict_identity is False


@pytest.mark.parametrize(
    "header",
    [
        "OtherTime,ROI1-410,ROI1-470",
        "Time(s),ROI1-410,ROI2-470",
    ],
)
def test_missing_time_or_pair_raises_header_not_found(tmp_path: Path, header: str):
    path = _write(tmp_path / "fluorescence.csv", [header])
    with pytest.raises(RwdHeaderInspectionError) as excinfo:
        inspect_rwd_header_contract(str(path), parsing_contract=_contract())
    assert excinfo.value.category == "rwd_header_not_found"


def test_duplicate_roi_base_and_ambiguous_uv_suffix_pairing_block(tmp_path: Path):
    path = _write(
        tmp_path / "fluorescence.csv",
        ["Time(s),ROI1-410,ROI1-415,ROI1-470"],
    )
    result = inspect_rwd_header_contract(str(path), parsing_contract=_contract())
    assert result.duplicate_roi_bases
    assert result.ambiguous_pairings
    assert result.acceptable_for_strict_identity is False


def test_ambiguous_signal_suffix_pairing_blocks(tmp_path: Path):
    path = _write(
        tmp_path / "fluorescence.csv",
        ["Time(s),ROI1-410,ROI1-470,ROI1-480"],
    )
    contract = _contract(signal_suffix_candidates=("-470", "-480"))
    result = inspect_rwd_header_contract(str(path), parsing_contract=contract)
    assert result.ambiguous_pairings
    assert result.acceptable_for_strict_identity is False


def test_reused_column_ambiguity_blocks(tmp_path: Path):
    path = _write(
        tmp_path / "fluorescence.csv",
        ["Time(s),ROI-410,ROI-470,ROI--470"],
    )
    contract = _contract(uv_suffix_candidates=("-410", "410"))
    result = inspect_rwd_header_contract(str(path), parsing_contract=contract)
    assert result.reused_columns
    assert result.acceptable_for_strict_identity is False


def test_overlapping_uv_signal_candidates_are_contract_invalid():
    with pytest.raises(RwdHeaderInspectionError) as excinfo:
        _contract(signal_suffix_candidates=("-470", "-410"))
    assert excinfo.value.category == "invalid_rwd_parsing_contract"


def test_casefold_roi_collision_blocks(tmp_path: Path):
    path = _write(
        tmp_path / "fluorescence.csv",
        ["Time(s),ROI1-410,ROI1-470,roi1-410,roi1-470"],
    )
    result = inspect_rwd_header_contract(str(path), parsing_contract=_contract())
    assert result.casefold_roi_collisions
    assert result.acceptable_for_strict_identity is False


@pytest.mark.parametrize(
    "header,reason",
    [
        ("Time(s),-410,-470", "empty"),
        ("Time(s),ROI\x01-410,ROI\x01-470", "control_character"),
        ("Time(s),ROI/1-410,ROI/1-470", "path_separator"),
        ("Time(s),ROI\\\\1-410,ROI\\\\1-470", "path_separator"),
        ("Time(s),ROI1 -410,ROI1 -470", "surrounding_whitespace"),
    ],
)
def test_invalid_roi_ids_block(tmp_path: Path, header: str, reason: str):
    path = _write(tmp_path / "fluorescence.csv", [header])
    result = inspect_rwd_header_contract(str(path), parsing_contract=_contract())
    assert result.invalid_roi_ids
    assert reason in {finding.context["reason"] for finding in result.invalid_roi_ids}
    assert result.acceptable_for_strict_identity is False


def test_header_beyond_search_limit_raises_not_found(tmp_path: Path):
    path = _write(
        tmp_path / "fluorescence.csv",
        ["metadata 1", "metadata 2", "Time(s),ROI1-410,ROI1-470"],
    )
    with pytest.raises(RwdHeaderInspectionError) as excinfo:
        inspect_rwd_header_contract(
            str(path),
            parsing_contract=_contract(header_search_line_limit=2),
        )
    assert excinfo.value.category == "rwd_header_not_found"


def test_parser_stops_after_candidate_header_and_does_not_parse_signal_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = _write(
        tmp_path / "fluorescence.csv",
        [
            "metadata",
            "Time(s),ROI1-410,ROI1-470",
            '"malformed,signal,row',
            "0,1,2",
        ],
    )
    line_count = 0
    original_parse = rwd_contract._parse_line

    def tracking_parse(line: str, line_index: int):
        nonlocal line_count
        line_count += 1
        return original_parse(line, line_index)

    monkeypatch.setattr(rwd_contract, "_parse_line", tracking_parse)
    result = inspect_rwd_header_contract(str(path), parsing_contract=_contract())
    assert result.acceptable_for_strict_identity is True
    assert line_count == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"header_search_line_limit": 0},
        {"time_column_candidates": ()},
        {"uv_suffix_candidates": ()},
        {"signal_suffix_candidates": ()},
        {"time_column_candidates": ("Time(s)", "Time(s)")},
        {"time_column_candidates": "Time(s)"},
        {"uv_suffix_candidates": ("",)},
        {"unresolved_inputs": ("dataset_contract",)},
        {"column_normalization_rule": "unsupported"},
        {"roi_name_rule": "unsupported"},
        {"ambiguity_policy": "unsupported"},
        {"schema_name": "unsupported"},
        {"schema_version": "unsupported"},
    ],
)
def test_invalid_parsing_contracts_fail_closed(overrides: dict[str, object]):
    with pytest.raises(RwdHeaderInspectionError) as excinfo:
        _contract(**overrides)
    assert excinfo.value.category == "invalid_rwd_parsing_contract"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("time_column_candidates", None),
        ("uv_suffix_candidates", None),
        ("signal_suffix_candidates", None),
        ("unresolved_inputs", None),
        ("time_column_candidates", 123),
        ("uv_suffix_candidates", 123),
        ("signal_suffix_candidates", 123),
        ("unresolved_inputs", 123),
        ("time_column_candidates", {"Time(s)": True}),
        ("uv_suffix_candidates", {"-410": True}),
        ("signal_suffix_candidates", {"-470": True}),
        ("unresolved_inputs", {"dataset_contract": True}),
        ("time_column_candidates", {"Time(s)"}),
        ("header_search_line_limit", None),
        ("schema_name", None),
        ("schema_version", None),
    ],
)
def test_malformed_contract_field_types_raise_categorized_error(
    field_name: str,
    value: object,
):
    with pytest.raises(RwdHeaderInspectionError) as excinfo:
        _contract(**{field_name: value})
    assert excinfo.value.category == "invalid_rwd_parsing_contract"


def test_candidate_lists_are_accepted_and_normalized_to_tuples():
    contract = _contract(
        time_column_candidates=["Time(s)"],
        uv_suffix_candidates=["-410"],
        signal_suffix_candidates=["-470"],
        unresolved_inputs=[],
    )
    assert contract.time_column_candidates == ("Time(s)",)
    assert contract.uv_suffix_candidates == ("-410",)
    assert contract.signal_suffix_candidates == ("-470",)
    assert contract.unresolved_inputs == ()


def test_parser_contract_digest_is_deterministic_lowercase_sha256():
    contract = _contract()
    first = compute_rwd_header_parsing_contract_digest(contract)
    second = compute_rwd_header_parsing_contract_digest(contract)
    assert first == second
    assert len(first) == 64
    assert first == first.lower()
    int(first, 16)


@pytest.mark.parametrize(
    "overrides",
    [
        {"header_search_line_limit": 7},
        {"time_column_candidates": ("TimeStamp", "Time(s)")},
        {"uv_suffix_candidates": ("-415", "-410")},
        {"signal_suffix_candidates": ("-480", "-470")},
    ],
)
def test_parser_contract_digest_changes_with_semantic_fields(
    overrides: dict[str, object],
):
    assert compute_rwd_header_parsing_contract_digest(
        _contract()
    ) != compute_rwd_header_parsing_contract_digest(_contract(**overrides))


def test_parser_contract_digest_list_and_tuple_inputs_are_equivalent():
    tuple_contract = _contract()
    list_contract = _contract(
        time_column_candidates=list(tuple_contract.time_column_candidates),
        uv_suffix_candidates=list(tuple_contract.uv_suffix_candidates),
        signal_suffix_candidates=list(tuple_contract.signal_suffix_candidates),
        unresolved_inputs=[],
    )
    assert compute_rwd_header_parsing_contract_digest(
        tuple_contract
    ) == compute_rwd_header_parsing_contract_digest(list_contract)


@pytest.mark.parametrize("value", [None, {}, object()])
def test_parser_contract_digest_rejects_non_contract_inputs(value: object):
    with pytest.raises(RwdHeaderInspectionError) as excinfo:
        compute_rwd_header_parsing_contract_digest(value)  # type: ignore[arg-type]
    assert excinfo.value.category == "invalid_rwd_parsing_contract"


def test_parser_contract_digest_rejects_unresolved_contract_instance():
    contract = object.__new__(RwdHeaderParsingContract)
    for name, value in _contract().__dict__.items():
        object.__setattr__(contract, name, value)
    object.__setattr__(contract, "unresolved_inputs", ("dataset_contract",))

    with pytest.raises(RwdHeaderInspectionError) as excinfo:
        compute_rwd_header_parsing_contract_digest(contract)
    assert excinfo.value.category == "invalid_rwd_parsing_contract"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("column_normalization_rule", "unsupported"),
        ("roi_name_rule", "unsupported"),
        ("ambiguity_policy", "unsupported"),
    ],
)
def test_parser_contract_digest_rule_changes_must_be_supported(
    field_name: str,
    value: str,
):
    with pytest.raises(RwdHeaderInspectionError) as excinfo:
        _contract(**{field_name: value})
    assert excinfo.value.category == "invalid_rwd_parsing_contract"


def test_parser_module_has_no_forbidden_imports_or_config_instantiation():
    source = Path(rwd_contract.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert imported_roots.isdisjoint(
        {
            "gui",
            "subprocess",
            "pandas",
            "yaml",
            "photometry_pipeline",
        }
    )
    assert "Config(" not in source
    assert "RunSpec" not in source
