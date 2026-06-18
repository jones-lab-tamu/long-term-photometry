# Guided Workflow Draft Plan Import/Open Design

## Scope
This document designs the safety boundaries, validation layers, user review workflows, and state scoping required to support importing/opening an exported GuidedRunPlan JSON file back into the Guided Workflow. The primary objective is to define a strict read-only review model that prevents external JSON payloads from silently mutating active GUI widgets, current draft plan state, loaded completed-run contexts, or executing any logic or filesystem writes.

## Existing State
Currently, the Guided Workflow supports exporting an in-memory draft run plan to a JSON file. The UI layout has been split so that Confirm Strategy is the step where correction strategies are marked, and Draft Plan is the step where feature/event profiles, output policies, and readiness checklists reside. 

Key attributes of the existing system to preserve:
- `GuidedRunPlan` (defined in `photometry_pipeline/guided_run_plan.py`) is a planning and provenance container, not an execution engine.
- `RunSpec` is the separate, executable configuration object that translates plans into runtime execution parameters.
- `_current_run_dir` / `_current_guided_completed_run_dir()` represent the authoritative source-scoping boundary.
- Active widget values exist in volatile controls and do not mutate the in-memory draft plan state unless the user explicitly clicks "Apply" or "Mark strategy".
- Current draft plan state stores in `MainWindow` include:
  - `self._guided_strategy_choices` (mapped by run directory)
  - `self._guided_draft_feature_event_profiles_by_run`
  - `self._guided_draft_output_policy_by_run`

## Terminology
To ensure semantic clarity throughout future implementation, we define:
*   **Exported plan JSON**: The raw serialized file on disk created by a user action on the Draft Plan step containing a serialized `GuidedRunPlan`.
*   **Imported plan file**: A local JSON file selected by the user for import/open.
*   **Candidate imported plan**: An in-memory, read-only `GuidedRunPlan` object deserialized from an imported plan file. It is displayed in a dedicated candidate review panel but has no authority over live workflow widgets or current draft state.
*   **Current in-memory draft plan**: The active draft configuration representing the user's modifications in the current session.
*   **Adopted draft plan**: A candidate imported plan that has successfully passed all validation checks and has been explicitly copied into the current in-memory draft plan via a conscious user click in a later stage.
*   **Loaded completed run**: The active run directory selected via the "Open Results" mode that defines the source-scoping boundary.
*   **Plan source**: The completed run directory metadata stored within the imported plan (`source.completed_run_dir`).
*   **Current GUI source**: The actual loaded completed run folder active in the GUI.
*   **RunSpec**: The executable parameters container. This design does not touch or generate `RunSpec`.

A **candidate imported plan** is strictly distinct from the **current in-memory draft plan** and does not automatically become the active plan upon import/open.

## Recommended Model
We recommend a strict **Read-Only Candidate Review Model** for the initial import step (Stage 4H1):
1.  The user initiates import by selecting a JSON file.
2.  The application parses and validates the JSON file using contract rules.
3.  If valid, the plan is loaded into a separate candidate state store (e.g., `self._guided_imported_plan_candidate`).
4.  A read-only candidate review panel displays the plan details, validation warnings, source matching status, and ROI compatibility summary.
5.  **No automatic adoption occurs**. Active widgets, strategy choices, output policies, feature/event profiles, and loaded completed runs remain completely untouched.
6.  This design blocks execution and guarantees that simply opening a JSON plan has zero side effects on the workspace or current draft state.

## File Parsing and Validation
When a JSON file is selected, the application must handle the following failure/edge cases robustly:

*   **File too large**: Before reading the entire file into memory, check the file size (e.g. limit to 500 KB). If exceeded, reject the file immediately with "Import failed: File exceeds maximum size limit (500 KB)." No candidate is created, no state mutates, and no files are written.
*   **JSON top-level is not an object/dict**: If the parsed JSON is not a dictionary (e.g. it is a list, string, or number), reject with "Import failed: JSON root must be an object." No candidate is created.
*   **schema_version exists but is not a string**: Reject with "Import failed: schema_version must be a string." No candidate is created.
*   **Paths in JSON are non-string values**: Any path fields (e.g., `completed_run_dir`, `output_root`) must be strings. If they are other types, reject as a deserialization/contract error.
*   **Deeply nested JSON**: Reject JSON with nesting depth exceeding a safety threshold (e.g., depth > 10) to prevent parser stack overflow.
*   **Duplicate JSON keys**: Python's default `json.loads` keeps the last value for duplicate keys. To harden security, future implementation should optionally use `object_pairs_hook` to reject duplicate keys.
*   **Empty Path**: Reject selection immediately with "Import failed: Path cannot be empty."
*   **File Does Not Exist**: Reject selection immediately with "Import failed: File does not exist at <path>."
*   **Path is a Directory**: Reject selection immediately with "Import failed: Path points to a directory, not a file."
*   **Non-JSON Suffix**: Reject selection immediately with "Import failed: Plan file must have .json extension."
*   **Unreadable File**: Catch `OSError` / `PermissionError` and reject with "Import failed: File cannot be read (Permission denied)."
*   **Invalid JSON**: Catch `json.JSONDecodeError` and reject with "Import failed: Invalid JSON format. <error details>"
*   **Valid JSON, Not a Plan**: Check keys. Reject if schema is missing or invalid with "Import failed: JSON structure is not a valid GuidedRunPlan."
*   **Missing schema_version**: Reject deserialization with "Import failed: Missing plan schema version."
*   **Unsupported schema_version**: Reject deserialization with "Import failed: Unsupported schema version '<version>'."
*   **Deserialization Fails**: Catch `GuidedRunPlanContractError` and reject with "Import failed: Plan contract deserialization failed: <msg>"
*   **validate_plan_contract Fails**: Deserialize successfully, but mark candidate as invalid with "Candidate plan loaded with contract errors: <list of errors>"
*   **Contract Valid, Incomplete**: Deserialize successfully, mark candidate as incomplete but valid with "Candidate plan loaded. (Incomplete: <list of missing elements>)"
*   **Contract-Valid but Source-Incompatible**: Display candidate plan read-only, but flag source incompatibility and block adoption.
*   **Contract-Valid but Output-Unsafe**: Display candidate plan read-only, but flag output policy warnings/failures and block adoption.

## Schema Version Handling
*   **Exact match (`guided_run_plan.v1`)**: Load candidate plan and proceed to compatibility review.
*   **Older schema version**: Reject unless a migration layer is explicitly defined. No silent migration is allowed.
*   **Newer/Unknown/Non-string schema version**: Reject outright. Silently accepting or coercing newer/unknown schema versions into live draft state is forbidden. If read-only display is possible, it must clearly label the candidate plan as `Unsupported Schema - Adoption Blocked` and prevent any adoption actions.
*   **Missing schema version**: Reject for adoption.

## Source Matching
The imported plan source path (`plan.source.completed_run_dir`) must be checked against the current GUI loaded run (`self._current_guided_completed_run_dir()`).
1.  **Normalization**: Both paths must be normalized using `os.path.realpath` and `os.path.normpath` to resolve relative path segments, directory separators, and symlinks before comparison.
2.  **No Loaded Run**: If no run is currently loaded in the GUI, the candidate plan is marked `Mismatched Source - No Active Run Loaded`. Adoption is blocked. The current GUI source must not be changed automatically.
3.  **Source Path Matches**: If the normalized paths resolve to the same directory, they are treated as source matched.
4.  **Source Path Mismatch**: If the paths differ, the status is marked `Source Mismatched`. Adoption is blocked.
5.  **Source Path Missing**: If the source path in the JSON does not exist on the current filesystem, it is marked `Mismatched Source - Imported Path Does Not Exist`. Adoption is blocked.
6.  **Source Path Exists but is Not a Completed Run**: If the imported path exists but is not a valid completed run directory, mark as `Source path exists but is not a valid completed run`. Adoption is blocked.
7.  **Lacks Completed-Run Artifacts**: If the path exists but lacks required run artifacts (e.g., HDF5 traces, config_used.yaml), mark as `Source path lacks required completed-run artifacts`. Adoption is blocked.
8.  **Raw Input Mode**: If the imported plan uses `raw_input` rather than `completed_run`, it is rejected as incompatible because the current import/open design is strictly for completed-run planning. Raw-input plan import is a non-goal.

Source existence on disk is necessary but not sufficient. The candidate source must be a valid completed-run source for future adoption.

## ROI Compatibility
When the source matches, the candidate plan's ROI list must be verified against the active completed run's ROIs:
*   **ROI Discovery**: ROI names must be obtained using the authoritatively discovered ROIs from HDF5 metadata or completed-run cache. Do not invent or accept ROI names based solely on the imported JSON.
*   **No active run loaded**: Display imported ROIs but mark compatibility status as unknown and block adoption.
*   **All ROIs Match**: All imported ROIs exist in the loaded run. Marked as compatible.
*   **Missing ROIs**: Imported plan specifies an ROI name that does not exist in the loaded run. Mark candidate validation as `Failed (Missing ROIs)`. Adoption is blocked.
*   **Extra ROIs in Loaded Run**: The loaded run has ROIs not mentioned in the imported plan. This is compatible (the plan is partial). Mark as compatible but flag that some current ROIs will remain unconfigured. Future adoption may allow partial plans only as an incomplete draft state.
*   **Duplicate ROIs in Plan**: Reject as a contract/deserialization error. Adoption is blocked.
*   **Zero ROI Choices**: A plan with no ROI entries is contract-valid but marked as incomplete. Future adoption is blocked unless explicitly allowed as incomplete.
*   **Evidence Chunk Out of Range/Inventory Unavailable**: If the plan references a chunk index that is unavailable, display an evidence compatibility warning. Do not run analysis to recover or generate it.

## Correction Strategy Compatibility
The imported correction strategies must be checked against compatibility criteria:
*   **Valid Runnable Strategies**: Strategies in the `RUNNABLE_CORRECTION_STRATEGIES` list (e.g. `robust_global_event_reject`) are contract-valid. No execution occurs.
*   **Forbidden Strategies**: Strategies like `auto`, `needs_review`, or `no_correction` cause `validate_plan_contract` to fail, blocking adoption.
*   **Unknown Strategy**: Reject as contract error. Adoption is blocked.
*   **Valid but UI-Unavailable Strategy**: If a strategy is valid by contract but not supported by the current GUI version, display a UI-compatibility warning and block adoption.
*   **Signal-Only F0 Strategy**: This is valid only as an explicit reference-free strategy. It must be displayed clearly as explicit Signal-Only F0. It must not be treated as a fallback, must not be auto-selected because reference correction failed, and must not trigger or run the Signal-Only F0 diagnostic review.
*   **Evidence Chunk Provenance Missing**: Display as missing evidence provenance. The candidate may be incomplete/invalid depending on the contract.
*   **Evidence Chunk Provenance Stale/Unavailable**: Display stale/unavailable warning. Do not rerun previews or diagnostics. Adoption is blocked.
*   **Diagnostics/Previews Present in Evidence Summary**: Display as provenance only. Do not rerun diagnostics or previews, and do not treat diagnostics as auto-selection.

Candidate review must not run correction previews, diagnostics, applied-dF/F routing, manifests, or correction execution.

## Feature/Event Profile Compatibility
Imported feature/event profiles must be reviewed under strict safety guidelines:
*   **No profile**: Candidate review reports no feature/event profile configured. Plan is contract-valid but incomplete.
*   **One valid run-level profile**: Candidate review displays profile summary. Live widgets do not mutate.
*   **Multiple profiles**: If the contract allows, display and validate each profile separately showing profile_id, scope, status, and target/resolved ROIs. Adoption preserves these exact profiles only after source/ROI checks pass.
*   **Invalid config field**: Profile fails validation. Adoption is blocked.
*   **Unknown config field**: Profile is invalid. Adoption is blocked.
*   **Scope is chunk**: Profile is invalid under current contract. Adoption is blocked.
*   **Status is invalid, stale, needs_review, or unsupported**: Needs_review must not become an adoptable status. Stale/unsupported status must be displayed and block adoption.
*   **Stale or unsupported value relative to current helper validation**: Display invalid/stale value warning. Adoption is blocked.
*   **Profile references preview/evidence chunks**: Treated as provenance only. No previews or diagnostics are rerun, and no live widgets are mutated.
*   **Valid in JSON but GUI does not expose value**: Display as valid-by-contract but show a UI-compatibility warning. Adoption is blocked.

Feature/event import must reuse existing validation semantics and must not run feature extraction.

## Output Policy Safety
Validate_plan_contract is necessary but not sufficient for output-policy adoption safety. Import/open must run a separate output policy safety review against the current source before any future adoption.

### Imported OutputPolicy flags
*   **output_policy.output_root is None**: Candidate review reports no output destination configured. Candidate is contract-valid but incomplete. Future adoption allows this only as incomplete planning state, not executable readiness.
*   **output_policy.output_root is configured**:
    - Must not equal or be inside either the current completed-run directory or the imported plan source directory.
    - Must not point into legacy output directories.
    - Must not call `mkdir` or write probe files.
*   **overwrite is True**: Display a warning: "Safety warning: Overwrite is enabled." Future adoption should block overwrite=True initially, or require a later explicit remediation design.
*   **separate_from_source_required is False**: Display a warning: "Safety warning: separate_from_source_required is disabled." Future adoption should block this because it weakens the source/output separation contract.
*   **legacy_outputs_protected is False**: Display a warning: "Safety warning: legacy_outputs_protected is disabled." Future adoption should block this because it weakens the legacy-output protection contract.
*   **output_root is unsafe but other flags are safe**: Candidate review displays read-only, but adoption is blocked until the user fixes the output path.
*   **output_root is safe but any safety flag is unsafe**: Candidate review displays read-only, but adoption is blocked.

## Candidate Plan Review UI Concept
The Candidate Plan Review UI should be rendered as a new collapsible panel at the bottom of the `"Draft plan"` step named `"Imported plan review"`. It must be visually separate from the live Draft Plan readiness summary and checklist.

It must display:
*   Imported file path
*   Parse status
*   Validation status
*   Schema version
*   Plan ID
*   Plan mode
*   Imported source path
*   Imported source resolved path (if different)
*   Current source path
*   Current source resolved path (if different)
*   Source match/mismatch status
*   Source validity status
*   ROI compatibility summary
*   Missing imported ROIs
*   Extra current ROIs
*   Correction strategies summary (with Signal-Only F0 strategies clearly labeled)
*   Feature/event profiles summary
*   Output policy summary (with output policy flag warnings)
*   Contract errors
*   Compatibility warnings
*   Adoption status: `Adoption unavailable (Read-only review stage)`
*   Execution status: `Blocked: execution intentionally unavailable`
*   Files written: `none`

## Candidate State Storage
We compare two candidate state designs:
*   **Option A (Recommended for 4H1)**: Single current candidate state stored in GUI variables (`_guided_imported_plan_candidate`, `_guided_imported_plan_file_path`, `_guided_imported_plan_status`). The candidate does not affect source-scoped draft state, and is cleared when the user opens a different completed run or selects a new candidate.
*   **Option B**: Source-scoped candidate state (`_guided_imported_plan_candidate_by_run`). More complex, deferred unless cross-run comparison becomes a requirement.

We recommend **Option A** for its simplicity and safety. Candidate state must be kept strictly separate from:
- `_guided_strategy_choices`
- `_guided_draft_feature_event_profiles_by_run`
- `_guided_draft_output_policy_by_run`
- `_guided_export_editor_synced_run`
- Export path editor text
- Full Control state

## Future Adoption Model
*   Adoption is not part of Stage 4H1.
*   Adoption must be a separate explicit user click in Stage 4H2 or later.
*   Adoption is strictly source-matched and replace-only. Merge adoption is deferred because it creates conflict-resolution complexity.
*   Adoption warns if current draft state exists, and requires confirmation if replacing non-empty draft state.
*   Adoption re-runs contract validation, source compatibility, ROI compatibility, and output policy safety.
*   Adoption blocks unsafe OutputPolicy flags.
*   Adoption does not execute anything, write files, change Full Control, or change the export path.
*   GUI widgets synchronize only after a successful explicit adoption.

## Interaction With Export
*   Importing/opening a candidate must not alter the export path text edit.
*   Importing/opening a candidate must not trigger automatic export.
*   Importing/opening a candidate must not change the OutputPolicy.
*   Exporting while a candidate is merely open exports the current live draft plan, not the candidate.
*   After future adoption, exporting exports the adopted live draft state, not the original file blindly.
*   Import/open must not overwrite an existing export file.

## Interaction With Readiness and Checklist
*   The active live readiness summary and checklist remain about the current live draft plan.
*   The candidate imported plan gets its own separate candidate validation/readiness display.
*   A source-mismatched or incomplete candidate plan must not make the live readiness appear configured or complete.
*   Candidate properties (output policy, profile, ROI choices) must not satisfy or make live readiness summaries configured.

## Security and Robustness
The import process must implement the following safeguards:
*   **Very large JSON files**: Rejection before reading the entire file to avoid memory issues.
*   **Deeply nested JSON**: Reject to prevent stack overflow.
*   **Duplicate JSON keys**: Documented future hardening requirement to reject using `object_pairs_hook`.
*   **Non-string paths**: Deserialization error, no adoption.
*   **Stale paths**: Display warnings and block adoption.
*   **Platform differences in path resolution**: Always normalize paths using resolved paths (`os.path.realpath`) before comparison.
*   **Symlinks**: Resolve to real paths.
*   **Path traversal**: Normalize paths to prevent traversing up parent directories.
*   **Malicious/malformed JSON**: Caught safely and reported without crash.
*   **Unexpected types**: Deserialization helpers type-check fields.
*   **No code execution**: Never eval or dynamically import modules based on JSON content.
*   **No filesystem writes**: No permission probe files or temporary directory creation.
*   **No automatic loading**: Never load external files referenced by JSON beyond path comparison and metadata checks.
*   **Never trust JSON paths**: Treat all paths as raw data, never execute them.

## Future Test Plan

### Required 4H1 read-only candidate review tests:
*   `test_gui_import_invalid_json_rejected`: Malformed JSON is caught and reported.
*   `test_gui_import_top_level_non_dict_rejected`: Root list/string rejected.
*   `test_gui_import_unsupported_schema_rejected`: Incompatible schema version rejected.
*   `test_gui_import_missing_schema_rejected`: Missing schema version rejected.
*   `test_gui_import_non_string_schema_rejected`: Non-string schema version rejected.
*   `test_gui_import_file_too_large_rejected`: Files over 500 KB rejected.
*   `test_gui_import_unreadable_file_handled`: Handle permission/OS errors.
*   `test_gui_import_contract_invalid_plan_rejected`: Bad contracts block candidate creation.
*   `test_gui_import_contract_valid_incomplete_plan_opens_readonly`: Valid incomplete plan displays but marks incomplete status.
*   `test_gui_import_source_matched_valid_plan_opens_readonly_no_mutation`: Matches active run, does not change draft state or widgets.
*   `test_gui_import_source_mismatched_valid_plan_opens_readonly_no_mutation`: Mismatched source displays warning, does not mutate.
*   `test_gui_import_no_completed_run_loaded_opens_readonly_no_mutation`: Reviewable without loaded run, does not change source.
*   `test_gui_import_source_exists_but_not_completed_run`: Displays error and blocks.
*   `test_gui_import_source_path_missing`: Path not found warning.
*   `test_gui_import_roi_mismatch_display`: Missing/extra ROIs correctly listed.
*   `test_gui_import_evidence_chunk_unavailable_display`: Unavailable chunk index flags warning.
*   `test_gui_import_signal_only_f0_display_no_execution`: Verified no diagnostic run occurs.
*   `test_gui_import_feature_event_no_profile_display`: Configured profile absence shown.
*   `test_gui_import_feature_event_valid_profile_display`: Profile fields correctly listed read-only.
*   `test_gui_import_feature_event_invalid_profile_display`: Invalid profile fields show error.
*   `test_gui_import_output_policy_unsafe_root_display`: Bad paths show safety warning.
*   `test_gui_import_output_policy_unsafe_flags_display`: Warning shown for overwrite=True or unsafe flags.
*   `test_gui_import_readonly_guarantee`: Assert no files written, no directories created, no RunSpec generated, and no Full Control changes.
*   `test_gui_import_stepper_readiness_unchanged`: Live readiness/checklist labels unaffected.
*   `test_gui_import_candidate_cleared_on_run_switch`: Candidate state cleared when run changes.
*   `test_gui_import_candidate_replaced_on_new_import`: New candidate replaces old one.

### Required 4H2 future adoption tests:
*   `test_gui_adopt_button_absent_in_4h1`: Verify no adopt button exists in Stage 4H1.
*   `test_gui_adopt_button_explicit_only`: Verify adoption requires button click in 4H2.
*   `test_gui_adopt_source_mismatch_blocked`: Mismatched sources block adoption.
*   `test_gui_adopt_roi_mismatch_blocked`: Missing ROIs block adoption.
*   `test_gui_adopt_unsafe_output_root_blocked`: Unsafe paths block adoption.
*   `test_gui_adopt_unsafe_output_flags_blocked`: Unsafe flags block adoption.
*   `test_gui_adopt_contract_invalid_candidate_blocked`: Invalid contracts block adoption.
*   `test_gui_adopt_replaces_draft_state_after_confirmation`: Overwrites draft state, confirms replacement.
*   `test_gui_adopt_does_not_merge`: Replaces rather than merging.
*   `test_gui_adopt_does_not_change_export_path`: Export path remains unchanged.
*   `test_gui_adopt_does_not_write_files_or_run_analysis`: No file writes or execution.
*   `test_gui_adopt_does_not_affect_full_control`: Full control unchanged.

## Non-goals
*   No plan import/open implementation in Stage 4H.
*   No read-only import panel UI layout implementation in Stage 4H.
*   No adoption implementation in Stage 4H.
*   No RunSpec generation or RunSpec mapping.
*   No Guided Run or pipeline execution.
*   No feature extraction, correction execution, or preview generation.
*   No output directory creation or permissions preflight writing.
*   No Full Control changes.
*   No GUI controls added in Stage 4H.
*   No tests added in Stage 4H.

## Recommended Next Stages
1.  **Stage 4H1 (Next)**: Implement the read-only imported plan review panel on the `"Draft plan"` step. Users can select a JSON file, parse it, and view its details/compatibility status. No adoption, no draft state mutation, and no execution.
2.  **Stage 4H2**: Implement the explicit `"Adopt Plan"` button. If the candidate plan passes all source, ROI, and output policy validation checks, clicking `"Adopt Plan"` mutates the current GUI draft plan state to match the imported plan.
