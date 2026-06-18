# Guided Workflow run-plan contract design

This Stage 4D6 document defines the future Guided Workflow run-plan layer. It
does not implement Guided production execution, write manifests, create
applied-dF/F outputs, create features, run validation, run the pipeline, change
Full Control behavior, or change backend analysis behavior.

## 1. Executive summary

Guided Workflow needs a durable run-plan layer because long-duration
photometry analysis cannot be safely represented by whichever ROI, chunk, or
diagnostic control is currently visible in the GUI. The plan must be an
explicit, reproducible object that describes what will run, which sources it
uses, which ROIs it targets, which strategy choices were explicitly marked,
which evidence was reviewed, and which existing backend machinery will execute
it later.

The concrete problem this prevents is current visible GUI selection becoming
production scope. Evidence chunks are useful for preview and QC, but they must
not define what a future Run step executes. Similarly, a current ROI combo-box
selection must not overwrite or imply a run plan entry unless the user performs
an explicit plan-edit action.

This stage defines the contract only. It does not execute anything.

## 2. Existing source-of-truth inventory

The Stage 4D5 audit established that the app already has the backend and Full
Control machinery needed for long-duration analysis. The Guided run plan should
reuse those source-of-truth components rather than reimplementing them.

| Capability | Existing source of truth | Files/functions/classes | Current scope | How Guided run plan should reuse it | What must not be duplicated |
| --- | --- | --- | --- | --- | --- |
| Raw input source | Full Control input state and RunSpec | `gui/main_window.py`, `gui/run_spec.py::RunSpec.input_dir`, pipeline CLI args | Raw input folder for new analysis | Store a resolved source reference and later feed it into RunSpec | New raw discovery or loader logic |
| Completed-run source | Completed-run parser/loader | `gui/run_report_parser.py`, `gui/run_report_viewer.py`, `MainWindow._open_completed_results_dir`, `is_successful_completed_run_dir` | Completed run for review, diagnostics, and planning | Store completed-run reference when plan is derived from existing outputs | Treat completed run as raw input without explicit reanalysis boundary |
| Recording structure | RunSpec, Config, discovery/pipeline | `gui/run_spec.py`, `photometry_pipeline/config.py`, `photometry_pipeline/pipeline.py::discover_files`, GUI RWD/NPM contract inference | Run/source setup | Reference RunSpec-derived setup or a recorded config snapshot | Second discovery/contract system |
| ROI include/exclude | Pipeline and RunSpec include/exclude semantics | `RunSpec.include_roi_ids`, `RunSpec.exclude_roi_ids`, `Pipeline.run(... include_rois, exclude_rois ...)` | Selected ROI set for a run | Represent target ROIs explicitly and later map to RunSpec include/exclude | Per-widget ROI selection as implicit run scope |
| Correction/dynamic fit settings | Config plus regression dispatcher | `Config.dynamic_fit_mode`, `photometry_pipeline.core.regression.fit_chunk_dynamic`, dynamic fitting modules, Guided correction cards | Config/run setup and future ROI-level strategy planning | Store explicit ROI-level correction choices and map production-compatible choices to existing Config/applied tools | New correction implementation or silent fallback |
| Signal-Only F0/reference-free workflow | Core candidate diagnostics and applied-dF/F tools | `photometry_pipeline.core.signal_only_f0_candidate`, `tools/write_applied_dff_cache.py`, `tools/run_applied_dff_pipeline.py`, `tools/run_applied_dff_batch.py`, diagnostic backends | Explicit applied strategy and diagnostic evidence | Store only explicit `signal_only_f0` choices; reuse applied tools for later execution | Automatic Signal-Only F0 selection or denominator recomputation semantics |
| Feature/event config fields | Config, knob registry, parsing helpers, feature backend | `Config` peak/event fields, `gui/knobs_registry.py`, `parse_and_validate_event_feature_knobs`, `core.feature_extraction.extract_features` | Run/profile-like settings; chunk-internal execution | Reference future feature/event profiles using existing Config fields | New detector, new peak semantics, or chunk-scoped production settings |
| Validation | GUI RunSpec generation and runner validation path | `MainWindow._build_run_spec`, `RunSpec.generate_derived_config`, `tools/run_full_pipeline_deliverables.py`, GUI Validate Only | Run-level validation | Future plan validation should check plan completeness before using existing validation machinery | Independent validation with different pass/fail semantics |
| Run execution | Runner and pipeline | `tools/run_full_pipeline_deliverables.py`, `photometry_pipeline.pipeline.Pipeline.run` | Dataset/run execution | Future Guided Run should translate a validated plan into RunSpec/config/runner calls | Direct execution from combo-box state |
| Output/provenance writing | Existing writers and reports | Pipeline `run_report.json`, `MANIFEST.json`, `status.json`, preview/diagnostic summaries, applied-dF/F summaries/provenance | Output contract by run/tool | Plan should carry IDs/references that downstream outputs can echo | New uncoordinated provenance format for production outputs |
| Completed-run review/loading | Existing parser/viewer and artifact resolution | `gui/run_report_parser.py`, `gui/run_report_viewer.py`, completed-run artifact helpers | Completed run and ROI/output review | Future Guided Review should read plan-linked outputs through existing readers | Rebuilt partial review viewer |
| Applied-dF/F production | Explicit applied-dF/F toolchain | `write_applied_dff_cache`, `verify_applied_dff_cache`, `run_applied_dff_features`, `verify_applied_dff_feature_outputs`, `run_applied_dff_pipeline`, `run_applied_dff_batch` | ROI plus explicit strategy | A future plan can reference these as explicit ROI strategy execution stages | Executable `auto`, `needs_review`, or `no_correction` production routing |

## 3. Proposed run-plan scope model

The run plan should separate user decisions from evidence context.

### Plan-level scope

Plan-level fields describe the plan itself: schema version, plan ID, creation
time, app/git version if available, mode, status, and intended execution class.
Plan-level state is not tied to current widgets.

### Source-level scope

Source-level fields describe where the plan comes from:

- `new_analysis`: raw input folder plus RunSpec-derived setup.
- `completed_run_planning`: completed run plus source/cache/config references
  for review, diagnostics, applied-dF/F, or reference-free reanalysis.

The plan must distinguish these modes so Open Results review cannot be confused
with raw-data execution.

### ROI-level scope

ROI-level entries are the normal unit for correction strategy planning. Each
ROI entry can store an explicit correction strategy choice, its status, and the
evidence that was reviewed.

### Selected-ROI-group-level scope

Future Guided actions may apply a choice to selected ROIs or all compatible
ROIs. That should be represented as an explicit plan edit that expands to ROI
entries or records a group rule with resolved ROI membership. It must not mean
"whatever ROI is visible now."

### Feature/event-profile-level scope

Feature/event settings should be profile-level, run-level, selected-ROI-level,
or ROI-level. Chunks can be used to preview/QC those settings but cannot be the
production scope for feature/event configuration.

### Evidence/provenance-level scope

Evidence entries describe what the user inspected: preview IDs, diagnostic IDs,
artifact paths, source hashes, ROI, evidence chunk, stale state, and summary
text. Evidence is provenance, not routing.

### Output-destination-level scope

Output-destination fields describe where future outputs may be written and
which safety policy applies. They must be checked before any write and must not
default to source or legacy output locations.

### Explicit scope rules

- Evidence chunks may be recorded as provenance.
- Evidence chunks must not define production scope.
- Current selected ROI/chunk widgets must not define production scope.
- Future execution must consume the run plan, not live widget state.

## 4. Proposed run-plan object shape

This is a contract sketch, not an implemented schema.

```json
{
  "schema_version": "guided_run_plan.v1",
  "plan_id": "uuid-or-stable-id",
  "created_at_utc": "2026-06-18T00:00:00Z",
  "updated_at_utc": "2026-06-18T00:00:00Z",
  "app_version": "",
  "git_commit": "",
  "mode": "new_analysis | completed_run_planning",
  "status": "draft | complete_for_validation | validated | execution_ready | executed",
  "source": {
    "source_mode": "raw_input | completed_run",
    "raw_input_dir": "",
    "completed_run_dir": "",
    "phasic_out_dir": "",
    "source_config_path": "",
    "source_config_sha256": "",
    "source_run_report_path": "",
    "source_manifest_path": "",
    "source_hashes": {}
  },
  "recording_structure": {
    "run_spec_reference": {
      "format": "",
      "acquisition_mode": "",
      "sessions_per_hour": null,
      "session_duration_s": null,
      "continuous_window_sec": null,
      "include_roi_ids": [],
      "exclude_roi_ids": []
    },
    "config_overrides": {},
    "data_contract_overrides": {},
    "discovery_summary": {},
    "recording_structure_status": "not_checked | checked | stale"
  },
  "roi_plan": [
    {
      "roi": "CH1",
      "roi_status": "planned | needs_review | excluded",
      "correction_strategy": {
        "strategy": "robust_global_event_reject | adaptive_event_gated_regression | global_linear_regression | signal_only_f0",
        "strategy_label": "",
        "choice_source": "explicit_user_mark",
        "choice_status": "marked | stale | superseded",
        "marked_at_utc": "",
        "marked_from_completed_run_dir": "",
        "no_auto_selection": true
      },
      "feature_event_profile_id": "default_profile",
      "evidence": {
        "reviewed_chunks": [
          {
            "chunk_id": 0,
            "role": "representative_evidence",
            "preview_artifact_paths": [],
            "signal_only_f0_diagnostic_paths": [],
            "summary": "",
            "stale": false
          }
        ],
        "diagnostic_artifacts": [],
        "evidence_notes": ""
      }
    }
  ],
  "roi_group_rules": [
    {
      "rule_id": "",
      "action": "apply_strategy_to_selected_rois | apply_profile_to_selected_rois",
      "resolved_rois": [],
      "created_by_explicit_user_action": true
    }
  ],
  "feature_event_profiles": [
    {
      "profile_id": "default_profile",
      "scope": "run | roi | selected_roi_group",
      "config_fields": {
        "event_signal": "dff",
        "signal_excursion_polarity": "positive",
        "peak_threshold_method": "mean_std",
        "peak_threshold_k": 2.5,
        "peak_threshold_percentile": 95.0,
        "peak_threshold_abs": 0.0,
        "peak_min_distance_sec": 1.0,
        "peak_min_prominence_k": 2.0,
        "peak_min_width_sec": 0.3,
        "peak_pre_filter": "none",
        "event_auc_baseline": "zero"
      },
      "evidence_previews": []
    }
  ],
  "validation": {
    "plan_completeness_status": "not_checked | passed | failed",
    "source_status": "not_checked | passed | failed",
    "roi_coverage_status": "not_checked | passed | failed",
    "strategy_coverage_status": "not_checked | passed | failed",
    "feature_event_profile_status": "not_checked | passed | failed",
    "output_destination_status": "not_checked | passed | failed",
    "run_spec_generation_status": "not_checked | passed | failed",
    "messages": []
  },
  "output_policy": {
    "output_root": "",
    "overwrite": false,
    "separate_from_source_required": true,
    "legacy_outputs_protected": true,
    "applied_dff_output_policy": "separate_output_root_only"
  },
  "provenance": {
    "created_by": "Guided Workflow",
    "source_of_truth_references": [],
    "evidence_artifact_references": [],
    "no_manifest_written": true,
    "no_pipeline_execution": true,
    "no_feature_extraction": true,
    "no_auto_strategy_selection": true
  }
}
```

The initial implementation should not add all fields at once. The first model
can be a minimal draft object if it preserves the scope boundaries above.

## 5. ROI-level correction plan semantics

Current Confirm Strategy UI-state should eventually map into plan entries as
follows:

- Key strategy choices by source context and ROI, not chunk.
- Store `evidence_chunk` only inside evidence/provenance fields.
- Allow multiple ROIs to have independent choices.
- Allow future explicit apply-to-selected-ROIs or
  apply-to-all-compatible-ROIs actions.
- Preserve the fact that Signal-Only F0 is explicit and never automatic.
- Reject `auto`, `needs_review`, and `no_correction` as runnable plan
  strategies.
- Store the selected correction strategy as an explicit user mark with timestamp
  and source references.
- Preserve stale-state detection when evidence was generated from another
  completed run, ROI, or evidence chunk.

Mapping from current 4D4 UI-state:

```text
self._guided_strategy_choices[(completed_run_dir, roi)] = {
    "strategy": "...",
    "strategy_label": "...",
    "confirmed": True,
    "completed_run_dir": "...",
    "roi": "...",
    "evidence_chunk": 0,
    "evidence_summary": {...}
}
```

Future plan mapping:

```text
roi_plan[roi].correction_strategy.strategy = entry["strategy"]
roi_plan[roi].correction_strategy.choice_source = "explicit_user_mark"
roi_plan[roi].evidence.reviewed_chunks += entry["evidence_chunk"]
```

The evidence chunk must not appear in the key for production routing.

## 6. Feature/event planning placeholder semantics

Guided mode should not design a new detector. Future feature/event planning
should refer to existing Config fields and `feature_extraction` behavior.

Required semantics:

- Feature/event settings are profile-level, run-level, selected-ROI-level, or
  ROI-level.
- Feature/event settings are not chunk-level production decisions.
- Chunks may be used for preview/QC of a profile.
- Existing fields must be reused:
  - `event_signal`
  - `signal_excursion_polarity`
  - `peak_threshold_method`
  - `peak_threshold_k`
  - `peak_threshold_percentile`
  - `peak_threshold_abs`
  - `peak_min_distance_sec`
  - `peak_min_prominence_k`
  - `peak_min_width_sec`
  - `peak_pre_filter`
  - `event_auc_baseline`
- Existing output semantics must be reused:
  - standard pipeline `phasic_out/features/features.csv`
  - applied-dF/F feature output `features.csv/json`
  - feature summary/provenance JSON/CSV where present
- Missing from Guided is planning/exposure, not detector backend.

The plan should store a feature/event profile reference on ROI entries rather
than copying feature parameters into every ROI entry unless a ROI override is
explicitly created.

## 7. Validation and execution boundary

Before future Guided Run can execute, the plan must pass explicit checks:

- Plan completeness:
  - schema version supported
  - source mode valid
  - required source references present
  - target ROI set resolved
- Source existence:
  - raw input directory exists for new analysis
  - completed run exists and passes completed-run checks for completed-run
    planning
  - referenced phasic cache/config/report paths exist when required
- ROI coverage:
  - planned ROIs are present in discovery or completed-run cache
  - include/exclude semantics are resolved through existing pipeline/RunSpec
    behavior
- Strategy coverage:
  - all required planned ROIs have explicit runnable strategy choices
  - forbidden strategies are rejected
  - Signal-Only F0 entries are explicit user choices
- Feature/event profile checks:
  - profile fields are valid Config keys
  - profile values pass existing Config/parser validation
  - profile scope is run/ROI/group, not chunk
- Output destination safety:
  - output root is separate from source data, completed-run production outputs,
    `phasic_out`, and legacy features
  - overwrite policy is explicit and path-safe
- RunSpec/config generation checks:
  - future execution maps plan data into `RunSpec`
  - `RunSpec.generate_derived_config` remains the config writer
  - `RunSpec.build_runner_argv` remains the runner argv builder
- Provenance checks:
  - plan ID/schema is recorded in future outputs
  - source hashes/references are preserved where available
  - evidence artifact references are carried forward as provenance

Future execution must consume the plan, not current GUI selection.

## 8. Open Results boundary

Open Results currently loads a completed run and supports Diagnostics and
Confirm Strategy planning. A future run plan must distinguish completed-run
planning from raw-input new-analysis planning.

Open questions to resolve before implementation:

- Is completed-run planning limited to applied-dF/F/reference-free reanalysis
  from an existing `phasic_trace_cache.h5`?
- Can a completed-run plan ever trigger raw-data rerun, or must that require
  switching to new-analysis mode with raw input and RunSpec setup?
- Should ROI-level correction marks made from completed-run diagnostics map only
  to applied-dF/F production tools, or can they update a future new-analysis
  correction plan?

Boundary requirements:

- `completed_run_dir` must be explicit in the plan source.
- `raw_input_dir` must be explicit for raw-data execution.
- A plan derived from a completed run must not silently write into the completed
  run production directories.
- Open Results mode must remain read-only unless a future explicit reanalysis
  plan permits separate-output writes.
- Any completed-run reanalysis plan must use completed-run artifacts as inputs
  and separate output roots as outputs.
- Guided Run should remain unavailable in Open Results mode until this boundary
  is implemented and tested.

## 9. Review/provenance implications

Future Guided Review needs plan references to make long-duration output review
revisitable:

- plan ID
- schema version
- source mode and source paths
- raw input or completed-run references
- ROI strategy choices
- feature/event profile settings
- evidence chunks and diagnostic artifact paths
- output root and generated artifacts
- source hashes and config hashes where available
- validation status and messages
- execution status
- stale-state checks comparing the current loaded source to plan source

Review should summarize and navigate outputs across ROIs and chunks. It should
not require the user to discover every chunk manually.

## 10. Acceptance criteria for future implementation

A future run-plan implementation must satisfy these criteria:

- Changing selected evidence chunk cannot change production scope.
- Changing visible ROI cannot change plan entries unless the user explicitly
  edits the plan.
- Diagnostics cannot auto-populate runnable strategy choices.
- Signal-Only F0 diagnostic success cannot auto-select Signal-Only F0.
- Plan serialization round-trip preserves ROI-level choices and evidence
  provenance.
- Future Run consumes the plan, not combo-box state.
- Open Results mode cannot write production outputs unless a future explicit
  reanalysis plan allows it.
- Forbidden strategies (`auto`, `needs_review`, `no_correction`) are rejected
  from runnable plan entries.
- Feature/event settings are profile/run/ROI scoped, not evidence-chunk scoped.
- Output destination checks run before any write.
- Plan provenance is included in future outputs.
- Existing RunSpec/config/pipeline machinery is reused for execution.

## 11. Tests needed before wiring production Run

Required tests before any production Run wiring:

- Plan serialization round-trip with multiple ROIs.
- Evidence chunk recorded as provenance, not production scope.
- Changing selected widget state does not alter plan without explicit edit.
- Forbidden strategies are rejected.
- Signal-Only F0 remains explicit only.
- Feature/event settings are profile/run/ROI scoped.
- Feature/event profile validation uses existing Config/parser rules.
- RunSpec/config generation from plan uses existing `RunSpec` machinery.
- Future Run consumes plan entries, not live combo-box state.
- Open Results production-write guard.
- Output destination safety for raw-input and completed-run planning.
- Completed-run plan cannot mutate completed-run production outputs.
- Plan stale-state detection when source, ROI list, evidence artifacts, or
  config references change.
- Multiple ROI choices remain independent.
- Apply-to-selected-ROIs resolves explicit ROI membership at plan-edit time.

## 12. Recommended next implementation stages

Recommended sequence:

1. Stage 4D6a: Add plan contract tests and a minimal dataclass/schema only.
   No GUI, no writes beyond test fixtures, no execution.
2. Stage 4D6b: Map Confirm Strategy UI-state into an in-memory plan preview.
   No persistence, no manifests, no outputs.
3. Stage 4D6c: Add optional draft plan export if desired. It must be clearly
   non-executable and separate from production manifests.
4. Stage 4E: Design Guided feature/event planning using existing detector and
   Config machinery. Include profile/run/ROI scope before UI.
5. Later: Wire Guided Run only after plan validation, output safety, RunSpec
   mapping, and provenance tests exist.

## 13. Non-goals for this contract

- No Guided production execution.
- No manifests.
- No applied-dF/F outputs.
- No feature extraction.
- No validation or pipeline calls.
- No new detector.
- No new correction implementation.
- No Full Control changes.
- No automatic strategy selection.
- No per-chunk production decisions.

## 14. Files inspected for this contract

- `guided_workflow_long_duration_audit.md`
- `gui/main_window.py`
- `gui/run_spec.py`
- `gui/knobs_registry.py`
- `gui/run_report_parser.py`
- `gui/run_report_viewer.py`
- `photometry_pipeline/config.py`
- `photometry_pipeline/pipeline.py`
- `photometry_pipeline/core/feature_extraction.py`
- `photometry_pipeline/preview/correction_preview.py`
- `photometry_pipeline/signal_only_f0_diagnostics/contract.py`
- `photometry_pipeline/signal_only_f0_diagnostics/generate.py`
- `tools/write_applied_dff_cache.py`
- `tools/verify_applied_dff_cache.py`
- `tools/run_applied_dff_features.py`
- `tools/verify_applied_dff_feature_outputs.py`
- `tools/run_applied_dff_pipeline.py`
- `tools/run_applied_dff_batch.py`
- Guided, run-spec, feature/event, preview, diagnostic, applied-dF/F, and
  pipeline tests under `tests/`.
