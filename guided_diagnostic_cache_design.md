# Guided Diagnostic Cache Design

## 1. Executive Summary

Recommended architecture: implement Model B as a Guided-owned "Build diagnostic cache" stage that starts from raw Guided setup state, launches preliminary cache generation using Tuning Prep semantics, records a diagnostic-cache artifact record, and then lets correction preview, Signal-Only F0 diagnostic review, and Confirm Strategy consume that cache for strategy decisions. This avoids Bad Model C because Guided must create the cache from raw input inside the new-analysis workflow; an already completed production run may remain a secondary review/refinement source, but it must not be required for first-pass strategy planning.

The diagnostic cache may be completed-run-like internally because current consumers need `phasic_trace_cache.h5`, `config_used.yaml`, success evidence, ROI inventory, and chunk/session inventory. The user-facing product must not call this a final analysis. It should be labeled as preliminary diagnostic data that supports correction decisions before a separate future production Run action.

## 2. User Workflow

Intended Guided flow:

```text
Select input data
-> Confirm or repair acquisition structure
-> Discover ROIs and optionally exclude ROIs
-> Build diagnostic cache
-> Review correction preview / Signal-Only F0 diagnostic / Confirm Strategy
-> Choose per-ROI correction strategies
-> Edit feature/event settings
-> Choose final output destination
-> Review analysis plan
-> Run final production analysis in a future stage
```

Recommended placement in the Guided stepper:

- Keep raw setup in the existing early setup steps: Select Data and Recording Structure.
- Keep ROI discovery before diagnostics.
- Add "Build diagnostic cache" as the gate between ROI setup and Diagnostics. It can either be its own step or the first required action inside the Diagnostics step. The cleaner architecture is a distinct step because it performs a write and has its own readiness/staleness state.
- Diagnostics, Signal-Only F0 review, and Confirm Strategy should read from the active diagnostic cache record, not from an unrelated completed production run.

Readiness conditions before "Build diagnostic cache" is enabled:

- Raw input path is set and exists.
- Input format is selected and supported for the selected acquisition mode.
- Acquisition mode is valid.
- Intermittent fields are valid when intermittent mode is selected: sessions/hour and session duration if provided.
- Continuous fields are valid when continuous mode is selected: window > 0, step equals window in the current implementation, allow partial final window captured.
- ROI discovery has completed successfully.
- At least one ROI is included.
- Active baseline/custom config resolves and validates.
- Diagnostic output/cache destination passes shared output safety checks.
- No unsupported format/mode combination is selected, especially continuous NPM.

Proposed status/help text:

```text
Build diagnostic cache
Creates preliminary diagnostic artifacts from the selected raw input and included ROIs.
This is not the final production analysis. It is used only to review correction evidence
and choose per-ROI strategies. Final analysis still requires a later explicit Run action.
```

After cache generation:

```text
Diagnostic cache ready.
Correction previews and strategy decisions will use this preliminary cache.
If input, structure, ROI selection, config, or diagnostic scope changes, rebuild the cache before using it for decisions.
```

On stale cache:

```text
Diagnostic cache is stale because setup changed. Existing strategy choices are preserved for review but are marked stale until the cache is rebuilt or choices are reconfirmed.
```

## 3. Diagnostic Cache Artifact Contract

| Artifact / metadata | Required? | Current producer | Current consumer | Provided by Tuning Prep? | Needed adaptation | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Success/status marker | Required | `tools/run_full_pipeline_deliverables.py` writes `status.json`; completed-run resolvers also accept equivalent success evidence | Completed-run source resolvers in correction preview and Signal-Only F0 diagnostics | Yes | Add diagnostic-cache-specific profile/status marker so UI distinguishes preliminary cache from production run | Must indicate preliminary diagnostic cache, not final production analysis. |
| `run_report.json` or equivalent metadata | Required | Runner/root report generation | Completed-run loading, result workspace, tuning workflows | Yes | Add or stamp diagnostic-cache purpose, raw source fingerprint, selected ROIs, diagnostic scope, and cache build request id | Existing completed-run-like metadata is useful but needs product-specific labeling. |
| Source/input provenance | Required | Runner manifest/report and GUI `RunSpec` artifacts | Future Guided plan, staleness checks, audit review | Partially | Store raw input path, resolved path, format, source fingerprint if available, and config source | Required to prevent Model B from becoming "open any completed run". |
| Acquisition structure metadata | Required | RunSpec/effective config and runner status/manifest | Pipeline, continuous planning, future Guided plan | Partially | Promote into diagnostic-cache artifact record and staleness signature | Include intermittent/continuous mode, sessions/hour, session duration, continuous window, continuous step, allow partial final window. |
| ROI inventory | Required | Discovery and `PhotometryPipeline.run()` ROI selection metadata | Confirm Strategy, future plan readiness | Yes via cache/report | Record discovered ROIs, included ROIs, excluded ROIs, and selection source | Strategy choices must be scoped to included ROIs. |
| Session/chunk inventory | Required | Phasic trace cache | Correction preview, Signal-Only F0 diagnostic, Confirm Strategy chunk combos | Yes | Add diagnostic scope metadata so users know whether chunks represent full input or first-N subset | Evidence chunks are provenance, not production scope. |
| `_analysis/phasic_out/phasic_trace_cache.h5` | Required | Phasic analysis path in runner/pipeline | Correction preview, Signal-Only F0 diagnostic, Confirm Strategy | Yes | For product clarity, expose through a diagnostic-cache record instead of calling it a completed run | This is the core cache current consumers need. |
| `_analysis/phasic_out/config_used.yaml` | Required | Phasic analysis path | Correction preview and Signal-Only F0 source validators | Yes | Ensure it is derived from the active Guided/Full Control baseline plus explicit setup overrides | Prevent feature/default drift and hidden config differences. |
| Enough metadata for correction preview comparison | Required | Phasic cache plus config snapshot | `run_guided_correction_preview_comparison()` | Yes | Add `diagnostic_cache` or adapted `phasic_cache` source handling in GUI/backend | Preview writes remain preview-only artifacts. |
| Enough metadata for Signal-Only F0 diagnostic review | Required | Phasic cache plus completed-run resolver | `run_signal_only_f0_diagnostic_review()` | Yes internally, but public API currently wants completed-run source | Add source handling for diagnostic cache or phasic-cache backend path in public Guided API | Avoid pretending diagnostic cache is final completed run. |
| Enough metadata for Confirm Strategy ROI/chunk population | Required | Phasic cache | `MainWindow._refresh_guided_confirm_strategy_panel()` | Yes | Refactor source resolver to read active diagnostic-cache artifact record | Confirm Strategy should be enabled by cache readiness, not completed-run load. |
| Diagnostic cache build request | Required | New Guided state | Future staleness and provenance checks | No | New object/state | Records raw setup state used to launch cache generation. |
| Diagnostic cache artifact record | Required | New Guided state after success | Guided diagnostics, confirm strategy, plan export/provenance | No | New object/state | Holds paths, source signature, scope signature, status, warnings, and artifact contract version. |
| Feature/event profile settings | Future/optional for cache | Full Control active defaults and Guided draft profile | Final plan, feature/event settings step | Not needed for traces-only cache except config provenance | Decide whether cache config stamps event defaults for provenance only | Diagnostic cache should not require final feature/event choices unless backend actually needs them. |

## 4. Diagnostic Namespace and Lifecycle

Recommendation: create a clearly named diagnostic-cache subfolder under the user-selected analysis output base, not inside the raw input folder and not inside a final production run folder.

Recommended shape:

```text
<analysis_output_base>/
  _guided_diagnostic_cache/
    <cache_id>/
      diagnostic_cache_record.json
      config_effective.yaml
      gui_diagnostic_cache_request.json
      command_invoked.txt
      status.json
      run_report.json
      MANIFEST.json
      _analysis/
        phasic_out/
          phasic_trace_cache.h5
          config_used.yaml
```

Naming:

- `cache_id` should be stable enough for provenance but unique enough to avoid accidental overwrite, for example `diagcache_YYYYMMDD_HHMMSS_<short_hash>`.
- The short hash should be derived from the source/staleness signature, not from arbitrary visible widget state alone.
- Do not use "run" in the primary folder name. Avoid "Tuning Prep Run" in Guided-facing labels.

Association with raw input:

- `diagnostic_cache_record.json` should store the resolved raw input path, selected format, acquisition structure fields, included/excluded ROIs, active config source path/hash, diagnostic scope, output base, cache id, creation time, and artifact paths.
- The record should include `purpose: "guided_diagnostic_cache"` and `production_analysis: false`.

Staleness:

- Compute a diagnostic cache signature from the fields listed in Section 5.
- If the current Guided setup signature differs from the artifact record signature, mark the cache stale.
- Stale caches remain inspectable as historical evidence but must not enable new strategy decisions without explicit acknowledgement or rebuild.

Rebuild behavior:

- New build creates a new `cache_id` folder by default.
- Do not overwrite an existing diagnostic cache folder by default.
- A future cleanup action may delete old caches, but deletion should be explicit and out of scope for first implementation.

Retention:

- First implementation should retain caches until user deletion or project cleanup. Temporary-only caches are risky for long-duration work because evidence provenance may vanish between planning sessions.
- If temporary caches are later added, saved plans must clearly indicate that evidence artifacts may be unavailable.

UI distinction from completed production runs:

- Diagnostic cache panels should display "Preliminary diagnostic cache" and "Not final production analysis".
- Completed-run review should remain a separate path.
- Opening a completed production run should not satisfy the new-analysis diagnostic-cache requirement unless the user is explicitly in completed-run review/refinement mode.

## 5. Scope and Performance

First implementation recommendation:

- Process all included ROIs by default.
- Allow a conservative optional "Limit sessions for diagnostic cache" using the existing `preview_first_n` backend support.
- Do not implement arbitrary chunk/window selection or representative time-range selection in the first implementation.
- Record diagnostic scope explicitly in provenance: full selected input versus first N sessions/files, included ROIs, acquisition mode, and continuous window settings.

Tradeoff:

- Full selected input gives stronger evidence but may be slower.
- First-N diagnostic scope is faster and already supported, but it may miss later-session drift, bleaching, artifacts, or ROI behavior changes.
- The UI should not imply that first-N evidence represents the full production scope. It should say:

```text
Diagnostic scope is limited to the first N sessions/files. Use this for faster preliminary review.
Final production analysis scope is defined later by the analysis plan, not by the currently viewed diagnostic chunk.
```

Selected ROIs:

- The diagnostic cache should honor included ROIs from Guided ROI selection.
- Excluded ROIs should be recorded with status `excluded_before_diagnostics` or equivalent.
- Changing the ROI include/exclude set makes the cache stale.

Deferred:

- Arbitrary chunk/window selection.
- Representative time-range selection.
- Stratified diagnostic sampling across long recordings.
- Per-ROI diagnostic scope differences.

These require new backend scope contracts and stronger provenance before they are safe.

## 6. Compatibility and Validation

Format/mode compatibility must come from shared backend-derived validation, not duplicated Guided-only checks.

Current support from audits:

| Format / mode | Diagnostic-cache recommendation | Current support basis | User-facing message |
| --- | --- | --- | --- |
| Intermittent NPM | Allow | Runner and pipeline support NPM in intermittent mode | None beyond normal validation. |
| Intermittent custom_tabular | Allow | Runner and pipeline support custom_tabular in intermittent mode | None beyond normal validation. |
| Continuous custom_tabular | Allow | Continuous planning accepts `custom_tabular` | None beyond normal validation. |
| Continuous RWD | Allow while RWD remains an actual app format | Continuous planning accepts `rwd` | None beyond normal validation. |
| Continuous NPM | Block before launch | Runner and pipeline explicitly reject NPM continuous mode | `Continuous acquisition mode is not yet implemented for NPM/interleaved inputs.` |
| Continuous auto | Block unless validation resolves unambiguously to RWD or custom_tabular | Runner can sniff but rejects ambiguous or NPM-like auto | `Continuous mode with format auto is ambiguous for mixed/unknown inputs. Use format rwd or format custom_tabular.` |

Shared validation should live in a small non-UI helper that can be called by:

- Full Control validate/run preflight.
- Guided diagnostic-cache readiness.
- Future Guided final production readiness.

The helper should return structured status:

- `ok`
- `code`
- `message`
- `resolved_format` when applicable
- `warnings`

Unsupported combinations should disable Build Diagnostic Cache and display the stable message before any subprocess starts.

## 7. Output Safety Policy

Use shared output safety helpers, not one-off Guided checks.

Operations needing policy:

- Full Control production runs.
- Guided diagnostic-cache generation.
- Future Guided final production runs.
- Completed-run review/read-only actions.

Recommended shared concepts:

- `source_root`: raw input or completed-run source being read.
- `output_base`: user-selected base for new writes.
- `target_path`: exact folder to be created or written.
- `operation_kind`: `production_run`, `diagnostic_cache`, `preview_artifact`, `read_only_review`, `plan_export`.
- `allow_existing_target`: false by default for diagnostic cache and production run target folders.
- `protected_roots`: raw input root, completed-run roots, legacy output directories, active production run directories, applied-dF/F roots, preview artifact roots where relevant.

Cases:

- Output base inside input/source: block for production and diagnostic-cache writes.
- Input/source inside output base: block unless the output base is a parent workspace explicitly designed to contain both source and outputs; first implementation should block because provenance and cleanup risk are high.
- Existing output base: allow if it is a directory and target writes go to a unique new subfolder.
- Existing target diagnostic-cache folder: fail by default; create a new cache id instead.
- Existing production run folder: fail by default for future Guided; Full Control currently uses unique run IDs but should share the same guard.
- Overwrite: do not expose in Guided first implementation. Full Control can retain expert overwrite semantics, but the shared helper should make overwrite explicit and auditable.
- Append: do not append diagnostic cache or production outputs into an existing target.
- Version/unique subfolder: prefer unique subfolders for both diagnostic caches and production runs.
- Protected completed-run and legacy output directories: block new diagnostic-cache or production writes inside them unless the operation is a scoped preview artifact that already has a validated preview namespace.
- Completed-run review/read-only: no output target needed; only verify source readability and do not create directories.

The helper should also produce user-facing messages that name the exact path and reason.

## 8. Feature/Event Defaults Policy

Guided should initialize feature/event defaults from the same effective baseline source as Full Control:

- Use the active lab default or custom YAML selected in the setup/config state.
- Reuse the equivalent of `_event_feature_defaults_from_active_baseline()` rather than bare `Config()`.
- If the baseline path is invalid, block readiness before cache generation or plan review instead of silently falling back.

Explicit user edits:

- Edits become live Guided plan state only after an explicit Apply action.
- Store only the explicit config fields used for feature/event settings and record the baseline source/path/hash used to initialize them.
- Record whether each value came from baseline default or explicit user edit if the plan schema supports it.

Reset behavior:

- "Reset to defaults" should reset to the current active baseline, not bare `Config()`.
- If the baseline source changes after a user applied edits, preserve the edits but mark the feature/event profile as needing review or ask for explicit reset/rebase.

Diagnostic-cache dependency:

- The first diagnostic-cache implementation should not depend on final feature/event settings unless the backend requires those fields to build traces/cache.
- Because Tuning Prep is traces-only and skips feature extraction, feature/event settings should remain later plan settings. They may still be stamped into config provenance if the shared config generation path includes them, but strategy evidence should not require final event thresholds.

## 9. Plan State and Execution Boundary

Design-level state objects:

| State | Durable? | Written to disk? | Can affect production analysis? | Notes |
| --- | --- | --- | --- | --- |
| Raw Guided setup state | Should become durable live draft state | Not by itself, unless plan/export/cache request writes it | Yes, only after incorporated into explicit plan/executable config | Includes source, format, acquisition structure, ROI selection, active config source. |
| Diagnostic-cache build request | Durable for provenance | Yes, as `gui_diagnostic_cache_request.json` or equivalent | No direct production effect | Captures exactly what was used to create preliminary cache. |
| Diagnostic-cache artifact record | Durable | Yes, as `diagnostic_cache_record.json` | No direct production effect | Enables evidence tools and staleness checks. |
| Per-ROI correction strategy choices | Durable live draft state | Exportable in future plan; may remain in memory before save | Yes, after final plan-to-execution mapping exists | Must include evidence cache id/chunk provenance and stale status. |
| Feature/event settings | Durable live draft state after Apply | Exportable in future plan | Yes, after final plan-to-execution mapping exists | Defaults from active baseline, edits explicit. |
| Output policy | Durable live draft state after Apply | Exportable in future plan | Yes, defines future production destination | Planning state only until Run. |
| GuidedRunPlan or successor new_analysis plan | Durable planning object | Optional export; future provenance | Not executable by itself | Should represent raw source, cache evidence, strategies, feature/event, output policy. |
| Executable RunSpec or equivalent | Durable execution config | Yes at final Run boundary | Yes | Generated only in future explicit Run/readiness stage. |
| Final production Run action | N/A | Writes production outputs | Yes | Must be explicit; not triggered by diagnostics or review. |
| Completed-run review | Read-only review state | No new writes except explicit preview/diagnostic artifacts | No | Secondary workflow only. |

Visible-widget rule:

- Visible ROI, selected chunk, and current preview panel state must never define production scope.
- Evidence chunk/window selections are provenance for strategy choices only.
- Final production execution must come from explicit plan state translated to executable configuration, not from whichever widget is visible.

## 10. Existing Tool Reuse

Recommended reuse path:

- Add or adapt a source abstraction for diagnostic caches.
- Internally, a diagnostic cache can expose `phasic_out`, `phasic_trace_cache_path`, `config_path`, source provenance, and cache id.
- Existing correction preview code can likely reuse the direct `phasic_cache` path if the GUI supplies `preview_output_dir` and a validated diagnostic-cache source record.
- Signal-Only F0 diagnostic review currently exposes completed-run as the public source. It should gain a diagnostic-cache or phasic-cache source path rather than requiring the GUI to pretend the cache is a completed production run.
- Confirm Strategy should use a shared source resolver that can list ROIs/chunks from an active diagnostic cache record.

Source type recommendation:

- User-facing Guided state should call it a diagnostic cache.
- Backend source handling should support `diagnostic_cache` as a first-class source type or adapter that resolves to phasic-cache artifacts plus diagnostic metadata.
- Do not rely solely on `completed_run` source handling for Guided new-analysis diagnostics. That would recreate Model C semantics under a different label.

Evidence provenance:

- Every preview or diagnostic artifact should reference the diagnostic cache id, source signature, ROI, chunk/window, method, and config path/hash.
- Strategy choices should store evidence references to the cache and selected chunk/window.
- If cache becomes stale, strategy choices should be marked stale until reconfirmed against a current cache.

Completed-run review step:

- Completed-run review remains separate.
- It may continue to use existing completed-run consumers.
- It should not be the source of first-pass new-analysis strategy planning unless the user explicitly chooses a rerun/refinement workflow.

## 11. Saved-Plan Import/Adoption Boundary

Do not resume saved-plan restore/adoption work as part of diagnostic-cache implementation.

Future saved plans may reference diagnostic caches, but the reference should be evidence provenance, not a hard requirement to run final analysis.

Rules:

- A saved new-analysis plan may include diagnostic cache id, path, artifact hashes, and evidence references.
- Diagnostic cache paths may not be portable across machines; imported plans should treat missing caches as missing evidence, not as invalid production configuration.
- Cache artifacts are required to review the original evidence and reconfirm stale choices; they should not be required to run final production analysis if the plan already contains explicit strategies and executable configuration can be built from raw input.
- Saved-plan restore/adoption stays paused until raw Guided setup state, diagnostic-cache state, and new-analysis plan state exist.
- Opening a saved plan must not launch cache generation, build RunSpec, run analysis, or mutate live draft state without explicit restore in a future stage.

## 12. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Diagnostic cache is mistaken for final production run | Use "diagnostic cache" naming, `production_analysis: false` metadata, separate namespace, and status text that final analysis still requires Run. |
| Tuning Prep terminology leaks into Guided workflow | Keep "Tuning Prep" as backend/profile terminology only; use "Build diagnostic cache" or "Prepare correction diagnostics" in Guided. |
| Completed-run-centered drift returns under another name | Require cache creation from raw Guided setup for new-analysis flow; keep completed-run sources in a separate review/refinement branch. |
| Cache staleness creates invalid strategy decisions | Signature all relevant setup/config/scope fields; disable new decisions on stale cache; preserve old decisions as stale evidence until reconfirmed. |
| Feature/event defaults drift from Full Control | Reuse active baseline/default config helpers; record config source/hash; avoid bare `Config()` defaults for Guided editor initialization. |
| Output safety differs between Full Control and Guided | Implement shared output safety helper before Guided writes diagnostic caches. |
| Continuous unsupported modes fail too late | Shared compatibility validation disables Build Diagnostic Cache before subprocess launch and uses stable backend messages. |
| Diagnostic scope is too small to support correction decisions | Default to all selected input; if first-N is used, label it as limited evidence and record scope in provenance. |
| Diagnostic cache writes too much or is too slow | Use Tuning Prep traces-only semantics, selected ROIs, optional first-N; defer arbitrary windows until backend scope contract exists. |
| Future final Run accidentally consumes widget state instead of plan state | Define explicit new-analysis plan and later plan-to-RunSpec boundary; final Run reads plan/executable config only. |
| Diagnostic cache folder is overwritten or mixed with production output | Unique cache id folders, no append, no overwrite by default, separate `_guided_diagnostic_cache` namespace. |
| Signal-Only F0 API forces completed-run semantics | Add diagnostic-cache/phasic-cache source handling rather than routing through completed-run labels. |

## 13. Recommended Staged Roadmap

1. 4J1: shared output safety and compatibility validation helpers.
   - Classification: core workflow.
   - Small first implementation because it writes no new diagnostic cache and can be tested around pure path/format/mode decisions.

2. 4J2: diagnostic-cache state model and artifact contract.
   - Classification: core workflow.
   - Define build request and artifact record objects, signatures, status values, and staleness rules without launching the runner.

3. 4J3: Guided Build Diagnostic Cache action using Tuning Prep semantics.
   - Classification: core workflow.
   - Launch from raw setup state into `_guided_diagnostic_cache/<cache_id>`, write request/record/provenance, and display preliminary status. No final Run.

4. 4J4: allow correction preview, Signal-Only F0, and Confirm Strategy to consume diagnostic-cache source.
   - Classification: core workflow.
   - Add source adapter/resolver and keep completed-run review separate.

5. 4J5: new_analysis Guided plan state from raw setup plus diagnostic evidence.
   - Classification: core workflow.
   - Store raw source, acquisition structure, ROI inclusion, diagnostic evidence references, correction choices, feature/event settings, and output policy.

6. 4J6: feature/event defaults alignment with Full Control baseline.
   - Classification: core workflow.
   - Reuse active baseline defaults in Guided, record baseline provenance, and test reset/edit behavior.

7. 4J7: plan-to-RunSpec design.
   - Classification: core workflow.
   - Design first; implementation should wait until plan state and diagnostic evidence boundaries are stable.

8. Saved-plan restore/adoption.
   - Classification: defer.
   - Resume only after the core new-analysis path exists.

9. Model A raw-input preview research.
   - Classification: defer.
   - Consider only if backend can reuse pipeline loading/correction semantics without duplicating substantial logic.

10. Diagnostic cache cleanup/retention management.
   - Classification: secondary utility.
   - Add after cache generation and provenance behavior are accepted.

## 14. Open Questions for Jeff

1. Should the first Guided diagnostic cache default to full selected input, or should the UI offer first-N scope immediately?
2. Should Guided allow users to choose a separate diagnostic-cache root, or should caches always live under the selected analysis output base?
3. Should excluded ROIs require a reason/comment before diagnostic-cache generation?
4. Should a stale strategy choice be blocked from final plan readiness, or allowed with an explicit warning and reconfirmation requirement?
5. Should completed-run refinement use the same diagnostic-cache source abstraction later, or remain a separate completed-run branch?

## Commands Run

- `Get-Content -Raw C:\Users\Jeff\Desktop\task69.txt`
- `git status --short`
- `Get-Content -Raw guided_diagnostic_cache_followup.md`
- `Get-Content -Raw guided_full_contract_wiring_audit.md`
- `Get-Content -Raw C:\Users\Jeff\Desktop\guided_full_app_workflow_contract_draft.md`

No runtime code was changed and no tests were run because this was design-only.
