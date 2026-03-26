# Dynamic Fit Rolling-Regression Contract

## Scope
This note defines the active dynamic-fit contract used by the production pipeline and GUI.
It is an internal stabilization reference for regression safety, not a design proposal.

## Active Algorithm
- Inputs for fitting: lowpass-filtered `sig_filt` and `uv_filt`.
- Engine: centered rolling local linear regression across the chunk.
- Outputs per sample: dense time-varying slope/intercept `(a(t), b(t))`.
- Fitted reference: `uv_fit(t) = a(t) * uv_raw(t) + b(t)`.
- Fallback: if rolling windows cannot produce finite local parameters, a global fit fallback is used for that ROI/chunk.

## Boundary Contract
- Fit generation boundary: `uv_fit` is generated in the regression layer.
- Canonical numerator assembly: `delta_f = sig_raw - uv_fit` in one place.
- Denominator policy: unchanged from existing baseline/F0 path.
- dF/F path: unchanged downstream of fit replacement (`dff = 100 * delta_f / F0` where valid).
- Downstream consumers continue to read: `uv_fit`, `delta_f`, `dff`.

## Controls
### Active controls
- `lowpass_hz` (through preprocessing into `uv_filt/sig_filt`)
- `window_sec`
- `min_samples_per_window`

### Legacy/inactive controls (compatibility only)
- `step_sec`
- `min_valid_windows`
- `r_low`
- `r_high`
- `g_min`

These legacy fields remain in config/metadata for backward compatibility, but the active rolling-fit engine does not use them.

## Format Notes
- Supported: both `RWD` and `NPM` on the active path.
- NPM strict/admission and plotting contracts were aligned earlier; this note only records dynamic-fit behavior at the correction boundary.
- RWD and NPM share the same rolling-fit semantics once chunked/resampled data reaches regression.

## QC / Diagnostics Outputs
- Correction-impact figure remains a 4-panel diagnostic:
  - raw absolute sig/iso
  - centered common-gain sig/iso
  - dynamic fit panel
  - final corrected dF/F
- Correction retune inspection remains a 4-image navigable set in canonical order:
  - `raw`, `centered`, `fit`, `dff`
- Sig/iso centered panel semantics remain median-centered common-gain display (no per-trace amplitude equalization).

## Legacy Telemetry Note
- Some timing bucket names still include legacy window-gating terminology for backward-compatible metric parsing.
- Under the active rolling engine these legacy-gating buckets are expected to remain zero or near-zero.
