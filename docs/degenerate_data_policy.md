# Degenerate Data Policy

This policy defines how the photometry pipeline behaves when numerical assumptions fail, ensuring deterministic outputs and explicit Quality Control (QC) markers instead of silent NumPy runtime warnings.

| Case ID | Condition | Component | Behavior | QC Emission |
| :--- | :--- | :--- | :--- | :--- |
| **DD1** | Fewer than 2 finite samples for regression fit (per ROI, per window) | `regression.py` | Skip the window for this ROI (do not append to valid stats). | `chunk.metadata.setdefault('qc_warnings', []).append("DEGENERATE[DD1] ...")` |
| **DD2** | Denominator unsafe for slope computation (`var_u` non-finite or &lt;= 1e-12) | `regression.py` | Skip the window for this ROI (do not append to valid stats). | `chunk.metadata.setdefault('qc_warnings', []).append("DEGENERATE[DD2] ...")` |
| **DD3** | All-NaN or < 2 finite samples for percentile/quantile/median operations | `feature_extraction.py` | Return a null row (mean/median/std/mad = NaN, peak_count = 0, auc = NaN) and safely continue. | `chunk.metadata.setdefault('qc_warnings', []).append("DEGENERATE[DD3] ...")` |
| **DD4** | Flatline / zero-variability trace used in peak thresholding (`sigma` or `mad` == 0) | `feature_extraction.py` | If threshold method relies on variance (e.g., `mean_std`, `median_mad`), set threshold to `inf` (no peaks). | `chunk.metadata.setdefault('qc_warnings', []).append("DEGENERATE[DD4] ...")` |
| **DD5** | Segment/window becomes empty or single-element after NaN masking | `feature_extraction.py` (Segment Iteration) | Skip evaluating peaks and AUC for the degenerate segment entirely. | `chunk.metadata.setdefault('qc_warnings', []).append("DEGENERATE[DD5] ...")` |
