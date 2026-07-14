# B2-A intermittent NPM characterization fixtures

These are small, deterministic flat CSV files in the vendor-style NPM shape
currently consumed by `photometry_pipeline.io.adapters._load_npm`:

- `Timestamp` is the configured system timestamp column;
- `LedState=1` is the UV/control sample and `LedState=2` is the signal sample;
- `Region*G` columns contain the time-multiplexed ROI values;
- filenames contain the vendor timestamp fragment used by `sort_npm_files`.

The `basic` filenames intentionally use prefixes whose lexical order differs
from the embedded timestamp order. `multi_roi` covers two independent ROI
columns. `malformed` contains a missing-signal case that must be rejected by
the current strict/permissive loader contract.
