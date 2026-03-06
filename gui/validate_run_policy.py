"""
validate_run_policy.py

Pure logic for deciding whether a "Run" attempt is consistent with 
a prior "Validate" success.

The policy as of Fix B1v4:
1. Validate and Run use DIFFERENT directories.
2. Run is only allowed if settings are identical to a prior successful Validate.
3. Signature matching is the mechanism for this consistency check.
"""

import hashlib
import os
import json

def compute_run_signature(run_dir: str) -> str:
    """
    Compute a SHA256 signature of the user intent artifacts in run_dir.
    
    Includes:
    - gui_run_spec.json (stable intent fields only)
    - config_effective.yaml
    
    Returns hex signature.
    Raises FileNotFoundError if artifacts are missing.
    """
    spec_path = os.path.join(run_dir, "gui_run_spec.json")
    config_path = os.path.join(run_dir, "config_effective.yaml")
    
    if not os.path.isfile(spec_path):
        raise FileNotFoundError(f"Missing artifact: {spec_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing artifact: {config_path}")
        
    with open(spec_path, "r", encoding="utf-8") as f:
        spec_data = json.load(f)
    
    # Filter out dynamic/metadata fields to ensure signature is stable 
    # for identical user intent regardless of timing or directory name.
    STABLE_FIELDS = {
        "input_dir", "format", "sessions_per_hour", "session_duration_s",
        "smooth_window_s", "config_source_path", "config_overrides",
        "mode", "traces_only", "preview_first_n", "representative_session_index",
        "include_roi_ids", "exclude_roi_ids"
    }
    stable_spec = {k: v for k, v in spec_data.items() if k in STABLE_FIELDS}
    spec_bytes = json.dumps(stable_spec, sort_keys=True).encode("utf-8")

    with open(config_path, "rb") as f:
        config_bytes = f.read()
        
    hasher = hashlib.sha256()
    hasher.update(spec_bytes)
    hasher.update(b"\n---\n")
    hasher.update(config_bytes)
    return hasher.hexdigest()

def is_validation_current(
    validated_signature: str | None,
    current_run_signature: str
) -> bool:
    """
    Check if the current run intent matches the last successful validation.
    """
    if not validated_signature:
        return False
    return validated_signature == current_run_signature
