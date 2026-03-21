"""
Session and ROI Discovery.

Reuses the exact same file-discovery, format-sniffing, and ROI-intersection
logic used by Pipeline.discover_files() and Pipeline.run(), so the GUI
preview always matches what a real run will process.

This module MUST NOT write any files, emit events, or create directories.
"""

import os
import sys
import glob
import logging
import pathlib
from typing import Optional, Dict, Any

from photometry_pipeline.config import Config
from photometry_pipeline.io.adapters import sniff_format, load_chunk
from photometry_pipeline.core.utils import natural_sort_key


def _session_entry_to_id(entry: str) -> str:
    """Returns a stable session ID for a file path or RWD folder.

    Mirrors Pipeline._session_entry_to_id exactly.
    """
    p = pathlib.Path(entry)
    if p.name == "fluorescence.csv":
        return p.parent.name
    return p.stem


def discover_inputs(
    input_dir: str,
    config: Config,
    force_format: str = "auto",
    preview_first_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Discover sessions and ROIs using the same resolution logic as the pipeline.

    Mirrors Pipeline.discover_files() for session ordering and
    Pipeline.run() for ROI intersection.  Does not mutate pipeline
    state, emit events, or write artifacts.

    Returns a dict matching the strict discovery JSON schema (v1).
    """
    if not os.path.exists(input_dir):
        raise ValueError(f"Input path does not exist: {input_dir}")

    # ------------------------------------------------------------------
    # 1. File discovery  (mirrors Pipeline.discover_files)
    # ------------------------------------------------------------------
    file_list: list = []
    resolved_format = force_format.lower() if force_format else "auto"

    if resolved_format == "rwd":
        from photometry_pipeline.io.adapters import discover_rwd_chunks
        file_list = discover_rwd_chunks(input_dir)
    elif os.path.isfile(input_dir):
        file_list = [input_dir]
        if resolved_format == "auto":
            resolved_format = sniff_format(input_dir, config)
    else:
        # Default non-recursive *.csv search — same as Pipeline.discover_files
        search_pattern = os.path.join(input_dir, "*.csv")
        file_list = glob.glob(search_pattern)
        if resolved_format == "auto" and not file_list:
            from photometry_pipeline.io.adapters import discover_csv_or_rwd_chunks
            file_list = discover_csv_or_rwd_chunks(input_dir, file_glob="*.csv")

    file_list.sort(key=natural_sort_key)

    if not file_list:
        raise ValueError(f"No files found in {input_dir}")

    # ------------------------------------------------------------------
    # 2. Format resolution  (mirrors Pipeline._get_format)
    # ------------------------------------------------------------------
    if resolved_format == "auto":
        resolved_format = sniff_format(file_list[0], config)
        if resolved_format is None:
            raise ValueError(
                f"Could not automatically detect format for {file_list[0]}"
            )

    n_total_discovered = len(file_list)

    # ------------------------------------------------------------------
    # 3. Build sessions list (strict schema)
    # ------------------------------------------------------------------
    sessions = []
    for idx, fpath in enumerate(file_list):
        session_id = _session_entry_to_id(fpath)
        included = True
        if preview_first_n is not None:
            included = idx < preview_first_n

        sessions.append({
            "index": idx,
            "session_id": session_id,
            "path": os.path.abspath(fpath),
            "included_in_preview": included,
        })

    # ------------------------------------------------------------------
    # 4. ROI discovery  (mirrors Pipeline.run lines 664-679)
    # ------------------------------------------------------------------
    # Load a few chunks and intersect channel names, preserving discovered
    # order from the first valid chunk — identical to Pipeline.run().
    channels_seen: list = []
    max_sniff = min(len(file_list), 5)

    for i in range(max_sniff):
        fpath = file_list[i]
        try:
            chunk = load_chunk(fpath, resolved_format, config, chunk_id=i)
            channels_seen.append(chunk.channel_names)
        except Exception:
            logging.debug("discover_inputs: load_chunk %s failed", fpath,
                          exc_info=True)
            continue

    if not channels_seen:
        raise RuntimeError(
            "No valid data files could be parsed to discover ROIs."
        )

    channel_sets = [set(cx) for cx in channels_seen]
    discovered_rois = [
        r for r in channels_seen[0] if all(r in cs for cs in channel_sets)
    ]

    rois = [{"roi_id": roi_id, "label": roi_id} for roi_id in discovered_rois]
    n_preview = sum(1 for s in sessions if s["included_in_preview"])

    # ------------------------------------------------------------------
    # 5. Assemble strict JSON payload
    # ------------------------------------------------------------------
    return {
        "schema_version": 1,
        "input_dir": os.path.abspath(input_dir),
        "resolved_format": resolved_format.upper(),
        "sessions": sessions,
        "n_total_discovered": n_total_discovered,
        "preview_first_n": preview_first_n,
        "n_preview": n_preview,
        "rois": rois,
    }
