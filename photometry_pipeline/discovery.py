import os
import glob
import pathlib
from typing import List, Optional, Dict, Any

from photometry_pipeline.config import Config
from photometry_pipeline.io.adapters import sniff_format, load_chunk
from photometry_pipeline.core.utils import natural_sort_key


def _session_entry_to_id(entry: str) -> str:
    """Returns a stable session ID for a file path or RWD folder."""
    p = pathlib.Path(entry)
    if p.name == "fluorescence.csv":
        return p.parent.name
    return p.stem


def discover_inputs(
    input_dir: str, 
    config: Config, 
    force_format: str = "auto", 
    preview_first_n: Optional[int] = None
) -> Dict[str, Any]:
    """
    Discover sessions and ROIs using the same resolution logic as the pipeline.
    Does not mutate pipeline state, emit events, or write artifacts.
    """
    if not os.path.exists(input_dir):
        raise ValueError(f"Input path does not exist: {input_dir}")

    # 1. Discover physical files/directories
    file_list = []
    resolved_format = force_format
    
    if force_format == 'rwd':
        from photometry_pipeline.io.adapters import discover_rwd_chunks
        file_list = discover_rwd_chunks(input_dir)
    elif os.path.isfile(input_dir):
        file_list = [input_dir]
        if resolved_format == 'auto':
            resolved_format = sniff_format(input_dir, config)
    else:
        # Default behavior for non-recursive pipeline search is *.csv
        search_pattern = os.path.join(input_dir, "*.csv")
        file_list = glob.glob(search_pattern)
        
    file_list.sort(key=natural_sort_key)
    
    if not file_list:
        raise ValueError(f"No files found in {input_dir}")

    # Resolve format if auto and we have a list of files
    if resolved_format == 'auto':
        resolved_format = sniff_format(file_list[0], config)
        if resolved_format is None:
            raise ValueError(f"Could not automatically detect format for {file_list[0]}")

    n_total_discovered = len(file_list)

    # 2. Build Sessions list
    sessions = []
    for idx, fpath in enumerate(file_list):
        session_id = _session_entry_to_id(fpath)
        included = True
        if preview_first_n is not None:
            included = (idx < preview_first_n)
            
        sessions.append({
            "index": idx,
            "session_id": session_id,
            "path": os.path.abspath(fpath),
            "included_in_preview": included
        })

    # 3. Discover ROIs
    # The pipeline uses the intersection of channels across all valid chunks.
    # To avoid loading all chunks in discovery (which might take too long), we load only chunks included in preview,
    # or just the first few chunks if no preview. Since the pipeline throws on missing ROIs later anyway,
    # reading the first valid chunk's header is sufficient for GUI discovery.
    
    channels_seen = []
    # Only sniff up to 5 chunks to avoid expensive reads over network drives
    max_sniff = min(len(file_list), 5)
    
    for i in range(max_sniff):
        fpath = file_list[i]
        try:
            # We enforce chunk_duration_sec validation but we just want headers.
            # RWD and NPM adapters read the header fast.
            chunk = load_chunk(fpath, resolved_format, config, chunk_id=i)
            channels_seen.append(chunk.channel_names)
        except Exception as e:
            print(f"DEBUG load_chunk {fpath} failed: {type(e).__name__}: {e}")
            continue
            
    if not channels_seen:
        raise RuntimeError("No valid data files could be parsed to discover ROIs.")
        
    channel_sets = [set(cx) for cx in channels_seen]
    discovered_rois = [r for r in channels_seen[0] if all(r in cs for cs in channel_sets)]
    
    rois = []
    for roi_id in discovered_rois:
        rois.append({
            "roi_id": roi_id,
            "label": roi_id  # UI can augment this later, but default to canonical ID
        })

    n_preview = sum(1 for s in sessions if s["included_in_preview"])

    return {
        "schema_version": 1,
        "input_dir": os.path.abspath(input_dir),
        "resolved_format": resolved_format,
        "sessions": sessions,
        "n_total_discovered": n_total_discovered,
        "preview_first_n": preview_first_n,
        "n_preview": n_preview,
        "rois": rois
    }
