import os
from typing import List
from photometry_pipeline.io.adapters import (
    discover_csv_or_rwd_chunks,
    sniff_format,
    sort_npm_files
)
from photometry_pipeline.core.utils import natural_sort_key

def map_cached_sources_to_schedule_positions(
    raw_input_dir: str, 
    fmt: str, 
    cached_source_files: List[str],
    cids: List[int]
) -> List[int]:
    """
    Recover actual physical chunk indices by mapping the stored trace cache 
    source references back onto the authoritative timeline file discovery.
    
    Arguments:
        raw_input_dir: The dataset directory string.
        fmt: 'rwd', 'npm', or 'auto'.
        cached_source_files: The decoded string paths recovered from cache meta 'source_files'.
        cids: The sequential chunk_ids matched against the cached_source_files length.
        
    Returns:
        List of integers representing the actual corrected schedule positions.
        Fallback to `cids[i]` for true lookup misses.
    """
    if not raw_input_dir:
        return list(cids)
        
    file_list = discover_csv_or_rwd_chunks(raw_input_dir)
    if not file_list:
        return list(cids)
        
    resolved_fmt = fmt if (fmt and fmt != 'auto') else sniff_format(file_list[0], None)
    if resolved_fmt == 'npm':
        file_list = sort_npm_files(file_list)
    else:
        file_list.sort(key=natural_sort_key)
        
    def _normalize_path(p: str) -> str:
        return os.path.normcase(os.path.normpath(str(p))) if p else ""
        
    normalized_file_list: List[str] = [_normalize_path(f) for f in file_list]
    
    actual_positions: List[int] = []
    for i, cid in enumerate(cids):
        fpath_cached = cached_source_files[i] if i < len(cached_source_files) else ""
        norm_fpath_cached = _normalize_path(fpath_cached)
        try:
            actual_idx = normalized_file_list.index(norm_fpath_cached)
        except ValueError:
            actual_idx = cid
        actual_positions.append(actual_idx)
        
    return actual_positions
