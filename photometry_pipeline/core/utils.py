import re
import os
import logging
from typing import Tuple, Any

_warned_natural_sort = False

def natural_sort_key(path: str) -> Tuple[Any, ...]:
    """
    Extracts the LAST integer group from the filename stem for sorting.
    Falls back to lexicographic string comparison if no integer is found,
    emitting a warning once per run.
    """
    global _warned_natural_sort
    
    stem = os.path.splitext(os.path.basename(path))[0]
    
    # Extract all integer groups
    matches = list(re.finditer(r'\d+', stem))
    
    if matches:
        # Use the last integer found in the stem
        last_match = matches[-1]
        num = int(last_match.group())
        # Return a tuple that sorts by the extracted integer, then the full original string to break ties
        return (0, num, path)
    else:
        if not _warned_natural_sort:
            logging.warning(f"natural_sort_key: No integer found in '{stem}'. Falling back to lexicographical sort. (This warning is printed once)")
            _warned_natural_sort = True
        return (1, path)
