import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from ..config import Config
from .types import SessionStats

@dataclass
class ReservoirSampler:
    """
    Fixed-size reservoir sampling for session-wide percentile estimation.
    """
    capacity: int = 200_000 # Enough for ~1.5h at 40Hz.
    # If session is longer, we downsample representation.
    # For median/percentile, 200k points is plenty for statistical stability.
    
    buffer: Dict[str, np.ndarray] = field(default_factory=dict) # channel -> array
    count: Dict[str, int] = field(default_factory=dict)
    
    def add(self, channel: str, data: np.ndarray):
        """Streaming reservoir update."""
        if channel not in self.buffer:
            self.buffer[channel] = np.zeros(self.capacity, dtype=np.float32)
            self.count[channel] = 0
            
        n = len(data)
        if n == 0:
            return
            
        current_count = self.count[channel]
        
        # If buffer not full, fill it
        if current_count < self.capacity:
            available = self.capacity - current_count
            take = min(n, available)
            self.buffer[channel][current_count : current_count+take] = data[:take]
            self.count[channel] += take
            
            # If we still have data left, handle reservoir logic
            remaining_data = data[take:]
            remaining_n = len(remaining_data)
            if remaining_n > 0:
                self._reservoir_update(channel, remaining_data, current_count + take)
        else:
             self._reservoir_update(channel, data, current_count)

    def _reservoir_update(self, channel: str, data: np.ndarray, total_seen: int):
        # Algorithm L or simple replacement
        # Simple R: for each new item i at index k (where k is global index),
        # probability of keeping is capacity/k.
        # Vectorized implementation: generate random indices for all new items?
        # Efficient approach for large blocks:
        # for each item in data:
        #   m = total_seen + i + 1
        #   if random < capacity/m:
        #       replace random slot
        
        # This is slow in python loops. Vectorized approx:
        # Generate randoms for all new items
        n = len(data)
        # We need to process sequentially to respect probability? 
        # Actually for global distribution, order doesn't matter much.
        # But let's be rigorous.
        
        # We can just randomly select which specific items from 'data' to keep?
        # No, that's not quite reservoir sampling.
        # Correct way for block updates:
        
        start_n = total_seen
        # Generate probabilities for each item being included: P = capacity / (start_n + i + 1)
        # Logic:
        # 1. Decide how many items from this block enter the reservoir.
        # This is hyper-geometric? Too complex.
        
        # Simple loop is safest:
        for val in data:
            start_n += 1
            if np.random.rand() < (self.capacity / start_n):
                idx = np.random.randint(0, self.capacity)
                self.buffer[channel][idx] = val
        
        self.count[channel] = start_n

    def get_percentile(self, channel: str, p: float) -> float:
        if channel not in self.buffer or self.count[channel] == 0:
            return np.nan
        
        n_valid = min(self.count[channel], self.capacity)
        view = self.buffer[channel][:n_valid]
        return np.percentile(view, p)

@dataclass
class GlobalFitAccumulator:
    """
    Accumulates stats for Method B Pass 1a:
    Σuv, Σsig, Σuv², Σuv·sig, n
    """
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict) # channel -> stats
    
    def add(self, channel: str, uv: np.ndarray, sig: np.ndarray):
        if channel not in self.stats:
            self.stats[channel] = {
                'sum_u': 0.0, 'sum_s': 0.0, 
                'sum_uu': 0.0, 'sum_us': 0.0, 
                'n': 0
            }
        
        s = self.stats[channel]
        # Ignore NaNs?
        valid = np.isfinite(uv) & np.isfinite(sig)
        u_clean = uv[valid]
        s_clean = sig[valid]
        
        s['n'] += len(u_clean)
        s['sum_u'] += np.sum(u_clean)
        s['sum_s'] += np.sum(s_clean)
        s['sum_uu'] += np.sum(u_clean**2)
        s['sum_us'] += np.sum(u_clean * s_clean)

    def solve(self) -> Dict[str, Dict[str, float]]:
        # OLS: a = (n*Σxy - ΣxΣy) / (n*Σxx - (Σx)^2)
        # b = (Σy - aΣx)/n
        results = {}
        for ch, s in self.stats.items():
            n = s['n']
            if n < 2:
                results[ch] = {'a': 1.0, 'b': 0.0}
                continue
                
            num = n * s['sum_us'] - s['sum_u'] * s['sum_s']
            den = n * s['sum_uu'] - s['sum_u']**2
            
            if abs(den) < 1e-9:
                a = 0.0 # Vertical or singular
            else:
                a = num / den
                
            b = (s['sum_s'] - a * s['sum_u']) / n
            results[ch] = {'a': a, 'b': b}
            
        return results

