
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from ..config import Config
from .types import SessionStats

@dataclass
class DeterministicReservoir:
    """
    Seeded, deterministic reservoir sampler.
    Guarantees identical outputs for identical input sequences.
    """
    seed: int
    capacity: int = 200_000
    
    buffer: Dict[str, np.ndarray] = field(default_factory=dict)
    count: Dict[str, int] = field(default_factory=dict)
    _rng: np.random.Generator = field(init=False)
    
    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        
    def add(self, channel: str, data: np.ndarray):
        if channel not in self.buffer:
            self.buffer[channel] = np.zeros(self.capacity, dtype=np.float32)
            self.count[channel] = 0
            
        data = np.asarray(data)
        data = data[np.isfinite(data)]
        n = len(data)
        if n == 0: return
        
        current = self.count[channel]
        
        if current < self.capacity:
            take = min(n, self.capacity - current)
            self.buffer[channel][current:current+take] = data[:take]
            self.count[channel] += take
            
            remaining = data[take:]
            if len(remaining) > 0:
                self._update_existing(channel, remaining)
        else:
            self._update_existing(channel, data)
            
    def _update_existing(self, channel: str, data: np.ndarray):
        total_seen = self.count[channel]
        n_new = len(data)
        
        probs = self._rng.random(n_new)
        denominators = np.arange(total_seen + 1, total_seen + n_new + 1)
        threshs = self.capacity / denominators
        
        mask = probs < threshs
        n_replace = np.sum(mask)
        
        if n_replace > 0:
            replace_indices = self._rng.integers(0, self.capacity, size=n_replace)
            vals_to_insert = data[mask]
            self.buffer[channel][replace_indices] = vals_to_insert
            
        self.count[channel] += n_new

    def get_percentile(self, channel: str, p: float) -> float:
        if channel not in self.buffer or self.count[channel] == 0:
            return np.nan
        valid = min(self.count[channel], self.capacity)
        return np.percentile(self.buffer[channel][:valid], p)

@dataclass
class GlobalFitAccumulator:
    """
    Accumulates stats for Method B Pass 1a:
    Sum_uv, Sum_sig, Sum_uv^2, Sum_uv*sig, n
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
        valid = np.isfinite(uv) & np.isfinite(sig)
        u_clean = uv[valid]
        s_clean = sig[valid]
        
        s['n'] += len(u_clean)
        s['sum_u'] += np.sum(u_clean)
        s['sum_s'] += np.sum(s_clean)
        s['sum_uu'] += np.sum(u_clean**2)
        s['sum_us'] += np.sum(u_clean * s_clean)

    def solve(self) -> Dict[str, Dict[str, float]]:
        results = {}
        for ch, s in self.stats.items():
            n = s['n']
            if n < 2:
                results[ch] = {'a': 1.0, 'b': 0.0}
                continue
                
            num = n * s['sum_us'] - s['sum_u'] * s['sum_s']
            den = n * s['sum_uu'] - s['sum_u']**2
            
            if abs(den) < 1e-9:
                a = 0.0 
            else:
                a = num / den
                
            b = (s['sum_s'] - a * s['sum_u']) / n
            results[ch] = {'a': a, 'b': b}
            
        return results
