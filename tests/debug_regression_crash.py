
import numpy as np
import traceback
from photometry_pipeline.config import Config
from photometry_pipeline.core.types import Chunk
from photometry_pipeline.core.regression import fit_chunk_dynamic

def run_debug():
    try:
        config = Config()
        config.target_fs_hz = 40.0
        config.chunk_duration_sec = 10.0
        config.window_sec = 2.0
        config.step_sec = 1.0
        
        n = 400
        t = np.arange(n)/40.0
        uv = np.random.randn(n, 1)
        sig = uv * 2.0 + 1.0
        
        chunk = Chunk(0, "dummy", "rwd", t, uv, sig, 40.0, ["Region0"], {})
        chunk.uv_filt = uv.copy()
        chunk.sig_filt = sig.copy()
        
        # Inject NaNs
        chunk.uv_filt[0:50, 0] = np.nan
        
        print("Starting fit...")
        fit_chunk_dynamic(chunk, config)
        print("Fit complete.")
    except:
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()
