import subprocess
import re
import numpy as np


def get_least_used_gpu():
    """Get the least used GPU device."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
        )
        gpu_memory_used = [int(x) for x in result.stdout.decode("utf-8").strip().split("\n")]
        return np.argmin(gpu_memory_used)
    except Exception:
        return -1