import os
from datetime import datetime

import torch


def get_amp_config(device):
    """Return AMP configuration for the given device.

    On CUDA we prefer bfloat16 (same exponent range as fp32 → no overflow
    in deep transformers; this fixes `inf`/`nan` logits that previously
    poisoned opponent-data generation through softmax). bf16 also doesn't
    need a GradScaler since it can't underflow gradients in any range
    fp32 reaches. Older GPUs that don't support bf16 fall back to fp16
    with a scaler, matching the previous behaviour exactly.

    Returns:
        (amp_enabled, device_type, amp_dtype, use_scaler)
    """
    device_str = str(device)
    if device_str.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return True, "cuda", torch.bfloat16, False
        return True, "cuda", torch.float16, True
    elif device_str == "mps":
        return True, "mps", torch.bfloat16, False
    else:
        return False, "cpu", torch.float32, False

def resolve_device(requested="auto"):
    """Resolve the compute device once, so it can be passed down.

    `CLAUDE.md` §3: no hardcoded `.cuda()`, no assumed CUDA. "auto" walks
    CUDA → MPS → CPU; anything else is taken literally so a run can be pinned
    to CPU on the execution box.
    """
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Logger:
    def __init__(self, base_dir):
        """
        Args:
            base_dir: root directory for this experiment, e.g. data/v0/my_exp
        """
        self.init_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.base_dir = base_dir

        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self.filename = os.path.join(logs_dir, f"{self.init_time}.txt")
        self._fh = open(self.filename, "a", encoding="utf-8")

    def __call__(self, obj):
        text = str(obj)
        print(text)
        self._fh.write(text + "\n")
        self._fh.flush()

    def close(self):
        """Close the underlying log file handle."""
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __del__(self):
        self.close()

    def run_dir(self, scenario_name):
        """Return a timestamped run directory for a training scenario.

        Creates: <base_dir>/<scenario_name>/<init_time>/
        """
        d = os.path.join(self.base_dir, scenario_name, self.init_time)
        os.makedirs(d, exist_ok=True)
        return d
