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

    def __call__(self, obj):
        text = str(obj)
        print(text)
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def run_dir(self, scenario_name):
        """Return a timestamped run directory for a training scenario.

        Creates: <base_dir>/<scenario_name>/<init_time>/
        """
        d = os.path.join(self.base_dir, scenario_name, self.init_time)
        os.makedirs(d, exist_ok=True)
        return d
