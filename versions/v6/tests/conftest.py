"""Session-level test configuration.

1. Ensures gpu_solver and gpu_solver_v2 are importable by adding
   agent/gto_utils/ to sys.path — no sys.modules stubs needed.
2. Seeds all RNGs before every test for full determinism.
"""
import sys
import os
import random

import numpy as np
import torch
import pytest

_gto_utils = os.path.join(os.path.dirname(__file__), "..", "agent", "gto_utils")
if _gto_utils not in sys.path:
    sys.path.insert(0, _gto_utils)

_project_root = os.path.join(os.path.dirname(__file__), "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


@pytest.fixture(autouse=True)
def _seed_rngs():
    """Reset all RNGs to a fixed seed before every test."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
