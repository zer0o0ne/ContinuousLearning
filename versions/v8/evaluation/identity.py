"""Content identities and atomic JSON for resumable evaluations."""

import hashlib
import json
import os
from pathlib import Path

import torch


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_digest(net):
    digest = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, path)


def check_identity(directory, identity):
    """Refuse to combine results from different policies/protocols."""
    directory = Path(directory)
    path = directory / "identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(f"Evaluation identity changed at {directory}; use a new "
                             "evaluation run name. Existing results were kept.")
    else:
        if directory.exists() and (any(directory.rglob("*.jsonl"))
                                   or any(directory.rglob("*.json"))):
            raise ValueError(f"Unversioned evaluation at {directory}; use a new "
                             "run name to keep historical results separate.")
        atomic_json(path, identity)
