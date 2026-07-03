"""gru_window propagation in phase-6 MCTS training (mcts_train.gru_window).

The GRU's truncated-BPTT window was previously hardcoded to the
perception default (1) in phase 6. `train_mcts` now reads
`mcts_train.gru_window` and threads it through `_mcts_forward` /
`_run_validation` into every `perception.forward_batch` call (root +
chain). These tests pin the propagation.

All tests are fully deterministic: fixed seeds, no probabilistic
assertions, no order dependence.

Run from versions/v6/:
    python3 -m pytest tests/test_mcts_gru_window.py -v
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
for _p in (_PKG_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.agent import ASI
from agent.train_scenarios.mcts_predict.train import _mcts_forward, _run_validation

from test_mcts_lm_loss import (
    _TINY_CONFIG, _seq_two_decisions, _chain_events, _chain_step,
)


def _make_agent(seed=0):
    torch.manual_seed(seed)
    agent = ASI(lambda m: None, config=_TINY_CONFIG)
    agent.set_device("cpu")
    return agent


def _record_gru_windows(agent):
    """Wrap perception.forward_batch to record the gru_window kwarg of
    every call while preserving behavior."""
    seen = []
    orig = agent.perception.forward_batch

    def wrapper(*args, **kwargs):
        seen.append(kwargs.get("gru_window", 1))
        return orig(*args, **kwargs)

    agent.perception.forward_batch = wrapper
    return seen


def _batch_with_chain():
    event_sequences = [_seq_two_decisions()]
    chains = [[_chain_step(action_taken=1, events=_chain_events())]]
    return event_sequences, chains


def test_mcts_forward_propagates_gru_window_to_all_perception_calls():
    agent = _make_agent()
    seen = _record_gru_windows(agent)
    event_sequences, chains = _batch_with_chain()
    _mcts_forward(agent, event_sequences, chains, "cpu", gru_window=7)
    # Root perception + chain perception both present, all with window 7.
    assert len(seen) >= 2
    assert all(w == 7 for w in seen)


def test_mcts_forward_default_gru_window_is_one():
    agent = _make_agent()
    seen = _record_gru_windows(agent)
    event_sequences, chains = _batch_with_chain()
    _mcts_forward(agent, event_sequences, chains, "cpu")
    assert len(seen) >= 2
    assert all(w == 1 for w in seen)


def test_run_validation_propagates_gru_window():
    agent = _make_agent()
    seen = _record_gru_windows(agent)
    event_sequences, chains = _batch_with_chain()
    val_targets = torch.zeros(1)
    act_targets = torch.full((1, 5), 0.2)
    val_loader = [(event_sequences, None, val_targets, act_targets,
                   chains, [[]])]
    weights = dict(value_weight=1.0, action_weight=1.0, chain_weight=1.0,
                   recon_weight=0.5, value_chain_weight=0.1,
                   chain_depth_gamma=0.9, entropy_weight=0.0,
                   terminal_value_weight=0.0, infonce_weight=0.5,
                   infonce_temperature=0.1)
    _run_validation(agent, val_loader, "cpu", weights, gru_window=5)
    assert len(seen) >= 2
    assert all(w == 5 for w in seen)


def test_train_cfg_gru_window_read():
    """train_mcts floors the config value at 1 (mirrors phase-5 k=max(1, ...))."""
    from agent.train_scenarios.mcts_predict import train as t
    # The read is `max(1, int(train_cfg.get("gru_window", 1)))` — pin the
    # source so a silent removal of the config knob fails loudly.
    import inspect
    src = inspect.getsource(t.train_mcts)
    assert 'train_cfg.get("gru_window", 1)' in src
