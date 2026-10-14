"""Independent per-player, per-hand style draws for training sessions.

The hand owns the draw: posterior queries, replay and oracle continuations
must see the policy that actually acted, including inside label workers.
"""

import numpy as np

from pool.style import StyleParams, sample_style


def augmentation_probability(style_cfg, iteration):
    """Linear decay from 1 at iteration 0 to 0 at the configured endpoint.

    Missing/zero ``augmentation_zero_iteration`` disables augmentation, keeping
    old configs unchanged. Iterations use the pipeline's zero-based numbering.
    """
    end = style_cfg.get("augmentation_zero_iteration", 0)
    if isinstance(end, bool) or not isinstance(end, int) or end < 0:
        raise ValueError("style.augmentation_zero_iteration must be a nonnegative integer")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("augmentation iteration must be a nonnegative integer")
    if end == 0 or iteration >= end:
        return 0.0
    return 1.0 - iteration / end


def augmentation_signature(style_cfg, iteration):
    """Effective draw distribution, used to reject incompatible label resumes."""
    probability = augmentation_probability(style_cfg, iteration)
    if probability == 0.0:
        return None
    return {
        "version": 1,
        "probability": probability,
        "style": {
            key: style_cfg.get(key, default)
            for key, default in (
                ("uncond_scale", 0.0), ("position_scale", 0.0),
                ("street_scale", 0.0), ("log_temperature_range", [0.0, 0.0]),
                ("uniform_mix_range", [0.0, 0.0]))
        },
    }


def augment_sessions(sessions, style_cfg, iteration, *, exclude_slot=None):
    """Attach compact 32-vectors by seat, without changing members or RNGs.

    Even repeated copies of one pool member get independent draws. The seed
    depends only on the original hand and seat, never on execution order or
    rollout seeds. ``exclude_slot=0`` leaves the current training hero alone;
    corpus self-play augments every seat, including its observer.
    """
    probability = augmentation_probability(style_cfg, iteration)
    identity = np.asarray(StyleParams.identity().to_list())
    for session in sessions:
        for h, spec in enumerate(session.specs):
            spec.style_augmentations = None
            if probability == 0.0:
                session.specs[h] = spec
                continue
            for seat, slot in enumerate(session.slot_of_seat(h)):
                if slot == exclude_slot:
                    continue
                rng = np.random.default_rng([int(spec.seed), seat, 7351])
                if rng.random() >= probability:
                    continue
                if spec.style_augmentations is None:
                    spec.style_augmentations = np.tile(identity, (spec.num_players, 1))
                spec.style_augmentations[seat] = sample_style(rng, style_cfg).to_list()
            session.specs[h] = spec
