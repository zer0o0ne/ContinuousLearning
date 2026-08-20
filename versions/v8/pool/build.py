"""Building the opponent pool from config (CONCEPT.md §4.1, §8.1).

The `bootstrap` config section is a list of entries. Each entry names one *base*
policy and how many style draws to make from it:

.. code-block:: json

    {"kind": "degenerate", "strategy": "always_call", "style": "identity"}
    {"kind": "degenerate", "strategy": "nit",  "n_variants": 4}
    {"kind": "v7", "checkpoint": "../../data/v7/…/best.pt",
     "arch_config": "../../data/v7/…/config.json", "n_variants": 8}

``style`` is optional and selects where the draw comes from:

``absent``
    draw ``n_variants`` styles from the `style` config section (§4.2);
``"identity"``
    no modifier at all — the base policy as it is;
``list of 32 floats``
    an explicit style vector, for the hand-designed corners of the space.

The last two describe a single member, so ``n_variants`` must be 1 with them.

`build_pool` returns members **and** the descriptor list that the G1 report
needs to say which style each member was, and which base it came from.
"""

import json
import os

from pool.degenerate import DEGENERATE_STRATEGIES
from pool.style import StyleParams, sample_style
from pool.v7_member import V7NetworkMember
from vendor.v7.agent import V7Agent, n_actions_from_config


def _resolve_style(entry, rng, style_cfg):
    """List of `StyleParams`, one per member this entry produces."""
    spec = entry.get("style")
    n_variants = int(entry.get("n_variants", 1))
    if spec is None:
        return [sample_style(rng, style_cfg) for _ in range(n_variants)]
    assert n_variants == 1, (
        f"entry {entry.get('label') or entry!r} fixes a style but asks for "
        f"{n_variants} variants — they would all be the same member")
    if spec == "identity":
        return [StyleParams.identity()]
    return [StyleParams.from_list(spec)]


def _load_v7_agent(entry, device, log, cache=None):
    """Load a v7 checkpoint into the vendored perception+action subset.

    `cache` memoises by (checkpoint, arch_config). D13 puts every base in the
    pool twice — once unmodified, once as `n_variants` style draws — and those
    are two `bootstrap` entries naming one file, so without the cache the same
    weights are loaded and held in memory twice. Sharing is already the rule
    inside an entry (`with_style` shares one loaded agent across every variant);
    this extends it across entries, and it is a cache and not a semantic change:
    the agent is never mutated after `set_device`.
    """
    ckpt = entry["checkpoint"]
    arch_config = entry.get("arch_config")
    if arch_config is None:
        base = ckpt if os.path.isdir(ckpt) else os.path.dirname(ckpt)
        arch_config = os.path.join(base, "config.json")
    if not os.path.exists(arch_config):
        raise FileNotFoundError(
            f"v7 pool entry {entry.get('label') or ckpt!r} needs the v7 config "
            f"that describes its architecture; looked for {arch_config!r}. Set "
            f"\"arch_config\" on the entry.")
    key = (str(ckpt), str(arch_config))
    if cache is not None and key in cache:
        return cache[key]
    with open(arch_config) as fh:
        v7_config = json.load(fh)
    agent = V7Agent(v7_config, log=log)
    agent.load_checkpoint(ckpt)
    agent.eval()
    agent.set_device(device)
    loaded = (agent, n_actions_from_config(v7_config))
    if cache is not None:
        cache[key] = loaded
    return loaded


def build_pool(config, rng, device="cpu", log=print):
    """Build the pool. Returns ``(members, descriptors)``.

    A descriptor is a dict ``{name, kind, base, style}`` — enough for the G1
    report to group results by base policy and to record the exact 32-scalar
    draw a member was playing.
    """
    style_cfg = config.get("style", {})
    n_actions = None
    members, descriptors = [], []
    v7_cache = {}

    for entry in config["bootstrap"]:
        kind = entry["kind"]
        label = entry.get("label")
        styles = _resolve_style(entry, rng, style_cfg)

        if kind == "degenerate":
            strategy = entry["strategy"]
            assert strategy in DEGENERATE_STRATEGIES, (
                f"unknown degenerate strategy {strategy!r}; have "
                f"{sorted(DEGENERATE_STRATEGIES)}")
            entry_n_actions = int(config["game"]["n_actions"])
            base_name = label or strategy
            factory = DEGENERATE_STRATEGIES[strategy]
            bases = [factory(base_name, entry_n_actions, styles[0])]
        elif kind == "v7":
            agent, entry_n_actions = _load_v7_agent(entry, device, log,
                                                     cache=v7_cache)
            base_name = label or os.path.basename(str(entry["checkpoint"]))
            bases = [V7NetworkMember(base_name, entry_n_actions, agent,
                                     styles[0])]
        else:
            raise ValueError(f"unknown pool entry kind {kind!r}")

        if n_actions is None:
            n_actions = entry_n_actions
        assert entry_n_actions == n_actions, (
            f"pool entry {base_name!r} uses {entry_n_actions} actions, the "
            f"pool already uses {n_actions}")

        base = bases[0]
        for v, style in enumerate(styles):
            name = base_name if len(styles) == 1 else f"{base_name}#{v}"
            member = base if v == 0 else base.with_style(name, style)
            member.name = name
            member.style = style
            members.append(member)
            descriptors.append({
                "name": name,
                "kind": kind,
                "base": base_name,
                "style": style.to_list(),
            })

    return members, descriptors


def fresh_style_variants(members, descriptors, n, rng, style_cfg, tag):
    """Fresh style draws off the same bases — the B1(b) test set (§14 G1.2).

    "Because styles are procedural, these cost nothing to generate and restrict
    nothing: this is not a held-out opponent pool, it is a fresh draw."

    **Cycling is over distinct bases, not over members.** A base that expanded
    into many style variants occupies many consecutive slots in `members`, so
    cycling over members would draw the first few bases over and over and reach
    the later ones only if `n` exceeded the whole pool. In the first pilot that
    silently produced an unseen set containing no v7 network at all — the one
    base that conditions on cards — which made §14.2 a comparison between a
    mixed set and an all-degenerate one rather than a test of style
    generalisation. Cycling over bases guarantees every base contributes as soon
    as `n` reaches the number of bases.

    The draws come from the same `style` distribution the training pool was drawn
    from; the only difference is that the embedding network never saw a hand
    played by these settings.
    """
    by_base = {}
    for member, desc in zip(members, descriptors):
        by_base.setdefault(desc["base"], (member, desc))
    bases = list(by_base)

    out_members, out_descriptors = [], []
    for i in range(n):
        src, desc = by_base[bases[i % len(bases)]]
        style = sample_style(rng, style_cfg)
        name = f"{tag}{i}:{desc['base']}"
        out_members.append(src.with_style(name, style))
        out_descriptors.append({
            "name": name,
            "kind": desc["kind"],
            "base": desc["base"],
            "style": style.to_list(),
        })
    return out_members, out_descriptors
