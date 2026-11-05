"""Evaluation plans: which suites, attacks and injection positions to run.

A plan is a list of blocks. Items for each suite come from one fixed, seeded pool and
every block takes a *prefix* of that pool, so the items of a small block (e.g. the
extra positions used for the adaptive best-of-K score) are always a subset of the
items of a larger block. That keeps per-item comparisons paired.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from injdrift.attacks.static import ATTACKS, build_payload, inject
from injdrift.data.suites import Case, TaskItem, injected_for, load_suite
from injdrift.formats import Dialect

POOL_SIZES = {"alpacafarm": None, "squad": 200, "toolout": 140}


@dataclass(frozen=True)
class Block:
    suite: str
    n: int
    attacks: tuple[str, ...]
    positions: tuple[str, ...] = ("end",)


ADAPTIVE = ("ignore", "completion", "combined")

PROFILES: dict[str, tuple[Block, ...]] = {
    # A few dozen prompts: wiring checks and CI smoke tests.
    "smoke": (
        Block("alpacafarm", 6, ("naive", "ignore", "combined")),
        Block("toolout", 4, ("none", "ignore")),
    ),
    # Cheap set evaluated at intermediate checkpoints while a fine-tune runs.
    "track": (Block("alpacafarm", 32, ADAPTIVE),),
    # Default for the CI gate: static ASR on all three suites.
    "gate": (
        Block("alpacafarm", 48, ATTACKS),
        Block("squad", 16, ("none",) + ATTACKS),
        Block("toolout", 16, ("none",) + ATTACKS),
    ),
    # Final evaluation of every run in the pilot grid.
    "standard": (
        Block("alpacafarm", 64, ATTACKS),
        Block("squad", 32, ("none",) + ATTACKS),
        Block("toolout", 32, ("none",) + ATTACKS),
        Block("alpacafarm", 32, ADAPTIVE, ("start", "middle")),
        Block("squad", 16, ADAPTIVE, ("start", "middle")),
        Block("toolout", 16, ADAPTIVE, ("start", "middle")),
    ),
    # Everything: all 208 AlpacaFarm items, every attack at every position.
    "full": (
        Block("alpacafarm", 208, ATTACKS, ("end", "start", "middle")),
        Block("squad", 100, ("none",) + ATTACKS, ("end", "start", "middle")),
        Block("toolout", 100, ("none",) + ATTACKS, ("end", "start", "middle")),
    ),
}


def profile_size(blocks: tuple[Block, ...]) -> int:
    """Upper bound on the number of cases a profile expands to (before de-duplication)."""
    total = 0
    for b in blocks:
        attacked = [a for a in b.attacks if a != "none"]
        total += b.n * (len(attacked) * len(b.positions) + ("none" in b.attacks))
    return total


def suite_pool(suite: str, seed: int = 0) -> list[TaskItem]:
    """The fixed, shuffled item pool a plan draws prefixes from."""
    items = load_suite(suite, n=POOL_SIZES[suite], seed=seed)
    order = list(range(len(items)))
    random.Random(f"pool-{suite}-{seed}").shuffle(order)
    return [items[i] for i in order]


def cases_for_items(
    items: list[TaskItem],
    attacks: tuple[str, ...],
    positions: tuple[str, ...],
    dialect: Dialect,
    *,
    seed: int = 0,
    index_offset: int = 0,
) -> list[Case]:
    cases: list[Case] = []
    for idx, item in enumerate(items):
        injected = injected_for(idx + index_offset)
        for attack in attacks:
            if attack == "none":
                cases.append(Case(item, "none", "-", None, ((item.data, "data"),)))
                continue
            for pos in positions:
                rng = random.Random(f"{seed}/{item.key}/{attack}/{pos}")
                payload = build_payload(
                    attack, injected.instruction, reference=item.reference, dialect=dialect, rng=rng
                )
                pieces = inject(
                    item.data, payload, pos, site=item.site, json_escape=item.json_escape
                )
                cases.append(Case(item, attack, pos, injected, pieces))
    return cases


def build_plan(profile: str | tuple[Block, ...], dialect: Dialect, *, seed: int = 0) -> list[Case]:
    """Expand a profile into concrete, de-duplicated cases."""
    blocks = PROFILES[profile] if isinstance(profile, str) else profile
    pools: dict[str, list[TaskItem]] = {}
    seen: set[str] = set()
    out: list[Case] = []
    for block in blocks:
        if block.suite not in pools:
            pools[block.suite] = suite_pool(block.suite, seed)
        items = pools[block.suite][: block.n]
        for case in cases_for_items(items, block.attacks, block.positions, dialect, seed=seed):
            if case.key not in seen:
                seen.add(case.key)
                out.append(case)
    return out
