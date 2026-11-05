"""Small, dependency-free statistics: Wilson intervals, exact McNemar, bootstrap."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed exactly."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def mcnemar_one_sided(newly_broken: int, newly_fixed: int) -> float:
    """Exact one-sided McNemar p-value for 'the candidate is *more* attackable'.

    Only discordant pairs carry information: ``newly_broken`` prompts were resisted by
    the reference but succeed on the candidate, ``newly_fixed`` the other way round.
    Under the null both are equally likely, so the p-value is a binomial tail.
    """
    n = newly_broken + newly_fixed
    if n == 0:
        return 1.0
    return binom_sf(newly_broken, n, 0.5)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``b - a`` over paired items with a percentile bootstrap 95% CI."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    if not a:
        return (math.nan, math.nan, math.nan)
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (mean, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1])


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 3:
        return math.nan
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return math.nan
    return sxy / math.sqrt(sxx * syy)
