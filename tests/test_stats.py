import math

import pytest

from injdrift.stats import binom_sf, mcnemar_one_sided, paired_bootstrap_diff, pearson, wilson


def test_wilson_matches_reference_values():
    lo, hi = wilson(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-3)
    assert hi == pytest.approx(0.1744, abs=1e-3)
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and hi == pytest.approx(0.0714, abs=1e-3)
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_binomial_tail_and_mcnemar():
    assert binom_sf(0, 10) == 1.0
    assert binom_sf(10, 10) == pytest.approx(0.5**10)
    assert mcnemar_one_sided(0, 0) == 1.0
    assert mcnemar_one_sided(8, 0) == pytest.approx(0.5**8)
    assert mcnemar_one_sided(5, 5) > 0.5


def test_paired_bootstrap_is_deterministic_and_brackets_mean():
    a = [0, 0, 0, 1, 0, 0, 1, 0]
    b = [1, 0, 1, 1, 0, 1, 1, 0]
    m, lo, hi = paired_bootstrap_diff(a, b, n_boot=500, seed=3)
    assert m == pytest.approx(0.375)
    assert lo <= m <= hi
    assert (m, lo, hi) == paired_bootstrap_diff(a, b, n_boot=500, seed=3)


def test_pearson():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert math.isnan(pearson([1, 1, 1], [1, 2, 3]))
