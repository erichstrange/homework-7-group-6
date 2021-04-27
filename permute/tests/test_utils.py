"""Test Suite for utils.py."""

import sys
import pytest

import numpy as np
from scipy.stats import hypergeom, binom
from cryptorandom.cryptorandom import SHA256
from cryptorandom.sample import random_sample, random_permutation

from ..utils import (binom_conf_interval,
                     hypergeom_conf_interval,
                     hypergeometric,
                     binomial_p,
                     get_prng,
                     permute,
                     permute_rows,
                     permute_within_groups,
                     permute_incidence_fixed_sums,
                     potential_outcomes)


def test_binom_conf_interval1():
    """Clopper-Pearson tests.

    Tests legal calls to binom_conf_interval, using the
    default Clopper-Pearson computation method. Asserts proper
    bounds are returned for both one-sided and two-sided.
    """
    res = binom_conf_interval(10, 3)
    expected = (0.05154625578928545, 0.6915018049393984)
    np.testing.assert_equal(res, expected)

    res2 = binom_conf_interval(10, 5, cl=0.95, alternative="upper")
    expected2 = (0.0, 0.7775588989918742)
    np.testing.assert_equal(res2, expected2)

    res3 = binom_conf_interval(10, 5, cl=0.95, alternative="lower")
    expected3 = (0.22244110100812578, 1.0)
    np.testing.assert_equal(res3, expected3)

    res4 = binom_conf_interval(10, 5, cl=0.95, alternative="upper", p=1)
    expected4 = (0.0, 0.7775588989918742)
    np.testing.assert_equal(res4, expected4)

    res5 = binom_conf_interval(10, 5, cl=0.95, alternative="lower", p=0)
    expected5 = (0.22244110100812578, 1.0)
    np.testing.assert_equal(res5, expected5)

    lower1, upper1 = binom_conf_interval(10, 4)
    assert(lower1 <= 0.4 <= upper1)

    lower2, upper2 = binom_conf_interval(10, 4, alternative="lower")
    assert(lower2 <= 0.4 <= upper2)

    lower3, upper3 = binom_conf_interval(10, 4, alternative="upper")
    assert(lower3 <= 0.4 <= upper3)


def test_binom_conf_interval2():
    """Sterne tests.

    Tests legal calls to binom_conf_interval, using the
    Sterne computation method. Asserts proper
    bounds are returned for both one-sided and two-sided.
    """
    res = binom_conf_interval(10, 3, 0.95, 'two-sided', None, 'sterne')
    assert(np.isclose(res[0], 0.087))
    assert(np.isclose(res[1], 0.62))

    res2 = binom_conf_interval(10, 5, 0.95, 'two-sided', None, 'sterne')
    assert(res2[0] >= 0 and res2[1] >= 0)

    lower2, upper2 = binom_conf_interval(10, 4, 0.95,
                                         'two-sided', None, 'sterne')
    assert(lower2 <= 0.4 <= upper2)


def test_binom_conf_badinput1():
    """Clopper-Pearson.

    Observed successes cannot be larger than sample size
    """
    pytest.raises(ValueError, binom_conf_interval, 4, 10)


def test_binom_conf_badinput2():
    """Clopper-Pearson.

    Observed successes cannot be negative
    """
    pytest.raises(ValueError, binom_conf_interval, 10, -4)


def test_binom_conf_badinput3():
    """Sterne.

    With current implementation, Sterne can only be used with two-sided CI
    """
    pytest.raises(ValueError, binom_conf_interval,
                  10, 3, 0.95, 'lower', None, 'sterne')


def test_binom_conf_badinput4():
    """Sterne.

    With current implementation, Sterne can only be used with two-sided CI
    """
    pytest.raises(ValueError, binom_conf_interval,
                  10, 3, 0.95, 'upper', None, 'sterne')


def test_binom_conf_badinput5():
    """Sterne.

    Observed successes cannot be larger than sample size
    """
    pytest.raises(ValueError, binom_conf_interval,
                  10, 11, 0.95, 'two-sided', None, 'sterne')


def test_binom_conf_badinput6():
    """Clopper-Pearson.

    With current implementation, Sterne can only be used with two-sided CI
    """
    pytest.raises(ValueError, binom_conf_interval,
                  10, 3, 0.95, 'lower', None, 'sterne')


def test_binom_conf_badinput7():
    """Clopper-Pearson.

    Observed successes cannot be larger than sample size
    when alternate == 'upper'
    """
    pytest.raises(ValueError, binom_conf_interval,
                  10, 11, 0.95, 'upper', None, 'clopper-pearson')


def test_binom_conf_badinput8():
    """Clopper-Pearson.

    Observed successes cannot be larger than sample size
    when alternate == 'lower'
    """
    pytest.raises(ValueError, binom_conf_interval,
                  10, 11, 0.95, 'lower', None, 'clopper-pearson')


def test_hypergeom_conf_interval1():
    """Clopper-Pearson Tests.

    Tests legal calls to hypergeom_conf_interval, using the
    default Clopper-Pearson computation method. Asserts proper
    bounds are returned for two-sided CI's.

    """
    res = hypergeom_conf_interval(2, 1, 5, cl=0.95, alternative="two-sided")
    expected = (1.0, 4.0)
    np.testing.assert_equal(res, expected)

    res2 = hypergeom_conf_interval(2, 1, 5, cl=0.95, alternative="upper")
    expected2 = (0.0, 4.0)
    np.testing.assert_equal(res2, expected2)

    res3 = hypergeom_conf_interval(2, 1, 5, cl=0.95, alternative="lower")
    expected3 = (1.0, 5.0)
    np.testing.assert_equal(res3, expected3)

    res4 = hypergeom_conf_interval(2, 2, 5, cl=0.95, alternative="two-sided")
    expected4 = (2.0, 5.0)
    np.testing.assert_equal(res4, expected4)
    assert(res4[0] >= 0 and res4[1] >= 0)

    cl = 0.95
    n = 10
    x = 5
    N = 20
    [lot, hit] = [6, 14]
    alternative = "two-sided"
    [lo, hi] = hypergeom_conf_interval(n, x, N, cl=cl, alternative=alternative)
    np.testing.assert_equal(lo, lot)
    np.testing.assert_equal(hi, hit)

    res5 = hypergeom_conf_interval(2, 1, 5, cl=0.95, alternative="upper", G=5)
    np.testing.assert_equal(res5, expected2)

    res6 = hypergeom_conf_interval(2, 1, 5, cl=0.95, alternative="lower", G=0)
    np.testing.assert_equal(res6, expected3)


def test_hypergeom_conf_interval2():
    """Sterne Tests.

    Tests legal calls to hypergeom_conf_interval, using the
    Sterne computation method. Asserts proper
    bounds are returned for two-sided CI's.
    """
    res = hypergeom_conf_interval(2, 1, 5, cl=0.95,
                                  alternative="two-sided",
                                  G=None, method='sterne')
    expected = (0.0, 5.0)
    np.testing.assert_equal(res, expected)

    res4 = hypergeom_conf_interval(2, 2, 5, cl=0.95,
                                   alternative="two-sided",
                                   G=None, method='sterne')
    expected4 = (1.0, 5.0)
    np.testing.assert_equal(res4, expected4)

    cl = 0.95
    n = 10
    x = 5
    N = 20
    [lot, hit] = [5, 15]
    alternative = "two-sided"
    [lo, hi] = hypergeom_conf_interval(n, x, N, cl=cl,
                                       alternative=alternative,
                                       G=None, method='sterne')
    np.testing.assert_equal(lo, lot)
    np.testing.assert_equal(hi, hit)


def test_hypergeom_conf_interval3():
    """Wang Tests.

    Tests legal calls to hypergeom_conf_interval,
    using the Wang computation method. Asserts proper
    bounds are returned for two-sided CI's.
    """
    res = hypergeom_conf_interval(2, 1, 5, cl=0.95,
                                  alternative="two-sided",
                                  G=None, method='wang')
    expected = (1, 4)
    np.testing.assert_equal(res, expected)

    res1 = hypergeom_conf_interval(2, 2, 5, cl=0.95,
                                   alternative="two-sided",
                                   G=None, method='wang')
    expected4 = (2, 5)
    np.testing.assert_equal(res1, expected4)

    cl = 0.95
    n = 10
    x = 5
    N = 20
    [lot, hit] = [6, 14]
    alternative = "two-sided"
    [lo, hi] = hypergeom_conf_interval(n, x, N, cl=cl,
                                       alternative=alternative,
                                       G=None, method='wang')
    np.testing.assert_equal(lo, lot)
    np.testing.assert_equal(hi, hit)


def test_hypergeometric_conf_badinput1():
    """Clopper-Pearson.

    Observed successes cannot be larger than sample size
    """
    pytest.raises(ValueError, hypergeom_conf_interval, 5, 6, 10)


def test_hypergeometric_conf_badinput2():
    """Clopper-Pearson.

    Population size cannot be smaller than size
    of sample taken w/o replacement
    """
    pytest.raises(ValueError, hypergeom_conf_interval, 5, 1, 4)


def test_hypergeometric_conf_badinput3():
    """Clopper-Pearson.

    Number of observed successes cannot be larger
    than size of population
    """
    pytest.raises(ValueError, hypergeom_conf_interval, 5, 11, 10)


def test_hypergeometric_conf_badinput4():
    """Clopper-Pearson.

    Number of observed successes cannot be negative
    """
    pytest.raises(ValueError, hypergeom_conf_interval, 5, -5, 10)


def test_hypergeometric_conf_badinput5():
    """Sterne.

    With current implementation,
    Sterne can only be used with two-sided CI
    """
    pytest.raises(ValueError, hypergeom_conf_interval,
                  10, 5, 100, 0.95, 'lower', None, 'sterne')


def test_hypergeometric_conf_badinput6():
    """Sterne.

    With current implementation,
    Sterne can only be used with two-sided CI
    """
    pytest.raises(ValueError, hypergeom_conf_interval,
                  10, 5, 100, 0.95, 'upper', None, 'sterne')


def test_hypergeometric_conf_badinput7():
    """Sterne.

    Sample size is too big when two-sided
    """
    pytest.raises(ValueError, hypergeom_conf_interval,
                  101, 5, 100, 0.95, 'two-sided', None, 'sterne')


def test_hypergeometric_conf_badinput8():
    """Sterne.

    Observed successes cannot be larger than sample size
    """
    pytest.raises(ValueError, hypergeom_conf_interval,
                  5, 6, 10, 0.95, 'two-sided', None, 'sterne')


def test_hypergeometric_conf_badinput9():
    """Sterne.

    Number of observed successes cannot be negative
    """
    pytest.raises(ValueError, hypergeom_conf_interval,
                  5, -5, 10, 0.95, 'two-sided', None, 'sterne')


def test_hypergeometric_conf_badinput10():
    """Wang.

    With current implementation,
    Wang can only be used with two-sided CI
    """
    pytest.raises(ValueError, hypergeom_conf_interval,
                  10, 5, 100, 0.95, 'lower', None, 'wang')


def test_hypergeometric_conf_badinput11():
    """Wang.

    With current implementation, Wang can only be used with two-sided CI
    """
    pytest.raises(ValueError, hypergeom_conf_interval,
                  10, 5, 100, 0.95, 'upper', None, 'wang')


def test_hypergeometric_conf_badinput12():
    """Wang.

    Sample size is too big when two-sided
    """
    pytest.raises(ValueError, hypergeom_conf_interval,
                  101, 5, 100, 0.95, 'two-sided', None, 'wang')


def test_hypergeometric_conf_badinput13():
    """Wang.

    Observed successes cannot be larger than sample size
    """
    pytest.raises(ValueError, hypergeom_conf_interval,
                  5, 6, 10, 0.95, 'two-sided', None, 'wang')


def test_hypergeometric_conf_badinput14():
    """Wang.

    Number of observed successes cannot be negative
    """
    pytest.raises(ValueError, hypergeom_conf_interval,
                  5, -5, 10, 0.95, 'two-sided', None, 'wang')


def test_hypergeometric():
    """Clopper-Pearson."""
    np.testing.assert_almost_equal(hypergeometric(4, 10, 5, 6, 'greater'),
                                   1-hypergeom.cdf(3, 10, 5, 6))
    np.testing.assert_almost_equal(hypergeometric(4, 10, 5, 6, 'less'),
                                   hypergeom.cdf(4, 10, 5, 6))
    np.testing.assert_almost_equal(hypergeometric(4, 10, 5, 6, 'two-sided'),
                                   2*(1-hypergeom.cdf(3, 10, 5, 6)))


def test_hypergeometric_badinput1():
    """Clopper-Pearson."""
    pytest.raises(ValueError, hypergeometric, 5, 10, 2, 6)


def test_hypergeometric_badinput2():
    """Clopper-Pearson."""
    pytest.raises(ValueError, hypergeometric, 5, 10, 18, 6)


def test_hypergeometric_badinput3():
    """Clopper-Pearson."""
    pytest.raises(ValueError, hypergeometric, 5, 10, 6, 16)


def test_hypergeometric_badinput4():
    """Clopper-Pearson."""
    pytest.raises(ValueError, hypergeometric, 5, 10, 6, 2)


def test_binomial_p():
    """Binomial Test."""
    np.testing.assert_almost_equal(binomial_p(5, 10, 0.5, 'greater'),
                                   1-binom.cdf(4, 10, 0.5))
    np.testing.assert_almost_equal(binomial_p(5, 10, 0.5, 'less'),
                                   binom.cdf(5, 10, 0.5))
    np.testing.assert_almost_equal(binomial_p(5, 10, 0.5, 'two-sided'), 1)


def test_binomial_badinput():
    """Clopper-Pearson."""
    pytest.raises(ValueError, binomial_p, 10, 5, 0.5)


def test_get_random_state():
    """Random State Test."""
    prng1 = np.random.RandomState(42)
    prng2 = get_prng(42)
    prng3 = get_prng(prng1)
    prng4 = get_prng(prng2)
    prng5 = get_prng()
    prng6 = get_prng(None)
    prng7 = get_prng(np.random)
    prng8 = get_prng(SHA256(42))
    assert(isinstance(prng1, np.random.RandomState))
    assert(isinstance(prng2, SHA256))
    assert(isinstance(prng3, np.random.RandomState))
    assert(isinstance(prng4, SHA256))
    assert(isinstance(prng5, SHA256))
    assert(isinstance(prng6, SHA256))
    assert(isinstance(prng7, np.random.RandomState))
    x1 = prng1.randint(0, 5, size=10)
    x2 = prng2.randint(0, 5, size=10)
    x3 = prng3.randint(0, 5, size=10)
    x4 = prng4.randint(0, 5, size=10)
    x5 = prng5.randint(0, 5, size=10)
    x6 = prng6.randint(0, 5, size=10)
    x7 = prng7.randint(0, 5, size=10)
    x8 = prng8.randint(0, 5, size=10)
    np.testing.assert_equal(x2, x8)
    assert prng2.counter == 1
    assert prng2.baseseed == 42
    assert prng2.baseseed == prng4.baseseed
    assert len(x5) == 10
    assert len(x6) == 10
    assert len(x7) == 10


def test_get_random_state_error():
    """Random State Error Test."""
    pytest.raises(ValueError, get_prng, [1, 1.11])


def test_permute_within_group():
    """Group Permute Test."""
    x = np.repeat([1, 2, 3] * 3, 3)
    group = np.repeat([1, 2, 3], 9)

    res1 = permute_within_groups(x, group, seed=42)
    res2 = permute_within_groups(x, group, seed=SHA256(42))
    np.testing.assert_equal(res1, res2)

    res3 = permute_within_groups(x, group)
    np.testing.assert_equal(res3.max(), 3)
    res3.sort()
    np.testing.assert_equal(group, res3)


@pytest.mark.skipif(sys.platform == "win32",
                    reason="need updated cryptorandom")
def test_permute():
    """Permute Test."""
    prng = SHA256(42)

    x = prng.randint(0, 10, size=20)
    actual = permute(x, prng)
    expected = np.array([6, 9, 5, 1, 3, 1, 4, 7, 6, 9,
                         8, 7, 2, 1, 9, 7, 8, 1, 8, 1])
    np.testing.assert_array_equal(actual, expected)

    actual = permute(x)
    np.testing.assert_equal(actual.max(), 9)
    np.testing.assert_equal(actual.min(), 1)


def test_permute_rows():
    """Permute Rows Test."""
    prng = SHA256(42)

    x = prng.randint(0, 10, size=20).reshape(2, 10)
    actual = permute_rows(x, prng)
    expected = np.array([[9, 1, 8, 6, 9, 1, 5, 4, 3, 6],
                         [1, 7, 2, 1, 9, 7, 8, 7, 8, 1]])
    np.testing.assert_array_equal(actual, expected)

    a = permute_rows(x)
    np.testing.assert_equal(a.max(), 9)
    np.testing.assert_equal(a.min(), 1)


def test_permute_incidence_fixed_sums():
    """Fixed Sums Permute Test."""
    prng = np.random.RandomState(42)
    x0 = prng.randint(2, size=80).reshape((8, 10))
    x1 = permute_incidence_fixed_sums(x0)

    K = 5

    m = []
    for i in range(1000):
        x2 = permute_incidence_fixed_sums(x0, k=K)
        m.append(np.sum(x0 != x2))

    np.testing.assert_(max(m) <= K * 4,
                       "Too many swaps occurred")

    for axis in (0, 1):
        for test_arr in (x1, x2):
            np.testing.assert_array_equal(x0.sum(axis=axis),
                                          test_arr.sum(axis=axis))


def test_permute_incidence_fixed_sums_ND_arr():
    """Fixed Sums Permute Test."""
    pytest.raises(ValueError, permute_incidence_fixed_sums,
                  np.random.random((1, 1, 1)))


def test_permute_incidence_fixed_sums_non_binary():
    """Fixed Sums Non Binary Permute Test."""
    pytest.raises(ValueError, permute_incidence_fixed_sums,
                  np.array([[1, 2], [3, 4]]))


def test_potential_outcomes():
    """Testing Potential Outcomes."""
    x = np.array(range(5)) + 1
    y = x + 4.5

    def f(u):
        return u + 3.5

    def finv(u):
        return u - 3.5

    def g(g):
        return np.exp(u * 2)

    def ginv(u):
        return np.log(u) / 2

    resf = potential_outcomes(x, y, f, finv)
    resg = potential_outcomes(x, y, g, ginv)
    expectedf = np.array([[1.,  -2.5],
                          [2.,  -1.5],
                          [3.,  -0.5],
                          [4.,   0.5],
                          [5.,   1.5],
                          [9.,   5.5],
                          [10.,   6.5],
                          [11.,   7.5],
                          [12.,   8.5],
                          [13.,   9.5]])
    expectedg = np.array([[1.00000000e+00,   0.00000000e+00],
                          [2.00000000e+00,   3.46573590e-01],
                          [3.00000000e+00,   5.49306144e-01],
                          [4.00000000e+00,   6.93147181e-01],
                          [5.00000000e+00,   8.04718956e-01],
                          [5.98741417e+04,   5.50000000e+00],
                          [4.42413392e+05,   6.50000000e+00],
                          [3.26901737e+06,   7.50000000e+00],
                          [2.41549528e+07,   8.50000000e+00],
                          [1.78482301e+08,   9.50000000e+00]])
    np.testing.assert_equal(resf, expectedf)
    np.testing.assert_almost_equal(resg[:5, :], expectedg[:5, :])
    np.testing.assert_almost_equal(resg[5:, :], expectedg[5:, :], 1)


def test_potential_outcomes_bad_inverse():
    """Bad inverse test."""
    def f(u):
        return u + 3.5

    def ginv(u):
        return np.log(u) / 2

    pytest.raises(AssertionError, potential_outcomes,
                  np.array([1, 2]), np.array([3, 4]), f, ginv)
