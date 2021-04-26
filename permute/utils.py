"""Various utilities and helper functions."""


import math
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, hypergeom
from cryptorandom.cryptorandom import SHA256
from cryptorandom.sample import random_sample, random_permutation


def binom_conf_interval(n, x, cl=0.975, alternative="two-sided", p=None,
                        method='clopper-pearson', **kwargs):
    """
    Compute a confidence interval for a binomial p.

    Parameters
    ----------
    n : int
        The number of Bernoulli trials.
    x : int
        The number of successes.
    cl : float in (0, 1)
        The desired confidence level.
    alternative : {"two-sided", "lower", "upper"}
        Indicates the alternative hypothesis.
    p : float in (0, 1)
        Starting point in search for confidence bounds
        for probability of success in each trial.
    method: {'clopper-pearson', 'wang', 'sterne'}
        The desired computation method
    kwargs : dict
        Key word arguments

    Returns
    -------
    tuple
        lower and upper confidence level with coverage (approximately)
        1-alpha.

    Notes
    -----
    xtol : float
        Tolerance
    rtol : float
        Tolerance
    maxiter : int
        Maximum number of iterations.
    """
    assert alternative in ("two-sided", "lower", "upper")
    if n < x:
        raise ValueError("Cannot observe more successes than "
                         + "the population size")
    if x < 0:
        raise ValueError("Cannot have negative successes cases")
    if method not in ['clopper-pearson', 'wang', 'sterne']:
        raise ValueError("Wrong Method!")

    if method == 'clopper-pearson':
        if p is None:
            p = x / n
        ci_low = 0.0
        ci_upp = 1.0
        if alternative == 'two-sided':
            cl = 1 - (1 - cl) / 2

        if alternative != "upper" and x > 0:
            def f(q):
                return cl - binom.cdf(x - 1, n, q)
            while f(p) < 0:
                p = (p+1)/2
            ci_low = brentq(f, 0.0, p, *kwargs)

        if alternative != "lower" and x < n:
            def f(q):
                return binom.cdf(x, n, q) - (1 - cl)
            while f(p) < 0:
                p = p/2
            ci_upp = brentq(f, 1.0, p, *kwargs)
        return ci_low, ci_upp

    if method == 'wang':
        if alternative != "two-sided":
            raise ValueError("Alternative should be 2-sided for this method")
        return wang_binom_conf(n, x, cl, p)

    if method == 'sterne':
        if alternative != "two-sided":
            raise ValueError("Alternative should be 2-sided for this method")
        return sterne_binom_conf(n, x, cl)


@lru_cache(maxsize=None)  # decorate the function to cache the results
# of calls to the function
def binom_accept(n, p, alpha=0.05, randomized=False):
    """
    Acceptance region for a randomized binomial test.

    If randomized==True, find the acceptance region for a randomized, exact
    level-alpha test of the null hypothesis X~Binomial(n,p).
    The acceptance region is the smallest possible.
    (And not, for instance, symmetric.)

    If randomized==False, find the smallest conservative acceptance region.

    Parameters
    ----------
    n : integer
        number of independent trials
    p : float
        probability of success in each trial
    alpha : float
        desired significance level
    ramndomized : Boolean
        return randomized exact test or conservative non-randomized test?

    Returns
    --------
    If randomized:
    never_rejects : list
        values for which the test never rejects
    never_rejects : list
        values for which the test sometimes rejects
    gamma : float
        probability the test does not reject when the value is in J

    If not randomized:
    never_rejects : list
        values for which the test does not reject

    """
    assert 0 < alpha < 1, "bad significance level"
    x = np.arange(0, n+1)
    never_rejects = list(x)  # start w/ all possible outcomes (then remove)
    pmf = binom.pmf(x, n, p)  # "frozen" binomial pmf
    bottom = 0  # smallest outcome still in never_rejects
    top = n  # largest outcome still in never_rejects
    J = []  # outcomes for which the test is randomized
    p_J = 0  # probability of outcomes for which test is randomized
    p_tail = 0  # probability of outcomes excluded from never_rejects
    while p_tail < alpha:  # need to remove outcomes from the acceptance region
        pb = pmf[bottom]
        pt = pmf[top]
        if pb < pt:  # the smaller possibility has smaller probability
            J = [bottom]
            p_J = pb
            bottom += 1
        elif pb > pt:  # the larger possibility has smaller probability
            J = [top]
            p_J = pt
            top -= 1
        else:
            if bottom < top:  # the two possibilities have equal probability
                J = [bottom, top]
                p_J = pb+pt
                bottom += 1
                top -= 1
            else:  # there is only one possibility left
                J = [bottom]
                p_J = pb
                bottom += 1
        p_tail += p_J
        for j in J:  # remove outcomes from acceptance region
            never_rejects.remove(j)
    return_val = None
    if randomized:
        """
        probability of accepting H_0 when X in J
        to get exact level alpha
        """
        gamma = (p_tail-alpha)/p_J
        return_val = never_rejects, J, gamma
    else:
        while p_tail > alpha:
            j = J.pop()  # move the outcome into the acceptance region
            p_tail -= pmf[j]
            never_rejects.append(j)
        return_val = never_rejects
    return return_val


def sterne_binom_conf(n, x, cl=0.95, eps=10**-3):
    """
    two-sided confidence bound for a binomial p.

    Assumes x is a draw from a binomial distribution with parameters
    n (known) and p (unknown). Finds a confidence interval for p
    at confidence level cl by inverting conservative tests

    Parameters
    ----------
    n : int
        number of trials, nonnegative integer
    x : int
        observed number of successes, nonnegative integer not larger than n
    cl : float
        confidence level, between 1/2 and 1
    eps : float in (0, 1)
        resolution of the grid search

    Returns
    -------
    lb : float
        lower confidence bound
    ub : float
        upper confidence bound
    """
    assert 0 <= x <= n, 'impossible arguments'
    assert 0 < cl < 1, 'silly confidence level'
    lb = 0
    ub = 1
    alpha = 1-cl
    if x > 0:
        while x not in binom_accept(n, lb, alpha, randomized=False):
            lb += eps
        lb -= eps
    if x < n:
        while x not in binom_accept(n, ub, alpha, randomized=False):
            ub -= eps
        ub += eps
    return lb, ub


def hypergeom_conf_interval(n, x, N, cl=0.975, alternative="two-sided", G=None,
                            method='clopper-pearson', **kwargs):
    """
    Confidence interval for a hypergeometric distribution parameter G.

    G is the number of good objects in a population in size N,
    based on the number x of good objects in a simple
    random sample of size n.

    Parameters
    ----------
    n : int
        The number of draws without replacement.
    x : int
        The number of "good" objects in the sample.
    N : int
        The number of objects in the population.
    cl : float in (0, 1)
        The desired confidence level.
    alternative : {"two-sided", "lower", "upper"}
        Indicates the alternative hypothesis.
    G : int in [0, N]
        Starting point in search for confidence bounds
        for the hypergeometric parameter G.
    method: {'clopper-pearson', 'wang', 'sterne'}
        The desired computation method
    kwargs : dict
        Key word arguments

    Returns
    -------
    tuple
        lower and upper confidence level with coverage (at least)
        1-alpha.

    Notes
    -----
    xtol : float
        Tolerance
    rtol : float
        Tolerance
    maxiter : int
        Maximum number of iterations.
    """
    assert alternative in ("two-sided", "lower", "upper")
    if n < x:
        raise ValueError("Cannot observe more good elements " +
                         "than the sample size")
    if x < 0:
        raise ValueError("Cannot have negative successes cases")
    if N < n:
        raise ValueError("Population size cannot be smaller than sample")
    if G:
        if N < G:
            raise ValueError("Can't have starting point for number of good "
                             + "elements in population exceed the "
                             + "population size")
    if method not in ['clopper-pearson', 'wang', 'sterne']:
        raise ValueError("Wrong Method!")

    if method == 'clopper-pearson':
        if G is None:
            G = (x / n) * N
        ci_low = 0
        ci_upp = N

        if alternative == 'two-sided':
            cl = 1 - (1 - cl) / 2

        if alternative != "upper" and x > 0:
            def f(q):
                return cl - hypergeom.cdf(x - 1, N, q, n)
            while f(G) < 0:
                G = (G+N)/2
            ci_low = math.ceil(brentq(f, 0.0, G, *kwargs))

        if alternative != "lower" and x < n:
            def f(q):
                return hypergeom.cdf(x, N, q, n) - (1 - cl)
            while f(G) < 0:
                G = G/2
            ci_upp = math.floor(brentq(f, G, N, *kwargs))

        return ci_low, ci_upp

    if method == 'wang':
        if alternative != "two-sided":
            raise ValueError("Alternative should be 2-sided for this method")
        alpha = 1 - cl
        return wang_hypergeom_conf(n, x, N, alpha)

    if method == 'sterne':
        if alternative != "two-sided":
            raise ValueError("Alternative should be 2-sided for this method")
        return sterne_hypergeom_conf(N, n, x, cl)


# Wang method
def wang_lci(n, alpha, N):
    """Wang method: Calculating lower bound of 1-alpha interval."""
    lci = list(range(n + 1))
    xx = list(range(n + 1))
    for i in range(n + 1):
        if xx[i] < 0.5:
            lci[i] = 0
        else:
            aa = list(range(N + 2))
            bb = [k + 1 for k in aa]
            for j in range(N + 1):
                bb[j+1] = hypergeom.cdf(xx[i] - 1, N, aa[j+1]-1, n)

            dd = list(map(list, zip(aa, bb)))
            dd = np.array([i for i in dd if i[1] >= 1 - alpha])
            if (len(dd) == 1):
                lci[i] = dd[0]
            else:
                lci[i] = max(dd[:, 0])
    return lci


def wang_uci(n, alpha, N):
    """Wang method: Calculating upper bound of 1-alpha interval."""
    lci = wang_lci(n, alpha, N)
    uci = [N - i for i in lci]
    uci.reverse()
    return uci


def wang_cal(n, alpha, N):
    """
    Wang method: Calculating needed intervals.

    Two-sided 1-alpha interval and two-sided 1-2alpha interval as well
    as initialize the final CI interval value
    """
    lcin1 = wang_lci(n, alpha / 2, N)
    ucin1 = wang_uci(n, alpha / 2, N)   # two-sided 1-alpha interval
    lcin2 = wang_lci(n, alpha, N)
    ucin2 = wang_uci(n, alpha, N)   # two-sided 1-2alpha interval

    lciw = lcin1  # lcin1<=lciw<=lcin2
    uciw = ucin1  # ucin2<=uciw<=ucin1
    return lcin1, ucin1, lcin2, ucin2, lciw, uciw


def wang_ind(x, a, b):
    """Wang method: indicator fucntion of interval."""
    return (x >= a) * (x <= b)


def wang_cpci(n, N, M, lciw, uciw):
    """Wang method: the coverage probability function."""
    kk = list(range(len(M)))
    for ii in kk:
        indp = list(range(n + 1))
        for j in range(n + 1):
            indp[j] = wang_ind(M[ii], lciw[j], uciw[j])\
                * hypergeom.pmf(j, N, M[ii], n)
        kk[ii] = sum(indp)
    return kk


def wang_hypergeom_conf(n, x, N, alpha):
    """
    Wang method for Confidence interval for a hypergeometric distribution.

    CI for parameter G, the number of good objects in a population in size
    N, based on the number x of good objects in a simple random sample of
    size n.

    Parameters
    ----------
    n : int
        The number of draws without replacement.
    x : int
        The number of "good" objects in the sample.
    N : int
        The number of objects in the population.
    alpha : float in (0, 1)
        1-alpha will be the desired confidence level.

    Returns
    -------
    lc: int
        lower confidence level with coverage (at least)
        1-alpha.
    uc: int
        upper confidence level with coverage (at least)
        1-alpha.

    Notes
    -----
    G : int in [0, N]
        Starting point in search for confidence bounds for the
        hypergeometric parameter G.
    """
    lcin1, ucin1, lcin2, ucin2, lciw, uciw = wang_cal(n, alpha, N)
    if n % 2 == 0:
        xvalue = int(n / 2)  # even n
    else:
        xvalue = int((n + 1) / 2)  # odd n
    aa = list(range(int(lciw[xvalue]), int(N / 2 + 1)))

    for i in range(len(aa)):
        lciw[xvalue] = aa[i]
        uciw[xvalue] = N - aa[i]
        M = list(range(N + 1))
        bb = min(wang_cpci(n, N, M, lciw, uciw))
        if bb >= 1 - alpha:
            i1 = i
        else:
            break

    lciw[xvalue] = aa[i1]
    uciw[xvalue] = N - lciw[xvalue]

    if n % 2 == 0:
        xvalue = int(n / 2 - 1)  # even n
    else:
        xvalue = int((n + 1) / 2 - 1)  # odd n

    while xvalue >= 0:
        al = int(lcin2[xvalue] - lciw[xvalue] + 1)
        au = int(uciw[xvalue] - ucin2[xvalue] + 1)

        if (al * au > 1):
            ff = np.zeros(shape=(al*au, 4))
            for i in range(al):
                ff[(i * au):((i + 1) * au), 0] = lciw[xvalue] + i
                ff[(i * au):((i + 1) * au), 1] = list(range(int(ucin2[xvalue]),
                                                      (int(uciw[xvalue]) + 1)))
                ff[(i * au):((i + 1) * au), 2] = (ff[i * au:((i + 1) * au), 1]
                                                  - ff[i * au:((i + 1) * au),
                                                       0])

            for ii in range(al * au):
                lciw[xvalue] = ff[ii, 0]
                uciw[xvalue] = ff[ii, 1]
                lciw[n + 2 - xvalue - 2] = N - uciw[xvalue]
                uciw[n + 2 - xvalue - 2] = N - lciw[xvalue]
                M = list(range(N + 1))
                ff[ii, 3] = min(wang_cpci(n, N, M, lciw, uciw))

            ff = np.array([k for k in ff if k[3] >= 1-alpha])

            if (len(ff) > 1):
                index = np.lexsort(ff.T[:3, :])
                ff = ff[index, :]
                lciw[xvalue] = ff[0, 0]
                uciw[xvalue] = ff[0, 1]
            else:
                lciw[xvalue] = ff[0, 0]
                uciw[xvalue] = ff[0, 1]
            lciw[n + 2 - xvalue - 2] = N - uciw[xvalue]
            uciw[n + 2 - xvalue - 2] = N - lciw[xvalue]

        xvalue = xvalue - 1
    lc = lciw[x]
    uc = uciw[x]
    return lc, uc


@lru_cache(maxsize=None)
def hypergeom_accept(k, M, n, N, alpha=0.05, randomized=False):
    """
    Acceptance region for a randomized hypergeometric test.

    If randomized==True, find the acceptance region for a randomized, exact
    level-alpha test of the null hypothesis X~Binomial(n,p).
    The acceptance region is the smallest possible.
    (And not, for instance, symmetric.)

    If randomized==False, find the smallest conservative acceptance region.

    Parameters
    ----------
    k : integer
        number of "good items" in sample
    M : integer
        size of population
    n: integer
        number of "good items" in population
    N: integer
        size of sample
    alpha : float
        desired significance level
    ramndomized : Boolean
        return randomized exact test or conservative non-randomized test?

    Returns
    --------
    If randomized:
    never_rejects : list
        values for which the test never rejects
    J : list
        values for which the test sometimes rejects
    gamma : float
        probability the test does not reject when the value is in J

    If not randomized:
    never_rejects : list
        values for which the test does not reject

    """
    assert 0 < alpha < 1, "bad significance level"
    x = np.arange(0, n+1)
    never_rejects = list(x)  # start w/ all possible outcomes (then remove)
    pmf = hypergeom.pmf(x, M, n, N)  # "frozen" hypergeometric pmf
    bottom = 0  # smallest outcome still in never_rejects
    top = n  # largest outcome still in never_rejects
    J = []  # outcomes for which the test is randomized
    p_J = 0  # probability of outcomes for which test is randomized
    p_tail = 0  # probability of outcomes excluded from never_rejects

    while p_tail < alpha:  # need to remove outcomes from the acceptance region
        pb = pmf[bottom]
        pt = pmf[top]
        if pb < pt:  # the smaller possibility has smaller probability
            J = [bottom]
            p_J = pb
            bottom += 1
        elif pb > pt:  # the larger possibility has smaller probability
            J = [top]
            p_J = pt
            top -= 1
        else:
            if bottom < top:  # the two possibilities have equal probability
                J = [bottom, top]
                p_J = pb+pt
                bottom += 1
                top -= 1
            else:  # there is only one possibility left
                J = [bottom]
                p_J = pb
                bottom += 1
        p_tail += p_J
        for j in J:  # remove outcomes from acceptance region
            never_rejects.remove(j)
    return_val = None
    if randomized:
        gamma = (p_tail-alpha)/p_J  # probability of accepting H_0 when X in J
        # to get exact level alpha
        return_val = never_rejects, J, gamma
    else:
        while p_tail > alpha:
            j = J.pop()  # move the outcome into the acceptance region
            p_tail -= pmf[j]
            never_rejects.append(j)
        return_val = never_rejects
    return return_val


def sterne_hypergeom_conf(N, n, x, cl=0.95):
    """
    two-sided confidence bound for a binomial p.

    Assumes x is a draw from a hypergeometric distribution with parameters
    N (known), n (known), and G (unknown). Finds a lower confidence bound for G
    at confidence level cl.

    Parameters
    ----------
    N : int
        population size, nonnegative integer
    n : int
        number of trials, nonnegative integer <= N
    x : int
        observed number of successes, nonnegative integer <= n
    cl : float
        confidence level, between 0 and 1

    Returns
    -------
    lb : float
        lower confidence bound
    ub : float
        upper confidence bound
    """
    assert 0 <= x <= n, 'impossible arguments'
    assert n <= N, 'impossible sample size'
    assert 0 < cl < 1, 'silly confidence level'
    lb = 0
    ub = N
    alpha = 1-cl
    if x > 0:
        while x not in hypergeom_accept(x, N, lb, n, alpha,  randomized=False):
            lb += 1
        lb -= 1
    if x < n:
        while x not in hypergeom_accept(x, N, ub, n, alpha, randomized=False):
            ub -= 1
        ub += 1
    return lb, ub


def hypergeometric(x, N, n, G, alternative='greater'):
    """
    Hypergeometric function.

    Parameters
    ----------
    x : int
        number of `good` elements observed in the sample
    N : int
        population size
    n : int
       sample size
    G : int
       hypothesized number of good elements in population
    alternative : {'greater', 'less', 'two-sided'}
       alternative hypothesis to test (default: 'greater')
    Returns
    -------
    float
       estimated p-value
    """
    if n < x:
        raise ValueError("Cannot observe more good elements "
                         + "than the sample size")
    if N < n:
        raise ValueError("Population size cannot be smaller than sample")
    if N < G:
        raise ValueError("Number of good elements can't "
                         + "exceed the population size")
    if G < x:
        raise ValueError("Number of observed good elements can't "
                         + "exceed the number in the population")

    assert alternative in ("two-sided", "less", "greater")
    if n < x:
        raise ValueError("Cannot observe more successes "
                         + "than the population size")

    plower = hypergeom.cdf(x, N, G, n)
    pupper = hypergeom.sf(x-1, N, G, n)
    if alternative == 'two-sided':
        pvalue = 2*np.min([plower, pupper, 0.5])
    elif alternative == 'greater':
        pvalue = pupper
    elif alternative == 'less':
        pvalue = plower
    return pvalue


def binomial_p(x, n, p, alternative='greater'):
    """
    Binomial.

    Parameters
    ----------
    x : array-like
       list of elements consisting of x in {0, 1} where 0
       represents a failure and 1 represents a success
    p : int
       hypothesized number of successes in n trials
    n : int
       number of trials
    alternative : {'greater', 'less', 'two-sided'}
       alternative hypothesis to test (default: 'greater')
    Returns
    -------
    float
       estimated p-value
    """
    assert alternative in ("two-sided", "less", "greater")
    if n < x:
        raise ValueError("Cannot observe more successes "
                         + "than the population size")

    plower = binom.cdf(x, n, p)
    pupper = binom.sf(x-1, n, p)
    if alternative == 'two-sided':
        pvalue = 2*np.min([plower, pupper, 0.5])
    elif alternative == 'greater':
        pvalue = pupper
    elif alternative == 'less':
        pvalue = plower
    return pvalue


def get_prng(seed=None):
    """Turn seed into a cryptorandom instance.

    Parameters
    ----------
    seed : {None, int, str, RandomState}
        If seed is None, return generate a pseudo-random
        63-bit seed using np.random and return a new
        SHA256 instance seeded with it. If seed is a
        number or str, return a new cryptorandom
        instance seeded with seed.If seed is already a
        numpy.random RandomState or SHA256 instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    RandomState
    """
    if seed is None:
        # Need to specify dtype (Windows defaults to int32)
        seed = np.random.randint(0, 10**10, dtype=np.int64)  # generate integer
    if seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer, float, str)):
        return SHA256(seed)
    if isinstance(seed, (np.random.RandomState, SHA256)):
        return seed
    raise ValueError('%r cannot be used to seed cryptorandom' % seed)


def permute_within_groups(x, group, seed=None):
    """
    Permutation of condition within each group.

    Parameters
    ----------
    x : array-like
        A 1-d array indicating treatment.
    group : array-like
        A 1-d array indicating group membership
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

    Returns
    -------
    permuted : array-like
        The within group permutation of x.
    """
    prng = get_prng(seed)
    permuted = x.copy()

    for g in np.unique(group):
        gg = group == g
        permuted[gg] = random_permutation(permuted[gg], prng=prng)
    return permuted


def permute(x, seed=None):
    """
    Permute an array in-place.

    Parameters
    ----------
    x : array-like
        A 1-d array
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

    Returns
    -------
    None
        Original array is permuted in-place, nothing is returned.
    """
    return random_permutation(x, prng=seed)


def permute_rows(m, seed=None):
    """
    Permute the rows of a matrix in-place.

    Parameters
    ----------
    m : array-like
        A 2-d array
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

    Returns
    -------
    None
        Original matrix is permuted in-place, nothing is returned.
    """
    prng = get_prng(seed)

    mprime = []
    for row in m:
        mprime.append(random_permutation(row, prng=prng))
    return np.array(mprime)


def permute_incidence_fixed_sums(incidence, k=1, seed=None):
    """
    Permute elements of a (binary) incidence matrix.

    Keeps the row and column sums in-tact.

    Parameters
    ----------
    incidence : 2D ndarray
        Incidence matrix to permute.
    k : int
        The number of successful pairwise swaps to perform.
    seed : RandomState instance or {None, int, RandomState instance}
        If None, the pseudorandom number generator is the RandomState
        instance used by `np.random`;
        If int, seed is the seed used by the random number generator;
        If RandomState instance, seed is the pseudorandom number generator

    Notes
    -----
    The row and column sums are kept fixed by always swapping elements
    two pairs at a time.

    Returns
    -------
    permuted : 2D ndarray
        The permuted incidence matrix.
    """
    if not incidence.ndim == 2:
        raise ValueError("Incidence matrix must be 2D")

    if incidence.min() != 0 or incidence.max() != 1:
        raise ValueError("Incidence matrix must be binary")

    prng = get_prng(seed)

    incidence = incidence.copy()
    n, m = incidence.shape
    rows = np.arange(n)
    cols = np.arange(m)
    K, k = k, 0

    while k < K:
        swappable = False
        while not swappable:
            chosen_rows = np.random.choice(rows, 2, replace=False)
            s0, s1 = chosen_rows

            potential_cols0, = np.where((incidence[s0, :] == 1) &
                                        (incidence[s1, :] == 0))

            potential_cols1, = np.where((incidence[s0, :] == 0) &
                                        (incidence[s1, :] == 1))

            potential_cols0 = np.setdiff1d(potential_cols0, potential_cols1)

            if (len(potential_cols0) == 0) or (len(potential_cols1) == 0):
                continue

            p0 = prng.choice(potential_cols0)
            p1 = prng.choice(potential_cols1)

            # These statements should always be true, so we should
            # never raise an assertion here
            assert incidence[s0, p0] == 1
            assert incidence[s0, p1] == 0
            assert incidence[s1, p0] == 0
            assert incidence[s1, p1] == 1
            swappable = True
        i0 = incidence.copy()
        incidence[[s0, s0, s1, s1],
                  [p0, p1, p0, p1]] = [0, 1, 1, 0]
        k += 1
    return incidence


def potential_outcomes(x, y, f, finverse):
    """
    Potential Outcomes.

    Given observations $x$ under treatment and $y$ under control conditions,
    returns the potential outcomes for units under their unobserved condition
    under the hypothesis that $x_i = f(y_i)$ for all units.

    Parameters
    ----------
    x : array-like
        Outcomes under treatment
    y : array-like
        Outcomes under control
    f : function
        An invertible function
    finverse : function
        The inverse function to f.

    Returns
    -------
    potential_outcomes : 2D array
        The first column contains all potential outcomes under the treatment,
        the second column contains all potential outcomes under the control.
    """
    tester = np.array(range(5)) + 1
    assert np.allclose(finverse(f(tester)),
                       tester), "f and finverse aren't inverses"
    assert np.allclose(f(finverse(tester)),
                       tester), "f and finverse aren't inverses"

    pot_treat = np.concatenate([x, f(y)])
    pot_ctrl = np.concatenate([finverse(x), y])

    return np.column_stack([pot_treat, pot_ctrl])
