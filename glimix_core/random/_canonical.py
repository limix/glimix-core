from numpy import ascontiguousarray, atleast_1d, atleast_2d, sqrt, std

from ..cov import EyeCov, LinearCov, SumCov
from ..lik import BernoulliProdLik, BinomialProdLik, PoissonProdLik
from ..link import LogitLink, LogLink
from ..mean import LinearMean, OffsetMean, SumMean
from ._ggp import GGPSampler


def bernoulli_sample(
    offset,
    G,
    heritability=0.5,
    causal_variants=None,
    causal_variance=0,
    random_state=None,
):
    r"""Bernoulli likelihood sampling.

    Sample according to

    .. math::

        \mathbf y \sim \prod_{i=1}^n
        \text{Bernoulli}(\mu_i = \text{logit}(z_i))
        \mathcal N(~ o \mathbf 1 + \mathbf a^\intercal \boldsymbol\alpha;
        ~ (h^2 - v_c)\mathrm G^\intercal\mathrm G +
        (1-h^2-v_c)\mathrm I ~)

    using the canonical Logit link function to define the conditional Bernoulli
    mean :math:`\mu_i`.

    The causal :math:`\mathbf a` covariates and the corresponding effect-sizes
    are randomly draw according to the following idea. The ``causal_variants``,
    if given, are first mean-zero and std-one normalized and then having
    its elements divided by the squared-root the the number of variances::

        causal_variants = _stdnorm(causal_variants, axis=0)
        causal_variants /= sqrt(causal_variants.shape[1])

    The causal effect-sizes :math:`\boldsymbol\alpha` are draw from
    :math:`\{-1, +1\}` and subsequently normalized for mean-zero and std-one""

    Parameters
    ----------
    random_state : random_state
        Set the initial random state.

    Example
    -------

    .. doctest::

        >>> from glimix_core.random import bernoulli_sample
        >>> from numpy.random import RandomState
        >>> offset = 5
        >>> G = [[1, -1], [2, 1]]
        >>> bernoulli_sample(offset, G, random_state=RandomState(0))
        array([1., 1.])
    """
    link = LogitLink()
    mean, cov = _mean_cov(
        offset, G, heritability, causal_variants, causal_variance, random_state
    )
    lik = BernoulliProdLik(link)
    sampler = GGPSampler(lik, mean, cov)

    return sampler.sample(random_state)


def binomial_sample(
    ntrials,
    offset,
    G,
    heritability=0.5,
    causal_variants=None,
    causal_variance=0,
    random_state=None,
):
    """Binomial likelihood sampling.

    Parameters
    ----------
    random_state : random_state
        Set the initial random state.

    Example
    -------

    .. doctest::

        >>> from glimix_core.random import binomial_sample
        >>> from numpy.random import RandomState
        >>> ntrials = [5, 15]
        >>> offset = 0.5
        >>> G = [[1, -1], [2, 1]]
        >>> binomial_sample(ntrials, offset, G, random_state=RandomState(0))
        array([ 2., 14.])
    """
    link = LogitLink()
    mean, cov = _mean_cov(
        offset, G, heritability, causal_variants, causal_variance, random_state
    )
    lik = BinomialProdLik(ntrials, link)
    sampler = GGPSampler(lik, mean, cov)

    return sampler.sample(random_state)


def poisson_sample(
    offset,
    G,
    heritability=0.5,
    causal_variants=None,
    causal_variance=0,
    random_state=None,
):
    """Poisson likelihood sampling.

    Parameters
    ----------
    random_state : random_state
        Set the initial random state.

    Example
    -------

    .. doctest::

        >>> from glimix_core.random import poisson_sample
        >>> from numpy.random import RandomState
        >>> offset = -0.5
        >>> G = [[0.5, -1], [2, 1]]
        >>> poisson_sample(offset, G, random_state=RandomState(0))
        array([0, 6])
    """
    mean, cov = _mean_cov(
        offset, G, heritability, causal_variants, causal_variance, random_state
    )
    link = LogLink()
    lik = PoissonProdLik(link)
    sampler = GGPSampler(lik, mean, cov)

    return sampler.sample(random_state)


def _causal_mean(causal_variants, causal_variance, random):
    causal_variants = atleast_2d(atleast_1d(causal_variants).T).T
    causal_variants = _stdnorm(causal_variants, axis=0)
    causal_variants /= sqrt(causal_variants.shape[1])
    p = causal_variants.shape[1]
    directions = random.randn(p)
    directions[directions < 0.5] = -1
    directions[directions >= 0.5] = +1
    s = std(directions)
    if s > 0:
        directions /= s
    directions *= sqrt(causal_variance)
    directions -= directions.mean()
    mean = LinearMean(causal_variants)
    mean.effsizes = directions
    return mean


def _mean_cov(offset, G, heritability, causal_variants, causal_variance, random_state):
    G = ascontiguousarray(G, dtype=float)
    nsamples = G.shape[0]
    G = _stdnorm(G, axis=0)

    G /= sqrt(G.shape[1])

    mean1 = OffsetMean(nsamples)
    mean1.offset = offset

    cov1 = LinearCov(G)
    cov2 = EyeCov(nsamples)
    cov = SumCov([cov1, cov2])

    cov1.scale = heritability - causal_variance
    cov2.scale = 1 - heritability - causal_variance

    means = []
    means.append(mean1)
    if causal_variants is not None:
        means.append(_causal_mean(causal_variants, causal_variance, random_state))

    mean = SumMean(means)

    return mean, cov


def _stdnorm(X, axis=None, out=None):
    X = ascontiguousarray(X, dtype=float)
    if out is None:
        out = X.copy()

    m = out.mean(axis)
    s = out.std(axis)
    ok = s > 0

    out -= m

    out[..., ok] /= s[ok]

    return out
