from __future__ import division

from numpy import sqrt, std, ascontiguousarray

from ..link import LogitLink
from ..link import LogLink

from ..lik import BernoulliProdLik
from ..lik import BinomialProdLik
from ..lik import PoissonProdLik

from ..mean import OffsetMean
from ..mean import LinearMean
from ..mean import SumMean

from ..cov import LinearCov
from ..cov import SumCov
from ..cov import EyeCov

from ._glmm import GLMMSampler

from ..fruits import Apples


def bernoulli_sample(offset,
                     G,
                     heritability=0.5,
                     causal_variants=None,
                     causal_variance=0,
                     random_state=None):
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


    """
    link = LogitLink()
    mean, cov = _mean_cov(offset, G, heritability, causal_variants,
                          causal_variance, random_state)
    lik = BernoulliProdLik(link)
    sampler = GLMMSampler(lik, mean, cov)

    return sampler.sample(random_state)


def binomial_sample(ntrials,
                    offset,
                    G,
                    heritability=0.5,
                    causal_variants=None,
                    causal_variance=0,
                    random_state=None):
    """Binomial likelihood sampling."""
    link = LogitLink()
    mean, cov = _mean_cov(offset, G, heritability, causal_variants,
                          causal_variance, random_state)
    lik = BinomialProdLik(ntrials, link)
    sampler = GLMMSampler(lik, mean, cov)

    return sampler.sample(random_state)


def poisson_sample(offset,
                   G,
                   heritability=0.5,
                   causal_variants=None,
                   causal_variance=0,
                   random_state=None):
    """Poisson likelihood sampling."""
    mean, cov = _mean_cov(offset, G, heritability, causal_variants,
                          causal_variance, random_state)
    link = LogLink()
    lik = PoissonProdLik(link)
    sampler = GLMMSampler(lik, mean, cov)

    return sampler.sample(random_state)


def _causal_mean(causal_variants, causal_variance, random):
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
    mean = LinearMean(p)
    mean.set_data((causal_variants, ), 'sample')
    mean.effsizes = directions
    return mean


def _mean_cov(offset, G, heritability, causal_variants, causal_variance,
              random_state):
    G = ascontiguousarray(G, dtype=float)
    nsamples = G.shape[0]
    G = _stdnorm(G, axis=0)

    G /= sqrt(G.shape[1])

    mean1 = OffsetMean()
    mean1.offset = offset

    cov1 = LinearCov()
    cov2 = EyeCov()
    cov = SumCov([cov1, cov2])

    mean1.set_data(nsamples, 'sample')
    cov1.set_data((G, G), 'sample')
    a = Apples(nsamples)
    cov2.set_data((a, a), 'sample')

    cov1.scale = heritability - causal_variance
    cov2.scale = 1 - heritability - causal_variance

    means = [mean1]
    if causal_variants is not None:
        means += [_causal_mean(causal_variants, causal_variance, random_state)]

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

    if out.ndim == 1:
        if s > 0:
            out /= s
    else:
        out[..., ok] /= s[ok]

    return out
