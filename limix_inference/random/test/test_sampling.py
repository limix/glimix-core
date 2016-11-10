from __future__ import division

from numpy.random import RandomState
from numpy.testing import (assert_equal, assert_array_less)

from limix_inference.random import GLMMSampler
from limix_inference.random import bernoulli_sample
from limix_inference.random import binomial_sample
from limix_inference.random import poisson_sample
from limix_inference.mean import OffsetMean
from limix_inference.cov import LinearCov
from limix_inference.cov import EyeCov
from limix_inference.cov import SumCov
from limix_inference.lik import BinomialLik
from limix_inference.lik import PoissonLik
from limix_inference.link import LogitLink
from limix_inference.link import LogLink
from limix_inference.fruits import Apples


def test_binomial_sampler():
    random = RandomState(4503)
    link = LogitLink()
    binom = BinomialLik(12, link)
    assert_equal(binom.sample(0, random), 7)


def test_poisson_sampler():
    random = RandomState(4503)
    link = LogLink()
    poisson = PoissonLik(link)
    assert_equal(poisson.sample(0, random), 1)
    assert_equal(poisson.sample(0, random), 0)
    assert_equal(poisson.sample(0, random), 2)
    assert_equal(poisson.sample(0, random), 0)
    assert_equal(poisson.sample(-10, random), 0)
    assert_equal(poisson.sample(+5, random), 158)


def test_GLMMSampler_poisson():
    random = RandomState(4503)
    X = random.randn(10, 15)
    link = LogLink()
    lik = PoissonLik(link)

    mean = OffsetMean()
    mean.offset = 1.2
    mean.set_data(10, 'sample')
    cov = LinearCov()
    cov.set_data((X, X), 'sample')
    sampler = GLMMSampler(lik, mean, cov)
    assert_equal(
        sampler.sample(random), [0, 289, 0, 11, 0, 0, 176, 0, 228, 82])

    mean = OffsetMean()
    mean.offset = 0.0
    mean.set_data(10, 'sample')

    cov1 = LinearCov()
    cov1.set_data((X, X), 'sample')

    cov2 = EyeCov()
    a = Apples(10)
    cov2.set_data((a, a), 'sample')

    cov1.scale = 1e-4
    cov2.scale = 1e-4

    cov = SumCov([cov1, cov2])

    sampler = GLMMSampler(lik, mean, cov)

    assert_equal(sampler.sample(random), [2, 0, 1, 2, 1, 1, 1, 2, 0, 0])

    cov2.scale = 100.
    sampler = GLMMSampler(lik, mean, cov)
    assert_equal(sampler.sample(random), [0, 0, 0, 0, 1, 0, 0, 1196, 0, 0])


def test_GLMMSampler_binomial():
    random = RandomState(4503)
    X = random.randn(10, 15)
    link = LogitLink()
    lik = BinomialLik(5, link)

    mean = OffsetMean()
    mean.offset = 1.2
    mean.set_data(10, 'sample')
    cov = LinearCov()
    cov.set_data((X, X), 'sample')
    sampler = GLMMSampler(lik, mean, cov)
    assert_equal(sampler.sample(random), [0, 5, 0, 5, 1, 1, 5, 0, 5, 5])

    mean.offset = 0.
    assert_equal(sampler.sample(random), [5, 4, 1, 0, 0, 1, 4, 5, 5, 0])

    mean = OffsetMean()
    mean.offset = 0.0
    mean.set_data(10, 'sample')

    cov1 = LinearCov()
    cov1.set_data((X, X), 'sample')

    cov2 = EyeCov()
    a = Apples(10)
    cov2.set_data((a, a), 'sample')

    cov1.scale = 1e-4
    cov2.scale = 1e-4

    cov = SumCov([cov1, cov2])

    lik = BinomialLik(100, link)
    sampler = GLMMSampler(lik, mean, cov)
    assert_equal(
        sampler.sample(random), [56, 56, 55, 51, 59, 45, 47, 43, 51, 38])

    cov2.scale = 100.
    sampler = GLMMSampler(lik, mean, cov)
    assert_equal(
        sampler.sample(random), [99, 93, 99, 75, 77, 0, 0, 100, 99, 12])


def test_canonical_bernoulli_sampler():
    random = RandomState(9)
    G = random.randn(10, 5)

    y = bernoulli_sample(0.1, G, random_state=random)
    assert_array_less(y, [2] * 10)


def test_canonical_binomial_sampler():
    random = RandomState(9)
    G = random.randn(10, 5)

    y = binomial_sample(5, 0.1, G, random_state=random)
    assert_array_less(y, [5 + 1] * 10)

    ntrials = [2, 3, 1, 1, 4, 5, 1, 2, 1, 1]
    y = binomial_sample(ntrials, -0.1, G, random_state=random)
    assert_array_less(y, [i + 1 for i in ntrials])


def test_canonical_poisson_sampler():
    random = RandomState(9)
    G = random.randn(10, 5)

    y = poisson_sample(0.1, G, random_state=random)
    assert_array_less(y, [20] * len(y))


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
