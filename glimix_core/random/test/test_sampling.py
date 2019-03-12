from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_array_less, assert_equal

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.lik import BinomialProdLik, PoissonProdLik
from glimix_core.link import LogitLink, LogLink
from glimix_core.mean import OffsetMean
from glimix_core.random import (
    GGPSampler,
    GPSampler,
    bernoulli_sample,
    binomial_sample,
    poisson_sample,
)


def test_binomial_sampler():
    random = RandomState(4503)
    link = LogitLink()
    binom = BinomialProdLik(12, link)
    assert_equal(binom.sample(0, random), 7)


def test_poisson_sampler():
    random = RandomState(4503)
    link = LogLink()
    poisson = PoissonProdLik(link)
    assert_equal(poisson.sample(0, random), 1)
    assert_equal(poisson.sample(0, random), 0)
    assert_equal(poisson.sample(0, random), 2)
    assert_equal(poisson.sample(0, random), 0)
    assert_equal(poisson.sample(-10, random), 0)
    assert_equal(poisson.sample(+5, random), 158)


def test_GGPSampler_poisson():
    random = RandomState(4503)
    X = random.randn(10, 15)
    link = LogLink()
    lik = PoissonProdLik(link)

    mean = OffsetMean(10)
    mean.offset = 1.2
    cov = LinearCov(X)
    sampler = GGPSampler(lik, mean, cov)
    assert_equal(sampler.sample(random), [0, 289, 0, 11, 0, 0, 176, 0, 228, 82])

    mean = OffsetMean(10)
    mean.offset = 0.0

    cov1 = LinearCov(X)

    cov2 = EyeCov(10)

    cov1.scale = 1e-4
    cov2.scale = 1e-4

    cov = SumCov([cov1, cov2])

    sampler = GGPSampler(lik, mean, cov)

    assert_equal(sampler.sample(random), [2, 0, 1, 2, 1, 1, 1, 2, 0, 0])

    cov2.scale = 20.0
    sampler = GGPSampler(lik, mean, cov)
    assert_equal(sampler.sample(random), [0, 0, 0, 2, 0, 0, 1, 22, 0, 0])

    sampler.sample()


def test_GGPSampler_binomial():
    random = RandomState(4503)
    X = random.randn(10, 15)
    link = LogitLink()
    lik = BinomialProdLik(5, link)

    mean = OffsetMean(10)
    mean.offset = 1.2
    cov = LinearCov(X)
    sampler = GGPSampler(lik, mean, cov)
    assert_equal(sampler.sample(random), [0, 5, 0, 5, 1, 1, 5, 0, 5, 5])

    mean.offset = 0.0
    assert_equal(sampler.sample(random), [5, 4, 1, 0, 0, 1, 4, 5, 5, 0])

    mean = OffsetMean(10)
    mean.offset = 0.0

    cov1 = LinearCov(X)

    cov2 = EyeCov(10)

    cov1.scale = 1e-4
    cov2.scale = 1e-4

    cov = SumCov([cov1, cov2])

    lik = BinomialProdLik(100, link)
    sampler = GGPSampler(lik, mean, cov)
    assert_equal(sampler.sample(random), [56, 56, 55, 51, 59, 45, 47, 43, 51, 38])

    cov2.scale = 100.0
    sampler = GGPSampler(lik, mean, cov)
    assert_equal(sampler.sample(random), [99, 93, 99, 75, 77, 0, 0, 100, 99, 12])


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

    X = random.randn(len(ntrials))
    y = binomial_sample(ntrials, -0.1, G, causal_variants=X, random_state=random)
    assert_array_less(y, [i + 1 for i in ntrials])

    X = random.randn(len(ntrials), 2)
    y = binomial_sample(ntrials, -0.1, G, causal_variants=X, random_state=random)
    assert_array_less(y, [i + 1 for i in ntrials])


def test_canonical_poisson_sampler():
    random = RandomState(10)
    G = random.randn(10, 5)

    y = poisson_sample(0.1, G, random_state=random)
    assert_array_less(y, [20] * len(y))


def test_GPSampler():
    random = RandomState(4503)
    X = random.randn(10, 15)

    mean = OffsetMean(10)
    mean.offset = 1.2
    cov = LinearCov(X)
    sampler = GPSampler(mean, cov)

    x = [
        -1.34664027,
        5.68720656,
        -6.01941814,
        2.70860707,
        -1.81883975,
        0.21338045,
        5.16462597,
        -2.23134206,
        5.49661095,
        4.38526077,
    ]
    assert_allclose(sampler.sample(random), x)
    sampler.sample()
