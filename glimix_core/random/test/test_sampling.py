import pytest
from numpy.random import default_rng
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
    random = default_rng(4503)
    link = LogitLink()
    binom = BinomialProdLik(12, link)
    assert_equal(binom.sample(0, random), 4.0)


def test_poisson_sampler():
    random = default_rng(4503)
    link = LogLink()
    poisson = PoissonProdLik(link)
    assert_equal(poisson.sample(0, random), 0.0)
    assert_equal(poisson.sample(0, random), 1.0)
    assert_equal(poisson.sample(0, random), 2.0)
    assert_equal(poisson.sample(0, random), 2.0)
    assert_equal(poisson.sample(-10, random), 0.0)
    assert_equal(poisson.sample(+5, random), 160.0)


@pytest.mark.skip(reason="non-deterministic")
def test_GGPSampler_poisson():
    random = default_rng(4503)
    X = random.normal(size=(10, 15))
    link = LogLink()
    lik = PoissonProdLik(link)

    mean = OffsetMean(10)
    mean.offset = 1.2
    cov = LinearCov(X)
    sampler = GGPSampler(lik, mean, cov)
    assert_equal(
        sampler.sample(random), [1.0, 0.0, 0.0, 10.0, 2.0, 618.0, 17.0, 3.0, 20.0, 9.0]
    )

    mean = OffsetMean(10)
    mean.offset = 0.0

    cov1 = LinearCov(X)

    cov2 = EyeCov(10)

    cov1.scale = 1e-4
    cov2.scale = 1e-4

    cov = SumCov([cov1, cov2])

    sampler = GGPSampler(lik, mean, cov)

    assert_equal(
        sampler.sample(random), [2.0, 2.0, 0.0, 3.0, 0.0, 7.0, 1.0, 1.0, 1.0, 1.0]
    )

    cov2.scale = 20.0
    sampler = GGPSampler(lik, mean, cov)
    assert_equal(
        sampler.sample(random), [12.0, 0.0, 29.0, 0.0, 0.0, 0.0, 38.0, 5.0, 0.0, 0.0]
    )

    sampler.sample()


@pytest.mark.skip(reason="non-deterministic")
def test_GGPSampler_binomial():
    random = default_rng(4503)
    X = random.normal(size=(10, 15))
    link = LogitLink()
    lik = BinomialProdLik(5, link)

    mean = OffsetMean(10)
    mean.offset = 1.2
    cov = LinearCov(X)
    sampler = GGPSampler(lik, mean, cov)
    assert_equal(
        sampler.sample(random), [2.0, 1.0, 3.0, 5.0, 4.0, 5.0, 5.0, 4.0, 5.0, 5.0]
    )

    mean.offset = 0.0
    assert_equal(
        sampler.sample(random), [0.0, 3.0, 1.0, 0.0, 5.0, 4.0, 0.0, 4.0, 1.0, 1.0]
    )

    mean = OffsetMean(10)
    mean.offset = 0.0

    cov1 = LinearCov(X)

    cov2 = EyeCov(10)

    cov1.scale = 1e-4
    cov2.scale = 1e-4

    cov = SumCov([cov1, cov2])

    lik = BinomialProdLik(100, link)
    sampler = GGPSampler(lik, mean, cov)
    assert_equal(
        sampler.sample(random),
        [40.0, 39.0, 50.0, 47.0, 56.0, 61.0, 54.0, 54.0, 50.0, 52.0],
    )

    cov2.scale = 100.0
    sampler = GGPSampler(lik, mean, cov)
    assert_equal(
        sampler.sample(random),
        [98.0, 0.0, 100.0, 44.0, 100.0, 90.0, 100.0, 99.0, 22.0, 16.0],
    )


def test_canonical_bernoulli_sampler():
    random = default_rng(9)
    G = random.normal(size=(10, 5))

    y = bernoulli_sample(0.1, G, random_state=random)
    assert_array_less(y, [2] * 10)


def test_canonical_binomial_sampler():
    random = default_rng(9)
    G = random.normal(size=(10, 5))

    y = binomial_sample(5, 0.1, G, random_state=random)
    assert_array_less(y, [5 + 1] * 10)

    ntrials = [2, 3, 1, 1, 4, 5, 1, 2, 1, 1]
    y = binomial_sample(ntrials, -0.1, G, random_state=random)
    assert_array_less(y, [i + 1 for i in ntrials])

    X = random.normal(size=len(ntrials))
    y = binomial_sample(ntrials, -0.1, G, causal_variants=X, random_state=random)
    assert_array_less(y, [i + 1 for i in ntrials])

    X = random.normal(size=(len(ntrials), 2))
    y = binomial_sample(ntrials, -0.1, G, causal_variants=X, random_state=random)
    assert_array_less(y, [i + 1 for i in ntrials])


def test_canonical_poisson_sampler():
    random = default_rng(10)
    G = random.normal(size=(10, 5))

    y = poisson_sample(0.1, G, random_state=random)
    assert_array_less(y, [20] * len(y))


@pytest.mark.skip(reason="non-deterministic")
def test_GPSampler():
    random = default_rng(4503)
    X = random.normal(size=(10, 15))

    mean = OffsetMean(10)
    mean.offset = 1.2
    cov = LinearCov(X)
    sampler = GPSampler(mean, cov)

    x = [
        -0.9328779739,
        -1.997157809,
        0.5276438923,
        2.3730973762,
        1.2032887445,
        6.4469744126,
        2.7004521738,
        0.5853558975,
        2.8682819171,
        2.2248157174,
    ]
    assert_allclose(sampler.sample(random), x)
    sampler.sample()
