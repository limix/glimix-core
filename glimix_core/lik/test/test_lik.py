from __future__ import unicode_literals

from numpy.random import RandomState
from numpy.testing import assert_, assert_allclose

from glimix_core.lik import (
    BernoulliProdLik,
    BinomialProdLik,
    DeltaProdLik,
    PoissonProdLik,
)
from glimix_core.link import ProbitLink


def test_delta_prod_lik():
    random = RandomState(0)

    lik = DeltaProdLik(ProbitLink())

    assert_(lik.name == "Delta")

    lik.outcome = [1, 0, 1]
    assert_allclose(lik.outcome, [1, 0, 1])

    assert_(lik.sample_size == 3)

    assert_allclose(lik.mean([-1, 0, 0.5]), [-1, 0, 0.5])
    assert_allclose(lik.sample([-10, 0, 0.5], random), [-10, 0, 0.5])


def test_bernoulli_prod_lik():
    random = RandomState(0)

    lik = BernoulliProdLik(ProbitLink())

    assert_(lik.name == "Bernoulli")

    lik.outcome = [1, 0, 1]
    assert_allclose(lik.outcome, [1, 0, 1])

    assert_(lik.sample_size == 3)
    assert_allclose(lik.mean([-1, 0, 0.5]), [0.15865525, 0.5, 0.69146246])
    assert_allclose(lik.sample([-10, 0, 0.5], random), [0, 1, 1])


def test_binomial_prod_lik():
    random = RandomState(0)

    lik = BinomialProdLik([6, 2, 3], ProbitLink())
    assert_allclose(lik.ntrials, [6, 2, 3])

    assert_(lik.name == "Binomial")

    lik.nsuccesses = [4, 0, 1]
    assert_allclose(lik.nsuccesses, [4, 0, 1])

    assert_(lik.sample_size == 3)
    assert_allclose(lik.mean([-1, 0, 0.5]), [0.15865525, 0.5, 0.69146246])
    assert_allclose(lik.sample([-10, 0, 0.5], random), [0, 1, 2])


def test_poisson_prod_lik():
    random = RandomState(0)

    lik = PoissonProdLik(ProbitLink())

    assert_(lik.name == "Poisson")

    lik.noccurrences = [1, 4, 3]
    assert_allclose(lik.noccurrences, [1, 4, 3])

    assert_(lik.sample_size == 3)
    assert_allclose(lik.mean([-1, 0, 0.5]), [0.15865525, 0.5, 0.69146246])
    assert_allclose(lik.sample([-10, 0, 0.5], random), [0, 1, 1])

    lik = PoissonProdLik()

    assert_(lik.name == "Poisson")

    lik.noccurrences = [1, 4, 3]
    assert_allclose(lik.noccurrences, [1, 4, 3])

    assert_(lik.sample_size == 3)
    assert_allclose(
        lik.mean([-1, 0, 0.5]), [0.36787944117144233, 1.0, 1.6487212707001282]
    )
    assert_allclose(lik.sample([-10, 0, 0.5], random), [0, 3, 4])
