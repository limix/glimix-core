from __future__ import division

from numpy import array, dot, ones, sqrt, zeros, exp
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from numpy_sugar.linalg import economic_qs_linear
from limix_inference.glmm import ExpFamEP
from limix_inference.lik import PoissonProdLik
from limix_inference.link import LogLink


def test_poisson_lml():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S0) = economic_qs_linear(G)
    noccurrences = array([1., 0., 5.])
    lik = PoissonProdLik(LogLink())
    lik.noccurrences = noccurrences
    ep = ExpFamEP(lik, M, Q[0], Q[1], S0 + 1)
    ep.beta = array([1.])
    assert_almost_equal(ep.beta, array([1.]))
    ep.v = 1.
    ep.delta = 0
    assert_almost_equal(ep.lml(), -6.793765561069963)


def test_poisson_optimize():
    random = RandomState(139)
    nsamples = 30
    nfeatures = 31

    G = random.randn(nsamples, nfeatures) / sqrt(nfeatures)

    u = random.randn(nfeatures)

    z = 0.1 + 2 * dot(G, u) + random.randn(nsamples)

    y = zeros(nsamples)
    for i in range(nsamples):
        y[i] = random.poisson(lam=exp(z[i]))
    (Q0, Q1), S0 = economic_qs_linear(G)

    M = ones((nsamples, 1))
    lik = PoissonProdLik(LogLink())
    lik.noccurrences = y
    ep = ExpFamEP(lik, M, Q0, Q1, S0)
    ep.optimize()
    assert_almost_equal(ep.lml(), -77.90919831238075, decimal=2)
    assert_almost_equal(ep.beta[0], 0.314709077094, decimal=1)
    assert_almost_equal(ep.heritability, 0.797775054939, decimal=1)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
