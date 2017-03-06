from __future__ import division

from numpy import array, c_, dot, ones, pi, sqrt, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose

from limix_inference.glmm import ExpFamEP
from limix_inference.lik import BinomialProdLik
from limix_inference.link import LogitLink
from numpy_sugar.linalg import economic_qs_linear


def test_binomial_get_normal_likelihood_trick():
    random = RandomState(139)
    nsamples = 30
    nfeatures = 31

    G = random.randn(nsamples, nfeatures) / sqrt(nfeatures)

    u = random.randn(nfeatures)

    z = 0.1 + 2 * dot(G, u) + random.randn(nsamples)

    ntrials = random.randint(10, 500, size=nsamples)

    y = zeros(nsamples)
    for i in range(len(ntrials)):
        y[i] = sum(
            z[i] + random.logistic(scale=pi / sqrt(3), size=ntrials[i]) > 0)
    (Q, S0) = economic_qs_linear(G)

    M = ones((nsamples, 1))
    lik = BinomialProdLik(ntrials, LogitLink())
    lik.nsuccesses = y
    ep = ExpFamEP(lik, M, Q[0], Q[1], S0)
    ep.learn(progress=False)

    nlt = ep.get_normal_likelihood_trick()
    assert_allclose(
        nlt.fast_scan(G)[0], [
            -143.48903288, -144.32031587, -144.03889888, -144.31806561,
            -143.90248659, -144.303103, -144.47854112, -144.44469341,
            -144.285027, -144.31240175, -143.11590263, -142.81623878,
            -141.67554141, -144.4780024, -144.47780285, -144.10317082,
            -142.10043322, -143.0813298, -143.99841663, -143.345783,
            -144.45458683, -144.37877612, -142.56846859, -144.32923028,
            -144.44116855, -144.45082936, -144.40932741, -143.0212886,
            -144.47902176, -143.94188634, -143.72765373
        ], rtol=1e-5)


def test_binomial_lml():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S0) = economic_qs_linear(G)
    nsuccesses = array([1., 0., 1.])
    ntrials = array([1., 1., 1.])
    lik = BinomialProdLik(ntrials, LogitLink())
    lik.nsuccesses = nsuccesses
    ep = ExpFamEP(lik, M, Q[0], Q[1], S0 + 1)
    ep.beta = array([1.])
    assert_allclose(ep.beta, array([1.]))
    ep.v = 1.
    ep.delta = 0
    assert_allclose(ep.lml(), -2.3202659215368935)


def test_binomial_gradient_over_v():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S0) = economic_qs_linear(G)
    nsuccesses = array([1., 0., 1.])
    ntrials = array([1., 1., 1.])
    lik = BinomialProdLik(ntrials, LogitLink())
    lik.nsuccesses = nsuccesses
    ep = ExpFamEP(lik, M, Q[0], Q[1], S0 + 1)
    ep.beta = array([1.])
    assert_allclose(ep.beta, array([1.]))
    ep.v = 1.
    ep.delta = 0.

    analytical_gradient = ep._gradient_over_v()

    lml0 = ep.lml()
    step = 1e-5
    ep.v = ep.v + step
    lml1 = ep.lml()

    empirical_gradient = (lml1 - lml0) / step

    assert_allclose(empirical_gradient, analytical_gradient, rtol=1e-4)

    ep.v = 0.5
    ep.delta = 0.0

    analytical_gradient = ep._gradient_over_v()

    lml0 = ep.lml()
    step = 1e-5
    ep.v = ep.v + step
    lml1 = ep.lml()

    empirical_gradient = (lml1 - lml0) / step

    assert_allclose(empirical_gradient, analytical_gradient, rtol=1e-4)


def test_binomial_gradient_over_delta():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S0) = economic_qs_linear(G)
    nsuccesses = array([1., 0., 1.])
    ntrials = array([1., 1., 1.])
    lik = BinomialProdLik(ntrials, LogitLink())
    lik.nsuccesses = nsuccesses
    ep = ExpFamEP(lik, M, Q[0], Q[1], S0 + 1.0)
    ep.beta = array([1.])
    assert_allclose(ep.beta, array([1.]))
    ep.v = 1.
    ep.delta = 0.5

    analytical_gradient = ep._gradient_over_delta()

    lml0 = ep.lml()
    step = 1e-5
    ep.delta = ep.delta + step
    lml1 = ep.lml()

    empirical_gradient = (lml1 - lml0) / step

    assert_allclose(empirical_gradient, analytical_gradient, rtol=1e-4)


def test_binomial_gradient_over_both():
    n = 3
    M = ones((n, 1)) * 1.
    G = array([[1.2, 3.4], [-.1, 1.2], [0.0, .2]])
    (Q, S0) = economic_qs_linear(G)
    nsuccesses = array([1., 0., 1.])
    ntrials = array([1., 1., 1.])
    lik = BinomialProdLik(ntrials, LogitLink())
    lik.nsuccesses = nsuccesses
    ep = ExpFamEP(lik, M, Q[0], Q[1], S0 + 1.0)
    ep.beta = array([1.])
    assert_allclose(ep.beta, array([1.]))
    ep.v = 1.5
    ep.delta = 0.3

    analytical_gradient = ep._gradient_over_both()[0]

    lml0 = ep.lml()
    step = 1e-5
    ep.v = ep.v + step
    lml1 = ep.lml()

    empirical_gradient = (lml1 - lml0) / step

    assert_allclose(empirical_gradient, analytical_gradient, rtol=1e-4)

    ep.v = 1.5
    ep.delta = 0.3

    analytical_gradient = ep._gradient_over_both()[1]

    lml0 = ep.lml()
    step = 1e-5
    ep.delta = ep.delta + step
    lml1 = ep.lml()

    empirical_gradient = (lml1 - lml0) / step

    assert_allclose(empirical_gradient, analytical_gradient, rtol=1e-4)


def test_binomial_optimize():
    random = RandomState(139)
    nsamples = 30
    nfeatures = 31

    G = random.randn(nsamples, nfeatures) / sqrt(nfeatures)

    u = random.randn(nfeatures)

    z = 0.1 + 2 * dot(G, u) + random.randn(nsamples)

    ntrials = random.randint(10, 500, size=nsamples)

    y = zeros(nsamples)
    for i in range(len(ntrials)):
        y[i] = sum(
            z[i] + random.logistic(scale=pi / sqrt(3), size=ntrials[i]) > 0)
    (Q, S0) = economic_qs_linear(G)

    M = ones((nsamples, 1))
    lik = BinomialProdLik(ntrials, LogitLink())
    lik.nsuccesses = y
    ep = ExpFamEP(lik, M, Q[0], Q[1], S0)
    ep.learn(progress=False)

    assert_allclose(ep.lml(), -144.2381842202486, rtol=1e-3)


def test_binomial_optimize_refit():
    random = RandomState(139)
    nsamples = 30
    nfeatures = 31

    G = random.randn(nsamples, nfeatures) / sqrt(nfeatures)

    u = random.randn(nfeatures)

    z = 0.1 + 2 * dot(G, u) + random.randn(nsamples)

    ntrials = random.randint(10, 500, size=nsamples)

    y = zeros(nsamples)
    for i in range(len(ntrials)):
        y[i] = sum(
            z[i] + random.logistic(scale=pi / sqrt(3), size=ntrials[i]) > 0)
    (Q, S0) = economic_qs_linear(G)

    M = ones((nsamples, 1))
    lik = BinomialProdLik(ntrials, LogitLink())
    lik.nsuccesses = y
    ep = ExpFamEP(lik, M, Q[0], Q[1], S0)
    ep.learn(progress=False)

    assert_allclose(ep.lml(), -144.2381842202486, rtol=1e-3)

    nep = ep.copy()

    assert_allclose(ep.lml(), -144.2381842202486, rtol=1e-3)
    assert_allclose(nep.lml(), -144.2381842202486, rtol=1e-3)

    nep.M = c_[M, random.randn(nsamples)]

    assert_allclose(nep.lml(), -145.7076758124364, rtol=1e-3)
    nep.learn(progress=False)
    assert_allclose(nep.lml(), -143.98475638974728, rtol=1e-3)


def test_binomial_extreme():
    import numpy as np
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = np.load(os.path.join(dir_path, 'data', 'extreme.npz'))
    from limix_inference.glmm import ExpFamEP
    from limix_inference.lik import BinomialProdLik
    from limix_inference.link import LogitLink
    link = LogitLink()
    lik = BinomialProdLik(data['ntrials'], link)
    lik.nsuccesses = data['nsuccesses']
    ep = ExpFamEP(
        lik, data['covariates'], Q0=data['Q0'], Q1=data['Q1'], S0=data['S0'])
    ep.learn(progress=False)
    assert_allclose(ep.lml(), -627.8816497398326, rtol=1e-3)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
