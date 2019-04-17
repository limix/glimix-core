import pytest
import scipy.stats as st
from numpy import concatenate, exp, eye, log, pi
from numpy.linalg import slogdet, solve
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_equal
from numpy_sugar.linalg import ddot, economic_qs_linear, economic_svd
from scipy.optimize import minimize

from glimix_core._util import assert_interface
from glimix_core.lmm import LMM


def test_lmm():
    random = RandomState(12)

    (y, X, G) = _full_rank(random)
    _test_lmm(random, y, X, G, _get_mvn(y, X, G), False)

    (y, X, G) = _low_rank(random)
    _test_lmm(random, y, X, G, _get_mvn(y, X, G), False)

    (y, X, G) = _full_rank(random)
    _test_lmm(random, y, X, G, _get_mvn_restricted(y, X, G), True)

    (y, X, G) = _low_rank(random)
    _test_lmm(random, y, X, G, _get_mvn_restricted(y, X, G), True)


def test_lm():
    random = RandomState(0)

    (y, X, _) = _full_rank(random)
    lmm = LMM(y, X)
    lmm.fit(verbose=False)
    assert_allclose(lmm.v0, 2.0129061033356781e-16, atol=1e-7)
    assert_allclose(lmm.v1, 0.9065323176914355)
    assert_allclose(lmm.beta, [0.24026567104188318, -0.17873180599015123])


def test_lmm_beta_covariance():
    random = RandomState(0)

    (y, X, G) = _full_rank(random)
    QS = economic_qs_linear(G)
    lmm = LMM(y, X, QS)
    lmm.fit(verbose=False)

    A = [
        [0.015685784760937037, 0.006509918649859495],
        [0.006509918649859495, 0.007975242272006645],
    ]
    assert_allclose(lmm.beta_covariance, A)

    (y, X, G) = _low_rank(random)
    QS = economic_qs_linear(G)
    lmm = LMM(y, X[:, :2], QS)
    lmm.fit(verbose=False)

    A = [
        [0.002763268929325623, 0.0006651810010328699],
        [0.0006651810010328708, 0.0016910004907565248],
    ]
    assert_allclose(lmm.beta_covariance, A)

    (y, X, G) = _low_rank(random)
    QS = economic_qs_linear(G)
    lmm = LMM(y, X, QS)
    lmm.fit(verbose=False)

    A = [
        [
            0.003892850639339253,
            0.0012112513279299796,
            0.003892850639339256,
            0.0012112513279299794,
        ],
        [
            0.0012112513279299794,
            0.009340423857663259,
            0.0012112513279299833,
            0.009340423857663257,
        ],
        [
            0.0038928506393392562,
            0.0012112513279299835,
            0.003892850639339259,
            0.0012112513279299833,
        ],
        [
            0.0012112513279299794,
            0.009340423857663257,
            0.0012112513279299833,
            0.009340423857663257,
        ],
    ]
    assert_allclose(lmm.beta_covariance, A)


def test_lmm_public_attrs():
    callables = [
        "covariance",
        "fit",
        "fix",
        "get_fast_scanner",
        "gradient",
        "lml",
        "mean",
        "unfix",
        "value",
    ]
    properties = [
        "X",
        "beta",
        "beta_covariance",
        "delta",
        "name",
        "ncovariates",
        "nsamples",
        "scale",
        "v0",
        "v1",
    ]
    assert_interface(LMM, callables, properties)


def test_lmm_interface():
    random = RandomState(1)
    n = 3
    G = random.randn(n, n + 1)
    X = random.randn(n, 2)
    y = X @ random.randn(2) + G @ random.randn(G.shape[1]) + random.randn(n)
    y -= y.mean(0)
    y /= y.std(0)

    QS = economic_qs_linear(G)
    lmm = LMM(y, X, QS, restricted=False)
    lmm.name = "lmm"
    lmm.fit(verbose=False)

    assert_allclose(
        lmm.covariance(),
        [
            [0.436311031439718, 2.6243891396439837e-16, 2.0432156171727483e-16],
            [2.6243891396439837e-16, 0.4363110314397185, 4.814313140426306e-16],
            [2.0432156171727483e-16, 4.814313140426305e-16, 0.43631103143971817],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.mean(),
        [0.6398184791042468, -0.8738254794097052, 0.7198112606871158],
        atol=1e-7,
    )
    assert_allclose(lmm.lml(), -3.012715726960625, atol=1e-7)
    assert_allclose(lmm.value(), lmm.lml(), atol=1e-7)
    assert_allclose(lmm.lml(), -3.012715726960625, atol=1e-7)
    assert_allclose(
        lmm.X,
        [
            [-0.3224172040135075, -0.38405435466841564],
            [1.1337694423354374, -1.0998912673140309],
            [-0.17242820755043575, -0.8778584179213718],
        ],
        atol=1e-7,
    )
    assert_allclose(lmm.beta, [-1.3155159120000266, -0.5615702941530938], atol=1e-7)
    assert_allclose(
        lmm.beta_covariance,
        [
            [0.44737305797088345, 0.20431961864892412],
            [0.20431961864892412, 0.29835835133251526],
        ],
        atol=1e-7,
    )
    assert_allclose(lmm.delta, 0.9999999999999998, atol=1e-7)
    assert_equal(lmm.ncovariates, 2)
    assert_equal(lmm.nsamples, 3)
    assert_allclose(lmm.scale, 0.43631103143971767, atol=1e-7)
    assert_allclose(lmm.v0, 9.688051060046502e-17, atol=1e-7)
    assert_allclose(lmm.v1, 0.43631103143971756, atol=1e-7)
    assert_equal(lmm.name, "lmm")

    with pytest.raises(NotImplementedError):
        lmm.gradient()


def _test_lmm(random, y, X, G, mvn, restricted):
    c = X.shape[1]
    QS = economic_qs_linear(G)
    lmm = LMM(y, X, QS, restricted=restricted)
    beta = lmm.beta
    v0 = lmm.v0
    v1 = lmm.v1

    K0 = G @ G.T
    assert_allclose(lmm.lml(), mvn(beta, v0, v1, y, X, K0))

    beta = random.randn(c)
    lmm.beta = beta
    assert_allclose(lmm.lml(), mvn(beta, v0, v1, y, X, K0))

    delta = random.rand(1).item()
    lmm.delta = delta
    v0 = lmm.v0
    v1 = lmm.v1
    assert_allclose(lmm.lml(), mvn(beta, v0, v1, y, X, K0))

    scale = random.rand(1).item()
    lmm.scale = scale
    v0 = lmm.v0
    v1 = lmm.v1
    assert_allclose(lmm.lml(), mvn(beta, v0, v1, y, X, K0))

    def fun(x):
        beta = x[:c]
        v0 = exp(x[c])
        v1 = exp(x[c + 1])
        return -mvn(beta, v0, v1, y, X, K0)

    res = minimize(fun, [0] * c + [0, 0])
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), -res.fun, rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.beta, res.x[:c], rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.v0, exp(res.x[c]), rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.v1, exp(res.x[c + 1]), rtol=1e-3, atol=1e-6)

    lmm = LMM(y, X, QS, restricted=restricted)
    beta = random.randn(c)
    lmm.beta = beta
    lmm.delta = random.rand(1).item()
    lmm.scale = random.rand(1).item()
    lmm.fix("beta")

    def fun(x):
        v0 = exp(x[0])
        v1 = exp(x[1])
        return -mvn(beta, v0, v1, y, X, K0)

    res = minimize(fun, [0, 0])
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), -res.fun, rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.v0, exp(res.x[0]), rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.v1, exp(res.x[1]), rtol=1e-3, atol=1e-6)

    lmm = LMM(y, X, QS, restricted=restricted)
    lmm.beta = random.randn(c)
    delta = random.rand(1).item()
    lmm.delta = delta
    lmm.scale = random.rand(1).item()
    lmm.fix("delta")

    def fun(x):
        beta = x[:c]
        scale = exp(x[c])
        v0 = scale * (1 - delta)
        v1 = scale * delta
        return -mvn(beta, v0, v1, y, X, K0)

    res = minimize(fun, [0] * c + [0])
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), -res.fun, rtol=1e-5, atol=1e-6)
    assert_allclose(lmm.beta, res.x[:c], rtol=1e-5, atol=1e-6)
    assert_allclose(lmm.scale, exp(res.x[c]), rtol=1e-5, atol=1e-6)

    lmm = LMM(y, X, QS, restricted=restricted)
    lmm.beta = random.randn(c)
    lmm.delta = random.rand(1).item()
    scale = random.rand(1).item()
    lmm.scale = scale
    lmm.fix("scale")

    def fun(x):
        beta = x[:c]
        delta = 1 / (1 + exp(-x[c]))
        v0 = scale * (1 - delta)
        v1 = scale * delta
        return -mvn(beta, v0, v1, y, X, K0)

    res = minimize(fun, [0] * c + [0])
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), -res.fun, rtol=1e-5, atol=1e-6)
    assert_allclose(lmm.beta, res.x[:c], rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.delta, 1 / (1 + exp(-res.x[c])), rtol=1e-3, atol=1e-6)


def _get_mvn(y, X, G):
    def mvn(beta, v0, v1, y, X, K0):
        m = X @ beta
        K = v0 * K0 + v1 * eye(K0.shape[0])
        return st.multivariate_normal(m, K).logpdf(y)

    return mvn


def _get_mvn_restricted(y, X, G):
    SVD = economic_svd(X)
    tX = ddot(SVD[0], SVD[1])

    def mvn_restricted(beta, v0, v1, y, X, K0):
        n = K0.shape[0]
        m = X @ beta
        K = v0 * K0 + v1 * eye(K0.shape[0])
        lml = st.multivariate_normal(m, K).logpdf(y)
        lml += slogdet(tX.T @ tX)[1] / 2 - slogdet(tX.T @ solve(K, tX))[1] / 2
        lml += n * log(2 * pi) / 2
        lml -= (n - tX.shape[1]) * log(2 * pi) / 2
        return lml

    return mvn_restricted


def _full_rank(random):
    n = 30
    G = random.randn(n, n + 1)
    X = random.randn(n, 2)
    y = X @ random.randn(2) + G @ random.randn(G.shape[1]) + random.randn(n)
    y -= y.mean(0)
    y /= y.std(0)

    return (y, X, G)


def _low_rank(random):
    n = 30
    G = random.randn(n, 5)
    X = random.randn(n, 2)
    X = concatenate((X, X), axis=1)
    y = X @ random.randn(4) + G @ random.randn(G.shape[1]) + random.randn(n)
    y -= y.mean(0)
    y /= y.std(0)
    return (y, X, G)
