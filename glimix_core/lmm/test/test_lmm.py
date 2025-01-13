import pytest
import scipy.stats as st
from numpy import concatenate, exp, eye, log, pi
from numpy.linalg import slogdet, solve
from numpy.random import Generator, default_rng
from numpy.testing import assert_allclose, assert_equal
from numpy_sugar.linalg import ddot, economic_qs_linear, economic_svd
from scipy.optimize import minimize

from glimix_core._util import assert_interface
from glimix_core.lmm import LMM


def test_lmm():
    random = default_rng(12)

    (y, X, G) = _full_rank(random)
    _test_lmm(random, y, X, G, _get_mvn(y, X, G), False)

    (y, X, G) = _low_rank(random)
    _test_lmm(random, y, X, G, _get_mvn(y, X, G), False)

    (y, X, G) = _full_rank(random)
    _test_lmm(random, y, X, G, _get_mvn_restricted(y, X, G), True)

    (y, X, G) = _low_rank(random)
    _test_lmm(random, y, X, G, _get_mvn_restricted(y, X, G), True)


def test_lm():
    random = default_rng(0)

    (y, X, _) = _full_rank(random)
    lmm = LMM(y, X)
    lmm.fit(verbose=False)
    assert_allclose(lmm.v0, 2.0129061033356781e-16, atol=1e-7)
    assert_allclose(lmm.v1, 0.8108150965)
    assert_allclose(lmm.beta, [0.3225896208, -0.2917152789])


def test_lmm_beta_covariance():
    random = default_rng(0)

    (y, X, G) = _full_rank(random)
    QS = economic_qs_linear(G)
    lmm = LMM(y, X, QS)
    lmm.fit(verbose=False)

    A = [[0.0016815025, -0.0001486849], [-0.0001486849, 0.0023484078]]
    assert_allclose(lmm.beta_covariance, A, rtol=1e-6)

    (y, X, G) = _low_rank(random)
    QS = economic_qs_linear(G)
    lmm = LMM(y, X[:, :2], QS)
    lmm.fit(verbose=False)

    A = [[0.0037514935, -0.00114132], [-0.00114132, 0.006491444]]
    assert_allclose(lmm.beta_covariance, A)

    (y, X, G) = _low_rank(random)
    QS = economic_qs_linear(G)
    lmm = LMM(y, X, QS)
    lmm.fit(verbose=False)

    A = [
        [0.0042288753, 0.0016227908, 0.0042288753, 0.0016227908],
        [0.0016227908, 0.0080978309, 0.0016227908, 0.0080978309],
        [0.0042288753, 0.0016227908, 0.0042288753, 0.0016227908],
        [0.0016227908, 0.0080978309, 0.0016227908, 0.0080978309],
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
    random = default_rng(1)
    n = 3
    G = random.normal(size=(n, n + 1))
    X = random.normal(size=(n, 2))
    y = (
        X @ random.normal(size=2)
        + G @ random.normal(size=G.shape[1])
        + random.normal(size=n)
    )
    y -= y.mean(0)
    y /= y.std(0)

    QS = economic_qs_linear(G)
    lmm = LMM(y, X, QS, restricted=False)
    lmm.name = "lmm"
    lmm.fit(verbose=False)

    assert_allclose(
        lmm.covariance(),
        [
            [1.0454767788, -0.1024992983, -0.1347709001],
            [-0.1024992983, 0.6609585597, 0.3069073261],
            [-0.1347709001, 0.3069073261, 0.2085936602],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.mean(),
        [-0.259134127, -1.4618988298, 0.5657654358],
        atol=1e-7,
    )
    assert_allclose(lmm.lml(), -2.6467351496, atol=1e-7)
    assert_allclose(lmm.value(), lmm.lml(), atol=1e-7)
    assert_allclose(lmm.lml(), -2.6467351496, atol=1e-7)
    assert_allclose(
        lmm.X,
        [
            [-0.736454087, -0.162909948],
            [-0.4821193127, 0.5988462126],
            [0.0397221075, -0.292456751],
        ],
        atol=1e-7,
    )
    assert_allclose(lmm.beta, [0.757055469, -1.8317019366], atol=1e-7)
    assert_allclose(
        lmm.beta_covariance,
        [[1.2672251036, 0.4685773896], [0.4685773896, 0.3343485385]],
        atol=1e-7,
    )
    assert_allclose(lmm.delta, 2.2204460493e-16, atol=1e-7)
    assert_equal(lmm.ncovariates, 2)
    assert_equal(lmm.nsamples, 3)
    assert_allclose(lmm.scale, 0.4018140195, atol=1e-7)
    assert_allclose(lmm.v0, 0.4018140195, atol=1e-7)
    assert_allclose(lmm.v1, 8.9220635204e-17, atol=1e-7)
    assert_equal(lmm.name, "lmm")

    with pytest.raises(NotImplementedError):
        lmm.gradient()


def _test_lmm(random: Generator, y, X, G, mvn, restricted):
    c = X.shape[1]
    QS = economic_qs_linear(G)
    lmm = LMM(y, X, QS, restricted=restricted)
    beta = lmm.beta
    v0 = lmm.v0
    v1 = lmm.v1

    K0 = G @ G.T
    assert_allclose(lmm.lml(), mvn(beta, v0, v1, y, X, K0))

    beta = random.normal(size=c)
    lmm.beta = beta
    assert_allclose(lmm.lml(), mvn(beta, v0, v1, y, X, K0))

    delta = random.uniform(size=1).item()
    lmm.delta = delta
    v0 = lmm.v0
    v1 = lmm.v1
    assert_allclose(lmm.lml(), mvn(beta, v0, v1, y, X, K0))

    scale = random.random(size=1).item()
    lmm.scale = scale
    v0 = lmm.v0
    v1 = lmm.v1
    assert_allclose(lmm.lml(), mvn(beta, v0, v1, y, X, K0))

    def fun0(x):
        beta = x[:c]
        v0 = exp(x[c])
        v1 = exp(x[c + 1])
        return -mvn(beta, v0, v1, y, X, K0)

    res = minimize(fun0, [0] * c + [0, 0])
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), -res.fun, rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.beta, res.x[:c], rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.v0, exp(res.x[c]), rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.v1, exp(res.x[c + 1]), rtol=1e-3, atol=1e-6)

    lmm = LMM(y, X, QS, restricted=restricted)
    beta = random.normal(size=c)
    lmm.beta = beta
    lmm.delta = random.random(size=1).item()
    lmm.scale = random.random(size=1).item()
    lmm.fix("beta")

    def fun1(x):
        v0 = exp(x[0])
        v1 = exp(x[1])
        return -mvn(beta, v0, v1, y, X, K0)

    res = minimize(fun1, [0, 0])
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), -res.fun, rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.v0, exp(res.x[0]), rtol=1e-3, atol=1e-6)
    assert_allclose(lmm.v1, exp(res.x[1]), rtol=1e-3, atol=1e-6)

    lmm = LMM(y, X, QS, restricted=restricted)
    lmm.beta = random.normal(size=c)
    delta = random.random(size=1).item()
    lmm.delta = delta
    lmm.scale = random.random(size=1).item()
    lmm.fix("delta")

    def fun2(x):
        beta = x[:c]
        scale = exp(x[c])
        v0 = scale * (1 - delta)
        v1 = scale * delta
        return -mvn(beta, v0, v1, y, X, K0)

    res = minimize(fun2, [0] * c + [0])
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), -res.fun, rtol=1e-5, atol=1e-6)
    assert_allclose(lmm.beta, res.x[:c], rtol=1e-5, atol=1e-6)
    assert_allclose(lmm.scale, exp(res.x[c]), rtol=1e-5, atol=1e-6)

    lmm = LMM(y, X, QS, restricted=restricted)
    lmm.beta = random.normal(size=c)
    lmm.delta = random.random(size=1).item()
    scale = random.random(size=1).item()
    lmm.scale = scale
    lmm.fix("scale")

    def fun3(x):
        beta = x[:c]
        delta = 1 / (1 + exp(-x[c]))
        v0 = scale * (1 - delta)
        v1 = scale * delta
        return -mvn(beta, v0, v1, y, X, K0)

    res = minimize(fun3, [0] * c + [0])
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
    G = random.normal(size=(n, n + 1))
    X = random.normal(size=(n, 2))
    y = (
        X @ random.normal(size=2)
        + G @ random.normal(size=G.shape[1])
        + random.normal(size=n)
    )
    y -= y.mean(0)
    y /= y.std(0)

    return (y, X, G)


def _low_rank(random):
    n = 30
    G = random.normal(size=(n, 5))
    X = random.normal(size=(n, 2))
    X = concatenate((X, X), axis=1)
    y = (
        X @ random.normal(size=4)
        + G @ random.normal(size=G.shape[1])
        + random.normal(size=n)
    )
    y -= y.mean(0)
    y /= y.std(0)
    return (y, X, G)
