import scipy.stats as st
from numpy import concatenate, corrcoef, dot, exp, eye, log, ones, pi, sqrt
from numpy.linalg import slogdet, solve
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import ddot, economic_qs_linear, economic_svd
from scipy.optimize import minimize

from glimix_core._util import assert_interface
from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.lik import DeltaProdLik
from glimix_core.lmm import LMM
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler


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


def test_lmm_prediction():
    random = RandomState(9458)
    n = 30

    X = random.randn(n, n + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])

    offset = 1.0

    mean = OffsetMean(n)
    mean.offset = offset

    cov_left = LinearCov(X)
    cov_left.scale = 1.5

    cov_right = EyeCov(n)
    cov_right.scale = 1.5

    cov = SumCov([cov_left, cov_right])

    lik = DeltaProdLik()

    y = GGPSampler(lik, mean, cov).sample(random)

    QS = economic_qs_linear(X)

    lmm = LMM(y, ones((n, 1)), QS)

    lmm.fit(verbose=False)

    K = dot(X, X.T)
    pm = lmm.predictive_mean(ones((n, 1)), K, K.diagonal())
    assert_allclose(corrcoef(y, pm)[0, 1], 0.8358820971891354)


def test_lmm_lmm_public_attrs():
    assert_interface(
        LMM,
        [
            "lml",
            "X",
            "beta",
            "delta",
            "scale",
            "mean_star",
            "variance_star",
            "covariance_star",
            "covariance",
            "predictive_covariance",
            "mean",
            "gradient",
            "v0",
            "v1",
            "fit",
            "value",
            "get_fast_scanner",
            "predictive_mean",
            "name",
            "unfix",
            "fix",
            "nsamples",
            "ncovariates",
        ],
    )


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
    assert_allclose(lmm.lml(), -res.fun, rtol=1e-5, atol=1e-6)
    assert_allclose(lmm.beta, res.x[:c], rtol=1e-5, atol=1e-6)
    assert_allclose(lmm.v0, exp(res.x[c]), rtol=1e-5, atol=1e-6)
    assert_allclose(lmm.v1, exp(res.x[c + 1]), rtol=1e-5, atol=1e-6)

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
    assert_allclose(lmm.lml(), -res.fun, rtol=1e-5, atol=1e-6)
    assert_allclose(lmm.v0, exp(res.x[0]), rtol=1e-5, atol=1e-6)
    assert_allclose(lmm.v1, exp(res.x[1]), rtol=1e-5, atol=1e-6)

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
    assert_allclose(lmm.beta, res.x[:c], rtol=1e-4, atol=1e-6)
    assert_allclose(lmm.delta, 1 / (1 + exp(-res.x[c])), rtol=1e-4, atol=1e-6)


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
