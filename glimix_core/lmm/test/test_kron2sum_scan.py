import scipy.stats as st
from numpy import concatenate, empty, kron
from numpy.random import RandomState
from numpy.testing import assert_allclose
from brent_search import minimize

from glimix_core._util import vec
from glimix_core.lmm import Kron2Sum


def test_lmm_kron_scan():
    random = RandomState(0)
    n = 5
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 6)
    lmm = Kron2Sum(Y, A, F, G, restricted=True)
    lmm.fit(verbose=False)
    scan = lmm.get_fast_scanner()

    m = lmm.mean()
    K = lmm.covariance()

    def func(scale):
        mv = st.multivariate_normal(m, scale * K)
        return -mv.logpdf(vec(Y))

    s = minimize(func, 1e-3, 5.0, 1e-5)[0]

    assert_allclose(scan.null_lml(), st.multivariate_normal(m, s * K).logpdf(vec(Y)))
    assert_allclose(kron(A, F) @ vec(scan.null_effsizes()), m)

    A1 = random.randn(3, 2)
    F1 = random.randn(n, 4)

    lml, effsizes0, effsizes1, scale = scan.scan(A1, F1)
    assert_allclose(scale, 0.3000748879939645, rtol=1e-3)

    m = kron(A, F) @ vec(effsizes0) + kron(A1, F1) @ vec(effsizes1)

    def func(scale):
        mv = st.multivariate_normal(m, scale * K)
        return -mv.logpdf(vec(Y))

    s = minimize(func, 1e-3, 5.0, 1e-5)[0]

    assert_allclose(lml, st.multivariate_normal(m, s * K).logpdf(vec(Y)))

    lml, effsizes0, effsizes1, scale = scan.scan(empty((3, 0)), F1)
    assert_allclose(lml, -10.96414417860732, rtol=1e-4)
    assert_allclose(scale, 0.5999931720566452, rtol=1e-3)
    assert_allclose(
        effsizes0,
        [
            [1.411082677273241, 0.41436234081257045, -1.5337251391408189],
            [-0.6753042112998789, -0.20299590400182352, 0.6723874047807074],
        ],
        rtol=1e-2,
        atol=1e-2,
    )
    assert_allclose(effsizes1, [])


def test_lmm_kron_scan_redundant():
    random = RandomState(0)
    n = 5
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 6)
    G = concatenate([G, G], axis=1)
    lmm = Kron2Sum(Y, A, F, G, restricted=True)
    lmm.fit(verbose=False)
    scan = lmm.get_fast_scanner()

    m = lmm.mean()
    K = lmm.covariance()

    def func(scale):
        mv = st.multivariate_normal(m, scale * K)
        return -mv.logpdf(vec(Y))

    s = minimize(func, 1e-3, 5.0, 1e-5)[0]

    assert_allclose(scan.null_lml(), st.multivariate_normal(m, s * K).logpdf(vec(Y)))
    assert_allclose(kron(A, F) @ vec(scan.null_effsizes()), m)

    A1 = random.randn(3, 2)
    F1 = random.randn(n, 4)
    F1 = concatenate([F1, F1], axis=1)

    lml, effsizes0, effsizes1, scale = scan.scan(A1, F1)
    assert_allclose(scale, 0.3005376956901813, rtol=1e-3)

    m = kron(A, F) @ vec(effsizes0) + kron(A1, F1) @ vec(effsizes1)

    def func(scale):
        mv = st.multivariate_normal(m, scale * K)
        return -mv.logpdf(vec(Y))

    s = minimize(func, 1e-3, 5.0, 1e-5)[0]

    assert_allclose(lml, st.multivariate_normal(m, s * K).logpdf(vec(Y)))
