import scipy.stats as st
from brent_search import minimize
from numpy import concatenate, empty, kron
from numpy.random import default_rng
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs

from glimix_core._util import assert_interface, vec
from glimix_core.lmm import FastScanner, Kron2Sum, KronFastScanner


def test_lmm_kron_scan():
    random = default_rng(0)
    n = 20
    Y = random.normal(size=(n, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 6))
    lmm = Kron2Sum(Y, A, F, G, restricted=True)
    lmm.fit(verbose=False)
    scan = lmm.get_fast_scanner()

    m = lmm.mean()
    K = lmm.covariance()

    def func0(scale):
        mv = st.multivariate_normal(m, scale * K)
        return -mv.logpdf(vec(Y))

    s = minimize(func0, 1e-3, 5.0, 1e-5)[0]

    assert_allclose(scan.null_lml(), st.multivariate_normal(m, s * K).logpdf(vec(Y)))
    assert_allclose(kron(A, F) @ scan.null_beta, m)

    A1 = random.normal(size=(3, 2))
    F1 = random.normal(size=(n, 4))

    r = scan.scan(A1, F1)
    assert_allclose(r["scale"], 0.7432575497, rtol=1e-3)

    m = kron(A, F) @ vec(r["effsizes0"]) + kron(A1, F1) @ vec(r["effsizes1"])

    def func1(scale):
        mv = st.multivariate_normal(m, scale * K)
        return -mv.logpdf(vec(Y))

    s = minimize(func1, 1e-3, 5.0, 1e-5)[0]

    assert_allclose(r["lml"], st.multivariate_normal(m, s * K).logpdf(vec(Y)))

    r = scan.scan(empty((3, 0)), F1)
    assert_allclose(r["lml"], -67.2013784411, rtol=1e-4)
    assert_allclose(r["scale"], 0.8999995537936586, rtol=1e-3)
    assert_allclose(
        r["effsizes0"],
        [
            [-0.1463056513, -0.8234240129, -0.1653985457],
            [0.164938537, 1.0066226167, 0.1740701106],
        ],
        rtol=1e-2,
        atol=1e-2,
    )
    assert_allclose(r["effsizes1"], [])


def test_lmm_kron_scan_with_lmm():
    random = default_rng(0)
    n = 15
    Y = random.normal(size=(n, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 6))

    klmm = Kron2Sum(Y, A, F, G, restricted=True)
    klmm.fit(verbose=False)
    kscan = klmm.get_fast_scanner()

    K = klmm.covariance()

    X = kron(A, F)
    QS = economic_qs(K)
    scan = FastScanner(vec(Y), X, QS, 0.0)

    assert_allclose(klmm.covariance(), K)
    assert_allclose(kscan.null_scale, scan.null_scale)
    assert_allclose(kscan.null_beta, scan.null_beta)
    assert_allclose(kscan.null_lml(), scan.null_lml())
    assert_allclose(kscan.null_beta_covariance, scan.null_beta_covariance)

    A1 = random.normal(size=(3, 2))
    F1 = random.normal(size=(n, 2))
    M = kron(A1, F1)

    kr = kscan.scan(A1, F1)
    r = scan.scan(M)
    assert_allclose(kr["lml"], r["lml"])
    assert_allclose(kr["scale"], r["scale"])
    assert_allclose(vec(kr["effsizes0"]), r["effsizes0"])
    assert_allclose(vec(kr["effsizes1"]), r["effsizes1"])
    assert_allclose(vec(kr["effsizes0_se"]), r["effsizes0_se"])
    assert_allclose(vec(kr["effsizes1_se"]), r["effsizes1_se"])


def test_lmm_kron_scan_unrestricted():
    random = default_rng(0)
    n = 15
    Y = random.normal(size=(n, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 6))
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm.fit(verbose=False)
    scan = lmm.get_fast_scanner()

    assert_allclose(scan.null_scale, 1.0, rtol=1e-3)
    assert_allclose(lmm.beta_covariance, scan.null_beta_covariance, rtol=1e-3)


def test_lmm_kron_scan_redundant():
    random = default_rng(0)
    n = 30
    Y = random.normal(size=(n, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 6))
    G = concatenate([G, G], axis=1)
    lmm = Kron2Sum(Y, A, F, G, restricted=True)
    lmm.fit(verbose=False)
    scan = lmm.get_fast_scanner()

    m = lmm.mean()
    K = lmm.covariance()

    def func0(scale):
        mv = st.multivariate_normal(m, scale * K)
        return -mv.logpdf(vec(Y))

    s = minimize(func0, 1e-3, 5.0, 1e-5)[0]

    assert_allclose(scan.null_lml(), st.multivariate_normal(m, s * K).logpdf(vec(Y)))
    assert_allclose(kron(A, F) @ scan.null_beta, m)

    A1 = random.normal(size=(3, 2))
    F1 = random.normal(size=(n, 4))
    F1 = concatenate([F1, F1], axis=1)

    r = scan.scan(A1, F1)
    assert_allclose(r["scale"], 0.8558704225, rtol=1e-3)

    m = kron(A, F) @ vec(r["effsizes0"]) + kron(A1, F1) @ vec(r["effsizes1"])

    def func1(scale):
        mv = st.multivariate_normal(m, scale * K)
        return -mv.logpdf(vec(Y))

    s = minimize(func1, 1e-3, 5.0, 1e-5)[0]

    assert_allclose(r["lml"], st.multivariate_normal(m, s * K).logpdf(vec(Y)))


def test_lmm_kron_scan_public_attrs():
    assert_interface(
        KronFastScanner,
        ["null_lml", "scan"],
        ["null_beta", "null_beta_covariance", "null_beta_se", "null_scale"],
    )
