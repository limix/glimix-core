import scipy.stats as st
from numpy import concatenate, kron
from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core._util import vec
from glimix_core.lmm import RKron2Sum


def test_lmm_kron_scan():
    random = RandomState(0)
    n = 5
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 6)
    lmm = RKron2Sum(Y, A, F, G)
    lmm.fit(verbose=False)
    scan = lmm.get_fast_scanner()

    m = lmm.mean.value()
    K = lmm.cov.value()
    assert_allclose(scan.null_lml(), st.multivariate_normal(m, K).logpdf(vec(Y)))
    assert_allclose(kron(A, F) @ scan.null_effsizes(), m)

    A1 = random.randn(3, 2)
    F1 = random.randn(n, 4)

    lml, effsizes0, effsizes1 = scan.scan(A1, F1)

    m = kron(A, F) @ vec(effsizes0) + kron(A1, F1) @ vec(effsizes1)
    assert_allclose(lml, st.multivariate_normal(m, K).logpdf(vec(Y)))


def test_lmm_kron_scan_redundant():
    random = RandomState(0)
    n = 5
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 6)
    G = concatenate([G, G], axis=1)
    lmm = RKron2Sum(Y, A, F, G)
    lmm.fit(verbose=False)
    scan = lmm.get_fast_scanner()

    m = lmm.mean.value()
    K = lmm.cov.value()
    assert_allclose(scan.null_lml(), st.multivariate_normal(m, K).logpdf(vec(Y)))
    assert_allclose(kron(A, F) @ vec(scan.null_effsizes()), m)

    A1 = random.randn(3, 2)
    F1 = random.randn(n, 4)
    F1 = concatenate([F1, F1], axis=1)

    lml, effsizes0, effsizes1 = scan.scan(A1, F1)

    m = kron(A, F) @ vec(effsizes0) + kron(A1, F1) @ vec(effsizes1)
    assert_allclose(lml, st.multivariate_normal(m, K).logpdf(vec(Y)))
