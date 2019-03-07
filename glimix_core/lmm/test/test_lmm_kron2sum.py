import pytest
import scipy.stats as st
from numpy import concatenate
from numpy.random import RandomState
from numpy.testing import assert_allclose
from scipy.optimize import check_grad

from glimix_core.lmm import Kron2Sum


def test_lmm_kron2sum():
    random = RandomState(0)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    lmm = Kron2Sum(Y, A, F, G)
    y = lmm._y

    m = lmm.mean.value()
    K = lmm.cov.value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm.mean.B = random.randn(2, 3)
    m = lmm.mean.value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm.cov.C0.Lu = random.randn(3)
    K = lmm.cov.value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm.cov.C1.Lu = random.randn(6)
    K = lmm.cov.value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))


def test_lmm_kron2sum_gradient():
    random = RandomState(1)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    lmm = Kron2Sum(Y, A, F, G)
    lmm.mean.B = random.randn(2, 3)
    lmm.cov.C0.Lu = random.randn(3)
    lmm.cov.C1.Lu = random.randn(6)

    def func(x):
        lmm.cov.C0.Lu = x[:3]
        lmm.cov.C1.Lu = x[3:9]
        lmm.mean.B = x[9:].reshape((2, 3), order="F")
        return lmm.lml()

    def grad(x):
        lmm.cov.C0.Lu = x[:3]
        lmm.cov.C1.Lu = x[3:9]
        lmm.mean.B = x[9:].reshape((2, 3), order="F")
        D = lmm.lml_gradient()
        return concatenate((D["C0.Lu"], D["C1.Lu"], D["M.vecB"]))

    assert_allclose(
        check_grad(func, grad, random.randn(15), epsilon=1e-8), 0, atol=1e-3
    )


def test_lmm_kron2sum_fit_ill_conditioned():
    random = RandomState(0)
    n = 30
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-161.0096475502658, -122.97307222429187])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 15, atol=1e-3)


def test_lmm_kron2sum_fit_C1_well_cond():
    random = RandomState(0)
    Y = random.randn(5, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 6)
    lmm = Kron2Sum(Y, A, F, G)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-21.459910525411757, -11.853021674440884])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 9, atol=1e-4)


def test_lmm_kron2sum_fit_C1_well_cond_C0_fullrank():
    random = RandomState(0)
    Y = random.randn(5, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 6)
    lmm = Kron2Sum(Y, A, F, G, rank=2)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-22.44348328519218, -11.853021674565952])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 11, atol=1e-3)


# def test_lmm_kron2sum_fit_C1_well_cond_redutant_G():
#     random = RandomState(0)
#     Y = random.randn(5, 2)
#     A = random.randn(2, 2)
#     A = A @ A.T
#     F = random.randn(5, 2)
#     G = random.randn(5, 2)
#     G = concatenate((G, G), axis=1)
#     lmm = Kron2Sum(Y, A, F, G)
#     lml0 = lmm.lml()
#     lmm.fit(verbose=False)
#     lml1 = lmm.lml()
#     assert_allclose([lml0, lml1], [-19.905930526833977, 2.9779698820017453])
#     grad = lmm.lml_gradient()
#     vars = grad.keys()
#     assert_allclose(concatenate([grad[var] for var in vars]), [0] * 9, atol=1e-2)


def test_lmm_kron2sum_fit_C1_well_cond_redutant_F():
    random = RandomState(0)
    Y = random.randn(5, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(5, 2)
    F = concatenate((F, F), axis=1)
    G = random.randn(5, 2)
    lmm = Kron2Sum(Y, A, F, G)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-19.423945522925802, 2.9779698820395035])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 13, atol=1e-2)


def test_lmm_kron2sum_fit_C1_well_cond_redundant_Y():
    random = RandomState(0)
    Y = random.randn(5, 2)
    Y = concatenate((Y, Y), axis=1)
    A = random.randn(4, 4)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 2)
    with pytest.warns(UserWarning):
        lmm = Kron2Sum(Y, A, F, G)
    lml = lmm.lml()
    assert_allclose(lml, -43.906141655466485)
