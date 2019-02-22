import scipy.stats as st
from numpy import concatenate
from numpy.random import RandomState
from numpy.testing import assert_allclose
from scipy.optimize import check_grad

from glimix_core.lmm import Kron2Sum
from glimix_core.util import vec


def test_kron2sum_lmm():
    random = RandomState(0)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    lmm = Kron2Sum(Y, A, F, G)
    y = lmm._y

    m = vec(lmm._mean.feed().value())
    K = lmm.cov.compact_value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm.mean.B = random.randn(2, 3)
    m = vec(lmm._mean.feed().value())
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm.variables().get("Cr_Lu").value = random.randn(3)
    K = lmm.cov.compact_value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm.variables().get("Cn_Llow").value = random.randn(3)
    K = lmm.cov.compact_value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm.variables().get("Cn_Llogd").value = random.randn(3)
    K = lmm.cov.compact_value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))


def test_kron2sum_lmm_gradient():
    random = RandomState(0)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    lmm = Kron2Sum(Y, A, F, G)
    lmm.mean.B = random.randn(2, 3)
    lmm.variables().get("Cr_Lu").value = random.randn(3)
    lmm.variables().get("Cn_Llow").value = random.randn(3)
    lmm.variables().get("Cn_Llogd").value = random.randn(3)

    def func(x):
        lmm.variables().get("Cr_Lu").value = x[:3]
        lmm.variables().get("Cn_Llow").value = x[3:6]
        lmm.variables().get("Cn_Llogd").value = x[6:]
        return lmm.lml()

    def grad(x):
        lmm.variables().get("Cr_Lu").value = x[:3]
        lmm.variables().get("Cn_Llow").value = x[3:6]
        lmm.variables().get("Cn_Llogd").value = x[6:]
        D = lmm.lml_gradient()
        return concatenate((D["Cr_Lu"], D["Cn_Llow"], D["Cn_Llogd"]))

    assert_allclose(check_grad(func, grad, random.randn(9), epsilon=1e-8), 0, atol=1e-2)


def test_kron2sum_lmm_fit_ill_conditioned():
    random = RandomState(0)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    lmm = Kron2Sum(Y, A, F, G)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-26.65532748835924, -13.870685577153672])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 9, atol=1e-3)

    random = RandomState(0)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    G = concatenate((G, G), axis=1)
    lmm = Kron2Sum(Y, A, F, G)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-27.165209858854023, -13.87070328181693])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 9, atol=1e-3)


def test_kron2sum_lmm_fit_Cn_well_cond():
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
    assert_allclose([lml0, lml1], [-21.459910525411757, -16.059333723514342])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 5, atol=1e-6)


def test_kron2sum_lmm_fit_Cn_well_cond_Cr_fullrank():
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
    assert_allclose([lml0, lml1], [-22.44348328519218, -16.059333723514357])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 7, atol=1e-6)


def test_kron2sum_lmm_fit_Cn_well_cond_redutant_G():
    random = RandomState(0)
    Y = random.randn(5, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 2)
    G = concatenate((G, G), axis=1)
    lmm = Kron2Sum(Y, A, F, G)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-19.905930526833977, -15.136265111585097])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 5, atol=1e-6)


def test_kron2sum_lmm_fit_Cn_well_cond_redutant_F():
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
    assert_allclose([lml0, lml1], [-19.423945522925802, -15.136265111599853])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 5, atol=1e-4)


def test_kron2sum_lmm_fit_Cn_well_cond_redutant_Y():
    random = RandomState(0)
    Y = random.randn(5, 2)
    Y = concatenate((Y, Y), axis=1)
    A = random.randn(4, 4)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 2)
    lmm = Kron2Sum(Y, A, F, G)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-43.906141655466485, 26.855864863521624])
    grad = lmm.lml_gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 14, atol=1e-2)
