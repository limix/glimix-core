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

    lmm.cov.Cr.Lu = random.randn(3)
    K = lmm.cov.compact_value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm.cov.Cn.Lu = random.randn(6)
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
    lmm.cov.Cr.Lu = random.randn(3)
    lmm.cov.Cn.Lu = random.randn(6)

    def func(x):
        n = lmm.cov.Cr.Lu.shape[0]
        lmm.cov.Cr.Lu = x[:n]
        lmm.cov.Cn.Lu = x[n:]
        return lmm.lml()

    def grad(x):
        n = lmm.cov.Cr.Lu.shape[0]
        lmm.cov.Cr.Lu = x[:n]
        lmm.cov.Cn.Lu = x[n:]
        D = lmm.lml_gradient()
        return concatenate((D["Cr_Lu"], D["Cn_Lu"]))

    assert_allclose(check_grad(func, grad, random.randn(9), epsilon=1e-7), 0, atol=1e-3)


# def test_kron2sum_lmm_fit():
#     random = RandomState(0)
#     Y = random.randn(5, 3)
#     A = random.randn(3, 3)
#     A = A @ A.T
#     F = random.randn(5, 2)
#     G = random.randn(5, 4)
#     lmm = Kron2Sum(Y, A, F, G)
#     lmm.fit()
