import pytest
import scipy.stats as st
from numpy import concatenate
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import check_grad

from glimix_core._util import assert_interface, vec
from glimix_core.lmm import Kron2Sum


def test_kron2sum_restricted():
    random = RandomState(0)
    n = 5
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=True)

    assert_allclose(lmm.lml(), -16.580821931417656)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 3)
    assert_equal(lmm.ncovariates, 2)

    n = 5
    Y = random.randn(n, 1)
    A = random.randn(1, 1)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=True)
    lmm.name = "KronSum"

    assert_allclose(lmm.lml(), -4.582089407009583)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_allclose(
        [lmm.mean[0], lmm.mean[1]], [0.0497438970225256, 0.5890598193072355]
    )

    assert_allclose(
        [
            lmm.cov.value()[0, 0],
            lmm.cov.value()[0, 1],
            lmm.cov.value()[1, 0],
            lmm.cov.value()[1, 1],
        ],
        [
            4.3712532668348185,
            -0.07239366121399138,
            -0.07239366121399138,
            2.7242131674614862,
        ],
    )

    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 1)
    assert_equal(lmm.name, "KronSum")
    lmm.fit(verbose=False)
    grad = lmm.gradient()
    assert_allclose(grad["C0.Lu"], [0], atol=1e-4)
    assert_allclose(grad["C1.Lu"], [0], atol=1e-4)
    assert_allclose(lmm.lml(), -0.6930197958236421, rtol=1e-5)


def test_kron2sum_unrestricted():
    random = RandomState(0)
    n = 5
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)

    assert_allclose(lmm.lml(), -22.700472625381742)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 3)
    assert_equal(lmm.ncovariates, 2)

    n = 5
    Y = random.randn(n, 1)
    A = random.randn(1, 1)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm.name = "KronSum"

    assert_allclose(lmm.lml(), -7.8032707190765525)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_allclose(
        [lmm.mean[0], lmm.mean[1]], [0.0497438970225256, 0.5890598193072355]
    )

    assert_allclose(
        [
            lmm.cov.value()[0, 0],
            lmm.cov.value()[0, 1],
            lmm.cov.value()[1, 0],
            lmm.cov.value()[1, 1],
        ],
        [
            4.3712532668348185,
            -0.07239366121399138,
            -0.07239366121399138,
            2.7242131674614862,
        ],
    )

    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 1)
    assert_equal(lmm.name, "KronSum")
    lmm.fit(verbose=False)
    grad = lmm.gradient()
    assert_allclose(grad["C0.Lu"], [0], atol=1e-4)
    assert_allclose(grad["C1.Lu"], [0], atol=1e-4)
    assert_allclose(lmm.lml(), 2.3394131683160992, rtol=1e-5)


def test_kron2sum_unrestricted_lml():
    random = RandomState(0)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    y = vec(lmm._Y)

    m = lmm.mean
    K = lmm.cov.value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm.cov.C0.Lu = random.randn(3)
    m = lmm.mean
    K = lmm.cov.value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm.cov.C1.Lu = random.randn(6)
    m = lmm.mean
    K = lmm.cov.value()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))


def test_kron2sum_public_attrs():
    assert_interface(
        Kron2Sum,
        [
            "fit",
            "ntraits",
            "lml",
            "mean",
            "nsamples",
            "value",
            "cov",
            "ncovariates",
            "name",
            "gradient",
            "B",
            "get_fast_scanner",
        ],
    )


def test_kron2sum_gradient_unrestricted():
    random = RandomState(2)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm.cov.C0.Lu = random.randn(3)
    lmm.cov.C1.Lu = random.randn(6)

    def func(x):
        lmm.cov.C0.Lu = x[:3]
        lmm.cov.C1.Lu = x[3:9]
        return lmm.lml()

    def grad(x):
        lmm.cov.C0.Lu = x[:3]
        lmm.cov.C1.Lu = x[3:9]
        D = lmm.gradient()
        return concatenate((D["C0.Lu"], D["C1.Lu"]))

    assert_allclose(check_grad(func, grad, random.randn(9), epsilon=1e-8), 0, atol=1e-3)


def test_kron2sum_fit_ill_conditioned_unrestricted():
    random = RandomState(0)
    n = 30
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-157.18713011032833, -122.97307224440634])
    grad = lmm.gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 9, atol=1e-3)


def test_kron2sum_fit_C1_well_cond_unrestricted():
    random = RandomState(0)
    Y = random.randn(5, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 6)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-19.12949904791771, -11.853021820832943])
    grad = lmm.gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 5, atol=1e-2)


def test_kron2sum_fit_C1_well_cond_C0_fullrank_unrestricted():
    random = RandomState(0)
    Y = random.randn(5, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 6)
    lmm = Kron2Sum(Y, A, F, G, rank=2, restricted=False)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-20.15199256730784, -11.853022074873408])
    grad = lmm.gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 7, atol=1e-2)


def test_kron2sum_fit_C1_well_cond_redutant_F_unrestricted():
    random = RandomState(0)
    Y = random.randn(5, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(5, 2)
    F = concatenate((F, F), axis=1)
    G = random.randn(5, 2)
    with pytest.warns(UserWarning):
        Kron2Sum(Y, A, F, G, restricted=False)


def test_kron2sum_fit_C1_well_cond_redundant_Y_unrestricted():
    random = RandomState(0)
    Y = random.randn(5, 2)
    Y = concatenate((Y, Y), axis=1)
    A = random.randn(4, 4)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 2)
    with pytest.warns(UserWarning):
        lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lml = lmm.lml()
    assert_allclose(lml, -40.5860882514021)
