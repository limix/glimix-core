from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_equal

from glimix_core._util import assert_interface
from glimix_core.lmm import RKron2Sum


def test_lmm_rkron2sum():
    random = RandomState(0)
    n = 5
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = RKron2Sum(Y, A, F, G)

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
    lmm = RKron2Sum(Y, A, F, G)
    lmm.name = "KronSum"

    assert_allclose(lmm.lml(), -4.582089407009583)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_allclose(
        [lmm.mean.value()[0], lmm.mean.value()[1]],
        [0.0497438970225256, 0.5890598193072355],
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
    assert_allclose(lmm.lml(), -0.6930197958236421)


def test_lmm_rkron2sum_public_attrs():
    assert_interface(
        RKron2Sum,
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
        ],
    )
