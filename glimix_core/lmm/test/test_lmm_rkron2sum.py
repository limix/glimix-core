from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core.lmm import RKron2Sum


def test_lmm_reml_rkron2sum():
    random = RandomState(0)
    n = 5
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = RKron2Sum(Y, A, F, G)

    assert_allclose(lmm.lml(), -16.580821931417656)
    # assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-3)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)

    n = 5
    Y = random.randn(n, 1)
    A = random.randn(1, 1)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = RKron2Sum(Y, A, F, G)

    assert_allclose(lmm.lml(), -4.582089407009583)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
