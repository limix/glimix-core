from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core.lmm import RKron2Sum


def test_lmm_reml_rkron2sum():
    random = RandomState(0)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    lmm = RKron2Sum(Y, A, F, G)

    assert_allclose(lmm.lml(), -31.187923426130165)
