from numpy import array, zeros
from numpy.testing import assert_allclose

from glimix_core.cov import FreeFormCov, Kron2SumCov, LRFreeFormCov
from optimix import Assertion


def test_kron2sumcov_optimix():
    # the first element is its id
    item0 = array([0, -1.5, 1.0])
    item1 = array([1, +1.24, 1.0])
    item2 = array([2, -1.5, 1.0])

    Cr = FreeFormCov(2)
    Cr.L = [[1, 0], [2, 1]]

    Cn = LRFreeFormCov(3, 2)
    Cn.L = [[1, 0], [2, 1], [-1, 3]]

    cov = Kron2SumCov(Cr, Cn)
    K = cov.value(item0, item1)
    assert_allclose(K, [[-0.86, -1.72], [-1.72, -4.3]])

    K = cov.value(item0, item0)
    assert_allclose(K, [[4.25, 8.5], [8.5, 21.25]])

    K = cov.value(item0, item2)
    assert_allclose(K, [[3.25, 6.5], [6.5, 16.25]])

    value_example = zeros((2, 2))
    a = Assertion(lambda: cov, item0, item1, value_example)
    a.assert_layout()
