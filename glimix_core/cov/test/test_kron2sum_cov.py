from numpy import array, stack, zeros
from numpy.linalg import slogdet
from numpy.testing import assert_, assert_allclose

from glimix_core.cov import FreeFormCov, Kron2SumCov, LRFreeFormCov
from optimix import Assertion


def test_kron2sumcov_optimix():
    # the first element is its id
    item0 = array([0, -1.5, 1.0])
    item1 = array([1, +1.24, 1.0])
    item2 = array([2, -1.5, 1.0])

    cov = Kron2SumCov(2, 1)
    cov.Cr.L = [[3], [2]]
    cov.Cn.L = [[1, 0], [2, 1]]

    K = cov.value(item0, item1)
    assert_allclose(K, [[-7.74, -5.16], [-5.16, -3.44]])

    K = cov.value(item0, item0)
    assert_allclose(K, [[30.25, 21.5], [21.5, 18.0]])

    K = cov.value(item0, item2)
    assert_allclose(K, [[29.25, 19.5], [19.5, 13.0]])

    value_example = zeros((2, 2))
    a = Assertion(lambda: cov, item0, item1, value_example)
    a.assert_layout()
    a.assert_gradient()

    G = stack([item0, item1, item2], axis=0)
    cov.G = G
    assert_allclose(
        cov.feed().value(),
        [
            [
                [[30.25, 21.5], [21.5, 18.0]],
                [[-7.74, -5.16], [-5.16, -3.44]],
                [[29.25, 19.5], [19.5, 13.0]],
            ],
            [
                [[-7.74, -5.16], [-5.16, -3.44]],
                [[23.8384, 17.2256], [17.2256, 15.1504]],
                [[-7.74, -5.16], [-5.16, -3.44]],
            ],
            [
                [[29.25, 19.5], [19.5, 13.0]],
                [[-7.74, -5.16], [-5.16, -3.44]],
                [[30.25, 21.5], [21.5, 18.0]],
            ],
        ],
    )


def test_kron2sumcov_logdet():
    item0 = array([0, -1.5, 1.0])
    item1 = array([1, +1.24, 1.0])
    item2 = array([2, -1.5, 1.0])
    G = stack([item0, item1, item2], axis=0)

    cov = Kron2SumCov(2, 1)
    cov.Cr.L = [[1], [2]]
    cov.Cn.L = [[3, 0], [2, 1]]

    cov.G = G
    K = cov.feed().value()
    K = [[K[0, 0, 0, 0], K[0, 1, 0, 0]], [K[1, 0, 0, 0], K[1, 1, 0, 0]]]
    assert_allclose(cov.compact_value()[:2][:, :2], K)
#     # ld = slogdet(cov.compact_value())
#     # assert_(ld[0] == 1)
#     # assert_allclose(cov.logdet(), ld[1])
