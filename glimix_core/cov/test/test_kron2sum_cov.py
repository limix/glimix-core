from numpy import array, eye, kron, stack, zeros
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

    G = stack([i[1:] for i in [item0, item1, item2]], axis=0)
    cov.G = G
    I = eye(G.shape[0])
    assert_allclose(
        cov.compact_value(),
        kron(cov.Cr.feed().value(), cov.G @ cov.G.T) + kron(cov.Cn.feed().value(), I),
    )


def test_kron2sumcov_solve():
    item0 = array([0, -1.5, 1.0, -2.5])
    item1 = array([1, +1.24, 1.0, -1.3])
    item2 = array([2, -1.5, 1.4, 0.0])

    cov = Kron2SumCov(2, 1)
    cov.Cr.L = [[1], [2]]
    cov.Cn.L = [[3, 0], [2, 1]]

    G = stack([i[1:] for i in [item0, item1, item2]], axis=0)
    cov.G = G
    assert_allclose(cov.solve(cov.compact_value()), eye(2 * G.shape[0]), atol=1e-7)

    item0 = array([0, -1.5, 1.0])
    item1 = array([1, +1.24, 1.0])
    item2 = array([2, -1.5, 1.4])

    cov = Kron2SumCov(2, 1)
    cov.Cr.L = [[1], [2]]
    cov.Cn.L = [[3, 0], [2, 1]]

    G = stack([i[1:] for i in [item0, item1, item2]], axis=0)
    cov.G = G
    assert_allclose(cov.solve(cov.compact_value()), eye(2 * G.shape[0]), atol=1e-7)


#     # ld = slogdet(cov.compact_value())
#     # assert_(ld[0] == 1)
#     # assert_allclose(cov.logdet(), ld[1])
