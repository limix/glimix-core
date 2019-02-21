from numpy import array, eye, kron, stack, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_
from numpy.linalg import slogdet

from glimix_core.cov import Kron2SumCov
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

def test_kron2sumcov_compact_value():
    item0 = array([0, -1.5, 1.0, -2.5])
    item1 = array([1, +1.24, 1.0, -1.3])
    item2 = array([2, -1.5, 1.4, 0.0])

    cov = Kron2SumCov(2, 1)
    cov.Cr.L = [[1], [2]]
    cov.Cn.L = [[3, 0], [2, 1]]

    G = stack([i[1:] for i in [item0, item1, item2]], axis=0)
    cov.G = G
    assert_allclose(cov.solve(cov.compact_value()), eye(2 * G.shape[0]), atol=1e-7)


def test_kron2sumcov_solve():
    def test_for_G(G):
        cov = Kron2SumCov(2, 1)
        cov.Cr.L = [[1], [2]]
        cov.Cn.L = [[3, 0], [2, 1]]
        cov.G = G
        assert_allclose(cov.solve(cov.compact_value()), eye(2 * G.shape[0]), atol=1e-7)

    random = RandomState(0)
    test_for_G(random.randn(3, 1))
    test_for_G(random.randn(3, 3))
    test_for_G(random.randn(3, 2))
    test_for_G(random.randn(3, 4))
    g = random.randn(3)
    test_for_G(stack((g, g), axis=1))
    test_for_G(stack((g, g, g), axis=1))
    test_for_G(stack((g, g, g, g), axis=1))


def test_kron2sumcov_logdet():
    def test_for_G(G):
        cov = Kron2SumCov(2, 1)
        cov.Cr.L = [[1], [2]]
        cov.Cn.L = [[3, 0], [2, 1]]
        cov.G = G
        K = cov.compact_value()
        assert_(slogdet(K)[0] == 1)
        assert_allclose(cov.logdet(), slogdet(K)[1], atol=1e-7)

    random = RandomState(0)
    test_for_G(random.randn(3, 1))
    test_for_G(random.randn(3, 3))
    test_for_G(random.randn(3, 2))
    test_for_G(random.randn(3, 4))
    g = random.randn(3)
    test_for_G(stack((g, g), axis=1))
    test_for_G(stack((g, g, g), axis=1))
    test_for_G(stack((g, g, g, g), axis=1))
