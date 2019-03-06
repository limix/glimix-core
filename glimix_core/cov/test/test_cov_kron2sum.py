from numpy import array, concatenate, eye, kron
from numpy.linalg import slogdet
from numpy.random import RandomState
from numpy.testing import assert_allclose
from scipy.optimize import check_grad

from glimix_core._util import vec
from glimix_core.cov import Kron2SumCov


def test_kron2sumcov():
    G = array([[-1.5, 1.0], [-1.5, 1.0], [-1.5, 1.0]])
    Lr = array([[3], [2]], float)
    Ln = array([[1, 0], [2, 1]], float)

    cov = Kron2SumCov(G, 2, 1)
    cov.C0.L = Lr
    cov.C1.L = Ln

    I = eye(G.shape[0])
    assert_allclose(
        cov.value(), kron(Lr @ Lr.T, G @ G.T) + kron(Ln @ Ln.T, I), atol=1e-4
    )
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
    assert_allclose(cov.solve(cov.value()), eye(2 * G.shape[0]), atol=1e-7)
    assert_allclose(cov.logdet(), slogdet(cov.value())[1], atol=1e-7)

    def func(x):
        cov.C0.Lu = x[:2]
        cov.C1.Lu = x[2:]
        return cov.logdet()

    def grad(x):
        cov.C0.Lu = x[:2]
        cov.C1.Lu = x[2:]
        D = cov.logdet_gradient()
        return concatenate((D["C0.Lu"], D["C1.Lu"]))

    random = RandomState(0)
    assert_allclose(check_grad(func, grad, random.randn(5)), 0, atol=1e-5)

    V = random.randn(3, 2)

    g = cov.C0.gradient()["Lu"]
    g0 = cov.gradient_dot(vec(V))["C0.Lu"]
    for i in range(2):
        assert_allclose(g0[..., i], kron(g[..., i], G @ G.T) @ vec(V))

    g = cov.C1.gradient()["Lu"]
    g0 = cov.gradient_dot(vec(V))["C1.Lu"]
    for i in range(3):
        assert_allclose(g0[..., i], kron(g[..., i], eye(3)) @ vec(V))

    V = random.randn(3, 2, 4)

    g = cov.C0.gradient()["Lu"]
    g0 = cov.gradient_dot(vec(V))["C0.Lu"]
    for i in range(2):
        for j in range(4):
            assert_allclose(g0[j, ..., i], kron(g[..., i], G @ G.T) @ vec(V[..., j]))

    g = cov.C1.gradient()["Lu"]
    g0 = cov.gradient_dot(vec(V))["C1.Lu"]
    for i in range(3):
        for j in range(4):
            assert_allclose(g0[j, ..., i], kron(g[..., i], eye(3)) @ vec(V[..., j]))

    M = random.randn(2 * G.shape[0], 2 * 4)

    R = cov.LdKL_dot(M)
    dK = cov.gradient()
    L = kron(cov.Lh, cov.Lx)

    for i in range(cov.C0.shape[0]):
        for j in range(M.shape[1]):
            expected = L @ dK["C0.Lu"][..., i] @ L.T @ M[:, [j]]
            assert_allclose(R["C0.Lu"][:, [j], i], expected, atol=1e-7)

    for i in range(cov.C1.shape[0]):
        for j in range(M.shape[1]):
            expected = L @ dK["C1.Lu"][..., i] @ L.T @ M[:, [j]]
            assert_allclose(R["C1.Lu"][:, [j], i], expected, atol=1e-7)


def test_kron2sumcov_g_full_col_rank():

    G = array([[-1.5, 1.0, 0.2, 0.5], [1.0, -0.25, -1.5, 1.0], [-0.1, -0.20, -2.5, 0]])
    Lr = array([[3], [2]], float)
    Ln = array([[1, 0], [2, 1]], float)

    cov = Kron2SumCov(G, 2, 1)
    cov.C0.L = Lr
    cov.C1.L = Ln

    I = eye(G.shape[0])
    assert_allclose(
        cov.value(), kron(Lr @ Lr.T, G @ G.T) + kron(Ln @ Ln.T, I), atol=1e-4
    )
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
    assert_allclose(cov.solve(cov.value()), eye(2 * G.shape[0]), atol=1e-7)
    assert_allclose(cov.logdet(), slogdet(cov.value())[1], atol=1e-7)

    cov = Kron2SumCov(G, 2, 2)
    Lr = array([[3, 0.0], [-2, 0.4]], float)
    Ln = array([[1, 0], [2, 1]], float)
    cov.C0.L = Lr
    cov.C1.L = Ln

    I = eye(G.shape[0])
    assert_allclose(
        cov.value(), kron(Lr @ Lr.T, G @ G.T) + kron(Ln @ Ln.T, I), atol=1e-4
    )
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
    assert_allclose(cov.solve(cov.value()), eye(2 * G.shape[0]), atol=1e-7)
    assert_allclose(cov.logdet(), slogdet(cov.value())[1], atol=1e-7)
