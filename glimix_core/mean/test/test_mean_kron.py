from numpy import kron, ravel
from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core.mean import KronMean


def test_mean_kron():

    random = RandomState(0)
    # number of trais
    p = 2
    # number of covariates
    c = 3
    # sample size
    n = 4

    A = random.randn(p, p)
    F = random.randn(n, c)
    B = random.randn(p, c)
    # vecB = random.randn(c, p).ravel()

    mean = KronMean(A, F)
    mean.B = B
    assert_allclose(mean.value(), kron(A, F) @ ravel(B, order="F"))
    assert_allclose(mean._check_grad(), 0, atol=1e-5)
