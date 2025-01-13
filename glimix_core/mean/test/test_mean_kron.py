from numpy import kron, ravel
from numpy.random import default_rng
from numpy.testing import assert_allclose

from glimix_core.mean import KronMean


def test_mean_kron():

    random = default_rng(0)
    # number of trais
    p = 2
    # number of covariates
    c = 3
    # sample size
    n = 4

    A = random.normal(size=(p, p))
    X = random.normal(size=(n, c))
    B = random.normal(size=(p, c))

    mean = KronMean(A, X)
    mean.B = B
    assert_allclose(mean.value(), kron(A, X) @ ravel(B, order="F"))
    assert_allclose(mean._check_grad(), 0, atol=1e-5)
