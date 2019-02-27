from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core.cov import LinearCov


def test_linearcov():
    X = RandomState(0).randn(3, 2)
    cov = LinearCov(X)
    assert_allclose(cov.value(), X @ X.T)
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
    assert_allclose(cov.scale, 1.0)
    cov.scale = 1.5
    assert_allclose(cov.scale, 1.5)
    assert_allclose(cov.value(), 1.5 * (X @ X.T))
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
