from numpy.random import default_rng
from numpy.testing import assert_allclose

from glimix_core.cov import LinearCov


def test_linearcov():
    X = default_rng(0).normal(size=(3, 2))
    cov = LinearCov(X)
    assert_allclose(cov.value(), X @ X.T)
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
    assert_allclose(cov.scale, 1.0)
    cov.scale = 1.5
    assert_allclose(cov.scale, 1.5)
    assert_allclose(cov.value(), 1.5 * (X @ X.T))
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
