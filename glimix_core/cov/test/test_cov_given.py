from numpy.random import default_rng
from numpy.testing import assert_allclose

from glimix_core.cov import GivenCov


def test_givencov():
    K = default_rng(0).normal(size=(5, 5))
    K = K @ K.T
    cov = GivenCov(K)
    assert_allclose(cov.value(), K, rtol=1e-4)
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
    assert_allclose(cov.scale, 1.0)
    cov.scale = 1.5
    assert_allclose(cov.scale, 1.5)
    assert_allclose(cov.value(), 1.5 * K, rtol=1e-4)
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
