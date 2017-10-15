from numpy import dot
from numpy.testing import assert_allclose
from numpy.random import RandomState

from glimix_core.cov import GivenCov
from optimix.testing import Assertion


def test_givencov_optimix():
    item0 = 0
    item1 = 1
    K = RandomState(0).randn(5, 5)
    K = dot(K, K.T)
    a = Assertion(lambda: GivenCov(K), item0, item1, 0.0, logscale=0.0)
    a.assert_layout()
    a.assert_gradient()


def test_givencov_interface():
    K = RandomState(0).randn(5, 5)
    K = dot(K, K.T)
    c = GivenCov(K)
    c.scale = 1.5
    assert_allclose(c.scale, 1.5)
