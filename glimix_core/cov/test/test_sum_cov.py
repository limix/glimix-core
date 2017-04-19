import numpy as np
from numpy.testing import assert_allclose

from glimix_core.cov import LinearCov, SumCov
from optimix import check_grad


def test_value():
    random = np.random.RandomState(0)
    cov_left = LinearCov()
    cov_right = LinearCov()

    X0 = random.randn(4, 20)
    X1 = random.randn(4, 15)

    cov_left.set_data((X0, X0))
    cov_right.set_data((X1, X1))

    cov = SumCov([cov_left, cov_right])
    K = cov.feed().value()
    assert_allclose(K[0, 0], 37.95568923)
    assert_allclose(K[3, 1], 4.53034295)


def test_gradient():
    random = np.random.RandomState(0)
    cov_left = LinearCov()
    cov_right = LinearCov()

    X0 = random.randn(4, 20)
    X1 = random.randn(4, 15)

    cov_left.set_data((X0, X0))
    cov_right.set_data((X1, X1))

    cov = SumCov([cov_left, cov_right])

    assert_allclose(check_grad(cov.feed()), 0, atol=6)
