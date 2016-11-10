import numpy as np
import numpy.testing as npt

from lim.cov import SumCov
from lim.cov import LinearCov
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
    npt.assert_almost_equal(K[0, 0], 37.95568923)
    npt.assert_almost_equal(K[3, 1], 4.53034295)


def test_gradient():
    random = np.random.RandomState(0)
    cov_left = LinearCov()
    cov_right = LinearCov()

    X0 = random.randn(4, 20)
    X1 = random.randn(4, 15)

    cov_left.set_data((X0, X0))
    cov_right.set_data((X1, X1))

    cov = SumCov([cov_left, cov_right])

    def func(x):
        cov_left.scale = np.exp(x[0])
        cov_right.scale = np.exp(x[1])
        return cov.feed().value()

    def grad(x):
        cov_left.scale = np.exp(x[0])
        cov_right.scale = np.exp(x[1])
        return cov.feed().gradient()

    npt.assert_almost_equal(check_grad(func, grad, [2.0, 1.5]), 0, decimal=6)
