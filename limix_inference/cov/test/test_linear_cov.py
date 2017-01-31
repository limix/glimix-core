import numpy as np
from numpy import array
import numpy.testing as npt

from limix_inference.cov import LinearCov
from optimix import check_grad

from optimix.testing import Assertion

def test_linearcov_optimix():
    item0 = array([-1.5, 1.0])
    item1 = array([+1.24, 1.0])
    a = Assertion(lambda: LinearCov(), item0, item1, 0.0, logscale=0.0)
    a.assert_layout()
    a.assert_gradient()

def test_value():
    random = np.random.RandomState(0)
    cov = LinearCov()

    x0 = random.randn(10)
    x1 = random.randn(10)

    npt.assert_almost_equal(cov.value(x0, x1), np.dot(x0, x1))
    cov.scale = 2.
    npt.assert_almost_equal(cov.value(x0, x1), 2 * np.dot(x0, x1))


def test_gradient():
    random = np.random.RandomState(0)
    cov = LinearCov()
    cov.scale = 2.

    x0 = random.randn(10)
    x1 = random.randn(10)

    def func(x):
        cov.scale = np.exp(x[0])
        return cov.value(x0, x1)

    def grad(x):
        cov.scale = np.exp(x[0])
        return [cov.derivative_logscale(x0, x1)]

    npt.assert_almost_equal(check_grad(func, grad, [2.0]), 0, decimal=6)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
