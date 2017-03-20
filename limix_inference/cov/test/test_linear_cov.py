import numpy as np
from numpy import array
from numpy.testing import assert_allclose

from limix_inference.cov import LinearCov
from optimix import check_grad
from optimix.testing import Assertion


def test_linearcov_optimix():
    item0 = array([-1.5, 1.0])
    item1 = array([+1.24, 1.0])
    a = Assertion(LinearCov, item0, item1, 0.0, logscale=0.0)
    a.assert_layout()
    a.assert_gradient()


def test_value():
    random = np.random.RandomState(0)
    cov = LinearCov()

    x0 = random.randn(10)
    x1 = random.randn(10)

    assert_allclose(cov.value(x0, x1), np.dot(x0, x1))
    cov.scale = 2.
    assert_allclose(cov.value(x0, x1), 2 * np.dot(x0, x1))


def test_gradient():
    random = np.random.RandomState(0)
    cov = LinearCov()
    cov.scale = 2.

    x0 = random.randn(10)
    x1 = random.randn(10)

    cov.set_data((x0, x1))

    assert_allclose(check_grad(cov.feed()), 0, atol=1e-7)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
