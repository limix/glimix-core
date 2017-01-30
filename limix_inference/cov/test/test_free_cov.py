from __future__ import division

from numpy import array, newaxis
from numpy.testing import assert_allclose

from limix_inference.cov import FreeFormCov
from numpy_sugar import cartesian
from optimix import check_grad


def test_freeform_value():
    cov = FreeFormCov(2)
    assert_allclose(cov.value(0, 0), 1)
    assert_allclose(cov.value(0, 1), 1)
    assert_allclose(cov.value(1, 0), 1)
    assert_allclose(cov.value(1, 1), 2)

    assert_allclose(cov.value([[0], [1]], [[0], [1]]), [[1, 1], [1, 2]])


def test_freeform_derivate():
    cov = FreeFormCov(2)

    step = 1e-7

    x0 = array([1, 1, 1], float)
    cov.set('Lu', x0)
    y0 = cov.value(0, 0)

    x1 = array([1 + step, 1, 1], float)
    cov.set('Lu', x1)
    y1 = cov.value(0, 0)

    assert_allclose((y1 - y0) / step, cov.derivative_Lu(0, 0)[0], rtol=1e-5)

    x0 = array([1, 1, 1], float)
    cov.set('Lu', x0)
    y0 = cov.value(0, 0)

    x1 = array([1, 1 + step, 1], float)
    cov.set('Lu', x1)
    y1 = cov.value(0, 0)

    assert_allclose((y1 - y0) / step, cov.derivative_Lu(0, 0)[1], rtol=1e-5)

    x0 = array([1, 1, 1], float)
    cov.set('Lu', x0)
    y0 = cov.value(0, 0)

    x1 = array([1, 1, 1 + step], float)
    cov.set('Lu', x1)
    y1 = cov.value(0, 0)

    assert_allclose((y1 - y0) / step, cov.derivative_Lu(0, 0)[2], rtol=1e-5)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
