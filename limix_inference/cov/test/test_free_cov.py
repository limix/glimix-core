from __future__ import division

from numpy import array, newaxis, zeros
from numpy.testing import assert_allclose

from limix_inference.cov import FreeFormCov
from optimix import check_grad

from optimix.testing import Assertion

# def test_freeform_optimix():
#     item0 = 0
#     item1 = 1
#     a = Assertion(lambda: FreeFormCov(2), item0, item1, 0.0)
#     a.assert_layout()
#
# def test_freeform_value():
#     cov = FreeFormCov(2)
#     assert_allclose(cov.value(0, 0), 1)
#     assert_allclose(cov.value(0, 1), 1)
#     assert_allclose(cov.value(1, 0), 1)
#     assert_allclose(cov.value(1, 1), 2)
#
#     assert_allclose(cov.value([[0], [1]], [[0], [1]]), [[1, 1], [1, 2]])
#
#
# def test_freeform_derivate():
#     cov = FreeFormCov(2)
#
#     step = 1e-7
#
#     x0 = array([1, 1, 1], float)
#     cov.set('Lu', x0)
#     y0 = cov.value(0, 0)
#
#     x1 = array([1 + step, 1, 1], float)
#     cov.set('Lu', x1)
#     y1 = cov.value(0, 0)
#
#     assert_allclose(
#         (y1 - y0) / step, cov.derivative_Lu(0, 0)[0, 0, 0], rtol=1e-5)
#
#     x0 = array([1, 1, 1], float)
#     cov.set('Lu', x0)
#     y0 = cov.value(0, 0)
#
#     x1 = array([1, 1 + step, 1], float)
#     cov.set('Lu', x1)
#     y1 = cov.value(0, 0)
#
#     assert_allclose(
#         (y1 - y0) / step, cov.derivative_Lu(0, 0)[0, 0, 1], rtol=1e-5)
#
#     x0 = array([1, 1, 1], float)
#     cov.set('Lu', x0)
#     y0 = cov.value(0, 0)
#
#     x1 = array([1, 1, 1 + step], float)
#     cov.set('Lu', x1)
#     y1 = cov.value(0, 0)
#
#     assert_allclose(
#         (y1 - y0) / step, cov.derivative_Lu(0, 0)[0, 0, 2], rtol=1e-5)
#
#
# def test_freeform_gradient():
#     cov = FreeFormCov(2)
#     idx = array([[0], [1]], int)
#     cov.set_data((idx, idx))
#
#     def func(x):
#         Lu = cov.Lu
#         Lu[0] = x[0]
#         cov.Lu = Lu
#         return cov.feed().value()
#
#     def grad(x):
#         Lu = cov.Lu
#         Lu[0] = x[0]
#         cov.Lu = Lu
#         return [cov.feed().gradient()[0][:, :, 0]]
#
#     assert_allclose(check_grad(func, grad, [1.0], step=1e-5), 0, atol=1e-5)

# def test_crap():
#     item0 = 0
#     item1 = 1
#     a = Assertion(lambda: FreeFormCov(2), item0, item1, 0.0, Lu=zeros(3))
#     a.assert_layout()

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
