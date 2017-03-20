from __future__ import division

from numpy import exp
from numpy.random import RandomState
from numpy.testing import assert_allclose

from limix_inference.cov import LinearCov
from limix_inference.gp import GP
from limix_inference.mean import LinearMean, OffsetMean
from optimix import check_grad


# def test_gp_value_1():
#     random = RandomState(94584)
#     N = 50
#     X = random.randn(N, 100)
#     offset = 0.5
#
#     mean = OffsetMean()
#     mean.offset = offset
#     mean.fix('offset')
#     mean.set_data(N)
#
#     cov = LinearCov()
#     cov.scale = 1.0
#     cov.set_data((X, X))
#
#     y = random.randn(N)
#
#     gp = GP(y, mean, cov)
#     assert_almost_equal(gp.feed().value(), -153.623791551399108)
#
#
# def test_gp_value_2():
#     random = RandomState(94584)
#     N = 50
#     X1 = random.randn(N, 3)
#     X2 = random.randn(N, 100)
#
#     mean = LinearMean(3)
#     mean.set_data(X1)
#
#     cov = LinearCov()
#     cov.scale = 1.0
#     cov.set_data((X2, X2))
#
#     y = random.randn(N)
#
#     gp = GP(y, mean, cov)
#     assert_almost_equal(gp.feed().value(), -153.091074766)
#
#     mean.effsizes = [3.4, 1.11, -6.1]
#     assert_almost_equal(gp.feed().value(), -178.273116338)
#
#
# def test_regression_gradient():
#     random = RandomState(94584)
#     N = 50
#     X = random.randn(N, 100)
#     offset = 0.5
#
#     mean = OffsetMean()
#     mean.offset = offset
#     mean.fix('offset')
#     mean.set_data(N)
#
#     cov = LinearCov()
#     cov.scale = 1.0
#     cov.set_data((X, X))
#
#     y = random.randn(N)
#
#     gp = GP(y, mean, cov)
#
#     def func(x):
#         cov.scale = exp(x[0])
#         return gp.value(mean.feed().value(), cov.feed().value())
#
#     def grad(x):
#         cov.scale = exp(x[0])
#         return gp.gradient(mean.feed().value(),
#                            cov.feed().value(),
#                            mean.feed().gradient(), cov.feed().gradient())
#
#     assert_almost_equal(check_grad(func, grad, [0]), 0)
#
#
# def test_maximize_1():
#     random = RandomState(94584)
#     N = 50
#     X = random.randn(N, 100)
#     offset = 0.5
#
#     mean = OffsetMean()
#     mean.offset = offset
#     mean.fix('offset')
#     mean.set_data(N)
#
#     cov = LinearCov()
#     cov.scale = 1.0
#     cov.set_data((X, X))
#
#     y = random.randn(N)
#
#     gp = GP(y, mean, cov)
#     m = mean.feed().value()
#     K = cov.feed().value()
#     assert_almost_equal(gp.value(m, K), -153.62379155139911)
#
#     gp.feed().maximize(progress=False)
#     assert_almost_equal(gp.feed().value(), -79.899212241487518)
#
#
def test_maximize_2():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(N)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X, X))

    y = random.randn(N)

    gp = GP(y, mean, cov)
    # import pdb; pdb.set_trace()
    assert_allclose(gp.feed().value(), -153.623791551)
    # print(gp.feed().gradient())
    check_grad(gp.feed())
    # gp.feed().maximize(progress=False)
    # assert_almost_equal(gp.feed().value(), -79.365136339619610)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
