from __future__ import division

from numpy import exp
from numpy.random import RandomState
from numpy.testing import assert_almost_equal

from optimix import check_grad

from limix_inference.lmm import SlowLMM
from limix_inference.cov import LinearCov
from limix_inference.mean import OffsetMean
from limix_inference.mean import LinearMean


def test_slowlmm_value_1():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.fix('offset')
    mean.set_data(N)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X, X))

    y = random.randn(N)

    lmm = SlowLMM(y, mean, cov)
    assert_almost_equal(lmm.feed().value(), -153.623791551399108)


def test_slowlmm_value_2():
    random = RandomState(94584)
    N = 50
    X1 = random.randn(N, 3)
    X2 = random.randn(N, 100)

    mean = LinearMean(3)
    mean.set_data(X1)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X2, X2))

    y = random.randn(N)

    lmm = SlowLMM(y, mean, cov)
    assert_almost_equal(lmm.feed().value(), -153.091074766)

    mean.effsizes = [3.4, 1.11, -6.1]
    assert_almost_equal(lmm.feed().value(), -178.273116338)


def test_regression_gradient():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.fix('offset')
    mean.set_data(N)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X, X))

    y = random.randn(N)

    lmm = SlowLMM(y, mean, cov)

    def func(x):
        cov.scale = exp(x[0])
        return lmm.value(mean.feed().value(), cov.feed().value())

    def grad(x):
        cov.scale = exp(x[0])
        return lmm.gradient(mean.feed().value(),
                            cov.feed().value(),
                            mean.feed().gradient(), cov.feed().gradient())

    assert_almost_equal(check_grad(func, grad, [0]), 0)


def test_maximize_1():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean()
    mean.offset = offset
    mean.fix('offset')
    mean.set_data(N)

    cov = LinearCov()
    cov.scale = 1.0
    cov.set_data((X, X))

    y = random.randn(N)

    lmm = SlowLMM(y, mean, cov)
    m = mean.feed().value()
    K = cov.feed().value()
    assert_almost_equal(lmm.value(m, K), -153.62379155139911)

    lmm.feed().maximize()
    assert_almost_equal(lmm.feed().value(), -79.899212241487518)


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

    lmm = SlowLMM(y, mean, cov)
    lmm.feed().maximize()
    assert_almost_equal(lmm.feed().value(), -79.365136339619610)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
