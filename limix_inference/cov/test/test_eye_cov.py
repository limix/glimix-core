from __future__ import division

from numpy.testing import assert_allclose

from numpy import exp
from numpy import arange

from optimix import check_grad

from limix_inference.cov import EyeCov

from optimix.testing import Assertion

def test_eyecov_optimix():
    item0 = 0
    item1 = 1
    a = Assertion(lambda: EyeCov(), item0, item1, 0.0, logscale=0.0)
    a.assert_layout()

def test_eye_value():
    cov = EyeCov()
    cov.scale = 2.1
    assert_allclose(cov.value(0, 0), 2.1)


def test_eye_gradient_1():
    cov = EyeCov()
    cov.scale = 2.1
    a = arange(1)
    cov.set_data((a, a))

    def func(x):
        cov.scale = exp(x[0])
        return cov.feed().value()

    def grad(x):
        cov.scale = exp(x[0])
        return cov.feed().gradient()

    assert_allclose(check_grad(func, grad, [0.1]), 0, atol=1e-7)


def test_eye_gradient_2():
    cov = EyeCov()
    cov.scale = 2.1
    a = arange(5)
    cov.set_data((a, a))

    def func(x):
        cov.scale = exp(x[0])
        return cov.feed().value()

    def grad(x):
        cov.scale = exp(x[0])
        return cov.feed().gradient()

    assert_allclose(check_grad(func, grad, [0.1]), 0, atol=1e-7)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
