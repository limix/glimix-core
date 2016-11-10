from __future__ import division

from numpy.testing import assert_allclose

from numpy import exp

from optimix import check_grad

from limix_inference.cov import EyeCov
from limix_inference.fruits import Oranges
from limix_inference.fruits import Apples


def test_eye_value():
    cov = EyeCov()
    cov.scale = 2.1
    o = Oranges(None)
    assert_allclose(2.1, cov.value(o, o))


def test_eye_gradient_1():
    cov = EyeCov()
    cov.scale = 2.1
    a = Apples(None)
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
    a = Apples(5)
    cov.set_data((a, a))

    def func(x):
        cov.scale = exp(x[0])
        return cov.feed().value()

    def grad(x):
        cov.scale = exp(x[0])
        return cov.feed().gradient()

    assert_allclose(check_grad(func, grad, [0.1]), 0, atol=1e-7)


def test_eye_gradient_3():
    cov = EyeCov()
    cov.scale = 2.1
    a = Apples(5)
    o = Oranges(4)
    cov.set_data((a, o))

    def func(x):
        cov.scale = exp(x[0])
        return cov.feed().value()

    def grad(x):
        cov.scale = exp(x[0])
        return cov.feed().gradient()

    assert_allclose(check_grad(func, grad, [0.1]), 0, atol=1e-7)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
