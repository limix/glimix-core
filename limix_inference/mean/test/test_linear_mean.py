from numpy.random import RandomState
from numpy.testing import assert_allclose

from optimix import check_grad

from limix_inference.mean import LinearMean


def test_value():
    random = RandomState(0)
    mean = LinearMean(5)
    effsizes = random.randn(5)
    mean.effsizes = effsizes

    x = random.randn(5)
    assert_allclose(mean.value(x), -0.956409566703)


def test_gradient():
    random = RandomState(1)
    mean = LinearMean(5)
    effsizes = random.randn(5)
    mean.effsizes = effsizes
    x = random.randn(2, 5)
    mean.set_data(x)
    assert_allclose(check_grad(mean.feed()), 0, atol=1e-6)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
