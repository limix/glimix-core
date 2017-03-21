from numpy import ones
from numpy.random import RandomState
from numpy.testing import assert_allclose

from limix_inference.mean import OffsetMean
from optimix import check_grad


def test_offset_mean_value():
    random = RandomState(0)
    mean = OffsetMean()
    offset = random.randn()
    mean.offset = offset

    n = 10
    oarr = offset * ones(n)

    assert_allclose(mean.value(n), oarr)


def test_offset_mean_gradient():
    mean = OffsetMean()
    mean.offset = 0.5
    mean.set_data(10)
    assert_allclose(check_grad(mean.feed()), 0, atol=1e-7)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
