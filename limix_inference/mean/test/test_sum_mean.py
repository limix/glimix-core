from numpy import arange
from numpy.testing import assert_allclose

from glimix_core.mean import OffsetMean, LinearMean, SumMean

from optimix import check_grad


def test_summean_optimix():
    X = [[5.1, 1.0],
         [2.1, -0.2]]

    mean0 = LinearMean(2)
    mean0.set_data((X, ))
    mean0.effsizes = [-1.0, 0.5]

    mean1 = OffsetMean()
    mean1.set_data((arange(2), ))
    mean1.offset = 2.0

    mean = SumMean([mean0, mean1])

    assert_allclose(mean.feed().value(), [-2.6, -0.2])
    assert_allclose(check_grad(mean.feed()), 0, atol=1e-6)
