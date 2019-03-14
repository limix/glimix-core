from numpy.testing import assert_allclose

from glimix_core.mean import LinearMean, OffsetMean, SumMean


def test_mean_sum():
    X = [[5.1, 1.0], [2.1, -0.2]]

    mean0 = LinearMean(X)
    mean0.effsizes = [-1.0, 0.5]

    mean1 = OffsetMean(2)
    mean1.offset = 2.0

    mean = SumMean([mean0, mean1])

    assert_allclose(mean.value(), [-2.6, -0.2])
    assert_allclose(mean._check_grad(), 0, atol=1e-5)
