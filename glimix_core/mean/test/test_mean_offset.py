from numpy.testing import assert_allclose

from glimix_core.mean import OffsetMean


def test_mean_offset():
    mean = OffsetMean(2)
    mean.offset = 1.5
    assert_allclose(mean.value(), [1.5, 1.5])
    assert_allclose(mean.offset, 1.5)
    assert_allclose(mean._check_grad(), 0, atol=1e-5)
