from numpy import array
from numpy.testing import assert_allclose

from glimix_core.mean import LinearMean


def test_mean_linear():
    X = array([[5.1, 1.0], [2.1, -0.2]])

    mean = LinearMean(X)
    mean.effsizes = [1.0, -1.0]
    assert_allclose(mean.value(), X @ [1.0, -1.0])
    assert_allclose(mean.effsizes, [1.0, -1.0])
    assert_allclose(mean._check_grad(), 0, atol=1e-5)
