from numpy.testing import assert_allclose

from glimix_core.mean import LinearMean
from optimix.testing import Assertion


def test_offsetmean_optimix():
    item0 = [5.1, 1.0]
    item1 = [2.1, -0.2]

    cov = LinearMean(2)

    a = Assertion(lambda: cov, item0, item1, 0.0, effsizes=[0.5, 1.0])
    a.assert_layout()
    a.assert_gradient()

    cov.effsizes = [1.0, -1.0]
    assert_allclose(cov.effsizes, [1.0, -1.0])
