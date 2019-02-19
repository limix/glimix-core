from numpy.testing import assert_allclose, assert_string_equal

from glimix_core.mean import LinearMean
from optimix import Assertion


def test_offsetmean_optimix():
    item0 = [5.1, 1.0]
    item1 = [2.1, -0.2]

    mean = LinearMean(2)

    a = Assertion(lambda: mean, item0, item1, 0.0, effsizes=[0.5, 1.0])
    a.assert_layout()
    a.assert_gradient()

    mean.effsizes = [1.0, -1.0]
    assert_allclose(mean.effsizes, [1.0, -1.0])


def test_offsetmean_str():
    mean = LinearMean(2)
    mean.effsizes = [0.5, 1.0]
    assert_string_equal(str(mean), "LinearMean(size=2)\n  effsizes: [0.5 1. ]")
