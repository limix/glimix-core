from glimix_core.mean import OffsetMean
from optimix.testing import Assertion


def test_offsetmean_optimix():
    item0 = 5
    item1 = 10

    a = Assertion(OffsetMean, item0, item1, 0.0, offset=0.5)
    a.assert_layout()
    a.assert_gradient()
