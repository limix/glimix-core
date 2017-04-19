from __future__ import division

from glimix_core.cov import EyeCov
from optimix.testing import Assertion


def test_eyecov_optimix():
    item0 = 0
    item1 = 1
    a = Assertion(EyeCov, item0, item1, 0.0, logscale=0.0)
    a.assert_layout()
    a.assert_gradient()
