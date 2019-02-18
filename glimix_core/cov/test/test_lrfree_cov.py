from __future__ import division

from numpy import zeros
from numpy.testing import assert_allclose

from glimix_core.cov import LRFreeFormCov
from optimix import Assertion


def test_lrfreeform_optimix():
    item0 = 0
    item1 = 1

    cov = LRFreeFormCov(3, 2)
    cov.L = [[1, 0], [2, 1], [-1, 3]]

    a = Assertion(lambda: cov, item0, item1, 0.0, Lu=zeros(3))

    a.assert_layout()
    a.assert_gradient()

    assert_allclose(cov.Lu, [1.0, 0.0, 2.0, 1.0, -1.0, 3.0])
    cov.Lu = [1] * 3 * 2
    assert_allclose(cov.Lu, [1] * 3 * 2)
