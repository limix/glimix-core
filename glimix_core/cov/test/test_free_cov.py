from __future__ import division

from numpy import zeros
from numpy.testing import assert_allclose

from glimix_core.cov import FreeFormCov
from optimix.testing import Assertion


def test_freeform_optimix():
    item0 = 0
    item1 = 1

    cov = FreeFormCov(2)
    cov.L = [[1, 0], [2, 1]]

    a = Assertion(lambda: cov, item0, item1, 0.0, Lu=zeros(3))

    a.assert_layout()
    a.assert_gradient()

    assert_allclose(cov.Lu, [1, 2, 1])
    cov.Lu = [1, 1, 1]
    assert_allclose(cov.Lu, [1, 1, 1])
