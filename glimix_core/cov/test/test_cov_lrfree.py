from __future__ import division

from numpy import array
from numpy.testing import assert_allclose

from glimix_core.cov import LRFreeFormCov


def test_lrfreeformcov():
    cov = LRFreeFormCov(3, 2)
    L = array([[1, 0], [2, 1], [-1, 3]], float)
    cov.L = L
    assert_allclose(cov.value(), L @ L.T)
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
