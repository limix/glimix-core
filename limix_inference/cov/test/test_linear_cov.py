from numpy import array

from glimix_core.cov import LinearCov
from optimix.testing import Assertion


def test_linearcov_optimix():
    item0 = array([-1.5, 1.0])
    item1 = array([+1.24, 1.0])
    a = Assertion(LinearCov, item0, item1, 0.0, logscale=0.0)
    a.assert_layout()
    a.assert_gradient()
