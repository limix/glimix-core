from __future__ import division

from numpy import zeros

from limix_inference.cov import FreeFormCov
from optimix.testing import Assertion


def test_freeform_optimix():
    item0 = 0
    item1 = 1
    a = Assertion(
        lambda: FreeFormCov(2), item0, item1, 0.0, Lu=zeros(3))
    a.assert_layout()
    a.assert_gradient()


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
