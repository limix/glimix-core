from __future__ import division

from numpy.testing import assert_allclose
from numpy import newaxis

from optimix import check_grad

from limix_inference.cov import FreeFormCov

from numpy_sugar import cartesian


def test_freeform_value():
    cov = FreeFormCov(2)
    assert_allclose(cov.value(0, 0), 1)
    assert_allclose(cov.value(0, 1), 1)
    assert_allclose(cov.value(1, 0), 1)
    assert_allclose(cov.value(1, 1), 2)

    assert_allclose(cov.value([[0], [1]], [[0], [1]]), [[1, 1],
                                                        [1, 2]])

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
