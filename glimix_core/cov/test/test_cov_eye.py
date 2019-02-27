from numpy import eye
from numpy.testing import assert_allclose

from glimix_core.cov import EyeCov


def test_eyecov():
    cov = EyeCov(2)
    cov.scale = 1.5
    assert_allclose(cov.value(), 1.5 * eye(2))
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
