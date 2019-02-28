from numpy import array
from numpy.testing import assert_allclose

from glimix_core.cov import FreeFormCov


def test_freeformcov():

    cov = FreeFormCov(3)
    L = array([[1.5, 0, 0], [2.5, 0.8, 0], [-0.3, 0.4, 2.1]], float)
    cov.L = L
    assert_allclose(cov.value(), L @ L.T, rtol=1e-4)
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
    assert_allclose(cov.logdet(), 1.8486202109176444)

    cov = FreeFormCov(2)
    L = array([[1, 0], [2.5, 1]], float)
    cov.L = L
    assert_allclose(cov.value(), L @ L.T, rtol=1e-4)
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
    assert_allclose(cov.logdet(), 0.00012292724603080174)

    cov = FreeFormCov(1)
    L = array([[0.5]], float)
    cov.L = L
    assert_allclose(cov.value(), L @ L.T, rtol=1e-4)
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
    assert_allclose(cov.logdet(), -1.3862347582514014)
