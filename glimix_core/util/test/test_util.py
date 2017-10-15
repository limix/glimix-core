from __future__ import division

from glimix_core.util import check_covariates, check_economic_qs, check_outcome
from numpy import ones, inf, nan, array
from numpy.testing import assert_allclose
import pytest


def test_util_check_economic_qs():

    A = ones((3, 2))
    B = ones((3, 1))
    C = ones(2)

    with pytest.raises(ValueError):
        check_economic_qs(A)

    with pytest.raises(ValueError):
        check_economic_qs((A, C))

    A[0, 0] = inf
    QS = ((A, B), C)

    with pytest.raises(ValueError):
        check_economic_qs(QS)

    A[0, 0] = 1
    C[0] = nan

    with pytest.raises(ValueError):
        check_economic_qs(QS)


def test_util_check_covariates():
    A = ones(2)
    B = ones((1, 2))

    with pytest.raises(ValueError):
        check_covariates(A)

    B[0, 0] = inf
    with pytest.raises(ValueError):
        check_covariates(B)


def test_util_check_outcome():
    y = ones(5)

    y[0] = nan
    with pytest.raises(ValueError):
        check_outcome((y, ), 'poisson')

    y[0] = 0.5
    want = (array([0.5, 1., 1., 1., 1.]), )
    assert_allclose(check_outcome((y, ), 'poisson'), want)

    x = ones(4)

    with pytest.raises(ValueError):
        check_outcome((y, x), 'bernoulli')

    x = ones(5)

    with pytest.raises(ValueError):
        check_outcome((y, ), 'normal')


def test_util_check_poisson_outcome():
    y = ones(5)
    y[0] = 25000 + 1
    want = (array([
        2.50000000e+04, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
        1.00000000e+00
    ]), )
    assert_allclose(check_outcome((y, ), 'poisson'), want)
