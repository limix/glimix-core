import numpy as np
import pytest
from numpy.testing import assert_allclose

from glimix_core.cov import GivenCov, LinearCov, SumCov


def test_sumcov():
    random = np.random.RandomState(0)
    X = random.randn(3, 2)
    cov_left = LinearCov(X)

    K = random.randn(3, 3)
    K = K @ K.T
    cov_right = GivenCov(K)

    cov = SumCov([cov_left, cov_right])
    assert_allclose(cov.value(), cov_left.value() + cov_right.value())
    assert_allclose(cov._check_grad(), 0, atol=1e-5)
    cov_left.scale = 0.1
    assert_allclose(cov._check_grad(), 0, atol=1e-5)

    with pytest.raises(ValueError):
        K = random.randn(3, 3)
        GivenCov(K)
