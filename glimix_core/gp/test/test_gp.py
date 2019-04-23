import pytest
from numpy import eye, inf, nan, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core.cov import EyeCov, GivenCov, LinearCov, SumCov
from glimix_core.gp import GP
from glimix_core.mean import LinearMean, OffsetMean


def test_gp_value_1():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean(N)
    mean.offset = offset
    mean.fix_offset()

    cov = LinearCov(X)
    cov.scale = 1.0

    y = random.randn(N)

    gp = GP(y, mean, cov)
    assert_allclose(gp.value(), -153.623791551399108)


def test_gp_value_2():
    random = RandomState(94584)
    N = 50
    X1 = random.randn(N, 3)
    X2 = random.randn(N, 100)

    mean = LinearMean(X1)

    cov = LinearCov(X2)
    cov.scale = 1.0

    y = random.randn(N)

    gp = GP(y, mean, cov)
    assert_allclose(gp.value(), -153.091074766)

    mean.effsizes = [3.4, 1.11, -6.1]
    assert_allclose(gp.value(), -178.273116338)


def test_gp_gradient():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean(N)
    mean.offset = offset
    mean.fix_offset()

    cov = LinearCov(X)
    cov.scale = 1.0

    y = random.randn(N)

    gp = GP(y, mean, cov)

    assert_allclose(gp._check_grad(), 0, atol=1e-5)


def test_gp_gradient2():
    random = RandomState(94584)
    N = 10
    X = random.randn(N, 2)

    K0 = random.randn(N, N + 1)
    K0 = K0 @ K0.T + eye(N) * 1e-5

    K1 = random.randn(N, N + 1)
    K1 = K1 @ K1.T + eye(N) * 1e-5

    mean = LinearMean(X)
    mean.effsizes = [-1.2, +0.9]

    cov0 = GivenCov(K0)
    cov1 = GivenCov(K1)
    cov2 = EyeCov(N)
    cov = SumCov([cov0, cov1, cov2])
    cov0.scale = 0.1
    cov1.scale = 2.3
    cov2.scale = 0.3

    y = random.randn(N)

    gp = GP(y, mean, cov)

    assert_allclose(gp._check_grad(), 0, atol=1e-5)


def test_gp_maximize():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean(N)
    mean.offset = offset
    mean.fix_offset()

    cov = LinearCov(X)
    cov.scale = 1.0

    y = random.randn(N)

    gp = GP(y, mean, cov)

    assert_allclose(gp.value(), -153.623791551)
    gp.fit(verbose=False)
    assert_allclose(gp.value(), -79.8992122415)


def test_gp_nonfinite_phenotype():
    random = RandomState(94584)
    N = 50
    X = random.randn(N, 100)
    offset = 0.5

    mean = OffsetMean(N)
    mean.offset = offset
    mean.fix_offset()

    cov = LinearCov(X)
    cov.scale = 1.0

    y = zeros(N)

    y[0] = nan
    with pytest.raises(ValueError):
        GP(y, mean, cov)

    y[0] = -inf
    with pytest.raises(ValueError):
        GP(y, mean, cov)

    y[0] = +inf
    with pytest.raises(ValueError):
        GP(y, mean, cov)
