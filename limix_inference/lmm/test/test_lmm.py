from __future__ import division

import pytest
from numpy import arange, concatenate, inf, nan, newaxis, ones, sqrt, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs_linear

from limix_inference.cov import EyeCov, LinearCov, SumCov
from limix_inference.lik import DeltaProdLik
from limix_inference.lmm import LMM
from limix_inference.lmm.core import LMMCore
from limix_inference.mean import OffsetMean
from limix_inference.random import GGPSampler


def test_fastlmm_fast_scan():  # pylint: disable=R0914
    random = RandomState(9458)
    N = 500
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    offset = 1.0

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(arange(N), purpose='sample')

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X), purpose='sample')

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((arange(N), arange(N)), purpose='sample')

    cov = SumCov([cov_left, cov_right])

    lik = DeltaProdLik()

    y = GGPSampler(lik, mean, cov).sample(random)

    QS = economic_qs_linear(X)

    lmm = LMM(y, ones((N, 1)), QS)

    lmm.learn(progress=False)

    markers = random.randn(N, 2)

    lmm_ = lmm.copy()
    lmm_.X = concatenate([lmm.X, markers[:, 0][:, newaxis]], axis=1)
    lmm_.fix('delta')
    lmm_.learn(progress=False)
    lml0 = lmm_.lml()

    lmm_ = lmm.copy()
    lmm_.X = concatenate([lmm.X, markers[:, 1][:, newaxis]], axis=1)
    lmm_.fix('delta')
    lmm_.learn(progress=False)
    lml1 = lmm_.lml()

    fast_scanner = lmm.get_fast_scanner()

    lmls = fast_scanner.fast_scan(markers)[0]
    assert_allclose(lmls, [lml0, lml1])


def test_lmm_learn():
    random = RandomState(9458)
    N = 500
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    offset = 1.0

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(arange(N), purpose='sample')

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X), purpose='sample')

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((arange(N), arange(N)), purpose='sample')

    cov = SumCov([cov_left, cov_right])

    lik = DeltaProdLik()

    y = GGPSampler(lik, mean, cov).sample(random)

    QS = economic_qs_linear(X)

    lmm = LMM(y, ones((N, 1)), QS)

    lmm.learn(progress=False)

    assert_allclose(lmm.beta[0], 0.8997652129631661, rtol=1e-5)
    assert_allclose(lmm.genetic_variance, 1.7303981309775553, rtol=1e-5)
    assert_allclose(lmm.environmental_variance, 1.2950028351268132, rtol=1e-5)

    lmm.beta = [-0.5]
    assert_allclose(lmm.beta[0], [-0.5])


def test_fastlmm_learn_fix():
    random = RandomState(9458)
    N = 500
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    offset = 1.0

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(arange(N), purpose='sample')

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X), purpose='sample')

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((arange(N), arange(N)), purpose='sample')

    cov = SumCov([cov_left, cov_right])

    lik = DeltaProdLik()

    y = GGPSampler(lik, mean, cov).sample(random)

    QS = economic_qs_linear(X)

    lmm = LMM(y, ones((N, 1)), QS)

    lmm.fix('delta')
    lmm.fix('scale')

    lmm.scale = 1.0
    lmm.delta = 0.5

    lmm.learn(progress=False)

    assert_allclose(lmm.beta[0], 0.899765212963)
    assert_allclose(lmm.scale, 1.0)
    assert_allclose(lmm.delta, 0.5)
    assert_allclose(lmm.genetic_variance, 0.5)
    assert_allclose(lmm.environmental_variance, 0.5)
    assert_allclose(lmm.lml(), -681.381571238)

    lmm.unfix('scale')
    lmm.learn(progress=False)

    assert_allclose(lmm.beta[0], 0.899765212963)
    assert_allclose(lmm.genetic_variance, 1.4614562029852856)
    assert_allclose(lmm.environmental_variance, 1.4614562029852856)
    assert_allclose(lmm.lml(), -949.526700867)

    lmm.unfix('delta')
    lmm.learn(progress=False)

    assert_allclose(lmm.beta[0], 0.899765212963)
    assert_allclose(lmm.genetic_variance, 1.73039821903)
    assert_allclose(lmm.environmental_variance, 1.29500280131)
    assert_allclose(lmm.lml(), -948.798268063)


def test_lmm_unique_outcome():
    random = RandomState(9458)
    N = 50
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])

    QS = economic_qs_linear(X)

    lmm = LMM(zeros(N), ones((N, 1)), QS)

    lmm.learn(progress=False)

    assert_allclose(lmm.beta[0], 0, atol=1e-7)
    assert_allclose(lmm.genetic_variance, 0, atol=1e-7)
    assert_allclose(lmm.environmental_variance, 0, atol=1e-7)


def test_lmm_nonfinite_phenotype():
    random = RandomState(9458)
    N = 50
    QS = economic_qs_linear(random.randn(N, N + 1))
    y = zeros(N)

    y[0] = nan
    with pytest.raises(ValueError):
        LMM(y, ones((N, 1)), QS)

    y[0] = -inf
    with pytest.raises(ValueError):
        LMM(y, ones((N, 1)), QS)

    y[0] = +inf
    with pytest.raises(ValueError):
        LMM(y, ones((N, 1)), QS)


def test_lmmcore_interface():
    random = RandomState(9458)
    N = 50
    QS = economic_qs_linear(random.randn(N, N + 1))
    y = zeros(N)

    lmmc = LMMCore(y, ones((N, 1)), QS)
    with pytest.raises(NotImplementedError):
        print(lmmc.delta)

    with pytest.raises(NotImplementedError):
        lmmc.delta = 1
