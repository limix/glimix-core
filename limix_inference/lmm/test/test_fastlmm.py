from __future__ import division

from numpy.random import RandomState
from numpy import ones, concatenate, sqrt
from numpy import arange, newaxis
from numpy.testing import assert_allclose

from numpy_sugar.linalg import economic_qs_linear

from limix_inference.lmm import LMM
from limix_inference.lik import DeltaProdLik
from limix_inference.cov import LinearCov
from limix_inference.cov import EyeCov
from limix_inference.cov import SumCov
from limix_inference.mean import OffsetMean
from limix_inference.random import GGPSampler

def test_fastlmm_fast_scan(): # pylint: disable=R0914
    random = RandomState(9458)
    N = 500
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    offset = 1.0

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(N, purpose='sample')

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

    flmm = LMM(y, ones((N, 1)), QS)

    flmm.learn(progress=False)

    markers = random.randn(N, 2)

    # flmm_ = flmm.copy()
    # flmm_.M = concatenate([flmm.M, markers[:, 0][:, newaxis]], axis=1)
    # flmm_.fix('delta')
    # flmm_.learn(progress=False)
    # lml0 = flmm_.lml()
    #
    # flmm_ = flmm.copy()
    # flmm_.M = concatenate([flmm.M, markers[:, 1][:, newaxis]], axis=1)
    # flmm_.fix('delta')
    # flmm_.learn(progress=False)
    # lml1 = flmm_.lml()
    #
    # fast_scanner = flmm.get_fast_scanner()
    #
    # lmls = fast_scanner.fast_scan(markers)[0]
    # assert_allclose(lmls, [lml0, lml1])

def test_learn():
    random = RandomState(9458)
    N = 500
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    offset = 1.0

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(N, purpose='sample')

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

    flmm = LMM(y, ones((N, 1)), QS)

    flmm.learn(progress=False)

    assert_allclose(flmm.beta[0], 0.8997652129631661, rtol=1e-5)
    assert_allclose(flmm.genetic_variance, 1.7303981309775553, rtol=1e-5)
    assert_allclose(flmm.environmental_variance, 1.2950028351268132, rtol=1e-5)


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
    mean.set_data(N, purpose='sample')

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

    flmm = LMM(y, ones((N, 1)), QS)

    flmm.fix('delta')
    flmm.fix('scale')

    flmm.scale = 1.0
    flmm.delta = 0.5

    flmm.learn(progress=False)

    assert_allclose(flmm.beta[0], 0.899765212963)
    assert_allclose(flmm.scale, 1.0)
    assert_allclose(flmm.delta, 0.5)
    assert_allclose(flmm.genetic_variance, 0.5)
    assert_allclose(flmm.environmental_variance, 0.5)
    assert_allclose(flmm.lml(), -681.381571238)

    flmm.unfix('scale')
    flmm.learn(progress=False)

    assert_allclose(flmm.beta[0], 0.899765212963)
    assert_allclose(flmm.genetic_variance, 1.4614562029852856)
    assert_allclose(flmm.environmental_variance, 1.4614562029852856)
    assert_allclose(flmm.lml(), -949.526700867)

    flmm.unfix('delta')
    flmm.learn(progress=False)

    assert_allclose(flmm.beta[0], 0.899765212963)
    assert_allclose(flmm.genetic_variance, 1.73039821903)
    assert_allclose(flmm.environmental_variance, 1.29500280131)
    assert_allclose(flmm.lml(), -948.798268063)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
