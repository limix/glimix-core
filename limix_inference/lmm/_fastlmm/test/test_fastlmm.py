from __future__ import division

import numpy as np
from numpy import ones
from numpy import arange
from numpy.testing import assert_allclose

from numpy_sugar.linalg import economic_qs_linear

from limix_inference.lmm import FastLMM
from limix_inference.lik import DeltaProdLik
from limix_inference.cov import LinearCov
from limix_inference.cov import EyeCov
from limix_inference.cov import SumCov
from limix_inference.mean import OffsetMean
from limix_inference.random import GLMMSampler


def test_learn():
    random = np.random.RandomState(9458)
    N = 500
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= np.sqrt(X.shape[1])
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

    y = GLMMSampler(lik, mean, cov).sample(random)

    (Q0, Q1), S0 = economic_qs_linear(X)

    flmm = FastLMM(y, Q0, Q1, S0, covariates=ones((N, 1)))

    flmm.learn()

    assert_allclose(flmm.beta[0], 0.8997652129631661, rtol=1e-5)
    assert_allclose(flmm.genetic_variance, 1.7303981309775553, rtol=1e-5)
    assert_allclose(flmm.environmental_variance, 1.2950028351268132, rtol=1e-5)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
