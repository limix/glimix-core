from __future__ import division

from numpy import arange, corrcoef, dot, ones, sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs_linear

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.lik import DeltaProdLik
from glimix_core.lmm import LMM
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler


def _outcome_sample(random, offset, X):
    n = X.shape[0]
    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(arange(n), purpose="sample")

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X), purpose="sample")

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((arange(n), arange(n)), purpose="sample")

    cov = SumCov([cov_left, cov_right])

    lik = DeltaProdLik()

    return GGPSampler(lik, mean, cov).sample(random)


def _covariates_sample(random, n, p):
    X = random.randn(n, p)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    return X


def test_lmm_prediction():
    random = RandomState(9458)
    n = 30
    X = _covariates_sample(random, n, n + 1)

    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    lmm = LMM(y, ones((n, 1)), QS)

    lmm.fit(verbose=False)

    K = dot(X, X.T)
    pm = lmm.predictive_mean(ones((n, 1)), K, K.diagonal())
    assert_allclose(corrcoef(y, pm)[0, 1], 0.8358820971891354)
