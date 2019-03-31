from numpy import corrcoef, dot, ones, sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs_linear

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.lik import DeltaProdLik
from glimix_core.lmm import LMM, LMMPredict
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler


def test_lmm_predict():
    random = RandomState(9458)
    n = 30

    X = random.randn(n, n + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])

    offset = 1.0

    mean = OffsetMean(n)
    mean.offset = offset

    cov_left = LinearCov(X)
    cov_left.scale = 1.5

    cov_right = EyeCov(n)
    cov_right.scale = 1.5

    cov = SumCov([cov_left, cov_right])

    lik = DeltaProdLik()

    y = GGPSampler(lik, mean, cov).sample(random)

    QS = economic_qs_linear(X)

    lmm = LMM(y, ones((n, 1)), QS)

    lmm.fit(verbose=False)

    plmm = LMMPredict(y, lmm.beta, lmm.v0, lmm.v1, lmm.mean(), lmm.covariance())

    K = dot(X, X.T)
    pm = plmm.predictive_mean(ones((n, 1)), K, K.diagonal())
    assert_allclose(corrcoef(y, pm)[0, 1], 0.8358820971891354)
