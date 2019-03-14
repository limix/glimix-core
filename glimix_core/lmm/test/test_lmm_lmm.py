from numpy import corrcoef, dot, ones, sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core._util import assert_interface
from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.lik import DeltaProdLik
from glimix_core.lmm import LMM
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler
from numpy_sugar.linalg import economic_qs_linear


def _outcome_sample(random, offset, X):
    n = X.shape[0]
    mean = OffsetMean(n)
    mean.offset = offset

    cov_left = LinearCov(X)
    cov_left.scale = 1.5

    cov_right = EyeCov(n)
    cov_right.scale = 1.5

    cov = SumCov([cov_left, cov_right])

    lik = DeltaProdLik()

    return GGPSampler(lik, mean, cov).sample(random)


def _covariates_sample(random, n, p):
    X = random.randn(n, p)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    return X


def test_lmm_lmm_prediction():
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


def test_lmm_lmm_public_attrs():
    assert_interface(
        LMM,
        [
            "lml",
            "X",
            "beta",
            "delta",
            "scale",
            "mean_star",
            "variance_star",
            "covariance_star",
            "covariance",
            "predictive_covariance",
            "mean",
            "isfixed",
            "fixed_effects_variance",
            "gradient",
            "v0",
            "v1",
            "fit",
            "copy",
            "value",
            "get_fast_scanner",
            "predictive_mean",
            "name",
            "unfix",
            "fix",
        ],
    )
