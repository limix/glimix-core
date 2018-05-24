from __future__ import division

import pytest
from numpy import arange, corrcoef, dot, inf, nan, ones, sqrt, zeros
from numpy.random import RandomState
from numpy.testing import assert_, assert_allclose

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.lik import DeltaProdLik
from glimix_core.lmm import LMM
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler
from numpy_sugar.linalg import economic_qs_linear


def test_lmm_fix_unfix():
    random = RandomState(9458)
    n = 30
    X = _covariates_sample(random, n, n + 1)

    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    lmm = LMM(y, ones((n, 1)), QS)

    assert_(not lmm.isfixed('delta'))
    lmm.fix('delta')
    assert_(lmm.isfixed('delta'))

    assert_(not lmm.isfixed('scale'))
    lmm.fix('scale')
    assert_(lmm.isfixed('scale'))

    lmm.scale = 1.0
    lmm.delta = 0.5

    lmm.fit(verbose=False)

    assert_allclose(lmm.beta[0], 0.7065598068496923)
    assert_allclose(lmm.scale, 1.0)
    assert_allclose(lmm.delta, 0.5)
    assert_allclose(lmm.v0, 0.5)
    assert_allclose(lmm.v1, 0.5)
    assert_allclose(lmm.lml(), -57.56642490856645)

    lmm.unfix('scale')
    lmm.fit(verbose=False)

    assert_allclose(lmm.beta[0], 0.7065598068496923)
    assert_allclose(lmm.v0, 1.060029052117017)
    assert_allclose(lmm.v1, 1.060029052117017)
    assert_allclose(lmm.lml(), -52.037205784544476)

    lmm.unfix('delta')
    lmm.fit(verbose=False)

    assert_allclose(lmm.beta[0], 0.7065598068496922, rtol=1e-5)
    assert_allclose(lmm.v0, 0.5667112269084563, rtol=1e-5)
    assert_allclose(lmm.v1, 1.3679269553495002, rtol=1e-5)
    assert_allclose(lmm.lml(), -51.84396136865774, rtol=1e-5)

    with pytest.raises(ValueError):
        lmm.fix('deltaa')

    with pytest.raises(ValueError):
        lmm.isfixed('deltaa')

    lmm = LMM(y, ones((n, 1)), QS)
    lmm.fix('beta')
    lmm.beta = [1.5]
    assert_allclose(lmm.lml(), -59.02992868385325)
    assert_allclose(lmm.beta[0], 1.5)
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), -56.56425503034913)
    assert_allclose(lmm.beta[0], 1.5)
    lmm.unfix('beta')
    assert_allclose(lmm.lml(), -52.2963882611983)
    assert_allclose(lmm.beta[0], 0.7065598068496929)
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), -51.84396136865775)
    assert_allclose(lmm.beta[0], 0.7065598068496929)
    lmm.fix('beta')
    lmm.beta = lmm.beta[0] + 0.01
    assert_allclose(lmm.lml(), -51.84505787833421)
    lmm.beta = lmm.beta[0] - 2 * 0.01
    assert_allclose(lmm.lml(), -51.84505787833422)


def test_lmm_unique_outcome():
    random = RandomState(9458)
    N = 5
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])

    QS = economic_qs_linear(X)

    lmm = LMM(zeros(N), ones((N, 1)), QS)

    lmm.fit(verbose=False)

    assert_allclose(lmm.beta[0], 0, atol=1e-5)
    assert_allclose(lmm.v0, 0, atol=1e-5)
    assert_allclose(lmm.v1, 0, atol=1e-5)


def test_lmm_nonfinite_outcome():
    random = RandomState(9458)
    N = 5
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


def test_lmm_redundant_covariates_fullrank():
    random = RandomState(9458)
    n = 30
    X = _covariates_sample(random, n, n + 1)

    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    lmm = LMM(y, ones((n, 1)), QS)
    lmm.fit(verbose=False)

    assert_allclose(lmm.scale, 1.93463817155, rtol=1e-5)
    assert_allclose(lmm.delta, 0.707071227475, rtol=1e-5)
    assert_allclose(lmm.beta, 0.70655980685, rtol=1e-5)

    M = ones((n, 10))
    lmm = LMM(y, M, QS)
    lmm.fit(verbose=False)

    assert_allclose(lmm.scale, 1.93463817155, rtol=1e-5)
    assert_allclose(lmm.delta, 0.707071227475, rtol=1e-5)
    assert_allclose(lmm.beta, 0.070655980685, rtol=1e-5)


def test_lmm_redundant_covariates_lowrank():
    random = RandomState(9458)
    n = 30
    X = _covariates_sample(random, n, n - 1)

    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    lmm = LMM(y, ones((n, 1)), QS)
    lmm.fit(verbose=False)

    assert_allclose(lmm.scale, 2.97311575698, rtol=1e-5)
    assert_allclose(lmm.delta, 0.693584745932, rtol=1e-5)
    assert_allclose(lmm.beta, 0.932326853301, rtol=1e-5)

    M = ones((n, 10))
    lmm = LMM(y, M, QS)
    lmm.fit(verbose=False)

    assert_allclose(lmm.scale, 2.97311575698, rtol=1e-5)
    assert_allclose(lmm.delta, 0.693584745932, rtol=1e-5)
    assert_allclose(lmm.beta, 0.0932326853301, rtol=1e-5)


def _outcome_sample(random, offset, X):
    n = X.shape[0]
    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(arange(n), purpose='sample')

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X), purpose='sample')

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((arange(n), arange(n)), purpose='sample')

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


def test_lmm_iid_prior():
    random = RandomState(9458)
    n = 30
    X = _covariates_sample(random, n, n + 1)

    offset = 1.0

    y = _outcome_sample(random, offset, X)

    lmm = LMM(y, ones((n, 1)), None)

    assert_(lmm.isfixed('delta'))

    assert_allclose(lmm.lml(), -52.29638826846387)
    lmm.fit(verbose=False)

    assert_allclose(lmm.beta[0], 0.7065598068496929)
    assert_allclose(lmm.scale, 1.9127630509027338)
    assert_allclose(lmm.delta, 0.9999999999999998)
    assert_allclose(lmm.lml(), -52.29638826846388)


def test_lmm_zero_rank_covariates():
    random = RandomState(9458)
    n = 30
    X = _covariates_sample(random, n, n + 1)

    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)
    lmm = LMM(y, zeros((n, 1)), QS)
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), -55.774936794143315)
    assert_allclose(lmm.scale, 2.411989807718206)
    assert_allclose(lmm.delta, 0.9999999979388463)
    assert_allclose(lmm.beta, [0])
