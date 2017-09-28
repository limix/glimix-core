from __future__ import division

from numpy import arange, concatenate, newaxis, ones, sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs_linear

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.lik import DeltaProdLik
from glimix_core.lmm import LMM
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler


def test_scan_fix_unfix():
    random = RandomState(12)
    n = 100
    X = _covariates_sample(random, n, n + 1)
    offset = 0.1

    y = _outcome_sample(random, offset, X)
    QS = economic_qs_linear(X)
    lmm = LMM(y, ones((n, 1)), QS)

    lmm.fit(verbose=False)

    lml0 = lmm.lml()
    assert_allclose(lml0, -193.88684605236722)
    lmm.fix('scale')
    lml1 = lmm.lml()

    assert_allclose(lml0, lml1)
    # lmm.fit(verbose=False)
    #
    # delta1 = lmm.delta
    # assert_allclose(delta0, delta1)
    # print("\n--------------------")
    #
    # print('lml', lmm.lml())
    # print(lmm.scale, lmm.delta, lmm.beta)
    #
    # # import pdb
    # # pdb.set_trace()
    #
    # print("\n--------------------")
    # fast_scanner = lmm.get_fast_scanner()
    # # fast_scanner.set_scale(0.8342)
    # fast_scanner.set_scale(1)
    #
    # print('lml', fast_scanner.null_lml())


def test_scan_fast_scan():
    random = RandomState(9458)
    n = 30
    X = _covariates_sample(random, n, n + 1)
    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    lmm = LMM(y, ones((n, 1)), QS)

    lmm.fit(verbose=False)

    markers = random.randn(n, 2)

    lmm_ = lmm.copy()
    lmm_.X = concatenate([lmm.X, markers[:, 0][:, newaxis]], axis=1)
    lmm_.fix('delta')
    lmm_.fit(verbose=False)
    lml0 = lmm_.lml()

    lmm_ = lmm.copy()
    lmm_.X = concatenate([lmm.X, markers[:, 1][:, newaxis]], axis=1)
    lmm_.fix('delta')
    lmm_.fit(verbose=False)
    lml1 = lmm_.lml()

    fast_scanner = lmm.get_fast_scanner()

    lmls = fast_scanner.fast_scan(markers, verbose=False)[0]
    assert_allclose(lmls, [lml0, lml1])


def test_scan_fastlmm_redundant_candidates():
    random = RandomState(9458)
    n = 10
    X = _covariates_sample(random, n, n + 1)
    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    M = ones((n, 5))
    lmm = LMM(y, M, QS)

    lmm.fit(verbose=False)

    markers = M.copy()

    lmm.fit(verbose=False)
    fast_scanner = lmm.get_fast_scanner()

    lmls = fast_scanner.fast_scan(markers, verbose=False)[0]
    assert_allclose(
        lmls, [-13.897468, -13.897468, -13.897468, -13.897468, -13.897468],
        rtol=1e-5)


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
