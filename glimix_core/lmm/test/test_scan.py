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

    scale = lmm.scale
    lmm.fit(verbose=False)
    assert_allclose(scale, lmm.scale)

    assert_allclose(lmm.delta, 0.546799614073)
    lmm.delta = 0.5
    assert_allclose(lmm.delta, 0.5)

    assert_allclose(lmm.lml(), -193.947028697)

    assert_allclose(lmm.scale, 3.10605443333)

    lmm.scale = 0.5

    lmm.fit(verbose=False)

    assert_allclose(lmm.scale, 0.5)
    assert_allclose(lmm.delta, 0.775021320328)
    assert_allclose(lmm.lml(), -351.381863666)

    lmm.fix('delta')
    lmm.delta = 0.1

    assert_allclose(lmm.scale, 0.5)
    assert_allclose(lmm.delta, 0.1)
    assert_allclose(lmm.lml(), -615.1757214529657)


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

    fast_scanner = lmm.get_fast_scanner()

    lmls = fast_scanner.fast_scan(markers, verbose=False)[0]
    assert_allclose(
        lmls, [-13.897468, -13.897468, -13.897468, -13.897468, -13.897468],
        rtol=1e-5)


def test_scan_fastlmm_set_scale_1covariate():
    random = RandomState(9458)
    n = 10
    X = _covariates_sample(random, n, n + 1)
    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    M = random.randn(n, 1)
    lmm = LMM(y, M, QS)

    lmm.fit(verbose=False)
    assert_allclose(lmm.scale, 5.3420241172305172)
    assert_allclose(lmm.delta, 0.74276416210589369)
    assert_allclose(lmm.beta, [-0.23917848433233424])

    markers = M.copy() + random.randn(n, 1)

    fast_scanner = lmm.get_fast_scanner()

    fast_scanner.set_scale(1.0)

    lmls, effsizes = fast_scanner.fast_scan(markers, verbose=False)

    assert_allclose(lmls, [-21.5857229586])
    assert_allclose(effsizes, [1.42553567113])

    fast_scanner.unset_scale()
    lmls, effsizes = fast_scanner.fast_scan(markers, verbose=False)

    assert_allclose(lmls, [-21.5007702134])
    assert_allclose(effsizes, [1.42553567113])


# def test_scan_fastlmm_set_scale_1covariate_redundant():
#     random = RandomState(9458)
#     n = 10
#     X = _covariates_sample(random, n, n + 1)
#     offset = 1.0
#
#     y = _outcome_sample(random, offset, X)
#
#     QS = economic_qs_linear(X)
#
#     M = random.randn(n, 1)
#     lmm = LMM(y, M, QS)
#
#     lmm.fit(verbose=False)
#
#     markers = M.copy()
#
#     fast_scanner = lmm.get_fast_scanner()
#
#     fast_scanner.set_scale(1.0)
#
#     import pdb
#     pdb.set_trace()
#     lmls, effsizes = fast_scanner.fast_scan(markers, verbose=False)
#
#     assert_allclose(lmls, [-22.3630065826])
#     assert_allclose(effsizes, [0.0274590587252])
#
#     fast_scanner.unset_scale()
#     lmls, effsizes = fast_scanner.fast_scan(markers, verbose=False)


def test_scan_fastlmm_set_scale_multicovariates():
    random = RandomState(9458)
    n = 10
    X = _covariates_sample(random, n, n + 1)
    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    M = random.randn(n, 3)
    lmm = LMM(y, M, QS)

    lmm.fit(verbose=False)

    markers = M.copy()

    fast_scanner = lmm.get_fast_scanner()

    fast_scanner.set_scale(1.0)

    lmls, effsizes = fast_scanner.fast_scan(markers, verbose=False)

    assert_allclose(
        lmls, [-21.906944123587948, -21.906944123587948, -21.906944123587948])
    assert_allclose(
        effsizes,
        [-0.016199083532862601, -0.13468405305329331, -0.24996061939828515])

    fast_scanner.unset_scale()
    lmls, effsizes = fast_scanner.fast_scan(markers, verbose=False)

    assert_allclose(
        lmls, [-19.533512354470659, -19.533512354470659, -19.533512354470659])
    assert_allclose(
        effsizes,
        [-0.016199083532862601, -0.13468405305329331, -0.24996061939828515])


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
