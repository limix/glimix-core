from __future__ import division

from numpy import arange, concatenate, newaxis, ones, sqrt, array
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs_linear, economic_qs

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.lik import DeltaProdLik
from glimix_core.lmm import LMM
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler
from glimix_core.lmm.scan import FastScanner


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


def test_scan_fastlmm_set_scale_1covariate_redundant():
    random = RandomState(9458)
    n = 10
    X = _covariates_sample(random, n, n + 1)
    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    M = random.randn(n, 1)
    lmm = LMM(y, M, QS)

    lmm.fit(verbose=False)

    markers = M.copy()

    fast_scanner = lmm.get_fast_scanner()

    fast_scanner.set_scale(1.0)

    lmls, effsizes = fast_scanner.fast_scan(markers, verbose=False)

    print(lmls[0])
    print(effsizes[0])
    return
    assert_allclose(lmls, [-22.3630065826])
    assert_allclose(effsizes, [0.0274590587252])

    fast_scanner.unset_scale()
    lmls, effsizes = fast_scanner.fast_scan(markers, verbose=False)
    assert_allclose(lmls[0], [-22.3622109073])
    assert_allclose(effsizes[0], [0.0274590587252])


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

    assert_allclose(effsizes, [-0.54393628378, -12.0444021706, -0.75])

    fast_scanner.unset_scale()
    lmls, effsizes = fast_scanner.fast_scan(markers, verbose=False)

    assert_allclose(
        lmls, [-19.533512354470659, -19.533512354470659, -19.533512354470659])

    assert_allclose(effsizes, [-0.54393628378, -12.0444021706, -0.75])


def test_scan_difficult_settings_offset():
    y = array([-1.0449132, 1.15229426, 0.79595129])
    low_rank_K = array([[5., 14., 23.], [14., 50., 86.], [23., 86., 149.]])
    full_rank_K = array([[6., 14., 23.], [14., 51., 86.], [23., 86., 150.]])
    low_rank_QS = economic_qs(low_rank_K)
    full_rank_QS = economic_qs(full_rank_K)
    X = ones((3, 1))

    M = array([[0.88766985, -1.80940339], [0.00822629, -0.4488265],
               [0.55807272, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [401.11650199229757, 1409.645649707018])
    assert_allclose(effsizes, [-5.5851957509459353, -9.2375605785935715])

    X = ones((3, 1))
    M = array([[0.88766985, -1.80940339], [0.00822629, -0.4488265],
               [0.55807272, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0.75)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [0.823505300549, -0.607251148058])
    assert_allclose(effsizes, [-1.79486991632, 0.872439001906])

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [349.86295749, 1409.64564971])
    assert_allclose(effsizes[1], -9.23756057859)

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0.75)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [-6.40108661588, -0.607251148058])
    assert_allclose(effsizes[1], 0.872439001906)

    M = array([[0.88766985, -1.80940339], [0.00822629, -0.4488265],
               [0.55807272, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [0.636421389625, -0.795311239268])
    assert_allclose(effsizes, [-1.79757295968, 0.871806031702])

    M = array([[0.88766985, -1.80940339], [0.00822629, -0.4488265],
               [0.55807272, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0.75)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [0.251547678451, -1.18305656511])
    assert_allclose(effsizes, [-1.80551427016, 0.86994164292])

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [-6.21920242113, -0.795311239268])
    assert_allclose(effsizes[1], 0.871806031702)

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0.75)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [-5.89837796719, -1.18305656511])
    assert_allclose(effsizes[1], 0.86994164292)


def test_scan_difficult_settings_multicovariates():
    y = array([-1.0449132, 1.15229426, 0.79595129])
    low_rank_K = array([[5., 14., 23.], [14., 50., 86.], [23., 86., 149.]])
    full_rank_K = array([[6., 14., 23.], [14., 51., 86.], [23., 86., 150.]])
    low_rank_QS = economic_qs(low_rank_K)
    full_rank_QS = economic_qs(full_rank_K)
    X = array([[-0.40592765, 1.04348945], [0.92275415, -0.32394197],
               [-0.98197991, 1.22912219]])

    M = array([[0.88766985, -1.80940339], [0.00822629, -0.4488265],
               [0.55807272, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [1409.64564971, 403.195943534])
    assert_allclose(effsizes, [-3.62953948465, -0.267605654385])

    X = ones((3, 1))
    M = array([[0.88766985, -1.80940339], [0.00822629, -0.4488265],
               [0.55807272, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0.75)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [0.823505300549, -0.607251148058])
    assert_allclose(effsizes, [-1.79486991632, 0.872439001906])

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [349.862957487, 1409.64564971])
    assert_allclose(effsizes[1], -9.23756057859)

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0.75)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [-6.40108661588, -0.607251148058])
    assert_allclose(effsizes[1], 0.872439001906)

    M = array([[0.88766985, -1.80940339], [0.00822629, -0.4488265],
               [0.55807272, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [0.636421389625, -0.795311239268])
    assert_allclose(effsizes, [-1.79757295968, 0.871806031702])

    M = array([[0.88766985, -1.80940339], [0.00822629, -0.4488265],
               [0.55807272, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0.75)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [0.251547678451, -1.18305656511])
    assert_allclose(effsizes, [-1.80551427016, 0.86994164292])

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [-6.21920242113, -0.795311239268])
    assert_allclose(effsizes[1], 0.871806031702)

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0.75)
    lmls, effsizes = scanner.fast_scan(M, verbose=False)
    assert_allclose(lmls, [-5.89837796719, -1.18305656511])


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
