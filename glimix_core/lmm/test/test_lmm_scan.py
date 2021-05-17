import pytest
import scipy.stats as st
from numpy import (
    array,
    concatenate,
    errstate,
    exp,
    eye,
    inf,
    nan,
    ones,
    reshape,
    sqrt,
    zeros,
)
from numpy.linalg import inv, pinv, solve
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs, economic_qs_linear
from scipy.linalg import toeplitz
from scipy.optimize import minimize

from glimix_core._util import assert_interface
from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.lik import DeltaProdLik
from glimix_core.lmm import LMM, FastScanner
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler


def test_fast_scanner_statsmodel_gls():
    from numpy.linalg import lstsq

    def _lstsq(A, B):
        return lstsq(A, B, rcond=None)[0]

    # data = sm.datasets.longley.load()
    # data.exog = sm.add_constant(data.exog)
    # ols_resid = sm.OLS(data.endog, data.exog).fit().resid
    # resid_fit = sm.OLS(ols_resid[1:], sm.add_constant(ols_resid[:-1])).fit()
    # rho = resid_fit.params[1]
    rho = -0.3634294908774683
    # order = toeplitz(range(len(ols_resid)))
    order = toeplitz(range(16))
    sigma = rho ** order

    QS = economic_qs(sigma)
    endog = reshape(
        [
            60323.0,
            61122.0,
            60171.0,
            61187.0,
            63221.0,
            63639.0,
            64989.0,
            63761.0,
            66019.0,
            67857.0,
            68169.0,
            66513.0,
            68655.0,
            69564.0,
            69331.0,
            70551.0,
        ],
        (16,),
    )
    exog = reshape(
        [
            1.0,
            83.0,
            234289.0,
            2356.0,
            1590.0,
            107608.0,
            1947.0,
            1.0,
            88.5,
            259426.0,
            2325.0,
            1456.0,
            108632.0,
            1948.0,
            1.0,
            88.2,
            258054.0,
            3682.0,
            1616.0,
            109773.0,
            1949.0,
            1.0,
            89.5,
            284599.0,
            3351.0,
            1650.0,
            110929.0,
            1950.0,
            1.0,
            96.2,
            328975.0,
            2099.0,
            3099.0,
            112075.0,
            1951.0,
            1.0,
            98.1,
            346999.0,
            1932.0,
            3594.0,
            113270.0,
            1952.0,
            1.0,
            99.0,
            365385.0,
            1870.0,
            3547.0,
            115094.0,
            1953.0,
            1.0,
            100.0,
            363112.0,
            3578.0,
            3350.0,
            116219.0,
            1954.0,
            1.0,
            101.2,
            397469.0,
            2904.0,
            3048.0,
            117388.0,
            1955.0,
            1.0,
            104.6,
            419180.0,
            2822.0,
            2857.0,
            118734.0,
            1956.0,
            1.0,
            108.4,
            442769.0,
            2936.0,
            2798.0,
            120445.0,
            1957.0,
            1.0,
            110.8,
            444546.0,
            4681.0,
            2637.0,
            121950.0,
            1958.0,
            1.0,
            112.6,
            482704.0,
            3813.0,
            2552.0,
            123366.0,
            1959.0,
            1.0,
            114.2,
            502601.0,
            3931.0,
            2514.0,
            125368.0,
            1960.0,
            1.0,
            115.7,
            518173.0,
            4806.0,
            2572.0,
            127852.0,
            1961.0,
            1.0,
            116.9,
            554894.0,
            4007.0,
            2827.0,
            130081.0,
            1962.0,
        ],
        (16, 7),
    )
    lmm = LMM(endog, exog, QS)
    lmm.fit(verbose=False)

    sigma = lmm.covariance()
    scanner = lmm.get_fast_scanner()
    best_beta_se = _lstsq(exog.T @ _lstsq(lmm.covariance(), exog), eye(7))
    best_beta_se = sqrt(best_beta_se.diagonal())
    assert_allclose(scanner.null_beta_se, best_beta_se, atol=1e-4)

    endog = endog.copy()
    endog -= endog.mean(0)
    endog /= endog.std(0)

    exog = exog.copy()
    exog -= exog.mean(0)
    with errstate(invalid="ignore", divide="ignore"):
        exog /= exog.std(0)
    exog[:, 0] = 1

    lmm = LMM(endog, exog, QS)
    lmm.fit(verbose=False)

    sigma = lmm.covariance()
    scanner = lmm.get_fast_scanner()

    # gls_model = sm.GLS(endog, exog, sigma=sigma)
    # gls_results = gls_model.fit()
    # scale = gls_results.scale
    scale = 1.7777777777782937
    # beta_se = gls_results.bse
    beta_se = array(
        [
            0.014636888951505144,
            0.21334653097414055,
            0.7428559936739378,
            0.10174713767252333,
            0.032745906589939845,
            0.3494488802468581,
            0.4644879873404213,
        ]
    )
    our_beta_se = sqrt(scanner.null_beta_covariance.diagonal())
    # statsmodels scales the covariance matrix we pass, that is why
    # we need to account for it here.
    assert_allclose(our_beta_se, beta_se / sqrt(scale), rtol=1e-6)
    assert_allclose(scanner.null_beta_se, beta_se / sqrt(scale), rtol=1e-6)


def test_fast_scanner_redundant_candidates():
    random = RandomState(9458)
    n = 10
    X = _covariates_sample(random, n, n + 1)
    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    M = ones((n, 5))
    lmm = LMM(y, M, QS, restricted=False)

    lmm.fit(verbose=False)

    markers = M.copy()

    scanner = lmm.get_fast_scanner()

    scanner.fast_scan(markers, verbose=False)


def test_fast_scanner_set_scale_1covariate():
    random = RandomState(9458)
    n = 10
    X = _covariates_sample(random, n, n + 1)
    offset = 1.0

    y = _outcome_sample(random, offset, X)

    QS = economic_qs_linear(X)

    M = random.randn(n, 1)
    lmm = LMM(y, M, QS)

    lmm.fit(verbose=False)
    assert_allclose(lmm.scale, 5.282731934070453)
    assert_allclose(lmm.delta, 0.7029974630034005)
    assert_allclose(lmm.beta, [0.0599712498212])

    markers = M.copy() + random.randn(n, 1)

    scanner = lmm.get_fast_scanner()
    r = scanner.fast_scan(markers, verbose=False)

    assert_allclose(r["lml"], [-21.509721], rtol=1e-6)
    assert_allclose(r["effsizes0"], [[-1.43206379971882]])
    assert_allclose(r["effsizes1"], [1.412239], rtol=1e-6)
    assert_allclose(r["scale"], [0.8440354018505616], rtol=1e-6)

    beta = lmm.beta
    assert_allclose(
        scanner.fast_scan(zeros((10, 1)), verbose=False)["effsizes0"][0], beta
    )


def test_fast_scanner_set_scale_1covariate_redundant():
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

    scanner = lmm.get_fast_scanner()
    r = scanner.fast_scan(markers, verbose=False)
    assert_allclose(r["lml"][0], -22.357525517597185, rtol=1e-6)
    assert_allclose(r["effsizes0"], [[0.029985622694805182]])
    assert_allclose(r["effsizes1"][0], 0.02998562491058301, rtol=1e-6, atol=1e-6)
    assert_allclose(r["scale"], [1.0], rtol=1e-6)


def test_fast_scanner_set_scale_multicovariates():
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

    scanner = lmm.get_fast_scanner()
    r = scanner.fast_scan(markers, verbose=False)

    want = [-19.318845, -19.318845, -19.318845]
    assert_allclose(r["lml"], want, rtol=1e-6, atol=1e-6)

    assert_allclose(
        r["effsizes0"][2],
        [-0.6923007382350215, 2.3550810825973034, -0.38157769653894497],
        rtol=1e-5,
    )

    want = [-0.34615, 1.177541, -0.381578]
    assert_allclose(r["effsizes1"], want, rtol=1e-6, atol=1e-6)
    assert_allclose(r["scale"], [1.0, 1.0, 1.0])


def test_fast_scanner_effsizes_se():
    full_rank_K = array([[6.0, 14.0, 23.0], [14.0, 51.0, 86.0], [23.0, 86.0, 150.0]])
    full_rank_QS = economic_qs(full_rank_K)
    _test_fast_scanner_effsizes_se(full_rank_K, full_rank_QS, 0.2)
    _test_fast_scanner_effsizes_se(full_rank_K, full_rank_QS, 0.0)

    low_rank_K = array([[5.0, 14.0, 23.0], [14.0, 50.0, 86.0], [23.0, 86.0, 149.0]])
    low_rank_QS = economic_qs(low_rank_K)
    _test_fast_scanner_effsizes_se(low_rank_K, low_rank_QS, 0.2)


def _test_fast_scanner_effsizes_se(K0, QS, v):

    X = ones((3, 1))
    y = array([-1.0449132, 1.15229426, 0.79595129])

    M = array([[0.887669, -1.809403], [0.008226, -0.44882], [0.558072, -2.008683]])
    scanner = FastScanner(y, X, QS, v)
    r = scanner.fast_scan(M, verbose=False)
    for i in range(2):
        K = r["scale"][i] * (K0 + v * eye(3))
        XM = concatenate((X, M[:, i : (i + 1)]), axis=1)
        effsizes_se = sqrt(abs(pinv(XM.T @ solve(K, XM)).diagonal()))
        se = concatenate((r["effsizes0_se"][i], r["effsizes1_se"][i : (i + 1)]))
        assert_allclose(se, effsizes_se, atol=1e-5)

    # Redundant covariates
    M = array([[0, 1], [0, 1], [0, 1]])
    scanner = FastScanner(y, X, QS, v)
    r = scanner.fast_scan(M, verbose=False)
    for i in range(2):
        K = r["scale"][i] * (K0 + v * eye(3))
        XM = concatenate((X, M[:, i : (i + 1)]), axis=1)
        effsizes_se = sqrt(abs(pinv(XM.T @ solve(K, XM)).diagonal()))
        se = concatenate((r["effsizes0_se"][i], r["effsizes1_se"][i : (i + 1)]))
        assert_allclose(se, effsizes_se, atol=1e-5)

    M = array([[0.887669, -1.809403], [0.008226, -0.44882], [0.558072, -2.008683]])
    scanner = FastScanner(y, X, QS, v)
    r = scanner.scan(M)
    K = r["scale"] * (K0 + v * eye(3))
    XM = concatenate((X, M), axis=1)
    effsizes_se = sqrt(abs(pinv(XM.T @ solve(K, XM)).diagonal()))
    se = concatenate((r["effsizes0_se"], r["effsizes1_se"]))
    assert_allclose(se, effsizes_se, atol=1e-5)

    # Redundant covariates
    M = array([[0, 1], [0, 1], [0, 1]])
    scanner = FastScanner(y, X, QS, v)
    r = scanner.scan(M)
    K = r["scale"] * (K0 + v * eye(3))
    XM = concatenate((X, M), axis=1)
    A = XM.T @ solve(K, XM) + eye(XM.shape[0]) * 1e-9
    effsizes_se = sqrt(pinv(A, hermitian=True).diagonal())
    se = concatenate((r["effsizes0_se"], r["effsizes1_se"]))
    assert se.min() >= effsizes_se.max()


def test_lmm_scan_difficult_settings_offset():
    y = array([-1.0449132, 1.15229426, 0.79595129])
    low_rank_K = array([[5.0, 14.0, 23.0], [14.0, 50.0, 86.0], [23.0, 86.0, 149.0]])
    full_rank_K = array([[6.0, 14.0, 23.0], [14.0, 51.0, 86.0], [23.0, 86.0, 150.0]])
    low_rank_QS = economic_qs(low_rank_K)
    full_rank_QS = economic_qs(full_rank_K)
    X = ones((3, 1))

    M = array(
        [[0.88766985, -1.80940339], [0.00822629, -0.4488265], [0.55807272, -2.00868376]]
    )
    scanner = FastScanner(y, X, low_rank_QS, 0)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["scale"], [-2.9605947323337506e-16, 0.0], atol=1e-6)
    assert_allclose(r["lml"], [28.896466, 28.896466], rtol=1e-6, atol=1e-6)
    assert_allclose(r["effsizes0"], [[3.008011016293077], [-12.837513658497771]])
    assert_allclose(r["effsizes1"], [-5.5851957509459353, -9.2375605785935715])

    X = ones((3, 1))
    M = array(
        [[0.88766985, -1.80940339], [0.00822629, -0.4488265], [0.55807272, -2.00868376]]
    )
    scanner = FastScanner(y, X, low_rank_QS, 0.75)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["scale"], [0.00476137752581085, 0.012358877799161524], atol=1e-6)
    assert_allclose(r["lml"], [0.823505300549, -0.607251148058])
    assert_allclose(r["effsizes0"], [[0.3495610486128161], [0.21723441738440089]])
    assert_allclose(r["effsizes1"], [-1.79486991632, 0.872439001906])

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["scale"], [0.010459203921428906, 0.0], atol=1e-6)
    assert_allclose(r["lml"], [8.704135, 28.896466], rtol=1e-6)
    assert_allclose(r["effsizes0"], [[-0.9261322100000001], [-12.837513658497771]])
    assert_allclose(r["effsizes1"][1], -9.23756057859)

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0.75)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["scale"], [0.49332480112228455, 0.012358877799161524], atol=1e-6)
    assert_allclose(r["lml"], [-6.137441, -0.607251], rtol=1e-6, atol=1e-6)
    assert_allclose(r["effsizes0"], [[-0.9093206621461191], [0.21723441738440089]])
    assert_allclose(r["effsizes1"][1], 0.872439001906)

    M = array(
        [[0.88766985, -1.80940339], [0.00822629, -0.4488265], [0.55807272, -2.00868376]]
    )
    scanner = FastScanner(y, X, full_rank_QS, 0)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["scale"], [0.004736354793149378, 0.012301930830306581], atol=1e-6)
    assert_allclose(r["lml"], [0.636421389625, -0.795311239268])
    assert_allclose(r["effsizes0"], [[0.355188079803773], [0.2224382967889082]])
    assert_allclose(r["effsizes1"], [-1.79757295968, 0.871806031702])

    M = array(
        [[0.88766985, -1.80940339], [0.00822629, -0.4488265], [0.55807272, -2.00868376]]
    )
    scanner = FastScanner(y, X, full_rank_QS, 0.75)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["scale"], [0.004662840142176661, 0.012134195778553977], atol=1e-6)
    assert_allclose(r["lml"], [0.251547678451, -1.18305656511])
    assert_allclose(r["effsizes0"], [[0.3717198167493795], [0.2377661184233066]])
    assert_allclose(r["effsizes1"], [-1.80551427016, 0.86994164292])

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["scale"], [0.3725256920525814, 0.012301930830306581], atol=1e-6)
    assert_allclose(r["lml"], [-5.911136, -0.795311], atol=1e-6, rtol=1e-6)
    assert_allclose(r["effsizes0"], [[-0.9038187010303032], [0.2224382967889082]])
    assert_allclose(r["effsizes1"][1], 0.871806031702)

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0.75)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["scale"], [0.21713469123194049, 0.012134195778553977], atol=1e-6)
    assert_allclose(r["lml"], [-5.509792, -1.183057], atol=1e-6, rtol=1e-6)
    assert_allclose(r["effsizes0"], [[-0.8876088873393126], [0.2377661184233066]])
    assert_allclose(r["effsizes1"][1], 0.86994164292)


def test_lmm_scan_difficult_settings_multicovariates():
    y = array([-1.0449132, 1.15229426, 0.79595129])
    low_rank_K = array([[5.0, 14.0, 23.0], [14.0, 50.0, 86.0], [23.0, 86.0, 149.0]])
    full_rank_K = array([[6.0, 14.0, 23.0], [14.0, 51.0, 86.0], [23.0, 86.0, 150.0]])
    low_rank_QS = economic_qs(low_rank_K)
    full_rank_QS = economic_qs(full_rank_K)
    X = array(
        [
            [-0.40592765, 1.04348945],
            [0.92275415, -0.32394197],
            [-0.98197991, 1.22912219],
        ]
    )

    M = array(
        [[0.88766985, -1.80940339], [0.00822629, -0.4488265], [0.55807272, -2.00868376]]
    )
    scanner = FastScanner(y, X, low_rank_QS, 0)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["lml"], [28.896466, 28.896466], atol=1e-6, rtol=1e-6)
    assert_allclose(
        r["effsizes0"],
        [
            [-2.247703714834354, 0.8190282920748556],
            [-3.1760541039273127, 0.40265319454969883],
        ],
    )
    assert_allclose(r["effsizes1"], [-1.195493, 0.318417], atol=1e-6, rtol=1e-6)
    assert_allclose(
        r["scale"], [1.850371707708594e-17, 1.3877787807814457e-17], atol=1e-6
    )

    X = ones((3, 1))
    M = array(
        [[0.88766985, -1.80940339], [0.00822629, -0.4488265], [0.55807272, -2.00868376]]
    )
    scanner = FastScanner(y, X, low_rank_QS, 0.75)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["lml"], [0.823505300549, -0.607251148058])
    assert_allclose(r["effsizes0"], [[0.3495610486128161], [0.21723441738440089]])
    assert_allclose(r["effsizes1"], [-1.79486991632, 0.872439001906])
    assert_allclose(r["scale"], [0.00476137752581085, 0.012358877799161524], atol=1e-6)

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["lml"], [8.704135, 28.896466])
    assert_allclose(r["effsizes0"], [[-0.9261322100000001], [-12.837513658497771]])
    assert_allclose(r["effsizes1"][1], -9.23756057859)
    assert_allclose(r["scale"], [0.010459203921428906, 0.0], atol=1e-6)

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, low_rank_QS, 0.75)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["lml"], [-6.137441, -0.607251], atol=1e-6, rtol=1e-6)
    assert_allclose(r["effsizes0"], [[-0.9093206621461191], [0.21723441738440089]])
    assert_allclose(r["effsizes1"][1], 0.872439001906)
    assert_allclose(r["scale"], [0.49332480112228455, 0.012358877799161524], atol=1e-6)

    M = array(
        [[0.88766985, -1.80940339], [0.00822629, -0.4488265], [0.55807272, -2.00868376]]
    )
    scanner = FastScanner(y, X, full_rank_QS, 0)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["lml"], [0.636421389625, -0.795311239268])
    assert_allclose(r["effsizes1"], [-1.79757295968, 0.871806031702])
    assert_allclose(r["scale"], [0.004736354793149378, 0.012301930830306581], atol=1e-6)

    M = array(
        [[0.88766985, -1.80940339], [0.00822629, -0.4488265], [0.55807272, -2.00868376]]
    )
    scanner = FastScanner(y, X, full_rank_QS, 0.75)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["lml"], [0.251547678451, -1.18305656511])
    assert_allclose(r["effsizes0"], [[0.3717198167493795], [0.2377661184233066]])
    assert_allclose(r["effsizes1"], [-1.80551427016, 0.86994164292])
    assert_allclose(r["scale"], [0.004662840142176661, 0.012134195778553977], atol=1e-6)

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["lml"], [-5.911136, -0.795311], atol=1e-6, rtol=1e-6)
    assert_allclose(r["effsizes0"], [[-0.9038187010303032], [0.2224382967889082]])
    assert_allclose(r["effsizes1"][1], 0.871806031702)
    assert_allclose(r["scale"], [0.3725256920525814, 0.012301930830306581], atol=1e-6)

    M = array([[0.0, -1.80940339], [0.0, -0.4488265], [0.0, -2.00868376]])
    scanner = FastScanner(y, X, full_rank_QS, 0.75)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["lml"], [-5.509792, -1.183057], atol=1e-6, rtol=1e-6)
    assert_allclose(r["effsizes0"], [[-0.8876088873393126], [0.2377661184233066]])
    assert_allclose(r["effsizes1"], [0.0, 0.8699416429204303])
    assert_allclose(r["scale"], [0.21713469123194049, 0.012134195778553977], atol=1e-6)


def test_lmm_scan_very_low_rank():
    y = array([-1.0449132, 1.15229426, 0.79595129, 2.1])
    X = array(
        [
            [-0.40592765, 1.04348945],
            [0.92275415, -0.32394197],
            [-0.98197991, 1.22912219],
            [-1.0007991, 2.22912219],
        ]
    )
    G = array(
        [
            [-0.14505449, -1.1000817],
            [0.45714984, 1.82214436],
            [-1.23763742, 1.38771103],
            [-2.27377329, 0.9577192],
        ]
    )
    K = G @ G.T
    low_rank_QS = economic_qs(K)

    M = array(
        [
            [0.88766985, -1.80940339],
            [0.00822629, -0.4488265],
            [0.55807272, -2.00868376],
            [3.2, 2.1],
        ]
    )
    scanner = FastScanner(y, X, low_rank_QS, 0)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(
        r["lml"], [46.512791016862764, 46.512791016862764], atol=1e-6, rtol=1e-6
    )
    assert_allclose(
        r["effsizes0"],
        [
            [3.8616635463341358, 0.43233789455471455],
            [4.534162667593971, 3.573393734139044],
        ],
    )
    assert_allclose(
        r["effsizes1"], [2.1553245206596263, -0.684698367443129], atol=1e-6, rtol=1e-6
    )
    assert_allclose(
        r["scale"], [5.551115123125783e-17, 2.5326962749261384e-16], atol=1e-6
    )

    X = ones((4, 1))
    M = array(
        [
            [0.88766985, -1.80940339],
            [0.00822629, -0.4488265],
            [0.55807272, -2.00868376],
            [3.2, 2.1],
        ]
    )
    scanner = FastScanner(y, X, low_rank_QS, 0.75)
    r = scanner.fast_scan(M, verbose=False)
    assert_allclose(r["lml"], [-3.988506684733393, -2.852200552237104])
    assert_allclose(r["effsizes0"], [[-0.4955288599792398], [0.36297469139979893]])
    assert_allclose(r["effsizes1"], [0.5929013274071214, 0.36216887594630626])
    assert_allclose(r["scale"], [0.18324637118292808, 0.10382205995195082], atol=1e-6)


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


def test_lmm_scan_lmm_iid_prior():
    random = RandomState(9458)
    n = 30
    X = _covariates_sample(random, n, n + 1)
    markers = random.randn(n, 2)

    offset = 1.0

    y = _outcome_sample(random, offset, X)

    lmm = LMM(y, ones((n, 1)), None)

    lmm.fit(verbose=False)
    scanner = lmm.get_fast_scanner()
    lmls = scanner.fast_scan(markers, verbose=False)["lml"]
    assert_allclose(lmls[:2], [-63.16019973550036, -62.489358539276715])


def test_lmm_scan_interface():
    y = array([-1.0449132, 1.15229426, 0.79595129])
    low_rank_K = array([[5.0, 14.0, 23.0], [14.0, 50.0, 86.0], [23.0, 86.0, 149.0]])
    QS = economic_qs(low_rank_K)
    X = ones((3, 1))

    y[0] = nan
    with pytest.raises(ValueError):
        FastScanner(y, X, QS, 0.5)

    y[0] = inf
    with pytest.raises(ValueError):
        FastScanner(y, X, QS, 0.5)

    y[0] = 1
    X[0, 0] = nan
    with pytest.raises(ValueError):
        FastScanner(y, X, QS, 0.5)

    y[0] = 1
    X[0, 0] = 1
    with pytest.raises(ValueError):
        FastScanner(y, X, QS, -1)

    with pytest.raises(ValueError):
        FastScanner(y, X, QS, nan)


def test_lmm_scan_public_attrs():
    assert_interface(
        FastScanner,
        ["null_lml", "fast_scan", "scan"],
        ["null_beta", "null_beta_covariance", "null_scale", "null_beta_se"],
    )


def test_lmm_scan():
    random = RandomState(9458)
    n = 30
    X = _covariates_sample(random, n, n + 1)
    offset = 1.0
    y = _outcome_sample(random, offset, X)
    QS = economic_qs_linear(X)
    M0 = random.randn(n, 2)
    M1 = random.randn(n, 2)

    lmm = LMM(y, M0, QS)
    lmm.fit(verbose=False)

    v0 = lmm.v0
    v1 = lmm.v1
    K = v0 * X @ X.T + v1 * eye(n)
    M = concatenate((M0, M1), axis=1)

    def fun(x):
        beta = x[:4]
        scale = exp(x[4])
        return -st.multivariate_normal(M @ beta, scale * K).logpdf(y)

    res = minimize(fun, [0, 0, 0, 0, 0])
    scanner = lmm.get_fast_scanner()
    r = scanner.scan(M1)

    assert_allclose(r["lml"], -res.fun)
    assert_allclose(r["effsizes0"], res.x[:2], rtol=1e-5)
    assert_allclose(r["effsizes1"], res.x[2:4], rtol=1e-5)
    assert_allclose(r["scale"], exp(res.x[4]), rtol=1e-5)
    K = r["scale"] * lmm.covariance()
    M = concatenate((M0, M1), axis=1)
    effsizes_se = sqrt(inv(M.T @ solve(K, M)).diagonal())
    assert_allclose(effsizes_se, concatenate((r["effsizes0_se"], r["effsizes1_se"])))

    assert_allclose(scanner.null_lml(), -53.805721275578456, rtol=1e-5)
    assert_allclose(
        scanner.null_beta, [0.26521964226797085, 0.4334778669761928], rtol=1e-5
    )
    assert_allclose(
        scanner.null_beta_covariance,
        [
            [0.06302553593799207, 0.00429640179038484],
            [0.004296401790384839, 0.05591392416235412],
        ],
        rtol=1e-5,
    )
    assert_allclose(scanner.null_scale, 1.0)

    assert_allclose(scanner.null_beta, lmm.beta, rtol=1e-5)
    assert_allclose(scanner.null_beta_covariance, lmm.beta_covariance, rtol=1e-5)


def test_lmm_scan_fast_scan():
    random = RandomState(9458)
    n = 30
    X = _covariates_sample(random, n, n + 1)
    offset = 1.0
    y = _outcome_sample(random, offset, X)
    QS = economic_qs_linear(X)
    M0 = random.randn(n, 2)
    M1 = random.randn(n, 2)

    lmm = LMM(y, M0, QS)
    lmm.fit(verbose=False)

    v0 = lmm.v0
    v1 = lmm.v1
    K = v0 * X @ X.T + v1 * eye(n)
    M = concatenate((M0, M1[:, [0]]), axis=1)

    def fun(x):
        beta = x[:3]
        scale = exp(x[3])
        return -st.multivariate_normal(M @ beta, scale * K).logpdf(y)

    res = minimize(fun, [0, 0, 0, 0])
    scanner = lmm.get_fast_scanner()
    r = scanner.fast_scan(M1, verbose=False)

    assert_allclose(r["lml"][0], -res.fun)
    assert_allclose(r["effsizes0"][0], res.x[:2], rtol=1e-5)
    assert_allclose(r["effsizes1"][0], res.x[2:3], rtol=1e-5)
    assert_allclose(r["scale"][0], exp(res.x[3]), rtol=1e-5)
