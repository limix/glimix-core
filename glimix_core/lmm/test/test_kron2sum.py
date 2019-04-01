import pytest
import scipy.stats as st
from numpy import concatenate
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import check_grad

from glimix_core._util import assert_interface, vec
from glimix_core.lmm import Kron2Sum


def test_kron2sum_restricted():
    random = RandomState(0)
    n = 5
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=True)

    assert_allclose(lmm.lml(), -16.580821931417656)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 3)
    assert_equal(lmm.ncovariates, 2)

    n = 5
    Y = random.randn(n, 1)
    A = random.randn(1, 1)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=True)
    lmm.name = "KronSum"

    assert_allclose(lmm.lml(), -4.582089407009583)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_allclose(
        [lmm.mean()[0], lmm.mean()[1]], [0.0497438970225256, 0.5890598193072355]
    )

    assert_allclose(
        [
            lmm.covariance()[0, 0],
            lmm.covariance()[0, 1],
            lmm.covariance()[1, 0],
            lmm.covariance()[1, 1],
        ],
        [
            4.3712532668348185,
            -0.07239366121399138,
            -0.07239366121399138,
            2.7242131674614862,
        ],
    )

    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 1)
    assert_equal(lmm.name, "KronSum")
    lmm.fit(verbose=False)
    grad = lmm.gradient()
    assert_allclose(grad["C0.Lu"], [0], atol=1e-4)
    assert_allclose(grad["C1.Lu"], [0], atol=1e-4)
    assert_allclose(lmm.lml(), -0.6930197958236421, rtol=1e-5)

    A = lmm.beta_covariance
    assert_allclose(
        A,
        [
            [4.831846800714217, -2.132053997665423],
            [-2.1320539976654262, 0.9438168959531027],
        ],
        atol=1e-5,
        rtol=1e-5,
    )


def test_kron2sum_unrestricted():
    random = RandomState(0)
    n = 5
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)

    assert_allclose(lmm.lml(), -22.700472625381742)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 3)
    assert_equal(lmm.ncovariates, 2)

    n = 5
    Y = random.randn(n, 1)
    A = random.randn(1, 1)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm.name = "KronSum"

    assert_allclose(lmm.lml(), -7.8032707190765525)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_allclose(
        [lmm.mean()[0], lmm.mean()[1]], [0.0497438970225256, 0.5890598193072355]
    )

    assert_allclose(
        [
            lmm.covariance()[0, 0],
            lmm.covariance()[0, 1],
            lmm.covariance()[1, 0],
            lmm.covariance()[1, 1],
        ],
        [
            4.3712532668348185,
            -0.07239366121399138,
            -0.07239366121399138,
            2.7242131674614862,
        ],
    )

    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 1)
    assert_equal(lmm.name, "KronSum")
    lmm.fit(verbose=False)
    grad = lmm.gradient()
    assert_allclose(grad["C0.Lu"], [0], atol=1e-4)
    assert_allclose(grad["C1.Lu"], [0], atol=1e-4)
    assert_allclose(lmm.lml(), 2.3394131683160992, rtol=1e-5)

    A = [
        [3.621697718251791, -1.5979868630471679],
        [-1.5979868630471754, 0.7081138470260712],
    ]
    assert_allclose(lmm.beta_covariance, A, atol=1e-5, rtol=1e-5)


def test_kron2sum_unrestricted_lml():
    random = RandomState(0)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    y = vec(lmm._Y)

    m = lmm.mean()
    K = lmm.covariance()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm._cov.C0.Lu = random.randn(3)
    m = lmm.mean()
    K = lmm.covariance()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm._cov.C1.Lu = random.randn(6)
    m = lmm.mean()
    K = lmm.covariance()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))


def test_kron2sum_public_attrs():
    callables = [
        "covariance",
        "fit",
        "get_fast_scanner",
        "gradient",
        "lml",
        "mean",
        "value",
    ]
    properties = [
        "A",
        "B",
        "C0",
        "C1",
        "M",
        "X",
        "beta",
        "beta_covariance",
        "name",
        "ncovariates",
        "nsamples",
        "ntraits",
    ]
    assert_interface(Kron2Sum, callables, properties)


def test_kron2sum_interface():
    random = RandomState(2)
    Y = random.randn(2, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(2, 2)
    G = random.randn(2, 4)
    with pytest.warns(UserWarning):
        lmm = Kron2Sum(Y, A, F, G, restricted=False)

    assert_allclose(
        lmm.covariance(),
        [
            [
                1.8614698749913918,
                0.16194208185167086,
                1.861454973830198,
                0.16194208185167086,
                1.861454973830198,
                0.16194208185167086,
            ],
            [
                0.16194208185167086,
                2.5548860444869783,
                0.16194208185167086,
                2.5548711433257845,
                0.16194208185167086,
                2.5548711433257845,
            ],
            [
                1.861454973830198,
                0.16194208185167086,
                2.861469874991392,
                0.16194208185167086,
                2.861454973830198,
                0.16194208185167086,
            ],
            [
                0.16194208185167086,
                2.5548711433257845,
                0.16194208185167086,
                3.5548860444869783,
                0.16194208185167086,
                3.5548711433257845,
            ],
            [
                1.861454973830198,
                0.16194208185167086,
                2.861454973830198,
                0.16194208185167086,
                3.861469874991392,
                0.16194208185167086,
            ],
            [
                0.16194208185167086,
                2.5548711433257845,
                0.16194208185167086,
                3.5548711433257845,
                0.16194208185167086,
                4.554886044486978,
            ],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.mean(),
        [
            -0.41675784808182925,
            1.6402708080344155,
            -0.056266827839408506,
            -1.7934355855434205,
            -2.136196095953572,
            -0.8417473658908108,
        ],
        atol=1e-7,
    )
    assert_allclose(lmm.lml(), -6.29061304020919, atol=1e-7)
    assert_allclose(lmm.value(), lmm.lml(), atol=1e-7)
    assert_allclose(lmm.lml(), -6.29061304020919, atol=1e-7)
    assert_allclose(
        lmm.A,
        [
            [2.9228950357645274, -3.568888742838519, 0.8427306809792194],
            [-3.568888742838519, 6.384613981254706, 0.58138966904566],
            [0.8427306809792194, 0.58138966904566, 1.5420666949903108],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.B,
        [
            [-155.3062975390569, -98.10726134921538, 124.0522362217501],
            [-371.5199529125092, -234.20234587581533, 295.5028754471357],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.X,
        [
            [-0.596159699806467, -0.019130496521151476],
            [1.175001219500291, -0.7478709492938624],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.M,
        [
            [
                -1.7425122270871933,
                -0.05591643331338421,
                2.127627641573291,
                0.06827461367924896,
                -0.5024020697902709,
                -0.016121856360740573,
            ],
            [
                3.4344052314946665,
                -2.1859482850835352,
                -4.193448625096121,
                2.6690682120308225,
                0.9902095778608936,
                -0.630253794382992,
            ],
            [
                2.127627641573291,
                0.06827461367924896,
                -3.8062495544449777,
                -0.12214083555728823,
                -0.34660109056884186,
                -0.011122273041111408,
            ],
            [
                -4.193448625096121,
                2.6690682120308225,
                7.501929214012888,
                -4.774867319035823,
                0.6831335701335212,
                -0.43480444369882226,
            ],
            [
                -0.5024020697902709,
                -0.016121856360740573,
                -0.34660109056884186,
                -0.011122273041111408,
                -0.9193180179669744,
                -0.029500501543895694,
            ],
            [
                0.9902095778608936,
                -0.630253794382992,
                0.6831335701335212,
                -0.43480444369882226,
                1.8119302471643985,
                -1.1532668830568527,
            ],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.C0, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], atol=1e-7
    )
    assert_allclose(
        lmm.C1,
        [
            [1.0000149011611938, 1.0, 1.0],
            [1.0, 2.000014901161194, 2.0],
            [1.0, 2.0, 3.000014901161194],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.beta,
        [
            -155.3062975390569,
            -371.5199529125092,
            -98.10726134921538,
            -234.20234587581533,
            124.0522362217501,
            295.5028754471357,
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.beta_covariance,
        [
            [
                33340.729151244064,
                53132.29845764565,
                21057.359168750776,
                33557.73148577052,
                -26074.486560392546,
                -41542.83161415311,
            ],
            [
                53132.29845315848,
                113983.22778712188,
                33557.73148293744,
                71992.48996769385,
                -41542.831610621106,
                -89070.58034910174,
            ],
            [
                21057.35916875069,
                33557.731485771306,
                13299.457033382136,
                21194.723227370494,
                -16467.88018346373,
                -26237.554441779976,
            ],
            [
                33557.73148293613,
                71992.48996769331,
                21194.72322558043,
                45471.00242505276,
                -26237.554439548312,
                -56256.685808786104,
            ],
            [
                -26074.486560394453,
                -41542.831614133334,
                -16467.880183465,
                -26237.55444176686,
                20395.557350283147,
                32487.137772705748,
            ],
            [
                -41542.831610652975,
                -89070.5803491147,
                -26237.55443956946,
                -56256.68580879469,
                32487.13776996609,
                69614.90810634504,
            ],
        ],
        atol=1e-7,
    )
    assert_equal(lmm.ncovariates, 2)
    assert_equal(lmm.nsamples, 2)
    assert_equal(lmm.ntraits, 3)
    assert_equal(lmm.name, "Kron2Sum")

    # print()
    # print(
    #     "assert_allclose(lmm.covariance(), "
    #     + str(lmm.covariance().tolist())
    #     + ", atol=1e-7)"
    # )
    # print("assert_allclose(lmm.mean(), " + str(lmm.mean().tolist()) + ", atol=1e-7)")

    # print("assert_allclose(lmm.lml(), " + str(lmm.lml()) + ", atol=1e-7)")
    # print("assert_allclose(lmm.value(), lmm.lml(), atol=1e-7)")
    # print("assert_allclose(lmm.lml(), " + str(lmm.lml()) + ", atol=1e-7)")
    # print("assert_allclose(lmm.A, " + str(lmm.A.tolist()) + ", atol=1e-7)")
    # print("assert_allclose(lmm.B, " + str(lmm.B.tolist()) + ", atol=1e-7)")
    # print("assert_allclose(lmm.X, " + str(lmm.X.tolist()) + ", atol=1e-7)")
    # print("assert_allclose(lmm.M, " + str(lmm.M.tolist()) + ", atol=1e-7)")
    # print("assert_allclose(lmm.C0, " + str(lmm.C0.tolist()) + ", atol=1e-7)")
    # print("assert_allclose(lmm.C1, " + str(lmm.C1.tolist()) + ", atol=1e-7)")
    # print("assert_allclose(lmm.beta, " + str(lmm.beta.tolist()) + ", atol=1e-7)")
    # print(
    #     "assert_allclose(lmm.beta_covariance, "
    #     + str(lmm.beta_covariance.tolist())
    #     + ", atol=1e-7)"
    # )
    # print("assert_allclose(lmm.ncovariates, " + str(lmm.ncovariates) + ", atol=1e-7)")
    # print("assert_allclose(lmm.nsamples, " + str(lmm.nsamples) + ", atol=1e-7)")
    # print("assert_allclose(lmm.ntraits, " + str(lmm.ntraits) + ", atol=1e-7)")
    # print("assert_allclose(lmm.name, " + str(lmm.name) + ", atol=1e-7)")


def test_kron2sum_gradient_unrestricted():
    random = RandomState(2)
    Y = random.randn(5, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm._cov.C0.Lu = random.randn(3)
    lmm._cov.C1.Lu = random.randn(6)

    def func(x):
        lmm._cov.C0.Lu = x[:3]
        lmm._cov.C1.Lu = x[3:9]
        return lmm.lml()

    def grad(x):
        lmm._cov.C0.Lu = x[:3]
        lmm._cov.C1.Lu = x[3:9]
        D = lmm.gradient()
        return concatenate((D["C0.Lu"], D["C1.Lu"]))

    assert_allclose(check_grad(func, grad, random.randn(9), epsilon=1e-8), 0, atol=1e-3)


def test_kron2sum_fit_ill_conditioned_unrestricted():
    random = RandomState(0)
    n = 30
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-157.18713011032833, -122.97307224440634])
    grad = lmm.gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 9, atol=1e-3)


def test_kron2sum_fit_C1_well_cond_unrestricted():
    random = RandomState(0)
    Y = random.randn(5, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 6)
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-19.12949904791771, -11.853021820832943], rtol=1e-5)
    grad = lmm.gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 5, atol=1e-2)


def test_kron2sum_fit_C1_well_cond_C0_fullrank_unrestricted():
    random = RandomState(0)
    Y = random.randn(5, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 6)
    lmm = Kron2Sum(Y, A, F, G, rank=2, restricted=False)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-20.15199256730784, -11.853022074873408])
    grad = lmm.gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 7, atol=1e-2)


def test_kron2sum_fit_C1_well_cond_redutant_F_unrestricted():
    random = RandomState(0)
    Y = random.randn(5, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(5, 2)
    F = concatenate((F, F), axis=1)
    G = random.randn(5, 2)
    with pytest.warns(UserWarning):
        Kron2Sum(Y, A, F, G, restricted=False)


def test_kron2sum_fit_C1_well_cond_redundant_Y_unrestricted():
    random = RandomState(0)
    Y = random.randn(5, 2)
    Y = concatenate((Y, Y), axis=1)
    A = random.randn(4, 4)
    A = A @ A.T
    F = random.randn(5, 2)
    G = random.randn(5, 2)
    with pytest.warns(UserWarning):
        lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lml = lmm.lml()
    assert_allclose(lml, -40.5860882514021)
