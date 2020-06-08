import pytest
import scipy.stats as st
from numpy import concatenate, eye, kron
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import check_grad

from glimix_core._util import assert_interface, multivariate_normal, unvec, vec
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

    assert_allclose(lmm.lml(), -16.081058762513514)
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

    assert_allclose(lmm.lml(), -3.7547099473445003)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_allclose(
        [lmm.mean()[0], lmm.mean()[1]], [0.06452826276050515, 0.4855196092646256]
    )

    assert_allclose(
        lmm.covariance(),
        [
            [
                1.9379300845374776,
                -0.02014070399890988,
                -0.7399969689595782,
                -0.1402228534612341,
                -0.4690219904509089,
            ],
            [
                -0.02014070399890988,
                1.4797056135059965,
                0.0916295591269426,
                -0.3210581381149237,
                0.2558220662032061,
            ],
            [
                -0.7399969689595782,
                0.0916295591269426,
                1.6313538475715865,
                -0.07164808824303559,
                0.5063738410283093,
            ],
            [
                -0.1402228534612341,
                -0.3210581381149237,
                -0.07164808824303559,
                3.333140431376828,
                0.3424485007527981,
            ],
            [
                -0.4690219904509089,
                0.2558220662032061,
                0.5063738410283093,
                0.3424485007527981,
                2.023907116315917,
            ],
        ],
    )

    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 1)
    assert_equal(lmm.name, "KronSum")
    lmm.fit(verbose=False)
    grad = lmm.gradient()
    assert_allclose(grad["C0.Lu"], [0], atol=1e-3)
    assert_allclose(grad["C1.Lu"], [0], atol=1e-3)
    assert_allclose(lmm.lml(), -0.6930197328322949, rtol=1e-5)

    A = lmm.beta_covariance
    assert_allclose(
        A,
        [
            [4.831901045051292, -2.1320785310203645],
            [-2.1320785310203645, 0.9438229054009741],
        ],
        atol=1e-4,
        rtol=1e-4,
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

    assert_allclose(lmm.lml(), -21.917751466118062)
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

    assert_allclose(lmm.lml(), -6.293806054115431)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_allclose(
        lmm.mean(),
        [
            0.06452826276050515,
            0.4855196092646256,
            0.1396748241908668,
            -0.264600249205993,
            -0.08238891336460354,
        ],
    )

    assert_allclose(
        lmm.covariance(),
        [
            [
                1.9379300845374776,
                -0.02014070399890988,
                -0.7399969689595782,
                -0.1402228534612341,
                -0.4690219904509089,
            ],
            [
                -0.02014070399890988,
                1.4797056135059965,
                0.0916295591269426,
                -0.3210581381149237,
                0.2558220662032061,
            ],
            [
                -0.7399969689595782,
                0.0916295591269426,
                1.6313538475715865,
                -0.07164808824303559,
                0.5063738410283093,
            ],
            [
                -0.1402228534612341,
                -0.3210581381149237,
                -0.07164808824303559,
                3.333140431376828,
                0.3424485007527981,
            ],
            [
                -0.4690219904509089,
                0.2558220662032061,
                0.5063738410283093,
                0.3424485007527981,
                2.023907116315917,
            ],
        ],
    )

    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 1)
    assert_equal(lmm.name, "KronSum")
    lmm.fit(verbose=False)
    grad = lmm.gradient()
    assert_allclose(grad["C0.Lu"], [0], atol=1e-3)
    assert_allclose(grad["C1.Lu"], [0], atol=1e-3)
    assert_allclose(lmm.lml(), 2.3394131683065957, rtol=1e-4)

    A = [
        [3.621700765362852, -1.5979882078099437],
        [-1.5979882078099474, 0.7081144405074323],
    ]
    assert_allclose(lmm.beta_covariance, A, atol=1e-4, rtol=1e-4)


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
                14.086388186569708,
                2.460064191520785,
                14.086373285408515,
                2.460064191520785,
                14.086373285408515,
                2.460064191520785,
            ],
            [
                2.460064191520785,
                24.620081892940938,
                2.460064191520785,
                24.620066991779744,
                2.460064191520785,
                24.620066991779744,
            ],
            [
                14.086373285408515,
                2.460064191520785,
                15.086388186569708,
                2.460064191520785,
                15.086373285408515,
                2.460064191520785,
            ],
            [
                2.460064191520785,
                24.620066991779744,
                2.460064191520785,
                25.620081892940938,
                2.460064191520785,
                25.620066991779744,
            ],
            [
                14.086373285408515,
                2.460064191520785,
                15.086373285408515,
                2.460064191520785,
                16.086388186569707,
                2.460064191520785,
            ],
            [
                2.460064191520785,
                24.620066991779744,
                2.460064191520785,
                25.620066991779744,
                2.460064191520785,
                26.620081892940938,
            ],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.mean(),
        [
            -0.4167578796040061,
            1.6402707883933658,
            -0.05626685903041562,
            -1.7934356050990345,
            -2.1361961254498922,
            -0.8417473850096115,
        ],
        atol=1e-7,
    )
    assert_allclose(lmm.lml(), -8.429274310765745, atol=1e-7)
    assert_allclose(lmm.value(), lmm.lml(), atol=1e-7)
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
            [-155.30629394331186, -98.10725907737903, 124.05223343020717],
            [-371.5199454125782, -234.2023411372212, 295.50286962569464],
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
        lmm.C0,
        [
            [15.191012511337567, 15.191012511337567, 15.191012511337567],
            [15.191012511337567, 15.191012511337567, 15.191012511337567],
            [15.191012511337567, 15.191012511337567, 15.191012511337567],
        ],
        atol=1e-7,
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
                180454.5259497599,
                301884.9772469587,
                114011.82157538977,
                190733.14284940425,
                -140134.13859849886,
                -234404.7154325302,
            ],
            [
                301884.97720675723,
                720240.068298505,
                190733.1428240255,
                455058.3880442711,
                -234404.7154008012,
                -559111.09170418,
            ],
            [
                114011.82157538975,
                190733.14284942497,
                72033.11947254256,
                120506.65851010302,
                -88536.94418846158,
                -148098.13529283906,
            ],
            [
                190733.14282400408,
                455058.38804427016,
                120506.6584940551,
                287512.77822660643,
                -148098.13527277583,
                -353253.7315868218,
            ],
            [
                -140134.13859850125,
                -234404.71543202398,
                -88536.94418846308,
                -148098.13529250308,
                108827.80847946613,
                182016.00480204573,
            ],
            [
                -234404.71540133483,
                -559111.0917042121,
                -148098.13527312962,
                -353253.7315868427,
                182016.00477781752,
                434044.7463389145,
            ],
        ],
        atol=1e-7,
    )
    assert_equal(lmm.ncovariates, 2)
    assert_equal(lmm.nsamples, 2)
    assert_equal(lmm.ntraits, 3)
    assert_equal(lmm.name, "Kron2Sum")


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
    assert_allclose([lml0, lml1], [-154.73966241953627, -122.97307227633186])
    grad = lmm.gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 9, atol=1e-2)


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
    assert_allclose([lml0, lml1], [-17.87016217772149, -11.853022179263597], rtol=1e-5)
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
    with pytest.warns(UserWarning):
        lmm = Kron2Sum(Y, A, F, G, rank=2, restricted=False)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-18.201106294121434, -11.853021889285362])
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
    assert_allclose(lml, -39.59627521826263)


def test_kron2sum_large_outcome():

    random = RandomState(2)
    n = 50
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    B = random.randn(2, 3)
    C0 = random.randn(3, 3)
    C0 = C0 @ C0.T
    C1 = random.randn(3, 3)
    C1 = C1 @ C1.T
    K = kron(C0, (G @ G.T)) + kron(C1, eye(n))
    y = multivariate_normal(random, kron(A, F) @ vec(B), K)
    Y = unvec(y, (n, 3))
    Y = Y / Y.std(0)

    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm.fit(verbose=False)

    assert_allclose(lmm.lml(), -12.163158697588926)
    assert_allclose(lmm.C0[0, 1], -0.004781646218546575, rtol=1e-3, atol=1e-5)
    assert_allclose(lmm.C1[0, 1], 0.03454122242999587, rtol=1e-3, atol=1e-5)
    assert_allclose(lmm.beta[2], -0.02553979383437496, rtol=1e-3, atol=1e-5)
    assert_allclose(
        lmm.beta_covariance[0, 1], 0.0051326042358990865, rtol=1e-3, atol=1e-5
    )
    assert_allclose(lmm.mean()[3], 0.3442913781854699, rtol=1e-2, atol=1e-5)
    assert_allclose(lmm.covariance()[0, 1], 0.0010745698663887468, rtol=1e-3, atol=1e-5)


def test_kron2sum_large_covariance():

    random = RandomState(0)
    n = 50
    Y = random.randn(n, 3)
    A = random.randn(3, 3)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 4)
    scale = 1e4

    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm.fit(verbose=False)

    lmm_large = Kron2Sum(Y, A, F, scale * G, restricted=False)
    lmm_large.fit(verbose=False)

    assert_allclose(lmm_large.lml(), lmm.lml())
    assert_allclose(lmm_large.C0, lmm.C0 / (scale ** 2), rtol=1e-3, atol=1e-5)
    assert_allclose(lmm_large.C1, lmm.C1, rtol=1e-3, atol=1e-5)
    assert_allclose(lmm_large.beta, lmm.beta, rtol=1e-3, atol=1e-5)
    assert_allclose(
        lmm_large.beta_covariance, lmm.beta_covariance, rtol=1e-3, atol=1e-5
    )
    assert_allclose(lmm_large.mean(), lmm.mean(), rtol=1e-2, atol=1e-5)
    assert_allclose(lmm_large.covariance(), lmm.covariance(), rtol=1e-3, atol=1e-5)


def test_kron2sum_insufficient_sample_size():
    random = RandomState(0)
    n = 2
    Y = random.randn(n, 2)
    A = random.randn(2, 2)
    A = A @ A.T
    F = random.randn(n, 2)
    G = random.randn(n, 6)
    with pytest.warns(UserWarning):
        Kron2Sum(Y, A, F, G)
