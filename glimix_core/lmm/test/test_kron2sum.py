import pytest
import scipy.stats as st
from numpy import concatenate, eye, kron
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import check_grad

from glimix_core._util import assert_interface, multivariate_normal, unvec, vec
from glimix_core.lmm import Kron2Sum


def test_kron2sum_restricted():
    random = default_rng(0)
    n = 5
    Y = random.normal(size=(n, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 4))
    lmm = Kron2Sum(Y, A, F, G, restricted=True)

    assert_allclose(lmm.lml(), -13.9278187557)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 3)
    assert_equal(lmm.ncovariates, 2)

    n = 5
    Y = random.normal(size=(n, 1))
    A = random.normal(size=(1, 1))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 4))
    lmm = Kron2Sum(Y, A, F, G, restricted=True)
    lmm.name = "KronSum"

    assert_allclose(lmm.lml(), -3.8604834622)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_allclose([lmm.mean()[0], lmm.mean()[1]], [-0.8736140657, 0.1044747382])

    assert_allclose(
        lmm.covariance(),
        [
            [1.4638748594, -0.5112474979, 0.4868548458, -0.608025196, -0.220970285],
            [-0.5112474979, 2.5900087221, -0.8159022113, 0.5455802776, 0.3561742975],
            [0.4868548458, -0.8159022113, 2.1094730061, -0.7091682879, 0.3513920903],
            [-0.608025196, 0.5455802776, -0.7091682879, 1.8387667433, 0.0990124051],
            [-0.220970285, 0.3561742975, 0.3513920903, 0.0990124051, 2.3558066469],
        ],
    )

    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 1)
    assert_equal(lmm.name, "KronSum")
    lmm.fit(verbose=False)
    grad = lmm.gradient()
    assert_allclose(grad["C0.Lu"], [0], atol=1e-3)
    assert_allclose(grad["C1.Lu"], [0], atol=1e-3)
    assert_allclose(lmm.lml(), -2.2327723647, rtol=1e-5)

    A = lmm.beta_covariance
    assert_allclose(
        A,
        [[0.2173794271, -0.0067751264], [-0.0067751264, 0.163627761]],
        atol=1e-4,
        rtol=1e-4,
    )


def test_kron2sum_unrestricted():
    random = default_rng(0)
    n = 5
    Y = random.normal(size=(n, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 4))
    lmm = Kron2Sum(Y, A, F, G, restricted=False)

    assert_allclose(lmm.lml(), -20.1124657252)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 3)
    assert_equal(lmm.ncovariates, 2)

    n = 5
    Y = random.normal(size=(n, 1))
    A = random.normal(size=(1, 1))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 4))
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm.name = "KronSum"

    assert_allclose(lmm.lml(), -6.400073889)
    assert_allclose(lmm._check_grad(step=1e-7), 0, atol=1e-4)
    assert_allclose(
        lmm.mean(),
        [-0.8736140657, 0.1044747382, -0.0881302921, 1.2552131472, -1.2864745361],
    )

    assert_allclose(
        lmm.covariance(),
        [
            [1.4638748594, -0.5112474979, 0.4868548458, -0.608025196, -0.220970285],
            [-0.5112474979, 2.5900087221, -0.8159022113, 0.5455802776, 0.3561742975],
            [0.4868548458, -0.8159022113, 2.1094730061, -0.7091682879, 0.3513920903],
            [-0.608025196, 0.5455802776, -0.7091682879, 1.8387667433, 0.0990124051],
            [-0.220970285, 0.3561742975, 0.3513920903, 0.0990124051, 2.3558066469],
        ],
    )

    assert_equal(lmm.nsamples, n)
    assert_equal(lmm.ntraits, 1)
    assert_equal(lmm.name, "KronSum")
    lmm.fit(verbose=False)
    grad = lmm.gradient()
    assert_allclose(grad["C0.Lu"], [0], atol=1e-3)
    assert_allclose(grad["C1.Lu"], [0], atol=1e-3)
    assert_allclose(lmm.lml(), -2.44422321522, rtol=1e-4)

    A = [
        [0.130427638, -0.0040650753],
        [-0.0040650753, 0.0981766429],
    ]
    assert_allclose(lmm.beta_covariance, A, atol=1e-4, rtol=1e-4)


def test_kron2sum_unrestricted_lml():
    random = default_rng(0)
    Y = random.normal(size=(5, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(5, 2))
    G = random.normal(size=(5, 4))
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    y = vec(lmm._Y)

    m = lmm.mean()
    K = lmm.covariance()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm._cov.C0.Lu = random.normal(size=(3))
    m = lmm.mean()
    K = lmm.covariance()
    assert_allclose(lmm.lml(), st.multivariate_normal(m, K).logpdf(y))

    lmm._cov.C1.Lu = random.normal(size=(6))
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
    random = default_rng(2)
    Y = random.normal(size=(2, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(2, 2))
    G = random.normal(size=(2, 4))
    with pytest.warns(UserWarning):
        lmm = Kron2Sum(Y, A, F, G, restricted=False)

    assert_allclose(
        lmm.covariance(),
        [
            [
                1.3898116696,
                -0.0921145976,
                1.3897967685,
                -0.0921145976,
                1.3897967685,
                -0.0921145976,
            ],
            [
                -0.0921145976,
                2.4263755083,
                -0.0921145976,
                2.4263606071,
                -0.0921145976,
                2.4263606071,
            ],
            [
                1.3897967685,
                -0.0921145976,
                2.3898116696,
                -0.0921145976,
                2.3897967685,
                -0.0921145976,
            ],
            [
                -0.0921145976,
                2.4263606071,
                -0.0921145976,
                3.4263755083,
                -0.0921145976,
                3.4263606071,
            ],
            [
                1.3897967685,
                -0.0921145976,
                2.3897967685,
                -0.0921145976,
                3.3898116696,
                -0.0921145976,
            ],
            [
                -0.0921145976,
                2.4263606071,
                -0.0921145976,
                3.4263606071,
                -0.0921145976,
                4.4263755083,
            ],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.mean(),
        [
            0.1890533818,
            -2.4414673826,
            -0.5227484415,
            1.7997073827,
            -0.4130635434,
            1.144165872,
        ],
        atol=1e-7,
    )
    assert_allclose(lmm.lml(), -6.1202145396, atol=1e-7)
    assert_allclose(lmm.value(), lmm.lml(), atol=1e-7)
    assert_allclose(
        lmm.A,
        [
            [0.7837560972, 0.8493429167, -0.3780225053],
            [0.8493429167, 1.3588032243, -0.7335569048],
            [-0.3780225053, -0.7335569048, 0.9426084885],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.B,
        [
            [28.569137354, -26.4006985758, -11.3407887073],
            [9.9013805057, -10.0820334998, -5.0886226006],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.X,
        [[-0.0991980517, 0.545288714], [-0.6071856999, 0.1268278471]],
        atol=1e-7,
    )
    assert_allclose(
        lmm.M,
        [
            [
                -0.0777470779,
                0.4273733543,
                -0.0842531626,
                0.4631371068,
                0.037499096,
                -0.2061314058,
            ],
            [
                -0.4758854944,
                0.0994020985,
                -0.5157088733,
                0.1077203336,
                0.2295298595,
                -0.0479437805,
            ],
            [
                -0.0842531626,
                0.4631371068,
                -0.1347906325,
                0.7409400627,
                0.0727674158,
                -0.4000003012,
            ],
            [
                -0.5157088733,
                0.1077203336,
                -0.8250458867,
                0.1723340876,
                0.4454052626,
                -0.093035443,
            ],
            [
                0.037499096,
                -0.2061314058,
                0.0727674158,
                -0.4000003012,
                -0.0935049256,
                0.5139937705,
            ],
            [
                0.2295298595,
                -0.0479437805,
                0.4454052626,
                -0.093035443,
                -0.5723383948,
                0.1195490053,
            ],
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.C0,
        [
            [0.2364051393, 0.2364051393, 0.2364051393],
            [0.2364051393, 0.2364051393, 0.2364051393],
            [0.2364051393, 0.2364051393, 0.2364051393],
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
            28.569137354,
            9.9013805057,
            -26.4006985758,
            -10.0820334998,
            -11.3407887073,
            -5.0886226006,
        ],
        atol=1e-7,
    )
    assert_allclose(
        lmm.beta_covariance,
        [
            [
                42.5707011974,
                17.6332114628,
                -46.1577363758,
                -19.8209753095,
                -27.5728424611,
                -12.3018127703,
            ],
            [
                17.6332114628,
                50.0469928942,
                -19.8209753095,
                -59.7913284342,
                -12.3018127703,
                -39.3510924252,
            ],
            [
                -46.1577363758,
                -19.8209753095,
                100.6414980719,
                40.5712408521,
                94.4461442836,
                37.2247082688,
            ],
            [
                -19.8209753095,
                -59.7913284342,
                40.5712408521,
                109.5329990419,
                37.2247082688,
                96.1051333454,
            ],
            [
                -27.5728424611,
                -12.3018127703,
                94.4461442836,
                37.2247082688,
                101.2452095823,
                38.9396030057,
            ],
            [
                -12.3018127703,
                -39.3510924252,
                37.2247082688,
                96.1051333454,
                38.9396030057,
                95.4263019998,
            ],
        ],
        atol=1e-7,
    )
    assert_equal(lmm.ncovariates, 2)
    assert_equal(lmm.nsamples, 2)
    assert_equal(lmm.ntraits, 3)
    assert_equal(lmm.name, "Kron2Sum")


def test_kron2sum_gradient_unrestricted():
    random = default_rng(5)
    Y = random.normal(size=(5, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(5, 2))
    G = random.normal(size=(5, 4))
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm._cov.C0.Lu = random.normal(size=3)
    lmm._cov.C1.Lu = random.normal(size=6)

    def func(x):
        lmm._cov.C0.Lu = x[:3]
        lmm._cov.C1.Lu = x[3:9]
        return lmm.lml()

    def grad(x):
        lmm._cov.C0.Lu = x[:3]
        lmm._cov.C1.Lu = x[3:9]
        D = lmm.gradient()
        return concatenate((D["C0.Lu"], D["C1.Lu"]))

    assert_allclose(
        check_grad(func, grad, random.normal(size=9), epsilon=1e-8), 0, atol=1e-3
    )


def test_kron2sum_fit_ill_conditioned_unrestricted():
    random = default_rng(0)
    n = 30
    Y = random.normal(size=(n, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 4))
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-152.5971290143, -118.7196733895])
    grad = lmm.gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 9, atol=1e-2)


def test_kron2sum_fit_C1_well_cond_unrestricted():
    random = default_rng(0)
    Y = random.normal(size=(5, 2))
    A = random.normal(size=(2, 2))
    A = A @ A.T
    F = random.normal(size=(5, 2))
    G = random.normal(size=(5, 6))
    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-12.1358805747, -6.6399563682], rtol=1e-5)
    grad = lmm.gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 5, atol=1e-2)


def test_kron2sum_fit_C1_well_cond_C0_fullrank_unrestricted():
    random = default_rng(0)
    Y = random.normal(size=(5, 2))
    A = random.normal(size=(2, 2))
    A = A @ A.T
    F = random.normal(size=(5, 2))
    G = random.normal(size=(5, 6))
    with pytest.warns(UserWarning):
        lmm = Kron2Sum(Y, A, F, G, rank=2, restricted=False)
    lml0 = lmm.lml()
    lmm.fit(verbose=False)
    lml1 = lmm.lml()
    assert_allclose([lml0, lml1], [-12.7636065645, -6.639956391])
    grad = lmm.gradient()
    vars = grad.keys()
    assert_allclose(concatenate([grad[var] for var in vars]), [0] * 7, atol=1e-2)


def test_kron2sum_fit_C1_well_cond_redutant_F_unrestricted():
    random = default_rng(0)
    Y = random.normal(size=(5, 2))
    A = random.normal(size=(2, 2))
    A = A @ A.T
    F = random.normal(size=(5, 2))
    F = concatenate((F, F), axis=1)
    G = random.normal(size=(5, 2))
    with pytest.warns(UserWarning):
        Kron2Sum(Y, A, F, G, restricted=False)


def test_kron2sum_fit_C1_well_cond_redundant_Y_unrestricted():
    random = default_rng(0)
    Y = random.normal(size=(5, 2))
    Y = concatenate((Y, Y), axis=1)
    A = random.normal(size=(4, 4))
    A = A @ A.T
    F = random.normal(size=(5, 2))
    G = random.normal(size=(5, 2))
    with pytest.warns(UserWarning):
        lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lml = lmm.lml()
    assert_allclose(lml, -22.3879477239)


def test_kron2sum_large_outcome():
    random = default_rng(2)
    n = 50
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 4))
    B = random.normal(size=(2, 3))
    C0 = random.normal(size=(3, 3))
    C0 = C0 @ C0.T
    C1 = random.normal(size=(3, 3))
    C1 = C1 @ C1.T
    K = kron(C0, (G @ G.T)) + kron(C1, eye(n))
    y = multivariate_normal(random, kron(A, F) @ vec(B), K)
    Y = unvec(y, (n, 3))
    Y = Y / Y.std(0)

    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm.fit(verbose=False)

    assert_allclose(lmm.lml(), -71.8213415517)
    assert_allclose(lmm.C0[0, 1], -0.1672105838, rtol=1e-3, atol=1e-5)
    assert_allclose(lmm.C1[0, 1], 0.0637083841, rtol=1e-3, atol=1e-5)
    assert_allclose(lmm.beta[2], 0.0191120584, rtol=1e-3, atol=1e-5)
    assert_allclose(lmm.beta_covariance[0, 1], -0.1393152897, rtol=1e-3, atol=1e-5)
    assert_allclose(lmm.mean()[3], -0.0663303304, rtol=1e-2, atol=1e-5)
    assert_allclose(lmm.covariance()[0, 1], 0.4083172663, rtol=1e-3, atol=1e-5)


def test_kron2sum_large_covariance():
    random = default_rng(0)
    n = 50
    Y = random.normal(size=(n, 3))
    A = random.normal(size=(3, 3))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 4))
    scale = 1e4

    lmm = Kron2Sum(Y, A, F, G, restricted=False)
    lmm.fit(verbose=False)

    lmm_large = Kron2Sum(Y, A, F, scale * G, restricted=False)
    lmm_large.fit(verbose=False)

    assert_allclose(lmm_large.lml(), lmm.lml())
    assert_allclose(lmm_large.C0, lmm.C0 / (scale**2), rtol=1e-3, atol=1e-5)
    assert_allclose(lmm_large.C1, lmm.C1, rtol=1e-3, atol=1e-5)
    assert_allclose(lmm_large.beta, lmm.beta, rtol=1e-3, atol=1e-5)
    assert_allclose(
        lmm_large.beta_covariance, lmm.beta_covariance, rtol=1e-3, atol=1e-5
    )
    assert_allclose(lmm_large.mean(), lmm.mean(), rtol=1e-2, atol=1e-5)
    assert_allclose(lmm_large.covariance(), lmm.covariance(), rtol=1e-3, atol=1e-5)


def test_kron2sum_insufficient_sample_size():
    random = default_rng(0)
    n = 2
    Y = random.normal(size=(n, 2))
    A = random.normal(size=(2, 2))
    A = A @ A.T
    F = random.normal(size=(n, 2))
    G = random.normal(size=(n, 6))
    with pytest.warns(UserWarning):
        Kron2Sum(Y, A, F, G)
