from numpy import asarray, eye, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs

from glimix_core.example import linear_eye_cov
from glimix_core.glmm import GLMMNormal

ATOL = 1e-3
RTOL = 1e-2


def test_glmmnormal_copy():
    nsamples = 10

    random = RandomState(0)

    X = random.randn(nsamples, 5)
    QS = economic_qs(linear_eye_cov().value())

    eta = random.randn(nsamples)
    tau = random.rand(nsamples) * 10

    glmm0 = GLMMNormal(eta, tau, X, QS)

    assert_allclose(glmm0.lml(), -12.646439806030257, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)

    v = -4.758450057194982
    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)

    glmm1 = glmm0.copy()
    assert_allclose(glmm1.lml(), v, atol=ATOL, rtol=RTOL)

    glmm1.scale = 0.92
    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), -10.986014936977927, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)
    glmm1.fit(verbose=False)

    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), v, atol=ATOL, rtol=RTOL)

    K = asarray(
        [
            [
                1.00000001e-03,
                -2.36582704e-12,
                -1.10745946e-12,
                -5.95414742e-12,
                1.32859621e-12,
                -1.97576469e-12,
                3.10002744e-12,
                -2.98276215e-12,
                2.21766464e-12,
                -4.11149765e-13,
            ],
            [
                -2.36582704e-12,
                1.00000000e-03,
                3.38869300e-12,
                -1.21382132e-12,
                -1.18303301e-12,
                2.99685550e-12,
                -1.12667797e-12,
                1.31258715e-14,
                -3.41562159e-12,
                -1.79390668e-12,
            ],
            [
                -1.10745946e-12,
                3.38869300e-12,
                1.00000001e-03,
                -4.35318561e-12,
                -2.94696950e-12,
                3.70379702e-12,
                1.15190236e-13,
                -4.09334846e-12,
                -6.56245537e-12,
                -9.57373359e-13,
            ],
            [
                -5.95414742e-12,
                -1.21382132e-12,
                -4.35318561e-12,
                1.00000002e-03,
                -9.18763856e-13,
                -4.95358110e-12,
                -2.44724110e-12,
                -1.70406838e-12,
                1.15381873e-13,
                3.10086508e-12,
            ],
            [
                1.32859621e-12,
                -1.18303301e-12,
                -2.94696950e-12,
                -9.18763856e-13,
                1.00000001e-03,
                -1.55985955e-12,
                -2.09090626e-12,
                5.35213097e-13,
                -2.05846838e-12,
                -7.77741240e-13,
            ],
            [
                -1.97576469e-12,
                2.99685550e-12,
                3.70379702e-12,
                -4.95358110e-12,
                -1.55985955e-12,
                1.00000001e-03,
                2.16017366e-12,
                -1.25544702e-12,
                -3.18748695e-12,
                -6.45778208e-12,
            ],
            [
                3.10002744e-12,
                -1.12667797e-12,
                1.15190236e-13,
                -2.44724110e-12,
                -2.09090626e-12,
                2.16017366e-12,
                1.00000001e-03,
                -4.07492827e-12,
                -2.49548897e-12,
                -5.02704611e-12,
            ],
            [
                -2.98276215e-12,
                1.31258715e-14,
                -4.09334846e-12,
                -1.70406838e-12,
                5.35213097e-13,
                -1.25544702e-12,
                -4.07492827e-12,
                1.00000002e-03,
                -2.16846857e-12,
                -1.08364499e-12,
            ],
            [
                2.21766464e-12,
                -3.41562159e-12,
                -6.56245537e-12,
                1.15381873e-13,
                -2.05846838e-12,
                -3.18748695e-12,
                -2.49548897e-12,
                -2.16846857e-12,
                1.00000001e-03,
                4.83828810e-12,
            ],
            [
                -4.11149765e-13,
                -1.79390668e-12,
                -9.57373359e-13,
                3.10086508e-12,
                -7.77741240e-13,
                -6.45778208e-12,
                -5.02704611e-12,
                -1.08364499e-12,
                4.83828810e-12,
                1.00000001e-03,
            ],
        ]
    )

    assert_allclose(glmm0.covariance(), K, rtol=1e-5, atol=1e-5)
    assert_allclose(glmm1.covariance(), K, rtol=1e-5, atol=1e-5)


def test_glmmnormal():
    nsamples = 10

    random = RandomState(0)
    X = random.randn(nsamples, 5)
    M = random.randn(nsamples, 3)
    K = linear_eye_cov().value()
    QS = economic_qs(K)

    eta = random.randn(nsamples)
    tau = 10 * random.rand(nsamples)

    glmm = GLMMNormal(eta, tau, X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    assert_allclose(glmm.lml(), -19.284378946701814)

    assert_allclose(glmm._check_grad(), 0, atol=1e-3, rtol=RTOL)

    flmm = glmm.get_fast_scanner()
    r = flmm.fast_scan(M, verbose=False)

    assert_allclose(r["lml"], [9.64605678059, 9.17041834, 9.56927990771])
    assert_allclose(
        r["effsizes1"], [-0.0758297759308, 0.0509863368859, 0.0876858800519]
    )
    assert_allclose(
        r["scale"], [0.0053192483818597395, 0.005850105527002988, 0.00540155776161286]
    )


def test_glmmnormal_qs_zeros():
    nsamples = 10

    random = RandomState(0)

    X = random.randn(nsamples, 5)

    eta = random.randn(nsamples)
    tau = random.rand(nsamples) * 10

    glmm = GLMMNormal(eta, tau, X)

    assert_allclose(glmm.lml(), -9.14905470951951, atol=ATOL, rtol=RTOL)

    K = eye(len(eta)) * 0.5
    assert_allclose(glmm.covariance(), K, atol=1e-7)

    glmm.fit(verbose=False)

    v = -4.742404032562819
    assert_allclose(glmm.lml(), v, atol=ATOL, rtol=RTOL)

    K = zeros((len(eta), len(eta)))
    assert_allclose(glmm.covariance(), K, atol=1e-7)
