from numpy import asarray
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy_sugar.linalg import economic_qs
from optimix import check_grad

from glimix_core.example import linear_eye_cov, nsamples
from glimix_core.glmm import GLMMNormal

ATOL = 1e-6
RTOL = 1e-6


def test_glmmnormal_copy():
    random = RandomState(0)

    X = random.randn(nsamples(), 5)
    QS = economic_qs(linear_eye_cov().feed().value())

    eta = random.randn(nsamples())
    tau = random.rand(nsamples()) * 10

    glmm0 = GLMMNormal(eta, tau, X, QS)

    assert_allclose(glmm0.lml(), -12.646439806030257, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)

    v = -4.758450057194982
    assert_allclose(glmm0.lml(), v)

    glmm1 = glmm0.copy()
    assert_allclose(glmm1.lml(), v)

    glmm1.scale = 0.92
    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), -10.986014936977927, atol=ATOL, rtol=RTOL)

    glmm0.fit(verbose=False)
    glmm1.fit(verbose=False)

    assert_allclose(glmm0.lml(), v, atol=ATOL, rtol=RTOL)
    assert_allclose(glmm1.lml(), v, atol=ATOL, rtol=RTOL)

    K = asarray([[
        1.00000000e-03, -1.21864053e-12, -5.70453783e-13, -3.06698895e-12,
        6.84361607e-13, -1.01771892e-12, 1.59682810e-12, -1.53642460e-12,
        1.14232189e-12, -2.11783770e-13
    ], [
        -1.21864053e-12, 1.00000000e-03, 1.74552010e-12, -6.25240913e-13,
        -6.09381818e-13, 1.54368410e-12, -5.80353261e-13, 6.76115317e-15,
        -1.75939105e-12, -9.24043622e-13
    ], [
        -5.70453783e-13, 1.74552010e-12, 1.00000001e-03, -2.24233147e-12,
        -1.51798776e-12, 1.90783058e-12, 5.93346380e-14, -2.10848903e-12,
        -3.38032915e-12, -4.93144241e-13
    ], [
        -3.06698895e-12, -6.25240913e-13, -2.24233147e-12, 1.00000001e-03,
        -4.73256437e-13, -2.55159596e-12, -1.26057702e-12, -8.77767803e-13,
        5.94333505e-14, 1.59725957e-12
    ], [
        6.84361607e-13, -6.09381818e-13, -1.51798776e-12, -4.73256437e-13,
        1.00000000e-03, -8.03485650e-13, -1.07702849e-12, 2.75688950e-13,
        -1.06031970e-12, -4.00615507e-13
    ], [
        -1.01771892e-12, 1.54368410e-12, 1.90783058e-12, -2.55159596e-12,
        -8.03485650e-13, 1.00000001e-03, 1.11270821e-12, -6.46682366e-13,
        -1.64187861e-12, -3.32641181e-12
    ], [
        1.59682810e-12, -5.80353261e-13, 5.93346380e-14, -1.26057702e-12,
        -1.07702849e-12, 1.11270821e-12, 1.00000001e-03, -2.09900076e-12,
        -1.28542956e-12, -2.58943788e-12
    ], [
        -1.53642460e-12, 6.76115317e-15, -2.10848903e-12, -8.77767803e-13,
        2.75688950e-13, -6.46682366e-13, -2.09900076e-12, 1.00000001e-03,
        -1.11698093e-12, -5.58186921e-13
    ], [
        1.14232189e-12, -1.75939105e-12, -3.38032915e-12, 5.94333505e-14,
        -1.06031970e-12, -1.64187861e-12, -1.28542956e-12, -1.11698093e-12,
        1.00000001e-03, 2.49220839e-12
    ], [
        -2.11783770e-13, -9.24043622e-13, -4.93144241e-13, 1.59725957e-12,
        -4.00615507e-13, -3.32641181e-12, -2.58943788e-12, -5.58186921e-13,
        2.49220839e-12, 1.00000000e-03
    ]])

    assert_allclose(glmm0.covariance(), K, atol=1e-10)
    assert_allclose(glmm1.covariance(), K, atol=1e-10)


def test_glmmnormal():
    random = RandomState(0)
    X = random.randn(nsamples(), 5)
    M = random.randn(nsamples(), 3)
    K = linear_eye_cov().feed().value()
    QS = economic_qs(K)

    eta = random.randn(nsamples())
    tau = 10 * random.rand(nsamples())

    glmm = GLMMNormal(eta, tau, X, QS)
    glmm.beta = asarray([1.0, 0, 0.5, 0.1, 0.4])

    assert_allclose(glmm.lml(), -19.284378946701814)

    assert_allclose(check_grad(glmm), 0, atol=1e-3, rtol=RTOL)

    flmm = glmm.get_fast_scanner()
    lmls, effsizes = flmm.fast_scan(M, verbose=False)

    assert_allclose(lmls, [9.64605678059, 9.17041834, 9.56927990771])
    assert_allclose(effsizes, [-0.0758297759308, 0.0509863368859,
                               0.0876858800519])

    assert_allclose(flmm.null_lml(), 8.89275310307)
    flmm.set_scale(1.5)
    assert_allclose(flmm.null_lml(), -13.5839989949)
    flmm.set_scale(0.5)
    assert_allclose(flmm.null_lml(), -8.13216534935)
