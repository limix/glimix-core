import pytest
from numpy import array, asarray, block, inf, nan, ones
from numpy.linalg import inv, pinv, svd
from numpy.testing import assert_allclose
from numpy_sugar.linalg import ddot

from glimix_core._util import check_covariates, check_economic_qs, check_outcome, hinv
from glimix_core._util.solve import heigvals, hsvd


def test_util_check_economic_qs():

    A = ones((3, 2))
    B = ones((3, 1))
    C = ones(2)

    with pytest.raises(ValueError):
        check_economic_qs(A)

    with pytest.raises(ValueError):
        check_economic_qs((A, C))

    A[0, 0] = inf
    QS = ((A, B), C)

    with pytest.raises(ValueError):
        check_economic_qs(QS)

    A[0, 0] = 1
    C[0] = nan

    with pytest.raises(ValueError):
        check_economic_qs(QS)


def test_util_check_covariates():
    A = ones(2)
    B = ones((1, 2))

    with pytest.raises(ValueError):
        check_covariates(A)

    B[0, 0] = inf
    with pytest.raises(ValueError):
        check_covariates(B)


def test_util_check_outcome():
    y = ones(5)

    y[0] = nan
    with pytest.raises(ValueError):
        check_outcome((y,), "poisson")

    y[0] = 0.5
    want = array([0.5, 1.0, 1.0, 1.0, 1.0])
    assert_allclose(check_outcome(y, "poisson"), want)

    x = ones(4)

    with pytest.raises(ValueError):
        check_outcome((y, x), "bernoulli")

    x = ones(5)

    with pytest.raises(ValueError):
        check_outcome(y, "normal")


def test_util_check_poisson_outcome():
    y = ones(5)
    y[0] = 25000 + 1
    want = array(
        [2.50000000e04, 1.00000000e00, 1.00000000e00, 1.00000000e00, 1.00000000e00]
    )
    with pytest.warns(UserWarning):
        assert_allclose(check_outcome(y, "poisson"), want)


def test_hsvd():

    A = asarray([[1.2, -0.2], [-0.2, 1.1]])
    SVD0 = hsvd(A[0, 0], A[0, 1], A[1, 1])
    SVD0 = [block(i) for i in SVD0]
    SVD1 = svd(A)
    assert_allclose(SVD0[0] @ ddot(SVD0[1], SVD0[2]), SVD1[0] @ ddot(SVD1[1], SVD1[2]))

    A = asarray([[0, 0], [0, 0]])
    SVD0 = hsvd(A[0, 0], A[0, 1], A[1, 1])
    SVD0 = [block(i) for i in SVD0]
    SVD1 = svd(A)
    assert_allclose(SVD0[0] @ ddot(SVD0[1], SVD0[2]), SVD1[0] @ ddot(SVD1[1], SVD1[2]))

    A = asarray([[1, 0], [0, 0]])
    SVD0 = hsvd(A[0, 0], A[0, 1], A[1, 1])
    SVD0 = [block(i) for i in SVD0]
    SVD1 = svd(A)
    assert_allclose(SVD0[0] @ ddot(SVD0[1], SVD0[2]), SVD1[0] @ ddot(SVD1[1], SVD1[2]))


def test_hinv():
    for A00 in [1.3, -1.3]:
        A01 = asarray([0.2, 0.1, 0.15, -0.3, 1.299, -0.1])
        A11 = asarray([0.9, 1.1, 0.75, 1.3, 1.3, 1000000.0])
        ai, bi, di = hinv(A00, A01, A11)
        for i in range(len(bi)):
            Ai = inv([[A00, A01[i]], [A01[i], A11[i]]])
            assert_allclose([[ai[i], bi[i]], [bi[i], di[i]]], Ai, atol=1e-5, rtol=1e-5)

    A = asarray([[0.3119092627599252, 0.0], [0.0, 0.0]])
    ai, bi, di = hinv(A[0, 0], A[0, 1:], A[1, 1:])
    assert_allclose([[ai[0], bi[0]], [bi[0], di[0]]], pinv(A), atol=1e-5, rtol=1e-5)

    A = asarray([[0.311909262759, 0.311909262759], [0.311909262759, 0.311909262759]])
    ai, bi, di = hinv(A[0, 0], A[0, 1:], A[1, 1:])
    assert_allclose([[ai[0], bi[0]], [bi[0], di[0]]], pinv(A), atol=1e-5, rtol=1e-5)

    A = [asarray([[0.3119092627599252, 0.0], [0.0, 0.0]])]
    A += [asarray([[0.311909262759, 0.311909262759], [0.311909262759, 0.311909262759]])]
    A00 = [0.3119092627599252, 0.311909262759]
    A01 = [0.0, 0.311909262759]
    A11 = [0.0, 0.311909262759]

    for i in range(2):
        ai, bi, di = hinv(A00[i], A01[i], A11[i])
        ai = ai[0]
        bi = bi[0]
        di = di[0]
        assert_allclose([[ai, bi], [bi, di]], pinv(A[i]), atol=1e-5, rtol=1e-5)


def test_heigvals():
    a = 1.0
    b = [1.0, 0.0, 100000000000.0, -0.03]
    d = [1.0, 0.0, 1.0, -0.4]
    heigvals(a, b, d)

    a = 0.0
    b = [1.0, 0.0, 100000000000.0, -0.03]
    d = [1.0, 0.0, 1.0, -0.4]
    heigvals(a, b, d)
