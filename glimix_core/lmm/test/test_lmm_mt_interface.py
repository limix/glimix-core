import pytest
from numpy import inf, nan, newaxis
from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core.lmm import MTLMM
from numpy_sugar.linalg import economic_qs_linear


def test_lmm_interface():
    random = RandomState(0)
    y = random.randn(4)
    X = random.randn(5, 2)
    G = random.randn(5, 6)

    QS = economic_qs_linear(G)
    with pytest.raises(ValueError):
        MTLMM(y, X, QS)

    y = random.randn(6)
    X = random.randn(6, 2)
    with pytest.raises(ValueError):
        MTLMM(y, X, QS)

    with pytest.raises(ValueError):
        MTLMM(y, X, X)

    X = random.randn(6)
    G = random.randn(6, 6)
    QS = economic_qs_linear(G)

    lmm = MTLMM(y, X, QS)
    lml0 = lmm.lml()

    lmm = MTLMM(y, X[:, newaxis], QS)
    lml1 = lmm.lml()

    assert_allclose(lml0, -5.88892141334)
    assert_allclose(lml0, lml1)

    y[0] = inf
    with pytest.raises(ValueError):
        MTLMM(y, X[:, newaxis], QS)

    y[0] = 0
    X[3] = nan
    with pytest.raises(ValueError):
        MTLMM(y, X[:, newaxis], QS)


def test_lmm_mt_interface():
    random = RandomState(0)
    y0 = random.randn(4)
    y1 = random.randn(5)
    X = random.randn(5, 2)
    G = random.randn(5, 6)

    QS = economic_qs_linear(G)
    with pytest.raises(ValueError):
        MTLMM([y0, y1], X, QS)

    with pytest.raises(ValueError):
        MTLMM([y1, y0], X, QS)

    y0 = random.randn(5)

    with pytest.raises(ValueError):
        MTLMM([y1, y0], X, QS)

    X0 = random.randn(5, 2)
    X1 = random.randn(5, 3)

    G1 = random.randn(6, 6)
    QS1 = economic_qs_linear(G1)

    with pytest.raises(ValueError):
        MTLMM([y1, y0], [X0, X1], QS1)

    lmm = MTLMM([y0, y1], [X0, X1], QS)
    lml = lmm.lml()

    assert_allclose(lml, -7.713526905947342)

    lmm = MTLMM(y0, X0, QS)
    lml = lmm.lml()
    assert_allclose(lml, -5.721633217309584)

    y0[3] = inf
    with pytest.raises(ValueError):
        MTLMM([y1, y0], [X0, X1], QS1)


def test_lmm_mt_interface_pandas():
    from pandas import Series, DataFrame

    random = RandomState(0)
    y = Series(random.randn(5))
    X = DataFrame(random.randn(5, 2))
    G = random.randn(5, 6)

    QS = economic_qs_linear(G)
    lmm = MTLMM([y, y], [X, X], QS)
    assert_allclose(lmm.lml(), -10.474952610114517)


def test_lmm_interface_pandas():
    from pandas import Series, DataFrame

    random = RandomState(0)
    y = Series(random.randn(5))
    X = DataFrame(random.randn(5, 2))
    G = random.randn(5, 6)

    QS = economic_qs_linear(G)
    lmm = MTLMM(y, X, QS)

    assert_allclose(lmm.lml(), -6.7985689491)
