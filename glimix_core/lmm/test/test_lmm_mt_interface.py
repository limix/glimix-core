from numpy import inf
from numpy.random import RandomState
from numpy.testing import assert_allclose

import pytest
from glimix_core.lmm import MTLMM
from numpy_sugar.linalg import economic_qs_linear


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


def test_lmm_interface_pandas1():
    from glimix_core.lmm import LMM
    from pandas import Series, DataFrame

    random = RandomState(0)
    y = Series(random.randn(5))
    X = DataFrame(random.randn(5, 2))
    G = random.randn(5, 6)

    QS = economic_qs_linear(G)
    lmm = LMM(y, X, QS)

    assert_allclose(lmm.lml(), -6.7985689491)


def test_lmm_interface_pandas2():
    from pandas import Series, DataFrame

    random = RandomState(0)
    y = Series(random.randn(5))
    X = DataFrame(random.randn(5, 2))
    G = random.randn(5, 6)

    QS = economic_qs_linear(G)
    lmm = MTLMM(y, X, QS)

    assert_allclose(lmm.lml(), -6.7985689491)
