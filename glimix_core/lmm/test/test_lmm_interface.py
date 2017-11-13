import pytest
from numpy import newaxis, inf, nan
from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core.lmm import LMM
from numpy_sugar.linalg import economic_qs_linear


def test_lmm_interface():
    random = RandomState(0)
    y = random.randn(4)
    X = random.randn(5, 2)
    G = random.randn(5, 6)

    QS = economic_qs_linear(G)
    with pytest.raises(ValueError):
        LMM(y, X, QS)

    y = random.randn(6)
    X = random.randn(6, 2)
    with pytest.raises(ValueError):
        LMM(y, X, QS)

    with pytest.raises(ValueError):
        LMM(y, X, X)

    X = random.randn(6)
    G = random.randn(6, 6)
    QS = economic_qs_linear(G)

    lmm = LMM(y, X, QS)
    lml0 = lmm.lml()

    lmm = LMM(y, X[:, newaxis], QS)
    lml1 = lmm.lml()

    assert_allclose(lml0, -5.88892141334)
    assert_allclose(lml0, lml1)

    y[0] = inf
    with pytest.raises(ValueError):
        LMM(y, X[:, newaxis], QS)

    y[0] = 0
    X[3] = nan
    with pytest.raises(ValueError):
        LMM(y, X[:, newaxis], QS)


def test_lmm_interface_pandas():
    from pandas import Series, DataFrame

    random = RandomState(0)
    y = Series(random.randn(5))
    X = DataFrame(random.randn(5, 2))
    G = random.randn(5, 6)

    QS = economic_qs_linear(G)
    lmm = LMM(y, X, QS)
    assert_allclose(lmm.lml(), -6.7985689491)
