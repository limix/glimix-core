from numpy import arange, concatenate, zeros
from numpy.random import RandomState
from numpy.testing import assert_allclose, assert_string_equal

from glimix_core.mean import KronMean
from optimix import Assertion


def test_kron_optimix():

    random = RandomState(0)
    # number of trais
    p = 2
    # number of covariates
    c = 3
    # sample size
    n = 4

    A = random.randn(p, p)
    F = random.randn(n, c)
    item0 = concatenate((A.ravel(), F.ravel()))
    A = random.randn(p, p)
    F = random.randn(n, c)
    item1 = concatenate((A.ravel(), F.ravel()))

    vecB = random.randn(c, p).ravel()

    mean = KronMean(c, p)
    output_example = zeros((n, p))
    a = Assertion(lambda: mean, item0, item1, output_example, vecB=vecB)
    a.assert_layout()
    a.assert_gradient()

    mean.B = [[1.0, -1.0], [0.5, +4.0], [1.0, 2.1]]
    assert_allclose(
        mean.value(item1),
        [
            [-2.5566130121, -4.816540753],
            [0.8335614751, 0.8458190667],
            [5.7447123647, 2.0348917581],
            [-2.4327915376, -5.3598099148],
        ],
    )
