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

    mean.A = A
    mean.F = F
    assert_allclose(
        mean.feed().value(),
        [
            [-2.5566130121, -4.816540753],
            [0.8335614751, 0.8458190667],
            [5.7447123647, 2.0348917581],
            [-2.4327915376, -5.3598099148],
        ],
    )

    assert_allclose(
        mean.AF,
        [
            [
                -3.8143686578,
                0.9765578653,
                1.2915360348,
                0.523766958,
                -0.1340952562,
                -0.1773462297,
            ],
            [
                -1.1088532258,
                3.3911928849,
                -2.1729373191,
                0.152261287,
                -0.4656589178,
                0.2983751367,
            ],
            [
                0.0683668431,
                -0.2796674732,
                2.2900933479,
                -0.009387738,
                0.0384023137,
                -0.3144623224,
            ],
            [
                2.1953381891,
                0.2315037062,
                0.5650047068,
                -0.3014510941,
                -0.0317887448,
                -0.0775831659,
            ],
            [
                -0.799258654,
                0.2046268714,
                0.270627054,
                2.1804977242,
                -0.5582528575,
                -0.7383112743,
            ],
            [
                -0.2323478972,
                0.7105868634,
                -0.4553149191,
                0.6338799818,
                -1.9385877536,
                1.2421675261,
            ],
            [
                0.0143255138,
                -0.0586012177,
                0.4798636658,
                -0.0390821547,
                0.1598729288,
                -1.3091401963,
            ],
            [
                0.460008773,
                0.0485090344,
                0.1183904709,
                -1.2549730649,
                -0.1323399361,
                -0.3229869968,
            ],
        ],
    )
