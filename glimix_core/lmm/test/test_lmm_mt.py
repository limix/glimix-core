from __future__ import division

from numpy import arange, corrcoef, dot, inf, nan, ones, sqrt, zeros
from numpy import concatenate
from numpy.random import RandomState
from numpy.testing import assert_, assert_allclose

import pytest
from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.lik import DeltaProdLik
from glimix_core.lmm import MTLMM
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler
from numpy_sugar.linalg import economic_qs_linear


def test_lmm_mt():
    random = RandomState(9458)
    n = 30
    X0 = _covariates_sample(random, n, 2)
    X1 = _covariates_sample(random, n, 2)
    G = _covariates_sample(random, n, n + 1)

    offset = 0.5

    y0 = _outcome_sample(random, offset, concatenate([X0, G], axis=1))
    y1 = _outcome_sample(random, offset, concatenate([X1, G], axis=1))

    y = [y0, y1]
    X = [X0, X1]
    X = [concatenate([x, ones((n, 1))], axis=1) for x in [X0, X1]]
    QS = economic_qs_linear(G)

    lmm = MTLMM(y, X, QS)

    assert_(not lmm.isfixed("delta"))
    lmm.fix("delta")
    assert_(lmm.isfixed("delta"))

    assert_(not lmm.isfixed("scale"))
    lmm.fix("scale")
    assert_(lmm.isfixed("scale"))

    lmm.scale = 1.0
    lmm.delta = 0.5

    lmm.fit(verbose=False)

    assert_allclose(
        lmm.beta[0], [1.428343850340893, -0.1237931925675461, 0.4402245351179559]
    )
    assert_allclose(
        lmm.beta[1], [-2.5806732580851466, 0.06110533548877167, 0.33064850937519386]
    )
    assert_allclose(lmm.scale, 1.0)
    assert_allclose(lmm.delta, 0.5)
    assert_allclose(lmm.v0, 0.5)
    assert_allclose(lmm.v1, 0.5)
    assert_allclose(lmm.lml(), -137.23576677088965)

    lmm.unfix("scale")
    lmm.fit(verbose=False)

    assert_allclose(
        lmm.beta[0], [1.428343850340893, -0.1237931925675461, 0.4402245351179559]
    )
    assert_allclose(
        lmm.beta[1], [-2.5806732580851466, 0.06110533548877167, 0.33064850937519386]
    )
    assert_allclose(lmm.v0, 1.4275698502922902)
    assert_allclose(lmm.v1, 1.4275698502922902)
    assert_allclose(lmm.lml(), -2.1096297207596546)

    lmm.unfix("delta")
    lmm.fit(verbose=False)

    assert_allclose(
        lmm.beta[0],
        [0.10805999290878535, -0.9196607986310403, 0.44022453511797177],
        rtol=1e-5,
    )
    assert_allclose(
        lmm.beta[1],
        [-3.755505547671947, 1.305711706302464, 0.3306485093751931],
        rtol=1e-5,
    )
    assert_allclose(lmm.v0, 19.034683831532796, rtol=1e-5)
    assert_allclose(lmm.v1, 3.923340745047039e-08, rtol=1e-5)
    assert_allclose(lmm.lml(), 16.83968479456045, rtol=1e-5)

    with pytest.raises(ValueError):
        lmm.fix("deltaa")

    with pytest.raises(ValueError):
        lmm.isfixed("deltaa")

    lmm = MTLMM(y, X, QS)
    lmm.fix("beta")
    lmm.beta = [[1.5, 1.0, -0.1], [1.1, -1.0, -0.2]]
    assert_allclose(lmm.lml(), -3.172805158447899)
    assert_allclose(lmm.beta[0][0], 1.5)
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), 6.649135553753846)
    assert_allclose(lmm.beta[0][0], 1.5)
    lmm.unfix("beta")
    assert_allclose(lmm.lml(), 10.64851984823196)
    assert_allclose(lmm.beta[0][0], 1.77367626062221)
    lmm.fit(verbose=False)
    assert_allclose(lmm.lml(), 16.83968479456045)
    assert_allclose(lmm.beta[0][0], 0.10805999290878535)
    lmm.fix("beta")
    lmm.beta = [lmm.beta[0] + 0.01, lmm.beta[1]]
    assert_allclose(lmm.lml(), 9.688653002124916)
    lmm.beta = [lmm.beta[0] - 2 * 1.01, lmm.beta[1]]
    assert_allclose(lmm.lml(), -0.9171747398890027)


def _covariates_sample(random, n, p):
    X = random.randn(n, p)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    return X


def _outcome_sample(random, offset, X):
    n = X.shape[0]
    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(arange(n), purpose="sample")

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X), purpose="sample")

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((arange(n), arange(n)), purpose="sample")

    cov = SumCov([cov_left, cov_right])

    lik = DeltaProdLik()

    return GGPSampler(lik, mean, cov).sample(random)
