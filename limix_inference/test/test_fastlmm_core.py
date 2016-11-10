from __future__ import division

from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy import sqrt
from numpy import ones

from limix_math import economic_qs_linear

from lim.inference.fastlmm_core import FastLMMCore
from lim.inference import SlowLMM
from lim.util.fruits import Apples
from lim.cov import LinearCov
from lim.cov import EyeCov
from lim.cov import SumCov
from lim.mean import OffsetMean
from lim.random import RegGPSampler
from lim.util import DesignMatrixTrans


def test_optimization():
    random = RandomState(9458)
    N = 50
    X = random.randn(N, 75)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    offset = 1.2

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(N)
    mean.set_data(N, purpose='sample')

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X))
    cov_left.set_data((X, X), purpose='sample')

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((Apples(N), Apples(N)))
    cov_right.set_data((Apples(N), Apples(N)), purpose='sample')

    cov = SumCov([cov_left, cov_right])

    y = RegGPSampler(mean, cov).sample(random)

    gp = SlowLMM(y, mean, cov)
    gp.feed().maximize()
    delta = cov_right.scale / (cov_left.scale + cov_right.scale)
    QS = economic_qs_linear(DesignMatrixTrans(X).transform(X))
    flmm = FastLMMCore(y, ones((N, 1)), QS[0][0], QS[0][1], QS[1])
    flmm.delta = delta
    assert_allclose(gp.feed().value(), flmm.lml(), rtol=1e-5)
    assert_allclose(mean.offset, flmm.beta[0], rtol=1e-5)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
