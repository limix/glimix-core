from __future__ import division

from numpy import arange, sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose

from limix_inference.cov import EyeCov, LinearCov, SumCov
from limix_inference.ep import SingleExpFamEP
from limix_inference.lik import DeltaProdLik
from limix_inference.mean import OffsetMean
from limix_inference.random import GLMMSampler


def test_single_expfam_ep():
    random = RandomState(458)
    N = 100
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    offset = 1.0

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(N, purpose='sample')
    mean.set_data(N, purpose='learn')

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X), purpose='sample')
    cov_left.set_data((X, X), purpose='learn')

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((arange(N), arange(N)), purpose='sample')
    cov_right.set_data((arange(N), arange(N)), purpose='learn')

    cov = SumCov([cov_left, cov_right])

    lik = DeltaProdLik()

    y = GLMMSampler(lik, mean, cov).sample(random)

    ep = SingleExpFamEP((y, ), 'bernoulli', mean, cov)
    assert_allclose(ep.feed().value(), 337.6554144526075)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
