from numpy import arange, sqrt
from numpy.random import RandomState

from limix_inference.cov import EyeCov, LinearCov, SumCov
from limix_inference.mean import OffsetMean


def offset_mean():
    N = 500
    mean = OffsetMean()
    mean.offset = 1.0
    mean.set_data(N, purpose='sample')
    mean.set_data(N, purpose='learn')

    return mean


def linear_eye_cov():
    random = RandomState(458)
    N = 500
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])

    cov_left = LinearCov()
    cov_left.scale = 1.0
    cov_left.set_data((X, X), purpose='sample')
    cov_left.set_data((X, X), purpose='learn')

    cov_right = EyeCov()
    cov_right.scale = 1.0
    cov_right.set_data((arange(N), arange(N)), purpose='sample')
    cov_right.set_data((arange(N), arange(N)), purpose='learn')

    cov_right.fix('logscale')

    return SumCov([cov_left, cov_right])
