from numpy import arange, sqrt
from numpy.random import RandomState

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.mean import OffsetMean


def offset_mean():
    N = 100
    mean = OffsetMean()
    mean.offset = 0.5
    mean.set_data(arange(N), purpose='sample')
    mean.set_data(arange(N), purpose='learn')

    return mean


def linear_eye_cov():
    random = RandomState(458)
    N = 100
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

    return SumCov([cov_left, cov_right])
