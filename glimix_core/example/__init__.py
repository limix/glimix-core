from numpy import arange, sqrt
from numpy.random import RandomState

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.mean import OffsetMean


def nsamples():
    return 10


def offset_mean():
    mean = OffsetMean(nsamples())
    mean.offset = 0.5

    return mean


def linear_eye_cov():
    random = RandomState(458)
    X = random.randn(nsamples(), nsamples() + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])

    cov_left = LinearCov()
    cov_left.scale = 1.0
    cov_left.X = X

    cov_right = EyeCov(nsamples())
    cov_right.scale = 1.0

    return SumCov([cov_left, cov_right])
