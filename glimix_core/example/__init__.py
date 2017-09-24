from numpy import arange, sqrt
from numpy.random import RandomState

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.mean import OffsetMean


def nsamples():
    return 10


def offset_mean():
    mean = OffsetMean()
    mean.offset = 0.5
    mean.set_data(arange(nsamples()), purpose='sample')
    mean.set_data(arange(nsamples()), purpose='learn')

    return mean


def linear_eye_cov():
    random = RandomState(458)
    X = random.randn(nsamples(), nsamples() + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])

    cov_left = LinearCov()
    cov_left.scale = 1.0
    cov_left.set_data((X, X), purpose='sample')
    cov_left.set_data((X, X), purpose='learn')

    cov_right = EyeCov()
    cov_right.scale = 1.0
    n = nsamples()
    cov_right.set_data((arange(n), arange(n)), purpose='sample')
    cov_right.set_data((arange(n), arange(n)), purpose='learn')

    return SumCov([cov_left, cov_right])
