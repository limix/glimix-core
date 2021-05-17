def _nsamples():
    return 10


def offset_mean():
    from glimix_core.mean import OffsetMean

    mean = OffsetMean(_nsamples())
    mean.offset = 0.5

    return mean


def linear_eye_cov():
    from numpy import sqrt
    from numpy.random import RandomState

    from glimix_core.cov import EyeCov, LinearCov, SumCov

    random = RandomState(458)
    X = random.randn(_nsamples(), _nsamples() + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])

    cov_left = LinearCov(X)
    cov_left.scale = 1.0

    cov_right = EyeCov(_nsamples())
    cov_right.scale = 1.0

    return SumCov([cov_left, cov_right])
