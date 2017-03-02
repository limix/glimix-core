from __future__ import division

from numpy import exp, log, pi

from numpy_sugar.special import normal_cdf, normal_icdf


class Link(object):
    def __init__(self):
        super(Link, self).__init__()

    def value(self, x):
        raise NotImplementedError

    def inv(self, x):
        raise NotImplementedError

    @property
    def latent_variance(self):
        raise NotImplementedError


class LogitLink(Link):
    def __init__(self):
        super(LogitLink, self).__init__()

    def value(self, x):
        return log(x / (1 - x))

    def inv(self, x):
        return 1 / (1 + exp(-x))

    @property
    def latent_variance(self):
        return pi**2 / 3.0


class ProbitLink(Link):
    def __init__(self):
        super(ProbitLink, self).__init__()

    def value(self, x):
        return normal_icdf(x)

    def inv(self, x):
        return normal_cdf(x)

    @property
    def latent_variance(self):
        return 1.0


class LogLink(Link):
    def __init__(self):
        super(LogLink, self).__init__()

    def value(self, x):
        return log(x)

    def inv(self, x):
        return exp(x)
