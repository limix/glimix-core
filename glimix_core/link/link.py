from __future__ import division

from numpy import asarray, exp, log, pi


class Link(object):
    def value(self, x):
        raise NotImplementedError

    def inv(self, x):
        raise NotImplementedError

    @property
    def latent_variance(self):
        raise NotImplementedError


class LogitLink(Link):
    def value(self, x):
        return log(x / (1 - x))

    def inv(self, x):
        return 1 / (1 + exp(-x))

    @property
    def latent_variance(self):
        return pi**2 / 3.0


class ProbitLink(Link):
    def value(self, x):
        return _normal_icdf(asarray(x, float))

    def inv(self, x):
        return _normal_cdf(asarray(x, float))

    @property
    def latent_variance(self):
        return 1.0


class LogLink(Link):
    def value(self, x):
        return log(x)

    def inv(self, x):
        return exp(x)

    @property
    def latent_variance(self):
        raise NotImplementedError


def _normal_cdf(x):
    import scipy.stats as st
    return st.norm.cdf(x)


def _normal_icdf(x):
    import scipy.stats as st
    return st.norm.isf(1 - x)
