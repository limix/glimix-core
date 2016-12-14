from __future__ import division

from numpy.random import RandomState
from numpy_sugar.linalg import sum2diag
from numpy_sugar import epsilon
from numpy_sugar.random import multivariate_normal


class GLMMSampler(object):
    def __init__(self, lik, mean, cov):
        self._lik = lik
        self._mean = mean
        self._cov = cov

    def sample(self, random_state=None):
        if random_state is None:
            random_state = RandomState()

        m = self._mean.feed('sample').value()
        K = self._cov.feed('sample').value()

        sum2diag(K, +epsilon.small, out=K)
        u = multivariate_normal(m, K, random_state)
        sum2diag(K, -epsilon.small, out=K)

        return self._lik.sample(u, random_state)
