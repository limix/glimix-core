from __future__ import division

from numpy import exp
from numpy import pi
from numpy import log
from numpy import log1p

from scipy.special import gammaln

from limix_math.special import logbinom


class ExpFamLik(object):
    def __init__(self):
        super(ExpFamLik, self).__init__()

    def pdf(self, x):
        return exp(self.logpdf(x))

    def logpdf(self, x):
        theta = self.theta(x)
        return (self.y * theta - self.b(theta)) / self.a() + self.c()

    @property
    def y(self):
        raise NotImplementedError

    def mean(self, x):
        raise NotImplementedError

    def theta(self, x):
        raise NotImplementedError

    @property
    def phi(self):
        raise NotImplementedError

    def a(self):
        raise NotImplementedError

    def b(self, theta):
        raise NotImplementedError

    def c(self):
        raise NotImplementedError

    def sample(self, x):
        raise NotImplementedError


class BernoulliLik(ExpFamLik):
    def __init__(self, outcome, link):
        super(BernoulliLik, self).__init__()
        self._outcome = outcome
        self._link = link

    @staticmethod
    def latent_variance(link):
        from ..link import ProbitLink
        from ..link import LogitLink
        if isinstance(link, ProbitLink):
            return 1
        elif isinstance(link, LogitLink):
            return pi**2 / 3
        raise ValueError()

    @property
    def y(self):
        return self._outcome

    def mean(self, x):
        return self._link.inv(x)

    def theta(self, x):
        m = self.mean(x)
        return log(m / (1 - m))

    @property
    def phi(self):
        return 1

    def a(self):
        return 1

    def b(self, theta):
        return theta + log1p(exp(-theta))

    def c(self):
        return 0

    def sample(self, x, random_state=None):
        import scipy.stats as st
        p = self.mean(x)
        return st.bernoulli(p).rvs(random_state=random_state)


class BinomialLik(ExpFamLik):
    def __init__(self, k, n, link):
        super(BinomialLik, self).__init__()
        self._k = k
        self._n = n
        self._link = link

    @staticmethod
    def latent_variance(link):
        from ..link import ProbitLink
        from ..link import LogitLink
        if isinstance(link, ProbitLink):
            return 1
        elif isinstance(link, LogitLink):
            return pi**2 / 3
        raise ValueError()

    @property
    def y(self):
        return self._k / self._n

    def mean(self, x):
        return self._link.inv(x)

    def theta(self, x):
        m = self.mean(x)
        return log(m / (1 - m))

    @property
    def phi(self):
        return self._n

    def a(self):
        return 1 / self.phi

    def b(self, theta):
        return theta + log1p(exp(-theta))

    def c(self):
        return logbinom(self.phi, self.y * self.phi)

    def sample(self, x, random_state=None):
        import scipy.stats as st
        p = self.mean(x)
        return st.binom(self._n, p).rvs(random_state=random_state)


class PoissonLik(ExpFamLik):
    def __init__(self, k, link):
        super(PoissonLik, self).__init__()
        self._k = k
        self._link = link

    @property
    def y(self):
        return self._k

    def mean(self, x):
        return self._link.inv(x)

    def theta(self, x):
        m = self.mean(x)
        return log(m)

    @property
    def phi(self):
        return 1

    def a(self):
        return self.phi

    def b(self, theta):
        return exp(theta)

    def c(self):
        return gammaln(self._k + 1)

    def sample(self, x, random_state=None):
        import scipy.stats as st
        lam = self.mean(x)
        return st.poisson(mu=lam).rvs(random_state=random_state)


if __name__ == '__main__':
    print(gammaln(3))
