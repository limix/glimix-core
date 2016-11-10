from __future__ import division

from limix_math.special import logbinom

from scipy.special import gammaln
import scipy.stats as st

from numpy import ascontiguousarray as aca
from numpy import log1p
from numpy import exp
from numpy import log


class ProdLik(object):
    def __init__(self, likelihoods):
        super(ProdLik, self).__init__()
        self._likelihoods = likelihoods

    def pdf(self, x):
        return exp(self.logpdf(x))

    def logpdf(self, x):
        theta = self.theta(x)
        return (self.y * theta - self.b(theta)) / self.a() + self.c()

    @property
    def y(self):
        liks = self._likelihoods
        n = len(liks)
        return aca([liks[i].y for i in range(n)])

    def mean(self, x):
        liks = self._likelihoods
        n = len(liks)
        return aca([liks[i].mean(x[i]) for i in range(n)])

    def theta(self, x):
        liks = self._likelihoods
        n = len(liks)
        return aca([liks[i].teta(x[i]) for i in range(n)])

    @property
    def phi(self):
        liks = self._likelihoods
        n = len(liks)
        return aca([liks[i].phi for i in range(n)])

    def a(self):
        liks = self._likelihoods
        n = len(liks)
        return aca([liks[i].a() for i in range(n)])

    def b(self, theta):
        liks = self._likelihoods
        n = len(liks)
        return aca([liks[i].b(theta[i]) for i in range(n)])

    def c(self):
        liks = self._likelihoods
        n = len(liks)
        return aca([liks[i].c() for i in range(n)])

    def sample(self, x):
        raise NotImplementedError


class BernoulliProdLik(ProdLik):
    def __init__(self, link):
        super(BernoulliProdLik, self).__init__(None)
        self._link = link
        self._outcome = None

    @property
    def outcome(self):
        return self._outcome

    @outcome.setter
    def outcome(self, v):
        self._outcome = aca(v)

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
        return 1 / self.phi

    def b(self, theta):
        return theta + log1p(exp(-theta))

    def c(self):
        return logbinom(self.phi, self.y * self.phi)

    def sample(self, x, random_state=None):
        p = self.mean(x)
        return st.bernoulli(p).rvs(random_state=random_state)


class BinomialProdLik(ProdLik):
    def __init__(self, link):
        super(BinomialProdLik, self).__init__(None)
        self._link = link
        self._nsuccesses = None
        self._ntrials = None

    @property
    def nsuccesses(self):
        return self._nsuccesses

    @nsuccesses.setter
    def nsuccesses(self, v):
        self._nsuccesses = aca(v)

    @property
    def ntrials(self):
        return self._ntrials

    @ntrials.setter
    def ntrials(self, v):
        self._ntrials = aca(v)

    @property
    def y(self):
        return self._nsuccesses / self._ntrials

    def mean(self, x):
        return self._link.inv(x)

    def theta(self, x):
        m = self.mean(x)
        return log(m / (1 - m))

    @property
    def phi(self):
        return self._ntrials

    def a(self):
        return 1 / self.phi

    def b(self, theta):
        return theta + log1p(exp(-theta))

    def c(self):
        return logbinom(self.phi, self.y * self.phi)

    def sample(self, x, random_state=None):
        p = self.mean(x)
        return st.binom(self._ntrials, p).rvs(random_state=random_state)


class PoissonProdLik(ProdLik):
    def __init__(self, link):
        super(PoissonProdLik, self).__init__(None)
        self._link = link
        self._noccurrences = None

    @property
    def noccurrences(self):
        return self._noccurrences

    @noccurrences.setter
    def noccurrences(self, v):
        self._noccurrences = aca(v)

    @property
    def y(self):
        return self._noccurrences

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
        return gammaln(self._noccurrences + 1)

    def sample(self, x, random_state=None):
        lam = self.mean(x)
        return st.poisson(mu=lam).rvs(random_state=random_state)
