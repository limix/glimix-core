from __future__ import division

from numpy_sugar.special import logbinom

from scipy.special import gammaln
import scipy.stats as st

from numpy import ascontiguousarray
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
        return _aca([liks[i].y for i in range(n)])

    @property
    def ytuple(self):
        liks = self._likelihoods
        n = len(liks)
        return _aca([liks[i].ytuple for i in range(n)])

    def mean(self, x):
        liks = self._likelihoods
        n = len(liks)
        return _aca([liks[i].mean(x[i]) for i in range(n)])

    def theta(self, x):
        liks = self._likelihoods
        n = len(liks)
        return _aca([liks[i].teta(x[i]) for i in range(n)])

    @property
    def phi(self):
        liks = self._likelihoods
        n = len(liks)
        return _aca([liks[i].phi for i in range(n)])

    def a(self):
        liks = self._likelihoods
        n = len(liks)
        return _aca([liks[i].a() for i in range(n)])

    def b(self, theta):
        liks = self._likelihoods
        n = len(liks)
        return _aca([liks[i].b(theta[i]) for i in range(n)])

    def c(self):
        liks = self._likelihoods
        n = len(liks)
        return _aca([liks[i].c() for i in range(n)])

    def sample(self, x):
        raise NotImplementedError


class DeltaProdLik(ProdLik):
    def __init__(self, link=None):
        super(DeltaProdLik, self).__init__(None)
        self._link = link
        self._outcome = None
        self.name = 'Delta'

    @property
    def outcome(self):
        return self._outcome

    @outcome.setter
    def outcome(self, v):
        self._outcome = _aca(v)

    @property
    def y(self):
        return self._outcome

    @property
    def ytuple(self):
        return (self._outcome,)

    def mean(self, x):
        return x

    def theta(self, x):
        return 0

    @property
    def phi(self):
        return 1

    def a(self):
        return 1

    def b(self, theta):
        return theta

    def c(self):
        return 0

    def sample(self, x, random_state=None):
        return self.mean(x)

    @property
    def sample_size(self):
        return len(self.outcome)


class BernoulliProdLik(ProdLik):
    def __init__(self, link):
        super(BernoulliProdLik, self).__init__(None)
        self._link = link
        self._outcome = None
        self.name = 'Bernoulli'

    @property
    def outcome(self):
        return self._outcome

    @outcome.setter
    def outcome(self, v):
        self._outcome = _aca(v)

    @property
    def y(self):
        return self._outcome

    @property
    def ytuple(self):
        return (self._outcome,)

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

    def to_normal(self):
        y = self.outcome / self.outcome.std()
        y -= y.mean()
        return y

    @property
    def sample_size(self):
        return len(self.outcome)


class BinomialProdLik(ProdLik):
    def __init__(self, ntrials, link):
        super(BinomialProdLik, self).__init__(None)
        self._link = link
        self._nsuccesses = None
        self._ntrials = _aca(ntrials)
        self.name = 'Binomial'

    @property
    def ntrials(self):
        return self._ntrials

    @property
    def nsuccesses(self):
        return self._nsuccesses

    @nsuccesses.setter
    def nsuccesses(self, v):
        self._nsuccesses = _aca(v)

    @property
    def y(self):
        return self._nsuccesses / self._ntrials

    @property
    def ytuple(self):
        return (self._nsuccesses, self._ntrials)

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
        nt = ascontiguousarray(self._ntrials, dtype=int)
        return st.binom(nt, p).rvs(random_state=random_state)

    def to_normal(self):
        y = self.nsuccesses / self.ntrials
        y = y / y.std()
        y -= y.mean()
        return y

    @property
    def sample_size(self):
        return len(self.nsuccesses)


class PoissonProdLik(ProdLik):
    def __init__(self, link):
        super(PoissonProdLik, self).__init__(None)
        self._link = link
        self._noccurrences = None
        self.name = 'Poisson'

    @property
    def noccurrences(self):
        return self._noccurrences

    @noccurrences.setter
    def noccurrences(self, v):
        self._noccurrences = _aca(v)

    @property
    def y(self):
        return self._noccurrences

    @property
    def ytuple(self):
        return (self._noccurrences,)

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

    def to_normal(self):
        y = self.noccurrences
        return (y - y.mean()) / y.std()

    @property
    def sample_size(self):
        return len(self.noccurrences)


def _aca(x):
    return ascontiguousarray(x, dtype=float)
