from __future__ import division

from numpy import exp
from numpy import log
from numpy import log1p

from scipy.special import gammaln
import scipy.stats as st

from numpy_sugar.special import logbinom


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

    @property
    def ytuple(self):
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


class DeltaLik(ExpFamLik):
    def __init__(self, link=None):
        super(DeltaLik, self).__init__()
        self._link = link
        self._outcome = None
        self.name = 'Delta'

    @property
    def outcome(self):
        return self._outcome

    @outcome.setter
    def outcome(self, v):
        self._outcome = v

    @property
    def y(self):
        return self._outcome

    @property
    def ytuple(self):
        raise (self._outcome,)

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


class BernoulliLik(ExpFamLik):
    def __init__(self, link):
        super(BernoulliLik, self).__init__()
        self._link = link
        self._outcome = None
        self.name = 'Bernoulli'

    @property
    def outcome(self):
        return self._outcome

    @outcome.setter
    def outcome(self, v):
        self._outcome = v

    @property
    def y(self):
        return self._outcome

    @property
    def ytuple(self):
        raise (self._outcome,)

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
        p = self.mean(x)
        return st.bernoulli(p).rvs(random_state=random_state)


class BinomialLik(ExpFamLik):
    def __init__(self, ntrials, link):
        super(BinomialLik, self).__init__()
        self._link = link
        self._nsuccesses = None
        self._ntrials = ntrials
        self.name = 'Binomial'

    @property
    def nsuccesses(self):
        return self._nsuccesses

    @nsuccesses.setter
    def nsuccesses(self, v):
        self._nsuccesses = v

    @property
    def y(self):
        return self._nsuccesses / self._ntrials

    @property
    def ytuple(self):
        raise (self._nsuccesses, self._ntrials)

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


class PoissonLik(ExpFamLik):
    def __init__(self, link):
        super(PoissonLik, self).__init__()
        self._link = link
        self._noccurrences = None
        self.name = 'Poisson'

    @property
    def noccurrences(self):
        return self._noccurrences

    @noccurrences.setter
    def noccurrences(self, v):
        self._noccurrences = v

    @property
    def y(self):
        return self._noccurrences

    @property
    def ytuple(self):
        raise (self._noccurrences,)

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
