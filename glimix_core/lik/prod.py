from __future__ import division, unicode_literals

import scipy.stats as st
from numpy import ascontiguousarray


class DeltaProdLik(object):
    r"""Represents a product of Kronecker delta likelihoods.

    The product can be written as

    .. math::

        \prod_i \delta[y_i = x_i]

    """

    def __init__(self, link=None):
        self._link = link
        self._outcome = None

    @property
    def name(self):
        return 'Delta'

    @property
    def outcome(self):
        return self._outcome

    @outcome.setter
    def outcome(self, v):
        self._outcome = _aca(v)

    def mean(self, x):
        return x

    def sample(self, x, random_state=None):  # pylint: disable=W0613
        return x

    @property
    def sample_size(self):
        return len(self.outcome)


class BernoulliProdLik(object):
    r"""Represents a product of Bernoulli likelihoods.

    The product can be written as

    .. math::

        \prod_i g(x_i)^{y_i} (1-g(x_i))^{1-y_i}

    where :math:`g(\cdot)` is the inverse of the link function.

    """

    def __init__(self, link):
        self._link = link
        self._outcome = None

    @property
    def name(self):
        return 'Bernoulli'

    @property
    def outcome(self):
        return self._outcome

    @outcome.setter
    def outcome(self, v):
        self._outcome = _aca(v)

    def mean(self, x):
        return self._link.inv(x)

    def sample(self, x, random_state=None):
        p = self.mean(x)
        return st.bernoulli(p).rvs(random_state=random_state)

    @property
    def sample_size(self):
        return len(self.outcome)


class BinomialProdLik(object):
    r"""Represents a product of Binomial likelihoods.

    The product can be written as

    .. math::

        \prod_i \binom{n_i}{n_i y_i} g(x_i)^{n_i y_i}
        (1-g(x_i))^{n_i - n_i y_i}

    where :math:`g(x)` is the inverse of the link function.

    """

    def __init__(self, ntrials, link):
        self._link = link
        self._nsuccesses = None
        self._ntrials = _aca(ntrials)

    @property
    def name(self):
        return 'Binomial'

    @property
    def ntrials(self):
        return self._ntrials

    @property
    def nsuccesses(self):
        return self._nsuccesses

    @nsuccesses.setter
    def nsuccesses(self, v):
        self._nsuccesses = _aca(v)

    def mean(self, x):
        return self._link.inv(x)

    def sample(self, x, random_state=None):
        p = self.mean(x)
        nt = ascontiguousarray(self._ntrials, dtype=int)
        return st.binom(nt, p).rvs(random_state=random_state)

    @property
    def sample_size(self):
        return len(self.nsuccesses)


class PoissonProdLik(object):
    r"""TODO."""

    def __init__(self, link):
        self._link = link
        self._noccurrences = None

    @property
    def name(self):
        return 'Poisson'

    @property
    def noccurrences(self):
        return self._noccurrences

    @noccurrences.setter
    def noccurrences(self, v):
        self._noccurrences = _aca(v)

    def mean(self, x):
        return self._link.inv(x)

    def sample(self, x, random_state=None):
        lam = self.mean(x)
        return st.poisson(mu=lam).rvs(random_state=random_state)

    @property
    def sample_size(self):
        return len(self.noccurrences)


def _aca(x):
    return ascontiguousarray(x, dtype=float)
