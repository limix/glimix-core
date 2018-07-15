from __future__ import division, unicode_literals

import scipy.stats as st
from numpy import ascontiguousarray


def _sample_doc(func):
    func.__doc__ = r"""Sample from the likelihood distribution.

        Parameters
        ----------
        x : array_like
            Array of likelihood parameters.
        random_state : random_state
            Set the initial random state.

        Returns
        -------
        numpy.ndarray
            Sampled outcome.
        """
    return func


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
        r"""Get the name of this likelihood."""
        return "Delta"

    @property
    def outcome(self):
        r"""Get or set an array of outcomes."""
        return self._outcome

    @outcome.setter
    def outcome(self, v):
        self._outcome = _aca(v)

    def mean(self, x):
        r"""Outcome mean."""
        return x

    @_sample_doc
    def sample(self, x, random_state=None):
        return _aca(x)

    @property
    def sample_size(self):
        r"""Get the number of samples."""
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
        r"""Get the name of this likelihood."""
        return "Bernoulli"

    @property
    def outcome(self):
        r"""Get or set an array of outcomes."""
        return self._outcome

    @outcome.setter
    def outcome(self, v):
        self._outcome = _aca(v)

    def mean(self, x):
        r"""Outcome mean."""
        return self._link.inv(x)

    @_sample_doc
    def sample(self, x, random_state=None):
        p = self.mean(x)
        return _aca(st.bernoulli(p).rvs(random_state=random_state))

    @property
    def sample_size(self):
        r"""Get the number of samples."""
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
        r"""Get the name of this likelihood."""
        return "Binomial"

    @property
    def ntrials(self):
        r"""Get the array of number of trials."""
        return self._ntrials

    @property
    def nsuccesses(self):
        r"""Get or set an array of successfully trials."""
        return self._nsuccesses

    @nsuccesses.setter
    def nsuccesses(self, v):
        self._nsuccesses = _aca(v)

    def mean(self, x):
        r"""Mean of the number of successfully trials."""
        return self._link.inv(x)

    @_sample_doc
    def sample(self, x, random_state=None):
        p = self.mean(x)
        nt = ascontiguousarray(self._ntrials, dtype=int)
        return _aca(st.binom(nt, p).rvs(random_state=random_state))

    @property
    def sample_size(self):
        r"""Get the number of samples."""
        return len(self.nsuccesses)


class PoissonProdLik(object):
    r"""TODO."""

    def __init__(self, link):
        self._link = link
        self._noccurrences = None

    @property
    def name(self):
        r"""Get the name of this likelihood."""
        return "Poisson"

    @property
    def noccurrences(self):
        r"""Get or set an array of number of occurrences."""
        return self._noccurrences

    @noccurrences.setter
    def noccurrences(self, v):
        self._noccurrences = _aca(v)

    def mean(self, x):
        r"""Mean of the number of occurrences."""
        return self._link.inv(x)

    @_sample_doc
    def sample(self, x, random_state=None):
        lam = self.mean(x)
        return st.poisson(mu=lam).rvs(random_state=random_state)

    @property
    def sample_size(self):
        r"""Get the number of samples."""
        return len(self.noccurrences)


def _aca(x):
    return ascontiguousarray(x, dtype=float)
