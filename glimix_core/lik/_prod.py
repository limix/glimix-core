from numpy import ascontiguousarray

from ..link import IdentityLink, LogitLink, LogLink


def _sample_doc(func):
    func.__doc__ = """Sample from the likelihood distribution.

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
    r"""
    Product of Kronecker delta likelihoods.

    The product can be written as

    .. math::

        \prod_i \delta[y_i = x_i]

    Parameters
    ----------
    link : link_func
        Link function establishing :math:`g(y_i) = x_i`. Defaults to ``None``, which
        leads to the identity link function.
    """

    def __init__(self, link=None):
        if link is None:
            link = IdentityLink()
        self._link = link
        self._outcome = None

    @property
    def name(_):
        r"""Get the name of this likelihood."""
        return "Delta"

    @property
    def outcome(self):
        r"""Get or set an array of outcomes."""
        return self._outcome

    @outcome.setter
    def outcome(self, v):
        self._outcome = _aca(v)

    @staticmethod
    def mean(x):
        r"""Outcome mean."""
        return x

    @_sample_doc
    @staticmethod
    def sample(x, *_):
        return _aca(x)

    @property
    def sample_size(self):
        r"""Get the number of samples."""
        assert self.outcome is not None
        return len(self.outcome)


class BernoulliProdLik(object):
    r"""
    Product of Bernoulli likelihoods.

    The product can be written as

    .. math::

        \prod_i p_i^{y_i} (1-p_i)^{1-y_i}

    where :math:`p_i` is the probability of success.

    Parameters
    ----------
    link : link_func
        Link function establishing :math:`g(p_i) = x_i`. Defaults to ``None``, which
        leads to the :class:`glimix_core.link.LogitLink` link function.
    """

    def __init__(self, link=None):
        if link is None:
            link = LogitLink
        self._link = link
        self._outcome = None

    @property
    def name(_):
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
        import scipy.stats as st

        p = self.mean(x)
        return _aca(st.bernoulli(p).rvs(random_state=random_state))

    @property
    def sample_size(self):
        r"""Get the number of samples."""
        assert self.outcome is not None
        return len(self.outcome)


class BinomialProdLik(object):
    r"""
    Product of Binomial likelihoods.

    The product can be written as

    .. math::

        \prod_i \binom{n_i}{n_i y_i} p_i^{n_i y_i}
        (1-p_i)^{n_i - n_i y_i}

    where :math:`p_i` is the probability of success.

    Parameters
    ----------
    ntrials : array_like
        Array of number of trials.
    link : link_func
        Link function establishing :math:`g(p_i) = x_i`. Defaults to ``None``, which
        leads to the :class:`glimix_core.link.LogitLink` link function.
    """

    def __init__(self, ntrials, link=None):
        if link is None:
            link = LogitLink
        self._link = link
        self._nsuccesses = None
        self._ntrials = _aca(ntrials)

    @property
    def name(_):
        r"""Get the name of this likelihood."""
        return "Binomial"

    @property
    def ntrials(self):
        r"""Get the array of number of trials."""
        return self._ntrials

    @property
    def nsuccesses(self):
        r"""Get or set an array of successful trials."""
        return self._nsuccesses

    @nsuccesses.setter
    def nsuccesses(self, v):
        self._nsuccesses = _aca(v)

    def mean(self, x):
        r"""Mean of the number of successful trials."""
        return self._link.inv(x)

    @_sample_doc
    def sample(self, x, random_state=None):
        import scipy.stats as st

        p = self.mean(x)
        nt = ascontiguousarray(self._ntrials, dtype=int)
        return _aca(st.binom(nt, p).rvs(random_state=random_state))

    @property
    def sample_size(self):
        r"""Get the number of samples."""
        assert self.nsuccesses is not None
        return len(self.nsuccesses)


class PoissonProdLik(object):
    r"""
    Product of Poisson likelihoods.

    Parameters
    ----------
    link : link_func
        Link function establishing :math:`g(y_i) = x_i`. Defaults to ``None``, which
        leads to the :class:`glimix_core.link.LogitLink` link function.
    """

    def __init__(self, link=None):
        if link is None:
            link = LogLink()
        self._link = link
        self._noccurrences = None

    @property
    def name(_):
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
        import scipy.stats as st

        lam = self.mean(x)
        return st.poisson(mu=lam).rvs(random_state=random_state)

    @property
    def sample_size(self):
        r"""Get the number of samples."""
        assert self.noccurrences is not None
        return len(self.noccurrences)


def _aca(x):
    return ascontiguousarray(x, dtype=float)
