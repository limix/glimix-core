from __future__ import division

from numpy import asarray, exp, log


def _value_doc(func):
    func.__doc__ = r"""Link function evaluated at the given points.

        Parameters
        ----------
        x : array_like
            Array of points.

        Returns
        -------
        :class:`numpy.ndarray`
            Link function values.
        """
    return func


def _inv_doc(func):
    func.__doc__ = r"""Inverse of the link function evaluated at the given points.

        Parameters
        ----------
        x : array_like
            Array of points.

        Returns
        -------
        :class:`numpy.ndarray`
            Inverse of the link function values.
        """
    return func


class IdentityLink(object):
    r"""Identity link function.

    Mathematically,

    .. math::

        g(x) = x.
    """

    @_value_doc
    def value(self, x):
        return asarray(x, float)

    @_inv_doc
    def inv(self, x):
        return asarray(x, float)


class LogitLink(object):
    r"""Logit link function.

    Mathematically,

    .. math::

        g(x) = \log(x/(1 - x)).
    """

    @_value_doc
    def value(self, x):
        return asarray(log(x / (1 - x)), float)

    @_inv_doc
    def inv(self, x):
        return asarray(1 / (1 + exp(-x)), float)


class ProbitLink(object):
    r"""Probit link function.

    Mathematically,

    .. math::

        g(x) = \Phi^{-1}(x).
    """

    @_value_doc
    def value(self, x):
        return asarray(_normal_icdf(asarray(x, float)), float)

    @_inv_doc
    def inv(self, x):
        return asarray(_normal_cdf(asarray(x, float)), float)


class LogLink(object):
    r"""Log link function.

    Mathematically,

    .. math::

        g(x) = \log(x).
    """

    @_value_doc
    def value(self, x):
        return asarray(log(x), float)

    @_inv_doc
    def inv(self, x):
        return asarray(exp(x), float)


def _normal_cdf(x):
    import scipy.stats as st

    return st.norm.cdf(x)


def _normal_icdf(x):
    import scipy.stats as st

    return st.norm.isf(1 - x)
