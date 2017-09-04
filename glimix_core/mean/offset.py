from __future__ import division

from numpy import asarray, full, ones

from optimix import Function, Scalar


class OffsetMean(Function):
    r"""Offset mean function.

    The mathematical representation is

    .. math::

        f(n) = o \mathbf 1

    where :math:`\mathbf 1` is a :math:`n`-sized vector of ones.
    """

    def __init__(self):
        Function.__init__(self, offset=Scalar(1.0))

    def value(self, x):
        r"""Offset function evaluated at ``x``.

        Parameters
        ----------
        x : array_like
            Sample ids.

        Returns
        -------
        :math:`o \mathbf 1`.
        """
        x = asarray(x)
        return full(x.shape, self.variables().get('offset').value)

    def gradient(self, x):
        r"""Offset function gradient.

        Parameters
        ----------
        x : array_like
            Sample ids.

        Returns
        -------
        :math:`\mathbf 1`.
        """
        x = asarray(x)
        return dict(offset=ones(x.shape))

    @property
    def offset(self):
        r"""Offset parameter."""
        return self.variables().get('offset').value

    @offset.setter
    def offset(self, v):
        self.variables().get('offset').value = v
