from __future__ import division

from numpy import (
    diag_indices_from,
    dot,
    exp,
    log,
    ones,
    stack,
    tril_indices_from,
    zeros,
    zeros_like,
)

from numpy_sugar import epsilon
from numpy_sugar.linalg import economic_qs
from optimix import Function, Vector

from ..util.classes import NamedClass


class FreeFormCov(NamedClass, Function):
    r"""General semi-definite positive matrix.

    A :math:`d`-by-:math:`d` covariance matrix :math:`\mathrm K` will have
    ``((d + 1) * d) / 2`` parameters defining the lower triangular part of
    the Cholesky matrix ``L``: :math:`\mathrm L\mathrm L^\intercal=\mathrm K`.

    Parameters
    ----------
    dim : int
        Dimension :math:`d` of the free-form covariance matrix.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import FreeFormCov
        >>>
        >>> cov = FreeFormCov(2)
        >>> print(cov.value([0, 1], [0, 1]))
        [[1. 1.]
         [1. 2.]]
        >>> print(cov.L)
        [[1. 0.]
         [1. 1.]]
        >>> print(cov.Lu)
        [1. 1. 1.]
        >>> g = cov.gradient([0, 1], [0, 1])
        >>> print(g['Lu'].shape)
        (2, 2, 3)
        >>> print(g['Lu'])
        [[[2. 0. 0.]
          [1. 1. 0.]]
        <BLANKLINE>
         [[1. 1. 0.]
          [0. 2. 2.]]]
        >>> cov.Lu[1] = -2
        >>> print(cov.L)
        [[ 1.  0.]
         [-2.  1.]]
        >>> print(cov.value([0, 1], [0, 1]))
        [[ 1. -2.]
         [-2.  5.]]
        >>> print(cov)
        FreeFormCov()
          Lu: [ 1. -2.  1.]
        >>> cov.name = "covname"
        >>> print(cov)
        FreeFormCov(): covname
          Lu: [ 1. -2.  1.]
    """

    def __init__(self, dim):
        dim = int(dim)
        tsize = ((dim + 1) * dim) // 2
        self._L = zeros((dim, dim))
        self._tril = tril_indices_from(self._L)
        self._tril1 = tril_indices_from(self._L, k=-1)
        self._diag = diag_indices_from(self._L)
        self._L[self._tril1] = 1
        self._L[self._diag] = 0
        self._jitter = epsilon.tiny
        Function.__init__(self, Llow=Vector(ones(tsize - dim)), Llogd=Vector(zeros(dim)))
        self.variables().get("Llogd").bounds = (-20.0, +10)
        NamedClass.__init__(self)

    def economic_qs(self):
        return economic_qs(self.feed().value())

    def eigh(self):
        from numpy.linalg import eigh

        Sn, Un = eigh(self.feed().value())
        Sn += self._jitter
        return Sn, Un

    @property
    def L(self):
        """Cholesky decomposition of the covariance matrix."""
        self._L[self._tril1] = self.variables().get("Llow").value
        self._L[self._diag] = exp(self.variables().get("Llogd").value)
        return self._L

    @L.setter
    def L(self, value):
        self._L[:] = value
        self.variables().get("Llow").value = self._L[self._tril1]
        self.variables().get("Llogd").value = log(self._L[self._diag])

    # @property
    # def Lu(self):
    #     return self.variables().get("Lu").value

    # @Lu.setter
    # def Lu(self, value):
    #     self.variables().get("Lu").value = value

    def value(self, x0, x1):
        r"""Covariance function evaluated at ``(x0, x1)``.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        array_like
            Submatrix of :math:`\mathrm L\mathrm L^\intercal`, row and
            column-indexed by ``x0`` and ``x1``.
        """
        return dot(self.L, self.L.T)[x0, ...][..., x1]

    def gradient(self, x0, x1):
        r"""Derivative of the covariance function evaluated at ``(x0, x1)``.

        Derivative over the lower triangular part of :math:`\mathrm L`.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        dict
            Dictionary having the `Lu` key for the lower triangular part
            of the derivative of :math:`\mathrm L\mathrm L^\intercal`, row and
            column-indexed by ``x0`` and ``x1``.
        """
        L = self.L
        Lo = zeros_like(L)
        grad = {"Llow": [], "Llogd": []}
        for ii in range(len(self._tril[0])):
            row = self._tril[0][ii]
            col = self._tril[1][ii]
            if row == col:
                Lo[row, col] = L[row, col]
                grad["Llogd"].append(dot(Lo, L.T) + dot(L, Lo.T))
            else:
                Lo[row, col] = 1
                grad["Llow"].append(dot(Lo, L.T) + dot(L, Lo.T))
            Lo[row, col] = 0

        grad["Llow"] = [g[x0, ...][..., x1] for g in grad["Llow"]]
        grad["Llogd"] = [g[x0, ...][..., x1] for g in grad["Llogd"]]
        return dict(Llow=stack(grad["Llow"], axis=-1),
                    Llogd=stack(grad["Llogd"], axis=-1))

    def __str__(self):
        tname = type(self).__name__
        msg = "{}()".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        msg += "  Lu: {}".format(self.Lu)
        return msg
