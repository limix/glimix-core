from __future__ import division

from numpy import ix_
from numpy import dot
from numpy import ones
from numpy import zeros
from numpy import zeros_like
from numpy import tril_indices_from
from numpy import atleast_1d
from numpy import ascontiguousarray

from optimix import Vector
from optimix import Function

class FreeFormCov(Function):
    """
    General semi-definite positive matrix with no contraints.
    A free-form covariance matrix of dimension d has 1/2 * d * (d + 1) params
    """
    def __init__(self, size):
        """
        Args:
            dim:        dimension of the free-form covariance
            jitter:     extent of diagonal offset which is added for numerical stability
                        (default value: 1e-4)
        """
        tsize = ((size + 1) * size) / 2;
        tsize = int(tsize)
        self._L = zeros((size, size))
        self._tril = tril_indices_from(self._L)
        self._L[self._tril] = 1
        Function.__init__(self, Lu=Vector(ones(tsize)))

    @property
    def L(self):
        self._L[self._tril] = self.get('Lu')
        return self._L

    @L.setter
    def L(self, value):
        self._L[:] = value
        self.set('Lu', self._L[self._tril])

    def value(self, x0, x1):
        x0 = ascontiguousarray(atleast_1d(x0), int)
        x1 = ascontiguousarray(atleast_1d(x1), int)

        L = self.L
        K = dot(L, L.T)

        x0 = x0.ravel()
        x1 = x1.ravel()

        return K[ix_(x0, x1)]

    def derivative_Lu(self, x0, x1):
        Lu = self.get('Lu')
        d = zeros_like(Lu)
        for ii in range(len(self._tril[0])):
            i0, j0 = self._tril[0][ii], self._tril[1][ii]
            if x0 == i0 and x1 == j0:
                d[ii] = 2 * Lu[ii]

        return d
