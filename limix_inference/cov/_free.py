from __future__ import division

from numpy import dot
from numpy import ones
from numpy import empty
from numpy import empty_like
from numyp import tril_indices_from

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
        self._L = empty((size, size))
        self._tril = tril_indices_from(self._L)
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
        L = self.L()[x0,:][:,x1]
        return dot(L, L.T)

    def derivative_Lu(self, x0, x1):
        Lu = self.get('Lu')
        d = empty_like(Lu)
        for ii in range(len(self._tril)):
            i0, j0 = self._tril[0][ii], self._tril[1][ii]
            if x0 == i0 and x1 == j0:
                d[ii] = 2 * Lu[ii]
            elif x0 == i0 and x1 != j0:
                d[ii] = Lu[ii]
            elif x0 != i0 and x1 == j0:
                d[ii] = Lu[ii]
            else:
                d[ii] = 0

        return d
