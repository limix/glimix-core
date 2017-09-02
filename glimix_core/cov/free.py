# pylint: disable=E1101
from __future__ import division

from numpy import dot, ones, stack, tril_indices_from, zeros, zeros_like

from optimix import Function, Vector


class FreeFormCov(Function):
    """
    General semi-definite positive matrix with no contraints.
    A free-form covariance matrix of dimension d has 1/2 * d * (d + 1) params
    """

    def __init__(self, size):
        """
        Args:
            dim:        dimension of the free-form covariance
            jitter:     extent of diagonal offset which is added for numerical
                        stability (default value: 1e-4)
        """
        tsize = ((size + 1) * size) / 2
        tsize = int(tsize)
        self._L = zeros((size, size))
        self._tril = tril_indices_from(self._L)
        self._L[self._tril] = 1
        Function.__init__(self, Lu=Vector(ones(tsize)))

    @property
    def L(self):
        self._L[self._tril] = self.variables().get('Lu').value
        return self._L

    @L.setter
    def L(self, value):
        self._L[:] = value
        self.variables().get('Lu').value = self._L[self._tril]

    @property
    def Lu(self):
        return self.variables().get('Lu').value

    @Lu.setter
    def Lu(self, value):
        self.variables().get('Lu').value = value

    def value(self, x0, x1):
        return dot(self.L, self.L.T)[x0, ...][..., x1]

    def gradient(self, x0, x1):
        L = self.L
        Lo = zeros_like(L)
        grad = []
        for ii in range(len(self._tril[0])):
            row = self._tril[0][ii]
            col = self._tril[1][ii]
            Lo[row, col] = 1
            grad.append(dot(Lo, L.T) + dot(L, Lo.T))
            Lo[row, col] = 0

        grad = [g[x0, ...][..., x1] for g in grad]
        return dict(Lu=stack(grad, axis=-1))
