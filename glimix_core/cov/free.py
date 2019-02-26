from numpy import (
    diag_indices_from,
    dot,
    exp,
    eye,
    log,
    ones,
    tril_indices_from,
    zeros,
    zeros_like,
)
from numpy_sugar import epsilon
from optimix import Func, Vector

from ..util import format_function, format_named_arr


class FreeFormCov(Func):
    """
    General definite positive matrix.

    A d×d covariance matrix K will have ((d+1)⋅d)/2 parameters defining the lower
    triangular elements of a Cholesky matrix L such that:

        K = LLᵗ + ϵI,

    for a very small positive number ϵ. That additional term is necessary to avoid
    singular and ill conditioned covariance matrices.

    Parameters
    ----------
    dim : int
        Dimension d of the free-form covariance matrix.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import FreeFormCov
        >>>
        >>> cov = FreeFormCov(2)
        >>> print(cov.value([0, 1], [0, 1]))
        [[1.0000149 1.       ]
            [1.        2.0000149]]
        >>> print(cov.L)
        [[1. 0.]
         [1. 1.]]
        >>> g = cov.gradient([0, 1], [0, 1])
        >>> print(cov.value([0, 1], [0, 1]))
        [[1.0000149 1.       ]
         [1.        2.0000149]]
        >>> print(cov)
        FreeFormCov(dim=2)
        L: [[1. 0.]
            [1. 1.]]
        >>> cov.name = "covname"
        >>> print(cov)
        FreeFormCov(dim=2): covname
        L: [[1. 0.]
            [1. 1.]]
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
        self._epsilon = epsilon.small * 1000
        self._Llow = Vector(ones(tsize - dim))
        self._Llogd = Vector(zeros(dim))
        Func.__init__(self, "FreeCov", Llow=self._Llow, Llogd=self._Llogd)
        self._Llogd.bounds = [(log(epsilon.small * 1000), +15)] * dim

    def eigh(self):
        """
        Eigen decomposition of K.

        Returns
        -------
        S : ndarray
            The eigenvalues in ascending order, each repeated according to its multiplicity.
        U : ndarray
            Normalized eigenvectors.
        """
        from numpy.linalg import svd

        U, S = svd(self.L)[:2]
        S *= S
        S += self._epsilon
        return S, U

    @property
    def Llow(self):
        """
        Strictly lower-triangular, flat part of L.
        """
        return self._Llow.value

    @Llow.setter
    def Llow(self, v):
        self._Llow.value = v

    @property
    def Llogd(self):
        """
        Diagonal of L in log-space.
        """
        return self._Llogd.value

    @Llogd.setter
    def Llogd(self, v):
        self._Llogd.value = v

    @property
    def L(self):
        """
        Lower-triangular matrix L such that K = LLᵗ + ϵI.

        Returns
        -------
        L : (d, d) ndarray
            Lower-triangular matrix.
        """
        self._L[self._tril1] = self._Llow.value
        self._L[self._diag] = exp(self._Llogd.value)
        return self._L

    @L.setter
    def L(self, value):
        self._L[:] = value
        self._Llow.value = self._L[self._tril1]
        self._Llogd.value = log(self._L[self._diag])

    def logdet(self):
        r"""
        Log of \|K\|.

        Returns
        -------
        float
            Log-determinant of K.
        """
        from numpy.linalg import slogdet

        K = self.value()

        sign, logdet = slogdet(K)
        if sign != 1.0:

            raise RuntimeError(
                "The estimated determinant of K is not positive: "
                + f" ({sign}, {logdet})."
            )
        return logdet

    def value(self):
        """
        Covariance matrix.

        Returns
        -------
        ndarray
            Matrix K = LLᵗ + ϵI.
        """
        K = dot(self.L, self.L.T)
        return K + self._epsilon * eye(K.shape[0])

    def gradient(self):
        r"""
        Derivative of the covariance matrix over Llow and Llogd.

        Returns
        -------
        Llow : ndarray
            Derivative of K over Llow.
        Llogd : ndarray
            Derivative of K over Llogd.
        """
        L = self.L
        Lo = zeros_like(L)
        n = self.L.shape[0]
        grad = {
            "Llow": zeros((n, n, self._Llow.shape[0])),
            "Llogd": zeros((n, n, self._Llogd.shape[0])),
        }
        i = 0
        j = 0
        for ii in range(len(self._tril[0])):
            row = self._tril[0][ii]
            col = self._tril[1][ii]
            if row == col:
                Lo[row, col] = L[row, col]
                grad["Llogd"][..., i] = dot(Lo, L.T) + dot(L, Lo.T)
                i += 1
            else:
                Lo[row, col] = 1
                grad["Llow"][..., j] = dot(Lo, L.T) + dot(L, Lo.T)
                j += 1
            Lo[row, col] = 0

        return grad

    def __str__(self):
        msg = format_function(self, dim=self._L.shape[0]) + "\n"
        msg += format_named_arr("L", self.L)
        return msg
