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
from optimix import Function, Vector

from .._util import format_function


class FreeFormCov(Function):
    """
    General definite positive matrix, K = LLᵗ + ϵI.

    A d×d covariance matrix K will have ((d+1)⋅d)/2 parameters defining the lower
    triangular elements of a Cholesky matrix L such that:

        K = LLᵗ + ϵI,

    for a very small positive number ϵ. That additional term is necessary to avoid
    singular and ill conditioned covariance matrices.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import FreeFormCov
        >>>
        >>> cov = FreeFormCov(2)
        >>> cov.L = [[1., .0], [0.5, 2.]]
        >>> print(cov.gradient()["L0"])
        [[[0.]
          [1.]]
        <BLANKLINE>
         [[1.]
          [1.]]]
        >>> print(cov.gradient()["L1"])
        [[[2.  0. ]
          [0.5 0. ]]
        <BLANKLINE>
         [[0.5 0. ]
          [0.  8. ]]]
        >>> cov.name = "K"
        >>> print(cov)
        FreeFormCov(dim=2): K
          L: [[1.  0. ]
              [0.5 2. ]]
    """

    def __init__(self, dim):
        """
        Constructor.

        Parameters
        ----------
        dim : int
            Dimension d of the free-form covariance matrix.
        """
        dim = int(dim)
        tsize = ((dim + 1) * dim) // 2
        self._L = zeros((dim, dim))
        self._tril = tril_indices_from(self._L)
        self._tril1 = tril_indices_from(self._L, k=-1)
        self._diag = diag_indices_from(self._L)
        self._L[self._tril1] = 1
        self._L[self._diag] = 0
        self._epsilon = epsilon.small * 1000
        self._L0 = Vector(ones(tsize - dim))
        self._L1 = Vector(zeros(dim))
        Function.__init__(self, "FreeCov", L0=self._L0, L1=self._L1)
        self._L1.bounds = [(log(epsilon.small * 1000), +15)] * dim

    def eigh(self):
        """
        Eigen decomposition of K.

        Returns
        -------
        S : ndarray
            The eigenvalues in ascending order, each repeated according to its
            multiplicity.
        U : ndarray
            Normalized eigenvectors.
        """
        from numpy.linalg import svd

        U, S = svd(self.L)[:2]
        S *= S
        S += self._epsilon
        return S, U

    @property
    def L0(self):
        """
        Strictly lower-triangular, flat part of L.
        """
        return self._L0.value

    @L0.setter
    def L0(self, v):
        self._L0.value = v

    @property
    def L1(self):
        """
        Diagonal of L in log-space.
        """
        return self._L1.value

    @L1.setter
    def L1(self, v):
        self._L1.value = v

    @property
    def L(self):
        """
        Lower-triangular matrix L such that K = LLᵗ + ϵI.

        Returns
        -------
        L : (d, d) ndarray
            Lower-triangular matrix.
        """
        self._L[self._tril1] = self._L0.value
        self._L[self._diag] = exp(self._L1.value)
        return self._L

    @L.setter
    def L(self, value):
        self._L[:] = value
        self._L0.value = self._L[self._tril1]
        self._L1.value = log(self._L[self._diag])

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
        K : ndarray
            Matrix K = LLᵗ + ϵI, for a very small positive number ϵ.
        """
        K = dot(self.L, self.L.T)
        return K + self._epsilon * eye(K.shape[0])

    def gradient(self):
        r"""
        Derivative of the covariance matrix over L₀ and L₁.

        Returns
        -------
        L0 : ndarray
            Derivative of K over L0.
        L1 : ndarray
            Derivative of K over L1.
        """
        L = self.L
        Lo = zeros_like(L)
        n = self.L.shape[0]
        grad = {
            "L0": zeros((n, n, self._L0.shape[0])),
            "L1": zeros((n, n, self._L1.shape[0])),
        }
        i = 0
        j = 0
        for ii in range(len(self._tril[0])):
            row = self._tril[0][ii]
            col = self._tril[1][ii]
            if row == col:
                Lo[row, col] = L[row, col]
                grad["L1"][..., i] = dot(Lo, L.T) + dot(L, Lo.T)
                i += 1
            else:
                Lo[row, col] = 1
                grad["L0"][..., j] = dot(Lo, L.T) + dot(L, Lo.T)
                j += 1
            Lo[row, col] = 0

        return grad

    def __str__(self):
        return format_function(self, {"dim": self._L.shape[0]}, [("L", self.L)])
