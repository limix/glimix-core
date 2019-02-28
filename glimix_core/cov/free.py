from numpy import (
    diag_indices_from,
    dot,
    exp,
    eye,
    inf,
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
        self._Lu = Vector(zeros(tsize))
        Function.__init__(self, "FreeCov", Lu=self._Lu)
        bounds = [-inf, +inf] * (tsize - dim) + [(log(epsilon.small * 1000), +15)] * dim
        self._Lu.bounds = bounds

    @property
    def shape(self):
        """
        Array shape.
        """
        n = self._L.shape[0]
        return (n, n)

    def fix(self):
        """
        Disable parameter optimisation.
        """
        self._Lu.fix()

    def unfix(self):
        """
        Enable parameter optimisation.
        """
        self._Lu.unfix()

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
    def L(self):
        """
        Lower-triangular matrix L such that K = LLᵗ + ϵI.

        Returns
        -------
        L : (d, d) ndarray
            Lower-triangular matrix.
        """
        m = len(self._tril1[0])
        self._L[self._tril1] = self._Lu.value[:m]
        self._L[self._diag] = exp(self._Lu.value[m:])
        return self._L

    @L.setter
    def L(self, value):
        self._L[:] = value
        m = len(self._tril1[0])
        self._Lu.value[:m] = self._L[self._tril1]
        self._Lu.value[m:] = log(self._L[self._diag])

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
        m = len(self._tril1[0])
        grad = {"Lu": zeros((n, n, self._Lu.shape[0]))}
        i = 0
        j = 0
        for ii in range(len(self._tril[0])):
            row = self._tril[0][ii]
            col = self._tril[1][ii]
            if row == col:
                Lo[row, col] = L[row, col]
                grad["Lu"][..., m + i] = dot(Lo, L.T) + dot(L, Lo.T)
                i += 1
            else:
                Lo[row, col] = 1
                grad["Lu"][..., j] = dot(Lo, L.T) + dot(L, Lo.T)
                j += 1
            Lo[row, col] = 0

        return grad

    def __str__(self):
        return format_function(self, {"dim": self._L.shape[0]}, [("L", self.L)])
