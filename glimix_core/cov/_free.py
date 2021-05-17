from typing import Any, Dict

from numpy import diag_indices_from, dot, exp, eye, inf, log, tril_indices_from, zeros
from optimix import Function, Vector

from .._util import format_function


class FreeFormCov(Function):
    """
    General definite positive matrix, K = LLᵀ + ϵI.

    A d×d covariance matrix K will have ((d+1)⋅d)/2 parameters defining the lower
    triangular elements of a Cholesky matrix L such that:

        K = LLᵀ + ϵI,

    for a very small positive number ϵ. That additional term is necessary to avoid
    singular and ill conditioned covariance matrices.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import FreeFormCov
        >>>
        >>> cov = FreeFormCov(2)
        >>> cov.L = [[1., .0], [0.5, 2.]]
        >>> print(cov.gradient()["Lu"])
        [[[0.  2.  0. ]
          [1.  0.5 0. ]]
        <BLANKLINE>
         [[1.  0.5 0. ]
          [1.  0.  8. ]]]
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
        from numpy_sugar import epsilon

        dim = int(dim)
        tsize = ((dim + 1) * dim) // 2
        self._L = zeros((dim, dim))
        self._tril1 = tril_indices_from(self._L, k=-1)
        self._diag = diag_indices_from(self._L)
        self._L[self._tril1] = 1
        self._L[self._diag] = 0
        self._epsilon = epsilon.small * 1000
        self._Lu = Vector(zeros(tsize))
        self._Lu.value[: tsize - dim] = 1
        n = self.L.shape[0]
        self._grad_Lu = zeros((n, n, self._Lu.shape[0]))
        Function.__init__(self, "FreeCov", Lu=self._Lu)
        bounds = [(-inf, +inf)] * (tsize - dim)
        bounds += [(log(epsilon.small * 1000), +11)] * dim
        self._Lu.bounds = bounds
        self._cache: Dict[str, Any] = {"eig": None}
        self.listen(self._parameters_update)
        self._nparams = tsize

    def _parameters_update(self):
        self._cache["eig"] = None

    @property
    def nparams(self):
        """
        Number of parameters.
        """
        return self._nparams

    def listen(self, func):
        """
        Listen to parameters change.

        Parameters
        ----------
        func : callable
            Function to be called when a parameter changes.
        """
        self._Lu.listen(func)

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

        if self._cache["eig"] is not None:
            return self._cache["eig"]

        U, S = svd(self.L)[:2]
        S *= S
        S += self._epsilon
        self._cache["eig"] = S, U

        return self._cache["eig"]

    @property
    def Lu(self):
        """
        Lower-triangular, flat part of L.
        """
        return self._Lu.value

    @Lu.setter
    def Lu(self, v):
        self._Lu.value = v

    @property
    def L(self):
        """
        Lower-triangular matrix L such that K = LLᵀ + ϵI.

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
        """
        Log of ｜K｜.

        Returns
        -------
        float
            Log-determinant of K.
        """
        from numpy.linalg import slogdet

        K = self.value()

        sign, logdet = slogdet(K)
        if sign != 1.0:
            msg = "The estimated determinant of K is not positive: "
            msg += f" ({sign}, {logdet})."
            raise RuntimeError(msg)

        return logdet

    def value(self):
        """
        Covariance matrix.

        Returns
        -------
        K : ndarray
            Matrix K = LLᵀ + ϵI, for a very small positive number ϵ.
        """
        K = dot(self.L, self.L.T)
        return K + self._epsilon * eye(K.shape[0])

    def gradient(self):
        """
        Derivative of the covariance matrix over the parameters of L.

        Returns
        -------
        Lu : ndarray
            Derivative of K over the lower triangular part of L.
        """
        L = self.L
        self._grad_Lu[:] = 0

        for i in range(len(self._tril1[0])):
            row = self._tril1[0][i]
            col = self._tril1[1][i]
            self._grad_Lu[row, :, i] = L[:, col]
            self._grad_Lu[:, row, i] += L[:, col]

        m = len(self._tril1[0])
        for i in range(len(self._diag[0])):
            row = self._diag[0][i]
            col = self._diag[1][i]
            self._grad_Lu[row, :, m + i] = L[row, col] * L[:, col]
            self._grad_Lu[:, row, m + i] += L[row, col] * L[:, col]

        return {"Lu": self._grad_Lu}

    def __str__(self):
        return format_function(self, {"dim": self._L.shape[0]}, [("L", self.L)])
