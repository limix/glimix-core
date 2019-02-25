from numpy import (
    diag_indices_from,
    dot,
    exp,
    eye,
    log,
    ones,
    stack,
    tril_indices_from,
    zeros,
    zeros_like,
)
from numpy_sugar import epsilon
from optimix import Function, Vector

from ..util.classes import NamedClass
from ..util import format_function, format_named_arr


class FreeFormCov(NamedClass, Function):
    r"""
    General definite positive matrix.

    A d×d covariance matrix K will have ((d+1)⋅d)/2 parameters defining the lower
    triangular elements of a Cholesky matrix L such that:

        K = LLᵗ + ϵI,

    for a very small positive number ϵ. That additional term is necessary to avoid
    singular and ill conditioned covariance matrices.

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
        Function.__init__(
            self, Llow=Vector(ones(tsize - dim)), Llogd=Vector(zeros(dim))
        )
        self.variables().get("Llogd").bounds = [(log(epsilon.small * 1000), +15)] * dim
        NamedClass.__init__(self)

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
    def L(self):
        """
        Lower-triangular matrix L such that K = LLᵗ + ϵI.

        Returns
        -------
        L : (d, d) ndarray
            Lower-triangular matrix.
        """
        self._L[self._tril1] = self.variables().get("Llow").value
        self._L[self._diag] = exp(self.variables().get("Llogd").value)
        return self._L

    @L.setter
    def L(self, value):
        self._L[:] = value
        self.variables().get("Llow").value = self._L[self._tril1]
        self.variables().get("Llogd").value = log(self._L[self._diag])

    def logdet(self):
        r"""
        Log of \|K\|.

        Returns
        -------
        float
            Log-determinant of K.
        """
        from numpy.linalg import slogdet

        K = self.feed().value()

        sign, logdet = slogdet(K)
        if sign != 1.0:

            raise RuntimeError("The estimated determinant of K is not positive: "
                               + f" ({sign}, {logdet}).")
        return logdet

    def value(self, x0, x1):
        """
        Covariance function evaluated at (x₀,x₁).

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        ndarray
            Submatrix of K, row and column-indexed by ``x0`` and ``x1``.
        """
        K = dot(self.L, self.L.T)[x0, ...][..., x1]
        if K.ndim == 2 and K.shape[0] == K.shape[1]:
            # TODO: fix this hacky
            return K + self._epsilon * eye(K.shape[0])
        return K

    def gradient(self, x0, x1):
        r"""
        Derivative of the covariance function evaluated at ``(x0, x1)``.

        Derivative over the lower-triangular part of :math:`\mathrm L`.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        dict
            Dictionary having the `Lu` key for the lower-triangular part
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
        return dict(
            Llow=stack(grad["Llow"], axis=-1), Llogd=stack(grad["Llogd"], axis=-1)
        )

    def __str__(self):
        msg = format_function(self, dim=self._L.shape[0]) + "\n"
        msg += format_named_arr("L", self.L)
        return msg
