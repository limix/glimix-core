from functools import lru_cache

from numpy import (
    asarray,
    atleast_2d,
    concatenate,
    diag,
    eye,
    kron,
    log,
    sqrt,
    tensordot,
    zeros_like,
)
from numpy.linalg import eigh, svd

from numpy_sugar.linalg import ddot, dotd
from optimix import Function

from .._util import format_function, unvec
from .free import FreeFormCov
from .lrfree import LRFreeFormCov


class Kron2SumCov(Function):
    """
    Implements K =  C₀ ⊗ GGᵗ + C₁ ⊗ I.

    C₀ and C₁ are d×d symmetric matrices. C₀ is a semi-definite positive matrix while C₁
    is a positive definite one. G is a n×m matrix and I is a n×n identity matrix. Let
    M = Uₘ Sₘ Uₘᵗ be the eigen decomposition for any matrix M. The documentation and
    implementation of this class make use of the following definitions:

    - X = GGᵗ = Uₓ Sₓ Uₓᵗ
    - C₁ = U₁ S₁ U₁ᵗ
    - Cₕ = S₁⁻½ U₁ᵗ C₀ U₁ S₁⁻½
    - Cₕ = Uₕ Sₕ Uₕᵗ
    - D = (Sₕ ⊗ Sₓ + Iₕₓ)⁻¹
    - Lₓ = Uₓᵗ
    - Lₕ = Uₕᵗ S₁⁻½ U₁ᵗ
    - L = Lₕ ⊗ Lₓ

    The above definitions allows us to write the inverse of the covariance matrix as:

        K⁻¹ = LᵗDL,

    where D is a diagonal matrix.

    Example
    -------

    .. doctest::

        >>> from numpy import array
        >>> from glimix_core.cov import Kron2SumCov
        >>>
        >>> G = array([[-1.5, 1.0], [-1.5, 1.0], [-1.5, 1.0]])
        >>> Lr = array([[3], [2]], float)
        >>> Ln = array([[1, 0], [2, 1]], float)
        >>>
        >>> cov = Kron2SumCov(G, 2, 1)
        >>> cov.C0.L = Lr
        >>> cov.C1.L = Ln
        >>> print(cov)
        Kron2SumCov(G=..., dim=2, rank=1): Kron2SumCov
          LRFreeFormCov(n=2, m=1): C₀
            L: [[3.]
                [2.]]
          FreeFormCov(dim=2): C₁
            L: [[1. 0.]
                [2. 1.]]
    """

    def __init__(self, G, dim, rank):
        """
        Constructor.

        Parameters
        ----------
        dim : int
            Dimension d for the square matrices C₀ and C₁.
        rank : int
            Maximum rank of the C₁ matrix.
        """
        self._C0 = LRFreeFormCov(dim, rank)
        self._C0.name = "C₀"
        self._C1 = FreeFormCov(dim)
        self._C1.name = "C₁"
        self._G = atleast_2d(asarray(G, float))
        U, S, _ = svd(G)
        S = concatenate((S, [0.0] * (U.shape[0] - S.shape[0])))
        self._USx = U, S * S
        self._I = eye(self.G.shape[0])
        Function.__init__(
            self, "Kron2SumCov", composite=[("C0", self._C0), ("C1", self._C1)]
        )

    @property
    def _eUSx(self):
        from numpy_sugar.linalg import economic_svd

        US = economic_svd(self._G)
        return US[0], US[1] * US[1]

    @property
    @lru_cache(maxsize=None)
    def _LxG(self):
        Qx = self._USx[0]
        Lx = Qx.T
        return Lx @ self._G

    @property
    @lru_cache(maxsize=None)
    def _T0(self):
        return dotd(self._LxG, self._LxG.T)

    @property
    @lru_cache(maxsize=None)
    def _T1(self):
        Qx = self._USx[0]
        Lx = Qx.T
        return dotd(Lx, Lx.T)

    @property
    def G(self):
        """
        User-provided matrix G, n×m.
        """
        return self._G

    @property
    def C0(self):
        """
        Semi-definite positive matrix C₀.
        """
        return self._C0

    @property
    def C1(self):
        """
        Definite positive matrix C₁.
        """
        return self._C1

    @property
    def L(self):
        """
        L = Lₕ ⊗ Lₓ
        """
        Sn, Un = self.C1.eigh()
        C0 = self.C0.value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Ch = UnSn.T @ C0 @ UnSn
        Uh = eigh(Ch)[1]
        Ux, Sx = self._USx
        Lh = (UnSn @ Uh).T
        Lx = Ux.T
        return kron(Lh, Lx)

    def value(self):
        """
        Covariance matrix K = C₀ ⊗ GGᵗ + C₁ ⊗ I.

        Returns
        -------
        K : ndarray
            C₀ ⊗ GGᵗ + C₁ ⊗ I.
        """
        X = self.G @ self.G.T
        C0 = self._C0.value()
        C1 = self._C1.value()
        return kron(C0, X) + kron(C1, self._I)

    def gradient(self):
        """
        Gradient of K.

        Returns
        -------
        C0 : ndarray
            Derivative of C₀ over its parameters.
        C1 : ndarray
            Derivative of C₁ over its parameters.
        """
        I = self._I
        X = self.G @ self.G.T

        C0 = self._C0.gradient()["Lu"].transpose([2, 0, 1])
        C1_grad = self._C1.gradient()

        C1 = C1_grad["Lu"].transpose([2, 0, 1])

        grad = {
            "C0.Lu": kron(C0, X).transpose([1, 2, 0]),
            "C1.Lu": kron(C1, I).transpose([1, 2, 0]),
        }
        return grad

    def gradient_dot(self, v, var):
        G = self.G
        V = unvec(v, (self.G.shape[0], -1) + v.shape[1:])

        if var == "C0.Lu":
            C = self._C0.gradient()["Lu"]
            R = tensordot(V.T @ G @ G.T, C, axes=([-2], [0])).reshape(
                V.shape[2:] + (-1,) + (C.shape[-1],), order="F"
            )
            return R
        elif var == "C1.Lu":
            C = self._C1.gradient()["Lu"]
            R = tensordot(V.T, C, axes=([-2], [0])).reshape(
                V.shape[2:] + (-1,) + (C.shape[-1],), order="F"
            )
            return R

    def solve(self, v):
        """
        Implements the product K⁻¹v.

        Parameters
        ----------
        v : array_like
            Array to be multiplied.

        Returns
        -------
        x : ndarray
            Solution x to the equation K x = y.
        """
        Sn, Un = self.C1.eigh()
        C0 = self.C0.value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Ch = UnSn.T @ C0 @ UnSn
        Sh, Uh = eigh(Ch)
        Ux, Sx = self._USx
        D = 1 / (kron(Sh, Sx) + 1)
        Ln = (UnSn @ Uh).T
        Lx = Ux.T
        L = kron(Ln, Lx)
        return L.T @ ddot(D, L @ v, left=True)

    def logdet(self):
        """
        Implements log|K| = - log|D| + n⋅log|C₁|.

        Returns
        -------
        logdet : float
            Log-determinant of K.
        """
        Sn, Un = self.C1.eigh()
        C0 = self.C0.value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Ch = UnSn.T @ C0 @ UnSn
        Sh, Urs = eigh(Ch)
        Sx = self._USx[1]
        D = 1 / (kron(Sh, Sx) + 1)
        N = self.G.shape[0]
        logdetC = self.C1.logdet()
        return -log(D).sum() + N * logdetC

    def logdet_gradient(self):
        """
        Implements ∂log|K| = Tr[K⁻¹ ∂K].

        It can be shown that::

            ∂log|K| = diag(D)ᵗ diag(L ∂K Lᵗ).

        Let L = Lₕ⊗Lₓ and C₀ = E₀E₀ᵀ. Note that::

            L∂KLᵗ = 2 ((Lₕ∂E₀) ⊗ (LₓG)) ((LₕE₀)ᵀ ⊗ (LₓG)ᵀ),

        when the derivative is over the elements of E₀. Similarly,

            L∂KLᵗ = 2 ((Lₕ∂E₁) ⊗ Lₓ) ((LₕE₁)ᵀ ⊗ Lₓᵀ),

        when the derivative is over the elements of E₁, C₁ = E₁E₁ᵀ.
        From the property

            diag(A ⊗ B) = diag(A) ⊗ diag(B),

        we can rewrite the previous expressions in order to achieve computational speed
        up.

        Returns
        -------
        C0 : ndarray
            Derivative of C₀ over its parameters.
        C1 : ndarray
            Derivative of C₁ over its parameters.
        """
        Sn, Un = self.C1.eigh()
        C0 = self.C0.value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Ch = UnSn.T @ C0 @ UnSn
        Sh, Uh = eigh(Ch)
        Qx, Sx = self._USx
        D = 1 / (kron(Sh, Sx) + 1)
        Lh = (UnSn @ Uh).T

        dE = zeros_like(self._C0.L)
        E = self._C0.L
        ELhT = E.T @ Lh.T
        grad_C0 = zeros_like(self._C0.Lu)
        for i in range(self._C0.Lu.shape[0]):
            row = i // E.shape[1]
            col = i % E.shape[1]
            dE[row, col] = 1
            UU = kron(dotd(Lh @ dE, ELhT), self._T0)
            grad_C0[i] = (2 * UU * D).sum()
            dE[row, col] = 0

        dE = zeros_like(self._C1.L)
        E = self._C1.L
        LhET = (Lh @ E).T
        grad_C1 = zeros_like(self._C1.Lu)
        for i in range(len(self._C1._tril1[0])):
            row = self._C1._tril1[0][i]
            col = self._C1._tril1[1][i]
            dE[row, col] = 1
            UU = kron(dotd(Lh @ dE, LhET), self._T1)
            grad_C1[i] = (2 * UU * D).sum()
            dE[row, col] = 0

        m = len(self._C1._tril1[0])
        for i in range(len(self._C1._diag[0])):
            row = self._C1._diag[0][i]
            col = self._C1._diag[1][i]
            dE[row, col] = E[row, col]
            UU = kron(dotd(Lh @ dE, LhET), self._T1)
            grad_C1[m + i] = (2 * UU * D).sum()
            dE[row, col] = 0
        return {"C0.Lu": grad_C0, "C1.Lu": grad_C1}

    def __str__(self):
        dim = self._C0.L.shape[0]
        rank = self._C0.L.shape[1]
        msg0 = format_function(self, {"G": "...", "dim": dim, "rank": rank})
        msg1 = str(self._C0) + "\n" + str(self._C1)
        msg1 = "  " + "\n  ".join(msg1.split("\n"))
        return (msg0 + msg1).rstrip()
