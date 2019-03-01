from numpy import (
    asarray,
    atleast_2d,
    concatenate,
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
    is positive definite one. G is a n×m matrix and I is a n×n identity matrix. Let
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
        >>> cov.Cr.L = Lr
        >>> cov.Cn.L = Ln
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
        self._Cr = LRFreeFormCov(dim, rank)
        self._Cr.name = "C₀"
        self._Cn = FreeFormCov(dim)
        self._Cn.name = "C₁"
        self._G = atleast_2d(asarray(G, float))
        U, S, _ = svd(G)
        S = concatenate((S, [0.0] * (U.shape[0] - S.shape[0])))
        self._USx = U, S * S
        self._I = eye(self.G.shape[0])
        Function.__init__(
            self, "Kron2SumCov", composite=[("Cr", self._Cr), ("Cn", self._Cn)]
        )

    @property
    def _LgG(self):
        Qx = self._USx[0]
        Lg = Qx.T
        return Lg @ self._G

    @property
    def G(self):
        """
        User-provided matrix G, n×m.
        """
        return self._G

    @property
    def Cr(self):
        """
        Semi-definite positive matrix C₀.
        """
        return self._Cr

    @property
    def Cn(self):
        """
        Definite positive matrix C₁.
        """
        return self._Cn

    @property
    def L(self):
        """
        L = Lₕ ⊗ Lₓ
        """
        Sn, Un = self.Cn.eigh()
        Cr = self.Cr.value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Ch = UnSn.T @ Cr @ UnSn
        Uh = eigh(Ch)[1]
        Ux, Sx = self._USx
        Lc = (UnSn @ Uh).T
        Lx = Ux.T
        return kron(Lc, Lx)

    def value(self):
        """
        Covariance matrix K = C₀ ⊗ GGᵗ + C₁ ⊗ I.

        Returns
        -------
        K : ndarray
            C₀ ⊗ GGᵗ + C₁ ⊗ I.
        """
        X = self.G @ self.G.T
        Cr = self._Cr.value()
        Cn = self._Cn.value()
        return kron(Cr, X) + kron(Cn, self._I)

    def gradient(self):
        """
        Gradient of K.

        Returns
        -------
        Cr_Lu : ndarray
            Derivative of C₀ over the array Lu.
        Cn_Lu : ndarray
            Derivative of C₁ over the array Lu.
        """
        I = self._I
        X = self.G @ self.G.T

        Cr_Lu = self._Cr.gradient()["Lu"].transpose([2, 0, 1])
        Cn_grad = self._Cn.gradient()

        Cn_Lu = Cn_grad["Lu"].transpose([2, 0, 1])

        grad = {
            "Cr.Lu": kron(Cr_Lu, X).transpose([1, 2, 0]),
            "Cn.Lu": kron(Cn_Lu, I).transpose([1, 2, 0]),
        }
        return grad

    def gradient_dot(self, v, var):
        G = self.G
        V = unvec(v, (self.G.shape[0], -1) + v.shape[1:])

        if var == "Cr.Lu":
            C = self._Cr.gradient()["Lu"]
            R = tensordot(V.T @ G @ G.T, C, axes=([-2], [0])).reshape(
                V.shape[2:] + (-1,) + (C.shape[-1],), order="F"
            )
            return R
        elif var == "Cn.Lu":
            C = self._Cn.gradient()["Lu"]
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
        Sn, Un = self.Cn.eigh()
        Cr = self.Cr.value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Ch = UnSn.T @ Cr @ UnSn
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
        Sn, Un = self.Cn.eigh()
        Cr = self.Cr.value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Ch = UnSn.T @ Cr @ UnSn
        Sh, Urs = eigh(Ch)
        Sx = self._USx[1]
        D = 1 / (kron(Sh, Sx) + 1)
        N = self.G.shape[0]
        logdetC = self.Cn.logdet()
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

        Returns
        -------
        Cr_Lu : ndarray
            Derivative of C₀ over the array Lu.
        Cn_Lu : ndarray
            Derivative of C₁ over the array Lu.
        """
        Sn, Un = self.Cn.eigh()
        Cr = self.Cr.value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Ch = UnSn.T @ Cr @ UnSn
        Sh, Uh = eigh(Ch)
        Qx, Sx = self._USx
        D = 1 / (kron(Sh, Sx) + 1)
        Lc = (UnSn @ Uh).T
        Lg = Qx.T

        dE = zeros_like(self._Cr.L)
        E = self._Cr.L
        grad_Cr = zeros_like(self._Cr.Lu)
        for i in range(self._Cr.Lu.shape[0]):
            row = i // E.shape[1]
            col = i % E.shape[1]
            dE[row, col] = 1
            UU = dotd(kron(Lc @ dE, self._LgG), kron(E.T @ Lc.T, self._LgG.T))
            grad_Cr[i] = (2 * UU * D).sum()
            dE[row, col] = 0

        dE = zeros_like(self._Cn.L)
        E = self._Cn.L
        grad_Cn = zeros_like(self._Cn.Lu)
        for i in range(len(self._Cn._tril1[0])):
            row = self._Cn._tril1[0][i]
            col = self._Cn._tril1[1][i]
            dE[row, col] = 1
            UU = dotd(kron(Lc @ dE, Lg), kron(E.T @ Lc.T, Lg.T))
            grad_Cn[i] = (2 * UU * D).sum()
            dE[row, col] = 0

        m = len(self._Cn._tril1[0])
        for i in range(len(self._Cn._diag[0])):
            row = self._Cn._diag[0][i]
            col = self._Cn._diag[1][i]
            dE[row, col] = E[row, col]
            UU = dotd(kron(Lc @ dE, Lg), kron(E.T @ Lc.T, Lg.T))
            grad_Cn[m + i] = (2 * UU * D).sum()
            dE[row, col] = 0
        return {"Cr.Lu": grad_Cr, "Cn.Lu": grad_Cn}

    def __str__(self):
        dim = self._Cr.L.shape[0]
        rank = self._Cr.L.shape[1]
        msg0 = format_function(self, {"G": "...", "dim": dim, "rank": rank})
        msg1 = str(self._Cr) + "\n" + str(self._Cn)
        msg1 = "  " + "\n  ".join(msg1.split("\n"))
        return (msg0 + msg1).rstrip()
