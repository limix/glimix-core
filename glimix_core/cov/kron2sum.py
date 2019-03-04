from functools import lru_cache

from numpy import (
    asarray,
    atleast_2d,
    concatenate,
    eye,
    kron,
    log,
    newaxis,
    sqrt,
    tensordot,
    zeros_like,
)
from numpy.linalg import eigh, svd

from numpy_sugar.linalg import ddot, dotd
from optimix import Function

from .._util import format_function, unvec, vec
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

    The above definitions allows us to write the inverse of the covariance matrix as::

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
        self._Lx = U.T
        LxG = self._Lx @ G
        self._LXL = LxG @ LxG.T
        self._LL = self._Lx @ self._Lx.T
        self._I = eye(self.G.shape[0])
        Function.__init__(
            self, "Kron2SumCov", composite=[("C0", self._C0), ("C1", self._C1)]
        )

    @property
    def Lx(self):
        return self._Lx

    @property
    def LL(self):
        return self._LL

    @property
    def LXL(self):
        return self._LXL

    @property
    def _eUSx(self):
        from numpy_sugar.linalg import economic_svd

        US = economic_svd(self._G)
        return US[0], US[1] * US[1]

    @property
    @lru_cache(maxsize=None)
    def _LxG(self):
        return self._Lx @ self._G

    @property
    @lru_cache(maxsize=None)
    def _T0(self):
        return dotd(self._LxG, self._LxG.T)

    @property
    @lru_cache(maxsize=None)
    def _T1(self):
        return dotd(self._Lx, self._Lx.T)

    @property
    def _LD(self):
        """
        Implements Lₕ, Lₓ, and D.

        Returns
        -------
        Lh : ndarray
            Uₕᵗ S₁⁻½ U₁ᵗ.
        Lx : ndarray
            Uₓᵗ.
        D : ndarray
            (Sₕ ⊗ Sₓ + Iₕₓ)⁻¹.
        """
        S1, U1 = self.C1.eigh()
        U1S1 = ddot(U1, 1 / sqrt(S1))
        Sh, Uh = eigh(U1S1.T @ self.C0.value() @ U1S1)
        Sx = self._USx[1]
        return {"Lh": (U1S1 @ Uh).T, "Lx": self._Lx, "D": 1 / (kron(Sh, Sx) + 1)}

    @property
    def LhD(self):
        """
        Implements Lₕ and D.

        Returns
        -------
        Lh : ndarray
            Uₕᵗ S₁⁻½ U₁ᵗ.
        D : ndarray
            (Sₕ ⊗ Sₓ + Iₕₓ)⁻¹.
        """
        S1, U1 = self.C1.eigh()
        U1S1 = ddot(U1, 1 / sqrt(S1))
        Sh, Uh = eigh(U1S1.T @ self.C0.value() @ U1S1)
        Sx = self._USx[1]
        return {"Lh": (U1S1 @ Uh).T, "D": 1 / (kron(Sh, Sx) + 1)}

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

        C0 = self._C0.gradient()["Lu"].T
        C1 = self._C1.gradient()["Lu"].T

        grad = {"C0.Lu": kron(C0, X).T, "C1.Lu": kron(C1, I).T}
        return grad

    def gradient_dot(self, v):
        """
        Implements ∂K⋅v.

        Parameters
        ----------
        v : array_like
            Vector from ∂K⋅v.

        Returns
        -------
        C0.Lu : ndarray
            ∂K⋅v, where the gradient is taken over the C₀ parameters.
        C1.Lu : ndarray
            ∂K⋅v, where the gradient is taken over the C₁ parameters.
        """
        V = unvec(v, (self.G.shape[0], -1) + v.shape[1:])
        r = {}

        C = self._C0.gradient()["Lu"]
        r["C0.Lu"] = tensordot(V.T @ self.G @ self.G.T, C, axes=([-2], [0]))
        r["C0.Lu"] = r["C0.Lu"].reshape(V.shape[2:] + (-1,) + (C.shape[-1],), order="F")

        C = self._C1.gradient()["Lu"]
        r["C1.Lu"] = tensordot(V.T, C, axes=([-2], [0]))
        r["C1.Lu"] = r["C1.Lu"].reshape(V.shape[2:] + (-1,) + (C.shape[-1],), order="F")

        return r

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
        L = kron(self._LD["Lh"], self._LD["Lx"])
        return L.T @ ddot(self._LD["D"], L @ v, left=True)

    def logdet(self):
        """
        Implements log|K| = - log|D| + n⋅log|C₁|.

        Returns
        -------
        logdet : float
            Log-determinant of K.
        """
        return -log(self._LD["D"]).sum() + self.G.shape[0] * self.C1.logdet()

    def logdet_gradient(self):
        """
        Implements ∂log|K| = Tr[K⁻¹ ∂K].

        It can be shown that::

            ∂log|K| = diag(D)ᵗ diag(L ∂K Lᵗ).

        Let C₀ = E₀E₀ᵀ. Note that::

            L∂KLᵗ = 2 ((Lₕ∂E₀) ⊗ (LₓG)) ((LₕE₀)ᵀ ⊗ (LₓG)ᵀ),

        when the derivative is over the elements of E₀. Similarly, ::

            L∂KLᵗ = 2 ((Lₕ∂E₁) ⊗ Lₓ) ((LₕE₁)ᵀ ⊗ Lₓᵀ),

        when the derivative is over the elements of E₁ for C₁ = E₁E₁ᵀ.
        From the property ::

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
        Lh = self._LD["Lh"]
        D = self._LD["D"]

        dE = zeros_like(self._C0.L)
        E = self._C0.L
        ELhT = E.T @ Lh.T
        grad_C0 = zeros_like(self._C0.Lu)
        for i in range(self._C0.Lu.shape[0]):
            row = i // E.shape[1]
            col = i % E.shape[1]
            dE[row, col] = 1
            grad_C0[i] = (2 * kron(dotd(Lh @ dE, ELhT), self._T0) * D).sum()
            dE[row, col] = 0

        dE = zeros_like(self._C1.L)
        E = self._C1.L
        LhET = (Lh @ E).T
        grad_C1 = zeros_like(self._C1.Lu)
        for i in range(len(self._C1._tril1[0])):
            row = self._C1._tril1[0][i]
            col = self._C1._tril1[1][i]
            dE[row, col] = 1
            grad_C1[i] = (2 * kron(dotd(Lh @ dE, LhET), self._T1) * D).sum()
            dE[row, col] = 0

        m = len(self._C1._tril1[0])
        for i in range(len(self._C1._diag[0])):
            row = self._C1._diag[0][i]
            col = self._C1._diag[1][i]
            dE[row, col] = E[row, col]
            grad_C1[m + i] = (2 * kron(dotd(Lh @ dE, LhET), self._T1) * D).sum()
            dE[row, col] = 0
        return {"C0.Lu": grad_C0, "C1.Lu": grad_C1}

    def LdKL_dot(self, v):
        """
        Let::

            L∂KLᵗ = 2 ((Lₕ∂E₀) ⊗ (LₓG)) ((LₕE₀)ᵀ ⊗ (LₓG)ᵀ)

        We have::

            L∂KLᵗvec(V) = 2 ((Lₕ∂E₀) ⊗ (LₓG)) ((LₓG)ᵀV(LₕE₀))
            = 2 ((LₓG) unvec((LₓG)ᵀV(LₕE₀)) (Lₕ∂E₀)ᵀ)

        Let::

            L∂KLᵗ = 2 ((Lₕ∂E₁) ⊗ Lₓ) ((LₕE₁)ᵀ ⊗ Lₓᵀ)

        We have::

            L∂KLᵗvec(V) = 2 ((Lₕ∂E₁) ⊗ Lₓ) (LₓᵀV(LₕE₁))
            = 2 (Lₓ unvec(LₓᵀV(LₕE₁)) (Lₕ∂E₁)ᵀ)
        """
        from numpy import stack

        Lh = self._LD["Lh"]

        V = unvec(v, (self._G.shape[0], -1))
        dE = zeros_like(self._C0.L)
        E = self._C0.L
        LhE = Lh @ E
        LdKL_dot = {"C0.Lu": [], "C1.Lu": []}
        for i in range(self._C0.Lu.shape[0]):
            row = i // E.shape[1]
            col = i % E.shape[1]
            dE[row, col] = 1
            LhdE = Lh @ dE
            t = self._LxG @ (self._LxG.T @ ((V @ LhdE) @ LhE.T)) + self._LxG @ (
                self._LxG.T @ ((V @ LhE) @ LhdE.T)
            )
            # L = kron(self._LD["Lh"], self._Lx)
            # dK = self.gradient()["C0.Lu"][..., i]
            # t1 = L @ dK @ L.T @ v
            LdKL_dot["C0.Lu"].append(vec(t)[:, newaxis])
            dE[row, col] = 0

        dE = zeros_like(self._C1.L)
        E = self._C1.L
        LhE = Lh @ E
        for i in range(len(self._C1._tril1[0])):
            row = self._C1._tril1[0][i]
            col = self._C1._tril1[1][i]
            dE[row, col] = 1
            LhdE = Lh @ dE
            t = self._Lx @ (self._Lx.T @ ((V @ LhdE) @ LhE.T)) + self._Lx @ (
                self._Lx.T @ ((V @ LhE) @ LhdE.T)
            )
            # L = kron(self._LD["Lh"], self._Lx)
            # dK = self.gradient()["C1.Lu"][..., i]
            # t1 = L @ dK @ L.T @ v
            LdKL_dot["C1.Lu"].append(vec(t)[:, newaxis])
            dE[row, col] = 0

        for i in range(len(self._C1._diag[0])):
            row = self._C1._diag[0][i]
            col = self._C1._diag[1][i]
            dE[row, col] = E[row, col]
            LhdE = Lh @ dE
            t = self._Lx @ (self._Lx.T @ ((V @ LhE) @ LhdE.T)) + self._Lx @ (
                self._Lx.T @ ((V @ LhdE) @ LhE.T)
            )
            # L = kron(self._LD["Lh"], self._Lx)
            # dK = self.gradient()["C1.Lu"][..., i + 1]
            # t1 = L @ dK @ L.T @ v
            # breakpoint()
            LdKL_dot["C1.Lu"].append(vec(t)[:, newaxis])
            dE[row, col] = 0

        LdKL_dot["C0.Lu"] = stack(LdKL_dot["C0.Lu"], axis=2)
        LdKL_dot["C1.Lu"] = stack(LdKL_dot["C1.Lu"], axis=2)

        return LdKL_dot

    def __str__(self):
        dim = self._C0.L.shape[0]
        rank = self._C0.L.shape[1]
        msg0 = format_function(self, {"G": "...", "dim": dim, "rank": rank})
        msg1 = str(self._C0) + "\n" + str(self._C1)
        msg1 = "  " + "\n  ".join(msg1.split("\n"))
        return (msg0 + msg1).rstrip()
