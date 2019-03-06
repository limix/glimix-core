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
    Implements K =  C₀ ⊗ GGᵀ + C₁ ⊗ I.

    C₀ and C₁ are d×d symmetric matrices. C₀ is a semi-definite positive matrix while C₁
    is a positive definite one. G is a n×m matrix and I is a n×n identity matrix. Let
    M = Uₘ Sₘ Uₘᵀ be the eigen decomposition for any matrix M. The documentation and
    implementation of this class make use of the following definitions:

    - X = GGᵀ = Uₓ Sₓ Uₓᵀ
    - C₁ = U₁ S₁ U₁ᵀ
    - Cₕ = S₁⁻½ U₁ᵀ C₀ U₁ S₁⁻½
    - Cₕ = Uₕ Sₕ Uₕᵀ
    - D = (Sₕ ⊗ Sₓ + Iₕₓ)⁻¹
    - Lₓ = Uₓᵀ
    - Lₕ = Uₕᵀ S₁⁻½ U₁ᵀ
    - L = Lₕ ⊗ Lₓ

    The above definitions allows us to write the inverse of the covariance matrix as::

        K⁻¹ = LᵀDL,

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
        self._Sx = S * S
        self._Lx = U.T
        self._LxG = self._Lx @ G
        self._diag_LxGGLx = dotd(self._LxG, self._LxG.T)
        self._I = eye(self.G.shape[0])
        Function.__init__(
            self, "Kron2SumCov", composite=[("C0", self._C0), ("C1", self._C1)]
        )

    @property
    def Lx(self):
        return self._Lx

    @property
    @lru_cache(maxsize=None)
    def _X(self):
        return self.G @ self.G.T

    @property
    def LhD(self):
        """
        Implements Lₕ and D.

        Returns
        -------
        Lh : ndarray
            Uₕᵀ S₁⁻½ U₁ᵀ.
        D : ndarray
            (Sₕ ⊗ Sₓ + Iₕₓ)⁻¹.
        """
        S1, U1 = self.C1.eigh()
        U1S1 = ddot(U1, 1 / sqrt(S1))
        Sh, Uh = eigh(U1S1.T @ self.C0.value() @ U1S1)
        return {"Lh": (U1S1 @ Uh).T, "D": 1 / (kron(Sh, self._Sx) + 1)}

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
        Covariance matrix K = C₀ ⊗ GGᵀ + C₁ ⊗ I.

        Returns
        -------
        K : ndarray
            C₀ ⊗ GGᵀ + C₁ ⊗ I.
        """
        C0 = self._C0.value()
        C1 = self._C1.value()
        return kron(C0, self._X) + kron(C1, self._I)

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
        C0 = self._C0.gradient()["Lu"].T
        C1 = self._C1.gradient()["Lu"].T
        grad = {"C0.Lu": kron(C0, self._X).T, "C1.Lu": kron(C1, self._I).T}
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
        Implements the product K⁻¹⋅v.

        Parameters
        ----------
        v : array_like
            Array to be multiplied.

        Returns
        -------
        x : ndarray
            Solution x to the equation K⋅x = y.
        """
        L = kron(self.LhD["Lh"], self.Lx)
        return L.T @ ddot(self.LhD["D"], L @ v, left=True)

    def logdet(self):
        """
        Implements log|K| = - log|D| + n⋅log|C₁|.

        Returns
        -------
        logdet : float
            Log-determinant of K.
        """
        return -log(self.LhD["D"]).sum() + self.G.shape[0] * self.C1.logdet()

    def logdet_gradient(self):
        """
        Implements ∂log|K| = Tr[K⁻¹∂K].

        It can be shown that::

            ∂log|K| = diag(D)ᵀdiag(L(∂K)Lᵀ) = diag(D)ᵀ(diag(Lₕ∂C₀Lₕᵀ)⊗diag(LₓGGᵀLₓᵀ)),

        when the derivative is over the parameters of C₀. Similarly,

            ∂log|K| = diag(D)ᵀdiag(L(∂K)Lᵀ) = diag(D)ᵀ(diag(Lₕ∂C₁Lₕᵀ)⊗diag(I)),

        over the parameters of C₁.

        Returns
        -------
        C0 : ndarray
            Derivative of C₀ over its parameters.
        C1 : ndarray
            Derivative of C₁ over its parameters.
        """
        Lh = self.LhD["Lh"]
        D = self.LhD["D"]

        dC0 = self._C0.gradient()["Lu"]
        grad_C0 = zeros_like(self._C0.Lu)
        for i in range(self._C0.Lu.shape[0]):
            t = kron(dotd(Lh, dC0[..., i] @ Lh.T), self._diag_LxGGLx)
            grad_C0[i] = (D * t).sum()

        dC1 = self._C1.gradient()["Lu"]
        grad_C1 = zeros_like(self._C1.Lu)
        for i in range(self._C1.Lu.shape[0]):
            t = kron(dotd(Lh, dC1[..., i] @ Lh.T), self._I.diagonal())
            grad_C1[i] = (D * t).sum()

        return {"C0.Lu": grad_C0, "C1.Lu": grad_C1}

    def LdKL_dot(self, v):
        """
        Implements L(∂K)Lᵀv.

        The array v can have one or two dimensions and the first dimension has to have
        size n⋅p.

        Let vec(V) = v. We have

            L(∂K)Lᵀ⋅v = ((Lₕ∂C₀Lₕᵀ) ⊗ (LₓGGᵀLₓᵀ))vec(V) = vec(LₓGGᵀLₓᵀVLₕ∂C₀Lₕᵀ),

        when the derivative is over the parameters of C₀. Similarly,

            L(∂K)Lᵀv = ((Lₕ∂C₁Lₕᵀ) ⊗ (LₓLₓᵀ))vec(V) = vec(LₓLₓᵀVLₕ∂C₁Lₕᵀ),

        over the parameters of C₁.
        """
        from numpy import stack, einsum

        def dot(a, b):
            le = "ijk"[: a.ndim]
            ri = "jlk"[: b.ndim]
            re = "ilk"[: max(a.ndim, b.ndim)]
            return einsum(f"{le},{ri}->{re}", a, b)

        Lh = self.LhD["Lh"]
        V = unvec(v, (self.G.shape[0], -1) + v.shape[1:])
        LdKL_dot = {"C0.Lu": [], "C1.Lu": []}

        dC0 = self._C0.gradient()["Lu"]
        for i in range(self._C0.Lu.shape[0]):
            t = dot(self._LxG, dot(self._LxG.T, dot(V, Lh @ dC0[..., i] @ Lh.T)))
            LdKL_dot["C0.Lu"].append(t.reshape((-1,) + t.shape[2:], order="F"))

        dC1 = self._C1.gradient()["Lu"]
        for i in range(self._C1.Lu.shape[0]):
            t = dot(self._Lx, dot(self._Lx.T, dot(V, Lh @ dC1[..., i] @ Lh.T)))
            LdKL_dot["C1.Lu"].append(t.reshape((-1,) + t.shape[2:], order="F"))

        LdKL_dot["C0.Lu"] = stack(LdKL_dot["C0.Lu"], axis=-1)
        LdKL_dot["C1.Lu"] = stack(LdKL_dot["C1.Lu"], axis=-1)

        return LdKL_dot

    def __str__(self):
        dim = self._C0.L.shape[0]
        rank = self._C0.L.shape[1]
        msg0 = format_function(self, {"G": "...", "dim": dim, "rank": rank})
        msg1 = str(self._C0) + "\n" + str(self._C1)
        msg1 = "  " + "\n  ".join(msg1.split("\n"))
        return (msg0 + msg1).rstrip()
