from numpy import asarray, atleast_2d, concatenate, diagonal, eye, kron, log, sqrt
from numpy.linalg import eigh, svd

from numpy_sugar.linalg import ddot
from optimix import Function

from .._util import format_function
from .free import FreeFormCov
from .lrfree import LRFreeFormCov


class Kron2SumCov(Function):
    """
    Implements K = Cᵣ ⊗ GGᵗ + Cₙ ⊗ I.

    Cᵣ and Cₙ are d×d symmetric matrices. Cᵣ is a semi-definite positive matrix while Cₙ
    is positive definite one. G is a n×m matrix and I is a n×n identity matrix. Let
    M = Uₘ Sₘ Uₘᵗ be the eigen decomposition for any matrix M. The documentation and
    implementation of this class make use of the following definitions:

    - X = GGᵗ = Uₓ Sₓ Uₓᵗ
    - Cₙ = Uₙ Sₙ Uₙᵗ
    - Cₕ = Sₙ⁻½ Uₙᵗ Cᵣ Uₙ Sₙ⁻½
    - Cₕ = Uₕ Sₕ Uₕᵗ
    - D = (Sₕ ⊗ Sₓ + Iₕₓ)⁻¹
    - Lₓ = Uₓᵗ
    - Lₙ = Uₕᵗ Sₙ⁻½ Uₙᵗ
    - L = Lₙ ⊗ Lₓ

    The above definitions allows us to write the inverse of the covariance matrix as:

        K⁻¹ = LᵗDL,

    where D is a diagonal matrix.

    Parameters
    ----------
    dim : int
        Dimension d for the square matrices Cᵣ and Cₙ.
    rank : int
        Maximum rank of the Cₙ matrix.

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
        >>> cov = Kron2SumCov(2, 1)
        >>> cov.G = G
        >>> cov.Cr.L = Lr
        >>> cov.Cn.L = Ln
        >>> print(cov)
        Kron2SumCov(dim=2, rank=1): Kron2SumCov
          LRFreeFormCov(n=2, m=1): Cᵣ
            L: [[3.]
                [2.]]
          FreeFormCov(dim=2): Cₙ
            L: [[1. 0.]
                [2. 1.]]
    """

    def __init__(self, dim, rank):
        self._Cr = LRFreeFormCov(dim, rank)
        self._Cr.name = "Cᵣ"
        self._Cn = FreeFormCov(dim)
        self._Cn.name = "Cₙ"
        self._G = None
        self._I = None
        Function.__init__(
            self, "Kron2SumCov", composite=[("Cr", self._Cr), ("Cn", self._Cn)]
        )

    @property
    def G(self):
        """
        User-provided matrix G, n×m, used to evaluate this covariance function.
        """
        return self._G

    @G.setter
    def G(self, G):
        self._G = atleast_2d(asarray(G, float))
        U, S, _ = svd(G)
        S = concatenate((S, [0.0] * (U.shape[0] - S.shape[0])))
        self._USx = U, S * S
        self._I = eye(self.G.shape[0])

    @property
    def Cr(self):
        """
        Semi-definite positive matrix Cᵣ, .
        """
        return self._Cr

    @property
    def Cn(self):
        """
        Definite positive matrix Cᵣ.
        """
        return self._Cn

    @property
    def L(self):
        """
        L = Lₙ ⊗ Lₓ
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
        Covariance matrix K = Cᵣ ⊗ GGᵗ + Cₙ ⊗ I.
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
            Derivative over the array Lu of Cᵣ.
        Cn_L0 : ndarray
            Derivative over the array L0 of Cₙ.
        Cn_L1 : ndarray
            Derivative over the array L1 of Cₙ.
        """
        I = self._I
        X = self.G @ self.G.T

        Cr_Lu = self._Cr.gradient()["Lu"].transpose([2, 0, 1])
        Cn_grad = self._Cn.gradient()
        Cn_L0 = Cn_grad["L0"].transpose([2, 0, 1])
        Cn_L1 = Cn_grad["L1"].transpose([2, 0, 1])

        return {
            "Cr.Lu": kron(Cr_Lu, X).transpose([1, 2, 0]),
            "Cn.L0": kron(Cn_L0, I).transpose([1, 2, 0]),
            "Cn.L1": kron(Cn_L1, I).transpose([1, 2, 0]),
        }

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
            Solution x of the equation K x = y.
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
        Implements log|K| = - log|D| + N log|Cₙ|.
        """
        Sn, Un = self.Cn.eigh()
        Cr = self.Cr.value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Ch = UnSn.T @ Cr @ UnSn
        Sh, Urs = eigh(Ch)
        Qx, Sx = self._USx
        D = 1 / (kron(Sh, Sx) + 1)
        N = self.G.shape[0]
        logdetC = self.Cn.logdet()
        return -log(D).sum() + N * logdetC

    def logdet_gradient(self):
        """
        Implements ∂log|K| = Tr[K⁻¹ ∂K].

        It can be shown that

            ∂log|K| = diag(D)ᵗ diag(L ∂K Lᵗ).

        Returns
        -------
        Cr_Lu : ndarray
            Derivative over the array Lu of Cᵣ.
        Cn_L0 : ndarray
            Derivative over the array L0 of Cₙ.
        Cn_L1 : ndarray
            Derivative over the array L1 of Cₙ.
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
        L = kron(Lc, Lg)

        Kgrad = self.gradient()

        r0 = (
            D
            * diagonal(L @ Kgrad["Cr.Lu"].transpose([2, 0, 1]) @ L.T, axis1=1, axis2=2)
        ).sum(1)
        r1 = (
            D
            * diagonal(L @ Kgrad["Cn.L0"].transpose([2, 0, 1]) @ L.T, axis1=1, axis2=2)
        ).sum(1)
        r2 = (
            D
            * diagonal(L @ Kgrad["Cn.L1"].transpose([2, 0, 1]) @ L.T, axis1=1, axis2=2)
        ).sum(1)
        return {"Cr.Lu": r0, "Cn.L0": r1, "Cn.L1": r2}

    def __str__(self):
        dim = self._Cr.L.shape[0]
        rank = self._Cr.L.shape[1]
        msg0 = format_function(self, {"dim": dim, "rank": rank})
        msg1 = str(self._Cr) + "\n" + str(self._Cn)
        msg1 = "  " + "\n  ".join(msg1.split("\n"))
        return (msg0 + msg1).rstrip()
