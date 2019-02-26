from numpy import asarray, atleast_2d, concatenate, eye, diagonal, kron, log, sqrt
from numpy.linalg import eigh, svd

from numpy_sugar.linalg import ddot
from optimix import Func

from .free import FreeFormCov
from .lrfree import LRFreeFormCov


class Kron2SumCov(Func):
    """
    Implements K = Cᵣ ⊗ GGᵗ + Cₙ ⊗ I.

    Cᵣ and Cₙ are d×d symmetric matrices. Cᵣ is a semi-definite positive matrix while Cₙ
    is positive definite one. G is a m×n matrix and I is a m×m identity matrix. Let
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
    """

    def __init__(self, dim, rank):
        self._Cr = LRFreeFormCov(dim, rank)
        self._Cn = FreeFormCov(dim)
        self._G = None
        self._I = None
        Func.__init__(self, "Kron2SumCov", composite=[self._Cr, self._Cn])

    @property
    def G(self):
        """
        User-provided matrix G used to evaluate this covariance function.
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
        """ Semi-definite positive matrix Cᵣ. """
        return self._Cr

    @property
    def Cn(self):
        """ Definite positive matrix Cᵣ. """
        return self._Cn

    @property
    def L(self):
        Sn, Un = self.Cn.eigh()
        Cr = self.Cr.value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Crs = UnSn.T @ Cr @ UnSn
        Srs, Urs = eigh(Crs)
        Qx, Sx = self._USx
        Lc = (UnSn @ Urs).T
        Lg = Qx.T
        return kron(Lc, Lg)

    def value(self):
        """
        Covariance matrix K.
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
        Cn_Llow : ndarray
            Derivative over the array Llow of Cₙ.
        Cn_Llogd : ndarray
            Derivative over the array Llogd of Cₙ.
        """
        I = self._I
        X = self.G @ self.G.T

        Cr_Lu = self._Cr.gradient()["Lu"].transpose([2, 0, 1])
        Cn_grad = self._Cn.gradient()
        Cn_Llow = Cn_grad["Llow"].transpose([2, 0, 1])
        Cn_Llogd = Cn_grad["Llogd"].transpose([2, 0, 1])

        return {
            "Kron2SumCov[0].Lu": kron(Cr_Lu, X).transpose([1, 2, 0]),
            "Kron2SumCov[1].Llow": kron(Cn_Llow, I).transpose([1, 2, 0]),
            "Kron2SumCov[1].Llogd": kron(Cn_Llogd, I).transpose([1, 2, 0]),
        }

    def solve(self, v):
        """ Implements the product K⁻¹v.

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
        """ Implements log|K| = - log|D| + N log|Cₙ| """
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
        """ Implements ∂log|K| = Tr[K⁻¹ ∂K].

        It can be shown that

            ∂log|K| = diag(D)ᵗ diag(L ∂K Lᵗ).

        Returns
        -------
        Cr_Lu : ndarray
            Derivative over the array Lu of Cᵣ.
        Cn_Llow : ndarray
            Derivative over the array Llow of Cₙ.
        Cn_Llogd : ndarray
            Derivative over the array Llogd of Cₙ.
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
            * diagonal(
                L @ Kgrad["Kron2SumCov[0].Lu"].transpose([2, 0, 1]) @ L.T,
                axis1=1,
                axis2=2,
            )
        ).sum(1)
        r1 = (
            D
            * diagonal(
                L @ Kgrad["Kron2SumCov[1].Llow"].transpose([2, 0, 1]) @ L.T,
                axis1=1,
                axis2=2,
            )
        ).sum(1)
        r2 = (
            D
            * diagonal(
                L @ Kgrad["Kron2SumCov[1].Llogd"].transpose([2, 0, 1]) @ L.T,
                axis1=1,
                axis2=2,
            )
        ).sum(1)
        return {
            "Kron2SumCov[0].Lu": r0,
            "Kron2SumCov[1].Llow": r1,
            "Kron2SumCov[1].Llogd": r2,
        }
