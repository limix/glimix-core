from numpy import (
    arange,
    asarray,
    atleast_2d,
    concatenate,
    diagonal,
    kron,
    log,
    newaxis,
    sqrt,
    stack,
)
from numpy.linalg import eigh, svd

from glimix_core.util.classes import NamedClass
from numpy_sugar.linalg import ddot
from optimix import Function

from .eye import EyeCov
from .free import FreeFormCov
from .lrfree import LRFreeFormCov


class Kron2SumCov(NamedClass, Function):
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
        items = arange(dim)
        self._Cr = LRFreeFormCov(dim, rank)
        self._Cr.set_data((items, items))
        self._Cn = FreeFormCov(dim)
        self._Cn.set_data((items, items))
        self._G = None
        self._eye = EyeCov()
        Cr_Lu = self._Cr.variables().get("Lu")
        Cn_Llow = self._Cn.variables().get("Llow")
        Cn_Llogd = self._Cn.variables().get("Llogd")
        Function.__init__(self, Cr_Lu=Cr_Lu, Cn_Llow=Cn_Llow, Cn_Llogd=Cn_Llogd)
        NamedClass.__init__(self)

    def _I(self, items0, items1):
        return self._eye.value(items0, items1)

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
        ids = arange(G.shape[0])[:, newaxis]
        Gids = concatenate((ids, G), axis=1)
        self.set_data((Gids, Gids))

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
        Cr = self.Cr.feed().value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Crs = UnSn.T @ Cr @ UnSn
        Srs, Urs = eigh(Crs)
        Qx, Sx = self._USx
        Lc = (UnSn @ Urs).T
        Lg = Qx.T
        return kron(Lc, Lg)

    def value(self, x0, x1):
        """
        Covariance matrix K.
        """
        id0, x0, = _input_split(x0)
        id1, x1 = _input_split(x1)

        I = self._I(id0, id1)
        X = x0.dot(x1.T)

        Cr = _prepend_dims(self._Cr.feed().value(), X.ndim)
        Cn = _prepend_dims(self._Cn.feed().value(), X.ndim)

        return kron(X, Cr.T).T + kron(I, Cn.T).T

    def compact_value(self):
        """
        Covariance matrix K as a bi-dimensional array.
        """
        return _compact_form(self.feed().value())

    def compact_gradient(self):
        """
        Gradient of K as a bi-dimensional array.

        Returns
        -------
        Cr_Lu : ndarray
            Derivative over the array Lu of Cᵣ.
        Cn_Llow : ndarray
            Derivative over the array Llow of Cₙ.
        Cn_Llogd : ndarray
            Derivative over the array Llogd of Cₙ.
        """
        Gids = _input_join(self.G)

        Kgrad = self.gradient(Gids, Gids)
        Kgrad["Cr_Lu"] = _compact_form_grad(Kgrad["Cr_Lu"])
        Kgrad["Cn_Llow"] = _compact_form_grad(Kgrad["Cn_Llow"])
        Kgrad["Cn_Llogd"] = _compact_form_grad(Kgrad["Cn_Llogd"])

        return Kgrad

    def gradient(self, x0, x1):
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
        id0, x0, = _input_split(x0)
        id1, x1 = _input_split(x1)

        I = self._I(id0, id1)
        X = x0.dot(x1.T)

        Cr_Lu = self._Cr.feed().gradient()["Lu"]
        Cr_Lu = _prepend_dims(Cr_Lu, X.ndim)

        Cn_Llow = self._Cn.feed().gradient()["Llow"]
        Cn_Llow = _prepend_dims(Cn_Llow, X.ndim)

        Cn_Llogd = self._Cn.feed().gradient()["Llogd"]
        Cn_Llogd = _prepend_dims(Cn_Llogd, X.ndim)

        return {
            "Cr_Lu": kron(X, Cr_Lu.T).T,
            "Cn_Llow": kron(I, Cn_Llow.T).T,
            "Cn_Llogd": kron(I, Cn_Llogd.T).T,
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
        Cr = self.Cr.feed().value()
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
        Cr = self.Cr.feed().value()
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
        Cr = self.Cr.feed().value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Ch = UnSn.T @ Cr @ UnSn
        Sh, Uh = eigh(Ch)
        Qx, Sx = self._USx
        D = 1 / (kron(Sh, Sx) + 1)
        Lc = (UnSn @ Uh).T
        Lg = Qx.T
        L = kron(Lc, Lg)

        Kgrad = self.compact_gradient()

        r0 = (D * diagonal(L @ Kgrad["Cr_Lu"] @ L.T, axis1=1, axis2=2)).sum(1)
        r1 = (D * diagonal(L @ Kgrad["Cn_Llow"] @ L.T, axis1=1, axis2=2)).sum(1)
        r2 = (D * diagonal(L @ Kgrad["Cn_Llogd"] @ L.T, axis1=1, axis2=2)).sum(1)
        return {"Cr_Lu": r0, "Cn_Llow": r1, "Cn_Llogd": r2}


def _input_split(x):
    x = stack(x, axis=0)
    ids = x[..., 0].astype(int)
    x = x[..., 1:]
    return ids, x


def _input_join(G):
    ids = arange(G.shape[0])[:, newaxis]
    return concatenate((ids, G), axis=1)


def _prepend_dims(x, ndims):
    return x.reshape((1,) * ndims + x.shape)


def _compact_form(K):
    d = K.shape[0] * K.shape[2]
    return K.transpose((2, 0, 3, 1)).reshape(d, d)


def _compact_form_grad(K):
    K = K.transpose([4, 0, 1, 2, 3])
    mats = []
    for M in K:
        mats.append(_compact_form(M))
    return stack(mats, axis=0)
