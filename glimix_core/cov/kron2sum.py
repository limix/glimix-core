from numpy import (
    arange,
    asarray,
    atleast_2d,
    concatenate,
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
    """ Implements K = Cᵣ ⊗ GGᵗ + Cₙ ⊗ I. """

    def __init__(self, dim, rank):
        items = arange(dim)
        self._Cr = LRFreeFormCov(dim, rank)
        self._Cr.set_data((items, items))
        self._Cn = FreeFormCov(dim)
        self._Cn.set_data((items, items))
        self._G = None
        self._eye = EyeCov()
        Cr_Lu = self._Cr.variables().get("Lu")
        Cn_Lu = self._Cn.variables().get("Lu")
        Function.__init__(self, Cr_Lu=Cr_Lu, Cn_Lu=Cn_Lu)
        NamedClass.__init__(self)

    def _I(self, items0, items1):
        return self._eye.value(items0, items1)

    @property
    def G(self):
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
        return self._Cr

    @property
    def Cn(self):
        return self._Cn

    def value(self, x0, x1):
        id0, x0, = _input_split(x0)
        id1, x1 = _input_split(x1)

        I = self._I(id0, id1)
        X = x0.dot(x1.T)

        Cr = _prepend_dims(self._Cr.feed().value(), X.ndim)
        Cn = _prepend_dims(self._Cn.feed().value(), X.ndim)

        return kron(X, Cr.T).T + kron(I, Cn.T).T

    def compact_value(self):
        return _compact_form(self.feed().value())

    def gradient(self, x0, x1):
        id0, x0, = _input_split(x0)
        id1, x1 = _input_split(x1)

        I = self._I(id0, id1)
        X = x0.dot(x1.T)

        Cr_Lu = self._Cr.feed().gradient()["Lu"]
        Cr_Lu = _prepend_dims(Cr_Lu, X.ndim)

        Cn_Lu = self._Cn.feed().gradient()["Lu"]
        Cn_Lu = _prepend_dims(Cn_Lu, X.ndim)

        return {"Cr_Lu": kron(X, Cr_Lu.T).T, "Cn_Lu": kron(I, Cn_Lu.T).T}

    def solve(self, v):
        """ Implements the product K⁻¹v.

        Let X = GGᵗ, K = Cᵣ ⊗ X + Cₙ ⊗ I, and let us use the notation M = Uₘ Sₘ Uₘᵗ for
        eigen decomposition. Let Cᵣ* = Sₙ⁻½ Uₙᵗ Cᵣ Uₙ Sₙ⁻½. We then define
        Lₙ = Uᵣᵗ* Sₙ⁻½ Uₙᵗ* and Lᵣ = Uₓᵗ. The inverse of the covariance matrix is

            K⁻¹ = Lᵗ D L,

        for which D = (Sᵣ* ⊗ Sₓ + I)⁻¹ is a block diagonal matrix and L = Lₙ ⊗ Lᵣ.
        """
        Sn, Un = eigh(self.Cn.feed().value())
        Cr = self.Cr.feed().value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Crs = UnSn.T @ Cr @ UnSn
        Srs, Urs = eigh(Crs)
        Qx, Sx = self._USx
        D = 1 / (kron(Srs, Sx) + 1)
        Lc = (UnSn @ Urs).T
        Lg = Qx.T
        L = kron(Lc, Lg)
        return L.T @ ddot(D, L @ v)

    def logdet(self):
        """ log|K| = - log|D| + N log|Cₙ| """
        from numpy.linalg import slogdet

        Sn, Un = eigh(self.Cn.feed().value())
        Cr = self.Cr.feed().value()
        UnSn = ddot(Un, 1 / sqrt(Sn))
        Crs = UnSn.T @ Cr @ UnSn
        Srs, Urs = eigh(Crs)
        Qx, Sx = self._USx
        D = 1 / (kron(Srs, Sx) + 1)
        N = self.G.shape[0]
        logdetC = slogdet(self.Cn.feed().value())
        assert logdetC[0] == 1
        return -log(D).sum() + N * logdetC[1]


def _input_split(x):
    x = stack(x, axis=0)
    ids = x[..., 0].astype(int)
    x = x[..., 1:]
    return ids, x


def _prepend_dims(x, ndims):
    return x.reshape((1,) * ndims + x.shape)


def _compact_form(K):
    d = K.shape[0] * K.shape[2]
    return K.transpose((2, 0, 3, 1)).reshape(d, d)
