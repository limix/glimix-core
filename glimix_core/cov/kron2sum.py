from numpy import arange, asarray, atleast_2d, dot, eye, kron, log, newaxis, sqrt, stack, concatenate
# from scipy.linalg import eigh
from numpy.linalg import eigh, svd

from glimix_core.util.classes import NamedClass
from numpy_sugar.linalg import ddot, economic_qs, economic_svd
from optimix import Function

from .eye import EyeCov
from .free import FreeFormCov
from .lrfree import LRFreeFormCov


class Kron2SumCov(NamedClass, Function):
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
        Q, S, V = svd(G, full_matrices=False)
        self._QSg = Q, S * S
        ids = arange(G.shape[0])[:, newaxis]
        X = concatenate((ids, G), axis=1)
        self.set_data((X, X))

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
        Sn, Qn = eigh(self.Cn.feed().value())
        Cr = self.Cr.feed().value()
        QnSn = ddot(Qn, 1 / sqrt(Sn))
        Crs = QnSn.T @ Cr @ QnSn
        Srs, Qrs = eigh(Crs)
        Qg, Sg = self._QSg
        D = 1 / (kron(Srs, Sg) + 1)
        Lc = (QnSn @ Qrs).T
        Lg = Qg.T
        L = kron(Lc, Lg)
        Ki = L.T @ ddot(D, L)
        K0 = _compact_form(self.feed().value())
        I = eye(self.G.shape[0])
        K1 = kron(Cr, self.G @ self.G.T) + kron(self.Cn.feed().value(), I)
        return L.T @ ddot(D, L @ v)


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
