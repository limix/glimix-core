from numpy import arange, asarray, atleast_2d, dot, kron, stack, eye, sqrt, log, newaxis
from scipy.linalg import svd, eigh

from glimix_core.util.classes import NamedClass
from optimix import Function
from numpy_sugar.linalg import ddot, economic_qs, economic_svd

from .eye import EyeCov


def svd_reduce(X, tol=1e-9):
    U, Sh, V = svd(X, full_matrices=0)
    I = Sh < tol
    if I.any():
        # warnings.warn('G has dependent columns, dimensionality reduced')
        Sh = Sh[~I]
        U = U[:, ~I]
        V = eye(Sh.shape[0])
        X = U * Sh[newaxis, :]
    S = Sh ** 2
    return X, U, S, V


class Kron2SumCov(NamedClass, Function):
    def __init__(self, Cr, Cn):
        self._Cr = Cr
        self._Cn = Cn
        self._G = None
        Cr_Lu = Cr.variables().get("Lu")
        Cn_Lu = Cn.variables().get("Lu")
        Function.__init__(self, Cr_Lu=Cr_Lu, Cn_Lu=Cn_Lu)
        NamedClass.__init__(self)

    @property
    def G(self):
        return self._G

    def set_data_G(self, G):
        self._G = atleast_2d(asarray(G, float))
        self._G = svd_reduce(self._G)[0]
        self._dim_r = self._G.shape[0]
        self._rank_r = self._G.shape[1]
        self.set_data((self._G, self._G))

    @property
    def Cr(self):
        return self._Cr

    @property
    def Cn(self):
        return self._Cn

    def value(self, x0, x1):
        Cr = self._Cr
        Cn = self._Cn

        x0 = stack(x0, axis=0)
        id0 = x0[..., 0].astype(int)
        x0 = x0[..., 1:]

        x1 = stack(x1, axis=0)
        id1 = x1[..., 0].astype(int)
        x1 = x1[..., 1:]

        p = Cr.size
        item0 = arange(p)
        item1 = arange(p)
        X = x0.dot(x1.T)
        I = EyeCov().value(id0, id1)
        shape = (1,) * X.ndim + (p, p)

        Cr = Cr.value(item0, item1).reshape(shape)
        Cn = Cn.value(item0, item1).reshape(shape)

        return kron(X, Cr.T).T + kron(I, Cn.T).T

    def compact_value(self):
        K = self.feed().value()
        d = K.shape[0] * K.shape[2]
        return K.transpose((2, 0, 3, 1)).reshape(d, d)

    def gradient(self, x0, x1):
        Cr = self._Cr
        Cn = self._Cn

        x0 = stack(x0, axis=0)
        id0 = x0[..., 0].astype(int)
        x0 = x0[..., 1:]

        x1 = stack(x1, axis=0)
        id1 = x1[..., 0].astype(int)
        x1 = x1[..., 1:]

        p = Cr.size
        item0 = arange(p)
        item1 = arange(p)
        X = x0.dot(x1.T)
        I = EyeCov().value(id0, id1)

        Cr_Lu = Cr.gradient(item0, item1)["Lu"]
        Cr_Lu = Cr_Lu.reshape((1,) * X.ndim + Cr_Lu.shape)

        Cn_Lu = Cn.gradient(item0, item1)["Lu"]
        Cn_Lu = Cn_Lu.reshape((1,) * X.ndim + Cn_Lu.shape)

        return {"Cr_Lu": kron(X, Cr_Lu.T).T, "Cn_Lu": kron(I, Cn_Lu.T).T}

    def logdet(self):
        breakpoint()
        Cr = self._Cr
        Cn = self._Cn
        E = self.Cr.L
        p = Cr.size
        item0 = arange(p)
        item1 = arange(p)
        C = Cn.value(item0, item1)
        C = C + eye(C.shape[0]) * 1e-4
        QS = economic_qs(C)
        Lc = ddot(QS[0][0], 1 / sqrt(QS[1])).T
        Estar = dot(Lc, E)

        rank_c = Cr.rank

        Ue, Seh, Ve = svd(Estar, full_matrices=0)
        Se = Seh ** 2
        _, _Sg, _ = economic_svd(self.G)
        SpI = kron(1.0 / Se, 1.0 / _Sg) + 1

        logdetSg = log(_Sg).sum()

        # dim_r: rank of G after is has been reduced by svd

        CnS = eigh(C)[0]
        rv = sum(log(CnS)) * self._dim_r
        rv += log(SpI).sum()
        rv += log(Se).sum() * self._rank_r
        rv += logdetSg * rank_c

        # breakpoint()
        return rv
