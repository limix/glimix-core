from numpy import arange, kron, stack

from glimix_core.util.classes import NamedClass
from optimix import Function

from .eye import EyeCov


class Kron2SumCov(NamedClass, Function):
    def __init__(self, Cr, Cn):
        self._Cr = Cr
        self._Cn = Cn
        Cr_Lu = Cr.variables().get("Lu")
        Cn_Lu = Cn.variables().get("Lu")
        Function.__init__(self, Cr_Lu=Cr_Lu, Cn_Lu=Cn_Lu)
        NamedClass.__init__(self)

    @property
    def G(self):
        return self._G

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
