from numpy import arange, kron, stack

from glimix_core.util.classes import NamedClass
from optimix import Function

from .eye import EyeCov


class Kron2SumCov(NamedClass, Function):
    def __init__(self, Cr, Cn):
        self._Cr = Cr
        self._Cn = Cn
        Function.__init__(self)
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
        ndim = X.ndim
        Crr = Cr.value(item0, item1)
        Crr = Crr.reshape((1,) * ndim + Crr.shape)
        L = kron(X, Crr.T).T

        eye = EyeCov()
        I = eye.value(id0, id1)
        Cnn = Cn.value(item0, item1)
        Cnn = Cnn.reshape((1,) * ndim + Cnn.shape)
        R = kron(I, Cnn.T).T

        return L + R
