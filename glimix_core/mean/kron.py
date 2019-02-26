from numpy import asarray, atleast_2d, concatenate, dot, kron, ones, stack, zeros
from glimix_core.util import vec, unvec
from optimix import Func, Vector


class KronMean(Func):
    r""" Kronecker mean function.

    Let
    - n be the number of samples;
    - p the number of traits;
    - c the number of covariates.

    The mathematical representation is

        f(𝐀,𝐅)=(𝐀⊗𝐅)vec(𝐁)

    where 𝐀 is a p×p trait design matrix of fixed effects and 𝐅 is a n×c sample design
    matrix of fixed effects. 𝐁 is a c×p matrix of fixed-effect sizes.
    """

    def __init__(self, c, p):
        vecB = zeros((c, p)).ravel()
        self._c = c
        self._p = p
        self._A = None
        self._F = None
        self._vecB = Vector(vecB)
        Func.__init__(self, "KronMean", vecB=self._vecB)

    # def _set_data(self):
    #     if self._A is None or self._F is None:
    #         return
    #     item = concatenate((self._A.ravel(), self._F.ravel()))
    #     c = self._c
    #     p = self._p
    #     self._A = item[: p * p].reshape((p, p))
    #     self._F = item[p * p :].reshape((-1, c))
    #     assert self._A.base is item
    #     assert self._F.base is item
    #     self.set_data(item)

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, A):
        self._A = A

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, F):
        self._F = F

    @property
    def AF(self):
        r""" A ⊗ F. """
        return kron(self.A, self.F)

    # def _value(self, A, F):
    #     """ reshape((A ⊗ F) vec(B), n, p) """
    #     b = self._vecB.value
    #     n = F.shape[0]
    #     p = A.shape[0]
    #     return dot(kron(A, F), b).reshape((n, p))

    def value(self):
        """
        Kronecker mean function.
        """
        return self.AF @ self._vecB.value

    def gradient(self):
        r"""Gradient of the linear mean function.

        Returns
        -------
        dict
            Dictionary having the `effsizes` key for :math:`\mathbf x`.
        """
        return {"vecB": self.AF}
        # from numpy import diag

        # x = atleast_2d(asarray(x, float))
        # p = self._p
        # c = self._c
        # n = (len(x[0]) - p * p) // c
        # As = [i[: p * p].reshape((p, p)) for i in x]
        # Fs = [i[p * p :].reshape((n, c)) for i in x]
        # one = diag(ones(p * c))
        # AF1s = [dot(kron(A, F), one).reshape((n, p, p * c)) for A, F in zip(As, Fs)]
        # r = stack(AF1s)
        # return dict(vecB=r)

    @property
    def B(self):
        """
        Effect-sizes parameter.
        """
        return unvec(self._vecB.value, (self._c, self._p))

    @B.setter
    def B(self, v):
        self._vecB.value = vec(asarray(v, float))

    def __str__(self):
        tname = type(self).__name__
        p = self._p
        c = self._c
        msg = "{}(c={},p={})".format(tname, c, p)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        mat = format(self.B)
        msg += "  B: " + "\n     ".join(mat.split("\n"))
        return msg


# def _compact_form(x):
#     d = x.shape[0] * x.shape[2]
#     return x.transpose((2, 0, 3, 1)).reshape(d, d)


# def _compact_form_grad(x):
#     breakpoint()
#     assert x.shape[0] == 1
#     x = x[0, ...]
#     nparams = x.shape[2]
#     return x.ravel(order="F").reshape((-1, nparams), )
#     # x = x.transpose([2, 0, 1])
#     # mats = []
#     # for i in x:
#     #     mats.append(_compact_form(i))
#     # return stack(mats, axis=0)
