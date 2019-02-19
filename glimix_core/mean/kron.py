from __future__ import division

from numpy import asarray, atleast_2d, dot, kron, ones, stack, zeros

from optimix import Function, Vector

from ..util.classes import NamedClass


class KronMean(NamedClass, Function):
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
        Function.__init__(self, vecB=Vector(vecB))
        NamedClass.__init__(self)

    def value_AF(self, A, F):
        r""" Kronecker mean function. """
        b = self.variables().get("vecB").value
        n = F.shape[0]
        p = A.shape[0]
        return dot(kron(A, F), b).reshape((n, p))

    def value(self, x):
        r""" Kronecker mean function. """

        x = asarray(x, float)
        orig_ndim = x.ndim
        x = atleast_2d(x)
        p = self._p
        c = self._c
        n = (len(x[0]) - p * p) // c
        As = [i[: p * p].reshape((p, p)) for i in x]
        Fs = [i[p * p :].reshape((n, c)) for i in x]
        h = x.ndim - orig_ndim
        r = stack([self.value_AF(A, F) for A, F in zip(As, Fs)])
        return r.reshape(r.shape[h:])

    def gradient(self, x):
        r"""Gradient of the linear mean function.

        Parameters
        ----------
        x : array_like
            Covariates.

        Returns
        -------
        dict
            Dictionary having the `effsizes` key for :math:`\mathbf x`.
        """
        from numpy import diag

        x = atleast_2d(asarray(x, float))
        p = self._p
        c = self._c
        n = (len(x[0]) - p * p) // c
        As = [i[: p * p].reshape((p, p)) for i in x]
        Fs = [i[p * p :].reshape((n, c)) for i in x]
        one = diag(ones(p * c))
        AF1s = [dot(kron(A, F), one).reshape((n, p, p * c)) for A, F in zip(As, Fs)]
        r = stack(AF1s)
        return dict(vecB=r)

    @property
    def B(self):
        r""" Effect-sizes parameter. """
        return self.variables().get("vecB").value.reshape((self._c, self._p))

    @B.setter
    def B(self, v):
        v = asarray(v, float)
        self.variables().get("vecB").value[:] = v.ravel()

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
