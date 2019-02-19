from __future__ import division

from numpy import ascontiguousarray, dot, zeros

from optimix import Function, Vector

from ..util.classes import NamedClass


class KronMean(NamedClass, Function):
    r""" Kronecker mean function.

    Let
    - n be the number of samples;
    - p the number of traits;
    - c the number of covariates.

    The mathematical representation is

        f(𝐗)=(𝐀ᵗ⊗𝐅)vec(𝐗)

    where 𝐀 is a c×p trait design matrix of fixed effects and 𝐅 is a n×c sample design
    matrix of fixed effects.
    """

    def __init__(self):
        Function.__init__(self)
        NamedClass.__init__(self)

    def value(self, x):
        r""" Kronecker  mean function. """
        return dot(x, self.variables().get("effsizes").value)

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
        return dict(effsizes=x)

    @property
    def effsizes(self):
        r"""Effect-sizes parameter."""
        return self.variables().get("effsizes").value

    @effsizes.setter
    def effsizes(self, v):
        self.variables().get("effsizes").value = ascontiguousarray(v)

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(size={})".format(tname, len(self.effsizes))
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        msg += "  effsizes: {}".format(self.effsizes)
        return msg
