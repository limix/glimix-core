from __future__ import division

from numpy import ascontiguousarray, dot, zeros

from optimix import Function, Vector

from ..util.classes import NamedClass


class LinearMean(NamedClass, Function):
    r"""Linear mean function.

    The mathematical representation is

    .. math::

        f(\mathbf x) = \mathbf x^\intercal \boldsymbol\alpha

    where :math:`\boldsymbol\alpha` is a vector of effect sizes.

    Parameters
    ----------
    size : int
        Number of effects.

    Example
    -------

    .. doctest::

        >>> from numpy import asarray
        >>> from glimix_core.mean import LinearMean
        >>>
        >>> mean = LinearMean(2)
        >>> mean.effsizes = [1.0, -1.0]
        >>> x = [5.1, 1.0]
        >>> print(mean.value(x))
        4.1
        >>> print(mean.gradient(x))
        {'effsizes': [5.1, 1.0]}
        >>> print(mean)
        LinearMean(size=2)
          effsizes: [ 1. -1.]
        >>> mean.name = "covariates"
        >>> print(mean)
        LinearMean(size=2): covariates
          effsizes: [ 1. -1.]
        >>> X = asarray([[5.1,  1.0],
        ...              [0.3, -9.0]])
        >>> mean.set_data(X)
        >>> print(mean.feed().value())
        [4.1 9.3]
        >>> X = asarray([[5.1,  1.0],
        ...              [0.3, -9.0],
        ...              [4.9,  0.0]])
        >>> mean.set_data(X, "dataB")
        >>> print(mean.feed("dataB").value())
        [4.1 9.3 4.9]
    """

    def __init__(self, size):
        Function.__init__(self, effsizes=Vector(zeros(size)))
        self.variables().get("effsizes").bounds = [(-200.0, +200)] * size
        NamedClass.__init__(self)

    def value(self, x):
        r"""Linear mean function.

        Parameters
        ----------
        x : array_like
            Covariates.

        Returns
        -------
        float
            :math:`\mathbf x^\intercal \boldsymbol\alpha`.
        """
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
