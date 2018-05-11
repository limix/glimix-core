from __future__ import division

from numpy import ascontiguousarray, dot, zeros

from optimix import Function, Vector


class LinearMean(Function):
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

        >>> from glimix_core.mean import LinearMean
        >>>
        >>> mean = LinearMean(2)
        >>> mean.effsizes = [1.0, -1.0]
        >>> x = [5.1, 1.0]
        >>> print(mean.value(x))
        4.1
        >>> print(mean.gradient(x))
        {'effsizes': [5.1, 1.0]}
    """

    def __init__(self, size):
        Function.__init__(self, effsizes=Vector(zeros(size)))

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
        return dot(x, self.variables().get('effsizes').value)

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
        return self.variables().get('effsizes').value

    @effsizes.setter
    def effsizes(self, v):
        self.variables().get('effsizes').value = ascontiguousarray(v)
