from numpy import zeros
from numpy import ascontiguousarray

from optimix import Function
from optimix import Vector


class LinearMean(Function):
    r"""Linear mean function.

    The mathematical representation is

    .. math::

        f(\mathbf x) = \mathbf x^\intercal \boldsymbol\alpha

    where :math:`\boldsymbol\alpha` is a vector of effect sizes.
    """
    def __init__(self, size):
        Function.__init__(self, effsizes=Vector(zeros(size)))

    def value(self, x):
        r"""Mean function evaluated at `x`.

        Args:
            x (array_like): sample or samples.

        Returns:
            :math:`\mathbf x^\intercal \boldsymbol\alpha`.
        """
        return x.dot(self.get('effsizes'))

    def derivative_effsizes(self, x):
        r"""Linear mean function derivative.

        Args:
            size (int): sample size.

        Returns:
            :math:`\mathbf x`.
        """
        return x

    @property
    def effsizes(self):
        r"""Effect-sizes parameter."""
        return self.get('effsizes')

    @effsizes.setter
    def effsizes(self, v):
        self.set('effsizes', ascontiguousarray(v))
