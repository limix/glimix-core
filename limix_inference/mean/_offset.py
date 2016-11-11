from numpy import full
from numpy import ones

from optimix import Function
from optimix import Scalar


class OffsetMean(Function):
    r"""Offset mean function.

    The mathematical representation is

    .. math::

        f(n) = o \mathbf 1

    where :math:`\mathbf 1` is a :math:`n`-sized vector of ones.
    """
    def __init__(self):
        Function.__init__(self, offset=Scalar(1.0))

    def value(self, size):
        r"""Offset function evaluated for `size` samples.

        Args:
            size (int): sample size.

        Returns:
            :math:`o \mathbf 1`.
        """
        return full(size, self.get('offset'))

    def derivative_offset(self, size):
        r"""Offset function derivative.

        Args:
            size (int): sample size.

        Returns:
            :math:`\mathbf 1`.
        """
        return ones(size)

    @property
    def offset(self):
        r"""Offset parameter."""
        return self.get('offset')

    @offset.setter
    def offset(self, v):
        self.set('offset', v)
