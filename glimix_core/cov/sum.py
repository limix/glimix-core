from __future__ import division

from numpy import add

from optimix import FunctionReduce


class SumCov(FunctionReduce):
    r"""Sum covariance function.

    The mathematical representation is

    .. math::

        f(f_0, f_1, \dots) = f_0 + f_1 + \dots
    """

    def __init__(self, covariances):
        self._covariances = [c for c in covariances]
        FunctionReduce.__init__(self, self._covariances, 'sum')

    def value_reduce(self, values):  # pylint: disable=R0201
        r"""Sum covariance function evaluated at `(f_0, f_1, ...)`."""
        return add.reduce(list(values.values()))

    def gradient_reduce(self, _, gradients):  # pylint: disable=R0201
        r"""Sum of covariance function derivatives.

        Returns:
            :math:`f_0' + f_1' + \dots`
        """
        grad = dict()
        for (gn, gv) in iter(gradients.items()):
            for n, v in iter(gv.items()):
                grad[gn + '.' + n] = v
        return grad
