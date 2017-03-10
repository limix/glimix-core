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

    def value_reduce(self, values):
        r"""Sum covariance function evaluated at `(f_0, f_1, ...)`."""
        return add.reduce(values)

    def derivative_reduce(self, derivatives):
        r"""Sum of covariance function derivatives.

        Returns:
            :math:`f_0' + f_1' + \dots`
        """
        return add.reduce(derivatives)
