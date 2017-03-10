from numpy import add

from optimix import FunctionReduce


class SumMean(FunctionReduce):
    r"""Sum mean function.

    The mathematical representation is

    .. math::

        f(f_0, f_1, \dots) = f_0 + f_1 + \dots
    """
    def __init__(self, means):
        self._means = [c for c in means]
        FunctionReduce.__init__(self, self._means, 'sum')

    def value_reduce(self, values):
        r"""Sum mean function evaluated at `(f_0, f_1, ...)`."""
        return add.reduce(values)

    def derivative_reduce(self, derivatives):
        r"""Sum of mean function derivatives.

        Returns:
            :math:`f_0' + f_1' + \dots`
        """
        return add.reduce(derivatives)
