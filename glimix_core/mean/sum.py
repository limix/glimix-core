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

    def value_reduce(self, values):  # pylint: disable=R0201
        r"""Sum mean function evaluated at `(f_0, f_1, ...)`."""
        return add.reduce(list(values.values()))

    def gradient_reduce(self, _, gradients):  # pylint: disable=R0201
        r"""Sum of mean function derivatives.

        Returns:
            :math:`f_0' + f_1' + \dots`
        """
        grad = dict()
        for (gn, gv) in iter(gradients.items()):
            for n, v in iter(gv.items()):
                grad[gn + '.' + n] = v
        return grad
