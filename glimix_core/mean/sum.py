from numpy import add

from optimix import FunctionReduce


class SumMean(FunctionReduce):
    r"""Sum mean function.

    The mathematical representation is

    .. math::

        f(f_0, f_1, \dots) = f_0 + f_1 + \dots

    Parameters
    ----------
    means : list
        List of mean functions.

    Example
    -------

    .. doctest::

        >>> from numpy import arange
        >>> from glimix_core.mean import OffsetMean, LinearMean, SumMean
        >>>
        >>> X = [[5.1, 1.0],
        ...      [2.1, -0.2]]
        >>>
        >>> mean0 = LinearMean(2)
        >>> mean0.set_data((X, ))
        >>> mean0.effsizes = [-1.0, 0.5]
        >>>
        >>> mean1 = OffsetMean()
        >>> mean1.set_data((arange(2), ))
        >>> mean1.offset = 2.0
        >>>
        >>> mean = SumMean([mean0, mean1])
        >>>
        >>> print(mean.feed().value())
        [-2.6 -0.2]
        >>> g = mean.feed().gradient()
        >>> print(g['sum[0].effsizes'])
        [[5.1, 1.0], [2.1, -0.2]]
        >>> print(g['sum[1].offset'])
        [1. 1.]
    """

    def __init__(self, means):
        self._means = [c for c in means]
        FunctionReduce.__init__(self, self._means, 'sum')

    def value_reduce(self, values):
        r"""Sum mean function evaluated at :math:`(f_0, f_1, \dots)`."""
        return add.reduce(list(values.values()))

    def gradient_reduce(self, _, gradients):
        r"""Sum of mean function derivatives.

        Returns
        -------
        dict
            Dictionary having a key for each parameter of the underlying
            mean functions.
        """
        grad = dict()
        for (gn, gv) in iter(gradients.items()):
            for n, v in iter(gv.items()):
                grad[gn + '.' + n] = v
        return grad
