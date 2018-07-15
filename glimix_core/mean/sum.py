from numpy import add

from optimix import FunctionReduce

from ..util.classes import NamedClass


class SumMean(NamedClass, FunctionReduce):
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
        >>> print(mean)
        SumMean(means=...)
          LinearMean(size=2)
            effsizes: [-1.   0.5]
          OffsetMean()
            offset: 2.0
        >>> mean0.name = "A"
        >>> mean1.name = "B"
        >>> mean.name = "A+B"
        >>> print(mean)
        SumMean(means=...): A+B
          LinearMean(size=2): A
            effsizes: [-1.   0.5]
          OffsetMean(): B
            offset: 2.0
    """

    def __init__(self, means):
        self._means = [c for c in means]
        FunctionReduce.__init__(self, self._means, "sum")
        NamedClass.__init__(self)

    def value_reduce(self, values):
        r"""Sum mean function evaluated at :math:`(f_0, f_1, \dots)`.

        Parameters
        ----------
        values : dict
            A value for each function involved in the summation.

        Returns
        -------
        dict
            :math:`f_0 + f_1 + \dots`
        """
        return add.reduce(list(values.values()))

    def gradient_reduce(self, values, gradients):
        r"""Sum of mean function derivatives.

        Parameters
        ----------
        values : dict
            Its value is not used in this particular function. We suggest you to simply
            pass ``None``.
        gradients : dict
            A gradient for each function involved in the summation.

        Returns
        -------
        dict
            Dictionary having a key for each parameter of the underlying
            mean functions.
        """
        grad = dict()
        for (gn, gv) in iter(gradients.items()):
            for n, v in iter(gv.items()):
                grad[gn + "." + n] = v
        return grad

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(means=...)".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        for m in self._means:
            spl = str(m).split("\n")
            msg = msg + "\n" + "\n".join(["  " + s for s in spl])
        return msg
