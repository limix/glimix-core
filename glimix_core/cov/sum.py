from __future__ import division

from numpy import add

from optimix import FunctionReduce

from ..util.classes import NamedClass


class SumCov(NamedClass, FunctionReduce):
    r"""Sum covariance function.

    The mathematical representation is

    .. math::

        f(f_0, f_1, \dots) = f_0 + f_1 + \dots

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import LinearCov, SumCov
        >>> from numpy.random import RandomState
        >>>
        >>> random = RandomState(0)
        >>> cov_left = LinearCov()
        >>> cov_right = LinearCov()
        >>>
        >>> cov_left.scale = 0.5
        >>> X0 = random.randn(4, 20)
        >>> X1 = random.randn(4, 15)
        >>>
        >>> cov_left.set_data((X0, X0))
        >>> cov_right.set_data((X1, X1))
        >>>
        >>> cov = SumCov([cov_left, cov_right])
        >>> print(cov)
        SumCov(covariances=...)
          LinearCov()
            scale: 0.5
          LinearCov()
            scale: 1.0
        >>> cov_left.name = "A"
        >>> cov_right.name = "B"
        >>> cov.name = "A+B"
        >>> print(cov)
        SumCov(covariances=...): A+B
          LinearCov(): A
            scale: 0.5
          LinearCov(): B
            scale: 1.0
    """

    def __init__(self, covariances):
        self._covariances = [c for c in covariances]
        FunctionReduce.__init__(self, self._covariances, "sum")
        NamedClass.__init__(self)

    def value_reduce(self, values):
        r"""Sum covariance function evaluated at ``(f_0, f_1, ...)``.

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
        r"""Sum of covariance function derivatives.

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
            :math:`f_0' + f_1' + \dots`
        """
        grad = dict()
        for (gn, gv) in iter(gradients.items()):
            for n, v in iter(gv.items()):
                grad[gn + "." + n] = v
        return grad

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(covariances=...)".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        for c in self._covariances:
            spl = str(c).split("\n")
            msg = msg + "\n" + "\n".join(["  " + s for s in spl])
        return msg
