from __future__ import division

from numpy import log, pi
from numpy.linalg import slogdet, solve
from numpy_sugar import is_all_finite
from optimix import FunctionReduce


class GP(FunctionReduce):
    r"""Gaussian Process inference via maximum likelihood.

    Args:
        y (array_like): outcome variable.
        mean (:class:`optimix.Function`): mean function.
                                          (Refer to :doc:`mean`.)
        cov (:class:`optimix.Function`): covariance function.
                                         (Refer to :doc:`cov`.)

    Example
    -------

    .. doctest::

        >>> from numpy.random import RandomState
        >>>
        >>> from glimix_core.example import offset_mean
        >>> from glimix_core.example import linear_eye_cov
        >>> from glimix_core.gp import GP
        >>> from glimix_core.random import GPSampler
        >>>
        >>> random = RandomState(94584)
        >>>
        >>> mean = offset_mean()
        >>> cov = linear_eye_cov()
        >>>
        >>> y = GPSampler(mean, cov).sample(random)
        >>>
        >>> gp = GP(y, mean, cov)
        >>> print('Before: %.4f' % gp.feed().value())
        Before: -15.5582
        >>> gp.feed().maximize(verbose=False)
        >>> print('After: %.4f' % gp.feed().value())
        After: -13.4791
    """

    def __init__(self, y, mean, cov):
        super(GP, self).__init__([mean, cov], name='GP')

        if not is_all_finite(y):
            raise ValueError("There are non-finite values in the phenotype.")

        self._y = y
        self._cov = cov
        self._mean = mean

    def _lml_gradient_mean(self, mean, cov, gmean):
        Kiym = solve(cov, self._y - mean)
        return gmean.T.dot(Kiym)

    def _lml_gradient_cov(self, mean, cov, gcov):
        Kiym = solve(cov, self._y - mean)
        return (
            -solve(cov, gcov).diagonal().sum() + Kiym.dot(gcov.dot(Kiym))) / 2

    def value_reduce(self, values):  # pylint: disable=R0201
        mean = values['GP[0]']
        cov = values['GP[1]']
        ym = self._y - mean
        Kiym = solve(cov, ym)

        (s, logdet) = slogdet(cov)
        if not s == 1.0:
            raise RuntimeError("This determinant should not be negative.")

        n = len(self._y)
        return -(logdet + ym.dot(Kiym) + n * log(2 * pi)) / 2

    def gradient_reduce(self, values, gradients):
        mean = values['GP[0]']
        cov = values['GP[1]']
        gmean = gradients['GP[0]']
        gcov = gradients['GP[1]']

        grad = dict()
        for n, g in iter(gmean.items()):
            grad['GP[0].' + n] = self._lml_gradient_mean(mean, cov, g)

        for n, g in iter(gcov.items()):
            grad['GP[1].' + n] = self._lml_gradient_cov(mean, cov, g)

        return grad
