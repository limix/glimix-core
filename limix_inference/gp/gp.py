from __future__ import division

from numpy import log
from numpy import pi
from numpy import var
from numpy.linalg import solve
from numpy.linalg import slogdet

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
        >>> from limix_inference.example import offset_mean
        >>> from limix_inference.example import linear_eye_cov
        >>> from limix_inference.gp import GP
        >>> from limix_inference.random import GPSampler
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
        Before: -164.0064
        >>> gp.feed().maximize(progress=False)
        >>> print('After: %.4f' % gp.feed().value())
        After: -163.4192
    """
    def __init__(self, y, mean, cov):
        if var(y) < 1e-8:
            raise ValueError("The phenotype variance is too low: %e." % var(y))

        super(GP, self).__init__([mean, cov], name='GP')
        self._y = y
        self._cov = cov
        self._mean = mean

    def _lml_gradient_mean(self, mean, cov, gmean):
        Kiym = solve(cov, self._y - mean)
        return gmean.T.dot(Kiym)

    def _lml_gradient_cov(self, mean, cov, gcov):
        Kiym = solve(cov, self._y - mean)
        return -solve(cov, gcov).diagonal().sum() + Kiym.dot(gcov.dot(Kiym))/2

    def value_reduce(self, values): # pylint: disable=R0201
        mean = values['GP[0]']
        cov = values['GP[1]']
        ym = self._y - mean
        Kiym = solve(cov, ym)

        (s, logdet) = slogdet(cov)
        assert s == 1.

        n = len(self._y)
        return -(logdet + ym.dot(Kiym) + n * log(2 * pi)) / 2

    # def value(self, mean, cov):
    #     import pdb; pdb.set_trace()
    #     ym = self._y - mean
    #     Kiym = solve(cov, ym)
    #
    #     (s, logdet) = slogdet(cov)
    #     assert s == 1.
    #
    #     n = len(self._y)
    #     return -(logdet + ym.dot(Kiym) + n * log(2 * pi)) / 2

    # # pylint: disable=W0221
    # def gradient(self, mean, cov, gmean, gcov):
    #     import pdb; pdb.set_trace()
    #     grad_cov = self._lml_gradient_cov(mean, cov, gcov)
    #     grad_mean = self._lml_gradient_mean(mean, cov, gmean)
    #     return grad_cov + grad_mean

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
