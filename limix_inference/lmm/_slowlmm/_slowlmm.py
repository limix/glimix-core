from __future__ import division

from numpy import log
from numpy import pi
from numpy import var
from numpy.linalg import solve
from numpy.linalg import slogdet

from optimix import Composite


class SlowLMM(Composite):
    r"""General Linear Mixed Models.

    Models

    .. math::

        \mathbf y \sim \mathcal N\big(~ \mathbf m;~ \mathrm K ~\big)

    for any mean :math:`\mathbf m` and covariance matrix :math:`\mathrm K`.

    Args:
        y (array_like): real-valued outcome.
        mean (mean_function): mean function defined. (Refer to :doc:`mean`.)
        cov (covariance_function): covariance function defined. (Refer to :doc:`cov`.)
    """
    def __init__(self, y, mean, cov):
        if var(y) < 1e-8:
            raise ValueError("The phenotype variance is too low: %e." % var(y))

        super(SlowLMM, self).__init__(mean=mean, cov=cov)
        self._y = y
        self._cov = cov
        self._mean = mean

    def _lml_gradient_mean(self, mean, cov, gmean):

        Kiym = solve(cov, self._y - mean)

        g = []
        for i in range(len(gmean)):
            g.append(gmean[i].T.dot(Kiym))
        return g

    def _lml_gradient_cov(self, mean, cov, gcov):

        Kiym = solve(cov, self._y - mean)

        g = []
        for i in range(len(gcov)):
            g.append(-solve(cov, gcov[i]).diagonal().sum() + Kiym.dot(gcov[
                i].dot(Kiym)))
        return [gi / 2 for gi in g]

    def value(self, mean, cov):
        ym = self._y - mean
        Kiym = solve(cov, ym)

        (s, logdet) = slogdet(cov)
        assert s == 1.

        n = len(self._y)
        return -(logdet + ym.dot(Kiym) + n * log(2 * pi)) / 2

    def gradient(self, mean, cov, gmean, gcov):
        grad_cov = self._lml_gradient_cov(mean, cov, gcov)
        grad_mean = self._lml_gradient_mean(mean, cov, gmean)
        return grad_cov + grad_mean
