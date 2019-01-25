from __future__ import absolute_import, division, unicode_literals

import warnings

from liknorm import LikNormMachine
from numpy import ascontiguousarray, sign
from numpy.linalg import LinAlgError

from optimix import FunctionReduce

from ..ep import EP
from ..util import check_outcome


class ExpFamGP(FunctionReduce):
    r"""Expectation Propagation for Generalised Gaussian Processes.

    Parameters
    ----------
    y : array_like
        Outcome variable.
    lik_name : str
        Likelihood name.
    mean : function
        Mean function. (Refer to :doc:`mean`.)
    cov : function
        Covariance function. (Refer to :doc:`cov`.)

    Example
    -------

    .. doctest::

        >>> from numpy.random import RandomState
        >>>
        >>> from glimix_core.example import offset_mean
        >>> from glimix_core.example import linear_eye_cov
        >>> from glimix_core.ggp import ExpFamGP
        >>> from glimix_core.lik import BernoulliProdLik
        >>> from glimix_core.link import LogitLink
        >>> from glimix_core.random import GGPSampler
        >>>
        >>> random = RandomState(1)
        >>>
        >>> lik = BernoulliProdLik(LogitLink())
        >>> mean = offset_mean()
        >>> cov = linear_eye_cov()
        >>>
        >>> y = GGPSampler(lik, mean, cov).sample(random)
        >>>
        >>> ggp = ExpFamGP(y, 'bernoulli', mean, cov)
        >>> print('Before: %.4f' % ggp.lml())
        Before: -6.9802
        >>> ggp.fit(verbose=False)
        >>> print('After: %.2f' % ggp.lml())
        After: -6.55
    """

    def __init__(self, y, lik, mean, cov):
        if isinstance(y, tuple):
            n = len(y[0])
        else:
            n = len(y)
        FunctionReduce.__init__(self, [mean, cov], name="ExpFamGP")

        if not isinstance(lik, (tuple, list)):
            lik = (lik,)

        self._lik = (lik[0].lower(),) + tuple(ascontiguousarray(i) for i in lik[1:])
        self._y = check_outcome(y, self._lik)

        self._mean = mean
        self._cov = cov
        self._ep = EP(n)
        self._ep.set_compute_moments(self.compute_moments)

        self._mean_value = None
        self._cov_value = None

        self._machine = LikNormMachine(lik[0], 500)

    def __del__(self):
        if hasattr(self, "_machine"):
            self._machine.finish()

    def fit(self, verbose=True, factr=1e5, pgtol=1e-7):
        r"""Maximise the marginal likelihood.

        Parameters
        ----------
        verbose : bool
            ``True`` for progress output; ``False`` otherwise.
            Defaults to ``True``.
        factr : float, optional
            The iteration stops when
            ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``, where ``eps`` is
            the machine precision.
        pgtol : float, optional
            The iteration will stop when ``max{|proj g_i | i = 1, ..., n} <= pgtol``
            where ``pg_i`` is the i-th component of the projected gradient.

        Notes
        -----
        Please, refer to :func:`scipy.optimize.fmin_l_bfgs_b` for further information
        about ``factr`` and ``pgtol``.
        """
        self.feed().maximize(verbose=verbose, factr=factr, pgtol=pgtol)

    def lml(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            :math:`\log p(\mathbf y)`
        """
        return self.feed().value()

    def compute_moments(self, eta, tau, moments):
        y = (self._y,) + self._lik[1:]
        self._machine.moments(y, eta, tau, moments)

    def value_reduce(self, values):
        from numpy_sugar import epsilon
        from numpy_sugar.linalg import economic_qs

        mean = values["ExpFamGP[0]"]
        cov = values["ExpFamGP[1]"]
        try:
            self._ep.set_prior(mean, dict(QS=economic_qs(cov)))
            lml = self._ep.lml()
        except (ValueError, LinAlgError) as e:
            warnings.warn(str(e), RuntimeWarning)
            lml = -1 / epsilon.small
        return lml

    def gradient_reduce(self, values, gradients):
        from numpy_sugar import epsilon
        from numpy_sugar.linalg import economic_qs

        mean = values["ExpFamGP[0]"]
        cov = values["ExpFamGP[1]"]
        gmean = gradients["ExpFamGP[0]"]
        gcov = gradients["ExpFamGP[1]"]

        try:
            self._ep.set_prior(mean, dict(QS=economic_qs(cov)))

            grad = dict()

            for n, g in iter(gmean.items()):
                grad["ExpFamGP[0]." + n] = self._ep.lml_derivative_over_mean(g)

            for n, g in iter(gcov.items()):
                QS = economic_qs(g)
                grad["ExpFamGP[1]." + n] = self._ep.lml_derivative_over_cov(QS)

            return grad
        except (ValueError, LinAlgError) as e:
            warnings.warn(str(e), RuntimeWarning)
            v = self.variables().select(fixed=False)
            return {i: -sign(v.get(i).value) / epsilon.small for i in v}
