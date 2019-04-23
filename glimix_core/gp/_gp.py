from numpy import log, pi
from numpy.linalg import slogdet, solve
from optimix import Function


class GP(Function):
    r"""Gaussian Process inference via maximum likelihood.

    Parameters
    ----------
    y : array_like
        Outcome variable.
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
        >>> print('Before: %.4f' % gp.lml())
        Before: -15.5582
        >>> gp.fit(verbose=False)
        >>> print('After: %.4f' % gp.lml())
        After: -13.4791
        >>> print(gp)  # doctest: +FLOAT_CMP
        GP(...)
          lml: -13.47907874997517
          OffsetMean(): OffsetMean
            offset: 0.7755803668772308
          SumCov(covariances=...): SumCov
            LinearCov(): LinearCov
              scale: 2.061153622438558e-09
            EyeCov(dim=10): EyeCov
              scale: 0.8675680523425126
    """

    def __init__(self, y, mean, cov):
        from numpy_sugar import is_all_finite

        super(GP, self).__init__("GP", composite=[mean, cov])

        if not is_all_finite(y):
            raise ValueError("There are non-finite values in the phenotype.")

        self._y = y
        self._cov = cov
        self._mean = mean

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
        self._maximize(verbose=verbose, factr=factr, pgtol=pgtol)

    def lml(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            :math:`\log p(\mathbf y)`
        """
        return self.value()

    def _lml_gradient_mean(self, mean, cov, gmean):
        Kiym = solve(cov, self._y - mean)
        return gmean.T.dot(Kiym)

    def _lml_gradient_cov(self, mean, cov, gcov):
        Kiym = solve(cov, self._y - mean)
        return (-solve(cov, gcov).diagonal().sum() + Kiym.dot(gcov.dot(Kiym))) / 2

    def value(self):
        mean = self._mean.value()
        cov = self._cov.value()
        ym = self._y - mean
        Kiym = solve(cov, ym)

        (s, logdet) = slogdet(cov)
        if not s == 1.0:
            raise RuntimeError("This determinant should not be negative.")

        n = len(self._y)
        return -(logdet + ym.dot(Kiym) + n * log(2 * pi)) / 2

    def gradient(self):
        mean = self._mean.value()
        cov = self._cov.value()
        gmean = self._mean.gradient()
        gcov = self._cov.gradient()

        grad = dict()
        for n, g in iter(gmean.items()):
            grad["GP[0]." + n] = self._lml_gradient_mean(mean, cov, g)

        for n, g in iter(gcov.items()):
            grad["GP[1]." + n] = self._lml_gradient_cov(mean, cov, g)

        return grad

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(...)\n".format(tname)
        msg += "  lml: {}\n".format(self.lml())

        mmsg = str(self._mean).split("\n")
        mmsg = "\n".join(["  " + m for m in mmsg])

        cmsg = str(self._cov).split("\n")
        cmsg = "\n".join(["  " + m for m in cmsg])

        return msg + mmsg + "\n" + cmsg
