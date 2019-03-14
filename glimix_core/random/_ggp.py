from numpy.random import RandomState


class GGPSampler(object):
    r"""Sample from a Generalised Gaussian Process.

    Outcome modelled via

    .. math::

        \mathbf y \sim \int \prod_i \mathrm{ExpFam}(y_i ~|~ g_i(z_i))
        \mathcal N(\mathbf z ~|~ \mathbf m; \mathrm K) \mathrm d\mathbf z.

    Parameters
    ----------
    link : str
        Likelihood name.
    mean : function
        Mean function.
    cov : function
        Covariance function.

    Example
    -------

    .. doctest::

        >>> from numpy.random import RandomState
        >>>
        >>> from glimix_core.example import offset_mean
        >>> from glimix_core.example import linear_eye_cov
        >>> from glimix_core.random import GGPSampler
        >>> from glimix_core.lik import DeltaProdLik
        >>>
        >>> random = RandomState(1)
        >>>
        >>> mean = offset_mean()
        >>> cov = linear_eye_cov()
        >>>
        >>> lik = DeltaProdLik()
        >>>
        >>> y = GGPSampler(lik, mean, cov).sample(random)
        >>> print(y[:5])  # doctest: +FLOAT_CMP
        [-2.42181498  0.50720447 -1.01053967  0.736624    1.64019063]
    """

    def __init__(self, lik, mean, cov):
        self._lik = lik
        self._mean = mean
        self._cov = cov

    def sample(self, random_state=None):
        r"""Sample from the specified distribution.

        Parameters
        ----------
        random_state : random_state
            Set the initial random state.

        Returns
        -------
        numpy.ndarray
            Sample.
        """
        from numpy_sugar import epsilon
        from numpy_sugar.linalg import sum2diag
        from numpy_sugar.random import multivariate_normal

        if random_state is None:
            random_state = RandomState()

        m = self._mean.value()
        K = self._cov.value().copy()

        sum2diag(K, +epsilon.small, out=K)

        return self._lik.sample(multivariate_normal(m, K, random_state), random_state)
