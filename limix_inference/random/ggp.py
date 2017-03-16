from __future__ import division

from numpy.random import RandomState
from numpy_sugar.linalg import sum2diag
from numpy_sugar import epsilon
from numpy_sugar.random import multivariate_normal


class GGPSampler(object): # pylint: disable=R0903
    r"""Sample from a Generalised Gaussian Process.

    Sample from

    .. math::

        \mathbf y \sim \int \prod_i \mathrm{ExpFam}(y_i ~|~ g_i(z_i))
        \mathcal N(\mathbf z ~|~ \mathbf m; \mathrm K) \mathrm d\mathbf z.

    Args:
        lik (object): likelihood product. (Refer to :doc:`lik`.)
        mean (:class:`optimix.Function`): mean function.
                                          (Refer to :doc:`mean`.)
        cov (:class:`optimix.Function`): covariance function.
                                         (Refer to :doc:`cov`.)

    Example
    -------

    .. doctest::

        >>> from numpy import arange, sqrt
        >>> from numpy.random import RandomState
        >>>
        >>> from limix_inference.random import GGPSampler
        >>> from limix_inference.mean import OffsetMean
        >>> from limix_inference.cov import LinearCov, EyeCov, SumCov
        >>> from limix_inference.lik import DeltaProdLik
        >>>
        >>> random = RandomState(9458)
        >>> N = 500
        >>> X = random.randn(N, N + 1)
        >>> X -= X.mean(0)
        >>> X /= X.std(0)
        >>> X /= sqrt(X.shape[1])
        >>> offset = 1.0
        >>>
        >>> mean = OffsetMean()
        >>> mean.offset = offset
        >>> mean.set_data(N, purpose='sample')
        >>>
        >>> cov_left = LinearCov()
        >>> cov_left.scale = 1.5
        >>> cov_left.set_data((X, X), purpose='sample')
        >>>
        >>> cov_right = EyeCov()
        >>> cov_right.scale = 1.5
        >>> cov_right.set_data((arange(N), arange(N)), purpose='sample')
        >>>
        >>> cov = SumCov([cov_left, cov_right])
        >>>
        >>> lik = DeltaProdLik()
        >>>
        >>> y = GGPSampler(lik, mean, cov).sample(random)
        >>> print(y[:5])
        [ 2.17393302  0.27067607 -1.08349329  1.32031279  2.15242283]
    """
    def __init__(self, lik, mean, cov):
        self._lik = lik
        self._mean = mean
        self._cov = cov

    def sample(self, random_state=None):
        if random_state is None:
            random_state = RandomState()

        m = self._mean.feed('sample').value()
        K = self._cov.feed('sample').value().copy()

        sum2diag(K, +epsilon.small, out=K)

        return self._lik.sample(multivariate_normal(m, K, random_state),
                                random_state)
