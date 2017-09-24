from __future__ import division

from numpy.random import RandomState
from numpy_sugar import epsilon
from numpy_sugar.linalg import sum2diag
from numpy_sugar.random import multivariate_normal


class GPSampler(object):  # pylint: disable=R0903
    r"""Sample from a Gaussian Process.

    Outcome modelled via

    .. math::

        \mathbf y \sim \mathcal N(\mathbf z ~|~ \mathbf m, \mathrm K)
        \mathrm d\mathbf z.

    Args:
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
        >>> from glimix_core.random import GPSampler
        >>>
        >>> random = RandomState(1)
        >>>
        >>> mean = offset_mean()
        >>> cov = linear_eye_cov()
        >>>
        >>> y = GPSampler(mean, cov).sample(random)
        >>> print(y[:5])
        [-2.42181498  0.50720447 -1.01053967  0.736624    1.64019063]
    """

    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov

    def sample(self, random_state=None):
        if random_state is None:
            random_state = RandomState()

        m = self._mean.feed('sample').value()
        K = self._cov.feed('sample').value().copy()

        sum2diag(K, +epsilon.small, out=K)

        return multivariate_normal(m, K, random_state)
