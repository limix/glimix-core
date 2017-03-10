"""
***********************
limix_inference package
***********************

Example:

.. doctest::

    >>> from numpy import arange, sqrt
    >>> from numpy.random import RandomState
    >>> from limix_inference.random import GLMMSampler
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
    >>> y = GLMMSampler(lik, mean, cov).sample(random)
    >>> print(y[:5])

"""
from __future__ import absolute_import as _absolute_import

from . import lmm
from . import glmm
from . import cov
from . import lik
from . import mean
from . import link
from . import random

from pkg_resources import get_distribution as _get_distribution
from pkg_resources import DistributionNotFound as _DistributionNotFound

try:
    __version__ = _get_distribution('limix_inference').version
except _DistributionNotFound:
    __version__ = 'unknown'


def test():
    import os
    p = __import__('limix_inference').__path__[0]
    src_path = os.path.abspath(p)
    old_path = os.getcwd()
    os.chdir(src_path)

    try:
        return_code = __import__('pytest').main(['-q', '--doctest-modules'])
    finally:
        os.chdir(old_path)

    if return_code == 0:
        print("Congratulations. All tests have passed!")

    return return_code
