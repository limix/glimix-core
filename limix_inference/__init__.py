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
        return_code = __import__('pytest').main(['-q'])
    finally:
        os.chdir(old_path)

    if return_code == 0:
        print("Congratulations. All tests have passed!")

    return return_code
