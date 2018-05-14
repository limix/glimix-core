"""Covariance functions.

EyeCov       Indentity covariance function.
FreeFormCov  Free-form covariance based on Cholesky decomposition.
GivenCov     Fixed covariance function with scaling factor.
LinearCov    Dot-product covariance function.
SumCov       Sum of multiple covariance functions.
"""
from .eye import EyeCov
from .free import FreeFormCov
from .given import GivenCov
from .linear import LinearCov
from .sum import SumCov

__all__ = ['EyeCov', 'FreeFormCov', 'GivenCov', 'LinearCov', 'SumCov']
