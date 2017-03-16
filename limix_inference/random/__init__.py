"""
**************
Random sampler
**************

Introduction
^^^^^^^^^^^^

Helper classes and functions for sampling: :class:`.GGPSampler`,
:class:`.GPSampler`.

Public interface
^^^^^^^^^^^^^^^^
"""

from .ggp import GGPSampler
from .gp import GPSampler
from .canonical import bernoulli_sample, binomial_sample, poisson_sample

__all__ = ['GGPSampler', 'GPSampler', 'bernoulli_sample', 'binomial_sample',
           'poisson_sample']
