"""
Generalized Linear Mixed Models
-------------------------------

Example:


    import numpy as np

.. doctest::

    >>> from limix_inference.random import bernoulli_sample
    >>> from numpy.random import RandomState
    >>> offset = 5
    >>> G = [[1, -1], [2, 1]]
    >>> bernoulli_sample(offset, G, random_state=RandomState(0))
    array([1, 1])

ExpFamEP
^^^^^^^^

.. autoclass:: ExpFamEP
    :members: covariates_variance, genetic_variance, environmental_variance,
              heritability, K, m, beta, v, delta, lml, optimize, fixed_ep
"""
from __future__ import absolute_import

from ._expfam import ExpFamEP
