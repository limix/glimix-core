"""
**************
Random sampler
**************

- :func:`.bernoulli_sample`
- :func:`.binomial_sample`
- :func:`.poisson_sample`

Bernoulli
^^^^^^^^^

.. autofunction:: bernoulli_sample

Example:

.. doctest::

    >>> from limix_inference.random import bernoulli_sample
    >>> from numpy.random import RandomState
    >>> offset = 5
    >>> G = [[1, -1], [2, 1]]
    >>> bernoulli_sample(offset, G, random_state=RandomState(0))
    array([1, 1])

Binomial
^^^^^^^^

.. autofunction:: binomial_sample

Example:

.. doctest::

    >>> from limix_inference.random import binomial_sample
    >>> from numpy.random import RandomState
    >>> ntrials = [5, 15]
    >>> offset = 0.5
    >>> G = [[1, -1], [2, 1]]
    >>> binomial_sample(ntrials, offset, G, random_state=RandomState(0))
    array([ 2, 14])

Poisson
^^^^^^^

.. autofunction:: poisson_sample

Example:

.. doctest::

    >>> from limix_inference.random import poisson_sample
    >>> from numpy.random import RandomState
    >>> offset = -0.5
    >>> G = [[0.5, -1], [2, 1]]
    >>> poisson_sample(offset, G, random_state=RandomState(0))
    array([0, 6])

"""

from ._glmm import GLMMSampler
from ._canonical import bernoulli_sample, binomial_sample, poisson_sample
