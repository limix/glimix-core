"""
*******************************
Generalized Linear Mixed Models
*******************************

Example
^^^^^^^

.. doctest::

    >>> from limix_inference.random import bernoulli_sample
    >>> from limix_inference.glmm import ExpFamEP
    >>> from limix_inference.lik import BernoulliProdLik
    >>> from limix_inference.link import LogLink
    >>> from numpy_sugar.linalg import economic_qs_linear
    >>> from numpy.random import RandomState
    >>>
    >>> offset = 0.2
    >>> random = RandomState(0)
    >>> G = random.randn(100, 200)
    >>> QS = economic_qs_linear(G)
    >>> y = bernoulli_sample(offset, G, random_state=random)
    >>> covariates = random.randn(100, 1)
    >>> lik = BernoulliProdLik(LogLink)
    >>> lik.outcome = y
    >>> glmm = ExpFamEP(lik, covariates, QS)
    >>> glmm.learn(progress=False)
    >>> '%.2f' % glmm.lml()
    '-69.06'

Expectation propagation
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ExpFamEP
    :members: covariates_variance, genetic_variance, environmental_variance,
              heritability, K, m, beta, v, delta, lml, learn, fixed_ep
"""

from .ep import ExpFamEP

__all__ = ['ExpFamEP']
