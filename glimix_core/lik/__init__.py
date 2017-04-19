"""
***********
Likelihoods
***********

- Likelihood products: :class:`.DeltaProdLik`, :class:`.BernoulliProdLik`,
  :class:`.BinomialProdLik`, and :class:`.PoissonProdLik`.

Likelihood products
^^^^^^^^^^^^^^^^^^^

.. autoclass:: DeltaProdLik
  :members:

.. autoclass:: BernoulliProdLik
  :members:

.. autoclass:: BinomialProdLik
  :members:

.. autoclass:: PoissonProdLik
  :members:

"""

from .prod import (BernoulliProdLik, BinomialProdLik, DeltaProdLik,
                   PoissonProdLik)

__all__ = [
    'DeltaProdLik', 'BernoulliProdLik', 'BinomialProdLik', 'PoissonProdLik'
]
