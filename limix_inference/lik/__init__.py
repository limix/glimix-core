"""
***********
Likelihoods
***********

- Single likelihoods: :class:`.DeltaLik`, :class:`.BernoulliLik`,
  :class:`.BinomialLik`, and :class:`.PoissonLik`.
- Likelihood products: :class:`.DeltaProdLik`, :class:`.BernoulliProdLik`,
  :class:`.BinomialProdLik`, and :class:`.PoissonProdLik`.

Single likelihoods
^^^^^^^^^^^^^^^^^^

.. autoclass:: DeltaLik
  :members:

.. autoclass:: BernoulliLik
  :members:

.. autoclass:: BinomialLik
  :members:

.. autoclass:: PoissonLik
  :members:

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

from ._expfam import (DeltaLik, BernoulliLik, BinomialLik, PoissonLik)
from ._prod import (DeltaProdLik, BernoulliProdLik, BinomialProdLik,
                    PoissonProdLik)

__all__ = [
    'DeltaLik', 'BernoulliLik', 'BinomialLik', 'PoissonLik', 'DeltaProdLik',
    'BernoulliProdLik', 'BinomialProdLik', 'PoissonProdLik'
]
