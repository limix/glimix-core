"""
***********
Likelihoods
***********

- :class:`.LinearMean`
- :class:`.OffsetMean`
- :class:`.SumMean`

Linear mean
^^^^^^^^^^^

.. autoclass:: LinearMean
  :members:

Offset mean
^^^^^^^^^^^

.. autoclass:: OffsetMean
  :members:

Sum mean
^^^^^^^^

.. autoclass:: SumMean
  :members:
"""

from ._expfam import (DeltaLik, BernoulliLik, BinomialLik, PoissonLik)
from ._prod import (DeltaProdLik, BernoulliProdLik, BinomialProdLik,
                    PoissonProdLik)
