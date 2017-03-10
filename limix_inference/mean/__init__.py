"""
**************
Mean functions
**************

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

from .offset import OffsetMean
from .linear import LinearMean
from .sum import SumMean
from .kron import KronSumMean
