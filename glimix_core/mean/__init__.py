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

from .linear import LinearMean
from .offset import OffsetMean
from .sum import SumMean

__all__ = ['OffsetMean', 'LinearMean', 'SumMean']
