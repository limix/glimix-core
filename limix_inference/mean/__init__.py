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

from ._offset import OffsetMean
from ._linear import LinearMean
from ._sum import SumMean
