"""
********************
Covariance functions
********************

- :class:`.LinearCov`
- :class:`.SumCov`
- :class:`.EyeCov`

Linear covariance
^^^^^^^^^^^^^^^^^

.. autoclass:: LinearCov
  :members:

Sum covariance
^^^^^^^^^^^^^^

.. autoclass:: SumCov
  :members:

Identity covariance
^^^^^^^^^^^^^^^^^^^

.. autoclass:: EyeCov
  :members:
"""

from ._linear import LinearCov
from ._sum import SumCov
from ._eye import EyeCov
