"""
********************
Covariance functions
********************

- :class:`.LinearCov`
- :class:`.SumCov`
- :class:`.EyeCov`
- :class:`.FreeFormCov`

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

Free-form covariance
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FreeFormCov
  :members:
"""

from .linear import LinearCov
from .sum import SumCov
from .eye import EyeCov
from .free import FreeFormCov

__all__ = ['LinearCov', 'SumCov', 'EyeCov', 'FreeFormCov']
