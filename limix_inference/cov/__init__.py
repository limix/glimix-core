"""
********************
Covariance functions
********************

- :class:`.FreeFormCov`
- :class:`.GivenCov`
- :class:`.EyeCov`
- :class:`.LinearCov`
- :class:`.SumCov`

Free-form covariance
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FreeFormCov
  :members:

Given covariance
^^^^^^^^^^^^^^^^

.. autoclass:: GivenCov
  :members:

Identity covariance
^^^^^^^^^^^^^^^^^^^

.. autoclass:: EyeCov
  :members:

Linear covariance
^^^^^^^^^^^^^^^^^

.. autoclass:: LinearCov
  :members:

Sum covariance
^^^^^^^^^^^^^^

.. autoclass:: SumCov
  :members:
"""

from .eye import EyeCov
from .free import FreeFormCov
from .given import GivenCov
from .linear import LinearCov
from .sum import SumCov

__all__ = ['GivenCov', 'LinearCov', 'SumCov', 'EyeCov', 'FreeFormCov']
