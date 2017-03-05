"""
*******************
Linear Mixed Models
*******************

- :class:`.FastLMM`
- :class:`.SlowLMM`
- :func:`.fast_scan`

Fast implementation
^^^^^^^^^^^^^^^^^^^

.. autoclass:: FastLMM
  :members:

.. autofunction:: fast_scan

General implementation
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SlowLMM
  :members:
"""

from ._fastlmm import FastLMM, fast_scan
from ._slowlmm import SlowLMM
