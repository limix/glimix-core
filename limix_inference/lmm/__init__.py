"""
*******************
Linear Mixed Models
*******************

- :class:`.FastLMM`
- :class:`.SlowLMM`

Fast implementation
^^^^^^^^^^^^^^^^^^^

.. autoclass:: FastLMM
  :members:

General implementation
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SlowLMM
  :members:
"""

from ._fastlmm import FastLMM, fast_scan
from ._slowlmm import SlowLMM
