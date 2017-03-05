"""
*******************
Linear Mixed Models
*******************

- :class:`.FastLMM`
- :class:`.SlowLMM`
- :class:`.NormalLikTrick`

Fast implementation
^^^^^^^^^^^^^^^^^^^

.. autoclass:: FastLMM
  :members:

.. autoclass:: NormalLikTrick
  :members:

General implementation
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SlowLMM
  :members:
"""

from ._fastlmm import FastLMM, NormalLikTrick
from ._slowlmm import SlowLMM
