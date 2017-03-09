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

from .fastlmm import FastLMM, NormalLikTrick
from .slowlmm import SlowLMM

__all__ = ['SlowLMM', 'FastLMM', 'NormalLikTrick']
