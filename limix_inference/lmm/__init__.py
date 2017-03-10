"""
*******************
Linear Mixed Models
*******************

We have two implementations: :class:`.SlowLMM` and :class:`.FastLMM`.
"""

from .fastlmm import FastLMM, NormalLikTrick
from .slowlmm import SlowLMM

__all__ = ['SlowLMM', 'FastLMM', 'NormalLikTrick']
