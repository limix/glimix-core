"""Mean functions.

LinearMean  Dot-product mean function.
OffsetMean  Offset mean.
SumMean     Sum of multiple mean functions.
"""
from .linear import LinearMean
from .offset import OffsetMean
from .sum import SumMean

__all__ = ["OffsetMean", "LinearMean", "SumMean"]
