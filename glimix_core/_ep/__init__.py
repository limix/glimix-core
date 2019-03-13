"""
Expectation Propagation
=======================

Introduction
------------

This module implements the building-blocks for EP inference: EP parameter
fitting, log of the marginal likelihood, and derivative of the log of the
marginal likelihood.
"""

from .ep import EP
from .linear_kernel import EPLinearKernel

__all__ = ["EP", "EPLinearKernel"]
