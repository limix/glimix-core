r"""
*******************
Linear Mixed Models
*******************

Introduction
^^^^^^^^^^^^

A LMM can be described as

.. math::

    \mathbf y = \mathrm X\boldsymbol\beta + \mathrm G\mathbf u
    + \boldsymbol\epsilon,

where :math:`\mathbf u \sim \mathcal N(\mathbf 0, \sigma_r^2\mathrm I)` is a
vector of random effects and :math:`\epsilon_i` are iid Normal random variables
with zero-mean and variance :math:`\sigma_e^2` each.
The outcome-vector is thus distributed according to

.. math::

    \mathbf y \sim \mathcal N(\mathrm X\boldsymbol\beta,
                              \sigma_r^2 \mathrm G \mathrm G^{\intercal}
                              + \sigma_e^2\mathrm I)

This module provides :class:`.LMM`, a FastLMM implementation to perform
inference.
:class:`.FastScanner` can be used to perform fast inference over multiple
covariates.

Public interface
^^^^^^^^^^^^^^^^
"""

from .lmm import LMM
from .scan import FastScanner

__all__ = ['LMM', 'FastScanner']
