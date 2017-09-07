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

where :math:`\mathbf u \sim \mathcal N(\mathbf 0, \sigma_g^2\mathrm I)` is a
vector of random effects and :math:`\epsilon_i` are iid Normal random variables
with zero-mean and variance :math:`\sigma_e^2` each.
The outcome-vector is thus distributed according to

.. math::

    \mathbf y \sim \mathcal N(\mathrm X\boldsymbol\beta,
                              \sigma_g^2 \mathrm G \mathrm G^{\intercal}
                              + \sigma_e^2\mathrm I)

We refer to :math:`\mathrm G \mathrm G^{\intercal}` simply as :math:`\mathrm K`,
if not otherwise mentioned.
This module provides :class:`.LMM`, a FastLMM :cite:`lippert2011fast`
implementation to perform inference over the variance parameters
:math:`\sigma_g^2` and :math:`\sigma_e2` and over the vector
:math:`\boldsymbol\beta` of fixed-effect sizes.
:class:`.FastScanner` is typically used for performing an even faster inference
across several (millions, for example) covariates independently.
This is achieved by not fitting the ratio between the total variance of ``K``
and ``I``.

.. bibliography:: refs.bib
    :style: unsrt

LMM class
^^^^^^^^^

.. autoclass:: glimix_core.lmm.LMM
  :members:

FastScanner class
^^^^^^^^^^^^^^^^^

.. autoclass:: glimix_core.lmm.FastScanner
  :members:
"""

from .lmm import LMM
from .scan import FastScanner

__all__ = ['LMM', 'FastScanner']
