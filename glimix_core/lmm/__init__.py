r"""
*******************
Linear Mixed Models
*******************

.. _lmm-intro:

Introduction
^^^^^^^^^^^^

A LMM can be described as

.. math::
    :label: lmm1

    \mathbf y = \mathrm X\boldsymbol\beta + \mathrm G\mathbf u
    + \boldsymbol\epsilon,

where :math:`\mathbf u \sim \mathcal N(\mathbf 0, v_0\mathrm I)` is a
vector of random effects and :math:`\epsilon_i` are iid Normal random variables
with zero-mean and variance :math:`v_1` each.
The outcome-vector is thus distributed according to

.. math::
    :label: lmm2

    \mathbf y \sim \mathcal N(\mathrm X\boldsymbol\beta,
                              v_0 \mathrm G \mathrm G^{\intercal}
                              + v_1\mathrm I)


We refer to :math:`\mathrm G \mathrm G^{\intercal}` simply as
:math:`\mathrm K` in some occasions for didactic purpose.
This module provides :class:`.LMM`, a FastLMM [Lip11]_
implementation to perform inference over the variance parameters
:math:`v_0` and :math:`v_1` and over the vector
:math:`\boldsymbol\beta` of fixed-effect sizes.
:class:`.FastScanner` is typically used for performing an even faster inference
across several (millions, for example) covariates independently.
This is achieved by not fitting the ratio between the overall variance of ``K``
and ``I``.

LMM class
^^^^^^^^^

This is the class that the user will be using to perform LMM inference.
The base model was already described in the :ref:`lmm-intro`.

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
