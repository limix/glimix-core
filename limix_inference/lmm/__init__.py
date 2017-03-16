r"""
*******************
Linear Mixed Models
*******************

Introduction
^^^^^^^^^^^^

A LMM can be described as

.. math::

    \mathbf y = \mathrm X\boldsymbol\beta + \mathrm Z\mathbf u
    + \boldsymbol\epsilon,

where :math:`\mathbf u \sim \mathcal N(\mathbf 0, \mathrm G)` is a vector of
random effects and :math:`\epsilon_i` are iid Normal random variables with
zero-mean and variance :math:`\sigma_{\epsilon}^2` each.
The outcome-vector is thus distributed according to

.. math::

    \mathbf y \sim \mathcal N(\mathrm X\boldsymbol\beta,
                              \mathrm Z \mathrm G \mathrm Z^{\intercal}
                              + \sigma_{\epsilon}^2\mathrm I)

This module provides two methods for fitting LMMs via maximum
likelihood: :class:`.SlowLMM` and :class:`.LMM`.
The former is more general but slower than the latter;
the latter assumes a scaled covariance matrix and linear fixed-effects.

:class:`.FastScanner` can be used to perform fast inference over multiple
covariates;
it is meant to perform such inference over millions of covariates in seconds.

Public interface
^^^^^^^^^^^^^^^^
"""

from .lmm import LMM
from .scan import FastScanner

__all__ = ['LMM', 'FastScanner']
