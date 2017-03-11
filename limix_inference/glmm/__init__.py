r"""
*******************************
Generalized Linear Mixed Models
*******************************

Introduction
^^^^^^^^^^^^

A GLMM can be described in two parts.
The first part consists in latent effects:

.. math::

    \mathbf z = \mathbf m + \mathbf u + \boldsymbol\epsilon

where :math:`\mathbf u \sim \mathcal N(\mathbf 0, \mathrm K)` and
:math:`\epsilon_i` are iid Normal random variables.
The second part connects the latent variable to the observed one:

.. math::

    y_i ~|~ z_i \sim \text{ExpFam}(y_i ~|~ \mu_i = g(z_i))

where :math:`g(\cdot)` is a link function and :math:`\text{ExpFam}(\cdot)` is
an exponential-family distribution.

This module implements the Expectation Propagation algorithm:
:class:`.ExpFamEP`.

Public interface
^^^^^^^^^^^^^^^^
"""

from .ep import ExpFamEP

__all__ = ['ExpFamEP']
