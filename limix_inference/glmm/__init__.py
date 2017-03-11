r"""
*******************************
Generalized Linear Mixed Models
*******************************

Introduction
^^^^^^^^^^^^

A GLMM can be described in two parts.
The first part consists in the latent variable

.. math::

    \mathbf z = \mathrm X\boldsymbol\beta + \mathrm Z\mathbf u,

where :math:`\mathbf u \sim \mathcal N(\mathbf 0, \mathrm G)` is a vector of
random effects.
The second part connects the latent variable to the observed one:

.. math::

    y_i ~|~ z_i \sim \text{ExpFam}(\mu_i = g(z_i)),

where :math:`g(\cdot)` is a link function and :math:`\text{ExpFam}(\cdot)` is
an exponential-family distribution.
The marginal likelihood is thus given by

.. math::

    p(\mathbf y) = \int
      \prod_i \text{ExpFam}(y_i ~|~ \mu_i = g(z_i))
      \mathcal N(\mathbf z ~|~ \mathrm X\boldsymbol\beta,
                 \mathrm Z \mathrm G \mathrm Z^{\intercal})
    d\mathbf z

This module implements the Expectation Propagation algorithm for parameter
fitting via Maximum Likelihood: :class:`.ExpFamEP`.

Public interface
^^^^^^^^^^^^^^^^
"""

from .ep import ExpFamEP

__all__ = ['ExpFamEP']
