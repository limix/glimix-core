r"""
*******************************
Generalized Linear Mixed Models
*******************************

Introduction
^^^^^^^^^^^^

A GLMM can be described in two parts.
The first part consists in the latent variable

.. math::

    \mathbf z = \mathrm X\boldsymbol\beta + \mathrm G\mathbf u +
                \boldsymbol\epsilon,

where :math:`\mathbf u \sim \mathcal N(\mathbf 0, \sigma_g^2\mathrm I)` is a
vector of random effects and :math:`\epsilon_i` are iid Normal random
variables with variance :math:`\sigma_e^2` each.
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
                 \sigma_g^2\mathrm G\mathrm G^{\intercal}
                 + \sigma_e^2\mathrm I)
    \mathrm d\mathbf z

This module implements the Expectation Propagation algorithm for parameter
fitting via Maximum Likelihood: :class:`.ExpFamEP`.

Public interface
^^^^^^^^^^^^^^^^
"""

from .glmm import GLMM

__all__ = ['GLMM']
