r"""
*******************************
Generalised Linear Mixed Models
*******************************

Introduction
^^^^^^^^^^^^

A GLMM can be described in two parts.
The first part consists in the latent variable

.. math::

    \mathbf z = \mathrm X\boldsymbol\beta + \mathrm G\mathbf u +
                \boldsymbol\epsilon,

where :math:`\mathbf u \sim \mathcal N(\mathbf 0, v_0\mathrm I)` is a
vector of random effects and :math:`\epsilon_i` are iid Normal random
variables with variance :math:`v_1` each.
The second part connects the latent variable to the observed one:

.. math::

    y_i ~|~ z_i \sim \text{ExpFam}(\mu_i = g(z_i)),

where :math:`g(\cdot)` is a link function and :math:`\text{ExpFam}(\cdot)` is
an exponential-family distribution with mean :math:`\mu_i`.
The marginal likelihood is thus given by

.. math::

    p(\mathbf y) = \int
      \prod_i \text{ExpFam}(y_i ~|~ \mu_i = g(z_i))
      \mathcal N(\mathbf z ~|~ \mathrm X\boldsymbol\beta,
                 v_0\mathrm G\mathrm G^{\intercal}
                 + v_1\mathrm I)
    \mathrm d\mathbf z

We use :math:`\mathrm K` for refering to
:math:`\mathrm G\mathrm G^{\intercal}` when appropriate.
This module implements two algorithms for parameter fitting via Maximum
Likelihood [ME17]_.
Expectation Propagation approximation [Min01]_ when the likelihood is
not normally distributed (refer to :class:`glimix_core.glmm.GLMMExpFam`);
a closed-form solution otherwise (refer to
:class:`glimix_core.glmm.GLMMNormal`).
The class :class:`glimix_core.glmm.GLMM` is a base classe for those two
implementations, providing common methods to query the model.

GLMM class
^^^^^^^^^^

.. autoclass:: glimix_core.glmm.GLMM
    :members:

GLMMExpFam class
^^^^^^^^^^^^^^^^

.. autoclass:: glimix_core.glmm.GLMMExpFam
    :members:

GLMMNormal class
^^^^^^^^^^^^^^^^

.. autoclass:: glimix_core.glmm.GLMMNormal
    :members:
"""

from .expfam import GLMMExpFam
from .glmm import GLMM
from .normal import GLMMNormal

__all__ = ['GLMM', 'GLMMNormal', 'GLMMExpFam']
