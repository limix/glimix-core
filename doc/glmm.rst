*******************************
Generalised Linear Mixed Models
*******************************

.. currentmodule:: glimix_core.glmm

Introduction
============

A GLMM can be described in two parts.
The first part consists in the latent variable

.. math::

    \mathbf z = \mathrm X\boldsymbol\beta + \mathrm G\mathbf u +
                \boldsymbol\epsilon,

where :math:`\mathbf u \sim \mathcal N(\mathbf 0, v_0\mathrm I_k)` is a
vector of random effects and :math:`\epsilon_i` are iid Normal random
variables with variance :math:`v_1` each.
The second part connects the latent variable to the observed one:

.. math::

    y_i ~|~ z_i \sim \text{ExpFam}(\mu_i = g(z_i)),

where :math:`g(\cdot)` is a link function [#f3]_ and :math:`\text{ExpFam}(\cdot)` is
an exponential-family distribution [#f4]_ with mean :math:`\mu_i`.
The marginal likelihood is thus given by

.. math::

    p(\mathbf y) = \int
      \prod_i \text{ExpFam}(y_i ~|~ \mu_i = g(z_i))
      \mathcal N(\mathbf z ~|~ \mathrm X\boldsymbol\beta,
                 v_0\mathrm G\mathrm G^{\intercal}
                 + v_1\mathrm I_n)
    \mathrm d\mathbf z

We use :math:`\mathrm K` for refering to
:math:`\mathrm G\mathrm G^{\intercal}` when appropriate.
The module :mod:`glimix_core.glmm` implements two algorithms for parameter fitting via
Maximum Likelihood [#f1]_: Expectation Propagation approximation [#f2]_ when the
likelihood is not normally distributed (refer to :class:`.GLMMExpFam`);
a closed-form solution otherwise (refer to :class:`.GLMMNormal`).

Exponential family likelihood
=============================

.. autoclass:: glimix_core.glmm.GLMMExpFam
    :members:
    :inherited-members:
    :exclude-members: value, set_data, set_nodata, unset_data, variables, feed

Heterogeneous Normal likelihood
===============================

.. autoclass:: glimix_core.glmm.GLMMNormal
    :members:
    :inherited-members:
    :exclude-members: value, set_data, set_nodata, unset_data, variables, feed

.. rubric:: References

.. [#f1] Maximum likelihood estimation. (2017, September 9). In Wikipedia,
         The Free Encyclopedia. Retrieved 14:48, September 22, 2017, from
         https://en.wikipedia.org/w/index.php?title=Maximum_likelihood_estimation&oldid=799704694
.. [#f2] Minka, T. P. (2001, August). Expectation propagation for approximate
         Bayesian inference. In Proceedings of the Seventeenth conference on
         Uncertainty in artificial intelligence (pp. 362-369).
         Morgan Kaufmann Publishers Inc..
.. [#f3] Wikipedia contributors. (2018, June 21). Generalized linear model. In
         Wikipedia, The Free Encyclopedia. Retrieved 09:45, August 6, 2018, from
         https://en.wikipedia.org/w/index.php?title=Generalized_linear_model&oldid=846910075#Link_function
.. [#f4] Wikipedia contributors. (2018, June 29). Exponential family. In Wikipedia, The
         Free Encyclopedia. Retrieved 09:48, August 6, 2018, from
         https://en.wikipedia.org/w/index.php?title=Exponential_family&oldid=848114709