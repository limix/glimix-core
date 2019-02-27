.. _lmm-intro:

Introduction
============

A LMM can be described as

.. math::
    :label: lmm1

    \mathbf y = \mathrm X\boldsymbol\beta + \mathrm G\mathbf u
    + \boldsymbol\epsilon,

where :math:`\mathbf u \sim \mathcal N(\mathbf 0, v_0\mathrm I_k)` is a
vector of random effects and :math:`\epsilon_i` are iid Normal random variables
with zero-mean and variance :math:`v_1` each.
The outcome-vector is thus distributed according to

.. math::

    \mathbf y \sim \mathcal N(\mathrm X\boldsymbol\beta,
                              v_0 \mathrm G \mathrm G^{\intercal}
                              + v_1\mathrm I_n)


We refer to :math:`\mathrm G \mathrm G^{\intercal}` as
:math:`\mathrm K`, the covariance-matrix.

The :class:`.LMM` class provides a FastLMM [#f3]_
implementation to perform inference over the variance parameters
:math:`v_0` and :math:`v_1` and over the vector
:math:`\boldsymbol\beta` of fixed-effect sizes.
An instance of this class is created by providing the outcome ``y``,
the covariates ``X``, and the covariance ``K`` via its economic eigendecomposition.
Here is an example:

.. doctest::

    >>> from numpy import array, ones
    >>> from numpy_sugar.linalg import economic_qs_linear
    >>> from glimix_core.lmm import LMM
    >>>
    >>> G = array([[1, 2], [3, -1], [1.1, 0.5], [0.5, -0.4]], float)
    >>> QS = economic_qs_linear(G)
    >>> X = ones((4, 1))
    >>> y = array([-1, 2, 0.3, 0.5])
    >>> lmm = LMM(y, X, QS)
    >>> lmm.fit(verbose=False)
    >>> lmm.lml()  # doctest: +FLOAT_CMP
    -2.2726234086180557

The method :func:`.LMM.fit` is called to optimise the marginal
likelihood over the fixed-effect sizes :math:`\boldsymbol\beta` and over the
variances :math:`v_0` and :math:`v_1`.
The resulting values for the above inference are:

.. doctest::

    >>> lmm.beta[0]  # doctest: +FLOAT_CMP
    0.0664650291693258
    >>> lmm.v0  # doctest: +FLOAT_CMP
    0.33736446158226896
    >>> lmm.v1  # doctest: +FLOAT_CMP
    0.012503600451739165

This module also provides :class:`.FastScanner`,
an approximated LMM implementation for performing an even faster
inference across several (millions, for example) covariates independently.
More detail about this approach is given in the next section.