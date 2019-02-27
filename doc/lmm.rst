*******************
Linear Mixed Models
*******************

Linear mixed models (LMMs) are a generalisation of linear models [#f1]_ to allow the
ouctome to be described as a summation of both fixed and random effects [#f2]_.
LMM inference is implemented by the :mod:`glimix_core.lmm` module and described here.

.. note::

   Please refer to the Variables_ section for the definition of the
   otherwise-unspecified mathematical symbols.

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

Association scan
================

Let :math:`\mathrm X` be a |n|-by-|d| matrix,
:math:`\mathrm M` a |n|-by-|c| matrix, and
:math:`\mathbf y` an array of outcome:

.. math::

    \mathbf y \sim \mathcal N\big(~ \mathrm X\mathbf b
    + \mathbf{m}_j \alpha_j,~
    s (\mathrm K + v \mathrm I_n) ~\big)

Note that :math:`\alpha_j` is a scalar multiplying a column-matrix
:math:`\mathbf{m}_j`.
The variable :math:`s` is a scaling factor that, if not set, is jointly
adjusted with :math:`\alpha_j` in order to maximise the marginal
likelihood.
The variable :math:`v` is held fixed and that is the reason why this inference
can be performed quickly over millions of candidates.

The user provides the outcome ``y``, the covariates ``X``, the covariance ``K``
via its eigendecomposition, and the variance ``v`` to create an instance of
the :class:`.FastScanner` class.
After that, the user is able to call the
:func:`.FastScanner.fast_scan` method and pass the matrix of
candidates ``M`` to compute the log of the marginal likelihood and the
fixed-effect size corresponding to each column of ``M``.
For example:

.. doctest::

    >>> from glimix_core.lmm import FastScanner
    >>>
    >>> scanner = FastScanner(y, X, QS, lmm.v1)
    >>> M = array([[1, 3, -1.5], [0, -2, 4], [-2, -2.5, 3], [0.2, +2, 2]])
    >>> lmls, effect_sizes = scanner.fast_scan(M, verbose=False)
    >>>
    >>> lmls[0]  # doctest: +FLOAT_CMP
    0.25435129186857885
    >>> lmls[1]  # doctest: +FLOAT_CMP
    0.4399332250146468
    >>> lmls[2]  # doctest: +FLOAT_CMP
    0.8654753462599309
    >>> effect_sizes[0]  # doctest: +FLOAT_CMP
    -0.07463997107886489
    >>> effect_sizes[1]  # doctest: +FLOAT_CMP
    -0.044137681512885746
    >>> effect_sizes[2]  # doctest: +FLOAT_CMP
    0.09065047700251282

API
===

LMM
---

.. autoclass:: glimix_core.lmm.LMM
  :members:
  :exclude-members: value

FastScanner
-----------

.. autoclass:: glimix_core.lmm.FastScanner
  :members:

MTLMM
-----

.. autoclass:: glimix_core.lmm.MTLMM
  :members:
  :exclude-members: value, gradient

Kron2Sum
--------

.. autoclass:: glimix_core.lmm.Kron2Sum
  :members:
  :exclude-members: value, gradient

Implementation
==============

Single-trait
------------

The LMM model :eq:`lmm1` can be equivalently written as

.. math::

    \mathbf y \sim \mathcal N\Big(~ \mathrm X\boldsymbol\beta;~
      s \big(
        (1-\delta)
          \mathrm K +
        \delta \mathrm I
      \big)
    ~\Big),

and we thus have :math:`v_0 = s (1 - \delta)` and :math:`v_1 = s \delta`.
Consider the economic eigendecomposition of :math:`\mathrm K`:

.. math::

    \overbrace{[\mathrm Q_0 \quad \mathrm Q_1]}^{\mathrm Q}
        \overbrace{\left[\begin{array}{cc}
            \mathrm S_0 & \mathbf 0\\
            \mathbf 0 & \mathbf 0
        \end{array}\right]}^{\mathrm S}
    \left[\begin{array}{c}
        \mathrm Q_0^{\intercal} \\
        \mathrm Q_1^{\intercal}
    \end{array}\right] = \mathrm K

and let

.. math::

    \mathrm D = \left[
        \begin{array}{cc}
          (1-\delta)\mathrm S_0 + \delta\mathrm I_r & \mathbf 0\\
          \mathbf 0 & \delta\mathrm I_{n-r}
        \end{array}
        \right].

We thus have

.. math::

    ((1-\delta)\mathrm K + \delta\mathrm I_n)^{-1} =
        \mathrm Q \mathrm D^{-1}
        \mathrm Q^{\intercal}.

A diagonal covariance-matrix can then be used to define an equivalent
marginal likelihood:

.. math::

    \mathcal N\left(\mathrm Q^{\intercal} \mathbf y ~|~
               \mathrm Q^{\intercal} \mathrm X\boldsymbol\beta,~
               s \mathrm D \right).


Taking the logarithm and expanding it gives us

.. math::

   \log p(\mathbf y) &=
       -\frac{n}{2} \log 2\pi - \frac{n}{2} \log s
           - \frac{1}{2}\log|\mathrm D|\\
       &- \frac{1}{2} (\mathrm Q^{\intercal}\mathbf y)^{\intercal} s^{-1}
           \mathrm D^{-1}(\mathrm Q^{\intercal} \mathbf y)\\
       &+ (\mathrm Q^{\intercal}\mathbf y)^{\intercal}
           s^{-1} \mathrm D^{-1}
           (\mathrm Q^{\intercal} \mathrm X \boldsymbol\beta)\\
       &- \frac{1}{2} (\mathrm Q^{\intercal}
           \mathrm X \boldsymbol\beta)^{\intercal} s^{-1} \mathrm D^{-1}
           (\mathrm Q^{\intercal} \mathrm X \boldsymbol\beta).

Setting the derivative of :math:`\log p(\mathbf y)` over effect sizes equal
to zero leads to solutions :math:`\boldsymbol\beta^*` from equation

.. math::

   (\mathrm Q^{\intercal}\mathrm X \boldsymbol\beta^*)^{\intercal}
       \mathrm D^{-1} (\mathrm Q^{\intercal} \mathrm X) =
       (\mathrm Q^{\intercal}\mathbf y)^{\intercal}\mathrm D^{-1}
       (\mathrm Q^{\intercal}\mathrm X).

Replacing it back to the log of the marginal likelihood gives us the simpler
formulae

.. math::

   \log p(\mathbf y) &=
       -\frac{n}{2} \log 2\pi - \frac{n}{2} \log s
           - \frac{1}{2}\log|\mathrm D|\\
       & +\frac{1}{2} (\mathrm Q^{\intercal}\mathbf y)^{\intercal} s^{-1}
           \mathrm D^{-1}\mathrm Q^{\intercal}
           (\mathrm X\boldsymbol\beta^* - \mathbf y).


In the extreme case where :math:`\boldsymbol\beta^*` is such that
:math:`\mathbf y = \mathrm X\boldsymbol\beta^*`, the maximum is attained
as :math:`s \rightarrow 0`.

Setting the derivative of :math:`\log p(\mathbf y; \boldsymbol\beta^*)` over
scale equal to zero leads to the maximum

.. math::

   s^* = n^{-1}
       (\mathrm Q^{\intercal}\mathbf y)^{\intercal}
           \mathrm D^{-1}\mathrm Q^{\intercal}
           (\mathbf y - \mathrm X\boldsymbol\beta^*).


We offer the possibility to use either :math:`s^*` found via the
above equation or a scale defined by the user.
In the first case we have a further simplification of the log of the marginal
likelihood:

.. math::

   \log p(\mathbf y; \boldsymbol\beta^*, s^*) &=
       -\frac{n}{2} \log 2\pi - \frac{n}{2} \log s^*
           - \frac{1}{2}\log|\mathrm D| - \frac{n}{2}\\
           &= \log \mathcal N(\text{Diag}(\sqrt{s^*\mathrm D})
               ~|~ \mathbf 0, s^*\mathrm D).

.. _mtlmm-impl:

Uncorrelated multi-trait
------------------------

The extension to multiple traits becomes easy under the assumption that the traits are
uncorrelated, as assumed in this section.
Let :math:`m` be the number of traits.
We stack all the different traits into

.. math::

    \mathbf y = \text{vec}\left(\left[ \mathbf y_0 ~ \mathbf y_1 ~\cdots~ \mathbf y_m
        \right] \right)

Similarly, we have the covariates, fixed-effect sizes, and the assembled covariance
matrix :math:`\tilde{\mathrm K}` as

.. math::

    \mathrm X = \text{vec}\left(\left[ \mathrm X_0 ~ \mathrm X_1 ~\cdots~ \mathrm X_m
        \right] \right),

.. math::

    \boldsymbol\beta = \text{vec}\left(
        \left[
            \boldsymbol\beta_0 ~ \boldsymbol\beta_1 ~\cdots~ \boldsymbol\beta_m
        \right]
    \right),

and

.. math::

    \tilde{\mathrm K} =
        \left[
            \begin{array}{ccc}
                \mathrm K & \mathbf 0 & \cdots \\
                \vdots    & \ddots    &        \\
                \mathbf 0 &           & \mathrm K
            \end{array}
        \right],

where :math:`\mathrm K` is repeated |m| times in :math:`\tilde{\mathrm K}`.
We thus consider the model

.. math::
    :label: mtlmm1

    \mathbf y \sim \mathcal N\Big(~
        \mathrm X\boldsymbol\beta;~
        v_0 \tilde{\mathrm K} + v_1 \mathrm I_{nm}
    ~\Big),

which is the model :eq:`lmm1` with multi-trait structure and uncorrelated traits.

We use the fact that the eigendecomposition of :math:`\tilde{\mathrm K}` can be
computed as fast as the eigendecomposition of :math:`\mathrm K`:

.. math::

    \overbrace{[\mathrm Q ~ \cdots ~ \mathrm Q]}^{\tilde{\mathrm Q}}
        \overbrace{\left[\begin{array}{ccc}
            \mathrm S & \mathbf 0 & \cdots \\
            \vdots    & \ddots    &   \\
            \mathbf 0 &           & \mathrm S
        \end{array}\right]}^{\tilde{\mathrm S}}
    \left[\begin{array}{c}
        \mathrm Q^{\intercal} \\
        \vdots \\
        \mathrm Q^{\intercal}
    \end{array}\right] = \tilde{\mathrm K}.

Let

.. math::

    \tilde{\mathrm D} = \left[
        \begin{array}{cc}
          (1-\delta)\tilde{\mathrm S}_0 + \delta\mathrm I_r & \mathbf 0\\
          \mathbf 0 & \delta\mathrm I_{n-r}
        \end{array}
        \right].

We thus have

.. math::

    ((1-\delta)\tilde{\mathrm K} + \delta\mathrm I_{nm})^{-1} =
            \left[\begin{array}{ccc}
                \mathrm Q\mathrm D^{-1}\mathrm Q^{\intercal} & \mathbf 0 & \cdots\\
                \vdots    & \ddots    & \\
                \mathbf 0 &           & \mathrm Q\mathrm D^{-1}\mathrm Q^{\intercal}
            \end{array}\right]

A diagonal covariance-matrix can then be used to define an equivalent marginal
likelihood:

.. math::

    \mathcal N\left(\tilde{\mathrm Q}^{\intercal} \mathbf y ~|~
               \tilde{\mathrm Q}^{\intercal}
               \mathrm X\boldsymbol\beta
               ,~
               s\left[\begin{array}{ccc}
                    \mathrm D & \mathbf 0 & \cdots\\
                    \vdots    & \ddots    & \\
                    \mathbf 0 &           & \mathrm D
                \end{array}\right]
               \right).

The optimal effect sizes are solutions to the equation

.. math::

   \left[(\mathrm Q^{\intercal}\mathrm X_i\boldsymbol\beta_i^*)^{\intercal}
       \mathrm D^{-1} (\mathrm Q^{\intercal} \mathrm X_i)\right] =
       \left[(\mathrm Q^{\intercal}\mathbf y_i)^{\intercal}\mathrm D^{-1}
       (\mathrm Q^{\intercal}\mathrm X_i)\right],

for :math:`i \in \{1, \dots, m\}`.
Setting the derivative of :math:`\log p(\mathbf y; \boldsymbol\beta^*)` over scale equal
to zero leads to the maximum

.. math::

   s^* = (nm)^{-1}
       \left[(\mathrm Q^{\intercal}\mathbf y_i)^{\intercal}
           \mathrm D^{-1}\right]\left[\mathrm Q^{\intercal}
           (\mathbf y_i - \mathrm X_i\boldsymbol\beta_i^*)\right].


Using the above, optimal scale leads to a further simplification of the log of the
marginal likelihood:

.. math::

   \log p(\mathbf y; \boldsymbol\beta^*, s^*) &=
       -\frac{nm}{2} \log 2\pi - \frac{nm}{2} \log s^*
           - \frac{1}{2}\log|\tilde{\mathrm D}| - \frac{nm}{2}\\
           &= \log \mathcal N(\text{Diag}(\sqrt{s^*\tilde{\mathrm D}})
               ~|~ \mathbf 0, s^*\tilde{\mathrm D}).

.. _Variables:

.. rubric:: Variables

:|n|: Number of samples.
:|m|: Number of traits.
:|c|: Number of candidates.
:|d|: Number of covariates.
:|k|: Number of random effects.
:|r|: Covariance-matrix rank.

.. |n| replace:: :math:`n`
.. |m| replace:: :math:`m`
.. |c| replace:: :math:`c`
.. |d| replace:: :math:`d`
.. |k| replace:: :math:`k`
.. |r| replace:: :math:`r`

Multi-trait
-----------

TODO:

.. rubric:: References

.. [#f1] Wikipedia contributors. (2018, May 22). Linear model. In Wikipedia, The Free
         Encyclopedia. Retrieved 16:00, August 5, 2018, from
         https://en.wikipedia.org/w/index.php?title=Linear_model&oldid=842479751.

.. [#f2] Introduction to linear mixed models. UCLA: Institute for Digital Research and
         Education. Retrieved from August 5, 2018, from
         https://stats.idre.ucla.edu/other/mult-pkg/introduction-to-linear-mixed-models/.

.. [#f3] Lippert, Christoph, Listgarten, Jennifer, Liu, Ying, Kadie, Carl M,
         Davidson, Robert I & Heckerman, David (2011). FaST linear mixed
         models for genome-wide association studies. Nature methods, 8,
         833-835.