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

    \mathbf y \sim \mathcal N(\mathrm X\boldsymbol\beta,
                              v_0 \mathrm G \mathrm G^{\intercal}
                              + v_1\mathrm I)


We refer to :math:`\mathrm G \mathrm G^{\intercal}` simply as
:math:`\mathrm K` in some occasions for didactic purpose.

This module provides :class:`.LMM`, a FastLMM [Lip11]_
implementation to perform inference over the variance parameters
:math:`v_0` and :math:`v_1` and over the vector
:math:`\boldsymbol\beta` of fixed-effect sizes.
An instance of this class is created by provided the outcome ``y``,
the covariates ``X``, and the covariance ``K`` via its egonomic
eigendecomposition.
Here is an example:

.. doctest::

    >>> from numpy import array, ones
    >>> from numpy_sugar.linalg import economic_qs_linear
    >>> from glimix_core.lmm import LMM
    >>>
    >>> X = array([[1, 2], [3, -1], [1.1, 0.5], [0.5, -0.4]], float)
    >>> QS = economic_qs_linear(X)
    >>> X = ones((4, 1))
    >>> y = array([-1, 2, 0.3, 0.5])
    >>> lmm = LMM(y, X, QS)
    >>> lmm.fit(verbose=False)
    >>> lmm.lml()  # doctest: +NPY_FLEX_NUMS
    -2.2727396924914833

The method :func:`glimix_core.lmm.LMM.fit` is called to optimise the marginal
likelihood over the fixed-effect sizes :math:`\boldsymbol\beta` and over the
variances :math:`v_0` and :math:`v_1`.
The resulting values for the above inference are:

.. doctest::

    >>> lmm.beta  # doctest: +NPY_FLEX_NUMS
    array([ 0.06646503])
    >>> lmm.v0  # doctest: +NPY_FLEX_NUMS
    0.33744719539335433
    >>> lmm.v1  # doctest: +NPY_FLEX_NUMS
    0.012502012057504848

This module also provides :class:`.FastScanner`,
an approximated LMM implementation for performing an even faster
inference across several (millions, for example) covariates independently.
More detail about this approach is given in the next section.

Association scan
^^^^^^^^^^^^^^^^

Let :math:`\mathrm X` be a samples-by-covariates matrix,
:math:`\mathrm M` a samples-by-candidates matrix, and
:math:`\mathbf y` an array of outcome:

.. math::

    \mathbf y \sim \mathcal N\big(~ \mathrm X\mathbf b
    + \mathbf{m}_j \alpha_j,~
    s (\mathrm K + v \mathrm I) ~\big)

Note that :math:`\alpha_j` is a scalar multiplying a column-matrix
:math:`\mathbf{m}_j`.
The variable :math:`s` is a scaling factor that, if not set, is jointly
adjusted with :math:`\alpha_j` in order to maximise the marginal
likelihood.
The variable :math:`v` is held fixed and that is the reason why this inference
can be performed quickly over millions of candidates.

The user provides the outcome ``y``, the covariates ``X``, the covariance ``K``
via its eigendecomposition, and the variance ``v`` to create an instance of
the :class:`glimix_core.lmm.FastScanner` class.
After that, the user can call the
:func:`glimix_core.lmm.FastScanner.fast_scan` method and pass the matrix of
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
    >>> lmls  # doctest: +NPY_FLEX_NUMS
    array([ 0.25447806,  0.4400597 ,  0.86560024])
    >>> effect_sizes  # doctest: +NPY_FLEX_NUMS
    array([-0.07463993, -0.04413758,  0.09065001])


Implementation
^^^^^^^^^^^^^^

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

Consider the economic eigendecomposition of ``K``:

.. math::

    [\mathrm Q_0 \quad \mathrm Q_1]
        \left[\begin{array}{cc}
            \mathrm S_0 & \mathbf 0\\
            \mathbf 0 & \mathbf 0
        \end{array}\right]
    \left[\begin{array}{c}
        \mathrm Q_0^{\intercal} \\
        \mathrm Q_1^{\intercal}
    \end{array}\right] = \mathrm K.

We thus have

.. math::

    s^{-1}((1-\delta)\mathrm K + \delta\mathrm I)^{-1} = s^{-1}
        \mathrm Q \mathrm D^{-1}
        \mathrm Q^{\intercal},

where

.. math::

    \mathrm D = \left(
        \begin{array}{cc}
          (1-\delta)\mathrm S_0 + \delta\mathrm I_0 & \mathbf 0\\
          \mathbf 0 & \delta\mathrm I_1
        \end{array}
        \right).

A diagonal covariance-matrix can then be used to define an equivalent
marginal likelihood:

.. math::

    \mathcal N\left(\mathrm Q^{\intercal} \mathbf y ~|~
               \mathrm Q^{\intercal} \mathrm X\boldsymbol\beta,~
               s \mathrm D \right).

Taking the logarithm and expanding it gives us

.. math::

   \log p(\mathbf y) &=
       -\frac{n}{2} \log 2\pi - \frac{1}{2}n \log s
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
       -\frac{n}{2} \log 2\pi - \frac{1}{2}n \log s
           - \frac{1}{2}\log|\mathrm D|\\
       &  \frac{1}{2} (\mathrm Q^{\intercal}\mathbf y)^{\intercal} s^{-1}
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

   \log p(\mathbf y) &=
       -\frac{n}{2} \log 2\pi - \frac{n}{2} \log s^*
           - \frac{1}{2}\log|\mathrm D| - \frac{n}{2}\\
           &= \log \mathcal N(\text{Diag}(\sqrt{s^*\mathrm D})
               ~|~ \mathbf 0, s^*\mathrm D).

LMM class
^^^^^^^^^

.. autoclass:: glimix_core.lmm.LMM
  :members:

FastScanner class
^^^^^^^^^^^^^^^^^

.. autoclass:: glimix_core.lmm.FastScanner
  :members:
