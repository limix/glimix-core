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

    >>> from numpy import array
    >>> from numpy_sugar.linalg import economic_qs_linear
    >>> from glimix_core.lmm import LMM
    >>>
    >>> X = array([[1, 2], [3, -1]], float)
    >>> QS = economic_qs_linear(X)
    >>> X = array([[1], [1]])
    >>> y = array([-1, 2], float)
    >>> lmm = LMM(y, X, QS)
    >>> lmm.fit(verbose=False)
    >>> print('%.3f' % lmm.lml())
    -3.649

The method :func:`glimix_core.lmm.LMM.fit` is called to optimise the marginal
likelihood over the fixed-effect sizes :math:`\boldsymbol\beta` and over the
variances :math:`v_0` and :math:`v_1`.
The resulting values for the above inference are:

.. doctest::

    >>> print("{:.5f}".format(lmm.beta[0]))
    0.49996
    >>> print("{:.5f}".format(lmm.v0))
    0.00002
    >>> print("{:.5f}".format(lmm.v1))
    2.24985

This module also provides :class:`.FastScanner`,
an approximated LMM implementation for performing an even faster
inference across several (millions, for example) covariates independently.
More detail about this approach is given in the next section.

Association scan
^^^^^^^^^^^^^^^^

Let :math:`\mathrm X` be a samples-by-covariates matrix,
:math:`\mathrm M` a samples-by-markers matrix, and
:math:`\mathbf y` an array of outcome.

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
can be performed quickly over millions of markers.

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
    >>> M = array([[1, 3, 2], [0, -5, 1]])
    >>> lmls, effect_sizes = scanner.fast_scan(M, verbose=False)
    >>>
    >>> from numpy import array2string
    >>>
    >>> print(array2string(lmls, precision=4, separator=','))
    [  33.0476, 703.321 ,  33.0476]
    >>> print(array2string(effect_sizes, precision=4, separator=','))
    [-3.   ,-0.375,-3.   ]


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

The marginal likelihood can thus be written as

.. math::

    \mathcal N\left(\mathrm Q^{\intercal} \mathbf y ~|~
               \mathrm Q^{\intercal} \mathrm X\boldsymbol\beta,~
               s D \right).

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
to zero leads to the solutions :math:`\boldsymbol\beta^*` for equation

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



Let :math:`\mathrm E_j = [\mathrm X; \quad \mathbf{m}_j]`
be the concatenated matrix of covariates :math:`\mathrm X` and the
candidate for association :math:`\mathbf{m}_j`.
Let

.. math::

    \mathrm D = \left(
        \begin{array}{cc}
            \mathrm S_0 + v\mathrm I_0 & \mathbf 0\\
            \mathbf 0 & v\mathrm I_1
        \end{array}\right)

be a diagonal matrix with the eigenvalues of the covariance matrix
when :math:`s=1`, and also :math:`D_0 = \mathrm S_0 + v\mathrm I_0` and
:math:`D_1 = v\mathrm I_1`

We make use of the identity

.. math::

    (\mathrm K + v\mathrm I)^{-1}=\mathrm Q\mathrm D^{-1}\mathrm Q^{\intercal}

in the implementation for performance reasons.
We can thus write the marginal likelihood as

.. math::

    \mathcal N\left(\mathrm Q^{\intercal} \mathbf y ~|~
               \mathrm Q^{\intercal} \mathrm E_j \boldsymbol\beta_j,~
               s_j \mathrm D\right),

where the covariance is now given by a diagonal matrix.

.. automodule:: glimix_core.lmm

Let :math:`\mathrm X` be a samples-by-covariates matrix,
:math:`\mathrm M` a samples-by-markers matrix, and
:math:`\mathbf y` an array of outcome.
A covariance :math:`\mathrm K` will be provided via its economic eigen
decomposition ``((Q0, Q1), S0)`` and ``v`` will define the variance
due to the identity in the formulae

.. math::

    \mathbf y \sim \mathcal N\big(~ \mathrm X\mathbf b
    + \mathbf{m}_j \alpha_j,~
    s (\mathrm K + v \mathrm I) ~\big)

Note that :math:`\alpha_j` is a scalar multiplying a column-matrix
:math:`\mathbf{m}_j`.
The variable :math:`s` is a scaling factor that, if not set, is jointly
adjusted with :math:`\alpha_j` in order to maximise the marginal
likelihood;
ultimately providing the degree of association between the
marker :math:`\mathbf{m}_j` with the outcome :math:`\mathbf y` via an
p-value.

Let :math:`\mathrm E_j = [\mathrm X; \quad \mathbf{m}_j]`
be the concatenated matrix of covariates :math:`\mathrm X` and the
candidate for association :math:`\mathbf{m}_j`.
Let

.. math::

    \mathrm D = \left(
        \begin{array}{cc}
            \mathrm S_0 + v\mathrm I_0 & \mathbf 0\\
            \mathbf 0 & v\mathrm I_1
        \end{array}\right)

be a diagonal matrix with the eigenvalues of the covariance matrix
when :math:`s=1`, and also :math:`D_0 = \mathrm S_0 + v\mathrm I_0` and
:math:`D_1 = v\mathrm I_1`

We make use of the identity

.. math::

    (\mathrm K + v\mathrm I)^{-1}=\mathrm Q\mathrm D^{-1}\mathrm Q^{\intercal}

in the implementation for performance reasons.
We can thus write the marginal likelihood as

.. math::

    \mathcal N\left(\mathrm Q^{\intercal} \mathbf y ~|~
               \mathrm Q^{\intercal} \mathrm E_j \boldsymbol\beta_j,~
               s_j \mathrm D\right),

where the covariance is now given by a diagonal matrix.
