*******************
Linear Mixed Models
*******************

Linear mixed models (LMMs) are a generalisation of linear models [#f1]_ to allow the
ouctome to be described as a summation of both fixed and random effects [#f2]_.
LMM inference is implemented by the :mod:`glimix_core.lmm` module and described here.

.. |n| replace:: :math:`n`
.. |m| replace:: :math:`m`
.. |c| replace:: :math:`c`
.. |d| replace:: :math:`d`
.. |k| replace:: :math:`k`
.. |r| replace:: :math:`r`

.. _lmm-intro:

Introduction
============

A LMM can be described as ::

    𝐲 = X𝜷 + G𝐮 + 𝛜,

where 𝐮 ∼ 𝓝(𝟎, v₀I) is a
vector of random effects and \epsilonᵢ are iid Normal random variables
with zero-mean and variance v₁ each.
The outcome-vector is thus distributed according to ::

    𝐲 ∼ 𝓝(X𝜷, v₀GGᵀ + v₁I)

The :class:`.LMM` class provides a FastLMM [#f3]_
implementation to perform inference over the variance parameters
v₀ and v₁ and over the vector
𝜷 of fixed-effect sizes.
An instance of this class is created by providing the outcome ``y``,
the covariates ``X``, and the covariance ``K`` via its economic eigendecomposition.
Here is an example:

.. doctest::

    >>> from numpy import ones
    >>> from numpy_sugar.linalg import economic_qs_linear
    >>>
    >>> from glimix_core.lmm import LMM
    >>>
    >>> G = [[1, 2], [3, -1], [1.1, 0.5], [0.5, -0.4]]
    >>> QS = economic_qs_linear(G)
    >>> X = ones((4, 1))
    >>> y = [-1, 2, 0.3, 0.5]
    >>> lmm = LMM(y, X, QS)
    >>> lmm.fit(verbose=False)
    >>> lmm.lml()  # doctest: +FLOAT_CMP
    -2.2726234086180557

The method :func:`.LMM.fit` is called to optimise the marginal
likelihood over the fixed-effect sizes 𝜷 and over the
variances v₀ and v₁.
The resulting values for the above inference are:

.. doctest::

    >>> lmm.beta[0]  # doctest: +FLOAT_CMP
    0.0664650291693258
    >>> lmm.v0  # doctest: +FLOAT_CMP
    0.33736446158226896
    >>> lmm.v1  # doctest: +FLOAT_CMP
    0.012503600451739165

We also provide :class:`.FastScanner`,
an approximated LMM implementation for performing an even faster
inference across several (millions, for example) covariates independently.
More detail about this approach is given in the `Association scan`_ section.

Multi-Trait
===========

This package also provides a variant of LMM that models multiple outcomes (or traits) of
the same set of samples.
Let p be the number of traits.
The outcome matrix Y is the concatenation of p vectors::

    Y = [𝐲₀ 𝐲₁ ... 𝐲ₚ].

The mean definition will involve three matrices::

    M = (A ⊗ F) vec(B),

where vec(·) stacks the columns of the input matrix into a single-column matrix.
B is a c×p matrix of effect sizes for c being the number of covariates.
A is a p×p design matrix that determines the covariance between the traits over the mean
vector.
F is a n×p design matrix of covariates.

The covariance matrix will be::

    K = C₀ ⊗ GGᵀ + C₁ ⊗ I.

C₀ and C₁ are p×p symmetric matrices whose values will be optimized.
GGᵀ gives the covariance between samples, while (C₀ ⊗ GGᵀ) gives the covariance between
samples when traits are taken into account.

Putting the outcome, mean, and covariance-matrix together, we have the distribution ::

    vec(Y) ~ N((A ⊗ F) vec(B), K = C₀ ⊗ GGᵀ + C₁ ⊗ I).

The parameters of the multi-trait LMM to be fit via maximum likelihood are the matrices
B, C₀, and C₁.

.. doctest::

    >>> from numpy.random import RandomState
    >>> from glimix_core.lmm import Kron2Sum
    >>>
    >>> random = RandomState(0)
    >>> n = 5
    >>> p = 2
    >>> c = 3
    >>> Y = random.randn(n, p)
    >>> A = random.randn(p, p)
    >>> A = A @ A.T
    >>> F = random.randn(n, c)
    >>> G = random.randn(n, 4)
    >>>
    >>> mlmm = Kron2Sum(Y, A, F, G)
    >>> mlmm.fit(verbose=False)
    >>> mlmm.lml()  # doctest: +FLOAT_CMP
    -5.666702537532974
    >>> mlmm.B  # doctest: +FLOAT_CMP
    array([[-0.17170011,  0.45565163],
           [ 0.57532031, -0.86070064],
           [ 0.21050686, -0.02573517]])
    >>> mlmm.cov.C0.value()  # doctest: +FLOAT_CMP
    array([[ 0.01598945, -0.04374046],
           [-0.04374046,  0.11965561]])
    >>> mlmm.cov.C1.value()  # doctest: +FLOAT_CMP
    array([[1.2051213 , 1.49844327],
           [1.49844327, 1.86319675]])

We also provide :class:`.KronFastScanner` for performing an even faster
inference across several (millions, for example) covariates independently.
Please, follow the next section for details.

Association scan
================

Let X be a n×c matrix, Mⱼ a n×mⱼ matrix for the j-th candidate set, and 𝐲 an array of
outcome::

    𝐲 ∼ 𝓝(X𝜷ⱼ + Mⱼ𝛂ⱼ, sⱼ(v₀GGᵀ + v₁I))

The parameters 𝜷ⱼ, 𝛂ⱼ, and sⱼ are fit via maximum likelihood, while the remaining
parameters v₀ and v₁ are held fixed. The v₀ and v₁ values are first found by applying
:class:`.LMM`.

.. doctest::

    >>> scanner = lmm.get_fast_scanner()
    >>> M = [[1.5, 0.1], [-0.2, 0.4], [0.0, 1.0], [-3.4, 0.6]]
    >>> lml, eff0, eff1, scale = scanner.scan(M)
    >>> lml  # doctest: +FLOAT_CMP
    -0.7322976913217882
    >>> print(eff0)  # doctest: +FLOAT_CMP
    [-0.42323051]
    >>> print(eff1)  # doctest: +FLOAT_CMP
    [-0.05913491  0.37079162]
    >>> scale  # doctest: +FLOAT_CMP
    0.4629376687687552

For the null case (i.e., when there is not candidate set Mⱼ), the log of the marginal
likelihood and the values of 𝜷 and s can be found as follows.

.. doctest::

    >>> scanner.null_lml()  # doctest: +FLOAT_CMP
    -2.272623408618055
    >>> scanner.null_effsizes()  # doctest: +FLOAT_CMP
    array([0.06646503])
    >>> scanner.null_scale()  # doctest: +FLOAT_CMP
    1.0

We also provide a fast scanner for the multi-trait case, :class:`.KronFastScanner`.
Its model is given by ::

    vec(Y) ∼ 𝓝(vec(Y) | (A ⊗ F)vec(𝚩ⱼ) + (Aⱼ ⊗ Fⱼ)vec(𝚨ⱼ), sⱼ(C₀ ⊗ GGᵀ + C₁ ⊗ I)).

As before, the parameters C₀ and C₁ are set to the values found by :class:`.Kron2Sum`.
A candidate set is defined by providing the matrices Aⱼ and Fⱼ.
The parameters 𝚩ⱼ, 𝚨ⱼ, and sⱼ are found via maximum likelihood.

.. doctest::

    >>> mscanner = mlmm.get_fast_scanner()
    >>> A = random.randn(2, 5)
    >>> F = random.randn(5, 3)
    >>> lml, eff0, eff1, scale = mscanner.scan(A, F)
    >>> lml
    81.87502470339223
    >>> eff0
    array([[ 0.01482133,  0.45189275],
           [ 0.43706748, -0.71162517],
           [ 0.52595486, -1.59740035]])
    >>> eff1
    array([[ 0.03868156, -0.77199913, -0.09643554, -0.53973775,  1.03149564],
           [ 0.05780863, -0.24744739, -0.11882984, -0.19331759,  0.74964805],
           [ 0.01051071, -1.61751886, -0.0654883 , -1.09931899,  1.51034738]])
    >>> scale
    2.220446049250313e-16

API
===

.. currentmodule:: glimix_core.lmm

.. autosummary::
  :toctree: _autosummary
  :template: class.rst

  FastScanner
  Kron2Sum
  KronFastScanner
  LMM

Implementation
==============

Single-trait
------------

The LMM model :eq:`lmm1` can be equivalently written as

.. math::

    𝐲 ∼ 𝓝\Big(~ X𝜷;~
      s \big(
        (1-𝛿)
          K +
        𝛿 I
      \big)
    ~\Big),

and we thus have v₀ = s (1 - 𝛿) and v₁ = s 𝛿.
Consider the economic eigendecomposition of K:

.. math::

    \overbrace{[\mathrm Q₀ \quad \mathrm Q₁]}^{\mathrm Q}
        \overbrace{\left[\begin{array}{cc}
            \mathrm S₀ & 𝟎\\
            𝟎 & 𝟎
        \end{array}\right]}^{\mathrm S}
    \left[\begin{array}{c}
        \mathrm Q₀ᵀ \\
        \mathrm Q₁ᵀ
    \end{array}\right] = K

and let

.. math::

    \mathrm D = \left[
        \begin{array}{cc}
          (1-𝛿)\mathrm S₀ + 𝛿I_r & 𝟎\\
          𝟎 & 𝛿I_{n-r}
        \end{array}
        \right].

We thus have

.. math::

    ((1-𝛿)K + 𝛿I)⁻¹ =
        \mathrm Q \mathrm D⁻¹
        \mathrm Qᵀ.

A diagonal covariance-matrix can then be used to define an equivalent
marginal likelihood:

.. math::

    𝓝\left(\mathrm Qᵀ 𝐲 ~|~
               \mathrm Qᵀ X𝜷,~
               s \mathrm D \right).


Taking the logarithm and expanding it gives us

.. math::

   log p(𝐲) &=
       -\frac{n}{2} log 2\pi - \frac{n}{2} log s
           - \frac{1}{2}log|\mathrm D|\\
       &- \frac{1}{2} (\mathrm Qᵀ𝐲)ᵀ s⁻¹
           \mathrm D⁻¹(\mathrm Qᵀ 𝐲)\\
       &+ (\mathrm Qᵀ𝐲)ᵀ
           s⁻¹ \mathrm D⁻¹
           (\mathrm Qᵀ X 𝜷)\\
       &- \frac{1}{2} (\mathrm Qᵀ
           X 𝜷)ᵀ s⁻¹ \mathrm D⁻¹
           (\mathrm Qᵀ X 𝜷).

Setting the derivative of log(p(𝐲)) over effect sizes equal
to zero leads to solutions 𝜷^* from equation

.. math::

   (\mathrm QᵀX 𝜷^*)ᵀ
       \mathrm D⁻¹ (\mathrm Qᵀ X) =
       (\mathrm Qᵀ𝐲)ᵀ\mathrm D⁻¹
       (\mathrm QᵀX).

Replacing it back to the log of the marginal likelihood gives us the simpler
formulae

.. math::

   log p(𝐲) &=
       -\frac{n}{2} log 2\pi - \frac{n}{2} log s
           - \frac{1}{2}log|\mathrm D|\\
       & +\frac{1}{2} (\mathrm Qᵀ𝐲)ᵀ s⁻¹
           \mathrm D⁻¹\mathrm Qᵀ
           (X𝜷^* - 𝐲).


In the extreme case where 𝜷^* is such that
𝐲 = X𝜷^*, the maximum is attained
as :math:`s \rightarrow 0`.

Setting the derivative of :math:`log p(𝐲; 𝜷^*)` over
scale equal to zero leads to the maximum

.. math::

   s^* = n⁻¹
       (\mathrm Qᵀ𝐲)ᵀ
           \mathrm D⁻¹\mathrm Qᵀ
           (𝐲 - X𝜷^*).


We offer the possibility to use either :math:`s^*` found via the
above equation or a scale defined by the user.
In the first case we have a further simplification of the log of the marginal
likelihood:

.. math::

   log p(𝐲; 𝜷^*, s^*) &=
       -\frac{n}{2} log 2\pi - \frac{n}{2} log s^*
           - \frac{1}{2}log|\mathrm D| - \frac{n}{2}\\
           &= log 𝓝(\text{Diag}(\sqrt{s^*\mathrm D})
               ~|~ 𝟎, s^*\mathrm D).

.. _mtlmm-impl:

Uncorrelated multi-trait
------------------------

The extension to multiple traits becomes easy under the assumption that the traits are
uncorrelated, as assumed in this section.
Let m be the number of traits.
We stack all the different traits into

.. math::

    𝐲 = \text{vec}\left(\left[ 𝐲₀ ~ 𝐲₁ ~\cdots~ 𝐲_m
        \right] \right)

Similarly, we have the covariates, fixed-effect sizes, and the assembled covariance
matrix :math:`\tilde{K}` as

.. math::

    X = \text{vec}\left(\left[ X₀ ~ X₁ ~\cdots~ X_m
        \right] \right),

.. math::

    𝜷 = \text{vec}\left(
        \left[
            𝜷₀ ~ 𝜷₁ ~\cdots~ 𝜷_m
        \right]
    \right),

and

.. math::

    \tilde{K} =
        \left[
            \begin{array}{ccc}
                K & 𝟎 & \cdots \\
                \vdots    & \ddots    &        \\
                𝟎 &           & K
            \end{array}
        \right],

where K is repeated \|m\| times in :math:`\tilde{K}`.
We thus consider the model

.. math::
    :label: mtlmm1

    𝐲 ∼ 𝓝\Big(~
        X𝜷;~
        v₀ \tilde{K} + v₁ I_{nm}
    ~\Big),

which is the model :eq:`lmm1` with multi-trait structure and uncorrelated traits.

We use the fact that the eigendecomposition of :math:`\tilde{K}` can be
computed as fast as the eigendecomposition of K:

.. math::

    \overbrace{[\mathrm Q ~ \cdots ~ \mathrm Q]}^{\tilde{\mathrm Q}}
        \overbrace{\left[\begin{array}{ccc}
            \mathrm S & 𝟎 & \cdots \\
            \vdots    & \ddots    &   \\
            𝟎 &           & \mathrm S
        \end{array}\right]}^{\tilde{\mathrm S}}
    \left[\begin{array}{c}
        \mathrm Qᵀ \\
        \vdots \\
        \mathrm Qᵀ
    \end{array}\right] = \tilde{K}.

Let

.. math::

    \tilde{\mathrm D} = \left[
        \begin{array}{cc}
          (1-𝛿)\tilde{\mathrm S}₀ + 𝛿I_r & 𝟎\\
          𝟎 & 𝛿I_{n-r}
        \end{array}
        \right].

We thus have

.. math::

    ((1-𝛿)\tilde{K} + 𝛿I_{nm})⁻¹ =
            \left[\begin{array}{ccc}
                \mathrm Q\mathrm D⁻¹\mathrm Qᵀ & 𝟎 & \cdots\\
                \vdots    & \ddots    & \\
                𝟎 &           & \mathrm Q\mathrm D⁻¹\mathrm Qᵀ
            \end{array}\right]

A diagonal covariance-matrix can then be used to define an equivalent marginal
likelihood:

.. math::

    𝓝\left(\tilde{\mathrm Q}ᵀ 𝐲 ~|~
               \tilde{\mathrm Q}ᵀ
               X𝜷
               ,~
               s\left[\begin{array}{ccc}
                    \mathrm D & 𝟎 & \cdots\\
                    \vdots    & \ddots    & \\
                    𝟎 &           & \mathrm D
                \end{array}\right]
               \right).

The optimal effect sizes are solutions to the equation

.. math::

   \left[(\mathrm QᵀXᵢ𝜷ᵢ^*)ᵀ
       \mathrm D⁻¹ (\mathrm Qᵀ Xᵢ)\right] =
       \left[(\mathrm Qᵀ𝐲ᵢ)ᵀ\mathrm D⁻¹
       (\mathrm QᵀXᵢ)\right],

for :math:`i \in \{1, \dots, m\}`.
Setting the derivative of log(p(𝐲; 𝜷^*)) over scale equal
to zero leads to the maximum

.. math::

   s^* = (nm)⁻¹
       \left[(\mathrm Qᵀ𝐲ᵢ)ᵀ
           \mathrm D⁻¹\right]\left[\mathrm Qᵀ
           (𝐲ᵢ - Xᵢ𝜷ᵢ^*)\right].


Using the above, optimal scale leads to a further simplification of the log of the
marginal likelihood:

.. math::

   log p(𝐲; 𝜷^*, s^*) &=
       -\frac{nm}{2} log 2\pi - \frac{nm}{2} log s^*
           - \frac{1}{2}log|\tilde{\mathrm D}| - \frac{nm}{2}\\
           &= log 𝓝(\text{Diag}(\sqrt{s^*\tilde{\mathrm D}})
               ~|~ 𝟎, s^*\tilde{\mathrm D}).

Multi-trait
-----------

Please, refer to :class:`glimix_core.lmm.Kron2Sum`.

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
