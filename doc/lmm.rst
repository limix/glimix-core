.. py:currentmodule:: glimix_core.lmm

*******************
Linear Mixed Models
*******************

Linear mixed models (LMMs) are a generalisation of linear models [#f1]_ to allow the
outcome to be described as a summation of both fixed and random effects [#f2]_.
LMM inference is implemented by the :mod:`glimix_core.lmm` module and is described here.

Introduction
============

A LMM can be described as ::

    𝐲 = X𝜷 + G𝐮 + 𝛜,

where 𝐮 ∼ 𝓝(𝟎, v₀I) is a
vector of random effects and 𝛜 are iid Normal random variables
with zero-mean and variance v₁ each.
The outcome-vector is thus distributed according to ::

    𝐲 ∼ 𝓝(X𝜷, v₀GGᵀ + v₁I).

The :class:`.LMM` class provides a FastLMM [#f3]_
implementation to perform inference over the variance parameters
v₀ and v₁ and over the vector
𝜷 of fixed-effect sizes.
Let K = GGᵀ and observe that K can be any symmetric positive definite matrix.
An instance of this class is created by providing the outcome 𝐲,
the covariates X, and the covariance K via its economic eigendecomposition.
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

The method :meth:`.LMM.fit` is called to optimise the marginal likelihood over the
fixed-effect sizes 𝜷 and over the variances v₀ and v₁.
The resulting values for the above inference are:

.. doctest::

    >>> lmm.beta  # doctest: +FLOAT_CMP
    array([0.06646503])
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
the same set of samples [#f4]_.
Let p be the number of traits.
The outcome matrix Y is the concatenation of p vectors::

    Y = [𝐲₀ 𝐲₁ ... 𝐲ₚ].

The mean definition will involve three matrices::

    M = (A ⊗ X) vec(B),

where vec(·) stacks the columns of the input matrix into a single-column matrix.
B is a c×p matrix of effect sizes for which c is the number of covariates.
A is a p×p design matrix that determines the covariance between the traits over the mean
vector.
X is a n×p design matrix of covariates.

The covariance matrix is defined by ::

    K = C₀ ⊗ GGᵀ + C₁ ⊗ I.

C₀ and C₁ are p×p symmetric matrices whose values will be optimized.
GGᵀ gives the covariance between samples, while (C₀ ⊗ GGᵀ) gives the covariance between
samples when traits are taken into account.

The outcome, mean, and covariance-matrix together define the distribution ::

    vec(Y) ~ N((A ⊗ X) vec(B), K = C₀ ⊗ GGᵀ + C₁ ⊗ I).

The parameters of the multi-trait LMM to be fit via maximum likelihood are the matrices
B, C₀, and C₁.

.. doctest::

    >>> from numpy.random import default_rng
    >>> from glimix_core.lmm import Kron2Sum
    >>>
    >>> random = default_rng(0)
    >>> n = 15
    >>> p = 2
    >>> c = 3
    >>> Y = random.normal(size=(n, p))
    >>> A = random.normal(size=(p, p))
    >>> A = A.dot(A.T)
    >>> X = random.normal(size=(n, c))
    >>> G = random.normal(size=(n, 4))
    >>>
    >>> mlmm = Kron2Sum(Y, A, X, G, restricted=False)
    >>> mlmm.fit(verbose=False)
    >>> mlmm.lml()  # doctest: +FLOAT_CMP
    -42.08309530985927

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
    >>> r = scanner.scan(M)
    >>> r["lml"]
    -0.7322976913217882
    >>> r["effsizes0"]
    array([-0.42323051])
    >>> r["effsizes1"]
    array([-0.05913491,  0.37079162])
    >>> r["scale"]
    0.4629376687687552

For the null case (i.e., when there is not candidate set Mⱼ), the log of the marginal
likelihood and the values of 𝜷 and s can be found as follows.

.. doctest::

    >>> scanner.null_lml()  # doctest: +FLOAT_CMP
    -2.272623408618055
    >>> scanner.null_beta  # doctest: +FLOAT_CMP
    array([0.06646503])
    >>> scanner.null_scale  # doctest: +FLOAT_CMP
    1.0

We also provide a fast scanner for the multi-trait case, :class:`.KronFastScanner`.
Its model is given by ::

    vec(Y) ∼ 𝓝(vec(Y) | (A ⊗ X)vec(𝚩ⱼ) + (Aⱼ ⊗ Xⱼ)vec(𝚨ⱼ), sⱼ(C₀ ⊗ GGᵀ + C₁ ⊗ I)).

As before, the parameters C₀ and C₁ are set to the values found by :class:`.Kron2Sum`.
A candidate set is defined by providing the matrices Aⱼ and Xⱼ.
The parameters 𝚩ⱼ, 𝚨ⱼ, and sⱼ are found via maximum likelihood.

.. doctest::

    >>> mscanner = mlmm.get_fast_scanner()
    >>> A = random.normal(size=(2, n))
    >>> X = random.normal(size=(n, 3))
    >>> r = mscanner.scan(A, X)
    >>> r["lml"]
    -42.74668875515792
    >>> r["effsizes0"][0, 0]
    0.0027782236846561684
    >>> r["effsizes1"][0, 0]
    -0.012738296905015198
    >>> r["scale"]
    1.0452327204999015

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

.. [#f4] Casale, F. P., Horta, D., Rakitsch, B., & Stegle, O. (2017). Joint genetic
         analysis using variant sets reveals polygenic gene-context interactions. PLoS
         genetics, 13(4), e1006693.
