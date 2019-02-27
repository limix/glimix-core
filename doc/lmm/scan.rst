Association scan
================

Let :math:`\mathrm X` be a \|n\|-by-\|d\| matrix,
:math:`\mathrm M` a \|n\|-by-\|c\| matrix, and
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