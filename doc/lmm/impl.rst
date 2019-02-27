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

where :math:`\mathrm K` is repeated \|m\| times in :math:`\tilde{\mathrm K}`.
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

Multi-trait
-----------

TODO: