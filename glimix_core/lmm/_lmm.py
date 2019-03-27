from numpy import asarray, clip, dot, exp
from numpy.linalg import solve

from ._core import LMMCore
from ._scan import FastScanner


class LMM(LMMCore):
    r"""
    Fast Linear Mixed Models inference via maximum likelihood.

    Examples
    --------
    .. doctest::

        >>> from numpy import array
        >>> from numpy_sugar.linalg import economic_qs_linear
        >>> from glimix_core.lmm import LMM
        >>>
        >>> X = array([[1, 2], [3, -1]], float)
        >>> QS = economic_qs_linear(X)
        >>> covariates = array([[1], [1]])
        >>> y = array([-1, 2], float)
        >>> lmm = LMM(y, covariates, QS)
        >>> lmm.fit(verbose=False)
        >>> print('%.3f' % lmm.lml())
        -3.649

    One can also specify which parameters should be fitted:

    .. doctest::

        >>> from numpy import array
        >>> from numpy_sugar.linalg import economic_qs_linear
        >>> from glimix_core.lmm import LMM
        >>>
        >>> X = array([[1, 2], [3, -1]], float)
        >>> QS = economic_qs_linear(X)
        >>> covariates = array([[1], [1]])
        >>> y = array([-1, 2], float)
        >>> lmm = LMM(y, covariates, QS)
        >>> lmm.fix('delta')
        >>> lmm.fix('scale')
        >>> lmm.delta = 0.5
        >>> lmm.scale = 1
        >>> lmm.fit(verbose=False)
        >>> print('%.3f' % lmm.lml())
        -3.832
        >>> lmm.unfix('delta')
        >>> lmm.fit(verbose=False)
        >>> print('%.3f' % lmm.lml())
        -3.713

    Notes
    -----

    The LMM model can be equivalently written as ::

        𝐲 ∼ 𝓝(X𝜷, s((1-𝛿)K + 𝛿I)),

    and we thus have v₀ = s (1 - 𝛿) and v₁ = s 𝛿.
    Consider the economic eigendecomposition of K:

    .. math::

        \overbrace{[\mathrm Q₀ \quad \mathrm Q₁]}^{\mathrm Q}
            \overbrace{\left[\begin{array}{cc}
                \mathrm S₀ & 𝟎\\
                        𝟎  & 𝟎
            \end{array}\right]}^{\mathrm S}
        \left[\begin{array}{c}
            \mathrm Q₀ᵀ \\
            \mathrm Q₁ᵀ
        \end{array}\right] = \mathrm K

    and let

    .. math::

        \mathrm D = \left[
            \begin{array}{cc}
                (1-𝛿)\mathrm S₀ + 𝛿\mathrm I & 𝟎\\
                𝟎                            & 𝛿\mathrm I
            \end{array}
        \right].

    We thus have

    .. math::

        ((1-𝛿)\mathrm K + 𝛿I)⁻¹ = \mathrm Q\mathrm D⁻¹\mathrm Qᵀ.

    A diagonal covariance-matrix can then be used to define an equivalent
    marginal likelihood:

    .. math::

        𝓝\left(\mathrm Qᵀ 𝐲 ~|~ \mathrm Qᵀ \mathrm X𝜷,~ s \mathrm D \right).

    """

    def __init__(self, y, X, QS=None):
        """
        Constructor.

        Parameters
        ----------
        y : array_like
            Outcome.
        X : array_like
            Covariates as a two-dimensional array.
        QS : tuple
            Economic eigendecompositon in form of ``((Q0, Q1), S0)`` of a
            covariance matrix ``K``.
        """
        from numpy_sugar import is_all_finite

        LMMCore.__init__(self, y, X, QS)

        if not is_all_finite(y):
            raise ValueError("There are non-finite values in the outcome.")

    def _get_delta(self):
        from numpy_sugar import epsilon

        v = clip(self._variables.get("logistic").value, -20, 20)
        x = 1 / (1 + exp(-v))
        return clip(x, epsilon.tiny, 1 - epsilon.tiny)

    @property
    def beta(self):
        """
        Get or set fixed-effect sizes.
        """
        return LMMCore.beta.fget(self)

    @beta.setter
    def beta(self, beta):
        LMMCore.beta.fset(self, beta)

    def copy(self):
        """
        Return a copy of this object.

        This is useful for performing new inference based on the results
        of the copied object, as the new LMM object will start its inference
        from the initial solution found in the copied object.

        Returns
        -------
        :class:`.LMM`
            Copy of this object.
        """
        o = LMM.__new__(LMM)

        LMMCore.__init__(o, self._y, self.X, self._QS)
        o.delta = self.delta
        if self.isfixed("delta"):
            o.fix("delta")

        setattr(o, "_fix_scale", self._fix_scale)
        setattr(o, "_scale", self._scale)
        setattr(o, "_fix_beta", self._fix_beta)
        setattr(o, "_verbose", self._verbose)

        return o

    def isfixed(self, var_name):
        """
        Return whether a variable it is fixed or not.

        Parameters
        ----------
        var_name : str
            Possible values are ``"delta"`` and ``"scale"``.

        Returns
        -------
        bool
            ``True`` if fixed; ``False`` otherwise.
        """
        if var_name not in ["delta", "scale", "beta"]:
            msg = "Possible values are 'delta', 'scale', and 'beta'."
            raise ValueError(msg)
        if var_name == "delta":
            return super(LMM, self)._isfixed("logistic")
        if var_name == "beta":
            return self._fix_beta
        return self._fix_scale

    @property
    def v0(self):
        """
        First variance.

        Returns
        -------
        float
            1 - 𝛿.
        """
        return self.scale * (1 - self.delta)

    @property
    def v1(self):
        """
        Second variance.

        Returns
        -------
        float
            𝛿.
        """
        return self.scale * self.delta

    def fit(self, verbose=True):
        """
        Maximise the marginal likelihood.

        Parameters
        ----------
        verbose : bool, optional
            ``True`` for progress output; ``False`` otherwise.
            Defaults to ``True``.
        """
        self._verbose = verbose
        if not self.isfixed("delta"):
            self._maximize_scalar(desc="LMM", verbose=verbose)
        self.delta = self._get_delta()
        self._update_fixed_effects()
        self._verbose = False

    def fix(self, var_name):
        """
        Disable the optimisation of a given variable.

        Parameters
        ----------
        var_name : str
            Possible values are ``"delta"`` and ``"scale"``.
        """
        if var_name not in ["delta", "scale", "beta"]:
            msg = "Possible values are 'delta', 'scale', and 'beta'."
            raise ValueError(msg)

        if var_name == "delta":
            super(LMM, self)._fix("logistic")
        elif var_name == "beta":
            self._fix_beta = True
        else:
            if not self._fix_scale:
                self._scale = self.scale
            self._fix_scale = True

    @property
    def fixed_effects_variance(self):
        """
        Variance of the fixed-effects.

        It is defined as the empirical variance of the prior mean.

        Returns
        -------
        var : float
            Estimated variance of the fixed-effects.
        """
        return self.mean.var()

    def get_fast_scanner(self):
        """
        Return :class:`.FastScanner` for association scan.

        Returns
        -------
        :class:`.FastScanner`
            Instance of a class designed to perform very fast association scan.
        """
        v0 = self.v0
        v1 = self.v1
        QS = (self._QS[0], v0 * self._QS[1])
        return FastScanner(self._y, self.X, QS, v1)

    def lml(self):
        self.delta = self._get_delta()
        return LMMCore.lml(self)

    @property
    def mean(self):
        """
        Estimated mean X𝞫.

        Returns
        -------
        mean : ndarray
            Mean of the prior.
        """
        return LMMCore.mean.fget(self)

    @property
    def scale(self):
        if self._fix_scale:
            return self._scale
        return LMMCore.scale.fget(self)

    @scale.setter
    def scale(self, v):
        self._scale = v

    def unfix(self, var_name):
        """
        Enable the optimisation of a given variable.

        Parameters
        ----------
        var_name : str
            Possible values are ``"delta"`` and ``"scale"``.
        """
        if var_name not in ["delta", "scale", "beta"]:
            msg = "Possible values are 'delta', 'scale', and 'beta'."
            raise ValueError(msg)
        if var_name == "delta":
            super(LMM, self)._unfix("logistic")
        elif var_name == "beta":
            self._fix_beta = False
        else:
            self._fix_scale = False

    def value(self, *_):
        self.delta = self._get_delta()
        return self.lml()

    @property
    def X(self):
        return LMMCore.X.fget(self)

    @X.setter
    def X(self, X):
        LMMCore.X.fset(self, X)

    def predictive_mean(self, Xstar, ks, kss):
        mstar = self.mean_star(Xstar)
        ks = self.covariance_star(ks)
        m = self.mean
        K = LMMCore.covariance(self)
        return mstar + dot(ks, solve(K, self._y - m))

    def predictive_covariance(self, Xstar, ks, kss):
        kss = self.variance_star(kss)
        ks = self.covariance_star(ks)
        K = LMMCore.covariance(self)
        ktk = solve(K, ks.T)
        b = []
        for i in range(len(kss)):
            b += [dot(ks[i, :], ktk[:, i])]
        b = asarray(b)
        return kss - b
