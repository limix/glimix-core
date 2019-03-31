from numpy import (
    asarray,
    atleast_2d,
    clip,
    concatenate,
    dot,
    errstate,
    exp,
    full,
    log,
    maximum,
    sum as npsum,
    zeros,
)
from numpy.linalg import inv, lstsq, slogdet

from glimix_core._util import cache, log2pi
from optimix import Function, Scalar

from .._util import economic_qs_zeros, numbers
from ._lmm_scan import FastScanner


class LMM(Function):
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

    We thus have ::

        ((1-𝛿)K + 𝛿I)⁻¹ = QD⁻¹Qᵀ.

    A diagonal covariance-matrix can then be used to define an equivalent
    marginal likelihood::

        𝓝(Qᵀ𝐲|QᵀX𝜷, sD).

    """

    def __init__(self, y, X, QS=None, restricted=False):
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
        restricted : bool
            ``True`` for restricted maximum likelihood optimization; ``False``
            otherwise. Defaults to ``False``.
        """
        from numpy_sugar import is_all_finite
        from numpy_sugar.linalg import ddot, economic_svd

        delta = Scalar(0.0)
        delta.listen(self._delta_update)
        delta.bounds = (-numbers.logmax, +numbers.logmax)
        Function.__init__(self, "LMM", logistic=delta)

        y = asarray(y, float).ravel()
        if not is_all_finite(y):
            raise ValueError("There are non-finite values in the outcome.")

        X = atleast_2d(asarray(X, float).T).T
        if not is_all_finite(X):
            raise ValueError("There are non-finite values in the covariates matrix.")

        self._optimal = {"beta": False, "scale": False}
        if QS is None:
            QS = economic_qs_zeros(len(y))
            delta.value = 1.0
            delta.fix()
        else:
            delta.value = 0.5

        if QS[0][0].shape[0] != len(y):
            msg = "Sample size differs between outcome and covariance decomposition."
            raise ValueError(msg)

        if y.shape[0] != X.shape[0]:
            msg = "Sample size differs between outcome and covariates."
            raise ValueError(msg)

        self._y = y
        self._QS = QS
        SVD = economic_svd(X)
        self._X = {"X": X, "tX": ddot(SVD[0], SVD[1]), "VT": SVD[2]}
        self._tbeta = zeros(len(SVD[1]))
        self._scale = 1.0
        self._fix = {"beta": False, "scale": False}
        self._restricted = restricted

    @property
    def beta(self):
        """
        Fixed-effect sizes.

        Returns
        -------
        effect-sizes : numpy.ndarray
            Optimal fixed-effect sizes.

        Notes
        -----
        Setting the derivative of log(p(𝐲)) over effect sizes equal
        to zero leads to solutions 𝜷 from equation ::

            (QᵀX)ᵀD⁻¹(QᵀX)𝜷 = (QᵀX)ᵀD⁻¹(Qᵀ𝐲).
        """
        from numpy_sugar.linalg import rsolve

        return rsolve(self._X["VT"], rsolve(self._X["tX"], self.mean()))

    @beta.setter
    def beta(self, beta):
        beta = asarray(beta, float).ravel()
        self._tbeta[:] = self._X["VT"] @ beta
        self._optimal["beta"] = False
        self._optimal["scale"] = False

    @property
    def beta_covariance(self):
        """
        Estimates the covariance-matrix of the optimal beta.

        Returns
        -------
        beta-covariance : ndarray
            (Xᵀ(s((1-𝛿)K + 𝛿I))⁻¹X)⁻¹.

        References
        ----------
        .. Rencher, A. C., & Schaalje, G. B. (2008). Linear models in statistics. John
           Wiley & Sons.
        """
        from numpy_sugar.linalg import ddot

        tX = self._X["tX"]
        Q = concatenate(self._QS[0], axis=1)
        S0 = self._QS[1]
        D = self.v0 * S0 + self.v1
        D = D.tolist() + [self.v1] * (len(self._y) - len(D))
        D = asarray(D)
        A = inv(tX.T @ (Q @ ddot(1 / D, Q.T @ tX)))
        VT = self._X["VT"]
        H = lstsq(VT, A, rcond=None)[0]
        return lstsq(VT, H.T, rcond=None)[0]

    def fix(self, param):
        """
        Disable parameter optimization.

        Parameters
        ----------
        param : str
            Possible values are ``"delta"``, ``"beta"``, and ``"scale"``.
        """
        if param == "delta":
            super()._fix("logistic")
        else:
            self._fix[param] = True

    def unfix(self, param):
        """
        Enable parameter optimization.

        Parameters
        ----------
        param : str
            Possible values are ``"delta"``, ``"beta"``, and ``"scale"``.
        """
        if param == "delta":
            self._unfix("logistic")
        else:
            self._fix[param] = False

    @property
    def v0(self):
        """
        First variance.

        Returns
        -------
        v0 : float
            1 - 𝛿.
        """
        return self.scale * (1 - self.delta)

    @property
    def v1(self):
        """
        Second variance.

        Returns
        -------
        v1 : float
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
        if not self._isfixed("logistic"):
            self._maximize_scalar(desc="LMM", verbose=verbose)

        if not self._fix["beta"]:
            self._update_beta()

        if not self._fix["scale"]:
            self._update_scale()

    def get_fast_scanner(self):
        """
        Return :class:`.FastScanner` for association scan.

        Returns
        -------
        fast-scanner : :class:`.FastScanner`
            Instance of a class designed to perform very fast association scan.
        """
        v0 = self.v0
        v1 = self.v1
        QS = (self._QS[0], v0 * self._QS[1])
        return FastScanner(self._y, self.X, QS, v1)

    def value(self):
        """
        Internal use only.
        """
        if not self._fix["beta"]:
            self._update_beta()

        if not self._fix["scale"]:
            self._update_scale()

        return self.lml()

    def gradient(self):
        """
        Not implemented.
        """
        raise NotImplementedError

    @property
    def nsamples(self):
        """
        Number of samples, n.
        """
        return len(self._y)

    @property
    def ncovariates(self):
        """
        Number of covariates, c.
        """
        return self._X["X"].shape[1]

    def lml(self):
        """
        Log of the marginal likelihood.

        Returns
        -------
        lml : float
            Log of the marginal likelihood.

        Notes
        -----
        The log of the marginal likelihood is given by ::

            2⋅log(p(𝐲)) = -n⋅log(2π) - n⋅log(s) - log|D| - (Qᵀ𝐲)ᵀs⁻¹D⁻¹(Qᵀ𝐲)
                        + (Qᵀ𝐲)ᵀs⁻¹D⁻¹(QᵀX𝜷)/2 - (QᵀX𝜷)ᵀs⁻¹D⁻¹(QᵀX𝜷).

        By using the optimal 𝜷, the log of the marginal likelihood can be rewritten
        as::

            2⋅log(p(𝐲)) = -n⋅log(2π) - n⋅log(s) - log|D| + (Qᵀ𝐲)ᵀs⁻¹D⁻¹Qᵀ(X𝜷-𝐲).


        In the extreme case where 𝜷 is such that 𝐲 = X𝜷, the maximum is attained as
        s→0.

        For optimals 𝜷 and s, the log of the marginal likelihood can be further
        simplified to ::

            2⋅log(p(𝐲; 𝜷, s)) = -n⋅log(2π) - n⋅log s - log|D| - n.
        """
        reml = (self._logdetXX() - self._logdetH()) / 2
        if self._optimal["scale"]:
            lml = self._lml_optimal_scale()
        else:
            lml = self._lml_arbitrary_scale()
        return lml + reml

    @property
    def X(self):
        """
        Covariates matrix.

        Returns
        -------
        X : ndarray
            Covariates.
        """
        return self._X["X"]

    @property
    def delta(self):
        """
        Variance ratio between ``K`` and ``I``.
        """
        from numpy_sugar import epsilon

        v = float(self._variables.get("logistic").value)
        with errstate(over="ignore", under="ignore"):
            v = 1 / (1 + exp(-v))
        return clip(v, epsilon.tiny, 1 - epsilon.tiny)

    @delta.setter
    def delta(self, delta):
        from numpy_sugar import epsilon

        delta = clip(delta, epsilon.tiny, 1 - epsilon.tiny)
        self._variables.set(dict(logistic=log(delta / (1 - delta))))
        self._optimal["beta"] = False
        self._optimal["scale"] = False

    @property
    def scale(self):
        """
        Scaling factor.

        Returns
        -------
        scale : float
            Scaling factor.

        Notes
        -----
        Setting the derivative of log(p(𝐲; 𝜷)), for which 𝜷 is optimal, over
        scale equal to zero leads to the maximum ::

            s = n⁻¹(Qᵀ𝐲)ᵀD⁻¹ Qᵀ(𝐲-X𝜷).

        In the case of restricted marginal likelihood ::

            s = (n-c)⁻¹(Qᵀ𝐲)ᵀD⁻¹ Qᵀ(𝐲-X𝜷),

        where s is the number of covariates.
        """
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale
        self._optimal["scale"] = False

    def mean(self):
        """
        Mean of the prior.

        Formally, 𝐦 = X𝜷.

        Returns
        -------
        mean : ndarray
            Mean of the prior.
        """
        return self._X["tX"] @ self._tbeta

    def covariance(self):
        """
        Covariance of the prior.

        Returns
        -------
        covariance : ndarray
            v₀K + v₁I.
        """
        from numpy_sugar.linalg import ddot, sum2diag

        Q0 = self._QS[0][0]
        S0 = self._QS[1]
        return sum2diag(dot(ddot(Q0, self.v0 * S0), Q0.T), self.v1)

    def _delta_update(self):
        self._optimal["beta"] = False
        self._optimal["scale"] = False

    @cache
    def _logdetXX(self):
        """
        log(｜XᵀX｜).
        """
        if not self._restricted:
            return 0.0

        ldet = slogdet(self._X["tX"].T @ self._X["tX"])
        if ldet[0] != 1.0:
            raise ValueError("The determinant of XᵀX should be positive.")
        return ldet[1]

    def _logdetH(self):
        """
        log(｜H｜) for H = s⁻¹XᵀQD⁻¹QᵀX.
        """
        if not self._restricted:
            return 0.0
        ldet = slogdet(sum(self._XTQDiQTX) / self.scale)
        if ldet[0] != 1.0:
            raise ValueError("The determinant of H should be positive.")
        return ldet[1]

    def _lml_optimal_scale(self):
        """
        Log of the marginal likelihood for optimal scale.

        Implementation for unrestricted LML::

        Returns
        -------
        lml : float
            Log of the marginal likelihood.
        """
        assert self._optimal["scale"]

        n = len(self._y)
        lml = -self._df * log2pi - self._df - n * log(self.scale)
        lml -= sum(npsum(log(D)) for D in self._D)
        return lml / 2

    def _lml_arbitrary_scale(self):
        """
        Log of the marginal likelihood for arbitrary scale.

        Returns
        -------
        lml : float
            Log of the marginal likelihood.
        """
        s = self.scale
        n = len(self._y)
        lml = -self._df * log2pi - n * log(s)
        lml -= sum(npsum(log(D)) for D in self._D)
        d = (mTQ - yTQ for (mTQ, yTQ) in zip(self._mTQ, self._yTQ))
        lml -= sum((i / j) @ i for (i, j) in zip(d, self._D)) / s

        return lml / 2

    @property
    def _df(self):
        """
        Degrees of freedom.
        """
        if not self._restricted:
            return self.nsamples
        return self.nsamples - self._X["tX"].shape[1]

    def _optimal_scale_using_optimal_beta(self):
        from numpy_sugar import epsilon

        assert self._optimal["beta"]

        yTQDiQTy = self._yTQDiQTy
        yTQDiQTm = self._yTQDiQTX
        s = sum(i - j @ self._tbeta for (i, j) in zip(yTQDiQTy, yTQDiQTm))
        return maximum(s / self._df, epsilon.small)

    def _update_beta(self):
        from numpy_sugar.linalg import rsolve

        assert not self._fix["beta"]
        if self._optimal["beta"]:
            return

        yTQDiQTm = list(self._yTQDiQTX)
        mTQDiQTm = list(self._XTQDiQTX)
        nominator = yTQDiQTm[0]
        denominator = mTQDiQTm[0]

        if len(yTQDiQTm) > 1:
            nominator += yTQDiQTm[1]
            denominator += mTQDiQTm[1]

        self._tbeta[:] = rsolve(denominator, nominator)
        self._optimal["beta"] = True
        self._optimal["scale"] = False

    def _update_scale(self):
        from numpy_sugar import epsilon

        if self._optimal["beta"]:
            self._scale = self._optimal_scale_using_optimal_beta()
        else:
            yTQDiQTy = self._yTQDiQTy
            yTQDiQTm = self._yTQDiQTX
            b = self._tbeta
            p0 = sum(i - 2 * j @ b for (i, j) in zip(yTQDiQTy, yTQDiQTm))
            p1 = sum((b @ i) @ b for i in self._XTQDiQTX)
            self._scale = maximum((p0 + p1) / self._df, epsilon.small)

        self._optimal["scale"] = True

    @property
    def _D(self):
        D = []
        n = self._y.shape[0]
        if self._QS[1].size > 0:
            D += [self._QS[1] * (1 - self.delta) + self.delta]
        if self._QS[1].size < n:
            D += [full(n - self._QS[1].size, self.delta)]
        return D

    @property
    def _XTQDiQTX(self):
        return (i / j @ i.T for (i, j) in zip(self._tXTQ, self._D))

    @property
    def _mTQ(self):
        return (self.mean().T @ Q for Q in self._QS[0] if Q.size > 0)

    @property
    def _tXTQ(self):
        return (self._X["tX"].T @ Q for Q in self._QS[0] if Q.size > 0)

    @property
    def _XTQ(self):
        return (self._X["tX"].T @ Q for Q in self._QS[0] if Q.size > 0)

    @property
    def _yTQ(self):
        return (self._y.T @ Q for Q in self._QS[0] if Q.size > 0)

    @property
    def _yTQQTy(self):
        return (yTQ ** 2 for yTQ in self._yTQ)

    @property
    def _yTQDiQTy(self):
        return (npsum(i / j) for (i, j) in zip(self._yTQQTy, self._D))

    @property
    def _yTQDiQTX(self):
        yTQ = self._yTQ
        D = self._D
        tXTQ = self._tXTQ
        return (i / j @ l.T for (i, j, l) in zip(yTQ, D, tXTQ))
