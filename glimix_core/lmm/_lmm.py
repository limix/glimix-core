from math import exp

from numpy import asarray, atleast_2d, dot, log, maximum, ndarray, zeros
from numpy.linalg import multi_dot, slogdet
from optimix import Function, Scalar

from .._util import (
    SVD,
    cached_property,
    economic_qs_zeros,
    log2pi,
    nice_inv,
    numbers,
    rsolve,
)
from ._b import B
from ._lmm_scan import FastScanner


class LMM(Function):
    """
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

        𝐲 ∼ 𝓝(𝚇𝜷, 𝑠((1-𝛿)𝙺 + 𝛿𝙸)),

    and we thus have v₀ = s (1 - 𝛿) and v₁ = s 𝛿. Consider the economic
    eigendecomposition of 𝙺::

        𝙺 = [𝚀₀  𝚀₁] [𝚂₀  𝟎] [𝚀₀ᵀ]
                     [ 𝟎  𝟎] [𝚀₁ᵀ]

    and let

        𝙳 = [(1-𝛿)𝚂₀ + 𝛿𝙸₀   𝟎 ]
            [      𝟎        𝛿𝙸₁].

    In order to eliminate the need of 𝚀₁, note that 𝚀𝚀ᵀ = 𝙸 implies that ::

        𝚀₁𝚀₁ᵀ = 𝙸 - 𝚀₀𝚀₀ᵀ.

    We will need to solve ((1-𝛿)𝙺 + 𝛿𝙸)𝐱 = 𝐮 for 𝐱. Let 𝙳₀ = ((1-𝛿)𝚂₀ + 𝛿𝙸₀) and let us
    define ::

        𝙱 = 𝚀₀𝙳₀⁻¹𝚀₀ᵀ                   if 𝛿=0, and
        𝙱 = 𝚀₀𝙳₀⁻¹𝚀₀ᵀ + 𝛿⁻¹(𝙸 - 𝚀₀𝚀₀ᵀ)  if 𝛿>0.

    We therefore have ::

        𝐱 = 𝙱𝐮.
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
            Economic eigendecompositon in form of ``((Q0, ), S0)`` of a
            covariance matrix ``K``.
        restricted : bool
            ``True`` for restricted maximum likelihood optimization; ``False``
            otherwise. Defaults to ``False``.
        """
        from numpy_sugar import is_all_finite

        logistic = Scalar(0.0)
        logistic.listen(self._delta_update)
        logistic.bounds = (-numbers.logmax, +numbers.logmax)
        Function.__init__(self, "LMM", logistic=logistic)
        self._logistic = logistic

        y = asarray(y, float).ravel()
        if not is_all_finite(y):
            raise ValueError("There are non-finite values in the outcome.")

        if len(y) == 0:
            raise ValueError("The outcome array is empty.")

        X = atleast_2d(asarray(X, float).T).T
        if not is_all_finite(X):
            raise ValueError("There are non-finite values in the covariates matrix.")

        self._optimal = {"beta": False, "scale": False}
        if QS is None:
            QS = economic_qs_zeros(len(y))
            self._B = B(QS[0][0], QS[1], 0.0, 1.0)
            self.delta = 1.0
            logistic.fix()
        else:
            self._B = B(QS[0][0], QS[1], 0.5, 0.5)
            self.delta = 0.5

        if QS[0][0].shape[0] != len(y):
            msg = "Sample size differs between outcome and covariance decomposition."
            raise ValueError(msg)

        if y.shape[0] != X.shape[0]:
            msg = "Sample size differs between outcome and covariates."
            raise ValueError(msg)

        self._y = y
        self._Q0 = QS[0][0]
        self._S0 = QS[1]
        self._Xsvd = SVD(X)
        self._tbeta = zeros(self._Xsvd.rank)
        self._scale = 1.0
        self._fix = {"beta": False, "scale": False}
        self._restricted = restricted

    @property
    def beta(self) -> ndarray:
        """
        Fixed-effect sizes.

        Returns
        -------
        effect-sizes
            Optimal fixed-effect sizes.

        Notes
        -----
        Setting the derivative of log(p(𝐲)) over effect sizes equal
        to zero leads to solutions 𝜷 from equation ::

            𝚇ᵀ𝙱𝚇𝜷 = 𝚇ᵀ𝙱𝐲
        """
        return rsolve(self._Xsvd.Vt, rsolve(self._Xsvd.US, self.mean()))

    @beta.setter
    def beta(self, beta):
        beta = asarray(beta, float).ravel()
        self._tbeta[:] = self._Xsvd.Vt @ beta
        self._optimal["beta"] = False
        self._optimal["scale"] = False

    @property
    def beta_covariance(self):
        """
        Estimates the covariance-matrix of the optimal beta.

        Returns
        -------
        beta-covariance : ndarray
            𝑠(𝚇ᵀ𝙱𝚇)⁻¹.

        References
        ----------
        .. Rencher, A. C., & Schaalje, G. B. (2008). Linear models in statistics. John
           Wiley & Sons.
        """
        A = nice_inv(self._tXTBtX)
        VT = self._Xsvd.Vt
        H = rsolve(VT, A)
        return rsolve(VT, H.T) * self.scale

    def fix(self, param: str):
        """
        Disable parameter optimization.

        Parameters
        ----------
        param
            Possible values are ``"delta"``, ``"beta"``, and ``"scale"``.
        """
        if param == "delta":
            super()._fix("logistic")
        else:
            self._fix[param] = True

    def unfix(self, param: str):
        """
        Enable parameter optimization.

        Parameters
        ----------
        param
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
            s(1 - 𝛿).
        """
        return self.scale * (1 - self.delta)

    @property
    def v1(self) -> float:
        """
        Second variance.

        Returns
        -------
        v1
            s𝛿.
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
            self._maximize_scalar(desc="LMM", rtol=1e-6, atol=1e-6, verbose=verbose)

        if not self._fix["beta"]:
            self._update_beta()

        if not self._fix["scale"]:
            self._update_scale()

    def get_fast_scanner(self) -> FastScanner:
        """
        Return :class:`.FastScanner` for association scan.

        Returns
        -------
        fast-scanner
            Instance of a class designed to perform very fast association scan.
        """
        v0 = self.v0
        v1 = self.v1
        QS = ((self._Q0,), v0 * self._S0)
        return FastScanner(self._y, self.X, QS, v1)

    def value(self) -> float:
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
    def nsamples(self) -> int:
        """
        Number of samples.
        """
        return len(self._y)

    @property
    def ncovariates(self) -> int:
        """
        Number of covariates.
        """
        return self._Xsvd.A.shape[1]

    def lml(self) -> float:
        """
        Log of the marginal likelihood.

        Returns
        -------
        lml
            Log of the marginal likelihood.

        Notes
        -----
        The log of the marginal likelihood is given by ::

            2⋅log(p(𝐲)) = -n⋅log(2π) - n⋅log(s) - log|𝙳|
                -  (𝚀₀ᵀ𝐲)ᵀ(s𝙳₀)⁻¹(𝚀₀ᵀ𝐲)  -  (𝐲)ᵀ(s𝛿)⁻¹(𝐲)  +  (𝚀₀ᵀ𝐲)ᵀ(s𝛿)⁻¹(𝚀₀ᵀ𝐲)
                - (𝚀₀ᵀ𝚇𝜷)ᵀ(s𝙳₀)⁻¹(𝚀₀ᵀ𝚇𝜷) - (𝚇𝜷)ᵀ(s𝛿)⁻¹(𝚇𝜷) + (𝚀₀ᵀ𝚇𝜷)ᵀ(s𝛿)⁻¹(𝚀₀ᵀ𝚇𝜷)
                + 2(𝚀₀ᵀ𝐲)ᵀ(s𝙳₀)⁻¹(𝚇𝜷)    + 2(𝐲)ᵀ(s𝛿)⁻¹(𝚇𝜷) - 2(𝚀₀ᵀ𝐲)ᵀ(s𝛿)⁻¹(𝚀₀ᵀ𝚇𝜷)

        By using the optimal 𝜷, the log of the marginal likelihood can be rewritten
        as::

            2⋅log(p(𝐲)) = -n⋅log(2π) - n⋅log(s) - log|𝙳| + (𝚀₀ᵀ𝐲)ᵀ(s𝙳₀)⁻¹𝚀₀ᵀ(𝚇𝜷 - 𝐲)
                        + (𝐲)ᵀ(s𝛿)⁻¹(𝚇𝜷 - 𝐲) - (𝚀₀ᵀ𝐲)ᵀ(s𝛿)⁻¹𝚀₀ᵀ(𝚇𝜷 - 𝐲).

        In the extreme case where 𝜷 is such that 𝐲 = 𝚇𝜷, the maximum is attained as
        s→0.

        For optimals 𝜷 and s, the log of the marginal likelihood can be further
        simplified to ::

            2⋅log(p(𝐲; 𝜷, s)) = -n⋅log(2π) - n⋅log s - log|𝙳| - n.
        """
        reml = (self._logdetXX - self._logdetH()) / 2
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
        return self._Xsvd.A

    @property
    def delta(self) -> float:
        """
        Variance ratio between ``K`` and ``I``.
        """
        from numpy_sugar import epsilon

        v = float(self._logistic.value)

        if v > 0.0:
            v = 1 / (1 + exp(-v))
        else:
            v = exp(v)
            v = v / (v + 1.0)

        return min(max(v, epsilon.tiny), 1 - epsilon.tiny)

    @delta.setter
    def delta(self, delta):
        from numpy_sugar import epsilon

        delta = min(max(delta, epsilon.tiny), 1 - epsilon.tiny)
        self._logistic.value = log(delta / (1 - delta))
        self._optimal["beta"] = False
        self._optimal["scale"] = False

    @property
    def scale(self) -> float:
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

            s = n⁻¹(Qᵀ𝐲)ᵀD⁻¹ Qᵀ(𝐲-𝚇𝜷).

        In the case of restricted marginal likelihood ::

            s = (n-c)⁻¹(Qᵀ𝐲)ᵀD⁻¹ Qᵀ(𝐲-𝚇𝜷),

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

        Formally, 𝐦 = 𝚇𝜷.

        Returns
        -------
        mean : ndarray
            Mean of the prior.
        """
        return self._Xsvd.US @ self._tbeta

    def covariance(self):
        """
        Covariance of the prior.

        Returns
        -------
        covariance : ndarray
            v₀𝙺 + v₁𝙸.
        """
        from numpy_sugar.linalg import ddot, sum2diag

        Q0 = self._Q0
        S0 = self._S0
        return sum2diag(dot(ddot(Q0, self.v0 * S0), Q0.T), self.v1)

    def _delta_update(self):
        self._optimal["beta"] = False
        self._optimal["scale"] = False
        delta = self.delta
        self._B.set_variances(1 - delta, delta)

    @cached_property
    def _logdetXX(self):
        """
        log(｜XᵀX｜).
        """
        if not self._restricted:
            return 0.0

        ldet = slogdet(self._Xsvd.US.T @ self._Xsvd.US)
        if ldet[0] != 1.0:
            raise ValueError("The determinant of XᵀX should be positive.")
        return ldet[1]

    def _logdetH(self):
        """
        log(｜H｜) for H = s⁻¹XᵀQD⁻¹QᵀX.
        """
        if not self._restricted:
            return 0.0
        ldet = slogdet(self._tXTBtX / self.scale)
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
        lml -= self._logdetD
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
        lml -= self._logdetD
        my = self.mean() - self._y
        lml -= my.T @ self._B.dot(my) / s
        return lml / 2

    @property
    def _df(self):
        """
        Degrees of freedom.
        """
        if not self._restricted:
            return self.nsamples
        return self.nsamples - self._Xsvd.rank

    def _optimal_scale_using_optimal_beta(self):
        from numpy_sugar import epsilon

        assert self._optimal["beta"]
        s = self._yTBy - self._yTBtX @ self._tbeta
        return maximum(s / self._df, epsilon.small)

    def _update_beta(self):
        assert not self._fix["beta"]
        if self._optimal["beta"]:
            return

        self._tbeta[:] = rsolve(self._tXTBtX, self._yTBtX)
        self._optimal["beta"] = True
        self._optimal["scale"] = False

    def _update_scale(self):
        from numpy_sugar import epsilon

        if self._optimal["beta"]:
            self._scale = self._optimal_scale_using_optimal_beta()
        else:
            p0 = self._yTBy - 2 * self._yTBtX @ self._tbeta
            p1 = multi_dot((self._tbeta, self._tXTBtX, self._tbeta))
            self._scale = maximum((p0 + p1) / self._df, epsilon.small)

        self._optimal["scale"] = True

    @property
    def _logdetD(self):
        v = 0.0
        d = self.delta
        rank = self._S0.size
        if rank > 0:
            v += log((1 - d) * self._S0 + d).sum()
        rankdef = self._y.shape[0] - rank
        v += rankdef * log(d)
        return v

    @property
    def _yTBy(self):
        return self._y.T @ self._B.dot(self._y)

    @property
    def _yTBtX(self):
        return self._y.T @ self._B.dot(self._Xsvd.US)

    @property
    def _tXTBtX(self):
        return self._Xsvd.US.T @ self._B.dot(self._Xsvd.US)
