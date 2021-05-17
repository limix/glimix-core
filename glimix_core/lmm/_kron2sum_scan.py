from numpy import asarray, block, clip, inf, kron, sqrt
from numpy.linalg import pinv

from glimix_core._util import rsolve, unvec, vec

from .._util import cached_property, log2pi, safe_log


class KronFastScanner:
    """
    Approximated fast inference over several covariates.

    Specifically, it maximizes the log of the marginal likelihood ::

        log(p(Y)ⱼ) = log𝓝(vec(Y) | (A ⊗ X)vec(𝚩ⱼ) + (Aⱼ ⊗ Xⱼ)vec(𝚨ⱼ), sⱼK),

    where K = C₀ ⊗ GGᵀ + C₁ ⊗ I and ⱼ index the candidates set. For performance purpose,
    we optimise only the fixed-effect sizes and scale parameters. Therefore, K is fixed
    throughout the process.
    """

    def __init__(self, Y, A, X, G, terms):
        """
        Constructor.

        Parameters
        ----------
        Y : (n, p) array_like
            Outcome matrix.
        A : (n, n) array_like
            Trait-by-trait design matrix.
        X : (n, c) array_like
            Covariates design matrix.
        G : (n, r) array_like
            Matrix G from the GGᵀ term.
        terms : dict
            Pre-computed terms.
        """

        self._Y = asarray(Y, float)
        self._A = asarray(A, float)
        self._X = asarray(X, float)
        self._G = asarray(G, float)
        self._H = terms["H"]
        self._logdetK = terms["logdetK"]
        self._W = terms["W"]
        self._yKiy = terms["yKiy"]
        self._WA = terms["WA"]
        self._WL0 = terms["WL0"]
        self._Lz = terms["Lz"]
        self._XRiM = terms["XRiM"]
        self._ZiXRiy = terms["ZiXRiy"]
        self._ZiXRiM = terms["ZiXRiM"]
        self._MRiM = terms["MRiM"]
        self._MRiXZiXRiM = terms["MRiXZiXRiM"]
        self._MRiy = terms["MRiy"]
        self._MRiXZiXRiy = terms["MRiXZiXRiy"]

    def null_lml(self):
        return self._null_lml

    @cached_property
    def _null_lml(self):
        """
        Log of the marginal likelihood for the null hypothesis.

        It is implemented as ::

            2·log(p(Y)) = -n·p·log(2𝜋s) - log｜K｜ - n·p,

        for which s and 𝚩 are optimal.

        Returns
        -------
        lml : float
            Log of the marginal likelihood.
        """
        np = self._nsamples * self._ntraits
        scale = self.null_scale
        return self._static_lml / 2 - np * safe_log(scale) / 2 - np / 2

    @cached_property
    def null_beta(self):
        """
        Optimal 𝛃 according to the marginal likelihood.

        It is compute by solving the equation ::

            MᵀK⁻¹M𝛃 = MᵀK⁻¹𝐲,

        for 𝐲 = vec(Y) and M = (A ⊗ X)vec(𝚩).

        Returns
        -------
        effsizes : ndarray
            Optimal 𝛃.
        """
        return rsolve(self._MKiM, self._MKiy)

    @cached_property
    def null_beta_covariance(self):
        """
        Covariance of the optimal 𝛃 according to the marginal likelihood.

        Returns
        -------
        effsizes-covariance : ndarray
            s(MᵀK⁻¹M)⁻¹.
        """
        return self.null_scale * pinv(self._H)

    @cached_property
    def null_beta_se(self):
        """
        Standard errors of the optimal 𝛃.

        Returns
        -------
        beta_se : ndarray
            Square root of the diagonal of the beta covariance.
        """
        return sqrt(self.null_beta_covariance.diagonal())

    @cached_property
    def null_scale(self):
        """
        Optimal s according to the marginal likelihood.

        The optimal s is given by

            s = (n·p)⁻¹𝐲ᵀK⁻¹(𝐲 - 𝐦),

        where 𝐦 = (A ⊗ X)vec(𝚩) and 𝚩 is optimal.

        Returns
        -------
        scale : float
            Optimal scale.
        """
        np = self._nsamples * self._ntraits
        b = vec(self.null_beta)
        mKiy = b.T @ self._MKiy
        sqrtdot = self._yKiy - mKiy
        scale = sqrtdot / np
        return scale

    def scan(self, A1, X1):
        """
        LML, fixed-effect sizes, and scale of the candidate set.

        Parameters
        ----------
        A1 : (p, e) array_like
            Trait-by-environments design matrix.
        X1 : (n, m) array_like
            Variants set matrix.

        Returns
        -------
        lml : float
            Log of the marginal likelihood for the set.
        effsizes0 : (c, p) ndarray
            Fixed-effect sizes for the covariates.
        effsizes0_se : (c, p) ndarray
            Fixed-effect size standard errors for the covariates.
        effsizes1 : (m, e) ndarray
            Fixed-effect sizes for the candidates.
        effsizes1_se : (m, e) ndarray
            Fixed-effect size standard errors for the candidates.
        scale : float
            Optimal scale.
        """
        from numpy import empty
        from numpy.linalg import multi_dot
        from numpy_sugar import epsilon, is_all_finite
        from scipy.linalg import cho_solve

        A1 = asarray(A1, float)
        X1 = asarray(X1, float)

        if not is_all_finite(A1):
            raise ValueError("A1 parameter has non-finite elements.")

        if not is_all_finite(X1):
            raise ValueError("X1 parameter has non-finite elements.")

        if A1.shape[1] == 0:
            beta_se = sqrt(self.null_beta_covariance.diagonal())
            return {
                "lml": self._null_lml,
                "effsizes0": unvec(self.null_beta, (self._ncovariates, -1)),
                "effsizes0_se": unvec(beta_se, (self._ncovariates, -1)),
                "effsizes1": empty((0,)),
                "effsizes1_se": empty((0,)),
                "scale": self.null_scale,
            }

        X1X1 = X1.T @ X1
        XX1 = self._X.T @ X1
        AWA1 = self._WA.T @ A1
        A1W = A1.T @ self._W
        GX1 = self._G.T @ X1

        MRiM1 = kron(AWA1, XX1)
        M1RiM1 = kron(A1W @ A1, X1X1)

        M1Riy = vec(multi_dot([X1.T, self._Y, A1W.T]))
        XRiM1 = kron(self._WL0.T @ A1, GX1)
        ZiXRiM1 = cho_solve(self._Lz, XRiM1)

        MRiXZiXRiM1 = self._XRiM.T @ ZiXRiM1
        M1RiXZiXRiM1 = XRiM1.T @ ZiXRiM1
        M1RiXZiXRiy = XRiM1.T @ self._ZiXRiy

        T0 = [[self._MRiM, MRiM1], [MRiM1.T, M1RiM1]]
        T1 = [[self._MRiXZiXRiM, MRiXZiXRiM1], [MRiXZiXRiM1.T, M1RiXZiXRiM1]]
        T2 = [self._MRiy, M1Riy]
        T3 = [self._MRiXZiXRiy, M1RiXZiXRiy]

        MKiM = block(T0) - block(T1)
        MKiy = block(T2) - block(T3)
        beta = rsolve(MKiM, MKiy)

        mKiy = beta.T @ MKiy
        cp = self._ntraits * self._ncovariates
        effsizes0 = unvec(beta[:cp], (self._ncovariates, self._ntraits))
        effsizes1 = unvec(beta[cp:], (X1.shape[1], A1.shape[1]))

        np = self._nsamples * self._ntraits
        sqrtdot = self._yKiy - mKiy
        scale = clip(sqrtdot / np, epsilon.tiny, inf)
        lml = self._static_lml / 2 - np * safe_log(scale) / 2 - np / 2

        effsizes_se = sqrt(clip(scale * pinv(MKiM).diagonal(), epsilon.tiny, inf))
        effsizes0_se = unvec(effsizes_se[:cp], (self._ncovariates, self._ntraits))
        effsizes1_se = unvec(effsizes_se[cp:], (X1.shape[1], A1.shape[1]))

        return {
            "lml": lml,
            "effsizes0": effsizes0,
            "effsizes1": effsizes1,
            "scale": scale,
            "effsizes0_se": effsizes0_se,
            "effsizes1_se": effsizes1_se,
        }

    @cached_property
    def _static_lml(self):
        np = self._nsamples * self._ntraits
        static_lml = -np * log2pi - self._logdetK
        return static_lml

    @property
    def _nsamples(self):
        return self._Y.shape[0]

    @property
    def _ntraits(self):
        return self._Y.shape[1]

    @property
    def _ncovariates(self):
        return self._X.shape[1]

    @cached_property
    def _MKiM(self):
        return self._MRiM - self._XRiM.T @ self._ZiXRiM

    @cached_property
    def _MKiy(self):
        return self._MRiy - self._XRiM.T @ self._ZiXRiy
