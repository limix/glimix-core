from numpy import asarray, block, clip, inf, kron

from glimix_core._util import rsolve, unvec, vec

from .._util import cache, log2pi, safe_log


class KronFastScanner:
    """
    Approximated fast inference over several covariates.

    Specifically, it maximizes the log of the marginal likelihood ::

        log(p(Y)ⱼ) = log𝓝(vec(Y) | (A ⊗ F)vec(𝚩ⱼ) + (Aⱼ ⊗ Fⱼ)vec(𝚨ⱼ), sⱼK),

    where K = C₀ ⊗ GGᵀ + C₁ ⊗ I and ⱼ index the candidates set. For performance purpose,
    we optimise only the fixed-effect sizes and scale parameters. Therefore, K is fixed
    throughout the process.
    """

    def __init__(self, Y, A, F, G, terms):
        """
        Constructor.

        Parameters
        ----------
        Y : (n, p) array_like
            Outcome matrix.
        A : (n, n) array_like
            Trait-by-trait design matrix.
        F : (n, c) array_like
            Covariates design matrix.
        G : (n, r) array_like
            Matrix G from the GGᵀ term.
        terms : dict
            Pre-computed terms.
        """

        self._Y = asarray(Y, float)
        self._A = asarray(A, float)
        self._F = asarray(F, float)
        self._G = asarray(G, float)
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

    @cache
    def null_lml(self):
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
        scale = self.null_scale()
        return self._static_lml() / 2 - np * safe_log(scale) / 2 - np / 2

    @cache
    def null_effsizes(self):
        """
        Optimal 𝚩 according to the marginal likelihood.

        It is compute by solving the equation ::

            MᵀK⁻¹Mvec(𝚩) = MᵀK⁻¹𝐲,

        for 𝐲 = vec(Y) and M = (A ⊗ F)vec(𝚩).

        Returns
        -------
        effsizes : ndarray
            Optimal 𝚩.
        """
        ntraits = self._Y.shape[1]
        return unvec(rsolve(self._MKiM, self._MKiy), (-1, ntraits))

    @cache
    def null_scale(self):
        """
        Optimal s according to the marginal likelihood.

        The optimal s is given by

            s = (n·p)⁻¹𝐲ᵀK⁻¹(𝐲 - 𝐦),

        where 𝐦 = (A ⊗ F)vec(𝚩) and 𝚩 is optimal.

        Returns
        -------
        scale : float
            Optimal scale.
        """
        np = self._nsamples * self._ntraits
        b = vec(self.null_effsizes())
        mKiy = b.T @ self._MKiy
        sqrtdot = self._yKiy - mKiy
        scale = sqrtdot / np
        return scale

    def scan(self, A1, F1):
        """
        LML, fixed-effect sizes, and scale of the candidate set.

        Parameters
        ----------
        A1 : (p, e) array_like
            Trait-by-environments design matrix.
        F1 : (n, m) array_like
            Variants set matrix.

        Returns
        -------
        lml : float
            Log of the marginal likelihood for the set.
        effsizes0 : (c, p) ndarray
            Fixed-effect sizes for the covariates.
        effsizes1 : (m, e) ndarray
            Fixed-effect sizes for the candidates.
        scale : float
            Optimal scale.
        """
        from numpy import empty
        from numpy.linalg import multi_dot
        from numpy_sugar import epsilon
        from scipy.linalg import cho_solve

        A1 = asarray(A1, float)
        F1 = asarray(F1, float)

        if A1.shape[1] == 0:
            return self.null_lml(), self.null_effsizes(), empty((0,)), self.null_scale()

        F1F1 = F1.T @ F1
        FF1 = self._F.T @ F1
        AWA1 = self._WA.T @ A1
        A1W = A1.T @ self._W
        GF1 = self._G.T @ F1

        MRiM1 = kron(AWA1, FF1)
        M1RiM1 = kron(A1W @ A1, F1F1)

        M1Riy = vec(multi_dot([F1.T, self._Y, A1W.T]))
        XRiM1 = kron(self._WL0.T @ A1, GF1)
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
        effsizes1 = unvec(beta[cp:], (F1.shape[1], A1.shape[1]))

        np = self._nsamples * self._ntraits
        sqrtdot = self._yKiy - mKiy
        scale = clip(sqrtdot / np, epsilon.tiny, inf)
        lmls = self._static_lml() / 2 - np * safe_log(scale) / 2 - np / 2
        return lmls, effsizes0, effsizes1, scale

    @cache
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
        return self._F.shape[1]

    @property
    @cache
    def _MKiM(self):
        return self._MRiM - self._XRiM.T @ self._ZiXRiM

    @property
    @cache
    def _MKiy(self):
        return self._MRiy - self._XRiM.T @ self._ZiXRiy
