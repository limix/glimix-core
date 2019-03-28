import warnings
from functools import reduce

from numpy import asarray, asfortranarray, kron, log, sqrt, tensordot, trace
from numpy.linalg import matrix_rank, slogdet

from glimix_core._util import cache, log2pi, unvec, vec
from glimix_core.cov import Kron2SumCov
from glimix_core.mean import KronMean
from optimix import Function

from ._kron2sum_scan import KronFastScanner


class Kron2Sum(Function):
    """
    LMM for multi-traits fitted via restricted maximum likelihood.

    This implementation follows the work published in [CA05]_.
    Let n, c, and p be the number of samples, covariates, and traits, respectively.
    The outcome variable Y is a n×p matrix distributed according to::

        vec(Y) ~ N((A ⊗ F) vec(B), K = C₀ ⊗ GGᵀ + C₁ ⊗ I).

    A and F are design matrices of dimensions p×p and n×c provided by the user,
    where F is the usual matrix of covariates commonly used in single-trait models.
    B is a c×p matrix of fixed-effect sizes per trait.
    G is a n×r matrix provided by the user and I is a n×n identity matrices.
    C₀ and C₁ are both symmetric matrices of dimensions p×p, for which C₁ is
    guaranteed by our implementation to be of full rank.
    The parameters of this model are the matrices B, C₀, and C₁.

    For implementation purpose, we make use of the following definitions:

    - M = A ⊗ F
    - H = MᵀK⁻¹M
    - Yₓ = LₓY
    - Yₕ = YₓLₕᵀ
    - Mₓ = LₓF
    - Mₕ = (LₕA) ⊗ Mₓ
    - mₕ = Mₕvec(B)

    where Lₓ and Lₕ are defined in :class:`glimix_core.cov.Kron2SumCov`.

    References
    ----------
    .. [CA05] Casale, F. P., Rakitsch, B., Lippert, C., & Stegle, O. (2015). Efficient
       set tests for the genetic analysis of correlated traits. Nature methods, 12(8),
       755.
    """

    def __init__(self, Y, A, F, G, rank=1, restricted=False):
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
        rank : optional, int
            Maximum rank of matrix C₀. Defaults to ``1``.
        """
        Y = asfortranarray(Y, float)
        yrank = matrix_rank(Y)
        if Y.shape[1] > yrank:
            warnings.warn(
                f"Y is not full column rank: rank(Y)={yrank}. "
                + "Convergence might be problematic.",
                UserWarning,
            )

        A = asarray(A, float)
        F = asarray(F, float)
        Frank = matrix_rank(F)
        if F.shape[1] > Frank:
            warnings.warn(
                f"F is not full column rank: rank(F)={Frank}. "
                + "Convergence might be problematic.",
                UserWarning,
            )

        G = asarray(G, float)
        self._Y = Y
        self._cov = Kron2SumCov(G, Y.shape[1], rank)
        self._mean = KronMean(A, F)
        self._cache = {"terms": None}
        self._cov.listen(self._parameters_update)
        self._restricted = restricted
        composite = [("C0", self._cov.C0), ("C1", self._cov.C1)]
        Function.__init__(self, "Kron2Sum", composite=composite)

    def _parameters_update(self):
        self._cache["terms"] = None

    @property
    @cache
    def _GY(self):
        return self._cov.Ge.T @ self._Y

    @property
    @cache
    def _GG(self):
        return self._cov.Ge.T @ self._cov.Ge

    @property
    @cache
    def _trGG(self):
        from numpy_sugar.linalg import trace2

        return trace2(self._cov.Ge, self._cov.Ge.T)

    @property
    @cache
    def _GGGG(self):
        return self._GG @ self._GG

    @property
    @cache
    def _GGGY(self):
        return self._GG @ self._GY

    @property
    @cache
    def _FF(self):
        return self._mean.F.T @ self._mean.F

    @property
    @cache
    def _GF(self):
        return self._cov.Ge.T @ self._mean.F

    @property
    @cache
    def _FGGG(self):
        return self._GF.T @ self._GG

    @property
    @cache
    def _FGGY(self):
        return self._GF.T @ self._GY

    @property
    @cache
    def _FGGF(self):
        return self._GF.T @ self._GF

    @property
    @cache
    def _FY(self):
        return self._mean.F.T @ self._Y

    @property
    def _terms(self):
        from scipy.linalg import cho_factor, cho_solve
        from numpy_sugar.linalg import ddot, sum2diag

        if self._cache["terms"] is not None:
            return self._cache["terms"]

        L0 = self._cov.C0.L
        S, U = self._cov.C1.eigh()
        W = ddot(U, 1 / S) @ U.T
        S = 1 / sqrt(S)
        Y = self._Y
        A = self._mean.A

        WL0 = W @ L0
        YW = Y @ W
        WA = W @ A
        L0WA = L0.T @ WA

        Z = kron(L0.T @ WL0, self._GG)
        Z = sum2diag(Z, 1)
        Lz = cho_factor(Z, lower=True)

        # 𝐲ᵀR⁻¹𝐲 = vec(YW)ᵀ𝐲
        yRiy = (YW * self._Y).sum()
        # MᵀR⁻¹M = AᵀWA ⊗ FᵀF
        MRiM = kron(A.T @ WA, self._FF)
        # XᵀR⁻¹𝐲 = vec(GᵀYWL₀)
        XRiy = vec(self._GY @ WL0)
        # XᵀR⁻¹M = (L₀ᵀWA) ⊗ (GᵀF)
        XRiM = kron(L0WA, self._GF)
        # MᵀR⁻¹𝐲 = vec(FᵀYWA)
        MRiy = vec(self._FY @ WA)

        ZiXRiM = cho_solve(Lz, XRiM)
        ZiXRiy = cho_solve(Lz, XRiy)

        MRiXZiXRiy = ZiXRiM.T @ XRiy
        MRiXZiXRiM = XRiM.T @ ZiXRiM

        yKiy = yRiy - XRiy @ ZiXRiy
        MKiy = MRiy - MRiXZiXRiy
        H = MRiM - MRiXZiXRiM
        Lh = cho_factor(H)
        b = cho_solve(Lh, MKiy)
        B = unvec(b, (self.ncovariates, -1))
        self._mean.B = B
        XRim = XRiM @ b

        ZiXRim = ZiXRiM @ b
        mRiy = b.T @ MRiy
        mRim = b.T @ MRiM @ b

        logdetK = log(Lz[0].diagonal()).sum() * 2
        logdetK -= 2 * log(S).sum() * self.nsamples

        mKiy = mRiy - XRim.T @ ZiXRiy
        mKim = mRim - XRim.T @ ZiXRim

        self._cache["terms"] = {
            "logdetK": logdetK,
            "mKiy": mKiy,
            "mKim": mKim,
            "b": b,
            "Z": Z,
            "B": B,
            "Lz": Lz,
            "S": S,
            "W": W,
            "WA": WA,
            "YW": YW,
            "WL0": WL0,
            "yRiy": yRiy,
            "MRiM": MRiM,
            "XRiy": XRiy,
            "XRiM": XRiM,
            "ZiXRiM": ZiXRiM,
            "ZiXRiy": ZiXRiy,
            "ZiXRim": ZiXRim,
            "MRiy": MRiy,
            "mRim": mRim,
            "mRiy": mRiy,
            "XRim": XRim,
            "yKiy": yKiy,
            "Lh": Lh,
            "MRiXZiXRiy": MRiXZiXRiy,
            "MRiXZiXRiM": MRiXZiXRiM,
        }
        return self._cache["terms"]

    def get_fast_scanner(self):
        """
        Return :class:`.FastScanner` for association scan.

        Returns
        -------
        :class:`.FastScanner`
            Instance of a class designed to perform very fast association scan.
        """
        terms = self._terms
        return KronFastScanner(self._Y, self._mean.A, self._mean.F, self._cov.Ge, terms)

    @property
    def mean(self):
        """
        Mean 𝐦 = (A ⊗ F) vec(B).

        Returns
        -------
        mean : ndarray
            𝐦.
        """
        self._terms
        return self._mean.value()

    @property
    def B(self):
        """
        Fixed-effect sizes B from 𝐦 = (A ⊗ F) vec(B).

        Returns
        -------
        fixed-effects : ndarray
            B from 𝐦 = (A ⊗ F) vec(B).
        """
        self._terms
        return asarray(self._mean.B, float)

    @property
    def cov(self):
        """
        Covariance K = C₀ ⊗ GGᵀ + C₁ ⊗ I.

        Returns
        -------
        covariance : Kron2SumCov
        """
        return self._cov

    @property
    def nsamples(self):
        """
        Number of samples, n.
        """
        return self._Y.shape[0]

    @property
    def ntraits(self):
        """
        Number of traits, p.
        """
        return self._Y.shape[1]

    @property
    def ncovariates(self):
        """
        Number of covariates, c.
        """
        return self._mean.F.shape[1]

    def value(self):
        """
        Log of the marginal likelihood.
        """
        return self.lml()

    def gradient(self):
        """
        Gradient of the log of the marginal likelihood.
        """
        return self._lml_gradient()

    @property
    @cache
    def _logdet_MM(self):
        if not self._restricted:
            return 0.0

        M = self._mean.AF
        ldet = slogdet(M.T @ M)
        if ldet[0] != 1.0:
            raise ValueError("The determinant of MᵀM should be positive.")
        return ldet[1]

    @property
    def _logdetH(self):
        if not self._restricted:
            return 0.0
        terms = self._terms
        MKiM = terms["MRiM"] - terms["XRiM"].T @ terms["ZiXRiM"]
        return slogdet(MKiM)[1]

    @property
    def _logdetK(self):
        terms = self._terms
        S = terms["S"]
        Lz = terms["Lz"]

        cov_logdet = log(Lz[0].diagonal()).sum() * 2
        cov_logdet -= 2 * log(S).sum() * self.nsamples
        return cov_logdet

    def lml(self):
        """
        Log of the marginal likelihood.

        Let 𝐲 = vec(Y), M = A⊗F, and H = MᵀK⁻¹M. The restricted log of the marginal
        likelihood is given by [R07]_::

            2⋅log(p(𝐲)) = -(n⋅p - c⋅p) log(2π) + log(｜MᵀM｜) - log(｜K｜) - log(｜H｜)
                - (𝐲-𝐦)ᵀ K⁻¹ (𝐲-𝐦),

        where 𝐦 = M𝛃 for 𝛃 = H⁻¹MᵀK⁻¹𝐲.

        For implementation purpose, let X = (L₀ ⊗ G) and R = (L₁ ⊗ I)(L₁ ⊗ I)ᵀ.
        The covariance can be written as::

            K = XXᵀ + R.

        From the Woodbury matrix identity, we have

            𝐲ᵀK⁻¹𝐲 = 𝐲ᵀR⁻¹𝐲 - 𝐲ᵀR⁻¹XZ⁻¹XᵀR⁻¹𝐲,

        where Z = I + XᵀR⁻¹X. Note that R⁻¹ = (U₁S₁⁻¹U₁ᵀ) ⊗ I and ::

            XᵀR⁻¹𝐲 = (L₀ᵀW ⊗ Gᵀ)𝐲 = vec(GᵀYWL₀),

        where W = U₁S₁⁻¹U₁ᵀ. The term GᵀY can be calculated only once and it will form a
        r×p matrix. We similarly have ::

            XᵀR⁻¹M = (L₀ᵀWA) ⊗ (GᵀF),

        for which GᵀF is pre-computed.

        The log-determinant of the covariance matrix is given by

            log(｜K｜) = log(｜Z｜) - log(｜R⁻¹｜) = log(｜Z｜) - 2·n·log(｜U₁S₁⁻½｜).

        The log of the marginal likelihood can be rewritten as::

            2⋅log(p(𝐲)) = -(n⋅p - c⋅p) log(2π) + log(｜MᵀM｜)
            - log(｜Z｜) + 2·n·log(｜U₁S₁⁻½｜)
            - log(｜MᵀR⁻¹M - MᵀR⁻¹XZ⁻¹XᵀR⁻¹M｜)
            - 𝐲ᵀR⁻¹𝐲 + (𝐲ᵀR⁻¹X)Z⁻¹(XᵀR⁻¹𝐲)
            - 𝐦ᵀR⁻¹𝐦 + (𝐦ᵀR⁻¹X)Z⁻¹(XᵀR⁻¹𝐦)
            + 2𝐲ᵀR⁻¹𝐦 - 2(𝐲ᵀR⁻¹X)Z⁻¹(XᵀR⁻¹𝐦).

        Returns
        -------
        lml : float
            Log of the marginal likelihood.

        References
        ----------
        .. [R07] LaMotte, L. R. (2007). A direct derivation of the REML likelihood
           function. Statistical Papers, 48(2), 321-327.
        """
        terms = self._terms
        yKiy = terms["yKiy"]
        mKiy = terms["mKiy"]
        mKim = terms["mKim"]

        lml = -self._df * log2pi + self._logdet_MM - self._logdetK
        lml -= self._logdetH
        lml += -yKiy - mKim + 2 * mKiy

        return lml / 2

    def _lml_gradient(self):
        """
        Gradient of the log of the marginal likelihood.

        Let 𝐲 = vec(Y), 𝕂 = K⁻¹∂(K)K⁻¹, and H = MᵀK⁻¹M. The gradient is given by::

            2⋅∂log(p(𝐲)) = -tr(K⁻¹∂K) - tr(H⁻¹∂H) + 𝐲ᵀ𝕂𝐲 - 𝐦ᵀ𝕂(2⋅𝐲-𝐦)
                - 2⋅(𝐦-𝐲)ᵀK⁻¹∂(𝐦).

        Observe that

            ∂𝛃 = -H⁻¹(∂H)𝛃 - H⁻¹Mᵀ𝕂𝐲 and ∂H = -Mᵀ𝕂M.

        Let Z = I + XᵀR⁻¹X and 𝓡 = R⁻¹(∂K)R⁻¹. We use Woodbury matrix identity to
        write ::

            𝐲ᵀ𝕂𝐲 = 𝐲ᵀ𝓡𝐲 - 2(𝐲ᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲) + (𝐲ᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
            Mᵀ𝕂M = Mᵀ𝓡M - 2(Mᵀ𝓡X)Z⁻¹(XᵀR⁻¹M) + (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹M)
            Mᵀ𝕂𝐲 = Mᵀ𝓡𝐲 - (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡𝐲) - (Mᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
                  + (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
            H⁻¹   = MᵀR⁻¹M - (MᵀR⁻¹X)Z⁻¹(XᵀR⁻¹M),

        where we have used parentheses to separate expressions
        that we will compute separately. For example, we have ::

            𝐲ᵀ𝓡𝐲 = 𝐲ᵀ(U₁S₁⁻¹U₁ᵀ ⊗ I)(∂C₀ ⊗ GGᵀ)(U₁S₁⁻¹U₁ᵀ ⊗ I)𝐲
                  = 𝐲ᵀ(U₁S₁⁻¹U₁ᵀ∂C₀ ⊗ G)(U₁S₁⁻¹U₁ᵀ ⊗ Gᵀ)𝐲
                  = vec(GᵀYU₁S₁⁻¹U₁ᵀ∂C₀)ᵀvec(GᵀYU₁S₁⁻¹U₁ᵀ),

        when the derivative is over the parameters of C₀. Otherwise, we have

            𝐲ᵀ𝓡𝐲 = vec(YU₁S₁⁻¹U₁ᵀ∂C₁)ᵀvec(YU₁S₁⁻¹U₁ᵀ).

        The above equations can be more compactly written as

            𝐲ᵀ𝓡𝐲 = vec(EᵢᵀYW∂Cᵢ)ᵀvec(EᵢᵀYW),

        where W = U₁S₁⁻¹U₁ᵀ, E₀ = G, and E₁ = I. We will now just state the results for
        the other instances of the aBc form, which follow similar derivations::

            Xᵀ𝓡X = (L₀ᵀW∂CᵢWL₀) ⊗ (GᵀEᵢEᵢᵀG)
            Mᵀ𝓡y = (AᵀW∂Cᵢ⊗FᵀEᵢ)vec(EᵢᵀYW) = vec(FᵀEᵢEᵢᵀYW∂CᵢWA)
            Mᵀ𝓡X = AᵀW∂CᵢWL₀ ⊗ FᵀEᵢEᵢᵀG
            Mᵀ𝓡M = AᵀW∂CᵢWA ⊗ FᵀEᵢEᵢᵀF
            Xᵀ𝓡𝐲 = GᵀEᵢEᵢᵀYW∂CᵢWL₀

        From Woodbury matrix identity and Kronecker product properties we have ::

            tr(K⁻¹∂K) = tr[W∂Cᵢ]tr[EᵢEᵢᵀ] - tr[Z⁻¹(Xᵀ𝓡X)]
            tr(H⁻¹∂H) = - tr[(MᵀR⁻¹M)(Mᵀ𝕂M)] + tr[(MᵀR⁻¹X)Z⁻¹(XᵀR⁻¹M)(Mᵀ𝕂M)]

        Note also that ::

            ∂𝛃 = H⁻¹Mᵀ𝕂M𝛃 - H⁻¹Mᵀ𝕂𝐲.

        Returns
        -------
        C0.Lu : ndarray
            Gradient of the log of the marginal likelihood over C₀ parameters.
        C1.Lu : ndarray
            Gradient of the log of the marginal likelihood over C₁ parameters.
        """
        from scipy.linalg import cho_solve

        terms = self._terms
        dC0 = self._cov.C0.gradient()["Lu"]
        dC1 = self._cov.C1.gradient()["Lu"]

        b = terms["b"]
        W = terms["W"]
        Lh = terms["Lh"]
        Lz = terms["Lz"]
        WA = terms["WA"]
        WL0 = terms["WL0"]
        YW = terms["YW"]
        MRiM = terms["MRiM"]
        MRiy = terms["MRiy"]
        XRiM = terms["XRiM"]
        XRiy = terms["XRiy"]
        ZiXRiM = terms["ZiXRiM"]
        ZiXRiy = terms["ZiXRiy"]

        WdC0 = mdot(W, dC0)
        WdC1 = mdot(W, dC1)

        AWdC0 = mdot(WA.T, dC0)
        AWdC1 = mdot(WA.T, dC1)

        # Mᵀ𝓡M
        MR0M = mkron(mdot(AWdC0, WA), self._FGGF)
        MR1M = mkron(mdot(AWdC1, WA), self._FF)

        # Mᵀ𝓡X
        MR0X = mkron(mdot(AWdC0, WL0), self._FGGG)
        MR1X = mkron(mdot(AWdC1, WL0), self._GF.T)

        # Mᵀ𝓡𝐲 = (AᵀW∂Cᵢ⊗FᵀEᵢ)vec(EᵢᵀYW) = vec(FᵀEᵢEᵢᵀYW∂CᵢWA)
        MR0y = vec(mdot(self._FGGY, mdot(WdC0, WA)))
        MR1y = vec(mdot(self._FY, WdC1, WA))

        # Xᵀ𝓡X
        XR0X = mkron(mdot(WL0.T, dC0, WL0), self._GGGG)
        XR1X = mkron(mdot(WL0.T, dC1, WL0), self._GG)

        # Xᵀ𝓡𝐲
        XR0y = vec(mdot(self._GGGY, WdC0, WL0))
        XR1y = vec(mdot(self._GY, WdC1, WL0))

        # 𝐲ᵀ𝓡𝐲 = vec(EᵢᵀYW∂Cᵢ)ᵀvec(EᵢᵀYW)
        yR0y = vec(mdot(self._GY, WdC0)).T @ vec(self._GY @ W)
        yR1y = (YW.T * mdot(self._Y, WdC1).T).T.sum(axis=(0, 1))

        ZiXR0X = cho_solve(Lz, XR0X)
        ZiXR1X = cho_solve(Lz, XR1X)
        ZiXR0y = cho_solve(Lz, XR0y)
        ZiXR1y = cho_solve(Lz, XR1y)

        # Mᵀ𝕂y = Mᵀ𝓡𝐲 - (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡𝐲) - (Mᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
        #       + (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
        MK0y = MR0y - mdot(XRiM.T, ZiXR0y) - mdot(MR0X, ZiXRiy)
        MK0y += mdot(XRiM.T, ZiXR0X, ZiXRiy)
        MK1y = MR1y - mdot(XRiM.T, ZiXR1y) - mdot(MR1X, ZiXRiy)
        MK1y += mdot(XRiM.T, ZiXR1X, ZiXRiy)

        # 𝐲ᵀ𝕂𝐲 = 𝐲ᵀ𝓡𝐲 - 2(𝐲ᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲) + (𝐲ᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹𝐲)
        yK0y = yR0y - 2 * XR0y.T @ ZiXRiy + ZiXRiy.T @ mdot(XR0X, ZiXRiy)
        yK1y = yR1y - 2 * XR1y.T @ ZiXRiy + ZiXRiy.T @ mdot(XR1X, ZiXRiy)

        # Mᵀ𝕂M = Mᵀ𝓡M - (Mᵀ𝓡X)Z⁻¹(XᵀR⁻¹M) - (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡M)
        #       + (MᵀR⁻¹X)Z⁻¹(Xᵀ𝓡X)Z⁻¹(XᵀR⁻¹M)
        MR0XZiXRiM = mdot(MR0X, ZiXRiM)
        MK0M = MR0M - MR0XZiXRiM - MR0XZiXRiM.transpose([1, 0, 2])
        MK0M += mdot(ZiXRiM.T, XR0X, ZiXRiM)
        MR1XZiXRiM = mdot(MR1X, ZiXRiM)
        MK1M = MR1M - MR1XZiXRiM - MR1XZiXRiM.transpose([1, 0, 2])
        MK1M += mdot(ZiXRiM.T, XR1X, ZiXRiM)

        MK0m = mdot(MK0M, b)
        mK0y = b.T @ MK0y
        mK0m = b.T @ MK0m
        MK1m = mdot(MK1M, b)
        mK1y = b.T @ MK1y
        mK1m = b.T @ MK1m
        XRim = XRiM @ b
        MRim = MRiM @ b

        db = {"C0.Lu": cho_solve(Lh, MK0m - MK0y), "C1.Lu": cho_solve(Lh, MK1m - MK1y)}

        grad = {
            "C0.Lu": -trace(WdC0) * self._trGG + trace(ZiXR0X),
            "C1.Lu": -trace(WdC1) * self.nsamples + trace(ZiXR1X),
        }

        if self._restricted:
            grad["C0.Lu"] += cho_solve(Lh, MK0M).diagonal().sum(1)
            grad["C1.Lu"] += cho_solve(Lh, MK1M).diagonal().sum(1)

        mKiM = MRim.T - XRim.T @ ZiXRiM
        yKiM = MRiy.T - XRiy.T @ ZiXRiM

        grad["C0.Lu"] += yK0y - 2 * mK0y + mK0m - 2 * mdot(mKiM, db["C0.Lu"])
        grad["C0.Lu"] += 2 * mdot(yKiM, db["C0.Lu"])
        grad["C1.Lu"] += yK1y - 2 * mK1y + mK1m - 2 * mdot(mKiM, db["C1.Lu"])
        grad["C1.Lu"] += 2 * mdot(yKiM, db["C1.Lu"])

        grad["C0.Lu"] /= 2
        grad["C1.Lu"] /= 2

        return grad

    def fit(self, verbose=True):
        """
        Maximise the marginal likelihood.

        Parameters
        ----------
        verbose : bool, optional
            ``True`` for progress output; ``False`` otherwise.
            Defaults to ``True``.
        """
        self._maximize(verbose=verbose, pgtol=1e-5, factr=1e8)

    @property
    def _df(self):
        np = self.nsamples * self.ntraits
        if not self._restricted:
            return np
        cp = self.ncovariates * self.ntraits
        return np - cp


def _dot(a, b):
    r = tensordot(a, b, axes=([min(1, a.ndim - 1)], [0]))
    if a.ndim > b.ndim:
        if r.ndim == 3:
            return r.transpose([0, 2, 1])
        return r
    return r


def mdot(*args):
    return reduce(_dot, args)


def _sum(a):
    return a.sum(axis=(0, 1))


def mkron(a, b):
    if a.ndim == 3:
        return kron(a.transpose([2, 0, 1]), b).transpose([1, 2, 0])
    return kron(a, b)
