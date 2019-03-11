import warnings
from functools import lru_cache, reduce

from numpy import asarray, asfortranarray, diagonal, eye, kron, log, sqrt, tensordot
from numpy.linalg import matrix_rank, slogdet, solve
from scipy.linalg import cho_factor, cho_solve

from glimix_core._util import log2pi, unvec, vec
from glimix_core.cov import Kron2SumCov
from glimix_core.mean import KronMean
from numpy_sugar.linalg import ddot
from optimix import Function


class RKron2Sum(Function):
    """
    LMM for multiple traits.

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
    """

    def __init__(self, Y, A, F, G, rank=1):
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
        G = asarray(G, float)
        self._Y = Y
        # self._y = Y.ravel(order="F")
        self._cov = Kron2SumCov(G, Y.shape[1], rank)
        self._mean = KronMean(A, F)
        self._Yx = self._cov.Lx @ Y
        self._Mx = self._cov.Lx @ self._mean.F
        # self._Yxe = self._cov._Lxe @ Y
        # self._Mxe = self._cov._Lxe @ self._mean.F
        self._cache = {"terms": None}
        self._cov.listen(self._parameters_update)
        composite = [("C0", self._cov.C0), ("C1", self._cov.C1)]
        Function.__init__(self, "Kron2Sum", composite=composite)

    def _parameters_update(self):
        self._cache["terms"] = None

    @property
    def GY(self):
        return self._cov._Ge.T @ self._Y

    @property
    def GYGY(self):
        return self.GY ** 2

    @property
    def GG(self):
        return self._cov._Ge.T @ self._cov._Ge

    @property
    def GGGG(self):
        return self.GG @ self.GG

    @property
    def _FF(self):
        return self._mean.F.T @ self._mean.F

    @property
    def _GF(self):
        return self._cov._Ge.T @ self._mean.F

    @property
    def _FY(self):
        return self._mean.F.T @ self._Y

    @property
    def _terms(self):
        if self._cache["terms"] is not None:
            return self._cache["terms"]

        Lh = self._cov.Lh
        D = self._cov.D
        yh = vec(self._Yx @ Lh.T)
        yl = D * yh
        A = self._mean.A
        Mh = kron(Lh @ A, self._Mx)
        Ml = ddot(D, Mh)

        # H = MᵀK⁻¹M.
        H = Mh.T @ Ml

        # 𝐦 = M𝛃 for 𝛃 = H⁻¹MᵀK⁻¹𝐲 and H = MᵀK⁻¹M.
        # 𝛃 = H⁻¹MᵀₕD𝐲ₗ
        b = solve(H, Mh.T @ yl)
        B = unvec(b, (self.ncovariates, -1))
        self._mean.B = B

        mh = Mh @ b
        ml = D * mh

        ldetH = slogdet(H)
        if ldetH[0] != 1.0:
            raise ValueError("The determinant of H should be positive.")
        ldetH = ldetH[1]

        L0 = self._cov.C0.L
        S, U = self._cov.C1.eigh()
        w = ddot(U, 1 / S) @ U.T
        S = 1 / sqrt(S)
        US = ddot(U, S)
        X = kron(self._cov.C0.L, self._cov._G)
        R = (
            kron(self._cov.C1.L, eye(self._cov._G.shape[0]))
            @ kron(self._cov.C1.L, eye(self._cov._G.shape[0])).T
        )
        Y = self._Y
        G = self._cov._Ge
        K = X @ X.T + R
        W = kron(ddot(U, S), eye(G.shape[0]))
        Ri = W @ W.T
        y = vec(self._Y)
        WY = Y @ US
        # yRiy = vec(WY).T @ vec(WY)
        F = self._mean.F
        A = self._mean.A
        WM = kron(US.T @ A, F)
        WB = F @ B @ A.T @ US
        # G = self._cov._G
        WX = kron(US.T @ L0, G)
        Z0 = kron(L0.T @ ddot(U, S * S) @ U.T @ L0, G.T @ G)
        Z = eye(G.shape[1]) + Z0
        Lz = cho_factor(Z, lower=True)
        MKiM = WM.T @ WM - WM.T @ WX @ solve(Z, WX.T @ WM)
        b = solve(MKiM, WM.T @ vec(WY) - WM.T @ WX @ solve(Z, WX.T @ vec(WY)))
        B = unvec(b, (self.ncovariates, -1))
        Wm = WM @ b
        WL0 = w @ L0
        YW = Y @ w
        WA = w @ A
        L0WA = L0.T @ WA

        # 𝐲ᵀR⁻¹𝐲 = vec(YW)ᵀ𝐲
        yRiy = vec(YW) @ y
        # MᵀR⁻¹M = AᵀWA ⊗ FᵀF
        MRiM = kron(A.T @ WA, self._FF)
        # XᵀR⁻¹𝐲 = vec(GᵀYWL₀)
        XRiy = vec(self.GY @ WL0)
        # XᵀR⁻¹M = (L₀ᵀWA) ⊗ (GᵀF)
        XRiM = kron(L0WA, self._GF)
        # MᵀR⁻¹𝐲 = vec(FᵀYWA)
        MRiy = vec(self._FY @ WA)
        XRim = XRiM @ b
        mRiy = b.T @ MRiy
        mRim = b.T @ MRiM @ b

        ZiXRiM = cho_solve(Lz, WX.T @ WM)
        ZiXRiy = cho_solve(Lz, WX.T @ vec(WY))
        ZiXRim = ZiXRiM @ b

        self._cache["terms"] = {
            "yh": yh,
            "yl": yl,
            "Mh": Mh,
            "Ml": Ml,
            "mh": mh,
            "ml": ml,
            "ldetH": ldetH,
            "H": H,
            "b": b,
            "Z": Z,
            "WM": WM,
            "WY": WY,
            "WX": WX,
            "W": W,
            "R": R,
            "K": K,
            "B": B,
            "Wm": Wm,
            "Ri": Ri,
            "X": X,
            "US": US,
            "Lz": Lz,
            "S": S,
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
            # "yhe": yhe,
            # "Mhe": Mhe,
            # "mhe": mhe,
        }
        return self._cache["terms"]

    @property
    def mean(self):
        """
        Mean 𝐦 = (A ⊗ F) vec(B).

        Returns
        -------
        mean : KronMean
        """
        return self._mean

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
        return self.lml()

    def gradient(self):
        return self.lml_gradient()

    @property
    @lru_cache(maxsize=None)
    def _logdet_MM(self):
        M = self._mean.AF
        ldet = slogdet(M.T @ M)
        if ldet[0] != 1.0:
            raise ValueError("The determinant of MᵀM should be positive.")
        return ldet[1]

    def lml(self):
        r"""
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
        np = self.nsamples * self.ntraits
        cp = self.ncovariates * self.ntraits
        terms = self._terms
        S = terms["S"]
        Lz = terms["Lz"]
        yRiy = terms["yRiy"]
        MRiM = terms["MRiM"]
        mRim = terms["mRim"]
        mRiy = terms["mRiy"]
        XRiy = terms["XRiy"]
        XRiM = terms["XRiM"]
        XRim = terms["XRim"]
        ZiXRiM = terms["ZiXRiM"]
        ZiXRim = terms["ZiXRim"]
        ZiXRiy = terms["ZiXRiy"]

        cov_logdet = log(Lz[0].diagonal()).sum() * 2
        cov_logdet -= 2 * log(S).sum() * self.nsamples
        lml = -(np - cp) * log2pi + self._logdet_MM - cov_logdet

        MKiM = MRiM - XRiM.T @ ZiXRiM
        lml -= slogdet(MKiM)[1]

        yKiy = yRiy - XRiy @ ZiXRiy
        mKiy = mRiy - XRim.T @ ZiXRiy
        mKim = mRim - XRim.T @ ZiXRim
        lml += -yKiy - mKim + 2 * mKiy

        return lml / 2

    def lml_gradient(self):
        """
        Gradient of the log of the marginal likelihood.

        Let 𝐲 = vec(Y), 𝕂 = K⁻¹∂(K)K⁻¹, and H = MᵀK⁻¹M. The gradient is given by::

            2⋅∂log(p(𝐲)) = -tr(K⁻¹∂K) - tr(H⁻¹∂H) + 𝐲ᵀ𝕂𝐲 - 𝐦ᵀ𝕂(2⋅𝐲-𝐦)
                - 2⋅(𝐦-𝐲)ᵀK⁻¹∂(𝐦).

        Observe that

            ∂𝛃 = -H⁻¹(∂H)𝛃 - H⁻¹Mᵀ𝕂𝐲 and ∂H = -Mᵀ𝕂M.

        Let Z = I + XᵀR⁻¹X and 𝓡 = R⁻¹∂(K)R⁻¹. We use Woodbury matrix identity to
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
            Mᵀ𝓡X = AᵀW∂CᵢWL₀⊗FᵀEᵢEᵢᵀG.

        From Woodbury matrix identity and Kronecker product properties we have ::

            tr(K⁻¹∂K) = tr[W∂Cᵢ]tr[EᵢEᵢᵀ] - tr[Z⁻¹(Xᵀ𝓡X)]
            tr(H⁻¹∂H) = - tr[(MᵀR⁻¹M)(Mᵀ𝕂M)] + tr[(MᵀR⁻¹X)Z⁻¹(XᵀR⁻¹M)(Mᵀ𝕂M)]

        Returns
        -------
        C0.Lu : ndarray
            Gradient of the log of the marginal likelihood over C₀ parameters.
        C1.Lu : ndarray
            Gradient of the log of the marginal likelihood over C₁ parameters.
        """

        def _dot(a, b):
            r = tensordot(a, b, axes=([1], [0]))
            if a.ndim > b.ndim:
                return r.transpose([0, 2, 1])
            return r

        def dot(*args):
            return reduce(_dot, args)

        def _sum(a):
            return a.sum(axis=(0, 1))

        terms = self._terms
        LdKLy = self._cov.LdKL_dot(terms["yl"])
        LdKLm = self._cov.LdKL_dot(terms["ml"])

        R = terms["R"]
        Ri = terms["Ri"]
        Riy = solve(terms["R"], vec(self._Y))
        r = Riy.T @ self._cov.gradient()["C0.Lu"][..., 0] @ Riy
        L0 = self._cov.C0.L
        L1 = self._cov.C1.L
        S, U = self._cov.C1.eigh()
        S = 1 / sqrt(S)
        US = ddot(U, S)
        G = self._cov.G
        Y = self._Y
        y = vec(Y)
        X = terms["X"]
        W = terms["W"]
        Ri = terms["Ri"]
        Z = terms["Z"]
        Ge = self._cov._Ge
        # X = kron(self._cov.C0.L, self._cov._G)
        # W = kron(ddot(U, S), eye(Ge.shape[0]))
        # Ri = W @ W.T

        XRiy = vec(Ge.T @ Y @ US @ US.T @ L0)

        M = self._mean.AF
        m = unvec(M @ vec(terms["B"]), (-1, self.ntraits))
        Gm = Ge.T @ m
        XRim = vec(Ge.T @ m @ US @ US.T @ L0)

        varnames = ["C0.Lu", "C1.Lu"]
        LdKLM = self._cov.LdKL_dot(terms["Ml"])
        dH = {n: -dot(terms["Ml"].T, LdKLM[n]).transpose([2, 0, 1]) for n in varnames}

        left = {n: (dH[n] @ terms["b"]).T for n in varnames}
        right = {n: terms["Ml"].T @ LdKLy[n] for n in varnames}
        db = {n: -solve(terms["H"], left[n] + right[n]) for n in varnames}

        grad = {}
        dmh = {n: terms["Mh"] @ db[n] for n in varnames}
        ld_grad = self._cov.logdet_gradient()
        ZiXRiy = solve(Z, XRiy)
        ZiXRim = solve(Z, XRim)
        for var in varnames:
            grad[var] = -ld_grad[var]

        SUL0 = US.T @ L0
        dC0 = self._cov.C0.gradient()["Lu"]
        SUdC0US = dot(US.T, dC0, US)
        dC1 = self._cov.C1.gradient()["Lu"]
        SUdC1US = dot(US.T, dC1, US)

        YUS = Y @ US
        GYUS = self.GY @ US
        GmUS = Gm @ US
        mUS = m @ US

        GG = self.GG
        GGGG = self.GGGG

        # Xᵀ𝓡X = (L₀ᵀU₁S₁⁻¹U₁ᵀ∂C₀U₁S₁⁻¹U₁ᵀL₀) ⊗ (GᵀGGᵀG)
        J0 = kron(dot(SUL0.T, SUdC0US, SUL0).T, GGGG)
        # Xᵀ𝓡X = (L₀ᵀU₁S₁⁻¹U₁ᵀ∂C₁U₁S₁⁻¹U₁ᵀL₀) ⊗ (GᵀG)
        J1 = kron(dot(SUL0.T, SUdC1US, SUL0).T, GG)

        # Xᵀ𝓡XZ⁻¹XᵀR⁻¹𝐲 over C₀ parameters
        J0ZiXRiy = J0 @ ZiXRiy
        # Xᵀ𝓡XZ⁻¹XᵀR⁻¹𝐦 over C₀ parameters
        J0ZiXRim = J0 @ ZiXRim
        # Xᵀ𝓡XZ⁻¹XᵀR⁻¹𝐲 over C₁ parameters
        J1ZiXRiy = J1 @ ZiXRiy
        # Xᵀ𝓡XZ⁻¹XᵀR⁻¹𝐦 over C₁ parameters
        J1ZiXRim = J1 @ ZiXRim

        GYC1idC0US = dot(GYUS, SUdC0US)
        # 𝐲ᵀ𝓡X over C₀ parameters
        yR0X = vec(dot(GG, GYC1idC0US, SUL0))
        GmC1idC0US = dot(GmUS, SUdC0US)
        # 𝐦ᵀ𝓡X over C₀ parameters
        mR0X = vec(dot(GG, GmC1idC0US, SUL0))

        YC1idC1US = dot(YUS, SUdC1US)
        # 𝐲ᵀ𝓡X over C₁ parameters
        yR1X = vec(dot(Ge.T, YC1idC1US, SUL0))
        mC1idC1US = dot(mUS, SUdC1US)
        # 𝐦ᵀ𝓡X over C₁ parameters
        mR1X = vec(dot(Ge.T, mC1idC1US, SUL0))

        # 𝐲ᵀ𝓡𝐲 over C₀ parameters
        yR0y = _sum((GYC1idC0US.T * GYUS.T).T)
        # 𝐲ᵀ𝕂𝐲 over C₀ parameters
        yKidKKiy = yR0y - 2 * yR0X.T @ ZiXRiy + ZiXRiy @ J0ZiXRiy.T
        grad["C0.Lu"] += yKidKKiy

        # 𝐲ᵀ𝓡𝐲 over C₁ parameters
        yR1y = _sum((YC1idC1US.T * YUS.T).T)
        # 𝐲ᵀ𝕂𝐲 over C₁ parameters
        yKidKKiy = yR1y - 2 * yR1X.T @ ZiXRiy + ZiXRiy @ J1ZiXRiy.T
        grad["C1.Lu"] += yKidKKiy

        # 𝐦ᵀ𝓡𝐦 over C₀ parameters
        mR0m = _sum((GmC1idC0US.T * GmUS.T).T)
        # 𝐦ᵀ𝕂𝐦 over C₀ parameters
        mKidKKim = mR0m - 2 * mR0X.T @ ZiXRim + ZiXRim @ J0ZiXRim.T
        grad["C0.Lu"] += mKidKKim

        # 𝐦ᵀ𝓡𝐦 over C₁ parameters
        mR1m = _sum((mC1idC1US.T * mUS.T).T)
        # 𝐦ᵀ𝕂𝐦 over C₁ parameters
        mKidKKim = mR1m - 2 * mR1X.T @ ZiXRim + ZiXRim @ J1ZiXRim.T
        grad["C1.Lu"] += mKidKKim

        # 𝐲ᵀ𝓡𝐦 over C₀ parameters
        yR0m = _sum((GYC1idC0US.T * GmUS.T).T)
        # 𝐲ᵀ𝕂𝐦 over C₀ parameters
        yKidKKim = yR0m - yR0X.T @ ZiXRim - mR0X.T @ ZiXRiy + ZiXRiy @ J0ZiXRim.T
        grad["C0.Lu"] -= 2 * yKidKKim

        # 𝐲ᵀ𝓡𝐦 over C₁ parameters
        yR1m = _sum((YC1idC1US.T * mUS.T).T)
        # 𝐲ᵀ𝕂𝐦 over C₁ parameters
        yKidKKim = yR1m - yR1X.T @ ZiXRim - mR1X.T @ ZiXRiy + ZiXRiy @ J1ZiXRim.T
        grad["C1.Lu"] -= 2 * yKidKKim

        for var in varnames:
            grad[var] -= diagonal(solve(terms["H"], dH[var]), axis1=1, axis2=2).sum(1)
            grad[var] += 2 * (terms["yl"] - terms["ml"]).T @ dmh[var]
            grad[var] /= 2

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
        self._maximize(verbose=verbose)
