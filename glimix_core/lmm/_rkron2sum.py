import warnings

from numpy import asfortranarray, diagonal, kron, tensordot
from numpy.linalg import matrix_rank, slogdet, solve

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

        vec(Y) ~ N((A ⊗ F) vec(B), C₀ ⊗ GGᵗ + C₁ ⊗ I).

    A and F are design matrices of dimensions p×p and n×c provided by the user,
    where F is the usual matrix of covariates commonly used in single-trait models.
    B is a c×p matrix of fixed-effect sizes per trait.
    G is a n×r matrix provided by the user and I is a n×n identity matrices.
    C₀ and C₁ are both symmetric matrices of dimensions p×p, for which C₁ is
    guaranteed by our implementation to be of full rank.
    The parameters of this model are the matrices B, C₀, and C₁.

    For implementation purpose, we make use of the following definitions:

    - Y₀ = LₓY
    - M₀ = LₓF
    - M₁ = (LₕA) ⊗ M₀
    - M₂ = M₁B

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
            Matrix G from the GGᵗ term.
        rank : optional, int
            Maximum rank of matrix C₀. Defaults to ``1``.
        """
        Y = asfortranarray(Y)
        yrank = matrix_rank(Y)
        if Y.shape[1] > yrank:
            warnings.warn(
                f"Y is not full column rank: rank(Y)={yrank}. "
                + "Convergence might be problematic.",
                UserWarning,
            )

        self._Y = Y
        self._y = Y.ravel(order="F")
        self._A = A
        self._F = F
        self._cov = Kron2SumCov(G, Y.shape[1], rank)
        self._mean = KronMean(F.shape[1], Y.shape[1])
        self._mean.A = A
        self._mean.F = F
        self._Y0 = self._cov.Lx @ Y
        self._M0 = self._cov.Lx @ self._mean.F
        composite = [("C0", self._cov.C0), ("C1", self._cov.C1)]
        Function.__init__(self, "Kron2Sum", composite=composite)

    @property
    def _M1(self):
        return kron(self._cov.LhD["Lh"] @ self._mean.A, self._M0)

    @property
    def _MKiy(self):
        y1 = vec(self._Y0 @ self._cov.LhD["Lh"].T)
        return self._M1.T @ (self._cov.LhD["D"] * y1)

    def _quad(self):
        LhD = self._cov.LhD

        y1 = vec(self._Y0 @ LhD["Lh"].T)
        MKiy = self._M1.T @ (LhD["D"] * y1)

        D = LhD["D"]
        M1 = self._M1
        MKiM = M1.T @ ddot(D, M1)

        # 𝐦 = M𝛃 for 𝛃 = H⁻¹MᵗK⁻¹𝐲 and H = MᵗK⁻¹M.
        beta = solve(MKiM, self._MKiy)

        shape = (self.ncovariates, self.ntraits)
        m = vec(self._M0 @ unvec(beta, shape) @ self._mean.A.T @ LhD["Lh"].T)

        yKiy = y1 @ (LhD["D"] * y1)
        mKiy = m @ (LhD["D"] * y1)
        mKim = m @ (LhD["D"] * m)

        # dC0 = self._cov.C0.gradient()["Lu"]
        # dC1 = self._cov.C1.gradient()["Lu"]

        # axes = ([1], [0])
        # Z0 = tensordot(LhD["Lh"] @ dC0, LhD["Lh"].T, axes=axes)
        # Z0 = kron(Z0.transpose([1, 0, 2]), self._cov.LXL).transpose([1, 2, 0])
        # Z1 = tensordot(LhD["Lh"] @ dC1, LhD["Lh"].T, axes=axes)
        # Z1 = kron(Z1.transpose([1, 0, 2]), self._cov.LL).transpose([1, 2, 0])

        return {
            "MKiM": MKiM,
            "MKiy": MKiy,
            "yKiy": yKiy,
            "mKiy": mKiy,
            "mKim": mKim,
            "beta": beta,
        }

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
        Covariance K = C₀ ⊗ GGᵗ + C₁ ⊗ I.

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
        return self._F.shape[1]

    def value(self):
        return self.lml()

    def gradient(self):
        return self.lml_gradient()

    def _H(self):
        D = self._cov.LhD["D"]
        M1 = self._M1
        return M1.T @ ddot(D, M1)
        # M = self._mean.AF
        # return M.T @ self._cov.solve(M)

    def _logdet_MM(self):
        M = self._mean.AF
        ldet = slogdet(M.T @ M)
        if ldet[0] != 1.0:
            raise ValueError("The determinant of MᵀM should be positive.")
        return ldet[1]

    @property
    def reml_B(self):
        H = self._H()
        M = self._mean.AF
        beta = solve(H, M.T @ self._cov.solve(self._y))
        return unvec(beta, (self.ncovariates, self.ntraits))

    def lml(self):
        r"""
        Log of the marginal likelihood.

        Let 𝐲 = vec(Y), M = A⊗F, and H = MᵀK⁻¹M. The restricted log of the marginal
        likelihood is given by [R07]_::

            2⋅log(p(𝐲)) = -(n⋅p - c⋅p) log(2π) + log(\|MᵗM\|) - log(\|K\|) - log(\|H\|)
                - (𝐲-𝐦)ᵗ K⁻¹ (𝐲-𝐦),

        where 𝐦 = M𝛃 for 𝛃 = H⁻¹MᵗK⁻¹𝐲 and H = MᵗK⁻¹M.

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
        lml = -(np - cp) * log2pi + self._logdet_MM() - self._cov.logdet()

        quad = self._quad()

        ldet = slogdet(quad["MKiM"])
        if ldet[0] != 1.0:
            raise ValueError("The determinant of H should be positive.")
        lml -= ldet[1]

        lml -= quad["yKiy"] - 2 * quad["mKiy"] + quad["mKim"]

        return lml / 2

    def lml_gradient(self):
        """
        Gradient of the log of the marginal likelihood.

        Let 𝐲 = vec(Y), 𝕂 = K⁻¹∂(K)K⁻¹, and H = MᵀK⁻¹M. The gradient is given by::

            2⋅∂log(p(𝐲)) = -tr(K⁻¹∂K) - tr(H⁻¹∂H) + 𝐲ᵗ𝕂𝐲 + 2⋅∂(𝐦)ᵗK⁻¹𝐲 - 2⋅𝐦ᵗ𝕂𝐲
                - ∂(𝐦)ᵗK⁻¹𝐦 + 𝐦ᵗ𝕂𝐦 - 𝐦K⁻¹∂(𝐦).

            2⋅∂log(p(𝐲)) = -tr(K⁻¹∂K) - tr(H⁻¹∂H) + 𝐲ᵗ𝕂𝐲 - 𝐦ᵗ𝕂(2⋅𝐲-𝐦)
                - 2⋅(𝐦-𝐲)ᵗK⁻¹∂(𝐦).

        Returns
        -------
        C0.Lu : ndarray
            Gradient of the log of the marginal likelihood over C₀ parameters.
        C1.Lu : ndarray
            Gradient of the log of the marginal likelihood over C₁ parameters.
        """
        ld_grad = self._cov.logdet_gradient()
        quad = self._quad()

        Kiy = self._cov.solve(self._y)

        self._mean.B = self.reml_B
        m = self._mean.value()
        Kim = self._cov.solve(m)
        M = self._mean.AF
        KiM = self._cov.solve(M)
        grad = {}
        varnames = ["C0.Lu", "C1.Lu"]

        # LhD = self._cov.LhD
        # D = LhD["D"]

        # dH = - M^t K^-1 dK K^-1 M
        # dH = - M^t L^t D (L dK L^t) D L M
        # breakpoint()
        # dH_ = ddot(D, self._M1).T @ self._cov.LdKL_dot(ddot(D, self._M1)["C0.Lu"])
        t = self._cov.gradient_dot(KiM)
        dH = {n: -(KiM.T @ t[n]).T for n in varnames}

        H = self._H()

        # dK K^-1 y
        dK0 = self._cov.gradient_dot(Kiy)
        # dK K^-1 m
        dK1 = self._cov.gradient_dot(Kim)

        dbeta = {
            n: -solve(H, (dH[n] @ quad["beta"]).T) - solve(H, KiM.T @ dK0[n])
            for n in varnames
        }
        # LdKL_dot

        dm = {n: M @ g for n, g in dbeta.items()}
        for var in varnames:
            grad[var] = -ld_grad[var]
            grad[var] -= diagonal(solve(H, dH[var]), axis1=1, axis2=2).sum(1)
            grad[var] += Kiy.T @ dK0[var]
            # - 𝐦ᵗ𝕂(2⋅𝐲-𝐦)
            grad[var] -= Kim.T @ (2 * dK0[var] - dK1[var])
            # - 2⋅(𝐦-𝐲)ᵗK⁻¹∂(𝐦)
            grad[var] -= 2 * (m - self._y).T @ self._cov.solve(dm[var])
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
