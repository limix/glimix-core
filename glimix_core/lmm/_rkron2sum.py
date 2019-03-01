import warnings

from numpy import asfortranarray, diagonal
from numpy.linalg import matrix_rank, slogdet, solve

from glimix_core._util import log2pi, unvec, vec
from glimix_core.cov import Kron2SumCov
from glimix_core.mean import KronMean
from optimix import Function


class RKron2Sum(Function):
    """
    LMM for multiple traits.

    Let n, c, and p be the number of samples, covariates, and traits, respectively.
    The outcome variable Y is a n×p matrix distributed according to::

        vec(Y) ~ N((A ⊗ F) vec(B), Cᵣ ⊗ GGᵗ + Cₙ ⊗ I).

    A and F are design matrices of dimensions p×p and n×c provided by the user,
    where F is the usual matrix of covariates commonly used in single-trait models.
    B is a c×p matrix of fixed-effect sizes per trait.
    G is a n×r matrix provided by the user and I is a n×n identity matrices.
    Cᵣ and Cₙ are both symmetric matrices of dimensions p×p, for which Cₙ is
    guaranteed by our implementation to be of full rank.
    The parameters of this model are the matrices B, Cᵣ, and Cₙ.
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
            Maximum rank of matrix Cᵣ. Defaults to ``1``.
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
        self._cov = Kron2SumCov(Y.shape[1], rank)
        self._cov.G = G
        self._mean = KronMean(F.shape[1], Y.shape[1])
        self._mean.A = A
        self._mean.F = F
        composite = [("Cr", self._cov.Cr), ("Cn", self._cov.Cn)]
        Function.__init__(self, "Kron2Sum", composite=composite)

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
        Covariance K = Cᵣ ⊗ GGᵗ + Cₙ ⊗ I.

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
        M = self._mean.AF
        return M.T @ self._cov.solve(M)

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

        where 𝐦 = M𝛃 for 𝛃 = H⁻¹MᵗK⁻¹𝐲.

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

        H = self._H()
        ldet = slogdet(H)
        if ldet[0] != 1.0:
            raise ValueError("The determinant of H should be positive.")
        lml -= ldet[1]

        M = self._mean.AF
        beta = solve(H, M.T @ self._cov.solve(self._y))
        m = M @ beta
        d = self._y - m
        dKid = d @ self._cov.solve(d)

        lml -= dKid

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
        Cr.Lu : ndarray
            Gradient of the log of the marginal likelihood over Cᵣ parameters.
        Cn.L0 : ndarray
            Gradient of the log of the marginal likelihood over Cₙ parameter L₀.
        Cn.L1 : ndarray
            Gradient of the log of the marginal likelihood over Cₙ parameter L₁.
        """
        ld_grad = self._cov.logdet_gradient()

        Kiy = self._cov.solve(self._y)
        self._mean.B = self.reml_B
        m = self._mean.value()
        Kim = self._cov.solve(m)
        M = self._mean.AF
        KiM = self._cov.solve(M)
        grad = {}
        varnames = ["Cr.Lu", "Cn.Lu"]
        dK = {n: g.transpose([2, 0, 1]) for (n, g) in self._cov.gradient().items()}
        # TT = -KiM.T @ dK["Cn.Lu"] @ KiM
        # breakpoint()
        dH = {n: -KiM.T @ g @ KiM for n, g in dK.items()}
        H = self._H()
        beta = solve(H, M.T @ Kiy)

        def gdot(v, var):
            return self._cov.gradient_dot(v, var)

        dbeta = {
            n: -solve(H, (dH[n] @ beta).T) - solve(H, KiM.T @ gdot(Kiy, n))
            for n in varnames
        }

        dm = {n: M @ g for n, g in dbeta.items()}
        # dm
        for var in varnames:
            grad[var] = -ld_grad[var]
            grad[var] -= diagonal(solve(H, dH[var]), axis1=1, axis2=2).sum(1)
            # grad[var] += Kiy.T @ dK[var] @ Kiy
            grad[var] += Kiy.T @ gdot(Kiy, var)
            # self._cov.gradient_dot(Kiy)[var]
            # - 𝐦ᵗ𝕂(2⋅𝐲-𝐦)
            # grad[var] -= Kim.T @ dK[var] @ (2 * Kiy - Kim)
            grad[var] -= Kim.T @ gdot(2 * Kiy - Kim, var)
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
