import warnings

from numpy import asfortranarray
from numpy.linalg import matrix_rank, slogdet, solve

from glimix_core._util import log2pi, unvec
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
        lml = -(np - cp) * log2pi + self._logdet_MM() + self._cov.logdet()

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

        Let 𝐲 = vec(Y) and 𝕂 = K⁻¹∂(K)K⁻¹. The gradient is given by::

            2⋅∂log(p(𝐲)) = -tr(K⁻¹∂K) + 𝐲ᵗ𝕂𝐲 + (𝐦-2⋅𝐲)ᵗ𝕂𝐦 - 2⋅(𝐲-𝐦)ᵗK⁻¹∂(𝐦).

        Returns
        -------
        M.vecB : ndarray
            Gradient of the log of the marginal likelihood over vec(B).
        Cr.Lu : ndarray
            Gradient of the log of the marginal likelihood over Cᵣ parameters.
        Cn.L0 : ndarray
            Gradient of the log of the marginal likelihood over Cₙ parameter L₀.
        Cn.L1 : ndarray
            Gradient of the log of the marginal likelihood over Cₙ parameter L₁.
        """
        ld_grad = self._cov.logdet_gradient()
        dK = {n: g.transpose([2, 0, 1]) for (n, g) in self._cov.gradient().items()}
        Kiy = self._cov.solve(self._y)
        m = self._mean.value()
        Kim = self._cov.solve(m)
        grad = {}
        dm = self._mean.gradient()["vecB"]
        grad["M.vecB"] = dm.T @ Kiy - dm.T @ Kim
        for var in ["Cr.Lu", "Cn.L0", "Cn.L1"]:
            grad[var] = -ld_grad[var]
            grad[var] += Kiy.T @ dK[var] @ Kiy
            grad[var] -= 2 * (Kim.T @ dK[var] @ Kiy)
            grad[var] += Kim.T @ dK[var] @ Kim
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
