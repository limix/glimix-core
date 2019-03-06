import warnings
from functools import lru_cache

from numpy import asarray, asfortranarray, diagonal, kron, tensordot
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

        vec(Y) ~ N((A ⊗ F) vec(B), K = C₀ ⊗ GGᵗ + C₁ ⊗ I).

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
            Matrix G from the GGᵗ term.
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
        self._y = Y.ravel(order="F")
        self._cov = Kron2SumCov(G, Y.shape[1], rank)
        self._mean = KronMean(A, F)
        self._Yx = self._cov.Lx @ Y
        self._Mx = self._cov.Lx @ self._mean.F
        self._cache = {"terms": None}
        self._cov.listen(self._parameters_update)
        composite = [("C0", self._cov.C0), ("C1", self._cov.C1)]
        Function.__init__(self, "Kron2Sum", composite=composite)

    def _parameters_update(self, _):
        self._cache["terms"] = None

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

        # H = MᵗK⁻¹M.
        H = Mh.T @ Ml

        # 𝐦 = M𝛃 for 𝛃 = H⁻¹MᵗK⁻¹𝐲 and H = MᵗK⁻¹M.
        # 𝛃 = H⁻¹MᵗₕD𝐲ₗ
        b = solve(H, Mh.T @ yl)
        self._mean.B = unvec(b, (self.ncovariates, -1))

        mh = Mh @ b
        ml = D * mh

        ldetH = slogdet(H)
        if ldetH[0] != 1.0:
            raise ValueError("The determinant of H should be positive.")
        ldetH = ldetH[1]

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
        lml = -(np - cp) * log2pi + self._logdet_MM - self._cov.logdet()

        terms = self._terms
        lml -= terms["ldetH"]

        lml -= (
            terms["yh"] @ terms["yl"]
            - 2 * terms["ml"] @ terms["yh"]
            + terms["ml"] @ terms["mh"]
        )

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

        def dot(a, b):
            from numpy import einsum

            le = "ijk"[: a.ndim]
            ri = "jlk"[: b.ndim]
            re = "ilk"[: max(a.ndim, b.ndim)]
            return einsum(f"{le},{ri}->{re}", a, b)

        terms = self._terms
        LdKLy = self._cov.LdKL_dot(terms["yl"])
        LdKLm = self._cov.LdKL_dot(terms["ml"])

        varnames = ["C0.Lu", "C1.Lu"]
        LdKLM = self._cov.LdKL_dot(terms["Ml"])
        dH = {n: -dot(terms["Ml"].T, LdKLM[n]).transpose([2, 0, 1]) for n in varnames}

        left = {n: solve(terms["H"], (dH[n] @ terms["b"]).T) for n in varnames}
        right = {n: solve(terms["H"], terms["Ml"].T @ LdKLy[n]) for n in varnames}
        db = {n: -left[n] - right[n] for n in varnames}

        grad = {}
        dmh = {n: terms["Mh"] @ db[n] for n in varnames}
        ld_grad = self._cov.logdet_gradient()
        for var in varnames:
            grad[var] = -ld_grad[var]
            grad[var] -= diagonal(solve(terms["H"], dH[var]), axis1=1, axis2=2).sum(1)
            grad[var] += terms["yl"].T @ LdKLy[var]
            grad[var] -= 2 * terms["ml"].T @ LdKLy[var]
            grad[var] += terms["ml"].T @ LdKLm[var]
            grad[var] -= 2 * terms["ml"].T @ dmh[var]
            grad[var] += 2 * terms["yl"].T @ dmh[var]
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
