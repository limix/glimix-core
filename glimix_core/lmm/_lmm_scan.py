from numpy import (
    all,
    asarray,
    atleast_2d,
    clip,
    copyto,
    empty,
    full,
    inf,
    isfinite,
    log,
    newaxis,
    sqrt,
)

from .._util import cached_property, hinv, hsolve, log2pi, nice_inv, rsolve, safe_log
from ._b import B


class FastScanner:
    """
    Approximated fast inference over several covariates.

    Specifically, it maximizes the marginal likelihood ::

        p(𝐲)ⱼ = 𝓝(𝐲 | 𝚇𝜷ⱼ + 𝙼ⱼ𝜶ⱼ, 𝑠ⱼ(𝙺 + 𝑣𝙸)),

    over 𝜷ⱼ, 𝜶ⱼ, and sⱼ. Matrix Mⱼ is the candidate set defined by the user. Variance 𝑣
    is not optimised for performance reasons. The method assumes the user has provided a
    reasonable value for it.

    Parameters
    ----------
    y
        Real-valued outcome.
    X
        Matrix of covariates.
    QS
        Economic eigendecomposition ``((Q0, Q1), S0)`` of ``K``.
    v
        Variance due to iid effect.

    Notes
    -----
    The implementation requires further explanation as it is somehow obscure. Let
    𝚀𝚂𝚀ᵀ = 𝙺, where 𝚀𝚂𝚀ᵀ is the eigendecomposition of 𝙺. Let 𝙳 = (𝚂 + 𝑣𝙸) and
    𝙳₀ = (𝚂₀ + 𝑣𝙸₀), where 𝚂₀ is the part of 𝚂 with positive values. Therefore, solving

        (𝙺 + 𝑣𝙸)𝐱 = 𝐲

     for 𝐱 is equivalent to solving

        𝚀₀𝙳₀𝚀₀ᵀ𝐱 + 𝑣𝚀₁𝚀₁ᵀ𝐱 = 𝚀₀𝙳₀𝚀₀ᵀ𝐱 + 𝑣(𝙸 - 𝚀₀𝚀₀ᵀ)𝐱 = 𝐲.

    for 𝐱. Let

        𝙱 = 𝚀₀𝙳₀⁻¹𝚀₀ᵀ                    if 𝑣=0, and
        𝙱 = 𝚀₀𝙳₀⁻¹𝚀₀ᵀ + 𝑣⁻¹(𝙸 - 𝚀₀𝚀₀ᵀ)   if 𝑣>0.

    We therefore have

        𝐱 = 𝙱𝐲

    as the solution of (𝙺 + 𝑣𝙸)𝐱 = 𝐲.

    Let 𝐛ⱼ = [𝜷ⱼᵀ 𝜶ⱼᵀ]ᵀ and 𝙴ⱼ = [𝚇 𝙼ⱼ]. The optimal parameters according to the marginal
    likelihood are given by ::

        (𝙴ⱼᵀ𝙱𝙴ⱼ)𝐛ⱼ = 𝙴ⱼᵀ𝙱𝐲

    and ::

        𝑠 = 𝑛⁻¹𝐲ᵀ𝙱(𝐲 - 𝙴ⱼ𝐛ⱼ).
    """

    def __init__(self, y, X, QS, v):
        from numpy_sugar import epsilon

        y = asarray(y, float)
        X = atleast_2d(asarray(X, float).T).T

        if not all(isfinite(y)):
            raise ValueError("Not all values are finite in the outcome array.")

        if not all(isfinite(X)):
            raise ValueError("Not all values are finite in the `X` matrix.")

        if v < 0:
            raise ValueError("Variance has to be non-negative.")

        if not isfinite(v):
            raise ValueError("Variance has to be a finite value..")

        if v <= epsilon.small:
            v = 0.0

        Q0 = QS[0][0]
        S0 = QS[1]
        D0 = S0 + v
        self._rankdef = y.shape[0] - S0.shape[0]

        self._B = B(Q0, S0, 1.0, v)

        By = self._B.dot(y)
        self._yTBy = y.T @ By
        self._yTBX = By.T @ X

        # Used for performing association scan on single variants
        self._ETBE = ETBE(self._B, X)
        self._yTBE = yTBE(By.T @ X)

        self._Q0 = Q0
        self._D0 = D0
        self._X = X
        self._y = y
        self._v = v

    def null_lml(self) -> float:
        return self._null_lml

    @cached_property
    def _null_lml(self) -> float:
        """
        Log of the marginal likelihood for the null hypothesis.

        It is implemented as ::

            2·log(p(𝐲)) = -𝑛·log(2·𝜋·𝑠) - log|D| - 𝑛.

        Returns
        -------
        lml
            Log of the marginal likelihood.
        """
        n = self._nsamples
        scale = self.null_scale
        return (self._static_lml - n * log(scale)) / 2

    @cached_property
    def null_beta(self):
        """
        Optimal 𝜷 according to the marginal likelihood.

        It is compute by solving the equation ::

            (XᵀBX)𝜷 = XᵀB𝐲.

        Returns
        -------
        beta
            Optimal 𝜷.
        """
        return rsolve(self._ETBE.XTBX, self._yTBX)

    @cached_property
    def null_beta_covariance(self):
        """
        Covariance of the optimal 𝜷 according to the marginal likelihood.

        Returns
        -------
        beta_covariance
            (Xᵀ(s(K + vI))⁻¹X)⁻¹.
        """
        return self.null_scale * nice_inv(self._X.T @ self._B.dot(self._X))

    @cached_property
    def null_beta_se(self):
        """
        Standard errors of the optimal 𝜷.

        Returns
        -------
        beta_se
            Square root of the diagonal of the beta covariance.
        """
        return sqrt(self.null_beta_covariance.diagonal())

    @cached_property
    def null_scale(self) -> float:
        """
        Optimal s according to the marginal likelihood.

        The optimal s is given by ::

            s = n⁻¹𝐲ᵀB(𝐲 - X𝜷),

        where 𝜷 is optimal.

        Returns
        -------
        scale
            Optimal scale.
        """
        n = self._nsamples
        beta = self.null_beta
        sqrdot2 = self._yTBy - self._yTBX @ beta
        return sqrdot2 / n

    def fast_scan(self, M, verbose: bool = True):
        """
        LMLs, fixed-effect sizes, and scales for single-marker scan.

        Parameters
        ----------
        M : array_like
            Matrix of fixed-effects across columns.
        verbose
            ``True`` for progress information; ``False`` otherwise.
            Defaults to ``True``.

        Returns
        -------
        lmls
            Log of the marginal likelihoods.
        effsizes0
            Covariate fixed-effect sizes.
        effsizes1
            Candidate set fixed-effect sizes.
        scales
            Scales.
        """
        from tqdm import tqdm

        if M.ndim != 2:
            raise ValueError("`M` array must be bidimensional.")
        p = M.shape[1]

        lmls = empty(p)
        effsizes0 = empty((p, self._X.shape[1]))
        effsizes0_se = empty((p, self._X.shape[1]))
        effsizes1 = empty(p)
        effsizes1_se = empty(p)
        scales = empty(p)

        chunks = get_chunks(M)

        start = 0
        for i in tqdm(chunks, desc="Scanning", disable=not verbose):
            stop = start + i

            r = self._fast_scan_chunk(M[:, start:stop])

            lmls[start:stop] = r["lml"]
            effsizes0[start:stop, :] = r["effsizes0"]
            effsizes0_se[start:stop, :] = r["effsizes0_se"]
            effsizes1[start:stop] = r["effsizes1"]
            effsizes1_se[start:stop] = r["effsizes1_se"]
            scales[start:stop] = r["scale"]

            start = stop

        return {
            "lml": lmls,
            "effsizes0": effsizes0,
            "effsizes0_se": effsizes0_se,
            "effsizes1": effsizes1,
            "effsizes1_se": effsizes1_se,
            "scale": scales,
        }

    def scan(self, M):
        """
        LML, fixed-effect sizes, and scale of the candidate set.

        Parameters
        ----------
        M : array_like
            Fixed-effects set.

        Returns
        -------
        lml : float
            Log of the marginal likelihood.
        effsizes0 : ndarray
            Covariates fixed-effect sizes.
        effsizes0_se : ndarray
            Covariates fixed-effect size standard errors.
        effsizes1 : ndarray
            Candidate set fixed-effect sizes.
        effsizes1_se : ndarray
            Candidate fixed-effect size standard errors.
        scale : ndarray
            Optimal scale.
        """
        from numpy_sugar import is_all_finite

        M = asarray(M, float)

        if M.shape[1] == 0:
            return {
                "lml": self._null_lml,
                "effsizes0": self.null_beta,
                "effsizes0_se": self.null_beta_se,
                "effsizes1": empty((0)),
                "effsizes1_se": empty((0)),
                "scale": self.null_scale,
            }

        if not is_all_finite(M):
            raise ValueError("M parameter has non-finite elements.")

        BM = self._B.dot(M)
        yTBM = self._y.T @ BM
        XTBM = self._X.T @ BM
        MTBM = M.T @ BM

        return self._multicovariate_set(yTBM, XTBM, MTBM)

    @property
    def _nsamples(self):
        return self._y.shape[0]

    @property
    def _ncovariates(self):
        return self._X.shape[1]

    @cached_property
    def _static_lml(self):
        """
        Static part of the marginal likelihood.

        It is defined by ::

            -𝑛·log(2·𝜋) - 𝑛 - log|D|.
        """
        n = self._nsamples
        static_lml = -n * log2pi - n
        static_lml -= safe_log(self._D0).sum()
        static_lml -= self._rankdef * safe_log(self._v)
        return static_lml

    def _fast_scan_chunk(self, M):
        from numpy_sugar import dotd

        M = asarray(M, float)

        if not M.ndim == 2:
            raise ValueError("`M` array must be bidimensional.")

        if not all(isfinite(M)):
            raise ValueError("One or more variants have non-finite value.")

        BM = self._B.dot(M)
        yTBM = self._y.T @ BM
        XTBM = self._X.T @ BM
        dMTBM = dotd(M.T, BM)

        lmls = full(M.shape[1], self._static_lml)
        eff0 = empty((M.shape[1], self._X.shape[1]))
        eff0_se = empty((M.shape[1], self._X.shape[1]))
        eff1 = empty((M.shape[1]))
        eff1_se = empty((M.shape[1]))
        scales = empty(M.shape[1])

        effs = {"eff0": eff0, "eff0_se": eff0_se, "eff1": eff1, "eff1_se": eff1_se}

        if self._ncovariates == 1:
            self._1covariate_loop(lmls, effs, scales, yTBM, XTBM, dMTBM)
        else:
            self._multicovariate_loop(lmls, effs, scales, yTBM, XTBM, dMTBM)

        return {
            "lml": lmls,
            "effsizes0": eff0,
            "effsizes0_se": eff0_se,
            "effsizes1": eff1,
            "effsizes1_se": eff1_se,
            "scale": scales,
        }

    def _multicovariate_loop(self, lmls, effs, scales, yTBM, XTBM, diagMTBM):
        ETBE = self._ETBE
        yTBE = self._yTBE

        for i in range(XTBM.shape[1]):

            yTBE.set_yTBM(yTBM[i])
            ETBE.set_XTBM(XTBM[:, [i]])
            ETBE.set_MTBM(diagMTBM[i])

            left = ETBE.value
            right = yTBE.value
            x = rsolve(left, right)
            beta = x[:-1][:, newaxis]
            alpha = x[-1:]
            bstar = _bstar_unpack(beta, alpha, self._yTBy, yTBE, ETBE, bstar_1effect)

            se = sqrt(nice_inv(left).diagonal())

            scales[i] = bstar / self._nsamples
            lmls[i] -= self._nsamples * safe_log(scales[i])
            effs["eff0"][i, :] = beta.T
            effs["eff0_se"][i, :] = se[:-1]
            effs["eff1"][i] = alpha[0]
            effs["eff1_se"][i] = se[-1]

        lmls /= 2

    def _multicovariate_set(self, yTBM, XTBM, MTBM):

        yBE = yTBE(self._yTBX, yTBM.shape[0])
        yBE.set_yTBM(yTBM)

        set_size = yTBM.shape[0]
        EBE = ETBE(self._B, self._X, set_size)

        EBE.set_XTBM(XTBM)
        EBE.set_MTBM(MTBM)

        left = EBE.value
        right = yBE.value
        x = rsolve(left, right)

        beta = x[:-set_size]
        alpha = x[-set_size:]
        bstar = _bstar_unpack(beta, alpha, self._yTBy, yBE, EBE, bstar_set)

        lml = self._static_lml

        scale = bstar / self._nsamples
        lml -= self._nsamples * safe_log(scale)
        lml /= 2

        effsizes_se = sqrt(scale * nice_inv(left).diagonal())
        beta_se = effsizes_se[:-set_size]
        alpha_se = effsizes_se[-set_size:]

        return {
            "lml": lml,
            "effsizes0": beta,
            "effsizes0_se": beta_se,
            "effsizes1": alpha,
            "effsizes1_se": alpha_se,
            "scale": scale,
        }

    def _1covariate_loop(self, lmls, effs, scales, yTBM, XTBM, diagMTBM):
        ETBE = self._ETBE
        yTBX = self._yTBX
        XTBX = ETBE.XTBX
        yTBy = self._yTBy

        A00 = ETBE.XTBX[0, 0]
        A01 = XTBM[0, :]
        A11 = diagMTBM

        b0 = yTBX[0]
        b1 = yTBM

        x = hsolve(A00, A01, A11, b0, b1)
        beta = x[0][newaxis, :]
        alpha = x[1]
        bstar = bstar_1effect(beta, alpha, yTBy, yTBX, yTBM, XTBX, XTBM, diagMTBM)

        scales[:] = bstar / self._nsamples
        lmls -= self._nsamples * safe_log(scales)
        lmls /= 2
        effs["eff0"][:] = beta.T
        effs["eff1"][:] = alpha

        A00i, _, A11i = hinv(A00, A01, A11)
        effs["eff0_se"][:, 0] = sqrt(scales * A00i)
        effs["eff1_se"][:] = sqrt(scales * A11i)


class yTBE:
    """
    Represent 𝐲ᵀ𝙱𝙴 where 𝙴 = [𝚇 𝙼].
    """

    def __init__(self, yTBX, set_size=1):
        n = yTBX.shape[0] + set_size
        self._data = empty((n,))
        self._data[:-set_size] = yTBX
        self._m = set_size

    @property
    def value(self):
        return self._data

    @property
    def yTBX(self):
        return self._data[: -self._m]

    @property
    def yTBM(self):
        return self._data[-self._m :]

    def set_yTBM(self, yTBM):
        copyto(self.yTBM, yTBM)


class ETBE:
    """
    Represent 𝙴ᵀ𝙱𝙴 where 𝙴 = [𝚇 𝙼].
    """

    def __init__(self, B, X, set_size=1):
        n = X.shape[1] + set_size
        BX = B.dot(X)
        self._data = empty((n, n))
        self._data[:-set_size, :-set_size] = X.T @ BX
        self._m = set_size

    @property
    def value(self):
        return self._data

    @property
    def XTBX(self):
        return self._data[: -self._m, : -self._m]

    @property
    def XTBM(self):
        return self._data[: -self._m, -self._m :]

    @property
    def MTBX(self):
        return self._data[-self._m :, : -self._m]

    @property
    def MTBM(self):
        return self._data[-self._m :, -self._m :]

    def set_XTBM(self, XTBM):
        copyto(self.XTBM, XTBM)
        copyto(self.MTBX, XTBM.T)

    def set_MTBM(self, MTBM):
        copyto(self.MTBM, MTBM)


def bstar_1effect(beta, alpha, yTBy, yTBX, yTBM, XTBX, XTBM, MTBM):
    """
    Same as :func:`bstar_set` but for single-effect.
    """
    from numpy_sugar import epsilon
    from numpy_sugar.linalg import dotd

    r = full(MTBM.shape[0], yTBy)
    r -= 2 * yTBX @ beta
    r -= 2 * yTBM * alpha
    r += dotd(beta.T, XTBX @ beta)
    r += dotd(beta.T, XTBM * alpha)
    r += (alpha * XTBM * beta).sum(axis=0)
    r += alpha * MTBM.ravel() * alpha
    return clip(r, epsilon.tiny, inf)


def bstar_set(beta, alpha, yTBy, yTBX, yTBM, XTBX, XTBM, MTBM):
    """
    Compute 𝐲ᵀ𝙱𝐲 - 2𝐲ᵀ𝙱𝙴ⱼ𝐛ⱼ + 𝐛ⱼᵀ𝙴ⱼᵀ𝙱𝙴ⱼ𝐛ⱼ.

    For 𝐛ⱼ = [𝜷ⱼᵀ 𝜶ⱼᵀ]ᵀ.
    """
    r = yTBy
    r -= 2 * yTBX @ beta
    r -= 2 * yTBM @ alpha
    r += beta.T @ XTBX @ beta
    r += 2 * beta.T @ XTBM @ alpha
    r += alpha.T @ MTBM @ alpha
    return r


def _bstar_unpack(beta, alpha, yTBy, yTBE, ETBE, bstar):
    from numpy_sugar import epsilon

    yTBX = yTBE.yTBX
    yTBM = yTBE.yTBM
    XTBX = ETBE.XTBX
    XTBM = ETBE.XTBM
    MTBM = ETBE.MTBM
    bstar = bstar(beta, alpha, yTBy, yTBX, yTBM, XTBX, XTBM, MTBM)
    return clip(bstar, epsilon.tiny, inf)


def get_chunks(M):
    chunks = None
    if hasattr(M, "chunks") and M.chunks is not None:
        if len(M.chunks) == 2:
            return M.chunks[1]

    p = M.shape[1]
    siz = round(p / min(50, p))
    n = int(p / siz)
    chunks = [siz] * n
    if n * siz < p:
        chunks += [p - n * siz]

    return chunks
