import warnings
from functools import lru_cache

from numpy import asarray, block, kron, zeros
from numpy.linalg import LinAlgError

from glimix_core._util import unvec, vec

from .._util import log2pi


class KronFastScanner:
    def __init__(self, Y, A, F, G, terms):

        self._Y = Y
        self._A = A
        self._F = F
        self._G = G
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

    @lru_cache(maxsize=None)
    def _static_lml(self):
        np = self._nsamples * self._ntraits
        static_lml = -np * log2pi - self._logdetK - self._yKiy
        return static_lml / 2

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
    @lru_cache(maxsize=None)
    def _MKiM(self):
        return self._MRiM - self._XRiM.T @ self._ZiXRiM

    @property
    @lru_cache(maxsize=None)
    def _MKiy(self):
        return self._MRiy - self._XRiM.T @ self._ZiXRiy

    @lru_cache(maxsize=None)
    def null_effsizes(self):
        return _solve(self._MKiM, self._MKiy)

    @lru_cache(maxsize=None)
    def null_lml(self):
        r"""Log of the marginal likelihood for the null hypothesis.

        Returns
        -------
        float
            Log of the marginal likelihood.
        """
        b = self.null_effsizes()
        mKim = b.T @ self._MKiM @ b
        mKiy = b.T @ self._MKiy
        return self._static_lml() - mKim / 2 + mKiy

    def scan(self, A, G):
        from scipy.linalg import cho_solve

        A1 = asarray(A, float)
        F1 = asarray(G, float)

        F1F1 = F1.T @ F1
        FF1 = self._F.T @ F1
        AWA1 = self._WA.T @ A1
        A1W = A1.T @ self._W
        GF1 = self._G.T @ F1

        MRiM1 = kron(AWA1, FF1)
        M1RiM1 = kron(A1W @ A1, F1F1)

        M1Riy = vec(F1.T @ self._Y @ A1W.T)
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
        beta = _solve(MKiM, MKiy)

        mKim = beta.T @ MKiM @ beta
        mKiy = beta.T @ MKiy
        cp = self._ntraits * self._ncovariates
        effsizes0 = unvec(beta[:cp], (self._ncovariates, self._ntraits))
        effsizes1 = unvec(beta[cp:], (G.shape[1], self._ntraits))
        return self._static_lml() - mKim / 2 + mKiy, effsizes0, effsizes1


def _solve(A, y):
    from numpy_sugar.linalg import rsolve

    try:
        beta = rsolve(A, y)
    except LinAlgError:
        msg = "Could not converge to the optimal effect-size. "
        msg += "Setting its effect-size to zero."
        warnings.warn(msg, RuntimeWarning)
        beta = zeros(A.shape[0])

    return beta
