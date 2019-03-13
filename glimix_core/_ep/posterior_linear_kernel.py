from numpy import dot

from .posterior import Posterior


def _cho_factor(B):
    from scipy.linalg import cho_factor

    B = cho_factor(B, overwrite_a=True, lower=True, check_finite=False)[0]
    return B


class PosteriorLinearKernel(Posterior):
    """
    EP posterior.

    It is given by ::

        𝐳 ∼ 𝒩(Σ(T̃𝛍̃ + K⁻¹𝐦), (T̃ + K⁻¹)⁻¹).
    """

    @property
    def A(self):
        s = self._cov["scale"]
        d = self._cov["delta"]
        return 1 / (s * d * self._site.tau + 1)

    def AQ(self):
        from numpy_sugar.linalg import ddot

        if self._AQ_cache is not None:
            return self._AQ_cache

        Q = self._cov["QS"][0][0]
        A = self.A

        self._AQ_cache = ddot(A, Q)
        return self._AQ_cache

    def ATQ(self):
        from numpy_sugar.linalg import ddot

        if self._TAQ_cache is not None:
            return self._TAQ_cache

        self._TAQ_cache = ddot(self._site.tau, self.AQ())
        return self._TAQ_cache

    def QSQtATQLQtA(self):
        from numpy_sugar.linalg import ddot, dotd

        if self._QSQtATQLQtA_cache is not None:
            return self._QSQtATQLQtA_cache

        LQt = self.LQt()
        A = self.A
        Q = self._cov["QS"][0][0]
        LQtA = ddot(LQt, A)
        AQ = self.AQ()
        QS = self.QS()
        T = self._site.tau
        self._QSQtATQLQtA_cache = dotd(QS, dot(dot(ddot(AQ.T, T), Q), LQtA))
        return self._QSQtATQLQtA_cache

    def L(self):
        r"""Cholesky decomposition of :math:`\mathrm B`.

        .. math::

            \mathrm B = \mathrm Q^{\intercal}\tilde{\mathrm{T}}\mathrm Q
                + \mathrm{S}^{-1}
        """
        from numpy_sugar.linalg import ddot, sum2diag

        if self._L_cache is not None:
            return self._L_cache

        s = self._cov["scale"]
        d = self._cov["delta"]
        Q = self._cov["QS"][0][0]
        S = self._cov["QS"][1]

        ddot(self.A * self._site.tau, Q, left=True, out=self._NxR)
        B = dot(Q.T, self._NxR, out=self._RxR)
        B *= 1 - d
        sum2diag(B, 1.0 / S / s, out=B)
        self._L_cache = _cho_factor(B)
        return self._L_cache

    def update(self):
        from numpy_sugar.linalg import ddot, dotd

        self._flush_cache()

        s = self._cov["scale"]
        d = self._cov["delta"]
        Q = self._cov["QS"][0][0]

        A = self.A
        LQt = self.LQt()

        T = self._site.tau
        E = self._site.eta

        AQ = self.AQ()
        QS = self.QS()

        LQtA = ddot(LQt, A)
        D = self.QSQtATQLQtA()

        self.tau[:] = s * (1 - d) * dotd(QS, AQ.T)
        self.tau -= s * (1 - d) * D * (1 - d)
        self.tau += s * d * A
        self.tau -= s * d * dotd(self.ATQ(), LQtA) * (1 - d)
        self.tau **= -1

        v = s * (1 - d) * dot(Q, dot(QS.T, E)) + s * d * E + self._mean

        self.eta[:] = A * v
        self.eta -= dot(AQ, dot(LQtA, T * v)) * (1 - d)
        self.eta *= self.tau
