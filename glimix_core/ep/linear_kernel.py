from __future__ import absolute_import, division, unicode_literals

from math import fsum

from numpy import dot, isfinite, log, sqrt

from numpy_sugar.linalg import cho_solve, ddot, dotd

from .ep import EP
from .posterior_linear_kernel import PosteriorLinearKernel


def ldot(A, B):
    return ddot(A, B, left=True)


def dotr(A, B):
    return ddot(A, B, left=False)


class EPLinearKernel(EP):
    def __init__(self, nsites):
        super(EPLinearKernel, self).__init__(nsites, PosteriorLinearKernel)

    def lml(self):
        if self._cache['lml'] is not None:
            return self._cache['lml']

        self._update()

        L = self._posterior.L()
        cov = self._posterior.cov
        Q = cov['QS'][0][0]
        S = cov['QS'][1]
        ttau = self._site.tau
        teta = self._site.eta
        ctau = self._cav['tau']
        ceta = self._cav['eta']
        m = self._posterior.mean

        TS = ttau + ctau

        s = cov['scale']
        d = cov['delta']
        A = self._posterior._A
        tQ = sqrt(1 - d) * Q

        lml = [
            -log(L.diagonal()).sum(),
            -0.5 * sum(log(s * S)),
            +0.5 * sum(log(A)),
            # lml += 0.5 * sum(log(ttau)),
            +0.5 * dot(teta * A, dot(tQ, cho_solve(L, dot(tQ.T, teta * A)))),
            -0.5 * dot(teta, teta / TS),
            +dot(m, A * teta) - 0.5 * dot(m, A * ttau * m),
            -0.5 * dot(
                m * A * ttau,
                dot(tQ, cho_solve(L, dot(tQ.T, 2 * A * teta - A * ttau * m)))),
            +sum(self._moments['log_zeroth']),
            +0.5 * sum(log(TS)),
            # lml -= 0.5 * sum(log(ttau)),
            -0.5 * sum(log(ctau)),
            +0.5 * dot(ceta / TS, ttau * ceta / ctau - 2 * teta),
            0.5 * s * d * sum(teta * A * teta)
        ]
        lml = fsum(lml)

        if not isfinite(lml):
            raise ValueError("LML should not be %f." % lml)

        self._cache['lml'] = lml

        return lml

    def lml_derivatives(self, dm):
        self._update()

        L = self._posterior.L()
        ttau = self._site.tau
        teta = self._site.eta
        A = self._posterior._A

        cov = self._posterior.cov
        Q = cov['QS'][0][0]
        S = cov['QS'][1]
        s = cov['scale']
        d = cov['delta']
        tQ = sqrt(1 - d) * Q
        tQS = dotr(tQ, S)
        QS = dotr(Q, S)

        e_m = teta - ttau * self._posterior.mean
        Ae_m = A * e_m
        TA = ttau * A
        tQTAe_m = dot(tQ.T, Ae_m)
        dKAd_m = dot(tQS, tQTAe_m) + d * Ae_m
        w = TA * dot(tQ, cho_solve(L, tQTAe_m))
        QTAe_m = dot(Q.T, Ae_m)
        dKAs_m = -s * dot(QS, QTAe_m) + s * Ae_m
        TAtQ = ldot(TA, tQ)
        LQt = cho_solve(L, Q.T)
        TAQ = ldot(TA, Q)
        r = dotd(TAQ, dot(dot(LQt, TAQ), QS.T)).sum()

        dlml_mean = dot(e_m, ldot(A, dm)) - dot(
            Ae_m, dot(tQ, cho_solve(L, dot(tQ.T, ldot(TA, dm)))))

        dlml_scale = 0.5 * dot(Ae_m, dKAd_m)
        dlml_scale -= sum(w * dKAd_m)
        dlml_scale += 0.5 * dot(w, dot(tQS, dot(tQ.T, w)) + d * w)
        dlml_scale -= 0.5 * dotd(TAtQ, tQS.T).sum()
        dlml_scale -= 0.5 * sum(TA) * d
        dlml_scale += 0.5 * r * (1 - d)**2
        dlml_scale += 0.5 * d * dotd(TAQ, dotr(LQt, TA)).sum() * (1 - d)

        dlml_delta = 0.5 * dot(Ae_m, dKAs_m)
        dlml_delta -= sum(w * dKAs_m)
        dlml_delta += 0.5 * dot(w, -s * dot(QS, dot(Q.T, w)) + s * w)
        dlml_delta += 0.5 * s * dotd(ldot(TA, Q), QS.T).sum()
        dlml_delta -= 0.5 * sum(TA) * s
        dlml_delta -= 0.5 * s * r * (1 - d)
        dlml_delta += 0.5 * s * dotd(TAQ, dotr(LQt, TA)).sum() * (1 - d)

        return dict(mean=dlml_mean, scale=dlml_scale, delta=dlml_delta)
