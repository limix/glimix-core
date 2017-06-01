from __future__ import absolute_import, division, unicode_literals

import logging
from math import fsum

from numpy import dot, empty, inf, isfinite, log, maximum, sqrt, zeros
from numpy.linalg import norm
from numpy_sugar import epsilon
from numpy_sugar.linalg import cho_solve, ddot, dotd

from .ep import EP
from .posterior_linear_kernel import PosteriorLinearKernel
from .site import Site

MAX_ITERS = 100
RTOL = epsilon.small * 1000
ATOL = epsilon.small * 1000


def ldot(A, B):
    return ddot(A, B, left=True)


def dotr(A, B):
    return ddot(A, B, left=False)


class EPLinearKernel(EP):  # pylint: disable=R0903
    def __init__(self, nsites):
        super(EPLinearKernel, self).__init__(nsites, PosteriorLinearKernel)

    def _lml(self):
        self._params_update()

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
            -log(L.diagonal()).sum(),  #
            -0.5 * sum(log(s * S)),  #
            +0.5 * sum(log(A)),  #
            # lml += 0.5 * sum(log(ttau)),
            +0.5 * dot(teta * A, dot(tQ, cho_solve(L, dot(tQ.T,
                                                          teta * A)))),  #!=
            -0.5 * dot(teta, teta / TS),  #
            +dot(m, A * teta) - 0.5 * dot(m, A * ttau * m),  #
            -0.5 * dot(m * A * ttau,
                       dot(tQ,
                           cho_solve(L, dot(tQ.T,
                                            2 * A * teta - A * ttau * m)))),  #
            +sum(self._moments['log_zeroth']),  #
            +0.5 * sum(log(TS)),  #
            # lml -= 0.5 * sum(log(ttau)),
            -0.5 * sum(log(ctau)),  #
            +0.5 * dot(ceta / TS, ttau * ceta / ctau - 2 * teta),  #
            0.5 * s * d * sum(teta * A * teta)
        ]
        lml = fsum(lml)

        if not isfinite(lml):
            raise ValueError("LML should not be %f." % lml)

        return lml

    def _lml_derivatives(self, dm):
        self._params_update()

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

        e_m = teta - ttau * self._posterior.mean
        Ae_m = A * e_m
        TA = ttau * A
        QTAe_m = dot(tQ.T, Ae_m)
        dKAd_m = dot(tQS, QTAe_m) + d * Ae_m
        QLQAe_m = dot(tQ, cho_solve(L, QTAe_m))
        TAQLQAe_m = TA * QLQAe_m

        dlml_mean = dot(e_m, ldot(A, dm)) - dot(
            Ae_m, dot(tQ, cho_solve(L, dot(tQ.T, ldot(TA, dm)))))

        dlml_scale = 0.5 * dot(Ae_m, dKAd_m)
        dlml_scale -= sum(TAQLQAe_m * dKAd_m)
        dlml_scale += 0.5 * dot(TAQLQAe_m,
                          dot(tQS, dot(tQ.T, TAQLQAe_m)) + d * TAQLQAe_m)
        dlml_scale -= 0.5 * dotd(ldot(TA, tQ), tQS.T).sum()
        dlml_scale -= 0.5 * sum(TA * d)

        t0 = dot(cho_solve(L, dot(tQ.T, ldot(TA, tQ))), tQS.T)
        dlml_scale += 0.5 * dotd(ldot(TA, tQ), t0).sum()
        dlml_scale += 0.5 * d * dotd(ldot(TA, tQ), cho_solve(L, dotr(tQ.T, TA))).sum()

        return dict(mean=dlml_mean, scale=dlml_scale)

    def _lml_derivative_over_mean(self, dm):
        self._params_update()

        L = self._posterior.L()
        cov = self._posterior.cov
        ttau = self._site.tau
        teta = self._site.eta
        A = self._posterior._A

        Q = cov['QS'][0][0] * sqrt(1 - cov['delta'])

        di = teta - ttau * self._posterior.mean

        dlml = dot(di, ldot(A, dm))
        dlml -= dot(di * A,
                    dot(Q, cho_solve(L, dot(Q.T, ldot(A, (ttau * dm.T).T)))))

        return dlml

    def _lml_derivative_over_cov_scale(self):
        self._params_update()

        L = self._posterior.L()
        cov = self._posterior.cov
        T = self._site.tau
        A = self._posterior._A

        S = cov['QS'][1]
        d = cov['delta']
        Q = sqrt(1 - d) * cov['QS'][0][0]

        e_m = self._site.eta - T * self._posterior.mean
        Ae_m = A * e_m
        QTe_m = dot(Q.T, e_m)
        QS = dotr(Q, S)
        TA = T * A

        tQStQTdi = dot(QS, QTe_m)
        QTAe_m = dot(Q.T, Ae_m)

        dKAd_m = dot(QS, QTAe_m) + d * Ae_m

        QLQAd_m = dot(Q, cho_solve(L, QTAe_m))
        TAQLQAd_m = TA * QLQAd_m

        dlml = 0.5 * dot(Ae_m, dKAd_m)
        dlml -= sum(TAQLQAd_m * dKAd_m)
        dlml += 0.5 * dot(TAQLQAd_m,
                          dot(QS, dot(Q.T, TAQLQAd_m)) + d * TAQLQAd_m)

        dlml -= 0.5 * dotd(ldot(TA, Q), QS.T).sum()
        dlml -= 0.5 * sum(TA * d)

        t0 = dot(cho_solve(L, dot(Q.T, ldot(TA, Q))), QS.T)
        dlml += 0.5 * dotd(ldot(TA, Q), t0).sum()

        dlml += 0.5 * d * dotd(ldot(TA, Q), cho_solve(L, dotr(Q.T, TA))).sum()

        return dlml

    def _lml_derivative_over_cov_delta(self):
        self._params_update()

        L = self._posterior.L()
        cov = self._posterior.cov
        T = self._site.tau
        A = self._posterior._A

        S = cov['QS'][1]
        d = cov['delta']
        s = cov['scale']
        Q = cov['QS'][0][0]
        tQ = sqrt(1 - d) * cov['QS'][0][0]

        e_m = self._site.eta - T * self._posterior.mean
        Ae_m = A * e_m
        tQTe_m = dot(tQ.T, e_m)
        tQS = dotr(tQ, S)
        QS = dotr(Q, S)
        TA = T * A

        tQStQTdi = dot(tQS, tQTe_m)
        tQTAe_m = dot(tQ.T, Ae_m)
        QTAe_m = dot(Q.T, Ae_m)

        dKAd_m = -s * dot(QS, QTAe_m) + s * Ae_m

        QLQAd_m = dot(tQ, cho_solve(L, tQTAe_m))
        TAQLQAd_m = TA * QLQAd_m

        dlml = 0.5 * dot(Ae_m, dKAd_m)
        dlml -= sum(TAQLQAd_m * dKAd_m)
        dlml += 0.5 * dot(TAQLQAd_m,
                          -s * dot(QS, dot(Q.T, TAQLQAd_m)) + s * TAQLQAd_m)

        dlml += 0.5 * s * dotd(ldot(TA, Q), QS.T).sum()
        dlml -= 0.5 * sum(TA * s)

        t0 = dot(cho_solve(L, dot(tQ.T, ldot(TA, Q))), QS.T)
        dlml -= 0.5 * s * dotd(ldot(TA, tQ), t0).sum()

        dlml += 0.5 * s * dotd(ldot(TA, tQ), cho_solve(L, dotr(tQ.T,
                                                               TA))).sum()

        return dlml
