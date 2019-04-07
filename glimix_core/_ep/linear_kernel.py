from math import fsum

from numpy import dot, isfinite, log

from .ep import EP
from .posterior_linear_kernel import PosteriorLinearKernel


def ldot(A, B):
    from numpy_sugar.linalg import ddot

    return ddot(A, B, left=True)


def dotr(A, B):
    from numpy_sugar.linalg import ddot

    return ddot(A, B, left=False)


class EPLinearKernel(EP):
    def __init__(self, nsites, rtol=1.49e-05, atol=1.49e-08):
        super(EPLinearKernel, self).__init__(
            nsites, PosteriorLinearKernel, rtol=rtol, atol=atol
        )

    def lml(self):
        if self._cache["lml"] is not None:
            return self._cache["lml"]

        self._update()

        L = self._posterior.L()
        LQt = self._posterior.LQt()
        cov = self._posterior.cov
        Q = cov["QS"][0][0]
        S = cov["QS"][1]
        ttau = self._site.tau
        teta = self._site.eta
        ctau = self._cav["tau"]
        ceta = self._cav["eta"]
        m = self._posterior.mean

        TS = ttau + ctau

        s = cov["scale"]
        d = cov["delta"]
        A = self._posterior.A
        dif = 2 * A * teta - A * ttau * m

        lml = [
            -log(L.diagonal()).sum(),
            -0.5 * sum(log(s * S)),
            +0.5 * sum(log(A)),
            # lml += 0.5 * sum(log(ttau)),
            +0.5 * dot(teta * A, dot(Q, dot(LQt, teta * A))) * (1 - d),
            -0.5 * dot(teta, teta / TS),
            +dot(m, A * teta) - 0.5 * dot(m, A * ttau * m),
            -0.5 * dot(m * A * ttau, dot(Q, dot(LQt, dif))) * (1 - d),
            +sum(self._moments["log_zeroth"]),
            +0.5 * sum(log(TS)),
            # lml -= 0.5 * sum(log(ttau)),
            -0.5 * sum(log(ctau)),
            +0.5 * dot(ceta / TS, ttau * ceta / ctau - 2 * teta),
            0.5 * s * d * sum(teta * A * teta),
        ]
        lml = fsum(lml)

        if not isfinite(lml):
            raise ValueError("LML should not be %f." % lml)

        self._cache["lml"] = lml

        return lml

    def lml_derivatives(self, dm):
        from numpy_sugar.linalg import dotd

        if self._cache["grad"] is not None:
            return self._cache["grad"]

        self._update()

        LQt = self._posterior.LQt()
        ATQ = self._posterior.ATQ()
        ttau = self._site.tau
        teta = self._site.eta
        A = self._posterior.A

        cov = self._posterior.cov
        Q = cov["QS"][0][0]
        s = cov["scale"]
        d = cov["delta"]
        QS = self._posterior.QS()

        e_m = teta - ttau * self._posterior.mean
        Ae_m = A * e_m
        TA = ttau * A
        LtQTAe_m = dot(LQt, Ae_m)
        tQTAe_m = dot(Q.T, Ae_m)
        dKAd_m = dot(QS, tQTAe_m) * (1 - d) + d * Ae_m
        w = TA * dot(Q, LtQTAe_m) * (1 - d)
        QTAe_m = dot(Q.T, Ae_m)
        dKAs_m = -s * dot(QS, QTAe_m) + s * Ae_m

        r = (self._posterior.QSQtATQLQtA() * ttau).sum()

        dlml_mean = dot(e_m, ldot(A, dm)) - dot(
            Ae_m, dot(Q, dot(LQt, ldot(TA, dm)))
        ) * (1 - d)

        r1 = (TA * dotd(Q, LQt) * TA).sum()

        dlml_scale = 0.5 * dot(Ae_m, dKAd_m)
        dlml_scale -= sum(w * dKAd_m)
        dlml_scale += 0.5 * dot(w, dot(QS, dot(Q.T, w) * (1 - d)) + d * w)
        dlml_scale -= 0.5 * dotd(ATQ, QS.T).sum() * (1 - d)
        dlml_scale -= 0.5 * sum(TA) * d
        dlml_scale += 0.5 * r * (1 - d) * (1 - d)
        dlml_scale += 0.5 * d * r1 * (1 - d)

        dlml_delta = 0.5 * dot(Ae_m, dKAs_m)
        dlml_delta -= sum(w * dKAs_m)
        dlml_delta += 0.5 * dot(w, -s * dot(QS, dot(Q.T, w)) + s * w)
        dlml_delta += 0.5 * s * dotd(ATQ, QS.T).sum()
        dlml_delta -= 0.5 * sum(TA) * s
        dlml_delta -= 0.5 * s * r * (1 - d)
        dlml_delta += 0.5 * s * r1 * (1 - d)

        g = dict(mean=dlml_mean, scale=dlml_scale, delta=dlml_delta)

        self._cache["grad"] = g

        return g
