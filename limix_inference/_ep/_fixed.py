from __future__ import division

from numpy import sum
from numpy import dot
from numpy import newaxis
from numpy import nan_to_num
from numpy_sugar.linalg import cho_solve
from numpy import errstate
from numpy_sugar.linalg import ddot


class FixedEP(object):
    def __init__(self, lml_const, A, C, L, Q, QBiQtCteta, teta, beta_nom):
        self._lml_const = lml_const
        self._QBiQtCteta = QBiQtCteta
        self._teta = teta
        self._A = A
        self._C = C
        self._L = L
        self._Q = Q
        self._beta_nom = beta_nom

    def compute(self, covariates, X):
        A = self._A
        L = self._L
        Q = self._Q
        C = self._C

        AMs0 = ddot(A, covariates, left=True)
        dens0 = AMs0 - ddot(A, dot(Q, cho_solve(L, dot(Q.T, AMs0))), left=True)
        noms0 = dot(self._beta_nom, covariates)

        AMs1 = ddot(A, X, left=True)
        dens1 = AMs1 - ddot(A, dot(Q, cho_solve(L, dot(Q.T, AMs1))), left=True)
        noms1 = dot(self._beta_nom, X)

        row00 = sum(covariates * dens0, 0)
        row01 = sum(covariates * dens1, 0)
        row11 = sum(X * dens1, 0)

        betas0 = noms0 * row11
        betas0 -= noms1 * row01
        betas1 = -noms0 * row01
        betas1 += noms1 * row00

        denom = row00 * row11
        denom -= row01**2

        with errstate(divide='ignore', invalid='ignore'):
            betas0 /= denom
            betas1 /= denom

        betas0 = nan_to_num(betas0)
        betas1 = nan_to_num(betas1)

        ms = dot(covariates, betas0[newaxis, :])
        ms += X * betas1
        QBiQtCteta = self._QBiQtCteta
        teta = self._teta

        Am = ddot(A, ms, left=True)
        w4 = dot(C * teta, ms)
        w4 -= dot(QBiQtCteta, Am)
        QBiQtAm = dot(Q, cho_solve(L, dot(Q.T, Am)))
        w5 = -sum(Am * ms, 0)
        w5 += sum(Am * QBiQtAm, 0)
        w5 /= 2
        lmls = self._lml_const + w4 + w5

        return (lmls, betas1)
