from __future__ import division

from numpy import add, dot, tensordot


class Kron2Sum(object):
    r"""Sum covariance function.

    The mathematical representation is

    .. math::

        K = Cg \kron R + Cn \kron I
    """

    def __init__(self, Cg, R, Cn):
        self._Cg = Cg
        self._R = R
        self._Cn = Cn

    @property
    def dim_r(self):
        return self._R.shape[0]

    @property
    def dim_c(self):
        return self._Cg.shape[0]

    def solve_t(self, Mt):
        return RV

    def solve(self, b):
        B = b.reshape((self.dim_r, self.dim_c, b.shape[1]), order='F')

        RQS = economic_qs(self._R)
        CnQS = economic_qs(self._Cn)

        USi2 = CnQS[0] * CnQS[1]**(-1 / 2)
        Cstar = dot(USi2.T, dot(self._Cg.value(), USi2))
        CstarQS = economic_qs(Cstar)

        Lr = RQS[0][0].T
        Lc = dot(CstarQS[0][0].T, USi2.T)

        SpI = kron(QSa[1], QSb[1]) + 1
        d = 1. / SpI
        D = d.reshape((self._dim_r, self._dim_c), order='F')
        DLMt = D[:, :, newaxis] * vei_CoR_veX(B, Lc, Lr)

        x = vei_CoR_veX(DLMt, Lc.T, Lr.T)

        return x.reshape(b.shape, order='F')

    def logdet(self):
        QS = economic_qs(self._Cn)
        A = self._Cn.QS[0] * self._Cn.QS[1]**(-1 / 2)
        Cstar = dot(A.T, dot(self._Cg.value(), A))
        QSa = economic_qs(self.Cstar())
        QSb = economic_qs(self._R)
        SpI = kron(QSa[1], QSb[1]) + 1
        return sum(log(QS[1])) * self._R.shape[0] + sum(log(SpI))

    def value_reduce(self, values):  # pylint: disable=R0201
        r"""Sum covariance function evaluated at `(f_0, f_1, ...)`."""
        return add.reduce(list(values.values()))

    def gradient_reduce(self, _, gradients):  # pylint: disable=R0201
        r"""Sum of covariance function derivatives.

        Returns:
            :math:`f_0' + f_1' + \dots`
        """
        grad = dict()
        for (gn, gv) in iter(gradients.items()):
            for n, v in iter(gv.items()):
                grad[gn + '.' + n] = v
        return grad


def vei_CoR_veX(X, C, R):
    """
    Args:
        X:  NxPxS tensor
        C:  CxC row covariance (if None: C set to I_PP)
        R:  NxN row covariance (if None: R set to I_NN)
    Returns:
        NxPxS tensor obtained as ve^{-1}((C \kron R) ve(X))
        where ve(X) reshapes X as a NPxS matrix.
    """
    _X = X.transpose((0, 2, 1))
    if R is not None:
        RV = tensordot(R, _X, (1, 0))
    else:
        RV = _X
    if C is not None:
        RV = dot(RV, C.T)
    return RV.transpose((0, 2, 1))


def vec(M):
    return M.reshape((M.size, 1), order='F')
