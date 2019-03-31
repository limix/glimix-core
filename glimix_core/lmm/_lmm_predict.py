from numpy import asarray
from numpy.linalg import solve


class LMMPredict:
    """
    Predict new samples.

    Warning
    -------
    This has not been thoroughly tested and the interface is likely to change.
    """

    def __init__(self, y, beta, v0, v1, mean, covariance):
        self._y = y
        self._beta = beta
        self._v0 = v0
        self._v1 = v1
        self._mean = mean
        self._covariance = covariance

    def predictive_mean(self, Xstar, ks, kss):
        mstar = self.mean_star(Xstar)
        ks = self.covariance_star(ks)
        m = self._mean
        K = self._covariance
        return mstar + ks @ solve(K, self._y - m)

    def predictive_covariance(self, Xstar, ks, kss):
        kss = self.variance_star(kss)
        ks = self.covariance_star(ks)
        K = self._covariance
        ktk = solve(K, ks.T)
        b = []
        for i in range(len(kss)):
            b += [ks[i, :] @ ktk[:, i]]
        b = asarray(b)
        return kss - b

    def mean_star(self, Xstar):
        return Xstar @ self._beta

    def variance_star(self, kss):
        return kss * self._v0 + self._v1

    def covariance_star(self, ks):
        return ks * self._v0
