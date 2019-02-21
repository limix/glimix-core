from numpy import asfortranarray, log, sqrt

from glimix_core.cov import Kron2SumCov
from glimix_core.mean import KronMean
from glimix_core.util import log2pi
from optimix import Function


class Kron2Sum(Function):
    def __init__(self, Y, A, F, G, rank=1):
        """ LMM for multiple multiple traits.

        Let n, c, and p be the number of samples, covariates, and traits, respectively.
        The outcome variable is a n×p matrix distributed according to

            vec(Y) ~ N((𝐀 ⊗ 𝐅) vec(𝐁), Cᵣ ⊗ GGᵗ + Cₙ ⊗ I).

        𝐀 and 𝐅 are design matrices of dimensions p×p and n×c provided by the user,
        where 𝐅 is the usual matrix of covariates.
        𝐁 is a p×c matrix of fixed-effect sizes.
        G is a n×r matrix provided by the user and I is a n×n identity matrices.
        Cᵣ and Cₙ are both symmetric matrices of dimensions p×p, for which Cₙ is
        guaranteed by our implementation to be full rank.
        The parameters of this model are the matrices 𝐁, Cᵣ, and Cₙ.
        """
        Function.__init__(self)
        Y = asfortranarray(Y)
        self._Y = Y
        self._y = Y.ravel(order="F")
        self._A = A
        self._F = F
        self._cov = Kron2SumCov(Y.shape[1], rank)
        self._cov.G = G
        self._mean = KronMean(F.shape[1], Y.shape[1])
        self._mean.A = A
        self._mean.F = F

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    @property
    def nsamples(self):
        """ Number of samples. """
        return self._Y.shape[0]

    @property
    def ntraits(self):
        """ Number of traits. """
        return self._Y.shape[1]

    @property
    def ncovariates(self):
        """ Number of covariates. """
        return self._F.shape[1]

    def lml(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            Log of the marginal likelihood.
        """
        np = self.nsamples * self.ntraits
        lml = -np * log2pi - self._cov.logdet()

        m = asfortranarray(self._mean.feed().value()).ravel(order="F")
        d = self._y - m
        dKid = d @ self._cov.solve(d)
        lml -= dKid

        return lml / 2

    @property
    def z(self):
        return self._cov.L @ self._y
