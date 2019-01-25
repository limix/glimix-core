import numpy as np
from numpy.testing import assert_allclose

from glimix_core.cov import EyeCov
from glimix_core.ggp import ExpFamGP
from glimix_core.lik import BinomialProdLik
from glimix_core.link import LogitLink
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler


def test_ggp_binomial_tobi():
    random = np.random.RandomState(2)

    n = 30

    ntrials = random.randint(30, size=n)
    K = random.randn(n, n)
    K = np.matmul(K, K.T)

    lik = BinomialProdLik(ntrials=ntrials, link=LogitLink())

    mean = OffsetMean()
    mean.set_data(np.ones(n), purpose="sample")
    mean.set_data(np.ones(n), purpose="learn")

    cov2 = EyeCov()
    cov2.set_data((np.arange(n), np.arange(n)), purpose="sample")
    cov2.set_data((np.arange(n), np.arange(n)))

    y = GGPSampler(lik, mean, cov2).sample(random)

    mean.value(np.ones(n))

    ggp = ExpFamGP(y, ("binomial", ntrials), mean, cov2)
    assert_allclose(ggp.lml(), -67.84095700542488)

    ggp.fit(verbose=False)
    assert_allclose(ggp.lml(), -64.26701904994792)
