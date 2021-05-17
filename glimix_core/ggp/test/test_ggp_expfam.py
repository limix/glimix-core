from numpy import matmul, sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.ggp import ExpFamGP
from glimix_core.lik import BernoulliProdLik, BinomialProdLik
from glimix_core.link import LogitLink
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler


def _get_data():
    random = RandomState(0)
    N = 10
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    offset = 1.0

    mean = OffsetMean(N)
    mean.offset = offset

    cov_left = LinearCov(X)
    cov_left.scale = 1.5

    cov_right = EyeCov(N)
    cov_right.scale = 1.5

    cov = SumCov([cov_left, cov_right])

    lik = BernoulliProdLik(LogitLink())

    y = GGPSampler(lik, mean, cov).sample(random)

    return dict(
        mean=mean, cov=cov, lik=lik, y=y, cov_left=cov_left, cov_right=cov_right
    )


def test_ggp_expfam():
    data = _get_data()
    ep = ExpFamGP(data["y"], "bernoulli", data["mean"], data["cov"])
    assert_allclose(ep.value(), -5.031838893222976)
    assert_allclose(ep._check_grad(), 0, atol=1e-4)
    data["cov_left"].fix()
    ep.fit(verbose=False)
    assert_allclose(data["cov_right"].scale, 0.3814398504968659, atol=1e-5)
    assert_allclose(data["mean"].offset, 2.8339376861729737, rtol=1e-6)


def test_ggp_expfam_tobi():
    random = RandomState(2)

    n = 30

    ntrials = random.randint(30, size=n)
    K = random.randn(n, n)
    K = matmul(K, K.T)

    lik = BinomialProdLik(ntrials=ntrials, link=LogitLink())

    mean = OffsetMean(n)

    cov2 = EyeCov(n)

    y = GGPSampler(lik, mean, cov2).sample(random)

    ggp = ExpFamGP(y, ("binomial", ntrials), mean, cov2)
    assert_allclose(ggp.lml(), -67.84095700542488)

    ggp.fit(verbose=False)
    assert_allclose(ggp.lml(), -64.26701904994792)
