from numpy import matmul, sqrt
from numpy.random import default_rng
from numpy.testing import assert_allclose

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.ggp import ExpFamGP
from glimix_core.lik import BernoulliProdLik
from glimix_core.link import LogitLink
from glimix_core.mean import OffsetMean


def _get_data():
    random = default_rng(1)
    N = 10
    X = random.normal(size=(N, N + 1))
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

    # y = GGPSampler(lik, mean, cov).sample(random)
    y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    return dict(
        mean=mean, cov=cov, lik=lik, y=y, cov_left=cov_left, cov_right=cov_right
    )


def test_ggp_expfam():
    data = _get_data()
    ep = ExpFamGP(data["y"], "bernoulli", data["mean"], data["cov"])
    assert_allclose(ep.value(), -4.905982177317817)
    assert_allclose(ep._check_grad(), 0, atol=1e-4)
    data["cov_left"].fix()
    ep.fit(verbose=False)
    assert_allclose(data["cov_right"].scale, 1.133659133003428e-08, atol=1e-5)
    assert_allclose(data["mean"].offset, 2.7591672948875834, rtol=1e-6)


def test_ggp_expfam_tobi():
    random = default_rng(2)

    n = 30

    ntrials = random.integers(30, size=n)
    K = random.normal(size=(n, n))
    K = matmul(K, K.T)

    # lik = BinomialProdLik(ntrials=ntrials, link=LogitLink())

    mean = OffsetMean(n)

    cov2 = EyeCov(n)

    # y = GGPSampler(lik, mean, cov2).sample(random)
    y = [
        19.0,
        4.0,
        1.0,
        4.0,
        4.0,
        21.0,
        8.0,
        1.0,
        5.0,
        6.0,
        17.0,
        5.0,
        10.0,
        1.0,
        7.0,
        0.0,
        7.0,
        3.0,
        4.0,
        18.0,
        3.0,
        0.0,
        4.0,
        2.0,
        18.0,
        12.0,
        14.0,
        0.0,
        12.0,
        3.0,
    ]

    ggp = ExpFamGP(y, ("binomial", ntrials), mean, cov2)
    assert_allclose(ggp.lml(), -75.92472423326483)

    ggp.fit(verbose=False)
    assert_allclose(ggp.lml(), -75.08008540883655)
