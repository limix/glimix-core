from __future__ import division

from numpy import arange, sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.ggp import ExpFamGP
from glimix_core.lik import BernoulliProdLik
from glimix_core.link import LogitLink
from glimix_core.mean import OffsetMean
from glimix_core.random import GGPSampler
from optimix import check_grad


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

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.X = X

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.dim = N

    cov = SumCov([cov_left, cov_right])

    lik = BernoulliProdLik(LogitLink())

    y = GGPSampler(lik, mean, cov).sample(random)

    return dict(
        mean=mean, cov=cov, lik=lik, y=y, cov_left=cov_left, cov_right=cov_right
    )


def test_expfam_ep():
    data = _get_data()
    ep = ExpFamGP((data["y"],), "bernoulli", data["mean"], data["cov"])
    assert_allclose(ep.value(), -5.031838893222976)


def test_expfam_ep_function():
    data = _get_data()
    ep = ExpFamGP((data["y"],), "bernoulli", data["mean"], data["cov"])

    assert_allclose(ep._check_grad(), 0, atol=1e-4)


def test_expfam_ep_optimize():
    data = _get_data()
    ep = ExpFamGP((data["y"],), "bernoulli", data["mean"], data["cov"])
    data["cov_left"].fix_scale()
    ep.maximize(verbose=False)
    assert_allclose(data["cov_right"].scale, 0.38162494996579965, atol=1e-5)
    assert_allclose(data["mean"].offset, 2.8339908366727267, rtol=1e-6)
