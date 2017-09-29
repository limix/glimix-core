from __future__ import division

from numpy import arange, sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose
from optimix import check_grad

from glimix_core.cov import EyeCov, LinearCov, SumCov
from glimix_core.ggp import ExpFamGP
from glimix_core.lik import BernoulliProdLik
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

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(arange(N), purpose='sample')
    mean.set_data(arange(N), purpose='learn')

    cov_left = LinearCov()
    cov_left.scale = 1.5
    cov_left.set_data((X, X), purpose='sample')
    cov_left.set_data((X, X), purpose='learn')

    cov_right = EyeCov()
    cov_right.scale = 1.5
    cov_right.set_data((arange(N), arange(N)), purpose='sample')
    cov_right.set_data((arange(N), arange(N)), purpose='learn')

    cov = SumCov([cov_left, cov_right])

    lik = BernoulliProdLik(LogitLink())

    y = GGPSampler(lik, mean, cov).sample(random)

    return dict(
        mean=mean,
        cov=cov,
        lik=lik,
        y=y,
        cov_left=cov_left,
        cov_right=cov_right)


def test_expfam_ep():
    data = _get_data()
    ep = ExpFamGP((data['y'], ), 'bernoulli', data['mean'], data['cov'])
    assert_allclose(ep.feed().value(), -5.031838893222976)


def test_expfam_ep_function():
    data = _get_data()
    ep = ExpFamGP((data['y'], ), 'bernoulli', data['mean'], data['cov'])

    assert_allclose(check_grad(ep.feed()), 0, atol=1e-4)


def test_expfam_ep_optimize():
    data = _get_data()
    ep = ExpFamGP((data['y'], ), 'bernoulli', data['mean'], data['cov'])
    data['cov_left'].fix('logscale')
    ep.feed().maximize(verbose=False)
    assert_allclose(data['cov_right'].scale, 0.3815125853009603, atol=1e-5)
    assert_allclose(data['mean'].offset, 2.8339582691250604, rtol=1e-6)
