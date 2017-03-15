from __future__ import division

from numpy import arange, sqrt
from numpy.random import RandomState
from numpy.testing import assert_allclose

from limix_inference.cov import EyeCov, LinearCov, SumCov
from limix_inference.ggp import ExpFamGP
from limix_inference.lik import BernoulliProdLik
from limix_inference.link import LogitLink
from limix_inference.mean import OffsetMean
from limix_inference.random import GLMMSampler


def _get_data():
    random = RandomState(458)
    N = 100
    X = random.randn(N, N + 1)
    X -= X.mean(0)
    X /= X.std(0)
    X /= sqrt(X.shape[1])
    offset = 1.0

    mean = OffsetMean()
    mean.offset = offset
    mean.set_data(N, purpose='sample')
    mean.set_data(N, purpose='learn')

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

    y = GLMMSampler(lik, mean, cov).sample(random)

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
    assert_allclose(ep.feed().value(), -60.84029280372346)


def test_expfam_ep_function():
    data = _get_data()
    ep = ExpFamGP((data['y'], ), 'bernoulli', data['mean'], data['cov'])

    grad = ep.feed().gradient()

    x0 = ep.variables().flatten()
    f0 = ep.feed().value()
    step = 1e-4

    emp_grad = []
    for i, v in enumerate(x0):
        x1 = x0.copy()
        x1[i] = v + step
        ep.variables().from_flat(x1)
        f1 = ep.feed().value()
        emp_grad.append((f1 - f0) / step)

    assert_allclose(grad, emp_grad, rtol=1e-3)


def test_expfam_ep_optimize():
    data = _get_data()
    ep = ExpFamGP((data['y'], ), 'bernoulli', data['mean'], data['cov'])
    data['cov_left'].fix('logscale')
    ep.feed().maximize(progress=False)
    assert_allclose(data['cov_right'].scale, 4.165865119892221e-06)
    assert_allclose(data['mean'].offset, 1.0326586373049373)


if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
